"""
Access-code gating + subscription tiers for the DeepFaceLive web app.

This layer is intentionally OPTIONAL. It activates only when DATABASE_URL is
present (i.e. on the always-on Replit host). On the free Colab GPU clone there
is no DATABASE_URL, so ``enabled()`` returns False and the app runs fully open
— the smooth-GPU path is never blocked by the paywall.

Codes live in Postgres so they survive restarts and redeploys, and are shared
by both the web app (login) and the Telegram bot (generation).
"""
import os
import ssl
import secrets
from datetime import datetime, timedelta, timezone
from urllib.parse import urlparse, unquote

try:
    import pg8000.native as _pg          # pure-Python driver — cooperates with eventlet
    _DRIVER_OK = True
except Exception:                        # pragma: no cover
    _pg = None
    _DRIVER_OK = False

# Subscription tiers — prices in Naira (₦).
TIERS = {
    'weekly':  {'days': 7,   'price': 50000,  'label': 'Weekly'},
    'monthly': {'days': 30,  'price': 100000, 'label': 'Monthly'},
    'yearly':  {'days': 365, 'price': 150000, 'label': 'Yearly'},
}
TIERS_LIST = [dict(key=k, **v) for k, v in TIERS.items()]

# Unambiguous alphabet (no I/O/0/1) for codes people type on a phone.
_ALPHABET = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"


def enabled():
    """True only when a Postgres driver AND a database are available."""
    return _DRIVER_OK and bool(os.environ.get('DATABASE_URL'))


def _conn_params():
    u = urlparse(os.environ['DATABASE_URL'])
    return dict(
        user=unquote(u.username or ''),
        password=unquote(u.password or ''),
        host=u.hostname or 'localhost',
        port=u.port or 5432,
        database=(u.path or '/').lstrip('/') or 'postgres',
    )


def _connect():
    """Open a fresh short-lived connection. The dev Postgres refuses SSL while
    managed/hosted Postgres often requires it, so try plain first, then SSL
    (verified, then encrypted-but-unverified)."""
    p = _conn_params()
    try:
        return _pg.Connection(**p)
    except Exception:
        try:
            return _pg.Connection(ssl_context=ssl.create_default_context(), **p)
        except Exception:
            ctx = ssl.create_default_context()
            ctx.check_hostname = False
            ctx.verify_mode = ssl.CERT_NONE
            return _pg.Connection(ssl_context=ctx, **p)


def init_db():
    con = _connect()
    try:
        con.run("""
            CREATE TABLE IF NOT EXISTS access_codes (
                id           SERIAL PRIMARY KEY,
                code         VARCHAR(32) UNIQUE NOT NULL,
                tier         VARCHAR(16) NOT NULL,
                status       VARCHAR(16) NOT NULL DEFAULT 'unused',
                created_at   TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                activated_at TIMESTAMPTZ,
                expires_at   TIMESTAMPTZ,
                created_by   VARCHAR(64),
                note         TEXT
            )
        """)
        # A code locks to the first device that redeems it (anti-sharing).
        con.run("ALTER TABLE access_codes ADD COLUMN IF NOT EXISTS bound_device VARCHAR(64)")
        con.run("""
            CREATE TABLE IF NOT EXISTS bot_admins (
                chat_id  BIGINT PRIMARY KEY,
                username VARCHAR(64),
                added_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            )
        """)
    finally:
        con.close()


def _new_code():
    body = ''.join(secrets.choice(_ALPHABET) for _ in range(8))
    return f"TMFG-{body[:4]}-{body[4:]}"


def generate_code(tier, created_by=None, note=None):
    """Create an unused code for the given tier and return the code string."""
    tier = (tier or '').lower()
    if tier not in TIERS:
        raise ValueError(f"unknown tier: {tier}")
    con = _connect()
    try:
        for _ in range(8):               # retry on the (astronomically rare) collision
            code = _new_code()
            try:
                con.run(
                    "INSERT INTO access_codes (code, tier, status, created_by, note) "
                    "VALUES (:c, :t, 'unused', :b, :n)",
                    c=code, t=tier,
                    b=(str(created_by) if created_by is not None else None), n=note,
                )
                return code
            except Exception as e:
                if 'unique' in str(e).lower() or 'duplicate' in str(e).lower():
                    continue
                raise
        raise RuntimeError("could not generate a unique code")
    finally:
        con.close()


def redeem(code, device_id=None):
    """Validate a code and bind it to the first device that uses it.

    On first use the code is activated (expiry set) and locked to ``device_id``.
    Afterwards ONLY that same device may redeem it again — any other device is
    rejected with reason ``in_use`` (stops a buyer sharing one code around).

    Returns {'ok': True, 'tier': ..., 'expires_at': datetime} or
            {'ok': False, 'reason': 'invalid'|'expired'|'revoked'|'in_use'|'empty'}.
    """
    code = (code or '').strip().upper()
    if not code:
        return {'ok': False, 'reason': 'empty'}
    device_id = (device_id or '').strip() or None
    con = _connect()
    try:
        rows = con.run("SELECT tier, status, expires_at, bound_device "
                       "FROM access_codes WHERE code = :c", c=code)
        if not rows:
            return {'ok': False, 'reason': 'invalid'}
        tier, status, expires_at, bound_device = rows[0]
        now = datetime.now(timezone.utc)
        if status == 'revoked':
            return {'ok': False, 'reason': 'revoked'}

        if status == 'unused':
            # First use: activate AND lock to this device. The WHERE status guard
            # + RETURNING makes this atomic — if two devices race on a fresh code
            # only one UPDATE matches; the loser falls through to the binding
            # check below and is rejected.
            exp = now + timedelta(days=TIERS.get(tier, {}).get('days', 7))
            won = con.run("UPDATE access_codes SET status='active', activated_at=:a, "
                          "expires_at=:e, bound_device=:d "
                          "WHERE code=:c AND status='unused' RETURNING id",
                          a=now, e=exp, d=device_id, c=code)
            if won:
                return {'ok': True, 'tier': tier, 'expires_at': exp}
            rows = con.run("SELECT tier, status, expires_at, bound_device "
                           "FROM access_codes WHERE code = :c", c=code)
            if not rows:
                return {'ok': False, 'reason': 'invalid'}
            tier, status, expires_at, bound_device = rows[0]

        # already active — must carry a real expiry to be usable
        if expires_at is None:
            return {'ok': False, 'reason': 'invalid'}
        if expires_at.tzinfo is None:
            expires_at = expires_at.replace(tzinfo=timezone.utc)
        if expires_at <= now:
            con.run("UPDATE access_codes SET status='expired' WHERE code=:c", c=code)
            return {'ok': False, 'reason': 'expired'}
        # Single-device enforcement.
        if bound_device:
            if device_id != bound_device:
                return {'ok': False, 'reason': 'in_use'}
        else:
            # legacy code activated before binding existed — bind it now, but do
            # so atomically so a race can't bind two devices at once.
            bound = con.run("UPDATE access_codes SET bound_device=:d "
                            "WHERE code=:c AND bound_device IS NULL RETURNING id",
                            d=device_id, c=code)
            if not bound:
                rows2 = con.run("SELECT bound_device FROM access_codes WHERE code=:c", c=code)
                if rows2 and rows2[0][0] and rows2[0][0] != device_id:
                    return {'ok': False, 'reason': 'in_use'}
        return {'ok': True, 'tier': tier, 'expires_at': expires_at}
    finally:
        con.close()


# --- Telegram-bot admin helpers -----------------------------------------

def add_admin(chat_id, username=None):
    con = _connect()
    try:
        con.run("INSERT INTO bot_admins (chat_id, username) VALUES (:i, :u) "
                "ON CONFLICT (chat_id) DO UPDATE SET username = :u",
                i=int(chat_id), u=username)
    finally:
        con.close()


def is_admin(chat_id):
    con = _connect()
    try:
        return bool(con.run("SELECT 1 FROM bot_admins WHERE chat_id = :i", i=int(chat_id)))
    finally:
        con.close()


def recent_codes(limit=15):
    con = _connect()
    try:
        return con.run("SELECT code, tier, status, expires_at FROM access_codes "
                       "ORDER BY created_at DESC LIMIT :l", l=int(limit))
    finally:
        con.close()
