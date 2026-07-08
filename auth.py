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


def redeem(code):
    """Validate a code. Activates it (sets expiry) on first use.

    Returns {'ok': True, 'tier': ..., 'expires_at': datetime} or
            {'ok': False, 'reason': 'invalid'|'expired'|'revoked'|'empty'}.
    """
    code = (code or '').strip().upper()
    if not code:
        return {'ok': False, 'reason': 'empty'}
    con = _connect()
    try:
        rows = con.run("SELECT tier, status, expires_at FROM access_codes WHERE code = :c", c=code)
        if not rows:
            return {'ok': False, 'reason': 'invalid'}
        tier, status, expires_at = rows[0]
        now = datetime.now(timezone.utc)
        if status == 'revoked':
            return {'ok': False, 'reason': 'revoked'}
        if status == 'unused':
            exp = now + timedelta(days=TIERS.get(tier, {}).get('days', 7))
            con.run("UPDATE access_codes SET status='active', activated_at=:a, expires_at=:e WHERE code=:c",
                    a=now, e=exp, c=code)
            return {'ok': True, 'tier': tier, 'expires_at': exp}
        # already active — must carry a real expiry to be usable
        if expires_at is None:
            return {'ok': False, 'reason': 'invalid'}
        if expires_at.tzinfo is None:
            expires_at = expires_at.replace(tzinfo=timezone.utc)
        if expires_at <= now:
            con.run("UPDATE access_codes SET status='expired' WHERE code=:c", c=code)
            return {'ok': False, 'reason': 'expired'}
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
