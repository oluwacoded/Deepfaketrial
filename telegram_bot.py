"""
Owner-only Telegram bot that generates access codes for the DeepFaceLive app.

Runs as an eventlet green thread from web_server.py using long polling, so no
public webhook / extra port is needed. Activates only when TELEGRAM_BOT_TOKEN
is set AND the access database is available.

Owner setup:
  1. Create a bot with @BotFather and put the token in TELEGRAM_BOT_TOKEN.
  2. Pick a passphrase and put it in BOT_ADMIN_PASSPHRASE.
  3. Message your bot once:  /auth <passphrase>   -> you become the admin.
  4. Generate codes:         /gen weekly | monthly | yearly
"""
import os
import time

import requests

import auth

_API = "https://api.telegram.org/bot{token}/{method}"

_HELP = (
    "🤖 <b>Teddy MFG — code bot</b>\n\n"
    "Authorize yourself once:\n"
    "<code>/auth YOUR_PASSPHRASE</code>\n\n"
    "Then generate access codes:\n"
    "<code>/gen weekly</code>  — ₦50,000 · 7 days\n"
    "<code>/gen monthly</code> — ₦100,000 · 30 days\n"
    "<code>/gen yearly</code>  — ₦150,000 · 365 days\n\n"
    "<code>/codes</code> — list the 15 most recent codes"
)


def _token():
    return os.environ.get('TELEGRAM_BOT_TOKEN')


def _passphrase():
    return os.environ.get('BOT_ADMIN_PASSPHRASE')


def enabled():
    return bool(_token()) and auth.enabled()


def _call(method, **params):
    r = requests.post(_API.format(token=_token(), method=method), json=params, timeout=45)
    return r.json()


def _send(chat_id, text):
    try:
        _call('sendMessage', chat_id=chat_id, text=text, parse_mode='HTML',
              disable_web_page_preview=True)
    except Exception as e:                         # noqa: BLE001
        print(f'[bot] send error: {e}')


def _handle(update):
    msg = update.get('message') or update.get('edited_message')
    if not msg:
        return
    chat = msg.get('chat') or {}
    if chat.get('type') != 'private':
        return  # admin actions are DM-only — ignore groups/channels
    chat_id = chat['id']
    frm = msg.get('from') or {}
    user_id = frm.get('id', chat_id)          # bind admin identity to the sender
    username = frm.get('username') or chat.get('username')
    text = (msg.get('text') or '').strip()
    if not text:
        return
    parts = text.split()
    cmd = parts[0].lower().lstrip('/').split('@')[0]

    if cmd in ('start', 'help'):
        _send(chat_id, _HELP)
        return

    if cmd == 'auth':
        pw = _passphrase()
        if len(parts) < 2:
            _send(chat_id, "Send: <code>/auth YOUR_PASSPHRASE</code>")
        elif pw and parts[1] == pw:
            auth.add_admin(user_id, username)
            _send(chat_id, "✅ You are now the admin. Use <code>/gen weekly|monthly|yearly</code>.")
        else:
            _send(chat_id, "❌ Wrong passphrase.")
        return

    # everything below is admin-only
    if not auth.is_admin(user_id):
        _send(chat_id, "🔒 Not authorized. Send <code>/auth YOUR_PASSPHRASE</code> first.")
        return

    if cmd == 'gen':
        tier = parts[1].lower() if len(parts) > 1 else ''
        if tier not in auth.TIERS:
            _send(chat_id, "Usage: <code>/gen weekly | monthly | yearly</code>")
            return
        try:
            code = auth.generate_code(tier, created_by=user_id)
        except Exception as e:                     # noqa: BLE001
            _send(chat_id, f"⚠️ Could not create code: {e}")
            return
        info = auth.TIERS[tier]
        _send(chat_id,
              f"✅ <b>{info['label']}</b> code — ₦{info['price']:,} · {info['days']} days\n\n"
              f"<code>{code}</code>\n\n"
              "Send this to the buyer. It starts counting from their first login.")
        return

    if cmd == 'codes':
        try:
            rows = auth.recent_codes(15)
        except Exception as e:                     # noqa: BLE001
            _send(chat_id, f"⚠️ {e}")
            return
        if not rows:
            _send(chat_id, "No codes yet.")
            return
        lines = []
        for code, tier, status, exp in rows:
            exp_s = exp.strftime('%Y-%m-%d') if exp else '—'
            lines.append(f"<code>{code}</code> · {tier} · {status} · exp {exp_s}")
        _send(chat_id, "Recent codes:\n" + "\n".join(lines))
        return

    _send(chat_id, "Unknown command. Send /help")


def run():
    if not enabled():
        print('[bot] disabled (needs TELEGRAM_BOT_TOKEN + DATABASE_URL).')
        return
    offset = None
    try:                                           # skip any backlog on startup
        res = _call('getUpdates', offset=-1, timeout=0)
        if res.get('ok') and res.get('result'):
            offset = res['result'][-1]['update_id'] + 1
    except Exception as e:                          # noqa: BLE001
        print(f'[bot] init error: {e}')
    print('[bot] Telegram polling started.')
    while True:
        try:
            res = _call('getUpdates', offset=offset, timeout=30)
            if not res.get('ok'):
                time.sleep(3)
                continue
            for upd in res.get('result', []):
                offset = upd['update_id'] + 1
                try:
                    _handle(upd)
                except Exception as e:              # noqa: BLE001
                    print(f'[bot] handle error: {e}')
        except Exception as e:                      # noqa: BLE001
            print(f'[bot] poll error: {e}')
            time.sleep(3)
