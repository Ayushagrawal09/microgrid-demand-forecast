"""One-time helper: turns the 'code' from the Zoho API Console (Self Client) into a refresh token.
Run:  python get_refresh_token.py"""
import json
import urllib.parse
import urllib.request

accounts = input("Accounts URL [https://accounts.zoho.in]: ").strip() or "https://accounts.zoho.in"
client_id = input("Client ID: ").strip()
client_secret = input("Client Secret: ").strip()
code = input("Code (from Generate Code tab, valid only a few minutes): ").strip()

data = urllib.parse.urlencode({"grant_type": "authorization_code", "client_id": client_id,
                               "client_secret": client_secret, "code": code}).encode()
req = urllib.request.Request(accounts.rstrip("/") + "/oauth/v2/token", data=data, method="POST")
with urllib.request.urlopen(req, timeout=20) as resp:
    print(json.dumps(json.loads(resp.read().decode()), indent=2))
print("\nCopy the refresh_token value. That is your ZOHO_REFRESH_TOKEN.")
