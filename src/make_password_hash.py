"""Create a salted PBKDF2 hash for the dashboard sign-in (app/auth.py).

    python -m src.make_password_hash            # prompts, nothing is echoed or stored
Paste the printed value into `.streamlit/secrets.toml` under [auth] (see app/auth.py). Never commit that file.
"""
from __future__ import annotations

import getpass
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))


def main() -> None:
    import auth

    pw = getpass.getpass("Password: ")
    if len(pw) < 10:
        print("Use at least 10 characters (a passphrase of several words is best).")
        raise SystemExit(1)
    if pw != getpass.getpass("Repeat: "):
        print("The two entries differ.")
        raise SystemExit(1)
    print(auth.make_hash(pw))


if __name__ == "__main__":
    main()
