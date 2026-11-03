"""Create a salted PBKDF2 hash for the dashboard sign-in (app/auth.py).

    python -m src.make_password_hash            # prompts, nothing is echoed or stored
    python -m src.make_password_hash --password "..."
Paste the printed value into `.streamlit/secrets.toml` under [auth] (see app/auth.py). Never commit that file.
"""
from __future__ import annotations

import argparse
import getpass
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))


def main() -> None:
    import auth

    ap = argparse.ArgumentParser()
    ap.add_argument("--password", help="Password to hash (omit to be prompted interactively)")
    args = ap.parse_args()

    if args.password is not None:
        pw = args.password
    else:
        pw = getpass.getpass("Password: ")
        if pw != getpass.getpass("Repeat: "):
            print("The two entries differ.")
            raise SystemExit(1)
    if len(pw) < 8:
        print("Use at least 8 characters (a passphrase of several words is best).")
        raise SystemExit(1)
    print(auth.make_hash(pw))


if __name__ == "__main__":
    main()
