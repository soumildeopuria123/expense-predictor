#!/usr/bin/env bash
# Start the dashboard. Ctrl-C to stop.
#
#   ./start.sh                          # http://localhost:3000
#   ./start.sh <new-preview-host>       # after a restore changes the hostname
#   kill "$(cat .streamlit/server.pid)" # stop a detached instance
#
# All server settings (address, port, allowed hosts) live in
# .streamlit/config.toml, so a bare streamlit run works too:
#
#   ./.venv/bin/streamlit run app/dashboard.py
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

[ -x .venv/bin/streamlit ] || {
  echo "No virtualenv found. Create one first:" >&2
  echo "  python3 -m venv .venv && ./.venv/bin/pip install -e '.[dev]'" >&2
  exit 1
}

CONFIG=.streamlit/config.toml
PORT="$(./.venv/bin/python - <<'PY'
import re, pathlib
t = pathlib.Path(".streamlit/config.toml").read_text()
m = re.search(r"^\s*port\s*=\s*(\d+)", t, re.M)
print(m.group(1) if m else "3000")
PY
)"

# Optional: rotate the preview hostname recorded in allowedHosts.
if [ $# -ge 1 ] && [ -n "$1" ]; then
  HOSTS_CONFIG="$CONFIG" NEW_HOST="$1" ./.venv/bin/python - <<'PY'
import os, re, pathlib

p = pathlib.Path(os.environ["HOSTS_CONFIG"])
if not p.exists():
    raise SystemExit(f"missing {p}")
text = p.read_text()
new = os.environ["NEW_HOST"]
local = {"localhost", "127.0.0.1"}

m = re.search(r"(allowedHosts\s*=\s*\[)\s*(.*?)\s*(\])", text, re.S)
if not m:
    raise SystemExit(f"no allowedHosts array found in {p}")

entries = re.findall(r'"([^"]+)"', m.group(2))
if new in local:
    raise SystemExit(f"refusing to replace a local entry with {new}")
# Wildcard entries ("*", "*.example.com") cover hosted deployments and are
# never the preview host, so they are not rotation candidates.
foreign = [e for e in entries if e not in local and "*" not in e]
if len(foreign) != 1:
    raise SystemExit(
        f"expected exactly 1 non-local entry to rotate in {p}, found {foreign}"
    )
if foreign[0] == new:
    print(f"allowedHosts preview host already {new}")
    raise SystemExit(0)

# Replace in place so order and every other entry are preserved, rebuilding
# the array with fixed formatting so a round trip (rotate away, rotate back)
# restores the file byte-for-byte. This works in either direction (a plain
# hostname match, not a .style.dev pattern), so a rotated host can always be
# rotated back.
rotated = [new if e == foreign[0] else e for e in entries]
lines = "\n".join(f'  "{e}",' for e in rotated)
p.write_text(text[: m.end(1)] + "\n" + lines + "\n" + text[m.start(3) :])
print(f"allowedHosts: {foreign[0]} -> {new}")
PY
  ROTATED=1
fi

# Already serving? Say so instead of failing with "port already in use".
code="$(curl -s -o /dev/null -w '%{http_code}' --max-time 3 \
        "http://127.0.0.1:${PORT}/_stcore/health" || true)"
if [ "$code" != "000" ] && [ "$code" != "" ]; then
  echo "Dashboard is already running on port ${PORT} -> http://localhost:${PORT}"
  [ -f .streamlit/server.pid ] && echo "  pid $(cat .streamlit/server.pid) (stop: kill \$(cat .streamlit/server.pid))"
  # A running server keeps the allowedHosts it started with.
  [ "${ROTATED:-0}" = 1 ] && echo "  note: restart to apply the rotated hostname: kill \$(cat .streamlit/server.pid) && ./start.sh"
  exit 0
fi

echo "dashboard : http://localhost:${PORT}"
echo "config    : $CONFIG"
echo

# Record our own PID; `exec` makes this the server process, foreground or not.
echo $$ > .streamlit/server.pid
exec ./.venv/bin/streamlit run app/dashboard.py
