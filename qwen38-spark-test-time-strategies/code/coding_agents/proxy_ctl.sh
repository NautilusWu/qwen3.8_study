#!/usr/bin/env bash
# proxy_ctl.sh start <log.jsonl> | stop   (finds the proxy by its listening port, never by command-line match)
cd "$(dirname "$0")"
pid() { ss -ltnp 2>/dev/null | awk '/127.0.0.1:8099 / {print}' | grep -o 'pid=[0-9]*' | head -1 | cut -d= -f2; }
case "$1" in
  stop) p="$(pid)"; [ -n "$p" ] && kill "$p"; for i in 1 2 3 4 5; do [ -z "$(pid)" ] && break; sleep 0.5; done ;;
  start) "$0" stop; setsid nohup python3 normalize_proxy.py --listen 8099 --log "$2" > "${2%.jsonl}.proxy.out" 2>&1 < /dev/null &
         for i in $(seq 1 20); do curl -sf -m 1 http://127.0.0.1:8099/v1/models >/dev/null && break; sleep 0.25; done ;;
esac
echo "proxy pid: $(pid)"
