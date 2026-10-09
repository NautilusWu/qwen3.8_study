#!/usr/bin/env bash
# Turn the monitor off (DPMS) and keep it off while a measurement runs; every 15 s log the number of
# active monitors, whether it had to be turned off again, GPU temperature and GPU power.
#   scripts/display_off.sh display_log.csv &      # stop it with: kill %1
# X11 sessions only (xset, xrandr). A mouse move or key press wakes the monitor; this turns it off again
# within 15 s and records it. It does not stop anything on high temperature; it only logs it.
set -u
out="${1:-display_log.csv}"

active_monitors() {  # monitors the GPU is actively driving (0 = DPMS off or disconnected)
  local n=0 sock d c
  for sock in /tmp/.X11-unix/X*; do
    [ -e "$sock" ] || continue
    d=":${sock##*/X}"
    DISPLAY="$d" timeout 5 xset q 2>/dev/null | grep -q "Monitor is Off" && continue
    c="$(DISPLAY="$d" timeout 5 xrandr --current 2>/dev/null | grep -cE ' connected( primary)? [0-9]+x[0-9]+\+')"
    n=$(( n + ${c:-0} ))
  done
  echo "$n"
}

monitors_off() {
  local sock
  for sock in /tmp/.X11-unix/X*; do [ -e "$sock" ] && DISPLAY=":${sock##*/X}" timeout 5 xset dpms force off; done
}

[ -f "$out" ] || echo "utc,active_monitors,turned_off_again,gpu_temp_c,gpu_power_w" > "$out"
while true; do
  n="$(active_monitors)"; again=0
  if [ "${n:-0}" -gt 0 ]; then monitors_off && again=1; fi
  g="$(nvidia-smi --query-gpu=temperature.gpu,power.draw --format=csv,noheader,nounits 2>/dev/null | tr -d ' ')"
  printf '%s,%s,%s,%s\n' "$(date -u +%FT%TZ)" "$n" "$again" "${g:-,}" >> "$out"
  sleep 15
done
