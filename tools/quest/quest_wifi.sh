#!/usr/bin/env bash
# Switches the Quest to Wi-Fi ADB and keeps it awake when not worn. Run it with
# the USB cable plugged in, once after every headset restart; then the cable can
# come out. The Quest and this
# PC must be on the same network and allowed to reach each other (many
# university networks block that; a lab router or hotspot works).
set -euo pipefail
SERIAL=$(adb devices | awk 'NR > 1 && $2 == "device" && $1 !~ /:/ {print $1; exit}')
if [[ -z $SERIAL ]]; then
  echo "No Quest on USB (adb devices). Plug the cable in and allow USB debugging in the headset."
  exit 1
fi
IP=$(adb -s "$SERIAL" shell ip -f inet addr show wlan0 2>/dev/null | sed -n 's/.*inet \([0-9.]*\).*/\1/p' | head -1)
if [[ -z $IP ]]; then
  echo "The Quest has no Wi-Fi address. Connect it to a Wi-Fi network first."
  exit 1
fi
adb -s "$SERIAL" tcpip 5555 > /dev/null
sleep 2
adb connect "$IP:5555" > /dev/null 2>&1 || true
sleep 1
STATE=$(adb devices | awk -v d="$IP:5555" '$1 == d {print $2}')
if [[ $STATE == device ]]; then
  echo "Wi-Fi ADB is on. Headset IP: $IP"
  # Keep the headset awake when it is not worn (proximity-sensor override; the
  # teleop and monitor renew it while they run). Undo: quest_monitor.py --restore
  adb -s "$IP:5555" shell input keyevent KEYCODE_WAKEUP > /dev/null 2>&1 || true
  adb -s "$IP:5555" shell am broadcast -a com.oculus.vrpowermanager.prox_close > /dev/null 2>&1 || true
  echo "Headset set to stay awake when taken off (until it restarts)."
elif [[ $STATE == unauthorized ]]; then
  echo "The headset ($IP) asks to allow debugging for the Wi-Fi connection: put it on, tick"
  echo "\"Always allow\" and accept, then check with: adb devices"
else
  echo "Wi-Fi ADB is enabled on the headset ($IP), but this PC cannot reach it (ping $IP to check)."
  exit 1
fi
echo
echo "Use it with:"
echo "  quest_connection = $IP      (in config/teleop.conf)"
echo "  python3 tools/quest/quest_monitor.py --connection $IP"
