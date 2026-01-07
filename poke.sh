#!/usr/bin/env bash
set -euo pipefail

idx=""
delta="0.2"
duration="2.0"
rate="100"
ip="127.0.0.1"
port="20001"

usage() {
  cat <<'USAGE'
Usage: ./poke.sh --idx N [--delta D] [--duration S] [--rate HZ] [--ip IP] [--port PORT]

Sends a single-joint PD target packet to the MuJoCo UDP sim (port 20001).
Joint order:
  0 FL_HipX, 1 FL_HipY, 2 FL_Knee, 3 FR_HipX, 4 FR_HipY, 5 FR_Knee,
  6 HL_HipX, 7 HL_HipY, 8 HL_Knee, 9 HR_HipX, 10 HR_HipY, 11 HR_Knee
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --idx)
      idx="${2:-}"
      shift 2
      ;;
    --delta)
      delta="${2:-}"
      shift 2
      ;;
    --duration)
      duration="${2:-}"
      shift 2
      ;;
    --rate)
      rate="${2:-}"
      shift 2
      ;;
    --ip)
      ip="${2:-}"
      shift 2
      ;;
    --port)
      port="${2:-}"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      usage
      exit 1
      ;;
  esac
done

if [[ -z "${idx}" ]]; then
  echo "Missing --idx."
  usage
  exit 1
fi

python3 - "$idx" "$delta" "$duration" "$rate" "$ip" "$port" <<'PY'
import sys
import socket
import struct
import time

idx = int(sys.argv[1])
delta = float(sys.argv[2])
duration = float(sys.argv[3])
rate = float(sys.argv[4])
ip = sys.argv[5]
port = int(sys.argv[6])

dof = 12
if idx < 0 or idx >= dof:
    raise SystemExit(f"idx must be in [0, {dof-1}], got {idx}")

# Standup target from deploy defaults (hardware order).
pos = [
    -0.0154048, -0.76697,  1.53761,
     0.0159887, -0.768286, 1.53636,
    -0.0221317, -0.765865, 1.54788,
     0.0224431, -0.767203, 1.54679,
]
pos[idx] += delta

kp = [100.0] * dof
kd = [2.5] * dof
vel = [0.0] * dof
tau = [0.0] * dof

packet = struct.pack(f"{dof}f" * 5, *(kp + pos + kd + vel + tau))
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
period = 1.0 / rate if rate > 0.0 else 0.01
end = time.time() + duration
while time.time() < end:
    sock.sendto(packet, (ip, port))
    time.sleep(period)
PY
