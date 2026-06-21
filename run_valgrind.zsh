#!/usr/bin/env zsh

# Instructions:
# 1. For pyvale, make sure you're running the right venv, since it requires Python 3.11
# 2. For first time use, you may need to add the executable rights chmod +x run_valgrind.zsh
# 3. Then run as usual via ./run_valgrind.zsh
# Nb4, thos assumes that this script is in the same folder as the.py file to test

# Exit on error
set -euo pipefail

# 1) Run python under Valgrind/Callgrind
#    Adjust the name of the file as needed. Hardcoded here because I usually test the same file
valgrind \
  --tool=callgrind \
  --instr-atstart=no \
  --simulate-cache=yes \
  --collect-jumps=yes \
  --dump-instr=yes \
  python rttests.py

# 2) Get the PID Valgrind used in the callgrind.out.PID filename
#    Valgrind's callgrind output is named callgrind.out.<pid> by default.
last_pid=$(ls -t callgrind.out.* 2>/dev/null | head -n 1)

if [[ -z "${last_pid}" ]]; then
  echo "No callgrind.out.* file found."
  exit 1
fi

# Extract numeric PID from filename callgrind.out.PID
pid=${last_pid##callgrind.out.}

# 3) Annotate into callgrind_PID.txt
out_file="callgrind_${pid}.txt"
callgrind_annotate --auto=yes "callgrind.out.${pid}" > "${out_file}"

# 4) Echo the filename
echo "Annotated callgrind output written to: ${out_file}"
