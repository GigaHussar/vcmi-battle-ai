import os
import subprocess
import time
import socket
import json
import subprocess as sp

VCMI_BINARY = "/Users/syntaxerror/vcmi/build/bin/vcmiclient"

# Check if VCMI binary exists
if not os.path.isfile(VCMI_BINARY):
    print(f"Error: VCMI client not found at {VCMI_BINARY}")
    exit(1)


# Launch VCMI without arguments
print("Launching VCMI...")
vcmi_process = subprocess.Popen([VCMI_BINARY])

# Wait briefly for video to start, then send Return key to skip it
print("Waiting to skip intro video...")
time.sleep(3)
try:
    sp.run(["osascript", "-e", 'tell application "System Events" to key code 36'])  # Return
    print("Sent Return key to skip intro video.")
except Exception as e:
    print(f"Failed to send key press: {e}")

# Wait for main menu
print("Waiting for main menu...")
time.sleep(5)

# Send 'l' to open Load Game menu
try:
    sp.run(["osascript", "-e", 'tell application "System Events" to keystroke "l"'])
    print("Sent 'l' to open Load Game menu.")
except Exception as e:
    print(f"Failed to send 'l' key: {e}")

# Wait for load menu
time.sleep(3)

# Send 's' to select single scenario
try:
    sp.run(["osascript", "-e", 'tell application "System Events" to keystroke "s"'])
    print("Sent 's' to select single scenario.")
except Exception as e:
    print(f"Failed to send 's' key: {e}")

# Wait before confirming
time.sleep(2)

# Press Return to confirm selection
try:
    sp.run(["osascript", "-e", 'tell application "System Events" to key code 36'])
    print("Pressed Return to confirm selection.")
except Exception as e:
    print(f"Failed to send Return key: {e}")

# Wait for game world to load
time.sleep(5)

# Press Right Arrow to move hero
'''
try:
    sp.run(["osascript", "-e", 'tell application "System Events" to key code 124'])  # Right Arrow
    print("Pressed Right Arrow to move hero.")
except Exception as e:
    print(f"Failed to send Right Arrow key: {e}")
'''
# Wait for VCMI to finish
#vcmi_process.wait()
