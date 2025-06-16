import os
import subprocess
import time
import subprocess as sp

VCMI_BINARY = "/Users/syntaxerror/vcmi/build/bin/vcmiclient"
vcmi_process = None  # Global reference to the process

def open_vcmi_process():
    """
    Launches the VCMI client process.
    """
    global vcmi_process

    # Check if VCMI binary exists
    if not os.path.isfile(VCMI_BINARY):
        print(f"Error: VCMI client not found at {VCMI_BINARY}")
        return None

    try:
        print("Launching VCMI...")
        vcmi_process = subprocess.Popen([VCMI_BINARY])
        return vcmi_process
    except Exception as e:
        print(f"Failed to open VCMI process: {e}")
        return None

def control_vcmi_ui():
    """
    Automates UI keypresses to load a scenario in VCMI.
    Assumes VCMI is already running.
    """
    print("Waiting to skip intro video...")
    time.sleep(3)

    try:
        sp.run(["osascript", "-e", 'tell application "System Events" to key code 36'])  # Return
        print("Sent Return key to skip intro video.")
    except Exception as e:
        print(f"Failed to send Return key: {e}")

    print("Waiting for main menu...")
    time.sleep(5)

    try:
        sp.run(["osascript", "-e", 'tell application "System Events" to keystroke "l"'])
        print("Sent 'l' to open Load Game menu.")
    except Exception as e:
        print(f"Failed to send 'l' key: {e}")

    time.sleep(3)

    try:
        sp.run(["osascript", "-e", 'tell application "System Events" to keystroke "s"'])
        print("Sent 's' to select single scenario.")
    except Exception as e:
        print(f"Failed to send 's' key: {e}")

    time.sleep(2)

    try:
        sp.run(["osascript", "-e", 'tell application "System Events" to key code 36'])  # Return
        print("Pressed Return to confirm selection.")
    except Exception as e:
        print(f"Failed to send Return key: {e}")

    time.sleep(5)

    try:
        sp.run(["osascript", "-e", 'tell application "System Events" to key code 124'])  # Right Arrow
        print("Pressed Right Arrow to move hero.")
    except Exception as e:
        print(f"Failed to send Right Arrow key: {e}")
    
    time.sleep(5)

def close_vcmi_process():
    """
    Closes the VCMI client process.
    """
    time.sleep(3)
    print("Closing VCMI process...")
    global vcmi_process
    if vcmi_process:
        vcmi_process.kill()
        vcmi_process.wait()
        print("VCMI process closed.")
