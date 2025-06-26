import os
import subprocess
import time
import subprocess as sp
from paths import VCMI_BINARY

# Socket command list for VCMI EntryPoint:
#
# move_active_hero_left
#     Moves the currently selected hero one tile to the west.
#
# move_active_hero_right
#     Moves the currently selected hero one tile to the east.
#
# open_load_menu
#     Opens the “Load Game” screen from the main menu.
#
# lobby_accept
#     Simulates pressing Enter on the Load Game screen, triggering the load of the currently selected save.
#

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
        vcmi_process = None
        print("VCMI process closed.")
    else:
        print("VCMI process was already terminated.")
