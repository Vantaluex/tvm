import subprocess
import atexit

def lock_clocks_windows(freq=2617):
    print(f"\n[INFO] Attempting to lock Windows GPU clocks to {freq} MHz...")
    print("[INFO] Look for a Windows Admin (UAC) prompt on your taskbar...")
    
    # This tells WSL to ask Windows PowerShell to run nvidia-smi.exe as Administrator
    cmd = f"Start-Process nvidia-smi.exe -ArgumentList '-i 0 -lgc {freq},{freq}' -Verb RunAs -Wait"
    try:
        subprocess.run(["powershell.exe", "-Command", cmd], check=True)
        print("[INFO] GPU clocks locked successfully.")
    except subprocess.CalledProcessError as e:
        print(f"[WARN] Failed to lock clocks. Ensure you accepted the UAC prompt. Error: {e}")

def unlock_clocks_windows():
    print(f"\n[INFO] Attempting to unlock Windows GPU clocks...")
    cmd = "Start-Process nvidia-smi.exe -ArgumentList '-i 0 -rgc' -Verb RunAs -Wait"
    try:
        subprocess.run(["powershell.exe", "-Command", cmd], check=True)
        print("[INFO] GPU clocks unlocked successfully.")
    except subprocess.CalledProcessError as e:
        print(f"[WARN] Failed to unlock clocks. Error: {e}")

lock_clocks_windows(freq=2617)

# Register the unlock function to run automatically when the script exits
# This ensures clocks reset even if the script crashes or you hit Ctrl+C
atexit.register(unlock_clocks_windows)
