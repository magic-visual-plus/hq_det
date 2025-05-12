#!/usr/bin/env python3
import argparse
import subprocess
import time
import sys
import os
import signal

def monitor_process(pid, command):
    """
    Monitor a process with the given PID. Once the process ends,
    execute the specified command.
    
    Args:
        pid (int): Process ID to monitor
        command (str): Command to execute when the monitored process ends
    """
    print(f"Monitoring process with PID {pid}")
    
    try:
        while True:
            try:
                # Check if process exists by sending signal 0
                # This doesn't actually send a signal but checks if process exists
                os.kill(pid, 0)
                time.sleep(1)  # Check every second
            except OSError:
                # Process no longer exists
                print(f"Process {pid} has ended. Executing command: {command}")
                break
        
        # Execute the command
        subprocess.run(command, shell=True)
        print("Command executed successfully")
        
    except KeyboardInterrupt:
        print("Monitoring stopped by user")
        sys.exit(0)

def main():
    parser = argparse.ArgumentParser(description='Monitor a process and execute a command when it ends')
    parser.add_argument('pid', type=int, help='PID of the process to monitor')
    parser.add_argument('command', type=str, help='Command to execute when the process ends')
    
    args = parser.parse_args()
    
    monitor_process(args.pid, args.command)

if __name__ == "__main__":
    main()
