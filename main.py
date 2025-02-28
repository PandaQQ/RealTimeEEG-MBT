# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.

import subprocess
import os
import sys


def run_training():
    script_path = os.path.join(os.path.dirname(__file__), 'training', 'spa_export.py')
    subprocess.run([sys.executable, script_path], check=True)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    run_training()
