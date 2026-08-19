"""
Test runner for add_patient.py - provides sample input
"""

import subprocess
import sys

# Sample input for the add_patient.py script
sample_input = """John Smith
45
Male
Type 2 diabetes for 12 years
"""

# Run the add_patient.py script with the sample input
process = subprocess.Popen(
    [sys.executable, "add_patient.py"],
    stdin=subprocess.PIPE,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    text=True
)

stdout, stderr = process.communicate(input=sample_input)

print(stdout)
if stderr:
    print("STDERR:", stderr)

sys.exit(process.returncode)
