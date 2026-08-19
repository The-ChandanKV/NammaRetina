"""
Test runner for add_patient_history.py - demonstrates creating a new patient with visit history
"""

import subprocess
import sys

# Sample input: Create a new patient and add 2 visits
# Visit 1: Severity 1, Visit 2: Severity 2 (shows Worsened progression)
sample_input = """Alice Johnson
52
Female
Type 2 diabetes for 8 years
2
1
0.85

reports/alice_visit1.png

2026-06-01
y
2
0.90

reports/alice_visit2.png

2026-07-01
y
"""

# Run the add_patient_history.py script with the sample input
process = subprocess.Popen(
    [sys.executable, "add_patient_history.py"],
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
