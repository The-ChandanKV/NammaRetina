"""
Interactive script to add a new patient to the NammaRetina database.
Uses existing database.py helper functions.
Does not modify the database schema or existing records.
"""

import sqlite3
from pathlib import Path
from database import add_patient, DB_PATH, get_connection


def display_all_patients():
    """Query and display all patients in the Patients table."""
    conn = get_connection(DB_PATH)
    try:
        rows = conn.execute("SELECT * FROM Patients").fetchall()
        if not rows:
            print("\nNo patients in database.")
            return
        
        print("\n" + "=" * 80)
        print("ALL PATIENTS IN DATABASE")
        print("=" * 80)
        print("{:<12} {:<30} {:<6} {:<10} {:<22}".format(
            "Patient ID", "Name", "Age", "Gender", "Diabetes History"
        ))
        print("-" * 80)
        
        for row in rows:
            patient_id = row["patient_id"]
            name = row["name"]
            age = row["age"]
            gender = row["gender"]
            diabetes_history = row["diabetes_history"]
            
            # Truncate long fields for display
            name_display = name[:28] if name else "N/A"
            diabetes_display = diabetes_history[:20] if diabetes_history else "N/A"
            
            print("{:<12} {:<30} {:<6} {:<10} {:<22}".format(
                patient_id, name_display, age, gender, diabetes_display
            ))
        
        print("=" * 80)
    finally:
        conn.close()


def main():
    print("=" * 80)
    print("ADD NEW PATIENT TO NAMMA RETINA DATABASE")
    print("=" * 80)
    
    print("\nEnter patient information:")
    print("-" * 80)
    
    # Collect patient information
    name = input("Patient name: ").strip()
    if not name:
        print("Error: Patient name cannot be empty.")
        return
    
    while True:
        try:
            age = int(input("Age: ").strip())
            if age < 0 or age > 150:
                print("Error: Please enter a valid age (0-150).")
                continue
            break
        except ValueError:
            print("Error: Age must be a number.")
    
    gender = input("Gender: ").strip()
    if not gender:
        print("Error: Gender cannot be empty.")
        return
    
    diabetes_history = input("Diabetes history: ").strip()
    if not diabetes_history:
        print("Error: Diabetes history cannot be empty.")
        return
    
    # Insert patient into database
    print("\n" + "-" * 80)
    print("Inserting patient into database...")
    
    try:
        patient_id = add_patient(
            name=name,
            age=age,
            gender=gender,
            diabetes_history=diabetes_history,
            db_path=DB_PATH
        )
        print("[SUCCESS] Patient added with ID: {}".format(patient_id))
    except Exception as e:
        print("[ERROR] Failed to add patient: {}".format(e))
        return
    
    # Display all patients
    display_all_patients()
    
    print("\n[OK] New patient record has been successfully created and verified.")


if __name__ == "__main__":
    main()
