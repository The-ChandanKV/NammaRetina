"""
Interactive tool to create a new patient and add their visit history.
Uses existing database.py, progression.py, and treatment.py functions.
Does NOT modify the database schema or existing implementations.
"""

from datetime import date
from pathlib import Path

from database import (
    add_patient,
    add_report,
    get_patient,
    get_reports_for_patient,
    DB_PATH,
)
from progression import track_progression, plot_progression
from treatment import get_treatment_recommendation


def get_valid_patient_name():
    """Prompt for and validate patient name."""
    while True:
        name = input("Name: ").strip()
        if not name:
            print("Error: Name cannot be empty.")
            continue
        return name


def get_valid_patient_age():
    """Prompt for and validate patient age."""
    while True:
        try:
            age_input = input("Age: ").strip()
            if not age_input:
                print("Error: Age cannot be empty.")
                continue
            
            age = int(age_input)
            if age < 0 or age > 150:
                print("Error: Please enter a valid age (0-150).")
                continue
            
            return age
        except ValueError:
            print("Error: Age must be a number.")


def get_valid_patient_gender():
    """Prompt for and validate patient gender."""
    while True:
        gender = input("Gender: ").strip()
        if not gender:
            print("Error: Gender cannot be empty.")
            continue
        return gender


def get_valid_diabetes_history():
    """Prompt for and validate diabetes history."""
    while True:
        history = input("Diabetes history: ").strip()
        if not history:
            print("Error: Diabetes history cannot be empty.")
            continue
        return history


def create_new_patient():
    """Prompt for new patient information and create patient record."""
    print("\n" + "=" * 80)
    print("CREATE NEW PATIENT")
    print("=" * 80)
    print("\nEnter patient information:")
    print("-" * 80)
    
    name = get_valid_patient_name()
    age = get_valid_patient_age()
    gender = get_valid_patient_gender()
    diabetes_history = get_valid_diabetes_history()
    
    print("\n" + "-" * 80)
    print("Creating patient...")
    
    try:
        patient_id = add_patient(
            name=name,
            age=age,
            gender=gender,
            diabetes_history=diabetes_history,
            db_path=DB_PATH
        )
        print("Patient created successfully!")
        print("Patient ID: {}".format(patient_id))
        
        patient = get_patient(patient_id, db_path=DB_PATH)
        return patient_id, patient
    except Exception as e:
        print("Error creating patient: {}".format(e))
        return None, None


def get_valid_severity():
    """Prompt for and validate severity (0-4)."""
    while True:
        try:
            severity_input = input("  Severity (0-4): ").strip()
            if not severity_input:
                print("  Error: Severity cannot be empty.")
                continue
            
            severity = int(severity_input)
            if severity < 0 or severity > 4:
                print("  Error: Severity must be between 0 and 4.")
                continue
            
            return severity
        except ValueError:
            print("  Error: Severity must be a number.")


def get_optional_float(prompt):
    """Prompt for an optional float value."""
    while True:
        try:
            value_input = input("  {}: ".format(prompt)).strip()
            if not value_input:
                return None
            
            value = float(value_input)
            if value < 0.0 or value > 1.0:
                print("  Error: {} must be between 0.0 and 1.0.".format(prompt))
                continue
            
            return value
        except ValueError:
            print("  Error: {} must be a number.".format(prompt))


def get_optional_string(prompt):
    """Prompt for an optional string value (can be None/empty)."""
    return input("  {}: ".format(prompt)).strip() or None


def get_optional_date():
    """Prompt for an optional date (YYYY-MM-DD format)."""
    while True:
        date_input = input("  Date (YYYY-MM-DD, or leave blank for today): ").strip()
        if not date_input:
            return date.today().isoformat()
        
        try:
            # Validate date format
            date.fromisoformat(date_input)
            return date_input
        except ValueError:
            print("  Error: Date must be in YYYY-MM-DD format.")


def get_number_of_visits():
    """Prompt for and validate the number of visits to add."""
    while True:
        try:
            visits_input = input("\nHow many visits/scans do you want to add? ").strip()
            if not visits_input:
                print("Error: Number of visits cannot be empty.")
                continue
            
            num_visits = int(visits_input)
            if num_visits < 0:
                print("Error: Number of visits cannot be negative.")
                continue
            
            if num_visits == 0:
                print("No visits will be added.")
            
            return num_visits
        except ValueError:
            print("Error: Number of visits must be a number.")


def confirm_visit(visit_data):
    """Ask user to confirm visit data before adding."""
    print("\n" + "-" * 60)
    print("Visit Details:")
    print("  Severity: {}".format(visit_data["severity"]))
    print("  Confidence: {}".format(visit_data["confidence"] if visit_data["confidence"] is not None else "(not provided)"))
    print("  Image Path: {}".format(visit_data["image_path"] if visit_data["image_path"] else "(not provided)"))
    print("  Heatmap Path: {}".format(visit_data["heatmap_path"] if visit_data["heatmap_path"] else "(not provided)"))
    print("  Date: {}".format(visit_data["date"]))
    print("-" * 60)
    
    while True:
        confirm = input("Add this visit? (y/n): ").strip().lower()
        if confirm in ['y', 'yes']:
            return True
        elif confirm in ['n', 'no']:
            return False
        else:
            print("Error: Please enter 'y' or 'n'.")


def main():
    print("=" * 80)
    print("CREATE PATIENT AND ADD VISIT HISTORY")
    print("=" * 80)
    
    # Step 1: Create new patient
    patient_id, patient = create_new_patient()
    if patient_id is None:
        return
    
    patient_name = patient["name"]
    patient_age = patient["age"]
    patient_gender = patient["gender"]
    patient_diabetes_history = patient["diabetes_history"]
    
    # Step 2: Get number of visits
    print("\nStep 2: Add Visit History")
    print("-" * 80)
    num_visits = get_number_of_visits()
    
    if num_visits == 0:
        print("\nNo visits added. Patient created successfully.")
        return
    
    # Step 3: Collect visit information and add reports
    print("\nStep 3: Enter Visit Information")
    print("-" * 80)
    
    visits_added = []
    
    for visit_num in range(1, num_visits + 1):
        print("\nVisit {} of {}:".format(visit_num, num_visits))
        print("-" * 60)
        
        # Collect visit data
        severity = get_valid_severity()
        confidence = get_optional_float("Confidence (0.0-1.0)")
        image_path = get_optional_string("Image path (optional)")
        heatmap_path = get_optional_string("Heatmap path (optional)")
        visit_date = get_optional_date()
        
        visit_data = {
            "severity": severity,
            "confidence": confidence,
            "image_path": image_path,
            "heatmap_path": heatmap_path,
            "date": visit_date,
        }
        
        # Confirm before adding
        if not confirm_visit(visit_data):
            print("Visit skipped.")
            continue
        
        # Add report to database
        try:
            report_id = add_report(
                patient_id=patient_id,
                image_path=image_path if image_path else "reports/visit_{}.png".format(visit_num),
                severity=severity,
                confidence=confidence if confidence is not None else 0.5,
                heatmap_path=heatmap_path if heatmap_path else "heatmaps/visit_{}_gradcam.png".format(visit_num),
                recommendation="To be determined",
                llm_summary="Manual entry",
                date=visit_date,
                db_path=DB_PATH
            )
            print("Report added with ID: {}".format(report_id))
            
            # Track progression using existing progression.py function.
            # Stamp the progression record with this visit's date (not "today"),
            # so the timeline matches the report we just added.
            progression_result = track_progression(
                patient_id, severity, visit_date=visit_date, db_path=DB_PATH
            )
            
            visits_added.append({
                "visit_num": visit_num,
                "severity": severity,
                "status": progression_result["status"],
                "date": visit_date,
            })
            
        except Exception as e:
            print("Error adding report: {}".format(e))
            continue
    
    if not visits_added:
        print("\nNo visits were added.")
        return
    
    # Step 4: Display summary
    print("\n" + "=" * 80)
    print("PATIENT SUMMARY")
    print("=" * 80)
    
    print("\nPatient ID: {}".format(patient_id))
    print("Name: {}".format(patient_name))
    print("Age: {}".format(patient_age))
    print("Gender: {}".format(patient_gender))
    print("Diabetes History: {}".format(patient_diabetes_history))
    
    print("\nVisits:")
    print("-" * 80)
    for visit in visits_added:
        print("Visit {}: Severity {} - Status: {}".format(
            visit["visit_num"],
            visit["severity"],
            visit["status"]
        ))
    
    # Get current severity and treatment recommendation
    all_reports = get_reports_for_patient(patient_id, db_path=DB_PATH)
    if all_reports:
        current_severity = int(all_reports[-1]["severity"])
        treatment_rec = get_treatment_recommendation(current_severity)
        
        print("\nCurrent Status:")
        print("-" * 80)
        print("Current Severity: {}".format(current_severity))
        print("Current Progression: {}".format(visits_added[-1]["status"]))
        print("Treatment Recommendation: {}".format(treatment_rec))
    
    # Generate progression graph
    print("\nGenerating progression graph...")
    try:
        graph_path = plot_progression(patient_id)
        print("Progression Graph: {}".format(graph_path))
    except Exception as e:
        print("Warning: Could not generate progression graph: {}".format(e))
    
    print("\n" + "=" * 80)
    print("Patient and visit history have been successfully created.")
    print("=" * 80)


if __name__ == "__main__":
    main()
