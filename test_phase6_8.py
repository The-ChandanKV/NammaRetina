from pathlib import Path

from database import add_patient, add_progression_record, add_report, get_progression_history, get_reports_for_patient, init_db
from progression import plot_progression, track_progression
from simulation import simulate_progression
from treatment import get_treatment_recommendation


def run_phase6_8_tests():
    print("=== Phase 6-8 Test ===\n")

    db_path = Path("database.db")
    if db_path.exists():
        db_path.unlink()

    print("1. Initializing database...")
    init_db()

    # Test Patient 1: Single patient with 3 visits showing progression
    print("\n2. Testing Patient 1 with progression tracking...")
    patient_1_id = add_patient("Test Patient 1", 58, "Male", "Type 2 diabetes for 10 years")
    print(f"Patient 1 ID: {patient_1_id}")

    patient_1_visits = [
        {"severity": 1, "date": "2026-01-01", "image_path": "reports/visit1.png"},
        {"severity": 1, "date": "2026-02-01", "image_path": "reports/visit2.png"},
        {"severity": 2, "date": "2026-03-01", "image_path": "reports/visit3.png"},
    ]

    patient_1_results = []
    for visit in patient_1_visits:
        report_id = add_report(
            patient_id=patient_1_id,
            image_path=visit["image_path"],
            severity=visit["severity"],
            confidence=0.9,
            heatmap_path="heatmaps/sample_heatmap.png",
            recommendation="Follow up",
            llm_summary="placeholder",
            date=visit["date"],
        )

        result = track_progression(patient_1_id, visit["severity"])
        patient_1_results.append(result)

        # Derive visit number from patient's ordered report history
        reports = get_reports_for_patient(patient_1_id)
        visit_number = len(reports)
        
        print(f"Patient {patient_1_id} - Visit {visit_number} - Severity {visit['severity']} - Status: {result['status']}")

    expected_statuses = ["Initial", "Stable", "Worsened"]
    actual_statuses = [result["status"] for result in patient_1_results]
    assert actual_statuses == expected_statuses, f"Unexpected statuses for Patient 1: {actual_statuses}"
    print("✓ Patient 1 progression tracking correct")

    # Test Patient 2: Verify patient histories are isolated
    print("\n3. Testing Patient 2 with independent progression tracking...")
    patient_2_id = add_patient("Test Patient 2", 42, "Female", "Type 1 diabetes for 5 years")
    print(f"Patient 2 ID: {patient_2_id}")

    patient_2_visits = [
        {"severity": 1, "date": "2026-01-15", "image_path": "reports/p2_visit1.png"},
        {"severity": 3, "date": "2026-02-15", "image_path": "reports/p2_visit2.png"},
    ]

    patient_2_results = []
    for visit in patient_2_visits:
        report_id = add_report(
            patient_id=patient_2_id,
            image_path=visit["image_path"],
            severity=visit["severity"],
            confidence=0.85,
            heatmap_path="heatmaps/sample_heatmap.png",
            recommendation="Follow up",
            llm_summary="placeholder",
            date=visit["date"],
        )

        result = track_progression(patient_2_id, visit["severity"])
        patient_2_results.append(result)

        # Derive visit number from patient's ordered report history
        reports = get_reports_for_patient(patient_2_id)
        visit_number = len(reports)
        
        print(f"Patient {patient_2_id} - Visit {visit_number} - Severity {visit['severity']} - Status: {result['status']}")

    # Patient 2 should show: Initial (no previous), then Worsened (1 -> 3)
    expected_statuses_p2 = ["Initial", "Worsened"]
    actual_statuses_p2 = [result["status"] for result in patient_2_results]
    assert actual_statuses_p2 == expected_statuses_p2, f"Unexpected statuses for Patient 2: {actual_statuses_p2}"
    print("✓ Patient 2 progression tracking correct")
    print("✓ Patient histories are properly isolated")

    # Verify Patient 1 is unchanged
    print("\n4. Verifying Patient 1 history not affected by Patient 2...")
    patient_1_reports = get_reports_for_patient(patient_1_id)
    assert len(patient_1_reports) == 3, f"Patient 1 should have 3 reports, has {len(patient_1_reports)}"
    print(f"✓ Patient 1 has {len(patient_1_reports)} reports (unaffected by Patient 2)")

    patient_2_reports = get_reports_for_patient(patient_2_id)
    assert len(patient_2_reports) == 2, f"Patient 2 should have 2 reports, has {len(patient_2_reports)}"
    print(f"✓ Patient 2 has {len(patient_2_reports)} reports")

    print("\n5. Generating progression graphs...")
    print("Generating progression graph for Patient 1...")
    graph_path_p1 = plot_progression(patient_1_id)
    print(f"Graph saved to: {graph_path_p1}")
    assert Path(graph_path_p1).exists(), "Progression graph for Patient 1 was not created."

    print("Generating progression graph for Patient 2...")
    graph_path_p2 = plot_progression(patient_2_id)
    print(f"Graph saved to: {graph_path_p2}")
    assert Path(graph_path_p2).exists(), "Progression graph for Patient 2 was not created."
    
    # Verify graphs are separate files
    assert graph_path_p1 != graph_path_p2, "Progression graphs should be separate files"
    print("✓ Both progression graphs generated successfully")

    print("\n6. Testing treatment recommendations...")
    print("6a. Testing all severity level mappings:")
    for severity in range(5):
        rec = get_treatment_recommendation(severity)
        print(f"  Severity {severity}: {rec}")

    print("\n6b. Testing patient-specific treatment recommendations:")
    # Get most recent severity for each patient
    patient_1_reports = get_reports_for_patient(patient_1_id)
    patient_1_current_severity = int(patient_1_reports[-1]["severity"])
    patient_1_recommendation = get_treatment_recommendation(patient_1_current_severity)
    print(f"Patient {patient_1_id}: Severity {patient_1_current_severity} -> {patient_1_recommendation}")
    assert patient_1_current_severity == 2, "Patient 1 current severity should be 2"
    assert patient_1_recommendation == "Ophthalmologist consultation advised", f"Unexpected recommendation for Patient 1: {patient_1_recommendation}"

    patient_2_reports = get_reports_for_patient(patient_2_id)
    patient_2_current_severity = int(patient_2_reports[-1]["severity"])
    patient_2_recommendation = get_treatment_recommendation(patient_2_current_severity)
    print(f"Patient {patient_2_id}: Severity {patient_2_current_severity} -> {patient_2_recommendation}")
    assert patient_2_current_severity == 3, "Patient 2 current severity should be 3"
    assert patient_2_recommendation == "Immediate specialist consultation", f"Unexpected recommendation for Patient 2: {patient_2_recommendation}"
    print("✓ Patient-specific treatment recommendations verified")

    print("\n7. Testing simulation if suitable sample images exist...")
    sample_image = Path("reports") / "sample_retina.png"
    sample_heatmap = Path("heatmaps") / "sample_heatmap.png"
    if sample_image.exists() and sample_heatmap.exists():
        output_path = simulate_progression(
            str(sample_image),
            str(sample_heatmap),
            get_progression_history(patient_1_id),
            random_seed=7,
        )
        print(f"Simulation output: {output_path}")
        assert Path(output_path).exists(), "Simulation output was not created."
    else:
        print("Simulation skipped: no suitable retinal and Grad-CAM sample images detected.")

    print("\n=== All Phase 6-8 tests passed ===")


if __name__ == "__main__":
    run_phase6_8_tests()
