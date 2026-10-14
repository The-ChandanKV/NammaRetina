-- Run once in Supabase → SQL Editor after migrating data with explicit IDs.
-- Fixes "duplicate key value violates unique constraint ..._pkey" on insert.

SELECT setval(
  pg_get_serial_sequence('patients', 'patient_id'),
  COALESCE((SELECT MAX(patient_id) FROM patients), 1)
);

SELECT setval(
  pg_get_serial_sequence('reports', 'report_id'),
  COALESCE((SELECT MAX(report_id) FROM reports), 1)
);

SELECT setval(
  pg_get_serial_sequence('progression', 'progression_id'),
  COALESCE((SELECT MAX(progression_id) FROM progression), 1)
);
