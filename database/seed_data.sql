-- ============================================================
-- AI HOSPITAL OPERATING SYSTEM - SEED DATA
-- Synthetic but medically plausible data
-- ============================================================

-- ------------------------------------------------------------
-- PATIENTS  (20 synthetic patients)
-- ------------------------------------------------------------
INSERT INTO patients (mrn, first_name, last_name, date_of_birth, gender, race, ethnicity, insurance_id) VALUES
('MRN000001','James',    'Harrison',  '1948-03-12','M','White',              'Non-Hispanic','INS-AA-001'),
('MRN000002','Maria',    'Gonzalez',  '1965-07-22','F','Hispanic',           'Hispanic',    'INS-BB-002'),
('MRN000003','David',    'Kim',       '1972-11-05','M','Asian',              'Non-Hispanic','INS-CC-003'),
('MRN000004','Sarah',    'Thompson',  '1989-02-14','F','White',              'Non-Hispanic','INS-DD-004'),
('MRN000005','Robert',   'Williams',  '1955-09-30','M','Black or African Am','Non-Hispanic','INS-EE-005'),
('MRN000006','Patricia', 'Johnson',   '1942-06-18','F','White',              'Non-Hispanic','INS-FF-006'),
('MRN000007','Michael',  'Brown',     '1980-04-25','M','Black or African Am','Non-Hispanic','INS-GG-007'),
('MRN000008','Linda',    'Davis',     '1958-12-02','F','White',              'Non-Hispanic','INS-HH-008'),
('MRN000009','William',  'Martinez',  '1977-08-16','M','Hispanic',           'Hispanic',    'INS-II-009'),
('MRN000010','Barbara',  'Anderson',  '1961-01-28','F','White',              'Non-Hispanic','INS-JJ-010'),
('MRN000011','Charles',  'Taylor',    '1945-05-07','M','White',              'Non-Hispanic','INS-KK-011'),
('MRN000012','Susan',    'Jackson',   '1983-10-19','F','Asian',              'Non-Hispanic','INS-LL-012'),
('MRN000013','Joseph',   'White',     '1969-03-31','M','White',              'Non-Hispanic','INS-MM-013'),
('MRN000014','Jessica',  'Harris',    '1991-07-04','F','Black or African Am','Non-Hispanic','INS-NN-014'),
('MRN000015','Thomas',   'Clark',     '1953-11-22','M','White',              'Non-Hispanic','INS-OO-015'),
('MRN000016','Karen',    'Lewis',     '1967-09-13','F','White',              'Non-Hispanic','INS-PP-016'),
('MRN000017','Christopher','Robinson','1975-02-27','M','Black or African Am','Non-Hispanic','INS-QQ-017'),
('MRN000018','Nancy',    'Walker',    '1985-06-08','F','White',              'Non-Hispanic','INS-RR-018'),
('MRN000019','Daniel',   'Hall',      '1950-04-15','M','White',              'Non-Hispanic','INS-SS-019'),
('MRN000020','Lisa',     'Young',     '1993-12-20','F','Hispanic',           'Hispanic',    'INS-TT-020');

-- ------------------------------------------------------------
-- ENCOUNTERS
-- ------------------------------------------------------------
INSERT INTO encounters (patient_id, encounter_type, admit_time, discharge_time, chief_complaint, attending_md)
SELECT patient_id, 'icu', NOW() - INTERVAL '5 days', NULL,
       CASE mrn
           WHEN 'MRN000001' THEN 'Respiratory failure, hypoxia'
           WHEN 'MRN000002' THEN 'Septic shock, hypotension'
           WHEN 'MRN000003' THEN 'Acute MI, chest pain'
           WHEN 'MRN000004' THEN 'Post-surgical monitoring'
           WHEN 'MRN000005' THEN 'Hypertensive crisis'
           WHEN 'MRN000006' THEN 'COPD exacerbation'
           WHEN 'MRN000007' THEN 'Diabetic ketoacidosis'
           WHEN 'MRN000008' THEN 'Acute kidney injury'
           ELSE 'General monitoring'
       END,
       'Dr. Chen'
FROM patients WHERE mrn IN (
  'MRN000001','MRN000002','MRN000003','MRN000004',
  'MRN000005','MRN000006','MRN000007','MRN000008'
);

-- Discharged encounters
INSERT INTO encounters (patient_id, encounter_type, admit_time, discharge_time, chief_complaint, attending_md)
SELECT patient_id, 'inpatient',
       NOW() - INTERVAL '30 days',
       NOW() - INTERVAL '25 days',
       'Post-operative care', 'Dr. Patel'
FROM patients WHERE mrn IN ('MRN000009','MRN000010','MRN000011','MRN000012');

-- ------------------------------------------------------------
-- ADMISSIONS
-- ------------------------------------------------------------
INSERT INTO admissions (encounter_id, patient_id, admittime, admission_type, admission_location, insurance, hospital_expire_flag)
SELECT e.encounter_id, e.patient_id, e.admit_time, 'EMERGENCY', 'EMERGENCY ROOM ADMIT', 'Private', FALSE
FROM encounters e;

-- ------------------------------------------------------------
-- ICU STAYS  (for active ICU patients)
-- ------------------------------------------------------------
INSERT INTO icustays (admission_id, patient_id, icu_unit, intime, los, first_careunit, last_careunit)
SELECT a.admission_id, a.patient_id,
       CASE MOD(ROW_NUMBER() OVER (ORDER BY a.patient_id)::INT, 5)
           WHEN 0 THEN 'MICU' WHEN 1 THEN 'SICU'
           WHEN 2 THEN 'CSRU' WHEN 3 THEN 'CCU'
           ELSE 'TSICU'
       END,
       a.admittime,
       EXTRACT(EPOCH FROM (NOW() - a.admittime)) / 86400.0,
       'MICU', 'MICU'
FROM admissions a
JOIN encounters e ON a.encounter_id = e.encounter_id
WHERE e.discharge_time IS NULL;

-- ------------------------------------------------------------
-- VITAL SIGNS  (72 hours of hourly readings per ICU patient)
-- Realistic physiological ranges with simulated deterioration
-- ------------------------------------------------------------
INSERT INTO vital_signs (patient_id, icustay_id, charttime, heart_rate, sbp, dbp, map, temperature, spo2, resp_rate, gcs_total)
SELECT
    i.patient_id,
    i.icustay_id,
    i.intime + (n.n * INTERVAL '1 hour') AS charttime,
    -- Heart rate: base 75 +/- variability, deterioration modeled
    ROUND((
        CASE p.mrn
            WHEN 'MRN000002' THEN 110 + (n.n * 0.3) + (RANDOM() * 15 - 7)  -- sepsis: tachycardia rising
            WHEN 'MRN000003' THEN 88  + (RANDOM() * 20 - 10)                 -- MI: variable
            ELSE 72 + (RANDOM() * 16 - 8)
        END
    )::numeric, 1) AS heart_rate,
    -- Systolic BP
    ROUND((
        CASE p.mrn
            WHEN 'MRN000002' THEN 85  - (n.n * 0.2) + (RANDOM() * 10 - 5)  -- sepsis: dropping
            WHEN 'MRN000005' THEN 170 + (RANDOM() * 20 - 10)                 -- HTN crisis
            ELSE 118 + (RANDOM() * 20 - 10)
        END
    )::numeric, 1) AS sbp,
    -- Diastolic BP
    ROUND((
        CASE p.mrn
            WHEN 'MRN000002' THEN 52  - (n.n * 0.1) + (RANDOM() * 8 - 4)
            WHEN 'MRN000005' THEN 105 + (RANDOM() * 12 - 6)
            ELSE 74 + (RANDOM() * 12 - 6)
        END
    )::numeric, 1) AS dbp,
    -- MAP = (SBP + 2*DBP) / 3
    ROUND((
        CASE p.mrn
            WHEN 'MRN000002' THEN 63  - (n.n * 0.15)
            WHEN 'MRN000005' THEN 127
            ELSE 89 + (RANDOM() * 10 - 5)
        END
    )::numeric, 1) AS map,
    -- Temperature
    ROUND((
        CASE p.mrn
            WHEN 'MRN000002' THEN 38.8 + (RANDOM() * 0.6)                    -- fever in sepsis
            WHEN 'MRN000006' THEN 36.2 + (RANDOM() * 0.4)                    -- COPD: normal-low
            ELSE 36.8 + (RANDOM() * 0.6 - 0.3)
        END
    )::numeric, 2) AS temperature,
    -- SpO2
    ROUND((
        CASE p.mrn
            WHEN 'MRN000001' THEN 88  - (n.n * 0.05) + (RANDOM() * 4 - 2)  -- resp failure: declining
            WHEN 'MRN000006' THEN 91  + (RANDOM() * 3)                       -- COPD: borderline
            ELSE 97 + (RANDOM() * 2 - 1)
        END
    )::numeric, 1) AS spo2,
    -- Respiratory rate
    ROUND((
        CASE p.mrn
            WHEN 'MRN000001' THEN 24  + (n.n * 0.1) + (RANDOM() * 4 - 2)   -- tachypnea
            WHEN 'MRN000002' THEN 22  + (RANDOM() * 4)                       -- sepsis
            ELSE 16 + (RANDOM() * 4 - 2)
        END
    )::numeric, 0) AS resp_rate,
    -- GCS total
    CASE p.mrn
        WHEN 'MRN000002' THEN GREATEST(8, 15 - (n.n / 12))::INT             -- deteriorating consciousness
        ELSE 15
    END AS gcs_total
FROM icustays i
JOIN patients p ON i.patient_id = p.patient_id
CROSS JOIN GENERATE_SERIES(0, 71) AS n(n)
WHERE p.mrn IN ('MRN000001','MRN000002','MRN000003','MRN000004',
                'MRN000005','MRN000006','MRN000007','MRN000008');

-- ------------------------------------------------------------
-- LAB RESULTS
-- Creatinine, WBC, Lactate, Glucose, Hemoglobin, Platelets
-- ------------------------------------------------------------
INSERT INTO lab_results (patient_id, icustay_id, charttime, lab_name, value, value_uom, ref_range_lower, ref_range_upper, flag)
SELECT
    i.patient_id, i.icustay_id,
    i.intime + (n.n * INTERVAL '6 hours'),
    lab.lab_name,
    ROUND((lab.base_val + lab.noise * (RANDOM() - 0.5))::numeric, 2),
    lab.uom, lab.low, lab.high,
    CASE
        WHEN (lab.base_val + lab.noise * (RANDOM() - 0.5)) < lab.crit_low  THEN 'critical_low'
        WHEN (lab.base_val + lab.noise * (RANDOM() - 0.5)) > lab.crit_high THEN 'critical_high'
        WHEN (lab.base_val + lab.noise * (RANDOM() - 0.5)) < lab.low       THEN 'low'
        WHEN (lab.base_val + lab.noise * (RANDOM() - 0.5)) > lab.high      THEN 'high'
        ELSE 'normal'
    END AS flag
FROM icustays i
CROSS JOIN GENERATE_SERIES(0, 11) AS n(n)
CROSS JOIN (VALUES
    ('Creatinine',  1.4, 0.6, 0.6, 1.2, 0.3, 10.0, 'mg/dL'),
    ('WBC',        11.0, 4.0, 4.5,11.0, 1.0, 30.0, 'K/uL'),
    ('Lactate',     2.8, 1.5, 0.5, 2.0, 0.1,  8.0, 'mmol/L'),
    ('Glucose',   145.0,30.0,70.0,100.0,40.0,500.0, 'mg/dL'),
    ('Hemoglobin',  9.5, 1.5,12.0,17.5, 6.0, 20.0, 'g/dL'),
    ('Platelets', 165.0,50.0,150.0,400.0,50.0,1000.0,'K/uL'),
    ('Sodium',    138.0, 5.0,136.0,145.0,120.0,160.0,'mEq/L'),
    ('Potassium',   4.2, 0.8, 3.5, 5.0, 2.5,  7.0, 'mEq/L'),
    ('BUN',        28.0,10.0, 7.0,25.0, 2.0,100.0, 'mg/dL'),
    ('Troponin',    0.6, 0.3, 0.0, 0.04,0.0, 10.0, 'ng/mL')
) AS lab(lab_name, base_val, noise, low, high, crit_low, crit_high, uom)
JOIN patients p ON i.patient_id = p.patient_id
WHERE p.mrn IN ('MRN000001','MRN000002','MRN000003','MRN000004',
                'MRN000005','MRN000006','MRN000007','MRN000008');

-- ------------------------------------------------------------
-- MEDICATIONS
-- ------------------------------------------------------------
INSERT INTO medications (patient_id, encounter_id, drug_name, drug_class, dose_val, dose_uom, route, starttime, prescriber)
SELECT
    p.patient_id, e.encounter_id,
    med.drug_name, med.drug_class, med.dose_val, med.dose_uom, med.route,
    e.admit_time, 'Dr. Chen'
FROM patients p
JOIN encounters e ON p.patient_id = e.patient_id
CROSS JOIN (VALUES
    ('Norepinephrine',   'Vasopressor',      0.1,  'mcg/kg/min', 'IV'),
    ('Vancomycin',       'Antibiotic',      1500,  'mg',          'IV'),
    ('Piperacillin-Tazo','Antibiotic',      3375,  'mg',          'IV'),
    ('Heparin',          'Anticoagulant',   5000,  'units',       'SubQ'),
    ('Pantoprazole',     'PPI',               40,  'mg',          'IV'),
    ('Insulin',          'Antidiabetic',      10,  'units',       'SubQ'),
    ('Metoprolol',       'Beta Blocker',      25,  'mg',          'PO'),
    ('Furosemide',       'Diuretic',          40,  'mg',          'IV')
) AS med(drug_name, drug_class, dose_val, dose_uom, route)
WHERE p.mrn IN ('MRN000001','MRN000002','MRN000003','MRN000004');

-- ------------------------------------------------------------
-- DIAGNOSES
-- ------------------------------------------------------------
INSERT INTO diagnoses (patient_id, encounter_id, icd_code, icd_version, description, diag_priority)
SELECT p.patient_id, e.encounter_id, diag.icd_code, 10, diag.description, diag.priority
FROM patients p
JOIN encounters e ON p.patient_id = e.patient_id
CROSS JOIN (VALUES
    ('A41.9',  'Sepsis, unspecified organism',           1),
    ('J96.01', 'Acute respiratory failure with hypoxia', 1),
    ('I21.9',  'Acute myocardial infarction, unspecified',1),
    ('N17.9',  'Acute kidney failure, unspecified',      2),
    ('E11.65', 'Type 2 DM with hyperglycemia',           2),
    ('I10',    'Essential (primary) hypertension',       3),
    ('J44.1',  'COPD with acute exacerbation',           2),
    ('E87.1',  'Hypo-osmolality and hyponatremia',       3)
) AS diag(icd_code, description, priority)
WHERE p.mrn IN ('MRN000001','MRN000002','MRN000003','MRN000004',
                'MRN000005','MRN000006','MRN000007','MRN000008');

-- ------------------------------------------------------------
-- CLINICAL NOTES
-- ------------------------------------------------------------
INSERT INTO clinical_notes (patient_id, encounter_id, charttime, category, description, text, cgid)
SELECT
    p.patient_id, e.encounter_id,
    NOW() - INTERVAL '2 hours',
    notes.category, notes.description, notes.text, 'RN-001'
FROM patients p
JOIN encounters e ON p.patient_id = e.patient_id
CROSS JOIN (VALUES
  ('Physician','Admission note',
   'Patient is a 75-year-old male presenting with acute respiratory failure. History of COPD and CHF. On arrival: SpO2 82% on room air, RR 28, HR 108, BP 88/52. Started on non-invasive positive pressure ventilation (NIPPV). ABG showing pH 7.28, pCO2 62, pO2 55. Likely acute-on-chronic respiratory failure. Initiated broad-spectrum antibiotics for possible CAP. ICU admission warranted.'),
  ('Nursing','Nursing assessment',
   'Patient responsive to verbal stimuli. GCS 12 (E3V4M5). Appears in moderate distress. On supplemental O2 at 6L/min via nasal cannula. Bilateral crackles auscultated at lung bases. Peripheral edema 2+ bilateral lower extremities. IV access x2 established. Foley catheter placed, urine output 20 mL/hr past 2 hours - concerning for oliguria. Patient denies chest pain. Reports shortness of breath at rest.'),
  ('Physician','Progress note',
   'ICU Day 2. Hemodynamically unstable overnight. Norepinephrine 0.15 mcg/kg/min. MAP trending down to 58 mmHg. Lactate 4.2 mmol/L - septic physiology suspected. Blood cultures x2 sent. Chest X-ray: bilateral infiltrates consistent with ARDS. Procalcitonin elevated at 18 ng/mL. Decision made to intubate for airway protection. RSI performed without complications. PEEP 8, FiO2 0.6.')
) AS notes(category, description, text)
WHERE p.mrn = 'MRN000002';

-- Additional notes for patient MRN000001
INSERT INTO clinical_notes (patient_id, encounter_id, charttime, category, description, text, cgid)
SELECT p.patient_id, e.encounter_id, NOW() - INTERVAL '4 hours',
       'Radiology', 'Chest X-ray report',
       'CHEST X-RAY PORTABLE AP: Endotracheal tube in appropriate position. Bilateral patchy opacities consistent with multifocal pneumonia vs ARDS. Small bilateral pleural effusions. Cardiomegaly. No pneumothorax. Recommend CT chest if clinically stable.',
       'RAD-001'
FROM patients p JOIN encounters e ON p.patient_id = e.patient_id
WHERE p.mrn = 'MRN000001';
