/*====================================================================
  AI HOSPITAL OPERATING SYSTEM - SAS DATA PIPELINE
  Legacy Integration Layer
  clean_data.sas
  ====================================================================
  Purpose : Extract, validate, clean, and export EHR data for
            downstream ML pipelines.
  Inputs  : Raw CSV exports from legacy EHR systems
  Outputs : Cleaned SAS datasets + CSV exports for Python pipeline
  ====================================================================*/

options mprint mlogic symbolgen;
options noxwait noxsync;

%let RAWDATA_PATH  = /data/raw;
%let CLEAN_PATH    = /data/clean;
%let EXPORT_PATH   = /data/export;
%let LOG_PATH      = /logs;

/* ─────────────────────────────────────────────────────────────────
   MACRO: LOG_MSG - Timestamped logging
   ───────────────────────────────────────────────────────────────── */
%macro log_msg(msg, level=INFO);
    %let _ts = %sysfunc(datetime(), datetime20.);
    %put [&level.] &_ts. - &msg.;
%mend log_msg;

/* ─────────────────────────────────────────────────────────────────
   STEP 1: LOAD RAW VITAL SIGNS
   ───────────────────────────────────────────────────────────────── */
%log_msg(Loading raw vital signs data);

proc import
    datafile = "&RAWDATA_PATH./vitals_raw.csv"
    out      = work.vitals_raw
    dbms     = csv
    replace;
    guessingrows = MAX;
    getnames = yes;
run;

%log_msg(Raw vital signs loaded: %sysfunc(attrn(work.vitals_raw, nobs)) observations);

/* ─────────────────────────────────────────────────────────────────
   STEP 2: VALIDATE AND CLEAN VITAL SIGNS
   Physiological plausibility checks
   ───────────────────────────────────────────────────────────────── */
data work.vitals_clean work.vitals_rejected;
    set work.vitals_raw;

    /* Standardize column names to lowercase */
    patient_id_c = strip(upcase(patient_id));
    
    /* Convert charttime to SAS datetime */
    if charttime ne '' then do;
        charttime_dt = input(charttime, anydtdtm40.);
        format charttime_dt datetime20.;
    end;

    /* ── Physiological range validation ── */
    flag_hr   = 0; flag_sbp  = 0; flag_dbp = 0;
    flag_temp = 0; flag_spo2 = 0; flag_rr  = 0;
    
    /* Heart Rate: 20-300 bpm (wider than normal to catch extreme but real values) */
    if heart_rate < 20 or heart_rate > 300 or heart_rate = . then do;
        flag_hr = 1;
        heart_rate = .;  /* Nullify implausible value */
    end;
    
    /* Systolic BP: 40-300 mmHg */
    if sbp < 40 or sbp > 300 or sbp = . then do;
        flag_sbp = 1;
        sbp = .;
    end;
    
    /* Diastolic BP: 10-200 mmHg */
    if dbp < 10 or dbp > 200 or dbp = . then do;
        flag_dbp = 1;
        dbp = .;
    end;
    
    /* Temperature: 25-44 Celsius */
    if temperature < 25 or temperature > 44 then do;
        /* Try F to C conversion if looks like Fahrenheit */
        if temperature >= 86 and temperature <= 110 then
            temperature = (temperature - 32) * 5/9;
        else do;
            flag_temp = 1;
            temperature = .;
        end;
    end;
    
    /* SpO2: 50-100% */
    if spo2 < 50 or spo2 > 100 or spo2 = . then do;
        flag_spo2 = 1;
        spo2 = .;
    end;
    
    /* Respiratory Rate: 1-80 breaths/min */
    if resp_rate < 1 or resp_rate > 80 or resp_rate = . then do;
        flag_rr = 1;
        resp_rate = .;
    end;
    
    /* Calculate MAP if missing: (SBP + 2*DBP) / 3 */
    if map = . and sbp ne . and dbp ne . then
        map = round((sbp + 2*dbp) / 3, 0.1);
    
    /* Count total flags per record */
    total_flags = flag_hr + flag_sbp + flag_dbp + flag_temp + flag_spo2 + flag_rr;
    
    /* Route: >3 flags -> rejected, else -> clean */
    if total_flags > 3 then output work.vitals_rejected;
    else output work.vitals_clean;

    /* Track source */
    data_source = 'legacy_ehr';
    processed_dt = datetime();
    format processed_dt datetime20.;
    
    drop patient_id charttime;
    rename patient_id_c = patient_id
           charttime_dt = charttime;

run;

proc sql;
    select 'vitals_clean'    as dataset, count(*) as nobs from work.vitals_clean
    union all
    select 'vitals_rejected' as dataset, count(*) as nobs from work.vitals_rejected;
quit;

/* ─────────────────────────────────────────────────────────────────
   STEP 3: LOAD AND CLEAN LAB RESULTS
   ───────────────────────────────────────────────────────────────── */
%log_msg(Loading raw lab results);

proc import
    datafile = "&RAWDATA_PATH./labs_raw.csv"
    out      = work.labs_raw
    dbms     = csv
    replace;
    guessingrows = MAX;
    getnames = yes;
run;

/* Define lab reference ranges */
data work.lab_ref;
    length lab_name $50 uom $20;
    input lab_name $ low_normal high_normal crit_low crit_high min_plaus max_plaus uom $;
    datalines;
Creatinine   0.6  1.2  0.2  10.0  0.1  20.0  mg/dL
WBC          4.5 11.0  1.0  30.0  0.1  100.0 K/uL
Hemoglobin  12.0 17.5  4.0  20.0  1.0  25.0  g/dL
Platelets  150.0 400.0 20.0 1000.0 5.0 2000.0 K/uL
Sodium     136.0 145.0 115.0 160.0 100.0 180.0 mEq/L
Potassium    3.5   5.0  2.0   7.0  1.0  10.0  mEq/L
Glucose     70.0 100.0 40.0 500.0  20.0 800.0 mg/dL
BUN          7.0  25.0  2.0 100.0   1.0 200.0 mg/dL
Creatinine   0.6   1.2  0.2  10.0  0.1  20.0 mg/dL
Lactate      0.5   2.0  0.1   8.0  0.0  15.0 mmol/L
Troponin     0.0   0.04 0.0  10.0  0.0  50.0 ng/mL
;
run;

/* Clean labs with flag calculation */
proc sql;
    create table work.labs_flagged as
    select l.*,
           r.low_normal, r.high_normal, r.crit_low, r.crit_high,
           r.min_plaus,  r.max_plaus,
           case
               when l.value < r.min_plaus or l.value > r.max_plaus then 'IMPLAUSIBLE'
               when l.value < r.crit_low                           then 'critical_low'
               when l.value > r.crit_high                          then 'critical_high'
               when l.value < r.low_normal                         then 'low'
               when l.value > r.high_normal                        then 'high'
               else 'normal'
           end as flag
    from work.labs_raw l
    left join work.lab_ref r on upcase(l.lab_name) = upcase(r.lab_name)
    where l.value is not null
      and l.patient_id is not null;
quit;

data work.labs_clean work.labs_implausible;
    set work.labs_flagged;
    if flag = 'IMPLAUSIBLE' then output work.labs_implausible;
    else output work.labs_clean;
run;

/* ─────────────────────────────────────────────────────────────────
   STEP 4: SOFA SCORE CALCULATION
   Sequential Organ Failure Assessment - key sepsis indicator
   ───────────────────────────────────────────────────────────────── */
%log_msg(Calculating SOFA scores);

proc sql;
    create table work.sofa_components as
    select
        v.patient_id,
        v.charttime,
        /* PaO2/FiO2 ratio approximated from SpO2 */
        case
            when v.spo2 >= 95 then 0
            when v.spo2 >= 90 then 1
            when v.spo2 >= 85 then 2
            when v.spo2 >= 80 then 3
            else 4
        end as sofa_resp,
        /* MAP-based cardiovascular */
        case
            when v.map >= 70 then 0
            when v.map >= 65 then 1
            when v.map >= 50 then 2
            else 3
        end as sofa_cardio,
        /* Creatinine-based renal */
        case
            when l_cr.value < 1.2 then 0
            when l_cr.value < 2.0 then 1
            when l_cr.value < 3.5 then 2
            when l_cr.value < 5.0 then 3
            else 4
        end as sofa_renal,
        /* Platelet-based coagulation */
        case
            when l_plt.value >= 150 then 0
            when l_plt.value >= 100 then 1
            when l_plt.value >= 50  then 2
            when l_plt.value >= 20  then 3
            else 4
        end as sofa_coag,
        /* GCS-based neurological */
        case
            when v.gcs_total >= 15 then 0
            when v.gcs_total >= 13 then 1
            when v.gcs_total >= 10 then 2
            when v.gcs_total >= 6  then 3
            else 4
        end as sofa_neuro,
        v.heart_rate, v.sbp, v.spo2, v.map, v.resp_rate, v.temperature
    from work.vitals_clean v
    left join work.labs_clean l_cr  on v.patient_id = l_cr.patient_id
                                   and l_cr.lab_name = 'Creatinine'
    left join work.labs_clean l_plt on v.patient_id = l_plt.patient_id
                                   and l_plt.lab_name = 'Platelets';
quit;

data work.sofa_scores;
    set work.sofa_components;
    sofa_total = sofa_resp + sofa_cardio + sofa_renal + sofa_coag + sofa_neuro;
    
    /* SOFA >= 2 suggests organ dysfunction */
    sepsis_suspected = (sofa_total >= 2);
    
    label sofa_total = 'Total SOFA Score'
          sepsis_suspected = 'Sepsis Suspected (SOFA>=2)';
run;

/* ─────────────────────────────────────────────────────────────────
   STEP 5: FEATURE ENGINEERING
   Create ML-ready features
   ───────────────────────────────────────────────────────────────── */
%log_msg(Engineering features for ML pipeline);

proc sort data=work.vitals_clean; by patient_id charttime; run;

data work.vitals_features;
    set work.vitals_clean;
    by patient_id;
    
    /* Rolling statistics via lag */
    lag_hr   = lag(heart_rate);
    lag_sbp  = lag(sbp);
    lag_spo2 = lag(spo2);
    
    if first.patient_id then do;
        lag_hr = .; lag_sbp = .; lag_spo2 = .;
    end;
    
    /* Delta features (change from previous measurement) */
    delta_hr  = heart_rate - lag_hr;
    delta_sbp = sbp - lag_sbp;
    delta_spo2 = spo2 - lag_spo2;
    
    /* Shock Index = HR / SBP */
    if sbp > 0 then shock_index = heart_rate / sbp;
    else shock_index = .;
    
    /* Pulse pressure */
    if sbp ne . and dbp ne . then pulse_pressure = sbp - dbp;
    
    /* Hypotension flag */
    hypotension = (sbp < 90 and sbp ne .);
    tachycardia = (heart_rate > 100 and heart_rate ne .);
    hypoxia     = (spo2 < 92 and spo2 ne .);
    tachypnea   = (resp_rate > 22 and resp_rate ne .);
    fever       = (temperature > 38.3 and temperature ne .);
    
    drop lag_hr lag_sbp lag_spo2;
run;

/* Merge SOFA scores with features */
proc sql;
    create table work.ml_features as
    select f.*, s.sofa_total, s.sepsis_suspected
    from work.vitals_features f
    left join work.sofa_scores s
        on f.patient_id = s.patient_id and f.charttime = s.charttime;
quit;

/* ─────────────────────────────────────────────────────────────────
   STEP 6: NORMALIZATION
   Z-score normalization for continuous features
   ───────────────────────────────────────────────────────────────── */
proc means data=work.ml_features noprint;
    var heart_rate sbp dbp map temperature spo2 resp_rate
        delta_hr delta_sbp shock_index sofa_total;
    output out=work.norm_stats mean= std= / autoname;
run;

/* ─────────────────────────────────────────────────────────────────
   STEP 7: QUALITY REPORT
   ───────────────────────────────────────────────────────────────── */
%log_msg(Generating data quality report);

proc freq data=work.vitals_clean;
    tables flag_hr flag_sbp flag_spo2 flag_temp / nocum nopercent;
    title 'Vital Signs Validation Flags';
run;

proc freq data=work.labs_clean;
    tables flag / nocum;
    title 'Lab Results Flags Distribution';
run;

proc univariate data=work.ml_features noprint;
    var heart_rate sbp spo2 temperature resp_rate sofa_total;
    output out=work.feature_stats
        mean=mean_hr mean_sbp mean_spo2 mean_temp mean_rr mean_sofa
        std=std_hr std_sbp std_spo2 std_temp std_rr std_sofa
        min=min_hr min_sbp min_spo2 min_temp min_rr min_sofa
        max=max_hr max_sbp max_spo2 max_temp max_rr max_sofa;
run;

/* ─────────────────────────────────────────────────────────────────
   STEP 8: EXPORT FOR PYTHON PIPELINE
   ───────────────────────────────────────────────────────────────── */
%log_msg(Exporting cleaned data for Python ML pipeline);

proc export
    data    = work.ml_features
    outfile = "&EXPORT_PATH./ml_features.csv"
    dbms    = csv
    replace;
run;

proc export
    data    = work.labs_clean
    outfile = "&EXPORT_PATH./labs_clean.csv"
    dbms    = csv
    replace;
run;

proc export
    data    = work.sofa_scores
    outfile = "&EXPORT_PATH./sofa_scores.csv"
    dbms    = csv
    replace;
run;

%log_msg(SAS pipeline complete. Exported to &EXPORT_PATH.);

/* Summary counts */
proc sql;
    title 'Pipeline Output Summary';
    select 'ml_features'   as dataset, count(*) as records from work.ml_features
    union all
    select 'labs_clean',               count(*)             from work.labs_clean
    union all
    select 'sofa_scores',              count(*)             from work.sofa_scores
    union all
    select 'vitals_rejected',          count(*)             from work.vitals_rejected;
quit;
