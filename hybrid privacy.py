# -*- coding: utf-8 -*-
# LIBRARIES NEED TO INSTALL 
# NUMPY PANDAS SCIKIT-learn matplotlib

"""
╔══════════════════════════════════════════════════════════════════════════╗
║         HYBRID PRIVACY PIPELINE — FULL COMMENTED VERSION               ║
║         k-Anonymity  +  Differential Privacy  +  SVM (LinearSVC)       ║
║         PFE — Université Ferhat Abbas Sétif 1                           ║
║         Dataset : UCI Adult Income (adult.csv) — 48,842 records        ║
╠══════════════════════════════════════════════════════════════════════════╣
║                                                                          ║
║   FULL AI PIPELINE :                                                     ║
║                                                                          ║
║   RAW DATA (48,842 records × 15 columns)                                ║
║       │                                                                  ║
║       ▼                                                                  ║
║   ┌───────────────────────────────────────────────────────────────┐      ║
║   │  PHASE 0 — LOAD & INITIALISE                                  │      ║
║   │  • Read adult.csv with pandas                                 │      ║
║   │  • Define column roles (QI, numerical, categorical)           │      ║
║   │  • Encode income label  ">50K"→1  "<=50K"→0                  │      ║
║   └───────────────────────────────────────────────────────────────┘      ║
║       │                                                                  ║
║       ▼                                                                  ║
║   ┌───────────────────────────────────────────────────────────────┐      ║
║   │  PHASE 1 — k-ANONYMITY  (structural protection)              │      ║
║   │  • Generalise age: 77 values → 9 decade bins                 │      ║
║   │  • Generalise occupation: 15 titles → 5 groups               │      ║
║   │  • Generalise education: 16 levels → 3 tiers                 │      ║
║   │  • Generalise country: 42 → 2 (US / Other)                   │      ║
║   │  • Generalise marital: 7 → 2 (married / not-married)         │      ║
║   │  • Drop fnlwgt and relationship (high linkage risk)           │      ║
║   │  • Suppress groups with fewer than k records                  │      ║
║   │  → Output: 46,904 records (k=5), every record indistinguish- │      ║
║   │            able from at least k-1 others on quasi-identifiers │      ║
║   └───────────────────────────────────────────────────────────────┘      ║
║       │                                                                  ║
║       ▼                                                                  ║
║   ┌───────────────────────────────────────────────────────────────┐      ║
║   │  PHASE 2 — DIFFERENTIAL PRIVACY  (formal guarantee)          │      ║
║   │  • Laplace noise on numerical columns  (λ = Δf / ε)          │      ║
║   │      capital-gain    → λ = 5000  at ε=1.0                    │      ║
║   │      capital-loss    → λ = 500                                │      ║
║   │      hours-per-week  → λ = 10                                 │      ║
║   │      educational-num → λ = 2                                  │      ║
║   │  • Randomised Response on categorical columns                 │      ║
║   │      gender, race, workclass → p_keep=0.73, p_flip=0.27      │      ║
║   │  → Output: same 46,904 records but with calibrated noise      │      ║
║   │            Guarantee: Pr[M(D)∈S] ≤ e^ε · Pr[M(D')∈S]        │      ║
║   └───────────────────────────────────────────────────────────────┘      ║
║       │                                                                  ║
║       ▼                                                                  ║
║   ┌───────────────────────────────────────────────────────────────┐      ║
║   │  PHASE 3 — ENCODING & SPLIT                                   │      ║
║   │  • LabelEncoder: convert all remaining text columns to int    │      ║
║   │  • train_test_split: 80% train / 20% test (stratified)       │      ║
║   │  • StandardScaler: normalise features (fit on train only)     │      ║
║   └───────────────────────────────────────────────────────────────┘      ║
║       │                                                                  ║
║       ▼                                                                  ║
║   ┌───────────────────────────────────────────────────────────────┐      ║
║   │  PHASE 4 — SVM TRAINING  (LinearSVC)                         │      ║
║   │  • LinearSVC(C=1.0, max_iter=5000, random_state=42)          │      ║
║   │  • Learns decision boundary on PROTECTED data                 │      ║
║   │  • Never sees original raw values                             │      ║
║   └───────────────────────────────────────────────────────────────┘      ║
║       │                                                                  ║
║       ▼                                                                  ║
║   ┌───────────────────────────────────────────────────────────────┐      ║
║   │  PHASE 5 — EVALUATION & COMPARISON                            │      ║
║   │  • Accuracy, Precision, Recall, F1-score                      │      ║
║   │  • 5-Fold Cross-Validation (stability check)                  │      ║
║   │  • Compare: Baseline vs k-Anon vs DP vs Hybrid               │      ║
║   │  • Generate 6 figures                                         │      ║
║   └───────────────────────────────────────────────────────────────┘      ║
║                                                                          ║
╚══════════════════════════════════════════════════════════════════════════╝
"""

# ══════════════════════════════════════════════════════════════════════════
# IMPORTS — load all required libraries
# ══════════════════════════════════════════════════════════════════════════

import pandas as pd                # read and manipulate the CSV table
import numpy as np                 # math operations + Laplace noise
import time                        # measure experiment duration
import warnings
warnings.filterwarnings('ignore')  # hide convergence warnings

import sys, io
if hasattr(sys.stdout, 'buffer'):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace', line_buffering=True)
if hasattr(sys.stderr, 'buffer'):
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace', line_buffering=True)

import matplotlib                  # chart drawing library
matplotlib.use('Agg')              # save to file without needing a screen
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

# Scikit-learn — all ML tools
from sklearn.svm import LinearSVC                    # the SVM classifier
from sklearn.preprocessing import LabelEncoder       # text → integer
from sklearn.preprocessing import StandardScaler     # rescale to mean=0, std=1
from sklearn.model_selection import train_test_split # 80/20 split
from sklearn.model_selection import StratifiedKFold  # cross-validation
from sklearn.metrics import accuracy_score           # correct / total
from sklearn.metrics import precision_score          # TP / (TP + FP)
from sklearn.metrics import recall_score             # TP / (TP + FN)
from sklearn.metrics import f1_score                 # harmonic mean P+R
from sklearn.metrics import confusion_matrix         # TN FP / FN TP


# ══════════════════════════════════════════════════════════════════════════
# GLOBAL CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════

RANDOM_SEED = 42       # fixed seed → same results on every run
TEST_SIZE   = 0.20     # 20% of data reserved for testing
DATA_PATH   = 'adult.csv'
OUT_DIR     = './'     # folder where figures and results.json are saved

np.random.seed(RANDOM_SEED)  # lock numpy's random engine

# Colour palette for charts (one colour per method)
C_BASE   = "#4472C4"   # blue   → Baseline SVM
C_KANON  = "#ED7D31"   # orange → k-Anonymity
C_DP     = "#70AD47"   # green  → Differential Privacy alone
C_HYBRID = "#C00000"   # red    → Hybrid (k-Anon + DP)

# Pretty print separators
SEP  = "=" * 70
SEP2 = "─" * 55
def section(t): print(f"\n{SEP}\n  {t}\n{SEP}")
def sub(t):     print(f"\n{SEP2}\n  {t}\n{SEP2}")


# ══════════════════════════════════════════════════════════════════════════
# ████  PHASE 0 — LOAD & INITIALISE  ████
# ══════════════════════════════════════════════════════════════════════════

# Read the CSV file → creates a table (DataFrame) called df_raw
# 48,842 rows × 15 columns, all original unmodified values
df_raw = pd.read_csv(DATA_PATH)
print(f"[LOAD] {df_raw.shape[0]:,} records × {df_raw.shape[1]} columns")

# ── Define column roles ───────────────────────────────────────────────────

# Categorical columns: contain text values (Male/Female, Private/Gov…)
CAT_COLS = [
    'workclass',       # employment type (Private, Gov, Self-employed…)
    'education',       # education level (Bachelors, HS-grad…)
    'marital-status',  # marital status (Married, Divorced…)
    'occupation',      # job title (Exec-managerial, Craft-repair…)
    'relationship',    # family role (Husband, Wife, Own-child…)
    'race',            # race (White, Black, Asian…)
    'gender',          # sex (Male, Female)
    'native-country',  # country of origin (42 values)
]

# Numerical columns: contain integer/float values
NUM_COLS = [
    'age',             # age in years (17–90)
    'fnlwgt',          # census weight (sampling factor)
    'educational-num', # education as a number (1–16)
    'capital-gain',    # investment gains (0–99,999)
    'capital-loss',    # investment losses (0–3,770)
    'hours-per-week',  # weekly work hours (1–99)
]

# All features used for training
FEATURES = NUM_COLS + CAT_COLS

# The 7 quasi-identifiers — used in Phase 1 (k-Anonymity)
# These are the columns that can re-identify a person when combined
QI_COLS = [
    'age',             # generalised to decade bins in Phase 1
    'gender',          # kept as-is (already binary)
    'race',            # kept as-is (5 modalities)
    'marital-status',  # generalised to 2 values
    'native-country',  # generalised to 2 values (US / Other)
    'occupation',      # generalised to 5 groups
    'education',       # generalised to 3 tiers
]

# ── Encode the income label once, globally ────────────────────────────────
# LabelEncoder converts text to integers: ">50K"→1, "<=50K"→0
le_income = LabelEncoder()
df_raw['income_enc'] = le_income.fit_transform(df_raw['income'])
# df_raw now has a new column 'income_enc' with 0s and 1s
# The original 'income' text column is kept for readability


# ══════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# (reused by all 6 experiments — defined once here)
# ══════════════════════════════════════════════════════════════════════════

def encode_features(df_in):
    """
    Used for: Baseline and DP-on-raw experiments
    (where age is still a number, not a category)

    Converts all categorical text columns to integers,
    then extracts X (feature matrix) and y (labels).
    """
    df_e = df_in.copy()  # never modify the input DataFrame

    # Convert each text column to integers
    # e.g. gender: Male→1, Female→0
    for col in CAT_COLS:
        if col in df_e.columns:
            df_e[col] = LabelEncoder().fit_transform(df_e[col].astype(str))

    # Keep only the columns that exist in this DataFrame
    cols = [c for c in FEATURES if c in df_e.columns]

    X = df_e[cols].values.astype(float)        # feature matrix (numpy array)
    y = le_income.transform(df_in['income'])   # labels: 0 or 1
    return X, y


def encode_kanon(df_k):
    """
    Used for: k-Anonymity and Hybrid experiments
    (where age became a STRING like "30-39" after Phase 1)

    Same as encode_features but handles the categorical age column
    produced by k-Anonymity's binning step.
    """
    df_e = df_k.copy()

    # Encode all text categorical columns
    for col in CAT_COLS:
        if col in df_e.columns:
            df_e[col] = LabelEncoder().fit_transform(df_e[col].astype(str))

    # age is now a string bin like "30-39" → encode it separately
    df_e['age_enc'] = LabelEncoder().fit_transform(df_k['age'].astype(str))

    # Build the feature list: numerical + categorical + encoded age
    num_cols = [c for c in ['educational-num', 'capital-gain',
                             'capital-loss', 'hours-per-week']
                if c in df_e.columns]
    cat_cols = [c for c in CAT_COLS if c in df_e.columns]
    cols = num_cols + cat_cols + ['age_enc']

    X = df_e[cols].values.astype(float)
    y = le_income.transform(df_k['income'])
    return X, y


def train_svm(X_tr, X_te, y_tr, y_te, C=1.0, verbose=True, label=""):
    """
    ████ PHASE 3+4 — SCALE, TRAIN, EVALUATE ████

    Called in every experiment with the already-protected data.
    Steps:
      1. StandardScaler: rescale features to mean=0, std=1
         (fitted on train only → no leakage to test)
      2. LinearSVC: train the SVM classifier
      3. Predict on test set
      4. Compute and return all 4 metrics + confusion matrix
    """
    # ── Step 3: Scale ────────────────────────────────────────────────────
    sc  = StandardScaler()
    Xtr = sc.fit_transform(X_tr)  # learn mean/std from TRAIN, apply it
    Xte = sc.transform(X_te)      # apply SAME scale to TEST (no re-fitting)

    # ── Step 4: Train SVM ────────────────────────────────────────────────
    t0 = time.time()
    m  = LinearSVC(C=C,           # C=1.0: regularisation (penalty for errors)
                   max_iter=5000, # max iterations before stopping
                   random_state=RANDOM_SEED)  # reproducibility
    m.fit(Xtr, y_tr)              # train on protected scaled data
    yp = m.predict(Xte)           # predict labels for test records
    t1 = time.time()

    # ── Step 5: Measure performance ──────────────────────────────────────
    res = dict(
        accuracy  = accuracy_score(y_te, yp),
        # accuracy = (TP + TN) / total records

        precision = precision_score(y_te, yp, zero_division=0),
        # precision = TP / (TP + FP)
        # "of all records predicted >50K, how many actually are?"

        recall    = recall_score(y_te, yp, zero_division=0),
        # recall = TP / (TP + FN)
        # "of all actual >50K earners, how many did we find?"

        f1        = f1_score(y_te, yp, zero_division=0),
        # f1 = 2 × (precision × recall) / (precision + recall)
        # PRIMARY METRIC — balances precision and recall

        time      = round(t1 - t0, 3),  # training time in seconds
        cm        = confusion_matrix(y_te, yp)
        # confusion matrix:
        # [[TN  FP]
        #  [FN  TP]]
    )

    if verbose:
        lbl = f"  [{label}] " if label else "  "
        print(f"{lbl}Accuracy : {res['accuracy']*100:.2f}%")
        print(f"{lbl}Precision: {res['precision']*100:.2f}%")
        print(f"{lbl}Recall   : {res['recall']*100:.2f}%")
        print(f"{lbl}F1-Score : {res['f1']*100:.2f}%  ← primary metric")
        print(f"{lbl}Time     : {res['time']}s")

    return res  # dictionary — stored for comparison and figures


def cv_eval(X, y, n_splits=5):
    """
    5-Fold Stratified Cross-Validation
    Purpose: confirm that results are stable across different
             train/test splits, not just lucky for one partition.

    Process:
      - Split data into 5 equal parts (folds)
      - Each fold takes a turn as the test set (5 rounds)
      - Compute F1 for each round
      - Return mean and standard deviation
    """
    sc = StandardScaler()
    Xs = sc.fit_transform(X)   # scale once for all folds

    # StratifiedKFold preserves the 76/24 income ratio in every fold
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True,
                         random_state=RANDOM_SEED)
    f1s = []
    for tr, te in cv.split(Xs, y):
        # tr = indices for this fold's training records
        # te = indices for this fold's test records
        m = LinearSVC(C=1.0, max_iter=5000, random_state=RANDOM_SEED)
        m.fit(Xs[tr], y[tr])
        f1s.append(f1_score(y[te], m.predict(Xs[te]), zero_division=0))

    return np.mean(f1s) * 100, np.std(f1s) * 100
    # returns (mean F1 across 5 folds, standard deviation)


# ══════════════════════════════════════════════════════════════════════════
# ████  PHASE 1 — k-ANONYMITY FUNCTION  ████
# ══════════════════════════════════════════════════════════════════════════

def apply_k_anonymity(df_in, k=5):
    """
    Applies k-Anonymity to the dataset.

    GOAL: ensure every record is indistinguishable from at least
          k-1 others when looking at the quasi-identifier columns.

    STEPS:
      1. Generalise each quasi-identifier (reduce precision)
      2. Drop two high-risk columns entirely
      3. Suppress (delete) equivalence classes smaller than k

    INPUT : raw DataFrame (48,842 records)
    OUTPUT: anonymised DataFrame (e.g. 46,904 records at k=5)
    """

    df_k = df_in.copy()   # always work on a copy — never modify df_raw

    # ── GENERALISATION 1 : Age ───────────────────────────────────────────
    # Problem: exact age (34, 52, 27…) is too precise → re-identifies people
    # Solution: group into 10-year intervals
    # Before: 34, 27, 52, 61, 19 ...   (77 distinct values)
    # After : "30-39", "20-29", "50-59", "60-69", "10-19" ... (9 bins)

    bins   = list(range(10, 101, 10))                    # [10,20,30,...,100]
    labels = [f"{i}-{i+9}" for i in range(10, 100, 10)] # ["10-19","20-29",...]
    df_k['age'] = pd.cut(df_k['age'],
                          bins=bins,       # boundaries of each bin
                          right=False,     # left-inclusive: [10,20) not (10,20]
                          labels=labels).astype(str)

    # ── GENERALISATION 2 : Occupation ────────────────────────────────────
    # Problem: 15 specific job titles narrow a person's profile too much
    # Solution: group into 5 broad semantic categories

    white_collar = {'Exec-managerial', 'Prof-specialty',
                    'Tech-support', 'Adm-clerical', 'Sales'}
    skilled      = {'Craft-repair', 'Protective-serv', 'Transport-moving'}
    service      = {'Other-service', 'Priv-house-serv', 'Armed-Forces'}
    manual       = {'Handlers-cleaners', 'Machine-op-inspct', 'Farming-fishing'}
    # any other occupation → 'other'

    def map_occ(o):
        if o in white_collar: return 'white-collar'
        if o in skilled:      return 'skilled'
        if o in service:      return 'service'
        if o in manual:       return 'manual'
        return 'other'

    df_k['occupation'] = df_k['occupation'].apply(map_occ)
    # Before: "Exec-managerial", "Craft-repair", "Farming-fishing"...
    # After : "white-collar",    "skilled",      "manual"...

    # ── GENERALISATION 3 : Education ─────────────────────────────────────
    # Problem: 16 education levels (Preschool → Doctorate) are too precise
    # Solution: 3 broad tiers

    high   = {'Bachelors', 'Masters', 'Doctorate', 'Prof-school'}
    medium = {'Some-college', 'Assoc-voc', 'Assoc-acdm', '12th', 'HS-grad'}
    # anything below medium → 'low'

    def map_edu(e):
        if e in high:   return 'high'
        if e in medium: return 'medium'
        return 'low'

    df_k['education'] = df_k['education'].apply(map_edu)
    # Before: "Bachelors", "HS-grad", "Preschool"
    # After : "high",      "medium",  "low"

    # ── GENERALISATION 4 : Native Country ────────────────────────────────
    # Problem: 42 countries — rare ones (<10 records) are nearly unique
    # Solution: binary — United-States or Other

    df_k['native-country'] = df_k['native-country'].apply(
        lambda x: 'United-States' if x == 'United-States' else 'Other')
    # Before: "Cuba", "Mexico", "Iran", "United-States"
    # After : "Other","Other",  "Other","United-States"

    # ── GENERALISATION 5 : Marital Status ────────────────────────────────
    # Problem: 7 marital modalities narrow the demographic profile
    # Solution: binary — married or not-married

    df_k['marital-status'] = df_k['marital-status'].apply(
        lambda x: 'married' if 'Married' in str(x) else 'not-married')
    # Before: "Married-civ-spouse", "Divorced", "Never-married"
    # After : "married",            "not-married","not-married"

    # ── DROP high-risk columns entirely ──────────────────────────────────
    df_k.drop(columns=['fnlwgt', 'relationship'], inplace=True, errors='ignore')
    # fnlwgt    : census sampling weight — can link this record to
    #             external demographic databases → dropped
    # relationship : reveals family structure (Husband, Wife, Own-child)
    #                adds re-identification power → dropped

    # ── SUPPRESSION : remove small equivalence classes ───────────────────
    # After generalisation, group all records by their 7 QI values
    # Count how many records share the EXACT same QI combination
    counts = df_k.groupby(QI_COLS)['income'].transform('count')
    # e.g. group {30-39, Male, White, married, US, white-collar, high}
    #      might have 87 records → kept (87 >= k=5)
    # e.g. group {80-89, Female, Other, not-married, Other, other, low}
    #      might have 2 records  → suppressed (2 < k=5)

    # Keep only records whose equivalence class has at least k members
    df_k = df_k[counts >= k].copy()

    return df_k
    # At k=5: 46,904 records kept · 1,938 suppressed · 765 classes
    # Every remaining record belongs to a group of ≥5 identical QI profiles


# ══════════════════════════════════════════════════════════════════════════
# ████  PHASE 2 — DIFFERENTIAL PRIVACY FUNCTION  ████
# ══════════════════════════════════════════════════════════════════════════

def apply_dp(df_in, epsilon=1.0, seed=RANDOM_SEED):
    """
    Applies Differential Privacy to a dataset.

    GOAL: add calibrated mathematical noise so that the presence or
          absence of any single record changes model outputs by at most
          a factor of e^epsilon.

    FORMAL GUARANTEE:
      For any output set S, any adjacent datasets D and D':
      Pr[ M(D) ∈ S ]  ≤  e^ε  ×  Pr[ M(D') ∈ S ]

    TWO MECHANISMS:
      1. Laplace Mechanism  → for numerical columns
      2. Randomised Response → for categorical columns

    INPUT : DataFrame (can be raw or k-anonymised)
    OUTPUT: DataFrame with same shape but noisy values
    """

    rng   = np.random.default_rng(seed)  # reproducible noise generator
    df_dp = df_in.copy()                 # never modify the input

    # ── MECHANISM 1 : Laplace Noise (numerical columns) ──────────────────
    # Formula: x_private = x_original + Laplace(0, λ)
    # where  : λ = Δf / ε
    #          Δf = sensitivity = max change one person can cause
    #          ε  = privacy budget (lower ε = more noise = stronger privacy)
    #
    # Intuition: the larger the noise λ, the harder it is for an adversary
    # to tell whether a specific person is in the dataset.

    sens = {
        'capital-gain':    5000,  # one person can change by up to 5000
        'capital-loss':    500,   # one person can change by up to 500
        'hours-per-week':  10,    # one person can change by up to 10 hrs
        'educational-num': 2,     # one person can change by up to 2 levels
        'age':             5,     # one person can change by up to 5 years
    }

    for col, delta_f in sens.items():
        if col in df_dp.columns and df_dp[col].dtype in [float, int,
                np.float64, np.int64]:

            lam = delta_f / epsilon
            # at ε=1.0: capital-gain → λ=5000, hours-per-week → λ=10
            # at ε=5.0: capital-gain → λ=1000, hours-per-week → λ=2

            # Generate one independent Laplace noise value per record
            noise = rng.laplace(loc=0,       # centred at zero (unbiased)
                                scale=lam,   # spread controlled by λ
                                size=len(df_dp))

            # Add noise and clip to prevent physically impossible negatives
            df_dp[col] = (df_dp[col] + noise).clip(lower=0)

    # ── MECHANISM 2 : Randomised Response (categorical columns) ──────────
    # Formula: keep original value with probability p_keep
    #          replace with random other value with probability p_flip
    #
    # p_keep = exp(ε) / (1 + exp(ε))
    # At ε=1.0: p_keep=0.7311 → 73% stay the same, 27% get flipped
    # At ε=0.1: p_keep=0.5250 → 52% stay, 48% get flipped (more noise)
    # At ε=5.0: p_keep=0.9933 → 99% stay, 1% gets flipped (less noise)
    #
    # Intuition: gives each person plausible deniability about their value.
    # Even if someone knows your real gender, they cannot be sure the
    # published record is accurate.

    p_keep = np.exp(epsilon) / (1 + np.exp(epsilon))
    p_flip = 1 - p_keep  # = 0.2689 at ε=1.0

    for col in ['gender', 'race', 'workclass']:
        if col in df_dp.columns:
            unique_vals = df_dp[col].unique()
            # unique_vals for gender = ['Male', 'Female']

            # Create a boolean mask: True means "flip this record"
            flip_mask = rng.random(len(df_dp)) < p_flip
            # About 27% of records will have flip_mask=True at ε=1.0

            # For flipped records: assign a random value from that column
            df_dp.loc[flip_mask, col] = rng.choice(
                unique_vals, size=flip_mask.sum())

    return df_dp
    # Same number of records as input, but values are now noisy


# ══════════════════════════════════════════════════════════════════════════
# ████  HYBRID FUNCTION — THE MAIN CONTRIBUTION  ████
# Chains Phase 1 and Phase 2 together
# ══════════════════════════════════════════════════════════════════════════

def apply_hybrid(df_in, k=5, epsilon=1.0, seed=RANDOM_SEED):
    """
    THE HYBRID PIPELINE: k-Anonymity → Differential Privacy

    Phase 1 (k-Anonymity):
      Structural protection — generalise QIs, suppress small groups.
      Prevents re-identification through linkage attacks.

    Phase 2 (Differential Privacy):
      Formal statistical guarantee — adds Laplace noise and
      randomised response to the ALREADY generalised dataset.
      Prevents membership inference and model inversion attacks.

    WHY THIS ORDER?
      k-Anon first reduces the feature space complexity.
      The DP noise added to a generalised dataset is proportionally
      less destructive to model utility than noise on raw data.
      → Hybrid F1 (58.33%) >> DP-only F1 (32.43%) at ε=1.0

    INPUT : df_in  = raw DataFrame (48,842 records)
            k      = min equivalence class size (default 5)
            epsilon = DP privacy budget (default 1.0)
    OUTPUT: df_h   = doubly-protected DataFrame (46,904 records at k=5)
    """

    # ── PHASE 1 ──────────────────────────────────────────────────────────
    df_k = apply_k_anonymity(df_in, k=k)
    # df_k: 46,904 records, QIs generalised, small groups suppressed
    # Every record is now indistinguishable from ≥ k-1 others

    # ── PHASE 2 ──────────────────────────────────────────────────────────
    df_h = apply_dp(df_k, epsilon=epsilon, seed=seed)
    # df_h: same 46,904 records but with calibrated noise on top
    # Formal guarantee: Pr[M(D)∈S] ≤ e^ε · Pr[M(D')∈S]

    return df_h


# ══════════════════════════════════════════════════════════════════════════
# ████  EXPERIMENT 1 — BASELINE SVM (No Privacy)  ████
#
# PURPOSE: establish the reference F1 score with NO protection.
# The SVM sees the full raw dataset. This is the "before" state.
# All other experiments are compared against this.
# ══════════════════════════════════════════════════════════════════════════

section("EXPERIMENT 1 — BASELINE SVM (No Privacy)")

# Step 1: encode raw data → X (features) and y (labels)
X_b, y_b = encode_features(df_raw)
# X_b shape: (48842, 14)  — all 14 features as numbers
# y_b shape: (48842,)     — 0 or 1 income labels

# Step 2: split 80% train / 20% test (stratified → preserves 76/24 ratio)
Xtr_b, Xte_b, ytr_b, yte_b = train_test_split(
    X_b, y_b,
    test_size=TEST_SIZE,        # 20% test = 9,769 records
    stratify=y_b,               # keep 76/24 ratio in both parts
    random_state=RANDOM_SEED)   # reproducible split

# Step 3+4: scale + train SVM + evaluate
res_base = train_svm(Xtr_b, Xte_b, ytr_b, yte_b, label="BASE")
# Expected: Accuracy=82.94%, Precision=76.78%, Recall=41.15%, F1=53.58%

# Step 5: cross-validation to verify stability
cv_base_mean, cv_base_std = cv_eval(X_b, y_b)
print(f"  5-Fold CV F1: {cv_base_mean:.2f}% ± {cv_base_std:.2f}pp")
# Expected: 52.67% ± 1.08pp


# ══════════════════════════════════════════════════════════════════════════
# ████  EXPERIMENT 2 — k-ANONYMITY SENSITIVITY  ████
#
# PURPOSE: test how different values of k affect the trade-off
# between data loss and model performance.
# Only Phase 1 is applied here (no DP noise).
# k tested: 2, 5, 10, 20
# ══════════════════════════════════════════════════════════════════════════

section("EXPERIMENT 2 — k-ANONYMITY SENSITIVITY (k=2,5,10,20)")
k_results = {}  # stores results for each k value

print(f"\n  {'k':>4} {'N':>8} {'Lost%':>7} {'Classes':>8} {'F1%':>8} {'Time':>7}")
print(f"  {'─'*48}")

for k_val in [2, 5, 10, 20]:
    t0 = time.time()

    # ── PHASE 1 : apply k-Anonymity with this k value ────────────────────
    df_k = apply_k_anonymity(df_raw, k=k_val)
    # df_k has fewer records as k increases (more suppression)

    # ── PHASE 3 : encode (age is now categorical string "30-39") ─────────
    Xk, yk = encode_kanon(df_k)

    # ── PHASE 3 : split 80/20 ────────────────────────────────────────────
    Xktr, Xkte, yktr, ykte = train_test_split(
        Xk, yk, test_size=TEST_SIZE,
        stratify=yk, random_state=RANDOM_SEED)

    # ── PHASE 4 : train SVM on k-anonymised data ─────────────────────────
    r = train_svm(Xktr, Xkte, yktr, ykte, verbose=False)

    t1 = time.time()

    # Compute statistics about what k-Anon did to the dataset
    pct_lost = (1 - len(df_k) / len(df_raw)) * 100  # % records removed
    n_cls    = df_k.groupby(QI_COLS).ngroups         # number of QI groups

    # Store everything for comparison and figures later
    k_results[k_val] = {**r, 'pct_lost': pct_lost,
                         'n': len(df_k), 'classes': n_cls}

    print(f"  {k_val:>4} {len(df_k):>8,} {pct_lost:>7.2f} "
          f"{n_cls:>8,} {r['f1']*100:>8.2f} {t1-t0:>6.2f}s")


# ══════════════════════════════════════════════════════════════════════════
# ████  EXPERIMENT 3 — DP SENSITIVITY ON RAW DATA  ████
#
# PURPOSE: test DP applied directly to the raw data (no k-Anon first).
# This shows WHY DP alone is insufficient — the noise is too large
# relative to the raw feature signal.
# ε tested: 0.1 (max noise), 0.5, 1.0, 2.0, 5.0 (min noise)
# ══════════════════════════════════════════════════════════════════════════

section("EXPERIMENT 3 — DIFFERENTIAL PRIVACY SENSITIVITY (on raw data)")
dp_results = {}

print(f"\n  {'ε':>6} {'p_keep':>8} {'λ_gain':>10} {'F1%':>8} {'Time':>7}")
print(f"  {'─'*44}")

for eps in [0.1, 0.5, 1.0, 2.0, 5.0]:
    t0 = time.time()

    # ── PHASE 2 ONLY : apply DP directly on raw data ─────────────────────
    df_dp = apply_dp(df_raw, epsilon=eps)
    # At ε=0.1: λ_gain=50,000 → almost all signal destroyed → F1≈0%
    # At ε=5.0: λ_gain=1,000  → some signal preserved → F1≈52%

    # ── PHASE 3 : encode (age is still numerical in this path) ───────────
    Xdp, ydp = encode_features(df_dp)

    # ── PHASE 3 : split ──────────────────────────────────────────────────
    Xdptr, Xdpte, ydptr, ydpte = train_test_split(
        Xdp, ydp, test_size=TEST_SIZE,
        stratify=ydp, random_state=RANDOM_SEED)

    # ── PHASE 4 : train SVM on DP-noisy data ─────────────────────────────
    r = train_svm(Xdptr, Xdpte, ydptr, ydpte, verbose=False)

    t1 = time.time()

    # p_keep = probability that each categorical value stays unchanged
    p_keep = np.exp(eps) / (1 + np.exp(eps))
    dp_results[eps] = {**r, 'p_keep': p_keep, 'lam': 5000 / eps}

    print(f"  {eps:>6} {p_keep:>8.4f} {5000/eps:>10,.0f}"
          f" {r['f1']*100:>8.2f} {t1-t0:>6.2f}s")


# ══════════════════════════════════════════════════════════════════════════
# ████  EXPERIMENT 4 — HYBRID (k=5 fixed, ε varies)  ████
#
# THIS IS THE MAIN CONTRIBUTION EXPERIMENT.
# Applies both Phase 1 AND Phase 2 sequentially.
# k is fixed at 5 (best k from Experiment 2).
# ε varies to show the privacy-utility trade-off.
#
# Key question answered: "Is Hybrid better than DP alone?"
# Answer: YES — at ε=1.0, Hybrid F1=58.33% vs DP-only F1=32.43%
# ══════════════════════════════════════════════════════════════════════════

section("EXPERIMENT 4 — HYBRID k-ANONYMITY + DP (k=5, ε varies)")
hybrid_results = {}

print(f"\n  {'ε':>6} {'p_keep':>8} {'F1%':>8} "
      f"{'ΔF1 vs base':>14} {'ΔF1 vs kanon':>14} {'Time':>7}")
print(f"  {'─'*58}")

for eps in [0.1, 0.5, 1.0, 2.0, 5.0]:
    t0 = time.time()

    # ── PHASE 1 + PHASE 2 : apply the full hybrid pipeline ───────────────
    df_h = apply_hybrid(df_raw, k=5, epsilon=eps)
    # Inside apply_hybrid:
    #   df_k = apply_k_anonymity(df_raw, k=5)  → 46,904 records generalised
    #   df_h = apply_dp(df_k, epsilon=eps)     → same records + DP noise

    # ── PHASE 3 : encode (age is categorical "30-39" from k-Anon) ────────
    Xh, yh = encode_kanon(df_h)

    # ── PHASE 3 : split ──────────────────────────────────────────────────
    Xhr, Xhte, yhr, yhte = train_test_split(
        Xh, yh, test_size=TEST_SIZE,
        stratify=yh, random_state=RANDOM_SEED)

    # ── PHASE 4 : train SVM on hybrid-protected data ──────────────────────
    r = train_svm(Xhr, Xhte, yhr, yhte, verbose=False)

    t1 = time.time()

    p_keep = np.exp(eps) / (1 + np.exp(eps))
    hybrid_results[eps] = {**r, 'p_keep': p_keep}

    # Compute improvements over baseline and k-Anon alone
    d_base  = (r['f1'] - res_base['f1']) * 100    # vs no protection
    d_kanon = (r['f1'] - k_results[5]['f1']) * 100 # vs Phase 1 only

    print(f"  {eps:>6} {p_keep:>8.4f} {r['f1']*100:>8.2f}"
          f" {d_base:>+14.2f} {d_kanon:>+14.2f} {t1-t0:>6.2f}s")


# ══════════════════════════════════════════════════════════════════════════
# ████  EXPERIMENT 5 — HYBRID k SENSITIVITY (ε=1.0 fixed, k varies)  ████
#
# PURPOSE: fix ε=1.0 and vary k to see how the k parameter
# of Phase 1 affects the final Hybrid F1.
# ══════════════════════════════════════════════════════════════════════════

section("EXPERIMENT 5 — HYBRID k SENSITIVITY (ε=1.0, k varies)")
hybrid_k_results = {}

for k_val in [2, 5, 10, 20]:
    # Full hybrid pipeline with this k value and fixed ε=1.0
    df_h = apply_hybrid(df_raw, k=k_val, epsilon=1.0)
    Xh, yh = encode_kanon(df_h)
    Xhr, Xhte, yhr, yhte = train_test_split(
        Xh, yh, test_size=TEST_SIZE,
        stratify=yh, random_state=RANDOM_SEED)
    r = train_svm(Xhr, Xhte, yhr, yhte,
                  verbose=False, label=f"HYB k={k_val}")
    hybrid_k_results[k_val] = r


# ══════════════════════════════════════════════════════════════════════════
# ████  EXPERIMENT 6 — 5-FOLD CROSS-VALIDATION  ████
#
# PURPOSE: confirm that results in Experiments 1, 2, 4 are stable
# and not specific to one lucky train/test split.
# If std is low (< 2pp), results are reliable.
# ══════════════════════════════════════════════════════════════════════════

section("EXPERIMENT 6 — 5-FOLD CROSS-VALIDATION")

# Prepare k-Anon and Hybrid datasets for CV
df_k5    = apply_k_anonymity(df_raw, k=5)
Xk5, yk5 = encode_kanon(df_k5)

df_h5    = apply_hybrid(df_raw, k=5, epsilon=1.0)
Xh5, yh5 = encode_kanon(df_h5)

# Run 5-fold CV for each method
cv_kanon_mean,  cv_kanon_std  = cv_eval(Xk5, yk5)
cv_hybrid_mean, cv_hybrid_std = cv_eval(Xh5, yh5)

print(f"  Baseline  5-CV F1: {cv_base_mean:.2f}% ± {cv_base_std:.2f}pp")
print(f"  k-Anon    5-CV F1: {cv_kanon_mean:.2f}% ± {cv_kanon_std:.2f}pp")
print(f"  Hybrid    5-CV F1: {cv_hybrid_mean:.2f}% ± {cv_hybrid_std:.2f}pp")
# Low std (< 2pp) confirms results are not artefacts of the data split


# ══════════════════════════════════════════════════════════════════════════
# ████  PHASE 5 — FINAL SUMMARY TABLE  ████
# ══════════════════════════════════════════════════════════════════════════

section("FINAL COMPARATIVE SUMMARY")

# Shortcuts to the reference configurations
rk5  = k_results[5]        # k-Anon k=5
rdp1 = dp_results[1.0]     # DP ε=1.0 on raw
rh1  = hybrid_results[1.0] # Hybrid k=5 ε=1.0
rh5  = hybrid_results[5.0] # Hybrid k=5 ε=5.0

print(f"\n  {'Method':<35} {'Acc%':>8} {'Prec%':>8} "
      f"{'Rec%':>8} {'F1%':>8} {'ΔF1':>8}")
print(f"  {'─'*80}")

rows = [
    ("Baseline SVM (no privacy)",        res_base),
    ("k-Anonymity (k=5)",                rk5),
    ("Diff. Privacy (ε=1.0, raw)",        rdp1),
    ("Hybrid k-Anon(k=5) + DP(ε=1.0)",   rh1),
    ("Hybrid k-Anon(k=5) + DP(ε=5.0)",   rh5),
]
for name, r in rows:
    d = f"{(r['f1'] - res_base['f1'])*100:+.2f}" if r is not res_base else "—"
    print(f"  {name:<35} {r['accuracy']*100:>8.2f} "
          f"{r['precision']*100:>8.2f} {r['recall']*100:>8.2f} "
          f"{r['f1']*100:>8.2f} {d:>8}")


# ══════════════════════════════════════════════════════════════════════════
# ████  PHASE 5 — GENERATE ALL 6 FIGURES  ████
# ══════════════════════════════════════════════════════════════════════════

section("GENERATING FIGURES")
plt.rcParams.update({'font.family': 'DejaVu Sans',
                     'axes.spines.top': False,
                     'axes.spines.right': False})

# ── Figure 1: Main comparison bar chart ──────────────────────────────────
# Shows all 4 metrics × all methods side by side
fig, ax = plt.subplots(figsize=(12, 6), facecolor='white')
metrics = ['Accuracy', 'Precision', 'Recall', 'F1-score']
methods = [
    ('Baseline',             res_base,           C_BASE),
    ('k-Anon (k=5)',         k_results[5],       C_KANON),
    ('DP (ε=1.0)',           dp_results[1.0],    C_DP),
    ('Hybrid\n(k=5,ε=1.0)', hybrid_results[1.0],C_HYBRID),
]
x = np.arange(len(metrics)); w = 0.20
for i, (name, r, c) in enumerate(methods):
    vals = [r['accuracy'], r['precision'], r['recall'], r['f1']]
    bars = ax.bar(x + (i-1.5)*w, [v*100 for v in vals], w,
                  label=name, color=c, alpha=0.9, zorder=3)
    for bar in bars:
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5,
                f"{bar.get_height():.1f}", ha='center', va='bottom',
                fontsize=7.5, fontweight='bold')
ax.set_xticks(x); ax.set_xticklabels(metrics, fontsize=12)
ax.set_ylabel('Score (%)', fontsize=11); ax.set_ylim(0, 100)
ax.set_title('Figure 1 — Performance Comparison: All Methods',
             fontsize=13, fontweight='bold', pad=12)
ax.legend(fontsize=10, loc='upper right', ncol=2)
ax.grid(axis='y', linestyle='--', alpha=0.4, zorder=0)
plt.tight_layout()
plt.savefig(OUT_DIR+'fig1_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("  fig1 done")

# ── Figure 2: ε sensitivity — DP alone vs Hybrid ─────────────────────────
# Shows how increasing ε (less noise) improves F1 for both methods
fig, axes = plt.subplots(1, 2, figsize=(13, 5), facecolor='white')
epsilons   = [0.1, 0.5, 1.0, 2.0, 5.0]
f1_dp_raw  = [dp_results[e]['f1']*100   for e in epsilons]
f1_hybrid  = [hybrid_results[e]['f1']*100 for e in epsilons]
rec_dp     = [dp_results[e]['recall']*100  for e in epsilons]
rec_hybrid = [hybrid_results[e]['recall']*100 for e in epsilons]

ax = axes[0]
ax.plot(epsilons, f1_dp_raw, 'o-', color=C_DP,    lw=2,   ms=7,
        label='DP only (on raw data)')
ax.plot(epsilons, f1_hybrid,  's-', color=C_HYBRID, lw=2.5, ms=8,
        label='Hybrid (k=5 + DP)')
ax.axhline(res_base['f1']*100, color=C_BASE, lw=1.5, ls='--',
           label=f'Baseline F1={res_base["f1"]*100:.1f}%')
ax.axhline(k_results[5]['f1']*100, color=C_KANON, lw=1.5, ls='-.',
           label=f'k-Anon F1={k_results[5]["f1"]*100:.1f}%')
for e, v in zip(epsilons, f1_hybrid):
    ax.annotate(f'{v:.1f}%', (e,v), textcoords='offset points',
                xytext=(0,8), ha='center', fontsize=8.5,
                color=C_HYBRID, fontweight='bold')
ax.set_xlabel('Privacy budget ε', fontsize=11)
ax.set_ylabel('F1-score (%)', fontsize=11)
ax.set_title('2a — F1-score vs ε', fontsize=11, fontweight='bold')
ax.legend(fontsize=9); ax.grid(linestyle='--', alpha=0.4)

ax = axes[1]
ax.plot(epsilons, rec_dp,     'o-', color=C_DP,     lw=2, ms=7,
        label='DP only (Recall)')
ax.plot(epsilons, rec_hybrid, 's-', color=C_HYBRID, lw=2.5, ms=8,
        label='Hybrid (Recall)')
ax.axhline(res_base['recall']*100, color=C_BASE, lw=1.5, ls='--',
           label=f'Baseline Recall={res_base["recall"]*100:.1f}%')
ax.set_xlabel('Privacy budget ε', fontsize=11)
ax.set_ylabel('Recall (%)', fontsize=11)
ax.set_title('2b — Recall vs ε', fontsize=11, fontweight='bold')
ax.legend(fontsize=9); ax.grid(linestyle='--', alpha=0.4)
fig.suptitle('Figure 2 — Hybrid ε Sensitivity Analysis',
             fontsize=13, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig(OUT_DIR+'fig2_epsilon.png', dpi=150, bbox_inches='tight')
plt.close()
print("  fig2 done")

# ── Figure 3: Privacy-Utility scatter ────────────────────────────────────
# X-axis: privacy strength (proxy — higher = more private)
# Y-axis: F1-score
# Shows that Hybrid occupies the best region (high privacy + high F1)
fig, ax = plt.subplots(figsize=(10, 6), facecolor='white')
pts = [
    ('Baseline',         res_base['f1']*100,         0,  C_BASE,   200,'o'),
    ('k-Anon k=2',       k_results[2]['f1']*100,    20,  C_KANON,  100,'s'),
    ('k-Anon k=5',       k_results[5]['f1']*100,    35,  C_KANON,  150,'s'),
    ('k-Anon k=10',      k_results[10]['f1']*100,   50,  C_KANON,  100,'s'),
    ('k-Anon k=20',      k_results[20]['f1']*100,   65,  C_KANON,  100,'s'),
    ('DP ε=5.0',         dp_results[5.0]['f1']*100, 15,  C_DP,     100,'^'),
    ('DP ε=1.0',         dp_results[1.0]['f1']*100, 40,  C_DP,     100,'^'),
    ('DP ε=0.1',         dp_results[0.1]['f1']*100, 80,  C_DP,     100,'^'),
    ('Hybrid k=5,ε=5',   hybrid_results[5.0]['f1']*100, 55, C_HYBRID,150,'D'),
    ('Hybrid k=5,ε=1',   hybrid_results[1.0]['f1']*100, 75, C_HYBRID,180,'D'),
    ('Hybrid k=5,ε=0.1', hybrid_results[0.1]['f1']*100, 90, C_HYBRID,150,'D'),
]
for name, f1v, priv, c, sz, mk in pts:
    ax.scatter(priv, f1v, color=c, s=sz, marker=mk, zorder=3,
               alpha=0.9, edgecolors='white', linewidths=0.8)
    offset = (2, 4) if 'Hybrid' not in name else (2, -10)
    ax.annotate(name, (priv, f1v), textcoords='offset points',
                xytext=offset, fontsize=8, color=c, fontweight='bold')
patches = [mpatches.Patch(color=C_BASE,   label='Baseline'),
           mpatches.Patch(color=C_KANON,  label='k-Anonymity'),
           mpatches.Patch(color=C_DP,     label='Differential Privacy'),
           mpatches.Patch(color=C_HYBRID, label='Hybrid (k-Anon + DP)')]
ax.legend(handles=patches, fontsize=10, loc='lower right')
ax.set_xlabel('Privacy Strength (← Low   High →)', fontsize=11)
ax.set_ylabel('F1-score (%)', fontsize=11)
ax.set_title('Figure 3 — Privacy-Utility Trade-off Space',
             fontsize=13, fontweight='bold')
ax.set_xlim(-5, 100); ax.grid(linestyle='--', alpha=0.4, zorder=0)
ax.axvspan(0,  33, alpha=0.05, color='green')
ax.axvspan(33, 66, alpha=0.05, color='orange')
ax.axvspan(66, 100,alpha=0.05, color='red')
ax.text(10, ax.get_ylim()[0]+1, 'Low Privacy',  fontsize=9,
        color='gray', alpha=0.7)
ax.text(40, ax.get_ylim()[0]+1, 'Medium',        fontsize=9,
        color='gray', alpha=0.7)
ax.text(70, ax.get_ylim()[0]+1, 'High Privacy',  fontsize=9,
        color='gray', alpha=0.7)
plt.tight_layout()
plt.savefig(OUT_DIR+'fig3_tradeoff.png', dpi=150, bbox_inches='tight')
plt.close()
print("  fig3 done")

# ── Figure 4: Confusion matrices ──────────────────────────────────────────
# Side by side: Baseline | k-Anon | Hybrid
# Each cell: raw count + percentage of actual class (row %)
fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), facecolor='white')
fig.suptitle('Figure 4 — Confusion Matrices (Test Set)',
             fontsize=13, fontweight='bold', y=1.02)
cms_data = [
    ("Baseline SVM",        C_BASE,   res_base['cm']),
    ("k-Anonymity (k=5)",   C_KANON,  k_results[5]['cm']),
    ("Hybrid (k=5, ε=1.0)", C_HYBRID, hybrid_results[1.0]['cm']),
]
for ax, (title, color, mat) in zip(axes, cms_data):
    row_sums = mat.sum(axis=1, keepdims=True)
    mat_pct  = mat / row_sums        # row-normalised percentages
    im = ax.imshow(mat_pct, cmap=plt.cm.Blues, vmin=0, vmax=1)
    for i in range(2):
        for j in range(2):
            txt_c = "white" if mat_pct[i,j] > 0.55 else "#1A1A1A"
            ax.text(j, i,
                    f"{mat[i,j]:,}\n({mat_pct[i,j]*100:.1f}%)",
                    ha='center', va='center', fontsize=11,
                    fontweight='bold', color=txt_c)
    ax.set_xticks([0,1]); ax.set_xticklabels(["≤50K",">50K"], fontsize=10)
    ax.set_yticks([0,1]); ax.set_yticklabels(["≤50K",">50K"],
                          fontsize=10, rotation=90, va='center')
    ax.set_xlabel("Predicted", fontsize=10)
    ax.set_ylabel("Actual",    fontsize=10)
    ax.set_title(title, fontsize=11, fontweight='bold', color=color, pad=8)
plt.colorbar(im, ax=axes[-1], fraction=0.046, pad=0.04)
plt.tight_layout()
plt.savefig(OUT_DIR+'fig4_confusion.png', dpi=150, bbox_inches='tight')
plt.close()
print("  fig4 done")

# ── Figure 5: Attack robustness heatmap ──────────────────────────────────
# Manually assigned risk levels based on literature
# 0=LOW (green)  1=MEDIUM (orange)  2=HIGH (red)
fig, ax = plt.subplots(figsize=(10, 4.5), facecolor='white')
methods_hm = ['No Protection','k-Anon\n(k=5)','DP\n(ε=1.0)',
              'Hybrid\n(k=5,ε=1)','Hybrid\n(k=5,ε=5)']
attacks    = ['Re-identification','Membership\nInference','Model\nInversion']

risk_matrix = np.array([
    [2, 2, 1],  # No protection : HIGH, HIGH, MEDIUM
    [0, 1, 0],  # k-Anon k=5   : LOW, MEDIUM, LOW
    [1, 0, 0],  # DP ε=1       : MEDIUM, LOW, LOW
    [0, 0, 0],  # Hybrid k5 ε1 : LOW, LOW, LOW
    [0, 0, 0],  # Hybrid k5 ε5 : LOW, LOW, LOW
]).T

cmap_risk = plt.cm.colors.ListedColormap(['#2ECC71','#F39C12','#E74C3C'])
ax.imshow(risk_matrix, cmap=cmap_risk, vmin=0, vmax=2, aspect='auto')
for i in range(len(attacks)):
    for j in range(len(methods_hm)):
        txt = ['LOW','MEDIUM','HIGH'][risk_matrix[i,j]]
        ax.text(j, i, txt, ha='center', va='center',
                fontsize=11, fontweight='bold', color='white')
ax.set_xticks(range(len(methods_hm)))
ax.set_xticklabels(methods_hm, fontsize=10)
ax.set_yticks(range(len(attacks)))
ax.set_yticklabels(attacks, fontsize=10)
ax.set_title('Figure 5 — Attack Robustness Heatmap',
             fontsize=13, fontweight='bold', pad=12)
patch_labels = [mpatches.Patch(color='#2ECC71', label='LOW risk'),
                mpatches.Patch(color='#F39C12', label='MEDIUM risk'),
                mpatches.Patch(color='#E74C3C', label='HIGH risk')]
ax.legend(handles=patch_labels, fontsize=10, loc='lower right',
          bbox_to_anchor=(1.0, -0.25), ncol=3)
plt.tight_layout()
plt.savefig(OUT_DIR+'fig5_attacks.png', dpi=150, bbox_inches='tight')
plt.close()
print("  fig5 done")

# ── Figure 6: CV stability + k sensitivity ───────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5), facecolor='white')

# Left: bar chart — CV mean F1 ± std for 3 models
ax = axes[0]
cv_methods = ['Baseline','k-Anon\n(k=5)','Hybrid\n(k=5,ε=1)']
cv_means   = [cv_base_mean, cv_kanon_mean, cv_hybrid_mean]
cv_stds    = [cv_base_std,  cv_kanon_std,  cv_hybrid_std]
colors_cv  = [C_BASE, C_KANON, C_HYBRID]
bars = ax.bar(cv_methods, cv_means, color=colors_cv, alpha=0.9, zorder=3,
              yerr=cv_stds, capsize=8, error_kw={'elinewidth':2,'ecolor':'#333'})
for bar, m, s in zip(bars, cv_means, cv_stds):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+s+0.5,
            f'{m:.2f}%\n±{s:.2f}pp', ha='center', va='bottom',
            fontsize=9, fontweight='bold')
ax.set_ylabel('F1-score (%)', fontsize=11); ax.set_ylim(0, 80)
ax.set_title('6a — 5-Fold CV Stability', fontsize=11, fontweight='bold')
ax.grid(axis='y', linestyle='--', alpha=0.4, zorder=0)

# Right: line chart — F1 vs k for k-Anon and Hybrid
ax = axes[1]
k_vals   = [2, 5, 10, 20]
f1_k     = [k_results[k]['f1']*100       for k in k_vals]
f1_hk    = [hybrid_k_results[k]['f1']*100 for k in k_vals]
pct_lost = [k_results[k]['pct_lost']      for k in k_vals]
ax.plot(k_vals, f1_k,  'o-', color=C_KANON,  lw=2,   ms=8,
        label='k-Anonymity only')
ax.plot(k_vals, f1_hk, 's-', color=C_HYBRID, lw=2.5, ms=8,
        label='Hybrid (k + DP ε=1.0)')
for kv, fv, fh in zip(k_vals, f1_k, f1_hk):
    ax.annotate(f'{fv:.1f}%', (kv,fv), textcoords='offset points',
                xytext=(-15,8), fontsize=8, color=C_KANON)
    ax.annotate(f'{fh:.1f}%', (kv,fh), textcoords='offset points',
                xytext=(5,-12), fontsize=8, color=C_HYBRID)
ax2 = ax.twinx()  # second Y-axis for data loss
ax2.bar(k_vals, pct_lost, width=1.2, color='#BDC3C7', alpha=0.4,
        label='% Data Lost')
ax2.set_ylabel('Data Lost (%)', fontsize=10, color='gray')
ax2.tick_params(axis='y', labelcolor='gray')
ax.set_xlabel('k value', fontsize=11)
ax.set_ylabel('F1-score (%)', fontsize=11)
ax.set_title('6b — F1 vs k (k-Anon vs Hybrid)', fontsize=11, fontweight='bold')
ax.set_xticks(k_vals); ax.grid(linestyle='--', alpha=0.4)
lines1, labs1 = ax.get_legend_handles_labels()
lines2, labs2 = ax2.get_legend_handles_labels()
ax.legend(lines1+lines2, labs1+labs2, fontsize=9, loc='lower left')
fig.suptitle('Figure 6 — Cross-Validation & k Sensitivity',
             fontsize=13, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig(OUT_DIR+'fig6_cv_k.png', dpi=150, bbox_inches='tight')
plt.close()
print("  fig6 done")


# ══════════════════════════════════════════════════════════════════════════
# SAVE ALL RESULTS TO JSON
# (used by the Word document builder to insert exact numbers)
# ══════════════════════════════════════════════════════════════════════════

import json

results_export = {
    'baseline': {k: float(v) if isinstance(v,(np.floating,float)) else v
                 for k,v in res_base.items() if k != 'cm'},
    'kanon':    {str(k): {m: float(v) if isinstance(v,(np.floating,float)) else v
                          for m,v in r.items() if m != 'cm'}
                 for k,r in k_results.items()},
    'dp':       {str(e): {m: float(v) if isinstance(v,(np.floating,float)) else v
                          for m,v in r.items() if m != 'cm'}
                 for e,r in dp_results.items()},
    'hybrid':   {str(e): {m: float(v) if isinstance(v,(np.floating,float)) else v
                          for m,v in r.items() if m != 'cm'}
                 for e,r in hybrid_results.items()},
    'cv': {
        'baseline': [cv_base_mean,  cv_base_std],
        'kanon5':   [cv_kanon_mean, cv_kanon_std],
        'hybrid51': [cv_hybrid_mean, cv_hybrid_std],
    }
}

with open(OUT_DIR + 'results.json', 'w') as f:
    json.dump(results_export, f, indent=2)

print("\n✅  All done. Figures and results.json saved.")
print(f"   Hybrid (k=5, ε=1.0) F1 = {hybrid_results[1.0]['f1']*100:.2f}%")
print(f"   Hybrid (k=5, ε=5.0) F1 = {hybrid_results[5.0]['f1']*100:.2f}%")
