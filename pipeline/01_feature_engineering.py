"""
VECTOR — Pre-Delinquency Intervention Engine
Step 1: Feature Engineering Pipeline

Builds features aligned to the 7 VECTOR behavioral signals + CAMELS framework:
  - Signal 1: Salary Timing Drift (payment delay patterns)
  - Signal 2: Income Erosion (income stability features)
  - Signal 3: Liquidity Momentum (savings/balance depletion speed)
  - Signal 4: Payment Integrity / Bill Drift (installment delay patterns)
  - Signal 5: Failed Auto-Debits (DPD / overdue counts)
  - Signal 6: Credit Exhaustion (credit utilization acceleration)
  - Signal 7: Engagement Decay (bureau activity decay signals)
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path

DATA_DIR = Path("/home/balahero03/credit")
OUT_DIR  = DATA_DIR / "pipeline" / "output"
OUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 60)
print("VECTOR Feature Engineering Pipeline")
print("=" * 60)

# ─────────────────────────────────────────────────────────────
# 1. LOAD MAIN APPLICATION DATA
# ─────────────────────────────────────────────────────────────
print("\n[1/7] Loading application data...")
app_train = pd.read_csv(DATA_DIR / "application_train.csv")
app_test  = pd.read_csv(DATA_DIR / "application_test.csv")

print(f"  Train: {app_train.shape} | Test: {app_test.shape}")
print(f"  Default rate: {app_train['TARGET'].mean():.2%}")

# ─────────────────────────────────────────────────────────────
# 2. APPLICATION-LEVEL FEATURES (Financial Historian - XGBoost)
# ─────────────────────────────────────────────────────────────
print("\n[2/7] Engineering application-level features (CAMELS: C, A, E)...")

def engineer_application_features(df):
    feat = pd.DataFrame()
    feat['SK_ID_CURR'] = df['SK_ID_CURR']

    # ── Income & Capital (CAMELS: C - Capital Adequacy) ──────
    feat['INCOME_TOTAL']            = df['AMT_INCOME_TOTAL']
    feat['CREDIT_TO_INCOME_RATIO']  = df['AMT_CREDIT'] / (df['AMT_INCOME_TOTAL'] + 1)
    feat['ANNUITY_TO_INCOME_RATIO'] = df['AMT_ANNUITY'] / (df['AMT_INCOME_TOTAL'] + 1)
    feat['CREDIT_TO_GOODS_RATIO']   = df['AMT_CREDIT'] / (df['AMT_GOODS_PRICE'] + 1)
    feat['INCOME_PER_PERSON']       = df['AMT_INCOME_TOTAL'] / (df['CNT_FAM_MEMBERS'] + 1)

    # Signal 2: Income Erosion proxy — employment gap
    feat['DAYS_EMPLOYED_RATIO']     = df['DAYS_EMPLOYED'] / (df['DAYS_BIRTH'] + 1)
    feat['EMPLOYMENT_GAP']          = (df['DAYS_EMPLOYED'] < 0).astype(int)   # unemployed flag
    feat['EMPLOYED_TO_AGE_RATIO']   = df['DAYS_EMPLOYED'].clip(upper=0).abs() / (df['DAYS_BIRTH'].abs() + 1)

    # ── Age & Demographics ───────────────────────────────────
    feat['AGE_YEARS']               = df['DAYS_BIRTH'].abs() / 365
    feat['YOUNG_BORROWER']          = (feat['AGE_YEARS'] < 30).astype(int)
    feat['SENIOR_BORROWER']         = (feat['AGE_YEARS'] > 55).astype(int)

    # ── External Risk Scores (CAMELS: A - Asset Quality) ─────
    feat['EXT_SOURCE_MEAN']         = df[['EXT_SOURCE_1','EXT_SOURCE_2','EXT_SOURCE_3']].mean(axis=1)
    feat['EXT_SOURCE_STD']          = df[['EXT_SOURCE_1','EXT_SOURCE_2','EXT_SOURCE_3']].std(axis=1)
    feat['EXT_SOURCE_MIN']          = df[['EXT_SOURCE_1','EXT_SOURCE_2','EXT_SOURCE_3']].min(axis=1)
    feat['EXT_SOURCE_PRODUCT']      = df['EXT_SOURCE_1'] * df['EXT_SOURCE_2'] * df['EXT_SOURCE_3']
    feat['MISSING_EXT_SOURCES']     = df[['EXT_SOURCE_1','EXT_SOURCE_2','EXT_SOURCE_3']].isnull().sum(axis=1)

    # ── Document flags (CAMELS: M - Management) ──────────────
    doc_cols = [c for c in df.columns if c.startswith('FLAG_DOCUMENT')]
    feat['TOTAL_DOCS_PROVIDED']     = df[doc_cols].sum(axis=1)

    # ── Contact & Region flags ────────────────────────────────
    feat['SOCIAL_CIRCLE_DEFAULT_MEAN'] = df[['DEF_30_CNT_SOCIAL_CIRCLE',
                                              'DEF_60_CNT_SOCIAL_CIRCLE']].mean(axis=1)
    feat['OBS_SOCIAL_CIRCLE_MEAN']  = df[['OBS_30_CNT_SOCIAL_CIRCLE',
                                           'OBS_60_CNT_SOCIAL_CIRCLE']].mean(axis=1)

    # ── Signal 3: Liquidity proxy via car/realty ownership ───
    feat['OWNS_CAR']                = (df['FLAG_OWN_CAR'] == 'Y').astype(int)
    feat['OWNS_REALTY']             = (df['FLAG_OWN_REALTY'] == 'Y').astype(int)
    feat['ASSET_OWNERSHIP_SCORE']   = feat['OWNS_CAR'] + feat['OWNS_REALTY']

    # ── Application timing (CAMELS: S - Sensitivity) ─────────
    feat['HOUR_APPR_PROCESS']       = df['HOUR_APPR_PROCESS_START']
    feat['NIGHT_APPLICATION']       = ((df['HOUR_APPR_PROCESS_START'] < 9) |
                                       (df['HOUR_APPR_PROCESS_START'] > 20)).astype(int)

    # ── Region risk ──────────────────────────────────────────
    feat['REGION_RATING']           = df['REGION_RATING_CLIENT']
    feat['REGION_CITY_DIFF']        = df['REG_CITY_NOT_LIVE_CITY'].astype(int)

    # ── Children burden ──────────────────────────────────────
    feat['CNT_CHILDREN']            = df['CNT_CHILDREN']
    feat['CHILDREN_INCOME_BURDEN']  = df['CNT_CHILDREN'] / (df['AMT_INCOME_TOTAL'] + 1)

    # ── Categorical encodings ────────────────────────────────
    for col in ['CODE_GENDER','NAME_INCOME_TYPE','NAME_EDUCATION_TYPE',
                'NAME_FAMILY_STATUS','NAME_HOUSING_TYPE','NAME_CONTRACT_TYPE']:
        feat[col] = df[col].astype('category').cat.codes

    return feat

app_train_feat = engineer_application_features(app_train)
app_test_feat  = engineer_application_features(app_test)
print(f"  Generated {app_train_feat.shape[1]-1} application features")

# ─────────────────────────────────────────────────────────────
# 3. BUREAU FEATURES (CAMELS: A — Asset Quality)
# ─────────────────────────────────────────────────────────────
print("\n[3/7] Engineering bureau features (Signal 5, 6)...")
bureau = pd.read_csv(DATA_DIR / "bureau.csv")
bureau_bal = pd.read_csv(DATA_DIR / "bureau_balance.csv")

# Bureau balance — status-based overdue momentum
bureau_bal['STATUS_OVERDUE'] = bureau_bal['STATUS'].isin(['1','2','3','4','5']).astype(int)
bureau_bal['STATUS_BAD']     = bureau_bal['STATUS'].isin(['3','4','5']).astype(int)
bb_agg = bureau_bal.groupby('SK_ID_BUREAU').agg(
    OVERDUE_MONTHS  = ('STATUS_OVERDUE', 'sum'),
    BAD_MONTHS      = ('STATUS_BAD', 'sum'),
    TOTAL_MONTHS    = ('MONTHS_BALANCE', 'count'),
).reset_index()
bb_agg['OVERDUE_RATIO']    = bb_agg['OVERDUE_MONTHS'] / (bb_agg['TOTAL_MONTHS'] + 1)
# Signal 7: Engagement Decay → recent overdue trend
bureau_bal_sorted = bureau_bal.sort_values(['SK_ID_BUREAU','MONTHS_BALANCE'], ascending=[True, False])
recent_bb = bureau_bal_sorted.groupby('SK_ID_BUREAU').head(6)
recent_agg = recent_bb.groupby('SK_ID_BUREAU').agg(
    RECENT_OVERDUE_MONTHS = ('STATUS_OVERDUE', 'sum'),
).reset_index()
bb_agg = bb_agg.merge(recent_agg, on='SK_ID_BUREAU', how='left')

bureau = bureau.merge(bb_agg, on='SK_ID_BUREAU', how='left')

# Signal 5: Failed Auto-Debits → credit overdue days
# Signal 6: Credit Exhaustion → utilization
bureau['CREDIT_UTILIZATION'] = bureau['AMT_CREDIT_SUM_DEBT'] / (bureau['AMT_CREDIT_SUM'] + 1)
bureau['OVERDUE_FLAG']       = (bureau['CREDIT_DAY_OVERDUE'] > 0).astype(int)
bureau['IS_ACTIVE']          = (bureau['CREDIT_ACTIVE'] == 'Active').astype(int)
bureau['IS_CLOSED']          = (bureau['CREDIT_ACTIVE'] == 'Closed').astype(int)

bureau_agg = bureau.groupby('SK_ID_CURR').agg(
    BUREAU_CREDIT_COUNT         = ('SK_ID_BUREAU', 'count'),
    BUREAU_ACTIVE_COUNT         = ('IS_ACTIVE', 'sum'),
    BUREAU_CLOSED_COUNT         = ('IS_CLOSED', 'sum'),
    BUREAU_AVG_OVERDUE_DAYS     = ('CREDIT_DAY_OVERDUE', 'mean'),
    BUREAU_MAX_OVERDUE_DAYS     = ('CREDIT_DAY_OVERDUE', 'max'),
    BUREAU_TOTAL_OVERDUE        = ('OVERDUE_FLAG', 'sum'),
    BUREAU_MAX_AMT_OVERDUE      = ('AMT_CREDIT_MAX_OVERDUE', 'max'),
    BUREAU_SUM_DEBT             = ('AMT_CREDIT_SUM_DEBT', 'sum'),
    BUREAU_AVG_CREDIT_UTIL      = ('CREDIT_UTILIZATION', 'mean'),
    BUREAU_MAX_CREDIT_UTIL      = ('CREDIT_UTILIZATION', 'max'),
    BUREAU_OVERDUE_RATIO        = ('OVERDUE_RATIO', 'mean'),
    BUREAU_RECENT_OVERDUE_MONTHS= ('RECENT_OVERDUE_MONTHS', 'sum'),
    BUREAU_BAD_MONTHS           = ('BAD_MONTHS', 'sum'),
    BUREAU_PROLONG_COUNT        = ('CNT_CREDIT_PROLONG', 'sum'),
    BUREAU_AVG_DAYS_CREDIT      = ('DAYS_CREDIT', 'mean'),
    BUREAU_CREDIT_ENDDATE_MEAN  = ('DAYS_CREDIT_ENDDATE', 'mean'),
).reset_index()

# Risk Velocity signal — credit utilization acceleration (recent vs overall)
bureau_agg['CREDIT_UTIL_RISK_VELOCITY'] = (
    bureau_agg['BUREAU_RECENT_OVERDUE_MONTHS'] / (bureau_agg['BUREAU_ACTIVE_COUNT'] + 1)
)
print(f"  Generated {bureau_agg.shape[1]-1} bureau features")

# ─────────────────────────────────────────────────────────────
# 4. INSTALLMENTS FEATURES (Signal 4: Bill Drift, Signal 1: Salary Timing)
# ─────────────────────────────────────────────────────────────
print("\n[4/7] Engineering installment payment features (Signal 1, 4)...")
inst = pd.read_csv(DATA_DIR / "installments_payments.csv")

# Signal 4: Payment Integrity — how late/early are payments?
inst['PAYMENT_DELAY']       = inst['DAYS_ENTRY_PAYMENT'] - inst['DAYS_INSTALMENT']
inst['PAID_LATE']           = (inst['PAYMENT_DELAY'] > 0).astype(int)
inst['PAID_EARLY']          = (inst['PAYMENT_DELAY'] < 0).astype(int)
inst['PAYMENT_SHORTFALL']   = inst['AMT_INSTALMENT'] - inst['AMT_PAYMENT']
inst['UNDERPAID']           = (inst['PAYMENT_SHORTFALL'] > 0).astype(int)

# Signal 1: Salary Timing Drift — variance in payment timing (proxy)
inst_agg = inst.groupby('SK_ID_CURR').agg(
    INST_COUNT               = ('SK_ID_PREV', 'count'),
    # Signal 4: Bill Drift
    INST_MEAN_DELAY          = ('PAYMENT_DELAY', 'mean'),
    INST_MAX_DELAY           = ('PAYMENT_DELAY', 'max'),
    INST_STD_DELAY           = ('PAYMENT_DELAY', 'std'),    # variability = Signal 1 proxy
    INST_LATE_COUNT          = ('PAID_LATE', 'sum'),
    INST_LATE_RATIO          = ('PAID_LATE', 'mean'),
    INST_EARLY_RATIO         = ('PAID_EARLY', 'mean'),
    # Signal 5: Shortfalls (Failed Auto-Debits proxy)
    INST_UNDERPAID_COUNT     = ('UNDERPAID', 'sum'),
    INST_UNDERPAID_RATIO     = ('UNDERPAID', 'mean'),
    INST_SHORTFALL_SUM       = ('PAYMENT_SHORTFALL', 'sum'),
    INST_SHORTFALL_MEAN      = ('PAYMENT_SHORTFALL', 'mean'),
    INST_AMT_PAYMENT_SUM     = ('AMT_PAYMENT', 'sum'),
    INST_AMT_INSTALMENT_SUM  = ('AMT_INSTALMENT', 'sum'),
).reset_index()

# Signal 1: Salary timing — std of delay (higher = more erratic payment timing)
inst_agg['PAYMENT_TIMING_ERRATICISM'] = inst_agg['INST_STD_DELAY'].fillna(0)

# Risk Velocity: Recent 12 installments vs all-time delay
inst_sorted = inst.sort_values(['SK_ID_CURR','DAYS_INSTALMENT'], ascending=[True, False])
recent_inst = inst_sorted.groupby('SK_ID_CURR').head(12)
recent_inst_agg = recent_inst.groupby('SK_ID_CURR').agg(
    RECENT_MEAN_DELAY   = ('PAYMENT_DELAY', 'mean'),
    RECENT_LATE_RATIO   = ('PAID_LATE', 'mean'),
    RECENT_UNDERPAID_R  = ('UNDERPAID', 'mean'),
).reset_index()
inst_agg = inst_agg.merge(recent_inst_agg, on='SK_ID_CURR', how='left')

# Risk Velocity = recent delay vs historical delay (acceleration of bill drift)
inst_agg['BILL_DRIFT_VELOCITY'] = (
    inst_agg['RECENT_MEAN_DELAY'] - inst_agg['INST_MEAN_DELAY']
)
print(f"  Generated {inst_agg.shape[1]-1} installment features")

# ─────────────────────────────────────────────────────────────
# 5. CREDIT CARD BALANCE (Signal 3: Liquidity, Signal 6: Credit Exhaustion)
# ─────────────────────────────────────────────────────────────
print("\n[5/7] Engineering credit card balance features (Signal 3, 6)...")
cc = pd.read_csv(DATA_DIR / "credit_card_balance.csv")

# Signal 6: Credit Exhaustion — utilization rate
cc['UTILIZATION_RATE'] = cc['AMT_BALANCE'] / (cc['AMT_CREDIT_LIMIT_ACTUAL'] + 1)
cc['ATM_RATIO']        = cc['AMT_DRAWINGS_ATM_CURRENT'] / (cc['AMT_DRAWINGS_CURRENT'] + 1)
cc['DPD_FLAG']         = (cc['SK_DPD'] > 0).astype(int)

cc_agg = cc.groupby('SK_ID_CURR').agg(
    CC_COUNT                = ('SK_ID_PREV', 'count'),
    CC_AVG_BALANCE          = ('AMT_BALANCE', 'mean'),
    CC_MAX_BALANCE          = ('AMT_BALANCE', 'max'),
    CC_MIN_BALANCE          = ('AMT_BALANCE', 'min'),
    CC_AVG_UTILIZATION      = ('UTILIZATION_RATE', 'mean'),
    CC_MAX_UTILIZATION      = ('UTILIZATION_RATE', 'max'),
    # Signal 3: Liquidity — balance drawdown
    CC_BALANCE_STD          = ('AMT_BALANCE', 'std'),
    CC_TOTAL_DRAWINGS       = ('AMT_DRAWINGS_CURRENT', 'sum'),
    CC_ATM_DRAWINGS_RATIO   = ('ATM_RATIO', 'mean'),          # Signal 7: cash hoarding
    CC_DPD_COUNT            = ('DPD_FLAG', 'sum'),
    CC_MAX_DPD              = ('SK_DPD', 'max'),
    CC_AVG_MIN_PAYMENT      = ('AMT_INST_MIN_REGULARITY', 'mean'),
).reset_index()

# Signal 6: Acceleration — recent 6 months vs all-time utilization
cc_sorted = cc.sort_values(['SK_ID_CURR','MONTHS_BALANCE'], ascending=[True, False])
recent_cc = cc_sorted.groupby('SK_ID_CURR').head(6)
recent_cc_agg = recent_cc.groupby('SK_ID_CURR').agg(
    RECENT_AVG_UTIL  = ('UTILIZATION_RATE', 'mean'),
    RECENT_DPD_COUNT = ('DPD_FLAG', 'sum'),
    RECENT_AVG_BAL   = ('AMT_BALANCE', 'mean'),
).reset_index()
cc_agg = cc_agg.merge(recent_cc_agg, on='SK_ID_CURR', how='left')

# Risk Velocity: credit exhaustion acceleration
cc_agg['CREDIT_EXHAUST_VELOCITY'] = cc_agg['RECENT_AVG_UTIL'] - cc_agg['CC_AVG_UTILIZATION']
# Signal 3: Liquidity Momentum — balance decay ratio
cc_agg['LIQUIDITY_MOMENTUM']      = (cc_agg['CC_MIN_BALANCE'] - cc_agg['CC_AVG_BALANCE']) / (cc_agg['CC_MAX_BALANCE'] - cc_agg['CC_MIN_BALANCE'] + 1)
print(f"  Generated {cc_agg.shape[1]-1} credit card features")

# ─────────────────────────────────────────────────────────────
# 6. POS CASH BALANCE (Loan payment stress signals)
# ─────────────────────────────────────────────────────────────
print("\n[6/7] Engineering POS cash balance features...")
pos = pd.read_csv(DATA_DIR / "POS_CASH_balance.csv")
pos['DPD_FLAG']       = (pos['SK_DPD'] > 0).astype(int)
pos['DPD_DEF_FLAG']   = (pos['SK_DPD_DEF'] > 0).astype(int)
pos['COMPLETED']      = (pos['NAME_CONTRACT_STATUS'] == 'Completed').astype(int)

pos_agg = pos.groupby('SK_ID_CURR').agg(
    POS_COUNT           = ('SK_ID_PREV', 'count'),
    POS_DPD_MEAN        = ('SK_DPD', 'mean'),
    POS_DPD_MAX         = ('SK_DPD', 'max'),
    POS_DPD_COUNT       = ('DPD_FLAG', 'sum'),
    POS_DPD_DEF_COUNT   = ('DPD_DEF_FLAG', 'sum'),
    POS_COMPLETED_RATIO = ('COMPLETED', 'mean'),
    POS_FUTURE_INST_MAX = ('CNT_INSTALMENT_FUTURE', 'max'),
    POS_FUTURE_INST_MEAN= ('CNT_INSTALMENT_FUTURE', 'mean'),
).reset_index()

# Signal 5: Failed Auto-Debits — DPD acceleration
pos_sorted  = pos.sort_values(['SK_ID_CURR','MONTHS_BALANCE'], ascending=[True,False])
recent_pos  = pos_sorted.groupby('SK_ID_CURR').head(6)
recent_pos_agg = recent_pos.groupby('SK_ID_CURR').agg(
    POS_RECENT_DPD_MEAN = ('SK_DPD', 'mean'),
    POS_RECENT_DPD_COUNT= ('DPD_FLAG', 'sum'),
).reset_index()
pos_agg = pos_agg.merge(recent_pos_agg, on='SK_ID_CURR', how='left')
pos_agg['DPD_VELOCITY'] = pos_agg['POS_RECENT_DPD_MEAN'] - pos_agg['POS_DPD_MEAN']
print(f"  Generated {pos_agg.shape[1]-1} POS cash features")

# ─────────────────────────────────────────────────────────────
# 7. MERGE ALL FEATURES + COMPUTE VECTOR RISK PHYSICS
# ─────────────────────────────────────────────────────────────
print("\n[7/7] Merging all features & computing VECTOR Risk Physics...")

def merge_all(app_feat, bureau_agg, inst_agg, cc_agg, pos_agg):
    df = app_feat.copy()
    df = df.merge(bureau_agg, on='SK_ID_CURR', how='left')
    df = df.merge(inst_agg,   on='SK_ID_CURR', how='left')
    df = df.merge(cc_agg,     on='SK_ID_CURR', how='left')
    df = df.merge(pos_agg,    on='SK_ID_CURR', how='left')
    return df

train_merged = merge_all(app_train_feat, bureau_agg, inst_agg, cc_agg, pos_agg)
test_merged  = merge_all(app_test_feat,  bureau_agg, inst_agg, cc_agg, pos_agg)

# ── VECTOR Physics Features ───────────────────────────────────
# Risk Velocity (V) = composite rate of behavioral change
# Risk Acceleration (A) = rate at which velocity is increasing

for df in [train_merged, test_merged]:
    # Composite Risk Velocity Score (normalized 0-1 proxy)
    velocity_components = []
    for col in ['BILL_DRIFT_VELOCITY','CREDIT_EXHAUST_VELOCITY',
                'CREDIT_UTIL_RISK_VELOCITY','DPD_VELOCITY']:
        if col in df.columns:
            c = df[col].fillna(0)
            normalized = (c - c.min()) / (c.max() - c.min() + 1e-9)
            velocity_components.append(normalized)

    if velocity_components:
        df['VECTOR_RISK_VELOCITY'] = sum(velocity_components) / len(velocity_components)

    # Risk Acceleration: diff between recent & historical overdue/delay rates
    df['VECTOR_RISK_ACCELERATION'] = (
        df.get('RECENT_LATE_RATIO', pd.Series(0, index=df.index)).fillna(0) -
        df.get('INST_LATE_RATIO',   pd.Series(0, index=df.index)).fillna(0)
    )

    # 9-Zone Risk Matrix classifier (Velocity x Acceleration quadrants)
    def zone_mapper(row):
        v = row.get('VECTOR_RISK_VELOCITY', 0) or 0
        a = row.get('VECTOR_RISK_ACCELERATION', 0) or 0
        v_level = 2 if v > 0.66 else (1 if v > 0.33 else 0)
        a_level = 2 if a > 0.05 else (1 if a > 0    else 0)
        return v_level * 3 + a_level   # 0-8 → 9 zones

    df['VECTOR_RISK_ZONE'] = df.apply(zone_mapper, axis=1)

    # Composite VECTOR Stress Score (for dashboard display)
    stress_signals = []
    for col, w in [
        ('INST_LATE_RATIO',       0.20),
        ('BUREAU_OVERDUE_RATIO',  0.20),
        ('CC_AVG_UTILIZATION',    0.15),
        ('BILL_DRIFT_VELOCITY',   0.15),
        ('VECTOR_RISK_VELOCITY',  0.15),
        ('DPD_VELOCITY',          0.10),
        ('CREDIT_EXHAUST_VELOCITY',0.05),
    ]:
        if col in df.columns:
            c = df[col].fillna(0)
            normalized = (c - c.min()) / (c.max() - c.min() + 1e-9)
            stress_signals.append(normalized * w)

    if stress_signals:
        df['VECTOR_STRESS_SCORE'] = sum(stress_signals)

print(f"  Train merged: {train_merged.shape}")
print(f"  Test  merged: {test_merged.shape}")

# ── Add TARGET ────────────────────────────────────────────────
train_merged['TARGET'] = app_train['TARGET'].values

# ── Handle Inf / NaN ─────────────────────────────────────────
train_merged.replace([np.inf, -np.inf], np.nan, inplace=True)
test_merged.replace([np.inf, -np.inf],  np.nan, inplace=True)

# ── Save ──────────────────────────────────────────────────────
train_out = OUT_DIR / "train_features.parquet"
test_out  = OUT_DIR / "test_features.parquet"
train_merged.to_parquet(train_out, index=False)
test_merged.to_parquet(test_out,  index=False)

print(f"\n✅ Saved:")
print(f"   {train_out}  ({train_merged.shape[1]} features, {len(train_merged):,} rows)")
print(f"   {test_out}   ({test_merged.shape[1]} features, {len(test_merged):,} rows)")

# ── Feature summary ───────────────────────────────────────────
print("\n── VECTOR Feature Groups ──")
vector_cols = [c for c in train_merged.columns if 'VECTOR' in c]
print(f"  Physics features:    {vector_cols}")
print(f"  Total features:      {train_merged.shape[1]-2}")   # -SK_ID, -TARGET
print(f"  Null % (train mean): {train_merged.isnull().mean().mean():.1%}")
print("\n" + "="*60)
print("Feature Engineering COMPLETE ✅")
print("="*60)
