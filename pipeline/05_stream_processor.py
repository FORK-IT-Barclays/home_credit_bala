import json
import time
from datetime import datetime

# Simulating an Apache Kafka / Amazon Kinesis stream consumer
file_path = "/home/balahero03/credit/pipeline/simulated_kafka_stream.jsonl"

print("="*60)
print("VECTOR STREAM PROCESSOR: LIVE KAFKA CONSUMER")
print("="*60)

# Stateful feature store mimicking Feast / Redis
customer_state = {}

def process_transaction(tx):
    cid = tx["customer_id"]
    if cid not in customer_state:
        customer_state[cid] = {
            "last_salary_date": None,
            "salary_drift_days": [],
            "last_utility_date": None,
            "utility_drift_days": [],
            "atm_count": 0,
            "failed_emi": 0,
            "current_balance": 0.0,
            "risk_alerts": []
        }
    
    state = customer_state[cid]
    tx_date_str = tx["timestamp"]
    # Parse format "2023-01-01T12:00:00Z"
    tx_date = datetime.strptime(tx_date_str, "%Y-%m-%dT%H:%M:%SZ")
    mcc = tx["merchant_category"]
    amount = tx["amount"]
    tx_type = tx["type"]
    state["current_balance"] = tx["balance_after"]
    
    print(f"\n[STREAM IN] UID: {cid} | {mcc} | {amount} | Bal: {tx['balance_after']} | Time: {tx_date_str}")

    # Rule 1: Salary Drift (Signal 1)
    if mcc == "SALARY":
        if state["last_salary_date"] is not None:
            days_diff = (tx_date - state["last_salary_date"]).days
            # Nominal is 30 days. Anything > 30 is drift.
            if days_diff > 31:
                state["salary_drift_days"].append(days_diff - 30)
                alert = f"🚨 EXCEPTION: Salary delayed by {days_diff - 30} days!"
                state["risk_alerts"].append(alert)
                print(alert)
        state["last_salary_date"] = tx_date
        
    # Rule 4: Utility Bill Drift
    if mcc == "UTILITIES":
        if state["last_utility_date"] is not None:
            days_diff = (tx_date - state["last_utility_date"]).days
            if days_diff > 31:
                state["utility_drift_days"].append(days_diff - 30)
                alert = f"⚠️ WARNING: Fixed obligations shifted later by {days_diff - 30} days."
                state["risk_alerts"].append(alert)
                print(alert)
        state["last_utility_date"] = tx_date
        
    # Rule 7: Cash Hoarding (ATM)
    if mcc == "ATM_WITHDRAWAL":
        state["atm_count"] += 1
        if state["atm_count"] >= 3:
            alert = f"⚠️ METRIC: High-Velocity Cash Hoarding detected"
            if alert not in state["risk_alerts"]:
                state["risk_alerts"].append(alert)
                print(alert)

    # Rule 5: Failed Auto Debit
    if tx_type == "DEBIT_FAILED_NSF":
        state["failed_emi"] += 1
        alert = f"💥 CRITICAL: Failed auto-debit (NSF)."
        state["risk_alerts"].append(alert)
        print(alert)
        
    # Vectors Evaluation (Physics of Risk)
    if len(state["salary_drift_days"]) > 0 or len(state["utility_drift_days"]) > 0 or state["failed_emi"] > 0:
        drift = sum(state["salary_drift_days"]) + sum(state["utility_drift_days"])
        if drift > 15 or state["failed_emi"] > 0:
            print(f"   => 🔮 VECTOR ENGINE: User {cid} pushed to CRITICAL SPIRAL (Risk Zone 9)")
        elif drift > 5:
            print(f"   => 🔮 VECTOR ENGINE: User {cid} pushed to HIGH DRIFT (Risk Zone 7)")
            
# Process stream line by line
with open(file_path, "r") as f:
    for line in f:
        # Simulate network latency
        time.sleep(0.05)
        tx = json.loads(line)
        process_transaction(tx)

print("\n" + "="*60)
print("FINAL CUSTOMER RISK STATE:")
import pprint
pprint.pprint(customer_state)

