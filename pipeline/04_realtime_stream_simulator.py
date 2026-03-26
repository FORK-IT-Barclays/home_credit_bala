import pandas as pd
import numpy as np
import random
import datetime
import time

def generate_transactions(customer_id, start_date, is_distressed=False):
    """
    Generates realistic daily transactions for a user over 3 months.
    If distressed=True, the user faces: late salary, reduced discretionary 
    savings, increased ATM withdrawals, and late utility payments.
    """
    tx_list = []
    current_date = start_date
    balance = 5000.0 if not is_distressed else 2000.0
    
    # 3 months of data
    for month in range(3):
        days_in_month = 30
        
        # Salary Event (Signal 1: Salary drift)
        salary_date = current_date + datetime.timedelta(days=1)
        if is_distressed and month > 0:
            # Salary gets progressively later
            salary_date = salary_date + datetime.timedelta(days=month*4) 
        
        balance += 3500.0
        tx_list.append({
            "timestamp": salary_date.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "customer_id": customer_id,
            "merchant_category": "SALARY",
            "amount": 3500.0,
            "type": "CREDIT",
            "balance_after": balance
        })
        
        # Utility Bill (Signal 4: Bill Drift)
        utility_date = current_date + datetime.timedelta(days=5)
        if is_distressed and month > 0:
            utility_date = utility_date + datetime.timedelta(days=month*7)
            
        balance -= 200.0
        tx_list.append({
            "timestamp": utility_date.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "customer_id": customer_id,
            "merchant_category": "UTILITIES",
            "amount": -200.0,
            "type": "DEBIT",
            "balance_after": balance
        })
        
        # Discretionary vs. Distressed Spends
        if is_distressed:
            # Signal 7: Cash hoarding / ATM
            for _ in range(3 + month):
                atm_date = current_date + datetime.timedelta(days=random.randint(5, 25))
                balance -= 100.0
                tx_list.append({
                    "timestamp": atm_date.strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "customer_id": customer_id,
                    "merchant_category": "ATM_WITHDRAWAL",
                    "amount": -100.0,
                    "type": "DEBIT",
                    "balance_after": balance
                })
            
            # Lending apps / High risk
            upi_date = current_date + datetime.timedelta(days=random.randint(10, 20))
            balance -= 300.0
            tx_list.append({
                "timestamp": upi_date.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "customer_id": customer_id,
                "merchant_category": "LENDING_APP_REPAYMENT",
                "amount": -300.0,
                "type": "DEBIT",
                "balance_after": balance
            })
            
            # Auto-Debit Bounce (Signal 5)
            if month == 2:
                bounce_date = current_date + datetime.timedelta(days=28)
                tx_list.append({
                    "timestamp": bounce_date.strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "customer_id": customer_id,
                    "merchant_category": "EMI_AUTO_DEBIT",
                    "amount": -400.0,
                    "type": "DEBIT_FAILED_NSF", # Non-sufficient funds
                    "balance_after": balance
                })
        else:
            # Healthy user spends on dining, entertainment
            for _ in range(4):
                dining_date = current_date + datetime.timedelta(days=random.randint(2, 28))
                balance -= 50.0
                tx_list.append({
                    "timestamp": dining_date.strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "customer_id": customer_id,
                    "merchant_category": "DINING",
                    "amount": -50.0,
                    "type": "DEBIT",
                    "balance_after": balance
                })

        current_date = current_date + datetime.timedelta(days=days_in_month)

    # Sort sequentially
    tx_list.sort(key=lambda x: x['timestamp'])
    return tx_list

print("Generating simulated real-time banking stream...")
start_date = datetime.datetime.now() - datetime.timedelta(days=90)

healthy_txs = generate_transactions("CUST_HEALTHY_01", start_date, is_distressed=False)
stressed_txs = generate_transactions("CUST_STRESSED_02", start_date, is_distressed=True)

all_txs = sorted(healthy_txs + stressed_txs, key=lambda x: x['timestamp'])

# Save to a jsonlines log to act as our "Kafka" source
import json
output_file = "/home/balahero03/credit/pipeline/simulated_kafka_stream.jsonl"
with open(output_file, "w") as f:
    for tx in all_txs:
        f.write(json.dumps(tx) + "\n")

print(f"Generated {len(all_txs)} transactions across 2 profiles.")
print(f"Saved payload to '{output_file}' ready for the real-time processor.")
