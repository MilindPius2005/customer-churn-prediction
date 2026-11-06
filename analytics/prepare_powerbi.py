#!/usr/bin/env python3
"""Prepare the original IBM Telco churn CSV for Excel / Power BI. Standard library only."""
from pathlib import Path
import csv
ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
DEST = ROOT / "analytics" / "data" / "customers_powerbi.csv"

def band_tenure(n):
    return "0-12" if n <= 12 else "13-24" if n <= 24 else "25-48" if n <= 48 else "49-72"

def band_charges(n):
    return "Under 40" if n < 40 else "40-69" if n < 70 else "70-99" if n < 100 else "100+"

def main():
    DEST.parent.mkdir(parents=True, exist_ok=True)
    fields = ["CustomerID", "Gender", "SeniorCitizen", "Partner", "Dependents",
              "TenureMonths", "PhoneService", "MultipleLines", "InternetService",
              "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport",
              "StreamingTV", "StreamingMovies", "Contract", "PaperlessBilling",
              "PaymentMethod", "MonthlyCharges", "TotalCharges", "Churn",
              "ChurnFlag", "TenureBand", "MonthlyChargeBand",
              "ChurnedMonthlyCharges", "DataQuality"]
    with SOURCE.open(newline="", encoding="utf-8-sig") as src, DEST.open("w", newline="", encoding="utf-8") as dest:
        reader = csv.DictReader(src)
        writer = csv.DictWriter(dest, fieldnames=fields)
        writer.writeheader()
        count = 0
        for d in reader:
            tenure = int(d["tenure"])
            monthly = float(d["MonthlyCharges"])
            churn = int(d["Churn"] == "Yes")
            total = d["TotalCharges"].strip()
            out = {"CustomerID": d["customerID"], "Gender": d["gender"],
                   "SeniorCitizen": d["SeniorCitizen"], "Partner": d["Partner"],
                   "Dependents": d["Dependents"], "TenureMonths": tenure,
                   "PhoneService": d["PhoneService"], "MultipleLines": d["MultipleLines"],
                   "InternetService": d["InternetService"], "OnlineSecurity": d["OnlineSecurity"],
                   "OnlineBackup": d["OnlineBackup"], "DeviceProtection": d["DeviceProtection"],
                   "TechSupport": d["TechSupport"], "StreamingTV": d["StreamingTV"],
                   "StreamingMovies": d["StreamingMovies"], "Contract": d["Contract"],
                   "PaperlessBilling": d["PaperlessBilling"], "PaymentMethod": d["PaymentMethod"],
                   "MonthlyCharges": monthly, "TotalCharges": total, "Churn": d["Churn"],
                   "ChurnFlag": churn, "TenureBand": band_tenure(tenure),
                   "MonthlyChargeBand": band_charges(monthly),
                   "ChurnedMonthlyCharges": monthly if churn else 0,
                   "DataQuality": "Valid" if total else "Missing TotalCharges"}
            writer.writerow(out)
            count += 1
    print(f"Wrote {count:,} records to {DEST}")
if __name__ == "__main__":
    main()
