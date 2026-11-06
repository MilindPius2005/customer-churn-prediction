# Customer Churn & Revenue Risk Analytics
## What is implemented
- Full original **7,043-row IBM Telco** dataset retained at repository root.
- A cleaned, Power BI-ready customer-level CSV with churn flags, tenure bands, charge bands, and a churned-customer monthly-charge metric.
- A reproducible **Python standard-library ETL** script: `python analytics/prepare_powerbi.py`.
- An interactive self-contained analytics dashboard: `analytics/retention_dashboard.html`.
- Ready-to-paste **Power BI DAX measures**: `analytics/powerbi_measures.dax`.
- Excel dashboard (distributed separately as `Customer_Churn_Excel_Dashboard.xlsx`, add to this folder when downloaded).

## Verified dataset metrics
| Metric | Actual dataset value |
|---|---:|
| Records | 7,043 |
| Churned | 1,869 |
| Observed churn rate | 26.54% |
| Monthly charges, all customers | $456,116.60 |
| Monthly charges of churned customers | $139,130.85 |
| TotalCharges missing | 11 |
| Month-to-month customers / churned | 3,875 / 1,655 |

## Power BI report (finish in Power BI Desktop)
1. Home > Get Data > Text/CSV > `analytics/data/customers_powerbi.csv`. Set imported table name to **Customers**.
2. In Power Query, type `MonthlyCharges`, `TotalCharges`, and `ChurnedMonthlyCharges` as decimals; `TenureMonths` and `ChurnFlag` as whole numbers. Empty TotalCharges values are null, NOT zero.
3. Create measures in `analytics/powerbi_measures.dax`, one measure per line.
4. **Executive overview**: Cards for Customers, Churned Customers, Churn Rate, Churned Monthly Charges. Chart churn rate by Contract.
5. **Driver analysis**: bar charts Churn Rate by TenureBand, InternetService, PaymentMethod, TechSupport; use Customer counts as tooltips.
6. **Revenue exposure**: Churned Monthly Charges by Contract and InternetService. Add Contract/TenureBand slicers.
7. Verify: 7,043 customers, 1,869 churned, 26.54% churn rate, $139,130.85 churned monthly charges. Save as `Customer_Churn_Retention.pbix` using Power BI Desktop.

**Interpretation guardrail:** The dataset has churn labels and monthly charges; it has no observed cancellation timestamp. Churned Monthly Charges is an exposure proxy, not recovered revenue, causal attribution, or validated forecast.

**Original model:** The source notebook builds a Keras ANN and the existing README reports approximately **78%** test accuracy. This report analyzes the observed churn target; it does **not** claim those Power BI visuals are derived from model probability estimates.

## Portfolio walkthrough
Situation: customer retention analytics requires a consolidated view of churn drivers. Task: build a reproducible customer segmentation and reporting layer. Action: clean and enrich the source in Python, define consistent DAX KPIs, create an Excel dashboard and interactive reporting view. Result: a 7,043-customer reporting model with auditable churn segmentation and quantified monthly charges linked to churned accounts.
