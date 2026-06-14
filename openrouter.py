import os
import requests
from dotenv import load_dotenv

load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

if not OPENROUTER_API_KEY:
    raise ValueError("OPENROUTER_API_KEY is missing. Check your .env file.")
response = requests.get(
    "https://openrouter.ai/api/v1/auth/key",
    headers={"Authorization": f"Bearer {OPENROUTER_API_KEY}"}
)

json_data = response.json()
key_data = json_data.get("data", {})
limit = key_data.get("limit") or 0.0
spent = key_data.get("usage") or 0.0
remaining = key_data.get("limit_remaining") or 0.0
daily = key_data.get("usage_daily") or 0.0
weekly = key_data.get("usage_weekly") or 0.0
monthly = key_data.get("usage_monthly") or 0.0
label = key_data.get("label", "Unknown Key")
expires = key_data.get("expires_at", "No expiration")
md_content = f"""# OpenRouter API Key Usage Report

### Budget & Remaining Balance
* **Total Limit:** ${limit:.2f}
* **Total Spent:** ${spent:.2f}
* **Remaining Balance:** **${remaining:.2f}**

### Recent Spending Breakdown
* **Spent Today:** ${daily:.4f}
* **Spent This Week:** ${weekly:.4f}
* **Spent This Month:** ${monthly:.4f}

### Key Details
* **Key Label:** `{label}`
* **Expiration Date:** {expires}
"""

filename = "api_usage_report.md"
with open(filename, "w", encoding="utf-8") as file:
    file.write(md_content)

print(f"Success! Your usage stats have been saved to {filename}")