import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer, util
from datetime import datetime, timedelta
from fastapi.middleware.cors import CORSMiddleware
import re

app = FastAPI()

# -------------------------------
# 1️⃣ CORS configuration
# -------------------------------
origins = [
    "http://localhost:5173",
    "https://dashboard-zzun.onrender.com/" # React + Vite frontend
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------------
# 2️⃣ Load CSVs and preprocess
# -------------------------------
df = pd.read_csv("Cart.csv")
df = df.iloc[:, :6]
df.columns = ["Product", "Quantity", "Base Price", "Total Price", "User", "Time"]
df = df[df["Product"] != "Product"]  # remove header rows
df["Time"] = pd.to_datetime(df["Time"], errors="coerce")

# Parse numeric price
def parse_price(price_str):
    if pd.isna(price_str):
        return 0.0
    price_str = (
        str(price_str)
        .replace("€", "")
        .replace("$", "")
        .replace("¥", "")
        .replace("₽", "")
        .replace(",", "")
    )
    try:
        return float(price_str)
    except:
        return 0.0

df["base_price_num"] = df["Base Price"].apply(parse_price)

# Stock & Price drop CSVs
try:
    stock_df = pd.read_csv("stock.csv")
except FileNotFoundError:
    stock_df = pd.DataFrame(columns=["Product"])

try:
    price_drop_df = pd.read_csv("price_drop.csv")
except FileNotFoundError:
    price_drop_df = pd.DataFrame(columns=["Product"])

# -------------------------------
# 3️⃣ Load embeddings
# -------------------------------
model = SentenceTransformer("all-MiniLM-L6-v2")

def make_text(row):
    return f"{row['Product']} | Quantity: {row['Quantity']} | Price: {row['Base Price']} | Time: {row['Time']}"

df["embedding_text"] = df.apply(make_text, axis=1)
df["embedding"] = df["embedding_text"].apply(
    lambda x: model.encode(x, convert_to_tensor=True)
)

# -------------------------------
# 4️⃣ Price and date synonyms
# -------------------------------
expensive_synonyms = ["expensive", "high priced", "costly", "high price", "premium"]
cheap_synonyms = ["cheap", "low priced", "inexpensive", "affordable", "budget"]

last_month_synonyms = ["last month", "previous month", "recently", "lately"]
this_month_synonyms = ["this month", "current month", "now"]

# -------------------------------
# 5️⃣ Helpers
# -------------------------------
def check_promotion(time_val):
    """Check if product is older than 2 months"""
    if pd.isna(time_val):
        return False
    try:
        cutoff = datetime.today() - timedelta(days=60)
        return time_val < cutoff
    except Exception:
        return False

def check_stock(product):
    return product in stock_df.get("Product", []).values

def check_price_drop(product):
    return product in price_drop_df.get("Product", []).values

# -------------------------------
# 6️⃣ API request model
# -------------------------------
class QueryRequest(BaseModel):
    user: str
    query: str

# -------------------------------
# 7️⃣ API endpoint
# -------------------------------
@app.post("/check_cart")
async def check_cart(req: QueryRequest):
    user_cart = df[df["User"] == req.user].copy()
    if user_cart.empty:
        return {"message": f"No cart found for {req.user}"}

    query_lower = req.query.lower()
    today = datetime.today()

    # -------- Date filtering --------
    date_filtered = user_cart.copy()

    # 1️⃣ Explicit month names
    month_match = re.search(
        r"\b(January|February|March|April|May|June|July|August|September|October|November|December)\b",
        req.query,
        re.I,
    )
    if month_match:
        month_num = datetime.strptime(month_match.group(0), "%B").month
        date_filtered = date_filtered[
            (date_filtered["Time"].dt.month == month_num)
            & (date_filtered["Time"].dt.year == today.year)
        ]

    # 2️⃣ Relative month phrases
    for phrase in last_month_synonyms:
        if phrase in query_lower:
            first_day_this_month = today.replace(day=1)
            last_month_end = first_day_this_month - timedelta(days=1)
            last_month_start = last_month_end.replace(day=1)
            date_filtered = date_filtered[
                (date_filtered["Time"] >= last_month_start)
                & (date_filtered["Time"] <= last_month_end)
            ]
            break

    for phrase in this_month_synonyms:
        if phrase in query_lower:
            start_date = today.replace(day=1)
            date_filtered = date_filtered[
                (date_filtered["Time"] >= start_date)
                & (date_filtered["Time"] <= today)
            ]
            break

    # 3️⃣ Last N months
    month_range_match = re.search(r"last (\d+) months?", query_lower)
    if month_range_match:
        num_months = int(month_range_match.group(1))
        first_day_current_month = today.replace(day=1)
        start_date = (
            first_day_current_month - pd.DateOffset(months=num_months)
        ).replace(day=1)
        end_date = first_day_current_month - pd.DateOffset(days=1)
        date_filtered = date_filtered[
            (date_filtered["Time"] >= start_date) & (date_filtered["Time"] <= end_date)
        ]

    if date_filtered.empty:
        return {"message": "No matching items found for the given date."}

    # -------- Price filtering --------
    price_filtered = date_filtered.copy()
    price_intent = None

    # Keyword-based price intent
    if any(phrase in query_lower for phrase in expensive_synonyms):
        price_intent = "expensive"
    elif any(phrase in query_lower for phrase in cheap_synonyms):
        price_intent = "cheap"

    # Numeric price filtering
    price_match = re.search(
        r"(above|over|greater than|more than|below|under|less than)\s*\$?(\d+(\.\d+)?)",
        query_lower,
    )
    if price_match:
        operator = price_match.group(1)
        amount = float(price_match.group(2))
        if operator in ["above", "over", "greater than", "more than"]:
            price_filtered = price_filtered[price_filtered["base_price_num"] > amount]
        else:
            price_filtered = price_filtered[price_filtered["base_price_num"] < amount]

    # Apply keyword-based price sorting
    if price_intent == "expensive":
        price_filtered = price_filtered.sort_values(
            by="base_price_num", ascending=False
        ).head(5)
    elif price_intent == "cheap":
        price_filtered = price_filtered.sort_values(
            by="base_price_num", ascending=True
        ).head(5)

    # If price intent or numeric price exists, return top items directly
    if price_intent or price_match:
        results = []
        for _, row in price_filtered.iterrows():
            results.append(
                {
                    "product": row["Product"],
                    "quantity": row["Quantity"],
                    "base_price": row["Base Price"],
                    "total_price": row["Total Price"],
                    "time": row["Time"].strftime("%Y-%m-%d %H:%M:%S"),
                    "promotion": check_promotion(row["Time"]),
                    "stock_available": check_stock(row["Product"]),
                    "price_drop": check_price_drop(row["Product"]),
                    "score": 0.0,
                }
            )
        if results:
            return {"results": results}

    # -------- Semantic search --------
    query_embedding = model.encode(req.query, convert_to_tensor=True)
    results = []
    for _, row in price_filtered.iterrows():
        score = float(util.cos_sim(query_embedding, row["embedding"]))
        if score > 0.25:  # similarity threshold
            results.append(
                {
                    "product": row["Product"],
                    "quantity": row["Quantity"],
                    "base_price": row["Base Price"],
                    "total_price": row["Total Price"],
                    "time": row["Time"].strftime("%Y-%m-%d %H:%M:%S"),
                    "promotion": check_promotion(row["Time"]),
                    "stock_available": check_stock(row["Product"]),
                    "price_drop": check_price_drop(row["Product"]),
                    "score": round(score, 2),
                }
            )

    # Fallback: top 5 recent items
    if not results:
        top_rows = price_filtered.sort_values(by="Time", ascending=False).head(5)
        for _, row in top_rows.iterrows():
            results.append(
                {
                    "product": row["Product"],
                    "quantity": row["Quantity"],
                    "base_price": row["Base Price"],
                    "total_price": row["Total Price"],
                    "time": row["Time"].strftime("%Y-%m-%d %H:%M:%S"),
                    "promotion": check_promotion(row["Time"]),
                    "stock_available": check_stock(row["Product"]),
                    "price_drop": check_price_drop(row["Product"]),
                    "score": 0.0,
                }
            )

    return {"results": results}
