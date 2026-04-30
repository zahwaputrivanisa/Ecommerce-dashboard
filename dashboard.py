import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# =========================
# CONFIG
# =========================
st.set_page_config(page_title="E-commerce Dashboard", layout="wide")

# =========================
# LOAD DATA
# =========================
@st.cache_data
def load_data():
    orders = pd.read_csv("orders_clean.csv")
    order_items = pd.read_csv("order_items_clean.csv")
    order_reviews = pd.read_csv("order_reviews_clean.csv")
    order_payments = pd.read_csv("order_payments_clean.csv")
    products = pd.read_csv("products_clean.csv")
    return orders, order_items, order_reviews, order_payments, products

orders, order_items, order_reviews, order_payments, products = load_data()

# =========================
# PREPROCESSING
# =========================
orders['order_purchase_timestamp'] = pd.to_datetime(orders['order_purchase_timestamp'])
orders['order_delivered_customer_date'] = pd.to_datetime(orders['order_delivered_customer_date'])
orders['order_estimated_delivery_date'] = pd.to_datetime(orders['order_estimated_delivery_date'])

# =========================
# MERGE DATA (NO TRANSLATION)
# =========================
df = order_items.merge(order_reviews, on="order_id", how="left")
df = df.merge(products, on="product_id", how="left")

# =========================
# SIDEBAR FILTER (INTERAKTIF)
# =========================
st.sidebar.header("🔎 Filter Data")

start_date = st.sidebar.date_input(
    "Start Date",
    orders['order_purchase_timestamp'].min()
)

end_date = st.sidebar.date_input(
    "End Date",
    orders['order_purchase_timestamp'].max()
)

filtered_orders = orders[
    (orders['order_purchase_timestamp'] >= pd.to_datetime(start_date)) &
    (orders['order_purchase_timestamp'] <= pd.to_datetime(end_date))
]

# =========================
# TITLE
# =========================
st.title("📊 E-commerce Dashboard")

# =========================
# KPI (PAKAI DATA FILTERED)
# =========================
col1, col2, col3 = st.columns(3)

col1.metric("Total Orders", filtered_orders['order_id'].nunique())
col2.metric("Total Customer", filtered_orders['customer_id'].nunique())
col3.metric("Total Products", products['product_id'].nunique())

st.divider()

# =========================
# 📦 KETERLAMBATAN
# =========================
st.subheader("📦 Keterlambatan Pengiriman")

filtered_orders['delay'] = (
    filtered_orders['order_delivered_customer_date'] - 
    filtered_orders['order_estimated_delivery_date']
).dt.days

filtered_orders['is_late'] = filtered_orders['delay'] > 0

delay_counts = filtered_orders['is_late'].value_counts()

fig, ax = plt.subplots()
delay_counts.plot(kind='bar', ax=ax)

ax.set_xticklabels(['On Time', 'Late'], rotation=0)
ax.set_title("Distribusi Keterlambatan")
ax.set_ylabel("Jumlah Order")

st.pyplot(fig)

st.markdown("""
💡 **Insight:**  
Sebagian besar pesanan dikirim tepat waktu, namun masih terdapat keterlambatan.
""")

st.divider()

# =========================
# ⭐ REVIEW PER KATEGORI (FIX)
# =========================
st.subheader("⭐ Review per Kategori Produk")

review_category = (
    df.groupby('product_category_name')['review_score']
    .mean()
    .sort_values()
)

lowest = review_category.head(5)
highest = review_category.tail(5)

combined = pd.concat([lowest, highest]).sort_values()

fig, ax = plt.subplots(figsize=(8,5))

sns.barplot(
    x=combined.values,
    y=combined.index,
    ax=ax
)

ax.set_title("Top 5 & Bottom 5 Kategori Review")
ax.set_xlabel("Rata-rata Review Score")
ax.set_ylabel("Kategori Produk")

ax.set_xlim(0,5)
ax.set_xticks(range(0,6))

plt.tight_layout()
st.pyplot(fig)

st.markdown(f"""
💡 **Insight:**  
Kategori dengan review terendah adalah **{lowest.index[0]}**,  
sedangkan kategori dengan review tertinggi adalah **{highest.index[-1]}**.
""")

st.divider()

# =========================
# 👤 RFM SEGMENTATION (INTERAKTIF)
# =========================
st.subheader("👤 Segmentasi Pelanggan (RFM)")

rfm_df = filtered_orders.merge(order_payments, on="order_id", how="left")

reference_date = rfm_df['order_purchase_timestamp'].max()

rfm = rfm_df.groupby('customer_id').agg({
    'order_purchase_timestamp': lambda x: (reference_date - x.max()).days,
    'order_id': 'count',
    'payment_value': 'sum'
})

rfm.columns = ['Recency', 'Frequency', 'Monetary']

# SCORING
rfm['R_score'] = pd.qcut(rfm['Recency'], 4, labels=[4,3,2,1])
rfm['F_score'] = pd.qcut(rfm['Frequency'].rank(method='first'), 4, labels=[1,2,3,4])
rfm['M_score'] = pd.qcut(rfm['Monetary'], 4, labels=[1,2,3,4])

# SEGMENT
def segment_customer(row):
    if row['R_score'] == 4 and row['F_score'] >= 3:
        return 'Loyal Customer'
    elif row['R_score'] >= 3 and row['F_score'] <= 2:
        return 'Recent Customer'
    elif row['R_score'] <= 2 and row['F_score'] >= 3:
        return 'Frequent Customer'
    else:
        return 'At Risk'

rfm['Segment'] = rfm.apply(segment_customer, axis=1)

# FILTER SEGMENT
segment_filter = st.sidebar.multiselect(
    "Pilih Segment Pelanggan",
    options=rfm['Segment'].unique(),
    default=rfm['Segment'].unique()
)

rfm_filtered = rfm[rfm['Segment'].isin(segment_filter)]
segment_counts = rfm_filtered['Segment'].value_counts()

fig, ax = plt.subplots()
segment_counts.plot(kind='bar', ax=ax)

ax.set_title("Distribusi Segment Pelanggan")
ax.set_xlabel("Segment")
ax.set_ylabel("Jumlah Pelanggan")

plt.xticks(rotation=0)
st.pyplot(fig)

st.markdown("""
💡 **Insight:**  
Mayoritas pelanggan berada pada segmen At Risk, menunjukkan potensi churn tinggi.
""")

st.divider()

# FOOTER
# ======================
st.caption("Dashboard E-commerce Brazil | Fundamental Analysis Data Submission 🚀")
