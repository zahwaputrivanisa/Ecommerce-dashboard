import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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
# PREPROCESS
# =========================
orders['order_purchase_timestamp'] = pd.to_datetime(orders['order_purchase_timestamp'])
orders['order_delivered_customer_date'] = pd.to_datetime(orders['order_delivered_customer_date'])
orders['order_estimated_delivery_date'] = pd.to_datetime(orders['order_estimated_delivery_date'])

# =========================
# SIDEBAR FILTER
# =========================
st.sidebar.header("🔍 Filter Data")

# FILTER TANGGAL
min_date = orders['order_purchase_timestamp'].min()
max_date = orders['order_purchase_timestamp'].max()

date_range = st.sidebar.date_input(
    "Pilih Rentang Tanggal",
    [min_date, max_date]
)

# FILTER REVIEW
min_review = st.sidebar.slider("Minimum Review Score", 1, 5, 1)
max_review = st.sidebar.slider("Maximum Review Score", 1, 5, 5)

# FILTER SEGMENT
segments = ['Loyal Customer', 'Recent Customer', 'Frequent Customer', 'At Risk']

segment_filter = st.sidebar.multiselect(
    "Pilih Segment Pelanggan",
    options=segments,
    default=segments
)

# =========================
# FILTER DATA
# =========================
filtered_orders = orders[
    (orders['order_purchase_timestamp'] >= pd.to_datetime(date_range[0])) &
    (orders['order_purchase_timestamp'] <= pd.to_datetime(date_range[1]))
]

# MERGE FILTERED DATA
filtered_df = order_items.merge(filtered_orders, on="order_id", how="inner")
filtered_df = filtered_df.merge(order_reviews, on="order_id", how="left")
filtered_df = filtered_df.merge(products, on="product_id", how="left")

# FILTER REVIEW KE SEMUA VISUAL
filtered_df = filtered_df[
    (filtered_df['review_score'] >= min_review) &
    (filtered_df['review_score'] <= max_review)
]

# =========================
# TITLE
# =========================
st.title("📊 E-commerce Dashboard")

# =========================
# KPI
# =========================
col1, col2, col3 = st.columns(3)

col1.metric("Total Orders", filtered_orders['order_id'].nunique())
col2.metric("Total Customer", filtered_orders['customer_id'].nunique())
col3.metric("Total Products", filtered_df['product_id'].nunique())

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
# ⭐ REVIEW PER KATEGORI
# =========================
st.subheader("⭐ Review per Kategori Produk")

review_category = (
    filtered_df.groupby('product_category_name')['review_score']
    .mean()
    .sort_values()
)

lowest = review_category.head(5)
highest = review_category.tail(5)

combined = pd.concat([lowest, highest]).sort_values()

fig, ax = plt.subplots()

sns.barplot(x=combined.values, y=combined.index, ax=ax)

ax.set_title("Top 5 & Bottom 5 Review Kategori")
ax.set_xlabel("Rata-rata Review Score")
ax.set_ylabel("Kategori")

ax.set_xlim(0, 5)
ax.set_xticks(range(0, 6))

plt.tight_layout()
st.pyplot(fig)

st.markdown(f"""
💡 **Insight:**  
Kategori dengan review terendah adalah **{lowest.index[0]}**,  
sedangkan kategori dengan review tertinggi adalah **{highest.index[-1]}**.
""")

st.divider()

# =========================
# 👤 RFM SEGMENTATION
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

rfm['R_score'] = pd.qcut(rfm['Recency'], 4, labels=[4,3,2,1])
rfm['F_score'] = pd.qcut(rfm['Frequency'].rank(method='first'), 4, labels=[1,2,3,4])
rfm['M_score'] = pd.qcut(rfm['Monetary'], 4, labels=[1,2,3,4])

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

# FILTER SEGMENT (INTERAKTIF)
rfm_filtered = rfm[rfm['Segment'].isin(segment_filter)]

segment_counts = rfm_filtered['Segment'].value_counts()

fig, ax = plt.subplots()

segment_counts.plot(kind='bar', ax=ax)

ax.set_title("Distribusi Segment Pelanggan")
ax.set_xlabel("Segment")
ax.set_ylabel("Jumlah")

plt.xticks(rotation=0)

plt.tight_layout()
st.pyplot(fig)

st.markdown("""
💡 **Insight:**  
Distribusi pelanggan paling banyak berada pada posisi At Risk.
""")

st.divider()

# FOOTER
# ======================
st.caption("Dashboard E-commerce Brazil | Fundamental Analysis Data Submission 🚀")
