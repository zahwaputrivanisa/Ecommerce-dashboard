import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(layout="wide")

# =========================
# LOAD DATA
# =========================
orders = pd.read_csv("orders_clean.csv")
order_items = pd.read_csv("order_items_clean.csv")
order_reviews = pd.read_csv("order_reviews_clean.csv")
order_payments = pd.read_csv("order_payments_clean.csv")
products = pd.read_csv("products_clean.csv")
translation = pd.read_csv("category_translation_clean.csv")

# =========================
# PREPROCESS
# =========================
orders['order_purchase_timestamp'] = pd.to_datetime(orders['order_purchase_timestamp'])

df = order_items.merge(order_reviews, on="order_id")
df = df.merge(products, on="product_id")
df = df.merge(translation, on="product_category_name")

# =========================
# 🎛️ SIDEBAR (INTERAKTIF)
# =========================
st.sidebar.title("🎛️ Filter Dashboard")

# FILTER TANGGAL
start_date = st.sidebar.date_input(
    "Tanggal Mulai",
    orders['order_purchase_timestamp'].min()
)

end_date = st.sidebar.date_input(
    "Tanggal Akhir",
    orders['order_purchase_timestamp'].max()
)

filtered_orders = orders[
    (orders['order_purchase_timestamp'] >= pd.to_datetime(start_date)) &
    (orders['order_purchase_timestamp'] <= pd.to_datetime(end_date))
]

# FILTER KATEGORI
selected_category = st.sidebar.selectbox(
    "Pilih Kategori Produk",
    df['product_category_name_english'].dropna().unique()
)

df_filtered = df[df['product_category_name_english'] == selected_category]

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
col3.metric("Total Products", products['product_id'].nunique())

st.divider()

# =========================
# 📦 KETERLAMBATAN
# =========================
st.subheader("📦 Keterlambatan Pengiriman")

filtered_orders['order_delivered_customer_date'] = pd.to_datetime(filtered_orders['order_delivered_customer_date'])
filtered_orders['order_estimated_delivery_date'] = pd.to_datetime(filtered_orders['order_estimated_delivery_date'])

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
Sebagian besar pesanan dikirim tepat waktu, namun masih terdapat keterlambatan yang menunjukkan perlunya peningkatan efisiensi logistik.
""")

st.divider()

# =========================
# ⭐ REVIEW PER KATEGORI
# =========================
st.subheader(f"⭐ Review untuk Kategori: {selected_category}")

review_category = df_filtered.groupby('product_category_name_english')['review_score'].mean().sort_values()

fig, ax = plt.subplots()

sns.barplot(x=review_category.values, y=review_category.index, ax=ax)

ax.set_title("Rata-rata Review Kategori")
ax.set_xlabel("Review Score")
ax.set_xlim(0,5)
ax.set_xticks(range(0,6))

st.pyplot(fig)

st.markdown("""
💡 **Insight:**  
Performa kategori dapat berubah tergantung pilihan filter, sehingga penting untuk fokus pada kategori dengan skor rendah.
""")

st.divider()

# =========================
# 👤 RFM SEGMENTATION
# =========================
st.subheader("👤 Segmentasi Pelanggan (RFM)")

rfm_df = filtered_orders.merge(order_payments, on="order_id")

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

# FILTER SEGMENT
selected_segment = st.sidebar.multiselect(
    "Pilih Segment",
    rfm['Segment'].unique(),
    default=rfm['Segment'].unique()
)

rfm_filtered = rfm[rfm['Segment'].isin(selected_segment)]

segment_counts = rfm_filtered['Segment'].value_counts()

fig, ax = plt.subplots()

segment_counts.plot(kind='bar', ax=ax)

ax.set_title("Distribusi Segment Pelanggan")
ax.set_xlabel("Segment")
ax.set_ylabel("Jumlah")

plt.xticks(rotation=0)

st.pyplot(fig)

st.markdown("""
💡 **Insight:**  
Mayoritas pelanggan berada pada segmen *At Risk*, sehingga strategi retensi sangat diperlukan.
""")

st.divider()

# ======================
# FOOTER
# ======================
st.caption("Dashboard E-commerce Brazil | Fundamental Analysis Data Submission 🚀")
