import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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
# MERGE DATA
# =========================
df = order_items.merge(order_reviews, on="order_id")
df = df.merge(products, on="product_id")
df = df.merge(translation, on="product_category_name")

# =========================
# TITLE
# =========================
st.title("📊 E-commerce Dashboard")

# =========================
# KPI
# =========================
col1, col2, col3 = st.columns(3)

col1.metric("Total Orders", orders['order_id'].nunique())
col2.metric("Total Customer", orders['customer_id'].nunique())
col3.metric("Total Products", products['product_id'].nunique())

st.divider()

# =========================
# 📦 KETERLAMBATAN
# =========================
st.subheader("📦 Keterlambatan Pengiriman")

orders['order_delivered_customer_date'] = pd.to_datetime(orders['order_delivered_customer_date'])
orders['order_estimated_delivery_date'] = pd.to_datetime(orders['order_estimated_delivery_date'])

orders['delay'] = (
    orders['order_delivered_customer_date'] - 
    orders['order_estimated_delivery_date']
).dt.days

orders['is_late'] = orders['delay'] > 0

delay_counts = orders['is_late'].value_counts()

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
st.subheader("⭐ Review per Kategori Produk")

review_category = df.groupby('product_category_name_english')['review_score'].mean().sort_values()

lowest = review_category.head(5)
highest = review_category.tail(5)

combined = pd.concat([lowest, highest]).sort_values()

fig, ax = plt.subplots()

sns.barplot(x=combined.values, y=combined.index, ax=ax)

ax.set_title("Kategori Review Terendah vs Tertinggi")
ax.set_xlabel("Rata-rata Review Score")
ax.set_ylabel("Kategori Produk")

ax.set_xlim(0, 5)
ax.set_xticks(range(0,6))

plt.tight_layout()
st.pyplot(fig)

st.markdown("""
💡 **Insight:**  
Terdapat perbedaan signifikan antara kategori dengan review terendah dan tertinggi, yang menunjukkan adanya variasi kualitas produk atau layanan antar kategori.
""")

st.divider()

# =========================
# 👤 RFM SEGMENTATION
# =========================
st.subheader("👤 Segmentasi Pelanggan (RFM)")

rfm_df = orders.merge(order_payments, on="order_id")

rfm_df['order_purchase_timestamp'] = pd.to_datetime(rfm_df['order_purchase_timestamp'])

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

# COUNT & SORT
segment_counts = rfm['Segment'].value_counts()

fig, ax = plt.subplots()

segment_counts.plot(kind='bar', ax=ax)

ax.set_title("Distribusi Segment Pelanggan")
ax.set_xlabel("Segment")
ax.set_ylabel("Jumlah Pelanggan")

plt.xticks(rotation=0)

plt.tight_layout()
st.pyplot(fig)

st.markdown("""
💡 **Insight:**  
Mayoritas pelanggan berada pada segmen *At Risk*, yang menunjukkan potensi churn yang tinggi, sementara jumlah pelanggan loyal masih relatif sedikit.
""")

st.divider()

# ======================
# FOOTER
# ======================
st.caption("Dashboard E-commerce Brazil | Fundamental Data Analysis Submission 🚀")
