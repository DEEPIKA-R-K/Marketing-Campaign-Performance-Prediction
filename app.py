import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_absolute_error, r2_score, accuracy_score, classification_report
import warnings
warnings.filterwarnings("ignore")


st.markdown("""
<style>
    .stButton>button {
        background-color: #c2185b;
        color: white;
        border-radius: 8px;
        border: none;
        padding: 0.5rem 1.5rem;
    }
    .stButton>button:hover { background-color: #880e4f; }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    df = pd.read_csv("combined_df.csv")
    prefix_map = {"NY": "Nykaa", "TI": "Tira", "PU": "Purplle"}
    df["Brand"] = df["Campaign_ID"].str[:2].map(prefix_map).fillna("Unknown")
    df["Profit"] = df["Revenue"] - df["Acquisition_Cost"]
    df["Profit_Loss_Category"] = df["Profit"].apply(lambda x: "Profit" if x > 0 else "Loss")
    return df


@st.cache_resource
def train_models(df):
    feature_exclude = ["Campaign_ID", "Revenue", "Profit", "Profit_Loss_Category", "Brand"]
    categorical_cols = [c for c in df.select_dtypes(include=["object"]).columns if c not in feature_exclude]

    df_enc = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

    X    = df_enc.drop(columns=[c for c in feature_exclude if c in df_enc.columns], errors="ignore")
    y_prof = df_enc["Profit"]
    y_pl   = df_enc["Profit_Loss_Category"]

    X_tr_p, X_te_p, y_tr_p, y_te_p = train_test_split(X, y_prof, test_size=0.2, random_state=42)
    X_tr_pl, X_te_pl, y_tr_pl, y_te_pl = train_test_split(X, y_pl, test_size=0.2, random_state=42, stratify=y_pl)

    m_prof = RandomForestRegressor(n_estimators=30, random_state=42, n_jobs=-1)
    m_prof.fit(X_tr_p, y_tr_p)
    y_pred_p = m_prof.predict(X_te_p)

    m_pl = RandomForestClassifier(n_estimators=10, random_state=42, n_jobs=-1)
    m_pl.fit(X_tr_pl, y_tr_pl)
    y_pred_pl = m_pl.predict(X_te_pl)

    metrics = {
        "profit_mae": mean_absolute_error(y_te_p, y_pred_p),
        "profit_r2":  r2_score(y_te_p, y_pred_p),
        "pl_accuracy": accuracy_score(y_te_pl, y_pred_pl),
        "pl_report":   classification_report(y_te_pl, y_pred_pl, output_dict=True),
    }

    return m_prof, m_pl, X.columns.tolist(), metrics


st.sidebar.title("Campaign Analyzer")
st.sidebar.markdown("**Indian Beauty Brands**  \n`Nykaa · Purplle · Tira`")
st.sidebar.divider()

page = st.sidebar.radio(
    "Navigate",
    ["🏠 Overview", "📊 EDA", "🤖 ML Models", "🔮 Predict"],
)

try:
    df = load_data()
except FileNotFoundError:
    st.error("❌ `combined_df.csv` not found. Place it in the same folder as `app.py` and re-run.")
    st.stop()


if page == "🏠 Overview":
    st.title("🌸 Indian Beauty Brand Campaign Dashboard")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Campaigns", f"{len(df):,}")
    col2.metric("Total Revenue", f"₹{df['Revenue'].sum()/1e7:.2f} Cr")
    col3.metric("Avg ROI", f"{df['ROI'].mean():.2f}x")
    col4.metric("Profitable Campaigns", f"{(df['Profit_Loss_Category']=='Profit').sum():,}")

    st.divider()
    st.subheader("Brand-wise Summary")

    brand_summary = df.groupby("Brand").agg(
        Campaigns=("Campaign_ID", "count"),
        Revenue=("Revenue", "sum"),
        Avg_ROI=("ROI", "mean"),
        Avg_Engagement=("Engagement_Score", "mean"),
        Profit_Count=("Profit_Loss_Category", lambda x: (x == "Profit").sum()),
    ).reset_index()
    brand_summary["Revenue (Cr)"] = (brand_summary["Revenue"] / 1e7).round(2)
    brand_summary["Profit %"] = ((brand_summary["Profit_Count"] / brand_summary["Campaigns"]) * 100).round(1)

    st.dataframe(
        brand_summary[["Brand", "Campaigns", "Revenue (Cr)", "Avg_ROI", "Avg_Engagement", "Profit %"]],
        use_container_width=True, hide_index=True,
    )

    st.divider()
    st.subheader("Raw Data Preview")
    brand_filter = st.selectbox("Filter by Brand", ["All"] + sorted(df["Brand"].unique().tolist()))
    preview = df if brand_filter == "All" else df[df["Brand"] == brand_filter]
    st.dataframe(preview.head(100), use_container_width=True)


elif page == "📊 EDA":
    st.title("📊 Exploratory Data Analysis")

    brand_sel = st.multiselect("Select Brands", df["Brand"].unique().tolist(), default=df["Brand"].unique().tolist())
    filtered = df[df["Brand"].isin(brand_sel)] if brand_sel else df

    tab1, tab2, tab3, tab4 = st.tabs(["Revenue & Profit", "Campaign Types", "Channels", "Correlations"])

    with tab1:
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Revenue Distribution by Brand**")
            fig, ax = plt.subplots(figsize=(6, 4))
            palette = {"Nykaa": "#c2185b", "Purplle": "#7b1fa2", "Tira": "#d84315"}
            sns.boxplot(data=filtered, x="Brand", y="Revenue", palette=palette, ax=ax)
            ax.set_ylabel("Revenue (₹)")
            plt.tight_layout()
            st.pyplot(fig); plt.close()

        with col2:
            st.markdown("**Profit/Loss Count by Brand**")
            pl_counts = filtered.groupby(["Brand", "Profit_Loss_Category"]).size().reset_index(name="Count")
            brands = pl_counts["Brand"].unique()
            x = np.arange(len(brands))
            width = 0.35
            profit_vals = [pl_counts[(pl_counts["Brand"]==b)&(pl_counts["Profit_Loss_Category"]=="Profit")]["Count"].values[0]
                           if len(pl_counts[(pl_counts["Brand"]==b)&(pl_counts["Profit_Loss_Category"]=="Profit")])>0 else 0
                           for b in brands]
            loss_vals   = [pl_counts[(pl_counts["Brand"]==b)&(pl_counts["Profit_Loss_Category"]=="Loss")]["Count"].values[0]
                           if len(pl_counts[(pl_counts["Brand"]==b)&(pl_counts["Profit_Loss_Category"]=="Loss")])>0 else 0
                           for b in brands]
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.bar(x - width/2, profit_vals, width, label="Profit", color="#4caf50")
            ax.bar(x + width/2, loss_vals,   width, label="Loss",   color="#f44336")
            ax.set_xticks(x); ax.set_xticklabels(brands)
            ax.set_ylabel("Count"); ax.legend()
            plt.tight_layout()
            st.pyplot(fig); plt.close()

        st.markdown("**ROI vs Profit (scatter)**")
        fig, ax = plt.subplots(figsize=(10, 4))
        for brand, grp in filtered.groupby("Brand"):
            ax.scatter(grp["ROI"], grp["Profit"], alpha=0.3, label=brand,
                       color={"Nykaa": "#c2185b", "Purplle": "#7b1fa2", "Tira": "#d84315"}.get(brand, "grey"), s=10)
        ax.set_xlabel("ROI"); ax.set_ylabel("Profit (₹)"); ax.legend()
        plt.tight_layout()
        st.pyplot(fig); plt.close()

    with tab2:
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Campaign Type Distribution**")
            ct_counts = filtered["Campaign_Type"].value_counts()
            fig, ax = plt.subplots(figsize=(5, 5))
            ax.pie(ct_counts.values, labels=ct_counts.index, autopct="%1.1f%%",
                   colors=sns.color_palette("Set2", len(ct_counts)))
            plt.tight_layout()
            st.pyplot(fig); plt.close()

        with col2:
            st.markdown("**Avg Profit by Campaign Type**")
            ct_prof = filtered.groupby("Campaign_Type")["Profit"].mean().sort_values(ascending=True)
            fig, ax = plt.subplots(figsize=(5, 5))
            ct_prof.plot(kind="barh", ax=ax, color="#c2185b")
            ax.set_xlabel("Avg Profit (₹)")
            plt.tight_layout()
            st.pyplot(fig); plt.close()

        st.markdown("**Avg Engagement Score by Campaign Type & Brand**")
        pivot = filtered.pivot_table(values="Engagement_Score", index="Campaign_Type", columns="Brand", aggfunc="mean")
        fig, ax = plt.subplots(figsize=(10, 4))
        pivot.plot(kind="bar", ax=ax, color=["#c2185b", "#7b1fa2", "#d84315"])
        ax.set_ylabel("Avg Engagement Score"); ax.legend(title="Brand")
        plt.xticks(rotation=30); plt.tight_layout()
        st.pyplot(fig); plt.close()

    with tab3:
        channel_df = filtered.copy()
        channel_df["Channel_Split"] = channel_df["Channel_Used"].str.split(", ")
        channel_exploded = channel_df.explode("Channel_Split")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Top 10 Channels by Usage**")
            top_channels = channel_exploded["Channel_Split"].value_counts().head(10)
            fig, ax = plt.subplots(figsize=(5, 5))
            top_channels.plot(kind="barh", ax=ax, color="#7b1fa2")
            ax.set_xlabel("Count")
            plt.tight_layout()
            st.pyplot(fig); plt.close()

        with col2:
            st.markdown("**Avg Profit by Channel**")
            ch_prof = channel_exploded.groupby("Channel_Split")["Profit"].mean().sort_values(ascending=True).tail(10)
            fig, ax = plt.subplots(figsize=(5, 5))
            ch_prof.plot(kind="barh", ax=ax, color="#d84315")
            ax.set_xlabel("Avg Profit (₹)")
            plt.tight_layout()
            st.pyplot(fig); plt.close()

    with tab4:
        st.markdown("**Correlation Heatmap (Numerical Features)**")
        num_cols = ["Duration", "Impressions", "Clicks", "Leads", "Conversions",
                    "Revenue", "Acquisition_Cost", "ROI", "Engagement_Score", "Conversion_Rate", "Profit"]
        corr = filtered[num_cols].corr()
        fig, ax = plt.subplots(figsize=(10, 7))
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdPu", ax=ax, linewidths=0.5)
        plt.tight_layout()
        st.pyplot(fig); plt.close()

elif page == "🤖 ML Models":
    st.title("🤖 Machine Learning Models")

    with st.spinner("Training models... (cached after first run)"):
        m_prof, m_pl, feature_cols, metrics = train_models(df)

    st.success("✅ Models trained successfully!")
    st.divider()

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 💰 Profit Predictor")
        st.markdown("**Model:** Random Forest Regressor (30 trees)")
        st.metric("MAE", f"₹{metrics['profit_mae']:,.0f}")
        st.metric("R² Score", f"{metrics['profit_r2']:.4f}")

    with col2:
        st.markdown("### 🏷️ Profit/Loss Classifier")
        st.markdown("**Model:** Random Forest Classifier (10 trees)")
        st.metric("Accuracy", f"{metrics['pl_accuracy']*100:.2f}%")
        report = metrics["pl_report"]
        st.metric("F1 (Profit)", f"{report.get('Profit', {}).get('f1-score', 0):.4f}")

    st.divider()
    st.subheader("Feature Importances")
    model_choice = st.selectbox("Choose model", ["Profit Regressor", "Profit/Loss Classifier"])
    chosen_model = m_prof if model_choice == "Profit Regressor" else m_pl
    importances = pd.Series(chosen_model.feature_importances_, index=feature_cols).sort_values(ascending=True).tail(15)

    fig, ax = plt.subplots(figsize=(8, 5))
    importances.plot(kind="barh", ax=ax, color="#c2185b")
    ax.set_xlabel("Importance")
    ax.set_title(f"Top 15 Features — {model_choice}")
    plt.tight_layout()
    st.pyplot(fig); plt.close()

    st.divider()
    st.subheader("Profit/Loss Classification Report")
    report_df = pd.DataFrame(metrics["pl_report"]).transpose().round(4)
    st.dataframe(report_df, use_container_width=True)


elif page == "🔮 Predict":
    st.title("🔮 Predict Profit & Outcome")
    

    with st.spinner("Loading models..."):
        m_prof, m_pl, feature_cols, _ = train_models(df)

    st.divider()

    col1, col2 = st.columns(2)

    with col1:
        campaign_type    = st.selectbox("Campaign Type", sorted(df["Campaign_Type"].dropna().unique()))
        target_audience  = st.selectbox("Target Audience", sorted(df["Target_Audience"].dropna().unique()))
        language         = st.selectbox("Language", sorted(df["Language"].dropna().unique()))
        customer_segment = st.selectbox("Customer Segment", sorted(df["Customer_Segment"].dropna().unique()))

    with col2:
        duration         = st.slider("Duration (days)", int(df["Duration"].min()), int(df["Duration"].max()), 15)
        impressions      = st.number_input("Impressions", min_value=10000, max_value=100000, value=50000, step=1000)
        clicks           = st.number_input("Clicks", min_value=100, max_value=50000, value=5000, step=100)
        leads            = st.number_input("Leads", min_value=10, max_value=10000, value=500, step=50)
        conversions      = st.number_input("Conversions", min_value=1, max_value=5000, value=200, step=10)
        roi              = st.number_input("ROI", min_value=0.0, max_value=20.0, value=3.5, step=0.1)
        engagement_score = st.slider("Engagement Score", 0.0, 10.0, 5.0, 0.1)
        acquisition_cost = st.number_input("Acquisition Cost (₹)", min_value=1000, max_value=500000, value=50000, step=1000)

    if st.button("🚀 Predict Now"):
        input_data = {
            "Duration": duration,
            "Impressions": impressions,
            "Clicks": clicks,
            "Leads": leads,
            "Conversions": conversions,
            "ROI": roi,
            "Engagement_Score": engagement_score,
            "Conversion_Rate": conversions / clicks if clicks > 0 else 0.0,
            "Campaign_Type": campaign_type,
            "Target_Audience": target_audience,
            "Language": language,
            "Customer_Segment": customer_segment,
        }

        feature_exclude = ["Campaign_ID", "Revenue", "Profit", "Profit_Loss_Category", "Brand", "Acquisition_Cost"]
        categorical_cols_train = [c for c in df.select_dtypes(include=["object"]).columns if c not in feature_exclude]

        input_df = pd.DataFrame([input_data])
        combined_for_encode = pd.concat([
            df.drop(columns=["Campaign_ID", "Revenue", "Profit", "Profit_Loss_Category", "Brand", "Acquisition_Cost"], errors="ignore"),
            input_df
        ], ignore_index=True)
        combined_encoded = pd.get_dummies(combined_for_encode, columns=categorical_cols_train, drop_first=True)
        X_input = combined_encoded.tail(1).reindex(columns=feature_cols, fill_value=0)

        prof_pred  = m_prof.predict(X_input)[0]
        pl_pred    = m_pl.predict(X_input)[0]
        pl_prob    = m_pl.predict_proba(X_input)[0]
        pl_classes = m_pl.classes_

        st.divider()
        st.subheader("📊 Prediction Results")

        c1, c2 = st.columns(2)
        c1.metric("💰 Predicted Revenue", f"₹{prof_pred:,.0f}")
        c2.metric("🏷️ Outcome", pl_pred)

        st.markdown("**Profit/Loss Probability**")
        prob_df = pd.DataFrame({"Category": pl_classes, "Probability": pl_prob})
        fig, ax = plt.subplots(figsize=(5, 2))
        colors = ["#4caf50" if c == "Profit" else "#f44336" for c in prob_df["Category"]]
        ax.barh(prob_df["Category"], prob_df["Probability"], color=colors)
        ax.set_xlim(0, 1)
        ax.set_xlabel("Probability")
        for i, v in enumerate(prob_df["Probability"]):
            ax.text(v + 0.01, i, f"{v:.1%}", va="center")
        plt.tight_layout()
        st.pyplot(fig); plt.close()


st.sidebar.divider()
st.sidebar.caption("Built with Streamlit · Scikit-learn  \nNykaa · Purplle · Tira Dataset")