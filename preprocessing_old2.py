import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from imblearn.over_sampling import SMOTE
from collections import Counter
import streamlit as st
from fpdf import FPDF
import time
import warnings
warnings.filterwarnings("ignore")

def preprocess_df(df, skip_smote=False):
    with st.spinner("Preprocessing in progress... please wait"):
        summary = []
        drop_cols = ['customer_id']
        df = df.drop(columns=[col for col in drop_cols if col in df.columns], errors='ignore')
        summary.append(f"Dropped unnecessary columns: {drop_cols}")

        if 'is_anomaly' not in df.columns:
            raise KeyError("Expected is_anomaly label column")

        st.write("Handling missing values...")
        missing_ratio = df.isnull().mean()
        cols_to_drop = missing_ratio[missing_ratio > 0.5].index.tolist()
        df = df.drop(columns=cols_to_drop, errors='ignore')
        df = df.fillna(df.median(numeric_only=True))
        summary.append(f"Dropped columns with > 50% missing: {cols_to_drop}")
        summary.append(f"Filled remaining missing values with median")

        st.write("Encoding categorical features...")
        cat_cols = df.select_dtypes(include=['object']).columns.tolist()
        if cat_cols:
            enc = OneHotEncoder(sparse_output=True, handle_unknown='ignore')
            ohe = enc.fit_transform(df[cat_cols])
            ohe_df = pd.DataFrame.sparse.from_spmatrix(ohe, columns=enc.get_feature_names_out(cat_cols), index=df.index)
            df = pd.concat([df.drop(columns=cat_cols), ohe_df], axis=1)
            st.session_state.encoder = enc
            summary.append(f"Applied one-hot encoding to: {cat_cols}")

        # Skip multicollinearity check for demo to save time
        summary.append("Skipped multicollinearity check for faster preprocessing")

        st.write("Balancing classes with SMOTE (if enabled)...")
        class_dist_before = dict(Counter(df['is_anomaly']))
        summary.append(f"Class distribution before balancing: {class_dist_before}")

        if not skip_smote:
            X = df.drop(columns=['is_anomaly'])
            y = df['is_anomaly']
            smote = SMOTE(random_state=42, k_neighbors=3, sampling_strategy=0.5)  # Limit oversampling
            X_resampled, y_resampled = smote.fit_resample(X, y)
            df = pd.concat([pd.DataFrame(X_resampled, columns=X.columns), pd.Series(y_resampled, name='is_anomaly')], axis=1)
            class_dist_after = dict(Counter(df['is_anomaly']))
            summary.append(f"Class distribution after SMOTE: {class_dist_after}")
        else:
            summary.append("Skipped SMOTE for faster preprocessing")

        st.write("Generating preprocessing summary...")
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        pdf.cell(200, 10, txt="Preprocessing Summary Report", ln=True, align='C')
        pdf.ln(10)
        for line in summary:
            pdf.multi_cell(0, 10, line)
        pdf.output("preprocessing_summary.pdf")

        with open("preprocessing_summary.pdf", "rb") as f:
            st.download_button(label="Download Preprocessing Summary PDF", data=f, file_name="preprocessing_summary.pdf", mime="application/pdf")
        time.sleep(1)
    
    st.success("Data Preprocessing completed, proceed with Model Training")
    return df
