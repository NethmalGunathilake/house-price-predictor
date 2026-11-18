from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="House Price Predictor", page_icon="🏠")

# ---------- 1. Load the saved model and typical values ----------
BASE = Path(__file__).parent


@st.cache_resource
def load_files():
    model = joblib.load(BASE / "models" / "gb_model.pkl")
    columns = joblib.load(BASE / "models" / "feature_columns.pkl")
    defaults = joblib.load(BASE / "models" / "defaults.pkl")
    return model, columns, defaults


model, columns, defaults = load_files()

# ---------- 2. Page title ----------
st.title("🏠 House Price Predictor")
st.write(
    "Enter a few details about a house and a machine learning model "
    "(Gradient Boosting, trained on the Ames, Iowa housing dataset) "
    "will estimate its sale price."
)

# ---------- 3. The 8 inputs ----------
col1, col2 = st.columns(2)

with col1:
    overall_qual = st.slider("Overall quality (1 = poor, 10 = excellent)", 1, 10, 6)
    living_area = st.slider("Living area above ground (sq ft)", 330, 4500, 1460)
    year_built = st.slider("Year built", 1872, 2010, 1972)
    basement = st.slider("Basement area (sq ft, 0 = none)", 0, 3200, 990)

with col2:
    garage = st.slider("Garage area (sq ft, 0 = none)", 0, 1400, 480)
    lot_area = st.slider("Lot size (sq ft)", 1300, 50000, 9500)
    fireplaces = st.slider("Fireplaces", 0, 3, 1)
    full_baths = st.slider("Full bathrooms", 0, 3, 2)

# ---------- 4. Predict ----------
if st.button("Predict price", type="primary"):
    # Start from typical values for all 258 columns, then overwrite the 8 chosen ones
    row = defaults.to_dict()
    row.update(
        {
            "OverallQual": overall_qual,
            "GrLivArea": living_area,
            "YearBuilt": year_built,
            "TotalBsmtSF": basement,
            "GarageArea": garage,
            "LotArea": lot_area,
            "Fireplaces": fireplaces,
            "FullBath": full_baths,
        }
    )
    input_df = pd.DataFrame([row])[columns]

    # The model predicts log(price), so convert back to dollars
    price = np.expm1(model.predict(input_df)[0])

    st.metric("Estimated sale price", f"${price:,.0f}")
    st.caption(
        "On houses it had never seen, the model's typical error was about $14,500. "
        "This demo uses only 8 details and fills in typical values for the rest."
    )
