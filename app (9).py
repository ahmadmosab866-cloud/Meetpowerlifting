import streamlit as st
import pandas as pd
import numpy as np
import pickle

# --- Page Configuration (Professional Look ke liye) ---
st.set_page_config(
    page_title="Powerlifting Event Analytics & Predictor 2030",
    page_icon="🏋️‍♂️",
    layout="wide"
)

# --- 1. Load Pre-trained Artifacts ---
@st.cache_resource  # App ko fast chalane ke liye caching use kar rahe hain
def load_ml_components():
    with open('powerlifting_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('powerlifting_scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    with open('le_country.pkl', 'rb') as f:
        le_country = pickle.load(f)
    with open('le_fed.pkl', 'rb') as f:
        le_fed = pickle.load(f)
    with open('unique_countries.pkl', 'rb') as f:
        unique_countries = pickle.load(f)
    with open('unique_federations.pkl', 'rb') as f:
        unique_federations = pickle.load(f)
        
    return model, scaler, le_country, le_fed, unique_countries, unique_federations

try:
    model, scaler, le_country, le_fed, unique_countries, unique_federations = load_ml_components()
except FileNotFoundError:
    st.error("❌ Error: Saved model ya encoder files nahi milin. Pehle model training script run karein!")
    st.stop()

# --- 2. Main UI Layout ---
st.title("🏋️‍♂️ Powerlifting Global Event Predictor & Demand Forecast")
st.markdown("""
    This AI-powered analytics tool predicts the market demand and number of powerlifting competitions (meets) 
    globally. It helps federations, event planners, and sports brands plan their logistics and expansion strategies.
""")

st.write("---")

# Layout columns bana rahe hain: Left side par inputs, Right side par results
col1, col2 = st.columns([1, 2])

with col1:
    st.header("🔮 Prediction Parameters")
    
    # User Inputs Dropdowns (Saved lists se unique values pick ho rahi hain)
    selected_country = st.selectbox("Select Target Country", sorted(unique_countries))
    selected_federation = st.selectbox("Select Federation", sorted(unique_federations))
    
    # Time Selection Inputs
    target_year = st.slider("Target Year", min_value=2026, max_value=2030, value=2026, step=1)
    target_month = st.slider("Target Month", min_value=1, max_value=12, value=6, step=1)
    
    predict_button = st.button("Predict Event Demand", type="primary")

with col2:
    st.header("📊 Market Insights & Forecast")
    
    if predict_button:
        try:
            # --- 3. Preprocessing & Encoding User Input ---
            # Agar user ne aisa country ya federation select kiya jo training mein nahi tha (unlikely), handle karein
            country_encoded = le_country.transform([selected_country])[0]
            fed_encoded = le_fed.transform([selected_federation])[0]
            
            # Feature array banana jaisa model ko chahiye: ['Year', 'Month', 'MeetCountry_Encoded', 'Federation_Encoded']
            input_features = np.array([[target_year, target_month, country_encoded, fed_encoded]])
            
            # --- 4. Scaling ---
            input_scaled = scaler.transform(input_features)
            
            # --- 5. Model Inference (Prediction) ---
            prediction = model.predict(input_scaled)[0]
            
            # Kyunke events points mein nahi ho sakte, hum round-off (ya floor ceil) karenge
            predicted_count = max(0, int(np.round(prediction))) # Negative values handle karne ke liye max(0, ...)
            
            # --- 6. Professional Output Display ---
            st.success(f"### Prediction Success!")
            
            # Metric card style display
            st.metric(
                label=f"Predicted Powerlifting Meets for {selected_federation} in {selected_country}", 
                value=f"{predicted_count} Events",
                delta=f"Month: {target_month} | Year: {target_year}",
                delta_color="normal"
            )
            
            # Business Context Dynamic Messages
            st.info("💡 **Business Actionable Insights:**")
            if predicted_count == 0:
                st.write(f"⚠️ Is mahine mein **{selected_federation}** ke meets ka trend bohot low hai. Yeh time marketing campaigns ya athletes ki training recovery ke liye behtar ho sakta hai.")
            elif predicted_count <= 2:
                st.write(f"🔵 **Low to Moderate Demand:** Venue bookings aur local sponsorships is period ke liye optimal rahenge.")
            else:
                st.write(f"🔥 **High Peak Season:** Expect high athlete turnout! Sports apparel brands aur supplement manufacturers ko is region mein active product placements karni chahiye.")
                
        except Exception as e:
            st.error(f"Prediction process mein koi masla aya: {str(e)}")
            
    else:
        # Default placeholder jab user ne button click nahi kiya hoga
        st.info("Parameters select karne ke baad 'Predict Event Demand' par click karein taake forecast load ho sake.")

# --- Footer (Professional branding) ---
st.markdown("---")
st.caption("🚀 Developed as a Professional Machine Learning Portfolio Project for Sports Analytics.")
