import streamlit as st
import pandas as pd
import joblib

# -----------------------------
# 1. Load models
# -----------------------------
home_goals_model = joblib.load("laliga_hgoals.joblib")
away_goals_model = joblib.load("laliga_agoals.joblib")
home_xg_model = joblib.load("laliga_hxg.joblib")
away_xg_model = joblib.load("laliga_axg.joblib")
home_sot_model = joblib.load("laliga_hsot.joblib")
away_sot_model = joblib.load("laliga_asot.joblib")
home_ppda_model = joblib.load("laliga_hppda.joblib")
away_ppda_model = joblib.load("laliga_appda.joblib")
result_model = joblib.load("laliga_res.joblib")

# -----------------------------
# 2. Load LabelEncoder
# -----------------------------
le_team = joblib.load("laliga_le.joblib")  # single encoder for all teams

# -----------------------------
# 3. Streamlit UI
# -----------------------------
st.title("Football Match Prediction Pipeline")

teams = ['Malaga', 'Sevilla', 'Granada', 'Almeria', 'Eibar', 'Barcelona',
         'Celta Vigo', 'Levante', 'Real Madrid', 'Rayo Vallecano', 'Getafe',
         'Valencia', 'Cordoba', 'Athletic Club', 'Atletico Madrid',
         'Espanyol', 'Villarreal', 'Deportivo La Coruna', 'Real Sociedad',
         'Elche', 'Sporting Gijon', 'Real Betis', 'Las Palmas', 'Osasuna',
         'Leganes', 'Alaves', 'Girona', 'Real Valladolid', 'SD Huesca',
         'Mallorca', 'Cadiz']

home_team = st.selectbox("Select Home Team", teams)
away_team = st.selectbox("Select Away Team", teams)

home_form_points_avg = st.number_input(
    "Enter Home Team Form (average points last 5 matches)",
    min_value=0.0, max_value=3.0, value=1.0, step=0.1
)
away_form_points_avg = st.number_input(
    "Enter Away Team Form (average points last 5 matches)",
    min_value=0.0, max_value=3.0, value=1.0, step=0.1
)

def predict_match(home_team, away_team, home_form_points_avg, away_form_points_avg):
    # -----------------------------
    # Prepare features
    # -----------------------------
    input_df = pd.DataFrame([{
        'home_team': le_team.fit_transform([home_team.strip()])[0],
        'away_team': le_team.fit_transform([away_team.strip()])[0],
        'home_form_points_avg': home_form_points_avg,
        'away_form_points_avg': away_form_points_avg
    }])

    # -----------------------------
    # 7-stage sequential predictions
    # -----------------------------
    input_df['home_ppda'] = home_ppda_model.predict(input_df)
    input_df['away_ppda'] = away_ppda_model.predict(input_df)
    input_df['home_xg'] = home_xg_model.predict(input_df)
    input_df['away_xg'] = away_xg_model.predict(input_df)
    input_df['home_sot'] = home_sot_model.predict(input_df)
    input_df['away_sot'] = away_sot_model.predict(input_df)
    input_df['home_goals'] = home_goals_model.predict(input_df)
    input_df['away_goals'] = away_goals_model.predict(input_df)

    pred_result = result_model.predict(input_df)[0]
    pred_proba = result_model.predict_proba(input_df)[0]
    result_map = {0: "Away Win", 1: "Draw", 2: "Home Win"}
    pred_result_label = result_map.get(pred_result, "Unknown")

    return input_df, pred_result_label, pred_proba

# -----------------------------
# Trigger prediction
# -----------------------------
if st.button("Predict Match"):
    predictions_df, match_result, match_probs = predict_match(home_team, away_team, home_form_points_avg, away_form_points_avg)

    st.subheader("Predicted Match Statistics")
    st.write(f"**Predicted Home Goals:** {int(round(predictions_df['home_goals'].values[0]))}")
    st.write(f"**Predicted Away Goals:** {int(round(predictions_df['away_goals'].values[0]))}")
    st.write(f"**Predicted Home xG:** {predictions_df['home_xg'].values[0]:.2f}")
    st.write(f"**Predicted Away xG:** {predictions_df['away_xg'].values[0]:.2f}")
    st.write(f"**Predicted Home SOT:** {int(round(predictions_df['home_sot'].values[0]))}")
    st.write(f"**Predicted Away SOT:** {int(round(predictions_df['away_sot'].values[0]))}")
    st.write(f"**Predicted Match Result:** {match_result}")


        # Show probabilities
    st.subheader("Result Probabilities")
    st.write(f"**Home Win:** {match_probs[2]:.2%}")
    st.write(f"**Draw:** {match_probs[1]:.2%}")
    st.write(f"**Away Win:** {match_probs[0]:.2%}")
    
    
