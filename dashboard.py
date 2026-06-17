import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

# --- Page Setup ---
st.set_page_config(page_title="IPL Analytics & Prediction Dashboard", layout="wide")

@st.cache_data
def load_and_process_data():
    # Load dataset
    df = pd.read_csv(r'c:\Users\PRITHAA\Downloads\ipl_2008_2024_complete (1).csv')
    
    # Clean city references
    df.loc[df['venue'].str.contains('Sharjah', case=False, na=False), 'city'] = 'Sharjah'
    df.loc[df['venue'].str.contains('Dubai', case=False, na=False), 'city'] = 'Dubai'
    
    # Standardize team rebrands
    team_mapping = {
        'Royal Challengers Bangalore': 'Royal Challengers Bengaluru',
        'Kings XI Punjab': 'Punjab Kings',
        'Delhi Daredevils': 'Delhi Capitals',
        'Rising Pune Supergiants': 'Rising Pune Supergiant'
    }
    for col in ['batting_team', 'bowling_team', 'toss_winner', 'winner']:
        df[col] = df[col].replace(team_mapping)
        
    cleaned_df = df.dropna(subset=['winner']).copy()
    
    # Isolate first innings for ML framing
    inn1_df = cleaned_df[cleaned_df['innings'] == 1].copy().rename(columns={
        'total_runs': 'inn1_runs',
        'wickets': 'inn1_wickets',
        'powerplay_runs': 'inn1_pp_runs',
        'death_over_runs': 'inn1_death_runs',
        'batting_team': 'team1',
        'bowling_team': 'team2'
    })
    inn1_df['team1_win'] = (inn1_df['team1'] == inn1_df['winner']).astype(int)
    
    # Train the core ML model pipeline
    features = ['venue', 'team1', 'team2', 'inn1_runs', 'inn1_wickets', 'inn1_pp_runs', 'inn1_death_runs']
    X = inn1_df[features]
    y = inn1_df['team1_win']
    
    categorical_features = ['venue', 'team1', 'team2']
    preprocessor = ColumnTransformer(
        transformers=[('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)],
        remainder='passthrough'
    )
    
    model_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42))
    ])
    model_pipeline.fit(X, y)
    
    return cleaned_df, inn1_df, model_pipeline

# Initialize application data
cleaned_df, inn1_df, model = load_and_process_data()
venues = sorted(inn1_df['venue'].unique().tolist())
teams = sorted(inn1_df['team1'].unique().tolist())

# --- Layout Configuration ---
st.title("📊 IPL Strategic Analytics & Match Predictor Dashboard")
st.markdown("An end-to-end data platform combining predictive machine learning with historical insights.")

# Create main dashboard layout split columns
col_sidebar, col_main = st.columns([1, 3])

with col_sidebar:
    st.header("⚙️ Configuration")
    selected_venue = st.selectbox("Select Venue Match Ground", venues)
    team1 = st.selectbox("Team Batting First (Team 1)", teams)
    available_opponents = [t for t in teams if t != team1]
    team2 = st.selectbox("Team Bowling First (Team 2)", available_opponents)
    
    st.subheader("Current Innings Progress")
    inn1_runs = st.slider("Total Runs Scored", 50, 270, 160)
    inn1_wickets = st.slider("Total Wickets Lost", 0, 10, 4)
    inn1_pp_runs = st.slider("Powerplay Runs (Overs 1-6)", 15, 90, 45)
    inn1_death_runs = st.slider("Death Over Runs (Overs 17-20)", 10, 90, 40)

with col_main:
    # Tab creation to organize dashboard sections cleanly
    tab1, tab2 = st.tabs(["🔮 Live Predictor Engine", "📈 Venue & Historical Context Insights"])
    
    with tab1:
        st.subheader("Live Probability Analytics")
        input_data = pd.DataFrame([{
            'venue': selected_venue, 'team1': team1, 'team2': team2,
            'inn1_runs': inn1_runs, 'inn1_wickets': inn1_wickets,
            'inn1_pp_runs': inn1_pp_runs, 'inn1_death_runs': inn1_death_runs
        }])
        
        # Calculate live ML outcomes
        probabilities = model.predict_proba(input_data)[0]
        team1_prob = probabilities[1] * 100
        team2_prob = probabilities[0] * 100
        
        # Visual metrics display cards
        c1, c2 = st.columns(2)
        c1.metric(f"🟢 {team1} (Batting First)", f"{team1_prob:.1f}% Win Prob")
        c2.metric(f"🔵 {team2} (Chasing)", f"{team2_prob:.1f}% Win Prob")
        
        # Build horizontal probability bar chart using Plotly
        prob_df = pd.DataFrame({
            'Team': [team1, team2],
            'Probability (%)': [team1_prob, team2_prob]
        })
        fig_prob = px.bar(prob_df, x='Probability (%)', y='Team', color='Team', 
                          orientation='h', color_discrete_sequence=['#2ecc71', '#3498db'],
                          text_auto='.1f')
        fig_prob.update_layout(height=250, showlegend=False)
        st.plotly_chart(fig_prob, use_container_width=True)
        
    with tab2:
        st.subheader(f"Ground Intelligence Profile: {selected_venue}")
        
        # Filter metrics belonging purely to the chosen ground
        venue_df = cleaned_df[cleaned_df['venue'] == selected_venue]
        
        if len(venue_df) > 0:
            # Metric Card Row
            v_matches = len(venue_df[venue_df['innings'] == 1])
            avg_1st_inn = venue_df[venue_df['innings'] == 1]['total_runs'].mean()
            avg_2nd_inn = venue_df[venue_df['innings'] == 2]['total_runs'].mean()
            
            m1, m2, m3 = st.columns(3)
            m1.metric("Historical Matches Played", int(v_matches))
            m2.metric("Average 1st Inn Score", f"{avg_1st_inn:.1f}")
            m3.metric("Average 2nd Inn Score", f"{avg_2nd_inn:.1f}")
            
            # Subplots
            st.markdown("#### Innings Score Spreads & Strategic Decisions")
            g1, g2 = st.columns(2)
            
            with g1:
                # Score distribution chart over time at this venue
                fig_box = px.box(venue_df, x='innings', y='total_runs', color='innings',
                                 labels={'innings': 'Innings Number', 'total_runs': 'Total Score'},
                                 title="Score Distribution by Innings")
                st.plotly_chart(fig_box, use_container_width=True)
                
            with g2:
                # Toss behaviors specific to this ground
                toss_data = venue_df.drop_duplicates(subset=['match_id'])['toss_decision'].value_counts().reset_index()
                fig_pie = px.pie(toss_data, names='toss_decision', values='count', 
                                 title="Toss Selection Patterns",
                                 color_discrete_sequence=['#e74c3c', '#f1c40f'])
                st.plotly_chart(fig_pie, use_container_width=True)
        else:
            st.warning("Insufficient historical data to parse deep profiles for this venue selection.")
