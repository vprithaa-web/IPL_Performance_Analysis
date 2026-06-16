import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 1. Load the dataset and clean it (consistent with Step 2)
df = pd.read_csv(r'c:\Users\PRITHAA\Downloads\ipl_2008_2024_complete (1).csv')
df.loc[df['venue'].str.contains('Sharjah', case=False, na=False), 'city'] = 'Sharjah'
df.loc[df['venue'].str.contains('Dubai', case=False, na=False), 'city'] = 'Dubai'

team_mapping = {
    'Royal Challengers Bangalore': 'Royal Challengers Bengaluru',
    'Kings XI Punjab': 'Punjab Kings',
    'Delhi Daredevils': 'Delhi Capitals',
    'Rising Pune Supergiants': 'Rising Pune Supergiant'
}
for col in ['batting_team', 'bowling_team', 'toss_winner', 'winner']:
    df[col] = df[col].replace(team_mapping)

# Exclude abandoned matches
cleaned_df = df.dropna(subset=['winner']).copy()

# --- ANALYSIS 1: Toss Success Metric ---
toss_win_match_win_pct = cleaned_df['toss_win_match_win'].mean() * 100
print(f"Overall Toss Win to Match Win Percentage: {toss_win_match_win_pct:.2f}%")

# --- VISUALIZATION 1: Toss Decision Over the Years ---
toss_decision_season = cleaned_df.groupby(['season', 'toss_decision']).size().unstack(fill_value=0)
fig, ax = plt.subplots(figsize=(10, 6))
toss_decision_season.plot(kind='bar', stacked=True, color=['#4C72B0', '#DD8452'], ax=ax)
ax.set_title('Toss Decision Trends Across Seasons', fontsize=14, fontweight='bold')
ax.set_xlabel('Season', fontsize=12)
ax.set_ylabel('Number of Matches', fontsize=12)
ax.legend(title='Toss Decision')
plt.tight_layout()
plt.savefig('toss_decision_trends.png')
plt.close()

# --- VISUALIZATION 2: Score Trends Over Seasons ---
runs_season_innings = cleaned_df.groupby(['season', 'innings'])['total_runs'].mean().unstack()
fig, ax = plt.subplots(figsize=(10, 6))
runs_season_innings[1].plot(kind='line', marker='o', label='1st Innings', color='blue', ax=ax)
runs_season_innings[2].plot(kind='line', marker='s', label='2nd Innings', color='orange', ax=ax)
ax.set_title('Average Innings Total Runs Over Seasons', fontsize=14, fontweight='bold')
ax.set_xlabel('Season', fontsize=12)
ax.set_ylabel('Average Total Runs', fontsize=12)
ax.set_xticks(runs_season_innings.index)
ax.set_xticklabels(runs_season_innings.index, rotation=45)
ax.legend(title='Innings')
ax.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig('avg_runs_over_seasons.png')
plt.close()

# --- VISUALIZATION 3: Top Venues by Score ---
first_inn = cleaned_df[cleaned_df['innings'] == 1]
venue_counts = first_inn['venue'].value_counts()
top_venues = venue_counts[venue_counts >= 20].index # Filter grounds with at least 20 matches

venue_avg_runs = first_inn[first_inn['venue'].isin(top_venues)].groupby('venue')['total_runs'].mean().sort_values(ascending=False).head(10)

fig, ax = plt.subplots(figsize=(12, 6))
venue_avg_runs.plot(kind='bar', color='teal', ax=ax)
ax.set_title('Top 10 High-Scoring Venues (1st Innings Avg Score)', fontsize=14, fontweight='bold')
ax.set_xlabel('Venue', fontsize=12)
ax.set_ylabel('Average 1st Innings Score', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('top_high_scoring_venues.png')
plt.close()