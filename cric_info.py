import pandas as pd
import numpy as np


# 1. Load the dataset
df = pd.read_csv(r'c:\Users\PRITHAA\Downloads\ipl_2008_2024_complete (1).csv')

print(f"Original shape of dataset: {df.shape}")

# 2. Fix Missing Cities based on Venues
# Sharjah and Dubai venues are missing city names; let's fill them explicitly
df.loc[df['venue'].str.contains('Sharjah', case=False, na=False), 'city'] = 'Sharjah'
df.loc[df['venue'].str.contains('Dubai', case=False, na=False), 'city'] = 'Dubai'

# 3. Standardize Team Names (Accounting for Rebranding & Typos)
team_mapping = {
    'Royal Challengers Bangalore': 'Royal Challengers Bengaluru',
    'Kings XI Punjab': 'Punjab Kings',
    'Delhi Daredevils': 'Delhi Capitals',
    'Rising Pune Supergiants': 'Rising Pune Supergiant'
}

# Apply mapping to all team-related columns
team_cols = ['batting_team', 'bowling_team', 'toss_winner', 'winner']
for col in team_cols:
    df[col] = df[col].replace(team_mapping)

# 4. Handle Abandoned Matches
# Drop or isolate matches where there is no winner (No Result matches)
cleaned_df = df.dropna(subset=['winner']).copy()

# 5. Convert Date Column to Datetime format
cleaned_df['date'] = pd.to_datetime(cleaned_df['date'])

print(f"Cleaned shape of dataset: {cleaned_df.shape}")
print("Unique teams now:", cleaned_df['batting_team'].unique())

# Save the cleaned dataset for the next steps
cleaned_df.to_csv('cleaned_ipl_data.csv', index=False)
print("Cleaned data successfully saved to 'cleaned_ipl_data.csv'!")