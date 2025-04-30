import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
import xgboost as xgb  # Add this import
import joblib
import pickle
import os

# Load datasets
print("Loading datasets...")
matches = pd.read_csv('Datasets/matches.csv')
deliveries = pd.read_csv('Datasets/deliveries.csv')
print("Datasets loaded successfully!")

# Merge datasets
print("Merging datasets...")
merged = deliveries.merge(matches, left_on='match_id', right_on='id')
print("Datasets merged successfully!")

# Feature Engineering
print("Starting feature engineering...")
# Calculate cumulative runs and wickets per innings
merged['current_score'] = merged.groupby(['match_id', 'inning'])['total_runs'].cumsum()

# Fix the wickets_fallen calculation
# Instead of using apply with cumsum, we'll use transform with cumsum
merged['wickets_fallen'] = merged.groupby(['match_id', 'inning'])['player_dismissed'].transform(
    lambda x: x.notna().cumsum()
)

# Calculate last 5 overs performance (30 balls)
print("Calculating last 5 overs performance...")
# Create a ball counter within each match and innings
merged['ball_count'] = merged.groupby(['match_id', 'inning']).cumcount() + 1

# Improve last 5 overs runs calculation
merged['last_5_overs_runs'] = merged.groupby(['match_id', 'inning'])['total_runs'].transform(
    lambda x: x.rolling(window=30, min_periods=1).sum()
)

# Simplify the last 5 overs wickets calculation
merged['last_5_overs_wickets'] = merged.groupby(['match_id', 'inning'])['player_dismissed'].transform(
    lambda x: x.notna().rolling(window=30, min_periods=1).sum()
)

# Add a weighted version of last 5 overs wickets (more recent wickets have higher impact)
merged['weighted_wickets'] = merged.groupby(['match_id', 'inning'])['player_dismissed'].transform(
    lambda x: x.notna().ewm(span=15, min_periods=1).mean() * 30
)

# Add wicket momentum feature (rate of wicket-taking in last 5 overs)
merged['wicket_momentum'] = merged.groupby(['match_id', 'inning'])['player_dismissed'].transform(
    lambda x: x.notna().rolling(window=30, min_periods=1).mean() * 30
)

# Add run rate features
merged['current_run_rate'] = merged['current_score'] / (merged['over'] + merged['ball']/6)
merged['current_run_rate'] = merged['current_run_rate'].fillna(0).replace([np.inf, -np.inf], 0)

# Create over progression feature (0-20 for T20)
merged['over_progress'] = (merged['over'] + merged['ball']/10) / 20

# Add remaining resources feature (based on wickets left and balls remaining)
merged['resources_remaining'] = (1 - merged['over_progress']) * (1 - merged['wickets_fallen']/10)

# Venue performance features
print("Calculating venue statistics...")
venue_stats = merged.groupby(['venue', 'batting_team'])['total_runs'].agg(['mean', 'std']).reset_index()
venue_stats.rename(columns={'mean': 'mean_venue', 'std': 'std_venue'}, inplace=True)
# Fill NaN values in std_venue with 0
venue_stats['std_venue'] = venue_stats['std_venue'].fillna(0)

merged = merged.merge(venue_stats, on=['venue', 'batting_team'], how='left')
# Fill any remaining NaN values
merged['mean_venue'] = merged['mean_venue'].fillna(merged['mean_venue'].mean())
merged['std_venue'] = merged['std_venue'].fillna(merged['std_venue'].mean())

print("Feature engineering completed!")

# Prepare final dataset for score prediction model
print("Preparing final dataset for score prediction...")
# Get the final score for each innings
final_scores = merged.groupby(['match_id', 'inning'])['total_runs'].sum().reset_index()
final_scores.rename(columns={'total_runs': 'final_score'}, inplace=True)

# Get the data at each ball
features = merged[['match_id', 'inning', 'batting_team', 'bowling_team', 'venue', 
                  'current_score', 'wickets_fallen', 'over_progress',
                  'last_5_overs_runs', 'last_5_overs_wickets', 'weighted_wickets',
                  'wicket_momentum', 'current_run_rate', 'resources_remaining',
                  'mean_venue', 'std_venue']]

# Merge with final scores
features = features.merge(final_scores, on=['match_id', 'inning'])

# Filter to include only data points from completed overs (to make predictions more realistic)
completed_overs = features[features['over_progress'].apply(lambda x: x == int(x * 20) / 20)]

# Prepare data for win probability model
print("Preparing data for win probability model...")
# For second innings only
second_innings = merged[merged['inning'] == 2].copy()

# Get target for each match
targets = merged[merged['inning'] == 1].groupby('match_id')['total_runs'].sum().reset_index()
targets.rename(columns={'total_runs': 'target'}, inplace=True)

second_innings = second_innings.merge(targets, on='match_id')

# Calculate balls left, runs left, and required run rate
second_innings['balls_left'] = 120 - (second_innings['over'] * 6 + second_innings['ball'])
second_innings['runs_left'] = second_innings['target'] - second_innings['current_score']
second_innings['wickets_left'] = 10 - second_innings['wickets_fallen']
second_innings['crr'] = second_innings['current_score'] / (120 - second_innings['balls_left'])
second_innings['rrr'] = second_innings['runs_left'] * 6 / second_innings['balls_left']
second_innings['rrr'] = second_innings['rrr'].replace([np.inf, -np.inf], 0)

# Add result column (1 if batting team wins, 0 if bowling team wins)
second_innings['result'] = (second_innings['winner'] == second_innings['batting_team']).astype(int)

# Prepare win probability features
win_prob_features = second_innings[['batting_team', 'bowling_team', 'venue', 
                                   'runs_left', 'balls_left', 'wickets_left', 
                                   'target', 'crr', 'rrr', 'result']]

# Remove rows with invalid values
win_prob_features = win_prob_features.replace([np.inf, -np.inf], np.nan).dropna()

# Train score prediction model
print("Training score prediction model...")
# Prepare features and target
X = completed_overs.drop(['match_id', 'final_score'], axis=1)
y = completed_overs['final_score']

# Add feature scaling for better model performance
from sklearn.preprocessing import StandardScaler
numeric_cols = ['current_score', 'wickets_fallen', 'over_progress', 
               'last_5_overs_runs', 'last_5_overs_wickets', 'weighted_wickets',
               'wicket_momentum', 'current_run_rate', 'resources_remaining',
               'mean_venue', 'std_venue']
scaler = StandardScaler()
X[numeric_cols] = scaler.fit_transform(X[numeric_cols])

# Encode categorical features - use one-hot encoding
X_encoded = pd.get_dummies(X, columns=['batting_team', 'bowling_team', 'venue'], drop_first=True)

# Train/test split with stratification based on inning
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42, stratify=X_encoded['inning'])

# Tune hyperparameters for better performance
score_model = XGBRegressor(
    n_estimators=200,
    learning_rate=0.01,
    max_depth=4,
    min_child_weight=2,
    subsample=0.7,
    colsample_bytree=0.7,
    gamma=0,
    reg_alpha=0.5,
    reg_lambda=1,
    random_state=42,
    objective='reg:squarederror'
)

# Train the model with cross-validation
from sklearn.model_selection import cross_val_score
cv_scores = cross_val_score(score_model, X_train, y_train, cv=5, scoring='r2')
print(f"Cross-validation R² scores: {cv_scores}")
print(f"Mean CV R² score: {cv_scores.mean():.4f}")

# Fit the model on the full training set
score_model.fit(
    X_train, 
    y_train,
    eval_set=[(X_test, y_test)],
    verbose=False
)

# Evaluate on test set
train_r2 = score_model.score(X_train, y_train)
test_r2 = score_model.score(X_test, y_test)
print(f"Score prediction model - Train R² score: {train_r2:.4f}")
print(f"Score prediction model - Test R² score: {test_r2:.4f}")

# If R² is still negative, try a simpler model
if test_r2 < 0:
    print("XGBoost model performing poorly, trying a simpler model...")
    from sklearn.ensemble import GradientBoostingRegressor
    
    # Try a different algorithm
    gb_model = GradientBoostingRegressor(
        n_estimators=100,
        learning_rate=0.05,
        max_depth=3,
        min_samples_split=5,
        random_state=42
    )
    gb_model.fit(X_train, y_train)
    
    # Evaluate the new model
    gb_train_r2 = gb_model.score(X_train, y_train)
    gb_test_r2 = gb_model.score(X_test, y_test)
    print(f"Gradient Boosting - Train R² score: {gb_train_r2:.4f}")
    print(f"Gradient Boosting - Test R² score: {gb_test_r2:.4f}")
    
    # Use the better model
    if gb_test_r2 > test_r2:
        score_model = gb_model
        print("Using Gradient Boosting model instead of XGBoost")

# Train win probability model
print("Training win probability model...")
# Prepare features and target
X_win = win_prob_features.drop('result', axis=1)
y_win = win_prob_features['result']

# Encode categorical features
X_win_encoded = pd.get_dummies(X_win, columns=['batting_team', 'bowling_team', 'venue'])

# Train/test split
X_win_train, X_win_test, y_win_train, y_win_test = train_test_split(X_win_encoded, y_win, test_size=0.2, random_state=42)

# Train model
from sklearn.ensemble import RandomForestClassifier
win_model = RandomForestClassifier(n_estimators=100, random_state=42)
win_model.fit(X_win_train, y_win_train)
print(f"Win probability model accuracy: {win_model.score(X_win_test, y_win_test):.4f}")

# Create models directory if it doesn't exist
os.makedirs('D:/ipl-score-predictor/models', exist_ok=True)

# Save models
print("Saving models...")
joblib.dump(score_model, 'D:/ipl-score-predictor/models/ipl_score_model.pkl')
with open('D:/ipl-score-predictor/models/pipe.pkl', 'wb') as f:
    pickle.dump(win_model, f)

print("Models saved successfully! Data processing complete!")
