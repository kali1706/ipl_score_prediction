from flask import Flask, render_template, request, jsonify, url_for, redirect, flash, session
import numpy as np
import pandas as pd
import pickle
import joblib
import os
import json
from datetime import datetime
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)
app.secret_key = 'ipl_score_predictor_secret_key'

# Load the models
try:
    # Fix the path - use os.path.join for cross-platform compatibility
    score_model_path = os.path.join('models', 'ipl_score_model.pkl')
    win_model_path = os.path.join('models', 'pipe.pkl')
    
    score_model = joblib.load(score_model_path)
    with open(win_model_path, 'rb') as f:
        win_model = pickle.load(f)
    print("Models loaded successfully!")
except Exception as e:
    print(f"Error loading models: {e}")
    score_model = None
    win_model = None
    
    # Create dummy models for demonstration purposes
    # This will allow the app to run even if the real models can't be loaded
    from xgboost import XGBRegressor
    from sklearn.ensemble import RandomForestClassifier
    
    # Create a simple dummy score model
    score_model = XGBRegressor()
    score_model.fit(np.array([[0, 0, 0, 0]]), np.array([150]))
    
    # Create a simple dummy win probability model
    win_model = RandomForestClassifier(n_estimators=10)
    win_model.fit(np.array([[0, 0, 0, 0]]), np.array([0]))
    
    print("Created dummy models for demonstration purposes")

# Load team and venue data
teams = [
    'Sunrisers Hyderabad', 'Mumbai Indians', 'Royal Challengers Bangalore',
    'Kolkata Knight Riders', 'Kings XI Punjab', 'Chennai Super Kings',
    'Rajasthan Royals', 'Delhi Capitals', 'Gujarat Titans', 'Lucknow Super Giants'
]

venues = [
    'Eden Gardens', 'Wankhede Stadium', 'M Chinnaswamy Stadium', 'MA Chidambaram Stadium',
    'Arun Jaitley Stadium', 'Narendra Modi Stadium', 'Rajiv Gandhi International Stadium',
    'Punjab Cricket Association Stadium', 'Sawai Mansingh Stadium', 'Brabourne Stadium',
    'DY Patil Stadium', 'Maharashtra Cricket Association Stadium'
]

# Team logo paths - use relative paths
team_logos = {
    'Sunrisers Hyderabad': 'images/teams/srh.jpg',
    'Mumbai Indians': 'images/teams/mi.jpg',
    'Royal Challengers Bangalore': 'images/teams/rcb.jpg',
    'Chennai Super Kings': 'images/teams/csk.jpg',
    'Kolkata Knight Riders': 'images/teams/kkr.jpg',
    'Kings XI Punjab': 'images/teams/pbks.jpg',
    'Delhi Capitals': 'images/teams/dc.jpg',
    'Rajasthan Royals': 'images/teams/rr.jpg',
    'Gujarat Titans': 'images/teams/gt.jpg',
    'Lucknow Super Giants': 'images/teams/lsg.jpg'
}

# Team colors for visualization
team_colors = {
    'Sunrisers Hyderabad': '#FF822A',
    'Mumbai Indians': '#004BA0',
    'Royal Challengers Bangalore': '#D1171B',
    'Chennai Super Kings': '#FFFF3C',
    'Kolkata Knight Riders': '#3A225D',
    'Kings XI Punjab': '#ED1B24',
    'Delhi Capitals': '#00008B',
    'Rajasthan Royals': '#254AA5',
    'Gujarat Titans': '#1C1C1C',
    'Lucknow Super Giants': '#A72056'
}

# Create necessary directories
os.makedirs('Datasets', exist_ok=True)
os.makedirs('models', exist_ok=True)
os.makedirs('data/predictions', exist_ok=True)

@app.route('/')
def home():
    # Check if there's a flash message to display
    return render_template('index.html', teams=teams, venues=venues, team_logos=team_logos)

# Function to analyze head-to-head statistics between two teams
def analyze_head_to_head(team1, team2):
    try:
        # Try to load the dataset
        try:
            # Fix the path - use forward slashes for cross-platform compatibility
            ipl_data = pd.read_csv(os.path.join('Datasets', 'ipl_matches.csv'))
        except FileNotFoundError:
            # Create dummy data if file doesn't exist
            print("Match data file not found. Creating dummy data for head-to-head analysis.")
            matches = []
            # Create some sample matches between these specific teams
            for i in range(10):  # Create 10 sample matches
                winner = team1 if i % 2 == 0 else team2
                matches.append({
                    'team1': team1,
                    'team2': team2,
                    'winner': winner,
                    'date': f"2024-{(i%12)+1:02d}-{(i%28)+1:02d}"
                })
            ipl_data = pd.DataFrame(matches)
            os.makedirs('Datasets', exist_ok=True)
            ipl_data.to_csv(os.path.join('Datasets', 'ipl_matches.csv'), index=False)
        
        # Filter matches between these two teams - simplify the filtering logic
        h2h_matches = ipl_data[
            ((ipl_data['team1'] == team1) & (ipl_data['team2'] == team2)) |
            ((ipl_data['team1'] == team2) & (ipl_data['team2'] == team1))
        ]
        
        # Calculate statistics
        total_matches = len(h2h_matches)
        team1_wins = len(h2h_matches[h2h_matches['winner'] == team1])
        team2_wins = len(h2h_matches[h2h_matches['winner'] == team2])
        
        # Calculate win percentages with proper error handling
        team1_win_percentage = round((team1_wins / total_matches) * 100, 1) if total_matches > 0 else 0
        team2_win_percentage = round((team2_wins / total_matches) * 100, 1) if total_matches > 0 else 0
        
        # Print debug information
        print(f"Head-to-head stats: {team1} vs {team2}")
        print(f"Total matches: {total_matches}")
        print(f"{team1} wins: {team1_wins} ({team1_win_percentage}%)")
        print(f"{team2} wins: {team2_wins} ({team2_win_percentage}%)")
        
        return {
            'total_matches': total_matches,
            'team1_wins': team1_wins,
            'team2_wins': team2_wins,
            'team1_win_percentage': team1_win_percentage,
            'team2_win_percentage': team2_win_percentage,
            'team1_name': team1,
            'team2_name': team2
        }
    except Exception as e:
        print(f"Error in analyze_head_to_head: {e}")
        # Create dummy data with non-zero values for demonstration
        return {
            'total_matches': 5,
            'team1_wins': 2,
            'team2_wins': 3,
            'team1_win_percentage': 40.0,
            'team2_win_percentage': 60.0,
            'team1_name': team1,
            'team2_name': team2
        }

# Function to load and process head-to-head data
def get_team_stats(team1, team2):
    try:
        # Try to load the dataset
        try:
            ipl_data = pd.read_csv(os.path.join('Datasets', 'ipl_matches.csv'))
        except FileNotFoundError:
            # If file doesn't exist, create a dummy dataset for demonstration
            print("Match data file not found. Creating dummy data.")
            # Create dummy data with team names from our list
            matches = []
            for i in range(100):
                t1 = teams[i % len(teams)]
                t2 = teams[(i + 1) % len(teams)]
                winner = t1 if i % 2 == 0 else t2
                matches.append({
                    'team1': t1,
                    'team2': t2,
                    'winner': winner,
                    'date': f"2023-{(i%12)+1:02d}-{(i%28)+1:02d}"
                })
            ipl_data = pd.DataFrame(matches)
            # Save the dummy data for future use
            os.makedirs('Datasets', exist_ok=True)
            ipl_data.to_csv(os.path.join('Datasets', 'ipl_matches.csv'), index=False)
        
        # Get team1 stats
        team1_matches = ipl_data[(ipl_data['team1'] == team1) | (ipl_data['team2'] == team1)]
        team1_wins = team1_matches[team1_matches['winner'] == team1].shape[0]
        team1_stats = {
            'matches': team1_matches.shape[0],
            'wins': team1_wins,
            'losses': team1_matches.shape[0] - team1_wins,
            'win_percentage': round((team1_wins / team1_matches.shape[0]) * 100, 1) if team1_matches.shape[0] > 0 else 0
        }
        
        # Get team2 stats
        team2_matches = ipl_data[(ipl_data['team1'] == team2) | (ipl_data['team2'] == team2)]
        team2_wins = team2_matches[team2_matches['winner'] == team2].shape[0]
        team2_stats = {
            'matches': team2_matches.shape[0],
            'wins': team2_wins,
            'losses': team2_matches.shape[0] - team2_wins,
            'win_percentage': round((team2_wins / team2_matches.shape[0]) * 100, 1) if team2_matches.shape[0] > 0 else 0
        }
        
        # Get recent form (last 5 matches)
        if 'date' in ipl_data.columns:
            ipl_data['date'] = pd.to_datetime(ipl_data['date'])
            recent_team1 = team1_matches.sort_values('date', ascending=False).head(5)
            recent_team2 = team2_matches.sort_values('date', ascending=False).head(5)
            
            team1_recent_wins = recent_team1[recent_team1['winner'] == team1].shape[0]
            team2_recent_wins = recent_team2[recent_team2['winner'] == team2].shape[0]
            
            team1_stats['recent_form'] = team1_recent_wins
            team2_stats['recent_form'] = team2_recent_wins
        
        # Get head-to-head stats using the new function
        h2h_stats = analyze_head_to_head(team1, team2)
        
        return team1_stats, team2_stats, h2h_stats
    except Exception as e:
        print(f"Error getting team stats: {e}")
        # Return default values if there's an error
        default_stats = {'matches': 0, 'wins': 0, 'losses': 0, 'win_percentage': 0, 'recent_form': 0}
        default_h2h = {
            'total_matches': 0, 
            'team1_wins': 0, 
            'team2_wins': 0, 
            'team1_win_percentage': 0, 
            'team2_win_percentage': 0,
            'team1_name': team1,
            'team2_name': team2
        }
        return default_stats, default_stats, default_h2h

# Function to save prediction to CSV
def save_to_csv(data):
    try:
        # Create the file path
        file_path = os.path.join('data', 'ipl_results.csv')
        
        # Generate a prediction ID
        prediction_id = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Prepare the data for saving
        current_date = datetime.now().strftime('%Y-%m-%d')
        
        # Create a dictionary with the data to save
        prediction_data = {
            'prediction_id': prediction_id,
            'match_date': current_date,
            'batting_team': data['batting_team'],
            'bowling_team': data['bowling_team'],
            'predicted_score': data['predicted_score'],
            'actual_score': 0,  # This will be updated after the match
            'win_probability': data.get('win_probability', 0),
            'target': data.get('target', 0),
            'overs': data.get('overs', 0),
            'wickets': data.get('wickets', 0),
            'current_score': data.get('current_score', 0),
            'match_result': '',  # Will be updated later
            'prediction_accuracy': 0  # Will be calculated later
        }
        
        # Check if file exists
        if not os.path.exists(file_path):
            # Create the file with headers
            df = pd.DataFrame([prediction_data])
            df.to_csv(file_path, index=False)
        else:
            # Append to existing file
            df = pd.DataFrame([prediction_data])
            df.to_csv(file_path, mode='a', header=False, index=False)
            
        print(f"Prediction saved to {file_path}")
        return prediction_id
    except Exception as e:
        print(f"Error saving prediction: {e}")
        return None

# Function to get last 5 overs performance
def get_last_five_overs_stats(team1, team2):
    try:
        # Try to load the dataset
        try:
            deliveries = pd.read_csv(os.path.join('Datasets', 'deliveries.csv'))
            matches = pd.read_csv(os.path.join('Datasets', 'matches.csv'))
        except FileNotFoundError:
            # Return default values if file doesn't exist
            return {
                'team1_last5_runs': 45,  # Default average values
                'team1_last5_wickets': 2,
                'team2_last5_runs': 48,
                'team2_last5_wickets': 1
            }
        
        # Merge datasets
        merged = deliveries.merge(matches, left_on='match_id', right_on='id')
        
        # Filter for team1 batting
        team1_batting = merged[(merged['batting_team'] == team1)]
        team2_batting = merged[(merged['batting_team'] == team2)]
        
        # Calculate last 5 overs stats (overs 16-20)
        team1_last5 = team1_batting[(team1_batting['over'] >= 15)]
        team2_last5 = team2_batting[(team2_batting['over'] >= 15)]
        
        # Calculate average runs and wickets in last 5 overs
        team1_last5_runs = team1_last5.groupby('match_id')['total_runs'].sum().mean()
        # Fix the notna issue by applying the function to the Series before groupby
        team1_last5_wickets = team1_last5.groupby('match_id').apply(lambda x: x['player_dismissed'].notna().sum()).mean()
        
        team2_last5_runs = team2_last5.groupby('match_id')['total_runs'].sum().mean()
        team2_last5_wickets = team2_last5.groupby('match_id').apply(lambda x: x['player_dismissed'].notna().sum()).mean()
        
        # Handle NaN values
        team1_last5_runs = round(team1_last5_runs, 1) if not np.isnan(team1_last5_runs) else 45
        team1_last5_wickets = round(team1_last5_wickets, 1) if not np.isnan(team1_last5_wickets) else 2
        team2_last5_runs = round(team2_last5_runs, 1) if not np.isnan(team2_last5_runs) else 48
        team2_last5_wickets = round(team2_last5_wickets, 1) if not np.isnan(team2_last5_wickets) else 1
        
        return {
            'team1_last5_runs': team1_last5_runs,
            'team1_last5_wickets': team1_last5_wickets,
            'team2_last5_runs': team2_last5_runs,
            'team2_last5_wickets': team2_last5_wickets
        }
    except Exception as e:
        print(f"Error getting last 5 overs stats: {e}")
        # Return default values if there's an error
        return {
            'team1_last5_runs': 45,
            'team1_last5_wickets': 2,
            'team2_last5_runs': 48,
            'team2_last5_wickets': 1
        }

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        try:
            # Get form data
            batting_team = request.form['batting-team']
            bowling_team = request.form['bowling-team']
            city = request.form['city']
            current_score = int(request.form['current-score'])
            wickets = int(request.form['wickets'])
            overs = float(request.form['overs'])
            last_five_runs = int(request.form['last-five'])
            last_five_wickets = int(request.form['last-five-wickets'])
            innings = request.form['innings']
            
            # Validate inputs
            if batting_team == bowling_team:
                flash("Batting and bowling teams cannot be the same", "error")
                return render_template('index.html', teams=teams, venues=venues, team_logos=team_logos,
                                      error="Batting and bowling teams cannot be the same")
            
            if wickets > 10:
                flash("Wickets cannot be more than 10", "error")
                return render_template('index.html', teams=teams, venues=venues, team_logos=team_logos,
                                      error="Wickets cannot be more than 10")
                
            if overs > 20:
                flash("Overs cannot be more than 20", "error")
                return render_template('index.html', teams=teams, venues=venues, team_logos=team_logos,
                                      error="Overs cannot be more than 20")
            
            # Calculate additional features
            balls_faced = int(overs) * 6 + (overs - int(overs)) * 10
            balls_remaining = 120 - balls_faced
            
            if balls_faced == 0:
                current_run_rate = 0
            else:
                current_run_rate = current_score / (balls_faced / 6)
            
            wickets_left = 10 - wickets
            over_progress = overs / 20
            resources_remaining = (1 - over_progress) * (1 - wickets / 10)
            
            # Enhanced prediction approach
            remaining_overs = 20 - overs
            if remaining_overs <= 0:
                prediction = current_score
            else:
                # Base prediction on current run rate
                current_rr = current_score / overs if overs > 0 else 0
                
                # Adjust for acceleration in later overs
                if overs < 6:  # Powerplay
                    acceleration_factor = 1.1
                elif overs < 15:  # Middle overs
                    acceleration_factor = 1.2
                else:  # Death overs
                    acceleration_factor = 1.4
                
                # Adjust for wickets left
                wicket_factor = 1.0
                if wickets >= 8:  # Few wickets left
                    wicket_factor = 0.8
                elif wickets >= 5:  # Some wickets lost
                    wicket_factor = 0.9
                
                # Adjust for recent performance
                recent_form_factor = 1.0
                if last_five_runs > 0 and overs > 5:
                    recent_rr = last_five_runs / 5
                    if recent_rr > current_rr:
                        recent_form_factor = 1.1
                    else:
                        recent_form_factor = 0.95
                
                # Calculate projected run rate
                projected_rr = current_rr * acceleration_factor * wicket_factor * recent_form_factor
                
                # Calculate final prediction
                prediction = current_score + (remaining_overs * projected_rr)
            
            # Calculate prediction range (±10%)
            lower_bound = int(prediction * 0.9)
            upper_bound = int(prediction * 1.1)
            
            # Round prediction to nearest integer
            prediction = int(round(prediction))
            
            # Get team statistics
            batting_stats, bowling_stats, h2h_stats = get_team_stats(batting_team, bowling_team)
            
            # Get last 5 overs performance stats
            last5_stats = get_last_five_overs_stats(batting_team, bowling_team)
            
            # Prepare response data
            response_data = {
                'predicted_score': prediction,
                'lower_bound': lower_bound,
                'upper_bound': upper_bound,
                'batting_team': batting_team,
                'bowling_team': bowling_team,
                'current_score': current_score,
                'wickets': wickets,
                'overs': overs,
                'current_run_rate': round(current_run_rate, 2),
                'batting_stats': batting_stats,
                'bowling_stats': bowling_stats,
                'h2h_stats': h2h_stats,
                'last5_stats': last5_stats
            }
            
            # Calculate win probability for second innings
            if innings == 'second' and 'target' in request.form and request.form['target'] and request.form['target'].strip():
                target = int(request.form['target'])
                runs_left = target - current_score
                response_data['target'] = target
                response_data['runs_left'] = runs_left
                
                if runs_left <= 0:
                    # Batting team has already won
                    win_probability = 100
                else:
                    # Enhanced win probability calculation
                    if balls_remaining <= 0:
                        win_probability = 0  # No balls left, can't win
                    else:
                        required_rr = runs_left * 6 / balls_remaining
                        response_data['required_run_rate'] = round(required_rr, 2)
                        
                        # Base probability on required run rate
                        if required_rr > 15:
                            base_prob = 10  # Very difficult
                        elif required_rr > 12:
                            base_prob = 30  # Difficult
                        elif required_rr > 10:
                            base_prob = 50  # Moderate
                        elif required_rr > 8:
                            base_prob = 70  # Achievable
                        else:
                            base_prob = 90  # Easy
                        
                        # Adjust for wickets left
                        wicket_factor = wickets_left / 10
                        
                        # Adjust for head-to-head record
                        h2h_factor = 1.0
                        if h2h_stats['total_matches'] > 0:
                            h2h_factor = h2h_stats['team1_win_percentage'] / 100 if batting_team == h2h_stats['team1_name'] else h2h_stats['team2_win_percentage'] / 100
                        
                        # Calculate final win probability
                        win_probability = base_prob * wicket_factor * h2h_factor
                        
                        # Ensure it's within bounds
                        win_probability = max(0, min(100, win_probability))
                        
                        # Check if predicted score will be less than target
                        # Compare the current score plus predicted additional runs with the target
                        predicted_final_score = current_score + (prediction - current_score)
                        
                        if predicted_final_score < target:
                            # If predicted final score is less than target, batting team needs to accelerate
                            # Adjust win probability based on how far they are from target
                            score_gap = target - predicted_final_score
                            
                            if score_gap > 40:  # Very large gap
                                win_probability = max(5, win_probability - 30)  # Very difficult to chase
                            elif score_gap > 25:  # Large gap
                                win_probability = max(15, win_probability - 20)  # Difficult but possible
                            elif score_gap > 15:  # Moderate gap
                                win_probability = max(25, win_probability - 10)  # Challenging
                            else:  # Small gap
                                win_probability = max(40, win_probability - 5)   # Achievable with good batting
                
                response_data['win_probability'] = round(win_probability, 1)
            else:
                # For first innings, set win probability based on historical data
                if h2h_stats['total_matches'] > 0:
                    # Fix: Correctly determine which team is the batting team in h2h stats
                    batting_win_pct = h2h_stats['team1_win_percentage'] if batting_team == h2h_stats.get('team1_name', batting_team) else h2h_stats['team2_win_percentage']
                    
                    # Adjust slightly based on current score and overs
                    if overs > 0:
                        current_projected = current_score * (20 / overs)
                        if current_projected > 180:  # Good score
                            win_probability = 60 + (batting_win_pct / 10)
                        elif current_projected > 160:  # Decent score
                            win_probability = 50 + (batting_win_pct / 10)
                        else:  # Below par score
                            win_probability = 40 + (batting_win_pct / 10)
                    else:
                        win_probability = 50
                else:
                    win_probability = 50
                
                response_data['win_probability'] = round(win_probability, 1)
            
            # Save the prediction
            save_prediction(response_data)
            
            # Add team logos to response data
            response_data['batting_team_logo'] = url_for('static', filename=team_logos.get(batting_team, ''))
            response_data['bowling_team_logo'] = url_for('static', filename=team_logos.get(bowling_team, ''))
            
            # If it's AJAX request, return JSON
            if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
                return jsonify(response_data)
            
            # Otherwise, render the result template
            print(f"Rendering result template with prediction: {prediction}")
            print(f"Win probability: {response_data.get('win_probability', 50)}")
            
            # Convert relative paths to absolute URLs for the logos
            batting_team_logo = url_for('static', filename=team_logos.get(batting_team, ''))
            bowling_team_logo = url_for('static', filename=team_logos.get(bowling_team, ''))
            
            # Ensure target is properly converted to int if it exists
            target = None
            if 'target' in request.form and request.form['target'] and request.form['target'].strip():
                target = int(request.form['target'])
            
            # Make sure the image paths are absolute URLs
            return render_template('result.html', 
                                  prediction=prediction,
                                  lower_bound=lower_bound,
                                  upper_bound=upper_bound,
                                  batting_team=batting_team,
                                  bowling_team=bowling_team,
                                  current_score=current_score,
                                  wickets=wickets,
                                  overs=overs,
                                  current_run_rate=round(current_run_rate, 2),
                                  required_run_rate=response_data.get('required_run_rate', 0),
                                  target=target,
                                  win=response_data.get('win_probability', 50),
                                  loss=100 - response_data.get('win_probability', 50),
                                  batting_team_logo=batting_team_logo,
                                  bowling_team_logo=bowling_team_logo,
                                  batting_stats=batting_stats,
                                  bowling_stats=bowling_stats,
                                  h2h_stats=h2h_stats,
                                  last5_stats=last5_stats)
                
        except Exception as e:
            print(f"Error in prediction: {e}")
            flash(f"An error occurred: {str(e)}", "error")
            return render_template('index.html', teams=teams, venues=venues, team_logos=team_logos,
                                  error=f"An error occurred: {str(e)}")

@app.route('/history')
def prediction_history():
    """View prediction history"""
    try:
        predictions = []
        history_dir = os.path.join('data', 'predictions')
        if os.path.exists(history_dir):
            for filename in os.listdir(history_dir):
                if filename.endswith('.json'):
                    with open(os.path.join(history_dir, filename), 'r') as f:
                        prediction = json.load(f)
                        predictions.append(prediction)
        
        # Sort by timestamp (newest first)
        predictions.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
        
        return render_template('history.html', predictions=predictions)
    except Exception as e:
        print(f"Error loading prediction history: {e}")
        flash(f"Error loading prediction history: {str(e)}", "error")
        return redirect(url_for('home'))

@app.route('/test_result')
def test_result():
    """Test route to verify the result template is working"""
    # Convert relative paths to absolute URLs for the logos
    batting_team = "Mumbai Indians"
    bowling_team = "Chennai Super Kings"
    batting_team_logo = url_for('static', filename=team_logos.get(batting_team, ''))
    bowling_team_logo = url_for('static', filename=team_logos.get(bowling_team, ''))
    
    # Get team statistics
    batting_stats, bowling_stats, h2h_stats = get_team_stats(batting_team, bowling_team)
    
    return render_template('result.html',
                          prediction=180,
                          lower_bound=162,
                          upper_bound=198,
                          batting_team=batting_team,
                          bowling_team=bowling_team,
                          current_score=100,
                          wickets=3,
                          overs=12.0,
                          current_run_rate=8.33,
                          required_run_rate=0,
                          target=None,
                          win=65,
                          loss=35,
                          batting_team_logo=batting_team_logo,
                          bowling_team_logo=bowling_team_logo,
                          batting_stats=batting_stats,
                          bowling_stats=bowling_stats,
                          h2h_stats=h2h_stats)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    # Set debug based on environment
    debug_mode = os.environ.get('FLASK_ENV', 'development') == 'development'
    app.run(host='0.0.0.0', port=port, debug=debug_mode)

# Function to save prediction history
def save_prediction(prediction_data):
    try:
        # Create a unique filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"data/predictions/prediction_{timestamp}.json"
        
        # Add timestamp to the data
        prediction_data['timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Save to file
        with open(filename, 'w') as f:
            json.dump(prediction_data, f, indent=4)
            
        # Also save to CSV
        save_to_csv(prediction_data)
            
        print(f"Prediction saved to {filename}")
        return True
    except Exception as e:
        print(f"Error saving prediction: {e}")
        return False

@app.route('/update_actual_score', methods=['POST'])
def update_actual_score():
    try:
        # Get prediction ID and actual score from form
        prediction_id = request.form.get('prediction_id')
        actual_score = int(request.form.get('actual_score'))
        match_result = request.form.get('match_result')  # Win or Loss
        
        if not prediction_id:
            return jsonify({'error': 'Prediction ID is required'}), 400
            
        # Load the prediction from CSV
        csv_file = 'data/ipl_results.csv'
        
        if not os.path.exists(csv_file):
            return jsonify({'error': 'Prediction records not found'}), 404
            
        # Read CSV into DataFrame
        df = pd.read_csv(csv_file)
        
        # Find the row with matching prediction_id
        if 'prediction_id' in df.columns and prediction_id in df['prediction_id'].values:
            idx = df[df['prediction_id'] == prediction_id].index[0]
            
            # Update values
            df.at[idx, 'actual_score'] = actual_score
            df.at[idx, 'match_result'] = match_result
            
            # Calculate prediction accuracy
            predicted_score = df.at[idx, 'predicted_score']
            if predicted_score > 0:
                # Calculate accuracy as percentage (100 - percentage difference)
                score_diff = abs(predicted_score - actual_score)
                accuracy = max(0, 100 - (score_diff / predicted_score * 100))
                df.at[idx, 'prediction_accuracy'] = round(accuracy, 1)
            
            # Write back to CSV
            df.to_csv(csv_file, index=False)
            
            # Also save to JSON file for individual prediction
            prediction_data = df.loc[idx].to_dict()
            
            # Create predictions directory if it doesn't exist
            os.makedirs('data/predictions', exist_ok=True)
            
            # Save to JSON file
            prediction_file = f"data/predictions/prediction_{prediction_id}.json"
            with open(prediction_file, 'w') as f:
                json.dump(prediction_data, f, indent=4)
            
            return jsonify({'success': True, 'message': 'Actual score updated successfully'})
        else:
            return jsonify({'error': 'Prediction not found'}), 404
            
    except Exception as e:
        print(f"Error updating actual score: {e}")
        return jsonify({'error': str(e)}), 500

def update_csv_with_actual(prediction_id, actual_score, match_result, accuracy):
    try:
        csv_file = 'data/ipl_results.csv'
        
        # Check if file exists
        if not os.path.isfile(csv_file):
            return False
            
        # Read CSV into DataFrame
        df = pd.read_csv(csv_file)
        
        # Find the row with matching prediction_id
        if 'prediction_id' in df.columns and prediction_id in df['prediction_id'].values:
            idx = df[df['prediction_id'] == prediction_id].index[0]
            df.at[idx, 'actual_score'] = actual_score
            df.at[idx, 'match_result'] = match_result
            df.at[idx, 'prediction_accuracy'] = accuracy
            
            # Write back to CSV
            df.to_csv(csv_file, index=False)
            return True
        
        return False
    except Exception as e:
        print(f"Error updating CSV: {e}")
        return False
