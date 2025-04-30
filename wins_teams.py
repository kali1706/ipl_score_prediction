import pandas as pd
from datetime import datetime
import os
import pickle

def analyze_team_stats(team1, team2):
    # Read the IPL matches dataset
    df = pd.read_csv('Datasets\ipl_matches.csv')
    
    # Filter matches between the two teams (considering both combinations)
    matches = df[
        ((df['team1'] == team1) & (df['team2'] == team2)) |
        ((df['team1'] == team2) & (df['team2'] == team1))
    ]
    
    # Calculate total matches
    total_matches = len(matches)
    
    # Calculate wins for each team
    team1_wins = len(matches[matches['winner'] == team1])
    team2_wins = len(matches[matches['winner'] == team2])
    
    # Calculate win percentages
    team1_win_percentage = (team1_wins / total_matches) * 100 if total_matches > 0 else 0
    team2_win_percentage = (team2_wins / total_matches) * 100 if total_matches > 0 else 0
    
    # Create the results dictionary
    results = {
        'total_matches': total_matches,
        'team1_stats': {
            'name': team1,
            'wins': team1_wins,
            'win_percentage': round(team1_win_percentage, 2)
        },
        'team2_stats': {
            'name': team2,
            'wins': team2_wins,
            'win_percentage': round(team2_win_percentage, 2)
        },
        'all_matches': []
    }
    
    # Add all match details
    for _, match in matches.iterrows():
        match_info = {
            'winner': match['winner'],
            'toss_winner': match.get('toss_winner', 'N/A'),
            'toss_decision': match.get('toss_decision', 'N/A')
        }
        results['all_matches'].append(match_info)
    
    return results

# List of all IPL teams
teams = [
    'Sunrisers Hyderabad', 'Mumbai Indians', 'Royal Challengers Bangalore',
    'Kolkata Knight Riders', 'Kings XI Punjab', 'Chennai Super Kings',
    'Rajasthan Royals', 'Delhi Capitals', 'Gujarat Titans', 'Lucknow Super Giants'
]

if __name__ == "__main__":
    # Create a dictionary to store all results
    all_results = {}
    
    # Analyze all possible team combinations
    for i in range(len(teams)):
        for j in range(i + 1, len(teams)):
            team1 = teams[i]
            team2 = teams[j]
            
            stats = analyze_team_stats(team1, team2)
            
            # Store results with a unique key for each team combination
            key = f"{team1}_vs_{team2}"
            all_results[key] = {
                'team1': team1,
                'team2': team2,
                'total_matches': stats['total_matches'],
                'team1_wins': stats['team1_stats']['wins'],
                'team2_wins': stats['team2_stats']['wins'],
                'team1_win_percentage': stats['team1_stats']['win_percentage'],
                'team2_win_percentage': stats['team2_stats']['win_percentage'],
                'all_matches': stats['all_matches']
            }
            
            # Print the results
            print(f"\nHead to Head Analysis: {team1} vs {team2}")
            print("-" * 50)
            print(f"Total matches played: {stats['total_matches']}")
            print(f"\n{team1}:")
            print(f"Wins: {stats['team1_stats']['wins']}")
            print(f"Win Percentage: {stats['team1_stats']['win_percentage']}%")
            print(f"\n{team2}:")
            print(f"Wins: {stats['team2_stats']['wins']}")
            print(f"Win Percentage: {stats['team2_stats']['win_percentage']}%")
            
            print("\nAll matches history:")
            for k, match in enumerate(stats['all_matches'], 1):
                print(f"\nMatch {k}:")
                print(f"Winner: {match['winner']}")
                print(f"Toss: {match['toss_winner']} chose to {match['toss_decision']}")
            print("=" * 50)
    
    # Create the directory if it doesn't exist
    output_dir = 'd:\\ipl-score-predictor\\ipl-score-predictor\\data'
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate filename with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    pkl_filename = os.path.join(output_dir, f'head_to_head_analysis_{timestamp}.pkl')
    
    # Save to pickle file
    with open(pkl_filename, 'wb') as f:
        pickle.dump(all_results, f)
    
    print(f"\nResults saved to: {pkl_filename}")