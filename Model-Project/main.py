#!/usr/bin/env python3

from GameFeatures import GameFeatures
import os

'''
*****************************************************************************
* CREATE_GAME_FEATURES:
* Initalize a game feature object
***************************************************************************** 
'''


def create_game_features():
    print("Creating GameFeatures object...")
    game_features = GameFeatures()
    print(f"Board size: {game_features.BOARD_WIDTH}x{game_features.BOARD_LENGTH}")
    print(f"Pieces: {game_features.PIECES}")
    print()
    return game_features


'''
*****************************************************************************
* READ_GAME_TRANSCRIPT:
* Used to read a game log file 
***************************************************************************** 
'''


def read_game_transcript(game_features, log_file):
    print(f"Reading game transcript from {log_file}...")
    moves_dict = game_features.read_game_transcript(log_file)
    winner_moves = moves_dict['winner_moves']
    loser_moves = moves_dict['loser_moves']
    print(f"Found {len(winner_moves)} winner moves and {len(loser_moves)} loser moves in the transcript.")
    print()

    if not winner_moves and not loser_moves:
        print("No moves found in the transcript.")
        return

    return moves_dict


'''
*****************************************************************************
* CREATE_FEATURE_LIST:
* Create a feature list, later to be used in the network
***************************************************************************** 
'''


def create_feature_list(game_features, moves_dict):
    print("Creating feature lists...")
    feature_dict = game_features.create_feature_lists(moves_dict)
    winner_features = feature_dict['winner_features']
    loser_features = feature_dict['loser_features']
    print(f"Created {len(winner_features)} winner feature lists and {len(loser_features)} loser feature lists.")
    print()
    return feature_dict


'''
*****************************************************************************
* PRINT_FEATURES:
* Prints the feature dictionary as parsed from the log states
***************************************************************************** 
'''


def print_features(feature_dict):
    winner_features = feature_dict['winner_features']
    loser_features = feature_dict['loser_features']

    print("Feature Lists Results:")
    print("-" * 40)

    print("Winner Moves:")
    print("-" * 20)
    for i, feature_list in enumerate(winner_features):
        print(f"Winner Move {i + 1}:")
        print(f"  Number of features: {len(feature_list)}")
        print(f"  Features: {feature_list}")
        print()

    print("Loser Moves:")
    print("-" * 20)
    for i, feature_list in enumerate(loser_features):
        print(f"Loser Move {i + 1}:")
        print(f"  Number of features: {len(feature_list)}")
        print(f"  Features: {feature_list}")
        print()


'''
*****************************************************************************
* PRINT_FEATURES_SUMMARY:
* Helper function, used to help the features and their indices
***************************************************************************** 
'''


def print_features_summary():
    print("Feature Summary:")
    print("-" * 40)
    print("Feature order:")
    print("0-2:   Column counts (1, 2, 3)")
    print("3-5:   Line counts (1, 2, 3)")
    print("6-8:   Player pieces on board (S, M, L)")
    print("9-11: Player pieces captured (S, M, L)")
    print("12-14: Enemy pieces on board (S, M, L)")
    print("15-17: Enemy pieces captured (S, M, L)")
    print("18:    Player average X location")
    print("19:    Player average Y location")
    print("20:    Player spread")
    print("21:    Total pieces on board")
    print("22-24: Game progress (1-hot encoded)")


def main():
    print("GameFeatures Testing System")
    print("=" * 40)

    # Get log directory from user or use default
    import sys
    if len(sys.argv) > 1:
        log_directory = sys.argv[1]
    else:
        log_directory = "."  # Current directory

    # Check if directory exists
    if not os.path.exists(log_directory):
        print(f"Error: Directory '{log_directory}' not found.")
        print("Usage: python main.py [log_directory]")
        print("If no directory is specified, the current directory will be used.")
        return

    try:
        game_features = create_game_features()

        # Find all log files in the directory
        log_files = []
        log_number = 0

        while True:
            log_filename = f"log {log_number}"
            log_path = os.path.join(log_directory, log_filename)

            if os.path.exists(log_path):
                log_files.append(log_path)
                log_number += 1
            else:
                break

        if not log_files:
            print(f"No log files found in directory: {log_directory}")
            return

        print(f"Found {len(log_files)} log files: {log_files}")
        print()

        # Process each log file
        for log_file in log_files:
            print(f"Processing {log_file}...")
            moves_dict = read_game_transcript(game_features, log_file)
            if moves_dict:
                feature_dict = create_feature_list(game_features, moves_dict)
                print_features(feature_dict)
                print_features_summary()
            print("-" * 60)
            print()

    except Exception as e:
        print(f"Error: {e}")
        print("Please check the log file format and try again.")


if __name__ == "__main__":
    main()
