#!/usr/bin/env python3

import torch
import torch.nn as nn
from Feature_net import FeatureNet, train_feature_net, evaluate_feature_net, create_data_loader, generate_advice
from GameFeatures import GameFeatures
import os
import random
import numpy as np
import glob


def prepare_training_data_from_directory(game_features, log_directory):
    features = []
    labels = []

    '''
    *****************************************************************************
    * Find all log files in the directory
    ***************************************************************************** 
    '''
    log_files = glob.glob(os.path.join(log_directory, "log *"))
    if not log_files:
        print(f"No log files found in directory: {log_directory}")
        return features, labels
    print(f"Found {len(log_files)} log files: {log_files}")

    '''
    *****************************************************************************
    * Process each log file
    ***************************************************************************** 
    '''
    for log_file in log_files:
        try:
            print(f"Processing {log_file}...")

            '''
            *****************************************************************************
            * Reads the game log
            ***************************************************************************** 
            '''
            moves_dict = game_features.read_game_transcript(log_file)
            winner_moves = moves_dict['winner_moves']
            loser_moves = moves_dict['loser_moves']

            '''
            *****************************************************************************
            * Create a feature list
            ***************************************************************************** 
            '''
            feature_dict = game_features.create_feature_lists(moves_dict)
            winner_features = feature_dict['winner_features']
            loser_features = feature_dict['loser_features']

            '''
            *****************************************************************************
            * Winner move - label is 1, loser label is 0  
            ***************************************************************************** 
            '''
            for feature_list in winner_features:
                features.append(feature_list)
                labels.append(1.0)

            for feature_list in loser_features:
                features.append(feature_list)
                labels.append(0.0)

            print(f"Added {len(winner_features)} winner moves and {len(loser_features)} loser moves")

        except Exception as e:
            print(f"Error processing {log_file}: {e}")
            continue

    return features, labels


def split_data(features, labels, train_ratio=0.8):
    data = list(zip(features, labels))
    random.shuffle(data)

    '''
    *****************************************************************************
    * Split into train and test sets
    ***************************************************************************** 
    '''
    split_idx = int(len(data) * train_ratio)
    train_data = data[:split_idx]
    test_data = data[split_idx:]

    '''
    *****************************************************************************
    * Seperate features and labels
    ***************************************************************************** 
    '''
    train_features, train_labels = zip(*train_data)
    test_features, test_labels = zip(*test_data)

    return list(train_features), list(train_labels), list(test_features), list(test_labels)


def train_feature_net_main():
    print("FeatureNet Training System")
    print("=" * 40)

    import sys
    if len(sys.argv) > 1:
        log_directory = sys.argv[1]
    else:
        log_directory = "."

    if not os.path.exists(log_directory):
        print(f"Error: Directory '{log_directory}' not found.")
        print("Usage: python train_feature_net.py [log_directory]")
        print("If no directory is specified, the current directory will be used.")
        return

    try:
        print("Creating GameFeatures object...")
        game_features = GameFeatures()

        '''
        *****************************************************************************
        * Prepare the data based on the folder's log files
        ***************************************************************************** 
        '''
        print(f"Preparing training data from directory: {log_directory}")
        features, labels = prepare_training_data_from_directory(game_features, log_directory)

        if not features:
            print("No features found in the transcript.")
            return

        print(f"Total samples: {len(features)}")
        print(f"Winner moves: {sum(1 for label in labels if label == 1.0)}")
        print(f"Loser moves: {sum(1 for label in labels if label == 0.0)}")
        print()

        '''
        *****************************************************************************
        * Split data into train and test
        ***************************************************************************** 
        '''
        train_features, train_labels, test_features, test_labels = split_data(features, labels)

        print(f"Training samples: {len(train_features)}")
        print(f"Testing samples: {len(test_features)}")
        print()

        print("Creating data loaders...")
        train_loader = create_data_loader(train_features, train_labels, batch_size=16)
        test_loader = create_data_loader(test_features, test_labels, batch_size=16, shuffle=False)

        print("Creating FeatureNet model...")
        model = FeatureNet()

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {device}")
        print()

        '''
        *****************************************************************************
        * Train the model
        ***************************************************************************** 
        '''
        print("Starting training...")
        trained_model = train_feature_net(
            model=model,
            train_loader=train_loader,
            num_epochs=50,
            learning_rate=0.001,
            device=device
        )

        print("\nEvaluating model...")
        test_loss = evaluate_feature_net(trained_model, test_loader, device)

        '''
        *****************************************************************************
        * Save model
        ***************************************************************************** 
        '''
        torch.save(trained_model.state_dict(), 'feature_net_model.pth')
        print("\nModel saved as 'feature_net_model.pth'")

        from Feature_net import extract_stage_heuristics, interpret_heuristics
        trained_model = trained_model.cpu()

        heuristics = extract_stage_heuristics(trained_model)
        interpret_heuristics(heuristics)

        '''
        *****************************************************************************
        * Save the parsed suggestion into JSON file
        ***************************************************************************** 
        '''
        import json
        with open('game_heuristics.json', 'w') as f:
            json.dump(heuristics, f, indent=2)
        print("Heuristics saved to game_heuristics.json")

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    train_feature_net_main()
