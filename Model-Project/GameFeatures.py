import json
from Processing import Processing


class GameFeatures:
    # Hardcoded game configuration
    BOARD_WIDTH = 3
    BOARD_LENGTH = 3
    PIECES = {'S': 2, 'M': 2, 'L': 2}

    def __init__(self):
        # Initialize empty features list
        self.features = []

        # Use hardcoded board dimensions and pieces
        self.width = self.BOARD_WIDTH
        self.length = self.BOARD_LENGTH
        self.pieces = self.PIECES.copy()

    '''
    *****************************************************************************
    * GET_FEATURE:
    * Returns a specific feature if exists
    ***************************************************************************** 
    '''

    def get_feature(self, feature_name):
        for feature in self.features:
            if feature['name'] == feature_name:
                return feature['value']
        return None

    '''
    *****************************************************************************
    * UPDATE_FEATURE:
    * Updates a single feature if exists, otherwise appends it
    ***************************************************************************** 
    '''

    def update_feature(self, feature_name, value):
        for feature in self.features:
            if feature['name'] == feature_name:
                feature['value'] = value
                return
        # If feature doesn't exist, add it
        self.features.append({'name': feature_name, 'value': value})

    '''
    *****************************************************************************
    * GET_ALL_FEATURES:
    * Returns all of the existing features
    ***************************************************************************** 
    '''

    def get_all_features(self):
        return self.features

    '''
    *****************************************************************************
    * READ_GAME_TRANSCRIPT:
    * Inner function of how to read a log file based on our structure
    ***************************************************************************** 
    '''

    def read_game_transcript(self, transcript_file):
        # Read the transcript file
        with open(transcript_file, 'r') as f:
            lines = f.readlines()

        if not lines:
            return []

        # Get winning player from first line
        first_line = lines[0].strip()
        if not first_line.startswith('winner:'):
            raise ValueError("First line must specify the winner in format 'winner:[player_id]'")

        winning_player = int(first_line.split(':')[1]) - 1
        if winning_player not in [0, 1]:
            raise ValueError("Winner player_id must be 0 or 1")

        # Parse the transcript
        board_state = {}  # Dictionary to store current board state
        current_turn = 0
        winner_moves = []  # List to store features for each winning player's move
        loser_moves = []  # List to store features for each losing player's move
        captured_pieces = {'player': [], 'enemy': []}  # Track captured pieces for each player
        positions_read_for_turn = 0  # Track how many positions we've read for current turn
        total_positions = self.width * self.length  # Total positions on the board

        for line in lines[1:]:  # Skip first line (winner)
            line = line.strip()
            if not line:
                continue

            # Parse turn number
            if line.startswith('turn:'):
                # Start new turn
                current_turn = int(line.split(':')[1])
                positions_read_for_turn = 0
                continue

            # Parse board position
            if line.startswith('pos'):
                pos, pieces = line.split(':', 1)  # Split only on first colon
                x, y = map(int, pos[3:].split('-'))

                # Update board state
                if pieces:
                    pieces_list = []
                    piece_parts = pieces.split(',')
                    for piece_part in piece_parts:
                        piece_part = piece_part.strip()
                        if piece_part:
                            player_id, piece_type = piece_part.split(':')
                            pieces_list.append({
                                'player': int(player_id),
                                'type': piece_type,
                                'size': {'S': 0, 'M': 1, 'L': 2}[piece_type]
                            })

                    pieces_list.sort(key=lambda p: p['size'], reverse=True)  # Larger pieces first

                    # Add smaller pieces to captured list
                    for piece in pieces_list[1:]:
                        if piece['player'] == winning_player:
                            captured_pieces['player'].append((x, y))
                        else:
                            captured_pieces['enemy'].append((x, y))

                    # Keep only the largest piece at each position
                    board_state[(x, y)] = [pieces_list[0]]
                else:
                    board_state[(x, y)] = []

                positions_read_for_turn += 1

            # After reading all lines, check if the last turn was complete
            if current_turn > 0 and positions_read_for_turn == total_positions:
                current_player = 0 if current_turn % 2 == 1 else 1

                # Generate features for this move using the Processing class
                # For winner's moves, use winning_player as the perspective
                if current_player == winning_player:
                    move_features = Processing.process_game_state(board_state, captured_pieces, current_turn,
                                                                  winning_player, self.width, self.length)
                    winner_moves.append(move_features)
                # For loser's moves, use the losing player as the perspective
                else:
                    losing_player = 1 - winning_player
                    move_features = Processing.process_game_state(board_state, captured_pieces, current_turn,
                                                                  losing_player, self.width, self.length)
                    loser_moves.append(move_features)

        return {'winner_moves': winner_moves, 'loser_moves': loser_moves}

    '''
    *****************************************************************************
    * EXTRACT_COUNTS_ROW_COLUMN:
    * If feature exists add it to the player's feature list
    ***************************************************************************** 
    '''

    @staticmethod
    def extract_counts_row_column(move_features, feature_list, column_or_lines):
        counts = None
        for feature in move_features:
            if feature['name'] == column_or_lines:
                counts = feature['value']
                break

        if counts:
            # Add column counts for player (assuming player is the winning player)
            feature_list.extend(counts['player'])

    '''
    *****************************************************************************
    * GET_PIECE_COUNTS:
    * Get the amount of pieces for player / enemy, captured and on board
    ***************************************************************************** 
    '''

    @staticmethod
    def get_piece_counts(move_features, role):
        on_board = {size: 0 for size in 'SML'}
        captured = {size: 0 for size in 'SML'}

        for feature in move_features:
            for size in 'SML':
                if feature['name'] == f'{role}_pieces_on_board_{size}':
                    on_board[size] = feature['value']
                elif feature['name'] == f'{role}_pieces_captured_{size}':
                    captured[size] = feature['value']

        return on_board, captured

    '''
    *****************************************************************************
    * EXTRACT_PIECE_COUNT:
    * Extract the piece count feature for player / enemy
    ***************************************************************************** 
    '''

    @staticmethod
    def extract_piece_count(move_features, feature_list):
        player_on_board, player_captured = GameFeatures.get_piece_counts(move_features, 'player')  # get player info
        enemy_on_board, enemy_captured = GameFeatures.get_piece_counts(move_features, 'enemy')  # get enemy info

        feature_list.extend([
            player_on_board['S'], player_on_board['M'], player_on_board['L'],
            player_captured['S'], player_captured['M'], player_captured['L'],
            enemy_on_board['S'], enemy_on_board['M'], enemy_on_board['L'],
            enemy_captured['S'], enemy_captured['M'], enemy_captured['L'],
        ])

        return player_on_board, enemy_on_board

    '''
    *****************************************************************************
    * ADD_HOT_VECTOR:
    * Append the hot vector to the feature list for early-mid-end game strategy
    ***************************************************************************** 
    '''

    @staticmethod
    def add_hot_vector(move_features, feature_list, max_turns):
        current_turn = 0
        for feature in move_features:
            if feature['name'] == 'current_turn':
                current_turn = feature['value']
                break

        # Calculate game progress based on actual maximum turns from moves
        progress_ratio = current_turn / max_turns if max_turns > 0 else 0

        if progress_ratio <= 1 / 3:
            game_progress = [0, 0, 1]
        elif progress_ratio <= 2 / 3:
            game_progress = [0, 1, 0]
        else:
            game_progress = [1, 0, 0]

        feature_list.extend(game_progress)

    '''
    *****************************************************************************
    * PROCESS_MOVES:
    * Process a single move
    ***************************************************************************** 
    '''

    def process_moves(self, moves, max_turns, user_feature_list):
        for move_features in moves:
            feature_list = []
            self.extract_counts_row_column(move_features, feature_list, 'column_counts')  # extract columns count
            self.extract_counts_row_column(move_features, feature_list, 'line_counts')  # extract rows count
            # extract pieces on board, player and enemy
            player_pieces_on_board, enemy_pieces_on_board = self.extract_piece_count(move_features, feature_list)

            # extract avg_x, avg_y and spread
            values = {'player_avg_x': 0, 'player_avg_y': 0, 'player_spread': 0}
            for feature in move_features:
                name = feature['name']
                if name in values:
                    values[name] = feature['value']

            feature_list.extend([values['player_avg_x'], values['player_avg_y'], values['player_spread']])

            total_pieces_on_board = sum(player_pieces_on_board.values()) + sum(enemy_pieces_on_board.values())
            feature_list.append(total_pieces_on_board)

            # Add 1-hot encoded game progress
            GameFeatures.add_hot_vector(move_features, feature_list, max_turns)
            user_feature_list.append(feature_list)


    '''
    *****************************************************************************
    * CREATE_FEATURE_LISTS:
    * Create the winner and loser lists
    ***************************************************************************** 
    '''
    def create_feature_lists(self, moves_dict):
        winner_feature_lists = []
        loser_feature_lists = []

        winner_moves = moves_dict['winner_moves']
        loser_moves = moves_dict['loser_moves']

        # Calculate maximum turns from the total number of moves
        total_moves = len(winner_moves) + len(loser_moves)
        max_turns = total_moves * 2

        # Process winner moves
        self.process_moves(winner_moves, max_turns, winner_feature_lists)
        self.process_moves(loser_moves, max_turns, loser_feature_lists)

        return {'winner_features': winner_feature_lists, 'loser_features': loser_feature_lists}
