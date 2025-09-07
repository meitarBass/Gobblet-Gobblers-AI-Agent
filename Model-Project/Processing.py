class Processing:

    @staticmethod
    def init_piece_positions():
        piece_positions = {
            'player': {
                'S': {'on_board': [], 'not_on_board': [], 'captured': []},
                'M': {'on_board': [], 'not_on_board': [], 'captured': []},
                'L': {'on_board': [], 'not_on_board': [], 'captured': []}
            },
            'enemy': {
                'S': {'on_board': [], 'not_on_board': [], 'captured': []},
                'M': {'on_board': [], 'not_on_board': [], 'captured': []},
                'L': {'on_board': [], 'not_on_board': [], 'captured': []}
            }
        }
        return piece_positions

    @staticmethod
    def init_row_column_counts(length, width):
        line_counts = {'player': [0] * length, 'enemy': [0] * length}
        column_counts = {'player': [0] * width, 'enemy': [0] * width}
        return line_counts, column_counts

    '''
    *****************************************************************************
    * UPDATE_PIECES_ON_BOARD:
    * Updates the pieces positions to the current state
    ***************************************************************************** 
    '''

    @staticmethod
    def update_pieces_on_board(board_state, winning_player, piece_positions):
        # Track pieces by size on the board
        for (pos_x, pos_y), pieces in board_state.items():
            for piece in pieces:
                piece_type = piece['type']
                if piece['player'] == winning_player:
                    piece_positions['player'][piece_type]['on_board'].append((pos_x, pos_y))
                else:
                    piece_positions['enemy'][piece_type]['on_board'].append((pos_x, pos_y))

    '''
    *****************************************************************************
    * UPDATE_ROW_COLUMN_COUNTS:
    * Updates the current count for each row and column
    ***************************************************************************** 
    '''

    @staticmethod
    def update_row_column_counts(board_state, winning_player, line_counts, column_counts):
        for (pos_x, pos_y), pieces in board_state.items():
            for piece in pieces:
                if piece['player'] == winning_player:
                    line_counts['player'][pos_y] += 1
                    column_counts['player'][pos_x] += 1
                else:
                    line_counts['enemy'][pos_y] += 1
                    column_counts['enemy'][pos_x] += 1

    '''
    *****************************************************************************
    * GET_AVERAGE_COORDINATES_AND_SPREAD:
    * Used to update the relevant features in the features list
    ***************************************************************************** 
    '''

    @staticmethod
    def get_average_coordinates_and_spread(piece_positions, width, length):
        all_player_positions = []
        for size in ['S', 'M', 'L']:
            all_player_positions.extend(piece_positions['player'][size]['on_board'])

        if all_player_positions:
            avg_x = sum(pos[0] for pos in all_player_positions) / len(all_player_positions)
            avg_y = sum(pos[1] for pos in all_player_positions) / len(all_player_positions)

            # Calculate spread (average distance from center)
            center_x = width / 2
            center_y = length / 2
            spread = sum(((x - center_x) ** 2 + (y - center_y) ** 2) ** 0.5 for x, y in all_player_positions) / len(
                all_player_positions)
        else:
            avg_x = 0
            avg_y = 0
            spread = 0

        return avg_x, avg_y, spread

    '''
    *****************************************************************************
    * GET_POSITION_FEATURES:
    * Returns the position based features - for both player and enemy
    ***************************************************************************** 
    '''

    @staticmethod
    def get_position_features(line_counts, column_counts, piece_positions, avg_x, avg_y, spread, current_turn):
        move_features = []
        move_features.extend([
            # Line and column counts
            {'name': 'line_counts', 'value': line_counts},
            {'name': 'column_counts', 'value': column_counts},

            # Player piece counts by size
            {'name': 'player_pieces_on_board_S', 'value': len(piece_positions['player']['S']['on_board'])},
            {'name': 'player_pieces_on_board_M', 'value': len(piece_positions['player']['M']['on_board'])},
            {'name': 'player_pieces_on_board_L', 'value': len(piece_positions['player']['L']['on_board'])},
            {'name': 'player_pieces_captured_S', 'value': 0},
            {'name': 'player_pieces_captured_M', 'value': 0},
            {'name': 'player_pieces_captured_L', 'value': 0},

            # Enemy piece counts by size
            {'name': 'enemy_pieces_on_board_S', 'value': len(piece_positions['enemy']['S']['on_board'])},
            {'name': 'enemy_pieces_on_board_M', 'value': len(piece_positions['enemy']['M']['on_board'])},
            {'name': 'enemy_pieces_on_board_L', 'value': len(piece_positions['enemy']['L']['on_board'])},
            {'name': 'enemy_pieces_captured_S', 'value': 0},
            {'name': 'enemy_pieces_captured_M', 'value': 0},
            {'name': 'enemy_pieces_captured_L', 'value': 0},

            # Player piece coordinates and spread
            {'name': 'player_avg_x', 'value': avg_x},
            {'name': 'player_avg_y', 'value': avg_y},
            {'name': 'player_spread', 'value': spread},

            # Turn number
            {'name': 'current_turn', 'value': current_turn}
        ])
        return move_features

    '''
    *****************************************************************************
    * PROCESS_GAME_STATE:
    * Process a single state
    ***************************************************************************** 
    '''

    @staticmethod
    def process_game_state(board_state, captured_pieces, current_turn, winning_player, width, length):
        piece_positions = Processing.init_piece_positions()
        Processing.update_pieces_on_board(board_state, winning_player, piece_positions)
        line_counts, column_counts = Processing.init_row_column_counts(length, width)
        Processing.update_row_column_counts(board_state, winning_player, line_counts, column_counts)
        avg_x, avg_y, spread = Processing.get_average_coordinates_and_spread(piece_positions, width, length)

        return Processing.get_position_features(line_counts, column_counts, piece_positions,
                                                avg_x, avg_y, spread, current_turn)
