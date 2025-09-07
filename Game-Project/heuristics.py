import random
import numpy as np
import Gobblet_Gobblers_Env as gge
import time

not_on_board = np.array([-1, -1])

'''
*****************************************************************************
* AUXILIARY:
* Check if there's a winning move
***************************************************************************** 
'''


def check_winning_move(curr_state, agent_id):
    neighbor_list = curr_state.get_neighbors()
    for move, next_state in neighbor_list:
        if gge.is_final_state(next_state) == agent_id + 1:
            return move
    return None


'''
*****************************************************************************
* AUXILIARY:
* Checks whether a pawn is hidden under another pawn
***************************************************************************** 
'''


def is_hidden(state, agent_id, pawn):
    pawn_location = gge.find_curr_location(state, pawn, agent_id)
    pawn_size = state.player1_pawns[pawn][1] if agent_id == 0 else state.player2_pawns[pawn][1]

    for player_pawns in [state.player1_pawns, state.player2_pawns]:
        for _, (loc, size) in player_pawns.items():
            if np.array_equal(loc, pawn_location) and gge.size_cmp(size, pawn_size) == 1:
                return True
    return False


'''
*****************************************************************************
* AUXILIARY:
* Counts player's total number of visible pieces, 
* Returns by: row, col, diagonal, big pieces, hidden
***************************************************************************** 
'''


def count_visible_pieces(state, player_pawns, agent_id):
    rows = np.zeros(3)
    cols = np.zeros(3)
    diags = np.zeros(2)
    size_values = {'S': 1, 'M': 2, 'B': 3}
    big_piece_score = 0
    hidden_penalty = 0

    for pawn, (pos, size) in player_pawns.items():
        if not np.array_equal(pos, not_on_board):
            hidden = is_hidden(state, agent_id, pawn)
            r, c = pos
            if not hidden:
                rows[r] += 1
                cols[c] += 1
                if r == c:
                    diags[0] += 1
                if r + c == 2:
                    diags[1] += 1
                big_piece_score += size_values[size]
            else:
                hidden_penalty += 1

    return rows, cols, diags, big_piece_score, hidden_penalty


def score_lines(rows, cols, diags):
    val = 0
    for cnt in np.concatenate((rows, cols, diags)):
        if cnt == 1:
            val += 1
        elif cnt == 2:
            val += 5
        elif cnt >= 3:
            val += 50
    return val


def two_in_a_row_score(rows, cols, diags):
    return sum(1 for cnt in np.concatenate((rows, cols, diags)) if cnt == 2)


def handle_final_state(state, agent_id, win_score=1000, tie_score=0, lose_score=-1000):
    is_final = gge.is_final_state(state)
    if is_final is None:
        return None  # Game not over
    if is_final == 0:
        return tie_score
    winner = int(is_final) - 1
    return win_score if winner == agent_id else lose_score


'''
*****************************************************************************
* AUXILIARY:
* Checks whether the opponent can win the next move
***************************************************************************** 
'''


def opponent_can_win_after(state, agent_id):
    return any(gge.is_final_state(s) == 1 - agent_id + 1 for _, s in state.get_neighbors())


'''
*****************************************************************************
* AUXILIARY:
* Returns the best move based on future winning (depth 2)
***************************************************************************** 
'''


def get_best_moves_by_future_win_chances(curr_state, agent_id):
    best_moves = []
    best_move_value = float('-inf')

    neighbor_list = curr_state.get_neighbors()
    if not neighbor_list:
        return []

    for move, next_state in neighbor_list:
        move_score = 0
        for _, opp_next_state in next_state.get_neighbors():
            if check_winning_move(opp_next_state, 1 - agent_id):
                move_score = -4
            for _, next_next_state in opp_next_state.get_neighbors():
                if gge.is_final_state(next_next_state) == agent_id + 1:
                    move_score += 1

        if move_score > best_move_value:
            best_move_value = move_score
            best_moves = [move]
        elif move_score == best_move_value:
            best_moves.append(move)

    return best_moves


'''
*****************************************************************************
* AUXILIARY:
* Returns the top piece on position (row, col) and its owner 
***************************************************************************** 
'''


def get_top_piece_owner_and_size(state, row, col):
    top_owner = None
    top_size_val = 0
    size_map = {'S': 1, 'M': 2, 'B': 3}

    for player_id, pawns in [(0, state.player1_pawns), (1, state.player2_pawns)]:
        for _, (loc, size) in pawns.items():
            if np.array_equal(loc, [row, col]):
                size_val = size_map[size]
                if size_val > top_size_val:
                    top_owner = player_id
                    top_size_val = size_val
    return top_owner, top_size_val


def is_new_piece(pawn_pos):
    return np.array_equal(pawn_pos, not_on_board)


'''
*****************************************************************************
* AUXILIARY:
* Checks whether the player is controlling the center
***************************************************************************** 
'''


def has_center_control(player_pawns):
    for pawn, (loc, size) in player_pawns.items():
        if np.array_equal(loc, [1, 1]) and size == 'B':
            return True, pawn
    return False, None


'''
*****************************************************************************
* AUXILIARY:
* Returns a list of possible block moves
***************************************************************************** 
'''


def get_safe_blocking_moves(state, agent_id, time_limit, start_time):
    blocking_moves = []
    for my_action, my_next_state in state.get_neighbors():
        if time.time() - start_time >= time_limit * 0.99:
            break
        opponent_wins = any(
            gge.is_final_state(opp_final_state) == (1 - agent_id) + 1
            for _, opp_final_state in my_next_state.get_neighbors()
        )
        if not opponent_wins:
            blocking_moves.append(my_action)
    return blocking_moves


'''
*****************************************************************************
* AUXILIARY:
* Returns score based on larger number of pieces, larger pieces 
***************************************************************************** 
'''


def alternative_score_calc_func(state, agent_id, time_limit=None):
    # Count visible structure for both players
    rows1, cols1, diags1, big1, hidden1 = \
        count_visible_pieces(state, state.player1_pawns, 0)
    rows2, cols2, diags2, big2, hidden2 = \
        count_visible_pieces(state, state.player2_pawns, 1)

    score_1 = score_lines(rows1, cols1, diags1) + big1 - hidden1
    score_2 = score_lines(rows2, cols2, diags2) + big2 - hidden2

    final_score = handle_final_state(state, agent_id)
    if final_score is not None:
        return final_score

    if agent_id == 0:
        return score_1 - score_2
    else:
        return score_2 - score_1


'''
*****************************************************************************
* AUXILIARY:
* Returns the game stage - used to determine the current heuristic fron the 
* neurtal network
***************************************************************************** 
'''


def get_stage(total_pieces):
    if total_pieces <= 4:
        return "early"
    elif total_pieces <= 8:
        return "mid"
    return "late"


def load_learned_heuristic(stage):
    import json
    import os
    heuristics = {}
    try:
        dir_path = os.path.dirname(os.path.realpath(__file__))
        with open(os.path.join(dir_path, 'game_heuristics.json')) as f:
            heuristics = json.load(f).get(stage, [])
    except:
        pass
    return heuristics


'''
*****************************************************************************
* AUXILIARY:
* Calculate the move score based on the learned heuristic
***************************************************************************** 
'''


def score_move(curr_state, move, next_state, agent_id, stage,
               center_controlled, center_piece, heuristics,
               start_time, time_limit):
    # Immediate win
    if gge.is_final_state(next_state) == agent_id + 1:
        return float("inf"), True

    # Opponent best response (most damaging)
    opponent_moves = next_state.get_neighbors()
    if not opponent_moves:
        score_after_opponent = alternative_score_calc_func(next_state, agent_id)
    else:
        min_heuristic = float("inf")
        for _, opp_state in opponent_moves:
            if time.time() - start_time > time_limit * 0.95:
                break
            h = alternative_score_calc_func(opp_state, agent_id)
            if h < min_heuristic:
                min_heuristic = h
        score_after_opponent = min_heuristic

    score = score_after_opponent

    # Extract move properties
    loc_idx = int(move[1])
    row, col = loc_idx // 3, loc_idx % 3
    piece_size = gge.get_piece_size(move[0])
    pawn_str = move[0]
    player_pawns = curr_state.player1_pawns if agent_id == 0 else curr_state.player2_pawns
    curr_loc = player_pawns[pawn_str][0]
    new_piece = is_new_piece(curr_loc)

    # New piece bonus
    if new_piece:
        if stage == "early":
            score += 5
        elif stage == "mid":
            score += 3
        else:
            score += 1

    # center control bonus
    if loc_idx == 4:
        if piece_size == "B":
            score += 8 if stage == "early" else 5
        elif piece_size == "M":
            score += 4 if stage == "early" else 2

    # prefer controlling center
    if center_controlled:
        if pawn_str == center_piece and loc_idx != 4:
            score -= 1000
        elif loc_idx == 4 and piece_size != "B" and pawn_str != center_piece:
            score += 3

    # avoid self cover
    top_owner, _ = get_top_piece_owner_and_size(curr_state, row, col)
    if top_owner == agent_id:
        score -= 15

    # using the heuristic from the neural network
    for heuristic in heuristics:
        if time.time() - start_time > time_limit * 0.95:
            break
        i, j = heuristic["feature_pair"]
        weights = heuristic["weights"]
        gamma = heuristic["gamma"]
        beta = heuristic["beta"]

        sim_feature_i = 1.0 if i in [2, 6, 8] else 0.2
        sim_feature_j = 1.0 if j in [3, 7, 9] else 0.2

        term = weights[0] * sim_feature_i + weights[1] * sim_feature_j
        contribution = gamma * term + beta
        score += contribution

    return score, False
