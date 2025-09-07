from heuristics import *

'''
*****************************************************************************
* HUMAN_AGENT:
* Lets the user play manually
***************************************************************************** 
'''


def human_agent(curr_state, agent_id, time_limit):
    print("insert action")
    pawn = str(input("insert pawn: "))
    if pawn.__len__() != 2:
        print("invalid input")
        return None
    location = str(input("insert location: "))
    if location.__len__() != 1:
        print("invalid input")
        return None
    return pawn, location


'''
*****************************************************************************
* GREEDY_IMPROVED_AGENT:
* This heuristic finds the best move based on the AlternativeScoreCalcFunc
***************************************************************************** 
'''


def greedy_improved_agent(curr_state, agent_id, time_limit):
    neighbor_list = curr_state.get_neighbors()
    max_heuristic = -float("inf")
    max_neighbor = None
    for move, state in neighbor_list:
        curr_heuristic = alternative_score_calc_func(state, agent_id)
        if curr_heuristic >= max_heuristic:
            max_heuristic = curr_heuristic
            max_neighbor = (move, state)
    return max_neighbor[0] if max_neighbor else None


'''
*****************************************************************************
* RANDOM_IMPROVED_AGENT:
* First check for a winning move, otherwise go one level deeper.
* Using this extra depth, find the best move based on winning chance.
***************************************************************************** 
'''


def random_improved_agent(curr_state, agent_id, time_limit):
    winning_move = check_winning_move(curr_state, agent_id)
    if winning_move:
        return winning_move

    best_moves = get_best_moves_by_future_win_chances(curr_state, agent_id)
    return random.choice(best_moves) if best_moves else None


'''
*****************************************************************************
* PROACTIVE_GOBBLER_AGENT:
* Sort of MiniMax based on the a
* Using this extra depth, find the best move based on
* AlternativeScoreCalcFunc.
***************************************************************************** 
'''


def proactive_gobbler_agent(curr_state, agent_id, time_limit):
    start_time = time.time()
    neighbor_list = curr_state.get_neighbors()
    if not neighbor_list:
        return None

    best_score_for_us = float('-inf')
    best_actions = []

    for action, next_state in neighbor_list:
        if gge.is_final_state(next_state) == agent_id + 1:
            return action

        opponent_possible_moves = next_state.get_neighbors()
        if not opponent_possible_moves:
            score_after_opponent_move = alternative_score_calc_func(next_state, agent_id)
        else:
            min_score_for_us_after_opp_move = float('inf')
            for opp_action, opp_next_state in opponent_possible_moves:
                score_if_opponent_makes_this_move = alternative_score_calc_func(opp_next_state, agent_id)
                if score_if_opponent_makes_this_move < min_score_for_us_after_opp_move:
                    min_score_for_us_after_opp_move = score_if_opponent_makes_this_move
            score_after_opponent_move = min_score_for_us_after_opp_move

        if score_after_opponent_move > best_score_for_us:
            best_score_for_us = score_after_opponent_move
            best_actions = [action]
        elif score_after_opponent_move == best_score_for_us:
            best_actions.append(action)

        if time.time() - start_time > time_limit * 0.95:
            break

    if not best_actions:
        return random.choice(neighbor_list)[0]

    return random.choice(best_actions)


'''
*****************************************************************************
* BLOCKING_RANDOM_AGENT:
* Win if possible, otherwise focus on blocking the opponent.
***************************************************************************** 
'''


def blocking_random_agent(curr_state, agent_id, time_limit):
    start_time = time.time()
    my_possible_moves = curr_state.get_neighbors()
    if not my_possible_moves:
        return None

    for action, next_state in my_possible_moves:
        if gge.is_final_state(next_state) == agent_id + 1:
            return action

    blocking_moves = get_safe_blocking_moves(curr_state, agent_id, time_limit, start_time)

    if time.time() - start_time >= time_limit * 0.99:
        return random.choice(blocking_moves) if blocking_moves else random.choice(my_possible_moves)[0]

    return random.choice(blocking_moves) if blocking_moves else random.choice(my_possible_moves)[0]


'''
*****************************************************************************
* BLOCKING_MORE_RANDOM_AGENT:
* Block the opponent.
***************************************************************************** 
'''


def blocking_more_random_agent(curr_state, agent_id, time_limit):
    start_time = time.time()
    my_possible_moves = curr_state.get_neighbors()
    if not my_possible_moves:
        return None

    blocking_moves = get_safe_blocking_moves(curr_state, agent_id, time_limit, start_time)

    if time.time() - start_time >= time_limit * 0.99:
        return random.choice(blocking_moves) if blocking_moves else random.choice(my_possible_moves)[0]

    return random.choice(blocking_moves) if blocking_moves else random.choice(my_possible_moves)[0]


'''
*****************************************************************************
* STAGE_ADAPTIVE_AGENT:
* Agent based on the neural network, presented in the PDF.
***************************************************************************** 
'''


def stage_adaptive_agent(curr_state, agent_id, time_limit):
    start_time = time.time()
    neighbors = curr_state.get_neighbors()
    if not neighbors:
        return None

    opponent_id = 1 - agent_id
    player_pawns = curr_state.player1_pawns if agent_id == 0 else curr_state.player2_pawns

    # Win if possible
    winning_move = check_winning_move(curr_state, agent_id)
    if winning_move:
        return winning_move

    blocking_moves = []
    center_blocking_moves = []
    new_piece_blocking = []
    existing_piece_blocking = []

    center_controlled, center_piece = has_center_control(player_pawns)

    # Track recently placed pieces to avoid moving them
    recently_placed = set()
    if hasattr(curr_state, 'last_move') and curr_state.last_move:
        _, last_action = curr_state.last_move
        if last_action and last_action[0] in player_pawns:
            recently_placed.add(last_action[0])

    for move, new_state in neighbors:
        pawn_str = move[0]
        curr_loc = player_pawns[pawn_str][0]
        new_piece = is_new_piece(curr_loc)

        # Skip moves that move recently placed pieces
        if pawn_str in recently_placed and not is_new_piece(curr_loc):
            continue

        # Check if opponent can win after this move
        opp_wins = False
        for opp_move, opp_state in new_state.get_neighbors():
            if gge.is_final_state(opp_state) == opponent_id + 1:
                opp_wins = True
                break

        if not opp_wins:
            blocking_moves.append(move)

            # Check if we're covering our own piece
            loc_idx = int(move[1])
            row, col = loc_idx // 3, loc_idx % 3

            top_owner, _ = get_top_piece_owner_and_size(curr_state, row, col)

            if top_owner != agent_id:
                if new_piece:
                    new_piece_blocking.append(move)
                else:
                    existing_piece_blocking.append(move)

                # Prioritize moves that protect center
                if center_controlled and move[0] != center_piece and loc_idx == 4:
                    center_blocking_moves.append(move)

    candidate_lists = [
        [m for m in center_blocking_moves if m in new_piece_blocking],
        new_piece_blocking,
        [m for m in center_blocking_moves if m in existing_piece_blocking],
        existing_piece_blocking,
        blocking_moves
    ]

    filtered_neighbors = []
    for candidate in candidate_lists:
        if candidate:
            filtered_neighbors = [(move, state) for move, state in neighbors if move in candidate]
            break

    if not filtered_neighbors:
        filtered_neighbors = neighbors

    # Use the learned based heuristic
    total_pieces = gge.num_of_pawns(curr_state, 0) + gge.num_of_pawns(curr_state, 1)
    stage = get_stage(total_pieces)
    heuristics = load_learned_heuristic(stage)

    best_score = float("-inf")
    best_actions = []

    for move, next_state in filtered_neighbors:
        if time.time() - start_time > time_limit * 0.95:
            break

        score, is_win = score_move(
            curr_state, move, next_state, agent_id, stage,
            center_controlled, center_piece, heuristics,
            start_time, time_limit
        )
        if is_win:
            return move

        if score > best_score:
            best_score = score
            best_actions = [move]
        elif score == best_score:
            best_actions.append(move)

    # Select best action
    if best_actions:
        return random.choice(best_actions)
    return random.choice(filtered_neighbors)[0]
