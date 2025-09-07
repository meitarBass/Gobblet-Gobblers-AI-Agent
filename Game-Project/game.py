import time
import os

import Gobblet_Gobblers_Env as gge
import agents

time_limit = 100
steps_limit = 100

agents = {
    "human_agent": agents.human_agent,
    "greedy_improved_agent": agents.greedy_improved_agent,
    "random_improved_agent": agents.random_improved_agent,
    "blocking_random_agent": agents.blocking_random_agent,
    "blocking_more_random_agent": agents.blocking_more_random_agent,
    "proactive_gobbler_agent": agents.proactive_gobbler_agent,
    "best_agent": agents.stage_adaptive_agent,
}


# gets two functions of the agents and plays according to their selection
# plays the game and returns 1 if agent1 won and -1 if agent2 won and 0 if there is a tie
def play_game(agent_1_str, agent_2_str, game_index):
    agent_1 = agents[agent_1_str]
    agent_2 = agents[agent_2_str]
    filename = "./logs/log " + str(game_index) + ".txt"
    if not os.path.exists("./logs"):
        os.makedirs("./logs")
    file = open(filename, "a")
    s = gge.State()
    env = gge.GridWorldEnv('human')
    winner = None
    env.reset()
    env.render()
    steps_per_game = 0
    while winner is None:
        if env.s.turn == 0:
            print("player 0")
            start_time = time.time()
            chosen_step = agent_1(env.get_state(), 0, time_limit)
            end_time = time.time()
            if chosen_step is None:
                continue
            action = chosen_step[0], int(chosen_step[1])
            if (end_time - start_time) > time_limit and (agent_1_str in ["minimax", "alpha_beta", "expectimax"]):
                raise RuntimeError("Agent used too much time!")
            env.step(action)
            env.render()
            steps_per_game += 1
            print("time for step was", end_time - start_time)
        else:
            print("player 1")
            start_time = time.time()
            chosen_step = agent_2(env.get_state(), 1, time_limit)
            end_time = time.time()
            if chosen_step is None:
                continue
            action = chosen_step[0], int(chosen_step[1])
            if (end_time - start_time) > time_limit and (agent_2_str in ["minimax", "alpha_beta", "expectimax"]):
                raise RuntimeError("Agent used too much time!")
            env.step(action)
            env.render()
            steps_per_game += 1
            print("time for step was", end_time - start_time)

        s = env.get_state()

        text = env.get_log_state()
        file.write(text)

        time.sleep(0.5)
        winner = gge.is_final_state(env.s)
        if steps_per_game >= steps_limit:
            winner = 0
    file.write("\n")
    if winner == 0:
        print("tie")
    else:
        print("winner is:", winner)
    # write winner at beginning of file
    file.close()
    with open(filename, 'r+') as f:
        content = f.read()
        line = "winner:" + str(winner) + '\n'
        f.seek(0, 0)
        f.write(line.rstrip('\r\n') + '\n' + content)
    return winner


# plays many games between two agents and returns in percentage win per player and ties
def play_tournament(agent_1_str, agent_2_str, num_of_games, existing_logs):
    # agent_1 = agents[agent_1_str]
    # agent_2 = agents[agent_2_str]
    # score is [ties, wins for agent1, wins for agent2]
    score = [0, 0, 0]
    index = existing_logs

    for i in range(num_of_games):
        tmp_score = int(play_game(agent_1_str, agent_2_str, index))
        score[int(tmp_score)] = score[int(tmp_score)] + 1
        index += 1

    for j in range(num_of_games):
        tmp_score = int(play_game(agent_2_str, agent_1_str, index))
        real_tmp_score = 0
        # player nums are flipped, flip back
        if tmp_score != 0:
            if tmp_score == 1:
                real_tmp_score = 2
            else:
                real_tmp_score = 1
        score[int(real_tmp_score)] = score[int(real_tmp_score)] + 1
        index += 1

    print("ties: ", (score[0] / (num_of_games * 2)) * 100, "% ", agent_1_str, "player1 wins: ",
          (score[1] / (num_of_games * 2)) * 100,
          "% ", agent_2_str, "player2 wins: ", (score[2] / (num_of_games * 2)) * 100)

    print("")
