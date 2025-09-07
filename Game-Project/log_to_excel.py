import os
import glob
from openpyxl import Workbook


def _extract_winner(content_lines):
    if not content_lines:
        return None
    winner_line = content_lines[0].strip()
    if not winner_line.startswith("winner:"):
        return None
    try:
        return int(winner_line.split(':')[1])
    except (ValueError, IndexError):
        return None


def _parse_cell_content(cell_content_raw):
    top_pawn_info = None
    if cell_content_raw:
        candidate_pawns = [p.strip() for p in cell_content_raw.split(',') if p.strip()]

        if candidate_pawns:
            visible_pawn = candidate_pawns[-1].strip()  # get only visible pawn - top pawn

            if ':' in visible_pawn:
                player_and_size_parts = visible_pawn.split(':')
                if len(player_and_size_parts) == 2:
                    player = player_and_size_parts[0]
                    size = player_and_size_parts[1]

                    if player.isdigit() and size in ['S', 'M', 'L']:
                        top_pawn_info = f"{player}:{size}"
    return top_pawn_info


def _parse_turn_board_state(content_lines, start_index):
    # Init an empty board
    board_state = {}
    for row in range(3):
        for col in range(3):
            cell_position = (row, col)
            board_state[cell_position] = None

    lines_read = 0
    for idx in range(9):  # 9 lines per turn
        pos_line_idx = start_index + idx

        if pos_line_idx >= len(content_lines):
            return None, 0

        pos_line = content_lines[pos_line_idx].strip()
        parts = pos_line.split(':')
        pos_str = parts[0]

        row, col = -1, -1

        if pos_str.startswith("pos"):
            try:
                hyphen_idx = pos_str.find('-')
                if hyphen_idx != -1:
                    row_str = pos_str[3:hyphen_idx]
                    col_str = pos_str[hyphen_idx + 1:]

                    if row_str.isdigit() and col_str.isdigit():
                        row = int(row_str)
                        col = int(col_str)
            except (ValueError, IndexError):
                pass

        cell_content_raw = ":".join(parts[1:]).strip()
        board_state[(row, col)] = _parse_cell_content(cell_content_raw)
        lines_read += 1

    return board_state, lines_read


def parse_log_file(log_file_path):
    with open(log_file_path, 'r') as f:
        content = f.read().splitlines()

    winner = _extract_winner(content)
    if winner is None:
        return None

    turns_data = []
    j = 0
    while j < len(content):
        line = content[j].strip()

        if line.startswith("turn:"):
            turn_num = int(line.split(':')[1])

            board_state, lines_read = _parse_turn_board_state(content, j + 1)
            turns_data.append({'turn_num': turn_num, 'board_state': board_state})
            j = (j + 1) + lines_read
        else:
            j += 1

    return {'winner': winner, 'turns': turns_data}


def format_pawn_for_excel(pawn_string):
    if pawn_string is None:
        return '-'
    return pawn_string


def _setup_excel_workbook():
    wb = Workbook()
    ws = wb.active
    ws.title = "Game Logs"

    headers = [
        "Game Index", "Agent Player 1", "Agent Player 2", "Winner", "Turn",
        "Cell(0,0)", "Cell(0,1)", "Cell(0,2)",
        "Cell(1,0)", "Cell(1,1)", "Cell(1,2)",
        "Cell(2,0)", "Cell(2,1)", "Cell(2,2)"
    ]
    ws.append(headers)

    return wb, ws


def _get_sorted_log_files(log_dir):
    log_files = glob.glob(os.path.join(log_dir, "log *.txt"))

    def get_game_index_from_filename(filepath):
        basename = os.path.basename(filepath)
        try:
            return int(basename.split(' ')[1].split('.')[0])
        except (ValueError, IndexError):
            return 0

    log_files.sort(key=get_game_index_from_filename)
    return log_files


def _process_single_game_to_excel(ws, file_path, agent_mapping):
    filename = os.path.basename(file_path)
    game_index = 0
    try:
        game_index_str = filename.replace('log ', '').replace('.txt', '')
        if game_index_str.isdigit():
            game_index = int(game_index_str)
    except (ValueError, IndexError):
        pass

    agents = agent_mapping.get(game_index, ("Agent 1", "Agent 2"))
    game_data = parse_log_file(file_path)
    if not game_data or not game_data['turns']:
        return

    winner = game_data['winner']

    for turn_info in game_data['turns']:
        turn_num = turn_info['turn_num']
        board_state = turn_info['board_state']

        row_data = [game_index, agents[0], agents[1], winner, turn_num]

        for row in range(3):
            for col in range(3):
                pawn_string = board_state.get((row, col))
                row_data.append(format_pawn_for_excel(pawn_string))

        ws.append(row_data)

    ws.append([])


def log_to_excel(log_dir, output_excel_file, agent_mapping):
    wb, ws = _setup_excel_workbook()
    log_files = _get_sorted_log_files(log_dir)

    for file_path in log_files:
        _process_single_game_to_excel(ws, file_path, agent_mapping)

    wb.save(output_excel_file)


if __name__ == '__main__':
    log_directory = "./logs"
    output_file_name = "tournament_results.xlsx"
    agent_mapping = {}

    for i in range(15400, 15600):  # Range is exclusive at the end, so 10000 to 10499
        agent_mapping[i] = ("best_agent", "blocking_random_agent")

    log_to_excel(
        log_dir=log_directory,
        output_excel_file=output_file_name,
        agent_mapping=agent_mapping
    )
