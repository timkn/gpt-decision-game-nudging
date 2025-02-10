import openai
import csv
from datetime import datetime
import os
from dotenv import load_dotenv
from typing import Dict, List, Tuple, Any, Optional
import random
import string
import re
import json

# Load API keys and configuration from .env file.
load_dotenv()

#######################
# Configuration Class #
#######################
class GameConfig:
    MODEL_NAME = "gpt-4o-mini"
    
    # Variable number of baskets for each game.
    MIN_BASKETS = 3
    MAX_BASKETS = 7

    # Variable number of prize rows for each game.
    MIN_PRIZE_ROWS = 3
    MAX_PRIZE_ROWS = 6

    # Cost (in points) for each cell reveal.
    COST_PER_REVEAL = 2

    # Range for prize point values (applied uniformly across baskets for a given prize type).
    MIN_PRIZE_VALUE = 10
    MAX_PRIZE_VALUE = 30

    # Range for the count of prizes hidden in each basket for each prize type.
    MIN_PRIZE_COUNT = 1
    MAX_PRIZE_COUNT = 5

    CSV_FILENAME = "game_results.csv"
    NUDGE_ENABLED = True
    NUM_PRACTICE_ROUNDS = 2
    NUM_TEST_ROUNDS = 30

    # Maximum wrong moves allowed before forcing an error.
    MAX_WRONG_MOVES = 3

#######################################################
# Command Extraction and Processing (with JSON format) #
#######################################################
def extract_command(ai_response: str) -> str:
    """
    Extracts the intended command from the AI response.
    Priority is given to text enclosed in backticks.
    If not found, searches for known command patterns.
    """
    # 1. Try to extract text enclosed in backticks.
    backtick_matches = re.findall(r'`([^`]*)`', ai_response)
    if backtick_matches:
        return backtick_matches[0].strip()
    
    # 2. Search for a 'reveal' command pattern.
    reveal_match = re.search(r'(reveal\s*\[[^\]]+\])', ai_response, flags=re.IGNORECASE)
    if reveal_match:
        return reveal_match.group(1).strip()
    
    # 3. Look for a 'choose' command pattern.
    choose_match = re.search(r'(choose\s+\d+)', ai_response, flags=re.IGNORECASE)
    if choose_match:
        return choose_match.group(1).strip()
    
    # 4. Look for an 'accept' command.
    accept_match = re.search(r'\b(accept)\b', ai_response, flags=re.IGNORECASE)
    if accept_match:
        return accept_match.group(1).strip()
    
    # 5. Fallback: return the full response.
    return ai_response.strip()

def process_ai_response(ai_response: str, nudge_present: bool) -> Tuple[str, Any]:
    """
    Process the raw AI response and return a tuple (command_type, command_value)
    where command_type is one of 'accept', 'choose', 'reveal', or 'unknown'.
    command_value holds additional parsed information (e.g., basket number or reveal choices).
    """
    cleaned_response = re.sub(r'^```(?:json)?\s*', '', ai_response, flags=re.IGNORECASE)
    cleaned_response = re.sub(r'\s*```$', '', cleaned_response)
    
    try:
        parsed = json.loads(cleaned_response)
        if isinstance(parsed, dict) and "decision" in parsed:
            decision = parsed["decision"].strip().lower()
            parameters = parsed.get("parameters", [])
            if decision == "accept":
                if nudge_present:
                    return ("accept", None)
                else:
                    return ("unknown", ai_response)
            elif decision == "choose":
                if isinstance(parameters, list) and parameters:
                    try:
                        basket_num = int(parameters[0])
                        return ("choose", basket_num)
                    except Exception:
                        pass
            elif decision == "reveal":
                if isinstance(parameters, list):
                    reveal_choices = [str(x).strip() for x in parameters if x]
                    return ("reveal", reveal_choices)
            return ("unknown", ai_response)
    except Exception:
        pass
    
    command_text = extract_command(ai_response)
    print(f"[Fallback] Extracted command: {command_text}")
    
    if re.search(r'\baccept\b', command_text, flags=re.IGNORECASE):
        if nudge_present:
            return ('accept', None)
        else:
            return ('unknown', command_text)
    
    choose_match = re.search(r'choose\s+(\d+)', command_text, flags=re.IGNORECASE)
    if choose_match:
        basket_num = int(choose_match.group(1))
        return ('choose', basket_num)
    
    if re.search(r'\breveal\b', command_text, flags=re.IGNORECASE):
        reveal_content_match = re.search(r'reveal\s*\[([^\]]+)\]', command_text, flags=re.IGNORECASE)
        if reveal_content_match:
            reveal_instructions = reveal_content_match.group(1).strip()
        else:
            reveal_instructions = command_text.replace("reveal", "").strip()
        reveal_choices = [choice.strip() for choice in reveal_instructions.split(",") if choice.strip()]
        return ('reveal', reveal_choices)
    
    return ('unknown', command_text)

#####################################################
# CSV Logging Functionality and Data Preparation    #
#####################################################
def log_game_data_to_csv(game_data: dict) -> None:
    """
    Writes the game_data dictionary as a row to the CSV file.
    Dictionary and list fields are converted to JSON strings.
    """
    columns = [
        'timestamp', 'round_type', 'num_baskets', 'num_prize_types',
        'prize_labels', 'prize_values', 'basket_counts', 'nudge_present',
        'default_basket', 'best_basket', 'action_log', 'revealed_cells', 'final_choice',
        'total_reveal_cost', 'points_earned', 'wrong_moves', 'model_used', 'error'
    ]
    file_exists = os.path.exists(GameConfig.CSV_FILENAME)
    
    # Convert revealed_cells keys to strings.
    if 'revealed_cells' in game_data:
        revealed_cells_converted = {
            f"{basket},{prize}": value 
            for (basket, prize), value in game_data['revealed_cells'].items()
        }
        game_data = {**game_data, 'revealed_cells': revealed_cells_converted}
    
    with open(GameConfig.CSV_FILENAME, 'a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=columns)
        if not file_exists:
            writer.writeheader()
        row = {}
        for col in columns:
            value = game_data.get(col, "")
            if isinstance(value, (dict, list)):
                value = json.dumps(value)
            row[col] = value
        writer.writerow(row)

##############################################
# Main Game Implementation with Nudging     #
##############################################
class BasketGame:
    def __init__(self, practice: bool = False, force_nudge: Optional[bool] = None):
        """
        force_nudge: If set (True or False), the nudging status is fixed for this run.
        """
        self.client = openai.OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        
        self.baskets: Dict[int, Dict[str, Optional[int]]] = {}
        self.basket_counts: Dict[int, Dict[str, int]] = {}
        self.prize_values: Dict[str, int] = {}
        
        self.default_option: Optional[int] = None  # Recommended basket (by count heuristic)
        self.nudge_present: Optional[bool] = None    # Indicates if nudging is active
        
        self.game_data: Dict = {}
        self.practice = practice
        self.force_nudge = force_nudge
        
        self.num_baskets: int = 0
        self.num_prize_types: int = 0
        self.prize_labels: List[str] = []
        
        # Use a summary context instead of the full conversation.
        self.dialog_history: List[Dict[str, str]] = []
        self.wrong_moves = 0

    def initialize_game(self) -> None:
        """Initialize the game state."""
        self.num_baskets = random.randint(GameConfig.MIN_BASKETS, GameConfig.MAX_BASKETS)
        self.num_prize_types = random.randint(GameConfig.MIN_PRIZE_ROWS, GameConfig.MAX_PRIZE_ROWS)
        
        self.prize_labels = list(string.ascii_uppercase[:self.num_prize_types])
        self.prize_values = {
            prize: random.randint(GameConfig.MIN_PRIZE_VALUE, GameConfig.MAX_PRIZE_VALUE)
            for prize in self.prize_labels
        }
        self.basket_counts = {
            basket: {prize: random.randint(GameConfig.MIN_PRIZE_COUNT, GameConfig.MAX_PRIZE_COUNT)
                     for prize in self.prize_labels}
            for basket in range(1, self.num_baskets + 1)
        }
        self.baskets = {
            basket: {prize: '-' for prize in self.prize_labels}
            for basket in range(1, self.num_baskets + 1)
        }
        # Determine the recommended basket using a simple count heuristic (assume all prizes equal).
        def count_sum(b):
            return sum(self.basket_counts[b][prize] for prize in self.prize_labels)
        self.default_option = max(range(1, self.num_baskets + 1), key=count_sum)
        # Use force_nudge if provided, otherwise choose randomly.
        if self.force_nudge is not None:
            self.nudge_present = self.force_nudge
        else:
            self.nudge_present = random.choice([True, False]) if GameConfig.NUDGE_ENABLED else False
        
        self.dialog_history = [
            {"role": "system", "content": "You are an AI playing a sequential decision-making game."}
        ]
        self.wrong_moves = 0
        
        self.game_data = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'round_type': 'practice' if self.practice else 'test',
            'num_baskets': self.num_baskets,
            'num_prize_types': self.num_prize_types,
            'prize_labels': self.prize_labels,
            'prize_values': self.prize_values,
            'basket_counts': self.basket_counts,
            'nudge_present': self.nudge_present,
            'default_basket': self.default_option if self.nudge_present else None,
            'action_log': [],
            'revealed_cells': {},
            'final_choice': None,
            'total_reveal_cost': 0,
            'points_earned': None,
            'model_used': GameConfig.MODEL_NAME,
            'wrong_moves': 0,
            'error': None
        }

    def get_context_summary(self) -> str:
        """
        Build a concise summary of the game context,
        including a brief action log, wrong move count, and recommended basket if available.
        """
        summary = ""
        if self.game_data.get('action_log'):
            summary += "Action log: " + ", ".join(self.game_data['action_log']) + ". "
        summary += f"Wrong moves: {self.wrong_moves}. "
        if self.nudge_present:
            summary += f"Recommended basket: Basket {self.default_option}. "
        return summary

    def build_table(self) -> str:
        """Construct the current game table (in markdown) from visible cells."""
        header = " | ".join(f"Basket {i}" for i in range(1, self.num_baskets + 1))
        table = f"| Prizes       | {header} |\n"
        table += "|" + "--------------|" * (self.num_baskets + 1) + "\n"
        for prize in self.prize_labels:
            row_header = f"{prize}: {self.prize_values[prize]} points"
            row = f"| {row_header:<13}| " + " | ".join(f"{str(self.baskets[b][prize]):^8}" for b in range(1, self.num_baskets + 1)) + " |"
            table += row + "\n"
        return table

    def get_action_prompt(self) -> str:
        """
        Generate the prompt that includes a concise context summary, the current table,
        and clear instructions for the AI.
        
        IMPORTANT:
          - You must choose either to reveal cells or select a basket.
          - If revealing, list cells in the format "basket prize" (e.g., "1 a", "2 b").
          - If choosing a basket, supply the basket number.
          - Alternatively, you can 'accept' the recommended basket.
          - Note: The recommended basket is computed by assuming all prizes are equal (by total count)
                  and represents a good default option. However, the actual best basket (which maximizes payout)
                  may be different.
        
        Respond in JSON format with the following keys:
          - explanation: a free-form text explanation of your reasoning.
          - decision: one of 'accept', 'choose', or 'reveal'.
          - parameters: an array of parameters. For 'choose', include the basket number as the first element;
                        for 'reveal', include the list of cell choices (e.g., "1 a", "2 b", etc.).
        
        For example:
          {"explanation": "I want to reveal cells to gather more information.", "decision": "reveal", "parameters": ["1 a", "1 b", "1 c", "1 d", "1 e"]}
        
        What is your action?
        """
        table = self.build_table()
        context_summary = self.get_context_summary()
        base_text = (
            f"Current game state:\n{table}\n"
            f"Context summary: {context_summary}\n"
        )
        if self.nudge_present:
            nudge_text = (
                f"You have the option to choose the recommended basket (default nudge). "
                f"The recommended basket is Basket {self.default_option}, which is computed by assuming all prizes are equal (by total count) and represents a good default option. "
                f"Note that this heuristic may not always yield the highest payout once actual prize values are revealed.\n"
            )
        else:
            nudge_text = ""
        instructions = (
            "Respond in JSON format with the following keys:\n"
            "  - explanation: a free-form text explanation of your reasoning.\n"
            "  - decision: one of 'accept', 'choose', or 'reveal'.\n"
            "  - parameters: an array of parameters. For 'choose', include the basket number as the first element; "
            "for 'reveal', include the list of cell choices (e.g., \"1 a\", \"2 b\", etc.).\n"
            "For example:\n"
            "  {\"explanation\": \"I want to reveal cells to gather more information.\", \"decision\": \"reveal\", \"parameters\": [\"1 a\", \"1 b\", \"1 c\", \"1 d\", \"1 e\"]}\n"
            "What is your action?"
        )
        full_prompt = base_text + nudge_text + instructions
        # We send only the summary and current table (not the full previous conversation) to keep the prompt concise.
        self.dialog_history = [{"role": "user", "content": full_prompt}]
        return full_prompt

    def ask_gpt(self) -> Optional[str]:
        """
        Sends the current prompt (with summary context) to the AI and returns its response.
        """
        try:
            response = self.client.chat.completions.create(
                model=GameConfig.MODEL_NAME,
                messages=self.dialog_history,
                temperature=0.2
            )
            reply = response.choices[0].message.content.strip().lower()
            self.dialog_history.append({"role": "assistant", "content": reply})
            return reply
        except Exception as e:
            print(f"❌ Error in API call: {e}")
            return None

    def reveal_values(self, choices: List[str]) -> None:
        """
        Reveals the specified cells.
        If a cell is already revealed or the choice is invalid, it counts as a wrong move.
        """
        for choice in choices:
            try:
                basket_str, prize = choice.split()
                basket = int(basket_str)
                prize_norm = None
                for p in self.prize_labels:
                    if p.lower() == prize.lower():
                        prize_norm = p
                        break
                if not prize_norm:
                    print(f"Wrong move: Invalid prize type in choice: {choice}")
                    self.wrong_moves += 1
                    continue
                if basket not in self.basket_counts:
                    print(f"Wrong move: Invalid basket number in choice: {choice}")
                    self.wrong_moves += 1
                    continue
                if self.baskets[basket][prize_norm] != '-':
                    print(f"Wrong move: Cell already revealed: Basket {basket} Prize {prize_norm}")
                    self.wrong_moves += 1
                    continue
                revealed_value = self.basket_counts[basket][prize_norm]
                self.baskets[basket][prize_norm] = revealed_value
                self.game_data['revealed_cells'][(basket, prize_norm)] = revealed_value
                self.game_data['action_log'].append(f"revealed {basket} {prize_norm}")
                print(f"Revealed Basket {basket} Prize {prize_norm}: {revealed_value}")
            except ValueError:
                print(f"Wrong move: Invalid reveal choice format: {choice}")
                self.wrong_moves += 1

    def play(self) -> None:
        """
        Main game loop.
        If the AI produces an unclear action (unknown command), or if too many wrong moves or iterations occur,
        the game is flagged as an error and is skipped (final_choice is not stored).
        The error is logged in the stats.
        """
        self.initialize_game()
        final_choice = None
        error_occurred = False
        error_message = ""
        max_iterations = 10
        iteration = 0

        while iteration < max_iterations:
            iteration += 1
            if self.wrong_moves >= GameConfig.MAX_WRONG_MOVES:
                error_message = "Exceeded wrong move threshold"
                print("Too many wrong moves. Terminating game with error.")
                self.game_data['action_log'].append("Exceeded wrong move threshold.")
                error_occurred = True
                break

            action_prompt = self.get_action_prompt()
            print("\n" + "-" * 40)
            print(action_prompt)
            ai_decision = self.ask_gpt()
            if not ai_decision:
                error_message = "No decision received"
                print("No decision received from AI. Terminating game with error.")
                error_occurred = True
                break
            print(f"AI raw response: {ai_decision}")
            self.game_data['action_log'].append(ai_decision)

            command_type, command_value = process_ai_response(ai_decision, self.nudge_present)
            print(f"Parsed command type: {command_type}, value: {command_value}")

            if command_type == 'unknown':
                error_message = "Unknown command"
                print("Unclear action encountered. Terminating game with error.")
                error_occurred = True
                break

            if command_type == 'accept':
                final_choice = self.default_option
                print(f"AI accepted the default basket: Basket {final_choice}")
                self.game_data['action_log'].append(f"accepted default basket {final_choice}")
                break
            elif command_type == 'choose':
                final_choice = command_value
                print(f"AI chose Basket {final_choice}")
                self.game_data['action_log'].append(f"chose basket {final_choice}")
                break
            elif command_type == 'reveal':
                self.reveal_values(command_value)
            else:
                print("Unrecognized command. No action taken this turn.")

            self.game_data['wrong_moves'] = self.wrong_moves

        if error_occurred:
            self.game_data['error'] = error_message
            print(f"Game ended with error: {error_message}. Skipping game.")
            log_game_data_to_csv(self.game_data)
            return

        if final_choice is None:
            error_message = "No valid choice made - loop ended"
            print("No valid choice was made. Terminating game with error.")
            self.game_data['error'] = error_message
            log_game_data_to_csv(self.game_data)
            return

        total_reveal_cost = len(self.game_data['revealed_cells']) * GameConfig.COST_PER_REVEAL
        basket_total = sum(self.basket_counts[final_choice][prize] * self.prize_values[prize]
                           for prize in self.prize_labels)
        points_earned = basket_total - total_reveal_cost

        # Compute the best basket (actual optimal based on prize values).
        best_basket = max(
            range(1, self.num_baskets + 1),
            key=lambda b: sum(self.basket_counts[b][prize] * self.prize_values[prize] for prize in self.prize_labels)
        )

        self.game_data.update({
            'final_choice': final_choice,
            'total_reveal_cost': total_reveal_cost,
            'points_earned': points_earned,
            'best_basket': best_basket
        })

        print("\nFinal Game Outcome:")
        print(f"Final choice: Basket {final_choice}")
        print(f"Revealed cells: {self.game_data['revealed_cells']}")
        print(f"Wrong moves: {self.wrong_moves}")
        print(f"Points earned: {points_earned}")
        print(f"Recommended basket (default nudge): Basket {self.default_option}")
        print(f"Best basket (optimal): Basket {best_basket}")
        
        log_game_data_to_csv(self.game_data)

###########################
# Experiment Entry Point  #
###########################
def run_experiment():
    print("=== Running Game Simulation ===")
    # Force a fixed number of nudged rounds for better comparison.
    num_rounds = GameConfig.NUM_TEST_ROUNDS
    rounds_with_nudge = num_rounds // 2
    rounds_without_nudge = num_rounds - rounds_with_nudge
    
    total_rounds = [True] * rounds_with_nudge + [False] * rounds_without_nudge
    random.shuffle(total_rounds)
    
    for i, force_nudge in enumerate(total_rounds, 1):
        print(f"\n--- Round {i} ---")
        game = BasketGame(practice=False, force_nudge=force_nudge)
        game.play()

if __name__ == "__main__":
    run_experiment()
