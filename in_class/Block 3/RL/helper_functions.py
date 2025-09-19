import numpy as np
import random
import matplotlib.pyplot as plt
import ipywidgets as widgets
from IPython.display import display, clear_output
import matplotlib.image as mpimg
import matplotlib.lines as lines
import ipywidgets as widgets

output = widgets.Output()

# Set seeds for reproducibility
# random.seed(20)
# np.random.seed(20)

# Load images for the FrozenLake tiles
img_frozen = mpimg.imread("./assets/frozen.png")    # frozen tile
img_hole = mpimg.imread("./assets/hole.png")        # hole tile
img_agent = mpimg.imread("./assets/agent.png")      # agent tile
img_reward = mpimg.imread("./assets/present.png")   # reward tile


directions = {0: 'L', 1: 'D', 2: 'R', 3: 'U'}
moves = {'L': (0, -1), 'D': (1, 0), 'R': (0, 1), 'U': (-1, 0)}

# ---------------------------
# Hidden Helper Functions
# ---------------------------

def plot_environment(q_table, state):
    """
    Plot the FrozenLake environment along with the Q-values and agent position.
    """
    n_rows = len(env_map)
    n_cols = len(env_map[0])

    fig, ax = plt.subplots(figsize=(8, 8))

    # Loop through each row and column of the grid
    for row in range(n_rows):
        for col in range(n_cols):
            cell_state = row * n_cols + col
            cell_type = env_map[row][col]

            # Choose image based on cell type
            if cell_type == 'F':
                img = img_frozen
            elif cell_type == 'H':
                img = img_hole
            else:
                img = img_frozen  # Default fallback

            ax.imshow(img, extent=[col, col + 1, n_rows - 1 - row, n_rows - row])

            # If the cell is a goal ('G'), overlay the reward image
            if cell_type == 'G':
                ax.imshow(img_reward, extent=[col, col + 1, n_rows - 1 - row, n_rows - row], alpha=1.0)

            # Retrieve Q-values for the current cell
            q_values = q_table[cell_state]
            max_q_value = max(q_values)

            # Only draw an arrow if exactly one action has the highest Q-value
            if (q_values == max_q_value).sum() == 1:
                best_action = np.argmax(q_values)
                arrow_length = 0.2
                center_x = col + 0.5
                center_y = n_rows - 1 - row + 0.5

                # Set dx, dy for arrow direction based on best_action.
                if best_action == 0:      # Left
                    dx, dy = -arrow_length, 0
                elif best_action == 1:    # Down
                    dx, dy = 0, -arrow_length
                elif best_action == 2:    # Right
                    dx, dy = arrow_length, 0
                elif best_action == 3:    # Up
                    dx, dy = 0, arrow_length

                # Only draw the arrow if it's a frozen or start cell
                # (Holes and goals don't need an arrow)
                if cell_type == 'F' or cell_type == 'S':
                    ax.arrow(center_x, center_y, dx, dy, head_width=0.1, head_length=0.05,
                            fc='green', ec='green', zorder=0)

            # A small helper function to determine the text color
            def get_text_color(value):
                if cell_type in ["H", "G"]:
                    return 'grey'
                return 'green' if value == max_q_value else 'black'

            # Display Q-values in each cell in their respective positions
            # Order: Up (3), Left (0), Right (2), Down (1)
            ax.text(col + 0.5, n_rows - 1 - row + 0.9, f'{q_values[3]:.2f}', ha='center', fontsize=8,
                    color=get_text_color(q_values[3]), fontweight='bold')  # Up
            ax.text(col + 0.05, n_rows - 1 - row + 0.5, f'{q_values[0]:.2f}', va='center', fontsize=8,
                    color=get_text_color(q_values[0]), fontweight='bold')  # Left
            ax.text(col + 0.78, n_rows - 1 - row + 0.5, f'{q_values[2]:.2f}', va='center', fontsize=8,
                    color=get_text_color(q_values[2]), fontweight='bold')  # Right
            ax.text(col + 0.5, n_rows - 1 - row + 0.05, f'{q_values[1]:.2f}', ha='center', fontsize=8,
                    color=get_text_color(q_values[1]), fontweight='bold')  # Down

    # Plot the agent at the current state
    row, col = divmod(state, n_cols)
    ax.imshow(img_agent, extent=[col, col + 1, n_rows - 1 - row, n_rows - row], alpha=1.0)
    
    # Draw grid lines for clarity
    for i in range(n_rows + 1):  # Draw horizontal grid lines
        ax.add_line(lines.Line2D([0, n_cols], [i, i], color="grey", linewidth=1.5))
    for i in range(n_cols + 1):  # Draw vertical grid lines
        ax.add_line(lines.Line2D([i, i], [0, n_rows], color="grey", linewidth=1.5))

    # Remove axis ticks and set the viewing area
    ax.set_xticks([])
    ax.set_yticks([])
    plt.xlim(0, n_cols)
    plt.ylim(0, n_rows)
    plt.show()

# def plot_environment(q_table, state):
#     """
#     Plot the FrozenLake environment along with the Q-values and agent position.
#     """
#     n_rows = len(env_map)
#     n_cols = len(env_map[0])

#     fig, ax = plt.subplots(figsize=(8, 8))
#     # Loop through each row and column of the 4x4 grid
#     for row in range(n_rows):
#         for col in range(n_cols):
#             cell_state = row * n_cols + col
#             cell_type = env_map[row][col]
#             # Choose image based on cell type
#             if cell_type == 'F':
#                 img = img_frozen
#             elif cell_type == 'H':
#                 img = img_hole
#             else:
#                 img = img_frozen  # Default fallback

#             ax.imshow(img, extent=[col, col + 1, 3 - row, 4 - row])

#             # If the cell is a goal ('G'), overlay the reward image
#             if cell_type == 'G':
#                 ax.imshow(img_reward, extent=[col, col + 1, 3 - row, 4 - row], alpha=1.0)

#             # Retrieve Q-values for the current cell
#             q_values = q_table[cell_state]
#             max_q_value = max(q_values)

#             # Only draw an arrow if exactly one action has the highest Q-value
#             if (q_values == max_q_value).sum() == 1:
#                 best_action = np.argmax(q_values)
#                 arrow_length = 0.2
#                 center_x = col + 0.5
#                 center_y = 3 - row + 0.5

#                 # Set dx, dy for arrow direction based on best_action.
#                 if best_action == 0:      # Left
#                     dx, dy = -arrow_length, 0
#                 elif best_action == 1:    # Down
#                     dx, dy = 0, -arrow_length
#                 elif best_action == 2:    # Right
#                     dx, dy = arrow_length, 0
#                 elif best_action == 3:    # Up
#                     dx, dy = 0, arrow_length

#                 # Only draw the arrow if it's a frozen or start cell
#                 # (Holes and goals don't need an arrow)
#                 if cell_type == 'F' or cell_type == 'S':
#                     ax.arrow(center_x, center_y, dx, dy, head_width=0.1, head_length=0.05,
#                             fc='green', ec='green', zorder=0)
                
#             # A small helper function to determine the text color
#             # - 'grey' for holes or goals
#             # - 'green' for the unique maximum Q-value
#             # - 'black' otherwise
#             def get_text_color(value):
#                 if cell_type in ["H", "G"]:
#                     return 'grey'
#                 return 'green' if value == max_q_value else 'black'

#             # Display Q-values in each cell in their respective positions
#             # Order: Up (3), Left (0), Right (2), Down (1)
#             ax.text(col + 0.5, 3 - row + 0.9, f'{q_values[3]:.2f}', ha='center', fontsize=8,
#                     color=get_text_color(q_values[3]), fontweight='bold')  # Up
#             ax.text(col + 0.05, 3 - row + 0.5, f'{q_values[0]:.2f}', va='center', fontsize=8,
#                     color=get_text_color(q_values[0]), fontweight='bold')  # Left
#             ax.text(col + 0.78, 3 - row + 0.5, f'{q_values[2]:.2f}', va='center', fontsize=8,
#                     color=get_text_color(q_values[2]), fontweight='bold')  # Right
#             ax.text(col + 0.5, 3 - row + 0.05, f'{q_values[1]:.2f}', ha='center', fontsize=8,
#                     color=get_text_color(q_values[1]), fontweight='bold')  # Down

#     # Plot the agent at the current state
#     row, col = divmod(state, n_cols)
#     ax.imshow(img_agent, extent=[col, col + 1, 3 - row, 4 - row], alpha=1.0)
    
#     # Draw grid lines for clarity
#     for i in range(n_row):
#         ax.add_line(lines.Line2D([0, n_col - 1], [i, i], color="grey", linewidth=1.5))
#     for i in range(n_col):
#         ax.add_line(lines.Line2D([i, i], [0, n_row - 1], color="grey", linewidth=1.5))

#     # Remove axis ticks and set the viewing area
#     ax.set_xticks([])
#     ax.set_yticks([])
#     plt.xlim(0, 4)
#     plt.ylim(0, 4)
#     plt.show()


def print_q_table(q_table):
    """
    Print the Q-table in a formatted way.
    """
    for row in q_table:
        print("[", end="")
        print(" ".join(f"{num:.2f}" for num in row), end="")
        print("]")


def calculate_new_row_and_col(row, col, action):
    """
    Calculate the new row and column for a given action, based on
    the moves dictionary and the current (row, col).
    """
    move = moves[directions[action]]
    new_row, new_col = row + move[0], col + move[1]
    return new_row, new_col


def get_next_state(state, action):
    """
    Given a state and an action, determine the next state, reward, and whether the episode is done.
    - Accounts for slipperiness if is_slippery is True (randomly picks left, forward, or right)
    - If the move is out of bounds, remain in place
    - Reward is 1 if the new cell is the goal ('G'), 0 otherwise
    - done is True if the new cell is a hole ('H') or goal ('G')
    """
    row, col = divmod(state, len(env_map[0]))
    if is_slippery:
        action = random.choices(
            [(action - 1) % 4, action, (action + 1) % 4],
            weights=[1/3, 1/3, 1/3]
        )[0]
    new_row, new_col = calculate_new_row_and_col(row, col, action)
    if not (0 <= new_row < len(env_map) and 0 <= new_col < len(env_map[0])):
        new_row, new_col = row, col  # Out of bounds; stay in place
    new_state = new_row * len(env_map[0]) + new_col
    reward = 1 if env_map[new_row][new_col] == 'G' else 0
    done = env_map[new_row][new_col] in ['H', 'G']
    return action, new_state, reward, done



# -------------------------------------------------------------------------
# INTERACTIVE WIDGETS
# -------------------------------------------------------------------------
# These widgets allow interactive control of the Q-learning parameters
# and the environment updates.

# Sliders for controlling various Q-learning parameters
num_episodes_slider = widgets.IntSlider(value=1000, min=100, max=5000, step=100, description='Episodes')
max_steps_slider = widgets.IntSlider(value=100, min=10, max=500, step=10, description='Max Steps')
learning_rate_slider = widgets.FloatSlider(value=0.1, min=0.01, max=1.0, step=0.01, description='Learning Rate')
discount_factor_slider = widgets.FloatSlider(value=0.9, min=0.1, max=1.0, step=0.1, description='Discount Factor')
exploration_rate_slider = widgets.FloatSlider(value=1.0, min=0.01, max=1.0, step=0.01, description='Exploration Rate')
decay_rate_slider = widgets.FloatSlider(value=0.001, min=0.0001, max=0.01, step=0.0001,
                                         description='Decay Rate', readout_format='.4f')

# Buttons for running steps, episodes, or multiple episodes
btn_one_step        = widgets.Button(description='Run 1 Step')
btn_one_episode     = widgets.Button(description='Run 1 Episode')
btn_run_n_episodes  = widgets.Button(description='Run N Episodes')

# Output widget for displaying logs or additional prints
output = widgets.Output()

def my_display_widgets():
    """
    Display all the interactive widgets (sliders, buttons, output).
    """
    display(num_episodes_slider, max_steps_slider, learning_rate_slider, discount_factor_slider,
            exploration_rate_slider, decay_rate_slider, btn_one_step, btn_one_episode, btn_run_n_episodes)

def on_one_step_clicked(b):
    """
    Callback for running a single step of Q-learning.
    - Calls q_learning_execute_one_step from the main notebook.
    - Updates the environment plot.
    """
    global state, q_table
    with output:
        # Retrieve visible functions from the notebook namespace via the imported module 'main'
        new_state, reward, done = q_learning_execute_one_step(
            q_table, state,
            exploration_rate_slider.value,
            learning_rate_slider.value,
            discount_factor_slider.value
        )
        state = new_state
        clear_output(wait=True)
        my_display_widgets()
        plot_environment(q_table, state)
        if done:
            state = 0

def on_one_episode_clicked(b):
    """
    Callback for running one entire episode of Q-learning.
    - Calls q_learning_execute_one_episode from the main notebook.
    - Updates the environment plot afterward.
    """
    global state, q_table
    with output:
        q_learning_execute_one_episode(
            q_table, state,
            max_steps_slider.value,
            exploration_rate_slider.value,
            learning_rate_slider.value,
            discount_factor_slider.value
        )
        clear_output(wait=True)
        my_display_widgets()
        plot_environment(q_table, state)

def on_run_n_episodes_clicked(b):
    """
    Callback for running N episodes of Q-learning.
    - Calls q_learning from the main notebook.
    - Updates the exploration rate slider value, environment plot.
    """
    global state, q_table
    with output:
        exploration_rate, episode_rewards = q_learning(
            q_table, state,
            num_episodes_slider.value,
            max_steps_slider.value,
            learning_rate_slider.value,
            discount_factor_slider.value,
            exploration_rate_slider.value,
            decay_rate_slider.value
        )
        exploration_rate_slider.value = exploration_rate
        clear_output(wait=True)
        my_display_widgets()
        plot_environment(q_table, state)

# Attach the callbacks to the buttons
btn_one_step.on_click(on_one_step_clicked)
btn_one_episode.on_click(on_one_episode_clicked)
btn_run_n_episodes.on_click(on_run_n_episodes_clicked)

def run_game():
    """
    Initialize and display the interactive widgets and the initial environment plot.
    """
    display(output)
    with output:
        my_display_widgets()
        plot_environment(q_table, state)