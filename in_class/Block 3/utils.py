import numpy as np
import time

urn = 12 * [0] + 17 * [1]
np.random.shuffle(urn)

second_urn = 5 * [0] + 14 * [1]
np.random.shuffle(second_urn)

third_urn = 20 * [0] + 9 * [1]
np.random.shuffle(third_urn)


def draw_from_urn(index, urn_num = 0):
    if urn_num == 0:
        return "white" if urn[index] == 0 else "orange"
    elif urn_num == 1:
        return "white" if second_urn[index] == 0 else "orange"
    elif urn_num == 2:
        return "white" if third_urn[index] == 0 else "orange"
    else:
        print("Please select a valid urn number (0-2)")
    

def pull_arm(arm):
    reward = np.random.normal(loc=[0.4, 0.5, 0.6][arm], scale=.2)
    return float(np.clip(reward, 0, 1))


def simulate_bandit(select_arm_func, steps=50, delay=0.3):
    history = {0: [], 1: [], 2: []}
    cumulative_rewards = []

    print("🎲 Bandit Simulation Started\n")

    for t in range(steps):
        choice = select_arm_func(history)
        reward = pull_arm(choice)
        history[choice].append(reward)
        cumulative_rewards.append(reward if t == 0 else cumulative_rewards[-1] + reward)

        print(f"Pull #{t+1}: Chose arm {choice}, reward = {reward:.2f}")
        print(f"  Cumulative reward: {cumulative_rewards[-1]:.2f}")
        print(f"  Current histories:")
        for arm in history:
            print(f"    Arm {arm}: {[round(r,2) for r in history[arm]]}")
        print("-" * 40)
        time.sleep(delay)

    total_reward = cumulative_rewards[-1]
    print("\n✅ Simulation Complete!")
    print(f"Total cumulative reward: {total_reward:.2f}")
    for arm in history:
        total = sum(history[arm])
        avg = total / len(history[arm]) if history[arm] else 0
        print(f"Arm {arm}: Total = {total:.2f}, Average = {avg:.2f}")