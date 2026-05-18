import stable_retro
from .agents.ppo_agent import PPO_Agent
from . import config as cfg
import numpy as np
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

GAME_NAME = "SuperMarioKart-Snes"
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

def plot_and_save(plot_steps, avg_returns, avg_lengths, out_dir="plots"):
    """Save a two-panel training curve PNG every time it's called."""
    os.makedirs(out_dir, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

    ax1.plot(plot_steps, avg_returns, color="steelblue", linewidth=1.5)
    ax1.axhline(0, color="gray", linewidth=0.8, linestyle="--")
    if len(avg_returns) > 0:
        y_min, y_max = min(avg_returns), max(avg_returns)
        margin = max((y_max - y_min) * 0.05, 1.0)
        ax1.set_ylim(y_min - margin, y_max + margin)
    
    ax1.set_ylabel("Avg Return")
    ax1.set_title("PPO Training Curve — Super Mario Kart")
    ax1.grid(True, alpha=0.3)

    ax2.plot(plot_steps, avg_lengths, color="darkorange", linewidth=1.5)
    if len(avg_lengths) > 0:
        y_min, y_max = min(avg_lengths), max(avg_lengths)
        margin = max((y_max - y_min) * 0.05, 1.0)
        ax2.set_ylim(y_min - margin, y_max + margin)
        
    ax2.set_xlabel("Global Step")
    ax2.set_ylabel("Avg Episode Length")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, "ppo_training_curve.png"), dpi=120)
    plt.close(fig)

def main():
    checkpoint_prefix = "models/ppo_discovery_ckpt" + f"_{cfg.state}"
    
    env = stable_retro.make(
        game=GAME_NAME,
        state=cfg.state,
        scenario=cfg.scenario if hasattr(cfg, "scenario") else 'scenario',
        render_mode=cfg.render_mode,
        inttype=stable_retro.data.Integrations.ALL
    )

    total_timesteps = 10_000_000
    rollout_steps = 8192

    agent = PPO_Agent(
        env,
        learning_rate=5e-5,
        rollout_steps=rollout_steps,
        minibatch_size=1024,
        n_epochs=2,
        ent_coef_start=0.03,
        ent_coef_end=0.01,
        gae_lambda=0.95,
        clip_coef=0.1,
        max_grad_norm=0.5,
        total_timesteps=total_timesteps,
        no_improve_tolerance=100,
    )
    
    # We load based on global updates instead of episodes now
    # Using 'latest' or a specific update number if you want to resume
    start_update = 0
    if start_update > 0:
        start_update = agent.load_checkpoint(checkpoint_prefix + "_" + str(start_update))
    
    env = agent.wrap_env(env)

    # Tracking metrics
    episode_returns = []
    episode_lengths = []
    episode_finished = []
    plot_steps = []
    plot_avg_returns = []
    plot_avg_lengths = []

    global_step = agent.steps
    num_updates = (total_timesteps // rollout_steps)

    state, info = env.reset()
    episode_return = 0
    episode_length = 0

    print(f"Starting PPO training from Update {start_update} to {num_updates} ({total_timesteps} total steps)...")
    
    for update in range(start_update + 1, num_updates + 1):
        # Collect exactly `rollout_steps` for this update
        for step in range(rollout_steps):
            action = agent.action_select(state)
            next_state, reward, terminated, truncated, info = env.step(action)
            
            episode_return += reward
            episode_length += 1
            global_step += 1

            # The agent.update will automatically trigger the PPO network update 
            # when its internal buffer hits `rollout_steps` (at the end of this inner loop)
            agent.update(state, action, reward, next_state, terminated or truncated)

            if terminated or truncated:
                lap = info.get("lap", 128) - 128
                is_finished = lap >= 5
                
                state, info = env.reset()
                episode_returns.append(episode_return)
                episode_lengths.append(episode_length)
                episode_finished.append(is_finished)
                episode_return = 0
                episode_length = 0
            else:
                state = next_state

        # --- Logging after each PPO Update ---
        print(f"Update {update}/{num_updates} completed. Total steps: {global_step}/{total_timesteps}")
        
        if len(episode_returns) > 0:
            # Average over the last 10 episodes
            avg_return = np.mean(episode_returns[-10:])
            avg_length = np.mean(episode_lengths[-10:])
            
            recent_finished = episode_finished[-10:]
            finish_pct = np.mean(recent_finished) * 100 if len(recent_finished) > 0 else 0.0
            
            recent_lengths = episode_lengths[-10:]
            finish_times = [length for length, fin in zip(recent_lengths, recent_finished) if fin]
            avg_finish_time = np.mean(finish_times) if len(finish_times) > 0 else 0.0
            
            progress = min(agent.steps / agent.total_timesteps, 1.0)
            current_ent_coef = agent.ent_coef_start + progress * (agent.ent_coef_end - agent.ent_coef_start)

            print(f"    Avg Return (last 10 eps): {avg_return:.2f}")
            print(f"    Avg Length (last 10 eps): {avg_length:.2f}")
            print(f"    Finish Pct (last 10 eps): {finish_pct:.1f}%")
            if len(finish_times) > 0:
                print(f"    Avg Finish Time (last 10 eps): {avg_finish_time:.2f} steps")
            else:
                print(f"    Avg Finish Time (last 10 eps): N/A")
            print(f"    Entropy: {current_ent_coef:.4f}")

            # Record and redraw the training curve
            plot_steps.append(global_step)
            plot_avg_returns.append(avg_return)
            plot_avg_lengths.append(avg_length)
            plot_and_save(plot_steps, plot_avg_returns, plot_avg_lengths)

            # Early stopping check
            agent.record_return(avg_return)
            if agent.should_stop:
                print("Stopping training early.")
                break
        else:
            print("    No episodes completed yet.")

        # Save checkpoints periodically (e.g., every 50 updates ~ 100k steps)
        if update % 50 == 0:
            print(f"Saving checkpoint at update {update}...")
            agent.save_checkpoint(checkpoint_prefix + f"_{update}", update)

    # Save final checkpoint after training completes
    print("Training complete. Saving final checkpoint...")
    agent.save_checkpoint(checkpoint_prefix + f"_final", num_updates)

if __name__ == "__main__":
    custom_path = os.path.abspath("custom_integrations")
    stable_retro.data.Integrations.add_custom_path(custom_path)

    main()
