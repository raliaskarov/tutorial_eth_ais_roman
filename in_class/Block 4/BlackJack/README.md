# SARSA Reinforcement Learning Notebooks

This folder contains two Jupyter notebooks demonstrating tabular SARSA:

1. **Minimal_Sarsa_Blackjack.ipynb** - Beginner-friendly introduction to SARSA with a simplified Blackjack game
2. **Minimal_Sarsa_CarRacing.ipynb** - SARSA applied to Gymnasium's CarRacing environment

## Setup Instructions

### 1. Create the Conda Environment

From this directory, run:

```bash
conda env create -f environment.yml
```

This creates a conda environment named `sarsa-rl` with all required dependencies.

### 2. Activate the Environment

```bash
conda activate sarsa-rl
```

### 3. Launch Jupyter

```bash
jupyter notebook
```

This will open Jupyter in your browser. Navigate to the notebook you want to run.

## Running the Notebooks

### Minimal_Sarsa_Blackjack.ipynb

- **No external dependencies** needed beyond Python's `random` module
- Run cells top to bottom
- Complete the two TODO functions (or reveal instructor solution)
- See the learned policy printed at the end

### Minimal_Sarsa_CarRacing.ipynb

- **Requires**: gymnasium[box2d], imageio, imageio-ffmpeg (all included in environment.yml)
- **Training time**: ~5-10 minutes for 100 episodes
- **Output**: Video file showing the learned policy
- For faster testing, reduce `episodes=100` to `episodes=10` in the training cell

## Troubleshooting

### Box2D Installation Issues

If you encounter issues with Box2D (CarRacing dependency), try:

```bash
conda install -c conda-forge swig
pip install gymnasium[box2d]
```

### Video Not Displaying

If the video doesn't display in the notebook:
- Check that `carracing_sarsa.mp4` was created in the same directory
- Open the MP4 file directly with a video player
- Try running in JupyterLab instead: `jupyter lab`

### macOS-specific

If on macOS and you get display issues, you may need:

```bash
conda install -c conda-forge python.app
```

## Deactivating the Environment

When you're done:

```bash
conda deactivate
```

## Removing the Environment

To completely remove the environment:

```bash
conda env remove -n sarsa-rl
```
