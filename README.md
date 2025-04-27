# Analyzing Latent Representations in Texas Holdem

## Data Collection

This project uses poker hand data from the following sources:

- [CardFight](https://cardfight.com/) - Provides Equity data
- [FlopturnRiver](https://flopturnriver.com/poker-strategy/) - Provides hand EV data
- https://archive.ics.uci.edu/dataset/158/poker+hand
The data collection scripts can be found in `data_collection/`. Collected data is stored in the `data/raw` directory.

Note: Running the data collection scripts will generate these files locally.


## Environment Setup

Uses Conda for environment management and a Python package defined in `pyproject.toml`.

### Setup Steps

1. Clone the repository
   ```bash
   git clone https://github.com/joncampbell29/poker_embeddings.git
   cd poker_embeddings
   ```

2. Create and activate the Conda environment
   ```bash
   conda env create -f environment.yml
   conda activate poker_env
   ```

3. Install the project package
   ```bash
   pip install -e .
   ```

   This installs the package in development mode, allowing changes to the source code to take effect without reinstallation.


1. poetry install
2. poetry run pip install https://download.pytorch.org/whl/cu126/torch-2.6.0%2Bcu126-cp312-cp312-manylinux_2_28_x86_64.whl
3. poetry run pip install https://download.pytorch.org/whl/cu126/torchaudio-2.6.0%2Bcu126-cp312-cp312-linux_x86_64.whl
4. poetry run pip install https://download.pytorch.org/whl/cu126/torchvision-0.21.0%2Bcu126-cp312-cp312-linux_x86_64.whl


