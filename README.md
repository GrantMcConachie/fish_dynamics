# Fish Dynamics

A graph neural network to predict fish behavior.

# Setup

On the SCC

```
git clone https://github.com/GrantMcConachie/fish_dynamics.git
cd fish_dynamics/
module load python3/3.12.4
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

# Running a training script
Ask me for the data and I can send it to you.
```
python scripts/train.py
```
The train script should save the fully trained model into `results/saved_models` with a time stamp. The configuration of the model can be found in `model/config.json' if you want to tweak any of the parameters.

# Running the evaluation script
Once a model has completed training
```
python scripts/evaluate.py
```
This will plot rollout predictions of the model as well as validation and test loss values.
