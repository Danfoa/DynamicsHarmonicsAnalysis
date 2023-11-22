
model="dae"
seeds="6,10"

shared_params="exp_name=mini_hp model=${model} system=mini_cheetah model.obs_pred_w=1.0 model.num_hidden_units=256 seed=${seeds}"
hydra_params="hydra.launcher.n_jobs=4"

echo "train_observables.py --multirun device=0 system.pred_horizon=10 system.eval_pred_horizon=20 system.obs_state_ratio=2,3 model.orth_w=0.0,0.1 ${shared_params} ${hydra_params} "
echo "train_observables.py --multirun device=1 system.pred_horizon=25 system.eval_pred_horizon=50 system.obs_state_ratio=2,3 model.orth_w=0.0,0.1 ${shared_params} ${hydra_params} "
echo "train_observables.py --multirun device=2 system.pred_horizon=50 system.eval_pred_horizon=100 system.obs_state_ratio=2,3 model.orth_w=0.0,0.1 ${shared_params} ${hydra_params} "
echo "train_observables.py --multirun device=4 system.pred_horizon=75 system.eval_pred_horizon=100 system.obs_state_ratio=2,3 model.orth_w=0.0,0.1 ${shared_params} ${hydra_params} "

