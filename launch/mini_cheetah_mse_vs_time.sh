
seeds="1,2,3,4"
exp_name="mse_vs_time"
shared_params="exp_name=${exp_name} system.terrain=uneven_easy system.max_epochs=400 model.obs_pred_w=1.0 model.num_hidden_units=256 seed=${seeds} model.activation=ReLU model.batch_size=512"
hydra_params="hydra.launcher.n_jobs=4"

#python train_observables.py --multirun  model=dae-aug exp_name=performance hydra.launcher.n_jobs=2 device=1 seed=0,1

python train_observables.py --multirun device=0 system=mini_cheetah model=dae        system.obs_state_ratio=3 ${hydra_params} ${shared_params} &
python train_observables.py --multirun device=1 system=mini_cheetah model=dae-aug    system.obs_state_ratio=3 ${hydra_params} ${shared_params} &
python train_observables.py --multirun device=2 system=mini_cheetah-c2 model=dae-aug system.obs_state_ratio=3 ${hydra_params} ${shared_params} &
python train_observables.py --multirun device=3 system=mini_cheetah-k4 model=dae-aug system.obs_state_ratio=3 ${hydra_params} ${shared_params} &
python train_observables.py --multirun device=4 system=mini_cheetah model=edae       system.obs_state_ratio=3 ${hydra_params} ${shared_params} &
python train_observables.py --multirun device=5 system=mini_cheetah-c2 model=edae    system.obs_state_ratio=3 ${hydra_params} ${shared_params} &
python train_observables.py --multirun device=6 system=mini_cheetah-k4 model=edae    system.obs_state_ratio=3 ${hydra_params} ${shared_params} &

wait