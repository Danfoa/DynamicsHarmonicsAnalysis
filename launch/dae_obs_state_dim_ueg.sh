seeds="0,1,2,3"
shared_params="exp_name=C10-ObsStateDim2 system=linear_system system.n_constraints=1 system.state_dim=30"
hydra_params="hydra.launcher.n_jobs=2"

#python train_observables.py --multirun device=0 model=dae   system.group=C5  seed=0 system.obs_state_ratio=1,2,3,4,6,8 ${hydra_params} ${shared_params} &
python train_observables.py --multirun device=0 model=dae   system.group=C5  seed=0,1 system.obs_state_ratio=6,8 ${hydra_params} ${shared_params} &
python train_observables.py --multirun device=1 model=dae   system.group=C5  seed=2,3 system.obs_state_ratio=6,8 ${hydra_params} ${shared_params} &
python train_observables.py --multirun device=2 model=edae   system.group=C5  seed=0,1 system.obs_state_ratio=6,8 ${hydra_params} ${shared_params} &
python train_observables.py --multirun device=3 model=edae   system.group=C5  seed=2,3 system.obs_state_ratio=6,8 ${hydra_params} ${shared_params} &
python train_observables.py --multirun device=4 model=dae   system.group=C10  seed=0,1 system.obs_state_ratio=6,8 ${hydra_params} ${shared_params} &
python train_observables.py --multirun device=5 model=dae   system.group=C10  seed=2,3 system.obs_state_ratio=6,8 ${hydra_params} ${shared_params} &
python train_observables.py --multirun device=6 model=edae   system.group=C10  seed=0,1 system.obs_state_ratio=6,8 ${hydra_params} ${shared_params} &
python train_observables.py --multirun device=7 model=edae   system.group=C10  seed=2,3 system.obs_state_ratio=6,8 ${hydra_params} ${shared_params} &
wait