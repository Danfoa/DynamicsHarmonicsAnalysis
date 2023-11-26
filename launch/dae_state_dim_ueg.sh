seeds="0,1,2,3"
shared_params="exp_name=C10-StateDim2 system=linear_system system.n_constraints=1 system.obs_state_ratio=3"
hydra_params="hydra.launcher.n_jobs=2"

#python train_observables.py --multirun device=0 model=dae  system.group=C10 system.state_dim=70  ${hydra_params} ${shared_params} &
#python train_observables.py --multirun device=1 model=dae  system.group=C10 system.state_dim=100 ${hydra_params} ${shared_params} &
#python train_observables.py --multirun device=2 model=edae system.group=C10 system.state_dim=70 ${hydra_params} ${shared_params} &
#python train_observables.py --multirun device=3 model=edae system.group=C10 system.state_dim=100 ${hydra_params} ${shared_params} &
python train_observables.py --multirun device=0 model=dae   system.group=C5 system.state_dim=100 seed=0 ${hydra_params} ${shared_params} &
python train_observables.py --multirun device=1 model=dae   system.group=C5 system.state_dim=100 seed=1 ${hydra_params} ${shared_params} &
python train_observables.py --multirun device=2 model=dae   system.group=C5 system.state_dim=100 seed=2 ${hydra_params} ${shared_params} &
python train_observables.py --multirun device=3 model=dae   system.group=C5 system.state_dim=100 seed=3 ${hydra_params} ${shared_params} &
python train_observables.py --multirun device=4 model=edae  system.group=C5 system.state_dim=100 seed=0 ${hydra_params} ${shared_params} &
python train_observables.py --multirun device=5 model=edae  system.group=C5 system.state_dim=100 seed=1 ${hydra_params} ${shared_params} &
python train_observables.py --multirun device=6 model=edae  system.group=C5 system.state_dim=100 seed=2 ${hydra_params} ${shared_params} &
python train_observables.py --multirun device=7 model=edae  system.group=C5 system.state_dim=100 seed=3 ${hydra_params} ${shared_params} &
wait