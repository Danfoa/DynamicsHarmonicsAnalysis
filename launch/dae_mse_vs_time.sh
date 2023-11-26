seeds="0,1,2,3,4"

shared_params="exp_name=MseVsTime system=linear_system seed=${seeds} system.n_constraints=1 system.obs_state_ratio=3 system.state_dim=100"
hydra_params="hydra.launcher.n_jobs=4"

python train_observables.py --multirun device=0 system.group=C5  model=dae  system.noise_level=4 seed=0,1,2,3 ${hydra_params} ${shared_params} &
python train_observables.py --multirun device=1 system.group=C5  model=dae  system.noise_level=4 seed=4,5,6,7 ${hydra_params} ${shared_params} &
python train_observables.py --multirun device=2 system.group=C5  model=edae system.noise_level=4 seed=0,1,2,3 ${hydra_params} ${shared_params} &
python train_observables.py --multirun device=3 system.group=C5  model=edae system.noise_level=4 seed=4,5,6,7 ${hydra_params} ${shared_params} &
python train_observables.py --multirun device=4 system.group=C10 model=dae  system.noise_level=4 seed=0,1,2,3 ${hydra_params} ${shared_params} &
python train_observables.py --multirun device=5 system.group=C10 model=dae  system.noise_level=4 seed=4,5,6,7 ${hydra_params} ${shared_params} &
python train_observables.py --multirun device=6 system.group=C10 model=edae system.noise_level=4 seed=0,1,2,3 ${hydra_params} ${shared_params} &
python train_observables.py --multirun device=7 system.group=C10 model=edae system.noise_level=4 seed=4,5,6,7 ${hydra_params} ${shared_params} &
wait