

seeds="0,1,2,3"
shared_params="exp_name=NoiseSensitivity system=linear_system seed=${seeds} system.n_constraints=1 system.obs_state_ratio=3 system.state_dim=70"
hydra_params="hydra.launcher.n_jobs=4"

python train_observables.py --multirun device=0 system.group=C5  model=dae  system.noise_level=0,2,4 ${hydra_params} ${shared_params} &
python train_observables.py --multirun device=1 system.group=C5  model=dae  system.noise_level=6,8   ${hydra_params} ${shared_params} &
python train_observables.py --multirun device=2 system.group=C5  model=edae system.noise_level=0,2,4 ${hydra_params} ${shared_params} &
python train_observables.py --multirun device=3 system.group=C5  model=edae system.noise_level=6,8   ${hydra_params} ${shared_params} &
python train_observables.py --multirun device=4 system.group=C10 model=dae  system.noise_level=0,2,4 ${hydra_params} ${shared_params} &
python train_observables.py --multirun device=5 system.group=C10 model=dae  system.noise_level=6,8   ${hydra_params} ${shared_params} &
python train_observables.py --multirun device=6 system.group=C10 model=edae system.noise_level=0,2,4 ${hydra_params} ${shared_params} &
python train_observables.py --multirun device=7 system.group=C10 model=edae system.noise_level=6,8   ${hydra_params} ${shared_params} &
wait