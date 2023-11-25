model="edae"
seeds="0,1,2,3"

shared_params="exp_name=LinSampleEff model=${model} system=linear_system seed=${seeds} system.n_constraints=1 system.obs_state_ratio=3 system.state_dim=50"
hydra_params="hydra.launcher.n_jobs=3"

python train_observables.py --multirun device=0 system.group=C5 system.train_ratio=0.1,0.25  ${hydra_params} ${shared_params} &
python train_observables.py --multirun device=1 system.group=C5 system.train_ratio=0.05,0.5  ${hydra_params} ${shared_params} &
python train_observables.py --multirun device=2 system.group=C5 system.train_ratio=0.15,0.75 ${hydra_params} ${shared_params} &
python train_observables.py --multirun device=3 system.group=C5 system.train_ratio=1.0  ${hydra_params} ${shared_params} &
python train_observables.py --multirun device=4 system.group=C10 system.train_ratio=0.1,0.25  ${hydra_params} ${shared_params} &
python train_observables.py --multirun device=5 system.group=C10 system.train_ratio=0.05,0.5  ${hydra_params} ${shared_params} &
python train_observables.py --multirun device=6 system.group=C10 system.train_ratio=0.15,0.75 ${hydra_params} ${shared_params} &
python train_observables.py --multirun device=7 system.group=C10 system.train_ratio=1.0  ${hydra_params} ${shared_params} &
wait
