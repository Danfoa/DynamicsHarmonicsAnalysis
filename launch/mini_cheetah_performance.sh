
shared_params="exp_name=performance system.pred_horizon=30 model.activation=ReLU model.batch_size=512"
hydra_params="hydra.launcher.n_jobs=1"

#python train_observables.py --multirun  model=dae-aug exp_name=performance hydra.launcher.n_jobs=2 device=1 seed=0,1

python train_observables.py --multirun device=0 system=mini_cheetah model=dae        seed=10 ${hydra_params} ${shared_params} &
python train_observables.py --multirun device=1 system=mini_cheetah model=dae-aug    seed=10 ${hydra_params} ${shared_params} &
python train_observables.py --multirun device=2 system=mini_cheetah-c2 model=dae-aug seed=10 ${hydra_params} ${shared_params} &
python train_observables.py --multirun device=3 system=mini_cheetah-k4 model=dae-aug seed=10 ${hydra_params} ${shared_params} &
python train_observables.py --multirun device=4 system=mini_cheetah model=dae        seed=11 ${hydra_params} ${shared_params} &
python train_observables.py --multirun device=5 system=mini_cheetah model=dae-aug    seed=11 ${hydra_params} ${shared_params} &
python train_observables.py --multirun device=6 system=mini_cheetah-c2 model=dae-aug seed=11 ${hydra_params} ${shared_params} &
python train_observables.py --multirun device=7 system=mini_cheetah-k4 model=dae-aug seed=11 ${hydra_params} ${shared_params} &

wait