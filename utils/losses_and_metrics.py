import torch


def observation_dynamics_error(z: torch.Tensor, z_pred: torch.Tensor):
    """
    Computes the average observation dynamics error between the true observations predicted by an embedding function and
     the predicted observations.
    :param z: torch.Tensor of shape (batch_size, time, dim(z)) holding the true embeddings of a trajectory of states in
        time.
    :param z_pred: torch.Tensor of same shape as `z` containing predictions of the evolution of `z`
    :return: L2 error of observation dynamics: || z - z_pred ||_2
    """
    assert z.shape == z_pred.shape, f"True obs shape {z.shape} differs from predictions shape {z_pred.shape}"
    n_frames = z.shape[1]
    z_mod, z_pred_mod = torch.abs(z), torch.abs(z_pred)

    obs_inner_prod = z.conj() * z_pred
    cos_similarity = torch.real(obs_inner_prod) / (z_mod * z_pred_mod)
    modulus_error = torch.abs(z_mod - z_pred_mod)

    norm_z_err = torch.norm(z - z_pred, dim=-1, p=2)

    avg_obs_pred_err = torch.mean(norm_z_err, dim=1)
    avg_obs_cos_sim = torch.mean(cos_similarity, dim=1)
    avg_obs_mod_err = torch.mean(modulus_error, dim=1)

    avg_single_step_error = torch.mean(norm_z_err[0], dim=0)
    metrics = {'obs_pred_err': avg_obs_pred_err.mean(),
               'obs_cos_sim': avg_obs_cos_sim.mean(),
               'obs_mod_err': avg_obs_mod_err.mean(),
               'obs_single_step_err': avg_single_step_error.mean()}
    window = 5
    for i in range(5, min(n_frames + 1, 21), window):
        avg_error = torch.mean(norm_z_err[i - window:i], dim=0)
        metrics[f'obs_{i - window:d}-{i:d}_err'] = avg_error.mean()
    return metrics