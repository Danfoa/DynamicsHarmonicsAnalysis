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

    z_err = torch.abs(torch.sub(z, z_pred))  # ∀ t: |z_t - K^t•ø(x_0)|

    norm_z_err = torch.norm(z_err, dim=-1, p=2)

    avg_obs_pred_err = torch.mean(norm_z_err[:, 1:], dim=1)

    return avg_obs_pred_err.mean()