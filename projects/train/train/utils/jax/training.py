import equinox as eqx
import jax
import jax.numpy as jnp
import optax

from jaxtyping import Array, PRNGKeyArray


@eqx.filter_vmap(
    in_axes=(None, 0, None, 0), out_axes=(0, None), axis_name="batch"
)
def jax_fwd_batch(
    model: eqx.Module,
    X_time: Array,
    state: eqx.nn.State,
    key: PRNGKeyArray,
) -> tuple[Array, eqx.nn.State]:
    return model(X_time, state, key=key)


def _bce_loss(logits: Array, y: Array) -> Array:
    """Binary cross entropy from combined outputs."""
    return jnp.mean(
        optax.sigmoid_binary_cross_entropy(logits, y.astype(logits.dtype))
    )


def jax_loss_fn(
    diff_model: eqx.Module,
    static_model: eqx.Module,
    state: eqx.nn.State,
    X_time: Array,
    y: Array,
    key: PRNGKeyArray,
) -> tuple[Array, tuple[eqx.nn.State, Array]]:
    model = eqx.combine(diff_model, static_model)
    logits, new_state = jax_fwd_batch(model, X_time, state, key)
    y = y.reshape(-1, 1)
    loss = _bce_loss(logits, y)
    return loss, (new_state, loss)


@eqx.filter_jit
def jax_apply_training_step(
    model: eqx.Module,
    model_filter_spec,
    state: eqx.nn.State,
    X_time: Array,
    y: Array,
    opt_state,
    opt_update,
    key: PRNGKeyArray,
) -> tuple[eqx.Module, eqx.nn.State, object, dict]:
    diff_model, static_model = eqx.partition(model, model_filter_spec)

    (loss, (new_state, _)), grads = eqx.filter_value_and_grad(
        jax_loss_fn, has_aux=True
    )(diff_model, static_model, state, X_time, y, key)

    updates, new_opt_state = opt_update(grads, opt_state, diff_model)
    new_model = eqx.combine(
        eqx.apply_updates(diff_model, updates), static_model
    )
    return new_model, new_state, new_opt_state, {"loss": loss}


def jax_snr_weighted_loss_fn(
    diff_model: eqx.Module,
    static_model: eqx.Module,
    state: eqx.nn.State,
    X_time: Array,
    y: Array,
    snr_weights: Array,
    fp_weight: float,
    snr_weight_power: float,
    key: PRNGKeyArray,
) -> tuple[Array, tuple[eqx.nn.State, Array]]:
    """BCE weighted by SNR for signals and ``fp_weight`` for background.

    Per-sample weights:
        - signal (y == 1): ``snr ^ snr_weight_power``
        - background (y == 0): ``fp_weight``

    Weights are normalised by their batch mean so the loss magnitude
    stays comparable to unweighted BCE regardless of hyperparameters.
    """
    model = eqx.combine(diff_model, static_model)
    logits, new_state = jax_fwd_batch(model, X_time, state, key)
    y_flat = y.reshape(-1, 1)
    per_sample = optax.sigmoid_binary_cross_entropy(
        logits, y_flat.astype(logits.dtype)
    ).squeeze(-1)  # (N,)

    fg_mask = y.reshape(-1).astype(jnp.bool_)
    raw_w = jnp.where(
        fg_mask,
        snr_weights**snr_weight_power,
        fp_weight,
    )
    w = raw_w / jnp.mean(raw_w)
    loss = jnp.mean(w * per_sample)
    return loss, (new_state, loss)


@eqx.filter_jit
def jax_apply_snr_weighted_training_step(
    model: eqx.Module,
    model_filter_spec,
    state: eqx.nn.State,
    X_time: Array,
    y: Array,
    snr_weights: Array,
    fp_weight: float,
    snr_weight_power: float,
    opt_state,
    opt_update,
    key: PRNGKeyArray,
) -> tuple[eqx.Module, eqx.nn.State, object, dict]:
    diff_model, static_model = eqx.partition(model, model_filter_spec)

    (loss, (new_state, _)), grads = eqx.filter_value_and_grad(
        jax_snr_weighted_loss_fn, has_aux=True
    )(
        diff_model,
        static_model,
        state,
        X_time,
        y,
        snr_weights,
        fp_weight,
        snr_weight_power,
        key,
    )

    updates, new_opt_state = opt_update(grads, opt_state, diff_model)
    new_model = eqx.combine(
        eqx.apply_updates(diff_model, updates), static_model
    )
    return new_model, new_state, new_opt_state, {"loss": loss}


def jax_focal_loss_fn(
    diff_model: eqx.Module,
    static_model: eqx.Module,
    state: eqx.nn.State,
    X_time: Array,
    y: Array,
    gamma: float,
    key: PRNGKeyArray,
) -> tuple[Array, tuple[eqx.nn.State, Array]]:
    """Focal loss: down-weights easy examples via (1 - p_t)^gamma factor."""
    model = eqx.combine(diff_model, static_model)
    logits, new_state = jax_fwd_batch(model, X_time, state, key)
    y_flat = y.reshape(-1, 1)
    p = jax.nn.sigmoid(logits)
    p_t = y_flat * p + (1.0 - y_flat) * (1.0 - p)
    bce = optax.sigmoid_binary_cross_entropy(
        logits, y_flat.astype(logits.dtype)
    )
    loss = jnp.mean((1.0 - p_t) ** gamma * bce)
    return loss, (new_state, loss)


@eqx.filter_jit
def jax_apply_focal_training_step(
    model: eqx.Module,
    model_filter_spec,
    state: eqx.nn.State,
    X_time: Array,
    y: Array,
    gamma: float,
    opt_state,
    opt_update,
    key: PRNGKeyArray,
) -> tuple[eqx.Module, eqx.nn.State, object, dict]:
    diff_model, static_model = eqx.partition(model, model_filter_spec)
    (loss, (new_state, _)), grads = eqx.filter_value_and_grad(
        jax_focal_loss_fn, has_aux=True
    )(diff_model, static_model, state, X_time, y, gamma, key)
    updates, new_opt_state = opt_update(grads, opt_state, diff_model)
    new_model = eqx.combine(
        eqx.apply_updates(diff_model, updates), static_model
    )
    return new_model, new_state, new_opt_state, {"loss": loss}


def jax_hnm_loss_fn(
    diff_model: eqx.Module,
    static_model: eqx.Module,
    state: eqx.nn.State,
    X_time: Array,
    y: Array,
    n_hard_negs: int,
    key: PRNGKeyArray,
) -> tuple[Array, tuple[eqx.nn.State, Array]]:
    """BCE over all positives + top-n_hard_negs hardest negatives per batch."""
    model = eqx.combine(diff_model, static_model)
    logits, new_state = jax_fwd_batch(model, X_time, state, key)
    y_flat = y.reshape(-1)
    logits_flat = logits.reshape(-1)
    per_sample = optax.sigmoid_binary_cross_entropy(
        logits_flat, y_flat.astype(logits_flat.dtype)
    )
    pos_mask = y_flat > 0.5
    neg_mask = ~pos_mask
    # Rank negatives by score descending; positives get -inf and rank last.
    masked_neg_logits = jnp.where(neg_mask, logits_flat, -jnp.inf)
    ranks = jnp.argsort(jnp.argsort(-masked_neg_logits))
    hard_neg_mask = neg_mask & (ranks < n_hard_negs)
    active = pos_mask | hard_neg_mask
    loss = jnp.sum(per_sample * active) / jnp.maximum(1.0, jnp.sum(active))
    return loss, (new_state, loss)


@eqx.filter_jit
def jax_apply_hnm_training_step(
    model: eqx.Module,
    model_filter_spec,
    state: eqx.nn.State,
    X_time: Array,
    y: Array,
    n_hard_negs: int,
    opt_state,
    opt_update,
    key: PRNGKeyArray,
) -> tuple[eqx.Module, eqx.nn.State, object, dict]:
    diff_model, static_model = eqx.partition(model, model_filter_spec)
    (loss, (new_state, _)), grads = eqx.filter_value_and_grad(
        jax_hnm_loss_fn, has_aux=True
    )(diff_model, static_model, state, X_time, y, n_hard_negs, key)
    updates, new_opt_state = opt_update(grads, opt_state, diff_model)
    new_model = eqx.combine(
        eqx.apply_updates(diff_model, updates), static_model
    )
    return new_model, new_state, new_opt_state, {"loss": loss}


def jax_pauc_loss_fn(
    diff_model: eqx.Module,
    static_model: eqx.Module,
    state: eqx.nn.State,
    X_time: Array,
    y: Array,
    n_hard_negs: int,
    margin: float,
    key: PRNGKeyArray,
) -> tuple[Array, tuple[eqx.nn.State, Array]]:
    """Squared-hinge pairwise loss over (positive, top-n hard-negative) pairs.

    Directly optimises a surrogate for partial AUC at low FPR. For each
    positive sample, the loss penalises any hard negative that scores within
    ``margin`` of it.
    """
    model = eqx.combine(diff_model, static_model)
    logits, new_state = jax_fwd_batch(model, X_time, state, key)
    y_flat = y.reshape(-1)
    logits_flat = logits.reshape(-1)
    pos_indicator = y_flat > 0.5
    neg_indicator = ~pos_indicator
    masked_neg_logits = jnp.where(neg_indicator, logits_flat, -jnp.inf)
    ranks = jnp.argsort(jnp.argsort(-masked_neg_logits))
    hard_neg = neg_indicator & (ranks < n_hard_negs)
    # Pairwise (pos_i, hard_neg_j) score differences — shape (N, N)
    score_diff = logits_flat[:, None] - logits_flat[None, :]
    valid_pairs = pos_indicator[:, None] & hard_neg[None, :]
    pair_loss = jnp.maximum(0.0, margin - score_diff) ** 2
    loss = jnp.sum(pair_loss * valid_pairs) / jnp.maximum(
        1.0, jnp.sum(valid_pairs)
    )
    return loss, (new_state, loss)


@eqx.filter_jit
def jax_apply_pauc_training_step(
    model: eqx.Module,
    model_filter_spec,
    state: eqx.nn.State,
    X_time: Array,
    y: Array,
    n_hard_negs: int,
    margin: float,
    opt_state,
    opt_update,
    key: PRNGKeyArray,
) -> tuple[eqx.Module, eqx.nn.State, object, dict]:
    diff_model, static_model = eqx.partition(model, model_filter_spec)
    (loss, (new_state, _)), grads = eqx.filter_value_and_grad(
        jax_pauc_loss_fn, has_aux=True
    )(diff_model, static_model, state, X_time, y, n_hard_negs, margin, key)
    updates, new_opt_state = opt_update(grads, opt_state, diff_model)
    new_model = eqx.combine(
        eqx.apply_updates(diff_model, updates), static_model
    )
    return new_model, new_state, new_opt_state, {"loss": loss}


def jax_focal_hnm_loss_fn(
    diff_model: eqx.Module,
    static_model: eqx.Module,
    state: eqx.nn.State,
    X_time: Array,
    y: Array,
    gamma: float,
    n_hard_negs: int,
    key: PRNGKeyArray,
) -> tuple[Array, tuple[eqx.nn.State, Array]]:
    """Focal loss on all positives + top-n_hard_negs hard negatives."""
    model = eqx.combine(diff_model, static_model)
    logits, new_state = jax_fwd_batch(model, X_time, state, key)
    y_flat = y.reshape(-1)
    logits_flat = logits.reshape(-1)
    p = jax.nn.sigmoid(logits_flat)
    p_t = y_flat * p + (1.0 - y_flat) * (1.0 - p)
    focal_weight = (1.0 - p_t) ** gamma
    per_sample = focal_weight * optax.sigmoid_binary_cross_entropy(
        logits_flat, y_flat.astype(logits_flat.dtype)
    )
    pos_mask = y_flat > 0.5
    neg_mask = ~pos_mask
    masked_neg_logits = jnp.where(neg_mask, logits_flat, -jnp.inf)
    ranks = jnp.argsort(jnp.argsort(-masked_neg_logits))
    hard_neg_mask = neg_mask & (ranks < n_hard_negs)
    active = pos_mask | hard_neg_mask
    loss = jnp.sum(per_sample * active) / jnp.maximum(1.0, jnp.sum(active))
    return loss, (new_state, loss)


@eqx.filter_jit
def jax_apply_focal_hnm_training_step(
    model: eqx.Module,
    model_filter_spec,
    state: eqx.nn.State,
    X_time: Array,
    y: Array,
    gamma: float,
    n_hard_negs: int,
    opt_state,
    opt_update,
    key: PRNGKeyArray,
) -> tuple[eqx.Module, eqx.nn.State, object, dict]:
    diff_model, static_model = eqx.partition(model, model_filter_spec)
    (loss, (new_state, _)), grads = eqx.filter_value_and_grad(
        jax_focal_hnm_loss_fn, has_aux=True
    )(diff_model, static_model, state, X_time, y, gamma, n_hard_negs, key)
    updates, new_opt_state = opt_update(grads, opt_state, diff_model)
    new_model = eqx.combine(
        eqx.apply_updates(diff_model, updates), static_model
    )
    return new_model, new_state, new_opt_state, {"loss": loss}


def _beta_nll_loss(
    mean: Array, target: Array, var: Array, beta: float
) -> Array:
    """Gaussian NLL with stop-gradient variance weighting (BetaNLL)."""
    nll = 0.5 * (jnp.log(var) + (target - mean) ** 2 / var)
    return jnp.mean(jax.lax.stop_gradient(var) ** beta * nll)


def jax_regression_loss_fn(
    diff_model: eqx.Module,
    static_model: eqx.Module,
    state: eqx.nn.State,
    X: Array,
    chirp_mass: Array,
    beta: float,
    lambda_spread: float,
    key: PRNGKeyArray,
) -> tuple[Array, tuple]:
    model = eqx.combine(diff_model, static_model)
    outputs, new_state = jax_fwd_batch(model, X, state, key)  # (B, d_output)
    n_vars = outputs.shape[-1] // 2
    mean = outputs[:, :n_vars]
    var = jax.nn.softplus(outputs[:, n_vars:])
    nll = _beta_nll_loss(mean, chirp_mass, var, beta)
    spread = jnp.mean(
        jax.nn.softplus(jnp.var(chirp_mass, axis=0) - jnp.var(mean, axis=0))
    )
    loss = nll + lambda_spread * spread
    return loss, (new_state, nll, spread, mean, var)


@eqx.filter_jit
def jax_apply_regression_training_step(
    model: eqx.Module,
    model_filter_spec,
    state: eqx.nn.State,
    X: Array,
    chirp_mass: Array,
    beta: float,
    lambda_spread: float,
    opt_state,
    opt_update,
    key: PRNGKeyArray,
) -> tuple[eqx.Module, eqx.nn.State, object, dict]:
    diff_model, static_model = eqx.partition(model, model_filter_spec)
    (loss, (new_state, nll, spread, mean, var)), grads = eqx.filter_value_and_grad(
        jax_regression_loss_fn, has_aux=True
    )(diff_model, static_model, state, X, chirp_mass, beta, lambda_spread, key)
    updates, new_opt_state = opt_update(grads, opt_state, diff_model)
    new_model = eqx.combine(eqx.apply_updates(diff_model, updates), static_model)
    return new_model, new_state, new_opt_state, {
        "loss": loss,
        "nll": nll,
        "spread": spread,
        "mean": mean,
        "var": var,
    }


@eqx.filter_jit
def jax_inference(
    model: eqx.Module,
    X_time: Array,
    state: eqx.nn.State,
    key: PRNGKeyArray,
) -> tuple[Array, eqx.nn.State]:
    inference_model = eqx.tree_inference(model, value=True)
    return jax_fwd_batch(inference_model, X_time, state, key)
