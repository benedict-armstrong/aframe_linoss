from architectures.base import JaxArchitecture


class RegressionTimeDomainLinOSS(JaxArchitecture):
    """JAX/Equinox LinOSS architecture for BNS parameter regression.

    Wraps ``LinOSS`` with ``d_output=2`` (chirp-mass mean + pre-Softplus
    log-variance) so it can be used with ``JaxRegressionAframe``.
    """

    linoss: "LinOSS"  # noqa: F821

    def __init__(
        self,
        num_ifos: int,
        sample_rate: float,
        kernel_length: float,
        time_in_features: int,
        time_hidden_dim: int,
        time_num_blocks: int,
        time_dropout_rate: float,
        time_state_dim: int,
        time_r_min: float,
        time_theta_max: float,
        time_num_res_blocks: int,
        time_conv_kernel_size: int,
        time_conv_position: str = "pre",
        resnet_layers: tuple[int, ...] = (2, 2, 2),
        resnet_latent_dim: int = 64,
        resnet_kernel_size: int = 21,
        resnet_norm_groups: int = 16,
        mlp_width: int = 64,
        mlp_depth: int = 2,
        d_output: int = 2,
        *,
        seed: int = 0,
    ) -> None:
        import jax.random as jr
        from architectures.networks.linoss import LinOSS

        self.linoss = LinOSS(
            time_in_features=time_in_features,
            time_hidden_dim=time_hidden_dim,
            time_num_blocks=time_num_blocks,
            time_dropout_rate=time_dropout_rate,
            time_state_dim=time_state_dim,
            time_r_min=time_r_min,
            time_theta_max=time_theta_max,
            time_num_res_blocks=time_num_res_blocks,
            time_conv_kernel_size=time_conv_kernel_size,
            time_conv_position=time_conv_position,
            resnet_layers=resnet_layers,
            resnet_latent_dim=resnet_latent_dim,
            resnet_kernel_size=resnet_kernel_size,
            resnet_norm_groups=resnet_norm_groups,
            mlp_width=mlp_width,
            mlp_depth=mlp_depth,
            d_output=d_output,
            key=jr.PRNGKey(seed),
        )

    def __call__(self, X, state, key=None):
        return self.linoss(X, state, key=key)

    def forward(self, X, state, key=None):
        return self.linoss(X, state, key=key)
