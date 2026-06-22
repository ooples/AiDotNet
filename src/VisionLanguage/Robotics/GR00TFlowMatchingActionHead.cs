using System;
using AiDotNet.Helpers;

namespace AiDotNet.VisionLanguage.Robotics;

/// <summary>
/// Flow-matching action head used by GR00T N1 to denoise continuous joint commands conditioned on
/// the System-2 vision-language latent.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Per GR00T N1 (NVIDIA, 2025, "GR00T N1: An Open Foundation Model for Generalist Humanoid Robots",
/// arXiv:2503.14734), the System-1 action policy is a DiT-style transformer trained with the
/// <b>flow-matching</b> objective (Lipman et al. 2023, arXiv:2210.02747). Inference reverses the
/// learned vector field by Euler-integrating from Gaussian noise at <c>t=0</c> to the data
/// distribution at <c>t=1</c>; each integration step queries the network for the velocity field
/// <c>v_θ(x_t, t | latent)</c> and updates <c>x_{t+Δt} = x_t + Δt · v_θ</c>.
/// </para>
/// <para>
/// This class implements the inference-time integrator. The per-step velocity network is supplied
/// as a callback (the GR00T-N1 model wires its DiT decoder there), keeping this class composable.
/// </para>
/// <para><b>References:</b></para>
/// <list type="bullet">
///   <item>Lipman et al., "Flow Matching for Generative Modeling", ICLR 2023, arXiv:2210.02747.</item>
///   <item>Bjorck et al., "GR00T N1: An Open Foundation Model for Generalist Humanoid Robots", NVIDIA 2025, arXiv:2503.14734 — §3.2 action head, §4.1 inference protocol (16 Euler steps default).</item>
///   <item>Black et al., "π0: A Vision-Language-Action Flow Model for General Robot Control", Physical Intelligence 2024 — established flow-matching as the leading VLA action-head choice.</item>
/// </list>
/// </remarks>
public sealed class GR00TFlowMatchingActionHead<T>
{
    private readonly INumericOperations<T> _numOps;
    private readonly Func<Tensor<T>, double, Tensor<T>, Tensor<T>> _velocityNetwork;
    private readonly int _numIntegrationSteps;
    private readonly Random _noiseSource;

    /// <summary>Number of Euler integration steps for inference. Paper §4.1 default: 16.</summary>
    public int NumIntegrationSteps => _numIntegrationSteps;

    /// <summary>
    /// Builds an action head.
    /// </summary>
    /// <param name="velocityNetwork">Per-step velocity field: <c>(x_t, t∈[0,1], latent) → v_θ(x_t,t|latent)</c>. Returns the velocity at the same shape as <c>x_t</c>.</param>
    /// <param name="numIntegrationSteps">Number of Euler steps. Paper default 16; smaller is faster but lower-quality.</param>
    /// <param name="seed">Optional seed for the Gaussian noise source. Default = time-based unseeded RNG (set this for reproducible inference).</param>
    public GR00TFlowMatchingActionHead(
        Func<Tensor<T>, double, Tensor<T>, Tensor<T>> velocityNetwork,
        int numIntegrationSteps = 16,
        int? seed = null
    )
    {
        if (numIntegrationSteps <= 0)
            throw new ArgumentOutOfRangeException(
                nameof(numIntegrationSteps),
                numIntegrationSteps,
                "numIntegrationSteps must be positive."
            );

        _velocityNetwork =
            velocityNetwork ?? throw new ArgumentNullException(nameof(velocityNetwork));
        _numIntegrationSteps = numIntegrationSteps;
        _noiseSource = seed.HasValue
            ? RandomHelper.CreateSeededRandom(seed.Value)
            : RandomHelper.CreateSecureRandom();
        _numOps = MathHelper.GetNumericOperations<T>();
    }

    /// <summary>
    /// Generates an action tensor of length <paramref name="actionDimension"/> by Euler-integrating
    /// the flow-matching vector field from Gaussian noise at t=0 to the data distribution at t=1,
    /// conditioned on <paramref name="system2Latent"/>.
    /// </summary>
    public Tensor<T> Generate(int actionDimension, Tensor<T> system2Latent)
    {
        if (actionDimension <= 0)
            throw new ArgumentOutOfRangeException(nameof(actionDimension));
        if (system2Latent is null)
            throw new ArgumentNullException(nameof(system2Latent));

        var x = SampleGaussian(actionDimension);
        double dt = 1.0 / _numIntegrationSteps;
        for (int step = 0; step < _numIntegrationSteps; step++)
        {
            double t = (step + 0.5) * dt;
            var velocity = _velocityNetwork(x, t, system2Latent);
            if (velocity.Length != actionDimension)
                throw new InvalidOperationException(
                    $"Velocity network returned length {velocity.Length} but expected {actionDimension}."
                );

            for (int d = 0; d < actionDimension; d++)
            {
                double xd = _numOps.ToDouble(x[d]);
                double vd = _numOps.ToDouble(velocity[d]);
                x[d] = _numOps.FromDouble(xd + dt * vd);
            }
        }
        return x;
    }

    /// <summary>
    /// Generates a horizon of <paramref name="horizon"/> action vectors, each of length
    /// <paramref name="actionDimension"/>, concatenated into a flat tensor. Each horizon step
    /// uses the same conditioning latent (paper §3.2: latent is held constant across an action
    /// chunk to preserve temporal coherence).
    /// </summary>
    public Tensor<T> GenerateHorizon(int actionDimension, int horizon, Tensor<T> system2Latent)
    {
        if (horizon <= 0)
            throw new ArgumentOutOfRangeException(nameof(horizon));
        var flat = new Tensor<T>([actionDimension * horizon]);
        for (int step = 0; step < horizon; step++)
        {
            var stepAction = Generate(actionDimension, system2Latent);
            for (int d = 0; d < actionDimension; d++)
                flat[step * actionDimension + d] = stepAction[d];
        }
        return flat;
    }

    private Tensor<T> SampleGaussian(int dim)
    {
        var x = new Tensor<T>([dim]);
        for (int i = 0; i < dim; i++)
        {
            double u1 = Math.Max(1e-12, _noiseSource.NextDouble());
            double u2 = _noiseSource.NextDouble();
            // Box-Muller transform for standard Normal.
            double n = Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2);
            x[i] = _numOps.FromDouble(n);
        }
        return x;
    }
}
