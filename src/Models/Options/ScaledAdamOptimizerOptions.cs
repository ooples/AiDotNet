namespace AiDotNet.Models.Options;

/// <summary>Configuration for <see cref="AiDotNet.Optimizers.ScaledAdamOptimizer{T, TInput, TOutput}"/>.</summary>
/// <typeparam name="T">The numeric type.</typeparam>
/// <typeparam name="TInput">The model input type.</typeparam>
/// <typeparam name="TOutput">The model output type.</typeparam>
/// <remarks>The defaults are icefall's <c>ScaledAdam</c> (and GPT-SoVITS's copy, <c>AR/modules/optim.py</c>).</remarks>
public class ScaledAdamOptimizerOptions<T, TInput, TOutput> : GradientBasedOptimizerOptions<T, TInput, TOutput>
{
    /// <summary>Gets or sets the mini-batch size of <c>Optimize</c> (32).</summary>
    public int BatchSize { get; set; } = 32;

    /// <summary>Gets or sets the learning rate (0.03).</summary>
    public override double InitialLearningRate { get; set; } = 0.03;

    /// <summary>Gets or sets the momentum decay β₁ (0.9).</summary>
    public double Beta1 { get; set; } = 0.9;

    /// <summary>Gets or sets the second-moment decay β₂ (0.98).</summary>
    public double Beta2 { get; set; } = 0.98;

    /// <summary>Gets or sets the clipping scale: gradients are clipped to this many times the median gradient norm of
    /// the last <see cref="ClippingUpdatePeriod"/> steps; null disables clipping (the default).</summary>
    public double? ClippingScale { get; set; }

    /// <summary>Gets or sets the learning-rate factor for parameter scales and scalar parameters (0.1).</summary>
    public double ScalarLearningRateScale { get; set; } = 0.1;

    /// <summary>Gets or sets ε (1e-8).</summary>
    public double Epsilon { get; set; } = 1e-8;

    /// <summary>Gets or sets the smallest root-mean-square a tensor's scale learning shrinks it to (1e-5).</summary>
    public double ParameterMinRms { get; set; } = 1e-5;

    /// <summary>Gets or sets the largest root-mean-square a tensor's scale learning grows it to (3).</summary>
    public double ParameterMaxRms { get; set; } = 3.0;

    /// <summary>Gets or sets the largest absolute value of a scalar parameter (10).</summary>
    public double ScalarMax { get; set; } = 10.0;

    /// <summary>Gets or sets how often, in steps, the parameter scales update (4).</summary>
    public int SizeUpdatePeriod { get; set; } = 4;

    /// <summary>Gets or sets how often, in steps, the clipping threshold updates (100).</summary>
    public int ClippingUpdatePeriod { get; set; } = 100;

    /// <summary>
    /// Gets or sets the gradient accumulation period: with a period above 1, each call's gradients are summed and the
    /// optimizer steps on calls n &gt; 0 with n divisible by the period (counting calls from 0), as GPT-SoVITS's s1 loop
    /// steps at <c>batch_idx &gt; 0 and batch_idx % 4 == 0</c> (so its first step sums five batches). 1 (the default)
    /// steps on every call.
    /// </summary>
    public int GradientAccumulationPeriod { get; set; } = 1;
}
