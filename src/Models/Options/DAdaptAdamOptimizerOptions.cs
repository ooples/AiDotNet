namespace AiDotNet.Models.Options;

/// <summary>
/// Options for D-Adaptation Adam (Defazio and Mishchenko 2023), which estimates the step size from the iterates
/// instead of taking it from a tuned learning rate.
/// </summary>
/// <remarks>
/// <para>The learning rate is a multiplier on the estimated distance d (1 by default, as the paper recommends), so a
/// schedule such as a warmup and cosine decay still shapes the step. Defaults follow the reference implementation
/// (facebookresearch/dadaptation <c>DAdaptAdam</c>, also vendored in audiocraft).</para>
/// <para><b>For Beginners:</b> Instead of guessing how big each update should be, the optimizer measures how far the
/// weights have travelled and grows its step to match, starting from a tiny one.</para>
/// </remarks>
/// <typeparam name="T">The numeric type.</typeparam>
/// <typeparam name="TInput">The model input type.</typeparam>
/// <typeparam name="TOutput">The model output type.</typeparam>
public class DAdaptAdamOptimizerOptions<T, TInput, TOutput> : GradientBasedOptimizerOptions<T, TInput, TOutput>
{
    /// <summary>Gets or sets the mini-batch size of <c>Optimize</c> (32).</summary>
    public int BatchSize { get; set; } = 32;

    /// <summary>Gets or sets the learning-rate multiplier on the estimated step size d (1).</summary>
    public override double InitialLearningRate { get; set; } = 1.0;

    /// <summary>Gets or sets the first-moment decay β₁ (0.9).</summary>
    public double Beta1 { get; set; } = 0.9;

    /// <summary>Gets or sets the second-moment decay β₂ (0.999).</summary>
    public double Beta2 { get; set; } = 0.999;

    /// <summary>Gets or sets the denominator's ε (1e-8).</summary>
    public double Epsilon { get; set; } = 1e-8;

    /// <summary>Gets or sets the weight decay (0).</summary>
    public double WeightDecay { get; set; }

    /// <summary>Gets or sets whether weight decay is decoupled, AdamW-style (true), rather than added to the
    /// gradient.</summary>
    public bool DecoupleWeightDecay { get; set; } = true;

    /// <summary>Gets or sets the initial distance estimate d₀ (1e-6).</summary>
    public double InitialDistance { get; set; } = 1e-6;

    /// <summary>Gets or sets the most d may grow per step, as a factor (unbounded).</summary>
    public double GrowthRate { get; set; } = double.PositiveInfinity;
}
