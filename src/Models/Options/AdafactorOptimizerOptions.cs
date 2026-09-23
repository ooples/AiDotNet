namespace AiDotNet.Models.Options;

/// <summary>
/// Settings for <see cref="AiDotNet.Optimizers.AdafactorOptimizer{T, TInput, TOutput}"/>.
/// </summary>
/// <remarks>
/// <para>
/// Adafactor (Shazeer and Stern 2018) replaces Adam's full second-moment matrix with two vectors —
/// a row sum and a column sum — so its optimizer state is O(n+m) rather than O(nm) for an n by m
/// weight matrix. On a large embedding or projection that is the difference between an optimizer
/// that fits in memory and one that does not.
/// </para>
/// <para><b>For Beginners:</b> Adam remembers two numbers per weight to adapt each one's step size.
/// For a 1000x1000 layer that is two million extra numbers. Adafactor keeps a summary per row and
/// per column instead — two thousand — and reconstructs an approximation when it needs one. It
/// gives up a little precision to train models that otherwise would not fit.
/// </para>
/// </remarks>
public class AdafactorOptimizerOptions<T, TInput, TOutput> : GradientBasedOptimizerOptions<T, TInput, TOutput>
{
    /// <summary>Samples per optimizer step.</summary>
    public int BatchSize { get; set; } = 32;

    /// <summary>
    /// The step size. Zero means use the paper's relative step size instead of a fixed rate.
    /// </summary>
    /// <remarks>
    /// Adafactor is normally run without a hand-set learning rate: it derives one per step from the
    /// step number and the size of the weights themselves (see <see cref="UseRelativeStepSize"/>).
    /// A paper that states a rate — AudioPaLM fine-tunes at a constant 5e-5 — sets it here and turns
    /// the relative rule off.
    /// </remarks>
    public override double InitialLearningRate { get; set; } = 0.0;

    /// <summary>
    /// Whether to derive the step size from the step number rather than use a fixed rate.
    /// </summary>
    /// <remarks>
    /// The paper's default. rho(t) = min(1e-2, 1/sqrt(t)), then scaled by the RMS of the parameters
    /// so that the update is proportional to the weights it is applied to. Turn this off when a
    /// paper states an explicit rate, or the stated rate is ignored.
    /// </remarks>
    public bool UseRelativeStepSize { get; set; } = true;

    /// <summary>Floor on the parameter RMS used to scale the relative step.</summary>
    /// <remarks>
    /// The paper's epsilon2. Without it, a layer initialised at or near zero would get a step size
    /// of zero and never begin to move.
    /// </remarks>
    public double ParameterScaleFloor { get; set; } = 1e-3;

    /// <summary>Added to the squared gradient before it is accumulated.</summary>
    /// <remarks>The paper's epsilon1. Keeps the second-moment estimate away from exactly zero.</remarks>
    public double Epsilon { get; set; } = 1e-30;

    /// <summary>Norm at which an update is scaled back down.</summary>
    /// <remarks>
    /// The paper's clipping threshold d. Applied to the UPDATE rather than the gradient, which is
    /// what makes Adafactor stable without a warmup in the schedules it was designed for.
    /// </remarks>
    public double UpdateClippingThreshold { get; set; } = 1.0;

    /// <summary>
    /// Second-moment decay. NaN uses the paper's schedule, 1 - t^-0.8, which starts fast and slows.
    /// </summary>
    public double Beta2 { get; set; } = double.NaN;

    /// <summary>
    /// First-moment decay. NaN keeps no first moment at all, which is the paper's default and the
    /// reason its state is so small.
    /// </summary>
    public double Beta1 { get; set; } = double.NaN;

    /// <summary>Decoupled weight decay, applied to the parameters rather than the gradient.</summary>
    public double WeightDecay { get; set; } = 0.0;
}
