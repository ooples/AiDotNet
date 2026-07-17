namespace AiDotNet.SelfSupervisedLearning;

/// <summary>
/// The outcome of a self-supervised pretraining pass: the loss trajectory, a representation-collapse
/// check, and an optional linear-probe quality estimate.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Beyond a plain pretraining loop, this reports whether the encoder <b>collapsed</b> — mapping every
/// input to nearly the same vector, the classic BYOL/SimSiam failure that a falling loss can hide — and,
/// when labels are available, a quick linear-probe score of how useful the learned representations are.
/// </para>
/// </remarks>
public sealed class SelfSupervisedLearningPretrainingResult<T>
{
    /// <summary>Gets the configured SSL method's name.</summary>
    public string MethodName { get; init; } = string.Empty;

    /// <summary>Gets the number of pretraining epochs run.</summary>
    public int EpochsRun { get; init; }

    /// <summary>Gets the total number of training steps (batches) run.</summary>
    public int StepsRun { get; init; }

    /// <summary>Gets the average loss of the final epoch.</summary>
    public double FinalLoss { get; init; }

    /// <summary>Gets the average loss per epoch, in order.</summary>
    public IReadOnlyList<double> EpochLosses { get; init; } = System.Array.Empty<double>();

    /// <summary>
    /// Gets whether representation collapse was detected (the encoder's outputs have near-zero spread).
    /// </summary>
    public bool CollapseDetected { get; init; }

    /// <summary>
    /// Gets the mean per-dimension standard deviation of the encoder's representations across a batch.
    /// A value near zero means collapse (all inputs map to the same point).
    /// </summary>
    public double RepresentationStdDev { get; init; }

    /// <summary>
    /// Gets the in-sample linear-probe R² of the learned representations against the targets, or
    /// <c>null</c> when a probe was not run (no targets available). Higher means the representations
    /// linearly capture more of the target signal.
    /// </summary>
    public double? LinearProbeR2 { get; init; }
}
