namespace AiDotNet.LearningRateSchedulers;

/// <summary>
/// Specifies when the learning rate scheduler should be stepped during training.
/// </summary>
/// <remarks>
/// <para>
/// Different training scenarios require different scheduling strategies. This enum
/// allows you to configure how frequently the learning rate is updated.
/// </para>
/// <para><b>For Beginners:</b> This controls when the learning rate changes:
/// - Per batch: Changes after every mini-batch (more frequent, smoother changes)
/// - Per epoch: Changes after each complete pass through the dataset (most common)
/// - Warmup then epoch: Increases LR during warmup (per batch), then switches to per-epoch
/// </para>
/// </remarks>
public enum SchedulerStepMode
{
    /// <summary>
    /// Step the scheduler after each mini-batch.
    /// </summary>
    /// <remarks>
    /// Use this for schedulers that need fine-grained control, such as warmup schedulers
    /// or cyclical learning rate policies.
    /// </remarks>
    StepPerBatch,

    /// <summary>
    /// Step the scheduler after each epoch (default).
    /// </summary>
    /// <remarks>
    /// This is the most common mode. The learning rate changes once per complete pass
    /// through the training dataset.
    /// </remarks>
    StepPerEpoch,

    /// <summary>
    /// Step per-batch during warmup phase, then switch to per-epoch.
    /// </summary>
    /// <remarks>
    /// This is useful for transformer training where a warmup phase gradually increases
    /// the learning rate, followed by a decay schedule that operates per-epoch.
    /// </remarks>
    WarmupThenEpoch
}
