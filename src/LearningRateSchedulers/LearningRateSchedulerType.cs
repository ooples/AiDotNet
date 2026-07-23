namespace AiDotNet.LearningRateSchedulers;

/// <summary>
/// Enumeration of available learning rate scheduler types.
/// </summary>
/// <remarks>
/// <para>
/// Use this enum with the <see cref="LearningRateSchedulerFactory"/> to create
/// schedulers by type without having to reference the concrete classes directly.
/// </para>
/// </remarks>
public enum LearningRateSchedulerType
{
    /// <summary>
    /// Constant learning rate (no decay).
    /// </summary>
    Constant,

    /// <summary>
    /// Step decay: multiply LR by gamma every step_size epochs.
    /// </summary>
    Step,

    /// <summary>
    /// Multi-step decay: multiply LR by gamma at specified milestones.
    /// </summary>
    MultiStep,

    /// <summary>
    /// Exponential decay: multiply LR by gamma every epoch.
    /// </summary>
    Exponential,

    /// <summary>
    /// Polynomial decay: LR follows polynomial curve to end value.
    /// </summary>
    Polynomial,

    /// <summary>
    /// Cosine annealing: smooth cosine-shaped decay.
    /// </summary>
    CosineAnnealing,

    /// <summary>
    /// Cosine annealing with warm restarts (SGDR).
    /// </summary>
    CosineAnnealingWarmRestarts,

    /// <summary>
    /// One cycle policy: warmup then annealing.
    /// </summary>
    OneCycle,

    /// <summary>
    /// Linear warmup followed by optional decay.
    /// </summary>
    LinearWarmup,

    /// <summary>
    /// Cyclic learning rate: oscillate between bounds.
    /// </summary>
    Cyclic,

    /// <summary>
    /// Reduce on plateau: decrease when metric stops improving.
    /// </summary>
    ReduceOnPlateau,

    /// <summary>
    /// Custom lambda function scheduler.
    /// </summary>
    Lambda,

    /// <summary>
    /// Sequential composition of multiple schedulers.
    /// </summary>
    Sequential
}
