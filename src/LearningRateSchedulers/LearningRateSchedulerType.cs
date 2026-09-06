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
    /// Linear warmup, then a constant hold, then linear decay.
    /// </summary>
    /// <remarks>
    /// The schedule wav2vec 2.0 fine-tunes with, and widely reused after it. Distinct from
    /// <see cref="LinearWarmup"/>, which has no hold phase, and from the Noam-hold schedule, whose
    /// final stage decays as a power of the step rather than linearly.
    /// </remarks>
    TriStage,

    /// <summary>
    /// The Noam schedule: linear warmup, then inverse-square-root decay.
    /// </summary>
    /// <remarks>
    /// The schedule of Vaswani et al. 2017 Sec. 5.3, and the one a large part of the transformer
    /// literature means by "the transformer learning rate schedule". Its peak is a function of the
    /// model dimension rather than a stated constant, so unlike every other member here it cannot
    /// be built from the recipe alone.
    /// </remarks>
    Noam,

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
