using AiDotNet.Interfaces;

namespace AiDotNet.Models.Options;

/// <summary>
/// Controls step-level behavior of the streaming training path (<c>ConfigureDataLoader</c> with an
/// <see cref="IStreamingDataLoader{T, TInput, TOutput}"/>): a reproducible data order, a global step budget,
/// periodic held-out validation, and resuming from the latest checkpoint.
/// </summary>
/// <remarks>
/// <para>
/// <see cref="ModelOptions.Seed"/> fixes the data order: epoch <c>e</c> is shuffled with <c>Seed + e</c>, so the
/// order is identical across runs and across a stop/resume. <c>null</c> keeps the loader's unseeded shuffle.
/// </para>
/// <para>
/// Periodic checkpoint cadence and retention come from the configured checkpoint manager
/// (<see cref="ICheckpointManager{T, TInput, TOutput}.ConfigureAutoCheckpointing"/>): the streaming loop offers a
/// checkpoint after every optimizer step and the manager decides whether to write it.
/// </para>
/// <para><b>For Beginners:</b> Long training runs (such as pretraining a language model) are measured in steps,
/// not epochs, and must survive being stopped. These options make the run reproducible (same data order every
/// time), let it stop after a fixed number of steps, check quality on held-out data every so often, and pick up
/// exactly where a previous run left off.</para>
/// <para><b>Reference:</b> Hu et al., "MiniCPM: Unveiling the Potential of Small Language Models with Scalable
/// Training Strategies" (2024), which introduced the warmup-stable-decay schedule these options extend on resume
/// and trains in fixed step budgets with periodic held-out evaluation.</para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <typeparam name="TInput">The input type of the model.</typeparam>
/// <typeparam name="TOutput">The output type of the model.</typeparam>
public class StreamingTrainingOptions<T, TInput, TOutput> : ModelOptions
{
    /// <summary>Initializes a new instance with default values.</summary>
    public StreamingTrainingOptions() { }

    /// <summary>Initializes a new instance by copying from another instance.</summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when <paramref name="other"/> is null.</exception>
    /// <remarks>The validation loader is a data source, not a setting, so the copy shares the same instance.</remarks>
    public StreamingTrainingOptions(StreamingTrainingOptions<T, TInput, TOutput> other)
        : base(other ?? throw new ArgumentNullException(nameof(other)))
    {
        MaxSteps = other.MaxSteps;
        ValidationLoader = other.ValidationLoader;
        ValidateEveryNSteps = other.ValidateEveryNSteps;
        ValidationMaxBatches = other.ValidationMaxBatches;
        ResumeFromLatestCheckpoint = other.ResumeFromLatestCheckpoint;
        UseConfiguredSchedulerOnResume = other.UseConfiguredSchedulerOnResume;
    }

    /// <summary>
    /// Gets or sets the global optimizer-step budget. Training stops after this many steps even mid-epoch.
    /// </summary>
    /// <value>The step budget, or <c>null</c> (the default) to train for the optimizer's configured epoch count.</value>
    /// <remarks>On resume the budget counts from step 0 of the original run, so raising it extends a run.</remarks>
    public long? MaxSteps { get; set; }

    /// <summary>
    /// Gets or sets a held-out loader evaluated during training. Its batches are never trained on.
    /// </summary>
    /// <value>The validation loader, or <c>null</c> (the default) for no validation.</value>
    /// <remarks>Required when <see cref="ValidateEveryNSteps"/> is positive.</remarks>
    public IStreamingDataLoader<T, TInput, TOutput>? ValidationLoader { get; set; }

    /// <summary>
    /// Gets or sets how often (in optimizer steps) to evaluate <see cref="ValidationLoader"/>.
    /// </summary>
    /// <value>The validation interval in steps. Defaults to <c>0</c>.</value>
    /// <remarks><c>0</c> evaluates only once, when training ends.</remarks>
    public int ValidateEveryNSteps { get; set; }

    /// <summary>
    /// Gets or sets the maximum number of validation batches per evaluation.
    /// </summary>
    /// <value>The batch cap, or <c>null</c> (the default) to evaluate the whole loader.</value>
    /// <remarks>Validation always reads the loader in its unshuffled order, so every evaluation sees the same
    /// batches.</remarks>
    public int? ValidationMaxBatches { get; set; }

    /// <summary>
    /// Gets or sets whether to resume from the checkpoint manager's latest checkpoint.
    /// </summary>
    /// <value><c>true</c> to resume; defaults to <c>false</c>.</value>
    /// <remarks>Restores model state, optimizer state including the learning-rate schedule, the global step and
    /// the data position. Requires <see cref="ModelOptions.Seed"/> so the skipped data is exactly the data already
    /// trained on.</remarks>
    public bool ResumeFromLatestCheckpoint { get; set; }

    /// <summary>
    /// Gets or sets whether, on resume, the learning-rate scheduler configured on this builder replaces the one
    /// stored in the checkpoint (fast-forwarded to the restored step).
    /// </summary>
    /// <value><c>true</c> to use the builder's scheduler; defaults to <c>false</c>, which restores the
    /// checkpoint's schedule exactly.</value>
    /// <remarks>Use this to EXTEND a run whose schedule does not depend on its length, e.g. moving the decay start
    /// of a <see cref="LearningRateSchedulers.WarmupStableDecayScheduler"/>.</remarks>
    public bool UseConfiguredSchedulerOnResume { get; set; }
    /// <summary>
    /// Validates the option values.
    /// </summary>
    /// <exception cref="ArgumentException">A value is out of range or the combination is inconsistent.</exception>
    public void Validate()
    {
        if (MaxSteps is <= 0)
            throw new ArgumentException("MaxSteps must be positive when set.", nameof(MaxSteps));
        if (ValidateEveryNSteps < 0)
            throw new ArgumentException("ValidateEveryNSteps cannot be negative.", nameof(ValidateEveryNSteps));
        if (ValidationMaxBatches is <= 0)
            throw new ArgumentException("ValidationMaxBatches must be positive when set.", nameof(ValidationMaxBatches));
        if (ValidateEveryNSteps > 0 && ValidationLoader is null)
            throw new ArgumentException("ValidateEveryNSteps requires a ValidationLoader.", nameof(ValidationLoader));
        if (ResumeFromLatestCheckpoint && Seed is null)
            throw new ArgumentException(
                "ResumeFromLatestCheckpoint requires Seed: without a fixed data order the batches skipped on resume " +
                "would not be the batches already trained on.", nameof(Seed));
    }
}
