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
/// order is identical across runs and across a stop/resume. <c>null</c> draws the base seed once per run (rank 0, shared
/// with every rank) and checkpoints it, so the order differs between fresh runs but a resumed run continues the order of
/// the run it resumes, as PyTorch's <c>DistributedSampler</c> resumes its seeded order.
/// </para>
/// <para>
/// Periodic checkpoint cadence and retention come from the configured checkpoint manager
/// (<see cref="ICheckpointManager{T, TInput, TOutput}.ConfigureAutoCheckpointing(AutoCheckpointingOptions)"/>): the streaming loop offers a
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
        GradientAccumulationSteps = other.GradientAccumulationSteps;
        ValidationLoader = other.ValidationLoader;
        ValidateEveryNSteps = other.ValidateEveryNSteps;
        ValidationMaxBatches = other.ValidationMaxBatches;
        ResumeFromLatestCheckpoint = other.ResumeFromLatestCheckpoint;
        UseConfiguredSchedulerOnResume = other.UseConfiguredSchedulerOnResume;
        MonitorEveryNSteps = other.MonitorEveryNSteps;
        TokensPerSample = other.TokensPerSample;
        MonitorGradNorm = other.MonitorGradNorm;
        MonitorResourceIntervalSeconds = other.MonitorResourceIntervalSeconds;
    }

    /// <summary>
    /// Gets or sets the number of micro-batches accumulated before each optimizer step.
    /// </summary>
    /// <value>The accumulation factor; defaults to <c>1</c> (step on every micro-batch).</value>
    /// <remarks>
    /// <para>
    /// In distributed runs, gradient synchronization (<c>AllReduce</c> / <c>ReduceScatter</c>), mixed-precision
    /// unscaling, and global-norm clipping run once per optimizer step after all <see cref="GradientAccumulationSteps"/>
    /// micro-batches have been accumulated locally.
    /// </para>
    /// <para><b>For Beginners:</b> If your GPU only fits a batch of 8 samples at a time, setting
    /// <see cref="GradientAccumulationSteps"/> to <c>4</c> adds up the gradients from 4 mini-batches before
    /// updating the weights once, giving you the stable learning behavior of a batch size of 32 without needing
    /// 4x the memory.</para>
    /// </remarks>
    public int GradientAccumulationSteps { get; set; } = 1;

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
    /// the data position, and the data order (the configured <see cref="ModelOptions.Seed"/>, or the seed the run drew),
    /// so the skipped data is exactly the data already trained on.</remarks>
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
    /// Gets or sets how often (in optimizer steps) the streaming loop reports step metrics to the configured
    /// training monitor.
    /// </summary>
    /// <value>The reporting interval in steps; defaults to <c>10</c>. <c>0</c> turns step reporting off, leaving
    /// only the end-of-epoch metrics.</value>
    /// <remarks>
    /// <para>Every N steps the monitor receives, at the global step number (counted across epochs):
    /// <c>train_loss</c> (the last step's loss), <c>train_loss_avg</c> (mean loss over the N steps),
    /// <c>learning_rate</c>, <c>step_ms</c> (wall-clock milliseconds per step over the window, measured with
    /// <see cref="System.Diagnostics.Stopwatch"/> and including data loading), <c>samples_per_second</c>,
    /// <c>tokens_per_second</c> when <see cref="TokensPerSample"/> is set, and <c>grad_norm</c> when the
    /// optimizer step computed one. It also receives <see cref="ITrainingMonitor{T}.UpdateProgress"/>.</para>
    /// <para>The loss is the value the loop already reads after each step, so reporting adds no device
    /// synchronization.</para>
    /// <para><b>For Beginners:</b> a long language-model run is often a single epoch of thousands of steps; this
    /// lets a dashboard show the loss curve while that epoch is still running.</para>
    /// </remarks>
    public int MonitorEveryNSteps { get; set; } = 10;

    /// <summary>
    /// Gets or sets the number of tokens in each training sample, used to report <c>tokens_per_second</c>.
    /// </summary>
    /// <value>Tokens per sample, or <c>null</c> (the default) to not report token throughput.</value>
    public int? TokensPerSample { get; set; }

    /// <summary>
    /// Gets or sets whether the optimizer reduces the global gradient norm on every step even when its clipping
    /// options would not, so <c>grad_norm</c> is always reported.
    /// </summary>
    /// <value><c>true</c> to force the reduction; defaults to <c>false</c>.</value>
    /// <remarks>
    /// <para>On an eager tape step with global-norm clipping on (the network's and the gradient-based optimizers'
    /// default), the clip already reduces the pre-clip norm; on a CPU engine it is a host value and <c>grad_norm</c>
    /// is reported for free whatever this is set to. On a GPU engine the network's clip leaves the norm on the
    /// device, and this allows reading it back (one scalar, which waits for the device). With clipping by norm off,
    /// no step computes it; setting this adds one sum of squares over every gradient per step and, for gradients
    /// resident on a GPU, a device reduction plus a device-to-host read of its scalar.</para>
    /// <para>A step taken by a fused compiled training plan reports no <c>grad_norm</c> either way: the plan clips
    /// internally and does not expose the norm.</para>
    /// </remarks>
    public bool MonitorGradNorm { get; set; }

    /// <summary>
    /// Gets or sets how often, in seconds of wall time, the streaming loop reports process and GPU resource usage
    /// to the training monitor.
    /// </summary>
    /// <value>The interval in seconds; defaults to <c>30</c>. <c>0</c> turns resource reporting off.</value>
    /// <remarks>Reports process CPU percent (from <see cref="System.Diagnostics.Process.TotalProcessorTime"/>
    /// deltas, normalized by the processor count) and working set in MB through
    /// <see cref="ITrainingMonitor{T}.LogResourceUsage"/>, and, when a GPU engine is active, the GPU allocator's
    /// allocated and reserved memory and the device's total memory as the metrics <c>gpu_memory_allocated_mb</c>,
    /// <c>gpu_memory_reserved_mb</c> and <c>gpu_memory_total_mb</c>. Checked on reporting steps only.</remarks>
    public double MonitorResourceIntervalSeconds { get; set; } = 30.0;
    /// <summary>
    /// Validates the option values.
    /// </summary>
    /// <exception cref="ArgumentException">A value is out of range or the combination is inconsistent.</exception>
    public void Validate()
    {
        if (MaxSteps is <= 0)
            throw new ArgumentException("MaxSteps must be positive when set.", nameof(MaxSteps));
        if (GradientAccumulationSteps <= 0)
            throw new ArgumentException("GradientAccumulationSteps must be at least 1.", nameof(GradientAccumulationSteps));
        if (ValidateEveryNSteps < 0)
            throw new ArgumentException("ValidateEveryNSteps cannot be negative.", nameof(ValidateEveryNSteps));
        if (ValidationMaxBatches is <= 0)
            throw new ArgumentException("ValidationMaxBatches must be positive when set.", nameof(ValidationMaxBatches));
        if (ValidateEveryNSteps > 0 && ValidationLoader is null)
            throw new ArgumentException("ValidateEveryNSteps requires a ValidationLoader.", nameof(ValidationLoader));
        if (MonitorEveryNSteps < 0)
            throw new ArgumentException("MonitorEveryNSteps cannot be negative.", nameof(MonitorEveryNSteps));
        if (TokensPerSample is <= 0)
            throw new ArgumentException("TokensPerSample must be positive when set.", nameof(TokensPerSample));
        if (MonitorResourceIntervalSeconds < 0 || double.IsNaN(MonitorResourceIntervalSeconds))
            throw new ArgumentException("MonitorResourceIntervalSeconds cannot be negative.", nameof(MonitorResourceIntervalSeconds));
    }
}
