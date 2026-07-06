using System.Collections.Generic;

namespace AiDotNet.Interfaces;

/// <summary>
/// A snapshot of training progress for a single completed epoch, handed to every
/// registered <see cref="ITrainingCallback{T}"/> so callers can plot live loss curves,
/// stream metrics to a dashboard, or decide whether training should keep going.
/// </summary>
/// <remarks>
/// <para>
/// This is an immutable value object produced by the training loop once per epoch. It
/// carries the epoch's aggregate loss, any additional named metrics, and timing
/// information so a callback can reason about both quality (is the loss falling?) and
/// speed (how fast are epochs completing?).
/// </para>
/// <para><b>For Beginners:</b> Think of this as a single row in a training log. After
/// every pass over your data (an "epoch"), the trainer fills one of these out and shows
/// it to you:
/// - which epoch just finished (<see cref="Epoch"/>) and how many are planned (<see cref="TotalEpochs"/>),
/// - the average error for that epoch (<see cref="Loss"/>) and any extras like accuracy (<see cref="Metrics"/>),
/// - how long training has been running (<see cref="Elapsed"/>) and roughly how fast it is going (<see cref="StepsPerSecond"/>).
/// You never construct this yourself — the framework hands it to your callback so you can
/// watch training happen in real time.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric data type used for calculations (e.g., float, double).</typeparam>
public sealed class TrainingProgress<T>
{
    /// <summary>
    /// Gets the zero-based index of the epoch that just completed.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> The first completed epoch reports <c>0</c>, the second <c>1</c>,
    /// and so on. Add one when displaying it to humans (e.g. "Epoch 1 of 10").
    /// </remarks>
    public int Epoch { get; }

    /// <summary>
    /// Gets the total number of epochs the training loop plans to run, or <c>0</c> when the
    /// count is not known in advance.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> Use this together with <see cref="Epoch"/> to render a progress
    /// bar. It reflects the configured maximum; training may still stop earlier if a callback
    /// requests an abort or an early-stopping rule fires.
    /// </remarks>
    public int TotalEpochs { get; }

    /// <summary>
    /// Gets the aggregate loss for the epoch (typically the mean loss across all batches).
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> Loss is a single number that says how wrong the model currently
    /// is — lower is better. Watching it fall epoch over epoch is the simplest sign that
    /// training is working. A value that becomes NaN (not-a-number) or shoots upward means
    /// training has diverged.
    /// </remarks>
    public T Loss { get; }

    /// <summary>
    /// Gets any additional named metrics recorded for the epoch (e.g. "accuracy",
    /// "validation_loss"). Never null; may be empty.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> Besides loss, a trainer might also track other measurements.
    /// They live here keyed by name so a dashboard can plot each series separately.
    /// </remarks>
    public IReadOnlyDictionary<string, T> Metrics { get; }

    /// <summary>
    /// Gets the wall-clock time elapsed since training began.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> How long the whole training run has been going, measured from
    /// the moment before the first epoch. Handy for estimating time remaining.
    /// </remarks>
    public TimeSpan Elapsed { get; }

    /// <summary>
    /// Gets the estimated training throughput in optimizer steps per second, or <c>null</c>
    /// when throughput cannot be estimated.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> A rough speedometer for training. It is optional because not
    /// every training path can measure it reliably.
    /// </remarks>
    public double? StepsPerSecond { get; }

    /// <summary>
    /// Initializes a new <see cref="TrainingProgress{T}"/> snapshot.
    /// </summary>
    /// <param name="epoch">The zero-based index of the completed epoch.</param>
    /// <param name="totalEpochs">The total number of epochs planned, or 0 if unknown.</param>
    /// <param name="loss">The aggregate loss for the epoch.</param>
    /// <param name="metrics">Additional named metrics; if null, an empty set is used.</param>
    /// <param name="elapsed">Wall-clock time elapsed since training began.</param>
    /// <param name="stepsPerSecond">Optional throughput estimate in steps per second.</param>
    public TrainingProgress(
        int epoch,
        int totalEpochs,
        T loss,
        IReadOnlyDictionary<string, T>? metrics,
        TimeSpan elapsed,
        double? stepsPerSecond = null)
    {
        Epoch = epoch;
        TotalEpochs = totalEpochs;
        Loss = loss;
        Metrics = metrics ?? new Dictionary<string, T>();
        Elapsed = elapsed;
        StepsPerSecond = stepsPerSecond;
    }
}
