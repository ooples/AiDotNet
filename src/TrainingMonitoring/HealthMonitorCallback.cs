using System.Diagnostics;
using AiDotNet.Interfaces;
using LogLevel = AiDotNet.Interfaces.LogLevel;

namespace AiDotNet.TrainingMonitoring;

/// <summary>
/// A ready-to-use <see cref="ITrainingCallback{T}"/> that watches training health and
/// aborts automatically when it detects a run that has gone wrong.
/// </summary>
/// <remarks>
/// <para>
/// The callback stops training when any of the following holds at the end of an epoch:
/// </para>
/// <list type="number">
///   <item><description>The epoch loss is <c>NaN</c> or infinite (numerical blow-up).</description></item>
///   <item><description>The epoch loss has risen by more than a configurable percentage relative
///   to the best loss seen inside a recent sliding window (divergence).</description></item>
///   <item><description>An optionally-supplied <see cref="ITrainingMonitor{T}"/> reports a
///   critical issue via <see cref="ITrainingMonitor{T}.CheckForIssues"/>.</description></item>
/// </list>
/// <para>
/// When it aborts, the reason is recorded in <see cref="AbortReason"/> and, when a monitor
/// and session id are supplied, also logged to that monitor as an error message.
/// </para>
/// <para><b>For Beginners:</b> Long training runs sometimes "explode" — the loss turns into
/// NaN or starts climbing instead of falling — and every remaining epoch is then wasted. Add
/// this callback and it plays the role of a safety cut-off: it watches the loss after each
/// epoch and, the moment things look unhealthy, it returns <c>false</c> so the trainer stops
/// early and hands you back the model as it was before the blow-up. You can tune how sensitive
/// it is with <paramref name="lossRisePercent"/> and <paramref name="windowSize"/>.
/// </para>
/// <para>
/// Register it with <c>AiModelBuilder.ConfigureTrainingCallback(new HealthMonitorCallback&lt;T&gt;(...))</c>;
/// its thresholds (<c>lossRisePercent</c>, <c>windowSize</c>) and optional monitor integration are
/// user-configurable, and <see cref="AbortReason"/> is readable after training.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric data type used for calculations (e.g., float, double).</typeparam>
public sealed class HealthMonitorCallback<T> : ITrainingCallback<T>
{
    private readonly double _lossRiseFraction;
    private readonly int _windowSize;
    private readonly ITrainingMonitor<T>? _monitor;
    private readonly string? _monitorSessionId;
    private readonly Queue<double> _recentLosses;

    /// <summary>
    /// Gets the human-readable reason training was aborted, or <c>null</c> if this callback has
    /// not requested an abort.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> After training, read this to find out whether (and why) the safety
    /// cut-off fired — for example "loss became NaN at epoch 3".
    /// </remarks>
    public string? AbortReason { get; private set; }

    /// <summary>
    /// Initializes a new <see cref="HealthMonitorCallback{T}"/>.
    /// </summary>
    /// <param name="lossRisePercent">
    /// How much the loss is allowed to rise, as a percentage of the best loss in the recent
    /// window, before training is considered diverging. Defaults to 50 (i.e. a 50% rise
    /// aborts). Values less than or equal to zero disable the rising-loss check.
    /// </param>
    /// <param name="windowSize">
    /// The number of most-recent epochs the rising-loss check compares against. Defaults to 5.
    /// Must be at least 1.
    /// </param>
    /// <param name="monitor">
    /// An optional training monitor whose <see cref="ITrainingMonitor{T}.CheckForIssues"/> is
    /// consulted each epoch. When supplied together with <paramref name="monitorSessionId"/>,
    /// any reported issue triggers an abort.
    /// </param>
    /// <param name="monitorSessionId">
    /// The monitor session id to query for issues and to log abort messages to. Required for the
    /// <paramref name="monitor"/> integration to be active.
    /// </param>
    /// <remarks>
    /// <b>For Beginners:</b> The defaults are sensible for most runs — you can construct it with
    /// no arguments (<c>new HealthMonitorCallback&lt;float&gt;()</c>) and it will guard against
    /// NaN/infinite losses and a runaway (&gt;50%) rise.
    /// </remarks>
    /// <exception cref="ArgumentOutOfRangeException">
    /// <paramref name="lossRisePercent"/> is <c>NaN</c> or infinite, or <paramref name="windowSize"/> is less than 1.
    /// </exception>
    public HealthMonitorCallback(
        double lossRisePercent = 50.0,
        int windowSize = 5,
        ITrainingMonitor<T>? monitor = null,
        string? monitorSessionId = null)
    {
        // Validate rather than silently accept: a NaN/infinite threshold would quietly disable or break
        // the divergence check, and windowSize < 1 was previously rewritten to 1 (hiding a caller mistake).
        if (double.IsNaN(lossRisePercent) || double.IsInfinity(lossRisePercent))
        {
            throw new ArgumentOutOfRangeException(nameof(lossRisePercent), lossRisePercent,
                "lossRisePercent must be a finite number (use a value <= 0 to disable the rising-loss check).");
        }
        if (windowSize < 1)
        {
            throw new ArgumentOutOfRangeException(nameof(windowSize), windowSize,
                "windowSize must be at least 1.");
        }

        _lossRiseFraction = lossRisePercent / 100.0;
        _windowSize = windowSize;
        _monitor = monitor;
        _monitorSessionId = monitorSessionId;
        _recentLosses = new Queue<double>(_windowSize + 1);
    }

    /// <inheritdoc/>
    public void OnTrainBegin(TrainingProgress<T> progress)
    {
        AbortReason = null;
        _recentLosses.Clear();
    }

    /// <inheritdoc/>
    public bool OnEpochEnd(TrainingProgress<T> progress)
    {
        double loss = ToDoubleSafe(progress.Loss);

        // (1) Numerical blow-up — the most important, unambiguous signal.
        if (double.IsNaN(loss) || double.IsInfinity(loss))
        {
            return Abort($"loss became {(double.IsNaN(loss) ? "NaN" : "infinite")} at epoch {progress.Epoch}", progress);
        }

        // (2) Divergence — loss rising well above the best value in the recent window.
        if (_lossRiseFraction > 0 && _recentLosses.Count > 0)
        {
            double bestRecent = double.PositiveInfinity;
            foreach (var l in _recentLosses)
            {
                if (l < bestRecent) bestRecent = l;
            }

            // Only meaningful when the baseline is a positive, finite magnitude.
            double baseline = Math.Abs(bestRecent);
            if (!double.IsInfinity(bestRecent) && baseline > 0
                && loss > bestRecent + baseline * _lossRiseFraction)
            {
                return Abort(
                    $"loss diverging at epoch {progress.Epoch}: {loss:E3} rose more than " +
                    $"{_lossRiseFraction * 100:F0}% above recent best {bestRecent:E3}",
                    progress);
            }
        }

        _recentLosses.Enqueue(loss);
        while (_recentLosses.Count > _windowSize)
        {
            _recentLosses.Dequeue();
        }

        // (3) Monitor-reported critical issues (stalls, resource exhaustion, logged errors).
        if (_monitor is not null && _monitorSessionId is not null)
        {
            List<string> issues;
            try
            {
                issues = _monitor.CheckForIssues(_monitorSessionId);
            }
            catch (Exception ex)
            {
                // Fail closed: a health guard that cannot run the monitor check it was configured with must
                // not silently treat the run as healthy. Abort with a clear reason and surface the failure
                // (the NaN/divergence guards above still ran; this only fires when a monitor was supplied).
                Trace.TraceWarning(
                    $"HealthMonitorCallback: ITrainingMonitor.CheckForIssues failed at epoch {progress.Epoch}: {ex}");
                return Abort(
                    $"training monitor health check failed at epoch {progress.Epoch}: {ex.GetType().Name}",
                    progress);
            }

            if (issues.Count > 0)
            {
                return Abort($"training monitor reported issue(s) at epoch {progress.Epoch}: {string.Join("; ", issues)}", progress);
            }
        }

        return true;
    }

    /// <inheritdoc/>
    public void OnTrainEnd(TrainingProgress<T> progress)
    {
        // Nothing to release; AbortReason (if set) remains available for inspection.
    }

    private bool Abort(string reason, TrainingProgress<T> progress)
    {
        AbortReason = reason;

        // Surface the reason through the monitor when we have one, so dashboards and
        // exported logs capture WHY the run stopped — not just that it did.
        if (_monitor is not null && _monitorSessionId is not null)
        {
            try
            {
                _monitor.LogMessage(_monitorSessionId, LogLevel.Error, $"HealthMonitorCallback aborting training: {reason}");
            }
            catch (Exception ex)
            {
                // Logging is best-effort and must never mask the abort decision — but do not hide that it
                // failed. Record it on AbortReason (which callers read) and to the trace listeners so a
                // broken monitor integration is observable rather than silent.
                AbortReason = $"{reason}; NOTE: failed to log abort to monitor ({ex.GetType().Name})";
                Trace.TraceWarning($"HealthMonitorCallback: ITrainingMonitor.LogMessage failed while recording abort: {ex}");
            }
        }

        return false; // request abort
    }

    private static double ToDoubleSafe(T value)
    {
        if (value is null) return double.NaN;
        try
        {
            return Convert.ToDouble(value, System.Globalization.CultureInfo.InvariantCulture);
        }
        catch
        {
            return double.NaN;
        }
    }
}
