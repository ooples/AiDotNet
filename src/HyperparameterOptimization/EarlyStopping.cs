using AiDotNet.Helpers;

namespace AiDotNet.HyperparameterOptimization;

/// <summary>
/// Provides early stopping functionality for hyperparameter optimization and training.
/// </summary>
/// <remarks>
/// <b>For Beginners:</b> Early stopping is a technique to:
/// - Stop training when performance stops improving
/// - Prevent overfitting by not training too long
/// - Save compute resources by terminating hopeless trials
///
/// Key concepts:
/// - Patience: How many checks without improvement before stopping
/// - Min Delta: Minimum improvement to count as "better"
/// - Best Value: The best score seen so far
/// - Counter: Tracks consecutive non-improvements
///
/// Early stopping is essential for efficient hyperparameter search because
/// it allows quick termination of poor configurations.
/// </remarks>
/// <typeparam name="T">The numeric data type used for calculations.</typeparam>
public class EarlyStopping<T>
{
    private readonly INumericOperations<T> _numOps;
    private readonly int _patience;
    private readonly double _minDelta;
    private readonly bool _maximize;
    private readonly EarlyStoppingMode _mode;

    private double _bestValue;
    private int _counter;
    private int _bestEpoch;
    private bool _stopped;
    private readonly List<double> _history;

    /// <summary>
    /// Gets whether early stopping has been triggered.
    /// </summary>
    public bool ShouldStop => _stopped;

    /// <summary>
    /// Gets the best value observed.
    /// </summary>
    public double BestValue => _bestValue;

    /// <summary>
    /// Gets the epoch at which the best value was observed.
    /// </summary>
    public int BestEpoch => _bestEpoch;

    /// <summary>
    /// Gets the number of epochs since the best value was observed.
    /// </summary>
    public int EpochsSinceBest => _counter;

    /// <summary>
    /// Gets the history of values that were checked.
    /// </summary>
    public IReadOnlyList<double> History => _history;

    /// <summary>
    /// Initializes a new instance of the EarlyStopping class.
    /// </summary>
    /// <param name="patience">Number of checks without improvement before stopping.</param>
    /// <param name="minDelta">Minimum improvement to qualify as improvement.</param>
    /// <param name="maximize">Whether higher values are better (true) or lower (false).</param>
    /// <param name="mode">Mode for determining improvement.</param>
    public EarlyStopping(
        int patience = 10,
        double minDelta = 0.0,
        bool maximize = true,
        EarlyStoppingMode mode = EarlyStoppingMode.Best)
    {
        if (patience < 1)
            throw new ArgumentException("Patience must be at least 1.", nameof(patience));
        if (minDelta < 0)
            throw new ArgumentException("minDelta must be non-negative.", nameof(minDelta));

        _numOps = MathHelper.GetNumericOperations<T>();
        _patience = patience;
        _minDelta = minDelta;
        _maximize = maximize;
        _mode = mode;

        _bestValue = _maximize ? double.NegativeInfinity : double.PositiveInfinity;
        _counter = 0;
        _bestEpoch = 0;
        _stopped = false;
        _history = new List<double>();
    }

    /// <summary>
    /// Checks if training should stop based on the new value.
    /// </summary>
    /// <param name="value">The new metric value to check.</param>
    /// <param name="epoch">The current epoch/step number.</param>
    /// <returns>True if training should stop, false otherwise.</returns>
    public bool Check(T value, int epoch = -1)
    {
        double doubleValue = _numOps.ToDouble(value);
        return Check(doubleValue, epoch);
    }

    /// <summary>
    /// Checks if training should stop based on the new value.
    /// </summary>
    /// <param name="value">The new metric value to check.</param>
    /// <param name="epoch">The current epoch/step number.</param>
    /// <returns>True if training should stop, false otherwise.</returns>
    public bool Check(double value, int epoch = -1)
    {
        if (epoch < 0)
            epoch = _history.Count;

        _history.Add(value);

        bool improved = IsImprovement(value);

        if (improved)
        {
            _bestValue = value;
            _counter = 0;
            _bestEpoch = epoch;
        }
        else
        {
            _counter++;

            if (_counter >= _patience)
            {
                _stopped = true;
            }
        }

        return _stopped;
    }

    /// <summary>
    /// Determines if the new value is an improvement over the best.
    /// </summary>
    private bool IsImprovement(double value)
    {
        return _mode switch
        {
            EarlyStoppingMode.Best => IsImprovementBest(value),
            EarlyStoppingMode.RelativeBest => IsImprovementRelative(value),
            EarlyStoppingMode.MovingAverage => IsImprovementMovingAverage(value),
            _ => IsImprovementBest(value)
        };
    }

    private bool IsImprovementBest(double value)
    {
        if (_maximize)
        {
            return value > _bestValue + _minDelta;
        }
        else
        {
            return value < _bestValue - _minDelta;
        }
    }

    private bool IsImprovementRelative(double value)
    {
        if (double.IsInfinity(_bestValue))
            return true;

        // Use absolute value of bestValue for relative delta to handle negative values correctly.
        // Without Math.Abs, negative bestValue * (1 + delta) moves threshold in the wrong direction.
        // Example: bestValue=-10, delta=0.1, maximize=true:
        //   Wrong: threshold = -10 * 1.1 = -11 (more negative, accepts worse values)
        //   Correct: threshold = -10 + |-10| * 0.1 = -9 (requires actual improvement)
        double absDelta = Math.Abs(_bestValue) * _minDelta;
        double threshold = _maximize
            ? _bestValue + absDelta
            : _bestValue - absDelta;

        return _maximize ? value > threshold : value < threshold;
    }

    private bool IsImprovementMovingAverage(double value)
    {
        if (_history.Count < 2)
            return true;

        int windowSize = Math.Min(_patience, _history.Count - 1);
        double recentAvg = _history.Skip(_history.Count - windowSize - 1).Take(windowSize).Average();

        if (_maximize)
        {
            return value > recentAvg + _minDelta;
        }
        else
        {
            return value < recentAvg - _minDelta;
        }
    }

    /// <summary>
    /// Resets the early stopping state.
    /// </summary>
    public void Reset()
    {
        _bestValue = _maximize ? double.NegativeInfinity : double.PositiveInfinity;
        _counter = 0;
        _bestEpoch = 0;
        _stopped = false;
        _history.Clear();
    }

    /// <summary>
    /// Gets a summary of the early stopping state.
    /// </summary>
    public EarlyStoppingState GetState()
    {
        return new EarlyStoppingState(
            _stopped,
            _bestValue,
            _bestEpoch,
            _counter,
            _patience,
            _history.Count
        );
    }
}

/// <summary>
/// Mode for determining improvement in early stopping.
/// </summary>
public enum EarlyStoppingMode
{
    /// <summary>
    /// Compare against the best value seen so far.
    /// </summary>
    Best,

    /// <summary>
    /// Compare using relative improvement (percentage-based).
    /// </summary>
    RelativeBest,

    /// <summary>
    /// Compare against a moving average of recent values.
    /// </summary>
    MovingAverage
}

/// <summary>
/// Represents the current state of early stopping.
/// </summary>
public class EarlyStoppingState
{
    /// <summary>
    /// Whether early stopping has been triggered.
    /// </summary>
    public bool Stopped { get; }

    /// <summary>
    /// The best value observed.
    /// </summary>
    public double BestValue { get; }

    /// <summary>
    /// The epoch at which the best value was observed.
    /// </summary>
    public int BestEpoch { get; }

    /// <summary>
    /// Number of epochs since the best value was observed.
    /// </summary>
    public int EpochsSinceBest { get; }

    /// <summary>
    /// The patience value configured.
    /// </summary>
    public int Patience { get; }

    /// <summary>
    /// Total number of checks performed.
    /// </summary>
    public int TotalChecks { get; }

    /// <summary>
    /// Initializes a new EarlyStoppingState.
    /// </summary>
    public EarlyStoppingState(bool stopped, double bestValue, int bestEpoch,
        int epochsSinceBest, int patience, int totalChecks)
    {
        Stopped = stopped;
        BestValue = bestValue;
        BestEpoch = bestEpoch;
        EpochsSinceBest = epochsSinceBest;
        Patience = patience;
        TotalChecks = totalChecks;
    }

    /// <inheritdoc />
    public override string ToString()
    {
        string status = Stopped ? "STOPPED" : "RUNNING";
        return $"[{status}] Best={BestValue:F4} at epoch {BestEpoch}, {EpochsSinceBest}/{Patience} patience used";
    }
}

/// <summary>
/// Builder for configuring early stopping with fluent API.
/// </summary>
/// <typeparam name="T">The numeric data type used for calculations.</typeparam>
public class EarlyStoppingBuilder<T>
{
    private int _patience = 10;
    private double _minDelta = 0.0;
    private bool _maximize = true;
    private EarlyStoppingMode _mode = EarlyStoppingMode.Best;

    /// <summary>
    /// Sets the patience (number of non-improving checks before stopping).
    /// </summary>
    public EarlyStoppingBuilder<T> WithPatience(int patience)
    {
        if (patience < 1)
            throw new ArgumentException("Patience must be at least 1.", nameof(patience));
        _patience = patience;
        return this;
    }

    /// <summary>
    /// Sets the minimum improvement delta.
    /// </summary>
    public EarlyStoppingBuilder<T> WithMinDelta(double minDelta)
    {
        if (minDelta < 0)
            throw new ArgumentException("minDelta must be non-negative.", nameof(minDelta));
        _minDelta = minDelta;
        return this;
    }

    /// <summary>
    /// Configures for maximization (higher is better).
    /// </summary>
    public EarlyStoppingBuilder<T> Maximize()
    {
        _maximize = true;
        return this;
    }

    /// <summary>
    /// Configures for minimization (lower is better).
    /// </summary>
    public EarlyStoppingBuilder<T> Minimize()
    {
        _maximize = false;
        return this;
    }

    /// <summary>
    /// Sets the improvement mode.
    /// </summary>
    public EarlyStoppingBuilder<T> WithMode(EarlyStoppingMode mode)
    {
        _mode = mode;
        return this;
    }

    /// <summary>
    /// Builds the EarlyStopping instance.
    /// </summary>
    public EarlyStopping<T> Build()
    {
        return new EarlyStopping<T>(_patience, _minDelta, _maximize, _mode);
    }

    /// <summary>
    /// Creates a new builder for configuring early stopping.
    /// </summary>
    public static EarlyStoppingBuilder<T> Create()
    {
        return new EarlyStoppingBuilder<T>();
    }
}
