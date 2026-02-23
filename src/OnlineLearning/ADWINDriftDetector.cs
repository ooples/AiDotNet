using AiDotNet.Helpers;
using AiDotNet.Interfaces;

namespace AiDotNet.OnlineLearning;

/// <summary>
/// ADWIN (ADaptive WINdowing) drift detector for concept drift detection in data streams.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// ADWIN maintains a variable-length window of recent data and automatically shrinks it
/// when a significant change in the mean is detected. It provides theoretical guarantees
/// on false positive/negative rates.
/// </para>
/// <para>
/// <b>For Beginners:</b> ADWIN is like a smart sliding window that automatically
/// adjusts its size based on whether the data is stable or changing:
///
/// When data is stable:
/// - Window grows to include more history
/// - More data = more accurate estimates
///
/// When drift occurs:
/// - Window shrinks to forget old (now irrelevant) data
/// - Model adapts quickly to new patterns
///
/// How it works:
/// 1. Maintains a window W of recent values
/// 2. For each new value, checks if there's a "cut point" where:
///    - W₁ = data before cut, W₂ = data after cut
///    - If |mean(W₁) - mean(W₂)| > threshold, drift is detected
/// 3. When drift detected, discards W₁ (old data)
///
/// The threshold is based on the Hoeffding bound, providing statistical guarantees:
/// - P(false alarm) bounded by δ
/// - P(missing real drift) bounded by δ
///
/// Key advantage: No need to set window size manually - ADWIN adapts!
///
/// Usage:
/// <code>
/// var detector = new ADWINDriftDetector&lt;double&gt;(delta: 0.002);
/// foreach (var error in modelErrors)
/// {
///     var status = detector.Update(error);
///     if (status == DriftStatus.Drift)
///     {
///         Console.WriteLine("Drift detected! Retrain model.");
///         model.Reset();
///     }
/// }
/// </code>
///
/// References:
/// - Bifet &amp; Gavaldà (2007). "Learning from Time-Changing Data with Adaptive Windowing"
/// </para>
/// </remarks>
public class ADWINDriftDetector<T> : IDriftDetector<T>
{
    /// <summary>
    /// Numeric operations helper for generic math.
    /// </summary>
    private readonly INumericOperations<T> _numOps = MathHelper.GetNumericOperations<T>();

    /// <summary>
    /// The confidence parameter (probability bound for false positives/negatives).
    /// </summary>
    private readonly double _delta;

    /// <summary>
    /// Maximum number of buckets per level.
    /// </summary>
    private readonly int _maxBucketsPerLevel;

    /// <summary>
    /// The bucket list representing the adaptive window.
    /// </summary>
    private readonly List<Bucket> _bucketList;

    /// <summary>
    /// Total sum of values in the window.
    /// </summary>
    private double _total;

    /// <summary>
    /// Total variance (sum of squared deviations) in the window.
    /// </summary>
    private double _variance;

    /// <summary>
    /// Total count of values in the window.
    /// </summary>
    private long _width;

    /// <summary>
    /// Count of total samples processed.
    /// </summary>
    private long _sampleCount;

    /// <summary>
    /// Index of the last detected change point.
    /// </summary>
    private long _changePoint;

    /// <summary>
    /// Whether drift was detected on the last update.
    /// </summary>
    private bool _driftDetected;

    /// <summary>
    /// Gets whether drift has been detected.
    /// </summary>
    public bool IsDriftDetected => _driftDetected;

    /// <summary>
    /// Gets whether a warning has been detected.
    /// ADWIN doesn't have explicit warnings, so this mirrors drift.
    /// </summary>
    public bool IsWarning => _driftDetected;

    /// <summary>
    /// Represents a bucket in the exponential histogram.
    /// </summary>
    private class Bucket
    {
        public double Total { get; set; }
        public double Variance { get; set; }
        public long Count { get; set; }
    }

    /// <summary>
    /// Initializes a new instance of the ADWINDriftDetector class.
    /// </summary>
    /// <param name="delta">Confidence parameter - lower values = higher confidence but slower detection. Default is 0.002.</param>
    /// <param name="maxBucketsPerLevel">Maximum buckets per level in exponential histogram. Default is 5.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Parameters:
    ///
    /// - delta: Controls sensitivity to drift
    ///   - Lower (e.g., 0.001): More confident but may miss subtle drifts
    ///   - Higher (e.g., 0.01): More sensitive but may have false alarms
    ///   - Default (0.002): Good balance for most applications
    ///
    /// - maxBucketsPerLevel: Memory/speed tradeoff
    ///   - Higher: More accurate but uses more memory
    ///   - Default (5): Works well for most cases
    /// </para>
    /// </remarks>
    public ADWINDriftDetector(double delta = 0.002, int maxBucketsPerLevel = 5)
    {
        _delta = delta;
        _maxBucketsPerLevel = maxBucketsPerLevel;
        _bucketList = new List<Bucket>();
        Reset();
    }

    /// <summary>
    /// Updates the detector with a new observation.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> For each new value:
    /// 1. Add it to the window
    /// 2. Check if any "cut point" shows significant mean difference
    /// 3. If yes, declare drift and shrink window
    /// 4. Return current drift status
    /// </para>
    /// </remarks>
    public DriftStatus Update(T value)
    {
        double x = _numOps.ToDouble(value);
        _sampleCount++;
        _driftDetected = false;

        // Add new bucket with single value
        InsertElement(x);

        // Compress buckets if needed
        CompressBuckets();

        // Check for drift using ADWIN cut detection
        _driftDetected = DetectDrift();

        if (_driftDetected)
        {
            _changePoint = _sampleCount;
        }

        return _driftDetected ? DriftStatus.Drift : DriftStatus.NoDrift;
    }

    /// <summary>
    /// Inserts a new element into the bucket list.
    /// </summary>
    private void InsertElement(double x)
    {
        // Create new bucket with single value
        _bucketList.Add(new Bucket
        {
            Total = x,
            Variance = 0,
            Count = 1
        });

        _total += x;
        _width++;

        // Estimate variance incrementally
        if (_width > 1)
        {
            double mean = _total / _width;
            _variance += (x - mean) * (x - mean);
        }
    }

    /// <summary>
    /// Compresses buckets using exponential histogram scheme.
    /// </summary>
    private void CompressBuckets()
    {
        // Group consecutive buckets of same size
        int i = _bucketList.Count - 1;
        while (i >= _maxBucketsPerLevel)
        {
            // Check if we have too many buckets of same size
            var similarBuckets = new List<int>();
            long targetCount = _bucketList[i].Count;

            for (int j = i; j >= 0; j--)
            {
                if (_bucketList[j].Count == targetCount)
                {
                    similarBuckets.Add(j);
                }
                else
                {
                    break;
                }
            }

            if (similarBuckets.Count > _maxBucketsPerLevel)
            {
                // Merge the two oldest buckets of this size
                int idx1 = similarBuckets[^1];
                int idx2 = similarBuckets[^2];

                MergeBuckets(idx1, idx2);
                _bucketList.RemoveAt(idx2);
                i--;
            }
            else
            {
                break;
            }
        }
    }

    /// <summary>
    /// Merges two buckets into the first one.
    /// </summary>
    private void MergeBuckets(int idx1, int idx2)
    {
        var b1 = _bucketList[idx1];
        var b2 = _bucketList[idx2];

        double newTotal = b1.Total + b2.Total;
        long newCount = b1.Count + b2.Count;

        // Combine variances using parallel algorithm
        double mean1 = b1.Count > 0 ? b1.Total / b1.Count : 0;
        double mean2 = b2.Count > 0 ? b2.Total / b2.Count : 0;
        double delta = mean2 - mean1;

        double newVariance = b1.Variance + b2.Variance +
            delta * delta * (b1.Count * b2.Count) / newCount;

        b1.Total = newTotal;
        b1.Variance = newVariance;
        b1.Count = newCount;
    }

    /// <summary>
    /// Detects drift using ADWIN's cut-based test.
    /// </summary>
    private bool DetectDrift()
    {
        if (_width < 2) return false;

        // Try different cut points
        double sumLeft = 0;
        double varianceLeft = 0;
        long countLeft = 0;

        for (int i = 0; i < _bucketList.Count - 1; i++)
        {
            // Add bucket i to left window
            sumLeft += _bucketList[i].Total;
            varianceLeft += _bucketList[i].Variance;
            countLeft += _bucketList[i].Count;

            // Right window is everything else
            double sumRight = _total - sumLeft;
            long countRight = _width - countLeft;

            if (countLeft > 0 && countRight > 0)
            {
                double meanLeft = sumLeft / countLeft;
                double meanRight = sumRight / countRight;
                double absDiff = Math.Abs(meanLeft - meanRight);

                // Compute ADWIN bound using Hoeffding inequality
                double bound = ComputeADWINBound(countLeft, countRight);

                if (absDiff > bound)
                {
                    // Drift detected - remove old (left) buckets
                    for (int j = 0; j <= i; j++)
                    {
                        _total -= _bucketList[0].Total;
                        _variance -= _bucketList[0].Variance;
                        _width -= _bucketList[0].Count;
                        _bucketList.RemoveAt(0);
                    }
                    return true;
                }
            }
        }

        return false;
    }

    /// <summary>
    /// Computes the ADWIN bound for cut detection.
    /// </summary>
    private double ComputeADWINBound(long n1, long n2)
    {
        // Harmonic mean of window sizes
        double m = 2.0 / (1.0 / n1 + 1.0 / n2);

        // Compute epsilon cut using Hoeffding-style bound
        double deltaP = _delta / Math.Log(n1 + n2);
        double epsilon = Math.Sqrt(2.0 * m * Math.Log(2.0 / deltaP)) / (n1 + n2);

        // Add variance term for tighter bound
        if (_width > 1 && _variance > 0)
        {
            double estVariance = _variance / (_width - 1);
            epsilon += Math.Sqrt(2.0 * estVariance * Math.Log(2.0 / deltaP) / m);
        }

        return epsilon;
    }

    /// <summary>
    /// Resets the detector to its initial state.
    /// </summary>
    public void Reset()
    {
        _bucketList.Clear();
        _total = 0;
        _variance = 0;
        _width = 0;
        _sampleCount = 0;
        _changePoint = -1;
        _driftDetected = false;
    }

    /// <summary>
    /// Gets the estimated change point.
    /// </summary>
    public long GetChangePoint() => _changePoint;

    /// <summary>
    /// Gets current detection statistics.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Statistics help you understand the detector's state:
    /// - WindowSize: How much history is being used
    /// - WindowMean: Average value in window
    /// - SampleCount: Total samples processed
    /// - NumBuckets: Internal structure size
    /// </para>
    /// </remarks>
    public Dictionary<string, T> GetStatistics()
    {
        double windowMean = _width > 0 ? _total / _width : 0;
        double windowVariance = _width > 1 ? _variance / (_width - 1) : 0;

        return new Dictionary<string, T>
        {
            { "WindowSize", _numOps.FromDouble(_width) },
            { "WindowMean", _numOps.FromDouble(windowMean) },
            { "WindowVariance", _numOps.FromDouble(windowVariance) },
            { "SampleCount", _numOps.FromDouble(_sampleCount) },
            { "NumBuckets", _numOps.FromDouble(_bucketList.Count) },
            { "ChangePoint", _numOps.FromDouble(_changePoint) }
        };
    }

    /// <summary>
    /// Gets the current window size.
    /// </summary>
    public long GetWindowSize() => _width;

    /// <summary>
    /// Gets the current mean of the window.
    /// </summary>
    public T GetWindowMean()
    {
        return _numOps.FromDouble(_width > 0 ? _total / _width : 0);
    }

    /// <summary>
    /// Gets the estimation of variance in the window.
    /// </summary>
    public T GetWindowVariance()
    {
        return _numOps.FromDouble(_width > 1 ? _variance / (_width - 1) : 0);
    }
}
