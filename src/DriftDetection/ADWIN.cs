namespace AiDotNet.DriftDetection;

/// <summary>
/// Implements ADWIN (ADaptive WINdowing) for concept drift detection.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> ADWIN is a popular drift detector that maintains a variable-size
/// window of recent observations. It automatically adjusts window size based on detected changes,
/// shrinking when drift occurs and growing when the data is stable.</para>
///
/// <para><b>How ADWIN works:</b>
/// <list type="number">
/// <item>Maintains a sliding window of observations with efficient compression</item>
/// <item>For each new observation, checks if the window can be split into two sub-windows
///   with significantly different means</item>
/// <item>If a significant difference is found, drops the older portion (drift detected)</item>
/// <item>Uses the Hoeffding bound to determine statistical significance</item>
/// </list>
/// </para>
///
/// <para><b>Key Features:</b>
/// <list type="bullet">
/// <item>Adapts window size automatically - no manual tuning needed</item>
/// <item>Provides theoretical guarantees on false positive/negative rates</item>
/// <item>Efficient O(log W) memory and time per observation (W = window size)</item>
/// </list>
/// </para>
///
/// <para><b>The delta parameter:</b> Controls sensitivity to drift. Smaller values mean:
/// <list type="bullet">
/// <item>More sensitive to small changes</item>
/// <item>Fewer false negatives (missed drifts)</item>
/// <item>More false positives (spurious drift detections)</item>
/// </list>
/// Typical values: 0.001 (sensitive) to 0.2 (conservative). Default: 0.002.
/// </para>
///
/// <para><b>Reference:</b> Bifet & Gavaldà, "Learning from Time-Changing Data with Adaptive Windowing" (2007)</para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class ADWIN<T> : DriftDetectorBase<T>
{
    private readonly double _delta;
    private readonly List<Bucket> _buckets;
    private int _windowSize;
    private double _total;
    private double _variance;
    private readonly int _maxBuckets;

    /// <summary>
    /// Represents a bucket in ADWIN's exponential histogram structure.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> ADWIN uses buckets to efficiently compress the observation
    /// history. Each bucket stores the sum and variance of multiple consecutive observations,
    /// allowing O(log W) memory instead of O(W).</para>
    /// </remarks>
    private class Bucket
    {
        public int Count { get; set; }
        public double Total { get; set; }
        public double Variance { get; set; }
    }

    /// <summary>
    /// Gets the delta parameter (confidence level for drift detection).
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Delta controls the trade-off between detecting real drifts
    /// and avoiding false alarms. Smaller delta = more sensitive but more false alarms.</para>
    /// </remarks>
    public double Delta => _delta;

    /// <summary>
    /// Gets the current window size (number of observations in memory).
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> This grows when data is stable and shrinks when drift
    /// is detected. A larger window provides better estimates but slower reaction to drift.</para>
    /// </remarks>
    public int WindowSize => _windowSize;

    /// <summary>
    /// Creates a new ADWIN drift detector.
    /// </summary>
    /// <param name="delta">Confidence parameter for drift detection (default: 0.002).</param>
    /// <param name="maxBuckets">Maximum number of buckets per level (default: 5).</param>
    /// <remarks>
    /// <para><b>For Beginners:</b> Use default parameters for most cases. Decrease delta
    /// if you want faster drift detection (more sensitive). Increase if you get too many
    /// false alarms.</para>
    /// </remarks>
    public ADWIN(double delta = 0.002, int maxBuckets = 5)
    {
        if (delta <= 0 || delta >= 1)
        {
            throw new ArgumentOutOfRangeException(nameof(delta), "Delta must be in range (0, 1).");
        }

        if (maxBuckets < 2)
        {
            throw new ArgumentOutOfRangeException(nameof(maxBuckets), "Max buckets must be at least 2.");
        }

        _delta = delta;
        _maxBuckets = maxBuckets;
        _buckets = new List<Bucket>();
        Reset();
    }

    /// <inheritdoc />
    public override void Reset()
    {
        base.Reset();
        _buckets.Clear();
        _windowSize = 0;
        _total = 0;
        _variance = 0;
    }

    /// <summary>
    /// Adds a new observation and checks for drift.
    /// </summary>
    /// <param name="value">The new observation (often a 0/1 error indicator).</param>
    /// <returns>True if drift was detected.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Feed this method with your model's predictions or errors.
    /// For classification, pass 1 for errors and 0 for correct predictions. ADWIN will
    /// detect when the error rate changes significantly.</para>
    /// </remarks>
    public override bool AddObservation(T value)
    {
        double val = ToDouble(value);
        ObservationCount++;

        // Add new observation
        InsertElement(val);

        // Check for drift by trying to cut the window
        bool driftDetected = CheckDrift();

        if (driftDetected)
        {
            IsInDrift = true;
            DriftProbability = 1.0;
        }
        else
        {
            IsInDrift = false;
            // Estimate drift probability based on how close we are to the threshold
            DriftProbability = EstimateDriftProbability();
        }

        // Update estimated mean
        EstimatedMean = _windowSize > 0 ? _total / _windowSize : 0;

        return driftDetected;
    }

    /// <summary>
    /// Inserts a new element into the bucket structure.
    /// </summary>
    /// <param name="value">The value to insert.</param>
    /// <remarks>
    /// <para><b>For Beginners:</b> This adds the value to the most recent bucket and
    /// compresses old buckets to maintain efficient storage.</para>
    /// </remarks>
    private void InsertElement(double value)
    {
        _windowSize++;
        _total += value;

        // Create a new bucket for this element
        var newBucket = new Bucket
        {
            Count = 1,
            Total = value,
            Variance = 0
        };

        // Insert at the beginning (most recent)
        _buckets.Insert(0, newBucket);

        // Compress buckets if needed
        CompressBuckets();
    }

    /// <summary>
    /// Compresses buckets when there are too many at a given level.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> This merging ensures memory stays at O(log W) by
    /// combining older buckets into larger ones. This is the key to ADWIN's efficiency.</para>
    /// </remarks>
    private void CompressBuckets()
    {
        int i = 0;
        int sameCountRun = 0;

        while (i < _buckets.Count)
        {
            // Count consecutive buckets of the same size
            int currentCount = _buckets[i].Count;
            sameCountRun = 0;

            while (i + sameCountRun < _buckets.Count &&
                   _buckets[i + sameCountRun].Count == currentCount)
            {
                sameCountRun++;
            }

            // If too many buckets of the same size, merge the oldest two
            if (sameCountRun > _maxBuckets)
            {
                int mergeIdx = i + sameCountRun - 2;
                MergeBuckets(mergeIdx, mergeIdx + 1);
                // Continue checking from the merged bucket
            }
            else
            {
                i += sameCountRun;
            }
        }
    }

    /// <summary>
    /// Merges two adjacent buckets.
    /// </summary>
    /// <param name="idx1">Index of first bucket.</param>
    /// <param name="idx2">Index of second bucket.</param>
    /// <remarks>
    /// <para><b>For Beginners:</b> When merging, we combine the counts and totals, and
    /// use Welford's algorithm to correctly combine variances.</para>
    /// </remarks>
    private void MergeBuckets(int idx1, int idx2)
    {
        if (idx1 >= _buckets.Count || idx2 >= _buckets.Count)
            return;

        var b1 = _buckets[idx1];
        var b2 = _buckets[idx2];

        // Combined statistics using Welford's method
        int newCount = b1.Count + b2.Count;
        double newTotal = b1.Total + b2.Total;

        double mean1 = b1.Count > 0 ? b1.Total / b1.Count : 0;
        double mean2 = b2.Count > 0 ? b2.Total / b2.Count : 0;
        double newMean = newCount > 0 ? newTotal / newCount : 0;

        double newVariance = b1.Variance + b2.Variance +
                            b1.Count * Math.Pow(mean1 - newMean, 2) +
                            b2.Count * Math.Pow(mean2 - newMean, 2);

        b1.Count = newCount;
        b1.Total = newTotal;
        b1.Variance = newVariance;

        _buckets.RemoveAt(idx2);
    }

    /// <summary>
    /// Checks if drift has occurred by finding a significant cut point.
    /// </summary>
    /// <returns>True if drift was detected.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This is ADWIN's core algorithm. It tries different
    /// ways to split the window into two parts and checks if their means are significantly
    /// different using the Hoeffding bound.</para>
    /// </remarks>
    private bool CheckDrift()
    {
        bool driftDetected = false;

        // Try cutting the window at different points (using bucket boundaries)
        double leftTotal = 0;
        double leftVariance = 0;
        int leftCount = 0;

        // Start from the most recent bucket
        for (int i = 0; i < _buckets.Count - 1; i++)
        {
            var bucket = _buckets[i];
            leftTotal += bucket.Total;
            leftVariance += bucket.Variance;
            leftCount += bucket.Count;

            double rightTotal = _total - leftTotal;
            int rightCount = _windowSize - leftCount;

            if (leftCount < MinimumObservations || rightCount < MinimumObservations)
                continue;

            double leftMean = leftTotal / leftCount;
            double rightMean = rightTotal / rightCount;

            // Hoeffding bound
            double epsilon = CalculateEpsilon(leftCount, rightCount);

            if (Math.Abs(leftMean - rightMean) >= epsilon)
            {
                // Drift detected - remove older portion
                driftDetected = true;

                // Remove buckets from index i+1 to end (older buckets)
                int removeCount = _buckets.Count - i - 1;
                for (int j = 0; j < removeCount; j++)
                {
                    int removeIdx = _buckets.Count - 1;
                    var removeBucket = _buckets[removeIdx];
                    _total -= removeBucket.Total;
                    _variance -= removeBucket.Variance;
                    _windowSize -= removeBucket.Count;
                    _buckets.RemoveAt(removeIdx);
                }

                break;
            }
        }

        return driftDetected;
    }

    /// <summary>
    /// Calculates the epsilon threshold using the Hoeffding bound.
    /// </summary>
    /// <param name="n1">Size of first window.</param>
    /// <param name="n2">Size of second window.</param>
    /// <returns>The epsilon threshold.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> The Hoeffding bound tells us how different two sample means
    /// need to be before we can confidently say they come from different distributions.
    /// This is the mathematical foundation of ADWIN's reliability.</para>
    /// </remarks>
    private double CalculateEpsilon(int n1, int n2)
    {
        double n = n1 + n2;
        double m = 1.0 / (1.0 / n1 + 1.0 / n2);

        // Hoeffding bound with delta correction
        double deltaPrime = _delta / Math.Log(n);
        double epsilon = Math.Sqrt(2 * m * Math.Log(2 / deltaPrime)) / n;

        return epsilon;
    }

    /// <summary>
    /// Estimates the probability of drift based on current window statistics.
    /// </summary>
    /// <returns>Estimated drift probability (0 to 1).</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This provides a continuous measure of drift likelihood,
    /// useful for monitoring even when the binary drift detection hasn't triggered.</para>
    /// </remarks>
    private double EstimateDriftProbability()
    {
        if (_buckets.Count < 2)
            return 0;

        double maxRatio = 0;

        double leftTotal = 0;
        int leftCount = 0;

        for (int i = 0; i < _buckets.Count - 1; i++)
        {
            var bucket = _buckets[i];
            leftTotal += bucket.Total;
            leftCount += bucket.Count;

            double rightTotal = _total - leftTotal;
            int rightCount = _windowSize - leftCount;

            if (leftCount < 5 || rightCount < 5)
                continue;

            double leftMean = leftTotal / leftCount;
            double rightMean = rightTotal / rightCount;
            double epsilon = CalculateEpsilon(leftCount, rightCount);

            if (epsilon > 0)
            {
                double ratio = Math.Abs(leftMean - rightMean) / epsilon;
                maxRatio = Math.Max(maxRatio, ratio);
            }
        }

        // Convert ratio to probability (approaches 1 as ratio approaches 1)
        return Math.Min(maxRatio, 1.0);
    }
}
