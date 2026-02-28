namespace AiDotNet.FederatedLearning.Compression;

/// <summary>
/// Implements FetchSGD — Count-Sketch + Top-k hybrid compression for massive models.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> FetchSGD combines two compression ideas: count-min sketches
/// (a probabilistic data structure) for aggregation and top-k for decompression. Each client
/// compresses their gradient into a small sketch (fixed-size regardless of model size). The
/// server merges sketches (just element-wise addition) and then recovers the top-k heavy hitters.
/// This is especially efficient for very large models (billions of parameters).</para>
///
/// <para>Algorithm:</para>
/// <code>
/// Client: sketch_k = CountSketch(gradient_k)     // O(sketch_size)
/// Server: merged = sum(sketch_k)                  // element-wise
/// Server: top_k = HeavyHitters(merged)            // recover large values
/// Server: update = top_k                          // sparse update
/// </code>
///
/// <para>Reference: Rothchild, D., et al. (2020). "FetchSGD: Communication-Efficient Federated
/// Learning with Sketching." ICML 2020.</para>
/// </remarks>
/// <typeparam name="T">The numeric type for model parameters.</typeparam>
public class FetchSGDCompressor<T> : Infrastructure.FederatedLearningComponentBase<T>
{
    private readonly int _sketchRows;
    private readonly int _sketchCols;
    private readonly int _topK;
    private readonly int _seed;
    private readonly long[] _hashA; // 2-universal hash: h(x) = (a*x + b) mod p mod m
    private readonly long[] _hashB;
    private readonly long[] _signA; // Separate 2-universal hash for sign
    private readonly long[] _signB;
    private const long LARGE_PRIME = 2147483647L; // 2^31 - 1 (Mersenne prime)
    private double[]? _errorAccumulator;

    /// <summary>
    /// Creates a new FetchSGD compressor.
    /// </summary>
    /// <param name="sketchRows">Number of hash functions (rows). Default: 5.</param>
    /// <param name="sketchCols">Width of each row. Default: 10000.</param>
    /// <param name="topK">Number of heavy hitters to recover. Default: 1000.</param>
    /// <param name="seed">Random seed. Default: 42.</param>
    public FetchSGDCompressor(int sketchRows = 5, int sketchCols = 10000, int topK = 1000, int seed = 42)
    {
        if (sketchRows < 1)
        {
            throw new ArgumentOutOfRangeException(nameof(sketchRows), "Must have at least 1 row.");
        }

        if (sketchCols < 1)
        {
            throw new ArgumentOutOfRangeException(nameof(sketchCols), "Must have at least 1 column.");
        }

        if (topK < 1)
        {
            throw new ArgumentOutOfRangeException(nameof(topK), "Top-k must be at least 1.");
        }

        _sketchRows = sketchRows;
        _sketchCols = sketchCols;
        _topK = topK;
        _seed = seed;

        // Initialize 2-universal hash families: h(x) = ((a*x + b) mod p) mod m
        // where a, b are random, p is a large prime, m is the table size.
        // This guarantees pairwise independence for collision bounds.
        var rng = new Random(seed);
        _hashA = new long[sketchRows];
        _hashB = new long[sketchRows];
        _signA = new long[sketchRows];
        _signB = new long[sketchRows];

        for (int r = 0; r < sketchRows; r++)
        {
            // Use two Next() calls combined for a wider range (net471 lacks NextInt64).
            _hashA[r] = ((long)rng.Next(1, int.MaxValue) << 16) | (long)rng.Next(0, 65536);
            _hashB[r] = ((long)rng.Next(0, int.MaxValue) << 16) | (long)rng.Next(0, 65536);
            _signA[r] = ((long)rng.Next(1, int.MaxValue) << 16) | (long)rng.Next(0, 65536);
            _signB[r] = ((long)rng.Next(0, int.MaxValue) << 16) | (long)rng.Next(0, 65536);
        }
    }

    /// <summary>
    /// Compresses a flattened gradient into a count sketch with error feedback.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Error feedback stores the difference between the original gradient
    /// and what was actually communicated (the sketch lossy reconstruction). This residual is added
    /// to the next round's gradient, ensuring no information is permanently lost. Over many rounds,
    /// all gradient information eventually gets communicated.</para>
    /// </remarks>
    /// <param name="gradient">The gradient values.</param>
    /// <param name="useErrorFeedback">Whether to apply error feedback. Default: true.</param>
    /// <returns>Count sketch matrix (rows x cols).</returns>
    public double[,] Sketch(T[] gradient, bool useErrorFeedback = true)
    {
        Guard.NotNull(gradient);
        var sketch = new double[_sketchRows, _sketchCols];

        // Add error feedback from previous round.
        if (useErrorFeedback && _errorAccumulator != null && _errorAccumulator.Length == gradient.Length)
        {
            for (int i = 0; i < gradient.Length; i++)
            {
                double val = NumOps.ToDouble(gradient[i]) + _errorAccumulator[i];
                for (int r = 0; r < _sketchRows; r++)
                {
                    int col = HashColumn(i, r);
                    int sign = HashSign(i, r);
                    sketch[r, col] += sign * val;
                }
            }
        }
        else
        {
            for (int i = 0; i < gradient.Length; i++)
            {
                double val = NumOps.ToDouble(gradient[i]);
                for (int r = 0; r < _sketchRows; r++)
                {
                    int col = HashColumn(i, r);
                    int sign = HashSign(i, r);
                    sketch[r, col] += sign * val;
                }
            }
        }

        // Update error accumulator if using feedback.
        if (useErrorFeedback)
        {
            _errorAccumulator = new double[gradient.Length];
            for (int i = 0; i < gradient.Length; i++)
            {
                // Error = original gradient - sketch reconstruction.
                // The error from the previous round was already folded into the sketch above,
                // so we only track the new residual for the next round.
                double estimate = EstimateFromSketch(sketch, i);
                _errorAccumulator[i] = NumOps.ToDouble(gradient[i]) - estimate;
            }
        }

        return sketch;
    }

    private int HashColumn(int index, int row)
    {
        long h = ((_hashA[row] * index + _hashB[row]) % LARGE_PRIME + LARGE_PRIME) % LARGE_PRIME;
        return (int)(h % _sketchCols);
    }

    private int HashSign(int index, int row)
    {
        long h = ((_signA[row] * index + _signB[row]) % LARGE_PRIME + LARGE_PRIME) % LARGE_PRIME;
        return (h % 2 == 0) ? 1 : -1;
    }

    private double EstimateFromSketch(double[,] sketch, int index)
    {
        var estimates = new double[_sketchRows];
        for (int r = 0; r < _sketchRows; r++)
        {
            int col = HashColumn(index, r);
            int sign = HashSign(index, r);
            estimates[r] = sign * sketch[r, col];
        }

        Array.Sort(estimates);
        return estimates[_sketchRows / 2]; // Median estimator.
    }

    /// <summary>
    /// Merges multiple sketches by element-wise addition.
    /// </summary>
    public double[,] MergeSketches(IReadOnlyList<double[,]> sketches)
    {
        Guard.NotNull(sketches);
        if (sketches.Count == 0)
        {
            return new double[_sketchRows, _sketchCols];
        }

        var merged = new double[_sketchRows, _sketchCols];
        foreach (var sketch in sketches)
        {
            for (int r = 0; r < _sketchRows; r++)
            {
                for (int c = 0; c < _sketchCols; c++)
                {
                    merged[r, c] += sketch[r, c];
                }
            }
        }

        return merged;
    }

    /// <summary>
    /// Recovers the top-k heavy hitters from a merged sketch.
    /// </summary>
    /// <param name="mergedSketch">Merged sketch.</param>
    /// <param name="gradientLength">Original gradient length.</param>
    /// <returns>Sparse recovery array (non-heavy-hitters are zero).</returns>
    public T[] RecoverTopK(double[,] mergedSketch, int gradientLength)
    {
        // Estimate each coordinate from the sketch using median of estimates.
        var estimates = new (int Index, double AbsValue, double Value)[gradientLength];
        for (int i = 0; i < gradientLength; i++)
        {
            double median = EstimateFromSketch(mergedSketch, i);
            estimates[i] = (i, Math.Abs(median), median);
        }

        // Top-k by magnitude.
        Array.Sort(estimates, (a, b) => b.AbsValue.CompareTo(a.AbsValue));
        var result = new T[gradientLength];
        int effectiveK = Math.Min(_topK, gradientLength);
        for (int i = 0; i < effectiveK; i++)
        {
            result[estimates[i].Index] = NumOps.FromDouble(estimates[i].Value);
        }

        return result;
    }

    /// <summary>Gets the sketch dimensions.</summary>
    public int SketchRows => _sketchRows;

    /// <summary>Gets the sketch width.</summary>
    public int SketchCols => _sketchCols;

    /// <summary>Gets the top-k value.</summary>
    public int TopK => _topK;
}
