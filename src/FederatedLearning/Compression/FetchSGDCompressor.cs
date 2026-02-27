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
    }

    /// <summary>
    /// Compresses a flattened gradient into a count sketch.
    /// </summary>
    /// <param name="gradient">The gradient values.</param>
    /// <returns>Count sketch matrix (rows x cols).</returns>
    public double[,] Sketch(T[] gradient)
    {
        var sketch = new double[_sketchRows, _sketchCols];

        for (int i = 0; i < gradient.Length; i++)
        {
            double val = NumOps.ToDouble(gradient[i]);
            for (int r = 0; r < _sketchRows; r++)
            {
                int col = Math.Abs((i * (_seed + r * 1000003) + 997) % _sketchCols);
                int sign = ((i * (_seed + r * 2000003)) % 2 == 0) ? 1 : -1;
                sketch[r, col] += sign * val;
            }
        }

        return sketch;
    }

    /// <summary>
    /// Merges multiple sketches by element-wise addition.
    /// </summary>
    public double[,] MergeSketches(IReadOnlyList<double[,]> sketches)
    {
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
            var rowEstimates = new double[_sketchRows];
            for (int r = 0; r < _sketchRows; r++)
            {
                int col = Math.Abs((i * (_seed + r * 1000003) + 997) % _sketchCols);
                int sign = ((i * (_seed + r * 2000003)) % 2 == 0) ? 1 : -1;
                rowEstimates[r] = sign * mergedSketch[r, col];
            }

            Array.Sort(rowEstimates);
            double median = rowEstimates[_sketchRows / 2];
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
