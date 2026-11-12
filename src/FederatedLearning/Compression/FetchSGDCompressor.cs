using AiDotNet.Attributes;
using AiDotNet.Enums;

namespace AiDotNet.FederatedLearning.Compression;

/// <summary>
/// Implements FetchSGD — Count-Sketch compression with momentum and error accumulation in sketch space.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> Each client compresses its gradient into a small count sketch (a fixed-size
/// table that is the same size whatever the model size). Because sketches are linear, the server can add them
/// up, and it keeps momentum and the not-yet-applied error <em>as sketches too</em>, so the server never stores
/// a full-size vector. Each round it recovers the k largest coordinates ("heavy hitters") from the error
/// sketch, applies them as a sparse update and subtracts their sketch from the error.</para>
///
/// <para>Algorithm 1 of the paper (per round t, W participating clients):</para>
/// <code>
/// Client i:  S_i = S(g_i)                              // count sketch of the gradient
/// Server:    S   = (1 / W) * sum_i S_i                 // aggregate
///            S_u = rho * S_u + S                       // momentum in sketch space
///            S_e = eta * S_u + S_e                     // error feedback in sketch space
///            Delta = Top-k(U(S_e))                     // unsketch: median estimates, keep top-k
///            S_e = S_e - S(Delta)                      // error accumulation
///            w = w - Delta
/// </code>
///
/// <para>Reference: Rothchild, D., et al. (2020). "FetchSGD: Communication-Efficient Federated
/// Learning with Sketching." ICML 2020.</para>
/// </remarks>
/// <typeparam name="T">The numeric type for model parameters.</typeparam>
[ComponentType(ComponentType.FederatedAggregator)]
[PipelineStage(PipelineStage.Training)]
public class FetchSGDCompressor<T> : Infrastructure.FederatedLearningComponentBase<T>
{
    private readonly int _sketchRows;
    private readonly int _sketchCols;
    private readonly int _topK;
    private readonly int _seed;
    private readonly double _momentum;
    private readonly double _learningRate;
    private readonly long[] _hashA; // 2-universal hash: h(x) = (a*x + b) mod p mod m
    private readonly long[] _hashB;
    private readonly long[] _signA; // Separate 2-universal hash for sign
    private readonly long[] _signB;
    private const long LARGE_PRIME = 2147483647L; // 2^31 - 1 (Mersenne prime)

    // Hash tables for the last gradient length: flat sketch index r * cols + h_r(i) and sign s_r(i),
    // both laid out [rows, length].
    private int _hashedLength = -1;
    private Tensor<int> _flatIndices = new Tensor<int>(new[] { 0 });
    private Tensor<T> _signs = new Tensor<T>(new[] { 0 });

    // Server state (sketch space), [rows, cols].
    private Tensor<T>? _momentumSketch;
    private Tensor<T>? _errorSketch;

    /// <summary>
    /// Creates a new FetchSGD compressor.
    /// </summary>
    /// <param name="sketchRows">Number of hash functions (rows). Default: 5.</param>
    /// <param name="sketchCols">Width of each row. Default: 10000.</param>
    /// <param name="topK">Number of heavy hitters to recover. Default: 1000.</param>
    /// <param name="seed">Random seed. Default: 42.</param>
    /// <param name="momentum">Server momentum rho applied in sketch space. Default: 0.9 (the paper's value).</param>
    /// <param name="learningRate">Server learning rate eta folded into the error sketch, so the returned
    /// update Delta already includes it. Default: 1.0.</param>
    public FetchSGDCompressor(
        int sketchRows = 5,
        int sketchCols = 10000,
        int topK = 1000,
        int seed = 42,
        double momentum = 0.9,
        double learningRate = 1.0)
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

        if (momentum < 0 || momentum >= 1)
        {
            throw new ArgumentOutOfRangeException(nameof(momentum), "Momentum must be in [0, 1).");
        }

        if (learningRate <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(learningRate), "Learning rate must be positive.");
        }

        _sketchRows = sketchRows;
        _sketchCols = sketchCols;
        _topK = topK;
        _seed = seed;
        _momentum = momentum;
        _learningRate = learningRate;

        // Initialize 2-universal hash families: h(x) = ((a*x + b) mod p) mod m
        // where a, b are random, p is a large prime, m is the table size.
        // This guarantees pairwise independence for collision bounds.
        var rng = RandomHelper.CreateSeededRandom(seed);
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
    /// Client step: compresses a flattened gradient into a count sketch S(g).
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Every coordinate i is added, with a pseudo-random sign s_r(i), into one
    /// bucket h_r(i) of every row r. The sketch is linear, so sketches from many clients can simply be
    /// added. FetchSGD keeps no error feedback on the client; momentum and error live on the server, in
    /// sketch space (see <see cref="ServerStep"/>).</para>
    /// </remarks>
    /// <param name="gradient">The gradient values.</param>
    /// <returns>Count sketch matrix (rows x cols).</returns>
    public Matrix<T> Sketch(T[] gradient)
    {
        Guard.NotNull(gradient);
        return SketchTensor(new Tensor<T>(gradient, new[] { gradient.Length })).ToMatrix();
    }

    /// <summary>
    /// Aggregates client sketches: S = (1 / W) * sum_i S_i.
    /// </summary>
    /// <param name="sketches">The client sketches.</param>
    /// <returns>The average sketch (all zeros when no sketches are given).</returns>
    public Matrix<T> MergeSketches(IReadOnlyList<Matrix<T>> sketches)
    {
        Guard.NotNull(sketches);
        var merged = new Tensor<T>(new[] { _sketchRows, _sketchCols });
        if (sketches.Count == 0)
        {
            return merged.ToMatrix();
        }

        foreach (var sketch in sketches)
        {
            Guard.NotNull(sketch);
            if (sketch.Rows != _sketchRows || sketch.Columns != _sketchCols)
            {
                throw new ArgumentException(
                    $"Sketch shape {sketch.Rows}x{sketch.Columns} differs from {_sketchRows}x{_sketchCols}.",
                    nameof(sketches));
            }

            merged = Engine.TensorAdd(merged, Tensor<T>.FromMatrix(sketch));
        }

        return Engine.TensorDivideScalar(merged, NumOps.FromDouble(sketches.Count)).ToMatrix();
    }

    /// <summary>
    /// Server step: momentum and error feedback in sketch space, top-k unsketch and error accumulation.
    /// </summary>
    /// <param name="aggregatedSketch">The aggregated client sketch S for this round
    /// (from <see cref="MergeSketches"/>).</param>
    /// <param name="gradientLength">Number of model parameters.</param>
    /// <returns>The sparse update Delta (dense array, zero outside the top-k coordinates); apply it as
    /// w = w - Delta.</returns>
    public T[] ServerStep(Matrix<T> aggregatedSketch, int gradientLength)
    {
        var sketch = ValidateSketch(aggregatedSketch, gradientLength);
        var momentumSketch = _momentumSketch ?? new Tensor<T>(new[] { _sketchRows, _sketchCols });
        var errorSketch = _errorSketch ?? new Tensor<T>(new[] { _sketchRows, _sketchCols });

        // S_u = rho * S_u + S;  S_e = eta * S_u + S_e.
        momentumSketch = Engine.TensorAdd(Engine.TensorMultiplyScalar(momentumSketch, NumOps.FromDouble(_momentum)), sketch);
        errorSketch = Engine.TensorAdd(Engine.TensorMultiplyScalar(momentumSketch, NumOps.FromDouble(_learningRate)), errorSketch);

        // Delta = Top-k(U(S_e));  S_e = S_e - S(Delta).
        var delta = UnsketchTopK(errorSketch, gradientLength);
        errorSketch = Engine.TensorSubtract(errorSketch, SketchTensor(delta));

        _momentumSketch = momentumSketch;
        _errorSketch = errorSketch;
        return delta.ToArray();
    }

    /// <summary>
    /// Clears the server's momentum and error sketches.
    /// </summary>
    public void ResetServerState()
    {
        _momentumSketch = null;
        _errorSketch = null;
    }

    /// <summary>
    /// Recovers the top-k heavy hitters from a sketch without touching the server state.
    /// </summary>
    /// <param name="mergedSketch">Merged sketch.</param>
    /// <param name="gradientLength">Original gradient length.</param>
    /// <returns>Sparse recovery array (non-heavy-hitters are zero).</returns>
    public T[] RecoverTopK(Matrix<T> mergedSketch, int gradientLength)
    {
        return UnsketchTopK(ValidateSketch(mergedSketch, gradientLength), gradientLength).ToArray();
    }

    private Tensor<T> ValidateSketch(Matrix<T> sketch, int gradientLength)
    {
        Guard.NotNull(sketch);
        if (sketch.Rows != _sketchRows || sketch.Columns != _sketchCols)
        {
            throw new ArgumentException(
                $"Sketch shape {sketch.Rows}x{sketch.Columns} differs from {_sketchRows}x{_sketchCols}.",
                nameof(sketch));
        }

        if (gradientLength < 0)
        {
            throw new ArgumentOutOfRangeException(nameof(gradientLength), "Gradient length must be non-negative.");
        }

        return Tensor<T>.FromMatrix(sketch);
    }

    /// <summary>Count sketch of a length-n vector as a [rows, cols] tensor.</summary>
    private Tensor<T> SketchTensor(Tensor<T> values)
    {
        int length = values.Length;
        var sketch = new Tensor<T>(new[] { _sketchRows * _sketchCols });
        if (length > 0)
        {
            EnsureHashTables(length);

            // Every row adds s_r(i) * g_i into bucket h_r(i); one index-add covers all rows.
            var signed = Engine.TensorMultiply(_signs, values.Reshape(new[] { 1, length }));
            sketch = Engine.TensorIndexAdd(
                sketch, 0, _flatIndices.Reshape(new[] { _sketchRows * length }), signed.Reshape(new[] { _sketchRows * length }));
        }

        return sketch.Reshape(new[] { _sketchRows, _sketchCols });
    }

    /// <summary>U(S) followed by Top-k: median-of-rows estimates, keeping the k largest magnitudes.</summary>
    private Tensor<T> UnsketchTopK(Tensor<T> sketch, int length)
    {
        if (length == 0)
        {
            return new Tensor<T>(new[] { 0 });
        }

        EnsureHashTables(length);

        // Estimates per row: s_r(i) * S[r, h_r(i)]; the estimate is the median over rows
        // (the upper middle value for an even row count).
        var bucketValues = Engine.TensorGather(
            sketch.Reshape(new[] { _sketchRows * _sketchCols }), _flatIndices.Reshape(new[] { _sketchRows * length }), 0);
        var rowEstimates = Engine.TensorMultiply(_signs, bucketValues.Reshape(new[] { _sketchRows, length }));
        var (sortedEstimates, _) = Engine.TensorSort(rowEstimates, 0, false);
        var estimates = Engine.TensorSlice(sortedEstimates, new[] { _sketchRows / 2, 0 }, new[] { 1, length })
            .Reshape(new[] { length });

        // Keep the k largest magnitudes: everything above the k-th largest, then ties in index order.
        int k = Math.Min(_topK, length);
        var magnitudes = Engine.TensorAbs(estimates);
        var (sortedMagnitudes, _) = Engine.TensorSort(magnitudes, 0, false);
        T threshold = sortedMagnitudes[length - k];
        var above = Engine.TensorGreaterThan(magnitudes, threshold).Cast<int>();
        int tieSlots = k - Engine.TensorCountNonzero(above);
        var ties = Engine.TensorEquals(magnitudes, threshold).Cast<int>();
        var keptTies = Engine.TensorMultiply(ties, Engine.TensorLessThan(Engine.TensorCumSum(ties, 0), tieSlots + 1));
        var keep = Engine.TensorAdd(above, keptTies).Cast<T>();

        return Engine.TensorWhere(keep, estimates, new Tensor<T>(new[] { length }));
    }

    private void EnsureHashTables(int length)
    {
        if (_hashedLength == length)
        {
            return;
        }

        var flatIndices = new int[_sketchRows * length];
        var signs = new int[_sketchRows * length];
        for (int r = 0; r < _sketchRows; r++)
        {
            for (int i = 0; i < length; i++)
            {
                flatIndices[r * length + i] = r * _sketchCols + HashColumn(i, r);
                signs[r * length + i] = HashSign(i, r);
            }
        }

        _flatIndices = new Tensor<int>(flatIndices, new[] { _sketchRows, length });
        _signs = new Tensor<int>(signs, new[] { _sketchRows, length }).Cast<T>();
        _hashedLength = length;
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

    /// <summary>Gets the sketch dimensions.</summary>
    public int SketchRows => _sketchRows;

    /// <summary>Gets the sketch width.</summary>
    public int SketchCols => _sketchCols;

    /// <summary>Gets the top-k value.</summary>
    public int TopK => _topK;

    /// <summary>Gets the server momentum rho.</summary>
    public double Momentum => _momentum;

    /// <summary>Gets the server learning rate eta.</summary>
    public double LearningRate => _learningRate;
}