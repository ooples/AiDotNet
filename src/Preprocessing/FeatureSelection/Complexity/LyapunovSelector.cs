using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Enums;
using AiDotNet.Attributes;

namespace AiDotNet.Preprocessing.FeatureSelection.Complexity;

/// <summary>
/// Lyapunov Exponent based Feature Selection.
/// </summary>
/// <remarks>
/// <para>
/// Selects features based on their largest Lyapunov exponent, which measures
/// sensitivity to initial conditions (chaos) in dynamical systems.
/// </para>
/// <para><b>For Beginners:</b> The Lyapunov exponent measures how fast nearby
/// trajectories diverge over time. Positive values indicate chaos (butterfly effect),
/// zero indicates marginal stability, negative indicates convergence. Features with
/// positive Lyapunov exponents may contain complex nonlinear dynamics.
/// </para>
/// </remarks>
[ComponentType(ComponentType.FeatureSelector)]
[PipelineStage(PipelineStage.Preprocessing)]
public class LyapunovSelector<T> : TransformerBase<T, Matrix<T>, Matrix<T>>, IFixedCountFeatureSelector<T>
{
    private readonly int _nFeaturesToSelect;
    private readonly int _embeddingDimension;
    private readonly int _delay;

    private Vector<T>? _lyapunovValues;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    /// <inheritdoc />
    public int NFeaturesToSelect => _nFeaturesToSelect;
    public int EmbeddingDimension => _embeddingDimension;
    public int Delay => _delay;
    public Vector<T>? LyapunovValues => _lyapunovValues;
    /// <inheritdoc />
    public int[]? SelectedIndices => _selectedIndices;
    /// <inheritdoc />
    public override bool SupportsInverseTransform => false;

    public LyapunovSelector(
        int nFeaturesToSelect = 10,
        int embeddingDimension = 3,
        int delay = 1,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));
        if (embeddingDimension < 2)
            throw new ArgumentException("Embedding dimension must be at least 2.", nameof(embeddingDimension));
        if (delay < 1)
            throw new ArgumentException("Delay must be at least 1.", nameof(delay));

        _nFeaturesToSelect = nFeaturesToSelect;
        _embeddingDimension = embeddingDimension;
        _delay = delay;
    }

    /// <inheritdoc />
    protected override void FitCore(Matrix<T> data)
    {
        _nInputFeatures = data.Columns;
        int n = data.Rows;
        int p = data.Columns;

        int minLength = (_embeddingDimension - 1) * _delay + 20;
        if (n < minLength)
            throw new ArgumentException($"Need at least {minLength} samples.");

        var engine = Engine;
        var X = Tensor<T>.FromMatrix(data);

        // Half and BFloat16 run in float: a squared coordinate difference past 256 overflows Half, and the inf
        // distance left the reference point untracked (its exponent became 0)
        bool narrow = PreprocessingTensorOps.IsNarrowFloat<T>();
        var perFeature = new Tensor<double>[p];
        for (int j = 0; j < p; j++)
        {
            var column = PreprocessingTensorOps.ColumnVector(X, j);
            perFeature[j] = narrow
                ? EstimateLyapunov(column.Cast<float>(), _embeddingDimension, _delay)
                : EstimateLyapunov(column, _embeddingDimension, _delay);
        }

        var lyapunov = p == 0 ? new Tensor<T>(new[] { 0 }) : engine.TensorConcatenate(perFeature, 0).Cast<T>();
        _lyapunovValues = lyapunov.ToVector();

        int numToSelect = Math.Min(_nFeaturesToSelect, p);
        _selectedIndices = PreprocessingTensorOps.TopKIndicesSorted(lyapunov, numToSelect);

        IsFitted = true;
    }

    // Upper bound on the elements of one block of reference vectors against every vector ([rows, vectors, m]).
    private const int NeighbourBlockElements = 1 << 22;

    /// <summary>
    /// The largest Lyapunov exponent of one series (Rosenstein's algorithm, simplified), as a one-element tensor.
    /// </summary>
    private static Tensor<double> EstimateLyapunov<TC>(Tensor<TC> data, int embeddingDimension, int delay)
    {
        var engine = AiDotNet.Tensors.Engines.AiDotNetEngine.Current;
        var ops = MathHelper.GetNumericOperations<TC>();
        int n = data.Length;
        int numVectors = n - (embeddingDimension - 1) * delay;
        var none = new Tensor<double>(new[] { 1 });
        if (numVectors < 10) return none;

        // Create embedded vectors: row i is (x[i], x[i + delay], ..., x[i + (m - 1) * delay])
        var lagged = new Tensor<TC>[embeddingDimension];
        for (int d = 0; d < embeddingDimension; d++)
            lagged[d] = engine.TensorSlice(data, new[] { d * delay }, new[] { numVectors });
        var vectors = engine.TensorStack(lagged, 1);

        // Rosenstein's algorithm (simplified)
        // For each point, find nearest neighbor (not too close in time, at a positive distance)
        int minTimeSep = embeddingDimension * delay;
        int reference = numVectors - 10;
        if (reference <= 0) return none;

        // Reference vectors are searched in blocks against every vector: all at once is a [reference, vectors]
        // distance matrix (3.2 GB of doubles for 20,000 rows) plus its masks. Each block keeps one minimum
        // distance and the first neighbour index at it per reference vector.
        TC maxValue = ops.MaxValue;
        TC zero = ops.Zero;
        int blockRows = Math.Max(1, NeighbourBlockElements / Math.Max(1, numVectors * embeddingDimension));
        var minDistBlocks = new List<Tensor<TC>>();
        var nearest = new int[reference];
        for (int start = 0; start < reference; start += blockRows)
        {
            int rows = Math.Min(blockRows, reference - start);

            // |k - i| >= minTimeSep for the global row i = start + r, as the union of the bands
            // k - r >= minTimeSep + start and k - r <= start - minTimeSep: built from the positions, not from time
            // indices held in TC (Half cannot hold every index above 2048)
            var shape = new[] { rows, numVectors };
            var ones = PreprocessingTensorOps.Filled(shape, ops.One);
            var separated = minTimeSep <= 0
                ? ones
                : engine.TensorAdd(
                    engine.TensorTriu(ones, minTimeSep + start), engine.TensorTril(ones, start - minTimeSep));
            var referenceVectors = engine.TensorSlice(vectors, new[] { start, 0 }, new[] { rows, embeddingDimension });
            var distances = engine.TensorSqrt(FeatureScoringTensorOps.SquaredDistances(referenceVectors, vectors));
            var allowed = engine.TensorMultiply(separated, engine.TensorGreaterThan(distances, zero));
            var candidates = engine.TensorWhere(allowed, distances, PreprocessingTensorOps.Filled(shape, maxValue));
            minDistBlocks.Add(engine.TensorNegate(engine.ReduceMax(engine.TensorNegate(candidates), new[] { 1 }, false)));
            Array.Copy(FeatureScoringTensorOps.FirstArgMinPerRow(candidates), 0, nearest, start, rows);
        }

        var minDist = minDistBlocks.Count == 1 ? minDistBlocks[0] : engine.TensorConcatenate(minDistBlocks.ToArray(), 0);

        // Points whose neighbour exists and has 5 successors are tracked
        int lastNeighbour = numVectors - 5;
        var tracked = engine.TensorMultiply(
            engine.TensorLessThan(minDist, maxValue),
            new Tensor<int>(Array.ConvertAll(nearest, k => k < lastNeighbour ? 1 : 0), new[] { reference }).Cast<TC>());
        var neighbour = Array.ConvertAll(nearest, k => k < lastNeighbour ? k : 0);

        // Track divergence over time steps: log(dist / minDist) / step
        var zeros = new Tensor<TC>(new[] { reference });
        Tensor<TC>? total = null;
        Tensor<int>? count = null;
        for (int step = 1; step <= 5; step++)
        {
            int offset = step;
            var ahead = engine.TensorSlice(vectors, new[] { step, 0 }, new[] { reference, embeddingDimension });
            var neighbourAhead = engine.TensorIndexSelect(vectors,
                new Tensor<int>(Array.ConvertAll(neighbour, k => k + offset), new[] { reference }), 0);
            var dist = engine.TensorSqrt(engine.ReduceSum(
                engine.TensorSquare(engine.TensorSubtract(ahead, neighbourAhead)), new[] { 1 }, false));
            var used = engine.TensorMultiply(tracked, engine.TensorGreaterThan(dist, zero));
            var terms = engine.TensorWhere(used, engine.TensorDivideScalar(
                engine.TensorLog(engine.TensorDivide(dist, minDist)), PreprocessingTensorOps.CountAsScalar<TC>(step)), zeros);
            total = total is null ? terms : engine.TensorAdd(total, terms);
            count = count is null ? used.Cast<int>() : engine.TensorAdd(count, used.Cast<int>());
        }

        if (total is null || count is null)
            return none;

        // Up to 5 terms per reference point: counted in int and averaged in double (a Half count stops at 2048)
        var divergenceSum = engine.ReduceSum(total.Cast<double>(), new[] { 0 }, true);
        var divergenceCount = engine.ReduceSum(count, new[] { 0 }, true).Cast<double>();
        return engine.TensorWhere(engine.TensorGreaterThan(divergenceCount, 0.0),
            engine.TensorDivide(divergenceSum, divergenceCount), new Tensor<double>(new[] { 1 }));
    }

    public new Matrix<T> FitTransform(Matrix<T> data)
    {
        Fit(data);
        return Transform(data);
    }

    /// <inheritdoc />
    protected override Matrix<T> TransformCore(Matrix<T> data)
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("LyapunovSelector has not been fitted.");

        return PreprocessingTensorOps.SelectColumns(data, _selectedIndices);
    }

    /// <inheritdoc />
    protected override Matrix<T> InverseTransformCore(Matrix<T> data)
    {
        throw new NotSupportedException("LyapunovSelector does not support inverse transformation.");
    }

    /// <inheritdoc />
    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("LyapunovSelector has not been fitted.");

        var mask = new bool[_nInputFeatures];
        foreach (int idx in _selectedIndices)
            mask[idx] = true;

        return mask;
    }

    /// <inheritdoc />
    public override string[] GetFeatureNamesOut(string[]? inputFeatureNames = null)
    {
        if (_selectedIndices is null) return [];

        if (inputFeatureNames is null)
            return _selectedIndices.Select(i => $"Feature{i}").ToArray();

        return _selectedIndices
            .Where(i => i < inputFeatureNames.Length)
            .Select(i => inputFeatureNames[i])
            .ToArray();
    }
}
