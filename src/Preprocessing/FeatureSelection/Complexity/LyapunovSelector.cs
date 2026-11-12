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

        var perFeature = new Tensor<T>[p];
        for (int j = 0; j < p; j++)
            perFeature[j] = EstimateLyapunov(PreprocessingTensorOps.ColumnVector(X, j));

        var lyapunov = p == 0 ? new Tensor<T>(new[] { 0 }) : engine.TensorConcatenate(perFeature, 0);
        _lyapunovValues = lyapunov.ToVector();

        int numToSelect = Math.Min(_nFeaturesToSelect, p);
        _selectedIndices = PreprocessingTensorOps.TopKIndicesSorted(lyapunov, numToSelect);

        IsFitted = true;
    }

    private Tensor<T> EstimateLyapunov(Tensor<T> data)
    {
        var engine = Engine;
        int n = data.Length;
        int numVectors = n - (_embeddingDimension - 1) * _delay;
        var none = new Tensor<T>(new[] { 1 });
        if (numVectors < 10) return none;

        // Create embedded vectors: row i is (x[i], x[i + delay], ..., x[i + (m - 1) * delay])
        var lagged = new Tensor<T>[_embeddingDimension];
        for (int d = 0; d < _embeddingDimension; d++)
            lagged[d] = engine.TensorSlice(data, new[] { d * _delay }, new[] { numVectors });
        var vectors = engine.TensorStack(lagged, 1);

        // Rosenstein's algorithm (simplified)
        // For each point, find nearest neighbor (not too close in time, at a positive distance)
        int minTimeSep = _embeddingDimension * _delay;
        int reference = numVectors - 10;
        if (reference <= 0) return none;

        var shape = new[] { reference, numVectors };
        var timeGap = engine.TensorAbs(engine.TensorSubtract(
            engine.TensorBroadcastTo(engine.Reshape(PreprocessingTensorOps.Range<T>(numVectors), new[] { 1, numVectors }), shape),
            engine.TensorBroadcastTo(engine.Reshape(PreprocessingTensorOps.Range<T>(reference), new[] { reference, 1 }), shape)));
        var separated = PreprocessingTensorOps.AtLeast(timeGap, NumOps.FromDouble(minTimeSep));
        var referenceVectors = engine.TensorSlice(vectors, new[] { 0, 0 }, new[] { reference, _embeddingDimension });
        var distances = engine.TensorSqrt(FeatureSelectionTensorOps.SquaredDistances(referenceVectors, vectors));
        T maxValue = NumOps.MaxValue;
        T zero = NumOps.Zero;
        var allowed = engine.TensorMultiply(separated, engine.TensorGreaterThan(distances, zero));
        var candidates = engine.TensorWhere(allowed, distances, PreprocessingTensorOps.Filled(shape, maxValue));
        var minDist = engine.TensorNegate(engine.ReduceMax(engine.TensorNegate(candidates), new[] { 1 }, false));
        var nearest = FeatureSelectionTensorOps.FirstArgMinPerRow(candidates);

        // Points whose neighbour exists and has 5 successors are tracked
        int lastNeighbour = numVectors - 5;
        var tracked = engine.TensorMultiply(
            engine.TensorLessThan(minDist, maxValue),
            new Tensor<int>(Array.ConvertAll(nearest, k => k < lastNeighbour ? 1 : 0), new[] { reference }).Cast<T>());
        var neighbour = Array.ConvertAll(nearest, k => k < lastNeighbour ? k : 0);

        // Track divergence over time steps: log(dist / minDist) / step
        var zeros = new Tensor<T>(new[] { reference });
        Tensor<T>? total = null;
        Tensor<T>? count = null;
        for (int step = 1; step <= 5; step++)
        {
            int offset = step;
            var ahead = engine.TensorSlice(vectors, new[] { step, 0 }, new[] { reference, _embeddingDimension });
            var neighbourAhead = engine.TensorIndexSelect(vectors,
                new Tensor<int>(Array.ConvertAll(neighbour, k => k + offset), new[] { reference }), 0);
            var dist = engine.TensorSqrt(engine.ReduceSum(
                engine.TensorSquare(engine.TensorSubtract(ahead, neighbourAhead)), new[] { 1 }, false));
            var used = engine.TensorMultiply(tracked, engine.TensorGreaterThan(dist, zero));
            var terms = engine.TensorWhere(used, engine.TensorDivideScalar(
                engine.TensorLog(engine.TensorDivide(dist, minDist)), PreprocessingTensorOps.CountAsScalar<T>(step)), zeros);
            total = total is null ? terms : engine.TensorAdd(total, terms);
            count = count is null ? used : engine.TensorAdd(count, used);
        }

        if (total is null || count is null)
            return none;

        var divergenceSum = engine.ReduceSum(total, new[] { 0 }, true);
        var divergenceCount = engine.ReduceSum(count, new[] { 0 }, true);
        return engine.TensorWhere(engine.TensorGreaterThan(divergenceCount, zero),
            engine.TensorDivide(divergenceSum, divergenceCount), none);
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
