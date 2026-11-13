using AiDotNet.Helpers;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Enums;
using AiDotNet.Attributes;

namespace AiDotNet.Preprocessing.FeatureSelection.Bioinformatics;

/// <summary>
/// Boruta feature selection algorithm.
/// </summary>
/// <remarks>
/// <para>
/// Boruta creates "shadow features" (shuffled copies of real features) and compares
/// feature importance against these shadows. Features consistently outperforming
/// the best shadow are confirmed as important.
/// </para>
/// <para><b>For Beginners:</b> Boruta asks: "Is this feature better than random noise?"
/// It shuffles your features to create meaningless versions (shadows), then compares
/// real features against these shadows. Features that consistently beat their shadows
/// are truly informative.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
[ComponentType(ComponentType.FeatureSelector)]
[PipelineStage(PipelineStage.Preprocessing)]
public class Boruta<T> : TransformerBase<T, Matrix<T>, Matrix<T>>, ISupervisedFeatureSelector<T>
{
    private readonly int _maxIterations;
    private readonly double _pValue;
    private readonly int? _randomState;
    private readonly Func<Matrix<T>, Vector<T>, Vector<T>>? _importanceFunction;

    private Vector<T>? _importances;
    private BorutaDecision[]? _decisions;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    public enum BorutaDecision
    {
        Confirmed,
        Rejected,
        Tentative
    }

    public int MaxIterations => _maxIterations;
    public Vector<T>? Importances => _importances;
    public BorutaDecision[]? Decisions => _decisions;
    /// <inheritdoc />
    public int[]? SelectedIndices => _selectedIndices;
    /// <inheritdoc />
    public override bool SupportsInverseTransform => false;

    public Boruta(
        int maxIterations = 100,
        double pValue = 0.05,
        Func<Matrix<T>, Vector<T>, Vector<T>>? importanceFunction = null,
        int? randomState = null,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (maxIterations < 10)
            throw new ArgumentException("Max iterations must be at least 10.", nameof(maxIterations));
        if (pValue <= 0 || pValue >= 1)
            throw new ArgumentException("P-value must be between 0 and 1.", nameof(pValue));

        _maxIterations = maxIterations;
        _pValue = pValue;
        _importanceFunction = importanceFunction;
        _randomState = randomState;
    }

    /// <inheritdoc />
    protected override void FitCore(Matrix<T> data)
    {
        throw new InvalidOperationException(
            "Boruta requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
    }

    /// <inheritdoc />
    public void Fit(Matrix<T> data, Vector<T> target)
    {
        if (data.Rows != target.Length)
            throw new ArgumentException("Target length must match rows in data.");

        _nInputFeatures = data.Columns;
        int n = data.Rows;
        int p = data.Columns;

        var random = _randomState.HasValue
            ? RandomHelper.CreateSeededRandom(_randomState.Value)
            : RandomHelper.CreateSecureRandom();

        var engine = Engine;
        var X = Tensor<T>.FromMatrix(data);

        // Track hits (times feature beats max shadow) and summed importances
        // Hits are counted in int and importances summed in double: over more than 2048 iterations a running
        // total in a narrow T such as Half stops growing (2048 + 1 rounds back to 2048)
        var hitCounts = new Tensor<int>(new[] { p });
        var importanceSum = new Tensor<double>(new[] { p });
        var shadowOrder = new int[n * p];

        for (int iter = 0; iter < _maxIterations; iter++)
        {
            // Create shadow features: column j shuffled by its own random permutation
            for (int j = 0; j < p; j++)
            {
                var indices = Enumerable.Range(0, n).OrderBy(_ => random.Next()).ToArray();
                for (int i = 0; i < n; i++)
                    shadowOrder[i * p + j] = indices[i];
            }

            var shadows = engine.TensorTakeAlongDim(X, new Tensor<int>((int[])shadowOrder.Clone(), new[] { n, p }), 0);
            var extendedMatrix = engine.TensorConcatenate(new[] { X, shadows }, 1).ToMatrix();

            // Get importances
            var allImportances = Tensor<T>.FromVector(GetImportances(extendedMatrix, target));
            var real = engine.TensorSlice(allImportances, new[] { 0 }, new[] { p });
            var maxShadow = engine.TensorMaxValue(engine.TensorSlice(allImportances, new[] { p }, new[] { p }));

            // Count hits
            hitCounts = engine.TensorAdd(hitCounts, engine.TensorGreaterThan(real, maxShadow).Cast<int>());
            importanceSum = engine.TensorAdd(importanceSum, real.Cast<double>());
        }

        // Average importances
        var importances = engine.TensorDivideScalar(importanceSum, (double)_maxIterations).Cast<T>();
        _importances = importances.ToVector();
        var hits = hitCounts.AsSpan().ToArray();

        // Make decisions using binomial test
        _decisions = new BorutaDecision[p];
        double threshold = BinomialThreshold(_maxIterations, _pValue);

        for (int j = 0; j < p; j++)
        {
            if (hits[j] >= threshold)
                _decisions[j] = BorutaDecision.Confirmed;
            else if (hits[j] <= _maxIterations - threshold)
                _decisions[j] = BorutaDecision.Rejected;
            else
                _decisions[j] = BorutaDecision.Tentative;
        }

        // Select confirmed features (and tentative if no confirmed)
        var selected = new List<int>();
        for (int j = 0; j < p; j++)
        {
            if (_decisions[j] == BorutaDecision.Confirmed)
                selected.Add(j);
        }

        if (selected.Count == 0)
        {
            // Include tentative features
            for (int j = 0; j < p; j++)
            {
                if (_decisions[j] == BorutaDecision.Tentative)
                    selected.Add(j);
            }
        }

        if (selected.Count == 0)
        {
            // Fall back to best features by importance
            selected = PreprocessingTensorOps.TopKIndicesSorted(importances, Math.Max(1, p / 10)).ToList();
        }

        _selectedIndices = selected.OrderBy(x => x).ToArray();
        IsFitted = true;
    }

    private Vector<T> GetImportances(Matrix<T> data, Vector<T> target)
    {
        if (_importanceFunction is not null)
            return _importanceFunction(data, target);

        return DefaultImportances(data, target);
    }

    private Vector<T> DefaultImportances(Matrix<T> data, Vector<T> target)
    {
        // Correlation-based importance
        return Engine.TensorAbs(
            PreprocessingTensorOps.TargetCorrelations(Tensor<T>.FromMatrix(data), Tensor<T>.FromVector(target))).ToVector();
    }

    private double BinomialThreshold(int n, double alpha)
    {
        // Approximate using normal approximation to binomial
        // Under null, hits ~ Binomial(n, 0.5)
        double mean = n * 0.5;
        double std = Math.Sqrt(n * 0.25);
        double z = NormalQuantile(1 - alpha);
        return mean + z * std;
    }

    private double NormalQuantile(double p)
    {
        // Approximate inverse normal CDF
        if (p <= 0) return double.NegativeInfinity;
        if (p >= 1) return double.PositiveInfinity;

        double t = Math.Sqrt(-2 * Math.Log(1 - p));
        double c0 = 2.515517;
        double c1 = 0.802853;
        double c2 = 0.010328;
        double d1 = 1.432788;
        double d2 = 0.189269;
        double d3 = 0.001308;

        return t - (c0 + c1 * t + c2 * t * t) / (1 + d1 * t + d2 * t * t + d3 * t * t * t);
    }

    /// <inheritdoc />
    public Matrix<T> FitTransform(Matrix<T> data, Vector<T> target)
    {
        Fit(data, target);
        return Transform(data);
    }

    /// <inheritdoc />
    protected override Matrix<T> TransformCore(Matrix<T> data)
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("Boruta has not been fitted.");

        return PreprocessingTensorOps.SelectColumns(data, _selectedIndices);
    }

    /// <inheritdoc />
    protected override Matrix<T> InverseTransformCore(Matrix<T> data)
    {
        throw new NotSupportedException("Boruta does not support inverse transformation.");
    }

    /// <inheritdoc />
    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("Boruta has not been fitted.");

        var mask = new bool[_nInputFeatures];
        foreach (int idx in _selectedIndices)
            mask[idx] = true;

        return mask;
    }

    /// <inheritdoc />
    public override string[] GetFeatureNamesOut(string[]? inputFeatureNames = null)
    {
        if (_selectedIndices is null) return Array.Empty<string>();

        if (inputFeatureNames is null)
            return _selectedIndices.Select(i => $"Feature{i}").ToArray();

        return _selectedIndices
            .Where(i => i < inputFeatureNames.Length)
            .Select(i => inputFeatureNames[i])
            .ToArray();
    }
}
