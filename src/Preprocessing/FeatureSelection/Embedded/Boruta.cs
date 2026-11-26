using AiDotNet.Helpers;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Enums;
using AiDotNet.Attributes;

namespace AiDotNet.Preprocessing.FeatureSelection.Embedded;

/// <summary>
/// Boruta feature selection algorithm.
/// </summary>
/// <remarks>
/// <para>
/// Boruta is an all-relevant feature selection method that uses shadow features
/// (randomized copies of original features) as a benchmark for importance.
/// </para>
/// <para>
/// The algorithm:
/// 1. Create shadow features by shuffling original features
/// 2. Train a model on [original + shadow] features
/// 3. Compare each feature's importance to the max shadow importance
/// 4. Features consistently beating shadows are "confirmed"
/// 5. Features consistently losing to shadows are "rejected"
/// 6. Repeat until all features are decided or max iterations reached
/// </para>
/// <para><b>For Beginners:</b> Boruta asks: "Is this feature better than random noise?"
///
/// It creates "shadow" features by shuffling your real features randomly.
/// If your original feature has higher importance than these random shadows,
/// it's probably useful. If it's worse than random noise, it's rejected.
///
/// Unlike SelectKBest which picks the "top K", Boruta finds ALL features
/// that are genuinely useful, which could be 3 or 30 depending on your data.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations (e.g., float, double).</typeparam>
[ComponentType(ComponentType.FeatureSelector)]
[PipelineStage(PipelineStage.Preprocessing)]
public class Boruta<T> : TransformerBase<T, Matrix<T>, Matrix<T>>, ISupervisedFeatureSelector<T>
{
    private readonly int _maxIterations;
    private readonly double _alpha;
    private readonly int? _randomState;
    private readonly Func<Matrix<T>, Vector<T>, Vector<T>>? _importanceFunc;

    // Fitted parameters
    private BorutaDecision[]? _decisions;
    private Vector<T>? _meanImportances;
    private int[]? _selectedIndices;
    private int _nInputFeatures;
    private int _iterations;

    /// <summary>
    /// Gets the decisions for each feature.
    /// </summary>
    public BorutaDecision[]? Decisions => _decisions;

    /// <summary>
    /// Gets the mean importance scores for each feature.
    /// </summary>
    public Vector<T>? MeanImportances => _meanImportances;

    /// <summary>
    /// Gets the indices of confirmed features.
    /// </summary>
    public int[]? SelectedIndices => _selectedIndices;

    /// <summary>
    /// Gets the number of iterations performed.
    /// </summary>
    public int Iterations => _iterations;

    /// <summary>
    /// Gets whether this transformer supports inverse transformation.
    /// </summary>
    public override bool SupportsInverseTransform => false;

    /// <summary>
    /// Creates a new instance of <see cref="Boruta{T}"/>.
    /// </summary>
    /// <param name="importanceFunc">Function to compute feature importances.</param>
    /// <param name="maxIterations">Maximum number of iterations. Defaults to 100.</param>
    /// <param name="alpha">Significance level for statistical test. Defaults to 0.05.</param>
    /// <param name="randomState">Random seed for reproducibility.</param>
    /// <param name="columnIndices">The column indices to evaluate, or null for all columns.</param>
    public Boruta(
        Func<Matrix<T>, Vector<T>, Vector<T>>? importanceFunc = null,
        int maxIterations = 100,
        double alpha = 0.05,
        int? randomState = null,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (maxIterations < 1)
        {
            throw new ArgumentException("Maximum iterations must be at least 1.", nameof(maxIterations));
        }

        if (alpha <= 0 || alpha >= 1)
        {
            throw new ArgumentException("Alpha must be between 0 and 1 exclusive.", nameof(alpha));
        }

        _importanceFunc = importanceFunc;
        _maxIterations = maxIterations;
        _alpha = alpha;
        _randomState = randomState;
    }

    /// <summary>
    /// Fits the selector (requires target via specialized Fit method).
    /// </summary>
    protected override void FitCore(Matrix<T> data)
    {
        throw new InvalidOperationException(
            "Boruta requires target values for fitting. Use Fit(Matrix<T> data, Vector<T> target) instead.");
    }

    /// <summary>
    /// Fits Boruta by comparing feature importances to shadow features.
    /// </summary>
    /// <param name="data">The feature matrix.</param>
    /// <param name="target">The target values.</param>
    public void Fit(Matrix<T> data, Vector<T> target)
    {
        if (data.Rows != target.Length)
        {
            throw new ArgumentException("Target length must match the number of rows in data.");
        }

        _nInputFeatures = data.Columns;
        int n = data.Rows;
        int p = data.Columns;

        var engine = Engine;
        var X = Tensor<T>.FromMatrix(data);
        var columns = new Tensor<T>[p];
        for (int j = 0; j < p; j++)
        {
            columns[j] = PreprocessingTensorOps.ColumnVector(X, j);
        }

        var random = _randomState.HasValue
            ? RandomHelper.CreateSeededRandom(_randomState.Value)
            : RandomHelper.CreateSecureRandom();

        // Initialize
        _decisions = new BorutaDecision[p];
        for (int j = 0; j < p; j++)
        {
            _decisions[j] = BorutaDecision.Tentative;
        }

        var hits = new int[p];  // Times feature beats max shadow
        var importanceSums = new Tensor<T>(new[] { p });

        // Indicator lookup: decision index 0 = decided, 1 = tentative
        var indicator = new Tensor<T>(new[] { NumOps.Zero, NumOps.One }, new[] { 2 });

        // Iterate
        for (_iterations = 0; _iterations < _maxIterations; _iterations++)
        {
            // Check if all features are decided
            bool allDecided = true;
            for (int j = 0; j < p; j++)
            {
                if (_decisions[j] == BorutaDecision.Tentative)
                {
                    allDecided = false;
                    break;
                }
            }

            if (allDecided) break;

            // Create shadow features (shuffled copies), drawing one permutation per feature in order
            var shadows = new Tensor<T>[p];
            for (int j = 0; j < p; j++)
            {
                var shuffledIndices = Enumerable.Range(0, n).OrderBy(_ => random.Next()).ToArray();
                shadows[j] = engine.Reshape(
                    engine.TensorIndexSelect(columns[j], new Tensor<int>(shuffledIndices, new[] { n }), 0), new[] { n, 1 });
            }

            var withShadow = engine.TensorConcatenate(new[] { X, engine.TensorConcatenate(shadows, 1) }, 1);

            // Get importances
            var importances = GetImportances(withShadow, target);
            var original = engine.TensorSlice(importances, new[] { 0 }, new[] { p });

            // Find max shadow importance
            T maxShadowImportance = engine.TensorMaxValue(engine.TensorSlice(importances, new[] { p }, new[] { p }));

            // Update hits and importance sums of tentative features
            var tentative = engine.TensorIndexSelect(indicator, new Tensor<int>(
                Array.ConvertAll(_decisions, d => d == BorutaDecision.Tentative ? 1 : 0), new[] { p }), 0);
            importanceSums = engine.TensorAdd(importanceSums, engine.TensorMultiply(original, tentative));
            foreach (int j in PreprocessingTensorOps.MaskPositions(
                engine.TensorMultiply(engine.TensorGreaterThan(original, maxShadowImportance), tentative)))
            {
                hits[j]++;
            }

            // Statistical test (binomial test approximation)
            int currentIter = _iterations + 1;
            double expectedHits = currentIter * 0.5;  // Under null: 50% chance of beating shadow
            double stdDev = Math.Sqrt(currentIter * 0.25);

            for (int j = 0; j < p; j++)
            {
                if (_decisions[j] == BorutaDecision.Tentative)
                {
                    // Z-score for binomial test
                    double z = (hits[j] - expectedHits) / stdDev;
                    double pValueHigh = 1 - NormalCDF(z);  // P(hitting more than observed)
                    double pValueLow = NormalCDF(z);  // P(hitting less than observed)

                    if (pValueHigh < _alpha)
                    {
                        _decisions[j] = BorutaDecision.Confirmed;
                    }
                    else if (pValueLow < _alpha)
                    {
                        _decisions[j] = BorutaDecision.Rejected;
                    }
                }
            }
        }

        // Calculate mean importances
        var meanImportances = _iterations > 0
            ? engine.TensorDivideScalar(importanceSums, NumOps.FromDouble(_iterations))
            : new Tensor<T>(new[] { p });
        _meanImportances = meanImportances.ToVector();

        // Select confirmed features
        var selectedList = new List<int>();
        for (int j = 0; j < p; j++)
        {
            if (_decisions[j] == BorutaDecision.Confirmed)
            {
                selectedList.Add(j);
            }
        }

        // If no features confirmed, take tentative ones with positive mean importance
        if (selectedList.Count == 0)
        {
            foreach (int j in PreprocessingTensorOps.MaskPositions(engine.TensorGreaterThan(meanImportances, NumOps.Zero)))
            {
                if (_decisions[j] == BorutaDecision.Tentative)
                {
                    selectedList.Add(j);
                }
            }
        }

        // If still no features, take the best one
        if (selectedList.Count == 0)
        {
            selectedList.Add(PreprocessingTensorOps.FirstArgMax(meanImportances));
        }

        _selectedIndices = selectedList.OrderBy(x => x).ToArray();
        IsFitted = true;
    }

    private Tensor<T> GetImportances(Tensor<T> data, Vector<T> target)
    {
        if (_importanceFunc is not null)
        {
            return Tensor<T>.FromVector(_importanceFunc(data.ToMatrix(), target));
        }

        // Default: absolute Pearson correlation, 0 when either centred sum of squares is not above 1e-10
        return Engine.TensorAbs(PreprocessingTensorOps.TargetCorrelations(data, Tensor<T>.FromVector(target)));
    }
    private double NormalCDF(double z)
    {
        // Approximation of standard normal CDF
        return 0.5 * (1 + Erf(z / Math.Sqrt(2)));
    }

    private double Erf(double x)
    {
        // Abramowitz and Stegun approximation
        double t = 1.0 / (1.0 + 0.5 * Math.Abs(x));
        double tau = t * Math.Exp(-x * x - 1.26551223 +
            t * (1.00002368 +
            t * (0.37409196 +
            t * (0.09678418 +
            t * (-0.18628806 +
            t * (0.27886807 +
            t * (-1.13520398 +
            t * (1.48851587 +
            t * (-0.82215223 +
            t * 0.17087277)))))))));

        return x >= 0 ? 1 - tau : tau - 1;
    }

    /// <summary>
    /// Fits and transforms the data.
    /// </summary>
    public Matrix<T> FitTransform(Matrix<T> data, Vector<T> target)
    {
        Fit(data, target);
        return Transform(data);
    }

    /// <summary>
    /// Transforms the data by selecting confirmed features.
    /// </summary>
    protected override Matrix<T> TransformCore(Matrix<T> data)
    {
        if (_selectedIndices is null)
        {
            throw new InvalidOperationException("Boruta has not been fitted.");
        }

        return PreprocessingTensorOps.SelectColumns(data, _selectedIndices);
    }

    /// <summary>
    /// Inverse transformation is not supported.
    /// </summary>
    protected override Matrix<T> InverseTransformCore(Matrix<T> data)
    {
        throw new NotSupportedException("Boruta does not support inverse transformation.");
    }

    /// <summary>
    /// Gets a boolean mask indicating which features are confirmed.
    /// </summary>
    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
        {
            throw new InvalidOperationException("Boruta has not been fitted.");
        }

        var mask = new bool[_nInputFeatures];
        foreach (int idx in _selectedIndices)
        {
            mask[idx] = true;
        }

        return mask;
    }

    /// <summary>
    /// Gets the output feature names after transformation.
    /// </summary>
    public override string[] GetFeatureNamesOut(string[]? inputFeatureNames = null)
    {
        if (_selectedIndices is null)
        {
            return Array.Empty<string>();
        }

        if (inputFeatureNames is null)
        {
            return _selectedIndices.Select(i => $"Feature{i}").ToArray();
        }

        return _selectedIndices
            .Where(i => i < inputFeatureNames.Length)
            .Select(i => inputFeatureNames[i])
            .ToArray();
    }
}

/// <summary>
/// Boruta decision for a feature.
/// </summary>
public enum BorutaDecision
{
    /// <summary>
    /// Feature is still being evaluated.
    /// </summary>
    Tentative,

    /// <summary>
    /// Feature is confirmed as important.
    /// </summary>
    Confirmed,

    /// <summary>
    /// Feature is rejected as unimportant.
    /// </summary>
    Rejected
}
