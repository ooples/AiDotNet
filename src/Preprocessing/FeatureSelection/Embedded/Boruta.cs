using AiDotNet.Helpers;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

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
public class Boruta<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly int _maxIterations;
    private readonly double _alpha;
    private readonly int? _randomState;
    private readonly Func<Matrix<T>, Vector<T>, double[]>? _importanceFunc;

    // Fitted parameters
    private BorutaDecision[]? _decisions;
    private double[]? _meanImportances;
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
    public double[]? MeanImportances => _meanImportances;

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
        Func<Matrix<T>, Vector<T>, double[]>? importanceFunc = null,
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

        // Convert to double arrays
        var X = new double[n, p];
        var y = new double[n];

        for (int i = 0; i < n; i++)
        {
            y[i] = NumOps.ToDouble(target[i]);
            for (int j = 0; j < p; j++)
            {
                X[i, j] = NumOps.ToDouble(data[i, j]);
            }
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
        var importanceSums = new double[p];

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

            // Create shadow features (shuffled copies)
            var XWithShadow = new double[n, p * 2];

            for (int i = 0; i < n; i++)
            {
                for (int j = 0; j < p; j++)
                {
                    XWithShadow[i, j] = X[i, j];
                }
            }

            // Shuffle for shadow features
            for (int j = 0; j < p; j++)
            {
                var shuffledIndices = Enumerable.Range(0, n).OrderBy(_ => random.Next()).ToArray();
                for (int i = 0; i < n; i++)
                {
                    XWithShadow[i, p + j] = X[shuffledIndices[i], j];
                }
            }

            // Get importances
            var matrixWithShadow = CreateMatrix(XWithShadow, n, p * 2);
            var targetVector = CreateVector(y, n);
            var importances = GetImportances(matrixWithShadow, targetVector, n, p * 2);

            // Find max shadow importance
            double maxShadowImportance = double.NegativeInfinity;
            for (int j = p; j < p * 2; j++)
            {
                if (importances[j] > maxShadowImportance)
                {
                    maxShadowImportance = importances[j];
                }
            }

            // Update hits for tentative features
            for (int j = 0; j < p; j++)
            {
                if (_decisions[j] == BorutaDecision.Tentative)
                {
                    importanceSums[j] += importances[j];

                    if (importances[j] > maxShadowImportance)
                    {
                        hits[j]++;
                    }
                }
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
        _meanImportances = new double[p];
        for (int j = 0; j < p; j++)
        {
            _meanImportances[j] = _iterations > 0 ? importanceSums[j] / _iterations : 0;
        }

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
            for (int j = 0; j < p; j++)
            {
                if (_decisions[j] == BorutaDecision.Tentative && _meanImportances[j] > 0)
                {
                    selectedList.Add(j);
                }
            }
        }

        // If still no features, take the best one
        if (selectedList.Count == 0)
        {
            int bestIdx = 0;
            double bestImportance = _meanImportances[0];
            for (int j = 1; j < p; j++)
            {
                if (_meanImportances[j] > bestImportance)
                {
                    bestImportance = _meanImportances[j];
                    bestIdx = j;
                }
            }
            selectedList.Add(bestIdx);
        }

        _selectedIndices = selectedList.OrderBy(x => x).ToArray();
        IsFitted = true;
    }

    private Matrix<T> CreateMatrix(double[,] X, int rows, int cols)
    {
        var data = new T[rows, cols];
        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < cols; j++)
            {
                data[i, j] = NumOps.FromDouble(X[i, j]);
            }
        }
        return new Matrix<T>(data);
    }

    private Vector<T> CreateVector(double[] y, int n)
    {
        var data = new T[n];
        for (int i = 0; i < n; i++)
        {
            data[i] = NumOps.FromDouble(y[i]);
        }
        return new Vector<T>(data);
    }

    private double[] GetImportances(Matrix<T> data, Vector<T> target, int n, int p)
    {
        if (_importanceFunc is not null)
        {
            return _importanceFunc(data, target);
        }

        // Default: correlation-based importance
        var importances = new double[p];

        for (int j = 0; j < p; j++)
        {
            double xMean = 0, yMean = 0;
            for (int i = 0; i < n; i++)
            {
                xMean += NumOps.ToDouble(data[i, j]);
                yMean += NumOps.ToDouble(target[i]);
            }
            xMean /= n;
            yMean /= n;

            double ssXY = 0, ssXX = 0, ssYY = 0;
            for (int i = 0; i < n; i++)
            {
                double x = NumOps.ToDouble(data[i, j]) - xMean;
                double y = NumOps.ToDouble(target[i]) - yMean;
                ssXY += x * y;
                ssXX += x * x;
                ssYY += y * y;
            }

            if (ssXX > 1e-10 && ssYY > 1e-10)
            {
                importances[j] = Math.Abs(ssXY / Math.Sqrt(ssXX * ssYY));
            }
        }

        return importances;
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

        int numRows = data.Rows;
        int numCols = _selectedIndices.Length;
        var result = new T[numRows, numCols];

        for (int i = 0; i < numRows; i++)
        {
            for (int j = 0; j < numCols; j++)
            {
                result[i, j] = data[i, _selectedIndices[j]];
            }
        }

        return new Matrix<T>(result);
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
