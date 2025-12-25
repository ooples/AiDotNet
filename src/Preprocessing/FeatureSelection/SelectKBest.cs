using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.FeatureSelection;

/// <summary>
/// Selects the K highest-scoring features according to a scoring function.
/// </summary>
/// <remarks>
/// <para>
/// SelectKBest computes a score for each feature based on the relationship between
/// the feature and the target variable, then selects the top K features with the
/// highest scores.
/// </para>
/// <para>
/// Built-in scoring functions include:
/// - F-score for regression (linear relationship)
/// - Mutual information (any relationship type)
/// </para>
/// <para><b>For Beginners:</b> Not all features are equally useful for prediction.
/// SelectKBest helps you:
/// - Reduce the number of features to improve model speed
/// - Remove noisy features that might hurt model accuracy
/// - Find the most informative features for understanding your problem
///
/// Example: From 100 features, select the 10 most related to your target.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations (e.g., float, double).</typeparam>
public class SelectKBest<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly int _k;
    private readonly SelectKBestScoreFunc _scoreFunc;

    // Fitted parameters
    private double[]? _scores;
    private double[]? _pvalues;
    private int[]? _selectedFeatures;
    private int _nInputFeatures;

    /// <summary>
    /// Gets the number of features to select.
    /// </summary>
    public int K => _k;

    /// <summary>
    /// Gets the scoring function used.
    /// </summary>
    public SelectKBestScoreFunc ScoreFunc => _scoreFunc;

    /// <summary>
    /// Gets the computed scores for each feature.
    /// </summary>
    public double[]? Scores => _scores;

    /// <summary>
    /// Gets the p-values for each feature (if applicable).
    /// </summary>
    public double[]? PValues => _pvalues;

    /// <summary>
    /// Gets the indices of selected features.
    /// </summary>
    public int[]? SelectedFeatures => _selectedFeatures;

    /// <summary>
    /// Gets whether this transformer supports inverse transformation.
    /// </summary>
    public override bool SupportsInverseTransform => false;

    /// <summary>
    /// Creates a new instance of <see cref="SelectKBest{T}"/>.
    /// </summary>
    /// <param name="k">Number of features to select. Defaults to 10.</param>
    /// <param name="scoreFunc">Scoring function to use. Defaults to FRegression.</param>
    /// <param name="columnIndices">The column indices to consider, or null for all columns.</param>
    public SelectKBest(
        int k = 10,
        SelectKBestScoreFunc scoreFunc = SelectKBestScoreFunc.FRegression,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (k < 1)
        {
            throw new ArgumentException("K must be at least 1.", nameof(k));
        }

        _k = k;
        _scoreFunc = scoreFunc;
    }

    /// <summary>
    /// Fits the selector by computing scores (requires target via specialized Fit method).
    /// </summary>
    protected override void FitCore(Matrix<T> data)
    {
        throw new InvalidOperationException(
            "SelectKBest requires target values for fitting. Use Fit(Matrix<T> data, Vector<T> target) instead.");
    }

    /// <summary>
    /// Fits the selector by computing feature scores based on the target.
    /// </summary>
    /// <param name="data">The feature matrix to fit.</param>
    /// <param name="target">The target values.</param>
    public void Fit(Matrix<T> data, Vector<T> target)
    {
        if (data.Rows != target.Length)
        {
            throw new ArgumentException("Target length must match the number of rows in data.");
        }

        _nInputFeatures = data.Columns;
        var columnsToProcess = GetColumnsToProcess(_nInputFeatures);

        _scores = new double[_nInputFeatures];
        _pvalues = new double[_nInputFeatures];

        // Compute scores for each feature
        for (int col = 0; col < _nInputFeatures; col++)
        {
            if (!columnsToProcess.Contains(col))
            {
                _scores[col] = double.NegativeInfinity;
                _pvalues[col] = 1.0;
                continue;
            }

            var (score, pvalue) = ComputeScore(data, target, col);
            _scores[col] = score;
            _pvalues[col] = pvalue;
        }

        // Select top K features
        int effectiveK = Math.Min(_k, columnsToProcess.Length);

        _selectedFeatures = columnsToProcess
            .OrderByDescending(col => _scores[col])
            .Take(effectiveK)
            .OrderBy(col => col) // Preserve original order
            .ToArray();

        IsFitted = true;
    }

    /// <summary>
    /// Fits the selector and transforms the data in one step.
    /// </summary>
    public Matrix<T> FitTransform(Matrix<T> data, Vector<T> target)
    {
        Fit(data, target);
        return Transform(data);
    }

    private (double Score, double PValue) ComputeScore(Matrix<T> data, Vector<T> target, int col)
    {
        switch (_scoreFunc)
        {
            case SelectKBestScoreFunc.FRegression:
                return ComputeFRegression(data, target, col);
            case SelectKBestScoreFunc.MutualInfoRegression:
                return (ComputeMutualInfo(data, target, col), 0);
            case SelectKBestScoreFunc.FClassif:
                return ComputeFClassif(data, target, col);
            case SelectKBestScoreFunc.Chi2:
                return ComputeChi2(data, target, col);
            default:
                return ComputeFRegression(data, target, col);
        }
    }

    private (double Score, double PValue) ComputeFRegression(Matrix<T> data, Vector<T> target, int col)
    {
        // Compute F-score for linear correlation with target
        int n = data.Rows;

        // Calculate means
        double meanX = 0, meanY = 0;
        for (int i = 0; i < n; i++)
        {
            meanX += NumOps.ToDouble(data[i, col]);
            meanY += NumOps.ToDouble(target[i]);
        }
        meanX /= n;
        meanY /= n;

        // Calculate correlation coefficient
        double sumXY = 0, sumX2 = 0, sumY2 = 0;
        for (int i = 0; i < n; i++)
        {
            double x = NumOps.ToDouble(data[i, col]) - meanX;
            double y = NumOps.ToDouble(target[i]) - meanY;
            sumXY += x * y;
            sumX2 += x * x;
            sumY2 += y * y;
        }

        if (sumX2 < 1e-10 || sumY2 < 1e-10)
        {
            return (0, 1);
        }

        double r = sumXY / Math.Sqrt(sumX2 * sumY2);

        // F-statistic: F = r^2 * (n-2) / (1 - r^2)
        double r2 = r * r;
        if (r2 >= 1)
        {
            return (double.MaxValue, 0);
        }

        double fScore = r2 * (n - 2) / (1 - r2);

        // Approximate p-value (simplified)
        double pValue = 1.0 / (1.0 + fScore);

        return (fScore, pValue);
    }

    private (double Score, double PValue) ComputeFClassif(Matrix<T> data, Vector<T> target, int col)
    {
        // ANOVA F-score for classification
        int n = data.Rows;

        // Group values by class
        var groups = new Dictionary<double, List<double>>();
        for (int i = 0; i < n; i++)
        {
            double classVal = NumOps.ToDouble(target[i]);
            double featureVal = NumOps.ToDouble(data[i, col]);

            if (!groups.TryGetValue(classVal, out var list))
            {
                list = new List<double>();
                groups[classVal] = list;
            }
            list.Add(featureVal);
        }

        // Calculate grand mean
        double grandMean = 0;
        for (int i = 0; i < n; i++)
        {
            grandMean += NumOps.ToDouble(data[i, col]);
        }
        grandMean /= n;

        // Calculate between-group and within-group sum of squares
        double ssBetween = 0;
        double ssWithin = 0;

        foreach (var kvp in groups)
        {
            var groupValues = kvp.Value;
            double groupMean = groupValues.Average();

            ssBetween += groupValues.Count * Math.Pow(groupMean - grandMean, 2);

            foreach (double val in groupValues)
            {
                ssWithin += Math.Pow(val - groupMean, 2);
            }
        }

        int k = groups.Count;
        double dfBetween = k - 1;
        double dfWithin = n - k;

        if (dfBetween < 1 || dfWithin < 1 || ssWithin < 1e-10)
        {
            return (0, 1);
        }

        double msBetween = ssBetween / dfBetween;
        double msWithin = ssWithin / dfWithin;

        double fScore = msBetween / msWithin;
        double pValue = 1.0 / (1.0 + fScore); // Simplified p-value

        return (fScore, pValue);
    }

    private (double Score, double PValue) ComputeChi2(Matrix<T> data, Vector<T> target, int col)
    {
        // Chi-squared test (feature must be non-negative)
        int n = data.Rows;

        // Get unique classes
        var classes = new HashSet<double>();
        for (int i = 0; i < n; i++)
        {
            classes.Add(NumOps.ToDouble(target[i]));
        }

        // Sum of feature values per class
        var classSums = new Dictionary<double, double>();
        var classCounts = new Dictionary<double, int>();
        double totalSum = 0;

        foreach (double c in classes)
        {
            classSums[c] = 0;
            classCounts[c] = 0;
        }

        for (int i = 0; i < n; i++)
        {
            double classVal = NumOps.ToDouble(target[i]);
            double featureVal = Math.Max(0, NumOps.ToDouble(data[i, col])); // Chi2 requires non-negative
            classSums[classVal] += featureVal;
            classCounts[classVal]++;
            totalSum += featureVal;
        }

        if (totalSum < 1e-10)
        {
            return (0, 1);
        }

        // Expected values
        double chi2 = 0;
        foreach (double c in classes)
        {
            double observed = classSums[c];
            double expected = totalSum * classCounts[c] / n;

            if (expected > 1e-10)
            {
                chi2 += Math.Pow(observed - expected, 2) / expected;
            }
        }

        double pValue = 1.0 / (1.0 + chi2); // Simplified p-value

        return (chi2, pValue);
    }

    private double ComputeMutualInfo(Matrix<T> data, Vector<T> target, int col)
    {
        // Simplified mutual information using discretization
        int n = data.Rows;
        int nBins = Math.Min(10, (int)Math.Sqrt(n));

        // Discretize feature
        var featureValues = new List<double>();
        for (int i = 0; i < n; i++)
        {
            featureValues.Add(NumOps.ToDouble(data[i, col]));
        }

        double minVal = featureValues.Min();
        double maxVal = featureValues.Max();
        double range = maxVal - minVal;

        if (range < 1e-10)
        {
            return 0;
        }

        // Calculate joint and marginal distributions
        var jointCounts = new Dictionary<(int, double), int>();
        var featureCounts = new int[nBins];
        var targetCounts = new Dictionary<double, int>();

        for (int i = 0; i < n; i++)
        {
            int featureBin = Math.Min(nBins - 1, (int)((featureValues[i] - minVal) / range * nBins));
            double targetVal = NumOps.ToDouble(target[i]);

            var key = (featureBin, targetVal);
            if (!jointCounts.TryGetValue(key, out int count))
            {
                count = 0;
            }
            jointCounts[key] = count + 1;

            featureCounts[featureBin]++;

            if (!targetCounts.TryGetValue(targetVal, out count))
            {
                count = 0;
            }
            targetCounts[targetVal] = count + 1;
        }

        // Calculate mutual information
        double mi = 0;
        foreach (var kvp in jointCounts)
        {
            int featureBin = kvp.Key.Item1;
            double targetVal = kvp.Key.Item2;
            int jointCount = kvp.Value;

            double pJoint = (double)jointCount / n;
            double pFeature = (double)featureCounts[featureBin] / n;
            double pTarget = (double)targetCounts[targetVal] / n;

            if (pJoint > 0 && pFeature > 0 && pTarget > 0)
            {
                mi += pJoint * Math.Log(pJoint / (pFeature * pTarget));
            }
        }

        return Math.Max(0, mi);
    }

    /// <summary>
    /// Selects the top K features from the data.
    /// </summary>
    protected override Matrix<T> TransformCore(Matrix<T> data)
    {
        if (_selectedFeatures is null)
        {
            throw new InvalidOperationException("Selector has not been fitted.");
        }

        int numRows = data.Rows;
        int numOutputCols = _selectedFeatures.Length;
        var result = new T[numRows, numOutputCols];

        for (int i = 0; i < numRows; i++)
        {
            for (int j = 0; j < numOutputCols; j++)
            {
                int sourceCol = _selectedFeatures[j];
                result[i, j] = data[i, sourceCol];
            }
        }

        return new Matrix<T>(result);
    }

    /// <summary>
    /// Inverse transformation is not supported.
    /// </summary>
    protected override Matrix<T> InverseTransformCore(Matrix<T> data)
    {
        throw new NotSupportedException("SelectKBest does not support inverse transformation.");
    }

    /// <summary>
    /// Gets a boolean mask indicating which features are selected.
    /// </summary>
    public bool[] GetSupportMask()
    {
        if (_selectedFeatures is null)
        {
            throw new InvalidOperationException("Selector has not been fitted.");
        }

        var mask = new bool[_nInputFeatures];
        var selectedSet = new HashSet<int>(_selectedFeatures);

        for (int i = 0; i < _nInputFeatures; i++)
        {
            mask[i] = selectedSet.Contains(i);
        }

        return mask;
    }

    /// <summary>
    /// Gets the output feature names after transformation.
    /// </summary>
    public override string[] GetFeatureNamesOut(string[]? inputFeatureNames = null)
    {
        if (_selectedFeatures is null)
        {
            return Array.Empty<string>();
        }

        var names = new string[_selectedFeatures.Length];
        for (int i = 0; i < _selectedFeatures.Length; i++)
        {
            int col = _selectedFeatures[i];
            names[i] = inputFeatureNames is not null && col < inputFeatureNames.Length
                ? inputFeatureNames[col]
                : $"x{col}";
        }

        return names;
    }
}

/// <summary>
/// Scoring functions for SelectKBest.
/// </summary>
public enum SelectKBestScoreFunc
{
    /// <summary>
    /// F-value for regression (correlation-based).
    /// </summary>
    FRegression,

    /// <summary>
    /// Mutual information for regression (captures non-linear relationships).
    /// </summary>
    MutualInfoRegression,

    /// <summary>
    /// ANOVA F-value for classification.
    /// </summary>
    FClassif,

    /// <summary>
    /// Chi-squared statistic for classification (features must be non-negative).
    /// </summary>
    Chi2
}
