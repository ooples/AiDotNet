using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.FeatureSelection;

/// <summary>
/// Recursive Feature Elimination with Cross-Validation to find optimal feature count.
/// </summary>
/// <remarks>
/// <para>
/// RFECV performs RFE multiple times using cross-validation to determine the optimal
/// number of features. It evaluates model performance at each step of elimination.
/// </para>
/// <para>
/// The algorithm:
/// 1. For each fold, run RFE and record validation scores at each number of features
/// 2. Average scores across folds
/// 3. Select the number of features with best average score
/// </para>
/// <para><b>For Beginners:</b> RFECV automatically finds how many features to keep:
/// - Regular RFE requires you to specify the number of features
/// - RFECV tests different numbers and picks the best one
/// - Uses cross-validation to avoid overfitting the feature selection
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations (e.g., float, double).</typeparam>
public class RFECV<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly int _minFeaturesToSelect;
    private readonly int _step;
    private readonly int _cv;
    private readonly RFEImportanceMethod _importanceMethod;
    private readonly Func<double[], double[], double>? _scoringFunc;

    // Fitted parameters
    private int _nFeaturesSelected;
    private double[]? _cvScores;
    private int[]? _ranking;
    private bool[]? _supportMask;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    /// <summary>
    /// Gets the number of features selected.
    /// </summary>
    public int NFeatures => _nFeaturesSelected;

    /// <summary>
    /// Gets the cross-validation scores at each number of features.
    /// </summary>
    public double[]? CVScores => _cvScores;

    /// <summary>
    /// Gets the feature ranking.
    /// </summary>
    public int[]? Ranking => _ranking;

    /// <summary>
    /// Gets the indices of selected features.
    /// </summary>
    public int[]? SelectedIndices => _selectedIndices;

    /// <summary>
    /// Gets whether this transformer supports inverse transformation.
    /// </summary>
    public override bool SupportsInverseTransform => false;

    /// <summary>
    /// Creates a new instance of <see cref="RFECV{T}"/>.
    /// </summary>
    /// <param name="minFeaturesToSelect">Minimum number of features to consider. Defaults to 1.</param>
    /// <param name="step">Number of features to remove at each iteration. Defaults to 1.</param>
    /// <param name="cv">Number of cross-validation folds. Defaults to 5.</param>
    /// <param name="importanceMethod">Method for computing feature importance. Defaults to Correlation.</param>
    /// <param name="scoringFunc">Custom scoring function (y_true, y_pred) => score. Null for R² score.</param>
    /// <param name="columnIndices">The column indices to evaluate, or null for all columns.</param>
    public RFECV(
        int minFeaturesToSelect = 1,
        int step = 1,
        int cv = 5,
        RFEImportanceMethod importanceMethod = RFEImportanceMethod.Correlation,
        Func<double[], double[], double>? scoringFunc = null,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (minFeaturesToSelect < 1)
        {
            throw new ArgumentException("Minimum features to select must be at least 1.", nameof(minFeaturesToSelect));
        }

        if (step < 1)
        {
            throw new ArgumentException("Step must be at least 1.", nameof(step));
        }

        if (cv < 2)
        {
            throw new ArgumentException("Number of CV folds must be at least 2.", nameof(cv));
        }

        _minFeaturesToSelect = minFeaturesToSelect;
        _step = step;
        _cv = cv;
        _importanceMethod = importanceMethod;
        _scoringFunc = scoringFunc;
    }

    /// <summary>
    /// Fits the selector (requires target via specialized Fit method).
    /// </summary>
    protected override void FitCore(Matrix<T> data)
    {
        throw new InvalidOperationException(
            "RFECV requires target values for fitting. Use Fit(Matrix<T> data, Vector<T> target) instead.");
    }

    /// <summary>
    /// Fits RFECV using cross-validation.
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

        // Generate CV fold indices
        var foldIndices = GenerateCVFolds(n, _cv);

        // Determine number of feature counts to evaluate
        int maxFeatures = p;
        var featureCounts = new List<int>();
        int nf = maxFeatures;
        while (nf >= _minFeaturesToSelect)
        {
            featureCounts.Add(nf);
            nf -= _step;
        }
        if (featureCounts.Count == 0 || featureCounts[featureCounts.Count - 1] != _minFeaturesToSelect)
        {
            featureCounts.Add(_minFeaturesToSelect);
        }
        featureCounts.Reverse();

        // Store scores for each feature count
        var scoresByFeatureCount = new Dictionary<int, List<double>>();
        foreach (int fc in featureCounts)
        {
            scoresByFeatureCount[fc] = new List<double>();
        }

        // Cross-validation
        for (int fold = 0; fold < _cv; fold++)
        {
            var (trainIdx, testIdx) = foldIndices[fold];

            // Extract train/test data
            var (XTrain, yTrain) = ExtractSubset(X, y, trainIdx);
            var (XTest, yTest) = ExtractSubset(X, y, testIdx);

            // Run RFE on training data
            var rfeResult = RunRFE(XTrain, yTrain, _minFeaturesToSelect);

            // Evaluate at each feature count
            foreach (int nFeatures in featureCounts)
            {
                var selectedFeatures = GetTopFeatures(rfeResult, nFeatures);
                double score = EvaluateFeatures(XTrain, yTrain, XTest, yTest, selectedFeatures);
                scoresByFeatureCount[nFeatures].Add(score);
            }
        }

        // Average scores and find optimal
        _cvScores = new double[featureCounts.Count];
        double bestScore = double.NegativeInfinity;
        int bestNFeatures = _minFeaturesToSelect;

        for (int i = 0; i < featureCounts.Count; i++)
        {
            int nFeatures = featureCounts[i];
            double avgScore = scoresByFeatureCount[nFeatures].Average();
            _cvScores[i] = avgScore;

            if (avgScore > bestScore)
            {
                bestScore = avgScore;
                bestNFeatures = nFeatures;
            }
        }

        _nFeaturesSelected = bestNFeatures;

        // Final RFE on full data
        var finalRfeResult = RunRFE(X, y, _minFeaturesToSelect);
        _ranking = finalRfeResult;

        // Create support mask and selected indices
        _selectedIndices = GetTopFeatures(finalRfeResult, _nFeaturesSelected);
        _supportMask = new bool[p];
        foreach (int idx in _selectedIndices)
        {
            _supportMask[idx] = true;
        }

        IsFitted = true;
    }

    private List<(int[] Train, int[] Test)> GenerateCVFolds(int n, int nFolds)
    {
        var folds = new List<(int[], int[])>();
        int foldSize = n / nFolds;
        var indices = Enumerable.Range(0, n).ToArray();

        for (int i = 0; i < nFolds; i++)
        {
            int start = i * foldSize;
            int end = (i == nFolds - 1) ? n : (i + 1) * foldSize;

            var testIdx = indices.Skip(start).Take(end - start).ToArray();
            var trainIdx = indices.Where((_, idx) => idx < start || idx >= end).ToArray();
            folds.Add((trainIdx, testIdx));
        }

        return folds;
    }

    private (double[,] X, double[] y) ExtractSubset(double[,] X, double[] y, int[] indices)
    {
        int n = indices.Length;
        int p = X.GetLength(1);
        var XSub = new double[n, p];
        var ySub = new double[n];

        for (int i = 0; i < n; i++)
        {
            ySub[i] = y[indices[i]];
            for (int j = 0; j < p; j++)
            {
                XSub[i, j] = X[indices[i], j];
            }
        }

        return (XSub, ySub);
    }

    private int[] RunRFE(double[,] X, double[] y, int minFeatures)
    {
        int n = X.GetLength(0);
        int p = X.GetLength(1);

        var ranking = new int[p];
        var remaining = new HashSet<int>(Enumerable.Range(0, p));
        int currentRank = p;

        while (remaining.Count > minFeatures)
        {
            var importances = ComputeImportances(X, y, remaining.ToArray(), n);

            int nToRemove = Math.Min(_step, remaining.Count - minFeatures);
            var toRemove = remaining
                .OrderBy(i => importances[i])
                .Take(nToRemove)
                .ToArray();

            foreach (int idx in toRemove)
            {
                ranking[idx] = currentRank--;
                remaining.Remove(idx);
            }
        }

        foreach (int idx in remaining)
        {
            ranking[idx] = 1;
        }

        return ranking;
    }

    private double[] ComputeImportances(double[,] X, double[] y, int[] featureIndices, int n)
    {
        var importances = new double[X.GetLength(1)];

        foreach (int j in featureIndices)
        {
            double xMean = 0, yMean = y.Average();
            for (int i = 0; i < n; i++)
            {
                xMean += X[i, j];
            }
            xMean /= n;

            double ssXY = 0, ssXX = 0, ssYY = 0;
            for (int i = 0; i < n; i++)
            {
                double dx = X[i, j] - xMean;
                double dy = y[i] - yMean;
                ssXY += dx * dy;
                ssXX += dx * dx;
                ssYY += dy * dy;
            }

            if (ssXX > 1e-10 && ssYY > 1e-10)
            {
                double r = ssXY / Math.Sqrt(ssXX * ssYY);
                importances[j] = Math.Abs(r);
            }
        }

        return importances;
    }

    private int[] GetTopFeatures(int[] ranking, int nFeatures)
    {
        return Enumerable.Range(0, ranking.Length)
            .Where(i => ranking[i] <= nFeatures)
            .OrderBy(i => ranking[i])
            .Take(nFeatures)
            .OrderBy(i => i)
            .ToArray();
    }

    private double EvaluateFeatures(double[,] XTrain, double[] yTrain, double[,] XTest, double[] yTest, int[] selectedFeatures)
    {
        int nTrain = XTrain.GetLength(0);
        int nTest = XTest.GetLength(0);
        int p = selectedFeatures.Length;

        if (p == 0) return double.NegativeInfinity;

        // Simple linear regression prediction
        var yPred = new double[nTest];

        // Compute mean of y_train as baseline
        double yMean = yTrain.Average();

        // For each selected feature, compute correlation-based prediction weight
        var weights = new double[p];
        for (int k = 0; k < p; k++)
        {
            int j = selectedFeatures[k];
            double xMean = 0;
            for (int i = 0; i < nTrain; i++)
            {
                xMean += XTrain[i, j];
            }
            xMean /= nTrain;

            double ssXY = 0, ssXX = 0;
            for (int i = 0; i < nTrain; i++)
            {
                double dx = XTrain[i, j] - xMean;
                double dy = yTrain[i] - yMean;
                ssXY += dx * dy;
                ssXX += dx * dx;
            }

            weights[k] = ssXX > 1e-10 ? ssXY / ssXX : 0;
        }

        // Predict on test set
        for (int i = 0; i < nTest; i++)
        {
            yPred[i] = yMean;
            for (int k = 0; k < p; k++)
            {
                int j = selectedFeatures[k];
                double xMean = 0;
                for (int t = 0; t < nTrain; t++)
                {
                    xMean += XTrain[t, j];
                }
                xMean /= nTrain;

                yPred[i] += weights[k] * (XTest[i, j] - xMean);
            }
        }

        // Compute R² score
        if (_scoringFunc is not null)
        {
            return _scoringFunc(yTest, yPred);
        }

        return ComputeR2Score(yTest, yPred);
    }

    private double ComputeR2Score(double[] yTrue, double[] yPred)
    {
        double yMean = yTrue.Average();
        double ssRes = 0, ssTot = 0;

        for (int i = 0; i < yTrue.Length; i++)
        {
            ssRes += (yTrue[i] - yPred[i]) * (yTrue[i] - yPred[i]);
            ssTot += (yTrue[i] - yMean) * (yTrue[i] - yMean);
        }

        if (ssTot < 1e-10) return 0;
        return 1 - ssRes / ssTot;
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
    /// Transforms the data by selecting optimal features.
    /// </summary>
    protected override Matrix<T> TransformCore(Matrix<T> data)
    {
        if (_selectedIndices is null)
        {
            throw new InvalidOperationException("RFECV has not been fitted.");
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
        throw new NotSupportedException("RFECV does not support inverse transformation.");
    }

    /// <summary>
    /// Gets the support mask indicating which features are selected.
    /// </summary>
    public bool[] GetSupportMask()
    {
        if (_supportMask is null)
        {
            throw new InvalidOperationException("RFECV has not been fitted.");
        }
        return (bool[])_supportMask.Clone();
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
