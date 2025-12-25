using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.FeatureSelection;

/// <summary>
/// Sequential feature selection using forward or backward selection.
/// </summary>
/// <remarks>
/// <para>
/// SequentialFeatureSelector performs feature selection by sequentially adding or removing
/// features based on cross-validation scores.
/// </para>
/// <para>
/// - Forward selection: Start with no features, add the best one at each step
/// - Backward selection: Start with all features, remove the worst one at each step
/// </para>
/// <para><b>For Beginners:</b> This is like building a team:
/// - Forward: Start with no players, add the best available each round
/// - Backward: Start with everyone, remove the weakest each round
/// - Stop when you have the desired team size
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations (e.g., float, double).</typeparam>
public class SequentialFeatureSelector<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly int _nFeaturesToSelect;
    private readonly SequentialDirection _direction;
    private readonly int _cv;
    private readonly Func<double[], double[], double>? _scoringFunc;

    // Fitted parameters
    private int[]? _selectedIndices;
    private bool[]? _supportMask;
    private int _nInputFeatures;

    /// <summary>
    /// Gets the number of features to select.
    /// </summary>
    public int NFeaturesToSelect => _nFeaturesToSelect;

    /// <summary>
    /// Gets the selection direction.
    /// </summary>
    public SequentialDirection Direction => _direction;

    /// <summary>
    /// Gets the indices of selected features.
    /// </summary>
    public int[]? SelectedIndices => _selectedIndices;

    /// <summary>
    /// Gets whether this transformer supports inverse transformation.
    /// </summary>
    public override bool SupportsInverseTransform => false;

    /// <summary>
    /// Creates a new instance of <see cref="SequentialFeatureSelector{T}"/>.
    /// </summary>
    /// <param name="nFeaturesToSelect">Number of features to select.</param>
    /// <param name="direction">Selection direction (Forward or Backward). Defaults to Forward.</param>
    /// <param name="cv">Number of cross-validation folds. Defaults to 5.</param>
    /// <param name="scoringFunc">Custom scoring function (y_true, y_pred) => score. Null for R² score.</param>
    /// <param name="columnIndices">The column indices to evaluate, or null for all columns.</param>
    public SequentialFeatureSelector(
        int nFeaturesToSelect,
        SequentialDirection direction = SequentialDirection.Forward,
        int cv = 5,
        Func<double[], double[], double>? scoringFunc = null,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
        {
            throw new ArgumentException("Number of features to select must be at least 1.", nameof(nFeaturesToSelect));
        }

        if (cv < 2)
        {
            throw new ArgumentException("Number of CV folds must be at least 2.", nameof(cv));
        }

        _nFeaturesToSelect = nFeaturesToSelect;
        _direction = direction;
        _cv = cv;
        _scoringFunc = scoringFunc;
    }

    /// <summary>
    /// Fits the selector (requires target via specialized Fit method).
    /// </summary>
    protected override void FitCore(Matrix<T> data)
    {
        throw new InvalidOperationException(
            "SequentialFeatureSelector requires target values for fitting. Use Fit(Matrix<T> data, Vector<T> target) instead.");
    }

    /// <summary>
    /// Fits the selector using sequential selection with cross-validation.
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

        int nToSelect = Math.Min(_nFeaturesToSelect, p);

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

        if (_direction == SequentialDirection.Forward)
        {
            _selectedIndices = ForwardSelection(X, y, foldIndices, nToSelect);
        }
        else
        {
            _selectedIndices = BackwardSelection(X, y, foldIndices, nToSelect);
        }

        // Create support mask
        _supportMask = new bool[p];
        foreach (int idx in _selectedIndices)
        {
            _supportMask[idx] = true;
        }

        IsFitted = true;
    }

    private int[] ForwardSelection(double[,] X, double[] y, List<(int[] Train, int[] Test)> folds, int nToSelect)
    {
        int p = X.GetLength(1);
        var selected = new HashSet<int>();
        var remaining = new HashSet<int>(Enumerable.Range(0, p));

        while (selected.Count < nToSelect && remaining.Count > 0)
        {
            int bestFeature = -1;
            double bestScore = double.NegativeInfinity;

            foreach (int candidate in remaining)
            {
                var testSet = selected.Concat(new[] { candidate }).ToArray();
                double score = EvaluateFeatureSetCV(X, y, folds, testSet);

                if (score > bestScore)
                {
                    bestScore = score;
                    bestFeature = candidate;
                }
            }

            if (bestFeature >= 0)
            {
                selected.Add(bestFeature);
                remaining.Remove(bestFeature);
            }
            else
            {
                break;
            }
        }

        return selected.OrderBy(i => i).ToArray();
    }

    private int[] BackwardSelection(double[,] X, double[] y, List<(int[] Train, int[] Test)> folds, int nToSelect)
    {
        int p = X.GetLength(1);
        var remaining = new HashSet<int>(Enumerable.Range(0, p));

        while (remaining.Count > nToSelect)
        {
            int worstFeature = -1;
            double bestScore = double.NegativeInfinity;

            foreach (int candidate in remaining)
            {
                var testSet = remaining.Where(f => f != candidate).ToArray();
                double score = EvaluateFeatureSetCV(X, y, folds, testSet);

                if (score > bestScore)
                {
                    bestScore = score;
                    worstFeature = candidate;
                }
            }

            if (worstFeature >= 0)
            {
                remaining.Remove(worstFeature);
            }
            else
            {
                break;
            }
        }

        return remaining.OrderBy(i => i).ToArray();
    }

    private double EvaluateFeatureSetCV(double[,] X, double[] y, List<(int[] Train, int[] Test)> folds, int[] features)
    {
        if (features.Length == 0) return double.NegativeInfinity;

        var scores = new List<double>();

        foreach (var (trainIdx, testIdx) in folds)
        {
            double score = EvaluateFold(X, y, trainIdx, testIdx, features);
            scores.Add(score);
        }

        return scores.Average();
    }

    private double EvaluateFold(double[,] X, double[] y, int[] trainIdx, int[] testIdx, int[] features)
    {
        int nTrain = trainIdx.Length;
        int nTest = testIdx.Length;
        int p = features.Length;

        // Extract training target
        var yTrain = new double[nTrain];
        for (int i = 0; i < nTrain; i++)
        {
            yTrain[i] = y[trainIdx[i]];
        }

        var yTest = new double[nTest];
        for (int i = 0; i < nTest; i++)
        {
            yTest[i] = y[testIdx[i]];
        }

        double yMean = yTrain.Average();

        // Simple linear regression for each feature
        var weights = new double[p];
        var xMeans = new double[p];

        for (int k = 0; k < p; k++)
        {
            int j = features[k];
            double xSum = 0;
            for (int i = 0; i < nTrain; i++)
            {
                xSum += X[trainIdx[i], j];
            }
            xMeans[k] = xSum / nTrain;

            double ssXY = 0, ssXX = 0;
            for (int i = 0; i < nTrain; i++)
            {
                double dx = X[trainIdx[i], j] - xMeans[k];
                double dy = yTrain[i] - yMean;
                ssXY += dx * dy;
                ssXX += dx * dx;
            }

            weights[k] = ssXX > 1e-10 ? ssXY / ssXX : 0;
        }

        // Predict on test set
        var yPred = new double[nTest];
        for (int i = 0; i < nTest; i++)
        {
            yPred[i] = yMean;
            for (int k = 0; k < p; k++)
            {
                int j = features[k];
                yPred[i] += weights[k] * (X[testIdx[i], j] - xMeans[k]);
            }
        }

        // Compute score
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

    /// <summary>
    /// Fits and transforms the data.
    /// </summary>
    public Matrix<T> FitTransform(Matrix<T> data, Vector<T> target)
    {
        Fit(data, target);
        return Transform(data);
    }

    /// <summary>
    /// Transforms the data by selecting the chosen features.
    /// </summary>
    protected override Matrix<T> TransformCore(Matrix<T> data)
    {
        if (_selectedIndices is null)
        {
            throw new InvalidOperationException("SequentialFeatureSelector has not been fitted.");
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
        throw new NotSupportedException("SequentialFeatureSelector does not support inverse transformation.");
    }

    /// <summary>
    /// Gets the support mask indicating which features are selected.
    /// </summary>
    public bool[] GetSupportMask()
    {
        if (_supportMask is null)
        {
            throw new InvalidOperationException("SequentialFeatureSelector has not been fitted.");
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

/// <summary>
/// Specifies the direction of sequential feature selection.
/// </summary>
public enum SequentialDirection
{
    /// <summary>
    /// Forward selection: Start with no features, add the best one at each step.
    /// </summary>
    Forward,

    /// <summary>
    /// Backward selection: Start with all features, remove the worst one at each step.
    /// </summary>
    Backward
}
