using AiDotNet.Helpers;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.FeatureSelection.Wrapper;

/// <summary>
/// Direction for sequential feature selection.
/// </summary>
public enum SelectionDirection
{
    /// <summary>Start with no features and add one at a time.</summary>
    Forward,
    /// <summary>Start with all features and remove one at a time.</summary>
    Backward
}

/// <summary>
/// Sequential Feature Selector using greedy forward or backward selection.
/// </summary>
/// <remarks>
/// <para>
/// Sequential Feature Selection (SFS) is a wrapper method that evaluates feature subsets
/// using a model's cross-validation score. Forward selection starts empty and adds features;
/// backward selection starts full and removes features.
/// </para>
/// <para><b>For Beginners:</b> Think of forward selection like building a team: you start
/// with no players and add the best candidate one at a time until you have enough.
/// Backward selection is the opposite: start with everyone and cut the worst performer
/// one at a time. The "best" is determined by how well a model performs with those features.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class SequentialFeatureSelector<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly int _nFeaturesToSelect;
    private readonly SelectionDirection _direction;
    private readonly Func<Matrix<T>, Vector<T>, int[], double>? _scorer;
    private readonly int _nFolds;
    private readonly int? _randomState;

    private double[]? _featureScores;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    public int NFeaturesToSelect => _nFeaturesToSelect;
    public SelectionDirection Direction => _direction;
    public double[]? FeatureScores => _featureScores;
    public int[]? SelectedIndices => _selectedIndices;
    public override bool SupportsInverseTransform => false;

    public SequentialFeatureSelector(
        int nFeaturesToSelect = 5,
        SelectionDirection direction = SelectionDirection.Forward,
        Func<Matrix<T>, Vector<T>, int[], double>? scorer = null,
        int nFolds = 5,
        int? randomState = null,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));

        _nFeaturesToSelect = nFeaturesToSelect;
        _direction = direction;
        _scorer = scorer;
        _nFolds = nFolds;
        _randomState = randomState;
    }

    protected override void FitCore(Matrix<T> data)
    {
        throw new InvalidOperationException(
            "SequentialFeatureSelector requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
    }

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

        _featureScores = new double[p];
        var scorer = _scorer ?? DefaultScorer;

        if (_direction == SelectionDirection.Forward)
        {
            _selectedIndices = ForwardSelection(data, target, p, scorer);
        }
        else
        {
            _selectedIndices = BackwardSelection(data, target, p, scorer);
        }

        IsFitted = true;
    }

    private int[] ForwardSelection(Matrix<T> data, Vector<T> target, int p,
        Func<Matrix<T>, Vector<T>, int[], double> scorer)
    {
        var selected = new List<int>();
        var remaining = Enumerable.Range(0, p).ToHashSet();

        int numToSelect = Math.Min(_nFeaturesToSelect, p);

        while (selected.Count < numToSelect && remaining.Count > 0)
        {
            double bestScore = double.NegativeInfinity;
            int bestFeature = -1;

            foreach (int candidate in remaining)
            {
                var testSet = selected.Concat([candidate]).ToArray();
                double score = scorer(data, target, testSet);

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
                _featureScores![bestFeature] = bestScore;
            }
            else
            {
                break;
            }
        }

        return selected.OrderBy(x => x).ToArray();
    }

    private int[] BackwardSelection(Matrix<T> data, Vector<T> target, int p,
        Func<Matrix<T>, Vector<T>, int[], double> scorer)
    {
        var remaining = Enumerable.Range(0, p).ToList();
        int numToRemove = p - Math.Min(_nFeaturesToSelect, p);

        for (int iteration = 0; iteration < numToRemove && remaining.Count > 1; iteration++)
        {
            double bestScore = double.NegativeInfinity;
            int worstFeature = -1;

            foreach (int candidate in remaining)
            {
                var testSet = remaining.Where(f => f != candidate).ToArray();
                double score = scorer(data, target, testSet);

                if (score > bestScore)
                {
                    bestScore = score;
                    worstFeature = candidate;
                }
            }

            if (worstFeature >= 0)
            {
                _featureScores![worstFeature] = -bestScore; // Lower score = worse feature
                remaining.Remove(worstFeature);
            }
            else
            {
                break;
            }
        }

        // Set positive scores for remaining features
        double finalScore = DefaultScorer(data, target, remaining.ToArray());
        foreach (int idx in remaining)
            _featureScores![idx] = finalScore;

        return remaining.OrderBy(x => x).ToArray();
    }

    private double DefaultScorer(Matrix<T> data, Vector<T> target, int[] featureIndices)
    {
        // Simple cross-validated R² score using linear regression
        if (featureIndices.Length == 0)
            return double.NegativeInfinity;

        int n = data.Rows;
        int foldSize = n / _nFolds;
        double totalScore = 0;

        for (int fold = 0; fold < _nFolds; fold++)
        {
            int testStart = fold * foldSize;
            int testEnd = fold == _nFolds - 1 ? n : (fold + 1) * foldSize;

            // Extract training data
            var trainX = new List<double[]>();
            var trainY = new List<double>();
            var testX = new List<double[]>();
            var testY = new List<double>();

            for (int i = 0; i < n; i++)
            {
                var row = new double[featureIndices.Length];
                for (int j = 0; j < featureIndices.Length; j++)
                    row[j] = NumOps.ToDouble(data[i, featureIndices[j]]);

                double y = NumOps.ToDouble(target[i]);

                if (i >= testStart && i < testEnd)
                {
                    testX.Add(row);
                    testY.Add(y);
                }
                else
                {
                    trainX.Add(row);
                    trainY.Add(y);
                }
            }

            // Fit simple linear regression
            double score = ComputeR2Score(trainX, trainY, testX, testY);
            totalScore += score;
        }

        return totalScore / _nFolds;
    }

    private static double ComputeR2Score(List<double[]> trainX, List<double> trainY,
        List<double[]> testX, List<double> testY)
    {
        if (trainX.Count == 0 || testX.Count == 0)
            return double.NegativeInfinity;

        int nFeatures = trainX[0].Length;
        int nTrain = trainX.Count;

        // Compute means
        double yMean = trainY.Average();
        var xMeans = new double[nFeatures];
        for (int j = 0; j < nFeatures; j++)
        {
            for (int i = 0; i < nTrain; i++)
                xMeans[j] += trainX[i][j];
            xMeans[j] /= nTrain;
        }

        // Simple multivariate linear regression using normal equations
        // For simplicity, use correlation-weighted average prediction
        var correlations = new double[nFeatures];
        var xStds = new double[nFeatures];

        for (int j = 0; j < nFeatures; j++)
        {
            double sxy = 0, sxx = 0, syy = 0;
            for (int i = 0; i < nTrain; i++)
            {
                double xDiff = trainX[i][j] - xMeans[j];
                double yDiff = trainY[i] - yMean;
                sxy += xDiff * yDiff;
                sxx += xDiff * xDiff;
                syy += yDiff * yDiff;
            }
            xStds[j] = Math.Sqrt(sxx / nTrain);
            double yStd = Math.Sqrt(syy / nTrain);
            correlations[j] = (sxx > 1e-10 && syy > 1e-10) ? sxy / Math.Sqrt(sxx * syy) : 0;
        }

        // Predict on test set using correlation-weighted features
        double ssTot = 0, ssRes = 0;
        double testYMean = testY.Average();

        for (int i = 0; i < testX.Count; i++)
        {
            double pred = yMean;
            double corrSum = 0;
            for (int j = 0; j < nFeatures; j++)
            {
                if (xStds[j] > 1e-10)
                {
                    double zScore = (testX[i][j] - xMeans[j]) / xStds[j];
                    pred += correlations[j] * zScore * Math.Sqrt(trainY.Select(y => (y - yMean) * (y - yMean)).Average());
                    corrSum += Math.Abs(correlations[j]);
                }
            }
            if (corrSum > 0 && nFeatures > 1)
                pred = yMean + (pred - yMean) / corrSum * Math.Abs(correlations.Max());

            double residual = testY[i] - pred;
            ssRes += residual * residual;
            ssTot += (testY[i] - testYMean) * (testY[i] - testYMean);
        }

        return ssTot > 1e-10 ? 1 - ssRes / ssTot : 0;
    }

    public Matrix<T> FitTransform(Matrix<T> data, Vector<T> target)
    {
        Fit(data, target);
        return Transform(data);
    }

    protected override Matrix<T> TransformCore(Matrix<T> data)
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("SequentialFeatureSelector has not been fitted.");

        int numRows = data.Rows;
        int numCols = _selectedIndices.Length;
        var result = new T[numRows, numCols];

        for (int i = 0; i < numRows; i++)
            for (int j = 0; j < numCols; j++)
                result[i, j] = data[i, _selectedIndices[j]];

        return new Matrix<T>(result);
    }

    protected override Matrix<T> InverseTransformCore(Matrix<T> data)
    {
        throw new NotSupportedException("SequentialFeatureSelector does not support inverse transformation.");
    }

    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("SequentialFeatureSelector has not been fitted.");

        var mask = new bool[_nInputFeatures];
        foreach (int idx in _selectedIndices)
            mask[idx] = true;

        return mask;
    }

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
