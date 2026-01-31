using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.FeatureSelection.Wrapper;

/// <summary>
/// Exhaustive Feature Selector that evaluates all possible feature subsets.
/// </summary>
/// <remarks>
/// <para>
/// The Exhaustive Feature Selector evaluates all possible combinations of features up to
/// a maximum subset size. This guarantees finding the optimal subset but has exponential
/// complexity O(2^p) where p is the number of features.
/// </para>
/// <para><b>For Beginners:</b> This is the brute-force approach: try every possible
/// combination of features and pick the best one. It's guaranteed to find the optimal
/// subset, but it's only practical for small numbers of features (typically under 20)
/// because the number of combinations grows exponentially.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class ExhaustiveFeatureSelector<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly int _minFeatures;
    private readonly int _maxFeatures;
    private readonly Func<Matrix<T>, Vector<T>, int[], double>? _scorer;
    private readonly int _nFolds;

    private double _bestScore;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    public int MinFeatures => _minFeatures;
    public int MaxFeatures => _maxFeatures;
    public double BestScore => _bestScore;
    public int[]? SelectedIndices => _selectedIndices;
    public override bool SupportsInverseTransform => false;

    public ExhaustiveFeatureSelector(
        int minFeatures = 1,
        int maxFeatures = 10,
        Func<Matrix<T>, Vector<T>, int[], double>? scorer = null,
        int nFolds = 5,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (minFeatures < 1)
            throw new ArgumentException("Minimum features must be at least 1.", nameof(minFeatures));
        if (maxFeatures < minFeatures)
            throw new ArgumentException("Maximum features must be >= minimum features.", nameof(maxFeatures));

        _minFeatures = minFeatures;
        _maxFeatures = maxFeatures;
        _scorer = scorer;
        _nFolds = nFolds;
    }

    protected override void FitCore(Matrix<T> data)
    {
        throw new InvalidOperationException(
            "ExhaustiveFeatureSelector requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
    }

    public void Fit(Matrix<T> data, Vector<T> target)
    {
        if (data.Rows != target.Length)
            throw new ArgumentException("Target length must match rows in data.");

        _nInputFeatures = data.Columns;
        int p = data.Columns;
        int maxFeaturesActual = Math.Min(_maxFeatures, p);

        var scorer = _scorer ?? DefaultScorer;
        _bestScore = double.NegativeInfinity;
        _selectedIndices = null;

        // Generate and evaluate all combinations
        for (int k = _minFeatures; k <= maxFeaturesActual; k++)
        {
            foreach (var combination in GetCombinations(p, k))
            {
                double score = scorer(data, target, combination);
                if (score > _bestScore)
                {
                    _bestScore = score;
                    _selectedIndices = combination.OrderBy(x => x).ToArray();
                }
            }
        }

        // Fallback if nothing was selected
        if (_selectedIndices is null || _selectedIndices.Length == 0)
        {
            _selectedIndices = Enumerable.Range(0, Math.Min(_minFeatures, p)).ToArray();
        }

        IsFitted = true;
    }

    private static IEnumerable<int[]> GetCombinations(int n, int k)
    {
        if (k > n || k < 0)
            yield break;

        int[] result = new int[k];
        var stack = new Stack<int>();
        stack.Push(0);

        while (stack.Count > 0)
        {
            int index = stack.Count - 1;
            int value = stack.Pop();

            while (value < n)
            {
                result[index++] = value++;
                stack.Push(value);

                if (index == k)
                {
                    yield return (int[])result.Clone();
                    break;
                }
            }
        }
    }

    private double DefaultScorer(Matrix<T> data, Vector<T> target, int[] featureIndices)
    {
        if (featureIndices.Length == 0)
            return double.NegativeInfinity;

        int n = data.Rows;
        int foldSize = n / _nFolds;
        double totalScore = 0;

        for (int fold = 0; fold < _nFolds; fold++)
        {
            int testStart = fold * foldSize;
            int testEnd = fold == _nFolds - 1 ? n : (fold + 1) * foldSize;

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

        double yMean = trainY.Average();
        var xMeans = new double[nFeatures];
        for (int j = 0; j < nFeatures; j++)
        {
            for (int i = 0; i < nTrain; i++)
                xMeans[j] += trainX[i][j];
            xMeans[j] /= nTrain;
        }

        var correlations = new double[nFeatures];
        var xStds = new double[nFeatures];
        double yVar = trainY.Select(y => (y - yMean) * (y - yMean)).Sum() / nTrain;
        double yStd = Math.Sqrt(yVar);

        for (int j = 0; j < nFeatures; j++)
        {
            double sxy = 0, sxx = 0;
            for (int i = 0; i < nTrain; i++)
            {
                double xDiff = trainX[i][j] - xMeans[j];
                double yDiff = trainY[i] - yMean;
                sxy += xDiff * yDiff;
                sxx += xDiff * xDiff;
            }
            xStds[j] = Math.Sqrt(sxx / nTrain);
            correlations[j] = (sxx > 1e-10 && yVar > 1e-10) ? sxy / Math.Sqrt(sxx * yVar * nTrain) : 0;
        }

        double ssTot = 0, ssRes = 0;
        double testYMean = testY.Average();

        for (int i = 0; i < testX.Count; i++)
        {
            double pred = yMean;
            for (int j = 0; j < nFeatures; j++)
            {
                if (xStds[j] > 1e-10)
                {
                    double zScore = (testX[i][j] - xMeans[j]) / xStds[j];
                    pred += correlations[j] * zScore * yStd;
                }
            }

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
            throw new InvalidOperationException("ExhaustiveFeatureSelector has not been fitted.");

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
        throw new NotSupportedException("ExhaustiveFeatureSelector does not support inverse transformation.");
    }

    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("ExhaustiveFeatureSelector has not been fitted.");

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
