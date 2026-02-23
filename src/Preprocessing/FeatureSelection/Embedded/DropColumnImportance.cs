using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.FeatureSelection.Embedded;

/// <summary>
/// Drop Column Importance feature selection by measuring score decrease when dropping features.
/// </summary>
/// <remarks>
/// <para>
/// Drop Column Importance measures the decrease in model performance when each feature is
/// completely removed from the dataset and the model is retrained. Unlike permutation
/// importance, this captures the feature's contribution to model learning, not just prediction.
/// </para>
/// <para><b>For Beginners:</b> Instead of shuffling a feature (like permutation importance),
/// you completely remove it and retrain the model from scratch. If the model performs much
/// worse without a feature, that feature is important for learning. This is more thorough
/// but also more computationally expensive since you retrain for each feature.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class DropColumnImportance<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly int _nFeaturesToSelect;
    private readonly Func<Matrix<T>, Vector<T>, double>? _scorer;
    private readonly int _nFolds;

    private double[]? _importances;
    private int[]? _selectedIndices;
    private int _nInputFeatures;
    private double _baselineScore;

    public int NFeaturesToSelect => _nFeaturesToSelect;
    public double[]? Importances => _importances;
    public double BaselineScore => _baselineScore;
    public int[]? SelectedIndices => _selectedIndices;
    public override bool SupportsInverseTransform => false;

    public DropColumnImportance(
        int nFeaturesToSelect = 10,
        Func<Matrix<T>, Vector<T>, double>? scorer = null,
        int nFolds = 5,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));

        _nFeaturesToSelect = nFeaturesToSelect;
        _scorer = scorer;
        _nFolds = nFolds;
    }

    protected override void FitCore(Matrix<T> data)
    {
        throw new InvalidOperationException(
            "DropColumnImportance requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
    }

    public void Fit(Matrix<T> data, Vector<T> target)
    {
        if (data.Rows != target.Length)
            throw new ArgumentException("Target length must match rows in data.");

        _nInputFeatures = data.Columns;
        int n = data.Rows;
        int p = data.Columns;

        var scorer = _scorer ?? DefaultCVScorer;

        // Compute baseline score with all features
        _baselineScore = scorer(data, target);

        _importances = new double[p];

        // For each feature, drop it and measure score decrease
        for (int j = 0; j < p; j++)
        {
            var droppedData = CreateDataWithoutFeature(data, j, n, p);
            double droppedScore = scorer(droppedData, target);
            _importances[j] = _baselineScore - droppedScore;
        }

        // Select top features by importance
        int numToSelect = Math.Min(_nFeaturesToSelect, p);
        _selectedIndices = _importances
            .Select((imp, idx) => (Importance: imp, Index: idx))
            .OrderByDescending(x => x.Importance)
            .Take(numToSelect)
            .Select(x => x.Index)
            .OrderBy(x => x)
            .ToArray();

        IsFitted = true;
    }

    private Matrix<T> CreateDataWithoutFeature(Matrix<T> data, int featureIdx, int n, int p)
    {
        var result = new T[n, p - 1];
        int newCol = 0;

        for (int j = 0; j < p; j++)
        {
            if (j == featureIdx) continue;

            for (int i = 0; i < n; i++)
                result[i, newCol] = data[i, j];
            newCol++;
        }

        return new Matrix<T>(result);
    }

    private double DefaultCVScorer(Matrix<T> data, Vector<T> target)
    {
        // Cross-validated R² score
        int n = data.Rows;
        int p = data.Columns;
        int foldSize = n / _nFolds;
        double totalScore = 0;

        for (int fold = 0; fold < _nFolds; fold++)
        {
            int testStart = fold * foldSize;
            int testEnd = fold == _nFolds - 1 ? n : (fold + 1) * foldSize;

            // Split data
            var trainX = new List<double[]>();
            var trainY = new List<double>();
            var testX = new List<double[]>();
            var testY = new List<double>();

            for (int i = 0; i < n; i++)
            {
                var row = new double[p];
                for (int j = 0; j < p; j++)
                    row[j] = NumOps.ToDouble(data[i, j]);

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

            if (trainX.Count > 0 && testX.Count > 0 && p > 0)
            {
                double score = ComputeR2Score(trainX, trainY, testX, testY);
                totalScore += score;
            }
        }

        return totalScore / _nFolds;
    }

    private static double ComputeR2Score(List<double[]> trainX, List<double> trainY,
        List<double[]> testX, List<double> testY)
    {
        if (trainX.Count == 0 || testX.Count == 0)
            return 0;

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

        // Compute correlations and standard deviations
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

        // Predict on test set
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
            throw new InvalidOperationException("DropColumnImportance has not been fitted.");

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
        throw new NotSupportedException("DropColumnImportance does not support inverse transformation.");
    }

    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("DropColumnImportance has not been fitted.");

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
