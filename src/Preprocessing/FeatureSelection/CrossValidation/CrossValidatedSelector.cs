using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Tensors.Helpers;

namespace AiDotNet.Preprocessing.FeatureSelection.CrossValidation;

/// <summary>
/// Cross-Validated Feature Selection.
/// </summary>
/// <remarks>
/// <para>
/// Selects features based on their predictive performance evaluated using
/// k-fold cross-validation with a simple linear model.
/// </para>
/// <para><b>For Beginners:</b> Cross-validation tests how well features predict
/// targets on data not used for training. Features that consistently help
/// predictions across multiple data splits are selected.
/// </para>
/// </remarks>
public class CrossValidatedSelector<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly int _nFeaturesToSelect;
    private readonly int _nFolds;
    private readonly int? _randomState;

    private double[]? _cvScores;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    public int NFeaturesToSelect => _nFeaturesToSelect;
    public int NFolds => _nFolds;
    public double[]? CVScores => _cvScores;
    public int[]? SelectedIndices => _selectedIndices;
    public override bool SupportsInverseTransform => false;

    public CrossValidatedSelector(
        int nFeaturesToSelect = 10,
        int nFolds = 5,
        int? randomState = null,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));

        _nFeaturesToSelect = nFeaturesToSelect;
        _nFolds = nFolds;
        _randomState = randomState;
    }

    protected override void FitCore(Matrix<T> data)
    {
        throw new InvalidOperationException(
            "CrossValidatedSelector requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
    }

    public void Fit(Matrix<T> data, Vector<T> target)
    {
        if (data.Rows != target.Length)
            throw new ArgumentException("Target length must match rows in data.");

        _nInputFeatures = data.Columns;
        int n = data.Rows;
        int p = data.Columns;

        var rand = _randomState.HasValue
            ? RandomHelper.CreateSeededRandom(_randomState.Value)
            : RandomHelper.CreateSecureRandom();

        var X = new double[n, p];
        var y = new double[n];
        for (int i = 0; i < n; i++)
        {
            y[i] = NumOps.ToDouble(target[i]);
            for (int j = 0; j < p; j++)
                X[i, j] = NumOps.ToDouble(data[i, j]);
        }

        // Create random fold assignments
        var indices = Enumerable.Range(0, n).ToArray();
        for (int i = n - 1; i > 0; i--)
        {
            int j = rand.Next(i + 1);
            (indices[i], indices[j]) = (indices[j], indices[i]);
        }

        var folds = new int[n];
        for (int i = 0; i < n; i++)
            folds[indices[i]] = i % _nFolds;

        _cvScores = new double[p];

        // Evaluate each feature individually
        for (int j = 0; j < p; j++)
        {
            double totalR2 = 0;

            for (int fold = 0; fold < _nFolds; fold++)
            {
                // Split data
                var trainX = new List<double>();
                var trainY = new List<double>();
                var testX = new List<double>();
                var testY = new List<double>();

                for (int i = 0; i < n; i++)
                {
                    if (folds[i] == fold)
                    {
                        testX.Add(X[i, j]);
                        testY.Add(y[i]);
                    }
                    else
                    {
                        trainX.Add(X[i, j]);
                        trainY.Add(y[i]);
                    }
                }

                if (trainX.Count < 2 || testX.Count == 0) continue;

                // Simple linear regression
                double xMean = trainX.Average();
                double yMean = trainY.Average();

                double numerator = 0, denominator = 0;
                for (int i = 0; i < trainX.Count; i++)
                {
                    numerator += (trainX[i] - xMean) * (trainY[i] - yMean);
                    denominator += (trainX[i] - xMean) * (trainX[i] - xMean);
                }

                double slope = denominator > 1e-10 ? numerator / denominator : 0;
                double intercept = yMean - slope * xMean;

                // Predict on test set
                double ssRes = 0, ssTot = 0;
                double testYMean = testY.Average();
                for (int i = 0; i < testX.Count; i++)
                {
                    double pred = slope * testX[i] + intercept;
                    ssRes += (testY[i] - pred) * (testY[i] - pred);
                    ssTot += (testY[i] - testYMean) * (testY[i] - testYMean);
                }

                double r2 = ssTot > 1e-10 ? 1 - ssRes / ssTot : 0;
                totalR2 += Math.Max(0, r2); // Avoid negative R2
            }

            _cvScores[j] = totalR2 / _nFolds;
        }

        int numToSelect = Math.Min(_nFeaturesToSelect, p);
        _selectedIndices = Enumerable.Range(0, p)
            .OrderByDescending(j => _cvScores[j])
            .Take(numToSelect)
            .OrderBy(x => x)
            .ToArray();

        IsFitted = true;
    }

    public Matrix<T> FitTransform(Matrix<T> data, Vector<T> target)
    {
        Fit(data, target);
        return Transform(data);
    }

    protected override Matrix<T> TransformCore(Matrix<T> data)
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("CrossValidatedSelector has not been fitted.");

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
        throw new NotSupportedException("CrossValidatedSelector does not support inverse transformation.");
    }

    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("CrossValidatedSelector has not been fitted.");

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
