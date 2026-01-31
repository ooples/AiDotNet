using AiDotNet.Helpers;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.FeatureSelection.Embedded;

/// <summary>
/// Permutation Importance feature selection by measuring score decrease when shuffling features.
/// </summary>
/// <remarks>
/// <para>
/// Permutation Importance measures the decrease in model performance when a single feature's
/// values are randomly shuffled, breaking the relationship between the feature and the target.
/// Features whose shuffling causes large performance drops are considered important.
/// </para>
/// <para><b>For Beginners:</b> Imagine you have a trained model and want to know which features
/// it relies on. You shuffle one feature's values randomly (like shuffling a deck of cards)
/// and see how much worse the model performs. If performance drops a lot, that feature was
/// important. If shuffling makes no difference, the feature wasn't useful.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class PermutationImportance<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly int _nFeaturesToSelect;
    private readonly int _nRepeats;
    private readonly Func<Matrix<T>, Vector<T>, double>? _scorer;
    private readonly int? _randomState;

    private double[]? _importances;
    private double[]? _importanceStds;
    private int[]? _selectedIndices;
    private int _nInputFeatures;
    private double _baselineScore;

    public int NFeaturesToSelect => _nFeaturesToSelect;
    public int NRepeats => _nRepeats;
    public double[]? Importances => _importances;
    public double[]? ImportanceStds => _importanceStds;
    public double BaselineScore => _baselineScore;
    public int[]? SelectedIndices => _selectedIndices;
    public override bool SupportsInverseTransform => false;

    public PermutationImportance(
        int nFeaturesToSelect = 10,
        int nRepeats = 5,
        Func<Matrix<T>, Vector<T>, double>? scorer = null,
        int? randomState = null,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));
        if (nRepeats < 1)
            throw new ArgumentException("Number of repeats must be at least 1.", nameof(nRepeats));

        _nFeaturesToSelect = nFeaturesToSelect;
        _nRepeats = nRepeats;
        _scorer = scorer;
        _randomState = randomState;
    }

    protected override void FitCore(Matrix<T> data)
    {
        throw new InvalidOperationException(
            "PermutationImportance requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
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

        var scorer = _scorer ?? DefaultScorer;

        // Compute baseline score
        _baselineScore = scorer(data, target);

        _importances = new double[p];
        _importanceStds = new double[p];

        // For each feature, permute and measure score decrease
        for (int j = 0; j < p; j++)
        {
            var scores = new double[_nRepeats];

            for (int r = 0; r < _nRepeats; r++)
            {
                // Create permuted data
                var permutedData = CreatePermutedData(data, j, n, p, random);
                double permutedScore = scorer(permutedData, target);
                scores[r] = _baselineScore - permutedScore;
            }

            _importances[j] = scores.Average();
            _importanceStds[j] = scores.Length > 1
                ? Math.Sqrt(scores.Select(s => (s - _importances[j]) * (s - _importances[j])).Sum() / (scores.Length - 1))
                : 0;
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

    private Matrix<T> CreatePermutedData(Matrix<T> data, int featureIdx, int n, int p, Random random)
    {
        var result = new T[n, p];

        // Copy all data
        for (int i = 0; i < n; i++)
            for (int j = 0; j < p; j++)
                result[i, j] = data[i, j];

        // Shuffle the target feature column
        var indices = Enumerable.Range(0, n).OrderBy(_ => random.Next()).ToArray();
        for (int i = 0; i < n; i++)
            result[i, featureIdx] = data[indices[i], featureIdx];

        return new Matrix<T>(result);
    }

    private double DefaultScorer(Matrix<T> data, Vector<T> target)
    {
        // Simple R² score using correlation-based prediction
        int n = data.Rows;
        int p = data.Columns;

        // Compute means
        double yMean = 0;
        for (int i = 0; i < n; i++)
            yMean += NumOps.ToDouble(target[i]);
        yMean /= n;

        var xMeans = new double[p];
        for (int j = 0; j < p; j++)
        {
            for (int i = 0; i < n; i++)
                xMeans[j] += NumOps.ToDouble(data[i, j]);
            xMeans[j] /= n;
        }

        // Compute correlations
        var correlations = new double[p];
        var xStds = new double[p];
        double yStd = 0;

        for (int j = 0; j < p; j++)
        {
            double sxy = 0, sxx = 0;
            for (int i = 0; i < n; i++)
            {
                double xDiff = NumOps.ToDouble(data[i, j]) - xMeans[j];
                double yDiff = NumOps.ToDouble(target[i]) - yMean;
                sxy += xDiff * yDiff;
                sxx += xDiff * xDiff;
            }
            xStds[j] = Math.Sqrt(sxx / n);
            correlations[j] = sxx > 1e-10 ? sxy / Math.Sqrt(sxx) : 0;
        }

        for (int i = 0; i < n; i++)
        {
            double yDiff = NumOps.ToDouble(target[i]) - yMean;
            yStd += yDiff * yDiff;
        }
        yStd = Math.Sqrt(yStd / n);

        // Predict and compute R²
        double ssTot = 0, ssRes = 0;
        for (int i = 0; i < n; i++)
        {
            double pred = yMean;
            for (int j = 0; j < p; j++)
            {
                if (xStds[j] > 1e-10 && yStd > 1e-10)
                {
                    double zScore = (NumOps.ToDouble(data[i, j]) - xMeans[j]) / xStds[j];
                    pred += (correlations[j] / Math.Sqrt(n)) * zScore * yStd / p;
                }
            }

            double actual = NumOps.ToDouble(target[i]);
            ssRes += (actual - pred) * (actual - pred);
            ssTot += (actual - yMean) * (actual - yMean);
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
            throw new InvalidOperationException("PermutationImportance has not been fitted.");

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
        throw new NotSupportedException("PermutationImportance does not support inverse transformation.");
    }

    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("PermutationImportance has not been fitted.");

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
