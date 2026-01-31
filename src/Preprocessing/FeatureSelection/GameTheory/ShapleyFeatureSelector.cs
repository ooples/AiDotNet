using AiDotNet.Helpers;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.FeatureSelection.GameTheory;

/// <summary>
/// Shapley Value-based Feature Selection.
/// </summary>
/// <remarks>
/// <para>
/// Uses Shapley values from cooperative game theory to fairly attribute
/// contribution to each feature based on all possible feature coalitions.
/// </para>
/// <para><b>For Beginners:</b> Shapley values come from game theory and measure
/// how much each "player" (feature) contributes to the team's success (model
/// accuracy). They consider all possible combinations of features, giving a
/// fair and complete picture of each feature's true importance.
/// </para>
/// </remarks>
public class ShapleyFeatureSelector<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly int _nFeaturesToSelect;
    private readonly int _nSamples;
    private readonly int? _randomState;

    private double[]? _shapleyValues;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    public int NFeaturesToSelect => _nFeaturesToSelect;
    public int NSamples => _nSamples;
    public double[]? ShapleyValues => _shapleyValues;
    public int[]? SelectedIndices => _selectedIndices;
    public override bool SupportsInverseTransform => false;

    public ShapleyFeatureSelector(
        int nFeaturesToSelect = 10,
        int nSamples = 100,
        int? randomState = null,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));

        _nFeaturesToSelect = nFeaturesToSelect;
        _nSamples = nSamples;
        _randomState = randomState;
    }

    protected override void FitCore(Matrix<T> data)
    {
        throw new InvalidOperationException(
            "ShapleyFeatureSelector requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
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

        _shapleyValues = new double[p];

        // Monte Carlo approximation of Shapley values
        for (int sample = 0; sample < _nSamples; sample++)
        {
            // Random permutation of features
            var permutation = Enumerable.Range(0, p).OrderBy(_ => rand.Next()).ToArray();
            var included = new HashSet<int>();

            double prevScore = 0;

            for (int i = 0; i < p; i++)
            {
                int feature = permutation[i];
                included.Add(feature);

                double score = EvaluateSubset(X, y, included, n);
                _shapleyValues[feature] += (score - prevScore);
                prevScore = score;
            }
        }

        // Average over samples
        for (int j = 0; j < p; j++)
            _shapleyValues[j] /= _nSamples;

        int numToSelect = Math.Min(_nFeaturesToSelect, p);
        _selectedIndices = Enumerable.Range(0, p)
            .OrderByDescending(j => _shapleyValues[j])
            .Take(numToSelect)
            .OrderBy(x => x)
            .ToArray();

        IsFitted = true;
    }

    private double EvaluateSubset(double[,] X, double[] y, HashSet<int> features, int n)
    {
        if (features.Count == 0) return 0;

        var featureList = features.ToList();
        int k = featureList.Count;

        // Simple linear regression R²
        var XtX = new double[k, k];
        var Xty = new double[k];
        double yMean = y.Average();

        for (int i = 0; i < k; i++)
        {
            int fi = featureList[i];
            for (int j = 0; j < k; j++)
            {
                int fj = featureList[j];
                for (int r = 0; r < n; r++)
                    XtX[i, j] += X[r, fi] * X[r, fj];
            }
            for (int r = 0; r < n; r++)
                Xty[i] += X[r, fi] * (y[r] - yMean);
        }

        // Add regularization
        for (int i = 0; i < k; i++)
            XtX[i, i] += 1e-6;

        var beta = SolveSystem(XtX, Xty, k);

        // Compute R²
        double ssTot = 0, ssRes = 0;
        for (int r = 0; r < n; r++)
        {
            double pred = yMean;
            for (int i = 0; i < k; i++)
                pred += beta[i] * X[r, featureList[i]];
            ssRes += (y[r] - pred) * (y[r] - pred);
            ssTot += (y[r] - yMean) * (y[r] - yMean);
        }

        return ssTot > 1e-10 ? Math.Max(0, 1 - ssRes / ssTot) : 0;
    }

    private double[] SolveSystem(double[,] A, double[] b, int n)
    {
        var aug = new double[n, n + 1];
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++)
                aug[i, j] = A[i, j];
            aug[i, n] = b[i];
        }

        for (int col = 0; col < n; col++)
        {
            int maxRow = col;
            for (int row = col + 1; row < n; row++)
                if (Math.Abs(aug[row, col]) > Math.Abs(aug[maxRow, col]))
                    maxRow = row;

            for (int j = 0; j <= n; j++)
                (aug[col, j], aug[maxRow, j]) = (aug[maxRow, j], aug[col, j]);

            if (Math.Abs(aug[col, col]) < 1e-10) continue;

            for (int row = col + 1; row < n; row++)
            {
                double factor = aug[row, col] / aug[col, col];
                for (int j = col; j <= n; j++)
                    aug[row, j] -= factor * aug[col, j];
            }
        }

        var x = new double[n];
        for (int i = n - 1; i >= 0; i--)
        {
            x[i] = aug[i, n];
            for (int j = i + 1; j < n; j++)
                x[i] -= aug[i, j] * x[j];
            x[i] /= (Math.Abs(aug[i, i]) > 1e-10 ? aug[i, i] : 1);
        }

        return x;
    }

    public Matrix<T> FitTransform(Matrix<T> data, Vector<T> target)
    {
        Fit(data, target);
        return Transform(data);
    }

    protected override Matrix<T> TransformCore(Matrix<T> data)
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("ShapleyFeatureSelector has not been fitted.");

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
        throw new NotSupportedException("ShapleyFeatureSelector does not support inverse transformation.");
    }

    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("ShapleyFeatureSelector has not been fitted.");

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
