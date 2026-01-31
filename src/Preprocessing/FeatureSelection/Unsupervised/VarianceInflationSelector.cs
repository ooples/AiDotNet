using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.FeatureSelection.Unsupervised;

/// <summary>
/// Variance Inflation Factor (VIF) Feature Selection.
/// </summary>
/// <remarks>
/// <para>
/// Removes features with high multicollinearity by computing the Variance
/// Inflation Factor for each feature and removing those above a threshold.
/// </para>
/// <para><b>For Beginners:</b> VIF measures how much a feature can be predicted
/// by other features. A high VIF means that feature is highly correlated with
/// others (redundant). We remove features with high VIF to keep only the
/// independent, non-redundant ones.
/// </para>
/// </remarks>
public class VarianceInflationSelector<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly double _vifThreshold;

    private double[]? _vifScores;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    public double VIFThreshold => _vifThreshold;
    public double[]? VIFScores => _vifScores;
    public int[]? SelectedIndices => _selectedIndices;
    public override bool SupportsInverseTransform => false;

    public VarianceInflationSelector(
        double vifThreshold = 5.0,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (vifThreshold <= 1)
            throw new ArgumentException("VIF threshold must be greater than 1.", nameof(vifThreshold));

        _vifThreshold = vifThreshold;
    }

    protected override void FitCore(Matrix<T> data)
    {
        _nInputFeatures = data.Columns;
        int n = data.Rows;
        int p = data.Columns;

        var X = new double[n, p];
        for (int i = 0; i < n; i++)
            for (int j = 0; j < p; j++)
                X[i, j] = NumOps.ToDouble(data[i, j]);

        // Standardize features
        var means = new double[p];
        var stds = new double[p];
        for (int j = 0; j < p; j++)
        {
            for (int i = 0; i < n; i++) means[j] += X[i, j];
            means[j] /= n;
            for (int i = 0; i < n; i++) stds[j] += (X[i, j] - means[j]) * (X[i, j] - means[j]);
            stds[j] = Math.Sqrt(stds[j] / (n - 1)) + 1e-10;
            for (int i = 0; i < n; i++) X[i, j] = (X[i, j] - means[j]) / stds[j];
        }

        _vifScores = new double[p];
        var selected = new List<int>(Enumerable.Range(0, p));

        // Iteratively remove features with highest VIF above threshold
        bool changed = true;
        while (changed && selected.Count > 1)
        {
            changed = false;

            // Compute VIF for each remaining feature
            int maxVifIdx = -1;
            double maxVif = 0;

            foreach (int j in selected)
            {
                double vif = ComputeVIF(X, selected, j, n);
                _vifScores[j] = vif;

                if (vif > maxVif)
                {
                    maxVif = vif;
                    maxVifIdx = j;
                }
            }

            // Remove feature with highest VIF if above threshold
            if (maxVif > _vifThreshold && maxVifIdx >= 0)
            {
                selected.Remove(maxVifIdx);
                changed = true;
            }
        }

        _selectedIndices = selected.OrderBy(x => x).ToArray();
        IsFitted = true;
    }

    private double ComputeVIF(double[,] X, List<int> features, int targetFeature, int n)
    {
        // Regress targetFeature on all other features
        var predictors = features.Where(f => f != targetFeature).ToList();
        if (predictors.Count == 0)
            return 1.0;

        // Get target column
        var y = new double[n];
        for (int i = 0; i < n; i++)
            y[i] = X[i, targetFeature];

        // Compute R-squared using simple linear regression approximation
        int k = predictors.Count;
        var XtX = new double[k, k];
        var Xty = new double[k];

        for (int j1 = 0; j1 < k; j1++)
        {
            int f1 = predictors[j1];
            for (int j2 = 0; j2 < k; j2++)
            {
                int f2 = predictors[j2];
                for (int i = 0; i < n; i++)
                    XtX[j1, j2] += X[i, f1] * X[i, f2];
            }
            for (int i = 0; i < n; i++)
                Xty[j1] += X[i, f1] * y[i];
        }

        // Regularization
        for (int j = 0; j < k; j++)
            XtX[j, j] += 1e-6;

        var beta = SolveSystem(XtX, Xty, k);

        // Compute R-squared
        double ssTot = 0, ssRes = 0;
        double yMean = y.Average();
        for (int i = 0; i < n; i++)
        {
            double pred = 0;
            for (int j = 0; j < k; j++)
                pred += beta[j] * X[i, predictors[j]];
            ssRes += (y[i] - pred) * (y[i] - pred);
            ssTot += (y[i] - yMean) * (y[i] - yMean);
        }

        double rSquared = ssTot > 1e-10 ? 1 - ssRes / ssTot : 0;
        rSquared = Math.Max(0, Math.Min(rSquared, 0.9999));

        return 1.0 / (1 - rSquared);
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

    protected override Matrix<T> TransformCore(Matrix<T> data)
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("VarianceInflationSelector has not been fitted.");

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
        throw new NotSupportedException("VarianceInflationSelector does not support inverse transformation.");
    }

    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("VarianceInflationSelector has not been fitted.");

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
