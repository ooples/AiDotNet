using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.FeatureSelection.Regression;

/// <summary>
/// Least Angle Regression (LARS) based Feature Selection.
/// </summary>
/// <remarks>
/// <para>
/// Uses LARS algorithm to select features by progressively adding features
/// along the direction equiangular to all active features.
/// </para>
/// <para><b>For Beginners:</b> LARS is like a compromise between forward selection
/// and Lasso. It adds features gradually, moving in a direction that's equally
/// correlated with all currently active features. This gives a natural ordering
/// of feature importance.
/// </para>
/// </remarks>
public class LARSSelector<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly int _nFeaturesToSelect;

    private double[]? _featureRanks;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    public int NFeaturesToSelect => _nFeaturesToSelect;
    public double[]? FeatureRanks => _featureRanks;
    public int[]? SelectedIndices => _selectedIndices;
    public override bool SupportsInverseTransform => false;

    public LARSSelector(
        int nFeaturesToSelect = 10,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));

        _nFeaturesToSelect = nFeaturesToSelect;
    }

    protected override void FitCore(Matrix<T> data)
    {
        throw new InvalidOperationException(
            "LARSSelector requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
    }

    public void Fit(Matrix<T> data, Vector<T> target)
    {
        if (data.Rows != target.Length)
            throw new ArgumentException("Target length must match rows in data.");

        _nInputFeatures = data.Columns;
        int n = data.Rows;
        int p = data.Columns;

        var X = new double[n, p];
        var y = new double[n];
        for (int i = 0; i < n; i++)
        {
            y[i] = NumOps.ToDouble(target[i]);
            for (int j = 0; j < p; j++)
                X[i, j] = NumOps.ToDouble(data[i, j]);
        }

        // Standardize
        var means = new double[p];
        var stds = new double[p];
        double yMean = y.Average();

        for (int j = 0; j < p; j++)
        {
            for (int i = 0; i < n; i++)
                means[j] += X[i, j];
            means[j] /= n;

            for (int i = 0; i < n; i++)
                stds[j] += (X[i, j] - means[j]) * (X[i, j] - means[j]);
            stds[j] = Math.Sqrt(stds[j] / n);
            if (stds[j] < 1e-10) stds[j] = 1;

            for (int i = 0; i < n; i++)
                X[i, j] = (X[i, j] - means[j]) / stds[j];
        }

        for (int i = 0; i < n; i++)
            y[i] -= yMean;

        // Simplified LARS-like forward selection
        var residual = (double[])y.Clone();
        var active = new HashSet<int>();
        var inactive = new HashSet<int>(Enumerable.Range(0, p));
        _featureRanks = new double[p];

        int numToSelect = Math.Min(_nFeaturesToSelect, p);
        var selectionOrder = new List<int>();

        while (selectionOrder.Count < numToSelect && inactive.Count > 0)
        {
            // Find feature most correlated with residual
            int bestFeature = -1;
            double maxCorr = 0;

            foreach (int j in inactive)
            {
                double corr = ComputeCorrelation(X, residual, j, n);
                if (Math.Abs(corr) > Math.Abs(maxCorr))
                {
                    maxCorr = corr;
                    bestFeature = j;
                }
            }

            if (bestFeature < 0) break;

            selectionOrder.Add(bestFeature);
            _featureRanks[bestFeature] = p - selectionOrder.Count + 1;
            active.Add(bestFeature);
            inactive.Remove(bestFeature);

            // Update residual by removing projection onto active set
            UpdateResidual(X, residual, active, n);
        }

        _selectedIndices = selectionOrder.Take(numToSelect).OrderBy(x => x).ToArray();

        IsFitted = true;
    }

    private double ComputeCorrelation(double[,] X, double[] y, int j, int n)
    {
        double xMean = 0, yMean = 0;
        for (int i = 0; i < n; i++)
        {
            xMean += X[i, j];
            yMean += y[i];
        }
        xMean /= n;
        yMean /= n;

        double sxy = 0, sxx = 0, syy = 0;
        for (int i = 0; i < n; i++)
        {
            double xd = X[i, j] - xMean;
            double yd = y[i] - yMean;
            sxy += xd * yd;
            sxx += xd * xd;
            syy += yd * yd;
        }

        return (sxx > 1e-10 && syy > 1e-10) ? sxy / Math.Sqrt(sxx * syy) : 0;
    }

    private void UpdateResidual(double[,] X, double[] residual, HashSet<int> active, int n)
    {
        if (active.Count == 0) return;

        var activeList = active.ToList();
        int k = activeList.Count;

        // Compute projection matrix and update residual
        var XtX = new double[k, k];
        var Xty = new double[k];

        for (int i = 0; i < k; i++)
        {
            int ji = activeList[i];
            for (int j = 0; j < k; j++)
            {
                int jj = activeList[j];
                for (int row = 0; row < n; row++)
                    XtX[i, j] += X[row, ji] * X[row, jj];
            }
            for (int row = 0; row < n; row++)
                Xty[i] += X[row, ji] * residual[row];
        }

        // Regularize
        for (int i = 0; i < k; i++)
            XtX[i, i] += 1e-6;

        // Solve for coefficients
        var beta = SolveSystem(XtX, Xty, k);

        // Compute fitted values and update residual
        for (int row = 0; row < n; row++)
        {
            double fitted = 0;
            for (int i = 0; i < k; i++)
                fitted += beta[i] * X[row, activeList[i]];
            residual[row] -= fitted;
        }
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
            throw new InvalidOperationException("LARSSelector has not been fitted.");

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
        throw new NotSupportedException("LARSSelector does not support inverse transformation.");
    }

    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("LARSSelector has not been fitted.");

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
