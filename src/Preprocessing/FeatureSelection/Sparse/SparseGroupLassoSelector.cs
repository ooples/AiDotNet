using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.FeatureSelection.Sparse;

/// <summary>
/// Sparse Group Lasso Feature Selection.
/// </summary>
/// <remarks>
/// <para>
/// Combines Lasso (individual feature sparsity) with Group Lasso (group sparsity)
/// to select features that are both individually and group-wise important.
/// </para>
/// <para><b>For Beginners:</b> Sometimes features naturally come in groups
/// (like one-hot encoded categories or polynomial terms). This method can select
/// entire groups together while also selecting individual features within groups,
/// giving you the best of both worlds.
/// </para>
/// </remarks>
public class SparseGroupLassoSelector<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly int _nFeaturesToSelect;
    private readonly double _alpha;
    private readonly double _groupRatio;
    private readonly int[]? _groupIndices;

    private double[]? _coefficients;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    public int NFeaturesToSelect => _nFeaturesToSelect;
    public double Alpha => _alpha;
    public double GroupRatio => _groupRatio;
    public double[]? Coefficients => _coefficients;
    public int[]? SelectedIndices => _selectedIndices;
    public override bool SupportsInverseTransform => false;

    public SparseGroupLassoSelector(
        int nFeaturesToSelect = 10,
        double alpha = 0.01,
        double groupRatio = 0.5,
        int[]? groupIndices = null,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));
        if (groupRatio < 0 || groupRatio > 1)
            throw new ArgumentException("Group ratio must be between 0 and 1.", nameof(groupRatio));

        _nFeaturesToSelect = nFeaturesToSelect;
        _alpha = alpha;
        _groupRatio = groupRatio;
        _groupIndices = groupIndices;
    }

    protected override void FitCore(Matrix<T> data)
    {
        throw new InvalidOperationException(
            "SparseGroupLassoSelector requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
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

        // Normalize features
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

        double yMean = y.Average();
        var yNorm = y.Select(yi => yi - yMean).ToArray();

        // Create groups (default: each feature is its own group)
        var groups = CreateGroups(p);

        // Coordinate descent with sparse group penalty
        _coefficients = new double[p];
        double l1Penalty = _alpha * (1 - _groupRatio);
        double groupPenalty = _alpha * _groupRatio;

        int maxIter = 100;
        for (int iter = 0; iter < maxIter; iter++)
        {
            for (int j = 0; j < p; j++)
            {
                // Compute residual
                double rho = 0;
                for (int i = 0; i < n; i++)
                {
                    double residual = yNorm[i];
                    for (int k = 0; k < p; k++)
                        if (k != j)
                            residual -= _coefficients[k] * X[i, k];
                    rho += X[i, j] * residual;
                }
                rho /= n;

                // Soft thresholding with group penalty
                int groupIdx = groups[j];
                double groupNorm = ComputeGroupNorm(_coefficients, groups, groupIdx, j);
                double effectivePenalty = l1Penalty + groupPenalty / (Math.Sqrt(groupNorm) + 1e-10);

                _coefficients[j] = SoftThreshold(rho, effectivePenalty);
            }
        }

        // Select features with non-zero coefficients
        var nonZero = Enumerable.Range(0, p)
            .Where(j => Math.Abs(_coefficients[j]) > 1e-10)
            .OrderByDescending(j => Math.Abs(_coefficients[j]))
            .ToList();

        int numToSelect = Math.Min(_nFeaturesToSelect, Math.Max(nonZero.Count, 1));
        if (nonZero.Count < numToSelect)
        {
            var additional = Enumerable.Range(0, p)
                .Where(j => !nonZero.Contains(j))
                .OrderByDescending(j => Math.Abs(_coefficients[j]))
                .Take(numToSelect - nonZero.Count);
            nonZero.AddRange(additional);
        }

        _selectedIndices = nonZero.Take(numToSelect).OrderBy(x => x).ToArray();
        IsFitted = true;
    }

    private int[] CreateGroups(int p)
    {
        if (_groupIndices is not null && _groupIndices.Length == p)
            return _groupIndices;

        // Default: each feature is its own group
        return Enumerable.Range(0, p).ToArray();
    }

    private double ComputeGroupNorm(double[] beta, int[] groups, int targetGroup, int excludeIdx)
    {
        double norm = 0;
        for (int j = 0; j < beta.Length; j++)
        {
            if (groups[j] == targetGroup && j != excludeIdx)
                norm += beta[j] * beta[j];
        }
        return norm + 1e-10;
    }

    private double SoftThreshold(double x, double lambda)
    {
        if (x > lambda) return x - lambda;
        if (x < -lambda) return x + lambda;
        return 0;
    }

    public Matrix<T> FitTransform(Matrix<T> data, Vector<T> target)
    {
        Fit(data, target);
        return Transform(data);
    }

    protected override Matrix<T> TransformCore(Matrix<T> data)
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("SparseGroupLassoSelector has not been fitted.");

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
        throw new NotSupportedException("SparseGroupLassoSelector does not support inverse transformation.");
    }

    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("SparseGroupLassoSelector has not been fitted.");

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
