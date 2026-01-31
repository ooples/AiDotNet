using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.FeatureSelection.Embedded;

/// <summary>
/// Group Lasso feature selection for selecting groups of related features.
/// </summary>
/// <remarks>
/// <para>
/// Extends Lasso to select entire groups of features together rather than individual
/// features. Useful when features naturally belong to groups (e.g., dummy variables
/// for a categorical feature, or related gene pathways).
/// </para>
/// <para><b>For Beginners:</b> Sometimes features come in natural groups. For example,
/// if you have a categorical variable converted to multiple dummy columns, you want
/// to keep or remove ALL columns for that category together. Group Lasso ensures
/// related features are selected or dropped as a unit.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class GroupLassoSelector<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly int _nGroupsToSelect;
    private readonly int[][]? _groups;
    private readonly double _alpha;
    private readonly int _maxIterations;
    private readonly double _tolerance;

    private double[]? _groupNorms;
    private int[]? _selectedGroups;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    public int NGroupsToSelect => _nGroupsToSelect;
    public double[]? GroupNorms => _groupNorms;
    public int[]? SelectedGroups => _selectedGroups;
    public int[]? SelectedIndices => _selectedIndices;
    public override bool SupportsInverseTransform => false;

    public GroupLassoSelector(
        int nGroupsToSelect = 5,
        int[][]? groups = null,
        double alpha = 1.0,
        int maxIterations = 1000,
        double tolerance = 1e-4,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nGroupsToSelect < 1)
            throw new ArgumentException("Number of groups must be at least 1.", nameof(nGroupsToSelect));

        _nGroupsToSelect = nGroupsToSelect;
        _groups = groups;
        _alpha = alpha;
        _maxIterations = maxIterations;
        _tolerance = tolerance;
    }

    protected override void FitCore(Matrix<T> data)
    {
        throw new InvalidOperationException(
            "GroupLassoSelector requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
    }

    public void Fit(Matrix<T> data, Vector<T> target)
    {
        if (data.Rows != target.Length)
            throw new ArgumentException("Target length must match rows in data.");

        _nInputFeatures = data.Columns;
        int n = data.Rows;
        int p = data.Columns;

        // Create default groups if not provided (one feature per group)
        var groups = _groups ?? Enumerable.Range(0, p).Select(i => new[] { i }).ToArray();
        int nGroups = groups.Length;

        // Standardize data
        var X = new double[n, p];
        var y = new double[n];
        var means = new double[p];
        var stds = new double[p];

        double yMean = 0;
        for (int i = 0; i < n; i++)
            yMean += NumOps.ToDouble(target[i]);
        yMean /= n;

        for (int j = 0; j < p; j++)
        {
            for (int i = 0; i < n; i++)
                means[j] += NumOps.ToDouble(data[i, j]);
            means[j] /= n;

            for (int i = 0; i < n; i++)
                stds[j] += Math.Pow(NumOps.ToDouble(data[i, j]) - means[j], 2);
            stds[j] = Math.Sqrt(stds[j] / n);
            if (stds[j] < 1e-10) stds[j] = 1;
        }

        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < p; j++)
                X[i, j] = (NumOps.ToDouble(data[i, j]) - means[j]) / stds[j];
            y[i] = NumOps.ToDouble(target[i]) - yMean;
        }

        // Block coordinate descent
        var beta = new double[p];
        var residuals = new double[n];
        Array.Copy(y, residuals, n);

        for (int iter = 0; iter < _maxIterations; iter++)
        {
            double maxChange = 0;

            foreach (var group in groups)
            {
                int groupSize = group.Length;

                // Compute gradient for this group
                var gradient = new double[groupSize];
                for (int g = 0; g < groupSize; g++)
                {
                    int j = group[g];
                    for (int i = 0; i < n; i++)
                        gradient[g] += X[i, j] * residuals[i];
                }

                // Compute old group coefficient norm
                double oldNorm = 0;
                for (int g = 0; g < groupSize; g++)
                    oldNorm += beta[group[g]] * beta[group[g]];
                oldNorm = Math.Sqrt(oldNorm);

                // Group soft thresholding
                double gradNorm = 0;
                for (int g = 0; g < groupSize; g++)
                {
                    double val = beta[group[g]] + gradient[g] / n;
                    gradNorm += val * val;
                }
                gradNorm = Math.Sqrt(gradNorm);

                double scale = Math.Max(0, 1 - _alpha * Math.Sqrt(groupSize) / (gradNorm + 1e-10));

                // Update coefficients
                for (int g = 0; g < groupSize; g++)
                {
                    double oldBeta = beta[group[g]];
                    beta[group[g]] = scale * (beta[group[g]] + gradient[g] / n);

                    // Update residuals
                    for (int i = 0; i < n; i++)
                        residuals[i] -= (beta[group[g]] - oldBeta) * X[i, group[g]];

                    maxChange = Math.Max(maxChange, Math.Abs(beta[group[g]] - oldBeta));
                }
            }

            if (maxChange < _tolerance)
                break;
        }

        // Compute group norms
        _groupNorms = new double[nGroups];
        for (int g = 0; g < nGroups; g++)
        {
            foreach (int j in groups[g])
                _groupNorms[g] += beta[j] * beta[j];
            _groupNorms[g] = Math.Sqrt(_groupNorms[g]);
        }

        // Select top groups
        int nToSelect = Math.Min(_nGroupsToSelect, nGroups);
        _selectedGroups = _groupNorms
            .Select((norm, idx) => (Norm: norm, Index: idx))
            .OrderByDescending(x => x.Norm)
            .Take(nToSelect)
            .Select(x => x.Index)
            .OrderBy(x => x)
            .ToArray();

        // Collect all selected feature indices
        var selectedSet = new HashSet<int>();
        foreach (int g in _selectedGroups)
            foreach (int j in groups[g])
                selectedSet.Add(j);

        _selectedIndices = selectedSet.OrderBy(x => x).ToArray();
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
            throw new InvalidOperationException("GroupLassoSelector has not been fitted.");

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
        throw new NotSupportedException("GroupLassoSelector does not support inverse transformation.");
    }

    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("GroupLassoSelector has not been fitted.");

        var mask = new bool[_nInputFeatures];
        foreach (int idx in _selectedIndices)
            mask[idx] = true;

        return mask;
    }

    public override string[] GetFeatureNamesOut(string[]? inputFeatureNames = null)
    {
        if (_selectedIndices is null) return Array.Empty<string>();

        if (inputFeatureNames is null)
            return _selectedIndices.Select(i => $"Feature{i}").ToArray();

        return _selectedIndices
            .Where(i => i < inputFeatureNames.Length)
            .Select(i => inputFeatureNames[i])
            .ToArray();
    }
}
