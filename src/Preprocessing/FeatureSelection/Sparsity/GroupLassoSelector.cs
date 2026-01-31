using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.FeatureSelection.Sparsity;

/// <summary>
/// Group Lasso based Feature Selection.
/// </summary>
/// <remarks>
/// <para>
/// Selects features using Group Lasso regularization, which selects or
/// removes entire groups of features together.
/// </para>
/// <para><b>For Beginners:</b> Group Lasso extends Lasso by treating features
/// in groups. Instead of selecting individual features, it selects entire groups
/// at once. This is useful when features naturally belong together, like one-hot
/// encoded categories or polynomial terms of the same variable.
/// </para>
/// </remarks>
public class GroupLassoSelector<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly int _nFeaturesToSelect;
    private readonly double _alpha;
    private readonly int _maxIterations;
    private readonly int[]? _groupAssignments;

    private double[]? _coefficients;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    public int NFeaturesToSelect => _nFeaturesToSelect;
    public double Alpha => _alpha;
    public double[]? Coefficients => _coefficients;
    public int[]? SelectedIndices => _selectedIndices;
    public override bool SupportsInverseTransform => false;

    public GroupLassoSelector(
        int nFeaturesToSelect = 10,
        double alpha = 0.1,
        int[]? groupAssignments = null,
        int maxIterations = 1000,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));
        if (alpha < 0)
            throw new ArgumentException("Alpha must be non-negative.", nameof(alpha));

        _nFeaturesToSelect = nFeaturesToSelect;
        _alpha = alpha;
        _groupAssignments = groupAssignments;
        _maxIterations = maxIterations;
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

        var X = new double[n, p];
        var y = new double[n];
        for (int i = 0; i < n; i++)
        {
            y[i] = NumOps.ToDouble(target[i]);
            for (int j = 0; j < p; j++)
                X[i, j] = NumOps.ToDouble(data[i, j]);
        }

        // Use provided groups or create individual groups
        var groups = _groupAssignments ?? Enumerable.Range(0, p).ToArray();

        // Standardize
        var means = new double[p];
        var stds = new double[p];
        for (int j = 0; j < p; j++)
        {
            double sum = 0;
            for (int i = 0; i < n; i++)
                sum += X[i, j];
            means[j] = sum / n;

            double sumSq = 0;
            for (int i = 0; i < n; i++)
            {
                double diff = X[i, j] - means[j];
                sumSq += diff * diff;
            }
            stds[j] = Math.Sqrt(sumSq / n);
            if (stds[j] < 1e-10) stds[j] = 1;

            for (int i = 0; i < n; i++)
                X[i, j] = (X[i, j] - means[j]) / stds[j];
        }

        // Group Lasso via block coordinate descent
        _coefficients = new double[p];
        var uniqueGroups = groups.Distinct().ToList();

        for (int iter = 0; iter < _maxIterations; iter++)
        {
            double maxChange = 0;

            foreach (var g in uniqueGroups)
            {
                var groupIndices = Enumerable.Range(0, p).Where(j => groups[j] == g).ToArray();
                int gSize = groupIndices.Length;
                if (gSize == 0) continue;

                // Compute gradient for group
                var gradient = new double[gSize];
                for (int gi = 0; gi < gSize; gi++)
                {
                    int j = groupIndices[gi];
                    for (int i = 0; i < n; i++)
                    {
                        double residual = y[i];
                        for (int k = 0; k < p; k++)
                            residual -= X[i, k] * _coefficients[k];
                        residual += X[i, j] * _coefficients[j];
                        gradient[gi] += X[i, j] * residual;
                    }
                }

                // Group L2 norm
                double gradNorm = Math.Sqrt(gradient.Sum(g => g * g));
                double shrinkage = _alpha * n * Math.Sqrt(gSize);

                // Group soft thresholding
                if (gradNorm > shrinkage)
                {
                    double scale = (gradNorm - shrinkage) / gradNorm;
                    for (int gi = 0; gi < gSize; gi++)
                    {
                        int j = groupIndices[gi];
                        double xNorm = 0;
                        for (int i = 0; i < n; i++)
                            xNorm += X[i, j] * X[i, j];
                        if (xNorm < 1e-10) continue;

                        double oldCoef = _coefficients[j];
                        _coefficients[j] = gradient[gi] * scale / xNorm;
                        maxChange = Math.Max(maxChange, Math.Abs(_coefficients[j] - oldCoef));
                    }
                }
                else
                {
                    for (int gi = 0; gi < gSize; gi++)
                    {
                        int j = groupIndices[gi];
                        maxChange = Math.Max(maxChange, Math.Abs(_coefficients[j]));
                        _coefficients[j] = 0;
                    }
                }
            }

            if (maxChange < 1e-6)
                break;
        }

        int numToSelect = Math.Min(_nFeaturesToSelect, p);
        _selectedIndices = Enumerable.Range(0, p)
            .OrderByDescending(j => Math.Abs(_coefficients[j]))
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
        if (_selectedIndices is null) return [];

        if (inputFeatureNames is null)
            return _selectedIndices.Select(i => $"Feature{i}").ToArray();

        return _selectedIndices
            .Where(i => i < inputFeatureNames.Length)
            .Select(i => inputFeatureNames[i])
            .ToArray();
    }
}
