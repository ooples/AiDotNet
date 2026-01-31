using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.FeatureSelection.Sparse;

/// <summary>
/// Sparse Group Lasso for feature selection with grouped structure.
/// </summary>
/// <remarks>
/// <para>
/// Sparse Group Lasso combines L1 (Lasso) and L2 (Group Lasso) penalties to select
/// both entire groups of features and individual features within groups. This is useful
/// when features have a natural grouping structure.
/// </para>
/// <para><b>For Beginners:</b> Imagine features organized into categories (e.g., color
/// features, shape features). Sometimes you want to select entire categories, other times
/// specific features within categories. Sparse Group Lasso does both: it can eliminate
/// entire groups or pick individual features within groups.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class SparseGroupLasso<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly int _nFeaturesToSelect;
    private readonly double _alpha;
    private readonly double _l1Ratio;
    private readonly int[] _groups;
    private readonly int _maxIterations;
    private readonly double _tolerance;

    private double[]? _coefficients;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    public int NFeaturesToSelect => _nFeaturesToSelect;
    public double Alpha => _alpha;
    public double L1Ratio => _l1Ratio;
    public int[] Groups => _groups;
    public double[]? Coefficients => _coefficients;
    public int[]? SelectedIndices => _selectedIndices;
    public override bool SupportsInverseTransform => false;

    public SparseGroupLasso(
        int nFeaturesToSelect = 10,
        double alpha = 1.0,
        double l1Ratio = 0.5,
        int[]? groups = null,
        int maxIterations = 1000,
        double tolerance = 1e-4,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));

        _nFeaturesToSelect = nFeaturesToSelect;
        _alpha = alpha;
        _l1Ratio = l1Ratio;
        _groups = groups ?? [];
        _maxIterations = maxIterations;
        _tolerance = tolerance;
    }

    protected override void FitCore(Matrix<T> data)
    {
        throw new InvalidOperationException(
            "SparseGroupLasso requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
    }

    public void Fit(Matrix<T> data, Vector<T> target)
    {
        if (data.Rows != target.Length)
            throw new ArgumentException("Target length must match rows in data.");

        _nInputFeatures = data.Columns;
        int n = data.Rows;
        int p = data.Columns;

        // Create default groups if not provided (each feature is its own group)
        var groups = _groups.Length == p ? _groups : Enumerable.Range(0, p).ToArray();

        // Get unique groups
        var uniqueGroups = groups.Distinct().OrderBy(x => x).ToArray();
        var groupIndices = new Dictionary<int, List<int>>();
        foreach (int g in uniqueGroups)
            groupIndices[g] = [];
        for (int j = 0; j < p; j++)
            groupIndices[groups[j]].Add(j);

        // Standardize features
        var means = new double[p];
        var stds = new double[p];
        var X = new double[n, p];

        for (int j = 0; j < p; j++)
        {
            double mean = 0;
            for (int i = 0; i < n; i++)
                mean += NumOps.ToDouble(data[i, j]);
            mean /= n;
            means[j] = mean;

            double variance = 0;
            for (int i = 0; i < n; i++)
            {
                double diff = NumOps.ToDouble(data[i, j]) - mean;
                variance += diff * diff;
            }
            double std = Math.Sqrt(variance / n);
            stds[j] = std > 1e-10 ? std : 1.0;

            for (int i = 0; i < n; i++)
                X[i, j] = (NumOps.ToDouble(data[i, j]) - mean) / stds[j];
        }

        // Standardize target
        double yMean = 0;
        for (int i = 0; i < n; i++)
            yMean += NumOps.ToDouble(target[i]);
        yMean /= n;

        var y = new double[n];
        for (int i = 0; i < n; i++)
            y[i] = NumOps.ToDouble(target[i]) - yMean;

        // Initialize coefficients
        _coefficients = new double[p];
        var residuals = (double[])y.Clone();

        double l1Weight = _alpha * _l1Ratio;
        double l2Weight = _alpha * (1 - _l1Ratio);

        // Block coordinate descent
        for (int iter = 0; iter < _maxIterations; iter++)
        {
            double maxChange = 0;

            foreach (int g in uniqueGroups)
            {
                var groupIdx = groupIndices[g];
                int groupSize = groupIdx.Count;

                // Compute group gradient
                var groupGrad = new double[groupSize];
                for (int k = 0; k < groupSize; k++)
                {
                    int j = groupIdx[k];
                    double oldCoef = _coefficients[j];

                    // Add back contribution
                    for (int i = 0; i < n; i++)
                        residuals[i] += X[i, j] * oldCoef;

                    // Compute gradient
                    double grad = 0;
                    for (int i = 0; i < n; i++)
                        grad += X[i, j] * residuals[i];
                    groupGrad[k] = grad / n;
                }

                // Group Lasso update
                double groupNorm = Math.Sqrt(groupGrad.Sum(g => g * g));
                double groupPenalty = l2Weight * Math.Sqrt(groupSize);

                if (groupNorm <= groupPenalty)
                {
                    // Shrink entire group to zero
                    foreach (int j in groupIdx)
                    {
                        maxChange = Math.Max(maxChange, Math.Abs(_coefficients[j]));
                        _coefficients[j] = 0;
                    }
                }
                else
                {
                    // Scale down group and apply L1 penalty
                    double scale = (groupNorm - groupPenalty) / groupNorm;

                    for (int k = 0; k < groupSize; k++)
                    {
                        int j = groupIdx[k];
                        double scaledGrad = groupGrad[k] * scale;

                        // L1 soft thresholding
                        double newCoef;
                        if (scaledGrad > l1Weight)
                            newCoef = scaledGrad - l1Weight;
                        else if (scaledGrad < -l1Weight)
                            newCoef = scaledGrad + l1Weight;
                        else
                            newCoef = 0;

                        maxChange = Math.Max(maxChange, Math.Abs(newCoef - _coefficients[j]));
                        _coefficients[j] = newCoef;
                    }
                }

                // Update residuals
                foreach (int j in groupIdx)
                {
                    for (int i = 0; i < n; i++)
                        residuals[i] -= X[i, j] * _coefficients[j];
                }
            }

            if (maxChange < _tolerance)
                break;
        }

        // Select features by absolute coefficient magnitude
        int numToSelect = Math.Min(_nFeaturesToSelect, p);
        _selectedIndices = _coefficients
            .Select((c, idx) => (Coef: Math.Abs(c), Index: idx))
            .OrderByDescending(x => x.Coef)
            .Take(numToSelect)
            .Select(x => x.Index)
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
            throw new InvalidOperationException("SparseGroupLasso has not been fitted.");

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
        throw new NotSupportedException("SparseGroupLasso does not support inverse transformation.");
    }

    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("SparseGroupLasso has not been fitted.");

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
