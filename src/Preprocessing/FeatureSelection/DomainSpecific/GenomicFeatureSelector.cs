using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.FeatureSelection.DomainSpecific;

/// <summary>
/// Feature selection for genomic/bioinformatics data.
/// </summary>
/// <remarks>
/// <para>
/// GenomicFeatureSelector is designed for high-dimensional genomic data such as
/// gene expression profiles, SNPs, or methylation data. It addresses the "large p,
/// small n" problem common in genomics and can incorporate gene grouping.
/// </para>
/// <para><b>For Beginners:</b> Genomic data often has millions of features (genes)
/// but only a few hundred samples (patients). This selector uses techniques
/// specifically designed for this challenging scenario, finding the few genes
/// that truly matter for distinguishing between conditions.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class GenomicFeatureSelector<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly int _nFeaturesToSelect;
    private readonly double _variancePercentile;
    private readonly bool _useFoldChangeFilter;
    private readonly double _foldChangeThreshold;

    private double[]? _featureScores;
    private double[]? _foldChanges;
    private double[]? _variances;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    public int NFeaturesToSelect => _nFeaturesToSelect;
    public double VariancePercentile => _variancePercentile;
    public bool UseFoldChangeFilter => _useFoldChangeFilter;
    public double FoldChangeThreshold => _foldChangeThreshold;
    public double[]? FeatureScores => _featureScores;
    public double[]? FoldChanges => _foldChanges;
    public double[]? Variances => _variances;
    public int[]? SelectedIndices => _selectedIndices;
    public override bool SupportsInverseTransform => false;

    public GenomicFeatureSelector(
        int nFeaturesToSelect = 100,
        double variancePercentile = 0.5,
        bool useFoldChangeFilter = true,
        double foldChangeThreshold = 1.5,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));
        if (variancePercentile < 0 || variancePercentile > 1)
            throw new ArgumentException("Variance percentile must be between 0 and 1.", nameof(variancePercentile));
        if (foldChangeThreshold < 0)
            throw new ArgumentException("Fold change threshold must be non-negative.", nameof(foldChangeThreshold));

        _nFeaturesToSelect = nFeaturesToSelect;
        _variancePercentile = variancePercentile;
        _useFoldChangeFilter = useFoldChangeFilter;
        _foldChangeThreshold = foldChangeThreshold;
    }

    protected override void FitCore(Matrix<T> data)
    {
        throw new InvalidOperationException(
            "GenomicFeatureSelector requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
    }

    public void Fit(Matrix<T> data, Vector<T> target)
    {
        if (data.Rows != target.Length)
            throw new ArgumentException("Target length must match rows in data.");
        if (data.Rows == 0 || data.Columns == 0)
            throw new ArgumentException("Data must contain at least one row and one column.", nameof(data));

        _nInputFeatures = data.Columns;
        int n = data.Rows;
        int p = data.Columns;

        // Split samples by class
        var class0 = new List<int>();
        var class1 = new List<int>();
        for (int i = 0; i < n; i++)
        {
            if (NumOps.ToDouble(target[i]) < 0.5)
                class0.Add(i);
            else
                class1.Add(i);
        }

        // Step 1: Compute variances and filter by percentile
        _variances = new double[p];
        for (int j = 0; j < p; j++)
        {
            double mean = 0;
            for (int i = 0; i < n; i++)
                mean += NumOps.ToDouble(data[i, j]);
            mean /= n;

            double variance = 0;
            for (int i = 0; i < n; i++)
            {
                double diff = NumOps.ToDouble(data[i, j]) - mean;
                variance += diff * diff;
            }
            _variances[j] = variance / n;
        }

        var sortedVariances = _variances.OrderBy(v => v).ToArray();
        int varianceIndex = (int)(p * _variancePercentile);
        if (varianceIndex >= p) varianceIndex = p - 1;
        if (varianceIndex < 0) varianceIndex = 0;
        double varianceThreshold = sortedVariances[varianceIndex];

        var candidates = new List<int>();
        for (int j = 0; j < p; j++)
        {
            if (_variances[j] >= varianceThreshold)
                candidates.Add(j);
        }

        // Step 2: Compute fold changes if binary classification
        _foldChanges = new double[p];
        if (class0.Count > 0 && class1.Count > 0 && _useFoldChangeFilter)
        {
            for (int j = 0; j < p; j++)
            {
                double mean0 = class0.Sum(i => NumOps.ToDouble(data[i, j])) / class0.Count;
                double mean1 = class1.Sum(i => NumOps.ToDouble(data[i, j])) / class1.Count;

                // Fold change (use log2 fold change)
                if (mean0 > 1e-10 && mean1 > 1e-10)
                    _foldChanges[j] = Math.Abs(Math.Log(mean1 / mean0) / Math.Log(2));
                else if (mean0 > 1e-10 || mean1 > 1e-10)
                    _foldChanges[j] = 10; // High value for zeros
                else
                    _foldChanges[j] = 0;
            }

            // Filter by fold change
            candidates = candidates
                .Where(j => _foldChanges[j] >= Math.Log(_foldChangeThreshold) / Math.Log(2))
                .ToList();
        }

        // Step 3: Compute t-statistic for remaining candidates
        _featureScores = new double[p];
        foreach (int j in candidates)
        {
            if (class0.Count >= 2 && class1.Count >= 2)
            {
                double mean0 = class0.Sum(i => NumOps.ToDouble(data[i, j])) / class0.Count;
                double mean1 = class1.Sum(i => NumOps.ToDouble(data[i, j])) / class1.Count;

                double var0 = class0.Sum(i => Math.Pow(NumOps.ToDouble(data[i, j]) - mean0, 2)) / (class0.Count - 1);
                double var1 = class1.Sum(i => Math.Pow(NumOps.ToDouble(data[i, j]) - mean1, 2)) / (class1.Count - 1);

                double se = Math.Sqrt(var0 / class0.Count + var1 / class1.Count);
                _featureScores[j] = se > 1e-10 ? Math.Abs(mean0 - mean1) / se : 0;
            }
            else
            {
                _featureScores[j] = _variances[j];
            }
        }

        // Select top features
        int numToSelect = Math.Min(_nFeaturesToSelect, candidates.Count);
        _selectedIndices = candidates
            .OrderByDescending(j => _featureScores[j])
            .Take(numToSelect)
            .OrderBy(x => x)
            .ToArray();

        // If not enough candidates, add high-variance features
        if (_selectedIndices.Length < _nFeaturesToSelect)
        {
            var remaining = Enumerable.Range(0, p)
                .Where(j => !candidates.Contains(j))
                .OrderByDescending(j => _variances[j])
                .Take(_nFeaturesToSelect - _selectedIndices.Length);

            _selectedIndices = _selectedIndices.Concat(remaining).OrderBy(x => x).ToArray();
        }

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
            throw new InvalidOperationException("GenomicFeatureSelector has not been fitted.");

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
        throw new NotSupportedException("GenomicFeatureSelector does not support inverse transformation.");
    }

    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("GenomicFeatureSelector has not been fitted.");

        var mask = new bool[_nInputFeatures];
        foreach (int idx in _selectedIndices)
            mask[idx] = true;

        return mask;
    }

    public override string[] GetFeatureNamesOut(string[]? inputFeatureNames = null)
    {
        if (_selectedIndices is null) return [];

        if (inputFeatureNames is null)
            return _selectedIndices.Select(i => $"Gene{i}").ToArray();

        return _selectedIndices
            .Where(i => i < inputFeatureNames.Length)
            .Select(i => inputFeatureNames[i])
            .ToArray();
    }
}
