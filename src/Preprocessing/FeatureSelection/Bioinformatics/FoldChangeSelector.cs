using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.FeatureSelection.Bioinformatics;

/// <summary>
/// Fold Change-based feature selection for differential analysis.
/// </summary>
/// <remarks>
/// <para>
/// Selects features based on fold change (ratio of means) between two conditions.
/// Commonly used in gene expression analysis to identify differentially expressed genes.
/// </para>
/// <para><b>For Beginners:</b> Fold change measures how much a feature's average value
/// changes between two groups. A fold change of 2 means the feature is twice as high
/// in one group. Large fold changes indicate features that behave very differently
/// between conditions.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class FoldChangeSelector<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly int _nFeaturesToSelect;
    private readonly double _minFoldChange;
    private readonly bool _logTransform;

    private double[]? _foldChanges;
    private double[]? _log2FoldChanges;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    public int NFeaturesToSelect => _nFeaturesToSelect;
    public double[]? FoldChanges => _foldChanges;
    public double[]? Log2FoldChanges => _log2FoldChanges;
    public int[]? SelectedIndices => _selectedIndices;
    public override bool SupportsInverseTransform => false;

    public FoldChangeSelector(
        int nFeaturesToSelect = 100,
        double minFoldChange = 1.5,
        bool logTransform = true,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));
        if (minFoldChange < 1)
            throw new ArgumentException("Minimum fold change must be at least 1.", nameof(minFoldChange));

        _nFeaturesToSelect = nFeaturesToSelect;
        _minFoldChange = minFoldChange;
        _logTransform = logTransform;
    }

    protected override void FitCore(Matrix<T> data)
    {
        throw new InvalidOperationException(
            "FoldChangeSelector requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
    }

    public void Fit(Matrix<T> data, Vector<T> target)
    {
        if (data.Rows != target.Length)
            throw new ArgumentException("Target length must match rows in data.");

        _nInputFeatures = data.Columns;
        int n = data.Rows;
        int p = data.Columns;

        // Separate samples by class
        var class0 = new List<int>();
        var class1 = new List<int>();

        for (int i = 0; i < n; i++)
        {
            if (NumOps.ToDouble(target[i]) < 0.5)
                class0.Add(i);
            else
                class1.Add(i);
        }

        if (class0.Count == 0 || class1.Count == 0)
            throw new ArgumentException("Both classes must have at least one sample.");

        _foldChanges = new double[p];
        _log2FoldChanges = new double[p];

        for (int j = 0; j < p; j++)
        {
            // Compute means for each class
            double mean0 = 0, mean1 = 0;
            foreach (int i in class0)
                mean0 += NumOps.ToDouble(data[i, j]);
            foreach (int i in class1)
                mean1 += NumOps.ToDouble(data[i, j]);

            mean0 /= class0.Count;
            mean1 /= class1.Count;

            // Add small value to avoid division by zero
            double eps = 1e-10;
            mean0 = Math.Max(eps, mean0);
            mean1 = Math.Max(eps, mean1);

            // Compute fold change (larger / smaller)
            if (mean1 >= mean0)
            {
                _foldChanges[j] = mean1 / mean0;
                _log2FoldChanges[j] = Math.Log(mean1 / mean0) / Math.Log(2);
            }
            else
            {
                _foldChanges[j] = mean0 / mean1;
                _log2FoldChanges[j] = -Math.Log(mean0 / mean1) / Math.Log(2);
            }
        }

        // Select features with high fold change
        var candidateFeatures = new List<(int Index, double AbsFC)>();
        for (int j = 0; j < p; j++)
        {
            if (_foldChanges[j] >= _minFoldChange)
                candidateFeatures.Add((j, Math.Abs(_log2FoldChanges[j])));
        }

        if (candidateFeatures.Count >= _nFeaturesToSelect)
        {
            _selectedIndices = candidateFeatures
                .OrderByDescending(x => x.AbsFC)
                .Take(_nFeaturesToSelect)
                .Select(x => x.Index)
                .OrderBy(x => x)
                .ToArray();
        }
        else
        {
            // Fall back to top by fold change
            _selectedIndices = _foldChanges
                .Select((fc, idx) => (FC: fc, Index: idx))
                .OrderByDescending(x => x.FC)
                .Take(_nFeaturesToSelect)
                .Select(x => x.Index)
                .OrderBy(x => x)
                .ToArray();
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
            throw new InvalidOperationException("FoldChangeSelector has not been fitted.");

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
        throw new NotSupportedException("FoldChangeSelector does not support inverse transformation.");
    }

    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("FoldChangeSelector has not been fitted.");

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
