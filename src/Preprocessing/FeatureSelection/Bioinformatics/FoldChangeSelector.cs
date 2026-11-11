using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Enums;
using AiDotNet.Attributes;

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
[ComponentType(ComponentType.FeatureSelector)]
[PipelineStage(PipelineStage.Preprocessing)]
public class FoldChangeSelector<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly int _nFeaturesToSelect;
    private readonly double _minFoldChange;
    private readonly bool _logTransform;

    private Vector<T>? _foldChanges;
    private Vector<T>? _log2FoldChanges;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    public int NFeaturesToSelect => _nFeaturesToSelect;
    public Vector<T>? FoldChanges => _foldChanges;
    public Vector<T>? Log2FoldChanges => _log2FoldChanges;
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

        var engine = Engine;
        var X = Tensor<T>.FromMatrix(data);

        // Separate samples by class
        var class1Mask = PreprocessingTensorOps.Not(engine.TensorLessThan(Tensor<T>.FromVector(target), NumOps.FromDouble(0.5)));
        var class0Mask = PreprocessingTensorOps.Not(class1Mask);
        int class1Count = PreprocessingTensorOps.CountNonZero(class1Mask);
        int class0Count = n - class1Count;

        if (class0Count == 0 || class1Count == 0)
            throw new ArgumentException("Both classes must have at least one sample.");

        // Class means, floored at a small value to avoid division by zero
        T eps = NumOps.FromDouble(1e-10);
        var mean0 = engine.TensorMax(engine.TensorDivideScalar(
            engine.Reshape(engine.TensorMatMul(engine.Reshape(class0Mask, new[] { 1, n }), X), new[] { p }),
            NumOps.FromDouble(class0Count)), eps);
        var mean1 = engine.TensorMax(engine.TensorDivideScalar(
            engine.Reshape(engine.TensorMatMul(engine.Reshape(class1Mask, new[] { 1, n }), X), new[] { p }),
            NumOps.FromDouble(class1Count)), eps);

        // Fold change (larger / smaller) and its signed log2
        T ln2 = NumOps.FromDouble(Math.Log(2));
        var up = PreprocessingTensorOps.Not(engine.TensorLessThan(mean1, mean0));
        var upRatio = engine.TensorDivide(mean1, mean0);
        var downRatio = engine.TensorDivide(mean0, mean1);
        var foldChanges = engine.TensorWhere(up, upRatio, downRatio);
        var log2FoldChanges = engine.TensorWhere(up,
            engine.TensorDivideScalar(engine.TensorLog(upRatio), ln2),
            engine.TensorNegate(engine.TensorDivideScalar(engine.TensorLog(downRatio), ln2)));
        _foldChanges = foldChanges.ToVector();
        _log2FoldChanges = log2FoldChanges.ToVector();

        // Select features with high fold change
        var candidates = PreprocessingTensorOps.AtLeast(foldChanges, NumOps.FromDouble(_minFoldChange));
        if (PreprocessingTensorOps.CountNonZero(candidates) >= _nFeaturesToSelect)
        {
            var ranked = engine.TensorWhere(candidates, engine.TensorAbs(log2FoldChanges),
                PreprocessingTensorOps.Filled(new[] { p }, NumOps.FromDouble(-1.0)));
            _selectedIndices = PreprocessingTensorOps.TopKIndicesSorted(ranked, _nFeaturesToSelect);
        }
        else
        {
            // Fall back to top by fold change
            _selectedIndices = PreprocessingTensorOps.TopKIndicesSorted(foldChanges, _nFeaturesToSelect);
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

        return PreprocessingTensorOps.SelectColumns(data, _selectedIndices);
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
