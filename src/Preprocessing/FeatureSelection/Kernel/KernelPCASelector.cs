using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Enums;
using AiDotNet.Attributes;
using AiDotNet.Preprocessing.FeatureSelection.Helpers;

namespace AiDotNet.Preprocessing.FeatureSelection.Kernel;

/// <summary>
/// Kernel PCA-based Feature Selection.
/// </summary>
/// <remarks>
/// <para>
/// Uses kernel PCA to map features to a higher-dimensional space and selects
/// features that contribute most to the principal components in kernel space.
/// </para>
/// <para><b>For Beginners:</b> Regular PCA finds linear patterns. Kernel PCA can
/// find non-linear patterns by implicitly mapping data to a higher dimension.
/// We select features that are most important in this richer representation,
/// capturing complex relationships that linear methods might miss.
/// </para>
/// </remarks>
[ComponentType(ComponentType.FeatureSelector)]
[PipelineStage(PipelineStage.Preprocessing)]
public class KernelPCASelector<T> : TransformerBase<T, Matrix<T>, Matrix<T>>, IFixedCountFeatureSelector<T>
{
    private readonly int _nFeaturesToSelect;
    private readonly double _gamma;
    private readonly int _nComponents;

    private Vector<T>? _featureImportances;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    /// <inheritdoc />
    public int NFeaturesToSelect => _nFeaturesToSelect;
    public double Gamma => _gamma;
    public Vector<T>? FeatureImportances => _featureImportances;
    /// <inheritdoc />
    public int[]? SelectedIndices => _selectedIndices;
    /// <inheritdoc />
    public override bool SupportsInverseTransform => false;

    public KernelPCASelector(
        int nFeaturesToSelect = 10,
        double gamma = 1.0,
        int nComponents = 5,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));

        _nFeaturesToSelect = nFeaturesToSelect;
        _gamma = gamma;
        _nComponents = nComponents;
    }

    /// <inheritdoc />
    protected override void FitCore(Matrix<T> data)
    {
        _nInputFeatures = data.Columns;
        int n = data.Rows;
        int p = data.Columns;

        var engine = Engine;
        var x = Tensor<T>.FromMatrix(data);

        // RBF kernel matrix, double-centred: Kc = K - rowMean - colMean + totalMean
        var kernel = engine.TensorExp(engine.TensorMultiplyScalar(
            FeatureSelectionTensorOps.PairwiseSquaredDistances(x), NumOps.FromDouble(-_gamma)));
        var centered = engine.TensorAddScalar(
            engine.TensorSubtract(
                engine.TensorSubtract(kernel, engine.ReduceMean(kernel, new[] { 1 }, true)),
                engine.ReduceMean(kernel, new[] { 0 }, true)),
            engine.TensorMean(kernel));

        // Top eigenvectors of the centred kernel (exact symmetric eigendecomposition)
        int nComp = Math.Min(_nComponents, Math.Min(n, p));
        var eigenvectors = FeatureSelectionTensorOps.TopEigenpairs(centered, nComp, out _); // [n, c]

        // Sensitivity of feature j: sum_k sum_ii' |v_ik v_i'k| (x_ij - x_i'j)^2 / n^2. With W = |V||V|' and
        // degrees D = W 1 that is 2 (sum_i D_i x_ij^2 - x_j' W x_j) / n^2.
        var importances = new Tensor<T>(new[] { p });
        if (nComp > 0)
        {
            var magnitude = engine.TensorAbs(eigenvectors);
            var weights = engine.TensorMatMul(magnitude, engine.TensorTranspose(magnitude));
            var degree = engine.ReduceSum(weights, new[] { 1 }, true);
            var spread = engine.TensorSubtract(
                engine.ReduceSum(engine.TensorMultiply(engine.TensorSquare(x), degree), new[] { 0 }, false),
                engine.ReduceSum(engine.TensorMultiply(x, engine.TensorMatMul(weights, x)), new[] { 0 }, false));
            importances = engine.TensorMax(
                engine.TensorMultiplyScalar(spread, NumOps.FromDouble(2.0 / ((double)n * n))), NumOps.Zero);
        }

        _featureImportances = importances.ToVector();

        int numToSelect = Math.Min(_nFeaturesToSelect, p);
        _selectedIndices = FeatureSelectionTensorOps.TopKIndicesSorted(importances, numToSelect);

        IsFitted = true;
    }

    /// <inheritdoc />
    protected override Matrix<T> TransformCore(Matrix<T> data)
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("KernelPCASelector has not been fitted.");

        return FeatureSelectionTensorOps.SelectColumns(data, _selectedIndices);
    }

    /// <inheritdoc />
    protected override Matrix<T> InverseTransformCore(Matrix<T> data)
    {
        throw new NotSupportedException("KernelPCASelector does not support inverse transformation.");
    }

    /// <inheritdoc />
    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("KernelPCASelector has not been fitted.");

        var mask = new bool[_nInputFeatures];
        foreach (int idx in _selectedIndices)
            mask[idx] = true;

        return mask;
    }

    /// <inheritdoc />
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
