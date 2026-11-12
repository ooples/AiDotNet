using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Enums;
using AiDotNet.Attributes;
using AiDotNet.Preprocessing.FeatureSelection.Helpers;

namespace AiDotNet.Preprocessing.FeatureSelection.Projection;

/// <summary>
/// Factor Analysis based Feature Selection.
/// </summary>
/// <remarks>
/// <para>
/// Selects features based on their factor loadings, identifying features that
/// load heavily on extracted latent factors.
/// </para>
/// <para><b>For Beginners:</b> Factor analysis finds hidden patterns in data by
/// identifying underlying factors. Features that are strongly connected to these
/// factors are selected, as they carry the most meaningful information.
/// </para>
/// </remarks>
[ComponentType(ComponentType.FeatureSelector)]
[PipelineStage(PipelineStage.Preprocessing)]
public class FactorAnalysisSelector<T> : TransformerBase<T, Matrix<T>, Matrix<T>>, IFixedCountFeatureSelector<T>
{
    private readonly int _nFeaturesToSelect;
    private readonly int _nFactors;

    private Vector<T>? _communalities;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    /// <inheritdoc />
    public int NFeaturesToSelect => _nFeaturesToSelect;
    public int NFactors => _nFactors;
    public Vector<T>? Communalities => _communalities;
    /// <inheritdoc />
    public int[]? SelectedIndices => _selectedIndices;
    /// <inheritdoc />
    public override bool SupportsInverseTransform => false;

    public FactorAnalysisSelector(
        int nFeaturesToSelect = 10,
        int nFactors = 3,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));

        _nFeaturesToSelect = nFeaturesToSelect;
        _nFactors = nFactors;
    }

    /// <inheritdoc />
    protected override void FitCore(Matrix<T> data)
    {
        _nInputFeatures = data.Columns;
        int n = data.Rows;
        int p = data.Columns;

        var engine = Engine;
        var x = Tensor<T>.FromMatrix(data);

        // Standardize (sample std, 1 for constant columns) and form the correlation matrix
        var centered = engine.TensorSubtract(x, engine.ReduceMean(x, new[] { 0 }, true));
        var std = engine.TensorSqrt(engine.TensorDivideScalar(
            engine.ReduceSum(engine.TensorSquare(centered), new[] { 0 }, true), NumOps.FromDouble(n - 1)));
        var safeStd = engine.TensorWhere(
            engine.TensorLessThan(std, NumOps.FromDouble(1e-10)), FeatureSelectionTensorOps.Filled(new[] { 1, p }, NumOps.One), std);
        var standardized = engine.TensorDivide(centered, safeStd);
        var correlation = engine.TensorDivideScalar(
            engine.TensorMatMul(engine.TensorTranspose(standardized), standardized), NumOps.FromDouble(n - 1));

        // Principal axis factoring: initial communality estimates (the squared largest absolute correlation
        // with another feature) replace the diagonal, and the factors are the top eigenpairs of that
        // reduced correlation matrix, with loadings sqrt(max(0, eigenvalue)) * eigenvector
        var eye = engine.TensorEye<T>(p);
        var offDiagonal = engine.TensorMultiply(correlation, FeatureSelectionTensorOps.OneMinus(eye));
        var maxCorrelation = engine.ReduceMax(engine.TensorAbs(offDiagonal), new[] { 1 }, true); // [p, 1]
        var initialCommunalities = engine.Reshape(engine.TensorSquare(maxCorrelation), new[] { 1, p });
        var reduced = engine.TensorAdd(offDiagonal, engine.TensorMultiply(eye, initialCommunalities));

        int numFactors = Math.Min(_nFactors, p);
        var factors = FeatureSelectionTensorOps.TopEigenpairs(reduced, numFactors, out var eigenvalues); // [p, f]

        // Communalities: sum of squared loadings
        var communalities = new Tensor<T>(new[] { p });
        if (numFactors > 0)
        {
            var factorVariance = engine.Reshape(engine.TensorMax(eigenvalues, NumOps.Zero), new[] { numFactors, 1 });
            communalities = engine.Reshape(engine.TensorMatMul(engine.TensorSquare(factors), factorVariance), new[] { p });
        }

        _communalities = communalities.ToVector();

        int numToSelect = Math.Min(_nFeaturesToSelect, p);
        _selectedIndices = FeatureSelectionTensorOps.TopKIndicesSorted(communalities, numToSelect);

        IsFitted = true;
    }

    public new Matrix<T> FitTransform(Matrix<T> data)
    {
        Fit(data);
        return Transform(data);
    }

    /// <inheritdoc />
    protected override Matrix<T> TransformCore(Matrix<T> data)
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("FactorAnalysisSelector has not been fitted.");

        return FeatureSelectionTensorOps.SelectColumns(data, _selectedIndices);
    }

    /// <inheritdoc />
    protected override Matrix<T> InverseTransformCore(Matrix<T> data)
    {
        throw new NotSupportedException("FactorAnalysisSelector does not support inverse transformation.");
    }

    /// <inheritdoc />
    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("FactorAnalysisSelector has not been fitted.");

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
