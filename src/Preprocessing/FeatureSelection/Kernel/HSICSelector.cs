using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Enums;
using AiDotNet.Attributes;
using AiDotNet.Preprocessing.FeatureSelection.Helpers;

namespace AiDotNet.Preprocessing.FeatureSelection.Kernel;

/// <summary>
/// Hilbert-Schmidt Independence Criterion (HSIC) based Feature Selection.
/// </summary>
/// <remarks>
/// <para>
/// Uses HSIC to measure the statistical dependency between features and target
/// in a reproducing kernel Hilbert space, capturing non-linear dependencies.
/// </para>
/// <para><b>For Beginners:</b> HSIC is a powerful way to measure if two variables
/// are related, even in complex non-linear ways. By computing HSIC between each
/// feature and the target, we find features that are truly informative, even when
/// the relationship isn't a straight line.
/// </para>
/// </remarks>
[ComponentType(ComponentType.FeatureSelector)]
[PipelineStage(PipelineStage.Preprocessing)]
public class HSICSelector<T> : TransformerBase<T, Matrix<T>, Matrix<T>>, ISupervisedFeatureSelector<T>, IFixedCountFeatureSelector<T>
{
    private readonly int _nFeaturesToSelect;
    private readonly double _gammaX;
    private readonly double _gammaY;

    private Vector<T>? _hsicScores;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    /// <inheritdoc />
    public int NFeaturesToSelect => _nFeaturesToSelect;
    public double GammaX => _gammaX;
    public double GammaY => _gammaY;
    public Vector<T>? HSICScores => _hsicScores;
    /// <inheritdoc />
    public int[]? SelectedIndices => _selectedIndices;
    /// <inheritdoc />
    public override bool SupportsInverseTransform => false;

    public HSICSelector(
        int nFeaturesToSelect = 10,
        double gammaX = 1.0,
        double gammaY = 1.0,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));

        _nFeaturesToSelect = nFeaturesToSelect;
        _gammaX = gammaX;
        _gammaY = gammaY;
    }

    /// <inheritdoc />
    protected override void FitCore(Matrix<T> data)
    {
        throw new InvalidOperationException(
            "HSICSelector requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
    }

    /// <inheritdoc />
    public void Fit(Matrix<T> data, Vector<T> target)
    {
        if (data.Rows != target.Length)
            throw new ArgumentException("Target length must match rows in data.");

        _nInputFeatures = data.Columns;
        int n = data.Rows;
        int p = data.Columns;

        var engine = Engine;
        var x = Tensor<T>.FromMatrix(data);

        // Centred RBF kernel of the target: H Ky H with H = I - 11'/n
        var y = engine.Reshape(Tensor<T>.FromVector(target), new[] { n, 1 });
        var ky = FeatureSelectionTensorOps.RbfColumnKernel(y, NumOps.FromDouble(-_gammaY));
        var centering = engine.TensorSubtract(
            engine.TensorEye<T>(n), PreprocessingTensorOps.Filled(new[] { n, n }, NumOps.FromDouble(1.0 / n)));
        var centeredKy = engine.TensorMatMul(engine.TensorMatMul(centering, ky), centering);

        // Biased HSIC = trace(Kx H Ky H) / (n-1)^2 = sum(Kx .* (H Ky H)') / (n-1)^2, clamped at 0
        T negativeGammaX = NumOps.FromDouble(-_gammaX);
        var sums = FeatureSelectionTensorOps.KernelWeightedSums(
            x,
            column => FeatureSelectionTensorOps.RbfColumnKernel(column, negativeGammaX),
            new[] { engine.TensorTranspose(centeredKy) },
            Array.Empty<Tensor<T>>());
        var hsic = engine.TensorMax(
            engine.TensorDivideScalar(engine.Reshape(sums, new[] { p }), NumOps.FromDouble((double)(n - 1) * (n - 1))), NumOps.Zero);
        _hsicScores = hsic.ToVector();

        int numToSelect = Math.Min(_nFeaturesToSelect, p);
        _selectedIndices = PreprocessingTensorOps.TopKIndicesSorted(hsic, numToSelect);

        IsFitted = true;
    }

    /// <inheritdoc />
    public Matrix<T> FitTransform(Matrix<T> data, Vector<T> target)
    {
        Fit(data, target);
        return Transform(data);
    }

    /// <inheritdoc />
    protected override Matrix<T> TransformCore(Matrix<T> data)
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("HSICSelector has not been fitted.");

        return PreprocessingTensorOps.SelectColumns(data, _selectedIndices);
    }

    /// <inheritdoc />
    protected override Matrix<T> InverseTransformCore(Matrix<T> data)
    {
        throw new NotSupportedException("HSICSelector does not support inverse transformation.");
    }

    /// <inheritdoc />
    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("HSICSelector has not been fitted.");

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
