using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Enums;
using AiDotNet.Attributes;
using AiDotNet.Preprocessing.FeatureSelection.Helpers;

namespace AiDotNet.Preprocessing.FeatureSelection.Nonlinear;

/// <summary>
/// Hilbert-Schmidt Independence Criterion (HSIC) based Feature Selection.
/// </summary>
/// <remarks>
/// <para>
/// Selects features based on HSIC, a kernel-based measure of independence
/// that can detect complex nonlinear relationships.
/// </para>
/// <para><b>For Beginners:</b> HSIC uses kernel methods to measure dependence
/// between variables in a high-dimensional space. It can detect subtle nonlinear
/// relationships that other methods might miss. Higher HSIC means stronger
/// dependence between the feature and target.
/// </para>
/// </remarks>
[ComponentType(ComponentType.FeatureSelector)]
[PipelineStage(PipelineStage.Preprocessing)]
public class HilbertSchmidtSelector<T> : TransformerBase<T, Matrix<T>, Matrix<T>>, ISupervisedFeatureSelector<T>, IFixedCountFeatureSelector<T>
{
    private readonly int _nFeaturesToSelect;
    private readonly double _kernelWidth;

    private Vector<T>? _hsicValues;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    /// <inheritdoc />
    public int NFeaturesToSelect => _nFeaturesToSelect;
    public double KernelWidth => _kernelWidth;
    public Vector<T>? HSICValues => _hsicValues;
    /// <inheritdoc />
    public int[]? SelectedIndices => _selectedIndices;
    /// <inheritdoc />
    public override bool SupportsInverseTransform => false;

    public HilbertSchmidtSelector(
        int nFeaturesToSelect = 10,
        double kernelWidth = 1.0,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));
        if (kernelWidth <= 0)
            throw new ArgumentException("Kernel width must be positive.", nameof(kernelWidth));

        _nFeaturesToSelect = nFeaturesToSelect;
        _kernelWidth = kernelWidth;
    }

    /// <inheritdoc />
    protected override void FitCore(Matrix<T> data)
    {
        throw new InvalidOperationException(
            "HilbertSchmidtSelector requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
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

        // Pairs i < j, for the median heuristic
        var upperPairs = new Tensor<int>(PreprocessingTensorOps.MaskPositions(FeatureSelectionTensorOps.StrictUpperMask<T>(n)),
            new[] { n * (n - 1) / 2 });

        // Kernel matrix for Y, centered: K_c = HKH with H = I - 1/n * 11^T
        var Ky = GaussianKernel(engine.Reshape(Tensor<T>.FromVector(target), new[] { n, 1 }), upperPairs);
        var KyC = engine.TensorAddScalar(
            engine.TensorSubtract(
                engine.TensorSubtract(Ky, engine.ReduceMean(Ky, new[] { 1 }, true)),
                engine.ReduceMean(Ky, new[] { 0 }, true)),
            engine.TensorMean(Ky));

        // HSIC = (1/n²) * trace(Kx_c * Ky_c) = (1/n²) * sum(Kx ∘ HKyH), since H is idempotent
        var traces = engine.Reshape(FeatureSelectionTensorOps.KernelWeightedSums(
            Tensor<T>.FromMatrix(data), column => GaussianKernel(column, upperPairs), new[] { KyC }, Array.Empty<Tensor<T>>()), new[] { p });
        var nSquared = NumOps.Multiply(FeatureSelectionTensorOps.Count<T>(n), FeatureSelectionTensorOps.Count<T>(n));
        var hsic = engine.TensorMax(engine.TensorDivideScalar(traces, nSquared), NumOps.Zero);
        _hsicValues = hsic.ToVector();

        int numToSelect = Math.Min(_nFeaturesToSelect, p);
        _selectedIndices = PreprocessingTensorOps.TopKIndicesSorted(hsic, numToSelect);

        IsFitted = true;
    }

    /// <summary>
    /// Gaussian kernel matrix ([n, n]) of an [n, 1] column, <c>exp(-(a_i - a_j)^2 / (2 width^2))</c>, with
    /// width = kernelWidth * the median pairwise distance (at least 1e-10; 1 when there is no pair).
    /// </summary>
    private Tensor<T> GaussianKernel(Tensor<T> column, Tensor<int> upperPairs)
    {
        var engine = Engine;
        int n = column.Shape[0];
        var difference = engine.TensorSubtract(
            engine.TensorBroadcastTo(column, new[] { n, n }), engine.TensorBroadcastTo(engine.TensorTranspose(column), new[] { n, n }));

        Tensor<T> median;
        if (upperPairs.Length == 0)
        {
            median = PreprocessingTensorOps.Filled(new[] { 1 }, NumOps.One);
        }
        else
        {
            var distances = engine.TensorIndexSelect(engine.Reshape(engine.TensorAbs(difference), new[] { n * n }), upperPairs, 0);
            median = engine.TensorMax(
                FilterStatisticTensorOps.ColumnQuantile(engine.Reshape(distances, new[] { upperPairs.Length, 1 }), 0.5),
                NumOps.FromDouble(1e-10));
        }

        // gamma = 1 / (2 width^2)
        var width = engine.TensorMultiplyScalar(median, NumOps.FromDouble(_kernelWidth));
        var negativeGamma = engine.TensorDivide(
            PreprocessingTensorOps.Filled(new[] { 1 }, NumOps.FromDouble(-0.5)), engine.TensorSquare(width));
        var scale = engine.TensorBroadcastTo(engine.Reshape(negativeGamma, new[] { 1, 1 }), new[] { n, n });
        return engine.TensorExp(engine.TensorMultiply(engine.TensorSquare(difference), scale));
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
            throw new InvalidOperationException("HilbertSchmidtSelector has not been fitted.");

        return PreprocessingTensorOps.SelectColumns(data, _selectedIndices);
    }

    /// <inheritdoc />
    protected override Matrix<T> InverseTransformCore(Matrix<T> data)
    {
        throw new NotSupportedException("HilbertSchmidtSelector does not support inverse transformation.");
    }

    /// <inheritdoc />
    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("HilbertSchmidtSelector has not been fitted.");

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
