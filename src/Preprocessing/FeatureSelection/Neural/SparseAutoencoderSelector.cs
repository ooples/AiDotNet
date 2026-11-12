using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Enums;
using AiDotNet.Attributes;
using AiDotNet.Preprocessing.FeatureSelection.Helpers;

namespace AiDotNet.Preprocessing.FeatureSelection.Neural;

/// <summary>
/// Sparse Autoencoder-based Feature Selection.
/// </summary>
/// <remarks>
/// <para>
/// Uses a sparse autoencoder with sparsity constraints on the hidden layer
/// to learn which input features are essential for reconstruction.
/// </para>
/// <para><b>For Beginners:</b> A sparse autoencoder forces most of its internal
/// neurons to be "off" most of the time. This makes it focus on the most
/// important input features. Features that activate the sparse neurons are
/// the ones we select.
/// </para>
/// </remarks>
[ComponentType(ComponentType.FeatureSelector)]
[PipelineStage(PipelineStage.Preprocessing)]
public class SparseAutoencoderSelector<T> : TransformerBase<T, Matrix<T>, Matrix<T>>, IFixedCountFeatureSelector<T>
{
    private readonly int _nFeaturesToSelect;
    private readonly int _hiddenSize;
    private readonly int _nEpochs;
    private readonly double _sparsityTarget;
    private readonly double _sparsityWeight;

    private Vector<T>? _featureImportances;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    /// <inheritdoc />
    public int NFeaturesToSelect => _nFeaturesToSelect;
    public Vector<T>? FeatureImportances => _featureImportances;
    /// <inheritdoc />
    public int[]? SelectedIndices => _selectedIndices;
    /// <inheritdoc />
    public override bool SupportsInverseTransform => false;

    public SparseAutoencoderSelector(
        int nFeaturesToSelect = 10,
        int hiddenSize = 20,
        int nEpochs = 100,
        double sparsityTarget = 0.05,
        double sparsityWeight = 1.0,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));

        _nFeaturesToSelect = nFeaturesToSelect;
        _hiddenSize = hiddenSize;
        _nEpochs = nEpochs;
        _sparsityTarget = sparsityTarget;
        _sparsityWeight = sparsityWeight;
    }

    /// <inheritdoc />
    protected override void FitCore(Matrix<T> data)
    {
        _nInputFeatures = data.Columns;
        int n = data.Rows;
        int p = data.Columns;
        var engine = Engine;

        // Normalize
        var x = NeuralSelectorOps.StandardizeSample(Tensor<T>.FromMatrix(data));
        var rows = NeuralSelectorOps.SplitRows(x);

        int h = Math.Min(_hiddenSize, p);
        var rand = RandomHelper.CreateSecureRandom();

        double scale = Math.Sqrt(2.0 / (p + h));
        var w1 = NeuralSelectorOps.SymmetricUniform<T>(rand, scale, p, h);
        var w2 = NeuralSelectorOps.SymmetricUniform<T>(rand, scale, h, p);
        T rate = NumOps.FromDouble(0.01);

        for (int epoch = 0; epoch < _nEpochs; epoch++)
        {
            // Average activation of every hidden unit over the data with the weights at the start of the epoch,
            // and the KL sparsity term of the hidden deltas
            var sparsity = SparsityDelta(MeanActivation(x, w1), _sparsityTarget, _sparsityWeight);

            for (int i = 0; i < n; i++)
            {
                var (gradW1, gradW2) = SampleGradients(rows[i], w1, w2, sparsity);
                w2 = NeuralSelectorOps.Step(w2, gradW2, rate);
                w1 = NeuralSelectorOps.Step(w1, gradW1, rate);
            }
        }

        var importance = engine.ReduceSum(engine.TensorAbs(w1), new[] { 1 }, false);
        _featureImportances = importance.ToVector();

        int numToSelect = Math.Min(_nFeaturesToSelect, p);
        _selectedIndices = FeatureSelectionTensorOps.TopKIndicesSorted(importance, numToSelect);

        IsFitted = true;
    }

    /// <summary>The mean sigmoid activation rho_hat [h] of the hidden units over the rows of x [n, p].</summary>
    internal static Tensor<T> MeanActivation(Tensor<T> x, Tensor<T> w1)
    {
        var engine = AiDotNetEngine.Current;
        return engine.ReduceMean(engine.TensorSigmoid(engine.TensorMatMul(x, w1)), new[] { 0 }, false);
    }

    /// <summary>
    /// The derivative beta (-rho / rho_hat + (1 - rho) / (1 - rho_hat)) of beta sum_j KL(rho || rho_hat_j) with
    /// respect to rho_hat, which back-propagation adds to the hidden deltas (Ng, sparse autoencoder notes).
    /// </summary>
    internal static Tensor<T> SparsityDelta(Tensor<T> meanActivation, double target, double weight)
    {
        var engine = AiDotNetEngine.Current;
        var ops = MathHelper.GetNumericOperations<T>();
        int h = meanActivation.Length;
        var eps = ops.FromDouble(1e-10);
        var first = engine.TensorDivide(
            FeatureSelectionTensorOps.Filled(new[] { h }, ops.FromDouble(-target)), engine.TensorAddScalar(meanActivation, eps));
        var second = engine.TensorDivide(
            FeatureSelectionTensorOps.Filled(new[] { h }, ops.FromDouble(1 - target)),
            engine.TensorAddScalar(FeatureSelectionTensorOps.OneMinus(meanActivation), eps));
        return engine.TensorMultiplyScalar(engine.TensorAdd(first, second), ops.FromDouble(weight));
    }

    /// <summary>
    /// The per-sample term of the gradient of J = (1/n) sum_i |x_i - sigmoid(x_i W1) W2|^2 / 2 +
    /// beta sum_j KL(rho || rho_hat_j): with hidden a = sigmoid(x W1) and output error e = a W2 - x, dW2 = a e^T
    /// and dW1 = x ((W2 e + sparsity) .* a .* (1 - a))^T, so the mean of the terms over the samples is the
    /// gradient of J. <paramref name="sparsity"/> is <see cref="SparsityDelta"/> at the current rho_hat.
    /// </summary>
    internal static (Tensor<T> W1, Tensor<T> W2) SampleGradients(Tensor<T> x, Tensor<T> w1, Tensor<T> w2, Tensor<T> sparsity)
    {
        var engine = AiDotNetEngine.Current;
        var hidden = engine.TensorSigmoid(NeuralSelectorOps.VecMat(x, w1));
        var outputError = engine.TensorSubtract(NeuralSelectorOps.VecMat(hidden, w2), x);
        var hiddenDelta = engine.TensorMultiply(
            engine.TensorAdd(NeuralSelectorOps.MatVec(w2, outputError), sparsity),
            engine.TensorMultiply(hidden, FeatureSelectionTensorOps.OneMinus(hidden)));
        return (NeuralSelectorOps.Outer(x, hiddenDelta), NeuralSelectorOps.Outer(hidden, outputError));
    }

    /// <inheritdoc />
    protected override Matrix<T> TransformCore(Matrix<T> data)
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("SparseAutoencoderSelector has not been fitted.");

        return FeatureSelectionTensorOps.SelectColumns(data, _selectedIndices);
    }

    /// <inheritdoc />
    protected override Matrix<T> InverseTransformCore(Matrix<T> data)
    {
        throw new NotSupportedException("SparseAutoencoderSelector does not support inverse transformation.");
    }

    /// <inheritdoc />
    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("SparseAutoencoderSelector has not been fitted.");

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
