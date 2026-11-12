using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Enums;
using AiDotNet.Attributes;
using AiDotNet.Preprocessing.FeatureSelection.Helpers;

namespace AiDotNet.Preprocessing.FeatureSelection.Neural;

/// <summary>
/// Variational Autoencoder-based Feature Selection.
/// </summary>
/// <remarks>
/// <para>
/// Uses a Variational Autoencoder (VAE) to learn a probabilistic latent space,
/// then selects features based on their contribution to the learned distribution.
/// </para>
/// <para><b>For Beginners:</b> A VAE learns not just a compressed representation
/// but a probability distribution over possible representations. Features that
/// contribute most to this learned distribution are the important ones.
/// </para>
/// </remarks>
[ComponentType(ComponentType.FeatureSelector)]
[PipelineStage(PipelineStage.Preprocessing)]
public class VariationalAutoencoderSelector<T> : TransformerBase<T, Matrix<T>, Matrix<T>>, IFixedCountFeatureSelector<T>
{
    private readonly int _nFeaturesToSelect;
    private readonly int _latentDim;
    private readonly int _nEpochs;

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

    public VariationalAutoencoderSelector(
        int nFeaturesToSelect = 10,
        int latentDim = 5,
        int nEpochs = 100,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));

        _nFeaturesToSelect = nFeaturesToSelect;
        _latentDim = latentDim;
        _nEpochs = nEpochs;
    }

    /// <inheritdoc />
    protected override void FitCore(Matrix<T> data)
    {
        _nInputFeatures = data.Columns;
        int n = data.Rows;
        int p = data.Columns;
        var engine = Engine;

        // Normalize to [0, 1]
        var raw = Tensor<T>.FromMatrix(data);
        var x = MinMaxScale(raw);
        var rows = NeuralSelectorOps.SplitRows(x);

        int z = Math.Min(_latentDim, p - 1);
        var rand = RandomHelper.CreateSecureRandom();

        var wMu = NeuralSelectorOps.SymmetricUniform<T>(rand, 0.1, p, z);
        var wLogVar = NeuralSelectorOps.SymmetricUniform<T>(rand, 0.1, p, z);
        var wDec = NeuralSelectorOps.SymmetricUniform<T>(rand, 0.1, z, p);
        T rate = NumOps.FromDouble(0.001);

        for (int epoch = 0; epoch < _nEpochs; epoch++)
        {
            for (int i = 0; i < n; i++)
            {
                var noise = StandardNormal(PopulationSearchOps.Uniform<T>(rand, z, 2));
                var (gradMu, gradLogVar, gradDec) = Gradients(rows[i], noise, wMu, wLogVar, wDec, KlWeight);
                wDec = NeuralSelectorOps.Step(wDec, gradDec, rate);
                wMu = NeuralSelectorOps.Step(wMu, gradMu, rate);
                wLogVar = NeuralSelectorOps.Step(wLogVar, gradLogVar, rate);
            }
        }

        var importance = engine.ReduceSum(
            engine.TensorAdd(engine.TensorAbs(wMu), engine.TensorAbs(wLogVar)), new[] { 1 }, false);
        _featureImportances = importance.ToVector();

        int numToSelect = Math.Min(_nFeaturesToSelect, p);
        _selectedIndices = PreprocessingTensorOps.TopKIndicesSorted(importance, numToSelect);

        IsFitted = true;
    }

    /// <summary>The weight of the KL term in the per-sample loss.</summary>
    internal const double KlWeight = 0.01;

    /// <summary>Columns of x [n, p] mapped to [0, 1] as (x - min) / (max - min + 1e-10); a constant column maps to 0.</summary>
    internal static Tensor<T> MinMaxScale(Tensor<T> raw)
    {
        var engine = AiDotNetEngine.Current;
        int n = raw.Shape[0];
        int p = raw.Shape[1];
        var min = PreprocessingTensorOps.ColumnMin(raw);
        var range = engine.TensorAddScalar(
            engine.TensorSubtract(PreprocessingTensorOps.ColumnMax(raw), min), PopulationSearchOps.Tiny<T>(1e-10));
        return engine.TensorDivide(
            engine.TensorSubtract(raw, engine.TensorBroadcastTo(engine.Reshape(min, new[] { 1, p }), new[] { n, p })),
            engine.TensorBroadcastTo(engine.Reshape(range, new[] { 1, p }), new[] { n, p }));
    }

    /// <summary>
    /// Box-Muller standard normals [z] from uniform pairs [z, 2]: sqrt(-2 ln(1 - u1)) sin(2 pi (1 - u2)). In float
    /// and Half a draw just below 1 rounds to 1, so 1 - u1 is floored at the smallest positive value a draw can leave
    /// (2^-53, or 2^-24 where T cannot hold that) instead of giving ln 0 and an infinite noise value.
    /// </summary>
    internal static Tensor<T> StandardNormal(Tensor<T> uniforms)
    {
        var engine = AiDotNetEngine.Current;
        var ops = MathHelper.GetNumericOperations<T>();
        int z = uniforms.Shape[0];
        T floor = ops.FromDouble(Math.Pow(2, -53));
        if (ops.Equals(floor, ops.Zero))
            floor = ops.FromDouble(Math.Pow(2, -24));
        var u1 = engine.TensorMax(
            PreprocessingTensorOps.OneMinus(engine.Reshape(engine.TensorSlice(uniforms, new[] { 0, 0 }, new[] { z, 1 }), new[] { z })),
            floor);
        var u2 = PreprocessingTensorOps.OneMinus(engine.Reshape(engine.TensorSlice(uniforms, new[] { 0, 1 }, new[] { z, 1 }), new[] { z }));
        var radius = engine.TensorSqrt(engine.TensorMultiplyScalar(engine.TensorLog(u1), ops.FromDouble(-2)));
        return engine.TensorMultiply(radius, engine.TensorSin(engine.TensorMultiplyScalar(u2, ops.FromDouble(2 * Math.PI))));
    }

    /// <summary>
    /// Gradients of the per-sample VAE loss (Kingma and Welling 2014) with respect to W_mu [p, z], W_logvar
    /// [p, z] and W_dec [z, p], for the noise eps [z]: mu = x W_mu, logvar = clamp(x W_logvar, -10, 10),
    /// latent = mu + exp(logvar / 2) eps, output = sigmoid(latent W_dec), and loss = binary cross-entropy of the
    /// output against x plus <paramref name="klWeight"/> KL(N(mu, exp(logvar)) || N(0, I)). With
    /// d = output - x and g = W_dec d: dW_dec = latent d^T, dW_mu = x (g + w mu)^T and
    /// dW_logvar = x (g .* eps .* exp(logvar / 2) / 2 + w (exp(logvar) - 1) / 2)^T, zero where logvar is clamped.
    /// </summary>
    internal static (Tensor<T> Mu, Tensor<T> LogVar, Tensor<T> Dec) Gradients(
        Tensor<T> x, Tensor<T> noise, Tensor<T> wMu, Tensor<T> wLogVar, Tensor<T> wDec, double klWeight)
    {
        var engine = AiDotNetEngine.Current;
        var ops = MathHelper.GetNumericOperations<T>();
        T half = ops.FromDouble(0.5);
        T weight = ops.FromDouble(klWeight);

        var mu = NeuralSelectorOps.VecMat(x, wMu);
        var rawLogVar = NeuralSelectorOps.VecMat(x, wLogVar);
        var logVar = engine.TensorClamp(rawLogVar, ops.FromDouble(-10), ops.FromDouble(10));
        var inside = engine.TensorMultiply(
            engine.TensorGreaterThan(rawLogVar, ops.FromDouble(-10)), engine.TensorLessThan(rawLogVar, ops.FromDouble(10)));
        var std = engine.TensorExp(engine.TensorMultiplyScalar(logVar, half));
        var latent = engine.TensorAdd(mu, engine.TensorMultiply(std, noise));
        var output = engine.TensorSigmoid(NeuralSelectorOps.VecMat(latent, wDec));

        var reconstruction = engine.TensorSubtract(output, x);
        var latentGrad = NeuralSelectorOps.MatVec(wDec, reconstruction);
        var gradDec = NeuralSelectorOps.Outer(latent, reconstruction);
        var muDelta = engine.TensorAdd(latentGrad, engine.TensorMultiplyScalar(mu, weight));
        var logVarDelta = engine.TensorMultiply(
            inside,
            engine.TensorAdd(
                engine.TensorMultiplyScalar(engine.TensorMultiply(latentGrad, engine.TensorMultiply(noise, std)), half),
                engine.TensorMultiplyScalar(
                    engine.TensorAddScalar(engine.TensorExp(logVar), ops.FromDouble(-1)), ops.FromDouble(0.5 * klWeight))));
        return (NeuralSelectorOps.Outer(x, muDelta), NeuralSelectorOps.Outer(x, logVarDelta), gradDec);
    }

    /// <inheritdoc />
    protected override Matrix<T> TransformCore(Matrix<T> data)
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("VariationalAutoencoderSelector has not been fitted.");

        return PreprocessingTensorOps.SelectColumns(data, _selectedIndices);
    }

    /// <inheritdoc />
    protected override Matrix<T> InverseTransformCore(Matrix<T> data)
    {
        throw new NotSupportedException("VariationalAutoencoderSelector does not support inverse transformation.");
    }

    /// <inheritdoc />
    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("VariationalAutoencoderSelector has not been fitted.");

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
