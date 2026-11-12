using System.Globalization;
using AiDotNet.Attributes;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Tensors.Engines;

namespace AiDotNet.NeuralNetworks.Layers;

/// <summary>
/// Matrix-free orthogonal projection <c>y = x Q</c> (or <c>y = x Qᵀ</c>) parameterized as a product of
/// <c>k</c> Householder reflections <c>Q = H₀ H₁ ⋯ H_{k−1}</c>, where each
/// <c>Hⱼ = I − (2 / (‖uⱼ‖₂² + ε)) uⱼ uⱼᵀ</c> uses only <c>O(k · D)</c> parameters and <c>4 k · M · D</c> FLOPs
/// without ever materializing a <c>D × D</c> matrix.
/// </summary>
/// <typeparam name="T">The numeric type used for inputs, reflector vectors, and outputs.</typeparam>
/// <remarks>
/// <para>
/// Every non-zero reflector vector <c>uⱼ ∈ ℝᴰ</c> defines an exact symmetric orthogonal reflection
/// (<c>Hⱼᵀ = Hⱼ</c>, <c>Hⱼ² = I</c>). Consequently <c>Qᵀ Q = I</c> holds by construction for arbitrary unconstrained
/// gradient updates to <c>uⱼ</c> without SVD, QR retraction, or Cayley matrix inversions, preserving activation
/// and gradient Euclidean norms across arbitrarily deep shared-parameter stacks (e.g. 100+ layers).
/// </para>
/// <para>
/// Setting <see cref="Transpose"/> to <c>true</c> applies the reflections in reverse order
/// (<c>H_{k−1} ⋯ H₁ H₀ = Qᵀ = Q⁻¹</c>), providing the exact inverse/transpose rotation.
/// </para>
/// </remarks>
[LayerCategory(LayerCategory.Dense)]
[LayerTask(LayerTask.Projection)]
[LayerProperty(IsTrainable = true, ChangesShape = false, TestInputShape = "2, 8", TestConstructorArgs = "8, 2")]
[TensorLayout(TensorAxis.Features, Direction = TensorLayoutDirection.Input)]
[TensorLayout(TensorAxis.Features, Direction = TensorLayoutDirection.Output)]
[TensorLayout(TensorAxis.Batch, TensorAxis.Features, BatchOptional = true, Direction = TensorLayoutDirection.Input)]
[TensorLayout(TensorAxis.Batch, TensorAxis.Features, BatchOptional = true, Direction = TensorLayoutDirection.Output)]
[TensorLayout(TensorAxis.Batch, TensorAxis.Time, TensorAxis.Features, Direction = TensorLayoutDirection.Input)]
[TensorLayout(TensorAxis.Batch, TensorAxis.Time, TensorAxis.Features, Direction = TensorLayoutDirection.Output)]
[AutoParameters]
public sealed partial class HouseholderSubspaceLayer<T> : LayerBase<T>, IShapeContract
{
    private readonly int _dimension;
    private readonly int _numReflectors;
    private readonly bool _transpose;

    [TrainableParameter(Role = PersistentTensorRole.Weights)]
    private readonly Tensor<T> _reflectors;

    /// <summary>Gets the feature dimension <c>D</c> rotated by this layer.</summary>
    public int Dimension => _dimension;

    /// <summary>Gets the number of Householder reflections <c>k</c> composed by this layer.</summary>
    public int NumReflectors => _numReflectors;

    /// <summary>Gets whether the reflection chain is applied in reverse order (<c>Qᵀ = Q⁻¹</c>).</summary>
    public bool Transpose => _transpose;

    /// <summary>Gets the underlying <c>[NumReflectors, Dimension]</c> trainable reflector matrix.</summary>
    public Tensor<T> Reflectors => _reflectors;

    /// <inheritdoc />
    public override bool SupportsTraining => true;

    /// <summary>
    /// Creates a matrix-free orthogonal Householder subspace rotation layer of width <paramref name="dimension"/>
    /// using <paramref name="numReflectors"/> reflections.
    /// </summary>
    /// <param name="dimension">Feature width <c>D</c> (must be positive).</param>
    /// <param name="numReflectors">Number of composed Householder reflections <c>k</c> (must be positive).</param>
    /// <param name="transpose">When <c>true</c>, applies reflections in reverse order (<c>Qᵀ = Q⁻¹</c>).</param>
    public HouseholderSubspaceLayer(int dimension, int numReflectors = 4, bool transpose = false)
        : base([dimension], [dimension])
    {
        if (dimension <= 0) throw new ArgumentOutOfRangeException(nameof(dimension));
        if (numReflectors <= 0) throw new ArgumentOutOfRangeException(nameof(numReflectors));

        _dimension = dimension;
        _numReflectors = numReflectors;
        _transpose = transpose;
        _reflectors = new Tensor<T>([numReflectors, dimension]);
        InitializeLayerWeights(_reflectors, numReflectors, dimension);
        NormalizeRowsInPlace(_reflectors, numReflectors, dimension);
        RegisterTrainableParameter(_reflectors, PersistentTensorRole.Weights);
    }

    /// <inheritdoc />
    public IReadOnlyList<OutputAxisContract>? OutputAxesFor(int inputRank)
    {
        if (_dimension <= 0 || inputRank < 1) return null;
        var features = new OutputAxisContract(TensorAxis.Features, AxisRelation.Fixed(_dimension));
        static OutputAxisContract Pass(TensorAxis axis) => new(axis, AxisRelation.Same(axis));
        return inputRank switch
        {
            1 => [features],
            2 => [Pass(TensorAxis.Batch), features],
            3 => [Pass(TensorAxis.Batch), Pass(TensorAxis.Time), features],
            4 => [Pass(TensorAxis.Batch), Pass(TensorAxis.Channels), Pass(TensorAxis.Height), features],
            _ => null
        };
    }

    /// <summary>
    /// Applies the matrix-free product of Householder reflections stored in <paramref name="reflectors"/>
    /// (<c>[k, D]</c>) to <paramref name="input"/> (<c>[..., D]</c>) using pure tape-tracked <see cref="IEngine"/> ops.
    /// </summary>
    public static Tensor<T> ApplyChain(IEngine engine, Tensor<T> input, Tensor<T> reflectors, bool transpose = false)
    {
        if (engine is null) throw new ArgumentNullException(nameof(engine));
        if (input is null) throw new ArgumentNullException(nameof(input));
        if (reflectors is null) throw new ArgumentNullException(nameof(reflectors));
        if (reflectors.Rank != 2)
            throw new ArgumentException($"Reflectors must be rank-2 [k, D], got [{string.Join(",", reflectors._shape)}].", nameof(reflectors));

        int k = reflectors.Shape[0];
        int d = reflectors.Shape[1];
        if (k <= 0 || d <= 0)
            throw new ArgumentException("Reflectors shape dimensions must be positive.", nameof(reflectors));
        if (input.Rank < 1 || input.Shape[^1] != d)
            throw new ArgumentException($"Expected last dimension {d}, got [{string.Join(",", input._shape)}].", nameof(input));

        var numOps = MathHelper.GetNumericOperations<T>();
        T two = numOps.FromDouble(2.0);
        // 1e-20 is a normal float (> 1.17e-38) and well below both FP32 (1.19e-7) and FP64 (2.22e-16) machine
        // epsilon at unit norm, preventing 0/0 on zero vectors with zero ULP perturbation on unit reflectors.
        T eps = numOps.FromDouble(1e-20);

        int rowCount = input.Length / d;
        Tensor<T> current = input.Rank == 2
            ? input
            : engine.Reshape(input, [rowCount, d]);

        int start = transpose ? k - 1 : 0;
        int end = transpose ? -1 : k;
        int step = transpose ? -1 : 1;

        for (int j = start; j != end; j += step)
        {
            Tensor<T> uRow = k == 1
                ? reflectors
                : engine.TensorSlice(reflectors, [j, 0], [1, d]);
            Tensor<T> uCol = engine.TensorTranspose(uRow);
            Tensor<T> proj = engine.TensorMatMul(current, uCol);
            Tensor<T> normSq = engine.ReduceSum(engine.TensorMultiply(uRow, uRow), null);
            Tensor<T> denom = engine.TensorAddScalar(normSq, eps);
            Tensor<T> scaledProj = engine.TensorDivide(engine.TensorMultiplyScalar(proj, two), denom);
            Tensor<T> outer = engine.TensorMatMul(scaledProj, uRow);
            current = engine.TensorSubtract(current, outer);
        }

        return input.Rank == 2
            ? current
            : engine.Reshape(current, input._shape);
    }

    /// <inheritdoc />
    protected override Tensor<T> ForwardTraced(Tensor<T> input)
        => ApplyChain(Engine, input, _reflectors, _transpose);

    /// <inheritdoc />
    public override void UpdateParameters(T learningRate)
    {
        var gradients = GetParameterGradients();
        if (gradients.Length != _reflectors.Length) return;
        using var noGrad = new AiDotNet.Tensors.Engines.Autodiff.NoGradScope<T>();
        var gradTensor = new Tensor<T>(_reflectors._shape, gradients);
        Engine.TensorSubtractInPlace(_reflectors, Engine.TensorMultiplyScalar(gradTensor, learningRate));
        Engine.InvalidatePersistentTensor(_reflectors);
    }

    /// <inheritdoc />
    public override void ResetState()
    {
    }

    internal override Dictionary<string, string> GetMetadata()
    {
        var metadata = base.GetMetadata();
        metadata["Dimension"] = _dimension.ToString(CultureInfo.InvariantCulture);
        metadata["NumReflectors"] = _numReflectors.ToString(CultureInfo.InvariantCulture);
        metadata["Transpose"] = _transpose.ToString(CultureInfo.InvariantCulture);
        return metadata;
    }

    private void NormalizeRowsInPlace(Tensor<T> reflectors, int rows, int cols)
    {
        if (rows <= 0 || cols <= 0) return;
        using var noGrad = new AiDotNet.Tensors.Engines.Autodiff.NoGradScope<T>();
        var sq = Engine.TensorMultiply(reflectors, reflectors);
        var rowNormSq = Engine.ReduceSum(sq, new[] { 1 }, keepDims: true);
        var safeNorm = Engine.TensorSqrt(Engine.TensorAddScalar(rowNormSq, NumOps.FromDouble(1e-24)));
        var normalized = Engine.TensorDivide(reflectors, safeNorm);
        normalized.AsSpan().Slice(0, reflectors.Length).CopyTo(reflectors.Data.Span);
        Engine.InvalidatePersistentTensor(reflectors);
    }
}
