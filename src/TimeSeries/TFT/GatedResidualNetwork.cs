using AiDotNet.Tensors;
using AiDotNet.Tensors.Engines;

namespace AiDotNet.TimeSeries.TFT;

/// <summary>
/// Gated Residual Network (GRN) as described in Lim et al. (2021).
/// GRN(a) = LayerNorm(a + GLU(η₁)) where η₁ = W₁·η₂ + b₁,
/// η₂ = ELU(W₂·a + b₂), and GLU(γ) = σ(W₄·γ + b₄) ⊙ (W₅·γ + b₅).
/// </summary>
/// <remarks>
/// This implementation is expressed entirely with batched <c>Engine.Tensor*</c> ops so a
/// <see cref="Tensors.Engines.Autodiff.GradientTape{T}"/> differentiates it automatically
/// (no hand-derived gradients) and every op is GPU-dispatchable. All layers operate on a
/// flattened <c>[N, d]</c> matrix where each of the N rows is processed independently
/// (token-wise), matching the Temporal Fusion Transformer tape-training campaign.
/// The final layer normalization uses learned affine parameters (γ, β).
/// </remarks>
internal class GatedResidualNetwork<T>
{
    private static IEngine Engine => AiDotNetEngine.Current;

    private readonly int _inputSize;
    private readonly int _hiddenSize;
    private readonly int _outputSize;

    // η₂ = ELU(W₂·a + b₂)
    private readonly Tensor<T> _w2; // [hiddenSize, inputSize]
    private readonly Tensor<T> _b2; // [hiddenSize]

    // η₁ = W₁·η₂ + b₁
    private readonly Tensor<T> _w1; // [outputSize, hiddenSize]
    private readonly Tensor<T> _b1; // [outputSize]

    // GLU: σ(W₄·γ + b₄) ⊙ (W₅·γ + b₅)
    private readonly Tensor<T> _w4; // [outputSize, outputSize]
    private readonly Tensor<T> _b4; // [outputSize]
    private readonly Tensor<T> _w5; // [outputSize, outputSize]
    private readonly Tensor<T> _b5; // [outputSize]

    // Skip connection projection (only when inputSize != outputSize)
    private readonly Tensor<T>? _skipProjection; // [outputSize, inputSize]

    // Learned LayerNorm affine parameters
    private readonly Tensor<T> _lnGamma; // [outputSize]
    private readonly Tensor<T> _lnBeta;  // [outputSize]

    public GatedResidualNetwork(int inputSize, int hiddenSize, int outputSize, int? seed = null)
    {
        _inputSize = inputSize;
        _hiddenSize = hiddenSize;
        _outputSize = outputSize;

        var random = seed.HasValue ? new Random(seed.Value) : RandomHelper.CreateSeededRandom(42);

        // He initialization for the ELU layer
        _w2 = CreateRandomTensor([_hiddenSize, _inputSize], Math.Sqrt(2.0 / _inputSize), random);
        _b2 = new Tensor<T>([_hiddenSize]);

        _w1 = CreateRandomTensor([_outputSize, _hiddenSize], Math.Sqrt(2.0 / _hiddenSize), random);
        _b1 = new Tensor<T>([_outputSize]);

        // GLU weights — small so the gate starts near 0.5 (neutral)
        double stdGlu = Math.Sqrt(1.0 / _outputSize);
        _w4 = CreateRandomTensor([_outputSize, _outputSize], stdGlu, random);
        _b4 = new Tensor<T>([_outputSize]);
        _w5 = CreateRandomTensor([_outputSize, _outputSize], stdGlu, random);
        _b5 = new Tensor<T>([_outputSize]);

        if (_inputSize != _outputSize)
        {
            _skipProjection = CreateRandomTensor([_outputSize, _inputSize], Math.Sqrt(1.0 / _inputSize), random);
        }

        _lnGamma = CreateConstantTensor([_outputSize], 1.0);
        _lnBeta = new Tensor<T>([_outputSize]);
    }

    /// <summary>
    /// Forward pass GRN(a) = LayerNorm(a + GLU(η₁)) on a flattened <c>[N, inputSize]</c> batch.
    /// Returns <c>[N, outputSize]</c>. Fully tape-differentiable.
    /// </summary>
    public Tensor<T> Forward(Tensor<T> input)
    {
        // η₂ = ELU(W₂·a + b₂)
        var eta2 = Engine.ELU(Linear(input, _w2, _b2), 1.0);

        // η₁ = W₁·η₂ + b₁
        var eta1 = Linear(eta2, _w1, _b1);

        // GLU(η₁) = σ(W₄·η₁ + b₄) ⊙ (W₅·η₁ + b₅)
        var gate = Engine.Sigmoid(Linear(eta1, _w4, _b4));
        var value = Linear(eta1, _w5, _b5);
        var gluOutput = Engine.TensorMultiply(gate, value);

        // Skip connection (projected when input/output dims differ)
        var skip = _skipProjection != null ? Linear(input, _skipProjection, null) : input;

        // Residual + LayerNorm(γ, β)
        var residual = Engine.TensorAdd(skip, gluOutput);
        return Engine.LayerNorm(residual, _lnGamma, _lnBeta, 1e-5, out _, out _);
    }

    /// <summary>Collects all trainable parameter tensors for tape-based autodiff.</summary>
    public IEnumerable<Tensor<T>> GetTrainableParameters()
    {
        yield return _w2;
        yield return _b2;
        yield return _w1;
        yield return _b1;
        yield return _w4;
        yield return _b4;
        yield return _w5;
        yield return _b5;
        if (_skipProjection != null) yield return _skipProjection;
        yield return _lnGamma;
        yield return _lnBeta;
    }

    // Row-wise affine map: x[N, in] · W[out, in]^T + b[out] -> [N, out].
    private static Tensor<T> Linear(Tensor<T> x, Tensor<T> weight, Tensor<T>? bias)
    {
        int outSize = weight.Shape[0];
        var result = Engine.TensorMatMul(x, Engine.TensorTranspose(weight));
        if (bias != null)
            result = Engine.TensorBroadcastAdd(result, Engine.Reshape(bias, [1, outSize]));
        return result;
    }

    private static Tensor<T> CreateRandomTensor(int[] shape, double stddev, Random random)
    {
        int size = 1;
        foreach (var s in shape) size *= s;
        var data = new T[size];
        var numOps = MathHelper.GetNumericOperations<T>();
        for (int i = 0; i < size; i++)
        {
            double u1 = 1.0 - random.NextDouble();
            double u2 = random.NextDouble();
            double normal = Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2);
            data[i] = numOps.FromDouble(normal * stddev);
        }
        return new Tensor<T>(shape, new Vector<T>(data));
    }

    private static Tensor<T> CreateConstantTensor(int[] shape, double value)
    {
        int size = 1;
        foreach (var s in shape) size *= s;
        var data = new T[size];
        var numOps = MathHelper.GetNumericOperations<T>();
        for (int i = 0; i < size; i++) data[i] = numOps.FromDouble(value);
        return new Tensor<T>(shape, new Vector<T>(data));
    }
}
