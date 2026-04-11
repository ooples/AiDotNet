using AiDotNet.Tensors;
using AiDotNet.Tensors.Engines;

namespace AiDotNet.TimeSeries.TFT;

/// <summary>
/// Gated Residual Network (GRN) as described in Lim et al. (2021).
/// GRN(a, c) = LayerNorm(a + GLU(η₁)) where η₁ = W₁·η₂ + b₁,
/// η₂ = ELU(W₂·a + W₃·c + b₂), and GLU(γ) = σ(W₄·γ + b₄) ⊙ (W₅·γ + b₅).
/// </summary>
internal class GatedResidualNetwork<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();
    private static IEngine Engine => AiDotNetEngine.Current;

    private readonly int _inputSize;
    private readonly int _hiddenSize;
    private readonly int _outputSize;
    private readonly int _contextSize;
    private readonly bool _hasContext;

    // η₂ = ELU(W₂·a + W₃·c + b₂)
    private Tensor<T> _w2; // [hiddenSize, inputSize]
    private Tensor<T> _b2; // [hiddenSize]
    private Tensor<T>? _w3; // [hiddenSize, contextSize] — only if context is used

    // η₁ = W₁·η₂ + b₁
    private Tensor<T> _w1; // [outputSize, hiddenSize]
    private Tensor<T> _b1; // [outputSize]

    // GLU: σ(W₄·γ + b₄) ⊙ (W₅·γ + b₅)
    private Tensor<T> _w4; // [outputSize, outputSize]
    private Tensor<T> _b4; // [outputSize]
    private Tensor<T> _w5; // [outputSize, outputSize]
    private Tensor<T> _b5; // [outputSize]

    // Skip connection projection (when inputSize != outputSize)
    private Tensor<T>? _skipProjection; // [outputSize, inputSize]

    public GatedResidualNetwork(int inputSize, int hiddenSize, int outputSize, int contextSize = 0, int? seed = null)
    {
        _inputSize = inputSize;
        _hiddenSize = hiddenSize;
        _outputSize = outputSize;
        _contextSize = contextSize;
        _hasContext = contextSize > 0;

        var random = seed.HasValue ? new Random(seed.Value) : RandomHelper.CreateSeededRandom(42);

        // He initialization for ELU
        double std2 = Math.Sqrt(2.0 / _inputSize);
        _w2 = CreateRandomTensor([_hiddenSize, _inputSize], std2, random);
        _b2 = new Tensor<T>([_hiddenSize]);

        if (_hasContext)
        {
            double std3 = Math.Sqrt(2.0 / _contextSize);
            _w3 = CreateRandomTensor([_hiddenSize, _contextSize], std3, random);
        }

        double std1 = Math.Sqrt(2.0 / _hiddenSize);
        _w1 = CreateRandomTensor([_outputSize, _hiddenSize], std1, random);
        _b1 = new Tensor<T>([_outputSize]);

        // GLU weights — initialize small so gate starts near 0.5 (neutral)
        double stdGlu = Math.Sqrt(1.0 / _outputSize);
        _w4 = CreateRandomTensor([_outputSize, _outputSize], stdGlu, random);
        _b4 = new Tensor<T>([_outputSize]);
        _w5 = CreateRandomTensor([_outputSize, _outputSize], stdGlu, random);
        _b5 = new Tensor<T>([_outputSize]);

        // Skip projection if dimensions don't match
        if (_inputSize != _outputSize)
        {
            _skipProjection = CreateRandomTensor([_outputSize, _inputSize], Math.Sqrt(1.0 / _inputSize), random);
        }
    }

    /// <summary>
    /// Forward pass: GRN(a, c) = LayerNorm(a + GLU(η₁))
    /// </summary>
    /// <param name="input">Primary input [batch, inputSize] or [inputSize]</param>
    /// <param name="context">Optional static context [batch, contextSize] or [contextSize]</param>
    public Tensor<T> Forward(Tensor<T> input, Tensor<T>? context = null)
    {
        // η₂ = ELU(W₂·a + W₃·c + b₂)
        var eta2 = LinearProject(input, _w2, _b2);

        if (_hasContext && context != null && _w3 != null)
        {
            var contextProj = LinearProject(context, _w3, null);
            eta2 = Engine.TensorAdd(eta2, contextProj);
        }

        eta2 = ApplyELU(eta2);

        // η₁ = W₁·η₂ + b₁
        var eta1 = LinearProject(eta2, _w1, _b1);

        // GLU(η₁) = σ(W₄·η₁ + b₄) ⊙ (W₅·η₁ + b₅)
        var gate = LinearProject(eta1, _w4, _b4);
        gate = ApplySigmoid(gate);
        var value = LinearProject(eta1, _w5, _b5);
        var gluOutput = Engine.TensorMultiply(gate, value);

        // Skip connection: a (or projected a if sizes differ)
        var skip = (_skipProjection != null)
            ? LinearProject(input, _skipProjection, null)
            : input;

        // Residual + LayerNorm
        var output = Engine.TensorAdd(skip, gluOutput);
        output = ApplyLayerNorm(output);

        return output;
    }

    /// <summary>
    /// Collects all trainable parameters for tape-based autodiff.
    /// </summary>
    public IEnumerable<Tensor<T>> GetTrainableParameters()
    {
        yield return _w2;
        yield return _b2;
        if (_w3 != null) yield return _w3;
        yield return _w1;
        yield return _b1;
        yield return _w4;
        yield return _b4;
        yield return _w5;
        yield return _b5;
        if (_skipProjection != null) yield return _skipProjection;
    }

    private static Tensor<T> LinearProject(Tensor<T> input, Tensor<T> weight, Tensor<T>? bias)
    {
        // input: [N] or [batch, N], weight: [outSize, inSize]
        int outSize = weight.Shape[0];
        int inSize = weight.Shape[1];

        Tensor<T> result;
        if (input.Shape.Length == 1 || (input.Shape.Length == 2 && input.Shape[0] == 1))
        {
            // Single vector: weight @ input
            var inputCol = input.Reshape(inSize, 1);
            result = Engine.TensorMatMul(weight, inputCol).Reshape(outSize);
        }
        else
        {
            // Batched: input @ weight^T
            var weightT = weight.Transpose(new[] { 1, 0 });
            result = Engine.TensorMatMul(input, weightT);
        }

        if (bias != null)
        {
            result = Engine.TensorAdd(result, bias.Reshape(result._shape));
        }

        return result;
    }

    private static Tensor<T> ApplyELU(Tensor<T> x)
    {
        var result = new Tensor<T>(x._shape);
        var xSpan = x.Data.Span;
        var rSpan = result.AsWritableSpan();
        for (int i = 0; i < xSpan.Length; i++)
        {
            double val = NumOps.ToDouble(xSpan[i]);
            rSpan[i] = NumOps.FromDouble(val >= 0 ? val : Math.Exp(val) - 1.0);
        }
        return result;
    }

    private static Tensor<T> ApplySigmoid(Tensor<T> x)
    {
        var result = new Tensor<T>(x._shape);
        var xSpan = x.Data.Span;
        var rSpan = result.AsWritableSpan();
        for (int i = 0; i < xSpan.Length; i++)
        {
            double val = NumOps.ToDouble(xSpan[i]);
            rSpan[i] = NumOps.FromDouble(1.0 / (1.0 + Math.Exp(-val)));
        }
        return result;
    }

    private static Tensor<T> ApplyLayerNorm(Tensor<T> x)
    {
        // Simple layer normalization over the last dimension
        var span = x.Data.Span;
        int len = span.Length;
        if (len == 0) return x;

        double sum = 0, sumSq = 0;
        for (int i = 0; i < len; i++)
        {
            double v = NumOps.ToDouble(span[i]);
            sum += v;
            sumSq += v * v;
        }

        double mean = sum / len;
        double variance = sumSq / len - mean * mean;
        double invStd = 1.0 / Math.Sqrt(variance + 1e-5);

        var result = new Tensor<T>(x._shape);
        var rSpan = result.AsWritableSpan();
        for (int i = 0; i < len; i++)
        {
            rSpan[i] = NumOps.FromDouble((NumOps.ToDouble(span[i]) - mean) * invStd);
        }
        return result;
    }

    private static Tensor<T> CreateRandomTensor(int[] shape, double stddev, Random random)
    {
        int size = 1;
        foreach (var s in shape) size *= s;
        var data = new T[size];
        for (int i = 0; i < size; i++)
        {
            // Normal distribution via Box-Muller
            double u1 = 1.0 - random.NextDouble();
            double u2 = random.NextDouble();
            double normal = Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2);
            data[i] = NumOps.FromDouble(normal * stddev);
        }
        return new Tensor<T>(shape, new Vector<T>(data));
    }
}
