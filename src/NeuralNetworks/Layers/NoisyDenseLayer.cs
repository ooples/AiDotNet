using AiDotNet.ActivationFunctions;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.NeuralNetworks.Layers;

/// <summary>
/// Noisy linear layer for exploration in reinforcement learning (Fortunato et al.
/// 2017 "Noisy Networks for Exploration", §3.2 Factorised Gaussian variant).
/// Replaces a conventional dense layer's deterministic weights with parametric
/// noise: <c>W = μ_w + σ_w ⊙ ε_w</c>, <c>b = μ_b + σ_b ⊙ ε_b</c>, where ε is
/// resampled on every forward pass. σ is learned jointly with μ, so the network
/// decides per-weight how much exploration noise to inject.
/// </summary>
/// <typeparam name="T">Numeric type for the layer.</typeparam>
/// <remarks>
/// <para>
/// Factorised noise generates two independent Gaussian vectors ε_in (size p)
/// and ε_out (size q), applies <c>f(x) = sign(x)·√|x|</c> to each, and forms
/// the weight noise matrix as the outer product
/// <c>ε_w[i,j] = f(ε_in[i]) · f(ε_out[j])</c>. This needs p+q random draws
/// instead of p·q for the independent variant (paper §3.2).
/// </para>
/// <para>
/// Initialisation follows Fortunato 2017 Eqs. 17–18:
/// <c>μ ~ U(-1/√p, 1/√p)</c>, <c>σ_init = 0.5/√p</c> for both weights and biases.
/// </para>
/// <para>
/// All forward arithmetic is routed through <see cref="LayerBase{T}.Engine"/>
/// ops on the same tensor instances returned by <see cref="GetTrainableParameters"/>,
/// so the gradient tape automatically captures gradients with respect to μ_w,
/// σ_w, μ_b, and σ_b. The ε tensors are rebuilt per forward and are NOT
/// trainable — the tape treats them as input data.
/// </para>
/// </remarks>
public class NoisyDenseLayer<T> : LayerBase<T>
{
    private readonly int _inputSize;
    private readonly int _outputSize;
    private readonly double _sigmaInit;
    private readonly Random _rng;

    // Trainable parameters — held as tensor fields so the tape can track them
    // by reference identity. Mutating their contents in-place (via SetParameters
    // or UpdateParameters) keeps the tape wiring intact.
    private readonly Tensor<T> _muWeights;
    private readonly Tensor<T> _sigmaWeights;
    private readonly Tensor<T> _muBiases;
    private readonly Tensor<T> _sigmaBiases;

    /// <summary>
    /// Creates a new noisy dense layer.
    /// </summary>
    /// <param name="inputSize">Number of input features.</param>
    /// <param name="outputSize">Number of output features.</param>
    /// <param name="activationFunction">Optional element-wise activation. Defaults to identity.</param>
    /// <param name="sigmaInit">Initial σ scale; Fortunato 2017 Eq. 18 uses 0.5/√p.</param>
    /// <param name="seed">RNG seed for noise + initialisation. Defaults to a non-deterministic seed.</param>
    public NoisyDenseLayer(
        int inputSize,
        int outputSize,
        IActivationFunction<T>? activationFunction = null,
        double? sigmaInit = null,
        int? seed = null)
        : base(
            inputShape: [inputSize],
            outputShape: [outputSize],
            scalarActivation: activationFunction ?? new IdentityActivation<T>())
    {
        if (inputSize <= 0) throw new ArgumentOutOfRangeException(nameof(inputSize));
        if (outputSize <= 0) throw new ArgumentOutOfRangeException(nameof(outputSize));

        _inputSize = inputSize;
        _outputSize = outputSize;
        _sigmaInit = sigmaInit ?? (0.5 / Math.Sqrt(inputSize));
        _rng = seed.HasValue ? new Random(seed.Value) : new Random();

        _muWeights = new Tensor<T>([inputSize, outputSize]);
        _sigmaWeights = new Tensor<T>([inputSize, outputSize]);
        _muBiases = new Tensor<T>([outputSize]);
        _sigmaBiases = new Tensor<T>([outputSize]);

        InitializeParameters();
    }

    /// <inheritdoc/>
    public override long ParameterCount => 2L * _inputSize * _outputSize + 2L * _outputSize;

    /// <inheritdoc/>
    public override bool SupportsTraining => true;

    private void InitializeParameters()
    {
        // Fortunato 2017 Eq. 17: μ ~ U(-range, range), σ = constant init.
        double range = 1.0 / Math.Sqrt(_inputSize);
        for (int i = 0; i < _inputSize; i++)
        {
            for (int j = 0; j < _outputSize; j++)
            {
                _muWeights[i, j] = NumOps.FromDouble((_rng.NextDouble() * 2.0 - 1.0) * range);
                _sigmaWeights[i, j] = NumOps.FromDouble(_sigmaInit);
            }
        }
        for (int j = 0; j < _outputSize; j++)
        {
            _muBiases[j] = NumOps.FromDouble((_rng.NextDouble() * 2.0 - 1.0) * range);
            _sigmaBiases[j] = NumOps.FromDouble(_sigmaInit);
        }
    }

    /// <inheritdoc/>
    public override IReadOnlyList<Tensor<T>> GetTrainableParameters() =>
        new[] { _muWeights, _sigmaWeights, _muBiases, _sigmaBiases };

    /// <inheritdoc/>
    public override void SetTrainableParameters(IReadOnlyList<Tensor<T>> parameters)
    {
        // The base contract expects the same shapes; we copy values in-place
        // so the existing tensor field references stay live (the tape keys
        // gradients by reference identity).
        if (parameters.Count != 4)
            throw new ArgumentException("Expected exactly 4 parameter tensors (μ_w, σ_w, μ_b, σ_b).", nameof(parameters));
        CopyTensorInPlace(parameters[0], _muWeights);
        CopyTensorInPlace(parameters[1], _sigmaWeights);
        CopyTensorInPlace(parameters[2], _muBiases);
        CopyTensorInPlace(parameters[3], _sigmaBiases);
    }

    private static void CopyTensorInPlace(Tensor<T> src, Tensor<T> dst)
    {
        if (src.Length != dst.Length)
            throw new ArgumentException(
                $"Shape mismatch: source has {src.Length} elements, destination has {dst.Length}.");
        for (int i = 0; i < src.Length; i++) dst[i] = src[i];
    }

    /// <inheritdoc/>
    public override Tensor<T> Forward(Tensor<T> input)
    {
        // Resample noise on every forward (Fortunato 2017 §3.3: one resample
        // per environment step / minibatch).
        var (epsW, epsB) = SampleFactorisedNoise();

        // Build effective weights/biases through engine ops so the tape captures
        // gradients ∂L/∂μ_w = ∂L/∂W_eff and ∂L/∂σ_w = ∂L/∂W_eff ⊙ ε_w.
        // W_eff = μ_w + σ_w ⊙ ε_w
        var wNoise = Engine.TensorMultiply(_sigmaWeights, epsW);
        var wEff = Engine.TensorAdd(_muWeights, wNoise);
        // b_eff = μ_b + σ_b ⊙ ε_b
        var bNoise = Engine.TensorMultiply(_sigmaBiases, epsB);
        var bEff = Engine.TensorAdd(_muBiases, bNoise);

        // Flatten input to [batch, inputSize] for matmul.
        int batchSize;
        Tensor<T> flatInput;
        if (input.Rank == 1)
        {
            batchSize = 1;
            flatInput = Engine.Reshape(input, [1, _inputSize]);
        }
        else if (input.Rank == 2)
        {
            batchSize = input.Shape[0];
            flatInput = input;
        }
        else
        {
            batchSize = 1;
            for (int i = 0; i < input.Rank - 1; i++) batchSize *= input.Shape[i];
            flatInput = Engine.Reshape(input, [batchSize, _inputSize]);
        }

        // pre = flatInput @ W_eff
        var pre = Engine.TensorMatMul(flatInput, wEff);
        // pre = pre + b_eff (broadcast over batch axis)
        var bBroadcast = Engine.Reshape(bEff, [1, _outputSize]);
        pre = Engine.TensorAdd(pre, bBroadcast);

        var activated = ApplyActivation(pre);

        // Restore output rank.
        if (input.Rank == 1) return Engine.Reshape(activated, [_outputSize]);
        if (input.Rank == 2) return activated;
        var outShape = new int[input.Rank];
        for (int i = 0; i < input.Rank - 1; i++) outShape[i] = input.Shape[i];
        outShape[^1] = _outputSize;
        return Engine.Reshape(activated, outShape);
    }

    /// <summary>
    /// Resamples ε_in (size p) and ε_out (size q), applies the signed-sqrt
    /// transform <c>f(x) = sign(x)·√|x|</c>, and builds the per-forward
    /// noise tensors ε_w (outer product, shape [p, q]) and ε_b (= f(ε_out)).
    /// Public for advanced callers that want to sync noise across multiple
    /// layers (e.g., per-rollout reset in deep RL).
    /// </summary>
    public (Tensor<T> EpsilonW, Tensor<T> EpsilonB) SampleFactorisedNoise()
    {
        var fIn = new T[_inputSize];
        for (int i = 0; i < _inputSize; i++)
        {
            double g = SampleStandardNormal();
            fIn[i] = NumOps.FromDouble(Math.Sign(g) * Math.Sqrt(Math.Abs(g)));
        }
        var fOut = new T[_outputSize];
        for (int j = 0; j < _outputSize; j++)
        {
            double g = SampleStandardNormal();
            fOut[j] = NumOps.FromDouble(Math.Sign(g) * Math.Sqrt(Math.Abs(g)));
        }

        var epsW = new Tensor<T>([_inputSize, _outputSize]);
        for (int i = 0; i < _inputSize; i++)
            for (int j = 0; j < _outputSize; j++)
                epsW[i, j] = NumOps.Multiply(fIn[i], fOut[j]);

        var epsB = new Tensor<T>([_outputSize]);
        for (int j = 0; j < _outputSize; j++) epsB[j] = fOut[j];

        return (epsW, epsB);
    }

    private double SampleStandardNormal()
    {
        // Box-Muller transform; one of the two outputs is discarded for clarity.
        double u1 = 1.0 - _rng.NextDouble();
        double u2 = 1.0 - _rng.NextDouble();
        return Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2);
    }

    /// <inheritdoc/>
    public override Vector<T> GetParameters()
    {
        var p = new Vector<T>((int)ParameterCount);
        int idx = 0;
        for (int i = 0; i < _muWeights.Length; i++) p[idx++] = _muWeights[i];
        for (int i = 0; i < _sigmaWeights.Length; i++) p[idx++] = _sigmaWeights[i];
        for (int j = 0; j < _muBiases.Length; j++) p[idx++] = _muBiases[j];
        for (int j = 0; j < _sigmaBiases.Length; j++) p[idx++] = _sigmaBiases[j];
        return p;
    }

    /// <inheritdoc/>
    public override void SetParameters(Vector<T> parameters)
    {
        if (parameters.Length != ParameterCount)
            throw new ArgumentException(
                $"Expected {ParameterCount} parameters, got {parameters.Length}.", nameof(parameters));
        int idx = 0;
        for (int i = 0; i < _muWeights.Length; i++) _muWeights[i] = parameters[idx++];
        for (int i = 0; i < _sigmaWeights.Length; i++) _sigmaWeights[i] = parameters[idx++];
        for (int j = 0; j < _muBiases.Length; j++) _muBiases[j] = parameters[idx++];
        for (int j = 0; j < _sigmaBiases.Length; j++) _sigmaBiases[j] = parameters[idx++];
    }

    /// <inheritdoc/>
    /// <remarks>
    /// Gradient-tape-driven training paths supply gradients via the optimizer
    /// step (which mutates μ/σ tensors in place using their tape-tracked grads).
    /// This SGD-style fallback is retained for callers that use the legacy
    /// per-layer UpdateParameters API; it reads gradients from
    /// <see cref="LayerBase{T}.ParameterGradients"/> in the same flat layout as
    /// <see cref="GetParameters"/>/<see cref="SetParameters"/>.
    /// </remarks>
    public override void UpdateParameters(T learningRate)
    {
        if (ParameterGradients is null || ParameterGradients.Length != ParameterCount) return;

        int idx = 0;
        for (int i = 0; i < _muWeights.Length; i++)
            _muWeights[i] = NumOps.Subtract(_muWeights[i],
                NumOps.Multiply(learningRate, ParameterGradients[idx++]));
        for (int i = 0; i < _sigmaWeights.Length; i++)
            _sigmaWeights[i] = NumOps.Subtract(_sigmaWeights[i],
                NumOps.Multiply(learningRate, ParameterGradients[idx++]));
        for (int j = 0; j < _muBiases.Length; j++)
            _muBiases[j] = NumOps.Subtract(_muBiases[j],
                NumOps.Multiply(learningRate, ParameterGradients[idx++]));
        for (int j = 0; j < _sigmaBiases.Length; j++)
            _sigmaBiases[j] = NumOps.Subtract(_sigmaBiases[j],
                NumOps.Multiply(learningRate, ParameterGradients[idx++]));
    }

    /// <inheritdoc/>
    public override void ResetState() { /* no per-forward state cached outside the tape */ }
}
