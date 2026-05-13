using AiDotNet.ActivationFunctions;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.Tensors.Helpers;
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
/// <para><b>For Beginners:</b> A regular dense (fully-connected) layer
/// learns one weight per (input, output) pair. NoisyDenseLayer learns TWO
/// per pair — a base value <c>μ</c> and a noise scale <c>σ</c> — and at
/// training time draws a fresh random ε every forward pass to form the
/// effective weight <c>W = μ + σ · ε</c>. The network learns when to make
/// σ small (confident, deterministic predictions) vs large (uncertain,
/// exploring different actions). This replaces hand-tuned exploration
/// strategies like ε-greedy in reinforcement-learning agents — the
/// exploration noise is built into the weights and decays naturally as
/// training converges. At evaluation time σ is zeroed out (paper §3.4)
/// so the network is deterministic given a fixed state.</para>
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
    /// <remarks>
    /// Constructor reports <c>InputShape = [-1]</c> to the layer-compatibility
    /// validator so the layer composes cleanly behind shape-agnostic predecessors
    /// (ActivationLayer, DropoutLayer, lazy DenseLayer). The validator skips
    /// strict <c>SequenceEqual</c> comparisons for layers with any non-positive
    /// dimension. The actual input size is fixed at construction time (the σ
    /// initialisation depends on it) so behaviour matches a fixed-shape layer.
    /// </remarks>
    public NoisyDenseLayer(
        int inputSize,
        int outputSize,
        IActivationFunction<T>? activationFunction = null,
        double? sigmaInit = null,
        int? seed = null)
        : base(
            inputShape: [-1],                 // lazy marker — layer validator skips shape-equality
            outputShape: [outputSize],
            scalarActivation: activationFunction ?? new IdentityActivation<T>())
    {
        if (inputSize <= 0) throw new ArgumentOutOfRangeException(nameof(inputSize));
        if (outputSize <= 0) throw new ArgumentOutOfRangeException(nameof(outputSize));

        _inputSize = inputSize;
        _outputSize = outputSize;
        _sigmaInit = sigmaInit ?? (0.5 / Math.Sqrt(inputSize));
        // Route RNG construction through RandomHelper rather than raw
        // `new Random()` — raw Random is not cryptographically secure
        // and its time-seeded default is predictable across closely-spaced
        // instances. RandomHelper.CreateSeededRandom preserves reproducibility
        // when a seed is supplied (Fortunato 2017 §3.3 noise tests rely on
        // deterministic resampling); CreateSecureRandom is the cryptographic
        // default otherwise.
        _rng = seed.HasValue
            ? RandomHelper.CreateSeededRandom(seed.Value)
            : RandomHelper.CreateSecureRandom();

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
        if (parameters.Count != 4)
            throw new ArgumentException("Expected exactly 4 parameter tensors (μ_w, σ_w, μ_b, σ_b).", nameof(parameters));
        CopyTensorInPlace(parameters[0], _muWeights);
        CopyTensorInPlace(parameters[1], _sigmaWeights);
        CopyTensorInPlace(parameters[2], _muBiases);
        CopyTensorInPlace(parameters[3], _sigmaBiases);
    }

    private static void CopyTensorInPlace(Tensor<T> src, Tensor<T> dst)
    {
        // Validate full shape, not just flat length. A [2, 6] source has the
        // same Length as a [3, 4] destination but copying the elements
        // straight across would silently scramble the layer's weight matrix
        // by row/column. Reject rank and per-dim mismatches explicitly.
        if (src.Rank != dst.Rank || src.Length != dst.Length)
            throw new ArgumentException(
                $"Shape mismatch: source rank={src.Rank} length={src.Length}, " +
                $"destination rank={dst.Rank} length={dst.Length}.");
        for (int dim = 0; dim < src.Rank; dim++)
        {
            if (src.Shape[dim] != dst.Shape[dim])
                throw new ArgumentException(
                    $"Shape mismatch at dim {dim}: source={src.Shape[dim]}, destination={dst.Shape[dim]}. " +
                    "Full shape must match — same-length tensors of different ranks/shapes would scramble weights.");
        }
        for (int i = 0; i < src.Length; i++) dst[i] = src[i];
    }

    /// <inheritdoc/>
    public override Tensor<T> Forward(Tensor<T> input)
    {
        // Training mode: resample noise (Fortunato 2017 §3.3 — one resample
        // per environment step / minibatch). Eval mode: use mean weights only
        // (paper §3.4: at evaluation the network uses μ and the σ term is set
        // to zero), which makes the policy deterministic given a fixed state.
        Tensor<T> wEff, bEff;
        if (IsTrainingMode)
        {
            var (epsW, epsB) = SampleFactorisedNoise();
            // W_eff = μ_w + σ_w ⊙ ε_w; tape captures ∂L/∂μ = ∂L/∂W_eff and
            // ∂L/∂σ = ∂L/∂W_eff ⊙ ε_w.
            var wNoise = Engine.TensorMultiply(_sigmaWeights, epsW);
            wEff = Engine.TensorAdd(_muWeights, wNoise);
            var bNoise = Engine.TensorMultiply(_sigmaBiases, epsB);
            bEff = Engine.TensorAdd(_muBiases, bNoise);
        }
        else
        {
            wEff = _muWeights;
            bEff = _muBiases;
        }

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

        var pre = Engine.TensorMatMul(flatInput, wEff);
        var bBroadcast = Engine.Reshape(bEff, [1, _outputSize]);
        pre = Engine.TensorAdd(pre, bBroadcast);

        var activated = ApplyActivation(pre);

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
    /// noise tensors via an engine-accelerated outer product:
    /// <c>ε_w = f(ε_in)ᵀ · f(ε_out)</c> (Fortunato 2017 §3.2). Public for
    /// advanced callers that want to sync noise across multiple layers.
    /// </summary>
    /// <remarks>
    /// Only the Gaussian RNG step runs on the CPU (Box-Muller via
    /// <see cref="Math"/>). The signed-sqrt transform and outer product are
    /// routed through <see cref="LayerBase{T}.Engine"/> ops so they pick up
    /// BLAS / GPU acceleration and the gradient tape sees them as part of
    /// the same compute graph as the rest of the forward pass.
    /// </remarks>
    public (Tensor<T> EpsilonW, Tensor<T> EpsilonB) SampleFactorisedNoise()
    {
        // Sample raw Gaussian vectors. RNG is intrinsically scalar; we
        // populate two Tensor<T>s in one pass per side.
        var rawIn = new Tensor<T>([_inputSize, 1]);
        for (int i = 0; i < _inputSize; i++)
            rawIn[i, 0] = NumOps.FromDouble(SampleStandardNormal());

        var rawOut = new Tensor<T>([1, _outputSize]);
        for (int j = 0; j < _outputSize; j++)
            rawOut[0, j] = NumOps.FromDouble(SampleStandardNormal());

        // Signed-sqrt transform f(x) = sign(x) · √|x|. Engine ops:
        //   f(x) = sign(x) · sqrt(|x|) = sign(x) · sqrt(x · sign(x))
        // Build via TensorAbs → TensorSqrt → TensorSign then multiply.
        var fIn = SignedSqrt(rawIn);
        var fOut = SignedSqrt(rawOut);

        // Outer product f(ε_in)ᵀ · f(ε_out): [p, 1] @ [1, q] → [p, q].
        // This is exactly Engine.TensorMatMul — BLAS-routed.
        var epsW = Engine.TensorMatMul(fIn, fOut);

        // ε_b is just f(ε_out) reshaped to [q].
        var epsB = Engine.Reshape(fOut, [_outputSize]);

        return (epsW, epsB);
    }

    /// <summary>
    /// Element-wise signed-square-root: <c>f(x) = sign(x) · √|x|</c>, the
    /// per-element transform Fortunato 2017 §3.2 specifies for factorised
    /// noise. Built from engine ops so it stays generic in T and runs on
    /// whatever hardware the engine is bound to.
    /// </summary>
    private Tensor<T> SignedSqrt(Tensor<T> x)
    {
        // |x|
        var absX = Engine.TensorAbs(x);
        // √|x|
        var sqrtAbs = Engine.TensorSqrt(absX);
        // sign(x) — engine sign is available; fall back to safe division
        // (x / (|x| + ε)) if the engine doesn't expose a TensorSign.
        var sign = Engine.TensorSign(x);
        return Engine.TensorMultiply(sign, sqrtAbs);
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
    /// SGD-style fallback retained for callers using the legacy per-layer
    /// UpdateParameters API; reads gradients from <see cref="LayerBase{T}.ParameterGradients"/>.
    /// Tape-based training (the standard path) updates μ/σ tensors in-place via
    /// the optimizer step and doesn't go through this method.
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

    /// <summary>
    /// Persists the constructor parameters needed by
    /// <c>DeserializationHelper</c> to reconstruct an identical layer
    /// post-Clone. Without this override, a Clone would fall back to the
    /// constructor's defaults and lose the configured InputSize /
    /// OutputSize / SigmaInit. The activation type is already covered by
    /// the base override (which writes ScalarActivationType /
    /// VectorActivationType).
    /// </summary>
    internal override Dictionary<string, string> GetMetadata()
    {
        var metadata = base.GetMetadata();
        var inv = System.Globalization.CultureInfo.InvariantCulture;
        metadata["InputSize"] = _inputSize.ToString(inv);
        metadata["OutputSize"] = _outputSize.ToString(inv);
        metadata["SigmaInit"] = _sigmaInit.ToString("R", inv);
        return metadata;
    }
}
