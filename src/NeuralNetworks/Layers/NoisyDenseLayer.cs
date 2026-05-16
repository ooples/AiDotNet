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
    private readonly object _rngLock = new();

    // Trainable parameters — held as tensor fields so the tape can track them
    // by reference identity. Fields are NOT readonly because
    // NeuralNetworkBase.GetOrCreateParameterBuffer rebinds them to views into
    // the contiguous ParameterBuffer via SetTrainableParameters; the tape's
    // TapeStepContext.ValidateBufferAlignment then requires every tensor it
    // sees during forward to be the same reference the buffer holds. Copying
    // data into the old standalone tensors would leave Forward() using
    // standalone-tensor references the tape rejects with "Parameter N is not
    // a view into the provided ParameterBuffer" — the path
    // RainbowDQNAgent.Train(state, target) takes when called for offline
    // pretraining or BC warm-start.
    private Tensor<T> _muWeights;
    private Tensor<T> _sigmaWeights;
    private Tensor<T> _muBiases;
    private Tensor<T> _sigmaBiases;

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
        double resolvedSigmaInit = sigmaInit ?? (0.5 / Math.Sqrt(inputSize));
        if (double.IsNaN(resolvedSigmaInit) || double.IsInfinity(resolvedSigmaInit) || resolvedSigmaInit < 0d)
            throw new ArgumentOutOfRangeException(
                nameof(sigmaInit),
                resolvedSigmaInit,
                "sigmaInit must be finite and non-negative — passing NaN / Inf / a negative " +
                "value would propagate into every _sigma* tensor and corrupt the layer state " +
                "long before an actionable exception surfaces.");
        _sigmaInit = resolvedSigmaInit;
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
    /// <remarks>
    /// Replaces the field tensor references with the supplied tensors rather
    /// than copying data into the old ones. The ParameterBuffer machinery in
    /// <see cref="NeuralNetworkBase{T}.GetOrCreateParameterBuffer"/> calls
    /// this with buffer-backed views; subsequent Forward() must use those
    /// view tensors so the tape's reference-identity alignment check passes
    /// (TapeStepContext.ValidateBufferAlignment, AiDotNet.Tensors).
    /// Validate shapes per-dim (rank + every dim) so a same-length but
    /// differently-shaped tensor doesn't silently scramble the layer's
    /// weights — a [2, 6] tensor and a [3, 4] tensor have the same flat
    /// length but rebinding _muWeights from the former to the latter would
    /// rotate every weight-index pair.
    /// </remarks>
    public override void SetTrainableParameters(IReadOnlyList<Tensor<T>> parameters)
    {
        if (parameters.Count != 4)
            throw new ArgumentException("Expected exactly 4 parameter tensors (μ_w, σ_w, μ_b, σ_b).", nameof(parameters));
        ValidateShapeMatch(parameters[0], _muWeights, nameof(_muWeights));
        ValidateShapeMatch(parameters[1], _sigmaWeights, nameof(_sigmaWeights));
        ValidateShapeMatch(parameters[2], _muBiases, nameof(_muBiases));
        ValidateShapeMatch(parameters[3], _sigmaBiases, nameof(_sigmaBiases));
        _muWeights = parameters[0];
        _sigmaWeights = parameters[1];
        _muBiases = parameters[2];
        _sigmaBiases = parameters[3];
    }

    private static void ValidateShapeMatch(Tensor<T> incoming, Tensor<T> existing, string paramName)
    {
        if (incoming.Rank != existing.Rank || incoming.Length != existing.Length)
            throw new ArgumentException(
                $"Shape mismatch for {paramName}: incoming rank={incoming.Rank} length={incoming.Length}, " +
                $"existing rank={existing.Rank} length={existing.Length}.");
        for (int dim = 0; dim < incoming.Rank; dim++)
        {
            if (incoming.Shape[dim] != existing.Shape[dim])
                throw new ArgumentException(
                    $"Shape mismatch for {paramName} at dim {dim}: incoming={incoming.Shape[dim]}, " +
                    $"existing={existing.Shape[dim]}. Full shape must match — same-length tensors " +
                    "of different ranks/shapes would scramble weights.");
        }
    }

    /// <inheritdoc/>
    public override Tensor<T> Forward(Tensor<T> input)
    {
        // Validate at the boundary — the ctor uses inputShape: [-1] so the
        // base class's Forward never gets a chance to catch a shape mismatch.
        // Without this guard, a wrong feature size first surfaces as an
        // engine-specific Reshape or TensorMatMul failure several lines down.
        if (input is null) throw new ArgumentNullException(nameof(input));
        if (input.Rank == 0)
            throw new ArgumentException(
                "NoisyDenseLayer expects an input with at least one dimension.",
                nameof(input));
        int featureSize = input.Rank == 1 ? input.Length : input.Shape[input.Rank - 1];
        if (featureSize != _inputSize)
            throw new ArgumentException(
                $"NoisyDenseLayer expects last-dim feature size {_inputSize}, got {featureSize} " +
                $"(input shape [{string.Join(",", input.Shape)}]).",
                nameof(input));

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
    /// <c>ε_w = f(ε_in)ᵀ · f(ε_out)</c> (Fortunato 2017 §3.2). Internal
    /// plumbing: Forward() always resamples its own noise, so an external
    /// "sync the same noise across layers" use case cannot be implemented
    /// just by calling this helper — exposing it publicly would only let
    /// callers desync the RNG.
    /// </summary>
    /// <remarks>
    /// Only the Gaussian RNG step runs on the CPU (Box-Muller via
    /// <see cref="Math"/>). The signed-sqrt transform and outer product are
    /// routed through <see cref="LayerBase{T}.Engine"/> ops so they pick up
    /// BLAS / GPU acceleration and the gradient tape sees them as part of
    /// the same compute graph as the rest of the forward pass.
    /// </remarks>
    internal (Tensor<T> EpsilonW, Tensor<T> EpsilonB) SampleFactorisedNoise()
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
        // sign(x) via the documented safe-division fallback x / (|x| + ε).
        // Avoids dependency on Engine.TensorSign which isn't part of the
        // baseline IEngine surface (LionOptimizer / FTRLOptimizer use the
        // same x/(|x|+ε) trick). ε keeps the denominator non-zero for the
        // exact zero element of the noise vectors.
        var eps = NumOps.FromDouble(1e-12);
        var absXPlusEps = Engine.TensorAddScalar(absX, eps);
        var sign = Engine.TensorDivide(x, absXPlusEps);
        return Engine.TensorMultiply(sign, sqrtAbs);
    }

    private double SampleStandardNormal()
    {
        // Box-Muller transform; one of the two outputs is discarded for clarity.
        // Guard the two _rng calls with a lock — System.Random is not
        // thread-safe and modern training frameworks (DataParallel, RLlib
        // workers, etc.) can call Forward concurrently on the same layer.
        // Without the lock, racing NextDouble calls can corrupt the RNG's
        // internal state and break reproducibility.
        double u1, u2;
        lock (_rngLock)
        {
            u1 = 1.0 - _rng.NextDouble();
            u2 = 1.0 - _rng.NextDouble();
        }
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
        // ParameterGradients is allowed to be null (e.g. before the first
        // backward pass), but a non-null gradient buffer with the wrong
        // length signals a real wiring bug — silently returning would stall
        // training without surfacing the cause.
        if (ParameterGradients is null) return;
        if (ParameterGradients.Length != ParameterCount)
            throw new InvalidOperationException(
                $"NoisyDenseLayer.UpdateParameters: gradient buffer length " +
                $"{ParameterGradients.Length} does not match ParameterCount {ParameterCount}.");

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
