using AiDotNet.Attributes;
using AiDotNet.Initialization;

namespace AiDotNet.NeuralNetworks.Layers;

/// <summary>
/// Implements the decoder of an Occupancy Network (Mescheder et al., CVPR 2019,
/// arXiv:1812.03828): the fully-connected, conditional-normalization ResNet that
/// maps a queried 3D point <c>p ∈ ℝ³</c> to an occupancy probability,
/// conditioned on a latent code <c>c</c> describing the shape.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// <b>Architecture (paper §4.1 / supplementary "Network Architecture", the
/// <c>DecoderCBN</c> module).</b> A linear point embedding feeds a stack of
/// pre-activation Conditional-ResNet blocks, then a final CBN → ReLU → linear
/// produces the occupancy logit:
/// <code>
///   net = fc_p(p)                                 # Linear(3 → hidden)
///   for each block:                               # pre-activation CResNet block
///       h   = fc_0(ReLU(CN_0(net, c)))            # Linear(hidden → hidden)
///       dx  = fc_1(ReLU(CN_1(h,   c)))            # Linear(hidden → hidden)
///       net = net + dx                            # residual skip (identity, in == out)
///   out = σ(fc_out(ReLU(CN(net, c))))             # Linear(hidden → 1) + sigmoid → occupancy probability
/// </code>
/// where Conditional Normalization is
/// <c>CN(x, c) = γ(c) ⊙ Norm(x) + β(c)</c> and <c>γ(c)</c>, <c>β(c)</c>
/// are linear projections of the latent code (the paper predicts the
/// normalization affine from the condition rather than learning it
/// unconditionally — this is what makes the block "conditional").
/// </para>
/// <para>
/// <b>Layer- vs Batch-Normalization.</b> The paper's <c>Norm</c> is Batch
/// Normalization, which is well-defined because it trains on large point
/// batches (T ≈ 2048 sampled points per shape). This model's public surface
/// evaluates a single point at a time (batch = 1), where batch statistics
/// collapse (variance = 0, so BatchNorm zeroes the signal and starves the
/// gradient — the exact failure that makes a single-point CBN untrainable).
/// We therefore normalize per-sample across the feature axis (Layer
/// Normalization, Ba et al. 2016) instead of across the batch. The conditional
/// affine, pre-activation residual structure, block count, and latent
/// conditioning are unchanged — only the normalization axis differs, so the
/// architecture stays faithful while remaining well-defined at any batch size.
/// The residual skips carry the input magnitude past each (scale-invariant)
/// normalization, so the decoder still responds to input scaling.
/// </para>
/// <para>
/// <b>Conditioning without an encoder (auto-decoder).</b> The paper produces
/// <c>c</c> from an encoder over an observation (image / point cloud). When the
/// model has no observation to encode — as here, where the public surface is
/// just point → occupancy — the standard substitute is a learnable latent code
/// (the DeepSDF auto-decoder of Park et al. 2019): a single trained vector that
/// the decoder conditions on, so the network represents one occupancy field.
/// The code is realized as a <see cref="DenseLayer{T}"/> driven by a constant
/// input of <c>1</c>, so it trains through the same tape/optimizer path as every
/// other parameter (no bespoke trainable-tensor wiring).
/// </para>
/// <para>
/// Backward is handled by the gradient tape: every operation here is either an
/// <c>Engine</c> op or a sub-layer <c>Forward</c>, so <c>TrainWithTape</c> records
/// the full graph and there is no hand-written backward to drift out of sync.
/// </para>
/// </remarks>
[LayerCategory(LayerCategory.Residual)]
[LayerCategory(LayerCategory.Dense)]
[LayerTask(LayerTask.FeatureExtraction)]
[LayerTask(LayerTask.Projection)]
[LayerProperty(IsTrainable = true, ChangesShape = true, ExpectedInputRank = 2, Cost = ComputeCost.Medium, TestInputShape = "1, 3", TestConstructorArgs = "3, 16, 8, 2")]
internal class OccupancyNetworkDecoder<T> : LayerBase<T>
{
    private readonly int _pointDim;
    private readonly int _hidden;
    private readonly int _latentDim;
    private readonly int _numBlocks;

    // Constant "1" input that drives the learnable latent-code generator. Cached
    // so each forward reuses the same buffer instead of allocating.
    private readonly Tensor<T> _one;

    // Learnable latent code c (auto-decoder): ones[1,1] → DenseLayer → c[1, latentDim].
    private readonly DenseLayer<T> _latentGen;

    // Point embedding: Linear(pointDim → hidden).
    private readonly DenseLayer<T> _fcP;

    // Per-block linear maps (Linear(hidden → hidden)) and the conditional affine
    // predictors γ/β (Linear(latentDim → hidden)) for the two conditional-norm sites.
    private readonly DenseLayer<T>[] _fc0;
    private readonly DenseLayer<T>[] _fc1;
    private readonly DenseLayer<T>[] _gamma0;
    private readonly DenseLayer<T>[] _beta0;
    private readonly DenseLayer<T>[] _gamma1;
    private readonly DenseLayer<T>[] _beta1;
    private readonly LayerNormalizationLayer<T>[] _norm0;
    private readonly LayerNormalizationLayer<T>[] _norm1;

    // Final conditional-norm site + output projection.
    private readonly LayerNormalizationLayer<T> _normF;
    private readonly DenseLayer<T> _gammaF;
    private readonly DenseLayer<T> _betaF;
    private readonly DenseLayer<T> _fcOut;

    /// <summary>Hidden width of the decoder's fully-connected ResNet trunk.</summary>
    public int Hidden => _hidden;

    /// <summary>Dimensionality of the (learnable) latent conditioning code.</summary>
    public int LatentDim => _latentDim;

    /// <summary>Number of Conditional-ResNet blocks in the trunk.</summary>
    public int NumBlocks => _numBlocks;

    /// <inheritdoc />
    public override bool SupportsTraining => true;

    /// <summary>
    /// Builds an Occupancy Network decoder.
    /// </summary>
    /// <param name="pointDim">Dimensionality of the queried point (3 for ℝ³). Must be positive.</param>
    /// <param name="hidden">Hidden width of the ResNet trunk (paper default 256). Must be positive.</param>
    /// <param name="latentDim">Dimensionality of the learnable latent code (paper c_dim default 128/256). Must be positive.</param>
    /// <param name="numBlocks">Number of Conditional-ResNet blocks (paper default 5). Must be positive.</param>
    public OccupancyNetworkDecoder(int pointDim = 3, int hidden = 256, int latentDim = 128, int numBlocks = 5)
        : base([pointDim], [1])
    {
        if (pointDim <= 0) throw new ArgumentOutOfRangeException(nameof(pointDim), pointDim, "pointDim must be positive.");
        if (hidden <= 0) throw new ArgumentOutOfRangeException(nameof(hidden), hidden, "hidden must be positive.");
        if (latentDim <= 0) throw new ArgumentOutOfRangeException(nameof(latentDim), latentDim, "latentDim must be positive.");
        if (numBlocks <= 0) throw new ArgumentOutOfRangeException(nameof(numBlocks), numBlocks, "numBlocks must be positive.");

        _pointDim = pointDim;
        _hidden = hidden;
        _latentDim = latentDim;
        _numBlocks = numBlocks;

        _one = new Tensor<T>([1, 1]);
        _one[0, 0] = NumOps.One;

        _latentGen = Linear(latentDim);
        _fcP = Linear(hidden);

        _fc0 = new DenseLayer<T>[numBlocks];
        _fc1 = new DenseLayer<T>[numBlocks];
        _gamma0 = new DenseLayer<T>[numBlocks];
        _beta0 = new DenseLayer<T>[numBlocks];
        _gamma1 = new DenseLayer<T>[numBlocks];
        _beta1 = new DenseLayer<T>[numBlocks];
        _norm0 = new LayerNormalizationLayer<T>[numBlocks];
        _norm1 = new LayerNormalizationLayer<T>[numBlocks];
        for (int i = 0; i < numBlocks; i++)
        {
            _fc0[i] = Linear(hidden);
            _fc1[i] = Linear(hidden);
            _gamma0[i] = Linear(hidden);
            _beta0[i] = Linear(hidden);
            _gamma1[i] = Linear(hidden);
            _beta1[i] = Linear(hidden);
            _norm0[i] = new LayerNormalizationLayer<T>();
            _norm1[i] = new LayerNormalizationLayer<T>();
        }

        _normF = new LayerNormalizationLayer<T>();
        _gammaF = Linear(hidden);
        _betaF = Linear(hidden);
        // Output head emits the occupancy probability o(p) ∈ [0,1] directly via a
        // sigmoid (the paper's occupancy is σ(logit); it folds the sigmoid into
        // BCEWithLogits for training, but AiDotNet's binary-classification loss
        // consumes a probability, so the sigmoid lives in the head here).
        _fcOut = new DenseLayer<T>(1, (IActivationFunction<T>)new SigmoidActivation<T>(), InitializationStrategies<T>.Lazy);

        // Register every sub-layer so the gradient-tape training system
        // discovers their trainable parameters via the recursive
        // GetSubLayers() walk (the nn.Module-attribute equivalent). Without
        // this the composite reports no trainable parameters and training is a
        // silent no-op.
        foreach (var layer in SubLayers())
            RegisterSubLayer(layer);

        // Materialize all lazy DenseLayer weights so they participate in
        // GetParameters / SetParameters immediately (otherwise a freshly cloned
        // decoder would lazy-init independent random weights at first forward,
        // breaking the SetParameters(GetParameters()) determinism contract that
        // clone / checkpoint reload depend on). One probe forward through the
        // whole module under NoGradScope does this, then ResetState clears the
        // per-step caches the probe populated.
        using (var _ = new AiDotNet.Tensors.Engines.Autodiff.NoGradScope<T>())
        {
            Forward(new Tensor<T>([1, pointDim]));
            ResetState();
        }
    }

    private static DenseLayer<T> Linear(int outputSize)
        => new(outputSize, (IActivationFunction<T>)new IdentityActivation<T>(), InitializationStrategies<T>.Lazy);

    /// <summary>
    /// Forward pass: queried point(s) <c>[B, pointDim]</c> (or <c>[pointDim]</c>)
    /// → occupancy logit(s) <c>[B, 1]</c>.
    /// </summary>
    public override Tensor<T> Forward(Tensor<T> input)
    {
        // Work in a consistent rank-2 [B, features] layout. A rank-1 [pointDim]
        // query is promoted to [1, pointDim] so the conditional affine (whose
        // γ(c)/β(c) are rank-2 [1, hidden]) and the residual add never mix ranks.
        var x = input.Rank == 1 ? Engine.Reshape(input, [1, input.Length]) : input;

        // Latent code c = generator(1). Shape [1, latentDim]; broadcasts across
        // the point batch in the conditional affine.
        var c = _latentGen.Forward(_one);

        var net = _fcP.Forward(x);

        for (int i = 0; i < _numBlocks; i++)
        {
            var h = _fc0[i].Forward(Engine.ReLU(ConditionalNorm(net, c, _norm0[i], _gamma0[i], _beta0[i])));
            var dx = _fc1[i].Forward(Engine.ReLU(ConditionalNorm(h, c, _norm1[i], _gamma1[i], _beta1[i])));
            // Residual skip — identity since the block preserves the hidden width.
            net = Engine.TensorAdd(net, dx);
        }

        var outPre = Engine.ReLU(ConditionalNorm(net, c, _normF, _gammaF, _betaF));
        var output = _fcOut.Forward(outPre);

        // Preserve the caller's rank: a rank-1 [pointDim] query returns a rank-1
        // [1] occupancy (matching DenseLayer's 1D-in/1D-out contract and the
        // [1] target the trainer compares against), not [1, 1].
        if (input.Rank == 1 && output.Rank == 2 && output.Shape[0] == 1)
            output = Engine.Reshape(output, [output.Shape[1]]);
        return output;
    }

    /// <summary>
    /// Conditional Normalization: <c>γ(c) ⊙ Norm(x) + β(c)</c>, where the affine
    /// parameters are linear projections of the latent code. The paper uses Batch
    /// Normalization; this uses Layer Normalization so the statistics are
    /// well-defined at batch = 1 (see the type remarks). γ(c) and β(c) are
    /// <c>[1, hidden]</c> and broadcast across the point batch.
    /// </summary>
    private Tensor<T> ConditionalNorm(
        Tensor<T> x, Tensor<T> c,
        LayerNormalizationLayer<T> norm, DenseLayer<T> gammaGen, DenseLayer<T> betaGen)
    {
        var normalized = norm.Forward(x);
        var gamma = gammaGen.Forward(c);
        var beta = betaGen.Forward(c);
        return Engine.TensorBroadcastAdd(Engine.TensorBroadcastMultiply(normalized, gamma), beta);
    }

    private IEnumerable<ILayer<T>> SubLayers()
    {
        yield return _latentGen;
        yield return _fcP;
        for (int i = 0; i < _numBlocks; i++)
        {
            yield return _fc0[i];
            yield return _fc1[i];
            yield return _gamma0[i];
            yield return _beta0[i];
            yield return _gamma1[i];
            yield return _beta1[i];
            yield return _norm0[i];
            yield return _norm1[i];
        }
        yield return _normF;
        yield return _gammaF;
        yield return _betaF;
        yield return _fcOut;
    }

    /// <inheritdoc />
    public override long ParameterCount
    {
        get
        {
            long total = 0;
            foreach (var layer in SubLayers())
                total += layer.ParameterCount;
            return total;
        }
    }

    /// <inheritdoc />
    public override Vector<T> GetParameters()
    {
        var parameters = new List<T>();
        foreach (var layer in SubLayers())
        {
            var p = layer.GetParameters();
            for (int i = 0; i < p.Length; i++)
                parameters.Add(p[i]);
        }
        return new Vector<T>(parameters.ToArray());
    }

    /// <inheritdoc />
    public override void SetParameters(Vector<T> parameters)
    {
        if (parameters.Length != ParameterCount)
            throw new ArgumentException(
                $"Expected {ParameterCount} parameters, got {parameters.Length}.",
                nameof(parameters));

        int idx = 0;
        foreach (var layer in SubLayers())
        {
            int count = checked((int)layer.ParameterCount);
            if (count == 0) continue;
            var sub = new Vector<T>(count);
            for (int i = 0; i < count; i++)
                sub[i] = parameters[idx++];
            layer.SetParameters(sub);
        }
    }

    /// <inheritdoc />
    public override void UpdateParameters(T learningRate)
    {
        foreach (var layer in SubLayers())
            layer.UpdateParameters(learningRate);
    }

    /// <inheritdoc />
    public override void SetTrainingMode(bool isTraining)
    {
        base.SetTrainingMode(isTraining);
        // Propagate to sub-layers so the BatchNorm sites switch between batch
        // statistics (training) and running statistics (inference) — without
        // this, Predict on a single point would use batch stats and collapse.
        foreach (var layer in SubLayers())
            layer.SetTrainingMode(isTraining);
    }

    /// <inheritdoc />
    public override void ResetState()
    {
        foreach (var layer in SubLayers())
            layer.ResetState();
    }

    internal override Dictionary<string, string> GetMetadata()
    {
        var inv = System.Globalization.CultureInfo.InvariantCulture;
        var meta = base.GetMetadata();
        meta["PointDim"] = _pointDim.ToString(inv);
        meta["Hidden"] = _hidden.ToString(inv);
        meta["LatentDim"] = _latentDim.ToString(inv);
        meta["NumBlocks"] = _numBlocks.ToString(inv);
        return meta;
    }
}
