using System.Diagnostics.CodeAnalysis;
using AiDotNet.ActivationFunctions;
using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks.Layers;

namespace AiDotNet.Diffusion.NoisePredictors;

/// <summary>
/// Scalable Interpolant Transformer (SiT) noise predictor for flow-based diffusion.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// SiT generalizes DiT by supporting arbitrary interpolants between data and noise,
/// unifying DDPM, flow matching, and stochastic interpolants within a single framework.
/// This enables flexible choice of forward process and sampling strategy.
/// </para>
/// <para>
/// <b>For Beginners:</b> SiT is a flexible version of DiT (Diffusion Transformer) that
/// can work with different types of noise processes. Instead of being locked to one way
/// of adding noise, SiT can smoothly interpolate between different strategies, choosing
/// the best approach for quality and speed.
///
/// Key advantage: You can pick different sampling methods at inference time without
/// retraining the model.
///
/// Used in: SiT research models, SANA (partially)
/// </para>
/// <para>
/// Reference: Ma et al., "SiT: Exploring Flow and Diffusion-based Generative Models with Scalable Interpolant Transformers", ECCV 2024
/// </para>
/// </remarks>
/// <example>
/// <code>
/// var predictor = new SiTPredictor&lt;float&gt;(inputChannels: 4, hiddenSize: 1152, numLayers: 28, numHeads: 16);
/// var noisyLatent = Tensor&lt;float&gt;.Random(new[] { 1, 4, 32, 32 });
/// var predicted = predictor.PredictNoise(noisyLatent, timestep: 500);
/// </code>
/// </example>
[ModelDomain(ModelDomain.Generative)]
[ModelCategory(ModelCategory.Diffusion)]
[ModelCategory(ModelCategory.Transformer)]
[ModelTask(ModelTask.Denoising)]
[ModelTask(ModelTask.Generation)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
    [ResearchPaper("SiT: Exploring Flow and Diffusion-based Generative Models with Scalable Interpolant Transformers", "https://arxiv.org/abs/2401.08740")]
public class SiTPredictor<T> : NoisePredictorBase<T>
{
    private int _inputChannels;
    private readonly int _hiddenSize;
    private readonly int _numLayers;
    private readonly int _numHeads;

    private DenseLayer<T>? _patchEmbed;
    private DenseLayer<T>[]? _blocks;
    private DenseLayer<T>? _finalLayer;
    private readonly int? _seed;
    private bool _initialized;

    /// <inheritdoc />
    public override int InputChannels => _inputChannels;
    /// <inheritdoc />
    public override int OutputChannels => _inputChannels;
    /// <inheritdoc />
    public override int BaseChannels => _hiddenSize;
    /// <inheritdoc />
    public override int TimeEmbeddingDim => _hiddenSize;
    /// <inheritdoc />
    public override bool SupportsCFG => true;
    /// <inheritdoc />
    public override bool SupportsCrossAttention => true;
    /// <inheritdoc />
    public override int ContextDimension => _hiddenSize;
    /// <inheritdoc />
    public override int ParameterCount => EnsureInitialized().paramCount;

    /// <summary>
    /// Initializes a new SiT predictor. Layers are lazily allocated on first use
    /// to avoid 700M+ parameter allocation at construction time.
    /// </summary>
    public SiTPredictor(
        int inputChannels = 4,
        int hiddenSize = 1152,
        int numLayers = 28,
        int numHeads = 16,
        int? seed = null)
        : base(seed: seed)
    {
        _inputChannels = inputChannels;
        _hiddenSize = hiddenSize;
        _numLayers = numLayers;
        _numHeads = numHeads;
        _seed = seed;
    }

    private (int paramCount, DenseLayer<T> embed, DenseLayer<T>[] blocks, DenseLayer<T> final_) EnsureInitialized()
    {
        if (!_initialized)
        {
            int patchDim = _inputChannels * 4;
            // LazyDense defers weight allocation to first Forward() call.
            _patchEmbed = LazyDense(patchDim, _hiddenSize, new GELUActivation<T>());

            _blocks = new DenseLayer<T>[_numLayers];
            for (int i = 0; i < _numLayers; i++)
                _blocks[i] = LazyDense(_hiddenSize, _hiddenSize, new GELUActivation<T>());

            _finalLayer = LazyDense(_hiddenSize, patchDim);
            _initialized = true;
        }

        int count = _patchEmbed?.ParameterCount ?? 0;
        if (_blocks is not null)
            foreach (var block in _blocks) count += block.ParameterCount;
        count += _finalLayer?.ParameterCount ?? 0;

        return (count, _patchEmbed ?? throw new InvalidOperationException(), _blocks ?? throw new InvalidOperationException(), _finalLayer ?? throw new InvalidOperationException());
    }

    /// <inheritdoc />
    public override Tensor<T> PredictNoise(Tensor<T> noisySample, int timestep, Tensor<T>? conditioning = null)
    {
        // SiT (per Ma et al. 2024) is a transformer over 2x2 patches: patchify
        // [B, C, H, W] → [B, (H/2)*(W/2), C*4], run the dense+block stack on the
        // token dim, then unpatchify back to [B, C, H, W] so the predicted noise
        // matches the input shape. Without patchify/unpatchify the dense layers
        // collapse on the wrong axis and PredictNoise returns a tensor whose
        // element count is patchDim/C × the input — failing the latent-length
        // check in DiffusionModelBase.Generate.
        // Resolve _inputChannels from the actual input on first forward — SiT
        // models constructed with the wrong default (latent_channels=4 vs the
        // diffusion test harness which passes 16-channel pixel input) would
        // otherwise build _finalLayer with the wrong patchDim, producing a
        // tensor that cannot be reshaped back to the input's channel count.
        // Lazy resolution keeps the model adaptive without breaking callers
        // that pass the documented default.
        if (!_initialized && noisySample.Rank >= 2 && noisySample.Shape[1] > 0)
        {
            _inputChannels = noisySample.Shape[1];
        }
        else if (_initialized && noisySample.Rank >= 2 && noisySample.Shape[1] != _inputChannels)
        {
            // Channel-consistency guard: once initialized, the patchify /
            // unpatchify dimensions are baked in. A subsequent call with a
            // different channel count would silently produce wrong-shape
            // output later in the layer stack. Fail fast with a clear
            // exception instead of letting the misconfiguration propagate.
            throw new ArgumentException(
                $"SiTPredictor was initialized for {_inputChannels} input channels, " +
                $"but received {noisySample.Shape[1]}. Construct a new SiTPredictor for " +
                "the new channel count or pass an input matching the original.",
                nameof(noisySample));
        }

        var (_, embed, blocks, final_) = EnsureInitialized();

        int b = noisySample.Shape[0];
        int c = noisySample.Shape[1];
        int h = noisySample.Shape[2];
        int w = noisySample.Shape[3];
        const int patchSize = 2;
        if (h % patchSize != 0 || w % patchSize != 0)
        {
            throw new ArgumentException(
                $"SiT requires spatial dims divisible by {patchSize}; got [{h},{w}].",
                nameof(noisySample));
        }
        int hp = h / patchSize;
        int wp = w / patchSize;

        // Patchify: [B, C, H, W] → [B, hp, patchSize, wp, patchSize, C] via
        // permute, then flatten to [B, hp*wp, C*patchSize*patchSize].
        var permuted = Engine.TensorPermute(noisySample, new[] { 0, 2, 3, 1 }).Contiguous();
        var reshaped = Engine.Reshape(permuted, new[] { b, hp, patchSize, wp, patchSize, c });
        var patchOrdered = Engine.TensorPermute(reshaped, new[] { 0, 1, 3, 2, 4, 5 }).Contiguous();
        var tokens = Engine.Reshape(patchOrdered, new[] { b, hp * wp, c * patchSize * patchSize });

        var x = embed.Forward(tokens);
        foreach (var block in blocks) x = block.Forward(x);
        var outTokens = final_.Forward(x); // [B, hp*wp, C*patchSize*patchSize]

        // Unpatchify: reverse of the above.
        var outPatched = Engine.Reshape(outTokens, new[] { b, hp, wp, patchSize, patchSize, c });
        var outOrdered = Engine.TensorPermute(outPatched, new[] { 0, 1, 3, 2, 4, 5 }).Contiguous();
        var outBhwc = Engine.Reshape(outOrdered, new[] { b, h, w, c });
        return Engine.TensorPermute(outBhwc, new[] { 0, 3, 1, 2 }).Contiguous();
    }

    /// <inheritdoc />
    public override Vector<T> GetParameters()
    {
        var (_, embed, blocks, final_) = EnsureInitialized();
        var allParams = new List<T>();
        AddParams(allParams, embed);
        foreach (var b in blocks) AddParams(allParams, b);
        AddParams(allParams, final_);
        return new Vector<T>(allParams.ToArray());
    }

    /// <inheritdoc />
    public override void SetParameters(Vector<T> parameters)
    {
        var (_, embed, blocks, final_) = EnsureInitialized();
        int offset = 0;
        offset = SetParams(embed, parameters, offset);
        foreach (var b in blocks) offset = SetParams(b, parameters, offset);
        SetParams(final_, parameters, offset);
    }

    /// <inheritdoc />
    public override IFullModel<T, Tensor<T>, Tensor<T>> DeepCopy() => Clone();

    /// <inheritdoc />
    public override INoisePredictor<T> Clone()
    {
        var clone = new SiTPredictor<T>(_inputChannels, _hiddenSize, _numLayers, _numHeads, _seed);
        if (_initialized)
            clone.SetParameters(GetParameters());
        return clone;
    }

    private static void AddParams(List<T> list, DenseLayer<T> layer)
    {
        var p = layer.GetParameters();
        for (int i = 0; i < p.Length; i++) list.Add(p[i]);
    }

    private static int SetParams(DenseLayer<T> layer, Vector<T> parameters, int offset)
    {
        int count = layer.ParameterCount;
        var p = new T[count];
        for (int i = 0; i < count; i++) p[i] = parameters[offset + i];
        layer.SetParameters(new Vector<T>(p));
        return offset + count;
    }

    protected override Vector<T> GetParameterGradients()
    {
        var (_, embed, blocks, final_) = EnsureInitialized();
        var allGrads = new List<T>();
        AddGrads(allGrads, embed);
        foreach (var b in blocks) AddGrads(allGrads, b);
        AddGrads(allGrads, final_);
        return new Vector<T>(allGrads.ToArray());
    }

    private static void AddGrads(List<T> list, DenseLayer<T> layer)
    {
        var g = layer.GetParameterGradients();
        for (int i = 0; i < g.Length; i++) list.Add(g[i]);
    }
}
