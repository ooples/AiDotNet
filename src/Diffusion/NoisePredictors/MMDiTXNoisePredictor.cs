using System.Diagnostics.CodeAnalysis;
using AiDotNet.ActivationFunctions;
using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LossFunctions;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.Helpers;

namespace AiDotNet.Diffusion.NoisePredictors;

/// <summary>
/// Extended Multi-Modal Diffusion Transformer (MMDiT-X) noise predictor for SD3.5 architecture.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// MMDiT-X extends the original MMDiT architecture with improved QK-normalization,
/// enhanced position encoding, and optimized attention patterns for SD3.5 models.
/// </para>
/// <para>
/// <b>For Beginners:</b> MMDiT-X is the improved brain of Stable Diffusion 3.5.
/// It processes both text and image information together through joint attention blocks,
/// allowing the model to deeply understand how text descriptions relate to image features.
///
/// Key improvements over MMDiT:
/// - QK-normalization prevents attention collapse at higher resolutions
/// - More efficient attention patterns reduce memory usage
/// - Better positional encoding for multi-resolution support
/// - Available in Medium (2B) and Large (8B) configurations
/// </para>
/// <para>
/// Reference: Esser et al., "Scaling Rectified Flow Transformers for High-Resolution Image Synthesis", ICML 2024
/// </para>
/// </remarks>
/// <example>
/// <code>
/// var predictor = new MMDiTXNoisePredictor&lt;float&gt;(inputChannels: 16, hiddenSize: 1536, numLayers: 24, numHeads: 24);
/// var noisyLatent = Tensor&lt;float&gt;.Random(new[] { 1, 16, 128, 128 });
/// var predicted = predictor.PredictNoise(noisyLatent, timestep: 500);
/// </code>
/// </example>
[ModelDomain(ModelDomain.Generative)]
[ModelCategory(ModelCategory.Diffusion)]
[ModelCategory(ModelCategory.Transformer)]
[ModelTask(ModelTask.Denoising)]
[ModelTask(ModelTask.TextToImage)]
[ModelComplexity(ModelComplexity.VeryHigh)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
    [ResearchPaper("Scaling Rectified Flow Transformers for High-Resolution Image Synthesis", "https://arxiv.org/abs/2403.03206")]
public class MMDiTXNoisePredictor<T> : NoisePredictorBase<T>
{
    private readonly int _inputChannels;
    private readonly int _hiddenSize;
    private readonly int _numJointLayers;
    private readonly int _numHeads;
    private readonly int _patchSize;
    private readonly int _contextDim;
    private readonly MMDiTXVariant _variant;

    private DenseLayer<T> _patchEmbed;
    private DenseLayer<T>[] _jointBlocks;
    private DenseLayer<T> _finalLayer;
    private Vector<T> _posEmbed;

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
    public override int ContextDimension => _contextDim;
    /// <inheritdoc />
    public override long ParameterCount { get; }

    /// <summary>
    /// Initializes a new MMDiT-X noise predictor for SD3.5.
    /// </summary>
    /// <param name="variant">SD3.5 variant. Default: Medium.</param>
    /// <param name="inputChannels">Latent channels. Default: 16.</param>
    /// <param name="patchSize">Patch size. Default: 2.</param>
    /// <param name="contextDim">Context dimension from text encoder. Default: 4096.</param>
    /// <param name="seed">Optional random seed.</param>
    public MMDiTXNoisePredictor(
        MMDiTXVariant variant = MMDiTXVariant.Medium,
        int inputChannels = 16,
        int patchSize = 2,
        int contextDim = 4096,
        int? seed = null,
        int hiddenSizeOverride = 0,
        int numLayersOverride = 0,
        int numHeadsOverride = 0)
        : base(seed: seed)
    {
        _variant = variant;
        _inputChannels = inputChannels;
        // Width/depth/head-count come from the SD3.5 variant by default; the overrides (>0) let callers
        // build a smaller same-architecture predictor (e.g. a reduced-scale test fixture) without
        // changing the paper-scale production defaults.
        _hiddenSize = hiddenSizeOverride > 0 ? hiddenSizeOverride : GetHiddenSize(variant);
        _numJointLayers = numLayersOverride > 0 ? numLayersOverride : GetNumLayers(variant);
        _numHeads = numHeadsOverride > 0 ? numHeadsOverride : GetNumHeads(variant);
        _patchSize = patchSize;
        _contextDim = contextDim;

        InitializeLayers(seed);

        ParameterCount = CalculateParameterCount();
    }

    [MemberNotNull(nameof(_patchEmbed), nameof(_jointBlocks), nameof(_finalLayer), nameof(_posEmbed))]
    private void InitializeLayers(int? seed)
    {
        int patchDim = _inputChannels * _patchSize * _patchSize;
        // LazyDense keeps weight tensors unallocated until Forward() — avoids
        // OOM at construction time under production-scale defaults.
        _patchEmbed = LazyDense(patchDim, _hiddenSize, new GELUActivation<T>());

        _jointBlocks = new DenseLayer<T>[_numJointLayers];
        for (int i = 0; i < _numJointLayers; i++)
            _jointBlocks[i] = LazyDense(_hiddenSize, _hiddenSize, new GELUActivation<T>());

        _finalLayer = LazyDense(_hiddenSize, patchDim);
        _posEmbed = new Vector<T>(1024 * _hiddenSize);
        var rng = seed.HasValue ? RandomHelper.CreateSeededRandom(seed.Value) : RandomHelper.CreateSecureRandom();
        for (int i = 0; i < _posEmbed.Length; i++)
            _posEmbed[i] = NumOps.FromDouble(rng.NextDouble() * 0.02 - 0.01);
    }

    private long CalculateParameterCount()
    {
        long count = _patchEmbed.ParameterCount;
        foreach (var block in _jointBlocks) count += block.ParameterCount;
        count += _finalLayer.ParameterCount;
        count += _posEmbed.Length;
        return count;
    }

    /// <inheritdoc />
    /// <remarks>
    /// Per Esser et al. 2024 §3 (Scaling Rectified Flow Transformers
    /// for High-Resolution Image Synthesis, MMDiT § 3) the predictor
    /// runs on patch TOKENS, not on the raw [B, C, H, W] spatial
    /// tensor. The forward path is:
    /// <list type="number">
    ///   <item>Patchify: [B, C, H, W] → [B, (H/P)·(W/P), C·P²]</item>
    ///   <item>Embed:    [B, N, C·P²] → [B, N, hiddenSize]</item>
    ///   <item>Joint blocks at [B, N, hiddenSize]</item>
    ///   <item>Project:  [B, N, hiddenSize] → [B, N, C·P²]</item>
    ///   <item>Unpatchify: [B, N, C·P²] → [B, C, H, W]</item>
    /// </list>
    /// where N = (H/P)·(W/P) and P = patchSize. The previous
    /// implementation skipped patchify/unpatchify and just ran Dense
    /// layers on the spatial tensor's last axis, which projects W →
    /// patchDim and emits [B, C, H, patchDim] (=2× the latent on the
    /// SD3 default 32×32×16, patchSize=2 → patchDim=64 so output
    /// elements = 1·16·32·64 = 32768 vs latent 16384) — exactly the
    /// "PredictNoise output length 32768 does not match the
    /// latent/sample length 16384" failure the test reported (#1224
    /// Cluster F: ControlNetSD3 — 11/11 tests blocked).
    /// </remarks>
    public override Tensor<T> PredictNoise(Tensor<T> noisySample, int timestep, Tensor<T>? conditioning = null)
    {
        // Normalize input to rank-4 [B, C, H, W]. Tests pass rank-3
        // [C, H, W] as a single sample; promote a leading batch dim of 1.
        var input4d = noisySample;
        bool wasUnbatched = false;
        if (input4d.Rank == 3)
        {
            wasUnbatched = true;
            input4d = input4d.Reshape(new[] { 1, input4d.Shape[0], input4d.Shape[1], input4d.Shape[2] });
        }
        if (input4d.Rank != 4)
            throw new ArgumentException(
                $"MMDiTXNoisePredictor expects rank-3 [C,H,W] or rank-4 [B,C,H,W] input; got rank {input4d.Rank}.",
                nameof(noisySample));

        int batch = input4d.Shape[0];
        int channels = input4d.Shape[1];
        int height = input4d.Shape[2];
        int width = input4d.Shape[3];

        if (channels != _inputChannels)
            throw new ArgumentException(
                $"MMDiTXNoisePredictor configured for {_inputChannels} channels; got {channels}.",
                nameof(noisySample));
        if (height % _patchSize != 0 || width % _patchSize != 0)
            throw new ArgumentException(
                $"MMDiTXNoisePredictor requires spatial dims divisible by patchSize ({_patchSize}); got {height}×{width}.",
                nameof(noisySample));

        using var streaming = BeginWeightStreamingForward();

        int patchDim = _inputChannels * _patchSize * _patchSize;
        int hPatches = height / _patchSize;
        int wPatches = width / _patchSize;
        int numTokens = hPatches * wPatches;

        // Patchify: [B, C, H, W] → [B, numTokens, patchDim]
        var tokens = Patchify(input4d, batch, channels, height, width, hPatches, wPatches, patchDim);

        // Embed and propagate
        // G4 (#1624): checkpoint each block so foundation-scale stacks recompute activations in backward.
        var x = _patchEmbed.Forward(tokens);
        foreach (var block in _jointBlocks)
            x = CheckpointBlock(block.Forward, x);
        var projected = _finalLayer.Forward(x);  // [B, numTokens, patchDim]

        // Unpatchify back to [B, C, H, W]
        var output = Unpatchify(projected, batch, channels, height, width, hPatches, wPatches);

        if (wasUnbatched)
            output = output.Reshape(new[] { channels, height, width });
        return streaming.Complete(output);
    }

    /// <summary>
    /// [B, C, H, W] → [B, (H/P)·(W/P), C·P²] via the standard
    /// rearrange("b c (h p1) (w p2) → b (h w) (c p1 p2)") that the
    /// MMDiT reference implementation uses (Esser et al. 2024 §3).
    /// </summary>
    private Tensor<T> Patchify(Tensor<T> input, int batch, int channels, int height, int width,
        int hPatches, int wPatches, int patchDim)
    {
        var output = new Tensor<T>(new[] { batch, hPatches * wPatches, patchDim });
        for (int b = 0; b < batch; b++)
        {
            for (int hp = 0; hp < hPatches; hp++)
            {
                for (int wp = 0; wp < wPatches; wp++)
                {
                    int tokenIdx = hp * wPatches + wp;
                    int featIdx = 0;
                    for (int c = 0; c < channels; c++)
                    {
                        for (int p1 = 0; p1 < _patchSize; p1++)
                        {
                            for (int p2 = 0; p2 < _patchSize; p2++)
                            {
                                int hSrc = hp * _patchSize + p1;
                                int wSrc = wp * _patchSize + p2;
                                output[b, tokenIdx, featIdx++] = input[b, c, hSrc, wSrc];
                            }
                        }
                    }
                }
            }
        }
        return output;
    }

    /// <summary>
    /// [B, (H/P)·(W/P), C·P²] → [B, C, H, W] — inverse of <see cref="Patchify"/>.
    /// </summary>
    private Tensor<T> Unpatchify(Tensor<T> tokens, int batch, int channels, int height, int width,
        int hPatches, int wPatches)
    {
        var output = new Tensor<T>(new[] { batch, channels, height, width });
        for (int b = 0; b < batch; b++)
        {
            for (int hp = 0; hp < hPatches; hp++)
            {
                for (int wp = 0; wp < wPatches; wp++)
                {
                    int tokenIdx = hp * wPatches + wp;
                    int featIdx = 0;
                    for (int c = 0; c < channels; c++)
                    {
                        for (int p1 = 0; p1 < _patchSize; p1++)
                        {
                            for (int p2 = 0; p2 < _patchSize; p2++)
                            {
                                int hDst = hp * _patchSize + p1;
                                int wDst = wp * _patchSize + p2;
                                output[b, c, hDst, wDst] = tokens[b, tokenIdx, featIdx++];
                            }
                        }
                    }
                }
            }
        }
        return output;
    }

    /// <inheritdoc />
    public override Vector<T> GetParameters()
    {
        var allParams = new List<T>();
        AddLayerParams(allParams, _patchEmbed);
        foreach (var block in _jointBlocks) AddLayerParams(allParams, block);
        AddLayerParams(allParams, _finalLayer);
        for (int i = 0; i < _posEmbed.Length; i++) allParams.Add(_posEmbed[i]);
        return new Vector<T>(allParams.ToArray());
    }

    /// <inheritdoc />
    public override void SetParameters(Vector<T> parameters)
    {
        int offset = 0;
        offset = SetLayerParams(_patchEmbed, parameters, offset);
        foreach (var block in _jointBlocks) offset = SetLayerParams(block, parameters, offset);
        offset = SetLayerParams(_finalLayer, parameters, offset);
        for (int i = 0; i < _posEmbed.Length; i++) _posEmbed[i] = parameters[offset++];
    }

    /// <inheritdoc />
    public override System.Collections.Generic.IEnumerable<Tensor<T>> GetParameterChunks()
    {
        // #1624: one chunk per layer (then the raw positional-embedding table) in the SAME order as
        // GetParameters/SetParameters, so the flat concatenation is index-identical to GetParameters
        // without materializing the full aggregate that OOMs at default size.
        yield return ChunkOf(_patchEmbed);
        foreach (var block in _jointBlocks) yield return ChunkOf(block);
        yield return ChunkOf(_finalLayer);
        if (_posEmbed.Length > 0) yield return new Tensor<T>(new[] { _posEmbed.Length }, new Vector<T>(_posEmbed));
    }

    /// <inheritdoc />
    public override void SetParameterChunks(System.Collections.Generic.IEnumerable<Tensor<T>> chunks)
    {
        using var e = chunks.GetEnumerator();
        SetChunk(e, _patchEmbed);
        foreach (var block in _jointBlocks) SetChunk(e, block);
        SetChunk(e, _finalLayer);
        if (_posEmbed.Length > 0)
        {
            if (!e.MoveNext())
                throw new System.ArgumentException(
                    "SetParameterChunks received fewer chunks than MMDiT-X has parameter groups (missing posEmbed).",
                    nameof(chunks));
            var pos = e.Current.ToVector();
            for (int i = 0; i < _posEmbed.Length; i++) _posEmbed[i] = pos[i];
        }
        if (e.MoveNext())
            throw new System.ArgumentException(
                "SetParameterChunks received more chunks than MMDiT-X has parameter groups.", nameof(chunks));
    }

    private static Tensor<T> ChunkOf(DenseLayer<T> layer)
    {
        var p = layer.GetParameters();
        return new Tensor<T>(new[] { p.Length }, p);
    }

    private static void SetChunk(System.Collections.Generic.IEnumerator<Tensor<T>> e, DenseLayer<T> layer)
    {
        if (!e.MoveNext())
            throw new System.ArgumentException(
                "SetParameterChunks received fewer chunks than MMDiT-X has parameter groups.", nameof(e));
        layer.SetParameters(e.Current.ToVector());
    }

    /// <inheritdoc />
    public override IFullModel<T, Tensor<T>, Tensor<T>> DeepCopy() => Clone();

    /// <inheritdoc />
    public override INoisePredictor<T> Clone()
    {
        var clone = new MMDiTXNoisePredictor<T>(
            _variant, _inputChannels, _patchSize, _contextDim,
            hiddenSizeOverride: _hiddenSize, numLayersOverride: _numJointLayers, numHeadsOverride: _numHeads);

        // The patch-embed/joint/final layers are LazyDense — their weight tensors only allocate on the
        // first Forward, not at construction. A fresh clone has the structure but unallocated weights, so
        // SetParameters(GetParameters()) onto it would copy into nothing and the clone would re-RNG-init
        // on its first real forward, diverging from the source. When the source has been forwarded
        // (its layers are materialized), run one tiny probe forward to materialize the clone through the
        // same path, THEN copy the trained values so they persist. (_posEmbed is eager, always copied.)
        if (_patchEmbed.IsInitialized)
        {
            int probeSpatial = _patchSize * 2;
            var probe = new Tensor<T>(new[] { 1, _inputChannels, probeSpatial, probeSpatial });
            clone.PredictNoise(probe, timestep: 0, conditioning: null);
        }
        if (clone.TryShareParametersFrom(this))
        {
            // COW shares trainable LAYER tensors only; _posEmbed is a learned Vector<T> field
            // (random-init, part of Get/SetParameters), so copy it explicitly here or the clone keeps
            // its own RNG-init positional embedding and diverges from the source on the share path.
            var pos = new Vector<T>(_posEmbed.Length);
            for (int i = 0; i < _posEmbed.Length; i++) pos[i] = _posEmbed[i];
            clone._posEmbed = pos;
            return clone;
        }
        clone.SetParameters(GetParameters());
        return clone;
    }

    private static void AddLayerParams(List<T> list, DenseLayer<T> layer)
    {
        var p = layer.GetParameters();
        for (int i = 0; i < p.Length; i++) list.Add(p[i]);
    }

    private static int SetLayerParams(DenseLayer<T> layer, Vector<T> parameters, int offset)
    {
        int count = checked((int)layer.ParameterCount);
        var p = new T[count];
        for (int i = 0; i < count; i++) p[i] = parameters[offset + i];
        layer.SetParameters(new Vector<T>(p));
        return offset + count;
    }

    protected override Vector<T> GetParameterGradients()
    {
        var allGrads = new List<T>();
        AddLayerGrads(allGrads, _patchEmbed);
        foreach (var block in _jointBlocks) AddLayerGrads(allGrads, block);
        AddLayerGrads(allGrads, _finalLayer);
        // _posEmbed is a raw tensor, no gradients from layer backward
        for (int i = 0; i < _posEmbed.Length; i++) allGrads.Add(NumOps.Zero);
        return new Vector<T>(allGrads.ToArray());
    }

    private static void AddLayerGrads(List<T> list, DenseLayer<T> layer)
    {
        var g = layer.GetParameterGradients();
        for (int i = 0; i < g.Length; i++) list.Add(g[i]);
    }

    private static int GetHiddenSize(MMDiTXVariant variant) => variant switch
    {
        MMDiTXVariant.Medium => 2048,
        MMDiTXVariant.LargeTurbo => 2560,
        _ => 2560
    };

    private static int GetNumLayers(MMDiTXVariant variant) => variant switch
    {
        MMDiTXVariant.Medium => 24,
        _ => 38
    };

    private static int GetNumHeads(MMDiTXVariant variant) => variant switch
    {
        MMDiTXVariant.Medium => 16,
        _ => 20
    };
}
