using System.Diagnostics.CodeAnalysis;
using AiDotNet.ActivationFunctions;
using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks.Layers;

namespace AiDotNet.Diffusion.NoisePredictors;

/// <summary>
/// Efficient MMDiT (E-MMDiT) noise predictor — lightweight 304M-parameter variant for fast generation.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// E-MMDiT is a compact version of the MMDiT architecture designed for efficient inference
/// with only 304M parameters while maintaining competitive image quality. It achieves this
/// through reduced hidden dimensions and fewer layers while preserving the joint attention
/// mechanism.
/// </para>
/// <para>
/// <b>For Beginners:</b> E-MMDiT is a "mini" version of the brain used in Stable Diffusion 3.
/// While the full SD3 has billions of parameters, E-MMDiT has only 304 million — making it
/// much faster to run on consumer hardware while still producing good images.
///
/// Think of it like a compact car vs a full-size sedan: smaller but still gets you there.
///
/// Key characteristics:
/// - Only 304M parameters (vs 2-8B for full MMDiT)
/// - Runs on consumer GPUs with limited VRAM
/// - Joint text-image attention preserved for quality
/// - Used in: Meissonic, other lightweight diffusion models
/// </para>
/// <para>
/// Reference: Based on MMDiT architecture with parameter-efficient design
/// </para>
/// </remarks>
/// <example>
/// <code>
/// var predictor = new EMMDiTPredictor&lt;float&gt;(inputChannels: 4, contextDim: 768);
/// var noisyLatent = Tensor&lt;float&gt;.Random(new[] { 1, 4, 64, 64 });
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
    [ResearchPaper("Scaling Rectified Flow Transformers for High-Resolution Image Synthesis", "https://arxiv.org/abs/2403.03206")]
public class EMMDiTPredictor<T> : NoisePredictorBase<T>
{
    private const int EMMDIT_HIDDEN_SIZE = 1024;
    private const int EMMDIT_NUM_LAYERS = 12;
    private const int EMMDIT_NUM_HEADS = 16;

    private readonly int _inputChannels;
    private readonly int _contextDim;

    private DenseLayer<T> _patchEmbed;
    private DenseLayer<T>[] _blocks;
    private DenseLayer<T> _finalLayer;

    /// <inheritdoc />
    public override int InputChannels => _inputChannels;
    /// <inheritdoc />
    public override int OutputChannels => _inputChannels;
    /// <inheritdoc />
    public override int BaseChannels => EMMDIT_HIDDEN_SIZE;
    /// <inheritdoc />
    public override int TimeEmbeddingDim => EMMDIT_HIDDEN_SIZE;
    /// <inheritdoc />
    public override bool SupportsCFG => true;
    /// <inheritdoc />
    public override bool SupportsCrossAttention => true;
    /// <inheritdoc />
    public override int ContextDimension => _contextDim;
    /// <inheritdoc />
    public override long ParameterCount { get; }

    /// <summary>
    /// Initializes a new E-MMDiT predictor.
    /// </summary>
    /// <param name="inputChannels">Input latent channels. Default: 4.</param>
    /// <param name="contextDim">Text conditioning dimension. Default: 768.</param>
    /// <param name="seed">Optional random seed.</param>
    public EMMDiTPredictor(
        int inputChannels = 4,
        int contextDim = 768,
        int? seed = null)
        : base(seed: seed)
    {
        _inputChannels = inputChannels;
        _contextDim = contextDim;

        InitializeLayers(seed);
        ParameterCount = CalculateParameterCount();
    }

    [MemberNotNull(nameof(_patchEmbed), nameof(_blocks), nameof(_finalLayer))]
    private void InitializeLayers(int? seed)
    {
        int patchDim = _inputChannels * 4;
        // LazyDense: weight tensors stay unallocated until first Forward() — avoids
        // eagerly materializing ~GB of weights at construction time.
        _patchEmbed = LazyDense(patchDim, EMMDIT_HIDDEN_SIZE, new GELUActivation<T>());

        _blocks = new DenseLayer<T>[EMMDIT_NUM_LAYERS];
        for (int i = 0; i < EMMDIT_NUM_LAYERS; i++)
            _blocks[i] = LazyDense(EMMDIT_HIDDEN_SIZE, EMMDIT_HIDDEN_SIZE, new GELUActivation<T>());

        _finalLayer = LazyDense(EMMDIT_HIDDEN_SIZE, patchDim);
    }

    private int CalculateParameterCount()
    {
        int count = checked((int)_patchEmbed.ParameterCount);
        foreach (var block in _blocks) count += (int)block.ParameterCount;
        count += (int)_finalLayer.ParameterCount;
        return count;
    }

    /// <inheritdoc />
    /// <remarks>
    /// E-MMDiT (Xie et al. 2024 §3.2 "SANA: Efficient High-Resolution Image
    /// Synthesis with Linear Diffusion Transformers", building on Esser et al.
    /// 2024 "Scaling Rectified Flow Transformers") operates on patch TOKENS,
    /// not raw [B, C, H, W] spatial tensors:
    ///   * Patchify:   [B, C, H, W] → [B, (H/P)·(W/P), C·P²]    (P = 2 here)
    ///   * Embed:      [B, N, C·P²] → [B, N, hidden]
    ///   * N joint blocks at [B, N, hidden]
    ///   * Project:    [B, N, hidden] → [B, N, C·P²]
    ///   * Unpatchify: [B, N, C·P²] → [B, C, H, W]
    /// The previous implementation skipped patchify/unpatchify and ran Dense
    /// layers on the spatial tensor's last axis — projecting W → patchDim and
    /// emitting [B, C, H, patchDim] (=4× the latent on the SANA default
    /// 32×32×32, patchSize=2 → patchDim=128 so output elements = 1·32·32·128 =
    /// 131072 vs latent 1·32·32·32 = 32768) — exactly the "PredictNoise output
    /// length does not match the latent/sample length" failure the test
    /// reported (#1224 Cluster: SANA — 10/10 tests blocked).
    ///
    /// Mirrors the MMDiT-X fix in c8483a153.
    /// </remarks>
    public override Tensor<T> PredictNoise(Tensor<T> noisySample, int timestep, Tensor<T>? conditioning = null)
    {
        // Normalize input to rank-4 [B, C, H, W]. Tests pass rank-3 [C, H, W]
        // as a single sample; promote a leading batch dim of 1.
        var input4d = noisySample;
        bool wasUnbatched = false;
        if (input4d.Rank == 3)
        {
            wasUnbatched = true;
            input4d = input4d.Reshape(new[] { 1, input4d.Shape[0], input4d.Shape[1], input4d.Shape[2] });
        }
        if (input4d.Rank != 4)
            throw new ArgumentException(
                $"EMMDiTPredictor expects rank-3 [C,H,W] or rank-4 [B,C,H,W] input; got rank {input4d.Rank}.",
                nameof(noisySample));

        const int patchSize = 2;  // Fixed: patchDim = inputChannels * 4 = inputChannels * patchSize²
        int batch = input4d.Shape[0];
        int channels = input4d.Shape[1];
        int height = input4d.Shape[2];
        int width = input4d.Shape[3];

        if (channels != _inputChannels)
            throw new ArgumentException(
                $"EMMDiTPredictor configured for {_inputChannels} channels; got {channels}.",
                nameof(noisySample));
        if (height % patchSize != 0 || width % patchSize != 0)
            throw new ArgumentException(
                $"EMMDiTPredictor requires spatial dims divisible by patchSize ({patchSize}); got {height}×{width}.",
                nameof(noisySample));

        using var streaming = BeginWeightStreamingForward();

        int patchDim = _inputChannels * patchSize * patchSize;
        int hPatches = height / patchSize;
        int wPatches = width / patchSize;
        int numTokens = hPatches * wPatches;

        // Patchify: [B, C, H, W] → [B, numTokens, patchDim]
        var tokens = new Tensor<T>(new[] { batch, numTokens, patchDim });
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
                        for (int p1 = 0; p1 < patchSize; p1++)
                        {
                            for (int p2 = 0; p2 < patchSize; p2++)
                            {
                                int hSrc = hp * patchSize + p1;
                                int wSrc = wp * patchSize + p2;
                                tokens[b, tokenIdx, featIdx++] = input4d[b, c, hSrc, wSrc];
                            }
                        }
                    }
                }
            }
        }

        // Embed and propagate
        var x = _patchEmbed.Forward(tokens);
        foreach (var block in _blocks) x = block.Forward(x);
        var projected = _finalLayer.Forward(x);  // [B, numTokens, patchDim]

        // Unpatchify: [B, numTokens, patchDim] → [B, C, H, W]
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
                        for (int p1 = 0; p1 < patchSize; p1++)
                        {
                            for (int p2 = 0; p2 < patchSize; p2++)
                            {
                                int hDst = hp * patchSize + p1;
                                int wDst = wp * patchSize + p2;
                                output[b, c, hDst, wDst] = projected[b, tokenIdx, featIdx++];
                            }
                        }
                    }
                }
            }
        }

        if (wasUnbatched)
            output = output.Reshape(new[] { channels, height, width });
        return streaming.Complete(output);
    }

    /// <inheritdoc />
    public override Vector<T> GetParameters()
    {
        var allParams = new List<T>();
        AddParams(allParams, _patchEmbed);
        foreach (var b in _blocks) AddParams(allParams, b);
        AddParams(allParams, _finalLayer);
        return new Vector<T>(allParams.ToArray());
    }

    /// <inheritdoc />
    public override void SetParameters(Vector<T> parameters)
    {
        int offset = 0;
        offset = SetParams(_patchEmbed, parameters, offset);
        foreach (var b in _blocks) offset = SetParams(b, parameters, offset);
        SetParams(_finalLayer, parameters, offset);
    }

    /// <inheritdoc />
    public override System.Collections.Generic.IEnumerable<Tensor<T>> GetParameterChunks()
    {
        // #1624: one chunk per layer in the SAME order as GetParameters/SetParameters, so the flat
        // concatenation is index-identical to GetParameters without materializing the full aggregate.
        yield return ChunkOf(_patchEmbed);
        foreach (var b in _blocks) yield return ChunkOf(b);
        yield return ChunkOf(_finalLayer);
    }

    /// <inheritdoc />
    public override void SetParameterChunks(System.Collections.Generic.IEnumerable<Tensor<T>> chunks)
    {
        using var e = chunks.GetEnumerator();
        SetChunk(e, _patchEmbed);
        foreach (var b in _blocks) SetChunk(e, b);
        SetChunk(e, _finalLayer);
        if (e.MoveNext())
            throw new System.ArgumentException(
                "SetParameterChunks received more chunks than the predictor has layers.", nameof(chunks));
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
                "SetParameterChunks received fewer chunks than the predictor has layers.", nameof(e));
        layer.SetParameters(e.Current.ToVector());
    }

    /// <inheritdoc />
    public override IFullModel<T, Tensor<T>, Tensor<T>> DeepCopy() => Clone();

    /// <inheritdoc />
    public override INoisePredictor<T> Clone()
    {
        var clone = new EMMDiTPredictor<T>(_inputChannels, _contextDim);
        if (!clone.TryShareParametersFrom(this)) clone.SetParameters(GetParameters());
        return clone;
    }

    private static void AddParams(List<T> list, DenseLayer<T> layer)
    {
        var p = layer.GetParameters();
        for (int i = 0; i < p.Length; i++) list.Add(p[i]);
    }

    private static int SetParams(DenseLayer<T> layer, Vector<T> parameters, int offset)
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
        AddGrads(allGrads, _patchEmbed);
        foreach (var b in _blocks) AddGrads(allGrads, b);
        AddGrads(allGrads, _finalLayer);
        return new Vector<T>(allGrads.ToArray());
    }

    private static void AddGrads(List<T> list, DenseLayer<T> layer)
    {
        var g = layer.GetParameterGradients();
        for (int i = 0; i < g.Length; i++) list.Add(g[i]);
    }
}
