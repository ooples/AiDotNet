using System.Diagnostics.CodeAnalysis;
using AiDotNet.ActivationFunctions;
using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks.Layers;

namespace AiDotNet.Diffusion.NoisePredictors;

/// <summary>
/// FLUX double-stream transformer noise predictor with joint and single-stream blocks.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// The FLUX architecture uses a hybrid design: double-stream blocks process text and image
/// tokens with joint attention but separate MLPs, followed by single-stream blocks that
/// process both modalities through a shared path.
/// </para>
/// <para>
/// <b>For Beginners:</b> This is the transformer architecture behind FLUX models.
/// It has two stages:
/// 1. Double-stream blocks: Text and image tokens interact through shared attention
///    but have their own separate processing paths (MLPs)
/// 2. Single-stream blocks: Both text and image tokens are processed together through
///    a shared path, enabling deep fusion
///
/// This hybrid design balances the benefits of modality-specific processing with
/// deep cross-modal interaction.
/// </para>
/// <para>
/// Reference: Black Forest Labs, "FLUX.1 Technical Report", 2024
/// </para>
/// </remarks>
/// <example>
/// <code>
/// var predictor = new FluxDoubleStreamPredictor&lt;float&gt;(inputChannels: 16, hiddenSize: 3072, numLayers: 19, numHeads: 24);
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
    [ResearchPaper("FLUX.1 Technical Report", "https://blackforestlabs.ai/announcing-black-forest-labs/")]
public class FluxDoubleStreamPredictor<T> : NoisePredictorBase<T>
{
    private readonly int _inputChannels;
    private readonly int _hiddenSize;
    private readonly int _numJointLayers;
    private readonly int _numSingleLayers;
    private readonly int _contextDim;
    private readonly FluxPredictorVariant _variant;

    /// <summary>FLUX patch size — every spatial 2×2 block becomes one token.
    /// Class-level constant so InitializeLayers and PredictNoise can't drift
    /// (per CodeRabbit PR #1396 review).</summary>
    private const int PatchSize = 2;

    private DenseLayer<T> _patchEmbed;
    private DenseLayer<T>[] _doubleBlocks;
    private DenseLayer<T>[] _singleBlocks;
    private DenseLayer<T> _finalLayer;

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
    /// Initializes a new FLUX double-stream predictor.
    /// </summary>
    /// <param name="variant">FLUX variant. Default: Dev.</param>
    /// <param name="inputChannels">Latent channels. Default: 16.</param>
    /// <param name="contextDim">Context dimension. Default: 4096.</param>
    /// <param name="seed">Optional random seed.</param>
    public FluxDoubleStreamPredictor(
        FluxPredictorVariant variant = FluxPredictorVariant.Dev,
        int inputChannels = 16,
        int contextDim = 4096,
        int? seed = null)
        : base(seed: seed)
    {
        _variant = variant;
        _inputChannels = inputChannels;
        _hiddenSize = 3072;
        _numJointLayers = 19;
        _numSingleLayers = 38;
        _contextDim = contextDim;

        InitializeLayers(seed);
        ParameterCount = CalculateParameterCount();
    }

    [MemberNotNull(nameof(_patchEmbed), nameof(_doubleBlocks), nameof(_singleBlocks), nameof(_finalLayer))]
    private void InitializeLayers(int? seed)
    {
        int patchDim = _inputChannels * PatchSize * PatchSize;
        // LazyDense defers weight allocation to first Forward() call.
        _patchEmbed = LazyDense(patchDim, _hiddenSize, new GELUActivation<T>());

        _doubleBlocks = new DenseLayer<T>[_numJointLayers];
        for (int i = 0; i < _numJointLayers; i++)
            _doubleBlocks[i] = LazyDense(_hiddenSize, _hiddenSize, new GELUActivation<T>());

        _singleBlocks = new DenseLayer<T>[_numSingleLayers];
        for (int i = 0; i < _numSingleLayers; i++)
            _singleBlocks[i] = LazyDense(_hiddenSize, _hiddenSize, new GELUActivation<T>());

        _finalLayer = LazyDense(_hiddenSize, patchDim);
    }

    private int CalculateParameterCount()
    {
        int count = checked((int)_patchEmbed.ParameterCount);
        foreach (var block in _doubleBlocks) count += (int)block.ParameterCount;
        foreach (var block in _singleBlocks) count += (int)block.ParameterCount;
        count += (int)_finalLayer.ParameterCount;
        return count;
    }

    /// <inheritdoc />
    /// <remarks>
    /// Patchify + DiT-style block stack + unpatchify. The previous
    /// implementation skipped patchify/unpatchify and just ran the
    /// Dense layers on the spatial tensor's last axis directly, which
    /// projects W → patchDim and emits [B, C, H, patchDim] (= 2× the
    /// latent on the FLUX default 32×32×16, since patchSize=2 →
    /// patchDim=64 so output elements = 1·16·32·64 = 32768 vs latent
    /// 16384) — exactly the "PredictNoise output length 32768 does
    /// not match the latent/sample length 16384" failure
    /// <c>Flux2SchnellModelTests.ScaledInput_ShouldChangeOutput</c>
    /// caught (#1305 cluster #6, sibling to the MMDiTXNoisePredictor
    /// fix in #1224 Cluster F).
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
                $"FluxDoubleStreamPredictor expects rank-3 [C,H,W] or rank-4 [B,C,H,W] input; got rank {input4d.Rank}.",
                nameof(noisySample));

        int batch = input4d.Shape[0];
        int channels = input4d.Shape[1];
        int height = input4d.Shape[2];
        int width = input4d.Shape[3];

        if (channels != _inputChannels)
            throw new ArgumentException(
                $"FluxDoubleStreamPredictor configured for {_inputChannels} channels; got {channels}.",
                nameof(noisySample));
        // FLUX uses 2x2 patches (patchDim = inputChannels * 4 in
        // InitializeLayers). Spatial dims must be divisible by 2 — at
        // smaller test fixtures (e.g. [1, 16, 32, 32]) this always
        // holds but we guard explicitly so a misconfigured caller gets
        // a clear shape error instead of an obscure index OOB inside
        // Patchify.
        if (height % PatchSize != 0 || width % PatchSize != 0)
            throw new ArgumentException(
                $"FluxDoubleStreamPredictor requires spatial dims divisible by patchSize ({PatchSize}); got {height}×{width}.",
                nameof(noisySample));

        using var streaming = BeginWeightStreamingForward();

        // Patchify: [B, C, H, W] → [B, numTokens, patchDim].
        var tokens = Patchify(input4d, PatchSize);

        // Embed + propagate through joint (double) + single block stack.
        var x = _patchEmbed.Forward(tokens);
        foreach (var block in _doubleBlocks)
            x = block.Forward(x);
        foreach (var block in _singleBlocks)
            x = block.Forward(x);
        var projected = _finalLayer.Forward(x);  // [B, numTokens, patchDim]

        // Unpatchify back to [B, C, H, W].
        var output = Unpatchify(projected, PatchSize, height, width);

        if (wasUnbatched)
            output = output.Reshape(new[] { channels, height, width });
        return streaming.Complete(output);
    }

    /// <summary>
    /// [B, C, H, W] → [B, (H/P)·(W/P), C·P²] via the standard
    /// rearrange("b c (h p1) (w p2) → b (h w) (c p1 p2)") that the
    /// FLUX / MMDiT reference implementation uses.
    /// </summary>
    private static Tensor<T> Patchify(Tensor<T> input, int patchSize)
    {
        int batch = input.Shape[0];
        int channels = input.Shape[1];
        int height = input.Shape[2];
        int width = input.Shape[3];
        int hPatches = height / patchSize;
        int wPatches = width / patchSize;
        int patchDim = channels * patchSize * patchSize;

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
                        for (int p1 = 0; p1 < patchSize; p1++)
                        {
                            for (int p2 = 0; p2 < patchSize; p2++)
                            {
                                int hSrc = hp * patchSize + p1;
                                int wSrc = wp * patchSize + p2;
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
    /// [B, (H/P)·(W/P), C·P²] → [B, C, H, W] — inverse of
    /// <see cref="Patchify"/>.
    /// </summary>
    private static Tensor<T> Unpatchify(Tensor<T> tokens, int patchSize, int height, int width)
    {
        int batch = tokens.Shape[0];
        int patchDim = tokens.Shape[2];
        int channels = patchDim / (patchSize * patchSize);
        int hPatches = height / patchSize;
        int wPatches = width / patchSize;

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
        // Pre-size the array (1 patch embed + N double + N single + 1 final) so we skip
        // the List growth/copy + ToArray allocation on every optimizer step.
        var vectors = new Vector<T>[2 + _doubleBlocks.Length + _singleBlocks.Length];
        int i = 0;
        vectors[i++] = _patchEmbed.GetParameters();
        foreach (var b in _doubleBlocks) vectors[i++] = b.GetParameters();
        foreach (var b in _singleBlocks) vectors[i++] = b.GetParameters();
        vectors[i] = _finalLayer.GetParameters();
        return Vector<T>.Concatenate(vectors);
    }

    /// <inheritdoc />
    public override void SetParameters(Vector<T> parameters)
    {
        // Reject vectors that don't match ParameterCount up-front so that an upstream
        // optimizer/state bug surfaces as an exception here instead of silently dropping
        // tail values (oversized) or leaving later layers with stale weights (undersized).
        if (parameters.Length != ParameterCount)
        {
            throw new ArgumentException(
                $"Expected {ParameterCount} parameters but got {parameters.Length}.",
                nameof(parameters));
        }

        int offset = 0;
        offset = SetParams(_patchEmbed, parameters, offset);
        foreach (var b in _doubleBlocks) offset = SetParams(b, parameters, offset);
        foreach (var b in _singleBlocks) offset = SetParams(b, parameters, offset);
        offset = SetParams(_finalLayer, parameters, offset);

        // Defense in depth: the per-layer SetParams calls already advance the offset
        // through every trainable parameter, so a final-offset/ParameterCount mismatch
        // would indicate that ParameterCount is out of sync with the layer composition
        // (e.g., a new sub-layer was added without updating the count). Surface that
        // class invariant violation immediately.
        if (offset != ParameterCount)
        {
            throw new InvalidOperationException(
                $"Internal invariant violation: SetParameters consumed {offset} elements " +
                $"but ParameterCount reports {ParameterCount}. This indicates a layer was " +
                "added or removed without updating ParameterCount.");
        }
    }

    /// <inheritdoc />
    public override IFullModel<T, Tensor<T>, Tensor<T>> DeepCopy() => Clone();

    /// <inheritdoc />
    public override INoisePredictor<T> Clone()
    {
        var clone = new FluxDoubleStreamPredictor<T>(_variant, _inputChannels, _contextDim);
        if (!clone.TryShareParametersFrom(this)) clone.SetParameters(GetParameters());
        return clone;
    }

    private static int SetParams(DenseLayer<T> layer, Vector<T> parameters, int offset)
    {
        int count = checked((int)layer.ParameterCount);
        layer.SetParameters(parameters.GetSubVector(offset, count));
        return offset + count;
    }

    protected override Vector<T> GetParameterGradients()
    {
        // Same fixed-size-array pattern as GetParameters above — avoids the per-call
        // List<T> allocation and ToArray copy on the gradient side too.
        var vectors = new Vector<T>[2 + _doubleBlocks.Length + _singleBlocks.Length];
        int i = 0;
        vectors[i++] = _patchEmbed.GetParameterGradients();
        foreach (var b in _doubleBlocks) vectors[i++] = b.GetParameterGradients();
        foreach (var b in _singleBlocks) vectors[i++] = b.GetParameterGradients();
        vectors[i] = _finalLayer.GetParameterGradients();
        return Vector<T>.Concatenate(vectors);
    }
}
