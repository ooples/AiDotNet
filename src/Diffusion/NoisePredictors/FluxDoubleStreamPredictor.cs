using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;

namespace AiDotNet.Diffusion.NoisePredictors;

/// <summary>
/// FLUX.1 double-stream transformer noise predictor (Black Forest Labs).
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// FLUX.1 is a 12B rectified-flow transformer with a hybrid MMDiT design:
/// <b>19 double-stream blocks</b> (separate image/text streams with joint
/// concatenated self-attention and per-stream MLPs, à la SD3 MMDiT) followed by
/// <b>38 single-stream blocks</b> (text and image concatenated into one sequence
/// through a unified path), at hidden size 3072 with 24 attention heads and
/// adaptive-LayerNorm timestep modulation throughout. It is realized on the
/// faithful <see cref="MMDiTNoisePredictor{T}"/> backbone, which natively supports
/// the joint-block + single-block split (numJointLayers + numSingleLayers). This
/// replaces the previous Dense-only placeholder that had no attention, no
/// timestep conditioning, and no dual stream.
/// </para>
/// <para>
/// <b>For Beginners:</b> FLUX first lets image and text tokens interact in two
/// separate "lanes" that still attend to each other (double-stream), then merges
/// them into one lane for deeper fusion (single-stream). Every layer is steered by
/// the diffusion timestep. This is the real architecture, not a stand-in.
/// </para>
/// <para>
/// Reference: Black Forest Labs, "FLUX.1", 2024.
/// </para>
/// </remarks>
/// <example>
/// <code>
/// var predictor = new FluxDoubleStreamPredictor&lt;float&gt;(inputChannels: 16);
/// var noisyLatent = Tensor&lt;float&gt;.Random(new[] { 1, 16, 32, 32 });
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
    [ResearchPaper("FLUX.1", "https://blackforestlabs.ai/announcing-black-forest-labs/")]
public class FluxDoubleStreamPredictor<T> : MMDiTNoisePredictor<T>
{
    // FLUX.1 scale: 19 double-stream (joint) + 38 single-stream blocks,
    // hidden 3072, 24 heads (3072 % 24 = 128).
    private const int FLUX_HIDDEN_SIZE = 3072;
    private const int FLUX_NUM_JOINT_LAYERS = 19;
    private const int FLUX_NUM_SINGLE_LAYERS = 38;
    private const int FLUX_NUM_HEADS = 24;

    // Retained for a type-correct Clone().
    private readonly FluxPredictorVariant _variant;
    private readonly int _fluxInputChannels;
    private readonly int _fluxContextDim;
    private readonly int? _fluxSeed;

    /// <summary>
    /// Initializes a new FLUX.1 predictor on the faithful MMDiT dual-stream
    /// backbone (19 joint + 38 single blocks, hidden 3072, 24 heads).
    /// </summary>
    /// <param name="variant">FLUX variant (Dev / Schnell). Default: Dev.</param>
    /// <param name="inputChannels">Latent channels (default: 16).</param>
    /// <param name="contextDim">Text-conditioning dimension (default: 4096, T5-XXL).</param>
    /// <param name="seed">Optional random seed.</param>
    public FluxDoubleStreamPredictor(
        FluxPredictorVariant variant = FluxPredictorVariant.Dev,
        int inputChannels = 16,
        int contextDim = 4096,
        int? seed = null)
        : base(
            inputChannels: inputChannels,
            hiddenSize: FLUX_HIDDEN_SIZE,
            numJointLayers: FLUX_NUM_JOINT_LAYERS,
            numSingleLayers: FLUX_NUM_SINGLE_LAYERS,
            numHeads: FLUX_NUM_HEADS,
            contextDim: contextDim,
            seed: seed)
    {
        _variant = variant;
        _fluxInputChannels = inputChannels;
        _fluxContextDim = contextDim;
        _fluxSeed = seed;
    }

    /// <inheritdoc />
    public override INoisePredictor<T> Clone()
    {
        var clone = new FluxDoubleStreamPredictor<T>(_variant, _fluxInputChannels, _fluxContextDim, _fluxSeed);
        // #1711: MMDiT LazyDense weights resolve via the FORWARD path; ProbeMaterializeAndCopyInto
        // probe-forwards the clone then copies, instead of a naive re-RNG-initializing copy.
        ProbeMaterializeAndCopyInto(clone);
        return clone;
    }

    /// <inheritdoc />
    public override IFullModel<T, Tensor<T>, Tensor<T>> DeepCopy() => Clone();
}
