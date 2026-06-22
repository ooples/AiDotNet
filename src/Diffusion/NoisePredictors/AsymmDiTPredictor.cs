using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;

namespace AiDotNet.Diffusion.NoisePredictors;

/// <summary>
/// Asymmetric Diffusion Transformer (AsymmDiT) noise predictor for video
/// generation (Genmo Mochi 1 architecture).
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// AsymmDiT (Mochi 1) is an MMDiT-family transformer: it jointly attends to text
/// and visual tokens with multi-modal self-attention and learns separate per-stream
/// MLPs, with adaptive-LayerNorm timestep modulation — exactly the MMDiT block.
/// It is therefore realized on the faithful <see cref="MMDiTNoisePredictor{T}"/>
/// dual-stream backbone (joint concatenated attention + per-stream MLPs + adaLN),
/// replacing the previous Dense-only placeholder that had no attention or timestep
/// conditioning.
/// </para>
/// <para>
/// <b>Known deviation from Mochi:</b> Mochi's defining feature is the
/// <i>asymmetry</i> — the visual stream carries ~4× the parameters of the text
/// stream (a wider visual hidden dim) via non-square QKV/output projections. The
/// MMDiT backbone here is <i>symmetric</i> (both streams share the hidden width),
/// so this is a faithful realization of Mochi's joint dual-stream MMDiT block but
/// not yet its stream-width asymmetry. Full asymmetry is tracked as a backbone
/// enhancement (non-square per-stream projections) rather than reverting to the
/// non-faithful Dense stub.
/// </para>
/// <para>
/// Reference: Genmo, "Mochi 1: A New SOTA in Open-Source Video Generation", 2024.
/// </para>
/// </remarks>
[ModelDomain(ModelDomain.Generative)]
[ModelCategory(ModelCategory.Diffusion)]
[ModelCategory(ModelCategory.Transformer)]
[ModelTask(ModelTask.Denoising)]
[ModelTask(ModelTask.TextToVideo)]
[ModelComplexity(ModelComplexity.VeryHigh)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
    [ResearchPaper("Mochi 1: A New SOTA in Open-Source Video Generation", "https://github.com/genmoai/mochi")]
public class AsymmDiTPredictor<T> : MMDiTNoisePredictor<T>
{
    // Retained for a type-correct Clone().
    private readonly int _asymInputChannels;
    private readonly int _asymHiddenSize;
    private readonly int _asymNumLayers;
    private readonly int _asymNumHeads;
    private readonly int _asymContextDim;
    private readonly int? _asymSeed;

    /// <summary>
    /// Initializes a new AsymmDiT (Mochi) predictor on the faithful MMDiT
    /// dual-stream backbone.
    /// </summary>
    /// <param name="inputChannels">Latent channels (default: 12, Mochi VAE).</param>
    /// <param name="hiddenSize">Transformer hidden dimension (default: 3072).</param>
    /// <param name="numLayers">Number of joint dual-stream blocks (default: 48).</param>
    /// <param name="numHeads">Number of attention heads (default: 24).</param>
    /// <param name="contextDim">Text-conditioning dimension (default: 4096, T5-XXL).</param>
    /// <param name="seed">Optional random seed.</param>
    public AsymmDiTPredictor(
        int inputChannels = 12,
        int hiddenSize = 3072,
        int numLayers = 48,
        int numHeads = 24,
        int contextDim = 4096,
        int? seed = null)
        : base(
            inputChannels: inputChannels,
            hiddenSize: hiddenSize,
            numJointLayers: numLayers,
            numSingleLayers: 0,
            numHeads: numHeads,
            contextDim: contextDim,
            seed: seed)
    {
        _asymInputChannels = inputChannels;
        _asymHiddenSize = hiddenSize;
        _asymNumLayers = numLayers;
        _asymNumHeads = numHeads;
        _asymContextDim = contextDim;
        _asymSeed = seed;
    }

    /// <inheritdoc />
    public override INoisePredictor<T> Clone()
    {
        var clone = new AsymmDiTPredictor<T>(
            _asymInputChannels, _asymHiddenSize, _asymNumLayers, _asymNumHeads, _asymContextDim, _asymSeed);
        if (!clone.TryShareParametersFrom(this)) clone.SetParameters(GetParameters());
        return clone;
    }

    /// <inheritdoc />
    public override IFullModel<T, Tensor<T>, Tensor<T>> DeepCopy() => Clone();
}
