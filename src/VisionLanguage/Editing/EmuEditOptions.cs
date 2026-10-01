namespace AiDotNet.VisionLanguage.Editing;

/// <summary>
/// Configuration options for Emu Edit: precise image editing via recognition and generation tasks.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> These options configure the EmuEdit model. Default values follow the original paper settings.</para>
/// </remarks>
public class EmuEditOptions : EditingVLMOptions
{
    /// <summary>Initializes a new instance by copying from another instance.</summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public EmuEditOptions(EmuEditOptions other)
    {
        if (other == null)
            throw new ArgumentNullException(nameof(other));

        Seed = other.Seed;
        ImageSize = other.ImageSize;
        VisionDim = other.VisionDim;
        DecoderDim = other.DecoderDim;
        NumVisionLayers = other.NumVisionLayers;
        NumDecoderLayers = other.NumDecoderLayers;
        EditHeadLayers = other.EditHeadLayers;
        NumHeads = other.NumHeads;
        VocabSize = other.VocabSize;
        MaxSequenceLength = other.MaxSequenceLength;
        MaxGenerationLength = other.MaxGenerationLength;
        DropoutRate = other.DropoutRate;
        ArchitectureType = other.ArchitectureType;
        ImageMean = other.ImageMean;
        ImageStd = other.ImageStd;
        ModelPath = other.ModelPath;
        OnnxOptions = other.OnnxOptions;
        LearningRate = other.LearningRate;
        WeightDecay = other.WeightDecay;
        OutputImageSize = other.OutputImageSize;
        NumDiffusionSteps = other.NumDiffusionSteps;
        GuidanceScale = other.GuidanceScale;
        EnablePreciseEditing = other.EnablePreciseEditing;
        UNetBaseChannels = other.UNetBaseChannels;
        UNetChannelMultipliers = (int[])other.UNetChannelMultipliers.Clone();
        UNetNumResBlocks = other.UNetNumResBlocks;
        UNetAttentionResolutions = (int[])other.UNetAttentionResolutions.Clone();
        CrossAttentionDim = other.CrossAttentionDim;
        VaeBaseChannels = other.VaeBaseChannels;
        VaeChannelMultipliers = (int[])other.VaeChannelMultipliers.Clone();
        VaeNumResBlocksPerLevel = other.VaeNumResBlocksPerLevel;
    }

    public EmuEditOptions()
    {
        VisionDim = 1024;
        DecoderDim = 4096;
        NumVisionLayers = 24;
        NumDecoderLayers = 32;
        EditHeadLayers = 4;
        NumHeads = 32;
        ImageSize = 512;
        VocabSize = 32000;
    }

    /// <summary>Gets or sets whether to use recognition-guided precise editing.</summary>
    public bool EnablePreciseEditing { get; set; } = true;

    /// <summary>
    /// Gets or sets the base channel width of the denoising U-Net. Defaults to the 320 of the
    /// Stable Diffusion U-Net geometry Emu Edit's denoiser follows.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> This is the single number that decides how big the image
    /// generator inside Emu Edit is. The default builds the full-size U-Net - roughly 860 million
    /// weights - which is what you want for real editing but far more than a smoke test or an
    /// experiment on a laptop can hold. Lower it (32 or 64) to get the same architecture at a size
    /// that fits in memory.</para>
    /// </remarks>
    public int UNetBaseChannels { get; set; } = 320;

    /// <summary>
    /// Gets or sets the per-level channel multipliers of the denoising U-Net. Defaults to the
    /// four-level <c>{1, 2, 4, 4}</c>.
    /// </summary>
    /// <remarks>
    /// <para>The array length is the number of resolution levels, so each entry both widens the
    /// network and halves the spatial size once more. A latent smaller than
    /// <c>2^(length - 1)</c> collapses to a single pixel before the last level.</para>
    /// </remarks>
    public int[] UNetChannelMultipliers { get; set; } = new[] { 1, 2, 4, 4 };

    /// <summary>
    /// Gets or sets the number of residual blocks per U-Net level. Defaults to 2.
    /// </summary>
    public int UNetNumResBlocks { get; set; } = 2;

    /// <summary>
    /// Gets or sets the spatial resolutions at which the U-Net applies self-attention. Defaults to
    /// <c>{4, 2, 1}</c>.
    /// </summary>
    public int[] UNetAttentionResolutions { get; set; } = new[] { 4, 2, 1 };

    /// <summary>
    /// Gets or sets the cross-attention width shared by the U-Net's conditioning projections and
    /// the edit head. Defaults to 768.
    /// </summary>
    /// <remarks>
    /// <para>This value must stay divisible by <c>NumHeads</c>, because the edit head splits it
    /// across attention heads.</para>
    /// </remarks>
    public int CrossAttentionDim { get; set; } = 768;

    /// <summary>
    /// Gets or sets the base channel width of the VAE. Defaults to 128.
    /// </summary>
    public int VaeBaseChannels { get; set; } = 128;

    /// <summary>
    /// Gets or sets the per-level channel multipliers of the VAE. Defaults to <c>{1, 2, 4, 4}</c>.
    /// </summary>
    public int[] VaeChannelMultipliers { get; set; } = new[] { 1, 2, 4, 4 };

    /// <summary>
    /// Gets or sets the number of residual blocks per VAE level. Defaults to 2.
    /// </summary>
    public int VaeNumResBlocksPerLevel { get; set; } = 2;
}
