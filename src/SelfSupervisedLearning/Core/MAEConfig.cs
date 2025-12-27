namespace AiDotNet.SelfSupervisedLearning.Core;

/// <summary>
/// MAE-specific configuration settings.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> MAE (Masked Autoencoder) learns by masking random patches
/// of an image and training the model to reconstruct them.</para>
/// </remarks>
public class MAEConfig
{
    /// <summary>
    /// Gets or sets the patch size.
    /// </summary>
    /// <remarks>
    /// <para>Default: <c>16</c></para>
    /// <para>Common sizes are 16 (for 224x224 images, giving 196 patches) or 14 (for ViT).</para>
    /// </remarks>
    public int? PatchSize { get; set; }

    /// <summary>
    /// Gets or sets the fraction of patches to mask.
    /// </summary>
    /// <remarks>
    /// <para>Default: <c>0.75</c></para>
    /// <para>MAE uses aggressive masking (75%) which is efficient and effective.</para>
    /// </remarks>
    public double? MaskRatio { get; set; }

    /// <summary>
    /// Gets or sets the decoder embedding dimension.
    /// </summary>
    /// <remarks>
    /// <para>Default: <c>512</c></para>
    /// <para>Decoder is typically smaller than encoder.</para>
    /// </remarks>
    public int? DecoderEmbedDimension { get; set; }

    /// <summary>
    /// Gets or sets the number of decoder transformer blocks.
    /// </summary>
    /// <remarks>
    /// <para>Default: <c>8</c></para>
    /// </remarks>
    public int? DecoderDepth { get; set; }

    /// <summary>
    /// Gets or sets the number of decoder attention heads.
    /// </summary>
    /// <remarks>
    /// <para>Default: <c>16</c></para>
    /// </remarks>
    public int? DecoderNumHeads { get; set; }

    /// <summary>
    /// Gets or sets whether to normalize reconstruction target.
    /// </summary>
    /// <remarks>
    /// <para>Default: <c>true</c></para>
    /// <para>Normalizing patches before reconstruction typically helps.</para>
    /// </remarks>
    public bool? NormalizeTarget { get; set; }

    /// <summary>
    /// Gets the configuration as a dictionary.
    /// </summary>
    public IDictionary<string, object> GetConfiguration()
    {
        var config = new Dictionary<string, object>();
        if (PatchSize.HasValue) config["patchSize"] = PatchSize.Value;
        if (MaskRatio.HasValue) config["maskRatio"] = MaskRatio.Value;
        if (DecoderEmbedDimension.HasValue) config["decoderEmbedDimension"] = DecoderEmbedDimension.Value;
        if (DecoderDepth.HasValue) config["decoderDepth"] = DecoderDepth.Value;
        if (DecoderNumHeads.HasValue) config["decoderNumHeads"] = DecoderNumHeads.Value;
        if (NormalizeTarget.HasValue) config["normalizeTarget"] = NormalizeTarget.Value;
        return config;
    }
}
