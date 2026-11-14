using AiDotNet.Attributes;

namespace AiDotNet.ComputerVision.Segmentation.PointCloud;

/// <summary>
/// Configuration options for Point Transformer V3 semantic segmentation.
/// </summary>
/// <remarks>
/// <para>
/// The defaults are Pointcept's <c>PT-v3m1</c> base configuration from the PTv3 paper (Wu et al.,
/// CVPR 2024, Table 9 / <c>semseg-pt-v3m1-0-base</c>): five encoder stages of depths (2, 2, 2, 6, 2)
/// and widths (32, 64, 128, 256, 512), a four-stage decoder of depths (2, 2, 2, 2) and widths
/// (64, 64, 128, 256), patch 1024, the four z/Hilbert orders, xCPE, pre-norm, MLP ratio 4 and drop
/// path 0.3 -- 46.2M backbone parameters.
/// </para>
/// <para><b>For Beginners:</b> Leave these alone to build the network from the paper; change the
/// arrays to make it smaller or larger.</para>
/// </remarks>
[PaperDefault(nameof(NumClasses), 20, Source = "Wu et al. 2024, Sec. 4.1: ScanNet v2 semantic segmentation, 20 classes")]
[PaperDefault(nameof(InputChannels), 6, Source = "Pointcept semseg-pt-v3m1-0-base: in_channels=6 (color, normal)")]
[PaperDefault(nameof(GridSize), 0.02, Source = "Pointcept semseg-pt-v3m1-0-base: GridSample grid_size=0.02")]
[PaperDefault(nameof(Strides), new[] { 2, 2, 2, 2 }, Source = "Wu et al. 2024, appendix model settings (Pointcept PT-v3m1 base): pooling stride (2, 2, 2, 2)")]
[PaperDefault(nameof(EncoderDepths), new[] { 2, 2, 2, 6, 2 }, Source = "Wu et al. 2024, appendix model settings (Pointcept PT-v3m1 base): encoder depth (2, 2, 2, 6, 2)")]
[PaperDefault(nameof(EncoderChannels), new[] { 32, 64, 128, 256, 512 }, Source = "Wu et al. 2024, appendix model settings (Pointcept PT-v3m1 base): encoder channels (32, 64, 128, 256, 512)")]
[PaperDefault(nameof(EncoderNumHeads), new[] { 2, 4, 8, 16, 32 }, Source = "Wu et al. 2024, appendix model settings (Pointcept PT-v3m1 base): encoder num heads (2, 4, 8, 16, 32)")]
[PaperDefault(nameof(EncoderPatchSizes), new[] { 1024, 1024, 1024, 1024, 1024 }, Source = "Wu et al. 2024, appendix model settings (Pointcept PT-v3m1 base): encoder patch size 1024")]
[PaperDefault(nameof(DecoderDepths), new[] { 2, 2, 2, 2 }, Source = "Wu et al. 2024, appendix model settings (Pointcept PT-v3m1 base): decoder depth (2, 2, 2, 2)")]
[PaperDefault(nameof(DecoderChannels), new[] { 64, 64, 128, 256 }, Source = "Wu et al. 2024, appendix model settings (Pointcept PT-v3m1 base): decoder channels (64, 64, 128, 256)")]
[PaperDefault(nameof(DecoderNumHeads), new[] { 4, 4, 8, 16 }, Source = "Wu et al. 2024, appendix model settings (Pointcept PT-v3m1 base): decoder num heads (4, 4, 8, 16)")]
[PaperDefault(nameof(DecoderPatchSizes), new[] { 1024, 1024, 1024, 1024 }, Source = "Wu et al. 2024, appendix model settings (Pointcept PT-v3m1 base): decoder patch size 1024")]
[PaperDefault(nameof(MlpRatio), 4.0, Source = "Wu et al. 2024, appendix model settings (Pointcept PT-v3m1 base): MLP ratio 4")]
[PaperDefault(nameof(DropPathRate), 0.3, Source = "Wu et al. 2024, appendix model settings (Pointcept PT-v3m1 base): drop path 0.3")]
[PaperDefault(nameof(EnableConditionalPositionalEncoding), true, Source = "Wu et al. 2024, Sec. 3.3: enhanced conditional positional encoding (xCPE)")]
[PaperDefault(nameof(PreNorm), true, Source = "Wu et al. 2024, Sec. 3.3 and Table 4: pre-norm with LayerNorm")]
[PaperDefault(nameof(ShuffleOrders), true, Source = "Wu et al. 2024, Sec. 3.2: shuffle order")]
[PaperDefault(nameof(SerializationOrders), new[] { "z", "z-trans", "hilbert", "hilbert-trans" }, Source = "Wu et al. 2024, appendix model settings (Pointcept PT-v3m1 base): serialization pattern Z + TZ + H + TH")]
public class PointTransformerV3Options : PointTransformerV3EncoderOptions
{
    /// <summary>Initializes a new instance with the PTv3 m1 base values.</summary>
    public PointTransformerV3Options()
    {
        NumClasses = 20;
    }

    /// <summary>Initializes a new instance by copying from another instance.</summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public PointTransformerV3Options(PointTransformerV3Options other)
        : base(other)
    {
    }

    /// <summary>
    /// Throws when a value on this instance cannot produce a working model.
    /// </summary>
    /// <exception cref="ArgumentException">Thrown when any backbone or segmentation setting is invalid.</exception>
    public void Validate() => ValidateBackbone(encoderOnly: false);
}
