namespace AiDotNet.Enums;

/// <summary>
/// The discriminator receptive-field sizes reported by pix2pix (Isola et al., 2017), section 6.1.2.
/// </summary>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> a PatchGAN discriminator does not judge a whole image as "real" or "fake";
/// it judges many overlapping square patches and averages the verdicts. This enum selects how big
/// each judged patch is. Smaller patches police fine texture only; larger patches also police
/// large-scale structure but need more parameters and are more prone to artifacts.
/// </para>
/// <para>
/// The paper varies the patch size purely by discriminator DEPTH — every variant uses the same
/// C64-C128-... pattern, just truncated or extended — so each member here maps to a layer count.
/// </para>
/// </remarks>
public enum PatchGANReceptiveField
{
    /// <summary>
    /// 1x1 "PixelGAN": <c>C64-C128</c> with 1x1 spatial filters. Judges each pixel independently,
    /// so it can encourage colour diversity but has no notion of spatial structure.
    /// </summary>
    Pixel1x1,

    /// <summary>
    /// 16x16 PatchGAN: <c>C64-C128</c>. Sharp local texture, but the paper reports tiling artifacts
    /// because the field is too small to see object-scale structure.
    /// </summary>
    Patch16x16,

    /// <summary>
    /// 70x70 PatchGAN: <c>C64-C128-C256-C512</c>. The paper's default and the setting used for all
    /// its headline results — the best quality/parameter trade-off.
    /// </summary>
    Patch70x70,

    /// <summary>
    /// 286x286 "ImageGAN": <c>C64-C128-C256-C512-C512-C512</c>. Covers the full 256x256 image. The
    /// paper found this scored WORSE than 70x70 despite more capacity, attributing it to the deeper
    /// network being harder to train.
    /// </summary>
    Image286x286
}
