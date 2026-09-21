namespace AiDotNet.Tests.IntegrationTests.Document;

/// <summary>
/// The one CI-scale contract for the Document integration tests: every capacity knob these tests
/// override lives here, with the published default it replaces recorded beside it.
/// </summary>
/// <remarks>
/// <para>
/// <b>Why this exists.</b> The Document models take their published dimensions as constructor
/// defaults — CRAFT is <c>imageSize: 768, backboneChannels: 512, upscaleChannels: 256</c>, Nougat
/// and Pix2Struct are <c>hiddenDim: 1024</c>, and so on. That is correct for the library: a caller
/// who names no size gets the paper's model. It means the TESTS carry the whole responsibility for
/// asking for a small one, and until this type existed they discharged it badly: the image size was
/// written out as a literal at 57 separate call sites, and the capacity knobs — the ones that
/// actually dominate memory — were not overridden at all.
/// </para>
/// <para>
/// <b>Why capacity matters more than image size.</b> Convolution weights scale with the square of
/// the channel count and not at all with the input resolution, so a test that shrinks
/// <c>imageSize</c> and leaves <c>backboneChannels</c> at 512 has shrunk its activations and kept
/// a publication-scale weight set. Shrinking the image alone took the Document namespace from
/// 6.87 GB to roughly 4 GB of peak working set; the channels are the rest of it, and Integration D
/// has to fit several namespaces like this one inside a 16 GB runner.
/// </para>
/// <para>
/// <b>Changing a value here changes every test at once.</b> That is the point — it is what makes
/// the next rung of the float → cap → shrink ladder a one-line experiment rather than a 57-site
/// edit. If one model genuinely cannot run at a value below, give that call site an explicit
/// argument and a comment saying why, rather than raising the shared floor for everything.
/// </para>
/// </remarks>
internal static class DocumentTestScale
{
    /// <summary>Square input edge. Published default is 768 (CRAFT and the OCR detectors).</summary>
    /// <remarks>
    /// 32 is the smallest edge the detection stacks survive: they downsample by 32, so a smaller
    /// input collapses the feature map to nothing and the model throws on an empty tensor.
    /// </remarks>
    public const int ImageSize = 32;

    /// <summary>Detector backbone width. Published defaults are 512 (CRAFT, EAST) and 256 (PSENet, DBNet).</summary>
    public const int BackboneChannels = 32;

    /// <summary>Upscale-head width. Published default is 256 (CRAFT).</summary>
    public const int UpscaleChannels = 16;

    /// <summary>Feature-pyramid width. Published defaults are 256 (PSENet) and 128 (EAST).</summary>
    public const int FeatureChannels = 32;

    /// <summary>Transformer width for the page-to-sequence models. Published default is 1024 (Nougat, Pix2Struct).</summary>
    public const int HiddenDim = 64;

    /// <summary>Embedding width for Donut. Published default is 128.</summary>
    public const int EmbedDim = 32;
}
