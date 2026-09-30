namespace AiDotNet.Models.Options;

/// <summary>
/// Hyperparameters of RT-DETR (Zhao et al., "DETRs Beat YOLOs on Real-time Object Detection", CVPR 2024).
/// The defaults are the paper's decoder and denoising values, which are the same for every size.
/// </summary>
/// <remarks>
/// <para>
/// <see cref="ObjectDetectionOptions{T}.Size"/> selects the paper's backbone and the hybrid encoder that goes
/// with it:
/// <list type="bullet">
/// <item>Nano = R18 and Small = R34, with CSP expansion 0.5 and 3 or 4 decoder layers.</item>
/// <item>Medium = R50, with 6 decoder layers.</item>
/// <item>Large and XLarge = R101, whose encoder is 384 wide with a 2048 feed-forward.</item>
/// </list>
/// The paper's HGNetv2 backbones (RT-DETR-L/X) are not available.
/// </para>
/// <para><b>For Beginners:</b> The defaults reproduce the published model; change them only to experiment.</para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public class RTDETROptions<T> : ObjectDetectionOptions<T>
{
    /// <summary>Creates the options with the paper's values.</summary>
    public RTDETROptions()
    {
        DecoderHiddenDimension = 256;
        NumHeads = 8;
        DecoderFeedForwardDimension = 1024;
        NumQueries = 300;
        NumSamplingPoints = 4;
        DenoisingQueries = 100;
        LabelNoiseRatio = 0.5;
        BoxNoiseScale = 1.0;
        PositionalTemperature = 10000;
    }

    /// <summary>Decoder width (hidden_dim). Paper: 256.</summary>
    public int DecoderHiddenDimension { get; set; }

    /// <summary>Attention heads in AIFI and the decoder. Paper: 8.</summary>
    public int NumHeads { get; set; }

    /// <summary>Decoder feed-forward width. Paper: 1024.</summary>
    public int DecoderFeedForwardDimension { get; set; }

    /// <summary>
    /// Queries chosen by IoU-aware query selection. Paper: 300. An image with fewer encoder tokens than this
    /// selects every token.
    /// </summary>
    public int NumQueries { get; set; }

    /// <summary>Sampling points per head and level in the decoder's deformable attention. Paper: 4.</summary>
    public int NumSamplingPoints { get; set; }

    /// <summary>Denoising query budget (num_denoising), split into positive/negative groups. Paper: 100.</summary>
    public int DenoisingQueries { get; set; }

    /// <summary>Label noise ratio. Half of it is the probability of flipping a label. Paper: 0.5.</summary>
    public double LabelNoiseRatio { get; set; }

    /// <summary>Box noise scale. Paper: 1.0.</summary>
    public double BoxNoiseScale { get; set; }

    /// <summary>Temperature of AIFI's 2-D sin-cos positional encoding. Paper: 10000.</summary>
    public double PositionalTemperature { get; set; }
}
