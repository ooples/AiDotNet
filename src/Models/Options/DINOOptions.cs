namespace AiDotNet.Models.Options;

/// <summary>
/// Hyperparameters of DINO (Zhang et al., "DINO: DETR with Improved DeNoising Anchor Boxes for End-to-End
/// Object Detection", ICLR 2023). The defaults are the paper's DINO-4scale configuration.
/// </summary>
/// <remarks>
/// <para>
/// <see cref="ObjectDetectionOptions{T}.Size"/> selects the paper's backbone: Nano, Small and Medium use
/// ResNet-50 at four scales, Large uses Swin-L at four scales, and XLarge uses Swin-L at five scales. The
/// transformer is the same for every size, as in the paper.
/// </para>
/// <para><b>For Beginners:</b> The defaults reproduce the published model; change them only to experiment.</para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public class DINOOptions<T> : ObjectDetectionOptions<T>
{
    /// <summary>Creates the options with the paper's DINO-4scale values.</summary>
    public DINOOptions()
    {
        HiddenDimension = 256;
        NumHeads = 8;
        NumEncoderLayers = 6;
        NumDecoderLayers = 6;
        FeedForwardDimension = 2048;
        NumQueries = 900;
        NumSamplingPoints = 4;
        DenoisingQueries = 100;
        LabelNoiseRatio = 0.5;
        BoxNoiseScale = 1.0;
        PositionalTemperature = 20;
    }

    /// <summary>Transformer width (d_model). Paper: 256.</summary>
    public int HiddenDimension { get; set; }

    /// <summary>Attention heads in every attention block. Paper: 8.</summary>
    public int NumHeads { get; set; }

    /// <summary>Deformable encoder layers. Paper: 6.</summary>
    public int NumEncoderLayers { get; set; }

    /// <summary>Decoder layers. Paper: 6.</summary>
    public int NumDecoderLayers { get; set; }

    /// <summary>Feed-forward width in the encoder and decoder layers. Paper: 2048.</summary>
    public int FeedForwardDimension { get; set; }

    /// <summary>
    /// Object queries selected from the encoder (mixed query selection). Paper: 900. An image with fewer
    /// encoder tokens than this selects every token.
    /// </summary>
    public int NumQueries { get; set; }

    /// <summary>Sampling points per head and level in deformable attention. Paper: 4.</summary>
    public int NumSamplingPoints { get; set; }

    /// <summary>
    /// Contrastive-denoising query budget (dn_number). It is split into groups of one positive and one
    /// negative copy of every target. Paper: 100.
    /// </summary>
    public int DenoisingQueries { get; set; }

    /// <summary>Label noise ratio. Half of it is the probability of flipping a label. Paper: 0.5.</summary>
    public double LabelNoiseRatio { get; set; }

    /// <summary>Box noise scale (lambda). Paper: 1.0.</summary>
    public double BoxNoiseScale { get; set; }

    /// <summary>Temperature of the 2-D sine positional encoding. DINO uses 20.</summary>
    public double PositionalTemperature { get; set; }
}
