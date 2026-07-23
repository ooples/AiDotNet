namespace AiDotNet.Preprocessing.Document;

/// <summary>
/// Options for document preprocessing.
/// </summary>
public class DocumentPreprocessingOptions
{
    /// <summary>
    /// Whether to apply deskewing. Default: true.
    /// </summary>
    public bool ApplyDeskew { get; set; } = true;

    /// <summary>
    /// Maximum deskew angle in degrees. Default: 45.
    /// </summary>
    public double DeskewMaxAngle { get; set; } = 45.0;

    /// <summary>
    /// Whether to apply binarization. Default: false.
    /// </summary>
    public bool ApplyBinarization { get; set; }

    /// <summary>
    /// Binarization method. Default: Otsu.
    /// </summary>
    public BinarizationMethod BinarizationMethod { get; set; } = BinarizationMethod.Otsu;

    /// <summary>
    /// Whether to apply noise removal. Default: true.
    /// </summary>
    public bool ApplyNoiseRemoval { get; set; } = true;

    /// <summary>
    /// Noise removal method. Default: Median.
    /// </summary>
    public NoiseRemovalMethod NoiseRemovalMethod { get; set; } = NoiseRemovalMethod.Median;

    /// <summary>
    /// Whether to apply layout normalization. Default: false.
    /// </summary>
    public bool ApplyLayoutNormalization { get; set; }

    /// <summary>
    /// Target width for layout normalization.
    /// </summary>
    public int TargetWidth { get; set; } = 224;

    /// <summary>
    /// Target height for layout normalization.
    /// </summary>
    public int TargetHeight { get; set; } = 224;

    /// <summary>
    /// Whether to normalize intensity to 0-1 range. Default: true.
    /// </summary>
    public bool NormalizeIntensity { get; set; } = true;
}
