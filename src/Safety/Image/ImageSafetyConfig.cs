namespace AiDotNet.Safety.Image;

/// <summary>
/// Configuration for image safety classification modules.
/// </summary>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> Use this to configure which types of harmful image content
/// to detect and how strict the detection should be.
/// </para>
/// </remarks>
public class ImageSafetyConfig
{
    /// <summary>NSFW detection threshold (0.0-1.0). Default: 0.5.</summary>
    public double? NSFWThreshold { get; set; }

    /// <summary>Violence detection threshold (0.0-1.0). Default: 0.5.</summary>
    public double? ViolenceThreshold { get; set; }

    /// <summary>Whether to detect CSAM content. Default: true (always recommended).</summary>
    public bool? CSAMDetection { get; set; }

    /// <summary>Classifier type to use. Default: Ensemble.</summary>
    public ImageClassifierType? ClassifierType { get; set; }

    internal double EffectiveNSFWThreshold => NSFWThreshold ?? 0.5;
    internal double EffectiveViolenceThreshold => ViolenceThreshold ?? 0.5;
    internal bool EffectiveCSAMDetection => CSAMDetection ?? true;
    internal ImageClassifierType EffectiveClassifierType => ClassifierType ?? ImageClassifierType.Ensemble;
}

/// <summary>
/// The type of image safety classifier to use.
/// </summary>
public enum ImageClassifierType
{
    /// <summary>CLIP embedding-based classifier.</summary>
    CLIP,
    /// <summary>Vision Transformer-based classifier.</summary>
    ViT,
    /// <summary>Scene graph-based classifier.</summary>
    SceneGraph,
    /// <summary>Ensemble combining multiple classifiers (recommended).</summary>
    Ensemble
}
