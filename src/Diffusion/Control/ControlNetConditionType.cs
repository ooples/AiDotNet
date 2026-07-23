namespace AiDotNet.Diffusion.Control;

/// <summary>
/// Specifies the type of conditioning input for multi-control ControlNet composition.
/// </summary>
/// <remarks>
/// <para>
/// ControlNet models can accept multiple simultaneous control conditions (e.g., Canny edges + depth map).
/// This enum identifies the semantic type of each conditioning input to enable proper routing,
/// weighting, and compositing of multiple control signals.
/// </para>
/// <para>
/// <b>For Beginners:</b> When using multiple ControlNet conditions at once (e.g., both a sketch
/// and a depth map), the model needs to know what each input represents so it can combine them
/// properly. This enum labels each input with its type, like putting labels on different ingredients
/// before mixing them together.
/// </para>
/// </remarks>
public enum ControlNetConditionType
{
    /// <summary>Canny edge detection map for structural guidance.</summary>
    CannyEdge,
    /// <summary>Monocular depth estimation for spatial layout.</summary>
    DepthMap,
    /// <summary>Surface normal map for lighting and geometry.</summary>
    NormalMap,
    /// <summary>OpenPose body and hand keypoints.</summary>
    OpenPose,
    /// <summary>DWPose whole-body keypoint detection.</summary>
    DWPose,
    /// <summary>Semantic segmentation label map.</summary>
    SemanticSegmentation,
    /// <summary>Instance segmentation with individual object masks.</summary>
    InstanceSegmentation,
    /// <summary>User-drawn scribble or sketch.</summary>
    Scribble,
    /// <summary>Line art or clean sketch drawing.</summary>
    LineArt,
    /// <summary>HED (Holistically-Nested Edge Detection) soft edges.</summary>
    HED,
    /// <summary>SoftEdge detection (PiDiNet or similar).</summary>
    SoftEdge,
    /// <summary>MLSD (Mobile Line Segment Detection) straight lines.</summary>
    MLSD,
    /// <summary>Content shuffle for structure-preserving randomization.</summary>
    ContentShuffle,
    /// <summary>Tile image for detail-preserving upscaling.</summary>
    Tile,
    /// <summary>QR code pattern for embedding readable codes.</summary>
    QRCode,
    /// <summary>Brightness/luminance map for lighting control.</summary>
    Brightness,
    /// <summary>Color palette for palette-guided generation.</summary>
    ColorPalette,
    /// <summary>Color map for per-pixel color guidance.</summary>
    ColorMap,
    /// <summary>Recoloring target for guided color transfer.</summary>
    Recolor,
    /// <summary>Binary inpainting mask (white = inpaint region).</summary>
    InpaintMask,
    /// <summary>FaceID embedding for identity-preserving generation.</summary>
    FaceID,
    /// <summary>MediaPipe face mesh landmarks.</summary>
    MediaPipeFace,
    /// <summary>Reference image features (no explicit spatial map).</summary>
    ReferenceImage,
    /// <summary>SAM (Segment Anything) segmentation mask.</summary>
    SAMSegmentation,
    /// <summary>Style reference for style-aligned generation.</summary>
    StyleReference
}

/// <summary>
/// Represents a single control condition input for multi-control ControlNet composition.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Pairs a conditioning image/tensor with its type identifier and a blending weight,
/// enabling weighted composition of multiple control signals. Used by ControlNet++ and
/// ControlNet-Union models that support simultaneous multi-condition inputs.
/// </para>
/// <para>
/// <b>For Beginners:</b> This is like a labeled ingredient with a measured amount. Each
/// control condition says "I'm a depth map with 80% influence" or "I'm a sketch with 50%
/// influence", so the model knows how much to pay attention to each control signal.
/// </para>
/// </remarks>
/// <param name="ConditionType">The semantic type of this control condition.</param>
/// <param name="ConditionImage">The conditioning tensor (preprocessed control image).</param>
/// <param name="Weight">Blending weight for this condition (0.0 to 1.0, default 1.0).</param>
public record ControlNetCondition<T>(
    ControlNetConditionType ConditionType,
    Tensor<T> ConditionImage,
    double Weight = 1.0);
