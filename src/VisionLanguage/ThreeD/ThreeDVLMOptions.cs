using AiDotNet.VisionLanguage.Generative;

namespace AiDotNet.VisionLanguage.ThreeD;

/// <summary>
/// Base configuration options for 3D vision-language models.
/// </summary>
public class ThreeDVLMOptions : GenerativeVLMOptions
{
    /// <summary>Gets or sets the maximum number of 3D points the model can process.</summary>
    public int MaxPoints { get; set; } = 8192;

    /// <summary>Gets or sets the number of channels per point (3=XYZ, 6=XYZ+RGB, 9=XYZ+RGB+normals).</summary>
    public int PointChannels { get; set; } = 6;

    /// <summary>Gets or sets the language model backbone name.</summary>
    public string LanguageModelName { get; set; } = "LLaMA";

    /// <summary>Gets or sets the point cloud encoder hidden dimension.</summary>
    public int PointEncoderDim { get; set; } = 512;
}
