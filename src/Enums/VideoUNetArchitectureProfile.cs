namespace AiDotNet.Enums;

/// <summary>Topology profiles supported by the shared video U-Net predictor.</summary>
public enum VideoUNetArchitectureProfile
{
    /// <summary>General video-diffusion U-Net behavior retained for existing callers.</summary>
    Generic = 0,

    /// <summary>
    /// Released Upscale-A-Video topology: two spatial ResNets per down block,
    /// three per up block, all-stage convolutional temporal modules, and the
    /// zero-initialized temporal attention embedded in Transformer3D blocks.
    /// </summary>
    UpscaleAVideo = 1
}
