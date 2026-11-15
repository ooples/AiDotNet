namespace AiDotNet.Enums;

/// <summary>
/// The input and output projections of an LDM <c>SpatialTransformer</c> (<c>ldm/modules/attention.py</c>).
/// </summary>
public enum SpatialTransformerProjection
{
    /// <summary><c>nn.Linear</c> over the flattened tokens (<c>use_linear_in_transformer: True</c>).</summary>
    Linear = 0,

    /// <summary>A 1x1 <c>nn.Conv2d</c> over the feature map (Stable Diffusion 1.x).</summary>
    Convolution = 1
}
