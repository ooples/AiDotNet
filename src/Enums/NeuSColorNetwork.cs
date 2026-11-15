namespace AiDotNet.Enums;

/// <summary>Which network predicts a NeuS sample's color.</summary>
public enum NeuSColorNetwork
{
    /// <summary>
    /// NeuS's IDR rendering network: point, embedded view direction, gradient and geometry feature into an MLP.
    /// </summary>
    Idr,

    /// <summary>
    /// instant-nsr-pl's <c>VolumeRadiance</c>: the SDF network's output (SDF and features), optionally the view
    /// direction's spherical-harmonics encoding, and the unit normal into a ReLU MLP with a sigmoid.
    /// </summary>
    VolumeRadiance,
}
