namespace AiDotNet.Enums;

/// <summary>
/// Defines the backbone size variants for SEEM (Segment Everything Everywhere All at Once).
/// </summary>
/// <remarks>
/// <para>
/// <b>Reference:</b> Zou et al., "Segment Everything Everywhere All at Once", NeurIPS 2023.
/// </para>
/// </remarks>
public enum SEEMModelSize
{
    /// <summary>Tiny backbone (~30M params).</summary>
    Tiny,
    /// <summary>Large backbone (~307M params).</summary>
    Large
}
