namespace AiDotNet.Enums;

/// <summary>
/// Defines the backbone size variants for MedSAM 2 (3D Medical SAM).
/// </summary>
/// <remarks>
/// <para>
/// <b>Reference:</b> "MedSAM 2: Segment Medical Images As Video Via Segment Anything Model 2",
/// arXiv 2025.
/// </para>
/// </remarks>
public enum MedSAM2ModelSize
{
    /// <summary>Tiny Hiera backbone. Fast inference.</summary>
    Tiny,
    /// <summary>Base Hiera backbone. +36.9% Dice vs vanilla SAM 2.</summary>
    Base,
    /// <summary>Large Hiera backbone. Best accuracy.</summary>
    Large
}
