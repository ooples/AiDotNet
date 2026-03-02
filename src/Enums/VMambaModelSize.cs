namespace AiDotNet.Enums;

/// <summary>
/// Defines the backbone size variants for VMamba visual state space model.
/// </summary>
/// <remarks>
/// <para>
/// <b>Reference:</b> Liu et al., "VMamba: Visual State Space Model", arXiv 2024.
/// </para>
/// </remarks>
public enum VMambaModelSize
{
    /// <summary>Tiny variant.</summary>
    Tiny,
    /// <summary>Small variant.</summary>
    Small,
    /// <summary>Base variant. Best accuracy with cross-scan mechanism.</summary>
    Base
}
