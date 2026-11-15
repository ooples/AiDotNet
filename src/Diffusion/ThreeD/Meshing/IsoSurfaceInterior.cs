namespace AiDotNet.Diffusion.ThreeD.Meshing;

/// <summary>
/// Which side of the iso-level of a scalar field is the solid interior.
/// </summary>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> A density or occupancy field is large inside an object, so its inside is
/// <see cref="AboveIsoLevel"/>. A signed distance function (SDF) is negative inside, so its inside is
/// <see cref="BelowIsoLevel"/>. The choice decides which way the triangles face: every extracted triangle is wound
/// counter-clockwise when seen from outside, so its normal points out of the solid.
/// </para>
/// </remarks>
public enum IsoSurfaceInterior
{
    /// <summary>
    /// Values greater than the iso-level are inside (density, occupancy, opacity fields).
    /// </summary>
    AboveIsoLevel,

    /// <summary>
    /// Values less than the iso-level are inside (signed distance functions).
    /// </summary>
    BelowIsoLevel
}
