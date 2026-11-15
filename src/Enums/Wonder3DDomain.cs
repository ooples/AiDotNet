namespace AiDotNet.Enums;

/// <summary>
/// The output domain Wonder3D's domain switcher selects: the shared U-Net produces a normal map or a color image of
/// a view depending on this.
/// </summary>
/// <remarks>
/// <para>
/// Wonder3D (Long et al. 2023) encodes the domain as a one-hot pair, normal <c>[1, 0]</c> and color <c>[0, 1]</c>,
/// that joins the camera values in the class embedding added to the timestep embedding.
/// </para>
/// <para><b>For Beginners:</b> The same network draws either the surface directions (normals) or the colors of a
/// view; this value tells it which one to draw.</para>
/// </remarks>
public enum Wonder3DDomain
{
    /// <summary>The normal map of a view.</summary>
    Normal,

    /// <summary>The color image of a view.</summary>
    Color,
}
