namespace AiDotNet.Enums;

/// <summary>
/// The shape EAST predicts for each detected piece of text.
/// </summary>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> When EAST finds text in a picture it has to describe WHERE that text is.
/// There are two ways it can do that, and they suit different pictures:
/// </para>
/// <list type="bullet">
/// <item><description>
/// A rotated box, which is a rectangle that can be tilted. This handles a sign photographed at an
/// angle, and it is what the paper uses by default.
/// </description></item>
/// <item><description>
/// A quadrilateral, which is any four-cornered shape. This handles text on a surface viewed from
/// the side, where the rectangle appears stretched into a wedge.
/// </description></item>
/// </list>
/// <para>
/// The choice changes the MODEL, not just the reporting: the geometry head predicts five numbers
/// per position for a rotated box (four edge distances plus an angle) and eight for a
/// quadrilateral (four corner offsets). Zhou et al. (CVPR 2017) describe both.
/// </para>
/// <para>
/// This was previously a <c>string</c> taking "RBOX" or "QUAD". A misspelling compiled fine and
/// silently fell through to the five-channel head, producing a model of the wrong shape with no
/// error -- which is exactly why a closed set of values belongs in an enum.
/// </para>
/// </remarks>
public enum EASTGeometryType
{
    /// <summary>
    /// A rotated rectangle: four distances from the centre to the edges, plus a rotation angle.
    /// Five geometry channels. This is the paper's default.
    /// </summary>
    RBox,

    /// <summary>
    /// An arbitrary quadrilateral: the offsets of its four corners. Eight geometry channels.
    /// </summary>
    Quad,
}
