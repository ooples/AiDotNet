namespace AiDotNet.Enums;

/// <summary>
/// How a box's width and height are measured from its (x1, y1, x2, y2) corners when computing areas and IoU.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> Some detectors treat coordinates as pixel indices, so a box from pixel 0 to
/// pixel 9 covers 10 pixels and its width is x2 - x1 + 1. Others treat coordinates as continuous positions,
/// so the same box has width x2 - x1 = 9. The two give slightly different IoU values for the same boxes.</para>
/// </remarks>
public enum BoxAreaConvention
{
    /// <summary>
    /// Corners are inclusive pixel indices: width = x2 - x1 + 1 and height = y2 - y1 + 1, as in
    /// cython_bbox <c>bbox_overlaps</c> used by the official ByteTrack implementation.
    /// </summary>
    PixelInclusive = 0,

    /// <summary>
    /// Corners are continuous coordinates: width = x2 - x1 and height = y2 - y1, as in ultralytics.
    /// </summary>
    Continuous = 1,
}
