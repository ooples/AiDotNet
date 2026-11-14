namespace AiDotNet.ComputerVision.Segmentation.PointCloud;

/// <summary>
/// A space-filling curve that maps a 3D integer grid coordinate to a position on a 1D curve.
/// </summary>
/// <remarks>
/// <para>
/// Point Transformer V3 (Wu et al., CVPR 2024, Sec. 3.1) replaces neighbourhood search with
/// <i>serialization</i>: every voxelised point gets a curve code, the points are sorted by that code,
/// and attention runs over contiguous patches of the sorted sequence. Points that are close on a
/// space-filling curve are close in space, so a patch of the sorted list is a spatial neighbourhood.
/// </para>
/// <para>
/// The paper uses four orders (<see cref="ZOrderCurve"/>, <see cref="HilbertCurve"/> and the
/// x/y-transposed variant of each, built with <see cref="TransposedSpaceFillingCurve"/>) and cycles
/// through them block by block. Implement this interface to add another curve; register it with
/// <see cref="SpaceFillingCurves.Register"/> so a saved or cloned model can rebuild it by
/// <see cref="Name"/>.
/// </para>
/// <para>
/// <b>For Beginners:</b> Imagine drawing one continuous line that visits every cell of a 3D grid.
/// The position along that line is the cell's "code". Sorting points by code lines them up so that
/// neighbours in the list are usually neighbours in space.
/// </para>
/// </remarks>
public interface ISpaceFillingCurve
{
    /// <summary>
    /// A stable identifier for this curve, used to rebuild it when a model is saved, loaded or cloned.
    /// </summary>
    string Name { get; }

    /// <summary>
    /// Encodes one grid coordinate.
    /// </summary>
    /// <param name="x">Grid x, in [0, 2^depth).</param>
    /// <param name="y">Grid y, in [0, 2^depth).</param>
    /// <param name="z">Grid z, in [0, 2^depth).</param>
    /// <param name="depth">Bits per axis; at most 16 so the code fits in 48 bits.</param>
    /// <returns>The curve code, in [0, 2^(3*depth)).</returns>
    long Encode(int x, int y, int z, int depth);
}
