namespace AiDotNet.ComputerVision.Segmentation.PointCloud;

/// <summary>
/// The Z-order (Morton) curve: interleaves the bits of x, y and z.
/// </summary>
/// <remarks>
/// <para>
/// Matches Pointcept's <c>z_order.xyz2key</c> (the octree key of OCNN that PTv3 uses): bit <c>i</c>
/// of x lands at code bit <c>3i + 2</c>, of y at <c>3i + 1</c> and of z at <c>3i</c>.
/// </para>
/// <para>
/// <b>For Beginners:</b> The Z-order curve visits the eight sub-cubes of a cube in a fixed "Z"
/// pattern, then repeats that pattern inside each sub-cube. It is cheap to compute and keeps most
/// nearby points close together.
/// </para>
/// </remarks>
public sealed class ZOrderCurve : ISpaceFillingCurve
{
    /// <summary>The name Pointcept uses for this order: <c>z</c>.</summary>
    public string Name => "z";

    /// <inheritdoc />
    public long Encode(int x, int y, int z, int depth)
    {
        SpaceFillingCurves.ValidateCoordinate(x, y, z, depth);
        long key = 0;
        for (int i = 0; i < depth; i++)
        {
            key |= ((long)((x >> i) & 1) << (3 * i + 2))
                   | ((long)((y >> i) & 1) << (3 * i + 1))
                   | ((long)((z >> i) & 1) << (3 * i));
        }

        return key;
    }
}
