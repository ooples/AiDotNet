namespace AiDotNet.NeuralRadianceFields.NeuS;

/// <summary>
/// An axis-aligned box over which a signed distance field is sampled on a dense grid.
/// </summary>
/// <remarks>
/// <para>
/// A grid of resolution R has R samples per axis, at the box corners and every
/// <c>(max - min) / (R - 1)</c> between them, as NeuS's <c>extract_fields</c> lays it out
/// (<c>torch.linspace(bound_min, bound_max, resolution)</c> per axis). Sample <c>[i, j, k]</c> lies at
/// <c>(MinX + i·dx, MinY + j·dy, MinZ + k·dz)</c>.
/// </para>
/// <para><b>For Beginners:</b> This is the box you want to turn into a mesh. NeuS normalizes every scene so the
/// object sits inside the unit sphere, so the default box is the cube from -1 to 1 on every axis.</para>
/// </remarks>
internal sealed class SdfGridBounds
{
    /// <summary>Creates a box from its two opposite corners.</summary>
    /// <param name="minX">The minimum x.</param>
    /// <param name="minY">The minimum y.</param>
    /// <param name="minZ">The minimum z.</param>
    /// <param name="maxX">The maximum x.</param>
    /// <param name="maxY">The maximum y.</param>
    /// <param name="maxZ">The maximum z.</param>
    public SdfGridBounds(double minX, double minY, double minZ, double maxX, double maxY, double maxZ)
    {
        if (!(maxX > minX) || !(maxY > minY) || !(maxZ > minZ))
            throw new ArgumentException("Each maximum must exceed its minimum.");
        Min = new[] { minX, minY, minZ };
        Max = new[] { maxX, maxY, maxZ };
    }

    /// <summary>The cube [-1, 1]³ that NeuS's normalized object frame occupies (the reference's default bounds).</summary>
    public static SdfGridBounds UnitCube => new(-1.0, -1.0, -1.0, 1.0, 1.0, 1.0);

    /// <summary>The minimum corner (x, y, z).</summary>
    public double[] Min { get; }

    /// <summary>The maximum corner (x, y, z).</summary>
    public double[] Max { get; }

    /// <summary>The spacing between grid samples along <paramref name="axis"/> for a grid of <paramref name="resolution"/>.</summary>
    /// <param name="resolution">Samples per axis (at least 2).</param>
    /// <param name="axis">0, 1 or 2.</param>
    public double Spacing(int resolution, int axis)
    {
        if (resolution < 2) throw new ArgumentOutOfRangeException(nameof(resolution), "A grid needs at least two samples per axis.");
        return (Max[axis] - Min[axis]) / (resolution - 1);
    }

    /// <summary>
    /// The world positions of every grid sample, <c>[R·R·R, 3]</c> as a row-major host array, x slowest and z fastest.
    /// </summary>
    /// <param name="resolution">Samples per axis.</param>
    public double[] GridPoints(int resolution)
    {
        double dx = Spacing(resolution, 0), dy = Spacing(resolution, 1), dz = Spacing(resolution, 2);
        long total = (long)resolution * resolution * resolution;
        if (total * 3 > int.MaxValue) throw new ArgumentOutOfRangeException(nameof(resolution), "The grid is too large.");
        var points = new double[total * 3];
        int p = 0;
        for (int i = 0; i < resolution; i++)
            for (int j = 0; j < resolution; j++)
                for (int k = 0; k < resolution; k++)
                {
                    points[p++] = Min[0] + i * dx;
                    points[p++] = Min[1] + j * dy;
                    points[p++] = Min[2] + k * dz;
                }

        return points;
    }
}
