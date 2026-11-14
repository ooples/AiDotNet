namespace AiDotNet.ComputerVision.Segmentation.PointCloud;

/// <summary>
/// The 3D Hilbert curve, computed with Skilling's transform exactly as Pointcept's
/// <c>serialization/hilbert.py</c> does.
/// </summary>
/// <remarks>
/// <para>
/// Skilling (2004), "Programming the Hilbert curve": the coordinate bits are taken most significant
/// first, transformed level by level into a Gray code (inverting or swapping the lower bits of each
/// axis against axis 0), interleaved level-major and finally converted from Gray to binary by a
/// prefix XOR. Unlike the Z-order curve, consecutive Hilbert codes are always face-adjacent cells,
/// which is why PTv3 mixes both orders.
/// </para>
/// <para>
/// <b>For Beginners:</b> The Hilbert curve is a line that winds through every cell of the grid
/// without ever jumping: each step moves to a cell that touches the previous one. That makes it
/// better than the Z-order curve at keeping neighbours together, at a slightly higher cost.
/// </para>
/// </remarks>
public sealed class HilbertCurve : ISpaceFillingCurve
{
    private const int Dimensions = 3;

    /// <summary>The name Pointcept uses for this order: <c>hilbert</c>.</summary>
    public string Name => "hilbert";

    /// <inheritdoc />
    public long Encode(int x, int y, int z, int depth)
    {
        SpaceFillingCurves.ValidateCoordinate(x, y, z, depth);
        if (depth == 0) return 0;

        // gray[dim * depth + bit]: bit 0 is the MOST significant of the depth bits, as in Pointcept,
        // which flips the little-endian bytes and keeps the last num_bits bits.
        var gray = new bool[Dimensions * depth];
        int[] coordinate = { x, y, z };
        for (int dim = 0; dim < Dimensions; dim++)
        {
            for (int bit = 0; bit < depth; bit++)
                gray[dim * depth + bit] = ((coordinate[dim] >> (depth - 1 - bit)) & 1) != 0;
        }

        for (int bit = 0; bit < depth; bit++)
        {
            for (int dim = 0; dim < Dimensions; dim++)
            {
                bool mask = gray[dim * depth + bit];
                for (int lower = bit + 1; lower < depth; lower++)
                {
                    // Where this bit is on, invert axis 0's lower bits ...
                    if (mask) gray[lower] = !gray[lower];
                    else
                    {
                        // ... where it is off, exchange axis 0's lower bits with this axis'.
                        bool flip = gray[lower] ^ gray[dim * depth + lower];
                        gray[dim * depth + lower] ^= flip;
                        gray[lower] ^= flip;
                    }
                }
            }
        }

        // Interleave level-major (swapaxes(1, 2)), then Gray -> binary as a prefix XOR from the top.
        long code = 0;
        bool running = false;
        int length = Dimensions * depth;
        for (int position = 0; position < length; position++)
        {
            int bit = position / Dimensions;
            int dim = position % Dimensions;
            running ^= gray[dim * depth + bit];
            if (running) code |= 1L << (length - 1 - position);
        }

        return code;
    }
}
