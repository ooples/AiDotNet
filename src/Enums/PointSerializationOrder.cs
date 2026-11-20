namespace AiDotNet.Enums;

/// <summary>
/// The built-in space-filling curve orders Point Transformer V3 serializes points with.
/// </summary>
/// <remarks>
/// <para>
/// PTv3 (Wu et al., CVPR 2024, Sec. 3.1) sorts voxelised points by a space-filling curve code and
/// runs attention over consecutive patches of the sorted list. The paper uses all four orders below,
/// cycling through them block by block (Pointcept <c>order=("z", "z-trans", "hilbert", "hilbert-trans")</c>).
/// The transposed variants swap the x and y axes before encoding.
/// </para>
/// <para>
/// <b>For Beginners:</b> Each value is a different way of drawing one line through every cell of a
/// 3D grid. Using several lines lets each attention patch see a different neighbourhood. To use a
/// curve of your own, implement <c>ISpaceFillingCurve</c> and pass its registered name to
/// <c>PointTransformerV3Encoder</c>.
/// </para>
/// </remarks>
public enum PointSerializationOrder
{
    /// <summary>The Z-order (Morton) curve, Pointcept <c>z</c>.</summary>
    ZOrder,

    /// <summary>The Z-order curve with x and y swapped, Pointcept <c>z-trans</c>.</summary>
    ZOrderTransposed,

    /// <summary>The Hilbert curve, Pointcept <c>hilbert</c>.</summary>
    Hilbert,

    /// <summary>The Hilbert curve with x and y swapped, Pointcept <c>hilbert-trans</c>.</summary>
    HilbertTransposed
}
