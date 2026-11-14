namespace AiDotNet.ComputerVision.Segmentation.PointCloud;

/// <summary>
/// Wraps a curve so it traverses the grid with the x and y axes exchanged.
/// </summary>
/// <remarks>
/// <para>
/// Pointcept's <c>z-trans</c> and <c>hilbert-trans</c> orders encode <c>grid_coord[:, [1, 0, 2]]</c>.
/// Mixing an order with its transpose gives the patch attention neighbourhoods that are elongated in
/// different directions, which PTv3's ablation (Table 5) shows is worth about a point of mIoU.
/// </para>
/// <para>
/// <b>For Beginners:</b> This is the same curve drawn after swapping the x and y axes, so the patches
/// it produces cut space differently from the original curve.
/// </para>
/// </remarks>
public sealed class TransposedSpaceFillingCurve : ISpaceFillingCurve
{
    private readonly ISpaceFillingCurve _inner;

    /// <summary>Creates the x/y-transposed variant of <paramref name="inner"/>.</summary>
    /// <param name="inner">The curve to transpose.</param>
    public TransposedSpaceFillingCurve(ISpaceFillingCurve inner)
    {
        _inner = inner ?? throw new ArgumentNullException(nameof(inner));
    }

    /// <summary>The wrapped curve's name followed by <c>-trans</c>, as in Pointcept.</summary>
    public string Name => _inner.Name + "-trans";

    /// <inheritdoc />
    public long Encode(int x, int y, int z, int depth) => _inner.Encode(y, x, z, depth);
}
