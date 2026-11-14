using System.Collections.Concurrent;

namespace AiDotNet.ComputerVision.Segmentation.PointCloud;

/// <summary>
/// The built-in serialization orders and a name registry for custom ones.
/// </summary>
/// <remarks>
/// <para>
/// A model stores its serialization orders by <see cref="ISpaceFillingCurve.Name"/> so that saving,
/// loading and cloning rebuild the same curves. The four orders of the PTv3 paper are always
/// available; a custom <see cref="ISpaceFillingCurve"/> must be registered before a model that uses
/// it is loaded. Configuring a model with a custom curve registers it automatically.
/// </para>
/// <para>
/// <b>For Beginners:</b> This is the list of curve types the library knows how to rebuild from a
/// saved file. The standard ones are always there; if you write your own curve, add it here.
/// </para>
/// </remarks>
public static class SpaceFillingCurves
{
    private static readonly ConcurrentDictionary<string, ISpaceFillingCurve> Registered =
        new(StringComparer.Ordinal);

    static SpaceFillingCurves()
    {
        Register(ZOrder);
        Register(ZOrderTransposed);
        Register(Hilbert);
        Register(HilbertTransposed);
    }

    /// <summary>The Z-order curve (<c>z</c>).</summary>
    public static ISpaceFillingCurve ZOrder { get; } = new ZOrderCurve();

    /// <summary>The x/y-transposed Z-order curve (<c>z-trans</c>).</summary>
    public static ISpaceFillingCurve ZOrderTransposed { get; } = new TransposedSpaceFillingCurve(new ZOrderCurve());

    /// <summary>The Hilbert curve (<c>hilbert</c>).</summary>
    public static ISpaceFillingCurve Hilbert { get; } = new HilbertCurve();

    /// <summary>The x/y-transposed Hilbert curve (<c>hilbert-trans</c>).</summary>
    public static ISpaceFillingCurve HilbertTransposed { get; } = new TransposedSpaceFillingCurve(new HilbertCurve());

    /// <summary>
    /// The four orders of the PTv3 base configuration and of Sonata, in the paper's order:
    /// <c>z, z-trans, hilbert, hilbert-trans</c>.
    /// </summary>
    /// <returns>A new array, so callers may modify it.</returns>
    public static ISpaceFillingCurve[] PaperOrders() => new[] { ZOrder, ZOrderTransposed, Hilbert, HilbertTransposed };

    /// <summary>The curve for a built-in <see cref="PointSerializationOrder"/>.</summary>
    /// <param name="order">The order.</param>
    /// <returns>The shared curve instance.</returns>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="order"/> is not a defined value.</exception>
    public static ISpaceFillingCurve Get(PointSerializationOrder order) => order switch
    {
        PointSerializationOrder.ZOrder => ZOrder,
        PointSerializationOrder.ZOrderTransposed => ZOrderTransposed,
        PointSerializationOrder.Hilbert => Hilbert,
        PointSerializationOrder.HilbertTransposed => HilbertTransposed,
        _ => throw new ArgumentOutOfRangeException(nameof(order), order, "Unknown serialization order.")
    };

    /// <summary>Registers a curve so models referring to its name can be rebuilt.</summary>
    /// <param name="curve">The curve; replaces any curve already registered under the same name.</param>
    public static void Register(ISpaceFillingCurve curve)
    {
        if (curve is null) throw new ArgumentNullException(nameof(curve));
        if (string.IsNullOrWhiteSpace(curve.Name))
            throw new ArgumentException("A space-filling curve needs a non-empty Name.", nameof(curve));
        if (curve.Name.IndexOf('\n') >= 0 || curve.Name.IndexOf('\r') >= 0)
            throw new ArgumentException("A space-filling curve Name must be a single line.", nameof(curve));
        Registered[curve.Name] = curve;
    }

    /// <summary>Whether <paramref name="curve"/> is registered, so a saved model can rebuild it by name.</summary>
    /// <param name="curve">The curve.</param>
    /// <returns>True when its name maps to this instance or to another instance of the same type.</returns>
    public static bool IsRegistered(ISpaceFillingCurve curve)
    {
        if (curve is null) throw new ArgumentNullException(nameof(curve));
        return curve.Name is not null && Registered.TryGetValue(curve.Name, out var registered)
            && (ReferenceEquals(registered, curve) || registered.GetType() == curve.GetType());
    }

    /// <summary>Looks up a registered curve by name.</summary>
    /// <param name="name">The curve's <see cref="ISpaceFillingCurve.Name"/>.</param>
    /// <returns>The registered curve.</returns>
    /// <exception cref="ArgumentException">No curve with that name is registered.</exception>
    public static ISpaceFillingCurve Resolve(string name)
    {
        if (name is not null && Registered.TryGetValue(name, out var curve)) return curve;
        throw new ArgumentException(
            $"No space-filling curve named '{name}' is registered. Register custom curves with "
            + $"{nameof(SpaceFillingCurves)}.{nameof(Register)} before building or loading the model.",
            nameof(name));
    }

    internal static void ValidateCoordinate(int x, int y, int z, int depth)
    {
        if (depth < 0 || depth > 16)
            throw new ArgumentOutOfRangeException(nameof(depth), depth, "Serialization depth must be in [0, 16].");
        int limit = 1 << depth;
        if (x < 0 || y < 0 || z < 0 || x >= limit || y >= limit || z >= limit)
            throw new ArgumentOutOfRangeException(nameof(x),
                $"Grid coordinate ({x}, {y}, {z}) lies outside [0, {limit}) for depth {depth}.");
    }
}
