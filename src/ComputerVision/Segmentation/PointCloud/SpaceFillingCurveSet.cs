using AiDotNet.Serialization;

namespace AiDotNet.ComputerVision.Segmentation.PointCloud;

/// <summary>
/// The ordered serialization curves of a Point Transformer V3 encoder, saved by registered name.
/// </summary>
/// <remarks>
/// <para>
/// The encoder cycles through these curves block by block (Wu et al., CVPR 2024, Sec. 3.1). A saved,
/// loaded or cloned encoder writes each curve's <see cref="ISpaceFillingCurve.Name"/> and rebuilds it
/// through <see cref="SpaceFillingCurves.Resolve(string)"/>, so every curve must be registered with
/// <see cref="SpaceFillingCurves.Register(ISpaceFillingCurve)"/> first. The four built-in curves
/// always are; saving a set that holds an unregistered custom curve throws instead of writing a
/// model that cannot be loaded.
/// </para>
/// <para>
/// <b>For Beginners:</b> This is just the list of curves the model uses, plus the logic that writes
/// their names into a saved file and finds them again when the file is loaded.
/// </para>
/// </remarks>
public sealed class SpaceFillingCurveSet : ILayerStatePersistable
{
    private const char Separator = '\n';
    private ISpaceFillingCurve[] _curves;

    /// <summary>Creates the paper's four orders: z, z-trans, hilbert, hilbert-trans.</summary>
    public SpaceFillingCurveSet()
    {
        _curves = SpaceFillingCurves.PaperOrders();
    }

    /// <summary>Creates a set holding <paramref name="curves"/> in order.</summary>
    /// <param name="curves">At least one curve; none may be null.</param>
    public SpaceFillingCurveSet(IEnumerable<ISpaceFillingCurve> curves)
    {
        _curves = Validate(curves);
    }

    /// <summary>The curves, in the order the blocks use them.</summary>
    public IReadOnlyList<ISpaceFillingCurve> Curves => _curves;

    /// <summary>Writes the curves' registered names.</summary>
    /// <returns>The names, one per line.</returns>
    /// <exception cref="InvalidOperationException">A curve is not registered under its name.</exception>
    public string SaveState()
    {
        var names = new string[_curves.Length];
        for (int i = 0; i < _curves.Length; i++)
        {
            var curve = _curves[i];
            if (!SpaceFillingCurves.IsRegistered(curve))
            {
                throw new InvalidOperationException(
                    $"Space-filling curve '{curve.Name}' ({curve.GetType().FullName}) is not registered, so a saved or "
                    + $"cloned model could not rebuild it. Call {nameof(SpaceFillingCurves)}.{nameof(SpaceFillingCurves.Register)} "
                    + "with the curve before saving or cloning the model, and again before loading it.");
            }

            names[i] = curve.Name;
        }

        return string.Join(Separator.ToString(), names);
    }

    /// <summary>Rebuilds the curves from names written by <see cref="SaveState"/>.</summary>
    /// <param name="text">The saved names.</param>
    /// <exception cref="ArgumentException">A name is not registered.</exception>
    public void LoadState(string text)
    {
        if (text is null) throw new ArgumentNullException(nameof(text));
        var names = text.Split(Separator);
        var curves = new ISpaceFillingCurve[names.Length];
        for (int i = 0; i < names.Length; i++) curves[i] = SpaceFillingCurves.Resolve(names[i]);
        _curves = Validate(curves);
    }

    private static ISpaceFillingCurve[] Validate(IEnumerable<ISpaceFillingCurve> curves)
    {
        if (curves is null) throw new ArgumentNullException(nameof(curves));
        var array = curves.ToArray();
        if (array.Length == 0) throw new ArgumentException("At least one serialization curve is required.", nameof(curves));
        foreach (var curve in array)
        {
            if (curve is null) throw new ArgumentException("Serialization curves must not be null.", nameof(curves));
        }

        return array;
    }
}
