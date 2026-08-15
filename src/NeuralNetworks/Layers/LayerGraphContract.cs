namespace AiDotNet.NeuralNetworks.Layers;

/// <summary>
/// Declares a non-sequential graph boundary inside an iterator-based layer factory. Plain
/// <c>yield return</c> remains the beginner-friendly sequential default; composite factories use
/// these markers where a layer starts an independent external or derived branch.
/// </summary>
/// <remarks>
/// The methods return the exact same layer instance and add no runtime wrapper. Their purpose is to
/// make the topology visible to the input-contract generator, which can then validate every real
/// sequential edge without inventing an edge between two independently registered components.
/// </remarks>
public static class LayerGraphContract
{
    private sealed class GraphRootMarker
    {
    }

    // The marker must survive past source analysis: runtime shape propagation also needs to know
    // that a list entry starts an independent branch. ConditionalWeakTable preserves the exact
    // layer identity without wrapping it, changing ownership, or keeping discarded graphs alive.
    private static readonly System.Runtime.CompilerServices.ConditionalWeakTable<object, GraphRootMarker>
        GraphRoots = new();

    /// <summary>Declares that <paramref name="layer"/> consumes a separate caller input.</summary>
    public static TLayer FromExternalInput<TLayer>(TLayer layer)
        where TLayer : class
    {
        if (layer is null) throw new ArgumentNullException(nameof(layer));
        GraphRoots.GetValue(layer, static _ => new GraphRootMarker());
        return layer;
    }

    /// <summary>
    /// Declares that <paramref name="layer"/> consumes a tensor derived from a named earlier input,
    /// such as position IDs derived from token IDs.
    /// </summary>
    public static TLayer FromDerivedInput<TLayer>(TLayer layer, string sourcePort)
        where TLayer : class
    {
        if (layer is null) throw new ArgumentNullException(nameof(layer));
        if (string.IsNullOrWhiteSpace(sourcePort))
            throw new ArgumentException("A derived graph root requires its source port name.", nameof(sourcePort));
        GraphRoots.GetValue(layer, static _ => new GraphRootMarker());
        return layer;
    }

    /// <summary>
    /// Returns whether a layer begins a non-sequential graph branch declared by this contract.
    /// </summary>
    internal static bool StartsIndependentBranch(object? layer)
        => layer is not null && GraphRoots.TryGetValue(layer, out _);
}
