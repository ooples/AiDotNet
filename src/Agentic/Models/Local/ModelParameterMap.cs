using System.Collections.Generic;
using System.Linq;
using AiDotNet.NeuralNetworks;

namespace AiDotNet.Agentic.Models.Local;

/// <summary>
/// A named segment of a model's flat parameter vector: the parameters contributed by one layer, in the order
/// the model lays them out.
/// </summary>
public sealed class ParameterSegment
{
    /// <summary>Initializes a new segment.</summary>
    /// <param name="name">The segment name (e.g. <c>blk.0</c>).</param>
    /// <param name="offset">The start index in the model's flat parameter vector.</param>
    /// <param name="length">The number of parameters in this segment.</param>
    public ParameterSegment(string name, int offset, int length)
    {
        Name = name;
        Offset = offset;
        Length = length;
    }

    /// <summary>Gets the segment name.</summary>
    public string Name { get; }

    /// <summary>Gets the start index in the model's flat parameter vector.</summary>
    public int Offset { get; }

    /// <summary>Gets the number of parameters in this segment.</summary>
    public int Length { get; }
}

/// <summary>
/// Builds a stable, named map over a network's flat parameter vector, derived from the model's actual layer
/// composition — so callers can address weights by name (and produce the ordered name list
/// <see cref="WeightImporter"/> needs) without hand-maintaining a per-architecture list.
/// </summary>
/// <remarks>
/// <para>
/// AiDotNet networks expose one flat parameter vector built by concatenating each layer's parameters in
/// <c>Layers</c> order (see <see cref="NeuralNetworkBase{T}.GetParameters"/>). This map walks the same layers
/// and assigns each parameter-bearing layer a recognizable name (<c>token_embd.0</c>, <c>blk.0</c>, <c>blk.1</c>,
/// <c>norm.0</c>, <c>output.0</c> for a Mamba stack; a generic <c>{layertype}.{n}</c> for anything else),
/// recording its offset and length. Because it is derived from the live model, the map is correct by
/// construction for every architecture in the library — no per-architecture table to drift.
/// </para>
/// <para>
/// Granularity is per layer (one segment per layer), matching the flat-vector design where a layer's weights
/// are a single contiguous block. Importing a foreign checkpoint that splits a block into many sub-tensors
/// therefore requires a converter that concatenates those sub-tensors into this per-layer order first; the map
/// gives the names and sizes to target.
/// </para>
/// <para><b>For Beginners:</b> A model stores all its numbers in one long list. This gives each chunk of that
/// list a human name (which layer it came from), so you can save, load, or inspect weights by name instead of
/// counting offsets by hand.
/// </para>
/// </remarks>
public static class ModelParameterMap
{
    // Recognizable base names for common layer types; anything else falls back to a sanitized type name.
    private static readonly IReadOnlyDictionary<string, string> KnownNames = new Dictionary<string, string>(StringComparer.Ordinal)
    {
        ["EmbeddingLayer"] = "token_embd",
        ["MambaBlock"] = "blk",
        ["Mamba2Block"] = "blk",
        ["LayerNormalizationLayer"] = "norm",
        ["RMSNormLayer"] = "norm",
        ["DenseLayer"] = "output",
        ["FullyConnectedLayer"] = "output",
    };

    /// <summary>
    /// Builds the ordered, named parameter segments for a model.
    /// </summary>
    /// <typeparam name="T">The model's numeric type.</typeparam>
    /// <param name="model">The network to map.</param>
    /// <returns>One segment per parameter-bearing layer, in flat-parameter order.</returns>
    /// <exception cref="ArgumentNullException">Thrown when <paramref name="model"/> is <c>null</c>.</exception>
    public static IReadOnlyList<ParameterSegment> Build<T>(NeuralNetworkBase<T> model)
    {
        Guard.NotNull(model);

        // Force any lazy layer shapes to resolve so per-layer ParameterCount is accurate — lazy-shape layers
        // (e.g. a final norm/dense whose input width is inferred) report 0 parameters until resolved, exactly
        // as NeuralNetworkBase.GetParameters does before summing. Idempotent.
        _ = model.GetParameters();

        var segments = new List<ParameterSegment>();
        var counters = new Dictionary<string, int>(StringComparer.Ordinal);
        var offset = 0;
        foreach (var layer in model.Layers)
        {
            var length = checked((int)layer.ParameterCount);
            if (length == 0)
            {
                continue;
            }

            var baseName = BaseName(layer.GetType().Name);
            var index = counters.TryGetValue(baseName, out var current) ? current : 0;
            counters[baseName] = index + 1;

            segments.Add(new ParameterSegment($"{baseName}.{index}", offset, length));
            offset += length;
        }

        return segments;
    }

    /// <summary>
    /// Returns the segment names in flat-parameter order — the list <see cref="WeightImporter.ImportInto{T}"/>
    /// expects.
    /// </summary>
    /// <typeparam name="T">The model's numeric type.</typeparam>
    /// <param name="model">The network to map.</param>
    public static IReadOnlyList<string> OrderedNames<T>(NeuralNetworkBase<T> model) =>
        Build(model).Select(segment => segment.Name).ToList();

    private static string BaseName(string typeName)
    {
        // Strip generic arity ("MambaBlock`1" -> "MambaBlock").
        var tick = typeName.IndexOf('`');
        if (tick >= 0)
        {
            typeName = typeName.Substring(0, tick);
        }

        if (KnownNames.TryGetValue(typeName, out var known))
        {
            return known;
        }

        // Generic fallback: drop a trailing "Layer" and lowercase ("ConvolutionalLayer" -> "convolutional").
        if (typeName.EndsWith("Layer", StringComparison.Ordinal) && typeName.Length > "Layer".Length)
        {
            typeName = typeName.Substring(0, typeName.Length - "Layer".Length);
        }

        return typeName.ToLowerInvariant();
    }
}
