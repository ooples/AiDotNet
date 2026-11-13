using System.Collections;
using System.Reflection;
using AiDotNet.Interfaces;

namespace AiDotNet.NeuralNetworks.Layers;

/// <summary>
/// A non-layer object a model holds that keeps references into the model's canonical layer list, such as a paper
/// network builder that created the layers, added them to the list and kept them as named blocks.
/// </summary>
/// <remarks>
/// A deserialize or eager clone replaces every canonical layer instance. The generated model rebinding repoints a
/// model's own layer fields and layer collections, but cannot see inside a helper object, which then keeps running
/// the constructor's untrained layers: the clone predicts like an untrained model while serializing the trained
/// weights. A helper implementing this interface is handed the replacement, and the generated rebinding calls it.
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
internal interface ILayerAliasOwner<T>
{
    /// <summary>Repoints every held reference to a layer of <paramref name="previousLayers"/> at the layer with the
    /// same index in <paramref name="replacementLayers"/>.</summary>
    void RebindLayerAliases(IReadOnlyList<ILayer<T>> previousLayers, IReadOnlyList<ILayer<T>> replacementLayers);
}

/// <summary>
/// Rebinds the layer references held anywhere in a helper object graph (fields, lists, arrays and nested helper
/// objects of this library), by reference identity.
/// </summary>
internal static class LayerAliasGraph
{
    private const BindingFlags InstanceFields = BindingFlags.Instance | BindingFlags.Public | BindingFlags.NonPublic;

    /// <summary>Repoints every reference to a layer of <paramref name="previousLayers"/> reachable from
    /// <paramref name="root"/> at the layer with the same index in <paramref name="replacementLayers"/>.</summary>
    public static void Rebind<T>(object root, IReadOnlyList<ILayer<T>> previousLayers, IReadOnlyList<ILayer<T>> replacementLayers)
    {
        if (root is null) throw new ArgumentNullException(nameof(root));
        if (previousLayers is null) throw new ArgumentNullException(nameof(previousLayers));
        if (replacementLayers is null) throw new ArgumentNullException(nameof(replacementLayers));

        var map = new Dictionary<ILayer<T>, ILayer<T>>(Helpers.TensorReferenceComparer<ILayer<T>>.Instance);
        int count = Math.Min(previousLayers.Count, replacementLayers.Count);
        for (int i = 0; i < count; i++)
        {
            if (!ReferenceEquals(previousLayers[i], replacementLayers[i]))
                map[previousLayers[i]] = replacementLayers[i];
        }

        if (map.Count == 0) return;
        Walk(root, map, new HashSet<object>(Helpers.TensorReferenceComparer<object>.Instance));
    }

    private static void Walk<T>(object node, Dictionary<ILayer<T>, ILayer<T>> map, HashSet<object> visited)
    {
        if (!visited.Add(node)) return;

        if (node is IList list)
        {
            for (int i = 0; i < list.Count; i++)
            {
                var item = list[i];
                if (item is ILayer<T> layer)
                {
                    if (map.TryGetValue(layer, out var replacement)) list[i] = replacement;
                }
                else if (item is not null && IsHelper(item))
                {
                    Walk(item, map, visited);
                }
            }

            return;
        }

        for (var type = node.GetType(); type is not null && type != typeof(object); type = type.BaseType)
        {
            foreach (var field in type.GetFields(InstanceFields | BindingFlags.DeclaredOnly))
            {
                var value = field.GetValue(node);
                if (value is ILayer<T> layer)
                {
                    if (!map.TryGetValue(layer, out var replacement)) continue;
                    if (!field.FieldType.IsInstanceOfType(replacement))
                    {
                        throw new InvalidOperationException(
                            $"{type.Name}.{field.Name} holds a {field.FieldType.Name}, but its replacement is a " +
                            $"{replacement.GetType().Name}.");
                    }

                    field.SetValue(node, replacement);
                }
                else if (value is not null && ((value is IList && HoldsReferences(value.GetType())) || IsHelper(value)))
                {
                    Walk(value, map, visited);
                }
            }
        }
    }

    // A helper is a plain object of this library: layers are rebound, not entered (their sublayers are their own),
    // and networks own their own rebinding.
    private static bool IsHelper(object? value)
        => value is not null
           && value is not string
           && value is not Delegate
           && value.GetType().IsClass
           && value.GetType().Assembly == typeof(LayerAliasGraph).Assembly
           && !IsLayerOrNetwork(value.GetType());

    // Value-type arrays and lists (weights, shapes, indices) cannot hold a layer; skipping them avoids boxing every
    // element of a large buffer.
    private static bool HoldsReferences(Type listType)
    {
        if (listType.IsArray) return !(listType.GetElementType()?.IsValueType ?? true);
        var arguments = listType.IsGenericType ? listType.GetGenericArguments() : Type.EmptyTypes;
        return arguments.Length != 1 || !arguments[0].IsValueType;
    }

    private static bool IsLayerOrNetwork(Type type)
    {
        for (var t = type; t is not null; t = t.BaseType)
        {
            if (t.IsGenericType)
            {
                var definition = t.GetGenericTypeDefinition();
                if (definition == typeof(LayerBase<>) || definition == typeof(NeuralNetworkBase<>)) return true;
            }
        }

        return false;
    }
}
