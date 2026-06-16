using System.Collections;
using System.Collections.Generic;
using System.Reflection;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;

namespace AiDotNet.Helpers;

/// <summary>
/// Copy-on-write clone lever (#1624): shares a model's trainable weight tensors with its clone via the
/// Tensors O(1)-until-write <see cref="Tensor{T}.CloneShared"/> (issue #624), instead of the
/// <c>SetParameters(GetParameters())</c> flatten round-trip that materializes the whole weight set a
/// second time (plus a giant intermediate flat vector) — the source of large-model <c>Clone()</c> OOMs
/// on the 16 GB CI runner.
/// </summary>
/// <remarks>
/// <para>Universal across base classes: any <c>Clone()</c> that builds a fresh same-typed instance can
/// replace <c>clone.SetParameters(GetParameters())</c> with
/// <c>if (!CopyOnWriteCloneHelper.TryShareTrainableParameters&lt;T&gt;(this, clone)) clone.SetParameters(GetParameters());</c>.
/// The first in-place write to either side privatizes that tensor, so the clone is observationally
/// identical to the flat-copy clone. Fidelity is equivalent to the flat copy because both transfer
/// exactly the model's trainable tensors.</para>
/// </remarks>
public static class CopyOnWriteCloneHelper
{
    /// <summary>
    /// Re-binds every trainable parameter of <paramref name="dest"/> to a copy-on-write share of the
    /// corresponding parameter of <paramref name="source"/>. Walks both object graphs in parallel by
    /// reflection (identical runtime type ⇒ identical field order ⇒ matching layer order). Returns
    /// <c>false</c> — leaving <paramref name="dest"/> untouched — if the trainable-layer structure does
    /// not line up 1:1 (e.g. a freshly-constructed clone whose lazy layers aren't resolved yet), so the
    /// caller can fall back to the eager flat copy.
    /// </summary>
    public static bool TryShareTrainableParameters<T>(object? source, object? dest)
    {
        if (source is null || dest is null || ReferenceEquals(source, dest)) return false;
        if (source.GetType() != dest.GetType()) return false;

        var srcLayers = CollectTrainableLayers<T>(source);
        var dstLayers = CollectTrainableLayers<T>(dest);
        if (srcLayers.Count == 0 || srcLayers.Count != dstLayers.Count) return false;

        // Verify the full structure matches BEFORE mutating anything, so we never leave a half-shared clone.
        for (int i = 0; i < srcLayers.Count; i++)
            if (srcLayers[i].GetTrainableParameters().Count != dstLayers[i].GetTrainableParameters().Count)
                return false;

        for (int i = 0; i < srcLayers.Count; i++)
        {
            var sp = srcLayers[i].GetTrainableParameters();
            if (sp.Count == 0) continue;
            var shared = new Tensor<T>[sp.Count];
            for (int p = 0; p < sp.Count; p++)
                shared[p] = (Tensor<T>)sp[p].CloneShared();
            dstLayers[i].SetTrainableParameters(shared);
        }

        return true;
    }

    private static List<ITrainableLayer<T>> CollectTrainableLayers<T>(object root)
    {
        var layers = new List<ITrainableLayer<T>>();
        CollectInto(root, layers, new HashSet<object>(TensorReferenceComparer<object>.Instance));
        return layers;
    }

    private static void CollectInto<T>(object? obj, List<ITrainableLayer<T>> layers, HashSet<object> visited)
    {
        if (obj is null || !visited.Add(obj)) return;
        if (obj is ITrainableLayer<T> trainable) layers.Add(trainable);

        var type = obj.GetType();
        if (type.IsPrimitive || type == typeof(string) || type.IsEnum) return;

        foreach (var field in type.GetFields(
            BindingFlags.Instance | BindingFlags.NonPublic | BindingFlags.Public))
        {
            // Tensor fields are leaves (their owning layer already exposed them via
            // GetTrainableParameters); skip primitives/strings/enums that can't hold a layer.
            if (field.FieldType.IsPrimitive || field.FieldType.IsEnum ||
                field.FieldType == typeof(string) || field.FieldType == typeof(Tensor<T>))
                continue;

            var val = field.GetValue(obj);
            if (val is null) continue;

            if (val is IEnumerable enumerable && val is not string)
            {
                foreach (var item in enumerable)
                    CollectInto(item, layers, visited);
            }
            else
            {
                CollectInto(val, layers, visited);
            }
        }
    }
}
