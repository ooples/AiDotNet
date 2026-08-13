using System.Collections;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.NeuralNetworks;
using AiDotNet.Training;

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
internal static class CopyOnWriteCloneHelper
{
    /// <summary>
    /// Re-binds every trainable parameter of <paramref name="dest"/> to a copy-on-write share of the
    /// corresponding parameter of <paramref name="source"/>. Walks both object graphs in parallel by
    /// reflection (identical runtime type ⇒ identical field order ⇒ matching layer order). Returns
    /// <c>false</c> — leaving <paramref name="dest"/> untouched — if the trainable-layer structure does
    /// not line up 1:1 (e.g. a freshly-constructed clone whose lazy layers aren't resolved yet), so the
    /// caller can fall back to the eager flat copy.
    /// </summary>
    internal static bool TryShareTrainableParameters<T>(
        IFullModel<T, Tensor<T>, Tensor<T>>? source,
        IFullModel<T, Tensor<T>, Tensor<T>>? dest)
    {
        if (source is null || dest is null || ReferenceEquals(source, dest)) return false;
        if (source.GetType() != dest.GetType()) return false;

        var srcLayers = CollectTrainableLayers<T>(source);
        var dstLayers = CollectTrainableLayers<T>(dest);
        if (srcLayers.Count == 0 || srcLayers.Count != dstLayers.Count) return false;

        // Verify the full structure — per-layer parameter COUNT and per-tensor SHAPE — matches BEFORE
        // mutating anything, so we never leave a half-shared clone and never rebind a shape-incompatible
        // tensor. A count-only check would let a same-count but differently-shaped graph (e.g. a custom
        // source whose predictor/VAE was built with different channel widths than the clone's defaults)
        // pass, share, and silently corrupt the clone; a shape mismatch must instead fall back to the
        // eager copy. A freshly-constructed clone whose lazy layers aren't resolved yet shows up here as
        // a zero-/mismatched-shape tensor and also falls back.
        for (int i = 0; i < srcLayers.Count; i++)
        {
            var sps = GetWithoutMaterialization(srcLayers[i]);
            var dps = GetWithoutMaterialization(dstLayers[i]);
            if (sps.Count != dps.Count) return false;
            for (int p = 0; p < sps.Count; p++)
                if (!ShapesEqual(sps[p], dps[p])) return false;
        }

        for (int i = 0; i < srcLayers.Count; i++)
        {
            var sp = GetWithoutMaterialization(srcLayers[i]);
            if (sp.Count == 0) continue;
            var shared = new Tensor<T>[sp.Count];
            for (int p = 0; p < sp.Count; p++)
                shared[p] = (Tensor<T>)sp[p].CloneShared();
            dstLayers[i].SetTrainableParameters(shared);
        }

        return true;
    }

    private static IReadOnlyList<Tensor<T>> GetWithoutMaterialization<T>(ITrainableLayer<T> layer) =>
        layer is AiDotNet.NeuralNetworks.Layers.LayerBase<T> layerBase
            ? layerBase.GetTrainableParametersWithoutMaterialization()
            : layer.GetTrainableParameters();

    private static bool ShapesEqual<T>(Tensor<T> a, Tensor<T> b)
    {
        var sa = a.Shape;
        var sb = b.Shape;
        if (sa.Length != sb.Length) return false;
        for (int i = 0; i < sa.Length; i++)
            if (sa[i] != sb[i]) return false;
        return true;
    }

    /// <summary>
    /// Collects every <see cref="ITrainableLayer{T}"/> reachable from <paramref name="root"/> by reflection,
    /// in a deterministic order. Captures layers held both in a base <c>_layers</c> list AND in dedicated
    /// fields (e.g. a tabular transformer's feature tokenizer / encoder stack / final layer-norm), which a
    /// <c>_layers</c>-only walk misses. Two instances of the same runtime type yield matching order, so the
    /// result pairs 1:1 between a model and its fresh clone.
    /// </summary>
    internal static List<ITrainableLayer<T>> CollectTrainableLayers<T>(IFullModel<T, Tensor<T>, Tensor<T>> root)
    {
        // NeuralNetworkBase owns an explicit module graph: top-level Layers plus
        // each LayerBase's registered sub-layers. Walk that graph directly, just
        // as training does. Reflection over only the concrete model type cannot
        // see NeuralNetworkBase's private _layers field; that made ordinary
        // sequential models appear empty and forced the lossy eager fallback.
        // More importantly, a reflective object-graph walk also wandered through
        // optimizers/options/caches, which are not modules and can differ between
        // a trained source and a fresh destination. The registered layer graph is
        // the stable, PyTorch-style ownership boundary for cloning.
        if (root is NeuralNetworkBase<T> neuralNetwork)
        {
            // `Layers` IS the registered graph, and it is what NeuralNetworkBase itself passes to this
            // same walk. This used to call a GetCopyOnWriteLayerRoots() that exists nowhere in the
            // repository -- a call that survived review because NeuralNetworkBase is an error type
            // while the #1789 split is mid-flight (its declaration depends on types slice 01 has not
            // landed yet), and Roslyn suppresses member lookup on an error type to avoid cascading
            // diagnostics. So the compiler could not report it and a search could not find it; it
            // would have failed the moment the branch built cleanly.
            //
            // structureVersion -1 keeps the caching disabled: a clone walks a graph the version
            // counter has never seen, so a cached answer would describe the wrong model.
            return new List<ITrainableLayer<T>>(
                TapeTrainingStep<T>.CollectTrainableLayers(
                    neuralNetwork.Layers,
                    structureVersion: -1));
        }

        var layers = new List<ITrainableLayer<T>>();
        // CollectInto walks arbitrary instance fields, so it is necessarily typed `object?` internally;
        // the public entry point constrains the root to a model so callers can't pass an unrelated graph.
        try
        {
            CollectInto(root, layers, new HashSet<object>(TensorReferenceComparer<object>.Instance), 0);
        }
        catch (CollectDepthExceededException)
        {
            // Absolute backstop: a pathologically deep/cyclic object graph that the visited-set could
            // not bound. Returning an empty list makes TryShareTrainableParameters fall back to the
            // eager flat copy (always correct, just not COW-shared) instead of risking a host stack
            // overflow. With the leaf-skips below this is unreachable for every real model graph — so a
            // trip is a genuine regression signal (a new self-regenerating field type), NOT a routine
            // event. Surface it via Trace (observable in Release; the codebase's existing diagnostic
            // idiom) rather than swallowing it silently, otherwise the COW-disabling fallback would
            // quietly reintroduce the large-Clone() OOM pressure (#1624) with zero observability.
            System.Diagnostics.Trace.TraceWarning(
                $"CopyOnWriteCloneHelper: trainable-layer walk for '{root.GetType().FullName}' exceeded " +
                $"MaxWalkDepth ({MaxWalkDepth}) and fell back to the eager Clone copy. This is expected to " +
                "be unreachable — investigate a newly-introduced self-regenerating field type in the model " +
                "object graph (the #1669 failure class).");
            layers.Clear();
        }
        return layers;
    }

    /// <summary>
    /// Hard ceiling on the reflective walk's recursion depth. Real model object graphs nest only a few
    /// dozen levels, so this is a conservative ~5–10× margin above any legitimate depth: it never trips
    /// for a real model, yet caps frames well short of a host stack overflow. The guard exists purely as
    /// a fail-safe so a previously-unseen self-regenerating field type can never stack-overflow the test
    /// host (the failure mode behind #1669) before the pointer/leaf skips below intercept it.
    /// </summary>
    private const int MaxWalkDepth = 256;

    private sealed class CollectDepthExceededException : System.Exception { }

    private static void CollectInto<T>(object? obj, List<ITrainableLayer<T>> layers, HashSet<object> visited, int depth)
    {
        if (obj is null || !visited.Add(obj)) return;
        if (depth > MaxWalkDepth) throw new CollectDepthExceededException();
        if (obj is ITrainableLayer<T> trainable) layers.Add(trainable);

        var type = obj.GetType();
        if (IsLeafType<T>(type)) return;

        // Walk the FULL inheritance chain with DeclaredOnly. Type.GetFields(Instance|NonPublic)
        // does NOT return a base class's PRIVATE fields, so a concrete-type-only enumeration could
        // never see NeuralNetworkBase<T>._layers (private readonly) or LayerBase<T>._registeredTensors
        // — making the walk return 0 layers for every model that keeps its layers in the base list
        // (i.e. most models) and silently disabling the COW clone fast path, so every Clone() fell
        // back to a full serialize/deserialize round-trip. DeclaredOnly also means each FieldInfo is
        // yielded exactly ONCE across the chain (no inherited-public duplicates, and a derived field
        // that shadows a base field is a distinct FieldInfo still visited once), so no double-visiting
        // is introduced. Order is derived-first then base — deterministic for a given runtime type,
        // which is what the src/dst pairing and the caller's parameter-coverage guard rely on.
        for (var t = type; t is not null && t != typeof(object); t = t.BaseType)
        {
        foreach (var field in t.GetFields(
            BindingFlags.Instance | BindingFlags.NonPublic | BindingFlags.Public | BindingFlags.DeclaredOnly))
        {
            // Tensor fields are leaves (their owning layer already exposed them via
            // GetTrainableParameters); skip primitives/strings/enums that can't hold a layer.
            if (IsLeafType<T>(field.FieldType))
                continue;

            // Skip unmanaged pointer fields (void*, byte*, T*, ...). Reflecting a pointer field via
            // FieldInfo.GetValue boxes it into a FRESH System.Reflection.Pointer object on EVERY call,
            // and that wrapper's own `_ptr` field is itself a pointer — so following it spirals into
            // unbounded recursion (a brand-new Pointer per level, never deduped by the visited set),
            // stack-overflowing the host. A pointer can never reference an ITrainableLayer, so it is
            // always a leaf. This was the single shared cycle behind #1669's NN/Generated shard host
            // crashes: any model whose graph transitively reached a native pointer (e.g. the
            // foundation-scale weight-streaming / mmap path) hit it.
            if (field.FieldType.IsPointer)
                continue;

            var val = field.GetValue(obj);
            if (val is null) continue;

            // Defensive companion to the IsPointer field-skip above: a pointer can also surface through
            // an object-/interface-typed field, where the declared FieldType is not IsPointer but the
            // runtime value is a System.Reflection.Pointer. Treat it as a leaf too so the same
            // wrap-per-call spiral cannot recur via that path.
            if (val is System.Reflection.Pointer) continue;

            if (val is IEnumerable enumerable && val is not string)
            {
                // Only enumerate sequences whose ELEMENTS could be a layer. Tensor<T>, Vector<T>,
                // Matrix<T> and T[]/int[]/List<double> are all IEnumerable over primitives: iterating
                // them boxes every scalar into the visited set (millions of allocations for a real
                // weight tensor). Critical now that the BaseType walk above reaches
                // LayerBase<T>._registeredTensors (List<Tensor<T>>) and every weight buffer behind it.
                if (!CanHoldLayers<T>(val.GetType())) continue;
                foreach (var item in enumerable)
                    CollectInto(item, layers, visited, depth + 1);
            }
            else
            {
                CollectInto(val, layers, visited, depth + 1);
            }
        }
        }
    }

    /// <summary>
    /// Types that can never transitively hold an <see cref="ITrainableLayer{T}"/>, so the walk stops.
    /// Includes the whole AiDotNet.Tensors assembly (Tensor/Vector/Matrix/ParameterBuffer/engines):
    /// that assembly does not reference AiDotNet, so no layer type can live inside one of its objects.
    /// Being wrong here is fail-safe: a missed layer only makes the caller fall back to the eager copy.
    /// </summary>
    private static bool IsLeafType<T>(Type t) =>
        t.IsPrimitive || t.IsEnum || t.IsPointer
        || t == typeof(string) || t == typeof(decimal)
        || t == typeof(IntPtr) || t == typeof(UIntPtr)
        || t.Assembly == typeof(Tensor<T>).Assembly;

    /// <summary>
    /// True when a sequence's element type could contain a trainable layer. Unknown (non-generic
    /// IEnumerable) is treated as walkable so nothing is lost silently.
    /// </summary>
    private static bool CanHoldLayers<T>(Type sequenceType)
    {
        if (sequenceType.IsArray)
        {
            var el = sequenceType.GetElementType();
            return el is null || !IsLeafType<T>(el);
        }

        bool sawGeneric = false;
        foreach (var iface in sequenceType.GetInterfaces()
                     .Where(i => i.IsGenericType && i.GetGenericTypeDefinition() == typeof(IEnumerable<>)))
        {
            sawGeneric = true;
            if (!IsLeafType<T>(iface.GetGenericArguments()[0])) return true;
        }
        return !sawGeneric;
    }
}
