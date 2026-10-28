using AiDotNet.Interfaces;

namespace AiDotNet.Models.Parameters;

/// <summary>Shared chunk restore policy for independent model base classes.</summary>
internal static class ParameterChunkRestore<T>
{
    /// <summary>Reads live registered chunks only when they describe the owner's ordinary parameter contract.</summary>
    internal static IEnumerable<ParameterChunk<T>> Read(object owner, Type baseType,
        ParameterComponentRegistry<T> registry, Func<Vector<T>> getParameters)
    {
        var getter = owner.GetType().GetMethod(nameof(IParameterSource<T>.GetParameters), Type.EmptyTypes);
        var state = owner.GetType().GetMethod(nameof(IParameterChunkSource<T>.GetParameterStateChunks), Type.EmptyTypes);
        // An explicit state override at or below the getter's owner opts into the registry's
        // live contract (for example, vision bases that materialize lazy layers then call base).
        // A more-derived flat-only override still requires a detached compatibility snapshot.
        var explicitStateContract = getter?.DeclaringType is Type getterOwner
            && state?.DeclaringType is Type stateOwner && stateOwner != baseType
            && getterOwner.IsAssignableFrom(stateOwner);
        if (registry.HasComponents && (getter?.DeclaringType == baseType || explicitStateContract))
            return registry.GetParameterStateChunks();
        var flat = getParameters();
        return flat.Length == 0 ? Array.Empty<ParameterChunk<T>>()
            : new[] { new ParameterChunk<T>("$", ParameterSlotRole.Trainable,
                new Tensor<T>(new[] { flat.Length }, flat), null, writableInPlace: false) };
    }

    /// <summary>Preserves custom setters and restore hooks while avoiding host copies for canonical dense chunks.</summary>
    internal static void Restore(object owner, Type baseType, IEnumerable<Tensor<T>> chunks,
        ParameterComponentRegistry<T> registry, Action<Vector<T>> setParameters, Action onRestored, Action? onRestoring = null)
    {
        if (chunks is null) throw new ArgumentNullException(nameof(chunks));
        var buffered = chunks.ToArray();
        if (buffered.Any(t => t is null))
            throw new ArgumentException("Chunk sequence contains a null tensor.", nameof(chunks));
        // A derived setter may rebuild caches, unpack a custom layout, or enforce constraints.
        // The base's fast path must not bypass that contract.
        var setter = owner.GetType().GetMethod(nameof(IParameterSource<T>.SetParameters), new[] { typeof(Vector<T>) });
        var getter = owner.GetType().GetMethod(nameof(IParameterSource<T>.GetParameters), Type.EmptyTypes);
        if (setter?.DeclaringType == baseType && getter?.DeclaringType == baseType
            && registry.TrySetParameterChunks(buffered, onRestoring))
        {
            onRestored();
            return;
        }
        setParameters(AiDotNet.Models.Parameters.ParameterChunkBuffer.ToVector(buffered));
    }
}
