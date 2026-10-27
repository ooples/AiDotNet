using AiDotNet.Interfaces;

namespace AiDotNet.Models.Parameters;

/// <summary>Copies fitted parameters through each model's streaming restore contract.</summary>
internal static class ParameterTransfer
{
    internal static void Copy<T, TInput, TOutput>(
        IParameterizable<T, TInput, TOutput> source, IParameterizable<T, TInput, TOutput> destination)
    {
        if (ReferenceEquals(source, destination)) return;
#if AIDN_LEGACY_TFM
        destination.SetParameters(source.GetParameters());
#else
        // A subclass can override its packed layout or restore hooks without overriding chunks.
        // Such an inherited chunk method must not silently replace the more specific contract.
        if (HasMoreSpecificFlatMethod(source.GetType(), "GetParameters", "GetParameterChunks", Type.EmptyTypes)
            || HasMoreSpecificFlatMethod(destination.GetType(), "SetParameters", "SetParameterChunks", new[] { typeof(Vector<T>) }))
            destination.SetParameters(source.GetParameters());
        else
            destination.SetParameterChunks(source.GetParameterChunks());
#endif
    }
    internal static void CopyLayer<T>(ILayer<T> source, ILayer<T> destination)
    {
        if (ReferenceEquals(source, destination)) return;
        if (source is AiDotNet.NeuralNetworks.Layers.LayerBase<T> from
            && destination is AiDotNet.NeuralNetworks.Layers.LayerBase<T> to
            && from.CanRestoreParameterChunksInPlace() && to.CanRestoreParameterChunksInPlace()
            && to.TryRestoreParameterChunks(from.GetParameterStateChunks("$").Select(x => x.Tensor).ToArray())) return;
        destination.SetParameters(source.GetParameters());
    }

    internal static bool UsesChunkContract<T, TInput, TOutput>(IParameterizable<T, TInput, TOutput> model)
        => !HasMoreSpecificFlatMethod(model.GetType(), "GetParameters", "GetParameterChunks", Type.EmptyTypes)
            && !HasMoreSpecificFlatMethod(model.GetType(), "SetParameters", "SetParameterChunks", new[] { typeof(Vector<T>) });

    internal static Action<IEnumerable<Tensor<T>>>? ChunkRestoreAction<T>(IParameterSource<T> source)
    {
        if (source is not IParameterChunkSource<T>) return null;
        var type = source.GetType();
        var chunks = type.GetMethod("GetParameterChunks", Type.EmptyTypes);
        var state = type.GetMethod("GetParameterStateChunks", Type.EmptyTypes);
        var restore = type.GetMethod("SetParameterChunks", new[] { typeof(IEnumerable<Tensor<T>>) });
        // The registry enumerates role-aware state. Only use a chunk setter when its public
        // read surface has the same owner, and no more-specific flat contract was overridden.
        if (chunks is null || state is null || restore is null || chunks.DeclaringType != state.DeclaringType
            || HasMoreSpecificFlatMethod(type, "GetParameters", "GetParameterChunks", Type.EmptyTypes)
            || HasMoreSpecificFlatMethod(type, "SetParameters", "SetParameterChunks", new[] { typeof(Vector<T>) })) return null;
        return (Action<IEnumerable<Tensor<T>>>)restore.CreateDelegate(typeof(Action<IEnumerable<Tensor<T>>>), source);
    }

    private static bool HasMoreSpecificFlatMethod(Type type, string flatName, string chunkName, Type[] parameters)
    {
        var flatOwner = type.GetMethod(flatName, parameters)?.DeclaringType;
        var chunkOwner = type.GetMethod(chunkName)?.DeclaringType;
        return flatOwner is not null && (chunkOwner is null
            || (flatOwner != chunkOwner && chunkOwner.IsAssignableFrom(flatOwner)));
    }
}
