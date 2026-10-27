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
    private static bool HasMoreSpecificFlatMethod(Type type, string flatName, string chunkName, Type[] parameters)
    {
        var flatOwner = type.GetMethod(flatName, parameters)?.DeclaringType;
        var chunkOwner = type.GetMethod(chunkName)?.DeclaringType;
        return flatOwner is not null && (chunkOwner is null
            || (flatOwner != chunkOwner && chunkOwner.IsAssignableFrom(flatOwner)));
    }
}
