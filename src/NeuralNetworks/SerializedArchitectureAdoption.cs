namespace AiDotNet.NeuralNetworks;

/// <summary>
/// Scopes when a network payload's architecture descriptor (format v10) is restored into the instance it loads into.
/// </summary>
/// <remarks>
/// <para>
/// Only a skeleton built from a type's defaults -- the instance <c>AiModelResult.Deserialize</c> restores into --
/// should adopt the saved architecture. A network a caller constructed has an architecture of its own that its
/// subclass may already depend on (a code model builds its tokenizer for its target language) and may validate
/// against the payload in <c>DeserializeNetworkSpecificData</c>; overwriting it in place would hide exactly the
/// mismatch that validation exists to report. Outside this scope the descriptor is read and discarded, so plain
/// <c>Deserialize</c> behaves as it did before v10.
/// </para>
/// </remarks>
internal static class SerializedArchitectureAdoption
{
    [ThreadStatic] private static int t_depth;

    /// <summary>True while the current thread is restoring into a skeleton that adopts the saved architecture.</summary>
    internal static bool IsActive => t_depth > 0;

    /// <summary>Restores saved architectures into the networks deserialized on this thread until disposed.</summary>
    internal static Scope Begin()
    {
        t_depth++;
        return new Scope();
    }

    internal readonly struct Scope : IDisposable
    {
        public void Dispose() => t_depth--;
    }
}
