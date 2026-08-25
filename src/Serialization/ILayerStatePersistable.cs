namespace AiDotNet.Serialization;

/// <summary>
/// Opt-in contract letting ANY type be carried as layer construction state, without the generator
/// having to learn about it.
/// </summary>
/// <remarks>
/// <para>
/// Every state type the generator understands today -- int, double[], bool[], string[], enums,
/// layer collections, expression trees -- required a ValueKind, a Classify case, emission sites and
/// a runtime reader. That is a per-type mechanism in the generator, and a layer author with a novel
/// type could do nothing but wait for the generator to be patched. This contract replaces N of those
/// with one: Classify asks a single question, and a type that answers it works with no generator
/// change at all.
/// </para>
/// <para>
/// INSTANCE methods, not C# 11 static abstract members: this assembly still targets net471, where
/// static abstract interface members do not exist. <see cref="LoadState"/> is therefore called on a
/// freshly constructed instance, so an implementing type needs a public parameterless constructor.
/// The generator verifies that at BUILD time and reports it, rather than failing when a model is
/// restored.
/// </para>
/// <para>
/// Implementations must round-trip exactly: <c>LoadState(SaveState())</c> has to reproduce every
/// value the layer will use. A partial implementation is worse than none, because the clone then
/// looks correct and computes something else.
/// </para>
/// </remarks>
public interface ILayerStatePersistable
{
    /// <summary>Writes this value as text the same type can read back.</summary>
    string SaveState();

    /// <summary>Populates a freshly constructed instance from <paramref name="text"/>.</summary>
    /// <param name="text">Text previously produced by <see cref="SaveState"/>.</param>
    void LoadState(string text);
}
