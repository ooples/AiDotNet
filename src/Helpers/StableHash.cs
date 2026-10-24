using System.Text;

namespace AiDotNet.Helpers;

/// <summary>
/// A string hash that is the same in every process, for anything that turns text into model input or a seed.
/// </summary>
/// <remarks>
/// <para>
/// <c>string.GetHashCode()</c> is randomized per process on .NET Core and .NET 5+, so a value derived from it
/// changes from one run to the next: a reloaded model looks up different rows, and the same prompt and seed
/// generate different output (#2361). This hash depends only on the text.
/// </para>
/// <para>
/// It is 32-bit FNV-1a over the UTF-8 bytes, each byte sign-extended before it is mixed in, which is exactly
/// fastText's <c>Dictionary::hash</c>. Subword buckets computed with it therefore match the reference
/// implementation's.
/// </para>
/// </remarks>
internal static class StableHash
{
    private const uint OffsetBasis = 2166136261;
    private const uint Prime = 16777619;

    /// <summary>The 32-bit FNV-1a hash of <paramref name="text"/>'s UTF-8 bytes, as fastText computes it.</summary>
    /// <param name="text">The text to hash.</param>
    /// <returns>The hash, unsigned.</returns>
    public static uint Fnv1a(string text)
    {
        if (text is null) throw new ArgumentNullException(nameof(text));
        uint hash = OffsetBasis;
        foreach (byte b in Encoding.UTF8.GetBytes(text))
        {
            hash ^= unchecked((uint)(sbyte)b);
            hash = unchecked(hash * Prime);
        }
        return hash;
    }

    /// <summary>The hash of <paramref name="text"/> as a non-negative <see cref="int"/>, for a seed.</summary>
    /// <param name="text">The text to hash.</param>
    /// <returns>A seed in [0, <see cref="int.MaxValue"/>].</returns>
    public static int Seed(string text) => (int)(Fnv1a(text) & int.MaxValue);

    /// <summary>An index in [0, <paramref name="count"/>) chosen by the hash of <paramref name="text"/>.</summary>
    /// <param name="text">The text to hash.</param>
    /// <param name="count">The number of slots; must be positive.</param>
    /// <returns>The slot index.</returns>
    public static int Index(string text, int count)
    {
        if (count <= 0) throw new ArgumentOutOfRangeException(nameof(count), "The slot count must be positive.");
        return (int)(Fnv1a(text) % (uint)count);
    }
}
