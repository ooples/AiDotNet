using AiDotNet.Helpers;
using AiDotNet.RetrievalAugmentedGeneration.QueryExpansion;
using Xunit;

namespace AiDotNet.Tests.UnitTests.Helpers;

/// <summary>
/// StableHash replaces string.GetHashCode() wherever text becomes model input, an index or a seed (#2361).
/// string.GetHashCode() is randomized per process, so a value derived from it cannot be pinned by a test at all;
/// these pin the published FNV-1a values, which are the same in every process.
/// </summary>
public class StableHashTests
{
    [Theory]
    [InlineData("", 0x811c9dc5u)]
    [InlineData("a", 0xe40c292cu)]
    [InlineData("foobar", 0xbf9cf968u)]
    public void Fnv1a_MatchesThePublishedVectors(string text, uint expected)
    {
        Assert.Equal(expected, StableHash.Fnv1a(text));
    }

    [Fact]
    public void Fnv1a_SignExtendsNonAsciiBytes_AsFastTextDoes()
    {
        // fastText's Dictionary::hash mixes uint32_t(int8_t(byte)). "é" is UTF-8 C3 A9: both bytes are above 0x7F,
        // so a hash that did not sign-extend them would differ, and its subword buckets would not match fastText's.
        Assert.Equal(0x3cfa68c1u, StableHash.Fnv1a("é"));
    }

    [Fact]
    public void Seed_IsNonNegative()
    {
        for (int i = 0; i < 1000; i++)
            Assert.InRange(StableHash.Seed("prompt " + i), 0, int.MaxValue);
    }

    [Theory]
    [InlineData(1)]
    [InlineData(3)]
    [InlineData(7)]
    public void Index_StaysInRange(int count)
    {
        for (int i = 0; i < 1000; i++)
            Assert.InRange(StableHash.Index("query " + i, count), 0, count - 1);
    }

    [Fact]
    public void MultiQueryExpansion_ExpandsEveryQuery()
    {
        // A negative string.GetHashCode() indexed its phrase and synonym tables out of range, so about half of all
        // queries threw IndexOutOfRangeException.
        var expansion = new MultiQueryExpansion("http://localhost", "key", numVariations: 6);
        for (int i = 0; i < 200; i++)
        {
            // "find" reaches the synonym table too, the second site that indexed by the raw hash.
            var expanded = expansion.ExpandQuery("find a fast model number " + i);
            Assert.NotEmpty(expanded);
        }
    }
}
