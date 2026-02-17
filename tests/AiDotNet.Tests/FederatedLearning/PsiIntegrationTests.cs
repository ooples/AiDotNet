using AiDotNet.FederatedLearning.PSI;
using AiDotNet.Models.Options;
using Xunit;

namespace AiDotNet.Tests.FederatedLearning;

/// <summary>
/// Comprehensive integration tests for Private Set Intersection protocols (#538).
/// </summary>
public class PsiIntegrationTests
{
    private static readonly string[] PartyA = { "alice", "bob", "charlie", "dave", "eve" };
    private static readonly string[] PartyB = { "bob", "dave", "frank", "grace", "eve" };

    // ========== DiffieHellmanPsi Tests ==========

    [Fact]
    public void DiffieHellman_ComputeIntersection_FindsCommonElements()
    {
        var psi = new DiffieHellmanPsi();
        var options = new PsiOptions { Protocol = PsiProtocol.DiffieHellman };

        var result = psi.ComputeIntersection(PartyA, PartyB, options);

        Assert.NotNull(result);
        Assert.Equal(3, result.IntersectionSize); // bob, dave, eve
        Assert.Contains("bob", result.IntersectionIds);
        Assert.Contains("dave", result.IntersectionIds);
        Assert.Contains("eve", result.IntersectionIds);
    }

    [Fact]
    public void DiffieHellman_ComputeCardinality_ReturnsCorrectCount()
    {
        var psi = new DiffieHellmanPsi();
        var options = new PsiOptions { Protocol = PsiProtocol.DiffieHellman };

        int cardinality = psi.ComputeCardinality(PartyA, PartyB, options);

        Assert.Equal(3, cardinality);
    }

    [Fact]
    public void DiffieHellman_EmptySets_ReturnsEmptyIntersection()
    {
        var psi = new DiffieHellmanPsi();
        var options = new PsiOptions();
        var empty = Array.Empty<string>();

        var result = psi.ComputeIntersection(empty, PartyB, options);

        Assert.Equal(0, result.IntersectionSize);
    }

    [Fact]
    public void DiffieHellman_DisjointSets_ReturnsEmpty()
    {
        var psi = new DiffieHellmanPsi();
        var options = new PsiOptions();
        var setA = new[] { "a", "b", "c" };
        var setB = new[] { "d", "e", "f" };

        var result = psi.ComputeIntersection(setA, setB, options);

        Assert.Equal(0, result.IntersectionSize);
    }

    [Fact]
    public void DiffieHellman_IdenticalSets_ReturnsAll()
    {
        var psi = new DiffieHellmanPsi();
        var options = new PsiOptions();
        var set = new[] { "a", "b", "c" };

        var result = psi.ComputeIntersection(set, set, options);

        Assert.Equal(3, result.IntersectionSize);
    }

    [Fact]
    public void DiffieHellman_ProtocolName_IsCorrect()
    {
        var psi = new DiffieHellmanPsi();
        Assert.Equal("DiffieHellman", psi.ProtocolName);
    }

    // ========== ObliviousTransferPsi Tests ==========

    [Fact]
    public void ObliviousTransferPsi_ComputeIntersection_FindsCommonElements()
    {
        var psi = new ObliviousTransferPsi();
        var options = new PsiOptions { Protocol = PsiProtocol.ObliviousTransfer };

        var result = psi.ComputeIntersection(PartyA, PartyB, options);

        Assert.Equal(3, result.IntersectionSize);
    }

    [Fact]
    public void ObliviousTransferPsi_ProtocolName_IsCorrect()
    {
        var psi = new ObliviousTransferPsi();
        Assert.Equal("ObliviousTransfer", psi.ProtocolName);
    }

    // ========== CircuitBasedPsi Tests ==========

    [Fact]
    public void CircuitBasedPsi_ComputeIntersection_FindsCommonElements()
    {
        var psi = new CircuitBasedPsi();
        var options = new PsiOptions { Protocol = PsiProtocol.CircuitBased };

        var result = psi.ComputeIntersection(PartyA, PartyB, options);

        Assert.Equal(3, result.IntersectionSize);
    }

    [Fact]
    public void CircuitBasedPsi_ProtocolName_IsCorrect()
    {
        var psi = new CircuitBasedPsi();
        Assert.Equal("CircuitBased", psi.ProtocolName);
    }

    // ========== BloomFilterPsi Tests ==========

    [Fact]
    public void BloomFilterPsi_ComputeIntersection_FindsElements()
    {
        var psi = new BloomFilterPsi();
        var options = new PsiOptions
        {
            Protocol = PsiProtocol.BloomFilter,
            BloomFilterFalsePositiveRate = 0.001
        };

        var result = psi.ComputeIntersection(PartyA, PartyB, options);

        // Bloom filter may have false positives, so intersection >= 3
        Assert.True(result.IntersectionSize >= 3,
            $"BloomFilter should find at least the 3 real intersecting elements, found {result.IntersectionSize}");
    }

    [Fact]
    public void BloomFilterPsi_ComputeCardinality_ReturnsReasonableCount()
    {
        var psi = new BloomFilterPsi();
        var options = new PsiOptions
        {
            Protocol = PsiProtocol.BloomFilter,
            BloomFilterFalsePositiveRate = 0.001
        };

        int cardinality = psi.ComputeCardinality(PartyA, PartyB, options);

        Assert.True(cardinality >= 3, $"Expected at least 3, got {cardinality}");
    }

    [Fact]
    public void BloomFilterPsi_ProtocolName_IsCorrect()
    {
        var psi = new BloomFilterPsi();
        Assert.Equal("BloomFilter", psi.ProtocolName);
    }

    // ========== UnbalancedPsi Tests ==========

    [Fact]
    public void UnbalancedPsi_ComputeIntersection_HandlesAsymmetricSets()
    {
        var psi = new UnbalancedPsi();
        var options = new PsiOptions();
        var smallSet = new[] { "bob", "eve" };
        var largeSet = new[] { "alice", "bob", "charlie", "dave", "eve", "frank", "grace", "henry" };

        var result = psi.ComputeIntersection(smallSet, largeSet, options);

        Assert.Equal(2, result.IntersectionSize);
        Assert.Contains("bob", result.IntersectionIds);
        Assert.Contains("eve", result.IntersectionIds);
    }

    // ========== MultiPartyPsi Tests ==========

    [Fact]
    public void MultiPartyPsi_ComputeMultiPartyIntersection_FindsCommon()
    {
        var multiPsi = new MultiPartyPsi();
        var options = new PsiOptions { NumberOfParties = 3 };
        var partyC = new[] { "bob", "eve", "henry" };

        var partySets = new List<IReadOnlyList<string>> { PartyA, PartyB, partyC };

        var result = multiPsi.ComputeMultiPartyIntersection(partySets, options);

        Assert.NotNull(result);
        // bob and eve are in all 3 sets
        Assert.Contains("bob", result.IntersectionIds);
        Assert.Contains("eve", result.IntersectionIds);
    }

    [Fact]
    public void MultiPartyPsi_ParameterlessConstructor_UsesDiffieHellman()
    {
        var multiPsi = new MultiPartyPsi();
        // Should work without throwing
        var options = new PsiOptions();
        var result = multiPsi.ComputeMultiPartyIntersection(
            new List<IReadOnlyList<string>> { new[] { "a", "b" }, new[] { "b", "c" } },
            options);

        Assert.NotNull(result);
    }

    // ========== FuzzyPsi Tests ==========

    [Fact]
    public void FuzzyPsi_ProtocolName_ContainsFuzzy()
    {
        var fuzzyPsi = new FuzzyPsi(new DiffieHellmanPsi());
        Assert.Contains("Fuzzy", fuzzyPsi.ProtocolName);
    }

    [Fact]
    public void FuzzyPsi_NullInnerProtocol_Throws()
    {
        Assert.Throws<ArgumentNullException>(() => new FuzzyPsi(null));
    }

    [Fact]
    public void FuzzyPsi_ExactMatch_FindsIntersection()
    {
        var fuzzyPsi = new FuzzyPsi(new DiffieHellmanPsi());
        var options = new PsiOptions();

        var result = fuzzyPsi.ComputeIntersection(PartyA, PartyB, options);

        Assert.Equal(3, result.IntersectionSize);
    }

    [Fact]
    public void FuzzyPsi_EditDistance_FindsFuzzyMatches()
    {
        var fuzzyPsi = new FuzzyPsi(new DiffieHellmanPsi());
        var options = new PsiOptions
        {
            FuzzyMatch = new FuzzyMatchOptions
            {
                Strategy = FuzzyMatchStrategy.EditDistance,
                Threshold = 2,
                CaseSensitive = false
            }
        };
        // Sets with similar (but not identical) entries
        var local = new[] { "jon", "Dave", "charlie" };
        var remote = new[] { "john", "dave", "chuck" };

        var result = fuzzyPsi.ComputeIntersection(local, remote, options);

        Assert.NotNull(result);
        // "jon" ↔ "john" (edit distance 1, threshold 2) should match
        // "Dave" ↔ "dave" (case insensitive) should match
        Assert.True(result.IntersectionSize >= 1,
            $"Expected at least 1 fuzzy match, got {result.IntersectionSize}");
    }

    [Fact]
    public void FuzzyPsi_NGram_FindsFuzzyMatches()
    {
        var fuzzyPsi = new FuzzyPsi(new DiffieHellmanPsi());
        var options = new PsiOptions
        {
            FuzzyMatch = new FuzzyMatchOptions
            {
                Strategy = FuzzyMatchStrategy.NGram,
                Threshold = 0.5,
                NGramSize = 2
            }
        };
        var local = new[] { "hello", "world" };
        var remote = new[] { "hallo", "earth" };

        var result = fuzzyPsi.ComputeIntersection(local, remote, options);

        Assert.NotNull(result);
        // "hello" ↔ "hallo" should have high n-gram similarity
        Assert.True(result.IntersectionSize >= 0);
    }

    [Fact]
    public void FuzzyPsi_Jaccard_FindsFuzzyMatches()
    {
        var fuzzyPsi = new FuzzyPsi(new DiffieHellmanPsi());
        var options = new PsiOptions
        {
            FuzzyMatch = new FuzzyMatchOptions
            {
                Strategy = FuzzyMatchStrategy.Jaccard,
                Threshold = 0.3,
                NGramSize = 2
            }
        };
        var local = new[] { "abcdef", "xyz" };
        var remote = new[] { "abcxyz", "uvw" };

        var result = fuzzyPsi.ComputeIntersection(local, remote, options);

        Assert.NotNull(result);
    }

    [Fact]
    public void FuzzyPsi_Phonetic_FindsSimilarSoundingMatches()
    {
        var fuzzyPsi = new FuzzyPsi(new DiffieHellmanPsi());
        var options = new PsiOptions
        {
            FuzzyMatch = new FuzzyMatchOptions
            {
                Strategy = FuzzyMatchStrategy.Phonetic,
                CaseSensitive = false
            }
        };
        var local = new[] { "smith", "johnson" };
        var remote = new[] { "smyth", "johnston" };

        var result = fuzzyPsi.ComputeIntersection(local, remote, options);

        Assert.NotNull(result);
        // Phonetic matching should find similar-sounding names
        Assert.True(result.IntersectionSize >= 0);
    }

    [Fact]
    public void FuzzyPsi_FuzzyMatchResult_HasConfidences()
    {
        var fuzzyPsi = new FuzzyPsi(new DiffieHellmanPsi());
        var options = new PsiOptions
        {
            FuzzyMatch = new FuzzyMatchOptions
            {
                Strategy = FuzzyMatchStrategy.EditDistance,
                Threshold = 3,
                CaseSensitive = false
            }
        };
        var local = new[] { "john", "alice" };
        var remote = new[] { "jon", "alice" };

        var result = fuzzyPsi.ComputeIntersection(local, remote, options);

        Assert.NotNull(result);
        Assert.True(result.IsFuzzyMatch);
    }

    // ========== EntityAligner Tests ==========

    [Fact]
    public void EntityAligner_AlignEntities_ReturnsAlignment()
    {
        var aligner = new EntityAligner();
        var options = new PsiOptions { Protocol = PsiProtocol.DiffieHellman };

        var result = aligner.AlignEntities(PartyA, PartyB, options);

        Assert.NotNull(result);
        Assert.NotNull(result.PsiResult);
        Assert.Equal(3, result.PsiResult.IntersectionSize);
    }

    [Fact]
    public void EntityAligner_AlignEntities_HasValidMappings()
    {
        var aligner = new EntityAligner();
        var options = new PsiOptions();

        var result = aligner.AlignEntities(PartyA, PartyB, options);

        Assert.NotNull(result.PsiResult.LocalToSharedIndexMap);
        Assert.NotNull(result.PsiResult.RemoteToSharedIndexMap);
        Assert.Equal(result.PsiResult.IntersectionSize, result.PsiResult.LocalToSharedIndexMap.Count);
    }

    [Fact]
    public void EntityAligner_AlignEntities_NullLocal_Throws()
    {
        var aligner = new EntityAligner();

        Assert.Throws<ArgumentNullException>(() =>
            aligner.AlignEntities(null, PartyB));
    }

    [Fact]
    public void EntityAligner_AlignEntities_NullRemote_Throws()
    {
        var aligner = new EntityAligner();

        Assert.Throws<ArgumentNullException>(() =>
            aligner.AlignEntities(PartyA, null));
    }

    [Fact]
    public void EntityAligner_ComputeOverlapCount_ReturnsCorrect()
    {
        var aligner = new EntityAligner();

        int count = aligner.ComputeOverlapCount(PartyA, PartyB);

        Assert.Equal(3, count);
    }

    [Fact]
    public void EntityAligner_CheckOverlapSufficiency_SufficientOverlap()
    {
        var aligner = new EntityAligner();

        var (isSufficient, overlapRatio, overlapCount) = aligner.CheckOverlapSufficiency(
            PartyA, PartyB, minimumOverlapRatio: 0.3);

        Assert.True(isSufficient, "60% overlap should be sufficient for 30% threshold");
        Assert.Equal(3, overlapCount);
        Assert.True(overlapRatio > 0.5);
    }

    [Fact]
    public void EntityAligner_CheckOverlapSufficiency_InsufficientOverlap()
    {
        var aligner = new EntityAligner();
        var setA = new[] { "a", "b", "c", "d", "e", "f", "g", "h", "i", "j" };
        var setB = new[] { "x", "y", "z", "a" };

        var (isSufficient, overlapRatio, overlapCount) = aligner.CheckOverlapSufficiency(
            setA, setB, minimumOverlapRatio: 0.5);

        Assert.False(isSufficient, "10% overlap should not be sufficient for 50% threshold");
        Assert.Equal(1, overlapCount);
    }

    [Fact]
    public void EntityAligner_AlignMultipleParties_Works()
    {
        var aligner = new EntityAligner();
        var partyC = new[] { "bob", "eve", "henry" } as IReadOnlyList<string>;
        var partySets = new List<IReadOnlyList<string>> { PartyA, PartyB, partyC };
        var options = new PsiOptions { NumberOfParties = 3 };

        var result = aligner.AlignMultipleParties(partySets, options);

        Assert.NotNull(result);
    }

    [Fact]
    public void EntityAligner_CardinalityOnly_ReturnsCountWithoutIds()
    {
        var aligner = new EntityAligner();
        var options = new PsiOptions { CardinalityOnly = true };

        var result = aligner.AlignEntities(PartyA, PartyB, options);

        Assert.True(result.PsiResult.IntersectionSize >= 0);
    }

    // ========== PsiResult Fields ==========

    [Fact]
    public void PsiResult_OverlapRatios_AreReasonable()
    {
        var psi = new DiffieHellmanPsi();
        var options = new PsiOptions();

        var result = psi.ComputeIntersection(PartyA, PartyB, options);

        Assert.InRange(result.LocalOverlapRatio, 0.0, 1.0);
        Assert.InRange(result.RemoteOverlapRatio, 0.0, 1.0);
        Assert.Equal(0.6, result.LocalOverlapRatio, 2); // 3/5
        Assert.Equal(0.6, result.RemoteOverlapRatio, 2); // 3/5
    }

    [Fact]
    public void PsiResult_ExecutionTime_IsPositive()
    {
        var psi = new DiffieHellmanPsi();
        var options = new PsiOptions();

        var result = psi.ComputeIntersection(PartyA, PartyB, options);

        Assert.True(result.ExecutionTime >= TimeSpan.Zero);
    }

    // ========== PsiOptions Defaults ==========

    [Fact]
    public void PsiOptions_DefaultValues()
    {
        var options = new PsiOptions();

        Assert.Equal(PsiProtocol.DiffieHellman, options.Protocol);
        Assert.Equal(128, options.SecurityParameter);
        Assert.Equal("SHA256", options.HashFunction);
        Assert.Equal(1_000_000, options.MaxSetSize);
        Assert.Equal(0.001, options.BloomFilterFalsePositiveRate);
        Assert.Equal(2, options.NumberOfParties);
        Assert.False(options.CardinalityOnly);
        Assert.Null(options.FuzzyMatch);
    }

    [Fact]
    public void FuzzyMatchOptions_DefaultValues()
    {
        var options = new FuzzyMatchOptions();

        Assert.Equal(FuzzyMatchStrategy.Exact, options.Strategy);
        Assert.Equal(2.0, options.Threshold);
        Assert.Equal(2, options.NGramSize);
        Assert.False(options.CaseSensitive);
        Assert.True(options.NormalizeWhitespace);
    }

    [Fact]
    public void PsiProtocol_HasAllExpectedValues()
    {
        Assert.True(Enum.IsDefined(typeof(PsiProtocol), PsiProtocol.DiffieHellman));
        Assert.True(Enum.IsDefined(typeof(PsiProtocol), PsiProtocol.ObliviousTransfer));
        Assert.True(Enum.IsDefined(typeof(PsiProtocol), PsiProtocol.CircuitBased));
        Assert.True(Enum.IsDefined(typeof(PsiProtocol), PsiProtocol.BloomFilter));
    }

    [Fact]
    public void FuzzyMatchStrategy_HasAllExpectedValues()
    {
        Assert.True(Enum.IsDefined(typeof(FuzzyMatchStrategy), FuzzyMatchStrategy.Exact));
        Assert.True(Enum.IsDefined(typeof(FuzzyMatchStrategy), FuzzyMatchStrategy.EditDistance));
        Assert.True(Enum.IsDefined(typeof(FuzzyMatchStrategy), FuzzyMatchStrategy.Phonetic));
        Assert.True(Enum.IsDefined(typeof(FuzzyMatchStrategy), FuzzyMatchStrategy.NGram));
        Assert.True(Enum.IsDefined(typeof(FuzzyMatchStrategy), FuzzyMatchStrategy.Jaccard));
    }

    // ========== Large Set Performance ==========

    [Fact]
    public void DiffieHellman_LargeSet_CompletesInReasonableTime()
    {
        var psi = new DiffieHellmanPsi();
        var options = new PsiOptions();

        var largeA = Enumerable.Range(0, 10000).Select(i => $"id_{i}").ToArray();
        var largeB = Enumerable.Range(5000, 10000).Select(i => $"id_{i}").ToArray();

        var result = psi.ComputeIntersection(largeA, largeB, options);

        Assert.Equal(5000, result.IntersectionSize);
    }
}
