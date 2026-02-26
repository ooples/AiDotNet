using System.Security.Cryptography;
using System.Text;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.DataVersionControl;

/// <summary>
/// Deep integration tests for DataVersionControl:
/// Hash-based integrity (SHA-256 properties, collision resistance, determinism),
/// Version numbering math, comparison metrics (row/column diff, size ratio),
/// Semantic versioning math, data drift detection (statistical distance, schema compatibility),
/// Storage math (deduplication ratio, compression overhead).
/// </summary>
public class DataVersionControlDeepMathIntegrationTests
{
    // ============================
    // Hash-Based Integrity: SHA-256 Properties
    // ============================

    [Fact]
    public void HashMath_SHA256_Deterministic()
    {
        byte[] data = Encoding.UTF8.GetBytes("Hello, dataset version control!");
        string hash1 = ComputeSHA256(data);
        string hash2 = ComputeSHA256(data);

        Assert.Equal(hash1, hash2);
    }

    [Fact]
    public void HashMath_SHA256_FixedLength()
    {
        // SHA-256 always produces 256 bits = 64 hex characters
        string hash1 = ComputeSHA256(Encoding.UTF8.GetBytes("short"));
        string hash2 = ComputeSHA256(Encoding.UTF8.GetBytes(new string('x', 10000)));

        Assert.Equal(64, hash1.Length);
        Assert.Equal(64, hash2.Length);
    }

    [Fact]
    public void HashMath_SHA256_AvalancheEffect()
    {
        // A single bit change should produce a completely different hash
        string hash1 = ComputeSHA256(Encoding.UTF8.GetBytes("data_v1"));
        string hash2 = ComputeSHA256(Encoding.UTF8.GetBytes("data_v2"));

        Assert.NotEqual(hash1, hash2);

        // Count differing hex digits - should be roughly half (avalanche property)
        int diffCount = 0;
        for (int i = 0; i < hash1.Length; i++)
        {
            if (hash1[i] != hash2[i]) diffCount++;
        }

        // At least 20% of hex digits should differ (very conservative threshold)
        Assert.True(diffCount > hash1.Length * 0.2,
            $"Only {diffCount}/{hash1.Length} hex digits differ - weak avalanche effect");
    }

    [Theory]
    [InlineData(256)]  // SHA-256 output bits
    [InlineData(64)]   // Hex character count
    public void HashMath_SHA256_OutputSize(int expectedValue)
    {
        string hash = ComputeSHA256(Encoding.UTF8.GetBytes("test"));
        if (expectedValue == 256)
        {
            // 256 bits = 32 bytes = 64 hex chars
            Assert.Equal(64, hash.Length);
        }
        else
        {
            Assert.Equal(expectedValue, hash.Length);
        }
    }

    [Fact]
    public void HashMath_CollisionResistance_BirthdayBound()
    {
        // Birthday bound for SHA-256: ~2^128 hashes needed for 50% collision probability
        // With n-bit hash, expected collisions after 2^(n/2) trials
        int hashBits = 256;
        double collisionBound = Math.Pow(2, hashBits / 2.0); // 2^128

        // This is astronomically large - log10(2^128) ≈ 38.5
        double log10Bound = (hashBits / 2.0) * Math.Log10(2);
        Assert.True(log10Bound > 38, $"Birthday bound should be > 10^38, got 10^{log10Bound:F1}");
    }

    // ============================
    // Version Numbering Math
    // ============================

    [Theory]
    [InlineData(0, 1)]    // First version
    [InlineData(5, 6)]    // Increment from 5
    [InlineData(99, 100)] // Increment from 99
    public void VersionMath_IncrementalVersioning(int currentMax, int expectedNext)
    {
        int nextVersion = currentMax + 1;
        Assert.Equal(expectedNext, nextVersion);
    }

    [Theory]
    [InlineData("1.0.0", "1.0.1", "patch")]    // Patch version bump
    [InlineData("1.0.0", "1.1.0", "minor")]    // Minor version bump
    [InlineData("1.0.0", "2.0.0", "major")]    // Major version bump
    public void VersionMath_SemanticVersioning(string current, string expected, string bumpType)
    {
        var parts = current.Split('.').Select(int.Parse).ToArray();
        int major = parts[0], minor = parts[1], patch = parts[2];

        switch (bumpType)
        {
            case "patch": patch++; break;
            case "minor": minor++; patch = 0; break;
            case "major": major++; minor = 0; patch = 0; break;
        }

        string result = $"{major}.{minor}.{patch}";
        Assert.Equal(expected, result);
    }

    // ============================
    // Comparison Metrics: Dataset Diff
    // ============================

    [Theory]
    [InlineData(1000, 1050, 50, 0)]       // 50 added, 0 removed
    [InlineData(1000, 980, 0, 20)]        // 0 added, 20 removed
    [InlineData(1000, 1030, 50, 20)]      // 50 added, 20 removed
    public void ComparisonMath_RowDiff(int originalRows, int newRows, int added, int removed)
    {
        int netChange = added - removed;
        Assert.Equal(newRows, originalRows + netChange);
    }

    [Theory]
    [InlineData(1000000, 1100000, 10.0)]   // 10% size increase
    [InlineData(1000000, 900000, -10.0)]   // 10% size decrease
    [InlineData(1000000, 1000000, 0.0)]    // No change
    public void ComparisonMath_SizeChangePercentage(long originalSize, long newSize, double expectedPercent)
    {
        double changePercent = ((double)(newSize - originalSize) / originalSize) * 100;
        Assert.Equal(expectedPercent, changePercent, 1e-10);
    }

    [Fact]
    public void ComparisonMath_SchemaCompatibility_ColumnsAdded()
    {
        string[] originalColumns = { "id", "name", "age" };
        string[] newColumns = { "id", "name", "age", "email" };

        var added = newColumns.Except(originalColumns).ToArray();
        var removed = originalColumns.Except(newColumns).ToArray();

        Assert.Single(added);
        Assert.Equal("email", added[0]);
        Assert.Empty(removed);
    }

    [Fact]
    public void ComparisonMath_SchemaCompatibility_ColumnsRemoved()
    {
        string[] originalColumns = { "id", "name", "age", "address" };
        string[] newColumns = { "id", "name", "age" };

        var removed = originalColumns.Except(newColumns).ToArray();
        Assert.Single(removed);
        Assert.Equal("address", removed[0]);
    }

    // ============================
    // Data Drift Detection
    // ============================

    [Theory]
    [InlineData(50.0, 50.0, 10.0, 10.0, 100, 100, 0.0)]     // Identical distributions
    [InlineData(50.0, 55.0, 10.0, 10.0, 100, 100, 0.5)]      // Shifted mean
    [InlineData(50.0, 50.0, 10.0, 20.0, 100, 100, 0.0)]      // Different variance (same mean)
    public void DriftMath_MeanShift(double mean1, double mean2, double std1, double std2,
        int n1, int n2, double expectedCohenD)
    {
        // Cohen's d effect size: d = (mean1 - mean2) / pooled_std
        double pooledStd = Math.Sqrt(((n1 - 1) * std1 * std1 + (n2 - 1) * std2 * std2) / (n1 + n2 - 2));
        double cohenD = Math.Abs(mean1 - mean2) / pooledStd;
        Assert.Equal(expectedCohenD, cohenD, 1e-1);
    }

    [Theory]
    [InlineData(0.0, "negligible")]
    [InlineData(0.15, "small")]
    [InlineData(0.5, "medium")]
    [InlineData(0.9, "large")]
    public void DriftMath_CohenD_Interpretation(double cohenD, string expectedEffect)
    {
        string effect;
        if (cohenD < 0.1) effect = "negligible";
        else if (cohenD < 0.35) effect = "small";
        else if (cohenD < 0.65) effect = "medium";
        else effect = "large";

        Assert.Equal(expectedEffect, effect);
    }

    [Fact]
    public void DriftMath_PopulationStabilityIndex()
    {
        // PSI = sum((actual_i - expected_i) * ln(actual_i / expected_i))
        double[] expected = { 0.10, 0.20, 0.30, 0.25, 0.15 }; // Reference distribution
        double[] actual = { 0.12, 0.18, 0.28, 0.27, 0.15 };   // Current distribution

        double psi = 0;
        for (int i = 0; i < expected.Length; i++)
        {
            psi += (actual[i] - expected[i]) * Math.Log(actual[i] / expected[i]);
        }

        // PSI < 0.1: no significant change
        // PSI 0.1-0.25: moderate change
        // PSI > 0.25: significant change
        Assert.True(psi >= 0, "PSI must be non-negative");
        Assert.True(psi < 0.1, $"Small distribution shift should have PSI < 0.1, got {psi}");
    }

    // ============================
    // Storage Math: Deduplication
    // ============================

    [Theory]
    [InlineData(1000000, 800000, 0.20)]   // 20% deduplication
    [InlineData(1000000, 500000, 0.50)]   // 50% deduplication
    [InlineData(1000000, 1000000, 0.00)]  // No deduplication
    public void StorageMath_DeduplicationRatio(long originalSize, long storedSize, double expectedRatio)
    {
        double ratio = 1.0 - (double)storedSize / originalSize;
        Assert.Equal(expectedRatio, ratio, 1e-10);
    }

    [Theory]
    [InlineData(10, 100000, 50, 1000500)]    // 10 * 100000 + 10 * 50 = 1000500
    [InlineData(5, 1000000, 200, 5001000)]   // 5 * 1000000 + 5 * 200 = 5001000
    public void StorageMath_TotalStorageEstimate(int numVersions, long avgVersionSizeBytes,
        long metadataOverheadPerVersion, long expectedTotalBytes)
    {
        long totalBytes = numVersions * avgVersionSizeBytes + numVersions * metadataOverheadPerVersion;
        Assert.Equal(expectedTotalBytes, totalBytes);
    }

    [Fact]
    public void StorageMath_DeltaStorage_SavingsEstimate()
    {
        // If only 10% of data changes between versions, delta storage saves 90%
        long fullVersionSize = 1000000; // 1 MB
        double changeRatio = 0.10; // 10% changes

        long deltaSize = (long)(fullVersionSize * changeRatio);
        long savings = fullVersionSize - deltaSize;
        double savingsPercent = (double)savings / fullVersionSize * 100;

        Assert.Equal(100000, deltaSize);
        Assert.Equal(900000, savings);
        Assert.Equal(90.0, savingsPercent, 1e-10);
    }

    // ============================
    // Lineage Math: DAG Properties
    // ============================

    [Fact]
    public void LineageMath_DAG_TransitiveReachability()
    {
        // Dataset lineage forms a DAG (Directed Acyclic Graph)
        // If A -> B -> C, then C is reachable from A (transitive closure)

        // Adjacency list representation
        var graph = new Dictionary<string, List<string>>
        {
            ["raw"] = new() { "cleaned" },
            ["cleaned"] = new() { "features", "sampled" },
            ["features"] = new() { "train_set", "test_set" },
            ["sampled"] = new() { "validation_set" },
            ["train_set"] = new(),
            ["test_set"] = new(),
            ["validation_set"] = new()
        };

        // BFS to find all reachable nodes from "raw"
        var reachable = new HashSet<string>();
        var queue = new Queue<string>();
        queue.Enqueue("raw");

        while (queue.Count > 0)
        {
            var node = queue.Dequeue();
            if (reachable.Contains(node)) continue;
            reachable.Add(node);
            foreach (var neighbor in graph[node])
                queue.Enqueue(neighbor);
        }

        Assert.Equal(7, reachable.Count); // All nodes reachable from root
        Assert.Contains("train_set", reachable);
        Assert.Contains("validation_set", reachable);
    }

    [Fact]
    public void LineageMath_DAG_NoCycles()
    {
        // A proper lineage DAG has no cycles
        // DFS-based cycle detection
        var graph = new Dictionary<string, List<string>>
        {
            ["v1"] = new() { "v2" },
            ["v2"] = new() { "v3" },
            ["v3"] = new()
        };

        bool hasCycle = DetectCycle(graph);
        Assert.False(hasCycle, "Lineage graph should not contain cycles");
    }

    [Fact]
    public void LineageMath_TopologicalSort_ValidOrdering()
    {
        // Topological sort gives valid processing order
        var graph = new Dictionary<string, List<string>>
        {
            ["raw"] = new() { "clean" },
            ["clean"] = new() { "feature" },
            ["feature"] = new() { "model" },
            ["model"] = new()
        };

        var order = TopologicalSort(graph);

        // "raw" must come before "clean", "clean" before "feature", etc.
        Assert.True(order.IndexOf("raw") < order.IndexOf("clean"));
        Assert.True(order.IndexOf("clean") < order.IndexOf("feature"));
        Assert.True(order.IndexOf("feature") < order.IndexOf("model"));
    }

    // ============================
    // Data Integrity: Checksum Verification
    // ============================

    [Fact]
    public void IntegrityMath_ChecksumVerification_ValidData()
    {
        byte[] data = Encoding.UTF8.GetBytes("important dataset contents");
        string expectedHash = ComputeSHA256(data);

        // Verify integrity
        string actualHash = ComputeSHA256(data);
        Assert.Equal(expectedHash, actualHash);
    }

    [Fact]
    public void IntegrityMath_ChecksumVerification_CorruptedData()
    {
        byte[] originalData = Encoding.UTF8.GetBytes("important dataset contents");
        string originalHash = ComputeSHA256(originalData);

        // Corrupt one byte
        byte[] corruptedData = (byte[])originalData.Clone();
        corruptedData[0] ^= 0x01;

        string corruptedHash = ComputeSHA256(corruptedData);
        Assert.NotEqual(originalHash, corruptedHash);
    }

    // ============================
    // Helper Methods
    // ============================

    private static string ComputeSHA256(byte[] data)
    {
        byte[] hash = SHA256.HashData(data);
        return BitConverter.ToString(hash).Replace("-", "").ToLowerInvariant();
    }

    private static bool DetectCycle(Dictionary<string, List<string>> graph)
    {
        var white = new HashSet<string>(graph.Keys); // Unvisited
        var gray = new HashSet<string>();  // In progress
        var black = new HashSet<string>(); // Completed

        while (white.Count > 0)
        {
            var node = white.First();
            if (DFSCycle(node, graph, white, gray, black))
                return true;
        }
        return false;
    }

    private static bool DFSCycle(string node, Dictionary<string, List<string>> graph,
        HashSet<string> white, HashSet<string> gray, HashSet<string> black)
    {
        white.Remove(node);
        gray.Add(node);

        foreach (var neighbor in graph[node])
        {
            if (gray.Contains(neighbor)) return true; // Back edge = cycle
            if (white.Contains(neighbor) && DFSCycle(neighbor, graph, white, gray, black))
                return true;
        }

        gray.Remove(node);
        black.Add(node);
        return false;
    }

    private static List<string> TopologicalSort(Dictionary<string, List<string>> graph)
    {
        var visited = new HashSet<string>();
        var result = new List<string>();

        foreach (var node in graph.Keys)
        {
            if (!visited.Contains(node))
                TopologicalDFS(node, graph, visited, result);
        }

        result.Reverse();
        return result;
    }

    private static void TopologicalDFS(string node, Dictionary<string, List<string>> graph,
        HashSet<string> visited, List<string> result)
    {
        visited.Add(node);
        foreach (var neighbor in graph[node])
        {
            if (!visited.Contains(neighbor))
                TopologicalDFS(neighbor, graph, visited, result);
        }
        result.Add(node);
    }
}
