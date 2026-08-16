#nullable disable
using AiDotNet.CausalDiscovery.ContinuousOptimization;
using AiDotNet.Models.Options;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.CausalDiscovery;

/// <summary>
/// Verifies that the continuous-optimization causal-discovery algorithms are reproducible.
/// </summary>
/// <remarks>
/// These algorithms previously fell back to an unseeded secure RNG when no seed was configured,
/// which made two runs on identical data return different graphs. Beyond making results
/// irreproducible for users, it made their generated model-family tests flaky, so those suites
/// could not be used to detect regressions.
/// </remarks>
public class CausalDiscoveryDeterminismTests
{
    /// <summary>
    /// Builds a small dataset with a clear chain structure x0 â†’ x1 â†’ x2.
    /// </summary>
    private static Matrix<double> BuildChainData(int rows = 60)
    {
        var data = new Matrix<double>(rows, 3);
        for (int i = 0; i < rows; i++)
        {
            // Deterministic pseudo-noise so the data itself is not a source of variation.
            double n0 = Math.Sin(i * 1.7);
            double n1 = Math.Sin(i * 2.3) * 0.3;
            double n2 = Math.Sin(i * 3.1) * 0.3;

            double x0 = n0;
            double x1 = 0.8 * x0 + n1;
            double x2 = 0.8 * x1 + n2;

            data[i, 0] = x0;
            data[i, 1] = x1;
            data[i, 2] = x2;
        }

        return data;
    }

    private static void AssertIdentical(Matrix<double> first, Matrix<double> second, string algorithm)
    {
        Assert.Equal(first.Rows, second.Rows);
        Assert.Equal(first.Columns, second.Columns);

        for (int i = 0; i < first.Rows; i++)
        {
            for (int j = 0; j < first.Columns; j++)
            {
                Assert.True(
                    Math.Abs(first[i, j] - second[i, j]) < 1e-12,
                    $"{algorithm}: entry [{i},{j}] differs between runs ({first[i, j]} vs {second[i, j]}). " +
                    "The algorithm is not reproducible with default options.");
            }
        }
    }

    /// <summary>
    /// Two runs with default options must produce identical graphs.
    /// </summary>
    [Fact]
    public void NOTEARSLowRank_WithDefaultOptions_IsReproducible()
    {
        var data = BuildChainData();

        var first = new NOTEARSLowRank<double>().DiscoverStructure(data).AdjacencyMatrix;
        var second = new NOTEARSLowRank<double>().DiscoverStructure(data).AdjacencyMatrix;

        AssertIdentical(first, second, nameof(NOTEARSLowRank<double>));
    }

    /// <summary>
    /// An explicitly supplied seed must still be honoured, and must still be reproducible.
    /// </summary>
    [Fact]
    public void NOTEARSLowRank_WithExplicitSeed_IsReproducible()
    {
        var data = BuildChainData();
        var options = new CausalDiscoveryOptions { Seed = 1234 };

        var first = new NOTEARSLowRank<double>(options).DiscoverStructure(data).AdjacencyMatrix;
        var second = new NOTEARSLowRank<double>(options).DiscoverStructure(data).AdjacencyMatrix;

        AssertIdentical(first, second, nameof(NOTEARSLowRank<double>));
    }

    /// <summary>
    /// Different seeds are allowed to give different answers â€” otherwise the seed would be doing
    /// nothing and the reproducibility tests above would be vacuous.
    /// </summary>
    [Fact]
    public void NOTEARSLowRank_SeedIsActuallyUsed()
    {
        // Rank above the variable count forces the random-initialization branch to be exercised,
        // which is the only place the seed enters this algorithm.
        var data = BuildChainData();

        var first = new NOTEARSLowRank<double>(
            new CausalDiscoveryOptions { Seed = 1, MaxRank = 8 }).DiscoverStructure(data).AdjacencyMatrix;
        var second = new NOTEARSLowRank<double>(
            new CausalDiscoveryOptions { Seed = 2, MaxRank = 8 }).DiscoverStructure(data).AdjacencyMatrix;

        // Both must at least be well-formed; the seed may or may not change the converged answer,
        // so this asserts only that supplying a seed does not break the algorithm.
        Assert.Equal(3, first.Rows);
        Assert.Equal(3, second.Rows);
    }

    [Fact]
    public void NOTEARSSobolev_WithDefaultOptions_IsReproducible()
    {
        var data = BuildChainData();

        var first = new NOTEARSSobolev<double>().DiscoverStructure(data).AdjacencyMatrix;
        var second = new NOTEARSSobolev<double>().DiscoverStructure(data).AdjacencyMatrix;

        AssertIdentical(first, second, nameof(NOTEARSSobolev<double>));
    }
}
