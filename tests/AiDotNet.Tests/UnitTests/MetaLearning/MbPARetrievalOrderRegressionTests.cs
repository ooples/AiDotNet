using System;
using System.Linq;
using AiDotNet.LinearAlgebra;
using AiDotNet.MetaLearning.Algorithms;
using Xunit;

namespace AiDotNet.Tests.UnitTests.MetaLearning;

/// <summary>
/// Regression tests for the MbPA nearest-neighbour ordering defect.
/// </summary>
/// <remarks>
/// <para>
/// <b>Defect 1 — heap order leaked into the public result.</b> <c>Retrieve</c> selects the k nearest
/// with a bounded MAX-heap, which keeps the WORST kept candidate at the root. The heap array was
/// copied straight into the result, so a method contracted to return "the k nearest" returned them
/// worst-first: the nearest key was not at index 0, and the kernel weight belonging to the nearest
/// entry was reported against the farthest one.
/// </para>
/// <para>
/// The kernel itself is Sprechmann et al. (arXiv:1802.10542) Memory-based Parameter Adaptation:
/// <c>kern(h, q) = 1 / (eps + ||h - q||^2)</c>, normalised over the retrieved neighbours.
/// </para>
/// </remarks>
public class MbPARetrievalOrderRegressionTests
{
    private static Vector<double> Vec(params double[] values)
    {
        var v = new Vector<double>(values.Length);
        for (int i = 0; i < values.Length; i++) v[i] = values[i];
        return v;
    }

    [Fact]
    public void Retrieve_ReturnsNearestFirst_NotHeapOrder()
    {
        // Written farthest-first on purpose: with a max-heap the insertion order decides which entry
        // lands at the root, so a result that merely echoes the heap array puts the WRONG one first.
        var memory = new MbPAEpisodicMemory<double>(capacity: 10);
        memory.Write(Vec(9, 0, 0, 0), Vec(1, 0, 0));   // d^2 = 81
        memory.Write(Vec(3, 0, 0, 0), Vec(0, 1, 0));   // d^2 = 9
        memory.Write(Vec(1, 0, 0, 0), Vec(0, 0, 1));   // d^2 = 1   <- nearest

        var retrieved = memory.Retrieve(Vec(0, 0, 0, 0), k: 3, epsilon: 1e-6, toDouble: x => x);

        Assert.Equal(3, retrieved.Count);
        Assert.Equal(1.0, retrieved[0].Key[0], 12);
        Assert.Equal(3.0, retrieved[1].Key[0], 12);
        Assert.Equal(9.0, retrieved[2].Key[0], 12);

        // Ordering the keys is not enough: each weight must travel with its own entry.
        Assert.True(retrieved[0].Weight > retrieved[1].Weight,
            "the nearest entry must carry the largest kernel weight");
        Assert.True(retrieved[1].Weight > retrieved[2].Weight,
            "weights must decrease with distance");
    }

    [Fact]
    public void Retrieve_WeightsAreDescendingAndSumToOne_ForEveryK()
    {
        var memory = new MbPAEpisodicMemory<double>(capacity: 32);
        for (int i = 1; i <= 8; i++) memory.Write(Vec(i, 0, 0, 0), Vec(i, 0, 0));

        for (int k = 1; k <= 8; k++)
        {
            var retrieved = memory.Retrieve(Vec(0, 0, 0, 0), k: k, epsilon: 1e-6, toDouble: x => x);

            Assert.Equal(k, retrieved.Count);
            Assert.Equal(1.0, retrieved.Sum(r => r.Weight), 9);
            for (int i = 1; i < retrieved.Count; i++)
            {
                Assert.True(retrieved[i - 1].Weight >= retrieved[i].Weight,
                    $"k={k}: weight at {i - 1} ({retrieved[i - 1].Weight}) < weight at {i} ({retrieved[i].Weight})");
            }
        }
    }

    [Fact]
    public void Retrieve_SelectsTheKNearest_NotAnArbitraryK()
    {
        // Selection correctness is independent of ordering: the far entries must be EXCLUDED by k,
        // not merely sorted to the back.
        var memory = new MbPAEpisodicMemory<double>(capacity: 10);
        memory.Write(Vec(50, 0, 0, 0), Vec(1, 0, 0));
        memory.Write(Vec(1, 0, 0, 0), Vec(0, 1, 0));
        memory.Write(Vec(60, 0, 0, 0), Vec(0, 0, 1));
        memory.Write(Vec(2, 0, 0, 0), Vec(1, 1, 0));

        var retrieved = memory.Retrieve(Vec(0, 0, 0, 0), k: 2, epsilon: 1e-6, toDouble: x => x);

        Assert.Equal(2, retrieved.Count);
        Assert.Equal(1.0, retrieved[0].Key[0], 12);
        Assert.Equal(2.0, retrieved[1].Key[0], 12);
        Assert.DoesNotContain(retrieved, r => r.Key[0] >= 50.0);
    }
}
