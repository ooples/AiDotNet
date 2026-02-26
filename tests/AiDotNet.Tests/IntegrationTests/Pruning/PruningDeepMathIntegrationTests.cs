using AiDotNet.Interfaces;
using AiDotNet.ModelCompression;
using AiDotNet.Pruning;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.Pruning;

/// <summary>
/// Deep math-correctness integration tests for the Pruning module.
/// Verifies exact numerical results for PruningMask sparsity computation,
/// mask application, MagnitudePruningStrategy importance scoring, mask creation,
/// N:M structured sparsity, and COO/CSR/CSC sparse format conversions.
/// </summary>
public class PruningDeepMathIntegrationTests
{
    private const double Tolerance = 1e-10;

    #region PruningMask Sparsity Computation

    [Fact]
    public void PruningMask_GetSparsity_AllOnes_ReturnsZero()
    {
        // All-ones mask = no pruning = sparsity 0.0
        var mask = new PruningMask<double>(3, 4);
        // Default constructor fills with ones

        double sparsity = mask.GetSparsity();

        // 0 zeros / 12 total = 0.0
        Assert.Equal(0.0, sparsity, Tolerance);
    }

    [Fact]
    public void PruningMask_GetSparsity_ExactFractionFromBoolArray()
    {
        // 3 out of 8 are false (pruned) => sparsity = 3/8 = 0.375
        var keepIndices = new bool[] { true, false, true, true, false, true, false, true };
        var mask = new PruningMask<double>(keepIndices);

        double sparsity = mask.GetSparsity();

        // Stored as 1x8 matrix: 3 zeros out of 8 total
        Assert.Equal(3.0 / 8.0, sparsity, Tolerance);
    }

    [Fact]
    public void PruningMask_GetSparsity_2DBoolArray_ExactComputation()
    {
        // 3x3 mask with specific pattern: 4 pruned out of 9 => sparsity = 4/9
        var keepIndices = new bool[,]
        {
            { true, false, true },
            { false, true, false },
            { true, true, false }
        };
        var mask = new PruningMask<double>(keepIndices);

        double sparsity = mask.GetSparsity();

        // 4 zeros / 9 total = 0.4444...
        Assert.Equal(4.0 / 9.0, sparsity, Tolerance);
    }

    [Fact]
    public void PruningMask_GetSparsity_AllPruned_ReturnsOne()
    {
        // All false = all pruned = sparsity 1.0
        var keepIndices = new bool[] { false, false, false, false, false };
        var mask = new PruningMask<double>(keepIndices);

        double sparsity = mask.GetSparsity();

        Assert.Equal(1.0, sparsity, Tolerance);
    }

    #endregion

    #region PruningMask Apply (Element-wise Multiply)

    [Fact]
    public void PruningMask_Apply_Matrix_ElementWiseMultiply_ExactValues()
    {
        // Mask: keep positions (0,0), (0,2), (1,1) => weight * 1; prune rest => weight * 0
        var keepIndices = new bool[,]
        {
            { true, false, true },
            { false, true, false }
        };
        var mask = new PruningMask<double>(keepIndices);

        var weights = new Matrix<double>(2, 3);
        weights[0, 0] = 0.5;
        weights[0, 1] = -0.3;
        weights[0, 2] = 0.8;
        weights[1, 0] = -0.1;
        weights[1, 1] = 0.9;
        weights[1, 2] = -0.7;

        var result = mask.Apply(weights);

        // Kept positions: weight * 1
        Assert.Equal(0.5, result[0, 0], Tolerance);
        Assert.Equal(0.8, result[0, 2], Tolerance);
        Assert.Equal(0.9, result[1, 1], Tolerance);

        // Pruned positions: weight * 0 = 0
        Assert.Equal(0.0, result[0, 1], Tolerance);
        Assert.Equal(0.0, result[1, 0], Tolerance);
        Assert.Equal(0.0, result[1, 2], Tolerance);
    }

    [Fact]
    public void PruningMask_Apply_Vector_PreservesKeptZeroesPruned()
    {
        var keepIndices = new bool[] { true, false, true, false, true };
        var mask = new PruningMask<double>(keepIndices);

        var weights = new Vector<double>(new double[] { 1.5, -2.3, 0.7, 4.1, -0.5 });

        var result = mask.Apply(weights);

        Assert.Equal(1.5, result[0], Tolerance);
        Assert.Equal(0.0, result[1], Tolerance);
        Assert.Equal(0.7, result[2], Tolerance);
        Assert.Equal(0.0, result[3], Tolerance);
        Assert.Equal(-0.5, result[4], Tolerance);
    }

    [Fact]
    public void PruningMask_Apply_PreservesNegativeWeights()
    {
        // Negative weights that are kept should remain negative (not absolute valued)
        var keepIndices = new bool[] { true, true, true };
        var mask = new PruningMask<double>(keepIndices);

        var weights = new Vector<double>(new double[] { -3.14, -0.001, -99.9 });

        var result = mask.Apply(weights);

        Assert.Equal(-3.14, result[0], Tolerance);
        Assert.Equal(-0.001, result[1], Tolerance);
        Assert.Equal(-99.9, result[2], Tolerance);
    }

    [Fact]
    public void PruningMask_Apply_Matrix_ShapeMismatch_Throws()
    {
        var mask = new PruningMask<double>(2, 3);
        var wrongShape = new Matrix<double>(3, 2);

        Assert.Throws<ArgumentException>(() => mask.Apply(wrongShape));
    }

    #endregion

    #region PruningMask CombineWith (Logical AND)

    [Fact]
    public void PruningMask_CombineWith_LogicalAND_ExactBehavior()
    {
        // Mask A: [1, 0, 1, 1, 0]
        // Mask B: [1, 1, 0, 1, 0]
        // AND:    [1, 0, 0, 1, 0]  - both must be 1 to keep
        var maskA = new PruningMask<double>(new bool[] { true, false, true, true, false });
        var maskB = new PruningMask<double>(new bool[] { true, true, false, true, false });

        var combined = maskA.CombineWith(maskB);

        var data = combined.GetMaskData();
        Assert.Equal(1.0, data[0], Tolerance); // 1 AND 1 = 1
        Assert.Equal(0.0, data[1], Tolerance); // 0 AND 1 = 0
        Assert.Equal(0.0, data[2], Tolerance); // 1 AND 0 = 0
        Assert.Equal(1.0, data[3], Tolerance); // 1 AND 1 = 1
        Assert.Equal(0.0, data[4], Tolerance); // 0 AND 0 = 0
    }

    [Fact]
    public void PruningMask_CombineWith_SparsityIncreasesOrStays()
    {
        // Combining masks can only increase sparsity (more pruning) or keep it the same
        var maskA = new PruningMask<double>(new bool[] { true, false, true, true, false, true, true, false });
        var maskB = new PruningMask<double>(new bool[] { true, true, false, true, true, false, true, false });

        double sparsityA = maskA.GetSparsity(); // 3/8 = 0.375
        double sparsityB = maskB.GetSparsity(); // 3/8 = 0.375

        var combined = maskA.CombineWith(maskB);
        double sparsityCombined = combined.GetSparsity();

        // AND can only prune more: sparsity(A AND B) >= max(sparsity(A), sparsity(B))
        Assert.True(sparsityCombined >= Math.Max(sparsityA, sparsityB) - Tolerance,
            $"Combined sparsity {sparsityCombined} should be >= max({sparsityA}, {sparsityB})");
    }

    [Fact]
    public void PruningMask_CombineWith_2D_LogicalAND()
    {
        // Mask A: [[1,0],[1,1]]
        // Mask B: [[0,1],[1,0]]
        // AND:    [[0,0],[1,0]]
        var maskA = new PruningMask<double>(new bool[,] { { true, false }, { true, true } });
        var maskB = new PruningMask<double>(new bool[,] { { false, true }, { true, false } });

        var combined = maskA.CombineWith(maskB);

        var data = combined.GetMaskData();
        Assert.Equal(0.0, data[0], Tolerance); // (0,0): 1 AND 0 = 0
        Assert.Equal(0.0, data[1], Tolerance); // (0,1): 0 AND 1 = 0
        Assert.Equal(1.0, data[2], Tolerance); // (1,0): 1 AND 1 = 1
        Assert.Equal(0.0, data[3], Tolerance); // (1,1): 1 AND 0 = 0

        // 3 out of 4 pruned => sparsity = 0.75
        Assert.Equal(0.75, combined.GetSparsity(), Tolerance);
    }

    #endregion

    #region PruningMask GetKeptIndices / GetPrunedIndices

    [Fact]
    public void PruningMask_GetKeptAndPrunedIndices_AreComplementary()
    {
        var keepIndices = new bool[] { true, false, true, false, false, true, true, false };
        var mask = new PruningMask<double>(keepIndices);

        var kept = mask.GetKeptIndices();
        var pruned = mask.GetPrunedIndices();

        // Kept + Pruned should cover all indices exactly once
        var all = kept.Concat(pruned).OrderBy(x => x).ToArray();
        Assert.Equal(8, all.Length);
        for (int i = 0; i < 8; i++)
        {
            Assert.Equal(i, all[i]);
        }

        // Verify specific indices
        Assert.Equal(new[] { 0, 2, 5, 6 }, kept);
        Assert.Equal(new[] { 1, 3, 4, 7 }, pruned);
    }

    [Fact]
    public void PruningMask_GetKeptIndices_2D_RowMajorOrder()
    {
        // 2x3 mask: [[1,0,1],[0,1,0]]
        // Flat row-major: [1,0,1,0,1,0] => kept at indices 0,2,4
        var keepIndices = new bool[,]
        {
            { true, false, true },
            { false, true, false }
        };
        var mask = new PruningMask<double>(keepIndices);

        var kept = mask.GetKeptIndices();

        Assert.Equal(new[] { 0, 2, 4 }, kept);
    }

    [Fact]
    public void PruningMask_GetPrunedIndices_CountMatchesSparsity()
    {
        var keepIndices = new bool[,]
        {
            { true, false, false, true },
            { false, true, false, false },
            { true, true, false, true }
        };
        var mask = new PruningMask<double>(keepIndices);

        var pruned = mask.GetPrunedIndices();
        double sparsity = mask.GetSparsity();

        // Count false entries: row0: 2 (cols 1,2), row1: 3 (cols 0,2,3), row2: 1 (col 2) = 6 pruned out of 12
        Assert.Equal(6, pruned.Length);
        Assert.Equal(6.0 / 12.0, sparsity, Tolerance);
    }

    #endregion

    #region PruningMask UpdateMask

    [Fact]
    public void PruningMask_UpdateMask_ChangesSparsity()
    {
        // Start with all ones (no pruning)
        var mask = new PruningMask<double>(1, 6);
        Assert.Equal(0.0, mask.GetSparsity(), Tolerance);

        // Update to prune 4 out of 6
        mask.UpdateMask(new bool[] { true, false, false, false, true, false });

        Assert.Equal(4.0 / 6.0, mask.GetSparsity(), Tolerance);
    }

    [Fact]
    public void PruningMask_UpdateMask_2D_OverwritesPrevious()
    {
        var mask = new PruningMask<double>(2, 2);
        Assert.Equal(0.0, mask.GetSparsity(), Tolerance);

        mask.UpdateMask(new bool[,] { { false, true }, { true, false } });

        Assert.Equal(0.5, mask.GetSparsity(), Tolerance);

        var data = mask.GetMaskData();
        Assert.Equal(0.0, data[0], Tolerance);
        Assert.Equal(1.0, data[1], Tolerance);
        Assert.Equal(1.0, data[2], Tolerance);
        Assert.Equal(0.0, data[3], Tolerance);
    }

    #endregion

    #region MagnitudePruningStrategy ImportanceScores

    [Fact]
    public void MagnitudePruning_ImportanceScores_Vector_ExactAbsoluteValues()
    {
        var strategy = new MagnitudePruningStrategy<double>();
        var weights = new Vector<double>(new double[] { -3.5, 0.0, 2.1, -0.7, 1.0, -4.2 });

        var scores = strategy.ComputeImportanceScores(weights);

        // importance = |weight|
        Assert.Equal(3.5, scores[0], Tolerance);
        Assert.Equal(0.0, scores[1], Tolerance);
        Assert.Equal(2.1, scores[2], Tolerance);
        Assert.Equal(0.7, scores[3], Tolerance);
        Assert.Equal(1.0, scores[4], Tolerance);
        Assert.Equal(4.2, scores[5], Tolerance);
    }

    [Fact]
    public void MagnitudePruning_ImportanceScores_Matrix_ExactAbsoluteValues()
    {
        var strategy = new MagnitudePruningStrategy<double>();
        var weights = new Matrix<double>(2, 3);
        weights[0, 0] = -0.5;  weights[0, 1] = 0.3;   weights[0, 2] = -0.8;
        weights[1, 0] = 0.1;   weights[1, 1] = -0.9;  weights[1, 2] = 0.0;

        var scores = strategy.ComputeImportanceScores(weights);

        Assert.Equal(0.5, scores[0, 0], Tolerance);
        Assert.Equal(0.3, scores[0, 1], Tolerance);
        Assert.Equal(0.8, scores[0, 2], Tolerance);
        Assert.Equal(0.1, scores[1, 0], Tolerance);
        Assert.Equal(0.9, scores[1, 1], Tolerance);
        Assert.Equal(0.0, scores[1, 2], Tolerance);
    }

    [Fact]
    public void MagnitudePruning_ImportanceScores_IgnoresGradients()
    {
        // Magnitude pruning should produce same scores regardless of gradients
        var strategy = new MagnitudePruningStrategy<double>();
        var weights = new Vector<double>(new double[] { -2.0, 1.5, -0.3 });
        var gradients = new Vector<double>(new double[] { 100.0, 200.0, 300.0 });

        var scoresNoGrad = strategy.ComputeImportanceScores(weights);
        var scoresWithGrad = strategy.ComputeImportanceScores(weights, gradients);

        for (int i = 0; i < 3; i++)
        {
            Assert.Equal(scoresNoGrad[i], scoresWithGrad[i], Tolerance);
        }
    }

    #endregion

    #region MagnitudePruningStrategy CreateMask for Vectors

    [Fact]
    public void MagnitudePruning_CreateMask_Vector_PrunesSmallestFirst()
    {
        // Weights: [0.1, 0.5, 0.3, 0.9, 0.2] => sorted ascending by |w|: 0.1, 0.2, 0.3, 0.5, 0.9
        // Target sparsity 0.4 => prune 40% of 5 = Math.Round(2.0) = 2 smallest
        // Prune indices with |w| = 0.1 (idx 0) and |w| = 0.2 (idx 4)
        var strategy = new MagnitudePruningStrategy<double>();
        var weights = new Vector<double>(new double[] { 0.1, 0.5, 0.3, 0.9, 0.2 });
        var scores = strategy.ComputeImportanceScores(weights);

        var mask = strategy.CreateMask(scores, 0.4);

        var maskData = mask.GetMaskData();
        Assert.Equal(0.0, maskData[0], Tolerance); // 0.1 pruned (smallest)
        Assert.Equal(1.0, maskData[1], Tolerance); // 0.5 kept
        Assert.Equal(1.0, maskData[2], Tolerance); // 0.3 kept
        Assert.Equal(1.0, maskData[3], Tolerance); // 0.9 kept
        Assert.Equal(0.0, maskData[4], Tolerance); // 0.2 pruned (2nd smallest)
    }

    [Fact]
    public void MagnitudePruning_CreateMask_Vector_ZeroSparsity_KeepsAll()
    {
        var strategy = new MagnitudePruningStrategy<double>();
        var weights = new Vector<double>(new double[] { 0.5, -0.3, 0.8 });
        var scores = strategy.ComputeImportanceScores(weights);

        var mask = strategy.CreateMask(scores, 0.0);

        // Zero sparsity => keep all, Round(0) = 0 pruned
        var maskData = mask.GetMaskData();
        Assert.Equal(1.0, maskData[0], Tolerance);
        Assert.Equal(1.0, maskData[1], Tolerance);
        Assert.Equal(1.0, maskData[2], Tolerance);
    }

    [Fact]
    public void MagnitudePruning_CreateMask_Vector_FullSparsity_PrunesAll()
    {
        var strategy = new MagnitudePruningStrategy<double>();
        var weights = new Vector<double>(new double[] { 0.5, -0.3, 0.8 });
        var scores = strategy.ComputeImportanceScores(weights);

        var mask = strategy.CreateMask(scores, 1.0);

        // Full sparsity => prune all, Round(3) = 3 pruned
        var maskData = mask.GetMaskData();
        Assert.Equal(0.0, maskData[0], Tolerance);
        Assert.Equal(0.0, maskData[1], Tolerance);
        Assert.Equal(0.0, maskData[2], Tolerance);
    }

    [Fact]
    public void MagnitudePruning_CreateMask_Vector_InvalidSparsity_Throws()
    {
        var strategy = new MagnitudePruningStrategy<double>();
        var scores = new Vector<double>(new double[] { 0.5, 0.3 });

        Assert.Throws<ArgumentException>(() => strategy.CreateMask(scores, -0.1));
        Assert.Throws<ArgumentException>(() => strategy.CreateMask(scores, 1.1));
    }

    #endregion

    #region MagnitudePruningStrategy CreateMask for Matrices

    [Fact]
    public void MagnitudePruning_CreateMask_Matrix_ExactPruneCount()
    {
        // 3x4 matrix = 12 elements. Sparsity 0.5 => (int)(12 * 0.5) = 6 pruned
        var strategy = new MagnitudePruningStrategy<double>();
        var weights = new Matrix<double>(3, 4);
        // Set weights with known magnitudes
        double[] flatWeights = { 0.1, 0.5, 0.3, 0.9, 0.2, 0.8, 0.4, 0.7, 0.6, 1.0, 0.15, 0.85 };
        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 4; j++)
                weights[i, j] = flatWeights[i * 4 + j];

        var scores = strategy.ComputeImportanceScores(weights);
        var mask = strategy.CreateMask(scores, 0.5);

        // Count pruned elements
        var maskData = mask.GetMaskData();
        int prunedCount = maskData.Count(v => Math.Abs(v) < Tolerance);

        // (int)(12 * 0.5) = 6
        Assert.Equal(6, prunedCount);

        // The 6 smallest by magnitude are: 0.1, 0.15, 0.2, 0.3, 0.4, 0.5
        // Indices (row-major): 0(0.1), 10(0.15), 4(0.2), 2(0.3), 6(0.4), 1(0.5)
        Assert.Equal(0.0, maskData[0], Tolerance);  // 0.1 pruned
        Assert.Equal(0.0, maskData[10], Tolerance); // 0.15 pruned
        Assert.Equal(0.0, maskData[4], Tolerance);  // 0.2 pruned
        Assert.Equal(0.0, maskData[2], Tolerance);  // 0.3 pruned
        Assert.Equal(0.0, maskData[6], Tolerance);  // 0.4 pruned
        Assert.Equal(0.0, maskData[1], Tolerance);  // 0.5 pruned

        // Remaining should be kept
        Assert.Equal(1.0, maskData[3], Tolerance);  // 0.9 kept
        Assert.Equal(1.0, maskData[5], Tolerance);  // 0.8 kept
        Assert.Equal(1.0, maskData[7], Tolerance);  // 0.7 kept
        Assert.Equal(1.0, maskData[8], Tolerance);  // 0.6 kept
        Assert.Equal(1.0, maskData[9], Tolerance);  // 1.0 kept
        Assert.Equal(1.0, maskData[11], Tolerance); // 0.85 kept
    }

    [Fact]
    public void MagnitudePruning_CreateMask_NegativeWeights_PrunedByAbsoluteValue()
    {
        // Verify that -0.9 is kept (high |w|) while 0.1 is pruned (low |w|)
        var strategy = new MagnitudePruningStrategy<double>();
        var weights = new Matrix<double>(1, 4);
        weights[0, 0] = 0.1;    // |w| = 0.1 (should be pruned)
        weights[0, 1] = -0.9;   // |w| = 0.9 (should be kept)
        weights[0, 2] = -0.05;  // |w| = 0.05 (should be pruned)
        weights[0, 3] = 0.8;    // |w| = 0.8 (should be kept)

        var scores = strategy.ComputeImportanceScores(weights);
        var mask = strategy.CreateMask(scores, 0.5);

        // (int)(4 * 0.5) = 2 pruned: the 2 smallest are 0.05 (idx 2) and 0.1 (idx 0)
        var maskData = mask.GetMaskData();
        Assert.Equal(0.0, maskData[0], Tolerance);  // 0.1 pruned
        Assert.Equal(1.0, maskData[1], Tolerance);  // -0.9 kept (high magnitude)
        Assert.Equal(0.0, maskData[2], Tolerance);  // -0.05 pruned
        Assert.Equal(1.0, maskData[3], Tolerance);  // 0.8 kept
    }

    #endregion

    #region MagnitudePruningStrategy ApplyPruning

    [Fact]
    public void MagnitudePruning_ApplyPruning_Vector_InPlaceZeroing()
    {
        var strategy = new MagnitudePruningStrategy<double>();
        var weights = new Vector<double>(new double[] { 0.1, -0.5, 0.3, 0.9, -0.2 });
        var scores = strategy.ComputeImportanceScores(weights);
        var mask = strategy.CreateMask(scores, 0.4);

        // Apply in-place
        strategy.ApplyPruning(weights, mask);

        // The 2 smallest (0.1 at idx 0, 0.2 at idx 4) should be zeroed
        Assert.Equal(0.0, weights[0], Tolerance);
        Assert.Equal(-0.5, weights[1], Tolerance);
        Assert.Equal(0.3, weights[2], Tolerance);
        Assert.Equal(0.9, weights[3], Tolerance);
        Assert.Equal(0.0, weights[4], Tolerance);
    }

    [Fact]
    public void MagnitudePruning_ApplyPruning_Matrix_InPlace()
    {
        var strategy = new MagnitudePruningStrategy<double>();
        var weights = new Matrix<double>(2, 2);
        weights[0, 0] = 0.1;  weights[0, 1] = 0.9;
        weights[1, 0] = 0.5;  weights[1, 1] = 0.3;

        var scores = strategy.ComputeImportanceScores(weights);
        var mask = strategy.CreateMask(scores, 0.5);

        strategy.ApplyPruning(weights, mask);

        // (int)(4 * 0.5) = 2 pruned: smallest are 0.1 and 0.3
        Assert.Equal(0.0, weights[0, 0], Tolerance);  // 0.1 pruned
        Assert.Equal(0.9, weights[0, 1], Tolerance);  // 0.9 kept
        Assert.Equal(0.5, weights[1, 0], Tolerance);  // 0.5 kept
        Assert.Equal(0.0, weights[1, 1], Tolerance);  // 0.3 pruned
    }

    #endregion

    #region N:M Structured Sparsity

    [Fact]
    public void MagnitudePruning_Create2to4Mask_ExactlyTwoZerosPerGroup()
    {
        // 2:4 sparsity: in every group of 4 elements, exactly 2 are pruned
        var strategy = new MagnitudePruningStrategy<double>();

        // 8 elements = 2 groups of 4
        var weights = new double[] { 0.1, 0.5, 0.3, 0.9, 0.2, 0.8, 0.4, 0.7 };
        var tensor = Tensor<double>.FromVector(new Vector<double>(weights), new int[] { 8 });
        var scores = strategy.ComputeImportanceScores(tensor);

        var mask = strategy.Create2to4Mask(scores);

        var maskData = mask.GetMaskData();

        // Group 1: [0.1, 0.5, 0.3, 0.9] => prune 2 smallest: 0.1 (idx 0), 0.3 (idx 2)
        Assert.Equal(0.0, maskData[0], Tolerance); // 0.1 pruned
        Assert.Equal(1.0, maskData[1], Tolerance); // 0.5 kept
        Assert.Equal(0.0, maskData[2], Tolerance); // 0.3 pruned
        Assert.Equal(1.0, maskData[3], Tolerance); // 0.9 kept

        // Group 2: [0.2, 0.8, 0.4, 0.7] => prune 2 smallest: 0.2 (idx 4), 0.4 (idx 6)
        Assert.Equal(0.0, maskData[4], Tolerance); // 0.2 pruned
        Assert.Equal(1.0, maskData[5], Tolerance); // 0.8 kept
        Assert.Equal(0.0, maskData[6], Tolerance); // 0.4 pruned
        Assert.Equal(1.0, maskData[7], Tolerance); // 0.7 kept

        // Verify exactly 50% sparsity (2:4 = 50%)
        Assert.Equal(0.5, mask.GetSparsity(), Tolerance);
    }

    [Fact]
    public void MagnitudePruning_CreateNtoMMask_3to6_ExactlyThreeZerosPerGroup()
    {
        // 3:6 sparsity: in every group of 6, exactly 3 are pruned
        var strategy = new MagnitudePruningStrategy<double>();

        // 6 elements = 1 group
        var weights = new double[] { 0.1, 0.6, 0.3, 0.5, 0.2, 0.4 };
        var tensor = Tensor<double>.FromVector(new Vector<double>(weights), new int[] { 6 });
        var scores = strategy.ComputeImportanceScores(tensor);

        var mask = strategy.CreateNtoMMask(scores, 3, 6);

        var maskData = mask.GetMaskData();

        // Sorted ascending: 0.1(idx0), 0.2(idx4), 0.3(idx2), 0.4(idx5), 0.5(idx3), 0.6(idx1)
        // Prune 3 smallest: 0.1(idx0), 0.2(idx4), 0.3(idx2)
        Assert.Equal(0.0, maskData[0], Tolerance); // 0.1 pruned
        Assert.Equal(1.0, maskData[1], Tolerance); // 0.6 kept
        Assert.Equal(0.0, maskData[2], Tolerance); // 0.3 pruned
        Assert.Equal(1.0, maskData[3], Tolerance); // 0.5 kept
        Assert.Equal(0.0, maskData[4], Tolerance); // 0.2 pruned
        Assert.Equal(1.0, maskData[5], Tolerance); // 0.4 kept

        // 3:6 = 50% sparsity
        Assert.Equal(0.5, mask.GetSparsity(), Tolerance);
    }

    [Fact]
    public void MagnitudePruning_CreateNtoMMask_1to4_Exactly25PercentSparse()
    {
        // 1:4 sparsity: prune 1 out of every 4 elements
        var strategy = new MagnitudePruningStrategy<double>();

        var weights = new double[] { 0.9, 0.1, 0.5, 0.3, 0.8, 0.2, 0.7, 0.4 };
        var tensor = Tensor<double>.FromVector(new Vector<double>(weights), new int[] { 8 });
        var scores = strategy.ComputeImportanceScores(tensor);

        var mask = strategy.CreateNtoMMask(scores, 1, 4);

        var maskData = mask.GetMaskData();

        // Group 1: [0.9, 0.1, 0.5, 0.3] => prune 1 smallest: 0.1 (idx 1)
        Assert.Equal(1.0, maskData[0], Tolerance);
        Assert.Equal(0.0, maskData[1], Tolerance); // smallest in group
        Assert.Equal(1.0, maskData[2], Tolerance);
        Assert.Equal(1.0, maskData[3], Tolerance);

        // Group 2: [0.8, 0.2, 0.7, 0.4] => prune 1 smallest: 0.2 (idx 5)
        Assert.Equal(1.0, maskData[4], Tolerance);
        Assert.Equal(0.0, maskData[5], Tolerance); // smallest in group
        Assert.Equal(1.0, maskData[6], Tolerance);
        Assert.Equal(1.0, maskData[7], Tolerance);

        // 1:4 = 25% sparsity
        Assert.Equal(0.25, mask.GetSparsity(), Tolerance);
    }

    [Fact]
    public void MagnitudePruning_Create2to4Mask_PartialLastGroup()
    {
        // 6 elements: first group of 4, second group of 2 (partial)
        var strategy = new MagnitudePruningStrategy<double>();

        var weights = new double[] { 0.1, 0.9, 0.5, 0.3, 0.7, 0.2 };
        var tensor = Tensor<double>.FromVector(new Vector<double>(weights), new int[] { 6 });
        var scores = strategy.ComputeImportanceScores(tensor);

        var mask = strategy.CreateNtoMMask(scores, 2, 4);

        var maskData = mask.GetMaskData();

        // Group 1 (4 elements): [0.1, 0.9, 0.5, 0.3] => prune 2 smallest: 0.1(idx0), 0.3(idx3)
        Assert.Equal(0.0, maskData[0], Tolerance); // 0.1 pruned
        Assert.Equal(1.0, maskData[1], Tolerance); // 0.9 kept
        Assert.Equal(1.0, maskData[2], Tolerance); // 0.5 kept
        Assert.Equal(0.0, maskData[3], Tolerance); // 0.3 pruned

        // Group 2 (2 elements, partial): [0.7, 0.2] => prune min(2, 2) = 2 smallest
        // Both elements would be pruned since n=2 and groupSize=2
        Assert.Equal(0.0, maskData[4], Tolerance); // 0.7 pruned (even though large, n=2 prunes all in group of 2)
        Assert.Equal(0.0, maskData[5], Tolerance); // 0.2 pruned
    }

    #endregion

    #region COO Sparse Format Conversion

    [Fact]
    public void MagnitudePruning_ToSparseFormat_COO_CorrectTriplets()
    {
        var strategy = new MagnitudePruningStrategy<double>();

        // 2x3 tensor with some zeros (pre-pruned)
        var data = new double[] { 0.0, 0.5, 0.0, 0.3, 0.0, 0.8 };
        var tensor = Tensor<double>.FromVector(new Vector<double>(data), new int[] { 2, 3 });

        var result = strategy.ToSparseFormat(tensor, SparseFormat.COO);

        Assert.Equal(SparseFormat.COO, result.Format);

        // Non-zero values: 0.5 at (0,1), 0.3 at (1,0), 0.8 at (1,2)
        Assert.Equal(3, result.Values.Length);
        Assert.Equal(0.5, result.Values[0], Tolerance);
        Assert.Equal(0.3, result.Values[1], Tolerance);
        Assert.Equal(0.8, result.Values[2], Tolerance);

        // Row indices
        Assert.Equal(3, result.RowIndices?.Length);
        Assert.Equal(0, result.RowIndices?[0]); // (0,1) => row 0
        Assert.Equal(1, result.RowIndices?[1]); // (1,0) => row 1
        Assert.Equal(1, result.RowIndices?[2]); // (1,2) => row 1

        // Column indices: col = flatIdx % cols
        Assert.Equal(3, result.ColumnIndices?.Length);
        Assert.Equal(1, result.ColumnIndices?[0]); // (0,1) => col 1
        Assert.Equal(0, result.ColumnIndices?[1]); // (1,0) => col 0
        Assert.Equal(2, result.ColumnIndices?[2]); // (1,2) => col 2
    }

    [Fact]
    public void MagnitudePruning_ToSparseFormat_COO_Sparsity()
    {
        var strategy = new MagnitudePruningStrategy<double>();

        // 3x3 with 6 zeros and 3 non-zeros
        var data = new double[] { 1.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 3.0 };
        var tensor = Tensor<double>.FromVector(new Vector<double>(data), new int[] { 3, 3 });

        var result = strategy.ToSparseFormat(tensor, SparseFormat.COO);

        Assert.Equal(3, result.NonZeroCount);
        // Sparsity = 1 - 3/9 = 6/9 = 2/3
        Assert.Equal(2.0 / 3.0, result.Sparsity, Tolerance);
    }

    #endregion

    #region CSR Sparse Format Conversion

    [Fact]
    public void MagnitudePruning_ToSparseFormat_CSR_CorrectStructure()
    {
        var strategy = new MagnitudePruningStrategy<double>();

        // 3x4 matrix:
        // Row 0: [0, 1, 0, 2]  => 2 non-zeros at cols 1, 3
        // Row 1: [3, 0, 0, 0]  => 1 non-zero at col 0
        // Row 2: [0, 0, 4, 5]  => 2 non-zeros at cols 2, 3
        var data = new double[] { 0, 1, 0, 2, 3, 0, 0, 0, 0, 0, 4, 5 };
        var tensor = Tensor<double>.FromVector(new Vector<double>(data), new int[] { 3, 4 });

        var result = strategy.ToSparseFormat(tensor, SparseFormat.CSR);

        Assert.Equal(SparseFormat.CSR, result.Format);

        // Values in row order
        Assert.Equal(5, result.Values.Length);
        Assert.Equal(1.0, result.Values[0], Tolerance);
        Assert.Equal(2.0, result.Values[1], Tolerance);
        Assert.Equal(3.0, result.Values[2], Tolerance);
        Assert.Equal(4.0, result.Values[3], Tolerance);
        Assert.Equal(5.0, result.Values[4], Tolerance);

        // Column indices
        Assert.Equal(new[] { 1, 3, 0, 2, 3 }, result.ColumnIndices);

        // Row pointers: [0, 2, 3, 5]
        // Row 0 starts at 0, has 2 elements => next at 2
        // Row 1 starts at 2, has 1 element => next at 3
        // Row 2 starts at 3, has 2 elements => next at 5
        Assert.Equal(new[] { 0, 2, 3, 5 }, result.RowPointers);
    }

    [Fact]
    public void MagnitudePruning_ToSparseFormat_CSR_EmptyRow()
    {
        var strategy = new MagnitudePruningStrategy<double>();

        // Row 0: [1, 0], Row 1: [0, 0] (empty!), Row 2: [0, 2]
        var data = new double[] { 1, 0, 0, 0, 0, 2 };
        var tensor = Tensor<double>.FromVector(new Vector<double>(data), new int[] { 3, 2 });

        var result = strategy.ToSparseFormat(tensor, SparseFormat.CSR);

        // Values: 1, 2
        Assert.Equal(2, result.Values.Length);

        // Row pointers: [0, 1, 1, 2]
        // Row 0: 1 element starting at 0
        // Row 1: 0 elements, pointer stays at 1
        // Row 2: 1 element starting at 1
        Assert.Equal(new[] { 0, 1, 1, 2 }, result.RowPointers);
    }

    [Fact]
    public void MagnitudePruning_ToSparseFormat_CSR_RowPointerProperty()
    {
        // CSR invariant: rowPointers has rows+1 entries
        // rowPointers[0] = 0 and rowPointers[rows] = nnz
        var strategy = new MagnitudePruningStrategy<double>();

        var data = new double[] { 1, 0, 3, 0, 5, 6, 0, 0, 9 };
        var tensor = Tensor<double>.FromVector(new Vector<double>(data), new int[] { 3, 3 });

        var result = strategy.ToSparseFormat(tensor, SparseFormat.CSR);

        Assert.NotNull(result.RowPointers);
        Assert.Equal(4, result.RowPointers?.Length); // rows + 1
        Assert.Equal(0, result.RowPointers?[0]); // always starts at 0
        Assert.Equal(result.NonZeroCount, result.RowPointers?[3]); // last = nnz
    }

    #endregion

    #region CSC Sparse Format Conversion

    [Fact]
    public void MagnitudePruning_ToSparseFormat_CSC_CorrectStructure()
    {
        var strategy = new MagnitudePruningStrategy<double>();

        // Same 3x4 matrix as CSR test:
        // Row 0: [0, 1, 0, 2]
        // Row 1: [3, 0, 0, 0]
        // Row 2: [0, 0, 4, 5]
        var data = new double[] { 0, 1, 0, 2, 3, 0, 0, 0, 0, 0, 4, 5 };
        var tensor = Tensor<double>.FromVector(new Vector<double>(data), new int[] { 3, 4 });

        var result = strategy.ToSparseFormat(tensor, SparseFormat.CSC);

        Assert.Equal(SparseFormat.CSC, result.Format);

        // CSC scans column-by-column:
        // Col 0: row 1 has 3 => values: [3], rowIdx: [1]
        // Col 1: row 0 has 1 => values: [1], rowIdx: [0]
        // Col 2: row 2 has 4 => values: [4], rowIdx: [2]
        // Col 3: row 0 has 2, row 2 has 5 => values: [2, 5], rowIdx: [0, 2]
        Assert.Equal(5, result.Values.Length);
        Assert.Equal(3.0, result.Values[0], Tolerance);
        Assert.Equal(1.0, result.Values[1], Tolerance);
        Assert.Equal(4.0, result.Values[2], Tolerance);
        Assert.Equal(2.0, result.Values[3], Tolerance);
        Assert.Equal(5.0, result.Values[4], Tolerance);

        // Row indices (in column order)
        Assert.Equal(new[] { 1, 0, 2, 0, 2 }, result.RowIndices);

        // Column pointers: [0, 1, 2, 3, 5]
        Assert.Equal(new[] { 0, 1, 2, 3, 5 }, result.ColumnPointers);
    }

    [Fact]
    public void MagnitudePruning_ToSparseFormat_CSC_ColumnPointerProperty()
    {
        // CSC invariant: colPointers has cols+1 entries
        // colPointers[0] = 0 and colPointers[cols] = nnz
        var strategy = new MagnitudePruningStrategy<double>();

        var data = new double[] { 1, 0, 3, 0, 5, 6, 0, 0, 9 };
        var tensor = Tensor<double>.FromVector(new Vector<double>(data), new int[] { 3, 3 });

        var result = strategy.ToSparseFormat(tensor, SparseFormat.CSC);

        Assert.NotNull(result.ColumnPointers);
        Assert.Equal(4, result.ColumnPointers?.Length); // cols + 1
        Assert.Equal(0, result.ColumnPointers?[0]); // always starts at 0
        Assert.Equal(result.NonZeroCount, result.ColumnPointers?[3]); // last = nnz
    }

    #endregion

    #region N:M Sparse Format Conversion

    [Fact]
    public void MagnitudePruning_ToSparseFormat_2to4_CorrectMaskBits()
    {
        var strategy = new MagnitudePruningStrategy<double>();

        // Pre-pruned 2:4 data: [0, 1, 0, 2, 3, 0, 4, 0]
        var data = new double[] { 0, 1, 0, 2, 3, 0, 4, 0 };
        var tensor = Tensor<double>.FromVector(new Vector<double>(data), new int[] { 8 });

        var result = strategy.ToSparseFormat(tensor, SparseFormat.Structured2to4);

        Assert.Equal(SparseFormat.Structured2to4, result.Format);

        // Non-zero values: 1, 2, 3, 4
        Assert.Equal(4, result.Values.Length);
        Assert.Equal(1.0, result.Values[0], Tolerance);
        Assert.Equal(2.0, result.Values[1], Tolerance);
        Assert.Equal(3.0, result.Values[2], Tolerance);
        Assert.Equal(4.0, result.Values[3], Tolerance);

        // Sparsity mask for groups of 4:
        // Group 0: [0, 1, 0, 2] => bits at positions 1,3 set => 0b1010 = 10
        // Group 1: [3, 0, 4, 0] => bits at positions 0,2 set => 0b0101 = 5
        Assert.NotNull(result.SparsityMask);
        Assert.Equal(2, result.SparsityMask?.Length);
        Assert.Equal((byte)10, result.SparsityMask?[0]); // 0b1010
        Assert.Equal((byte)5, result.SparsityMask?[1]);   // 0b0101
    }

    [Fact]
    public void MagnitudePruning_ToSparseFormat_NtoM_MaskBitEncoding()
    {
        var strategy = new MagnitudePruningStrategy<double>();

        // Group: [0, 0, 3, 4] => bits at positions 2,3 => 0b1100 = 12
        var data = new double[] { 0, 0, 3, 4 };
        var tensor = Tensor<double>.FromVector(new Vector<double>(data), new int[] { 4 });

        var result = strategy.ToSparseFormat(tensor, SparseFormat.StructuredNtoM);

        Assert.NotNull(result.SparsityMask);
        Assert.Equal(1, result.SparsityMask?.Length);
        Assert.Equal((byte)12, result.SparsityMask?[0]); // bit 2 (1<<2=4) + bit 3 (1<<3=8) = 12
    }

    #endregion

    #region End-to-End Pruning Pipeline

    [Fact]
    public void PruningPipeline_ComputeScores_CreateMask_Apply_CorrectResult()
    {
        var strategy = new MagnitudePruningStrategy<double>();

        // Original weights
        var weights = new Vector<double>(new double[] { -0.1, 0.5, -0.3, 0.9, 0.05, -0.7 });

        // Step 1: Compute importance scores = |w|
        var scores = strategy.ComputeImportanceScores(weights);
        Assert.Equal(0.1, scores[0], Tolerance);
        Assert.Equal(0.5, scores[1], Tolerance);
        Assert.Equal(0.3, scores[2], Tolerance);
        Assert.Equal(0.9, scores[3], Tolerance);
        Assert.Equal(0.05, scores[4], Tolerance);
        Assert.Equal(0.7, scores[5], Tolerance);

        // Step 2: Create mask with ~50% sparsity
        // 6 non-zero elements, Round(6 * 0.5) = 3 pruned
        // Sorted ascending: 0.05(idx4), 0.1(idx0), 0.3(idx2), 0.5(idx1), 0.7(idx5), 0.9(idx3)
        // Prune: idx 4, 0, 2
        var mask = strategy.CreateMask(scores, 0.5);

        var maskData = mask.GetMaskData();
        Assert.Equal(0.0, maskData[0], Tolerance); // 0.1 pruned
        Assert.Equal(1.0, maskData[1], Tolerance); // 0.5 kept
        Assert.Equal(0.0, maskData[2], Tolerance); // 0.3 pruned
        Assert.Equal(1.0, maskData[3], Tolerance); // 0.9 kept
        Assert.Equal(0.0, maskData[4], Tolerance); // 0.05 pruned
        Assert.Equal(1.0, maskData[5], Tolerance); // 0.7 kept

        // Step 3: Apply mask to original weights
        strategy.ApplyPruning(weights, mask);

        Assert.Equal(0.0, weights[0], Tolerance);   // was -0.1, pruned
        Assert.Equal(0.5, weights[1], Tolerance);    // kept
        Assert.Equal(0.0, weights[2], Tolerance);    // was -0.3, pruned
        Assert.Equal(0.9, weights[3], Tolerance);    // kept
        Assert.Equal(0.0, weights[4], Tolerance);    // was 0.05, pruned
        Assert.Equal(-0.7, weights[5], Tolerance);   // kept (sign preserved)
    }

    [Fact]
    public void PruningPipeline_Matrix_PruneAndConvertToCSR()
    {
        var strategy = new MagnitudePruningStrategy<double>();

        // 2x3 weight matrix
        var weights = new Matrix<double>(2, 3);
        weights[0, 0] = -0.1;  weights[0, 1] = 0.8;  weights[0, 2] = -0.05;
        weights[1, 0] = 0.6;   weights[1, 1] = -0.02; weights[1, 2] = 0.9;

        // Compute scores and create mask at 50% sparsity
        var scores = strategy.ComputeImportanceScores(weights);
        var mask = strategy.CreateMask(scores, 0.5);

        // Apply pruning
        strategy.ApplyPruning(weights, mask);

        // (int)(6 * 0.5) = 3 pruned: the 3 smallest magnitudes are:
        // 0.02(idx (1,1)), 0.05(idx (0,2)), 0.1(idx (0,0))
        Assert.Equal(0.0, weights[0, 0], Tolerance);  // 0.1 pruned
        Assert.Equal(0.8, weights[0, 1], Tolerance);   // kept
        Assert.Equal(0.0, weights[0, 2], Tolerance);   // 0.05 pruned
        Assert.Equal(0.6, weights[1, 0], Tolerance);   // kept
        Assert.Equal(0.0, weights[1, 1], Tolerance);   // 0.02 pruned
        Assert.Equal(0.9, weights[1, 2], Tolerance);   // kept

        // Convert to CSR
        var data = new double[6];
        for (int i = 0; i < 2; i++)
            for (int j = 0; j < 3; j++)
                data[i * 3 + j] = weights[i, j];
        var tensor = Tensor<double>.FromVector(new Vector<double>(data), new int[] { 2, 3 });
        var csr = strategy.ToSparseFormat(tensor, SparseFormat.CSR);

        // 3 non-zero values
        Assert.Equal(3, csr.NonZeroCount);
        Assert.Equal(0.5, csr.Sparsity, Tolerance); // 3/6 = 0.5

        // Row 0: 0.8 at col 1
        // Row 1: 0.6 at col 0, 0.9 at col 2
        Assert.Equal(new[] { 0, 1, 3 }, csr.RowPointers);
        Assert.Equal(new[] { 1, 0, 2 }, csr.ColumnIndices);
    }

    [Fact]
    public void PruningPipeline_IterativePruning_IncreasingSparsity()
    {
        var strategy = new MagnitudePruningStrategy<double>();

        // Start with 10 non-zero weights
        var weights = new Vector<double>(new double[]
            { 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0 });

        // Round 1: 30% sparsity
        var scores1 = strategy.ComputeImportanceScores(weights);
        var mask1 = strategy.CreateMask(scores1, 0.3);
        strategy.ApplyPruning(weights, mask1);

        // Round(10 * 0.3) = 3 pruned: 0.1, 0.2, 0.3
        int round1Zeros = 0;
        for (int i = 0; i < weights.Length; i++)
            if (Math.Abs(weights[i]) < Tolerance) round1Zeros++;
        Assert.Equal(3, round1Zeros);

        // Round 2: 30% of remaining non-zero (7 remaining)
        var scores2 = strategy.ComputeImportanceScores(weights);
        var mask2 = strategy.CreateMask(scores2, 0.3);
        strategy.ApplyPruning(weights, mask2);

        // Round(7 * 0.3) = 2 more pruned: 0.4, 0.5
        int round2Zeros = 0;
        for (int i = 0; i < weights.Length; i++)
            if (Math.Abs(weights[i]) < Tolerance) round2Zeros++;
        Assert.Equal(5, round2Zeros); // 3 + 2 = 5 total

        // The 5 smallest original weights should be zeroed
        Assert.Equal(0.0, weights[0], Tolerance); // 0.1
        Assert.Equal(0.0, weights[1], Tolerance); // 0.2
        Assert.Equal(0.0, weights[2], Tolerance); // 0.3
        Assert.Equal(0.0, weights[3], Tolerance); // 0.4
        Assert.Equal(0.0, weights[4], Tolerance); // 0.5

        // The 5 largest should be kept
        Assert.Equal(0.6, weights[5], Tolerance);
        Assert.Equal(0.7, weights[6], Tolerance);
        Assert.Equal(0.8, weights[7], Tolerance);
        Assert.Equal(0.9, weights[8], Tolerance);
        Assert.Equal(1.0, weights[9], Tolerance);
    }

    #endregion

    #region SparseCompressionResult Properties

    [Fact]
    public void SparseCompressionResult_Sparsity_ExactComputation()
    {
        // 2x3 = 6 total, 2 non-zero => sparsity = 1 - 2/6 = 4/6 = 2/3
        var result = new SparseCompressionResult<double>
        {
            Format = SparseFormat.COO,
            Values = new double[] { 1.0, 2.0 },
            RowIndices = new[] { 0, 1 },
            ColumnIndices = new[] { 1, 2 },
            OriginalShape = new[] { 2, 3 }
        };

        Assert.Equal(2, result.NonZeroCount);
        Assert.Equal(2.0 / 3.0, result.Sparsity, Tolerance);
    }

    [Fact]
    public void SparseCompressionResult_CompressedSizeBytes_Calculation()
    {
        var result = new SparseCompressionResult<double>
        {
            Format = SparseFormat.CSR,
            Values = new double[] { 1.0, 2.0, 3.0 },
            ColumnIndices = new[] { 1, 0, 2 },
            RowPointers = new[] { 0, 1, 3 },
            OriginalShape = new[] { 2, 3 }
        };

        // elementSize = 8 bytes (double)
        long size = result.GetCompressedSizeBytes(8);

        // Values: 3 * 8 = 24
        // ColumnIndices: 3 * 4 = 12
        // RowPointers: 3 * 4 = 12 (actually 3 entries for 2 rows + 1, but wait, the array has 3 entries)
        // Metadata: (2 + 4) * 4 = 24 (shape length 2 + 4 metadata ints)
        // Total = 24 + 12 + 12 + 24 = 72
        Assert.Equal(72, size);
    }

    #endregion

    #region Edge Cases

    [Fact]
    public void PruningMask_SingleElement_SparsityZeroOrOne()
    {
        // 1-element mask kept
        var kept = new PruningMask<double>(new bool[] { true });
        Assert.Equal(0.0, kept.GetSparsity(), Tolerance);

        // 1-element mask pruned
        var pruned = new PruningMask<double>(new bool[] { false });
        Assert.Equal(1.0, pruned.GetSparsity(), Tolerance);
    }

    [Fact]
    public void MagnitudePruning_AllSameWeights_PrunesArbitrarily()
    {
        // All weights have same magnitude => any subset can be pruned
        var strategy = new MagnitudePruningStrategy<double>();
        var weights = new Vector<double>(new double[] { 0.5, 0.5, 0.5, 0.5 });
        var scores = strategy.ComputeImportanceScores(weights);

        var mask = strategy.CreateMask(scores, 0.5);

        // Should prune exactly Round(4 * 0.5) = 2 elements
        var maskData = mask.GetMaskData();
        int prunedCount = maskData.Count(v => Math.Abs(v) < Tolerance);
        Assert.Equal(2, prunedCount);
    }

    [Fact]
    public void MagnitudePruning_AllZeroWeights_HandlesGracefully()
    {
        var strategy = new MagnitudePruningStrategy<double>();
        var weights = new Vector<double>(new double[] { 0.0, 0.0, 0.0, 0.0 });
        var scores = strategy.ComputeImportanceScores(weights);

        // All zero scores: special handling in CreateMask
        var mask = strategy.CreateMask(scores, 0.5);

        // Should have ~50% sparsity
        // Code: numToPrune = Round(4 * 0.5) = 2
        var maskData = mask.GetMaskData();
        int prunedCount = maskData.Count(v => Math.Abs(v) < Tolerance);
        Assert.Equal(2, prunedCount);
    }

    [Fact]
    public void PruningMask_GetMaskData_RowMajorFlatOrder()
    {
        // Verify GetMaskData returns in row-major order
        var keepIndices = new bool[,]
        {
            { true, false },
            { false, true },
            { true, true }
        };
        var mask = new PruningMask<double>(keepIndices);

        var data = mask.GetMaskData();

        // Row-major: (0,0)=1, (0,1)=0, (1,0)=0, (1,1)=1, (2,0)=1, (2,1)=1
        Assert.Equal(6, data.Length);
        Assert.Equal(1.0, data[0], Tolerance);
        Assert.Equal(0.0, data[1], Tolerance);
        Assert.Equal(0.0, data[2], Tolerance);
        Assert.Equal(1.0, data[3], Tolerance);
        Assert.Equal(1.0, data[4], Tolerance);
        Assert.Equal(1.0, data[5], Tolerance);
    }

    #endregion
}
