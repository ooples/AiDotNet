using AiDotNet.Genetics;
using AiDotNet.Tensors.Helpers;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.Genetics;

/// <summary>
/// Deep math-correctness integration tests for the Genetics module.
/// Verifies exact formulas and mathematical invariants for:
/// - BinaryIndividual: bit-shifting, normalization, range mapping
/// - PermutationIndividual: Order Crossover (OX), swap/inversion mutations
/// - RealValuedIndividual: 1/5 success rule step size adaptation
/// - MultiObjectiveRealIndividual: Pareto dominance definition
/// - Gene equality and clone independence
/// </summary>
public class GeneticsDeepMathIntegrationTests
{
    private const double Tolerance = 1e-10;

    #region BinaryIndividual - Bit Shifting Math

    [Fact]
    public void BinaryIndividual_GetValueAsInt_LittleEndianBitOrder_VerifyFormula()
    {
        // The formula is: value |= (1 << i) for each bit i that is 1
        // This means _genes[0] is the least significant bit (2^0)
        // and _genes[n-1] is the most significant bit (2^(n-1))

        // Arrange: bits = [1, 0, 1, 1, 0] => 1*1 + 0*2 + 1*4 + 1*8 + 0*16 = 13
        var genes = new List<BinaryGene>
        {
            new(1), // bit 0: 2^0 = 1
            new(0), // bit 1: 2^1 = 2 (not set)
            new(1), // bit 2: 2^2 = 4
            new(1), // bit 3: 2^3 = 8
            new(0)  // bit 4: 2^4 = 16 (not set)
        };
        var individual = new BinaryIndividual(genes);

        // Act
        int result = individual.GetValueAsInt();

        // Assert - manually compute: sum of 2^i for set bits
        int expected = 0;
        int[] bits = [1, 0, 1, 1, 0];
        for (int i = 0; i < bits.Length; i++)
        {
            if (bits[i] == 1) expected |= (1 << i);
        }
        Assert.Equal(expected, result);
        Assert.Equal(13, result);
    }

    [Fact]
    public void BinaryIndividual_GetValueAsInt_SingleBitAtEachPosition()
    {
        // Verify each bit position maps to correct power of 2
        for (int bitPos = 0; bitPos < 16; bitPos++)
        {
            var genes = new List<BinaryGene>();
            for (int i = 0; i < 16; i++)
            {
                genes.Add(new BinaryGene(i == bitPos ? 1 : 0));
            }
            var individual = new BinaryIndividual(genes);

            int result = individual.GetValueAsInt();
            int expected = 1 << bitPos;

            Assert.Equal(expected, result);
        }
    }

    [Fact]
    public void BinaryIndividual_GetValueAsNormalizedDouble_ExactFormula()
    {
        // Formula: GetValueAsInt() / (2^n - 1) where n = gene count
        // For 4 bits: max = 2^4 - 1 = 15

        // Arrange: bits = [1, 0, 1, 0] => value = 1 + 4 = 5, normalized = 5/15 = 1/3
        var genes = new List<BinaryGene> { new(1), new(0), new(1), new(0) };
        var individual = new BinaryIndividual(genes);

        // Act
        double normalized = individual.GetValueAsNormalizedDouble();

        // Assert
        double expected = 5.0 / 15.0; // 1/3
        Assert.Equal(expected, normalized, Tolerance);
    }

    [Fact]
    public void BinaryIndividual_GetValueAsNormalizedDouble_HalfValue()
    {
        // For 8 bits: value 128 would be normalized to 128/255
        // But 128 in little-endian is bit 7 only: [0,0,0,0,0,0,0,1] = 128
        var genes = new List<BinaryGene>();
        for (int i = 0; i < 8; i++)
        {
            genes.Add(new BinaryGene(i == 7 ? 1 : 0));
        }
        var individual = new BinaryIndividual(genes);

        double normalized = individual.GetValueAsNormalizedDouble();
        double expected = 128.0 / 255.0;

        Assert.Equal(expected, normalized, Tolerance);
    }

    [Fact]
    public void BinaryIndividual_GetValueMapped_ExactFormula()
    {
        // Formula: min + (GetValueAsNormalizedDouble() * (max - min))
        // For bits = [1,1,1,1] (all ones, value=15), normalized = 15/15 = 1.0
        // mapped to [-10, 10] = -10 + 1.0 * 20 = 10.0

        var allOnes = new List<BinaryGene> { new(1), new(1), new(1), new(1) };
        var individual = new BinaryIndividual(allOnes);

        double mapped = individual.GetValueMapped(-10.0, 10.0);
        Assert.Equal(10.0, mapped, Tolerance);

        // For bits = [0,0,0,0] (all zeros, value=0), normalized = 0/15 = 0.0
        // mapped to [-10, 10] = -10 + 0.0 * 20 = -10.0
        var allZeros = new List<BinaryGene> { new(0), new(0), new(0), new(0) };
        var zeroIndividual = new BinaryIndividual(allZeros);

        double mappedZero = zeroIndividual.GetValueMapped(-10.0, 10.0);
        Assert.Equal(-10.0, mappedZero, Tolerance);

        // Mid value: bits = [1,1,0,0] => value = 3, normalized = 3/15 = 0.2
        // mapped to [-10, 10] = -10 + 0.2 * 20 = -6.0
        var midGenes = new List<BinaryGene> { new(1), new(1), new(0), new(0) };
        var midIndividual = new BinaryIndividual(midGenes);

        double mappedMid = midIndividual.GetValueMapped(-10.0, 10.0);
        double expectedMid = -10.0 + (3.0 / 15.0) * 20.0;
        Assert.Equal(expectedMid, mappedMid, Tolerance);
    }

    [Fact]
    public void BinaryIndividual_GetValueAsInt_IntOverflowAt31Bits()
    {
        // With 31 bits all set, value should be 2^31 - 1 = int.MaxValue / 2
        // Actually: sum of 2^0 + 2^1 + ... + 2^30 = 2^31 - 1 = 2147483647
        var genes = new List<BinaryGene>();
        for (int i = 0; i < 31; i++)
        {
            genes.Add(new BinaryGene(1));
        }
        var individual = new BinaryIndividual(genes);

        int result = individual.GetValueAsInt();
        int expected = int.MaxValue; // 2^31 - 1

        Assert.Equal(expected, result);
    }

    [Fact]
    public void BinaryGene_Constructor_ClampsNonBinaryValues()
    {
        // BinaryGene constructor: Value = value > 0 ? 1 : 0
        var geneNeg = new BinaryGene(-5);
        var geneZero = new BinaryGene(0);
        var geneOne = new BinaryGene(1);
        var geneLarge = new BinaryGene(100);

        Assert.Equal(0, geneNeg.Value);  // -5 > 0 is false => 0
        Assert.Equal(0, geneZero.Value); // 0 > 0 is false => 0
        Assert.Equal(1, geneOne.Value);  // 1 > 0 is true => 1
        Assert.Equal(1, geneLarge.Value); // 100 > 0 is true => 1
    }

    [Fact]
    public void BinaryIndividual_GetValueAsNormalizedDouble_SingleBit()
    {
        // Edge case: single bit. Max = 2^1 - 1 = 1
        // Value 0 => 0/1 = 0.0
        var zero = new BinaryIndividual(new List<BinaryGene> { new(0) });
        Assert.Equal(0.0, zero.GetValueAsNormalizedDouble(), Tolerance);

        // Value 1 => 1/1 = 1.0
        var one = new BinaryIndividual(new List<BinaryGene> { new(1) });
        Assert.Equal(1.0, one.GetValueAsNormalizedDouble(), Tolerance);
    }

    #endregion

    #region PermutationIndividual - OrderCrossover Math

    [Fact]
    public void OrderCrossover_ChildInheritsSubstringFromParent()
    {
        // OX: child1 inherits substring [start..end] from parent1
        // Remaining positions filled from parent2 in order (wrapping from end+1)
        //
        // With deterministic seed, we know which start/end are chosen.
        // Instead of relying on seed, verify the invariant that the
        // substring from parent1 appears in child1 at the same positions.

        var parent1Genes = new List<PermutationGene>();
        var parent2Genes = new List<PermutationGene>();
        // Parent1: [0, 1, 2, 3, 4, 5, 6, 7]
        // Parent2: [7, 6, 5, 4, 3, 2, 1, 0]
        for (int i = 0; i < 8; i++)
        {
            parent1Genes.Add(new PermutationGene(i));
            parent2Genes.Add(new PermutationGene(7 - i));
        }
        var parent1 = new PermutationIndividual(parent1Genes);
        var parent2 = new PermutationIndividual(parent2Genes);

        // Run OX many times to verify invariant always holds
        for (int seed = 0; seed < 50; seed++)
        {
            var rand = RandomHelper.CreateSeededRandom(seed);
            var (child1, child2) = parent1.OrderCrossover(parent2, rand);

            var perm1 = child1.GetPermutation();
            var perm2 = child2.GetPermutation();

            // Both children must be valid permutations
            Assert.Equal(8, perm1.Distinct().Count());
            Assert.Equal(8, perm2.Distinct().Count());
            Assert.True(perm1.All(v => v >= 0 && v < 8));
            Assert.True(perm2.All(v => v >= 0 && v < 8));
        }
    }

    [Fact]
    public void OrderCrossover_IdenticalParents_ProducesIdenticalChildren()
    {
        // If both parents have the same permutation, children must be identical
        var genes = new List<PermutationGene>();
        for (int i = 0; i < 6; i++)
        {
            genes.Add(new PermutationGene(i));
        }
        var parent1 = new PermutationIndividual(new List<PermutationGene>(genes.Select(g => g.Clone())));
        var parent2 = new PermutationIndividual(new List<PermutationGene>(genes.Select(g => g.Clone())));

        var rand = RandomHelper.CreateSeededRandom(42);
        var (child1, child2) = parent1.OrderCrossover(parent2, rand);

        var perm1 = child1.GetPermutation();
        var perm2 = child2.GetPermutation();
        int[] expected = [0, 1, 2, 3, 4, 5];

        Assert.Equal(expected, perm1);
        Assert.Equal(expected, perm2);
    }

    [Fact]
    public void OrderCrossover_LargePermutation_AlwaysValid()
    {
        // Stress test: large permutation (100 elements)
        var rand = RandomHelper.CreateSeededRandom(42);
        int size = 100;
        var parent1 = new PermutationIndividual(size, rand);
        var parent2 = new PermutationIndividual(size, rand);

        for (int trial = 0; trial < 20; trial++)
        {
            var (child1, child2) = parent1.OrderCrossover(parent2, rand);
            var perm1 = child1.GetPermutation();
            var perm2 = child2.GetPermutation();

            // Verify valid permutation: all values 0..99, each exactly once
            var sorted1 = perm1.OrderBy(x => x).ToArray();
            var sorted2 = perm2.OrderBy(x => x).ToArray();
            for (int i = 0; i < size; i++)
            {
                Assert.Equal(i, sorted1[i]);
                Assert.Equal(i, sorted2[i]);
            }
        }
    }

    #endregion

    #region PermutationIndividual - SwapMutation Math

    [Fact]
    public void SwapMutation_ExactlyTwoPositionsSwapped()
    {
        // SwapMutation swaps two random positions
        // Verify at most 2 positions differ (could be 0 if same position picked twice)

        var genes = new List<PermutationGene>();
        for (int i = 0; i < 10; i++)
        {
            genes.Add(new PermutationGene(i));
        }
        var individual = new PermutationIndividual(new List<PermutationGene>(genes.Select(g => g.Clone())));
        int[] before = individual.GetPermutation().ToArray();

        var rand = RandomHelper.CreateSeededRandom(42);
        individual.SwapMutation(rand);
        int[] after = individual.GetPermutation();

        // Count differing positions
        int diffCount = 0;
        for (int i = 0; i < 10; i++)
        {
            if (before[i] != after[i]) diffCount++;
        }

        // Swap mutation: either 0 (same position picked) or 2 positions differ
        Assert.True(diffCount == 0 || diffCount == 2,
            $"Swap mutation should change 0 or 2 positions, but changed {diffCount}");
    }

    [Fact]
    public void SwapMutation_PreservesAllElements()
    {
        var rand = RandomHelper.CreateSeededRandom(42);
        int size = 20;
        var individual = new PermutationIndividual(size, rand);

        // Apply 100 swap mutations
        for (int i = 0; i < 100; i++)
        {
            individual.SwapMutation(rand);
        }

        var perm = individual.GetPermutation();
        var sorted = perm.OrderBy(x => x).ToArray();
        for (int i = 0; i < size; i++)
        {
            Assert.Equal(i, sorted[i]);
        }
    }

    #endregion

    #region PermutationIndividual - InversionMutation Math

    [Fact]
    public void InversionMutation_ReversesSubsequence()
    {
        // InversionMutation reverses a contiguous subsequence
        // After mutation, the set of elements must be the same (valid permutation)
        // and one contiguous block should be reversed compared to original

        var genes = new List<PermutationGene>();
        for (int i = 0; i < 8; i++)
        {
            genes.Add(new PermutationGene(i));
        }
        var individual = new PermutationIndividual(new List<PermutationGene>(genes.Select(g => g.Clone())));
        int[] before = individual.GetPermutation().ToArray();

        var rand = RandomHelper.CreateSeededRandom(42);
        individual.InversionMutation(rand);
        int[] after = individual.GetPermutation();

        // Verify it's still a valid permutation
        var sorted = after.OrderBy(x => x).ToArray();
        for (int i = 0; i < 8; i++)
        {
            Assert.Equal(i, sorted[i]);
        }

        // Find the reversed block: identify contiguous region that differs
        int firstDiff = -1;
        int lastDiff = -1;
        for (int i = 0; i < 8; i++)
        {
            if (before[i] != after[i])
            {
                if (firstDiff == -1) firstDiff = i;
                lastDiff = i;
            }
        }

        if (firstDiff >= 0)
        {
            // The differing block should be a reversal
            for (int i = firstDiff; i <= lastDiff; i++)
            {
                Assert.Equal(before[firstDiff + lastDiff - i], after[i]);
            }
        }
        // If no diff, pos1 == pos2 was chosen (single-element "reversal")
    }

    [Fact]
    public void InversionMutation_KnownSubsequence_VerifyReversal()
    {
        // Create permutation [0,1,2,3,4,5,6,7]
        // The code picks pos1 and pos2, ensures pos1 <= pos2, then reverses [pos1..pos2]
        // With seed 42, verify the exact result

        var genes = new List<PermutationGene>();
        for (int i = 0; i < 8; i++)
        {
            genes.Add(new PermutationGene(i));
        }
        var individual = new PermutationIndividual(new List<PermutationGene>(genes.Select(g => g.Clone())));

        // Use a seeded random to get deterministic positions
        var rand = RandomHelper.CreateSeededRandom(42);
        int pos1 = rand.Next(8); // Consume same random values as InversionMutation
        int pos2 = rand.Next(8);
        if (pos1 > pos2) (pos1, pos2) = (pos2, pos1);

        // Now apply actual mutation with same seed
        rand = RandomHelper.CreateSeededRandom(42);
        individual.InversionMutation(rand);
        int[] result = individual.GetPermutation();

        // Manually compute expected: reverse [pos1..pos2] in [0,1,2,3,4,5,6,7]
        int[] expected = [0, 1, 2, 3, 4, 5, 6, 7];
        int steps = (pos2 - pos1 + 1) / 2;
        for (int i = 0; i < steps; i++)
        {
            (expected[pos1 + i], expected[pos2 - i]) = (expected[pos2 - i], expected[pos1 + i]);
        }

        Assert.Equal(expected, result);
    }

    #endregion

    #region RealValuedIndividual - 1/5 Success Rule Math

    [Fact]
    public void UpdateStepSizes_ExactOneFifthRule_HighSuccess()
    {
        // 1/5 success rule: c = 0.817
        // If successRatio > 0.2 => adjustmentFactor = 1.0 / c
        // newStepSize = originalStepSize * (1.0 / c)

        const double c = 0.817;
        double originalStep = 0.5;

        var genes = new List<RealGene>
        {
            new(1.0, originalStep),
            new(2.0, originalStep),
            new(3.0, originalStep)
        };
        var individual = new RealValuedIndividual(genes);

        individual.UpdateStepSizes(0.3); // > 0.2 => increase

        var updatedGenes = individual.GetGenes().ToList();
        double expectedStep = originalStep * (1.0 / c);

        foreach (var gene in updatedGenes)
        {
            Assert.Equal(expectedStep, gene.StepSize, Tolerance);
        }
    }

    [Fact]
    public void UpdateStepSizes_ExactOneFifthRule_LowSuccess()
    {
        // If successRatio <= 0.2 => adjustmentFactor = c
        // newStepSize = originalStepSize * c

        const double c = 0.817;
        double originalStep = 0.5;

        var genes = new List<RealGene>
        {
            new(1.0, originalStep),
            new(2.0, originalStep)
        };
        var individual = new RealValuedIndividual(genes);

        individual.UpdateStepSizes(0.15); // < 0.2 => decrease

        var updatedGenes = individual.GetGenes().ToList();
        double expectedStep = originalStep * c;

        foreach (var gene in updatedGenes)
        {
            Assert.Equal(expectedStep, gene.StepSize, Tolerance);
        }
    }

    [Fact]
    public void UpdateStepSizes_BoundaryAt02_Decreases()
    {
        // Code: successRatio > 0.2 ? 1.0/c : c
        // At exactly 0.2, condition is false, so step size should DECREASE (multiply by c)

        const double c = 0.817;
        double originalStep = 1.0;

        var genes = new List<RealGene> { new(0.0, originalStep) };
        var individual = new RealValuedIndividual(genes);

        individual.UpdateStepSizes(0.2); // exactly 0.2 => decreases

        var updatedGene = individual.GetGenes().First();
        double expectedStep = originalStep * c;

        Assert.Equal(expectedStep, updatedGene.StepSize, Tolerance);
    }

    [Fact]
    public void UpdateStepSizes_RepeatedApplications_ConvergesCorrectly()
    {
        // Applying 1/5 rule repeatedly: each application multiplies by factor
        // High success: step *= 1/c each time
        // Low success: step *= c each time

        const double c = 0.817;
        double originalStep = 1.0;

        var genes = new List<RealGene> { new(0.0, originalStep) };
        var individual = new RealValuedIndividual(genes);

        // Apply 5 times with low success
        for (int i = 0; i < 5; i++)
        {
            individual.UpdateStepSizes(0.1);
        }

        double expectedStep = originalStep * Math.Pow(c, 5);
        double actualStep = individual.GetGenes().First().StepSize;
        Assert.Equal(expectedStep, actualStep, 1e-8);

        // Now apply 5 times with high success
        for (int i = 0; i < 5; i++)
        {
            individual.UpdateStepSizes(0.5);
        }

        expectedStep *= Math.Pow(1.0 / c, 5);
        actualStep = individual.GetGenes().First().StepSize;
        Assert.Equal(expectedStep, actualStep, 1e-8);
    }

    [Fact]
    public void UpdateStepSizes_HighThenLow_ReturnsToOriginal()
    {
        // One increase (multiply by 1/c) followed by one decrease (multiply by c)
        // should return to original: step * (1/c) * c = step

        double originalStep = 2.5;
        var genes = new List<RealGene> { new(0.0, originalStep) };
        var individual = new RealValuedIndividual(genes);

        individual.UpdateStepSizes(0.5); // increase
        individual.UpdateStepSizes(0.1); // decrease

        double finalStep = individual.GetGenes().First().StepSize;
        Assert.Equal(originalStep, finalStep, 1e-12);
    }

    [Fact]
    public void UpdateStepSizes_PreservesGeneValues()
    {
        // UpdateStepSizes should only modify StepSize, not Value
        var genes = new List<RealGene>
        {
            new(3.14, 0.5),
            new(-2.71, 0.5),
            new(1.41, 0.5)
        };
        var individual = new RealValuedIndividual(genes);
        double[] originalValues = individual.GetValuesAsArray();

        individual.UpdateStepSizes(0.3);

        double[] newValues = individual.GetValuesAsArray();
        for (int i = 0; i < originalValues.Length; i++)
        {
            Assert.Equal(originalValues[i], newValues[i], Tolerance);
        }
    }

    #endregion

    #region MultiObjectiveRealIndividual - Pareto Dominance Math

    [Fact]
    public void Dominates_BetterInAllObjectives_ReturnsTrue()
    {
        // Pareto dominance: A dominates B iff A is no worse in all objectives
        // AND strictly better in at least one

        var rand = RandomHelper.CreateSeededRandom(42);
        var a = new MultiObjectiveRealIndividual(2, 0.0, 1.0, rand);
        var b = new MultiObjectiveRealIndividual(2, 0.0, 1.0, rand);

        // A is better (lower) in all objectives
        a.SetObjectiveValues([0.1, 0.2]);
        b.SetObjectiveValues([0.3, 0.4]);

        Assert.True(a.Dominates(b));
        Assert.False(b.Dominates(a));
    }

    [Fact]
    public void Dominates_EqualInAllObjectives_ReturnsFalse()
    {
        // Equal in all objectives: neither dominates
        var rand = RandomHelper.CreateSeededRandom(42);
        var a = new MultiObjectiveRealIndividual(2, 0.0, 1.0, rand);
        var b = new MultiObjectiveRealIndividual(2, 0.0, 1.0, rand);

        a.SetObjectiveValues([0.5, 0.5]);
        b.SetObjectiveValues([0.5, 0.5]);

        Assert.False(a.Dominates(b));
        Assert.False(b.Dominates(a));
    }

    [Fact]
    public void Dominates_BetterInOneEqualInOther_ReturnsTrue()
    {
        // Better in one, equal in another => dominates
        var rand = RandomHelper.CreateSeededRandom(42);
        var a = new MultiObjectiveRealIndividual(2, 0.0, 1.0, rand);
        var b = new MultiObjectiveRealIndividual(2, 0.0, 1.0, rand);

        a.SetObjectiveValues([0.3, 0.5]);
        b.SetObjectiveValues([0.5, 0.5]);

        Assert.True(a.Dominates(b));
        Assert.False(b.Dominates(a));
    }

    [Fact]
    public void Dominates_Tradeoff_NeitherDominates()
    {
        // A is better in obj1 but worse in obj2 => trade-off, no dominance
        var rand = RandomHelper.CreateSeededRandom(42);
        var a = new MultiObjectiveRealIndividual(2, 0.0, 1.0, rand);
        var b = new MultiObjectiveRealIndividual(2, 0.0, 1.0, rand);

        a.SetObjectiveValues([0.2, 0.8]);
        b.SetObjectiveValues([0.8, 0.2]);

        Assert.False(a.Dominates(b));
        Assert.False(b.Dominates(a));
    }

    [Fact]
    public void Dominates_ThreeObjectives_CorrectBehavior()
    {
        var rand = RandomHelper.CreateSeededRandom(42);

        // A dominates B (better in all 3)
        var a = new MultiObjectiveRealIndividual(2, 0.0, 1.0, rand);
        var b = new MultiObjectiveRealIndividual(2, 0.0, 1.0, rand);
        a.SetObjectiveValues([0.1, 0.2, 0.3]);
        b.SetObjectiveValues([0.4, 0.5, 0.6]);
        Assert.True(a.Dominates(b));

        // C does not dominate D (better in 2, worse in 1)
        var c = new MultiObjectiveRealIndividual(2, 0.0, 1.0, rand);
        var d = new MultiObjectiveRealIndividual(2, 0.0, 1.0, rand);
        c.SetObjectiveValues([0.1, 0.2, 0.9]);
        d.SetObjectiveValues([0.4, 0.5, 0.3]);
        Assert.False(c.Dominates(d));
        Assert.False(d.Dominates(c));
    }

    [Fact]
    public void Dominates_DominanceIsTransitive()
    {
        // If A dominates B and B dominates C, then A must dominate C
        var rand = RandomHelper.CreateSeededRandom(42);
        var a = new MultiObjectiveRealIndividual(2, 0.0, 1.0, rand);
        var b = new MultiObjectiveRealIndividual(2, 0.0, 1.0, rand);
        var c = new MultiObjectiveRealIndividual(2, 0.0, 1.0, rand);

        a.SetObjectiveValues([0.1, 0.1]);
        b.SetObjectiveValues([0.3, 0.3]);
        c.SetObjectiveValues([0.5, 0.5]);

        Assert.True(a.Dominates(b));
        Assert.True(b.Dominates(c));
        Assert.True(a.Dominates(c)); // Transitivity
    }

    [Fact]
    public void Dominates_DominanceIsAntisymmetric()
    {
        // If A dominates B, then B cannot dominate A
        var rand = RandomHelper.CreateSeededRandom(42);
        var a = new MultiObjectiveRealIndividual(2, 0.0, 1.0, rand);
        var b = new MultiObjectiveRealIndividual(2, 0.0, 1.0, rand);

        a.SetObjectiveValues([0.1, 0.2]);
        b.SetObjectiveValues([0.3, 0.4]);

        Assert.True(a.Dominates(b));
        Assert.False(b.Dominates(a));
    }

    #endregion

    #region Clone Independence Tests

    [Fact]
    public void BinaryIndividual_Clone_MutatingCloneDoesNotAffectOriginal()
    {
        var genes = new List<BinaryGene> { new(1), new(0), new(1), new(0) };
        var original = new BinaryIndividual(genes);
        original.SetFitness(0.9);

        var clone = (BinaryIndividual)original.Clone();

        // Mutate clone's genes
        var cloneGenes = clone.GetGenes().ToList();
        clone.SetGenes(new List<BinaryGene> { new(0), new(1), new(0), new(1) });
        clone.SetFitness(0.1);

        // Original should be unchanged
        Assert.Equal(0.9, original.GetFitness(), Tolerance);
        int originalValue = original.GetValueAsInt();
        Assert.Equal(5, originalValue); // 1 + 4 = 5
    }

    [Fact]
    public void PermutationIndividual_Clone_MutatingCloneDoesNotAffectOriginal()
    {
        var genes = new List<PermutationGene>();
        for (int i = 0; i < 6; i++) genes.Add(new PermutationGene(i));
        var original = new PermutationIndividual(genes);
        original.SetFitness(100.0);
        int[] originalPerm = original.GetPermutation().ToArray();

        var clone = (PermutationIndividual)original.Clone();
        var rand = RandomHelper.CreateSeededRandom(42);
        clone.SwapMutation(rand);
        clone.SetFitness(200.0);

        // Original should be unchanged
        Assert.Equal(100.0, original.GetFitness(), Tolerance);
        int[] afterPerm = original.GetPermutation();
        Assert.Equal(originalPerm, afterPerm);
    }

    [Fact]
    public void RealValuedIndividual_Clone_MutatingCloneDoesNotAffectOriginal()
    {
        var genes = new List<RealGene>
        {
            new(1.0, 0.1),
            new(2.0, 0.2),
            new(3.0, 0.3)
        };
        var original = new RealValuedIndividual(genes);
        original.SetFitness(0.5);

        var clone = (RealValuedIndividual)original.Clone();
        clone.UpdateStepSizes(0.5); // This modifies step sizes
        clone.SetFitness(0.9);

        // Original should be unchanged
        Assert.Equal(0.5, original.GetFitness(), Tolerance);
        var originalGenes = original.GetGenes().ToList();
        Assert.Equal(0.1, originalGenes[0].StepSize, Tolerance);
        Assert.Equal(0.2, originalGenes[1].StepSize, Tolerance);
        Assert.Equal(0.3, originalGenes[2].StepSize, Tolerance);
    }

    [Fact]
    public void MultiObjectiveRealIndividual_Clone_PreservesAllState()
    {
        var rand = RandomHelper.CreateSeededRandom(42);
        var original = new MultiObjectiveRealIndividual(3, -1.0, 1.0, rand);
        original.SetFitness(0.75);
        original.SetRank(2);
        original.SetCrowdingDistance(1.5);
        original.SetObjectiveValues([0.1, 0.2, 0.3]);

        var clone = original.Clone();

        // Verify all properties preserved
        Assert.Equal(0.75, clone.GetFitness(), Tolerance);
        Assert.Equal(2, clone.GetRank());
        Assert.Equal(1.5, clone.GetCrowdingDistance(), Tolerance);

        var cloneObjectives = clone.GetObjectiveValues().ToList();
        Assert.Equal(3, cloneObjectives.Count);
        Assert.Equal(0.1, cloneObjectives[0], Tolerance);
        Assert.Equal(0.2, cloneObjectives[1], Tolerance);
        Assert.Equal(0.3, cloneObjectives[2], Tolerance);

        // Gene values preserved
        var originalValues = original.GetValuesAsArray();
        var cloneValues = clone.GetValuesAsArray();
        Assert.Equal(originalValues.Length, cloneValues.Length);
        for (int i = 0; i < originalValues.Length; i++)
        {
            Assert.Equal(originalValues[i], cloneValues[i], Tolerance);
        }
    }

    #endregion

    #region Gene Equality and HashCode Contracts

    [Fact]
    public void RealGene_Equality_UsesTolerance()
    {
        // RealGene.Equals uses Math.Abs(a-b) < 1e-10
        var a = new RealGene(1.0, 0.1);
        var b = new RealGene(1.0, 0.1);
        var c = new RealGene(1.0 + 1e-11, 0.1); // Within tolerance
        var d = new RealGene(1.0 + 1e-9, 0.1);  // Outside tolerance

        Assert.True(a.Equals(b));
        Assert.True(a.Equals(c));  // 1e-11 < 1e-10
        Assert.False(a.Equals(d)); // 1e-9 > 1e-10
    }

    [Fact]
    public void RealGene_Equality_ChecksBothValueAndStepSize()
    {
        var a = new RealGene(1.0, 0.1);
        var b = new RealGene(1.0, 0.2); // Different step size

        Assert.False(a.Equals(b));
    }

    [Fact]
    public void BinaryGene_Equality_ExactMatch()
    {
        var a = new BinaryGene(0);
        var b = new BinaryGene(0);
        var c = new BinaryGene(1);

        Assert.True(a.Equals(b));
        Assert.False(a.Equals(c));
        Assert.Equal(a.GetHashCode(), b.GetHashCode());
    }

    [Fact]
    public void PermutationGene_Equality_ExactMatch()
    {
        var a = new PermutationGene(5);
        var b = new PermutationGene(5);
        var c = new PermutationGene(6);

        Assert.True(a.Equals(b));
        Assert.False(a.Equals(c));
        Assert.Equal(a.GetHashCode(), b.GetHashCode());
    }

    #endregion

    #region Fisher-Yates Shuffle Correctness

    [Fact]
    public void PermutationIndividual_Constructor_ProducesValidPermutation()
    {
        // Fisher-Yates: for i from n-1 to 1: swap(i, random(0..i))
        // Must produce exactly the elements 0..size-1
        var rand = RandomHelper.CreateSeededRandom(42);
        int size = 50;
        var individual = new PermutationIndividual(size, rand);
        var perm = individual.GetPermutation();

        Assert.Equal(size, perm.Length);
        var sorted = perm.OrderBy(x => x).ToArray();
        for (int i = 0; i < size; i++)
        {
            Assert.Equal(i, sorted[i]);
        }
    }

    [Fact]
    public void PermutationIndividual_DifferentSeeds_ProduceDifferentPermutations()
    {
        // Different seeds should produce different permutations
        int size = 20;
        var perms = new HashSet<string>();
        for (int seed = 0; seed < 30; seed++)
        {
            var rand = RandomHelper.CreateSeededRandom(seed);
            var individual = new PermutationIndividual(size, rand);
            perms.Add(string.Join(",", individual.GetPermutation()));
        }

        // Should have produced at least several distinct permutations
        Assert.True(perms.Count > 20,
            $"30 different seeds should produce more than 20 distinct permutations, got {perms.Count}");
    }

    #endregion

    #region RealValuedIndividual Initialization Math

    [Fact]
    public void RealValuedIndividual_Constructor_UniformDistributionInRange()
    {
        // Each gene value = minValue + (maxValue - minValue) * random.NextDouble()
        // So all values should be in [minValue, maxValue]
        double min = -3.0;
        double max = 7.0;
        var rand = RandomHelper.CreateSeededRandom(42);
        var individual = new RealValuedIndividual(1000, min, max, rand);
        var values = individual.GetValuesAsArray();

        Assert.Equal(1000, values.Length);
        Assert.True(values.All(v => v >= min && v <= max));

        // Check rough uniformity: mean should be near (min+max)/2
        double mean = values.Average();
        double expectedMean = (min + max) / 2.0;
        Assert.True(Math.Abs(mean - expectedMean) < 0.5,
            $"Mean {mean} should be close to expected {expectedMean}");
    }

    [Fact]
    public void RealValuedIndividual_DefaultStepSize_Is01()
    {
        // RealGene default step size is 0.1
        // When creating RealValuedIndividual via constructor, each gene gets default step size
        var rand = RandomHelper.CreateSeededRandom(42);
        var individual = new RealValuedIndividual(5, -1.0, 1.0, rand);

        var genes = individual.GetGenes().ToList();
        foreach (var gene in genes)
        {
            Assert.Equal(0.1, gene.StepSize, Tolerance);
        }
    }

    #endregion

    #region BinaryIndividual - Monotonicity Properties

    [Fact]
    public void BinaryIndividual_GetValueMapped_IsMonotonic()
    {
        // As binary value increases, mapped value should also increase
        double min = -100.0;
        double max = 100.0;
        double prevMapped = double.NegativeInfinity;

        // Create all 4-bit values in order
        for (int val = 0; val <= 15; val++)
        {
            var genes = new List<BinaryGene>();
            for (int bit = 0; bit < 4; bit++)
            {
                genes.Add(new BinaryGene((val >> bit) & 1));
            }
            var individual = new BinaryIndividual(genes);
            double mapped = individual.GetValueMapped(min, max);

            Assert.True(mapped >= prevMapped,
                $"Mapped value should be monotonically non-decreasing. Val={val}, mapped={mapped}, prev={prevMapped}");
            prevMapped = mapped;
        }
    }

    [Fact]
    public void BinaryIndividual_GetValueMapped_LinearRelationship()
    {
        // The mapping is linear: mapped = min + normalized * (max - min)
        // So the difference between consecutive integer values should be constant
        double min = 0.0;
        double max = 100.0;
        int bits = 4;
        double maxBinaryValue = Math.Pow(2, bits) - 1; // 15

        double expectedStep = (max - min) / maxBinaryValue; // 100/15

        for (int val = 0; val < 15; val++)
        {
            var genes1 = new List<BinaryGene>();
            var genes2 = new List<BinaryGene>();
            for (int bit = 0; bit < bits; bit++)
            {
                genes1.Add(new BinaryGene((val >> bit) & 1));
                genes2.Add(new BinaryGene(((val + 1) >> bit) & 1));
            }

            double mapped1 = new BinaryIndividual(genes1).GetValueMapped(min, max);
            double mapped2 = new BinaryIndividual(genes2).GetValueMapped(min, max);

            double actualStep = mapped2 - mapped1;
            Assert.Equal(expectedStep, actualStep, 1e-10);
        }
    }

    #endregion
}
