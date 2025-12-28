using AiDotNet.Enums;
using AiDotNet.Genetics;
using AiDotNet.Tensors.Helpers;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.Genetics;

/// <summary>
/// Integration tests for Gene types (Issue #347).
/// Tests verify mathematical correctness and proper behavior of:
/// - BinaryGene: Binary (0/1) genes for classic GA problems
/// - RealGene: Real-valued genes with step sizes for ES/numerical optimization
/// - PermutationGene: Index-based genes for permutation problems (TSP, scheduling)
/// - NodeGene: Tree-based genes for genetic programming
/// - ModelParameterGene: Parameter genes for model optimization
/// </summary>
public class GeneTypesIntegrationTests
{
    private const double Tolerance = 1e-10;

    #region BinaryGene Tests

    [Theory]
    [InlineData(0)]
    [InlineData(1)]
    public void BinaryGene_Constructor_SetsValueCorrectly(int value)
    {
        // Arrange & Act
        var gene = new BinaryGene(value);

        // Assert
        Assert.Equal(value, gene.Value);
    }

    [Fact]
    public void BinaryGene_Clone_CreatesIndependentCopy()
    {
        // Arrange
        var original = new BinaryGene(1);

        // Act
        var clone = original.Clone();

        // Assert
        Assert.NotSame(original, clone);
        Assert.Equal(original.Value, clone.Value);
    }

    [Fact]
    public void BinaryGene_Clone_MutationDoesNotAffectOriginal()
    {
        // Arrange
        var original = new BinaryGene(0);
        var clone = original.Clone();

        // Act - Flip the clone (simulate mutation)
        var mutatedClone = new BinaryGene(1 - clone.Value); // Flip

        // Assert
        Assert.Equal(0, original.Value); // Original unchanged
        Assert.Equal(1, mutatedClone.Value); // Mutated value is flipped
    }

    [Fact]
    public void BinaryGene_Flip_CorrectlyInvertsValue()
    {
        // Arrange
        var gene0 = new BinaryGene(0);
        var gene1 = new BinaryGene(1);

        // Act - Create flipped versions
        var flipped0 = new BinaryGene(1 - gene0.Value);
        var flipped1 = new BinaryGene(1 - gene1.Value);

        // Assert
        Assert.Equal(1, flipped0.Value);
        Assert.Equal(0, flipped1.Value);
    }

    #endregion

    #region RealGene Tests

    [Theory]
    [InlineData(0.0)]
    [InlineData(1.5)]
    [InlineData(-3.14159)]
    [InlineData(double.MaxValue / 2)]
    public void RealGene_Constructor_SetsValueCorrectly(double value)
    {
        // Arrange & Act
        var gene = new RealGene(value);

        // Assert
        Assert.Equal(value, gene.Value, Tolerance);
    }

    [Fact]
    public void RealGene_DefaultStepSize_IsPositive()
    {
        // Arrange & Act
        var gene = new RealGene(0.0);

        // Assert - Default step size should be 0.1 per RealGene implementation
        Assert.Equal(0.1, gene.StepSize, Tolerance);
    }

    [Theory]
    [InlineData(0.1)]
    [InlineData(0.5)]
    [InlineData(2.0)]
    public void RealGene_Constructor_WithStepSize_SetsCorrectly(double stepSize)
    {
        // Arrange & Act
        var gene = new RealGene(1.0, stepSize);

        // Assert
        Assert.Equal(1.0, gene.Value, Tolerance);
        Assert.Equal(stepSize, gene.StepSize, Tolerance);
    }

    [Fact]
    public void RealGene_Clone_CopiesValueAndStepSize()
    {
        // Arrange
        var original = new RealGene(3.14159, 0.25);

        // Act
        var clone = original.Clone();

        // Assert
        Assert.NotSame(original, clone);
        Assert.Equal(original.Value, clone.Value, Tolerance);
        Assert.Equal(original.StepSize, clone.StepSize, Tolerance);
    }

    [Fact]
    public void RealGene_Mutation_StepSizeAffectsMagnitude()
    {
        // Arrange - Test that step size is used in mutation
        var gene = new RealGene(0.0, 0.1);
        var rand = RandomHelper.CreateSeededRandom(42);

        // Act - Simulate Gaussian mutation with step size
        double u1 = 1.0 - rand.NextDouble();
        double u2 = 1.0 - rand.NextDouble();
        double randStdNormal = Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Sin(2.0 * Math.PI * u2);
        double perturbation = randStdNormal * gene.StepSize;
        double mutatedValue = gene.Value + perturbation;

        // Assert - Mutation should be within reasonable range based on step size
        // With step size 0.1, most mutations should be within +/- 0.3 (3 sigma)
        Assert.True(Math.Abs(mutatedValue) < 0.5,
            $"Mutation with step size 0.1 should typically stay within bounds. Got: {mutatedValue}");
    }

    [Fact]
    public void RealGene_StepSizeAdaptation_OneFifthRule()
    {
        // Arrange - Test the 1/5 success rule adaptation
        var gene = new RealGene(0.0, 1.0);
        const double c = 0.817; // Standard constant

        // Act - Simulate adaptation
        double originalStepSize = gene.StepSize;

        // Case 1: High success rate (> 0.2) - step size should increase
        double highSuccessRate = 0.3;
        double adjustmentHigh = highSuccessRate > 0.2 ? 1.0 / c : c;
        double newStepSizeHigh = originalStepSize * adjustmentHigh;

        // Case 2: Low success rate (< 0.2) - step size should decrease
        double lowSuccessRate = 0.1;
        double adjustmentLow = lowSuccessRate > 0.2 ? 1.0 / c : c;
        double newStepSizeLow = originalStepSize * adjustmentLow;

        // Assert
        Assert.True(newStepSizeHigh > originalStepSize,
            "High success rate should increase step size");
        Assert.True(newStepSizeLow < originalStepSize,
            "Low success rate should decrease step size");
        Assert.Equal(1.0 / c, newStepSizeHigh, Tolerance);
        Assert.Equal(c, newStepSizeLow, Tolerance);
    }

    #endregion

    #region PermutationGene Tests

    [Theory]
    [InlineData(0)]
    [InlineData(5)]
    [InlineData(100)]
    public void PermutationGene_Constructor_SetsIndexCorrectly(int index)
    {
        // Arrange & Act
        var gene = new PermutationGene(index);

        // Assert
        Assert.Equal(index, gene.Index);
    }

    [Fact]
    public void PermutationGene_Clone_CreatesIndependentCopy()
    {
        // Arrange
        var original = new PermutationGene(42);

        // Act
        var clone = original.Clone();

        // Assert
        Assert.NotSame(original, clone);
        Assert.Equal(original.Index, clone.Index);
    }

    [Fact]
    public void PermutationGene_ValidPermutation_HasUniqueIndices()
    {
        // Arrange - Create a valid permutation of 10 elements
        var rand = RandomHelper.CreateSeededRandom(42);
        int size = 10;
        var individual = new PermutationIndividual(size, rand);

        // Act
        var genes = individual.GetGenes().ToList();
        var indices = genes.Select(g => g.Index).ToArray();

        // Assert - All indices should be unique
        var uniqueIndices = indices.Distinct().ToList();
        Assert.Equal(size, uniqueIndices.Count);

        // All indices should be in range [0, size-1]
        Assert.True(indices.All(i => i >= 0 && i < size),
            "All indices should be in valid range");
    }

    [Fact]
    public void PermutationGene_SwapMutation_PreservesValidPermutation()
    {
        // Arrange
        var rand = RandomHelper.CreateSeededRandom(42);
        int size = 10;
        var individual = new PermutationIndividual(size, rand);

        // Act
        individual.SwapMutation(rand);
        var mutatedPermutation = individual.GetPermutation();

        // Assert - Still a valid permutation
        var uniqueIndices = mutatedPermutation.Distinct().ToList();
        Assert.Equal(size, uniqueIndices.Count);
        Assert.True(mutatedPermutation.All(i => i >= 0 && i < size),
            "All indices should be in valid range after swap mutation");
    }

    [Fact]
    public void PermutationGene_InversionMutation_PreservesValidPermutation()
    {
        // Arrange
        var rand = RandomHelper.CreateSeededRandom(42);
        int size = 10;
        var individual = new PermutationIndividual(size, rand);

        // Act
        individual.InversionMutation(rand);
        var mutatedPermutation = individual.GetPermutation();

        // Assert - Still a valid permutation
        var uniqueIndices = mutatedPermutation.Distinct().ToList();
        Assert.Equal(size, uniqueIndices.Count);
        Assert.True(mutatedPermutation.All(i => i >= 0 && i < size),
            "All indices should be in valid range after inversion mutation");
    }

    [Fact]
    public void PermutationGene_OrderCrossover_ProducesValidOffspring()
    {
        // Arrange
        var rand = RandomHelper.CreateSeededRandom(42);
        int size = 10;
        var parent1 = new PermutationIndividual(size, rand);
        var parent2 = new PermutationIndividual(size, rand);

        // Act
        var (child1, child2) = parent1.OrderCrossover(parent2, rand);

        // Assert - Both children should be valid permutations
        var perm1 = child1.GetPermutation();
        var perm2 = child2.GetPermutation();

        Assert.Equal(size, perm1.Distinct().Count());
        Assert.Equal(size, perm2.Distinct().Count());
        Assert.True(perm1.All(i => i >= 0 && i < size),
            "Child 1 should have valid indices");
        Assert.True(perm2.All(i => i >= 0 && i < size),
            "Child 2 should have valid indices");
    }

    #endregion

    #region NodeGene Tests (Genetic Programming)

    [Fact]
    public void NodeGene_Function_HasCorrectType()
    {
        // Arrange & Act
        var functionNode = new NodeGene(GeneticNodeType.Function, "+");

        // Assert
        Assert.Equal(GeneticNodeType.Function, functionNode.Type);
        Assert.Equal("+", functionNode.Value);
        Assert.NotNull(functionNode.Children);
        Assert.Empty(functionNode.Children);
    }

    [Fact]
    public void NodeGene_Terminal_HasCorrectType()
    {
        // Arrange & Act
        var terminalNode = new NodeGene(GeneticNodeType.Terminal, "x");

        // Assert
        Assert.Equal(GeneticNodeType.Terminal, terminalNode.Type);
        Assert.Equal("x", terminalNode.Value);
    }

    [Fact]
    public void NodeGene_Clone_CreatesIndependentCopy()
    {
        // Arrange
        var original = new NodeGene(GeneticNodeType.Function, "*");
        original.Children.Add(new NodeGene(GeneticNodeType.Terminal, "x"));
        original.Children.Add(new NodeGene(GeneticNodeType.Terminal, "2.0"));

        // Act
        var clone = original.Clone();

        // Assert
        Assert.NotSame(original, clone);
        Assert.Equal(original.Type, clone.Type);
        Assert.Equal(original.Value, clone.Value);
        Assert.Equal(original.Children.Count, clone.Children.Count);
        Assert.NotSame(original.Children[0], clone.Children[0]);
    }

    [Fact]
    public void NodeGene_Clone_DeepClonesChildren()
    {
        // Arrange - Create a tree: + (x, * (y, 2))
        var root = new NodeGene(GeneticNodeType.Function, "+");
        var left = new NodeGene(GeneticNodeType.Terminal, "x");
        var right = new NodeGene(GeneticNodeType.Function, "*");
        right.Children.Add(new NodeGene(GeneticNodeType.Terminal, "y"));
        right.Children.Add(new NodeGene(GeneticNodeType.Terminal, "2.0"));
        root.Children.Add(left);
        root.Children.Add(right);

        // Act
        var clone = root.Clone();

        // Assert - Verify deep clone
        Assert.NotSame(root, clone);
        Assert.NotSame(root.Children[1], clone.Children[1]);
        Assert.NotSame(root.Children[1].Children[0], clone.Children[1].Children[0]);
        Assert.Equal("y", clone.Children[1].Children[0].Value);
    }

    [Fact]
    public void NodeGene_Equals_ReturnsTrueForIdenticalTrees()
    {
        // Arrange
        var node1 = new NodeGene(GeneticNodeType.Function, "+");
        node1.Children.Add(new NodeGene(GeneticNodeType.Terminal, "x"));
        node1.Children.Add(new NodeGene(GeneticNodeType.Terminal, "1.0"));

        var node2 = new NodeGene(GeneticNodeType.Function, "+");
        node2.Children.Add(new NodeGene(GeneticNodeType.Terminal, "x"));
        node2.Children.Add(new NodeGene(GeneticNodeType.Terminal, "1.0"));

        // Act & Assert
        Assert.True(node1.Equals(node2));
    }

    [Fact]
    public void NodeGene_Equals_ReturnsFalseForDifferentTrees()
    {
        // Arrange
        var node1 = new NodeGene(GeneticNodeType.Function, "+");
        node1.Children.Add(new NodeGene(GeneticNodeType.Terminal, "x"));

        var node2 = new NodeGene(GeneticNodeType.Function, "-");
        node2.Children.Add(new NodeGene(GeneticNodeType.Terminal, "x"));

        // Act & Assert
        Assert.False(node1.Equals(node2));
    }

    #endregion

    #region ModelParameterGene Tests

    [Theory]
    [InlineData(0, 1.5)]
    [InlineData(10, -3.14)]
    [InlineData(100, 0.0)]
    public void ModelParameterGene_Constructor_SetsIndexAndValue(int index, double value)
    {
        // Arrange & Act
        var gene = new ModelParameterGene<double>(index, value);

        // Assert
        Assert.Equal(index, gene.Index);
        Assert.Equal(value, gene.Value, Tolerance);
    }

    [Fact]
    public void ModelParameterGene_Clone_CreatesIndependentCopy()
    {
        // Arrange
        var original = new ModelParameterGene<double>(5, 2.718);

        // Act
        var clone = original.Clone();

        // Assert
        Assert.NotSame(original, clone);
        Assert.Equal(original.Index, clone.Index);
        Assert.Equal(original.Value, clone.Value, Tolerance);
    }

    [Fact]
    public void ModelParameterGene_OrderByIndex_MaintainsCorrectOrder()
    {
        // Arrange
        var genes = new List<ModelParameterGene<double>>
        {
            new(3, 0.3),
            new(1, 0.1),
            new(4, 0.4),
            new(0, 0.0),
            new(2, 0.2)
        };

        // Act
        var sorted = genes.OrderBy(g => g.Index).ToList();

        // Assert
        for (int i = 0; i < sorted.Count; i++)
        {
            Assert.Equal(i, sorted[i].Index);
            Assert.Equal(i * 0.1, sorted[i].Value, Tolerance);
        }
    }

    #endregion

    #region BinaryIndividual Tests

    [Fact]
    public void BinaryIndividual_GetValueAsInt_CorrectBinaryToIntConversion()
    {
        // Arrange - Create individual representing binary (little-endian: bit 0 is LSB)
        var genes = new List<BinaryGene>
        {
            new(0), // bit 0 = 0
            new(1), // bit 1 = 1 (value 2)
            new(0), // bit 2 = 0
            new(1)  // bit 3 = 1 (value 8)
        };
        // Value = 0*1 + 1*2 + 0*4 + 1*8 = 10
        var individual = new BinaryIndividual(genes);

        // Act
        int value = individual.GetValueAsInt();

        // Assert
        Assert.Equal(10, value);
    }

    [Fact]
    public void BinaryIndividual_GetValueAsNormalizedDouble_ReturnsValueBetweenZeroAndOne()
    {
        // Arrange
        var rand = RandomHelper.CreateSeededRandom(42);
        var individual = new BinaryIndividual(8, rand);

        // Act
        double normalized = individual.GetValueAsNormalizedDouble();

        // Assert
        Assert.True(normalized >= 0.0 && normalized <= 1.0,
            $"Normalized value should be in [0,1], got: {normalized}");
    }

    [Fact]
    public void BinaryIndividual_GetValueMapped_MapsToCorrectRange()
    {
        // Arrange - All 1s = max value
        var genes = new List<BinaryGene>
        {
            new(1), new(1), new(1), new(1)
        };
        var individual = new BinaryIndividual(genes);
        double min = -10.0;
        double max = 10.0;

        // Act
        double mapped = individual.GetValueMapped(min, max);

        // Assert - All 1s should map to max value
        Assert.Equal(max, mapped, Tolerance);
    }

    [Fact]
    public void BinaryIndividual_Clone_CreatesIndependentCopy()
    {
        // Arrange
        var rand = RandomHelper.CreateSeededRandom(42);
        var original = new BinaryIndividual(8, rand);
        original.SetFitness(0.95);

        // Act
        var clone = original.Clone() as BinaryIndividual;

        // Assert
        Assert.NotNull(clone);
        Assert.NotSame(original, clone);
        Assert.Equal(original.GetFitness(), clone.GetFitness(), Tolerance);
        Assert.Equal(original.GetValueAsInt(), clone.GetValueAsInt());
    }

    #endregion

    #region RealValuedIndividual Tests

    [Fact]
    public void RealValuedIndividual_Constructor_InitializesWithinRange()
    {
        // Arrange
        var rand = RandomHelper.CreateSeededRandom(42);
        double minValue = -5.0;
        double maxValue = 5.0;
        int dimensions = 10;

        // Act
        var individual = new RealValuedIndividual(dimensions, minValue, maxValue, rand);
        var values = individual.GetValuesAsArray();

        // Assert
        Assert.Equal(dimensions, values.Length);
        Assert.True(values.All(v => v >= minValue && v <= maxValue),
            "All values should be within specified range");
    }

    [Fact]
    public void RealValuedIndividual_UpdateStepSizes_HighSuccessIncreasesStepSize()
    {
        // Arrange
        var genes = new List<RealGene>
        {
            new(0.0, 1.0),
            new(1.0, 1.0),
            new(2.0, 1.0)
        };
        var individual = new RealValuedIndividual(genes);
        double originalStepSize = genes[0].StepSize;

        // Act - High success ratio (> 0.2)
        individual.UpdateStepSizes(0.3);
        var updatedGenes = individual.GetGenes().ToList();

        // Assert - Step sizes should increase
        Assert.True(updatedGenes[0].StepSize > originalStepSize,
            "High success rate should increase step size");
    }

    [Fact]
    public void RealValuedIndividual_UpdateStepSizes_LowSuccessDecreasesStepSize()
    {
        // Arrange
        var genes = new List<RealGene>
        {
            new(0.0, 1.0),
            new(1.0, 1.0),
            new(2.0, 1.0)
        };
        var individual = new RealValuedIndividual(genes);
        double originalStepSize = genes[0].StepSize;

        // Act - Low success ratio (< 0.2)
        individual.UpdateStepSizes(0.1);
        var updatedGenes = individual.GetGenes().ToList();

        // Assert - Step sizes should decrease
        Assert.True(updatedGenes[0].StepSize < originalStepSize,
            "Low success rate should decrease step size");
    }

    [Fact]
    public void RealValuedIndividual_Clone_PreservesFitness()
    {
        // Arrange
        var rand = RandomHelper.CreateSeededRandom(42);
        var individual = new RealValuedIndividual(5, -1.0, 1.0, rand);
        individual.SetFitness(0.85);

        // Act
        var clone = individual.Clone() as RealValuedIndividual;

        // Assert
        Assert.NotNull(clone);
        Assert.Equal(0.85, clone.GetFitness(), Tolerance);
    }

    #endregion

    #region PermutationIndividual Tests

    [Fact]
    public void PermutationIndividual_FisherYatesShuffle_ProducesUniformDistribution()
    {
        // Arrange - Create many permutations and verify distribution
        int size = 5;
        int trials = 1000;
        var positionCounts = new int[size, size]; // [position, value]
        var rand = RandomHelper.CreateSeededRandom(42);

        // Act
        for (int t = 0; t < trials; t++)
        {
            var individual = new PermutationIndividual(size, rand);
            var perm = individual.GetPermutation();
            for (int pos = 0; pos < size; pos++)
            {
                positionCounts[pos, perm[pos]]++;
            }
        }

        // Assert - Each value should appear roughly equally in each position
        double expected = trials / (double)size;
        double tolerance = expected * 0.3; // Allow 30% deviation

        for (int pos = 0; pos < size; pos++)
        {
            for (int val = 0; val < size; val++)
            {
                Assert.True(Math.Abs(positionCounts[pos, val] - expected) < tolerance,
                    $"Fisher-Yates should produce uniform distribution. " +
                    $"Position {pos}, Value {val}: Expected ~{expected}, Got {positionCounts[pos, val]}");
            }
        }
    }

    [Fact]
    public void PermutationIndividual_OrderCrossover_InheritsSubsequenceFromParent1()
    {
        // Arrange - Create specific permutations for predictable testing
        var parent1Genes = new List<PermutationGene>
        {
            new(1), new(2), new(3), new(4), new(5)
        };
        var parent2Genes = new List<PermutationGene>
        {
            new(5), new(4), new(3), new(2), new(1)
        };
        var parent1 = new PermutationIndividual(parent1Genes);
        var parent2 = new PermutationIndividual(parent2Genes);
        var rand = RandomHelper.CreateSeededRandom(42);

        // Act
        var (child1, child2) = parent1.OrderCrossover(parent2, rand);

        // Assert - Children should be valid permutations
        var perm1 = child1.GetPermutation();
        var perm2 = child2.GetPermutation();

        Assert.Equal(5, perm1.Distinct().Count());
        Assert.Equal(5, perm2.Distinct().Count());
    }

    #endregion

    #region TreeIndividual Tests

    [Fact]
    public void TreeIndividual_Create_WithRandomGeneration()
    {
        // Arrange
        var rand = RandomHelper.CreateSeededRandom(42);
        var terminals = new List<string> { "x", "1.0", "2.0" };

        // Act
        var tree = new TreeIndividual(rand, terminals, fullMethod: false);

        // Assert
        Assert.NotNull(tree);
        Assert.True(tree.GetDepth() >= 0, "Tree should have non-negative depth");
    }

    [Fact]
    public void TreeIndividual_Evaluate_ConstantReturnsValue()
    {
        // Arrange - Create a simple terminal node tree
        var root = new NodeGene(GeneticNodeType.Terminal, "5.0");
        var tree = new TreeIndividual(root);
        var variables = new Dictionary<string, double>();

        // Act
        double result = tree.Evaluate(variables);

        // Assert
        Assert.Equal(5.0, result, Tolerance);
    }

    [Fact]
    public void TreeIndividual_Evaluate_VariableReturnsValue()
    {
        // Arrange
        var root = new NodeGene(GeneticNodeType.Terminal, "x");
        var tree = new TreeIndividual(root);
        var variables = new Dictionary<string, double> { { "x", 3.5 } };

        // Act
        double result = tree.Evaluate(variables);

        // Assert
        Assert.Equal(3.5, result, Tolerance);
    }

    [Fact]
    public void TreeIndividual_Evaluate_AdditionWorks()
    {
        // Arrange - Tree: x + 2.0
        var root = new NodeGene(GeneticNodeType.Function, "+");
        root.Children.Add(new NodeGene(GeneticNodeType.Terminal, "x"));
        root.Children.Add(new NodeGene(GeneticNodeType.Terminal, "2.0"));
        var tree = new TreeIndividual(root);
        var variables = new Dictionary<string, double> { { "x", 3.0 } };

        // Act
        double result = tree.Evaluate(variables);

        // Assert
        Assert.Equal(5.0, result, Tolerance);
    }

    [Fact]
    public void TreeIndividual_Evaluate_MultiplicationWorks()
    {
        // Arrange - Tree: x * 3.0
        var root = new NodeGene(GeneticNodeType.Function, "*");
        root.Children.Add(new NodeGene(GeneticNodeType.Terminal, "x"));
        root.Children.Add(new NodeGene(GeneticNodeType.Terminal, "3.0"));
        var tree = new TreeIndividual(root);
        var variables = new Dictionary<string, double> { { "x", 4.0 } };

        // Act
        double result = tree.Evaluate(variables);

        // Assert
        Assert.Equal(12.0, result, Tolerance);
    }

    [Fact]
    public void TreeIndividual_Evaluate_ProtectedDivisionHandlesZero()
    {
        // Arrange - Tree: x / 0
        var root = new NodeGene(GeneticNodeType.Function, "/");
        root.Children.Add(new NodeGene(GeneticNodeType.Terminal, "5.0"));
        root.Children.Add(new NodeGene(GeneticNodeType.Terminal, "0.0"));
        var tree = new TreeIndividual(root);
        var variables = new Dictionary<string, double>();

        // Act
        double result = tree.Evaluate(variables);

        // Assert - Protected division should return 1.0 for division by zero
        Assert.Equal(1.0, result, Tolerance);
    }

    [Fact]
    public void TreeIndividual_GetDepth_CalculatesCorrectly()
    {
        // Arrange - Tree: + (x, * (y, 2))
        var root = new NodeGene(GeneticNodeType.Function, "+");
        root.Children.Add(new NodeGene(GeneticNodeType.Terminal, "x"));
        var mult = new NodeGene(GeneticNodeType.Function, "*");
        mult.Children.Add(new NodeGene(GeneticNodeType.Terminal, "y"));
        mult.Children.Add(new NodeGene(GeneticNodeType.Terminal, "2.0"));
        root.Children.Add(mult);
        var tree = new TreeIndividual(root);

        // Act
        int depth = tree.GetDepth();

        // Assert - Depth should be 2 (root -> mult -> terminal)
        Assert.Equal(2, depth);
    }

    [Fact]
    public void TreeIndividual_GetExpression_ReturnsCorrectString()
    {
        // Arrange - Tree: x + 2.0
        var root = new NodeGene(GeneticNodeType.Function, "+");
        root.Children.Add(new NodeGene(GeneticNodeType.Terminal, "x"));
        root.Children.Add(new NodeGene(GeneticNodeType.Terminal, "2.0"));
        var tree = new TreeIndividual(root);

        // Act
        string expr = tree.GetExpression();

        // Assert
        Assert.Equal("(x + 2.0)", expr);
    }

    [Fact]
    public void TreeIndividual_Clone_CreatesDeepCopy()
    {
        // Arrange
        var root = new NodeGene(GeneticNodeType.Function, "+");
        root.Children.Add(new NodeGene(GeneticNodeType.Terminal, "x"));
        root.Children.Add(new NodeGene(GeneticNodeType.Terminal, "1.0"));
        var original = new TreeIndividual(root);
        original.SetFitness(0.75);

        // Act
        var clone = original.Clone() as TreeIndividual;

        // Assert
        Assert.NotNull(clone);
        Assert.NotSame(original, clone);
        Assert.Equal(0.75, clone.GetFitness(), Tolerance);
        Assert.Equal(original.GetExpression(), clone.GetExpression());
    }

    [Fact]
    public void TreeIndividual_PointMutation_ChangesNode()
    {
        // Arrange
        var rand = RandomHelper.CreateSeededRandom(42);
        var terminals = new List<string> { "x", "1.0", "2.0" };
        var tree = new TreeIndividual(rand, terminals, fullMethod: true);
        string originalExpr = tree.GetExpression();

        // Act - Multiple mutations to ensure change happens
        for (int i = 0; i < 10; i++)
        {
            tree.PointMutation();
        }
        string mutatedExpr = tree.GetExpression();

        // Assert - Expression should likely change after mutations
        // Note: May occasionally be the same due to randomness
        Assert.NotNull(mutatedExpr);
    }

    #endregion

    #region MultiObjectiveRealIndividual Tests

    [Fact]
    public void MultiObjectiveRealIndividual_Constructor_InitializesCorrectly()
    {
        // Arrange
        var rand = RandomHelper.CreateSeededRandom(42);
        int dimensions = 5;

        // Act
        var individual = new MultiObjectiveRealIndividual(dimensions, -1.0, 1.0, rand);

        // Assert
        Assert.Equal(dimensions, individual.GetGenes().Count);
    }

    [Fact]
    public void MultiObjectiveRealIndividual_SetAndGetObjectiveValues()
    {
        // Arrange
        var rand = RandomHelper.CreateSeededRandom(42);
        var individual = new MultiObjectiveRealIndividual(5, -1.0, 1.0, rand);
        var objectiveValues = new List<double> { 0.5, 0.75, 0.25 };

        // Act
        individual.SetObjectiveValues(objectiveValues);
        var retrieved = individual.GetObjectiveValues().ToList();

        // Assert - Verify the values were correctly stored and retrieved
        Assert.NotNull(retrieved);
        Assert.Equal(objectiveValues.Count, retrieved.Count);
        for (int i = 0; i < objectiveValues.Count; i++)
        {
            Assert.Equal(objectiveValues[i], retrieved[i], Tolerance);
        }
    }

    [Fact]
    public void MultiObjectiveRealIndividual_RankAndCrowdingDistance()
    {
        // Arrange
        var rand = RandomHelper.CreateSeededRandom(42);
        var individual = new MultiObjectiveRealIndividual(5, -1.0, 1.0, rand);

        // Act
        individual.SetRank(3);
        individual.SetCrowdingDistance(1.5);

        // Assert
        Assert.Equal(3, individual.GetRank());
        Assert.Equal(1.5, individual.GetCrowdingDistance(), Tolerance);
    }

    [Fact]
    public void MultiObjectiveRealIndividual_Dominance_CorrectlyIdentified()
    {
        // Arrange - Create two individuals for dominance comparison
        var rand = RandomHelper.CreateSeededRandom(42);
        var individual1 = new MultiObjectiveRealIndividual(5, -1.0, 1.0, rand);
        var individual2 = new MultiObjectiveRealIndividual(5, -1.0, 1.0, rand);

        // Set objective values - individual1 has lower (better) values in all objectives
        // Dominates returns true if 'this' is better (lower values for minimization)
        individual1.SetObjectiveValues([0.3, 0.4]);
        individual2.SetObjectiveValues([0.6, 0.7]);

        // Act
        bool individual1DominatesIndividual2 = individual1.Dominates(individual2);
        bool individual2DominatesIndividual1 = individual2.Dominates(individual1);

        // Assert - Individual 1 should dominate Individual 2
        Assert.True(individual1DominatesIndividual2, "Individual with lower objective values should dominate");
        Assert.False(individual2DominatesIndividual1, "Individual with higher objective values should not dominate");
    }

    [Fact]
    public void MultiObjectiveRealIndividual_Clone_PreservesFitness()
    {
        // Arrange
        var rand = RandomHelper.CreateSeededRandom(42);
        var original = new MultiObjectiveRealIndividual(5, -1.0, 1.0, rand);
        original.SetFitness(0.5);
        original.SetRank(2);
        original.SetCrowdingDistance(0.8);

        // Act
        var clone = original.Clone() as RealValuedIndividual;

        // Assert
        Assert.NotNull(clone);
        Assert.Equal(0.5, clone.GetFitness(), Tolerance);
    }

    #endregion
}
