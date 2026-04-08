using AiDotNet.Enums;
using AiDotNet.Genetics;
using AiDotNet.Interfaces;
using AiDotNet.Tensors.Helpers;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.Genetics;

/// <summary>
/// Integration tests for Genetic Algorithm components including GeneticParameters,
/// EvolutionStats, and GA-related functionality.
/// </summary>
public class GeneticAlgorithmsIntegrationTests
{
    private const double Tolerance = 1e-10;

    #region GeneticParameters Tests

    [Fact]
    public void GeneticParameters_DefaultValues_AreCorrect()
    {
        // Arrange & Act
        var parameters = new GeneticParameters();

        // Assert - Verify default values
        Assert.Equal(100, parameters.PopulationSize);
        Assert.Equal(0.8, parameters.CrossoverRate, Tolerance);
        Assert.Equal(0.1, parameters.MutationRate, Tolerance);
        Assert.Equal(SelectionMethod.Tournament, parameters.SelectionMethod);
        Assert.Equal(3, parameters.TournamentSize);
        Assert.Equal(0.1, parameters.ElitismRate, Tolerance);
        Assert.Equal(100, parameters.MaxGenerations);
        Assert.Equal(double.MaxValue, parameters.FitnessThreshold);
        Assert.Equal(TimeSpan.FromMinutes(10), parameters.MaxTime);
        Assert.Equal(20, parameters.MaxGenerationsWithoutImprovement);
        Assert.Equal("SinglePoint", parameters.CrossoverOperator);
        Assert.Equal("Uniform", parameters.MutationOperator);
        Assert.True(parameters.UseParallelEvaluation);
        Assert.Equal(InitializationMethod.Random, parameters.InitializationMethod);
    }

    [Theory]
    [InlineData(50)]
    [InlineData(100)]
    [InlineData(500)]
    [InlineData(1000)]
    public void GeneticParameters_PopulationSize_CanBeConfigured(int size)
    {
        // Arrange
        var parameters = new GeneticParameters();

        // Act
        parameters.PopulationSize = size;

        // Assert
        Assert.Equal(size, parameters.PopulationSize);
    }

    [Theory]
    [InlineData(0.0)]
    [InlineData(0.5)]
    [InlineData(0.8)]
    [InlineData(1.0)]
    public void GeneticParameters_CrossoverRate_CanBeConfigured(double rate)
    {
        // Arrange
        var parameters = new GeneticParameters();

        // Act
        parameters.CrossoverRate = rate;

        // Assert
        Assert.Equal(rate, parameters.CrossoverRate, Tolerance);
    }

    [Theory]
    [InlineData(0.0)]
    [InlineData(0.05)]
    [InlineData(0.1)]
    [InlineData(0.5)]
    public void GeneticParameters_MutationRate_CanBeConfigured(double rate)
    {
        // Arrange
        var parameters = new GeneticParameters();

        // Act
        parameters.MutationRate = rate;

        // Assert
        Assert.Equal(rate, parameters.MutationRate, Tolerance);
    }

    [Theory]
    [InlineData(SelectionMethod.Tournament)]
    [InlineData(SelectionMethod.RouletteWheel)]
    [InlineData(SelectionMethod.Rank)]
    [InlineData(SelectionMethod.Truncation)]
    [InlineData(SelectionMethod.Uniform)]
    [InlineData(SelectionMethod.StochasticUniversalSampling)]
    [InlineData(SelectionMethod.Elitism)]
    public void GeneticParameters_SelectionMethod_CanBeConfigured(SelectionMethod method)
    {
        // Arrange
        var parameters = new GeneticParameters();

        // Act
        parameters.SelectionMethod = method;

        // Assert
        Assert.Equal(method, parameters.SelectionMethod);
    }

    [Theory]
    [InlineData(2)]
    [InlineData(3)]
    [InlineData(5)]
    [InlineData(10)]
    public void GeneticParameters_TournamentSize_CanBeConfigured(int size)
    {
        // Arrange
        var parameters = new GeneticParameters();

        // Act
        parameters.TournamentSize = size;

        // Assert
        Assert.Equal(size, parameters.TournamentSize);
    }

    [Theory]
    [InlineData(0.0)]
    [InlineData(0.05)]
    [InlineData(0.1)]
    [InlineData(0.2)]
    public void GeneticParameters_ElitismRate_CanBeConfigured(double rate)
    {
        // Arrange
        var parameters = new GeneticParameters();

        // Act
        parameters.ElitismRate = rate;

        // Assert
        Assert.Equal(rate, parameters.ElitismRate, Tolerance);
    }

    [Theory]
    [InlineData(InitializationMethod.Random)]
    [InlineData(InitializationMethod.CaseBased)]
    [InlineData(InitializationMethod.Heuristic)]
    [InlineData(InitializationMethod.Diverse)]
    [InlineData(InitializationMethod.Grid)]
    [InlineData(InitializationMethod.XavierUniform)]
    public void GeneticParameters_InitializationMethod_CanBeConfigured(InitializationMethod method)
    {
        // Arrange
        var parameters = new GeneticParameters();

        // Act
        parameters.InitializationMethod = method;

        // Assert
        Assert.Equal(method, parameters.InitializationMethod);
    }

    [Fact]
    public void GeneticParameters_MaxTime_CanBeConfigured()
    {
        // Arrange
        var parameters = new GeneticParameters();
        var customTime = TimeSpan.FromHours(1);

        // Act
        parameters.MaxTime = customTime;

        // Assert
        Assert.Equal(customTime, parameters.MaxTime);
    }

    [Fact]
    public void GeneticParameters_CrossoverOperator_CanBeConfigured()
    {
        // Arrange
        var parameters = new GeneticParameters();

        // Act
        parameters.CrossoverOperator = "Uniform";

        // Assert
        Assert.Equal("Uniform", parameters.CrossoverOperator);
    }

    [Fact]
    public void GeneticParameters_MutationOperator_CanBeConfigured()
    {
        // Arrange
        var parameters = new GeneticParameters();

        // Act
        parameters.MutationOperator = "Gaussian";

        // Assert
        Assert.Equal("Gaussian", parameters.MutationOperator);
    }

    [Fact]
    public void GeneticParameters_FullConfiguration_WorksCorrectly()
    {
        // Arrange & Act - Create a fully configured parameters object
        var parameters = new GeneticParameters
        {
            PopulationSize = 200,
            CrossoverRate = 0.9,
            MutationRate = 0.15,
            SelectionMethod = SelectionMethod.Rank,
            TournamentSize = 5,
            ElitismRate = 0.05,
            MaxGenerations = 500,
            FitnessThreshold = 0.99,
            MaxTime = TimeSpan.FromHours(2),
            MaxGenerationsWithoutImprovement = 50,
            CrossoverOperator = "Uniform",
            MutationOperator = "Gaussian",
            UseParallelEvaluation = false,
            InitializationMethod = InitializationMethod.XavierUniform
        };

        // Assert - All values are set correctly
        Assert.Equal(200, parameters.PopulationSize);
        Assert.Equal(0.9, parameters.CrossoverRate, Tolerance);
        Assert.Equal(0.15, parameters.MutationRate, Tolerance);
        Assert.Equal(SelectionMethod.Rank, parameters.SelectionMethod);
        Assert.Equal(5, parameters.TournamentSize);
        Assert.Equal(0.05, parameters.ElitismRate, Tolerance);
        Assert.Equal(500, parameters.MaxGenerations);
        Assert.Equal(0.99, parameters.FitnessThreshold, Tolerance);
        Assert.Equal(TimeSpan.FromHours(2), parameters.MaxTime);
        Assert.Equal(50, parameters.MaxGenerationsWithoutImprovement);
        Assert.Equal("Uniform", parameters.CrossoverOperator);
        Assert.Equal("Gaussian", parameters.MutationOperator);
        Assert.False(parameters.UseParallelEvaluation);
        Assert.Equal(InitializationMethod.XavierUniform, parameters.InitializationMethod);
    }

    #endregion

    #region ModelParameterGene Tests

    [Fact]
    public void ModelParameterGene_Constructor_SetsCorrectValues()
    {
        // Arrange & Act
        var gene = new ModelParameterGene<double>(5, 0.75);

        // Assert
        Assert.Equal(5, gene.Index);
        Assert.Equal(0.75, gene.Value, Tolerance);
    }

    [Theory]
    [InlineData(0, 0.0)]
    [InlineData(1, 1.5)]
    [InlineData(100, -3.14159)]
    [InlineData(999, double.MaxValue)]
    public void ModelParameterGene_Constructor_WithVariousValues(int index, double value)
    {
        // Arrange & Act
        var gene = new ModelParameterGene<double>(index, value);

        // Assert
        Assert.Equal(index, gene.Index);
        Assert.Equal(value, gene.Value, Tolerance);
    }

    [Fact]
    public void ModelParameterGene_Clone_CreatesDeepCopy()
    {
        // Arrange
        var original = new ModelParameterGene<double>(10, 0.5);

        // Act
        var clone = original.Clone();

        // Assert
        Assert.NotSame(original, clone);
        Assert.Equal(original.Index, clone.Index);
        Assert.Equal(original.Value, clone.Value, Tolerance);
    }

    [Fact]
    public void ModelParameterGene_Clone_IsIndependentCopy()
    {
        // Arrange
        var original = new ModelParameterGene<double>(10, 0.5);

        // Act
        var clone = original.Clone();

        // Assert - Clone has same values but is a different object
        // Note: ModelParameterGene is immutable, so we verify cloning works correctly
        Assert.NotSame(original, clone);
        Assert.Equal(original.Index, clone.Index);
        Assert.Equal(original.Value, clone.Value, Tolerance);
    }

    [Fact]
    public void ModelParameterGene_Equals_ReturnsTrueForEqualGenes()
    {
        // Arrange
        var gene1 = new ModelParameterGene<double>(5, 0.75);
        var gene2 = new ModelParameterGene<double>(5, 0.75);

        // Act & Assert
        Assert.True(gene1.Equals(gene2));
        Assert.Equal(gene1.GetHashCode(), gene2.GetHashCode());
    }

    [Fact]
    public void ModelParameterGene_Equals_ReturnsFalseForDifferentGenes()
    {
        // Arrange
        var gene1 = new ModelParameterGene<double>(5, 0.75);
        var gene2 = new ModelParameterGene<double>(5, 0.80);
        var gene3 = new ModelParameterGene<double>(6, 0.75);

        // Act & Assert
        Assert.False(gene1.Equals(gene2)); // Different value
        Assert.False(gene1.Equals(gene3)); // Different index
    }

    [Fact]
    public void ModelParameterGene_MultipleGenes_CanRepresentModelParameters()
    {
        // Arrange & Act - Simulate a simple model's parameters
        var genes = new List<ModelParameterGene<double>>
        {
            new ModelParameterGene<double>(0, 0.1),  // Weight 1
            new ModelParameterGene<double>(1, 0.2),  // Weight 2
            new ModelParameterGene<double>(2, 0.3),  // Weight 3
            new ModelParameterGene<double>(3, 0.5)   // Bias
        };

        // Assert
        Assert.Equal(4, genes.Count);
        Assert.All(genes, g => Assert.True(g.Index >= 0 && g.Index < 4));
        Assert.Equal(0.1, genes[0].Value, Tolerance);
        Assert.Equal(0.5, genes[3].Value, Tolerance);
    }

    #endregion

    #region MultiObjectiveRealIndividual Advanced Tests

    [Fact]
    public void MultiObjectiveRealIndividual_Dominates_CorrectlyIdentifiesDominance()
    {
        // Arrange
        var rand = RandomHelper.CreateSeededRandom(42);
        var individual1 = new MultiObjectiveRealIndividual(2, 0.0, 1.0, rand);
        var individual2 = new MultiObjectiveRealIndividual(2, 0.0, 1.0, rand);

        // Set objectives where individual1 dominates individual2
        // (better in at least one, not worse in any - for minimization)
        individual1.SetObjectiveValues([0.2, 0.3]);
        individual2.SetObjectiveValues([0.5, 0.6]);

        // Act
        bool dominates = individual1.Dominates(individual2);

        // Assert - individual1 should dominate individual2 (lower is better in minimization)
        Assert.True(dominates);
    }

    [Fact]
    public void MultiObjectiveRealIndividual_Dominates_ReturnsFalseWhenNotDominating()
    {
        // Arrange
        var rand = RandomHelper.CreateSeededRandom(42);
        var individual1 = new MultiObjectiveRealIndividual(2, 0.0, 1.0, rand);
        var individual2 = new MultiObjectiveRealIndividual(2, 0.0, 1.0, rand);

        // Neither dominates the other (trade-off)
        individual1.SetObjectiveValues([0.2, 0.8]);
        individual2.SetObjectiveValues([0.8, 0.2]);

        // Act
        bool dominates1Over2 = individual1.Dominates(individual2);
        bool dominates2Over1 = individual2.Dominates(individual1);

        // Assert - Neither should dominate the other
        Assert.False(dominates1Over2);
        Assert.False(dominates2Over1);
    }

    [Fact]
    public void MultiObjectiveRealIndividual_RankAndCrowdingDistance_CanBeSet()
    {
        // Arrange
        var rand = RandomHelper.CreateSeededRandom(42);
        var individual = new MultiObjectiveRealIndividual(3, -1.0, 1.0, rand);

        // Act
        individual.SetRank(2);
        individual.SetCrowdingDistance(0.75);

        // Assert
        Assert.Equal(2, individual.GetRank());
        Assert.Equal(0.75, individual.GetCrowdingDistance(), Tolerance);
    }

    [Theory]
    [InlineData(0)]
    [InlineData(1)]
    [InlineData(5)]
    [InlineData(10)]
    public void MultiObjectiveRealIndividual_Rank_CanBeSetToVarious(int rank)
    {
        // Arrange
        var rand = RandomHelper.CreateSeededRandom(42);
        var individual = new MultiObjectiveRealIndividual(2, 0.0, 1.0, rand);

        // Act
        individual.SetRank(rank);

        // Assert
        Assert.Equal(rank, individual.GetRank());
    }

    [Theory]
    [InlineData(0.0)]
    [InlineData(0.5)]
    [InlineData(1.0)]
    [InlineData(double.PositiveInfinity)]
    public void MultiObjectiveRealIndividual_CrowdingDistance_CanBeSetToVarious(double distance)
    {
        // Arrange
        var rand = RandomHelper.CreateSeededRandom(42);
        var individual = new MultiObjectiveRealIndividual(2, 0.0, 1.0, rand);

        // Act
        individual.SetCrowdingDistance(distance);

        // Assert
        Assert.Equal(distance, individual.GetCrowdingDistance(), Tolerance);
    }

    [Fact]
    public void MultiObjectiveRealIndividual_Clone_PreservesAllProperties()
    {
        // Arrange
        var rand = RandomHelper.CreateSeededRandom(42);
        var original = new MultiObjectiveRealIndividual(3, -1.0, 1.0, rand);
        original.SetObjectiveValues([0.1, 0.2, 0.3]);
        original.SetRank(1);
        original.SetCrowdingDistance(0.5);
        original.SetFitness(0.8);

        // Act
        var clone = original.Clone() as MultiObjectiveRealIndividual;

        // Assert
        Assert.NotNull(clone);
        Assert.NotSame(original, clone);
        Assert.Equal(original.GetRank(), clone.GetRank());
        Assert.Equal(original.GetCrowdingDistance(), clone.GetCrowdingDistance(), Tolerance);
        Assert.Equal(original.GetFitness(), clone.GetFitness(), Tolerance);

        // Check objective values
        var originalObjectives = original.GetObjectiveValues().ToList();
        var cloneObjectives = clone.GetObjectiveValues().ToList();
        Assert.Equal(originalObjectives.Count, cloneObjectives.Count);
        for (int i = 0; i < originalObjectives.Count; i++)
        {
            Assert.Equal(originalObjectives[i], cloneObjectives[i], Tolerance);
        }
    }

    #endregion

    #region PermutationIndividual Advanced Tests

    [Fact]
    public void PermutationIndividual_OrderCrossover_ProducesValidPermutations()
    {
        // Arrange
        var rand = RandomHelper.CreateSeededRandom(42);
        var parent1 = new PermutationIndividual(10, rand);
        var parent2 = new PermutationIndividual(10, rand);

        // Act
        var (child1, child2) = parent1.OrderCrossover(parent2, rand);

        // Assert - Each child should be a valid permutation
        Assert.NotNull(child1);
        Assert.NotNull(child2);

        var perm1 = child1.GetPermutation();
        var perm2 = child2.GetPermutation();

        // Each should contain all values 0-9 exactly once
        Assert.Equal(10, perm1.Length);
        Assert.Equal(10, perm2.Length);

        var sorted1 = perm1.OrderBy(x => x).ToArray();
        var sorted2 = perm2.OrderBy(x => x).ToArray();

        for (int i = 0; i < 10; i++)
        {
            Assert.Equal(i, sorted1[i]);
            Assert.Equal(i, sorted2[i]);
        }
    }

    [Fact]
    public void PermutationIndividual_SwapMutation_MaintainsValidPermutation()
    {
        // Arrange
        var rand = RandomHelper.CreateSeededRandom(42);
        var individual = new PermutationIndividual(10, rand);
        var originalPerm = individual.GetPermutation().ToArray();

        // Act
        individual.SwapMutation(rand);
        var mutatedPerm = individual.GetPermutation();

        // Assert - Still a valid permutation
        Assert.Equal(10, mutatedPerm.Length);
        var sorted = mutatedPerm.OrderBy(x => x).ToArray();
        for (int i = 0; i < 10; i++)
        {
            Assert.Equal(i, sorted[i]);
        }
    }

    [Fact]
    public void PermutationIndividual_InversionMutation_MaintainsValidPermutation()
    {
        // Arrange
        var rand = RandomHelper.CreateSeededRandom(42);
        var individual = new PermutationIndividual(10, rand);

        // Act
        individual.InversionMutation(rand);
        var mutatedPerm = individual.GetPermutation();

        // Assert - Still a valid permutation
        Assert.Equal(10, mutatedPerm.Length);
        var sorted = mutatedPerm.OrderBy(x => x).ToArray();
        for (int i = 0; i < 10; i++)
        {
            Assert.Equal(i, sorted[i]);
        }
    }

    [Fact]
    public void PermutationIndividual_MultipleSwapMutations_MaintainValidity()
    {
        // Arrange
        var rand = RandomHelper.CreateSeededRandom(42);
        var individual = new PermutationIndividual(20, rand);

        // Act - Apply many mutations
        for (int i = 0; i < 100; i++)
        {
            individual.SwapMutation(rand);
        }

        var perm = individual.GetPermutation();

        // Assert - Still valid after many mutations
        Assert.Equal(20, perm.Length);
        var sorted = perm.OrderBy(x => x).ToArray();
        for (int i = 0; i < 20; i++)
        {
            Assert.Equal(i, sorted[i]);
        }
    }

    #endregion

    #region TreeIndividual Advanced Tests

    [Fact]
    public void TreeIndividual_RandomGeneration_ProducesVariedExpressions()
    {
        // Arrange
        var terminals = new List<string> { "x", "1.0", "2.0" };
        var expressions = new HashSet<string>();

        // Act - Generate multiple trees with different seeds
        for (int i = 0; i < 20; i++)
        {
            var rand = RandomHelper.CreateSeededRandom(i);
            var tree = new TreeIndividual(rand, terminals, fullMethod: false);
            expressions.Add(tree.GetExpression());
        }

        // Assert - Should produce some variety
        Assert.True(expressions.Count > 1, "Should produce varied expressions");
    }

    [Fact]
    public void TreeIndividual_FullMethod_ProducesDeepTrees()
    {
        // Arrange
        var terminals = new List<string> { "x", "1.0" };
        var rand = RandomHelper.CreateSeededRandom(42);

        // Act
        var tree = new TreeIndividual(rand, terminals, fullMethod: true);

        // Assert - Full method should produce deeper trees
        Assert.True(tree.GetDepth() >= 1, "Full method should produce trees with depth >= 1");
    }

    [Fact]
    public void TreeIndividual_Evaluate_HandlesProtectedDivision()
    {
        // Arrange - Create a division by zero scenario
        var divNode = new NodeGene(GeneticNodeType.Function, "/");
        divNode.Children.Add(new NodeGene(GeneticNodeType.Terminal, "1.0"));
        divNode.Children.Add(new NodeGene(GeneticNodeType.Terminal, "0.0"));
        var tree = new TreeIndividual(divNode);

        // Act - Evaluate with x=anything
        var result = tree.Evaluate(new Dictionary<string, double> { { "x", 5.0 } });

        // Assert - Protected division should return 1.0 when dividing by zero
        Assert.Equal(1.0, result, Tolerance);
    }

    [Fact]
    public void TreeIndividual_Evaluate_HandlesProtectedLog()
    {
        // Arrange - Create a log of negative number scenario
        var root = new NodeGene(GeneticNodeType.Function, "log");
        root.Children.Add(new NodeGene(GeneticNodeType.Terminal, "-1.0"));
        var tree = new TreeIndividual(root);

        // Act - Evaluate
        var result = tree.Evaluate(new Dictionary<string, double>());

        // Assert - Protected log should return 0.0 for non-positive values
        Assert.Equal(0.0, result, Tolerance);
    }

    [Fact]
    public void TreeIndividual_SubtreeMutation_ChangesExpression()
    {
        // Arrange
        var rand = RandomHelper.CreateSeededRandom(42);
        var terminals = new List<string> { "x", "1.0", "2.0" };
        var tree = new TreeIndividual(rand, terminals, fullMethod: true);
        string originalExpr = tree.GetExpression();

        // Act - Apply many subtree mutations
        bool changed = false;
        for (int i = 0; i < 50; i++)
        {
            tree.SubtreeMutation();
            if (tree.GetExpression() != originalExpr)
            {
                changed = true;
                break;
            }
        }

        // Assert - Expression should eventually change
        Assert.True(changed, "Subtree mutation should change the expression");
    }

    [Fact]
    public void TreeIndividual_Evaluate_WithMultipleVariables()
    {
        // Arrange - Build: (x + y) * 2
        var mulNode = new NodeGene(GeneticNodeType.Function, "*");
        var addNode = new NodeGene(GeneticNodeType.Function, "+");
        addNode.Children.Add(new NodeGene(GeneticNodeType.Terminal, "x"));
        addNode.Children.Add(new NodeGene(GeneticNodeType.Terminal, "y"));
        mulNode.Children.Add(addNode);
        mulNode.Children.Add(new NodeGene(GeneticNodeType.Terminal, "2.0"));
        var tree = new TreeIndividual(mulNode);

        // Act
        var result = tree.Evaluate(new Dictionary<string, double> { { "x", 3.0 }, { "y", 2.0 } });

        // Assert - (3 + 2) * 2 = 10
        Assert.Equal(10.0, result, Tolerance);
    }

    #endregion

    #region RealValuedIndividual Advanced Tests

    [Fact]
    public void RealValuedIndividual_UpdateStepSizes_IncreasesWithHighSuccess()
    {
        // Arrange
        var rand = RandomHelper.CreateSeededRandom(42);
        var individual = new RealValuedIndividual(5, -1.0, 1.0, rand);
        var genes = individual.GetGenes().ToList();
        var originalStepSizes = genes.Select(g => g.StepSize).ToList();

        // Act - High success ratio (> 0.2) should increase step sizes
        individual.UpdateStepSizes(0.5);
        var updatedGenes = individual.GetGenes().ToList();

        // Assert - Step sizes should be larger
        for (int i = 0; i < genes.Count; i++)
        {
            Assert.True(updatedGenes[i].StepSize > originalStepSizes[i],
                $"Step size at index {i} should increase with high success ratio");
        }
    }

    [Fact]
    public void RealValuedIndividual_UpdateStepSizes_DecreasesWithLowSuccess()
    {
        // Arrange
        var rand = RandomHelper.CreateSeededRandom(42);
        var individual = new RealValuedIndividual(5, -1.0, 1.0, rand);
        var genes = individual.GetGenes().ToList();
        var originalStepSizes = genes.Select(g => g.StepSize).ToList();

        // Act - Low success ratio (< 0.2) should decrease step sizes
        individual.UpdateStepSizes(0.1);
        var updatedGenes = individual.GetGenes().ToList();

        // Assert - Step sizes should be smaller
        for (int i = 0; i < genes.Count; i++)
        {
            Assert.True(updatedGenes[i].StepSize < originalStepSizes[i],
                $"Step size at index {i} should decrease with low success ratio");
        }
    }

    [Fact]
    public void RealValuedIndividual_GetValuesAsArray_ReturnsCorrectValues()
    {
        // Arrange
        var genes = new List<RealGene>
        {
            new RealGene(0.1),
            new RealGene(0.2),
            new RealGene(0.3)
        };
        var individual = new RealValuedIndividual(genes);

        // Act
        var values = individual.GetValuesAsArray();

        // Assert
        Assert.Equal(3, values.Length);
        Assert.Equal(0.1, values[0], Tolerance);
        Assert.Equal(0.2, values[1], Tolerance);
        Assert.Equal(0.3, values[2], Tolerance);
    }

    #endregion

    #region BinaryIndividual Advanced Tests

    [Fact]
    public void BinaryIndividual_GetValueAsInt_ReturnsCorrectValue()
    {
        // Arrange - Binary 1010 (LSB first, so bits 0 and 2 are set)
        var genes = new List<BinaryGene>
        {
            new BinaryGene(0),   // bit 0 = 0
            new BinaryGene(1),   // bit 1 = 1 (contributes 2)
            new BinaryGene(0),   // bit 2 = 0
            new BinaryGene(1)    // bit 3 = 1 (contributes 8)
        };
        var individual = new BinaryIndividual(genes);

        // Act
        int decoded = individual.GetValueAsInt();

        // Assert - 2 + 8 = 10
        Assert.Equal(10, decoded);
    }

    [Theory]
    [InlineData(new int[] { 1, 1, 1, 1 }, 15)]      // All 1s = 1+2+4+8 = 15
    [InlineData(new int[] { 0, 0, 0, 0 }, 0)]       // All 0s = 0
    [InlineData(new int[] { 1, 0, 0, 0 }, 1)]       // Only bit 0 = 1
    [InlineData(new int[] { 0, 0, 0, 1 }, 8)]       // Only bit 3 = 8
    [InlineData(new int[] { 0, 1, 0, 0 }, 2)]       // Only bit 1 = 2
    public void BinaryIndividual_GetValueAsInt_VariousPatterns(int[] bits, int expected)
    {
        // Arrange
        var genes = bits.Select(b => new BinaryGene(b)).ToList();
        var individual = new BinaryIndividual(genes);

        // Act
        int decoded = individual.GetValueAsInt();

        // Assert
        Assert.Equal(expected, decoded);
    }

    [Fact]
    public void BinaryIndividual_GetValueAsNormalizedDouble_ReturnsValueInRange()
    {
        // Arrange
        var rand = RandomHelper.CreateSeededRandom(42);
        var individual = new BinaryIndividual(10, rand);

        // Act
        double normalized = individual.GetValueAsNormalizedDouble();

        // Assert - Should be in [0, 1]
        Assert.True(normalized >= 0.0 && normalized <= 1.0,
            $"Normalized value {normalized} should be between 0 and 1");
    }

    [Fact]
    public void BinaryIndividual_GetValueAsNormalizedDouble_AllZeros_ReturnsZero()
    {
        // Arrange
        var genes = Enumerable.Repeat(new BinaryGene(0), 8).ToList();
        var individual = new BinaryIndividual(genes);

        // Act
        double normalized = individual.GetValueAsNormalizedDouble();

        // Assert
        Assert.Equal(0.0, normalized, Tolerance);
    }

    [Fact]
    public void BinaryIndividual_GetValueAsNormalizedDouble_AllOnes_ReturnsOne()
    {
        // Arrange
        var genes = Enumerable.Repeat(new BinaryGene(1), 8).ToList();
        var individual = new BinaryIndividual(genes);

        // Act
        double normalized = individual.GetValueAsNormalizedDouble();

        // Assert
        Assert.Equal(1.0, normalized, Tolerance);
    }

    [Fact]
    public void BinaryIndividual_GetValueMapped_ReturnsValueInRange()
    {
        // Arrange
        var rand = RandomHelper.CreateSeededRandom(42);
        var individual = new BinaryIndividual(10, rand);
        double min = -5.0;
        double max = 5.0;

        // Act
        double mapped = individual.GetValueMapped(min, max);

        // Assert - Should be in [min, max]
        Assert.True(mapped >= min && mapped <= max,
            $"Mapped value {mapped} should be between {min} and {max}");
    }

    [Fact]
    public void BinaryIndividual_GetValueMapped_AllZeros_ReturnsMin()
    {
        // Arrange
        var genes = Enumerable.Repeat(new BinaryGene(0), 8).ToList();
        var individual = new BinaryIndividual(genes);

        // Act
        double mapped = individual.GetValueMapped(-10.0, 10.0);

        // Assert
        Assert.Equal(-10.0, mapped, Tolerance);
    }

    [Fact]
    public void BinaryIndividual_GetValueMapped_AllOnes_ReturnsMax()
    {
        // Arrange
        var genes = Enumerable.Repeat(new BinaryGene(1), 8).ToList();
        var individual = new BinaryIndividual(genes);

        // Act
        double mapped = individual.GetValueMapped(-10.0, 10.0);

        // Assert
        Assert.Equal(10.0, mapped, Tolerance);
    }

    [Fact]
    public void BinaryIndividual_Clone_CreatesDeepCopy()
    {
        // Arrange
        var rand = RandomHelper.CreateSeededRandom(42);
        var original = new BinaryIndividual(20, rand);
        original.SetFitness(0.75);

        // Act
        var clone = original.Clone() as BinaryIndividual;

        // Assert
        Assert.NotNull(clone);
        Assert.NotSame(original, clone);
        Assert.Equal(original.GetFitness(), clone.GetFitness(), Tolerance);
        Assert.Equal(original.GetGenes().Count, clone.GetGenes().Count);
        Assert.Equal(original.GetValueAsInt(), clone.GetValueAsInt());
    }

    [Fact]
    public void BinaryIndividual_SetGenes_UpdatesIndividual()
    {
        // Arrange
        var rand = RandomHelper.CreateSeededRandom(42);
        var individual = new BinaryIndividual(8, rand);
        var newGenes = new List<BinaryGene>
        {
            new BinaryGene(1), new BinaryGene(0), new BinaryGene(1), new BinaryGene(0),
            new BinaryGene(1), new BinaryGene(0), new BinaryGene(1), new BinaryGene(0)
        };

        // Act
        individual.SetGenes(newGenes);

        // Assert
        var genes = individual.GetGenes().ToList();
        Assert.Equal(8, genes.Count);
        Assert.Equal(1, genes[0].Value);
        Assert.Equal(0, genes[1].Value);
    }

    #endregion

    #region Integration Tests - GA Component Interactions

    [Fact]
    public void GeneticParameters_WithPermutationIndividual_ConfigurationScenario()
    {
        // Arrange - Simulate TSP configuration
        var parameters = new GeneticParameters
        {
            PopulationSize = 50,
            CrossoverRate = 0.85,
            MutationRate = 0.05,
            SelectionMethod = SelectionMethod.Tournament,
            TournamentSize = 5,
            ElitismRate = 0.1,
            MaxGenerations = 200,
            MaxGenerationsWithoutImprovement = 30
        };

        // Create a population of permutation individuals
        var rand = RandomHelper.CreateSeededRandom(42);
        var population = new List<PermutationIndividual>();
        for (int i = 0; i < parameters.PopulationSize; i++)
        {
            population.Add(new PermutationIndividual(10, rand));
        }

        // Assert - Population is properly configured
        Assert.Equal(50, population.Count);
        Assert.All(population, p =>
        {
            var perm = p.GetPermutation();
            Assert.Equal(10, perm.Length);
        });
    }

    [Fact]
    public void GeneticParameters_WithRealValuedIndividual_NumericalOptimization()
    {
        // Arrange - Simulate numerical optimization configuration
        var parameters = new GeneticParameters
        {
            PopulationSize = 100,
            CrossoverRate = 0.9,
            MutationRate = 0.15,
            SelectionMethod = SelectionMethod.Rank,
            ElitismRate = 0.05,
            MaxGenerations = 500,
            InitializationMethod = InitializationMethod.Diverse
        };

        // Create a population of real-valued individuals
        var rand = RandomHelper.CreateSeededRandom(42);
        var population = new List<RealValuedIndividual>();
        for (int i = 0; i < parameters.PopulationSize; i++)
        {
            population.Add(new RealValuedIndividual(10, -5.0, 5.0, rand));
        }

        // Assert - Population is properly configured
        Assert.Equal(100, population.Count);
        Assert.All(population, p =>
        {
            var values = p.GetValuesAsArray();
            Assert.Equal(10, values.Length);
            Assert.All(values, v => Assert.True(v >= -5.0 && v <= 5.0));
        });
    }

    [Fact]
    public void GeneticParameters_WithBinaryIndividual_FeatureSelection()
    {
        // Arrange - Simulate feature selection configuration
        var parameters = new GeneticParameters
        {
            PopulationSize = 30,
            CrossoverRate = 0.7,
            MutationRate = 0.02,
            SelectionMethod = SelectionMethod.RouletteWheel,
            ElitismRate = 0.1,
            MaxGenerations = 100
        };

        // Create a population where each bit represents whether a feature is selected
        var rand = RandomHelper.CreateSeededRandom(42);
        var numFeatures = 20;
        var population = new List<BinaryIndividual>();
        for (int i = 0; i < parameters.PopulationSize; i++)
        {
            population.Add(new BinaryIndividual(numFeatures, rand));
        }

        // Assert
        Assert.Equal(30, population.Count);
        Assert.All(population, p => Assert.Equal(20, p.GetGenes().Count));
    }

    [Fact]
    public void GeneticParameters_WithTreeIndividual_SymbolicRegression()
    {
        // Arrange - Simulate symbolic regression configuration
        var parameters = new GeneticParameters
        {
            PopulationSize = 200,
            CrossoverRate = 0.9,
            MutationRate = 0.1,
            SelectionMethod = SelectionMethod.Tournament,
            TournamentSize = 7,
            ElitismRate = 0.01,
            MaxGenerations = 50
        };

        var terminals = new List<string> { "x", "1.0", "2.0", "3.14159" };

        // Create a population of tree individuals
        var population = new List<TreeIndividual>();
        for (int i = 0; i < parameters.PopulationSize; i++)
        {
            var rand = RandomHelper.CreateSeededRandom(i);
            population.Add(new TreeIndividual(rand, terminals, fullMethod: i % 2 == 0));
        }

        // Assert
        Assert.Equal(200, population.Count);
        Assert.All(population, p =>
        {
            Assert.NotNull(p.GetExpression());
            Assert.True(p.GetDepth() >= 0);
        });
    }

    #endregion
}
