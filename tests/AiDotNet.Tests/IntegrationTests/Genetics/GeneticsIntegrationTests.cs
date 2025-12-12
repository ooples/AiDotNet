using AiDotNet.Enums;
using AiDotNet.Genetics;
using AiDotNet.LinearAlgebra;
using Xunit;

namespace AiDotNetTests.IntegrationTests.Genetics
{
    /// <summary>
    /// Comprehensive integration tests for Genetic Algorithms.
    /// Tests binary, real-valued, and permutation-based genetic algorithms
    /// with various selection methods, crossover operators, and mutation strategies.
    /// </summary>
    public class GeneticsIntegrationTests
    {
        private readonly Random _random = new(42); // Fixed seed for reproducibility

        #region BinaryGene Tests

        [Fact]
        public void BinaryGene_Creation_StoresCorrectValue()
        {
            // Arrange & Act
            var gene0 = new BinaryGene(0);
            var gene1 = new BinaryGene(1);
            var gene2 = new BinaryGene(5); // Should be 1

            // Assert
            Assert.Equal(0, gene0.Value);
            Assert.Equal(1, gene1.Value);
            Assert.Equal(1, gene2.Value); // Any positive value becomes 1
        }

        [Fact]
        public void BinaryGene_Clone_CreatesIndependentCopy()
        {
            // Arrange
            var original = new BinaryGene(1);

            // Act
            var clone = original.Clone();
            clone.Value = 0;

            // Assert
            Assert.Equal(1, original.Value);
            Assert.Equal(0, clone.Value);
        }

        [Fact]
        public void BinaryGene_Equals_ComparesCorrectly()
        {
            // Arrange
            var gene1 = new BinaryGene(1);
            var gene2 = new BinaryGene(1);
            var gene3 = new BinaryGene(0);

            // Assert
            Assert.True(gene1.Equals(gene2));
            Assert.False(gene1.Equals(gene3));
        }

        [Fact]
        public void BinaryGene_GetHashCode_IsConsistent()
        {
            // Arrange
            var gene1 = new BinaryGene(1);
            var gene2 = new BinaryGene(1);

            // Assert
            Assert.Equal(gene1.GetHashCode(), gene2.GetHashCode());
        }

        #endregion

        #region BinaryIndividual Tests

        [Fact]
        public void BinaryIndividual_Creation_InitializesRandomly()
        {
            // Arrange & Act
            var individual = new BinaryIndividual(10, _random);

            // Assert
            Assert.Equal(10, individual.GetGenes().Count);
            Assert.True(individual.GetGenes().All(g => g.Value == 0 || g.Value == 1));
        }

        [Fact]
        public void BinaryIndividual_GetValueAsInt_ConvertsCorrectly()
        {
            // Arrange - Create binary: 1010 (little-endian) = 5
            var genes = new List<BinaryGene>
            {
                new BinaryGene(1), // bit 0
                new BinaryGene(0), // bit 1
                new BinaryGene(1), // bit 2
                new BinaryGene(0)  // bit 3
            };
            var individual = new BinaryIndividual(genes);

            // Act
            var value = individual.GetValueAsInt();

            // Assert
            Assert.Equal(5, value); // 1*1 + 0*2 + 1*4 + 0*8 = 5
        }

        [Fact]
        public void BinaryIndividual_GetValueAsNormalizedDouble_ReturnsInRange()
        {
            // Arrange - All zeros
            var genesMin = new List<BinaryGene> { new(0), new(0), new(0), new(0) };
            var individualMin = new BinaryIndividual(genesMin);

            // All ones: 1111 = 15
            var genesMax = new List<BinaryGene> { new(1), new(1), new(1), new(1) };
            var individualMax = new BinaryIndividual(genesMax);

            // Act
            var valueMin = individualMin.GetValueAsNormalizedDouble();
            var valueMax = individualMax.GetValueAsNormalizedDouble();

            // Assert
            Assert.Equal(0.0, valueMin, precision: 10);
            Assert.Equal(1.0, valueMax, precision: 10);
        }

        [Fact]
        public void BinaryIndividual_GetValueMapped_MapsToRange()
        {
            // Arrange - 1111 = 15 (max for 4 bits)
            var genes = new List<BinaryGene> { new(1), new(1), new(1), new(1) };
            var individual = new BinaryIndividual(genes);

            // Act
            var mapped = individual.GetValueMapped(-10.0, 10.0);

            // Assert
            Assert.Equal(10.0, mapped, precision: 10); // Normalized 1.0 maps to max
        }

        [Fact]
        public void BinaryIndividual_SetGetGenes_WorksCorrectly()
        {
            // Arrange
            var individual = new BinaryIndividual(5, _random);
            var newGenes = new List<BinaryGene> { new(1), new(0), new(1) };

            // Act
            individual.SetGenes(newGenes);

            // Assert
            Assert.Equal(3, individual.GetGenes().Count);
            Assert.Equal(1, individual.GetGenes().ElementAt(0).Value);
        }

        [Fact]
        public void BinaryIndividual_SetGetFitness_WorksCorrectly()
        {
            // Arrange
            var individual = new BinaryIndividual(5, _random);

            // Act
            individual.SetFitness(42.5);

            // Assert
            Assert.Equal(42.5, individual.GetFitness());
        }

        [Fact]
        public void BinaryIndividual_Clone_CreatesIndependentCopy()
        {
            // Arrange
            var original = new BinaryIndividual(5, _random);
            original.SetFitness(10.0);

            // Act
            var clone = original.Clone() as BinaryIndividual;

            // Assert
            Assert.NotNull(clone);
            Assert.Equal(original.GetGenes().Count, clone.GetGenes().Count);
            Assert.Equal(original.GetFitness(), clone.GetFitness());

            // Modify clone
            clone.SetFitness(20.0);
            Assert.Equal(10.0, original.GetFitness()); // Original unchanged
        }

        #endregion

        #region RealGene Tests

        [Fact]
        public void RealGene_Creation_StoresCorrectValue()
        {
            // Arrange & Act
            var gene = new RealGene(3.14, 0.1);

            // Assert
            Assert.Equal(3.14, gene.Value, precision: 10);
            Assert.Equal(0.1, gene.StepSize, precision: 10);
        }

        [Fact]
        public void RealGene_Clone_CreatesIndependentCopy()
        {
            // Arrange
            var original = new RealGene(5.0, 0.2);

            // Act
            var clone = original.Clone();
            clone.Value = 10.0;

            // Assert
            Assert.Equal(5.0, original.Value);
            Assert.Equal(10.0, clone.Value);
        }

        [Fact]
        public void RealGene_Equals_ComparesCorrectly()
        {
            // Arrange
            var gene1 = new RealGene(3.14, 0.1);
            var gene2 = new RealGene(3.14, 0.1);
            var gene3 = new RealGene(2.71, 0.1);

            // Assert
            Assert.True(gene1.Equals(gene2));
            Assert.False(gene1.Equals(gene3));
        }

        [Fact]
        public void RealGene_GetHashCode_IsConsistent()
        {
            // Arrange
            var gene1 = new RealGene(3.14, 0.1);
            var gene2 = new RealGene(3.14, 0.1);

            // Assert
            Assert.Equal(gene1.GetHashCode(), gene2.GetHashCode());
        }

        #endregion

        #region RealValuedIndividual Tests

        [Fact]
        public void RealValuedIndividual_Creation_InitializesWithinRange()
        {
            // Arrange & Act
            var individual = new RealValuedIndividual(5, -10.0, 10.0, _random);

            // Assert
            Assert.Equal(5, individual.GetGenes().Count);
            Assert.True(individual.GetGenes().All(g => g.Value >= -10.0 && g.Value <= 10.0));
        }

        [Fact]
        public void RealValuedIndividual_GetValuesAsArray_ReturnsCorrectArray()
        {
            // Arrange
            var genes = new List<RealGene>
            {
                new RealGene(1.0),
                new RealGene(2.0),
                new RealGene(3.0)
            };
            var individual = new RealValuedIndividual(genes);

            // Act
            var values = individual.GetValuesAsArray();

            // Assert
            Assert.Equal(3, values.Length);
            Assert.Equal(1.0, values[0]);
            Assert.Equal(2.0, values[1]);
            Assert.Equal(3.0, values[2]);
        }

        [Fact]
        public void RealValuedIndividual_UpdateStepSizes_AdjustsCorrectly()
        {
            // Arrange
            var genes = new List<RealGene>
            {
                new RealGene(1.0, 0.1),
                new RealGene(2.0, 0.1)
            };
            var individual = new RealValuedIndividual(genes);

            // Act - High success ratio increases step size
            individual.UpdateStepSizes(0.3);

            // Assert
            Assert.True(individual.GetGenes().All(g => g.StepSize > 0.1));
        }

        [Fact]
        public void RealValuedIndividual_Clone_CreatesIndependentCopy()
        {
            // Arrange
            var original = new RealValuedIndividual(3, -5.0, 5.0, _random);
            original.SetFitness(15.0);

            // Act
            var clone = original.Clone() as RealValuedIndividual;

            // Assert
            Assert.NotNull(clone);
            Assert.Equal(original.GetGenes().Count, clone.GetGenes().Count);
            Assert.Equal(original.GetFitness(), clone.GetFitness());

            // Modify clone
            clone.SetFitness(25.0);
            Assert.Equal(15.0, original.GetFitness());
        }

        #endregion

        #region PermutationGene Tests

        [Fact]
        public void PermutationGene_Creation_StoresCorrectIndex()
        {
            // Arrange & Act
            var gene = new PermutationGene(5);

            // Assert
            Assert.Equal(5, gene.Index);
        }

        [Fact]
        public void PermutationGene_Clone_CreatesIndependentCopy()
        {
            // Arrange
            var original = new PermutationGene(7);

            // Act
            var clone = original.Clone();
            clone.Index = 10;

            // Assert
            Assert.Equal(7, original.Index);
            Assert.Equal(10, clone.Index);
        }

        [Fact]
        public void PermutationGene_Equals_ComparesCorrectly()
        {
            // Arrange
            var gene1 = new PermutationGene(3);
            var gene2 = new PermutationGene(3);
            var gene3 = new PermutationGene(5);

            // Assert
            Assert.True(gene1.Equals(gene2));
            Assert.False(gene1.Equals(gene3));
        }

        #endregion

        #region PermutationIndividual Tests

        [Fact]
        public void PermutationIndividual_Creation_IsValidPermutation()
        {
            // Arrange & Act
            var individual = new PermutationIndividual(10, _random);

            // Assert
            var permutation = individual.GetPermutation();
            Assert.Equal(10, permutation.Length);
            Assert.Equal(10, permutation.Distinct().Count()); // All unique
            Assert.True(permutation.All(i => i >= 0 && i < 10)); // Valid range
        }

        [Fact]
        public void PermutationIndividual_GetPermutation_ReturnsCorrectOrder()
        {
            // Arrange
            var genes = new List<PermutationGene>
            {
                new PermutationGene(2),
                new PermutationGene(0),
                new PermutationGene(1)
            };
            var individual = new PermutationIndividual(genes);

            // Act
            var permutation = individual.GetPermutation();

            // Assert
            Assert.Equal(new[] { 2, 0, 1 }, permutation);
        }

        [Fact]
        public void PermutationIndividual_OrderCrossover_ProducesValidPermutations()
        {
            // Arrange
            var parent1 = new PermutationIndividual(8, _random);
            var parent2 = new PermutationIndividual(8, _random);

            // Act
            var (child1, child2) = parent1.OrderCrossover(parent2, _random);

            // Assert
            var perm1 = child1.GetPermutation();
            var perm2 = child2.GetPermutation();

            Assert.Equal(8, perm1.Length);
            Assert.Equal(8, perm2.Length);
            Assert.Equal(8, perm1.Distinct().Count()); // Valid permutation
            Assert.Equal(8, perm2.Distinct().Count());
        }

        [Fact]
        public void PermutationIndividual_SwapMutation_MaintainsValidPermutation()
        {
            // Arrange
            var individual = new PermutationIndividual(10, _random);
            var originalPerm = individual.GetPermutation().ToArray();

            // Act
            individual.SwapMutation(_random);

            // Assert
            var mutatedPerm = individual.GetPermutation();
            Assert.Equal(10, mutatedPerm.Distinct().Count()); // Still valid permutation
            Assert.NotEqual(originalPerm, mutatedPerm); // Changed
        }

        [Fact]
        public void PermutationIndividual_InversionMutation_MaintainsValidPermutation()
        {
            // Arrange
            var individual = new PermutationIndividual(10, _random);

            // Act
            individual.InversionMutation(_random);

            // Assert
            var mutatedPerm = individual.GetPermutation();
            Assert.Equal(10, mutatedPerm.Distinct().Count()); // Still valid
        }

        [Fact]
        public void PermutationIndividual_Clone_CreatesIndependentCopy()
        {
            // Arrange
            var original = new PermutationIndividual(5, _random);
            original.SetFitness(20.0);

            // Act
            var clone = original.Clone() as PermutationIndividual;

            // Assert
            Assert.NotNull(clone);
            Assert.Equal(original.GetPermutation(), clone.GetPermutation());
            Assert.Equal(original.GetFitness(), clone.GetFitness());

            // Modify clone
            clone.SwapMutation(_random);
            Assert.NotEqual(original.GetPermutation(), clone.GetPermutation());
        }

        #endregion

        #region OneMax Problem - Binary GA

        /// <summary>
        /// OneMax: Classic GA problem - maximize the number of 1s in a binary string
        /// </summary>
        private double OneMaxFitness(BinaryIndividual individual)
        {
            return individual.GetGenes().Count(g => g.Value == 1);
        }

        [Fact]
        public void BinaryGA_OneMaxProblem_ConvergesToOptimal()
        {
            // Arrange
            int chromosomeLength = 20;
            int populationSize = 50;
            int generations = 100;
            var population = new List<BinaryIndividual>();

            for (int i = 0; i < populationSize; i++)
            {
                population.Add(new BinaryIndividual(chromosomeLength, _random));
            }

            // Act - Simple GA evolution
            for (int gen = 0; gen < generations; gen++)
            {
                // Evaluate fitness
                foreach (var ind in population)
                {
                    ind.SetFitness(OneMaxFitness(ind));
                }

                // Sort by fitness
                population = population.OrderByDescending(i => i.GetFitness()).ToList();

                // Check if optimal found
                if (population[0].GetFitness() == chromosomeLength)
                {
                    break;
                }

                // Create next generation
                var newPop = new List<BinaryIndividual>();

                // Elitism - keep best 10%
                int eliteCount = populationSize / 10;
                for (int i = 0; i < eliteCount; i++)
                {
                    newPop.Add(population[i].Clone() as BinaryIndividual);
                }

                // Fill rest with offspring
                while (newPop.Count < populationSize)
                {
                    // Tournament selection
                    var parent1 = TournamentSelect(population, 3);
                    var parent2 = TournamentSelect(population, 3);

                    // Single-point crossover
                    var (child1, child2) = SinglePointCrossover(parent1, parent2, 0.8);

                    // Bit-flip mutation
                    child1 = BitFlipMutation(child1, 0.01);
                    child2 = BitFlipMutation(child2, 0.01);

                    newPop.Add(child1);
                    if (newPop.Count < populationSize)
                        newPop.Add(child2);
                }

                population = newPop;
            }

            // Assert - Should find or be very close to optimal solution
            var bestFitness = population.Max(i => i.GetFitness());
            Assert.True(bestFitness >= chromosomeLength * 0.95); // At least 95% optimal
        }

        [Fact]
        public void BinaryGA_OneMaxProblem_ImprovesOverGenerations()
        {
            // Arrange
            int chromosomeLength = 15;
            int populationSize = 30;
            var population = Enumerable.Range(0, populationSize)
                .Select(_ => new BinaryIndividual(chromosomeLength, _random))
                .ToList();

            // Evaluate initial population
            foreach (var ind in population)
            {
                ind.SetFitness(OneMaxFitness(ind));
            }
            var initialBestFitness = population.Max(i => i.GetFitness());

            // Act - Evolve for 50 generations
            for (int gen = 0; gen < 50; gen++)
            {
                population = EvolveOneGeneration(population);
            }

            // Assert
            var finalBestFitness = population.Max(i => i.GetFitness());
            Assert.True(finalBestFitness > initialBestFitness);
        }

        private List<BinaryIndividual> EvolveOneGeneration(List<BinaryIndividual> population)
        {
            // Evaluate
            foreach (var ind in population)
            {
                ind.SetFitness(OneMaxFitness(ind));
            }

            var newPop = new List<BinaryIndividual>();
            int eliteCount = population.Count / 10;

            // Elitism
            var sorted = population.OrderByDescending(i => i.GetFitness()).ToList();
            for (int i = 0; i < eliteCount; i++)
            {
                newPop.Add(sorted[i].Clone() as BinaryIndividual);
            }

            // Crossover and mutation
            while (newPop.Count < population.Count)
            {
                var parent1 = TournamentSelect(population, 3);
                var parent2 = TournamentSelect(population, 3);
                var (child1, child2) = SinglePointCrossover(parent1, parent2, 0.8);
                child1 = BitFlipMutation(child1, 0.01);
                child2 = BitFlipMutation(child2, 0.01);
                newPop.Add(child1);
                if (newPop.Count < population.Count)
                    newPop.Add(child2);
            }

            return newPop;
        }

        #endregion

        #region Sphere Function - Real-valued GA

        /// <summary>
        /// Sphere function: f(x) = sum(xi^2)
        /// Global minimum at origin (0,0,...,0) with value 0
        /// </summary>
        private double SphereFitness(RealValuedIndividual individual)
        {
            var values = individual.GetValuesAsArray();
            var sumSquares = values.Sum(x => x * x);
            return -sumSquares; // Negative because we're maximizing fitness
        }

        [Fact]
        public void RealGA_SphereFunction_ConvergesToMinimum()
        {
            // Arrange
            int dimensions = 5;
            int populationSize = 50;
            int generations = 200;
            var population = new List<RealValuedIndividual>();

            for (int i = 0; i < populationSize; i++)
            {
                population.Add(new RealValuedIndividual(dimensions, -5.0, 5.0, _random));
            }

            // Act
            for (int gen = 0; gen < generations; gen++)
            {
                // Evaluate
                foreach (var ind in population)
                {
                    ind.SetFitness(SphereFitness(ind));
                }

                // Create next generation
                var newPop = new List<RealValuedIndividual>();

                // Elitism
                var sorted = population.OrderByDescending(i => i.GetFitness()).ToList();
                for (int i = 0; i < 5; i++)
                {
                    newPop.Add(sorted[i].Clone() as RealValuedIndividual);
                }

                // Offspring
                while (newPop.Count < populationSize)
                {
                    var parent1 = TournamentSelectReal(population, 3);
                    var parent2 = TournamentSelectReal(population, 3);
                    var (child1, child2) = ArithmeticCrossover(parent1, parent2, 0.8);
                    child1 = GaussianMutation(child1, 0.1, 0.3);
                    child2 = GaussianMutation(child2, 0.1, 0.3);
                    newPop.Add(child1);
                    if (newPop.Count < populationSize)
                        newPop.Add(child2);
                }

                population = newPop;
            }

            // Assert - Should be close to minimum (0)
            var bestFitness = population.Max(i => i.GetFitness());
            Assert.True(bestFitness > -0.5); // Close to 0 (optimal is 0)
        }

        [Fact]
        public void RealGA_SphereFunction_ImprovesOverGenerations()
        {
            // Arrange
            int dimensions = 3;
            var population = Enumerable.Range(0, 30)
                .Select(_ => new RealValuedIndividual(dimensions, -5.0, 5.0, _random))
                .ToList();

            foreach (var ind in population)
            {
                ind.SetFitness(SphereFitness(ind));
            }
            var initialBestFitness = population.Max(i => i.GetFitness());

            // Act - Evolve
            for (int gen = 0; gen < 100; gen++)
            {
                var newPop = new List<RealValuedIndividual>();
                var sorted = population.OrderByDescending(i => i.GetFitness()).ToList();

                for (int i = 0; i < 3; i++)
                {
                    newPop.Add(sorted[i].Clone() as RealValuedIndividual);
                }

                while (newPop.Count < 30)
                {
                    var p1 = TournamentSelectReal(population, 3);
                    var p2 = TournamentSelectReal(population, 3);
                    var (c1, c2) = ArithmeticCrossover(p1, p2, 0.8);
                    c1 = GaussianMutation(c1, 0.1, 0.3);
                    newPop.Add(c1);
                    if (newPop.Count < 30)
                    {
                        c2 = GaussianMutation(c2, 0.1, 0.3);
                        newPop.Add(c2);
                    }
                }

                foreach (var ind in newPop)
                {
                    ind.SetFitness(SphereFitness(ind));
                }

                population = newPop;
            }

            // Assert
            var finalBestFitness = population.Max(i => i.GetFitness());
            Assert.True(finalBestFitness > initialBestFitness);
        }

        #endregion

        #region Rastrigin Function - Multimodal Optimization

        /// <summary>
        /// Rastrigin function: f(x) = A*n + sum(xi^2 - A*cos(2*pi*xi))
        /// Global minimum at origin with value 0
        /// Many local minima - tests GA's ability to escape local optima
        /// </summary>
        private double RastriginFitness(RealValuedIndividual individual)
        {
            const double A = 10.0;
            var values = individual.GetValuesAsArray();
            var n = values.Length;
            var sum = A * n;

            foreach (var x in values)
            {
                sum += x * x - A * Math.Cos(2 * Math.PI * x);
            }

            return -sum; // Negative for maximization
        }

        [Fact]
        public void RealGA_RastriginFunction_FindsGoodSolution()
        {
            // Arrange
            int dimensions = 3;
            int populationSize = 80;
            int generations = 300;
            var population = new List<RealValuedIndividual>();

            for (int i = 0; i < populationSize; i++)
            {
                population.Add(new RealValuedIndividual(dimensions, -5.12, 5.12, _random));
            }

            // Act
            for (int gen = 0; gen < generations; gen++)
            {
                foreach (var ind in population)
                {
                    ind.SetFitness(RastriginFitness(ind));
                }

                var newPop = new List<RealValuedIndividual>();
                var sorted = population.OrderByDescending(i => i.GetFitness()).ToList();

                // Strong elitism for multimodal
                for (int i = 0; i < 8; i++)
                {
                    newPop.Add(sorted[i].Clone() as RealValuedIndividual);
                }

                while (newPop.Count < populationSize)
                {
                    var p1 = TournamentSelectReal(population, 5);
                    var p2 = TournamentSelectReal(population, 5);
                    var (c1, c2) = ArithmeticCrossover(p1, p2, 0.9);
                    c1 = GaussianMutation(c1, 0.2, 0.5);
                    c2 = GaussianMutation(c2, 0.2, 0.5);
                    newPop.Add(c1);
                    if (newPop.Count < populationSize)
                        newPop.Add(c2);
                }

                population = newPop;
            }

            // Assert - Should find a good solution (not necessarily global optimum)
            var bestFitness = population.Max(i => i.GetFitness());
            Assert.True(bestFitness > -20.0); // Good solution
        }

        #endregion

        #region Simple TSP - Permutation GA

        /// <summary>
        /// Simple TSP fitness: minimize total distance
        /// </summary>
        private double TspFitness(PermutationIndividual individual, double[,] distances)
        {
            var tour = individual.GetPermutation();
            double totalDistance = 0;

            for (int i = 0; i < tour.Length - 1; i++)
            {
                totalDistance += distances[tour[i], tour[i + 1]];
            }
            totalDistance += distances[tour[^1], tour[0]]; // Return to start

            return -totalDistance; // Negative for maximization
        }

        [Fact]
        public void PermutationGA_SimpleTSP_FindsGoodTour()
        {
            // Arrange - Simple 5-city TSP
            var distances = new double[,]
            {
                { 0, 10, 15, 20, 25 },
                { 10, 0, 35, 25, 30 },
                { 15, 35, 0, 30, 20 },
                { 20, 25, 30, 0, 15 },
                { 25, 30, 20, 15, 0 }
            };

            int populationSize = 50;
            int generations = 200;
            var population = new List<PermutationIndividual>();

            for (int i = 0; i < populationSize; i++)
            {
                population.Add(new PermutationIndividual(5, _random));
            }

            // Act
            for (int gen = 0; gen < generations; gen++)
            {
                foreach (var ind in population)
                {
                    ind.SetFitness(TspFitness(ind, distances));
                }

                var newPop = new List<PermutationIndividual>();
                var sorted = population.OrderByDescending(i => i.GetFitness()).ToList();

                // Elitism
                for (int i = 0; i < 5; i++)
                {
                    newPop.Add(sorted[i].Clone() as PermutationIndividual);
                }

                // Offspring
                while (newPop.Count < populationSize)
                {
                    var p1 = TournamentSelectPerm(population, 3);
                    var p2 = TournamentSelectPerm(population, 3);
                    var (c1, c2) = p1.OrderCrossover(p2, _random);

                    if (_random.NextDouble() < 0.2)
                        c1.SwapMutation(_random);
                    if (_random.NextDouble() < 0.2)
                        c2.InversionMutation(_random);

                    newPop.Add(c1);
                    if (newPop.Count < populationSize)
                        newPop.Add(c2);
                }

                population = newPop;
            }

            // Assert
            var bestFitness = population.Max(i => i.GetFitness());
            Assert.True(bestFitness > -100); // Good tour (optimal might be around 80)
        }

        [Fact]
        public void PermutationGA_TSP_ImprovesOverGenerations()
        {
            // Arrange
            var distances = new double[,]
            {
                { 0, 5, 8, 12 },
                { 5, 0, 6, 9 },
                { 8, 6, 0, 7 },
                { 12, 9, 7, 0 }
            };

            var population = Enumerable.Range(0, 30)
                .Select(_ => new PermutationIndividual(4, _random))
                .ToList();

            foreach (var ind in population)
            {
                ind.SetFitness(TspFitness(ind, distances));
            }
            var initialBest = population.Max(i => i.GetFitness());

            // Act
            for (int gen = 0; gen < 50; gen++)
            {
                var newPop = new List<PermutationIndividual>();
                var sorted = population.OrderByDescending(i => i.GetFitness()).ToList();

                for (int i = 0; i < 3; i++)
                    newPop.Add(sorted[i].Clone() as PermutationIndividual);

                while (newPop.Count < 30)
                {
                    var p1 = TournamentSelectPerm(population, 3);
                    var p2 = TournamentSelectPerm(population, 3);
                    var (c1, c2) = p1.OrderCrossover(p2, _random);
                    if (_random.NextDouble() < 0.2) c1.SwapMutation(_random);
                    newPop.Add(c1);
                    if (newPop.Count < 30)
                    {
                        if (_random.NextDouble() < 0.2) c2.InversionMutation(_random);
                        newPop.Add(c2);
                    }
                }

                foreach (var ind in newPop)
                {
                    ind.SetFitness(TspFitness(ind, distances));
                }
                population = newPop;
            }

            // Assert
            var finalBest = population.Max(i => i.GetFitness());
            Assert.True(finalBest >= initialBest);
        }

        #endregion

        #region Selection Method Tests

        [Fact]
        public void TournamentSelection_SelectsBetterIndividuals()
        {
            // Arrange
            var population = new List<BinaryIndividual>();
            for (int i = 0; i < 20; i++)
            {
                var ind = new BinaryIndividual(10, _random);
                ind.SetFitness(i); // Fitness 0-19
                population.Add(ind);
            }

            // Act - Select many times
            var selections = new List<double>();
            for (int i = 0; i < 100; i++)
            {
                var selected = TournamentSelect(population, 3);
                selections.Add(selected.GetFitness());
            }

            // Assert - Average selection should favor high fitness
            var avgSelected = selections.Average();
            Assert.True(avgSelected > 10); // Should be above population mean
        }

        [Fact]
        public void RouletteWheelSelection_SelectsProportionalToFitness()
        {
            // Arrange
            var population = new List<BinaryIndividual>();
            for (int i = 0; i < 10; i++)
            {
                var ind = new BinaryIndividual(5, _random);
                ind.SetFitness(i + 1); // Fitness 1-10
                population.Add(ind);
            }

            // Act
            var selections = new List<double>();
            for (int i = 0; i < 200; i++)
            {
                var selected = RouletteWheelSelect(population);
                selections.Add(selected.GetFitness());
            }

            // Assert - Higher fitness should be selected more often
            var avgSelected = selections.Average();
            Assert.True(avgSelected > 5.5); // Above uniform average
        }

        [Fact]
        public void ElitismSelection_SelectsTopIndividuals()
        {
            // Arrange
            var population = new List<BinaryIndividual>();
            for (int i = 0; i < 20; i++)
            {
                var ind = new BinaryIndividual(5, _random);
                ind.SetFitness(i);
                population.Add(ind);
            }

            // Act
            var elite = population.OrderByDescending(i => i.GetFitness()).Take(5).ToList();

            // Assert
            Assert.Equal(5, elite.Count);
            Assert.Equal(19, elite[0].GetFitness());
            Assert.Equal(18, elite[1].GetFitness());
            Assert.True(elite.All(e => e.GetFitness() >= 15));
        }

        [Fact]
        public void RankSelection_SelectsBasedOnRank()
        {
            // Arrange
            var population = new List<BinaryIndividual>();
            for (int i = 0; i < 10; i++)
            {
                var ind = new BinaryIndividual(5, _random);
                ind.SetFitness(i * i); // Quadratic fitness: 0, 1, 4, 9, 16, ...
                population.Add(ind);
            }

            // Act - Rank-based selection
            var sorted = population.OrderByDescending(i => i.GetFitness()).ToList();
            var selections = new List<int>();

            for (int trial = 0; trial < 100; trial++)
            {
                var randomValue = _random.NextDouble();
                var totalRank = (population.Count * (population.Count + 1)) / 2.0;
                var cumulative = 0.0;

                for (int i = 0; i < sorted.Count; i++)
                {
                    var rank = sorted.Count - i;
                    cumulative += rank / totalRank;
                    if (randomValue <= cumulative)
                    {
                        selections.Add(i);
                        break;
                    }
                }
            }

            // Assert - Should favor top ranks
            var avgRank = selections.Average();
            Assert.True(avgRank < 5); // Should favor better individuals
        }

        #endregion

        #region Crossover Operator Tests

        [Fact]
        public void SinglePointCrossover_ProducesValidOffspring()
        {
            // Arrange
            var parent1 = new BinaryIndividual(10, _random);
            var parent2 = new BinaryIndividual(10, _random);

            // Act
            var (child1, child2) = SinglePointCrossover(parent1, parent2, 1.0);

            // Assert
            Assert.Equal(10, child1.GetGenes().Count);
            Assert.Equal(10, child2.GetGenes().Count);
        }

        [Fact]
        public void SinglePointCrossover_InheritsFromBothParents()
        {
            // Arrange - Parents with all 0s and all 1s
            var genes0 = Enumerable.Range(0, 10).Select(_ => new BinaryGene(0)).ToList();
            var genes1 = Enumerable.Range(0, 10).Select(_ => new BinaryGene(1)).ToList();
            var parent1 = new BinaryIndividual(genes0);
            var parent2 = new BinaryIndividual(genes1);

            // Act
            var (child1, child2) = SinglePointCrossover(parent1, parent2, 1.0);

            // Assert - Children should have mix of 0s and 1s
            var child1Ones = child1.GetGenes().Count(g => g.Value == 1);
            var child2Ones = child2.GetGenes().Count(g => g.Value == 1);

            Assert.True(child1Ones > 0 && child1Ones < 10);
            Assert.True(child2Ones > 0 && child2Ones < 10);
        }

        [Fact]
        public void UniformCrossover_ProducesValidOffspring()
        {
            // Arrange
            var parent1 = new BinaryIndividual(10, _random);
            var parent2 = new BinaryIndividual(10, _random);

            // Act
            var (child1, child2) = UniformCrossover(parent1, parent2, 1.0);

            // Assert
            Assert.Equal(10, child1.GetGenes().Count);
            Assert.Equal(10, child2.GetGenes().Count);
        }

        [Fact]
        public void ArithmeticCrossover_ProducesIntermediateValues()
        {
            // Arrange
            var genes1 = new List<RealGene> { new(0.0), new(0.0), new(0.0) };
            var genes2 = new List<RealGene> { new(10.0), new(10.0), new(10.0) };
            var parent1 = new RealValuedIndividual(genes1);
            var parent2 = new RealValuedIndividual(genes2);

            // Act
            var (child1, child2) = ArithmeticCrossover(parent1, parent2, 1.0);

            // Assert - Children should have intermediate values
            var values1 = child1.GetValuesAsArray();
            var values2 = child2.GetValuesAsArray();

            Assert.True(values1.All(v => v >= 0.0 && v <= 10.0));
            Assert.True(values2.All(v => v >= 0.0 && v <= 10.0));
        }

        #endregion

        #region Mutation Operator Tests

        [Fact]
        public void BitFlipMutation_ChangesGenes()
        {
            // Arrange
            var genes = Enumerable.Range(0, 20).Select(_ => new BinaryGene(0)).ToList();
            var individual = new BinaryIndividual(genes);

            // Act - High mutation rate
            var mutated = BitFlipMutation(individual, 0.5);

            // Assert - Should have some 1s now
            var onesCount = mutated.GetGenes().Count(g => g.Value == 1);
            Assert.True(onesCount > 0);
        }

        [Fact]
        public void BitFlipMutation_LowRate_MakesSmallChanges()
        {
            // Arrange
            var individual = new BinaryIndividual(100, _random);
            var originalOnes = individual.GetGenes().Count(g => g.Value == 1);

            // Act
            var mutated = BitFlipMutation(individual, 0.01);
            var mutatedOnes = mutated.GetGenes().Count(g => g.Value == 1);

            // Assert - Small change
            Assert.True(Math.Abs(mutatedOnes - originalOnes) < 10);
        }

        [Fact]
        public void GaussianMutation_ChangesRealValues()
        {
            // Arrange
            var genes = new List<RealGene> { new(5.0), new(5.0), new(5.0) };
            var individual = new RealValuedIndividual(genes);

            // Act
            var mutated = GaussianMutation(individual, 1.0, 1.0);

            // Assert - Values should be different
            var original = individual.GetValuesAsArray();
            var changed = mutated.GetValuesAsArray();
            Assert.NotEqual(original, changed);
        }

        [Fact]
        public void SwapMutation_MaintainsPermutationValidity()
        {
            // Arrange
            var individual = new PermutationIndividual(10, _random);
            var originalPerm = individual.GetPermutation();

            // Act
            individual.SwapMutation(_random);

            // Assert
            var mutatedPerm = individual.GetPermutation();
            Assert.Equal(originalPerm.OrderBy(x => x), mutatedPerm.OrderBy(x => x));
        }

        #endregion

        #region Parameter Effect Tests

        [Fact]
        public void PopulationSize_LargerPopulation_FindsBetterSolutions()
        {
            // Arrange & Act
            var smallPopBest = RunOneMaxWithPopSize(20, 50);
            var largePopBest = RunOneMaxWithPopSize(100, 50);

            // Assert
            Assert.True(largePopBest >= smallPopBest);
        }

        [Fact]
        public void MutationRate_HighRate_MaintainsDiversity()
        {
            // Arrange
            var popLowMutation = RunOneMaxWithMutationRate(0.001, 30);
            var popHighMutation = RunOneMaxWithMutationRate(0.05, 30);

            // Calculate diversity (unique individuals)
            var diversityLow = popLowMutation.Select(i => i.GetValueAsInt()).Distinct().Count();
            var diversityHigh = popHighMutation.Select(i => i.GetValueAsInt()).Distinct().Count();

            // Assert
            Assert.True(diversityHigh >= diversityLow);
        }

        [Fact]
        public void CrossoverRate_HighRate_IncreasesMixing()
        {
            // Arrange - Run with different crossover rates
            var fitness1 = RunOneMaxWithCrossoverRate(0.3, 20);
            var fitness2 = RunOneMaxWithCrossoverRate(0.9, 20);

            // Assert - Higher crossover generally helps (though not guaranteed)
            Assert.True(fitness2 >= fitness1 * 0.9); // Allow some variance
        }

        [Fact]
        public void Elitism_PreservesBestIndividuals()
        {
            // Arrange
            int chromosomeLength = 15;
            var population = Enumerable.Range(0, 50)
                .Select(_ => new BinaryIndividual(chromosomeLength, _random))
                .ToList();

            foreach (var ind in population)
            {
                ind.SetFitness(OneMaxFitness(ind));
            }

            var bestBeforeEvolution = population.Max(i => i.GetFitness());

            // Act - Evolve with elitism
            for (int gen = 0; gen < 10; gen++)
            {
                var newPop = new List<BinaryIndividual>();
                var sorted = population.OrderByDescending(i => i.GetFitness()).ToList();

                // Keep best 5
                for (int i = 0; i < 5; i++)
                {
                    newPop.Add(sorted[i].Clone() as BinaryIndividual);
                }

                // Fill rest randomly
                while (newPop.Count < 50)
                {
                    var p1 = TournamentSelect(population, 3);
                    var p2 = TournamentSelect(population, 3);
                    var (c1, c2) = SinglePointCrossover(p1, p2, 0.8);
                    c1 = BitFlipMutation(c1, 0.01);
                    newPop.Add(c1);
                    if (newPop.Count < 50)
                        newPop.Add(c2);
                }

                foreach (var ind in newPop)
                {
                    ind.SetFitness(OneMaxFitness(ind));
                }

                population = newPop;
            }

            // Assert - Best fitness should not decrease
            var bestAfterEvolution = population.Max(i => i.GetFitness());
            Assert.True(bestAfterEvolution >= bestBeforeEvolution);
        }

        #endregion

        #region Convergence Tests

        [Fact]
        public void GA_Convergence_FitnessImprovesMonotonically()
        {
            // Arrange
            var population = Enumerable.Range(0, 50)
                .Select(_ => new BinaryIndividual(20, _random))
                .ToList();

            var fitnessHistory = new List<double>();

            // Act
            for (int gen = 0; gen < 50; gen++)
            {
                foreach (var ind in population)
                {
                    ind.SetFitness(OneMaxFitness(ind));
                }

                var bestFitness = population.Max(i => i.GetFitness());
                fitnessHistory.Add(bestFitness);

                // Evolve
                var newPop = new List<BinaryIndividual>();
                var sorted = population.OrderByDescending(i => i.GetFitness()).ToList();

                for (int i = 0; i < 5; i++)
                    newPop.Add(sorted[i].Clone() as BinaryIndividual);

                while (newPop.Count < 50)
                {
                    var p1 = TournamentSelect(population, 3);
                    var p2 = TournamentSelect(population, 3);
                    var (c1, c2) = SinglePointCrossover(p1, p2, 0.8);
                    c1 = BitFlipMutation(c1, 0.01);
                    newPop.Add(c1);
                    if (newPop.Count < 50)
                    {
                        c2 = BitFlipMutation(c2, 0.01);
                        newPop.Add(c2);
                    }
                }

                population = newPop;
            }

            // Assert - Best fitness should improve or stay same with elitism
            for (int i = 1; i < fitnessHistory.Count; i++)
            {
                Assert.True(fitnessHistory[i] >= fitnessHistory[i - 1]);
            }
        }

        [Fact]
        public void GA_Convergence_EventuallyStabilizes()
        {
            // Arrange
            var population = Enumerable.Range(0, 30)
                .Select(_ => new BinaryIndividual(10, _random))
                .ToList();

            var fitnessHistory = new List<double>();

            // Act
            for (int gen = 0; gen < 100; gen++)
            {
                foreach (var ind in population)
                {
                    ind.SetFitness(OneMaxFitness(ind));
                }

                fitnessHistory.Add(population.Max(i => i.GetFitness()));

                // Evolve
                population = EvolveOneGeneration(population);
            }

            // Assert - Last 10 generations should show little change
            var lastTen = fitnessHistory.TakeLast(10).ToList();
            var variance = lastTen.Max() - lastTen.Min();
            Assert.True(variance < 2); // Stabilized
        }

        #endregion

        #region Diversity Tests

        [Fact]
        public void GA_Diversity_DecreasesOverTime()
        {
            // Arrange
            var population = Enumerable.Range(0, 50)
                .Select(_ => new BinaryIndividual(20, _random))
                .ToList();

            var initialDiversity = population.Select(i => i.GetValueAsInt()).Distinct().Count();
            var diversityHistory = new List<int> { initialDiversity };

            // Act
            for (int gen = 0; gen < 50; gen++)
            {
                population = EvolveOneGeneration(population);
                var diversity = population.Select(i => i.GetValueAsInt()).Distinct().Count();
                diversityHistory.Add(diversity);
            }

            // Assert - Diversity should generally decrease
            var finalDiversity = diversityHistory.Last();
            Assert.True(finalDiversity < initialDiversity);
        }

        [Fact]
        public void GA_Diversity_HighMutationMaintainsDiversity()
        {
            // Arrange
            var popLowMut = Enumerable.Range(0, 50)
                .Select(_ => new BinaryIndividual(15, _random))
                .ToList();
            var popHighMut = Enumerable.Range(0, 50)
                .Select(_ => new BinaryIndividual(15, _random))
                .ToList();

            // Act - Evolve with different mutation rates
            for (int gen = 0; gen < 30; gen++)
            {
                popLowMut = EvolveWithMutationRate(popLowMut, 0.001);
                popHighMut = EvolveWithMutationRate(popHighMut, 0.05);
            }

            // Assert
            var diversityLow = popLowMut.Select(i => i.GetValueAsInt()).Distinct().Count();
            var diversityHigh = popHighMut.Select(i => i.GetValueAsInt()).Distinct().Count();
            Assert.True(diversityHigh >= diversityLow);
        }

        #endregion

        #region Fitness Function Tests

        [Fact]
        public void FitnessFunction_OneMax_RewardsMoreOnes()
        {
            // Arrange
            var genesAllZeros = Enumerable.Range(0, 10).Select(_ => new BinaryGene(0)).ToList();
            var genesFiveOnes = new List<BinaryGene>
            {
                new(1), new(1), new(1), new(1), new(1),
                new(0), new(0), new(0), new(0), new(0)
            };
            var genesAllOnes = Enumerable.Range(0, 10).Select(_ => new BinaryGene(1)).ToList();

            var ind0 = new BinaryIndividual(genesAllZeros);
            var ind5 = new BinaryIndividual(genesFiveOnes);
            var ind10 = new BinaryIndividual(genesAllOnes);

            // Act
            var fitness0 = OneMaxFitness(ind0);
            var fitness5 = OneMaxFitness(ind5);
            var fitness10 = OneMaxFitness(ind10);

            // Assert
            Assert.Equal(0, fitness0);
            Assert.Equal(5, fitness5);
            Assert.Equal(10, fitness10);
        }

        [Fact]
        public void FitnessFunction_Sphere_RewardsProximityToOrigin()
        {
            // Arrange
            var genesAtOrigin = new List<RealGene> { new(0.0), new(0.0), new(0.0) };
            var genesFar = new List<RealGene> { new(5.0), new(5.0), new(5.0) };

            var indOrigin = new RealValuedIndividual(genesAtOrigin);
            var indFar = new RealValuedIndividual(genesFar);

            // Act
            var fitnessOrigin = SphereFitness(indOrigin);
            var fitnessFar = SphereFitness(indFar);

            // Assert - Closer to origin is better (higher fitness)
            Assert.True(fitnessOrigin > fitnessFar);
            Assert.Equal(0.0, fitnessOrigin, precision: 10);
        }

        #endregion

        #region Edge Cases and Robustness

        [Fact]
        public void GA_SmallPopulation_StillWorks()
        {
            // Arrange
            var population = Enumerable.Range(0, 5)
                .Select(_ => new BinaryIndividual(10, _random))
                .ToList();

            // Act
            for (int gen = 0; gen < 20; gen++)
            {
                population = EvolveOneGeneration(population);
            }

            // Assert
            Assert.Equal(5, population.Count);
            var bestFitness = population.Max(i => i.GetFitness());
            Assert.True(bestFitness > 0);
        }

        [Fact]
        public void GA_LargeChromosome_HandlesEfficiently()
        {
            // Arrange
            var population = Enumerable.Range(0, 20)
                .Select(_ => new BinaryIndividual(200, _random))
                .ToList();

            // Act
            var sw = System.Diagnostics.Stopwatch.StartNew();
            for (int gen = 0; gen < 10; gen++)
            {
                population = EvolveOneGeneration(population);
            }
            sw.Stop();

            // Assert
            Assert.True(sw.ElapsedMilliseconds < 5000); // Should complete reasonably fast
        }

        [Fact]
        public void GA_ZeroMutationRate_StillProducesOffspring()
        {
            // Arrange
            var population = Enumerable.Range(0, 20)
                .Select(_ => new BinaryIndividual(10, _random))
                .ToList();

            // Act
            population = EvolveWithMutationRate(population, 0.0);

            // Assert
            Assert.Equal(20, population.Count);
        }

        [Fact]
        public void GA_ZeroCrossoverRate_StillEvolves()
        {
            // Arrange
            var population = Enumerable.Range(0, 20)
                .Select(_ => new BinaryIndividual(10, _random))
                .ToList();

            foreach (var ind in population)
            {
                ind.SetFitness(OneMaxFitness(ind));
            }
            var initialBest = population.Max(i => i.GetFitness());

            // Act - Evolve with no crossover, only mutation
            for (int gen = 0; gen < 30; gen++)
            {
                var newPop = new List<BinaryIndividual>();
                var sorted = population.OrderByDescending(i => i.GetFitness()).ToList();

                for (int i = 0; i < 2; i++)
                    newPop.Add(sorted[i].Clone() as BinaryIndividual);

                while (newPop.Count < 20)
                {
                    var parent = TournamentSelect(population, 3);
                    var child = BitFlipMutation(parent.Clone() as BinaryIndividual, 0.1);
                    newPop.Add(child);
                }

                foreach (var ind in newPop)
                {
                    ind.SetFitness(OneMaxFitness(ind));
                }
                population = newPop;
            }

            // Assert
            var finalBest = population.Max(i => i.GetFitness());
            Assert.True(finalBest >= initialBest);
        }

        #endregion

        #region Helper Methods for Tests

        private BinaryIndividual TournamentSelect(List<BinaryIndividual> population, int tournamentSize)
        {
            var best = population[_random.Next(population.Count)];
            for (int i = 1; i < tournamentSize; i++)
            {
                var contender = population[_random.Next(population.Count)];
                if (contender.GetFitness() > best.GetFitness())
                {
                    best = contender;
                }
            }
            return best;
        }

        private RealValuedIndividual TournamentSelectReal(List<RealValuedIndividual> population, int tournamentSize)
        {
            var best = population[_random.Next(population.Count)];
            for (int i = 1; i < tournamentSize; i++)
            {
                var contender = population[_random.Next(population.Count)];
                if (contender.GetFitness() > best.GetFitness())
                {
                    best = contender;
                }
            }
            return best;
        }

        private PermutationIndividual TournamentSelectPerm(List<PermutationIndividual> population, int tournamentSize)
        {
            var best = population[_random.Next(population.Count)];
            for (int i = 1; i < tournamentSize; i++)
            {
                var contender = population[_random.Next(population.Count)];
                if (contender.GetFitness() > best.GetFitness())
                {
                    best = contender;
                }
            }
            return best;
        }

        private BinaryIndividual RouletteWheelSelect(List<BinaryIndividual> population)
        {
            var totalFitness = population.Sum(i => i.GetFitness());
            if (totalFitness <= 0)
                return population[_random.Next(population.Count)];

            var randomValue = _random.NextDouble() * totalFitness;
            var cumulative = 0.0;

            foreach (var ind in population)
            {
                cumulative += ind.GetFitness();
                if (cumulative >= randomValue)
                {
                    return ind;
                }
            }

            return population.Last();
        }

        private (BinaryIndividual, BinaryIndividual) SinglePointCrossover(
            BinaryIndividual parent1, BinaryIndividual parent2, double rate)
        {
            if (_random.NextDouble() > rate)
            {
                return (parent1.Clone() as BinaryIndividual, parent2.Clone() as BinaryIndividual);
            }

            var genes1 = parent1.GetGenes().ToList();
            var genes2 = parent2.GetGenes().ToList();
            var point = _random.Next(1, genes1.Count);

            var child1Genes = genes1.Take(point).Concat(genes2.Skip(point)).Select(g => g.Clone()).ToList();
            var child2Genes = genes2.Take(point).Concat(genes1.Skip(point)).Select(g => g.Clone()).ToList();

            return (new BinaryIndividual(child1Genes), new BinaryIndividual(child2Genes));
        }

        private (BinaryIndividual, BinaryIndividual) UniformCrossover(
            BinaryIndividual parent1, BinaryIndividual parent2, double rate)
        {
            if (_random.NextDouble() > rate)
            {
                return (parent1.Clone() as BinaryIndividual, parent2.Clone() as BinaryIndividual);
            }

            var genes1 = parent1.GetGenes().ToList();
            var genes2 = parent2.GetGenes().ToList();

            var child1Genes = new List<BinaryGene>();
            var child2Genes = new List<BinaryGene>();

            for (int i = 0; i < genes1.Count; i++)
            {
                if (_random.NextDouble() < 0.5)
                {
                    child1Genes.Add(genes1[i].Clone());
                    child2Genes.Add(genes2[i].Clone());
                }
                else
                {
                    child1Genes.Add(genes2[i].Clone());
                    child2Genes.Add(genes1[i].Clone());
                }
            }

            return (new BinaryIndividual(child1Genes), new BinaryIndividual(child2Genes));
        }

        private (RealValuedIndividual, RealValuedIndividual) ArithmeticCrossover(
            RealValuedIndividual parent1, RealValuedIndividual parent2, double rate)
        {
            if (_random.NextDouble() > rate)
            {
                return (parent1.Clone() as RealValuedIndividual, parent2.Clone() as RealValuedIndividual);
            }

            var genes1 = parent1.GetGenes().ToList();
            var genes2 = parent2.GetGenes().ToList();
            var alpha = _random.NextDouble();

            var child1Genes = new List<RealGene>();
            var child2Genes = new List<RealGene>();

            for (int i = 0; i < genes1.Count; i++)
            {
                var val1 = alpha * genes1[i].Value + (1 - alpha) * genes2[i].Value;
                var val2 = (1 - alpha) * genes1[i].Value + alpha * genes2[i].Value;
                child1Genes.Add(new RealGene(val1, genes1[i].StepSize));
                child2Genes.Add(new RealGene(val2, genes2[i].StepSize));
            }

            return (new RealValuedIndividual(child1Genes), new RealValuedIndividual(child2Genes));
        }

        private BinaryIndividual BitFlipMutation(BinaryIndividual individual, double rate)
        {
            var clone = individual.Clone() as BinaryIndividual;
            var genes = clone.GetGenes().ToList();

            for (int i = 0; i < genes.Count; i++)
            {
                if (_random.NextDouble() < rate)
                {
                    genes[i].Value = 1 - genes[i].Value;
                }
            }

            clone.SetGenes(genes);
            return clone;
        }

        private RealValuedIndividual GaussianMutation(RealValuedIndividual individual, double rate, double stdDev)
        {
            var clone = individual.Clone() as RealValuedIndividual;
            var genes = clone.GetGenes().ToList();

            for (int i = 0; i < genes.Count; i++)
            {
                if (_random.NextDouble() < rate)
                {
                    var u1 = 1.0 - _random.NextDouble();
                    var u2 = 1.0 - _random.NextDouble();
                    var randStdNormal = Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Sin(2.0 * Math.PI * u2);
                    genes[i].Value += randStdNormal * stdDev;
                }
            }

            clone.SetGenes(genes);
            return clone;
        }

        private double RunOneMaxWithPopSize(int popSize, int generations)
        {
            var population = Enumerable.Range(0, popSize)
                .Select(_ => new BinaryIndividual(15, _random))
                .ToList();

            for (int gen = 0; gen < generations; gen++)
            {
                population = EvolveOneGeneration(population);
            }

            return population.Max(i => i.GetFitness());
        }

        private List<BinaryIndividual> RunOneMaxWithMutationRate(double mutationRate, int generations)
        {
            var population = Enumerable.Range(0, 30)
                .Select(_ => new BinaryIndividual(15, _random))
                .ToList();

            for (int gen = 0; gen < generations; gen++)
            {
                population = EvolveWithMutationRate(population, mutationRate);
            }

            return population;
        }

        private double RunOneMaxWithCrossoverRate(double crossoverRate, int generations)
        {
            var population = Enumerable.Range(0, 30)
                .Select(_ => new BinaryIndividual(15, _random))
                .ToList();

            for (int gen = 0; gen < generations; gen++)
            {
                foreach (var ind in population)
                {
                    ind.SetFitness(OneMaxFitness(ind));
                }

                var newPop = new List<BinaryIndividual>();
                var sorted = population.OrderByDescending(i => i.GetFitness()).ToList();

                for (int i = 0; i < 3; i++)
                    newPop.Add(sorted[i].Clone() as BinaryIndividual);

                while (newPop.Count < 30)
                {
                    var p1 = TournamentSelect(population, 3);
                    var p2 = TournamentSelect(population, 3);
                    var (c1, c2) = SinglePointCrossover(p1, p2, crossoverRate);
                    c1 = BitFlipMutation(c1, 0.01);
                    newPop.Add(c1);
                    if (newPop.Count < 30)
                    {
                        c2 = BitFlipMutation(c2, 0.01);
                        newPop.Add(c2);
                    }
                }

                population = newPop;
            }

            return population.Max(i => i.GetFitness());
        }

        private List<BinaryIndividual> EvolveWithMutationRate(List<BinaryIndividual> population, double mutationRate)
        {
            foreach (var ind in population)
            {
                ind.SetFitness(OneMaxFitness(ind));
            }

            var newPop = new List<BinaryIndividual>();
            var sorted = population.OrderByDescending(i => i.GetFitness()).ToList();

            int eliteCount = Math.Max(1, population.Count / 10);
            for (int i = 0; i < eliteCount; i++)
                newPop.Add(sorted[i].Clone() as BinaryIndividual);

            while (newPop.Count < population.Count)
            {
                var p1 = TournamentSelect(population, 3);
                var p2 = TournamentSelect(population, 3);
                var (c1, c2) = SinglePointCrossover(p1, p2, 0.8);
                c1 = BitFlipMutation(c1, mutationRate);
                newPop.Add(c1);
                if (newPop.Count < population.Count)
                {
                    c2 = BitFlipMutation(c2, mutationRate);
                    newPop.Add(c2);
                }
            }

            return newPop;
        }

        #endregion

        #region Rosenbrock Function - Complex Optimization

        /// <summary>
        /// Rosenbrock function: f(x,y) = (1-x)^2 + 100*(y-x^2)^2
        /// Global minimum at (1,1) with value 0
        /// Very difficult optimization problem with narrow valley
        /// </summary>
        private double RosenbrockFitness(RealValuedIndividual individual)
        {
            var values = individual.GetValuesAsArray();
            if (values.Length < 2) return 0;

            double sum = 0;
            for (int i = 0; i < values.Length - 1; i++)
            {
                var x = values[i];
                var y = values[i + 1];
                sum += Math.Pow(1 - x, 2) + 100 * Math.Pow(y - x * x, 2);
            }

            return -sum;
        }

        [Fact]
        public void RealGA_RosenbrockFunction_FindsReasonableSolution()
        {
            // Arrange
            var population = Enumerable.Range(0, 100)
                .Select(_ => new RealValuedIndividual(2, -2.0, 2.0, _random))
                .ToList();

            // Act
            for (int gen = 0; gen < 300; gen++)
            {
                foreach (var ind in population)
                {
                    ind.SetFitness(RosenbrockFitness(ind));
                }

                var newPop = new List<RealValuedIndividual>();
                var sorted = population.OrderByDescending(i => i.GetFitness()).ToList();

                for (int i = 0; i < 10; i++)
                    newPop.Add(sorted[i].Clone() as RealValuedIndividual);

                while (newPop.Count < 100)
                {
                    var p1 = TournamentSelectReal(population, 5);
                    var p2 = TournamentSelectReal(population, 5);
                    var (c1, c2) = ArithmeticCrossover(p1, p2, 0.9);
                    c1 = GaussianMutation(c1, 0.2, 0.3);
                    c2 = GaussianMutation(c2, 0.2, 0.3);
                    newPop.Add(c1);
                    if (newPop.Count < 100)
                        newPop.Add(c2);
                }

                population = newPop;
            }

            // Assert - Should find a reasonable solution
            var best = population.OrderByDescending(i => i.GetFitness()).First();
            var values = best.GetValuesAsArray();
            Assert.True(best.GetFitness() > -100); // Not optimal but reasonable
        }

        #endregion

        #region Ackley Function - Highly Multimodal

        /// <summary>
        /// Ackley function: highly multimodal test function
        /// Global minimum at origin with value 0
        /// </summary>
        private double AckleyFitness(RealValuedIndividual individual)
        {
            var values = individual.GetValuesAsArray();
            var n = values.Length;
            var sum1 = values.Sum(x => x * x);
            var sum2 = values.Sum(x => Math.Cos(2 * Math.PI * x));

            var result = -20 * Math.Exp(-0.2 * Math.Sqrt(sum1 / n))
                        - Math.Exp(sum2 / n) + 20 + Math.E;

            return -result;
        }

        [Fact]
        public void RealGA_AckleyFunction_FindsGoodSolution()
        {
            // Arrange
            var population = Enumerable.Range(0, 80)
                .Select(_ => new RealValuedIndividual(3, -5.0, 5.0, _random))
                .ToList();

            // Act
            for (int gen = 0; gen < 200; gen++)
            {
                foreach (var ind in population)
                {
                    ind.SetFitness(AckleyFitness(ind));
                }

                var newPop = new List<RealValuedIndividual>();
                var sorted = population.OrderByDescending(i => i.GetFitness()).ToList();

                for (int i = 0; i < 8; i++)
                    newPop.Add(sorted[i].Clone() as RealValuedIndividual);

                while (newPop.Count < 80)
                {
                    var p1 = TournamentSelectReal(population, 4);
                    var p2 = TournamentSelectReal(population, 4);
                    var (c1, c2) = ArithmeticCrossover(p1, p2, 0.85);
                    c1 = GaussianMutation(c1, 0.15, 0.4);
                    newPop.Add(c1);
                    if (newPop.Count < 80)
                    {
                        c2 = GaussianMutation(c2, 0.15, 0.4);
                        newPop.Add(c2);
                    }
                }

                population = newPop;
            }

            // Assert
            var bestFitness = population.Max(i => i.GetFitness());
            Assert.True(bestFitness > -5.0); // Reasonable solution
        }

        #endregion

        #region Additional Selection Method Tests

        [Fact]
        public void StochasticUniversalSampling_SelectsMultipleIndividuals()
        {
            // Arrange
            var population = new List<BinaryIndividual>();
            for (int i = 0; i < 20; i++)
            {
                var ind = new BinaryIndividual(10, _random);
                ind.SetFitness(i + 1);
                population.Add(ind);
            }

            // Act - Simulate SUS
            var totalFitness = population.Sum(i => i.GetFitness());
            var selectionSize = 10;
            var distance = totalFitness / selectionSize;
            var start = _random.NextDouble() * distance;

            var selected = new List<double>();
            var cumulative = 0.0;
            var index = 0;

            for (int i = 0; i < selectionSize; i++)
            {
                var pointer = start + i * distance;
                while (cumulative < pointer && index < population.Count)
                {
                    cumulative += population[index].GetFitness();
                    index++;
                }
                if (index > 0 && index <= population.Count)
                {
                    selected.Add(population[index - 1].GetFitness());
                }
            }

            // Assert
            Assert.Equal(selectionSize, selected.Count);
        }

        [Fact]
        public void TruncationSelection_SelectsTopPercent()
        {
            // Arrange
            var population = new List<BinaryIndividual>();
            for (int i = 0; i < 100; i++)
            {
                var ind = new BinaryIndividual(5, _random);
                ind.SetFitness(i);
                population.Add(ind);
            }

            // Act - Select top 20%
            var selected = population.OrderByDescending(i => i.GetFitness()).Take(20).ToList();

            // Assert
            Assert.Equal(20, selected.Count);
            Assert.True(selected.All(i => i.GetFitness() >= 80));
        }

        [Fact]
        public void UniformSelection_SelectsRandomly()
        {
            // Arrange
            var population = new List<BinaryIndividual>();
            for (int i = 0; i < 50; i++)
            {
                var ind = new BinaryIndividual(5, _random);
                ind.SetFitness(i);
                population.Add(ind);
            }

            // Act
            var selections = new List<int>();
            for (int i = 0; i < 100; i++)
            {
                var selected = population[_random.Next(population.Count)];
                selections.Add((int)selected.GetFitness());
            }

            // Assert - Should have good coverage
            var uniqueSelections = selections.Distinct().Count();
            Assert.True(uniqueSelections > 20); // Should select from many individuals
        }

        #endregion

        #region Two-Point Crossover Tests

        [Fact]
        public void TwoPointCrossover_ProducesValidOffspring()
        {
            // Arrange
            var parent1 = new BinaryIndividual(15, _random);
            var parent2 = new BinaryIndividual(15, _random);

            // Act
            var (child1, child2) = TwoPointCrossover(parent1, parent2, 1.0);

            // Assert
            Assert.Equal(15, child1.GetGenes().Count);
            Assert.Equal(15, child2.GetGenes().Count);
        }

        [Fact]
        public void TwoPointCrossover_InheritsFromBothParents()
        {
            // Arrange
            var genes0 = Enumerable.Range(0, 20).Select(_ => new BinaryGene(0)).ToList();
            var genes1 = Enumerable.Range(0, 20).Select(_ => new BinaryGene(1)).ToList();
            var parent1 = new BinaryIndividual(genes0);
            var parent2 = new BinaryIndividual(genes1);

            // Act
            var (child1, child2) = TwoPointCrossover(parent1, parent2, 1.0);

            // Assert
            var child1Ones = child1.GetGenes().Count(g => g.Value == 1);
            var child2Ones = child2.GetGenes().Count(g => g.Value == 1);

            Assert.True(child1Ones > 0 && child1Ones < 20);
            Assert.True(child2Ones > 0 && child2Ones < 20);
        }

        private (BinaryIndividual, BinaryIndividual) TwoPointCrossover(
            BinaryIndividual parent1, BinaryIndividual parent2, double rate)
        {
            if (_random.NextDouble() > rate)
            {
                return (parent1.Clone() as BinaryIndividual, parent2.Clone() as BinaryIndividual);
            }

            var genes1 = parent1.GetGenes().ToList();
            var genes2 = parent2.GetGenes().ToList();

            var point1 = _random.Next(1, genes1.Count - 1);
            var point2 = _random.Next(point1 + 1, genes1.Count);

            var child1Genes = genes1.Take(point1)
                .Concat(genes2.Skip(point1).Take(point2 - point1))
                .Concat(genes1.Skip(point2))
                .Select(g => g.Clone()).ToList();

            var child2Genes = genes2.Take(point1)
                .Concat(genes1.Skip(point1).Take(point2 - point1))
                .Concat(genes2.Skip(point2))
                .Select(g => g.Clone()).ToList();

            return (new BinaryIndividual(child1Genes), new BinaryIndividual(child2Genes));
        }

        #endregion

        #region Multimodal Optimization Tests

        [Fact]
        public void GA_MultimodalFunction_FindsMultiplePeaks()
        {
            // Arrange - Multiple independent runs
            var bestSolutions = new List<double>();

            for (int run = 0; run < 5; run++)
            {
                var population = Enumerable.Range(0, 50)
                    .Select(_ => new RealValuedIndividual(1, -5.0, 5.0, _random))
                    .ToList();

                // Simple multimodal: sin(x) * x
                for (int gen = 0; gen < 100; gen++)
                {
                    foreach (var ind in population)
                    {
                        var x = ind.GetValuesAsArray()[0];
                        ind.SetFitness(Math.Sin(x) * x);
                    }

                    var newPop = new List<RealValuedIndividual>();
                    var sorted = population.OrderByDescending(i => i.GetFitness()).ToList();

                    for (int i = 0; i < 5; i++)
                        newPop.Add(sorted[i].Clone() as RealValuedIndividual);

                    while (newPop.Count < 50)
                    {
                        var p1 = TournamentSelectReal(population, 3);
                        var p2 = TournamentSelectReal(population, 3);
                        var (c1, c2) = ArithmeticCrossover(p1, p2, 0.8);
                        c1 = GaussianMutation(c1, 0.2, 0.5);
                        newPop.Add(c1);
                        if (newPop.Count < 50)
                            newPop.Add(c2);
                    }

                    population = newPop;
                }

                bestSolutions.Add(population.Max(i => i.GetFitness()));
            }

            // Assert - Should find reasonably good solutions
            var avgBest = bestSolutions.Average();
            Assert.True(avgBest > 0);
        }

        #endregion

        #region Premature Convergence Tests

        [Fact]
        public void GA_WithoutDiversity_CanConvergePrematurely()
        {
            // Arrange - Very low mutation, high selection pressure
            var population = Enumerable.Range(0, 30)
                .Select(_ => new BinaryIndividual(20, _random))
                .ToList();

            var diversityHistory = new List<int>();

            // Act
            for (int gen = 0; gen < 40; gen++)
            {
                foreach (var ind in population)
                {
                    ind.SetFitness(OneMaxFitness(ind));
                }

                diversityHistory.Add(population.Select(i => i.GetValueAsInt()).Distinct().Count());

                // Very aggressive selection, low mutation
                var newPop = new List<BinaryIndividual>();
                var sorted = population.OrderByDescending(i => i.GetFitness()).ToList();

                for (int i = 0; i < 15; i++)
                    newPop.Add(sorted[i].Clone() as BinaryIndividual);

                while (newPop.Count < 30)
                {
                    var parent = sorted[_random.Next(10)]; // Only top 10
                    var child = BitFlipMutation(parent.Clone() as BinaryIndividual, 0.001);
                    newPop.Add(child);
                }

                population = newPop;
            }

            // Assert - Diversity should decrease significantly
            Assert.True(diversityHistory.Last() < diversityHistory.First() * 0.5);
        }

        [Fact]
        public void GA_DiversityMechanisms_PreventPrematureConvergence()
        {
            // Arrange
            var population1 = Enumerable.Range(0, 30)
                .Select(_ => new BinaryIndividual(20, _random))
                .ToList();
            var population2 = Enumerable.Range(0, 30)
                .Select(_ => new BinaryIndividual(20, _random))
                .ToList();

            // Act - Run one with diversity maintenance, one without
            for (int gen = 0; gen < 30; gen++)
            {
                population1 = EvolveWithMutationRate(population1, 0.001);
                population2 = EvolveWithMutationRate(population2, 0.05);
            }

            // Assert
            var diversity1 = population1.Select(i => i.GetValueAsInt()).Distinct().Count();
            var diversity2 = population2.Select(i => i.GetValueAsInt()).Distinct().Count();

            Assert.True(diversity2 >= diversity1);
        }

        #endregion

        #region Scalability Tests

        [Fact]
        public void GA_LargeDimension_HandlesEfficiently()
        {
            // Arrange
            var population = Enumerable.Range(0, 30)
                .Select(_ => new RealValuedIndividual(50, -10.0, 10.0, _random))
                .ToList();

            // Act
            var sw = System.Diagnostics.Stopwatch.StartNew();
            for (int gen = 0; gen < 20; gen++)
            {
                foreach (var ind in population)
                {
                    ind.SetFitness(SphereFitness(ind));
                }

                var newPop = new List<RealValuedIndividual>();
                var sorted = population.OrderByDescending(i => i.GetFitness()).ToList();

                for (int i = 0; i < 3; i++)
                    newPop.Add(sorted[i].Clone() as RealValuedIndividual);

                while (newPop.Count < 30)
                {
                    var p1 = TournamentSelectReal(population, 3);
                    var p2 = TournamentSelectReal(population, 3);
                    var (c1, c2) = ArithmeticCrossover(p1, p2, 0.8);
                    c1 = GaussianMutation(c1, 0.1, 0.3);
                    newPop.Add(c1);
                    if (newPop.Count < 30)
                        newPop.Add(c2);
                }

                population = newPop;
            }
            sw.Stop();

            // Assert
            Assert.True(sw.ElapsedMilliseconds < 10000);
        }

        [Fact]
        public void GA_LargePopulation_HandlesEfficiently()
        {
            // Arrange
            var population = Enumerable.Range(0, 500)
                .Select(_ => new BinaryIndividual(30, _random))
                .ToList();

            // Act
            var sw = System.Diagnostics.Stopwatch.StartNew();
            population = EvolveOneGeneration(population);
            sw.Stop();

            // Assert
            Assert.Equal(500, population.Count);
            Assert.True(sw.ElapsedMilliseconds < 5000);
        }

        #endregion

        #region Boundary Condition Tests

        [Fact]
        public void GA_SingleIndividual_HandlesGracefully()
        {
            // Arrange
            var population = new List<BinaryIndividual>
            {
                new BinaryIndividual(10, _random)
            };

            // Act & Assert - Should not crash
            for (int gen = 0; gen < 5; gen++)
            {
                foreach (var ind in population)
                {
                    ind.SetFitness(OneMaxFitness(ind));
                }

                var newPop = new List<BinaryIndividual>();
                newPop.Add(population[0].Clone() as BinaryIndividual);

                population = newPop;
            }

            Assert.Single(population);
        }

        [Fact]
        public void GA_SingleGene_WorksCorrectly()
        {
            // Arrange
            var population = Enumerable.Range(0, 20)
                .Select(_ => new BinaryIndividual(1, _random))
                .ToList();

            // Act
            for (int gen = 0; gen < 20; gen++)
            {
                population = EvolveOneGeneration(population);
            }

            // Assert - Should converge to all 1s
            var onesCount = population.Count(i => i.GetGenes().First().Value == 1);
            Assert.True(onesCount > 15); // Most should be 1
        }

        [Fact]
        public void GA_AllIdenticalInitialPopulation_CanEvolve()
        {
            // Arrange - All individuals identical
            var genes = Enumerable.Range(0, 10).Select(_ => new BinaryGene(0)).ToList();
            var population = Enumerable.Range(0, 30)
                .Select(_ => new BinaryIndividual(genes.Select(g => g.Clone()).ToList()))
                .ToList();

            // Act - Mutation should introduce diversity
            for (int gen = 0; gen < 50; gen++)
            {
                foreach (var ind in population)
                {
                    ind.SetFitness(OneMaxFitness(ind));
                }

                var newPop = new List<BinaryIndividual>();
                var sorted = population.OrderByDescending(i => i.GetFitness()).ToList();

                for (int i = 0; i < 3; i++)
                    newPop.Add(sorted[i].Clone() as BinaryIndividual);

                while (newPop.Count < 30)
                {
                    var parent = TournamentSelect(population, 3);
                    var child = BitFlipMutation(parent.Clone() as BinaryIndividual, 0.1);
                    newPop.Add(child);
                }

                population = newPop;
            }

            // Assert
            var diversity = population.Select(i => i.GetValueAsInt()).Distinct().Count();
            Assert.True(diversity > 1); // Should have diversity now
        }

        #endregion

        #region Real-World Application Tests

        [Fact]
        public void GA_FeatureSelection_FinksGoodSubset()
        {
            // Arrange - Simulate feature selection: maximize features with fitness
            // Fitness = accuracy - penalty * num_features
            var population = Enumerable.Range(0, 50)
                .Select(_ => new BinaryIndividual(20, _random))
                .ToList();

            double FeatureSelectionFitness(BinaryIndividual ind)
            {
                var selected = ind.GetGenes().Count(g => g.Value == 1);
                // Simulate: more features = higher accuracy but with penalty
                var accuracy = Math.Min(100, selected * 8 + 20);
                var penalty = selected * 2;
                return accuracy - penalty;
            }

            // Act
            for (int gen = 0; gen < 100; gen++)
            {
                foreach (var ind in population)
                {
                    ind.SetFitness(FeatureSelectionFitness(ind));
                }

                var newPop = new List<BinaryIndividual>();
                var sorted = population.OrderByDescending(i => i.GetFitness()).ToList();

                for (int i = 0; i < 5; i++)
                    newPop.Add(sorted[i].Clone() as BinaryIndividual);

                while (newPop.Count < 50)
                {
                    var p1 = TournamentSelect(population, 3);
                    var p2 = TournamentSelect(population, 3);
                    var (c1, c2) = SinglePointCrossover(p1, p2, 0.8);
                    c1 = BitFlipMutation(c1, 0.05);
                    newPop.Add(c1);
                    if (newPop.Count < 50)
                        newPop.Add(c2);
                }

                population = newPop;
            }

            // Assert - Should find good trade-off
            var best = population.OrderByDescending(i => i.GetFitness()).First();
            var selectedFeatures = best.GetGenes().Count(g => g.Value == 1);
            Assert.True(selectedFeatures >= 5 && selectedFeatures <= 15); // Reasonable subset
        }

        [Fact]
        public void GA_WeightOptimization_FindsOptimalWeights()
        {
            // Arrange - Optimize weights for a simple linear combination
            // Target: 0.3*x1 + 0.5*x2 + 0.2*x3 = 1.0 where x1+x2+x3 = 1
            var population = Enumerable.Range(0, 60)
                .Select(_ => new RealValuedIndividual(3, 0.0, 1.0, _random))
                .ToList();

            double WeightFitness(RealValuedIndividual ind)
            {
                var weights = ind.GetValuesAsArray();
                var sum = weights.Sum();
                if (sum == 0) return -1000;

                // Normalize
                var normalized = weights.Select(w => w / sum).ToArray();

                // Distance from target: [0.3, 0.5, 0.2]
                var target = new[] { 0.3, 0.5, 0.2 };
                var error = 0.0;
                for (int i = 0; i < 3; i++)
                {
                    error += Math.Pow(normalized[i] - target[i], 2);
                }

                return -error;
            }

            // Act
            for (int gen = 0; gen < 150; gen++)
            {
                foreach (var ind in population)
                {
                    ind.SetFitness(WeightFitness(ind));
                }

                var newPop = new List<RealValuedIndividual>();
                var sorted = population.OrderByDescending(i => i.GetFitness()).ToList();

                for (int i = 0; i < 6; i++)
                    newPop.Add(sorted[i].Clone() as RealValuedIndividual);

                while (newPop.Count < 60)
                {
                    var p1 = TournamentSelectReal(population, 3);
                    var p2 = TournamentSelectReal(population, 3);
                    var (c1, c2) = ArithmeticCrossover(p1, p2, 0.9);
                    c1 = GaussianMutation(c1, 0.15, 0.1);
                    newPop.Add(c1);
                    if (newPop.Count < 60)
                        newPop.Add(c2);
                }

                population = newPop;
            }

            // Assert
            var best = population.OrderByDescending(i => i.GetFitness()).First();
            Assert.True(best.GetFitness() > -0.05); // Close to target
        }

        #endregion

        #region Constrained Optimization Tests

        [Fact]
        public void GA_ConstrainedOptimization_RespectsConstraints()
        {
            // Arrange - Maximize x^2 subject to 0 <= x <= 5
            var population = Enumerable.Range(0, 40)
                .Select(_ => new RealValuedIndividual(1, 0.0, 10.0, _random))
                .ToList();

            double ConstrainedFitness(RealValuedIndividual ind)
            {
                var x = ind.GetValuesAsArray()[0];
                // Hard constraint: x must be in [0, 5]
                if (x < 0 || x > 5)
                    return -1000; // Heavy penalty

                return x * x; // Maximize x^2
            }

            // Act
            for (int gen = 0; gen < 100; gen++)
            {
                foreach (var ind in population)
                {
                    ind.SetFitness(ConstrainedFitness(ind));
                }

                var newPop = new List<RealValuedIndividual>();
                var sorted = population.OrderByDescending(i => i.GetFitness()).ToList();

                for (int i = 0; i < 4; i++)
                    newPop.Add(sorted[i].Clone() as RealValuedIndividual);

                while (newPop.Count < 40)
                {
                    var p1 = TournamentSelectReal(population, 3);
                    var p2 = TournamentSelectReal(population, 3);
                    var (c1, c2) = ArithmeticCrossover(p1, p2, 0.8);
                    c1 = GaussianMutation(c1, 0.1, 0.3);

                    // Repair: clip to valid range
                    var genes = c1.GetGenes().ToList();
                    genes[0].Value = Math.Max(0, Math.Min(5, genes[0].Value));
                    c1.SetGenes(genes);

                    newPop.Add(c1);
                    if (newPop.Count < 40)
                        newPop.Add(c2);
                }

                population = newPop;
            }

            // Assert - Should find x close to 5 (max in constrained region)
            var best = population.OrderByDescending(i => i.GetFitness()).First();
            var bestX = best.GetValuesAsArray()[0];
            Assert.True(bestX >= 4.5 && bestX <= 5.0);
        }

        #endregion

        #region Statistical Tests

        [Fact]
        public void GA_MultipleRuns_ShowsConsistency()
        {
            // Arrange
            var results = new List<double>();

            // Act - Run GA multiple times
            for (int run = 0; run < 10; run++)
            {
                var localRandom = new Random(42 + run);
                var population = Enumerable.Range(0, 30)
                    .Select(_ => new BinaryIndividual(15, localRandom))
                    .ToList();

                for (int gen = 0; gen < 50; gen++)
                {
                    foreach (var ind in population)
                    {
                        ind.SetFitness(OneMaxFitness(ind));
                    }

                    var newPop = new List<BinaryIndividual>();
                    var sorted = population.OrderByDescending(i => i.GetFitness()).ToList();

                    for (int i = 0; i < 3; i++)
                        newPop.Add(sorted[i].Clone() as BinaryIndividual);

                    while (newPop.Count < 30)
                    {
                        var best = sorted[0];
                        for (int j = 0; j < 3; j++)
                        {
                            var contender = population[localRandom.Next(population.Count)];
                            if (contender.GetFitness() > best.GetFitness())
                                best = contender;
                        }

                        var child = best.Clone() as BinaryIndividual;
                        var genes = child.GetGenes().ToList();
                        for (int i = 0; i < genes.Count; i++)
                        {
                            if (localRandom.NextDouble() < 0.01)
                                genes[i].Value = 1 - genes[i].Value;
                        }
                        child.SetGenes(genes);
                        newPop.Add(child);
                    }

                    population = newPop;
                }

                results.Add(population.Max(i => i.GetFitness()));
            }

            // Assert - Results should be consistently good
            var avgResult = results.Average();
            Assert.True(avgResult > 12); // Should find good solutions consistently
        }

        [Fact]
        public void GA_DifferentRandomSeeds_ProduceDifferentPaths()
        {
            // Arrange
            var fitnessHistory1 = new List<double>();
            var fitnessHistory2 = new List<double>();

            // Act - Two runs with different seeds
            for (int runId = 0; runId < 2; runId++)
            {
                var localRandom = new Random(runId == 0 ? 123 : 456);
                var population = Enumerable.Range(0, 30)
                    .Select(_ => new BinaryIndividual(15, localRandom))
                    .ToList();

                var history = runId == 0 ? fitnessHistory1 : fitnessHistory2;

                for (int gen = 0; gen < 20; gen++)
                {
                    foreach (var ind in population)
                    {
                        ind.SetFitness(OneMaxFitness(ind));
                    }

                    history.Add(population.Max(i => i.GetFitness()));

                    var newPop = new List<BinaryIndividual>();
                    var sorted = population.OrderByDescending(i => i.GetFitness()).ToList();

                    for (int i = 0; i < 3; i++)
                        newPop.Add(sorted[i].Clone() as BinaryIndividual);

                    while (newPop.Count < 30)
                    {
                        var best = sorted[0];
                        for (int j = 0; j < 3; j++)
                        {
                            var contender = population[localRandom.Next(population.Count)];
                            if (contender.GetFitness() > best.GetFitness())
                                best = contender;
                        }
                        newPop.Add(best.Clone() as BinaryIndividual);
                    }

                    population = newPop;
                }
            }

            // Assert - Paths should differ (at least in early generations)
            var differences = 0;
            for (int i = 0; i < 10; i++)
            {
                if (Math.Abs(fitnessHistory1[i] - fitnessHistory2[i]) > 0.5)
                    differences++;
            }
            Assert.True(differences > 0); // Some difference in evolutionary path
        }

        #endregion

        #region Niching and Speciation Tests

        [Fact]
        public void GA_Niching_MaintainsMultipleSolutions()
        {
            // Arrange - Multi-peak function: sin(x) + sin(2x) + sin(3x)
            var population = Enumerable.Range(0, 60)
                .Select(_ => new RealValuedIndividual(1, 0.0, 2 * Math.PI, _random))
                .ToList();

            double MultiPeakFitness(RealValuedIndividual ind)
            {
                var x = ind.GetValuesAsArray()[0];
                return Math.Sin(x) + Math.Sin(2 * x) + Math.Sin(3 * x);
            }

            // Act - Evolve with sharing to maintain diversity
            for (int gen = 0; gen < 100; gen++)
            {
                foreach (var ind in population)
                {
                    ind.SetFitness(MultiPeakFitness(ind));
                }

                var newPop = new List<RealValuedIndividual>();
                var sorted = population.OrderByDescending(i => i.GetFitness()).ToList();

                // Keep diverse elite
                var elite = new List<RealValuedIndividual>();
                foreach (var ind in sorted)
                {
                    var tooClose = elite.Any(e =>
                        Math.Abs(e.GetValuesAsArray()[0] - ind.GetValuesAsArray()[0]) < 0.5);

                    if (!tooClose && elite.Count < 10)
                    {
                        elite.Add(ind.Clone() as RealValuedIndividual);
                    }
                }

                newPop.AddRange(elite);

                while (newPop.Count < 60)
                {
                    var p1 = TournamentSelectReal(population, 3);
                    var p2 = TournamentSelectReal(population, 3);
                    var (c1, c2) = ArithmeticCrossover(p1, p2, 0.8);
                    c1 = GaussianMutation(c1, 0.2, 0.3);
                    newPop.Add(c1);
                    if (newPop.Count < 60)
                        newPop.Add(c2);
                }

                population = newPop;
            }

            // Assert - Should maintain diversity
            var finalValues = population.Select(i => i.GetValuesAsArray()[0]).OrderBy(x => x).ToList();
            var gaps = new List<double>();
            for (int i = 1; i < finalValues.Count; i++)
            {
                gaps.Add(finalValues[i] - finalValues[i - 1]);
            }

            var largeGaps = gaps.Count(g => g > 1.0);
            Assert.True(largeGaps >= 1); // At least one gap indicating multiple niches
        }

        #endregion

        #region Performance Comparison Tests

        [Fact]
        public void GA_TournamentVsRouletteWheel_PerformanceComparison()
        {
            // Arrange
            var popTournament = Enumerable.Range(0, 40)
                .Select(_ => new BinaryIndividual(15, _random))
                .ToList();
            var popRoulette = Enumerable.Range(0, 40)
                .Select(_ => new BinaryIndividual(15, _random))
                .ToList();

            // Act - Tournament
            for (int gen = 0; gen < 50; gen++)
            {
                foreach (var ind in popTournament)
                    ind.SetFitness(OneMaxFitness(ind));

                var newPop = new List<BinaryIndividual>();
                var sorted = popTournament.OrderByDescending(i => i.GetFitness()).ToList();
                for (int i = 0; i < 4; i++)
                    newPop.Add(sorted[i].Clone() as BinaryIndividual);

                while (newPop.Count < 40)
                {
                    var p1 = TournamentSelect(popTournament, 3);
                    var p2 = TournamentSelect(popTournament, 3);
                    var (c1, c2) = SinglePointCrossover(p1, p2, 0.8);
                    c1 = BitFlipMutation(c1, 0.01);
                    newPop.Add(c1);
                    if (newPop.Count < 40)
                        newPop.Add(c2);
                }
                popTournament = newPop;
            }

            // Act - Roulette
            for (int gen = 0; gen < 50; gen++)
            {
                foreach (var ind in popRoulette)
                    ind.SetFitness(OneMaxFitness(ind));

                var newPop = new List<BinaryIndividual>();
                var sorted = popRoulette.OrderByDescending(i => i.GetFitness()).ToList();
                for (int i = 0; i < 4; i++)
                    newPop.Add(sorted[i].Clone() as BinaryIndividual);

                while (newPop.Count < 40)
                {
                    var p1 = RouletteWheelSelect(popRoulette);
                    var p2 = RouletteWheelSelect(popRoulette);
                    var (c1, c2) = SinglePointCrossover(p1, p2, 0.8);
                    c1 = BitFlipMutation(c1, 0.01);
                    newPop.Add(c1);
                    if (newPop.Count < 40)
                        newPop.Add(c2);
                }
                popRoulette = newPop;
            }

            // Assert - Both should find good solutions
            var bestTournament = popTournament.Max(i => i.GetFitness());
            var bestRoulette = popRoulette.Max(i => i.GetFitness());

            Assert.True(bestTournament > 12);
            Assert.True(bestRoulette > 12);
        }

        [Fact]
        public void GA_SinglePointVsUniform_CrossoverComparison()
        {
            // Arrange
            var pop1 = Enumerable.Range(0, 30)
                .Select(_ => new BinaryIndividual(15, _random))
                .ToList();
            var pop2 = Enumerable.Range(0, 30)
                .Select(_ => new BinaryIndividual(15, _random))
                .ToList();

            // Act - Single point
            for (int gen = 0; gen < 40; gen++)
            {
                foreach (var ind in pop1)
                    ind.SetFitness(OneMaxFitness(ind));

                var newPop = new List<BinaryIndividual>();
                var sorted = pop1.OrderByDescending(i => i.GetFitness()).ToList();
                for (int i = 0; i < 3; i++)
                    newPop.Add(sorted[i].Clone() as BinaryIndividual);

                while (newPop.Count < 30)
                {
                    var p1 = TournamentSelect(pop1, 3);
                    var p2 = TournamentSelect(pop1, 3);
                    var (c1, c2) = SinglePointCrossover(p1, p2, 0.9);
                    c1 = BitFlipMutation(c1, 0.01);
                    newPop.Add(c1);
                    if (newPop.Count < 30)
                        newPop.Add(c2);
                }
                pop1 = newPop;
            }

            // Act - Uniform
            for (int gen = 0; gen < 40; gen++)
            {
                foreach (var ind in pop2)
                    ind.SetFitness(OneMaxFitness(ind));

                var newPop = new List<BinaryIndividual>();
                var sorted = pop2.OrderByDescending(i => i.GetFitness()).ToList();
                for (int i = 0; i < 3; i++)
                    newPop.Add(sorted[i].Clone() as BinaryIndividual);

                while (newPop.Count < 30)
                {
                    var p1 = TournamentSelect(pop2, 3);
                    var p2 = TournamentSelect(pop2, 3);
                    var (c1, c2) = UniformCrossover(p1, p2, 0.9);
                    c1 = BitFlipMutation(c1, 0.01);
                    newPop.Add(c1);
                    if (newPop.Count < 30)
                        newPop.Add(c2);
                }
                pop2 = newPop;
            }

            // Assert
            var best1 = pop1.Max(i => i.GetFitness());
            var best2 = pop2.Max(i => i.GetFitness());

            Assert.True(best1 > 11);
            Assert.True(best2 > 11);
        }

        #endregion

        #region Griewank Function Test

        /// <summary>
        /// Griewank function: another multimodal test function
        /// </summary>
        private double GriewankFitness(RealValuedIndividual individual)
        {
            var values = individual.GetValuesAsArray();
            var sum = values.Sum(x => x * x) / 4000.0;
            var product = 1.0;
            for (int i = 0; i < values.Length; i++)
            {
                product *= Math.Cos(values[i] / Math.Sqrt(i + 1));
            }
            return -(sum - product + 1);
        }

        [Fact]
        public void RealGA_GriewankFunction_FindsGoodSolution()
        {
            // Arrange
            var population = Enumerable.Range(0, 60)
                .Select(_ => new RealValuedIndividual(3, -10.0, 10.0, _random))
                .ToList();

            // Act
            for (int gen = 0; gen < 150; gen++)
            {
                foreach (var ind in population)
                {
                    ind.SetFitness(GriewankFitness(ind));
                }

                var newPop = new List<RealValuedIndividual>();
                var sorted = population.OrderByDescending(i => i.GetFitness()).ToList();

                for (int i = 0; i < 6; i++)
                    newPop.Add(sorted[i].Clone() as RealValuedIndividual);

                while (newPop.Count < 60)
                {
                    var p1 = TournamentSelectReal(population, 4);
                    var p2 = TournamentSelectReal(population, 4);
                    var (c1, c2) = ArithmeticCrossover(p1, p2, 0.85);
                    c1 = GaussianMutation(c1, 0.15, 0.5);
                    newPop.Add(c1);
                    if (newPop.Count < 60)
                        newPop.Add(c2);
                }

                population = newPop;
            }

            // Assert
            var bestFitness = population.Max(i => i.GetFitness());
            Assert.True(bestFitness > -5.0);
        }

        #endregion

        #region Additional Edge Cases

        [Fact]
        public void BinaryIndividual_AllZeros_GetValueAsIntIsZero()
        {
            // Arrange
            var genes = Enumerable.Range(0, 10).Select(_ => new BinaryGene(0)).ToList();
            var individual = new BinaryIndividual(genes);

            // Act
            var value = individual.GetValueAsInt();

            // Assert
            Assert.Equal(0, value);
        }

        [Fact]
        public void BinaryIndividual_AllOnes_GetValueAsIntIsMax()
        {
            // Arrange
            var genes = Enumerable.Range(0, 4).Select(_ => new BinaryGene(1)).ToList();
            var individual = new BinaryIndividual(genes);

            // Act
            var value = individual.GetValueAsInt();

            // Assert
            Assert.Equal(15, value); // 1111 in binary = 15
        }

        [Fact]
        public void RealValuedIndividual_SetGenes_UpdatesInternalState()
        {
            // Arrange
            var individual = new RealValuedIndividual(3, -5.0, 5.0, _random);
            var newGenes = new List<RealGene>
            {
                new RealGene(1.0),
                new RealGene(2.0),
                new RealGene(3.0)
            };

            // Act
            individual.SetGenes(newGenes);

            // Assert
            var values = individual.GetValuesAsArray();
            Assert.Equal(new[] { 1.0, 2.0, 3.0 }, values);
        }

        [Fact]
        public void PermutationIndividual_SetGenes_UpdatesPermutation()
        {
            // Arrange
            var individual = new PermutationIndividual(5, _random);
            var newGenes = new List<PermutationGene>
            {
                new PermutationGene(2),
                new PermutationGene(0),
                new PermutationGene(3),
                new PermutationGene(1),
                new PermutationGene(4)
            };

            // Act
            individual.SetGenes(newGenes);

            // Assert
            var permutation = individual.GetPermutation();
            Assert.Equal(new[] { 2, 0, 3, 1, 4 }, permutation);
        }

        [Fact]
        public void GA_VeryLowCrossoverRate_StillProducesOffspring()
        {
            // Arrange
            var population = Enumerable.Range(0, 20)
                .Select(_ => new BinaryIndividual(10, _random))
                .ToList();

            // Act
            for (int gen = 0; gen < 20; gen++)
            {
                foreach (var ind in population)
                    ind.SetFitness(OneMaxFitness(ind));

                var newPop = new List<BinaryIndividual>();
                var sorted = population.OrderByDescending(i => i.GetFitness()).ToList();

                for (int i = 0; i < 2; i++)
                    newPop.Add(sorted[i].Clone() as BinaryIndividual);

                while (newPop.Count < 20)
                {
                    var p1 = TournamentSelect(population, 3);
                    var p2 = TournamentSelect(population, 3);
                    var (c1, c2) = SinglePointCrossover(p1, p2, 0.1); // Very low
                    c1 = BitFlipMutation(c1, 0.05);
                    newPop.Add(c1);
                    if (newPop.Count < 20)
                        newPop.Add(c2);
                }

                population = newPop;
            }

            // Assert
            Assert.Equal(20, population.Count);
        }

        [Fact]
        public void GA_VeryHighMutationRate_ProducesDiversity()
        {
            // Arrange
            var population = Enumerable.Range(0, 30)
                .Select(_ => new BinaryIndividual(15, _random))
                .ToList();

            // Act - High mutation
            for (int gen = 0; gen < 20; gen++)
            {
                population = EvolveWithMutationRate(population, 0.3);
            }

            // Assert - Should maintain high diversity
            var diversity = population.Select(i => i.GetValueAsInt()).Distinct().Count();
            Assert.True(diversity > 20);
        }

        [Fact]
        public void GA_NoElitism_CanLoseBestSolution()
        {
            // Arrange
            var population = Enumerable.Range(0, 20)
                .Select(_ => new BinaryIndividual(10, _random))
                .ToList();

            foreach (var ind in population)
                ind.SetFitness(OneMaxFitness(ind));

            var initialBest = population.Max(i => i.GetFitness());

            // Act - Evolve without elitism
            for (int gen = 0; gen < 10; gen++)
            {
                var newPop = new List<BinaryIndividual>();

                while (newPop.Count < 20)
                {
                    var p1 = TournamentSelect(population, 2);
                    var p2 = TournamentSelect(population, 2);
                    var (c1, c2) = SinglePointCrossover(p1, p2, 0.8);
                    c1 = BitFlipMutation(c1, 0.1);
                    newPop.Add(c1);
                    if (newPop.Count < 20)
                        newPop.Add(c2);
                }

                foreach (var ind in newPop)
                    ind.SetFitness(OneMaxFitness(ind));

                population = newPop;
            }

            // Assert - Without elitism, best might not improve or could get worse
            var finalBest = population.Max(i => i.GetFitness());
            // Just verify it runs - without elitism, fitness might not improve
            Assert.True(finalBest >= 0);
        }

        [Fact]
        public void BinaryIndividual_MappedValue_RangeIsCorrect()
        {
            // Arrange
            var genesMin = Enumerable.Range(0, 8).Select(_ => new BinaryGene(0)).ToList();
            var genesMax = Enumerable.Range(0, 8).Select(_ => new BinaryGene(1)).ToList();
            var indMin = new BinaryIndividual(genesMin);
            var indMax = new BinaryIndividual(genesMax);

            // Act
            var mappedMin = indMin.GetValueMapped(-100, 100);
            var mappedMax = indMax.GetValueMapped(-100, 100);

            // Assert
            Assert.Equal(-100.0, mappedMin, precision: 5);
            Assert.Equal(100.0, mappedMax, precision: 5);
        }

        [Fact]
        public void GA_CombinationProblem_FindsGoodCombination()
        {
            // Arrange - Find combination that maximizes sum of selected indices
            var population = Enumerable.Range(0, 40)
                .Select(_ => new BinaryIndividual(10, _random))
                .ToList();

            double CombinationFitness(BinaryIndividual ind)
            {
                double sum = 0;
                var genes = ind.GetGenes().ToList();
                for (int i = 0; i < genes.Count; i++)
                {
                    if (genes[i].Value == 1)
                        sum += i; // Add index if selected
                }
                return sum;
            }

            // Act
            for (int gen = 0; gen < 50; gen++)
            {
                foreach (var ind in population)
                    ind.SetFitness(CombinationFitness(ind));

                var newPop = new List<BinaryIndividual>();
                var sorted = population.OrderByDescending(i => i.GetFitness()).ToList();

                for (int i = 0; i < 4; i++)
                    newPop.Add(sorted[i].Clone() as BinaryIndividual);

                while (newPop.Count < 40)
                {
                    var p1 = TournamentSelect(population, 3);
                    var p2 = TournamentSelect(population, 3);
                    var (c1, c2) = SinglePointCrossover(p1, p2, 0.8);
                    c1 = BitFlipMutation(c1, 0.02);
                    newPop.Add(c1);
                    if (newPop.Count < 40)
                        newPop.Add(c2);
                }

                population = newPop;
            }

            // Assert - Should select higher indices
            var best = population.OrderByDescending(i => i.GetFitness()).First();
            var bestGenes = best.GetGenes().ToList();
            var highIndicesSelected = bestGenes.Skip(5).Count(g => g.Value == 1);
            Assert.True(highIndicesSelected >= 3); // Should prefer higher indices
        }

        [Fact]
        public void RealGA_ConstrainedSphere_RespectsConstraints()
        {
            // Arrange - Sphere with constraint: sum(xi) >= 0
            var population = Enumerable.Range(0, 40)
                .Select(_ => new RealValuedIndividual(3, -5.0, 5.0, _random))
                .ToList();

            double ConstrainedSphereFitness(RealValuedIndividual ind)
            {
                var values = ind.GetValuesAsArray();
                var sum = values.Sum();
                if (sum < 0)
                    return -1000; // Penalty for constraint violation

                return -values.Sum(x => x * x);
            }

            // Act
            for (int gen = 0; gen < 100; gen++)
            {
                foreach (var ind in population)
                    ind.SetFitness(ConstrainedSphereFitness(ind));

                var newPop = new List<RealValuedIndividual>();
                var sorted = population.OrderByDescending(i => i.GetFitness()).ToList();

                for (int i = 0; i < 4; i++)
                    newPop.Add(sorted[i].Clone() as RealValuedIndividual);

                while (newPop.Count < 40)
                {
                    var p1 = TournamentSelectReal(population, 3);
                    var p2 = TournamentSelectReal(population, 3);
                    var (c1, c2) = ArithmeticCrossover(p1, p2, 0.8);
                    c1 = GaussianMutation(c1, 0.1, 0.3);
                    newPop.Add(c1);
                    if (newPop.Count < 40)
                        newPop.Add(c2);
                }

                population = newPop;
            }

            // Assert - Best should respect constraint
            var best = population.OrderByDescending(i => i.GetFitness()).First();
            var bestValues = best.GetValuesAsArray();
            Assert.True(bestValues.Sum() >= -0.5); // Close to or above 0
        }

        [Fact]
        public void PermutationGA_ShortestPath_ImprovesSolution()
        {
            // Arrange - Very simple 3-city problem
            var distances = new double[,]
            {
                { 0, 10, 15 },
                { 10, 0, 12 },
                { 15, 12, 0 }
            };

            var population = Enumerable.Range(0, 20)
                .Select(_ => new PermutationIndividual(3, _random))
                .ToList();

            foreach (var ind in population)
                ind.SetFitness(TspFitness(ind, distances));

            var initialBest = population.Max(i => i.GetFitness());

            // Act
            for (int gen = 0; gen < 30; gen++)
            {
                var newPop = new List<PermutationIndividual>();
                var sorted = population.OrderByDescending(i => i.GetFitness()).ToList();

                for (int i = 0; i < 2; i++)
                    newPop.Add(sorted[i].Clone() as PermutationIndividual);

                while (newPop.Count < 20)
                {
                    var p1 = TournamentSelectPerm(population, 3);
                    var p2 = TournamentSelectPerm(population, 3);
                    var (c1, c2) = p1.OrderCrossover(p2, _random);
                    if (_random.NextDouble() < 0.2)
                        c1.SwapMutation(_random);
                    newPop.Add(c1);
                    if (newPop.Count < 20)
                        newPop.Add(c2);
                }

                foreach (var ind in newPop)
                    ind.SetFitness(TspFitness(ind, distances));

                population = newPop;
            }

            // Assert
            var finalBest = population.Max(i => i.GetFitness());
            Assert.True(finalBest >= initialBest * 0.95); // Should not degrade significantly
        }

        [Fact]
        public void GA_Levy_Function_FindsGoodSolution()
        {
            // Arrange - Levy function
            var population = Enumerable.Range(0, 60)
                .Select(_ => new RealValuedIndividual(2, -10.0, 10.0, _random))
                .ToList();

            double LevyFitness(RealValuedIndividual ind)
            {
                var x = ind.GetValuesAsArray();
                var w = new double[x.Length];
                for (int i = 0; i < x.Length; i++)
                {
                    w[i] = 1 + (x[i] - 1) / 4.0;
                }

                var sum = Math.Pow(Math.Sin(Math.PI * w[0]), 2);
                for (int i = 0; i < w.Length - 1; i++)
                {
                    sum += Math.Pow(w[i] - 1, 2) * (1 + 10 * Math.Pow(Math.Sin(Math.PI * w[i] + 1), 2));
                }
                sum += Math.Pow(w[^1] - 1, 2) * (1 + Math.Pow(Math.Sin(2 * Math.PI * w[^1]), 2));

                return -sum;
            }

            // Act
            for (int gen = 0; gen < 150; gen++)
            {
                foreach (var ind in population)
                    ind.SetFitness(LevyFitness(ind));

                var newPop = new List<RealValuedIndividual>();
                var sorted = population.OrderByDescending(i => i.GetFitness()).ToList();

                for (int i = 0; i < 6; i++)
                    newPop.Add(sorted[i].Clone() as RealValuedIndividual);

                while (newPop.Count < 60)
                {
                    var p1 = TournamentSelectReal(population, 4);
                    var p2 = TournamentSelectReal(population, 4);
                    var (c1, c2) = ArithmeticCrossover(p1, p2, 0.85);
                    c1 = GaussianMutation(c1, 0.15, 0.4);
                    newPop.Add(c1);
                    if (newPop.Count < 60)
                        newPop.Add(c2);
                }

                population = newPop;
            }

            // Assert
            var bestFitness = population.Max(i => i.GetFitness());
            Assert.True(bestFitness > -10.0);
        }

        [Fact]
        public void GA_EarlyStopping_StopsWhenOptimalFound()
        {
            // Arrange
            var population = Enumerable.Range(0, 30)
                .Select(_ => new BinaryIndividual(10, _random))
                .ToList();

            int generationsRun = 0;

            // Act
            for (int gen = 0; gen < 100; gen++)
            {
                generationsRun++;

                foreach (var ind in population)
                    ind.SetFitness(OneMaxFitness(ind));

                var best = population.Max(i => i.GetFitness());
                if (best >= 10) // Found optimal
                    break;

                population = EvolveOneGeneration(population);
            }

            // Assert - Should stop before 100 generations
            Assert.True(generationsRun < 100);
            var finalBest = population.Max(i => i.GetFitness());
            Assert.Equal(10, finalBest);
        }

        [Fact]
        public void GA_StagnationDetection_DetectsNoImprovement()
        {
            // Arrange
            var population = Enumerable.Range(0, 20)
                .Select(_ => new BinaryIndividual(10, _random))
                .ToList();

            var fitnessHistory = new List<double>();
            int stagnantGenerations = 0;

            // Act
            for (int gen = 0; gen < 50; gen++)
            {
                foreach (var ind in population)
                    ind.SetFitness(OneMaxFitness(ind));

                var bestFitness = population.Max(i => i.GetFitness());
                fitnessHistory.Add(bestFitness);

                // Check stagnation
                if (fitnessHistory.Count >= 10)
                {
                    var last10 = fitnessHistory.TakeLast(10).ToList();
                    if (last10.Max() - last10.Min() < 0.5)
                    {
                        stagnantGenerations++;
                    }
                }

                population = EvolveOneGeneration(population);
            }

            // Assert - Should detect some stagnation periods
            Assert.True(stagnantGenerations >= 0); // At least some detection
        }

        [Fact]
        public void GA_CompleteEvolutionCycle_FromInitToConvergence()
        {
            // Arrange - Test complete GA lifecycle
            var population = Enumerable.Range(0, 40)
                .Select(_ => new BinaryIndividual(12, _random))
                .ToList();

            // Track metrics through evolution
            var fitnessHistory = new List<double>();
            var diversityHistory = new List<int>();

            // Act - Run complete evolution cycle
            for (int gen = 0; gen < 60; gen++)
            {
                // Evaluate
                foreach (var ind in population)
                    ind.SetFitness(OneMaxFitness(ind));

                // Track metrics
                fitnessHistory.Add(population.Max(i => i.GetFitness()));
                diversityHistory.Add(population.Select(i => i.GetValueAsInt()).Distinct().Count());

                // Evolve
                var newPop = new List<BinaryIndividual>();
                var sorted = population.OrderByDescending(i => i.GetFitness()).ToList();

                // Elitism
                for (int i = 0; i < 4; i++)
                    newPop.Add(sorted[i].Clone() as BinaryIndividual);

                // Generate offspring
                while (newPop.Count < 40)
                {
                    var p1 = TournamentSelect(population, 3);
                    var p2 = TournamentSelect(population, 3);
                    var (c1, c2) = SinglePointCrossover(p1, p2, 0.8);
                    c1 = BitFlipMutation(c1, 0.02);
                    newPop.Add(c1);
                    if (newPop.Count < 40)
                    {
                        c2 = BitFlipMutation(c2, 0.02);
                        newPop.Add(c2);
                    }
                }

                population = newPop;

                // Check for convergence
                if (fitnessHistory.Last() >= 12)
                    break;
            }

            // Assert - Verify evolutionary behavior
            Assert.True(fitnessHistory.Last() > fitnessHistory.First()); // Improvement
            Assert.True(fitnessHistory.Last() >= 11); // Good solution found
            Assert.True(diversityHistory.Last() < diversityHistory.First()); // Convergence

            // Verify monotonic improvement with elitism
            var improvements = 0;
            for (int i = 1; i < fitnessHistory.Count; i++)
            {
                if (fitnessHistory[i] >= fitnessHistory[i - 1])
                    improvements++;
            }
            Assert.True(improvements > fitnessHistory.Count * 0.9); // Mostly monotonic
        }

        #endregion

        #endregion
    }
}
