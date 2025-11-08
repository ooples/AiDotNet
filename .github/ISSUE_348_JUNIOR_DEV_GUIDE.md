# Issue #348: Junior Developer Implementation Guide
## Unit Tests for Genetic Algorithms

---

## Table of Contents
1. [Understanding the Problem](#understanding-the-problem)
2. [Understanding Genetic Algorithm Variants](#understanding-genetic-algorithm-variants)
3. [Algorithm Types Overview](#algorithm-types-overview)
4. [Testing Strategy](#testing-strategy)
5. [Step-by-Step Implementation Guide](#step-by-step-implementation-guide)
6. [Complete Test Examples](#complete-test-examples)

---

## Understanding the Problem

### What Are We Testing?

The `src/Genetics/` directory contains **genetic algorithm implementations** that currently have **0% test coverage**. These algorithms are responsible for evolving solutions through selection, crossover, and mutation.

### Why Is This Important?

Genetic algorithms power the evolution process. If these algorithms have bugs:
- **Evolution doesn't converge** - population never improves
- **Selection is biased** - best solutions are never chosen
- **Crossover fails** - offspring are invalid or identical to parents
- **Mutation breaks genes** - creates invalid solutions
- **Performance degrades** - algorithms run too slowly

### Current Code Organization

**Genetic Algorithm Files**:
```
src/Genetics/
├── GeneticBase.cs                     # Base class with common operations (1426 lines)
├── StandardGeneticAlgorithm.cs        # Classic genetic algorithm (480 lines)
├── SteadyStateGeneticAlgorithm.cs     # Continuous replacement strategy
├── AdaptiveGeneticAlgorithm.cs        # Self-adjusting parameters
├── NonDominatedSortingGeneticAlgorithm.cs  # NSGA-II for multi-objective
└── IslandModelGeneticAlgorithm.cs     # Parallel populations
```

---

## Understanding Genetic Algorithm Variants

### Classic Genetic Algorithm Flow

```
1. Initialize population randomly
2. Evaluate fitness for all individuals
3. REPEAT until stopping criteria:
   a. Select parents (e.g., tournament selection)
   b. Create offspring through crossover
   c. Mutate offspring
   d. Evaluate offspring fitness
   e. Replace old population with new generation
4. Return best individual found
```

### Why Different Variants?

Different problems need different evolutionary strategies:

**Standard GA** - Good for most problems
- Generational replacement (whole population replaced)
- Fixed crossover and mutation rates
- Simple, predictable behavior

**Steady-State GA** - Good for continuous optimization
- Replace worst individuals gradually
- Population always contains best solutions
- Better for online learning

**Adaptive GA** - Good for unknown problems
- Automatically adjusts crossover/mutation rates
- Prevents premature convergence
- Balances exploration vs exploitation

**NSGA-II** - Good for multi-objective problems
- Optimize multiple conflicting objectives
- Produces Pareto front of trade-off solutions
- Used when no single "best" solution exists

**Island Model GA** - Good for large-scale problems
- Multiple populations evolve independently
- Periodic migration between islands
- Parallel execution for speed

---

## Algorithm Types Overview

### GeneticBase<T, TInput, TOutput>

**Purpose**: Abstract base class providing common genetic algorithm operations

**Key Responsibilities**:
1. **Population Management**
   - Initialize population with different strategies
   - Track best individual across generations
   - Manage elitism (keeping best individuals)

2. **Evolution Operations**
   - Selection (tournament, roulette wheel, rank, etc.)
   - Crossover (single-point, uniform, etc.)
   - Mutation (uniform, Gaussian, etc.)

3. **Fitness Evaluation**
   - Evaluate all individuals in population
   - Support parallel evaluation
   - Track fitness statistics

4. **Stopping Criteria**
   - Maximum generations
   - Fitness threshold reached
   - Stagnation (no improvement for N generations)
   - Time limit exceeded

**Critical Methods to Test**:
```csharp
// Population initialization
ICollection<ModelIndividual<T, TInput, TOutput, ModelParameterGene<T>>>
    InitializePopulation(int populationSize, InitializationMethod method);

// Selection methods
ICollection<ModelIndividual<T, TInput, TOutput, ModelParameterGene<T>>>
    Select(int selectionSize, SelectionMethod method);

// Genetic operators
ICollection<ModelIndividual<T, TInput, TOutput, ModelParameterGene<T>>>
    Crossover(ModelIndividual parent1, ModelIndividual parent2, double rate);

ModelIndividual<T, TInput, TOutput, ModelParameterGene<T>>
    Mutate(ModelIndividual individual, double rate);

// Evolution
EvolutionStats<T, TInput, TOutput>
    Evolve(int generations, TInput trainingInput, TOutput trainingOutput);
```

### StandardGeneticAlgorithm<T, TInput, TOutput>

**Purpose**: Classic generational genetic algorithm

**Key Features**:
- Whole population replaced each generation
- Fixed crossover and mutation rates
- Elitism preserves best individuals
- Multiple initialization methods (random, Xavier, heuristic, etc.)

**Critical Methods to Test**:
```csharp
// Initialization methods
void InitializeParameters(Vector<T> parameters, IFullModel<T, TInput, TOutput> model,
    InitializationMethod method);

// Mutation strategies
ModelParameterGene<T> MutateGene(ModelParameterGene<T> gene);
ModelParameterGene<T> MutateGeneGaussian(ModelParameterGene<T> gene);

// Model conversion
IFullModel<T, TInput, TOutput> IndividualToModel(ModelIndividual individual);
```

### SteadyStateGeneticAlgorithm<T, TInput, TOutput>

**Purpose**: Continuous evolution with incremental replacement

**Key Features**:
- Replaces worst individuals instead of entire population
- Better convergence characteristics
- Preserves good solutions throughout evolution

**What to Test**:
- Replacement strategy (worst individuals replaced)
- Population size remains constant
- Best individuals preserved
- Convergence speed compared to standard GA

### AdaptiveGeneticAlgorithm<T, TInput, TOutput>

**Purpose**: Self-adjusting parameters based on evolution progress

**Key Features**:
- Crossover rate adapts (lower when converging)
- Mutation rate adapts (higher when stagnant)
- Population diversity monitoring
- Prevents premature convergence

**What to Test**:
- Parameter adaptation logic
- Diversity calculation
- Adaptation triggers (convergence detection)
- Parameter bounds (rates stay in [0,1])

### NonDominatedSortingGeneticAlgorithm (NSGA-II)

**Purpose**: Multi-objective optimization using Pareto dominance

**Key Concepts**:

**Pareto Dominance**:
- Solution A dominates B if A is better in ALL objectives
- Pareto front = solutions not dominated by any other

**Example**:
```
Objective 1: Maximize accuracy
Objective 2: Minimize inference time

Solution A: accuracy=0.95, time=100ms
Solution B: accuracy=0.90, time=50ms
Solution C: accuracy=0.85, time=60ms

A dominates nothing (worse time than B)
B dominates C (better accuracy, better time)
Pareto front: {A, B}
```

**Key Features**:
- Fast non-dominated sorting
- Crowding distance calculation
- Maintains diversity on Pareto front

**What to Test**:
- Non-dominated sorting correctness
- Crowding distance calculation
- Front assignment
- Diversity preservation

### IslandModelGeneticAlgorithm

**Purpose**: Multiple populations with periodic migration

**Key Features**:
- Multiple islands (sub-populations)
- Independent evolution on each island
- Periodic migration of best individuals
- Parallel execution support

**What to Test**:
- Island initialization
- Independent evolution
- Migration logic (topology, frequency)
- Best individual across all islands

---

## Testing Strategy

### Coverage Goals

**GeneticBase** (Core functionality):
1. **Initialization methods** (Random, Xavier, Heuristic, Diverse, Grid, CaseBased)
2. **Selection methods** (Tournament, RouletteWheel, Rank, Truncation, Uniform, SUS, Elitism)
3. **Crossover operators** (SinglePoint, Uniform)
4. **Mutation operators** (Uniform, Gaussian)
5. **Evolution loop** (multiple generations with fitness tracking)
6. **Stopping criteria** (generations, fitness threshold, stagnation, time limit)

**Algorithm-Specific Tests**:
1. **StandardGA** - Parameter initialization methods
2. **SteadyStateGA** - Replacement strategy
3. **AdaptiveGA** - Parameter adaptation
4. **NSGA-II** - Pareto sorting and crowding distance
5. **IslandModelGA** - Migration and parallelism

### Test File Structure

```
tests/Genetics/
├── GeneticBaseTests.cs                           # Base class tests
├── StandardGeneticAlgorithmTests.cs              # Standard GA tests
├── SteadyStateGeneticAlgorithmTests.cs           # Steady-state tests
├── AdaptiveGeneticAlgorithmTests.cs              # Adaptive tests
├── NonDominatedSortingGeneticAlgorithmTests.cs   # NSGA-II tests
└── IslandModelGeneticAlgorithmTests.cs           # Island model tests
```

### Test Naming Convention

Follow the pattern: `Method_Scenario_ExpectedBehavior`

Examples:
```csharp
[TestMethod]
public void InitializePopulation_WithRandomMethod_CreatesValidIndividuals()

[TestMethod]
public void TournamentSelection_WithTournamentSize3_SelectsBetterIndividuals()

[TestMethod]
public void Evolve_WithMaxGenerations_StopsAtCorrectGeneration()
```

---

## Step-by-Step Implementation Guide

### Step 1: Create StandardGeneticAlgorithmTests.cs

**What to test**:
1. Constructor validates parameters
2. InitializePopulation creates correct number of individuals
3. Different initialization methods work correctly
4. MutateGene produces different values
5. MutateGeneGaussian uses Gaussian distribution
6. IndividualToModel converts genes correctly

**Implementation checklist**:
- [ ] `Constructor_WithValidParameters_InitializesCorrectly()`
- [ ] `Constructor_WithNullModelFactory_ThrowsException()`
- [ ] `InitializePopulation_WithRandomMethod_CreatesCorrectCount()`
- [ ] `InitializePopulation_RandomMethod_CreatesValidGenes()`
- [ ] `InitializePopulation_XavierMethod_UsesCorrectDistribution()`
- [ ] `InitializePopulation_HeuristicMethod_ScalesWithFeatureCount()`
- [ ] `InitializePopulation_DiverseMethod_CreatesDifferentDistributions()`
- [ ] `InitializePopulation_GridMethod_CoversParameterSpace()`
- [ ] `MutateGene_ProducesDifferentValue()`
- [ ] `MutateGene_MaintainsGeneStructure()`
- [ ] `MutateGeneGaussian_UsesGaussianDistribution()`
- [ ] `IndividualToModel_CreatesValidModel()`
- [ ] `IndividualToModel_PreservesGeneValues()`

### Step 2: Test Selection Methods (in GeneticBaseTests.cs)

**Tournament Selection**:
```csharp
[TestMethod]
public void TournamentSelection_WithTournamentSize2_SelectsBetterIndividuals()
{
    // Arrange
    var ga = CreateTestGA();
    var population = new List<ModelIndividual>
    {
        CreateIndividual(fitness: 0.5),
        CreateIndividual(fitness: 0.9),  // Better
        CreateIndividual(fitness: 0.3)
    };
    ga.SetPopulation(population);
    ga.ConfigureGeneticParameters(new GeneticParameters
    {
        TournamentSize = 2,
        SelectionMethod = SelectionMethod.Tournament
    });

    // Act - Run selection many times
    var selectedFitnesses = new List<double>();
    for (int i = 0; i < 100; i++)
    {
        var selected = ga.Select(1, SelectionMethod.Tournament).First();
        selectedFitnesses.Add(selected.GetFitness());
    }

    // Assert - Better individuals selected more often
    double averageFitness = selectedFitnesses.Average();
    Assert.IsTrue(averageFitness > 0.6,
        "Average fitness should be above mean due to tournament selection");
}
```

**Roulette Wheel Selection**:
```csharp
[TestMethod]
public void RouletteWheelSelection_SelectsProportionalToFitness()
{
    // Arrange
    var ga = CreateTestGA();
    var population = new List<ModelIndividual>
    {
        CreateIndividual(fitness: 0.1),   // 10% chance
        CreateIndividual(fitness: 0.9)    // 90% chance
    };
    ga.SetPopulation(population);

    // Act - Run selection many times
    int countHighFitness = 0;
    for (int i = 0; i < 1000; i++)
    {
        var selected = ga.Select(1, SelectionMethod.RouletteWheel).First();
        if (Math.Abs(selected.GetFitness() - 0.9) < 0.01)
            countHighFitness++;
    }

    // Assert - High fitness individual selected ~90% of the time
    double proportion = countHighFitness / 1000.0;
    Assert.IsTrue(proportion > 0.85 && proportion < 0.95,
        $"Expected ~0.9, got {proportion}");
}
```

**Rank Selection**:
```csharp
[TestMethod]
public void RankSelection_SelectsBasedOnRankNotFitness()
{
    // Arrange
    var ga = CreateTestGA();
    var population = new List<ModelIndividual>
    {
        CreateIndividual(fitness: 0.1),   // Rank 3
        CreateIndividual(fitness: 0.5),   // Rank 2
        CreateIndividual(fitness: 0.9)    // Rank 1 (best)
    };
    ga.SetPopulation(population);

    // Act - Run selection many times
    var selectedIndices = new List<int>();
    for (int i = 0; i < 300; i++)
    {
        var selected = ga.Select(1, SelectionMethod.Rank).First();
        int index = population.IndexOf(selected);
        selectedIndices.Add(index);
    }

    // Assert - Best rank selected most often
    int bestCount = selectedIndices.Count(i => i == 2);
    Assert.IsTrue(bestCount > 150,
        "Highest rank should be selected most frequently");
}
```

### Step 3: Test Crossover Operations

**Single-Point Crossover**:
```csharp
[TestMethod]
public void Crossover_SinglePoint_ProducesValidOffspring()
{
    // Arrange
    var ga = CreateTestGA();
    var parent1 = CreateIndividual(genes: new[] { 1.0, 2.0, 3.0, 4.0 });
    var parent2 = CreateIndividual(genes: new[] { 5.0, 6.0, 7.0, 8.0 });

    ga.ConfigureGeneticParameters(new GeneticParameters
    {
        CrossoverOperator = "SinglePoint",
        CrossoverRate = 1.0  // Always crossover
    });

    // Act
    var offspring = ga.Crossover(parent1, parent2, 1.0);

    // Assert
    Assert.AreEqual(2, offspring.Count, "Should produce 2 offspring");

    var child1Genes = offspring.First().GetGenes().Select(g => g.Value).ToList();
    var child2Genes = offspring.Last().GetGenes().Select(g => g.Value).ToList();

    // Offspring should have mix of parent genes
    bool child1HasParent1Genes = child1Genes.Any(v => new[] { 1.0, 2.0, 3.0, 4.0 }.Contains(v));
    bool child1HasParent2Genes = child1Genes.Any(v => new[] { 5.0, 6.0, 7.0, 8.0 }.Contains(v));

    Assert.IsTrue(child1HasParent1Genes, "Child1 should have genes from parent1");
    Assert.IsTrue(child1HasParent2Genes, "Child1 should have genes from parent2");
}
```

**Uniform Crossover**:
```csharp
[TestMethod]
public void Crossover_Uniform_MixesGenesRandomly()
{
    // Arrange
    var ga = CreateTestGA();
    var parent1 = CreateIndividual(genes: new[] { 1.0, 1.0, 1.0, 1.0 });
    var parent2 = CreateIndividual(genes: new[] { 2.0, 2.0, 2.0, 2.0 });

    ga.ConfigureGeneticParameters(new GeneticParameters
    {
        CrossoverOperator = "Uniform",
        CrossoverRate = 1.0
    });

    // Act - Run crossover multiple times
    var allOffspring = new List<double>();
    for (int i = 0; i < 50; i++)
    {
        var offspring = ga.Crossover(parent1, parent2, 1.0);
        allOffspring.AddRange(offspring.First().GetGenes().Select(g => g.Value));
    }

    // Assert - Should have mix of 1.0 and 2.0 values
    int count1s = allOffspring.Count(v => Math.Abs(v - 1.0) < 0.01);
    int count2s = allOffspring.Count(v => Math.Abs(v - 2.0) < 0.01);

    Assert.IsTrue(count1s > 70 && count1s < 130,
        $"Expected ~100 genes from parent1, got {count1s}");
    Assert.IsTrue(count2s > 70 && count2s < 130,
        $"Expected ~100 genes from parent2, got {count2s}");
}
```

### Step 4: Test Mutation Operations

**Uniform Mutation**:
```csharp
[TestMethod]
public void Mutate_Uniform_ChangesGenesRandomly()
{
    // Arrange
    var ga = CreateTestGA();
    var original = CreateIndividual(genes: new[] { 1.0, 1.0, 1.0, 1.0 });

    ga.ConfigureGeneticParameters(new GeneticParameters
    {
        MutationOperator = "Uniform",
        MutationRate = 0.5  // 50% chance per gene
    });

    // Act - Run mutation multiple times
    int mutationCount = 0;
    for (int i = 0; i < 100; i++)
    {
        var mutated = ga.Mutate(original.Clone(), 0.5);
        var genes = mutated.GetGenes().Select(g => g.Value).ToList();

        // Count genes that changed
        for (int j = 0; j < 4; j++)
        {
            if (Math.Abs(genes[j] - 1.0) > 0.01)
                mutationCount++;
        }
    }

    // Assert - Approximately 50% of genes should mutate (200 out of 400)
    Assert.IsTrue(mutationCount > 150 && mutationCount < 250,
        $"Expected ~200 mutations, got {mutationCount}");
}
```

**Gaussian Mutation**:
```csharp
[TestMethod]
public void Mutate_Gaussian_UsesNormalDistribution()
{
    // Arrange
    var ga = CreateTestGA();
    var original = CreateIndividual(genes: new[] { 0.0, 0.0, 0.0, 0.0 });

    ga.ConfigureGeneticParameters(new GeneticParameters
    {
        MutationOperator = "Gaussian",
        MutationRate = 1.0  // Always mutate
    });

    // Act - Collect mutated values
    var mutatedValues = new List<double>();
    for (int i = 0; i < 1000; i++)
    {
        var mutated = ga.Mutate(original.Clone(), 1.0);
        mutatedValues.AddRange(mutated.GetGenes().Select(g => g.Value));
    }

    // Assert - Should follow normal distribution
    double mean = mutatedValues.Average();
    double variance = mutatedValues.Select(v => Math.Pow(v - mean, 2)).Average();
    double stdDev = Math.Sqrt(variance);

    Assert.IsTrue(Math.Abs(mean) < 0.02,
        $"Mean should be close to 0, got {mean}");
    Assert.IsTrue(stdDev > 0.08 && stdDev < 0.12,
        $"StdDev should be ~0.1, got {stdDev}");
}
```

### Step 5: Test Evolution Loop

**Basic Evolution**:
```csharp
[TestMethod]
public void Evolve_WithMaxGenerations_ImprovesF fitness()
{
    // Arrange
    var ga = CreateTestGA();
    var trainingX = CreateTrainingData();
    var trainingY = CreateTrainingLabels();

    ga.ConfigureGeneticParameters(new GeneticParameters
    {
        PopulationSize = 20,
        MaxGenerations = 50,
        ElitismRate = 0.1,
        CrossoverRate = 0.8,
        MutationRate = 0.1
    });

    // Act
    var stats = ga.Evolve(
        generations: 50,
        trainingInput: trainingX,
        trainingOutput: trainingY
    );

    // Assert
    Assert.IsTrue(stats.FitnessHistory.Count > 0, "Should track fitness");

    double initialFitness = stats.FitnessHistory[0];
    double finalFitness = stats.FitnessHistory[stats.FitnessHistory.Count - 1];

    Assert.IsTrue(finalFitness >= initialFitness,
        "Fitness should improve or stay same");
}
```

**Stopping Criteria - Fitness Threshold**:
```csharp
[TestMethod]
public void Evolve_WithFitnessThreshold_StopsEarly()
{
    // Arrange
    var ga = CreateTestGA();
    var trainingX = CreateTrainingData();
    var trainingY = CreateTrainingLabels();

    ga.ConfigureGeneticParameters(new GeneticParameters
    {
        PopulationSize = 20,
        FitnessThreshold = 0.95  // Stop when fitness >= 0.95
    });

    // Act
    var stats = ga.Evolve(
        generations: 1000,  // High max generations
        trainingInput: trainingX,
        trainingOutput: trainingY
    );

    // Assert
    Assert.IsTrue(stats.Generation < 1000,
        "Should stop before max generations");
    Assert.IsTrue(stats.BestFitness >= 0.95 || stats.Generation == 1000,
        "Should stop when threshold reached");
}
```

**Stopping Criteria - Stagnation**:
```csharp
[TestMethod]
public void Evolve_WithStagnation_StopsAfterNoImprovement()
{
    // Arrange
    var ga = CreateTestGA();
    var trainingX = CreateTrainingData();
    var trainingY = CreateTrainingLabels();

    ga.ConfigureGeneticParameters(new GeneticParameters
    {
        PopulationSize = 20,
        MaxGenerationsWithoutImprovement = 10  // Stop after 10 generations without improvement
    });

    // Act
    var stats = ga.Evolve(
        generations: 1000,
        trainingInput: trainingX,
        trainingOutput: trainingY
    );

    // Assert
    Assert.IsTrue(stats.GenerationsSinceImprovement <= 10,
        "Should stop after max generations without improvement");
}
```

### Step 6: Test Elitism

```csharp
[TestMethod]
public void GetElites_PreservesBestIndividuals()
{
    // Arrange
    var ga = CreateTestGA();
    var population = new List<ModelIndividual>
    {
        CreateIndividual(fitness: 0.9),  // Best
        CreateIndividual(fitness: 0.8),  // Second best
        CreateIndividual(fitness: 0.5),
        CreateIndividual(fitness: 0.3),
        CreateIndividual(fitness: 0.1)
    };
    ga.SetPopulation(population);

    // Act
    var elites = ga.GetElites(count: 2);

    // Assert
    Assert.AreEqual(2, elites.Count);

    var eliteFitnesses = elites.Select(e => e.GetFitness()).OrderByDescending(f => f).ToList();
    Assert.AreEqual(0.9, eliteFitnesses[0]);
    Assert.AreEqual(0.8, eliteFitnesses[1]);
}

[TestMethod]
public void CreateNextGeneration_WithElitism_PreservesBestIndividuals()
{
    // Arrange
    var ga = CreateTestGA();
    ga.ConfigureGeneticParameters(new GeneticParameters
    {
        PopulationSize = 10,
        ElitismRate = 0.2  // Keep top 2 individuals
    });

    var bestIndividual = CreateIndividual(fitness: 0.95);
    var population = new List<ModelIndividual> { bestIndividual };
    for (int i = 0; i < 9; i++)
        population.Add(CreateIndividual(fitness: 0.5));

    ga.SetPopulation(population);

    // Act
    var nextGen = ga.CreateNextGeneration(trainingX, trainingY);

    // Assert - Best individual should still be in population
    bool bestPreserved = nextGen.Any(ind =>
        Math.Abs(ind.GetFitness() - 0.95) < 0.01);
    Assert.IsTrue(bestPreserved, "Best individual should be preserved");
}
```

### Step 7: Test Diversity Calculation

```csharp
[TestMethod]
public void CalculateDiversity_WithIdenticalPopulation_ReturnsZero()
{
    // Arrange
    var ga = CreateTestGA();
    var genes = new[] { 1.0, 2.0, 3.0 };
    var population = new List<ModelIndividual>();
    for (int i = 0; i < 5; i++)
        population.Add(CreateIndividual(genes: genes));

    ga.SetPopulation(population);

    // Act
    var diversity = ga.CalculateDiversity();

    // Assert
    Assert.AreEqual(0.0, diversity, "Identical individuals should have zero diversity");
}

[TestMethod]
public void CalculateDiversity_WithDiversePopulation_ReturnsPositiveValue()
{
    // Arrange
    var ga = CreateTestGA();
    var population = new List<ModelIndividual>
    {
        CreateIndividual(genes: new[] { 1.0, 1.0, 1.0 }),
        CreateIndividual(genes: new[] { 2.0, 2.0, 2.0 }),
        CreateIndividual(genes: new[] { 3.0, 3.0, 3.0 })
    };
    ga.SetPopulation(population);

    // Act
    var diversity = ga.CalculateDiversity();

    // Assert
    Assert.IsTrue(diversity > 0.0, "Diverse population should have positive diversity");
}
```

### Step 8: Test NSGA-II Specific Features

**Non-Dominated Sorting**:
```csharp
[TestMethod]
public void NonDominatedSorting_AssignsCorrectRanks()
{
    // Arrange
    var nsga2 = CreateNSGA2();
    var individuals = new List<MultiObjectiveRealIndividual>
    {
        // Front 1 (non-dominated)
        CreateMultiObjective(objectives: new[] { 0.9, 100.0 }),  // High acc, fast
        CreateMultiObjective(objectives: new[] { 0.8, 50.0 }),   // Med acc, faster

        // Front 2 (dominated by front 1)
        CreateMultiObjective(objectives: new[] { 0.7, 120.0 }),  // Lower acc, slower
    };

    // Act
    nsga2.PerformNonDominatedSorting(individuals);

    // Assert
    Assert.AreEqual(1, individuals[0].Rank, "Should be in front 1");
    Assert.AreEqual(1, individuals[1].Rank, "Should be in front 1");
    Assert.AreEqual(2, individuals[2].Rank, "Should be in front 2");
}
```

**Crowding Distance**:
```csharp
[TestMethod]
public void CalculateCrowdingDistance_AssignsBoundaryInfinity()
{
    // Arrange
    var nsga2 = CreateNSGA2();
    var individuals = new List<MultiObjectiveRealIndividual>
    {
        CreateMultiObjective(objectives: new[] { 1.0, 0.0 }),  // Boundary
        CreateMultiObjective(objectives: new[] { 0.5, 0.5 }),  // Middle
        CreateMultiObjective(objectives: new[] { 0.0, 1.0 })   // Boundary
    };

    // Act
    nsga2.CalculateCrowdingDistance(individuals);

    // Assert
    Assert.AreEqual(double.PositiveInfinity, individuals[0].CrowdingDistance);
    Assert.IsTrue(individuals[1].CrowdingDistance < double.PositiveInfinity);
    Assert.AreEqual(double.PositiveInfinity, individuals[2].CrowdingDistance);
}
```

### Step 9: Test Island Model GA

**Island Initialization**:
```csharp
[TestMethod]
public void InitializeIslands_CreatesCorrectNumberOfPopulations()
{
    // Arrange
    var islandGA = CreateIslandModelGA();
    islandGA.ConfigureGeneticParameters(new GeneticParameters
    {
        NumberOfIslands = 4,
        IslandPopulationSize = 25
    });

    // Act
    islandGA.InitializeIslands();

    // Assert
    var islands = islandGA.GetIslands();
    Assert.AreEqual(4, islands.Count);
    Assert.IsTrue(islands.All(island => island.Count == 25));
}
```

**Migration**:
```csharp
[TestMethod]
public void PerformMigration_TransfersIndividualsBetweenIslands()
{
    // Arrange
    var islandGA = CreateIslandModelGA();
    islandGA.ConfigureGeneticParameters(new GeneticParameters
    {
        NumberOfIslands = 3,
        IslandPopulationSize = 20,
        MigrationRate = 0.1,  // 10% of individuals migrate
        MigrationInterval = 5  // Every 5 generations
    });

    islandGA.InitializeIslands();
    var initialIslands = islandGA.GetIslands().Select(i => i.ToList()).ToList();

    // Act - Evolve and trigger migration
    for (int gen = 0; gen < 5; gen++)
        islandGA.EvolveIslands(1);

    islandGA.PerformMigration();
    var migratedIslands = islandGA.GetIslands();

    // Assert - Some individuals should have moved between islands
    bool migrationOccurred = false;
    for (int i = 0; i < 3; i++)
    {
        var initial = initialIslands[i];
        var current = migratedIslands[i].ToList();

        // Check if populations changed
        if (!ArePopulationsIdentical(initial, current))
            migrationOccurred = true;
    }

    Assert.IsTrue(migrationOccurred, "Migration should transfer individuals");
}
```

---

## Complete Test Examples

### Example 1: GeneticBaseTests.cs (Selection Methods)

```csharp
using Microsoft.VisualStudio.TestTools.UnitTesting;
using AiDotNet.Genetics;
using AiDotNet.Models;

namespace AiDotNetTests.Genetics;

[TestClass]
public class GeneticBaseTests
{
    private TestGA _ga;
    private Matrix<double> _trainingX;
    private Vector<double> _trainingY;

    [TestInitialize]
    public void Setup()
    {
        _ga = new TestGA(
            modelFactory: () => new VectorModel<double>(Vector<double>.Empty()),
            fitnessCalculator: new TestFitnessCalculator(),
            modelEvaluator: new TestModelEvaluator()
        );

        // Create simple training data
        _trainingX = new Matrix<double>(10, 2);
        _trainingY = new Vector<double>(10);
        for (int i = 0; i < 10; i++)
        {
            _trainingX[i, 0] = i;
            _trainingX[i, 1] = i * 2;
            _trainingY[i] = i * 3;
        }
    }

    [TestMethod]
    public void TournamentSelection_WithTournamentSize2_SelectsBetterIndividuals()
    {
        // Arrange
        var population = new List<ModelIndividual<double, Matrix<double>, Vector<double>, ModelParameterGene<double>>>
        {
            CreateIndividual(fitness: 0.3),
            CreateIndividual(fitness: 0.7),
            CreateIndividual(fitness: 0.9),
            CreateIndividual(fitness: 0.5)
        };
        _ga.SetTestPopulation(population);
        _ga.ConfigureGeneticParameters(new GeneticParameters
        {
            TournamentSize = 2,
            SelectionMethod = SelectionMethod.Tournament
        });

        // Act - Run many selections
        var fitnesses = new List<double>();
        for (int i = 0; i < 100; i++)
        {
            var selected = _ga.Select(1, SelectionMethod.Tournament).First();
            fitnesses.Add(selected.GetFitness());
        }

        // Assert - Average should be above population mean
        double avgSelected = fitnesses.Average();
        double avgPopulation = population.Average(p => p.GetFitness());
        Assert.IsTrue(avgSelected > avgPopulation,
            $"Selected avg {avgSelected} should be > population avg {avgPopulation}");
    }

    [TestMethod]
    public void RouletteWheelSelection_WithHighAndLowFitness_SelectsProportionally()
    {
        // Arrange
        var highFitness = CreateIndividual(fitness: 0.9);
        var lowFitness = CreateIndividual(fitness: 0.1);
        var population = new List<ModelIndividual<double, Matrix<double>, Vector<double>, ModelParameterGene<double>>>
        {
            highFitness, lowFitness
        };
        _ga.SetTestPopulation(population);

        // Act
        int highCount = 0;
        for (int i = 0; i < 1000; i++)
        {
            var selected = _ga.Select(1, SelectionMethod.RouletteWheel).First();
            if (Math.Abs(selected.GetFitness() - 0.9) < 0.01)
                highCount++;
        }

        // Assert - ~90% should be high fitness
        double proportion = highCount / 1000.0;
        Assert.IsTrue(proportion > 0.85 && proportion < 0.95,
            $"Expected ~0.9, got {proportion}");
    }

    [TestMethod]
    public void Crossover_SinglePoint_CreatesDifferentOffspring()
    {
        // Arrange
        var parent1 = CreateIndividual(genes: new[] { 1.0, 2.0, 3.0, 4.0 });
        var parent2 = CreateIndividual(genes: new[] { 5.0, 6.0, 7.0, 8.0 });
        _ga.ConfigureGeneticParameters(new GeneticParameters
        {
            CrossoverOperator = "SinglePoint",
            CrossoverRate = 1.0
        });

        // Act
        var offspring = _ga.Crossover(parent1, parent2, 1.0).ToList();

        // Assert
        Assert.AreEqual(2, offspring.Count);

        var child1Genes = offspring[0].GetGenes().Select(g => g.Value).ToList();
        var child2Genes = offspring[1].GetGenes().Select(g => g.Value).ToList();

        // Children should differ from both parents
        Assert.IsFalse(AreGenesEqual(child1Genes, new[] { 1.0, 2.0, 3.0, 4.0 }),
            "Child1 should differ from parent1");
        Assert.IsFalse(AreGenesEqual(child1Genes, new[] { 5.0, 6.0, 7.0, 8.0 }),
            "Child1 should differ from parent2");
    }

    [TestMethod]
    public void Mutate_WithMutationRate1_ChangesAllGenes()
    {
        // Arrange
        var individual = CreateIndividual(genes: new[] { 0.0, 0.0, 0.0, 0.0 });
        _ga.ConfigureGeneticParameters(new GeneticParameters
        {
            MutationOperator = "Uniform",
            MutationRate = 1.0
        });

        // Act
        var mutated = _ga.Mutate(individual, 1.0);
        var genes = mutated.GetGenes().Select(g => g.Value).ToList();

        // Assert - All genes should have changed
        int changedCount = genes.Count(g => Math.Abs(g) > 0.01);
        Assert.AreEqual(4, changedCount, "All genes should be mutated");
    }

    [TestMethod]
    public void Evolve_ImprovesPopulationFitness()
    {
        // Arrange
        _ga.ConfigureGeneticParameters(new GeneticParameters
        {
            PopulationSize = 20,
            ElitismRate = 0.1,
            CrossoverRate = 0.8,
            MutationRate = 0.1
        });

        // Act
        var stats = _ga.Evolve(
            generations: 30,
            trainingInput: _trainingX,
            trainingOutput: _trainingY
        );

        // Assert
        Assert.IsTrue(stats.FitnessHistory.Count > 0);
        Assert.IsTrue(stats.BestFitness >= stats.FitnessHistory[0],
            "Final fitness should be >= initial fitness");
    }

    // Helper methods
    private ModelIndividual<double, Matrix<double>, Vector<double>, ModelParameterGene<double>>
        CreateIndividual(double fitness = 0.0, double[] genes = null)
    {
        genes = genes ?? new[] { 1.0, 2.0, 3.0 };
        var geneList = genes.Select((v, i) => new ModelParameterGene<double>(i, v)).ToList();

        var individual = new ModelIndividual<double, Matrix<double>, Vector<double>, ModelParameterGene<double>>(
            geneList,
            g => new VectorModel<double>(new Vector<double>(g.Select(gene => gene.Value).ToArray()))
        );
        individual.SetFitness(fitness);

        return individual;
    }

    private bool AreGenesEqual(List<double> genes1, double[] genes2)
    {
        if (genes1.Count != genes2.Length)
            return false;

        for (int i = 0; i < genes1.Count; i++)
        {
            if (Math.Abs(genes1[i] - genes2[i]) > 0.01)
                return false;
        }
        return true;
    }
}
```

### Example 2: StandardGeneticAlgorithmTests.cs

```csharp
using Microsoft.VisualStudio.TestTools.UnitTesting;
using AiDotNet.Genetics;
using AiDotNet.Models;

namespace AiDotNetTests.Genetics;

[TestClass]
public class StandardGeneticAlgorithmTests
{
    [TestMethod]
    public void Constructor_WithValidParameters_InitializesCorrectly()
    {
        // Arrange & Act
        var ga = new StandardGeneticAlgorithm<double, Matrix<double>, Vector<double>>(
            modelFactory: () => new VectorModel<double>(new Vector<double>(3)),
            fitnessCalculator: new TestFitnessCalculator(),
            modelEvaluator: new TestModelEvaluator()
        );

        // Assert
        Assert.IsNotNull(ga);
        var parameters = ga.GetGeneticParameters();
        Assert.IsNotNull(parameters);
    }

    [TestMethod]
    [ExpectedException(typeof(ArgumentNullException))]
    public void Constructor_WithNullModelFactory_ThrowsException()
    {
        // Arrange & Act
        var ga = new StandardGeneticAlgorithm<double, Matrix<double>, Vector<double>>(
            modelFactory: null,
            fitnessCalculator: new TestFitnessCalculator(),
            modelEvaluator: new TestModelEvaluator()
        );

        // Assert - Exception expected
    }

    [TestMethod]
    public void InitializePopulation_WithRandomMethod_CreatesValidIndividuals()
    {
        // Arrange
        var ga = new StandardGeneticAlgorithm<double, Matrix<double>, Vector<double>>(
            modelFactory: () => new VectorModel<double>(new Vector<double>(5)),
            fitnessCalculator: new TestFitnessCalculator(),
            modelEvaluator: new TestModelEvaluator()
        );

        // Act
        var population = ga.InitializePopulation(
            populationSize: 10,
            initializationMethod: InitializationMethod.Random
        );

        // Assert
        Assert.AreEqual(10, population.Count);
        Assert.IsTrue(population.All(ind => ind.GetGenes().Count == 5),
            "All individuals should have 5 genes");
    }

    [TestMethod]
    public void MutateGene_ProducesDifferentValue()
    {
        // Arrange
        var ga = new StandardGeneticAlgorithm<double, Matrix<double>, Vector<double>>(
            modelFactory: () => new VectorModel<double>(new Vector<double>(3)),
            fitnessCalculator: new TestFitnessCalculator(),
            modelEvaluator: new TestModelEvaluator()
        );
        var originalGene = new ModelParameterGene<double>(0, 5.0);

        // Act
        var mutatedGene = ga.MutateGene(originalGene);

        // Assert
        Assert.AreNotEqual(originalGene.Value, mutatedGene.Value,
            "Mutated gene should have different value");
    }

    [TestMethod]
    public void IndividualToModel_CreatesValidModel()
    {
        // Arrange
        var ga = new StandardGeneticAlgorithm<double, Matrix<double>, Vector<double>>(
            modelFactory: () => new VectorModel<double>(new Vector<double>(3)),
            fitnessCalculator: new TestFitnessCalculator(),
            modelEvaluator: new TestModelEvaluator()
        );
        var genes = new List<ModelParameterGene<double>>
        {
            new ModelParameterGene<double>(0, 1.0),
            new ModelParameterGene<double>(1, 2.0),
            new ModelParameterGene<double>(2, 3.0)
        };
        var individual = ga.CreateIndividual(genes);

        // Act
        var model = ga.IndividualToModel(individual);

        // Assert
        Assert.IsNotNull(model);
        var parameters = model.GetParameters();
        Assert.AreEqual(3, parameters.Length);
        Assert.AreEqual(1.0, parameters[0]);
        Assert.AreEqual(2.0, parameters[1]);
        Assert.AreEqual(3.0, parameters[2]);
    }
}
```

---

## Success Criteria

### Definition of Done

- [ ] 6 test files created (GeneticBase + 5 algorithm variants)
- [ ] Minimum 20 tests for GeneticBase
- [ ] Minimum 10 tests per algorithm variant (50 total)
- [ ] All tests passing (0 failures)
- [ ] Code coverage >= 70% for GeneticBase
- [ ] Code coverage >= 60% for each algorithm
- [ ] Integration tests verify evolution improves fitness
- [ ] Selection, crossover, mutation all tested independently

### Quality Checklist

- [ ] All selection methods tested (7 methods)
- [ ] All crossover operators tested
- [ ] All mutation operators tested
- [ ] Stopping criteria validated
- [ ] Elitism verified
- [ ] Diversity calculation tested
- [ ] NSGA-II Pareto sorting tested
- [ ] Island model migration tested

---

## Next Steps

After completing this issue:

1. **Run full test suite** and verify all tests pass
2. **Review code coverage** for gaps
3. **Performance test** evolution speed
4. **Move to Issue #349** (Math/Statistics Helper tests)

---

## Resources

- [Genetic Algorithms Tutorial](https://www.tutorialspoint.com/genetic_algorithms/index.htm)
- [NSGA-II Paper](https://ieeexplore.ieee.org/document/996017)
- [Island Model GAs](https://en.wikipedia.org/wiki/Island_model)

---

**Happy Testing!** Good genetic algorithm tests ensure reliable evolution.
