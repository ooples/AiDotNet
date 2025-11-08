# Issue #347: Junior Developer Implementation Guide
## Unit Tests for Gene Types and Individuals

---

## Table of Contents
1. [Understanding the Problem](#understanding-the-problem)
2. [Understanding Genetic Algorithms](#understanding-genetic-algorithms)
3. [Gene Types Overview](#gene-types-overview)
4. [Individual Types Overview](#individual-types-overview)
5. [Testing Strategy](#testing-strategy)
6. [Step-by-Step Implementation Guide](#step-by-step-implementation-guide)
7. [Test File Structure](#test-file-structure)
8. [Complete Test Examples](#complete-test-examples)

---

## Understanding the Problem

### What Are We Testing?

The `src/Genetics/` directory contains **gene types** and **individual types** that are the foundation of evolutionary algorithms. These currently have **0% test coverage**, meaning there are no unit tests validating their behavior.

### Why Is This Important?

Genetic algorithms are used to evolve solutions to complex problems. If the basic gene and individual types have bugs:
- **Evolution won't work correctly** - bad genes lead to bad solutions
- **Fitness evaluation fails** - can't tell good solutions from bad ones
- **Cloning/mutation breaks** - operations produce invalid individuals
- **Serialization corrupts data** - can't save/load evolved solutions

### Current Code Organization

**Gene Types** (Basic building blocks):
```
src/Genetics/
├── BinaryGene.cs          # Binary values (0 or 1)
├── RealGene.cs            # Real numbers with mutation step size
├── PermutationGene.cs     # Indices for ordering problems
└── NodeGene.cs            # Tree nodes for genetic programming
```

**Individual Types** (Collections of genes):
```
src/Genetics/
├── TreeIndividual.cs              # Genetic programming trees
├── ModelIndividual.cs             # ML model parameters
└── MultiObjectiveRealIndividual.cs # Multi-objective optimization
```

---

## Understanding Genetic Algorithms

### What Is a Genetic Algorithm?

A **genetic algorithm** mimics biological evolution to find good solutions to problems:

1. **Population** - A collection of candidate solutions (individuals)
2. **Genes** - The components that make up each individual
3. **Fitness** - A score measuring how good a solution is
4. **Selection** - Choosing better solutions to reproduce
5. **Crossover** - Combining two parents to create offspring
6. **Mutation** - Randomly changing genes to explore new solutions

### Real-World Example: Optimizing Neural Network Weights

**Problem**: Find the best weights for a neural network

```csharp
// Each individual represents a set of neural network weights
var individual = new ModelIndividual<double, Matrix<double>, Vector<double>, ModelParameterGene<double>>(
    genes: CreateWeightGenes(),
    genesConverter: genes => ConvertToNeuralNetwork(genes)
);

// Evaluate how well these weights perform
double fitness = EvaluateAccuracy(individual);
individual.SetFitness(fitness);

// Better individuals are more likely to be selected for breeding
// Their genes (weights) are combined and mutated to create new individuals
```

### Key Concepts for Testing

**1. Genes Must Be Cloneable**
```csharp
var original = new BinaryGene(1);
var clone = original.Clone();

// Clone must be independent - changing clone doesn't affect original
clone.Value = 0;
Assert.AreEqual(1, original.Value);  // Original unchanged
Assert.AreEqual(0, clone.Value);     // Clone changed
```

**2. Equality Must Work Correctly**
```csharp
var gene1 = new RealGene(5.0, 0.1);
var gene2 = new RealGene(5.0, 0.1);

Assert.IsTrue(gene1.Equals(gene2));  // Same values = equal
Assert.AreEqual(gene1.GetHashCode(), gene2.GetHashCode());  // Same hash
```

**3. Individuals Must Track Fitness**
```csharp
var individual = new TreeIndividual(rootNode);
individual.SetFitness(0.95);

Assert.AreEqual(0.95, individual.GetFitness());
```

---

## Gene Types Overview

### BinaryGene

**Purpose**: Represents a binary decision (yes/no, true/false, 0/1)

**Use Cases**:
- Feature selection (use this feature: yes/no)
- Configuration options (enable caching: yes/no)
- Boolean switches in optimization

**Key Properties**:
```csharp
public class BinaryGene
{
    public int Value { get; set; }  // Must be 0 or 1

    public BinaryGene(int value);   // Constructor ensures binary
    public BinaryGene Clone();      // Creates independent copy
    public override bool Equals(object? obj);
    public override int GetHashCode();
}
```

**Constructor Behavior**:
```csharp
var gene1 = new BinaryGene(5);   // Value > 0 becomes 1
Assert.AreEqual(1, gene1.Value);

var gene2 = new BinaryGene(-3);  // Value <= 0 becomes 0
Assert.AreEqual(0, gene2.Value);
```

### RealGene

**Purpose**: Represents a continuous numeric value with adaptive mutation

**Use Cases**:
- Neural network weights (e.g., 0.5, -1.2, 3.7)
- Hyperparameters (learning rate: 0.001)
- Continuous optimization variables

**Key Properties**:
```csharp
public class RealGene
{
    public double Value { get; set; }      // The actual value
    public double StepSize { get; set; }   // Mutation step size

    public RealGene(double value = 0.0, double stepSize = 0.1);
    public RealGene Clone();
    public override bool Equals(object? obj);
    public override int GetHashCode();
}
```

**Why StepSize Matters**:
```csharp
// Small step size = fine-grained search
var fineGene = new RealGene(1.0, stepSize: 0.01);  // Mutations are small

// Large step size = coarse exploration
var coarseGene = new RealGene(1.0, stepSize: 1.0);  // Mutations are large
```

**Equality Considerations**:
```csharp
// Uses tolerance for floating-point comparison
var gene1 = new RealGene(5.0, 0.1);
var gene2 = new RealGene(5.0000000001, 0.1);  // Nearly identical

Assert.IsTrue(gene1.Equals(gene2));  // Should be equal within tolerance
```

### PermutationGene

**Purpose**: Represents an index in a sequence (for ordering problems)

**Use Cases**:
- Traveling salesman problem (visit city 3, then city 1, then city 5...)
- Job scheduling (do task 2, then task 4, then task 1...)
- Feature importance ordering

**Key Properties**:
```csharp
public class PermutationGene
{
    public int Index { get; set; }  // Position in sequence

    public PermutationGene(int index);
    public PermutationGene Clone();
    public override bool Equals(object? obj);
    public override int GetHashCode();
}
```

**Example Usage**:
```csharp
// Represent the ordering: visit cities [2, 0, 3, 1]
var genes = new List<PermutationGene>
{
    new PermutationGene(2),  // Visit city 2 first
    new PermutationGene(0),  // Then city 0
    new PermutationGene(3),  // Then city 3
    new PermutationGene(1)   // Finally city 1
};
```

### NodeGene

**Purpose**: Represents a node in a genetic programming tree (functions and terminals)

**Use Cases**:
- Symbolic regression (evolve mathematical formulas)
- Automated program synthesis
- Expression tree optimization

**Key Properties**:
```csharp
public class NodeGene
{
    public GeneticNodeType Type { get; set; }   // Function or Terminal
    public string Value { get; set; }            // "+", "x", "5.0", etc.
    public List<NodeGene> Children { get; set; } // Child nodes

    public NodeGene(GeneticNodeType type, string value);
    public NodeGene Clone();
    public override bool Equals(object? obj);
    public override int GetHashCode();
}
```

**Example Tree Structure**:
```csharp
// Represents: (x + 2) * 3
var multiply = new NodeGene(GeneticNodeType.Function, "*");
var add = new NodeGene(GeneticNodeType.Function, "+");
var x = new NodeGene(GeneticNodeType.Terminal, "x");
var two = new NodeGene(GeneticNodeType.Terminal, "2");
var three = new NodeGene(GeneticNodeType.Terminal, "3");

add.Children.Add(x);
add.Children.Add(two);
multiply.Children.Add(add);
multiply.Children.Add(three);
```

**Clone Must Be Deep**:
```csharp
var original = new NodeGene(GeneticNodeType.Function, "+");
original.Children.Add(new NodeGene(GeneticNodeType.Terminal, "x"));

var clone = original.Clone();
clone.Children[0].Value = "y";  // Modify clone's child

// Original must be unchanged
Assert.AreEqual("x", original.Children[0].Value);
Assert.AreEqual("y", clone.Children[0].Value);
```

---

## Individual Types Overview

### TreeIndividual

**Purpose**: Represents a complete expression tree for genetic programming

**Use Cases**:
- Symbolic regression (find formula: y = x^2 + 3x - 1)
- Automated program generation
- Mathematical expression optimization

**Key Properties**:
```csharp
public class TreeIndividual : IEvolvable<NodeGene, double>
{
    // Core operations
    public ICollection<NodeGene> GetGenes();
    public void SetGenes(ICollection<NodeGene> genes);
    public double GetFitness();
    public void SetFitness(double fitness);
    public IEvolvable<NodeGene, double> Clone();

    // Tree-specific operations
    public double Evaluate(Dictionary<string, double> variables);
    public int GetDepth();
    public string GetExpression();
    public void PointMutation();
    public void SubtreeMutation();
    public void PermutationMutation();
    public NodeGene SelectRandomNode();
}
```

**Evaluation Example**:
```csharp
// Tree represents: (x + 2) * 3
var individual = CreateTreeIndividual("(x + 2) * 3");

// Evaluate with x = 5
var result = individual.Evaluate(new Dictionary<string, double> { { "x", 5.0 } });
Assert.AreEqual(21.0, result);  // (5 + 2) * 3 = 21
```

### ModelIndividual

**Purpose**: Represents an AI model's parameters as genes

**Use Cases**:
- Evolving neural network weights
- Optimizing regression coefficients
- Finding optimal hyperparameters

**Key Properties**:
```csharp
public class ModelIndividual<T, TInput, TOutput, TGene> : IEvolvable<TGene, T>
{
    public ICollection<TGene> GetGenes();
    public void SetGenes(ICollection<TGene> genes);
    public T GetFitness();
    public void SetFitness(T fitness);
    public IEvolvable<TGene, T> Clone();
    public IFullModel<T, TInput, TOutput> ToModel();  // Convert to usable model
}
```

### MultiObjectiveRealIndividual

**Purpose**: Represents a solution optimizing multiple conflicting objectives

**Use Cases**:
- Optimize accuracy AND speed simultaneously
- Balance precision AND recall
- Minimize cost AND maximize quality

**Key Properties**:
```csharp
public class MultiObjectiveRealIndividual : IEvolvable<RealGene, double>
{
    public ICollection<RealGene> GetGenes();
    public void SetGenes(ICollection<RealGene> genes);
    public double GetFitness();  // Aggregated fitness
    public void SetFitness(double fitness);
    public IEvolvable<RealGene, double> Clone();

    // Multi-objective specific
    public List<double> ObjectiveValues { get; set; }  // Individual objective scores
    public int DominationCount { get; set; }           // How many solutions dominate this
    public int Rank { get; set; }                      // Pareto front rank
}
```

**Pareto Dominance Example**:
```csharp
// Individual A: accuracy=0.9, speed=100ms
var individualA = new MultiObjectiveRealIndividual();
individualA.ObjectiveValues = new List<double> { 0.9, 100.0 };

// Individual B: accuracy=0.95, speed=120ms
var individualB = new MultiObjectiveRealIndividual();
individualB.ObjectiveValues = new List<double> { 0.95, 120.0 };

// A dominates B if A is better in ALL objectives
// Here: A has better speed, but B has better accuracy
// Neither dominates, both are on Pareto front
```

---

## Testing Strategy

### Coverage Goals

**Gene Types** - Test these aspects:
1. **Construction** - Valid initialization, edge cases
2. **Cloning** - Deep copy, independence from original
3. **Equality** - Value comparison, hash code consistency
4. **Value constraints** - Bounds, type safety

**Individual Types** - Test these aspects:
1. **Gene management** - Get/set genes, gene collections
2. **Fitness tracking** - Get/set fitness, fitness updates
3. **Cloning** - Deep copy with all genes and fitness
4. **Operations** - Mutation, evaluation, crossover
5. **Serialization** - Save/load state correctly

### Test Organization

Create test files in `tests/` directory matching source structure:
```
tests/Genetics/
├── BinaryGeneTests.cs
├── RealGeneTests.cs
├── PermutationGeneTests.cs
├── NodeGeneTests.cs
├── TreeIndividualTests.cs
├── ModelIndividualTests.cs
└── MultiObjectiveRealIndividualTests.cs
```

### Test Naming Convention

Use descriptive test names following AAA pattern:

```csharp
[TestMethod]
public void Constructor_WithPositiveValue_SetsValueToOne()
{
    // Arrange - Set up test data
    int inputValue = 5;

    // Act - Execute the method being tested
    var gene = new BinaryGene(inputValue);

    // Assert - Verify expected behavior
    Assert.AreEqual(1, gene.Value);
}
```

### Common Test Patterns

**Pattern 1: Value Initialization**
```csharp
[TestMethod]
public void Constructor_WithDefaultParameters_InitializesCorrectly()
{
    var gene = new RealGene();

    Assert.AreEqual(0.0, gene.Value);
    Assert.AreEqual(0.1, gene.StepSize);
}
```

**Pattern 2: Clone Independence**
```csharp
[TestMethod]
public void Clone_WhenModified_DoesNotAffectOriginal()
{
    var original = new BinaryGene(1);
    var clone = original.Clone();

    clone.Value = 0;

    Assert.AreEqual(1, original.Value, "Original should not change");
    Assert.AreEqual(0, clone.Value, "Clone should change");
}
```

**Pattern 3: Equality Validation**
```csharp
[TestMethod]
public void Equals_WithSameValues_ReturnsTrue()
{
    var gene1 = new PermutationGene(5);
    var gene2 = new PermutationGene(5);

    Assert.IsTrue(gene1.Equals(gene2));
    Assert.AreEqual(gene1.GetHashCode(), gene2.GetHashCode());
}
```

**Pattern 4: Null/Edge Cases**
```csharp
[TestMethod]
public void Equals_WithNull_ReturnsFalse()
{
    var gene = new BinaryGene(1);

    Assert.IsFalse(gene.Equals(null));
}

[TestMethod]
public void Equals_WithDifferentType_ReturnsFalse()
{
    var gene = new BinaryGene(1);
    var otherObject = new object();

    Assert.IsFalse(gene.Equals(otherObject));
}
```

---

## Step-by-Step Implementation Guide

### Step 1: Set Up Test Project Structure

Create test directory if it doesn't exist:
```bash
mkdir -p tests/Genetics
cd tests
```

### Step 2: Create BinaryGeneTests.cs

**What to test**:
- Constructor ensures binary values (0 or 1)
- Clone creates independent copy
- Equality compares values correctly
- GetHashCode consistent with Equals

**Implementation checklist**:
- [ ] `Constructor_WithZero_SetsValueToZero()`
- [ ] `Constructor_WithOne_SetsValueToOne()`
- [ ] `Constructor_WithPositiveValue_SetsValueToOne()`
- [ ] `Constructor_WithNegativeValue_SetsValueToZero()`
- [ ] `Clone_CreatesIndependentCopy()`
- [ ] `Clone_WhenModified_DoesNotAffectOriginal()`
- [ ] `Equals_WithSameValue_ReturnsTrue()`
- [ ] `Equals_WithDifferentValue_ReturnsFalse()`
- [ ] `Equals_WithNull_ReturnsFalse()`
- [ ] `Equals_WithDifferentType_ReturnsFalse()`
- [ ] `GetHashCode_WithSameValue_ReturnsSameHash()`
- [ ] `GetHashCode_WithDifferentValue_ReturnsDifferentHash()`

### Step 3: Create RealGeneTests.cs

**What to test**:
- Constructor with default and custom values
- StepSize initialized correctly
- Clone copies both Value and StepSize
- Equality uses floating-point tolerance (1e-10)

**Implementation checklist**:
- [ ] `Constructor_WithDefaults_InitializesCorrectly()`
- [ ] `Constructor_WithCustomValues_SetsPropertiesCorrectly()`
- [ ] `Clone_CopiesBothValueAndStepSize()`
- [ ] `Clone_WhenModified_DoesNotAffectOriginal()`
- [ ] `Equals_WithNearlyIdenticalValues_ReturnsTrue()` (tolerance test)
- [ ] `Equals_WithDifferentValues_ReturnsFalse()`
- [ ] `Equals_WithDifferentStepSizes_ReturnsFalse()`
- [ ] `Equals_WithNull_ReturnsFalse()`
- [ ] `GetHashCode_ConsistentWithEquals()`

### Step 4: Create PermutationGeneTests.cs

**What to test**:
- Constructor sets Index
- Clone creates independent copy
- Equality compares indices

**Implementation checklist**:
- [ ] `Constructor_SetsIndex()`
- [ ] `Constructor_WithNegativeIndex_SetsNegativeValue()` (no constraint)
- [ ] `Clone_CreatesIndependentCopy()`
- [ ] `Clone_WhenModified_DoesNotAffectOriginal()`
- [ ] `Equals_WithSameIndex_ReturnsTrue()`
- [ ] `Equals_WithDifferentIndex_ReturnsFalse()`
- [ ] `Equals_WithNull_ReturnsFalse()`
- [ ] `GetHashCode_ConsistentWithEquals()`

### Step 5: Create NodeGeneTests.cs

**What to test**:
- Constructor initializes Type, Value, Children
- Clone performs deep copy (children are cloned)
- Equality checks Type, Value, and all children recursively
- GetHashCode handles children correctly

**Implementation checklist**:
- [ ] `Constructor_InitializesProperties()`
- [ ] `Constructor_InitializesEmptyChildrenList()`
- [ ] `Clone_WithNoChildren_CreatesIndependentCopy()`
- [ ] `Clone_WithChildren_PerformsDeepCopy()`
- [ ] `Clone_ModifyingClonedChild_DoesNotAffectOriginal()`
- [ ] `Equals_WithSameTypeAndValue_ReturnsTrue()`
- [ ] `Equals_WithDifferentType_ReturnsFalse()`
- [ ] `Equals_WithDifferentValue_ReturnsFalse()`
- [ ] `Equals_WithDifferentChildCount_ReturnsFalse()`
- [ ] `Equals_WithDifferentChildren_ReturnsFalse()`
- [ ] `Equals_WithNull_ReturnsFalse()`
- [ ] `GetHashCode_WithChildren_ComputesCorrectly()`

### Step 6: Create TreeIndividualTests.cs

**What to test**:
- Construction creates valid tree
- GetGenes/SetGenes work correctly
- GetFitness/SetFitness track fitness
- Clone creates independent individual
- Evaluate computes expressions correctly
- GetDepth returns tree depth
- GetExpression returns string representation
- Mutation methods modify tree
- SelectRandomNode returns valid node

**Implementation checklist**:
- [ ] `Constructor_WithRandomGeneration_CreatesValidTree()`
- [ ] `Constructor_WithRootNode_SetsRootCorrectly()`
- [ ] `GetGenes_ReturnsRootNode()`
- [ ] `SetGenes_UpdatesRootNode()`
- [ ] `GetFitness_ReturnsFitnessValue()`
- [ ] `SetFitness_UpdatesFitness()`
- [ ] `Clone_CreatesIndependentIndividual()`
- [ ] `Clone_ModifyingClone_DoesNotAffectOriginal()`
- [ ] `Evaluate_WithSimpleExpression_ComputesCorrectly()`
- [ ] `Evaluate_WithVariables_SubstitutesValues()`
- [ ] `Evaluate_WithProtectedDivision_HandlesZero()`
- [ ] `Evaluate_WithProtectedLog_HandlesNegative()`
- [ ] `GetDepth_ReturnsCorrectDepth()`
- [ ] `GetExpression_ReturnsStringRepresentation()`
- [ ] `PointMutation_ModifiesNode()`
- [ ] `SubtreeMutation_ReplacesSubtree()`
- [ ] `PermutationMutation_ReordersChildren()`
- [ ] `SelectRandomNode_ReturnsValidNode()`

### Step 7: Create ModelIndividualTests.cs

**What to test**:
- Constructor with genes and converter function
- GetGenes/SetGenes manage gene collection
- GetFitness/SetFitness track fitness
- Clone creates independent individual
- ToModel converts genes to usable model

**Implementation checklist**:
- [ ] `Constructor_WithGenes_InitializesCorrectly()`
- [ ] `GetGenes_ReturnsGeneCollection()`
- [ ] `SetGenes_UpdatesGeneCollection()`
- [ ] `GetFitness_ReturnsFitnessValue()`
- [ ] `SetFitness_UpdatesFitness()`
- [ ] `Clone_CreatesIndependentIndividual()`
- [ ] `Clone_ClonesAllGenes()`
- [ ] `Clone_PreservesFitness()`
- [ ] `ToModel_ConvertsGenesToModel()`
- [ ] `ToModel_UsesConverterFunction()`

### Step 8: Create MultiObjectiveRealIndividualTests.cs

**What to test**:
- Constructor initializes with RealGenes
- ObjectiveValues collection management
- Rank and DominationCount properties
- Clone preserves all multi-objective data
- GetGenes/SetGenes for RealGene collection

**Implementation checklist**:
- [ ] `Constructor_InitializesProperties()`
- [ ] `ObjectiveValues_CanBeSetAndRetrieved()`
- [ ] `Rank_CanBeSetAndRetrieved()`
- [ ] `DominationCount_CanBeSetAndRetrieved()`
- [ ] `GetGenes_ReturnsRealGenes()`
- [ ] `SetGenes_UpdatesRealGenes()`
- [ ] `GetFitness_ReturnsAggregatedFitness()`
- [ ] `SetFitness_UpdatesFitness()`
- [ ] `Clone_CreatesIndependentIndividual()`
- [ ] `Clone_ClonesObjectiveValues()`
- [ ] `Clone_PreservesRankAndDominationCount()`

---

## Test File Structure

### Standard Template for Gene Tests

```csharp
using Microsoft.VisualStudio.TestTools.UnitTesting;
using AiDotNet.Genetics;

namespace AiDotNetTests.Genetics;

[TestClass]
public class [GeneName]Tests
{
    [TestMethod]
    public void Constructor_TestScenario_ExpectedBehavior()
    {
        // Arrange

        // Act

        // Assert
    }

    [TestMethod]
    public void Clone_TestScenario_ExpectedBehavior()
    {
        // Arrange

        // Act

        // Assert
    }

    [TestMethod]
    public void Equals_TestScenario_ExpectedBehavior()
    {
        // Arrange

        // Act

        // Assert
    }

    [TestMethod]
    public void GetHashCode_TestScenario_ExpectedBehavior()
    {
        // Arrange

        // Act

        // Assert
    }
}
```

### Standard Template for Individual Tests

```csharp
using Microsoft.VisualStudio.TestTools.UnitTesting;
using AiDotNet.Genetics;

namespace AiDotNetTests.Genetics;

[TestClass]
public class [IndividualName]Tests
{
    [TestMethod]
    public void Constructor_TestScenario_ExpectedBehavior()
    {
        // Arrange

        // Act

        // Assert
    }

    [TestMethod]
    public void GetGenes_TestScenario_ExpectedBehavior()
    {
        // Arrange

        // Act

        // Assert
    }

    [TestMethod]
    public void SetGenes_TestScenario_ExpectedBehavior()
    {
        // Arrange

        // Act

        // Assert
    }

    [TestMethod]
    public void GetFitness_TestScenario_ExpectedBehavior()
    {
        // Arrange

        // Act

        // Assert
    }

    [TestMethod]
    public void SetFitness_TestScenario_ExpectedBehavior()
    {
        // Arrange

        // Act

        // Assert
    }

    [TestMethod]
    public void Clone_TestScenario_ExpectedBehavior()
    {
        // Arrange

        // Act

        // Assert
    }
}
```

---

## Complete Test Examples

### Example 1: BinaryGeneTests.cs (Complete)

```csharp
using Microsoft.VisualStudio.TestTools.UnitTesting;
using AiDotNet.Genetics;

namespace AiDotNetTests.Genetics;

[TestClass]
public class BinaryGeneTests
{
    [TestMethod]
    public void Constructor_WithZero_SetsValueToZero()
    {
        // Arrange & Act
        var gene = new BinaryGene(0);

        // Assert
        Assert.AreEqual(0, gene.Value);
    }

    [TestMethod]
    public void Constructor_WithOne_SetsValueToOne()
    {
        // Arrange & Act
        var gene = new BinaryGene(1);

        // Assert
        Assert.AreEqual(1, gene.Value);
    }

    [TestMethod]
    public void Constructor_WithPositiveValue_SetsValueToOne()
    {
        // Arrange & Act
        var gene = new BinaryGene(42);

        // Assert
        Assert.AreEqual(1, gene.Value);
    }

    [TestMethod]
    public void Constructor_WithNegativeValue_SetsValueToZero()
    {
        // Arrange & Act
        var gene = new BinaryGene(-5);

        // Assert
        Assert.AreEqual(0, gene.Value);
    }

    [TestMethod]
    public void Clone_CreatesIndependentCopy()
    {
        // Arrange
        var original = new BinaryGene(1);

        // Act
        var clone = original.Clone();

        // Assert
        Assert.AreEqual(original.Value, clone.Value);
        Assert.AreNotSame(original, clone);
    }

    [TestMethod]
    public void Clone_WhenModified_DoesNotAffectOriginal()
    {
        // Arrange
        var original = new BinaryGene(1);
        var clone = original.Clone();

        // Act
        clone.Value = 0;

        // Assert
        Assert.AreEqual(1, original.Value, "Original should not change");
        Assert.AreEqual(0, clone.Value, "Clone should change");
    }

    [TestMethod]
    public void Equals_WithSameValue_ReturnsTrue()
    {
        // Arrange
        var gene1 = new BinaryGene(1);
        var gene2 = new BinaryGene(1);

        // Act
        bool result = gene1.Equals(gene2);

        // Assert
        Assert.IsTrue(result);
    }

    [TestMethod]
    public void Equals_WithDifferentValue_ReturnsFalse()
    {
        // Arrange
        var gene1 = new BinaryGene(0);
        var gene2 = new BinaryGene(1);

        // Act
        bool result = gene1.Equals(gene2);

        // Assert
        Assert.IsFalse(result);
    }

    [TestMethod]
    public void Equals_WithNull_ReturnsFalse()
    {
        // Arrange
        var gene = new BinaryGene(1);

        // Act
        bool result = gene.Equals(null);

        // Assert
        Assert.IsFalse(result);
    }

    [TestMethod]
    public void Equals_WithDifferentType_ReturnsFalse()
    {
        // Arrange
        var gene = new BinaryGene(1);
        var otherObject = new object();

        // Act
        bool result = gene.Equals(otherObject);

        // Assert
        Assert.IsFalse(result);
    }

    [TestMethod]
    public void GetHashCode_WithSameValue_ReturnsSameHash()
    {
        // Arrange
        var gene1 = new BinaryGene(1);
        var gene2 = new BinaryGene(1);

        // Act
        int hash1 = gene1.GetHashCode();
        int hash2 = gene2.GetHashCode();

        // Assert
        Assert.AreEqual(hash1, hash2);
    }

    [TestMethod]
    public void GetHashCode_WithDifferentValue_MayReturnDifferentHash()
    {
        // Arrange
        var gene1 = new BinaryGene(0);
        var gene2 = new BinaryGene(1);

        // Act
        int hash1 = gene1.GetHashCode();
        int hash2 = gene2.GetHashCode();

        // Assert
        // Note: Different values SHOULD have different hashes for good distribution,
        // but this is not strictly required by the contract
        Assert.AreNotEqual(hash1, hash2);
    }
}
```

### Example 2: RealGeneTests.cs (Complete)

```csharp
using Microsoft.VisualStudio.TestTools.UnitTesting;
using AiDotNet.Genetics;

namespace AiDotNetTests.Genetics;

[TestClass]
public class RealGeneTests
{
    private const double Tolerance = 1e-10;

    [TestMethod]
    public void Constructor_WithDefaults_InitializesCorrectly()
    {
        // Arrange & Act
        var gene = new RealGene();

        // Assert
        Assert.AreEqual(0.0, gene.Value);
        Assert.AreEqual(0.1, gene.StepSize);
    }

    [TestMethod]
    public void Constructor_WithCustomValues_SetsPropertiesCorrectly()
    {
        // Arrange & Act
        var gene = new RealGene(value: 5.5, stepSize: 0.5);

        // Assert
        Assert.AreEqual(5.5, gene.Value);
        Assert.AreEqual(0.5, gene.StepSize);
    }

    [TestMethod]
    public void Clone_CopiesBothValueAndStepSize()
    {
        // Arrange
        var original = new RealGene(3.14, 0.01);

        // Act
        var clone = original.Clone();

        // Assert
        Assert.AreEqual(original.Value, clone.Value);
        Assert.AreEqual(original.StepSize, clone.StepSize);
        Assert.AreNotSame(original, clone);
    }

    [TestMethod]
    public void Clone_WhenModified_DoesNotAffectOriginal()
    {
        // Arrange
        var original = new RealGene(1.0, 0.1);
        var clone = original.Clone();

        // Act
        clone.Value = 2.0;
        clone.StepSize = 0.2;

        // Assert
        Assert.AreEqual(1.0, original.Value, "Original value should not change");
        Assert.AreEqual(0.1, original.StepSize, "Original step size should not change");
        Assert.AreEqual(2.0, clone.Value, "Clone value should change");
        Assert.AreEqual(0.2, clone.StepSize, "Clone step size should change");
    }

    [TestMethod]
    public void Equals_WithExactlyEqualValues_ReturnsTrue()
    {
        // Arrange
        var gene1 = new RealGene(5.0, 0.1);
        var gene2 = new RealGene(5.0, 0.1);

        // Act
        bool result = gene1.Equals(gene2);

        // Assert
        Assert.IsTrue(result);
    }

    [TestMethod]
    public void Equals_WithNearlyIdenticalValues_ReturnsTrue()
    {
        // Arrange
        var gene1 = new RealGene(5.0, 0.1);
        var gene2 = new RealGene(5.0 + 1e-11, 0.1);  // Within tolerance

        // Act
        bool result = gene1.Equals(gene2);

        // Assert
        Assert.IsTrue(result, "Should be equal within tolerance 1e-10");
    }

    [TestMethod]
    public void Equals_WithDifferentValues_ReturnsFalse()
    {
        // Arrange
        var gene1 = new RealGene(5.0, 0.1);
        var gene2 = new RealGene(6.0, 0.1);

        // Act
        bool result = gene1.Equals(gene2);

        // Assert
        Assert.IsFalse(result);
    }

    [TestMethod]
    public void Equals_WithDifferentStepSizes_ReturnsFalse()
    {
        // Arrange
        var gene1 = new RealGene(5.0, 0.1);
        var gene2 = new RealGene(5.0, 0.2);

        // Act
        bool result = gene1.Equals(gene2);

        // Assert
        Assert.IsFalse(result);
    }

    [TestMethod]
    public void Equals_WithNull_ReturnsFalse()
    {
        // Arrange
        var gene = new RealGene(5.0, 0.1);

        // Act
        bool result = gene.Equals(null);

        // Assert
        Assert.IsFalse(result);
    }

    [TestMethod]
    public void Equals_WithDifferentType_ReturnsFalse()
    {
        // Arrange
        var gene = new RealGene(5.0, 0.1);
        var otherObject = new object();

        // Act
        bool result = gene.Equals(otherObject);

        // Assert
        Assert.IsFalse(result);
    }

    [TestMethod]
    public void GetHashCode_WithSameValues_ReturnsSameHash()
    {
        // Arrange
        var gene1 = new RealGene(5.0, 0.1);
        var gene2 = new RealGene(5.0, 0.1);

        // Act
        int hash1 = gene1.GetHashCode();
        int hash2 = gene2.GetHashCode();

        // Assert
        Assert.AreEqual(hash1, hash2);
    }
}
```

### Example 3: TreeIndividualTests.cs (Partial - Key Tests)

```csharp
using Microsoft.VisualStudio.TestTools.UnitTesting;
using AiDotNet.Genetics;

namespace AiDotNetTests.Genetics;

[TestClass]
public class TreeIndividualTests
{
    [TestMethod]
    public void Constructor_WithRootNode_SetsRootCorrectly()
    {
        // Arrange
        var rootNode = new NodeGene(GeneticNodeType.Terminal, "x");

        // Act
        var individual = new TreeIndividual(rootNode);
        var genes = individual.GetGenes();

        // Assert
        Assert.IsNotNull(genes);
        Assert.AreEqual(1, genes.Count);
        Assert.AreEqual("x", genes.First().Value);
    }

    [TestMethod]
    public void GetFitness_AfterSetFitness_ReturnsSameValue()
    {
        // Arrange
        var rootNode = new NodeGene(GeneticNodeType.Terminal, "5");
        var individual = new TreeIndividual(rootNode);

        // Act
        individual.SetFitness(0.95);
        double fitness = individual.GetFitness();

        // Assert
        Assert.AreEqual(0.95, fitness);
    }

    [TestMethod]
    public void Clone_CreatesIndependentIndividual()
    {
        // Arrange
        var rootNode = new NodeGene(GeneticNodeType.Terminal, "x");
        var original = new TreeIndividual(rootNode);
        original.SetFitness(0.8);

        // Act
        var clone = (TreeIndividual)original.Clone();

        // Assert
        Assert.AreNotSame(original, clone);
        Assert.AreEqual(original.GetFitness(), clone.GetFitness());
    }

    [TestMethod]
    public void Clone_ModifyingClone_DoesNotAffectOriginal()
    {
        // Arrange
        var rootNode = new NodeGene(GeneticNodeType.Terminal, "x");
        var original = new TreeIndividual(rootNode);
        original.SetFitness(0.8);

        var clone = (TreeIndividual)original.Clone();

        // Act
        clone.SetFitness(0.9);

        // Assert
        Assert.AreEqual(0.8, original.GetFitness(), "Original fitness should not change");
        Assert.AreEqual(0.9, clone.GetFitness(), "Clone fitness should change");
    }

    [TestMethod]
    public void Evaluate_WithConstant_ReturnsConstantValue()
    {
        // Arrange
        var rootNode = new NodeGene(GeneticNodeType.Terminal, "5.0");
        var individual = new TreeIndividual(rootNode);
        var variables = new Dictionary<string, double>();

        // Act
        double result = individual.Evaluate(variables);

        // Assert
        Assert.AreEqual(5.0, result);
    }

    [TestMethod]
    public void Evaluate_WithVariable_SubstitutesValue()
    {
        // Arrange
        var rootNode = new NodeGene(GeneticNodeType.Terminal, "x");
        var individual = new TreeIndividual(rootNode);
        var variables = new Dictionary<string, double> { { "x", 3.0 } };

        // Act
        double result = individual.Evaluate(variables);

        // Assert
        Assert.AreEqual(3.0, result);
    }

    [TestMethod]
    public void Evaluate_WithAddition_ComputesCorrectly()
    {
        // Arrange - Create tree: x + 2
        var add = new NodeGene(GeneticNodeType.Function, "+");
        var x = new NodeGene(GeneticNodeType.Terminal, "x");
        var two = new NodeGene(GeneticNodeType.Terminal, "2.0");
        add.Children.Add(x);
        add.Children.Add(two);

        var individual = new TreeIndividual(add);
        var variables = new Dictionary<string, double> { { "x", 5.0 } };

        // Act
        double result = individual.Evaluate(variables);

        // Assert
        Assert.AreEqual(7.0, result);  // 5 + 2 = 7
    }

    [TestMethod]
    public void Evaluate_WithDivisionByZero_ReturnsOne()
    {
        // Arrange - Create tree: 5 / 0
        var divide = new NodeGene(GeneticNodeType.Function, "/");
        var five = new NodeGene(GeneticNodeType.Terminal, "5.0");
        var zero = new NodeGene(GeneticNodeType.Terminal, "0.0");
        divide.Children.Add(five);
        divide.Children.Add(zero);

        var individual = new TreeIndividual(divide);
        var variables = new Dictionary<string, double>();

        // Act
        double result = individual.Evaluate(variables);

        // Assert
        Assert.AreEqual(1.0, result, "Protected division should return 1.0");
    }

    [TestMethod]
    public void GetDepth_WithSingleNode_ReturnsZero()
    {
        // Arrange
        var rootNode = new NodeGene(GeneticNodeType.Terminal, "x");
        var individual = new TreeIndividual(rootNode);

        // Act
        int depth = individual.GetDepth();

        // Assert
        Assert.AreEqual(0, depth);
    }

    [TestMethod]
    public void GetDepth_WithTwoLevels_ReturnsOne()
    {
        // Arrange - Create tree: x + 2
        var add = new NodeGene(GeneticNodeType.Function, "+");
        var x = new NodeGene(GeneticNodeType.Terminal, "x");
        var two = new NodeGene(GeneticNodeType.Terminal, "2");
        add.Children.Add(x);
        add.Children.Add(two);

        var individual = new TreeIndividual(add);

        // Act
        int depth = individual.GetDepth();

        // Assert
        Assert.AreEqual(1, depth);
    }

    [TestMethod]
    public void GetExpression_WithSimpleAddition_ReturnsCorrectString()
    {
        // Arrange - Create tree: x + 2
        var add = new NodeGene(GeneticNodeType.Function, "+");
        var x = new NodeGene(GeneticNodeType.Terminal, "x");
        var two = new NodeGene(GeneticNodeType.Terminal, "2");
        add.Children.Add(x);
        add.Children.Add(two);

        var individual = new TreeIndividual(add);

        // Act
        string expression = individual.GetExpression();

        // Assert
        Assert.AreEqual("(x + 2)", expression);
    }
}
```

---

## Running and Verifying Tests

### Step 1: Build the Test Project

```bash
cd tests
dotnet build
```

**Expected output**:
```
Build succeeded.
    0 Warning(s)
    0 Error(s)
```

### Step 2: Run All Genetics Tests

```bash
dotnet test --filter "FullyQualifiedName~AiDotNetTests.Genetics"
```

**Expected output** (after implementing all tests):
```
Passed!  - Failed:     0, Passed:   120, Skipped:     0, Total:   120
```

### Step 3: Check Code Coverage

```bash
dotnet test --collect:"XPlat Code Coverage"
```

Then generate report:
```bash
reportgenerator -reports:"**/coverage.cobertura.xml" -targetdir:"coveragereport"
```

**Target coverage**: Aim for 80%+ line coverage for genes and individuals.

### Step 4: Verify Each Gene Type

Run tests for individual gene types:
```bash
dotnet test --filter "FullyQualifiedName~BinaryGeneTests"
dotnet test --filter "FullyQualifiedName~RealGeneTests"
dotnet test --filter "FullyQualifiedName~PermutationGeneTests"
dotnet test --filter "FullyQualifiedName~NodeGeneTests"
```

### Step 5: Verify Each Individual Type

```bash
dotnet test --filter "FullyQualifiedName~TreeIndividualTests"
dotnet test --filter "FullyQualifiedName~ModelIndividualTests"
dotnet test --filter "FullyQualifiedName~MultiObjectiveRealIndividualTests"
```

---

## Common Issues and Solutions

### Issue 1: Floating-Point Equality

**Problem**: RealGene equality tests fail due to floating-point precision

**Solution**: Use tolerance-based comparison (already implemented in RealGene.Equals with 1e-10)

```csharp
// Correct
var gene1 = new RealGene(5.0, 0.1);
var gene2 = new RealGene(5.0 + 1e-11, 0.1);
Assert.IsTrue(gene1.Equals(gene2));  // Within tolerance
```

### Issue 2: NodeGene Deep Clone

**Problem**: Clone doesn't recursively clone children

**Verification test**:
```csharp
[TestMethod]
public void Clone_WithChildren_ClonesRecursively()
{
    var parent = new NodeGene(GeneticNodeType.Function, "+");
    var child = new NodeGene(GeneticNodeType.Terminal, "x");
    parent.Children.Add(child);

    var clone = parent.Clone();
    clone.Children[0].Value = "y";

    Assert.AreEqual("x", parent.Children[0].Value, "Original child unchanged");
    Assert.AreEqual("y", clone.Children[0].Value, "Clone child changed");
}
```

### Issue 3: TreeIndividual Mutations

**Problem**: Mutations don't maintain tree validity

**Verification tests**:
```csharp
[TestMethod]
public void SubtreeMutation_MaintainsMaxDepth()
{
    var individual = new TreeIndividual(/* large tree */);

    individual.SubtreeMutation();

    Assert.IsTrue(individual.GetDepth() <= 12, "Tree depth should not exceed max");
}

[TestMethod]
public void PointMutation_MaintainsTreeStructure()
{
    var individual = new TreeIndividual(rootNode);
    int originalDepth = individual.GetDepth();

    individual.PointMutation();

    Assert.AreEqual(originalDepth, individual.GetDepth(), "Depth should not change");
}
```

---

## Success Criteria

### Definition of Done

- [ ] All 7 test files created (4 genes + 3 individuals)
- [ ] Minimum 15 tests per gene type (60 total gene tests)
- [ ] Minimum 10 tests per individual type (30 total individual tests)
- [ ] All tests passing (0 failures)
- [ ] Code coverage >= 80% for genes
- [ ] Code coverage >= 70% for individuals
- [ ] No compiler warnings
- [ ] All tests use AAA (Arrange-Act-Assert) pattern
- [ ] Test names are descriptive and follow convention

### Quality Checklist

- [ ] Each test has clear purpose and description
- [ ] Edge cases are tested (null, empty, extremes)
- [ ] Clone independence verified for all types
- [ ] Equality and hash code consistency verified
- [ ] All public methods have at least one test
- [ ] Integration between genes and individuals tested

---

## Next Steps

After completing this issue:

1. **Run full test suite** to ensure no regressions
2. **Review code coverage report** to identify gaps
3. **Document any bugs found** during testing
4. **Create issues for bug fixes** if needed
5. **Move to Issue #348** (Genetic Algorithm tests)

---

## Resources

### Documentation
- [Genetic Algorithms Explained](https://en.wikipedia.org/wiki/Genetic_algorithm)
- [MSTest Framework](https://docs.microsoft.com/en-us/dotnet/core/testing/unit-testing-with-mstest)
- [C# Unit Testing Best Practices](https://docs.microsoft.com/en-us/dotnet/core/testing/unit-testing-best-practices)

### Related Issues
- Issue #348: Genetic Algorithm tests
- Issue #349: Math/Statistics Helper tests
- Issue #350: Model Helper tests

### Code Files
- `src/Genetics/BinaryGene.cs` (33 lines)
- `src/Genetics/RealGene.cs` (47 lines)
- `src/Genetics/PermutationGene.cs` (32 lines)
- `src/Genetics/NodeGene.cs` (77 lines)
- `src/Genetics/TreeIndividual.cs` (372 lines)
- `src/Genetics/ModelIndividual.cs` (needs creation)
- `src/Genetics/MultiObjectiveRealIndividual.cs` (needs creation)

---

**Happy Testing!** Remember: Good tests are the foundation of reliable evolutionary algorithms.
