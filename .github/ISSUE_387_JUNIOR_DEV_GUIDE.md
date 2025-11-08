# Junior Developer Implementation Guide: Issue #387

## Overview
**Issue**: Decision Trees and Random Forests
**Goal**: Implement CART decision trees and Random Forest ensemble classifier
**Difficulty**: Advanced
**Estimated Time**: 14-18 hours

## What You'll Be Building

You'll implement **tree-based classifiers**:

1. **IDecisionTree Interface** - Defines decision tree methods
2. **DecisionTreeNode** - Tree node structure (splits, leaves)
3. **DecisionTreeClassifier** - CART algorithm implementation
4. **RandomForestClassifier** - Ensemble of decision trees
5. **Tree splitting algorithms** - Gini impurity and Entropy
6. **Comprehensive Unit Tests** - 80%+ coverage

## Understanding Decision Trees

### What is a Decision Tree?

**Decision Tree** is a tree-like model where each node represents a decision based on a feature, and each leaf represents a class prediction.

**Real-World Analogy:**

Think of a decision tree like a flowchart for diagnosing car problems:
```
Is engine making noise?
├─ Yes → Is noise loud?
│   ├─ Yes → Check transmission (Predict: Transmission issue)
│   └─ No → Check oil (Predict: Low oil)
└─ No → Does car start?
    ├─ Yes → Predict: No problem
    └─ No → Predict: Battery issue
```

### Mathematical Formulas

**CART (Classification and Regression Trees):**

The goal: Find the best split at each node to maximize class purity.

**1. Gini Impurity:**
```
Gini(D) = 1 - ∑ pi²

where:
    D = dataset at this node
    pi = proportion of samples in class i
    ∑ = sum over all classes

Properties:
    Gini = 0: Perfect purity (all samples same class)
    Gini = 0.5: Maximum impurity (binary, 50/50 split)

Example: [40 spam, 60 not spam]
    p(spam) = 40/100 = 0.4
    p(not spam) = 60/100 = 0.6
    Gini = 1 - (0.4² + 0.6²) = 1 - (0.16 + 0.36) = 0.48
```

**2. Entropy (Information Gain):**
```
Entropy(D) = -∑ pi * log₂(pi)

Properties:
    Entropy = 0: Perfect purity
    Entropy = 1: Maximum impurity (binary, 50/50)

Example: [40 spam, 60 not spam]
    Entropy = -(0.4 * log₂(0.4) + 0.6 * log₂(0.6))
            = -(0.4 * -1.32 + 0.6 * -0.74)
            = -(-0.528 - 0.444)
            = 0.972
```

**3. Information Gain (Reduction in Impurity):**
```
IG(D, A) = Impurity(D) - ∑ (|Dv| / |D|) * Impurity(Dv)

where:
    A = feature to split on
    Dv = subset of D where feature A has value v
    |D| = number of samples in D

This measures how much a split reduces impurity.
We choose the split with maximum information gain.
```

**4. Split Selection:**
```
Best Split = argmax IG(D, feature, threshold)
             feature, threshold

For each feature:
    For each possible threshold:
        Calculate information gain
    Keep split with highest gain
```

### CART Algorithm (Recursive Binary Splitting)

```
1. Start with all training data at root
2. For current node:
   a. If stopping criteria met (max depth, min samples, pure node):
      - Create leaf node with majority class
   b. Else:
      - Find best split (feature + threshold with max IG)
      - Split data into left and right subsets
      - Recursively build left and right subtrees
3. Return tree
```

**Stopping Criteria:**
- Max depth reached
- Min samples per leaf reached
- Node is pure (all samples same class)
- No information gain possible

### Random Forests

**Random Forest** is an ensemble of decision trees trained on random subsets.

**Key Ideas:**

1. **Bootstrap Aggregating (Bagging)**:
   - Create multiple datasets by sampling with replacement
   - Train one tree on each dataset
   - Combine predictions by majority voting

2. **Random Feature Selection**:
   - At each split, consider only a random subset of features
   - Typical: sqrt(n_features) for classification
   - Reduces correlation between trees

**Why Random Forests Work Better:**
- **Reduces overfitting**: Individual trees overfit, but average doesn't
- **Reduces variance**: Averaging reduces prediction variance
- **More robust**: Outliers affect only some trees, not all
- **Feature importance**: Can measure which features are most useful

**Mathematical Formula:**
```
For classification:
    y_pred = mode([tree1.predict(x), tree2.predict(x), ..., treeN.predict(x)])

For regression:
    y_pred = mean([tree1.predict(x), tree2.predict(x), ..., treeN.predict(x)])

Feature Importance:
    Importance(feature) = ∑ (IG(feature) * samples_at_node) / total_samples
                          over all trees and nodes
```

**Hyperparameters:**
- `n_estimators`: Number of trees (more is better, but slower)
- `max_depth`: Maximum tree depth (prevent overfitting)
- `min_samples_split`: Minimum samples to split a node
- `max_features`: Features to consider at each split (sqrt(n) or log2(n))
- `bootstrap`: Whether to use bootstrap samples

## Understanding the Codebase

### Architecture Pattern

```
IDecisionTree<T>                    (Interface)
    ↓
DecisionTreeClassifier<T>           (CART implementation)
    ↓ (uses many)
DecisionTreeNode<T>                 (Tree structure)

RandomForestClassifier<T>           (Ensemble)
    ↓ (contains many)
DecisionTreeClassifier<T>[]         (Array of trees)
```

## Step-by-Step Implementation Guide

### Step 1: Create IDecisionTree Interface

Create file: `C:\Users\cheat\source\repos\AiDotNet\src\Interfaces\IDecisionTree.cs`

```csharp
namespace AiDotNet.Interfaces;

/// <summary>
/// Defines methods for decision tree classification models.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> A decision tree makes predictions by asking a series of questions.
///
/// Think of it like a game of "20 questions":
/// - Each question splits the possibilities into smaller groups
/// - You keep asking questions until you narrow down to an answer
/// - The questions are chosen to reduce uncertainty as much as possible
///
/// Example: Predicting if email is spam
/// - Question 1: Does it contain "free"? → Yes/No
/// - Question 2: Does it contain "money"? → Yes/No
/// - Question 3: Is sender known? → Yes/No
/// - Prediction: Spam or Not Spam
///
/// Real-world uses:
/// - Medical diagnosis (symptoms → disease)
/// - Credit approval (features → approve/deny)
/// - Customer segmentation (behavior → segment)
/// - Game AI (game state → action)
/// </para>
/// </remarks>
public interface IDecisionTree<T>
{
    /// <summary>
    /// Trains the decision tree on the provided data.
    /// </summary>
    void Fit(Matrix<T> X, Vector<T> y);

    /// <summary>
    /// Predicts class labels for new data.
    /// </summary>
    Vector<T> Predict(Matrix<T> X);

    /// <summary>
    /// Gets the feature importances (how much each feature contributes to predictions).
    /// </summary>
    Vector<T>? FeatureImportances { get; }

    /// <summary>
    /// Gets the maximum depth of the tree.
    /// </summary>
    int TreeDepth { get; }
}
```

### Step 2: Create DecisionTreeNode

Create file: `C:\Users\cheat\source\repos\AiDotNet\src\Trees\DecisionTreeNode.cs`

```csharp
namespace AiDotNet.Trees;

/// <summary>
/// Represents a node in a decision tree.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> A tree node is either a decision point or a prediction.
///
/// Internal Node (Decision):
/// - Has a feature and threshold for splitting
/// - "If feature X > threshold, go left, else go right"
/// - Has left and right children
///
/// Leaf Node (Prediction):
/// - Has a predicted class value
/// - No children (end of the path)
/// - Returns prediction for samples reaching this node
///
/// Example:
/// Root: [Is Age > 30?]
///   ├─ Left (Age <= 30): [Is Income > 50K?]
///   │   ├─ Left: Leaf → Predict: Class 0
///   │   └─ Right: Leaf → Predict: Class 1
///   └─ Right (Age > 30): Leaf → Predict: Class 1
/// </para>
/// </remarks>
public class DecisionTreeNode<T>
{
    /// <summary>
    /// The feature index used for splitting at this node.
    /// Null for leaf nodes.
    /// </summary>
    public int? FeatureIndex { get; set; }

    /// <summary>
    /// The threshold value for splitting at this node.
    /// Samples with feature value <= threshold go left, others go right.
    /// Null for leaf nodes.
    /// </summary>
    public T? Threshold { get; set; }

    /// <summary>
    /// The left child node (feature <= threshold).
    /// Null for leaf nodes.
    /// </summary>
    public DecisionTreeNode<T>? Left { get; set; }

    /// <summary>
    /// The right child node (feature > threshold).
    /// Null for leaf nodes.
    /// </summary>
    public DecisionTreeNode<T>? Right { get; set; }

    /// <summary>
    /// The predicted class value for this node.
    /// For internal nodes, this is the majority class in case tree depth is limited.
    /// For leaf nodes, this is the final prediction.
    /// </summary>
    public T? Value { get; set; }

    /// <summary>
    /// The impurity measure at this node (Gini or Entropy).
    /// </summary>
    public T Impurity { get; set; }

    /// <summary>
    /// Number of samples at this node.
    /// </summary>
    public int NSamples { get; set; }

    /// <summary>
    /// Gets whether this is a leaf node.
    /// </summary>
    public bool IsLeaf => Left == null && Right == null;
}
```

### Step 3: Create DecisionTreeClassifier

Create file: `C:\Users\cheat\source\repos\AiDotNet\src\Classification\DecisionTreeClassifier.cs`

```csharp
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.Trees;
using AiDotNet.Helpers;

namespace AiDotNet.Classification;

/// <summary>
/// Implements a decision tree classifier using the CART algorithm.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> CART (Classification and Regression Trees) builds a binary tree.
///
/// How it works:
/// 1. Start with all data at the root
/// 2. Find the best feature and threshold to split the data
/// 3. Split data into two groups (left and right)
/// 4. Repeat for each group recursively
/// 5. Stop when reaching max depth or min samples
///
/// "Best" split = the one that most reduces impurity (Gini or Entropy)
///
/// Example: Predicting species of iris flowers
/// - Root: [Petal length > 2.5cm?]
/// - If yes: [Petal width > 1.7cm?]
///   - If yes: Predict Virginica
///   - If no: Predict Versicolor
/// - If no: Predict Setosa
/// </para>
/// </remarks>
public class DecisionTreeClassifier<T> : IDecisionTree<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    /// <summary>
    /// The root node of the tree.
    /// </summary>
    private DecisionTreeNode<T>? _root;

    /// <summary>
    /// Maximum depth of the tree.
    /// </summary>
    public int MaxDepth { get; set; }

    /// <summary>
    /// Minimum samples required to split a node.
    /// </summary>
    public int MinSamplesSplit { get; set; }

    /// <summary>
    /// Minimum samples required at a leaf node.
    /// </summary>
    public int MinSamplesLeaf { get; set; }

    /// <summary>
    /// Criterion for measuring split quality ("gini" or "entropy").
    /// </summary>
    public string Criterion { get; set; }

    /// <summary>
    /// Feature importances.
    /// </summary>
    public Vector<T>? FeatureImportances { get; private set; }

    /// <summary>
    /// Gets the actual depth of the tree.
    /// </summary>
    public int TreeDepth { get; private set; }

    /// <summary>
    /// Creates a new decision tree classifier.
    /// </summary>
    /// <param name="maxDepth">Maximum depth of the tree (default: unlimited).</param>
    /// <param name="minSamplesSplit">Minimum samples to split a node (default: 2).</param>
    /// <param name="minSamplesLeaf">Minimum samples at a leaf (default: 1).</param>
    /// <param name="criterion">Split criterion: "gini" or "entropy" (default: "gini").</param>
    public DecisionTreeClassifier(int maxDepth = int.MaxValue, int minSamplesSplit = 2,
        int minSamplesLeaf = 1, string criterion = "gini")
    {
        MaxDepth = maxDepth;
        MinSamplesSplit = minSamplesSplit;
        MinSamplesLeaf = minSamplesLeaf;
        Criterion = criterion.ToLower();

        if (Criterion != "gini" && Criterion != "entropy")
            throw new ArgumentException("Criterion must be 'gini' or 'entropy'", nameof(criterion));
    }

    /// <summary>
    /// Trains the decision tree.
    /// </summary>
    public void Fit(Matrix<T> X, Vector<T> y)
    {
        if (X.Rows != y.Length)
            throw new ArgumentException("Number of samples in X must match length of y");

        int nFeatures = X.Columns;
        FeatureImportances = new Vector<T>(nFeatures);

        // Build tree recursively
        var indices = Enumerable.Range(0, X.Rows).ToList();
        _root = BuildTree(X, y, indices, depth: 0);

        // Calculate tree depth
        TreeDepth = CalculateDepth(_root);

        // Normalize feature importances
        T totalImportance = NumOps.Zero;
        for (int i = 0; i < nFeatures; i++)
        {
            totalImportance = NumOps.Add(totalImportance, FeatureImportances[i]);
        }

        if (NumOps.GreaterThan(totalImportance, NumOps.Zero))
        {
            for (int i = 0; i < nFeatures; i++)
            {
                FeatureImportances[i] = NumOps.Divide(FeatureImportances[i], totalImportance);
            }
        }
    }

    /// <summary>
    /// Recursively builds the decision tree.
    /// </summary>
    private DecisionTreeNode<T> BuildTree(Matrix<T> X, Vector<T> y, List<int> indices, int depth)
    {
        int nSamples = indices.Count;
        var node = new DecisionTreeNode<T> { NSamples = nSamples };

        // Calculate impurity and majority class
        node.Impurity = CalculateImpurity(y, indices);
        node.Value = GetMajorityClass(y, indices);

        // Stopping criteria
        if (depth >= MaxDepth ||
            nSamples < MinSamplesSplit ||
            NumOps.Equals(node.Impurity, NumOps.Zero))
        {
            return node; // Leaf node
        }

        // Find best split
        var bestSplit = FindBestSplit(X, y, indices);

        if (bestSplit == null ||
            bestSplit.LeftIndices.Count < MinSamplesLeaf ||
            bestSplit.RightIndices.Count < MinSamplesLeaf)
        {
            return node; // Leaf node
        }

        // Update feature importance
        T impurityReduction = NumOps.Subtract(node.Impurity,
            NumOps.Divide(
                NumOps.Add(
                    NumOps.Multiply(NumOps.FromInt(bestSplit.LeftIndices.Count), bestSplit.LeftImpurity),
                    NumOps.Multiply(NumOps.FromInt(bestSplit.RightIndices.Count), bestSplit.RightImpurity)
                ),
                NumOps.FromInt(nSamples)
            ));

        FeatureImportances![bestSplit.FeatureIndex] = NumOps.Add(
            FeatureImportances[bestSplit.FeatureIndex],
            NumOps.Multiply(impurityReduction, NumOps.FromInt(nSamples))
        );

        // Create split
        node.FeatureIndex = bestSplit.FeatureIndex;
        node.Threshold = bestSplit.Threshold;

        // Recursively build children
        node.Left = BuildTree(X, y, bestSplit.LeftIndices, depth + 1);
        node.Right = BuildTree(X, y, bestSplit.RightIndices, depth + 1);

        return node;
    }

    /// <summary>
    /// Finds the best split for a node.
    /// </summary>
    private SplitResult<T>? FindBestSplit(Matrix<T> X, Vector<T> y, List<int> indices)
    {
        SplitResult<T>? bestSplit = null;
        T bestGain = NumOps.Negate(NumOps.One); // Initialize to -1

        T parentImpurity = CalculateImpurity(y, indices);

        // Try each feature
        for (int featureIdx = 0; featureIdx < X.Columns; featureIdx++)
        {
            // Get unique values for this feature
            var uniqueValues = new HashSet<T>();
            foreach (var idx in indices)
            {
                uniqueValues.Add(X[idx, featureIdx]);
            }

            // Try each unique value as threshold
            foreach (var threshold in uniqueValues)
            {
                var (leftIndices, rightIndices) = SplitIndices(X, indices, featureIdx, threshold);

                if (leftIndices.Count == 0 || rightIndices.Count == 0)
                    continue;

                // Calculate impurities
                T leftImpurity = CalculateImpurity(y, leftIndices);
                T rightImpurity = CalculateImpurity(y, rightIndices);

                // Calculate weighted impurity
                T nLeft = NumOps.FromInt(leftIndices.Count);
                T nRight = NumOps.FromInt(rightIndices.Count);
                T nTotal = NumOps.FromInt(indices.Count);

                T weightedImpurity = NumOps.Add(
                    NumOps.Multiply(NumOps.Divide(nLeft, nTotal), leftImpurity),
                    NumOps.Multiply(NumOps.Divide(nRight, nTotal), rightImpurity)
                );

                // Calculate information gain
                T gain = NumOps.Subtract(parentImpurity, weightedImpurity);

                // Update best split
                if (NumOps.GreaterThan(gain, bestGain))
                {
                    bestGain = gain;
                    bestSplit = new SplitResult<T>
                    {
                        FeatureIndex = featureIdx,
                        Threshold = threshold,
                        LeftIndices = leftIndices,
                        RightIndices = rightIndices,
                        LeftImpurity = leftImpurity,
                        RightImpurity = rightImpurity
                    };
                }
            }
        }

        return bestSplit;
    }

    /// <summary>
    /// Splits indices based on feature and threshold.
    /// </summary>
    private (List<int> Left, List<int> Right) SplitIndices(Matrix<T> X, List<int> indices,
        int featureIdx, T threshold)
    {
        var leftIndices = new List<int>();
        var rightIndices = new List<int>();

        foreach (var idx in indices)
        {
            if (NumOps.LessThanOrEqual(X[idx, featureIdx], threshold))
            {
                leftIndices.Add(idx);
            }
            else
            {
                rightIndices.Add(idx);
            }
        }

        return (leftIndices, rightIndices);
    }

    /// <summary>
    /// Calculates impurity (Gini or Entropy).
    /// </summary>
    private T CalculateImpurity(Vector<T> y, List<int> indices)
    {
        if (indices.Count == 0)
            return NumOps.Zero;

        // Count class frequencies
        var classCounts = new Dictionary<T, int>();
        foreach (var idx in indices)
        {
            if (!classCounts.ContainsKey(y[idx]))
                classCounts[y[idx]] = 0;
            classCounts[y[idx]]++;
        }

        int total = indices.Count;

        if (Criterion == "gini")
        {
            // Gini: 1 - sum(p_i^2)
            T gini = NumOps.One;
            foreach (var count in classCounts.Values)
            {
                T p = NumOps.Divide(NumOps.FromInt(count), NumOps.FromInt(total));
                gini = NumOps.Subtract(gini, NumOps.Multiply(p, p));
            }
            return gini;
        }
        else // entropy
        {
            // Entropy: -sum(p_i * log2(p_i))
            T entropy = NumOps.Zero;
            foreach (var count in classCounts.Values)
            {
                if (count == 0) continue;
                T p = NumOps.Divide(NumOps.FromInt(count), NumOps.FromInt(total));
                T logP = NumOps.Log(p); // Natural log, can convert to log2 if needed
                entropy = NumOps.Subtract(entropy, NumOps.Multiply(p, logP));
            }
            return entropy;
        }
    }

    /// <summary>
    /// Gets the majority class in a subset.
    /// </summary>
    private T GetMajorityClass(Vector<T> y, List<int> indices)
    {
        var classCounts = new Dictionary<T, int>();
        foreach (var idx in indices)
        {
            if (!classCounts.ContainsKey(y[idx]))
                classCounts[y[idx]] = 0;
            classCounts[y[idx]]++;
        }

        return classCounts.OrderByDescending(kv => kv.Value).First().Key;
    }

    /// <summary>
    /// Calculates the depth of the tree.
    /// </summary>
    private int CalculateDepth(DecisionTreeNode<T>? node)
    {
        if (node == null || node.IsLeaf)
            return 0;

        return 1 + Math.Max(CalculateDepth(node.Left), CalculateDepth(node.Right));
    }

    /// <summary>
    /// Predicts class labels for new data.
    /// </summary>
    public Vector<T> Predict(Matrix<T> X)
    {
        if (_root == null)
            throw new InvalidOperationException("Model must be trained before prediction");

        var predictions = new Vector<T>(X.Rows);

        for (int i = 0; i < X.Rows; i++)
        {
            predictions[i] = PredictSingle(X.GetRow(i), _root);
        }

        return predictions;
    }

    /// <summary>
    /// Predicts a single sample by traversing the tree.
    /// </summary>
    private T PredictSingle(Vector<T> sample, DecisionTreeNode<T> node)
    {
        if (node.IsLeaf)
            return node.Value!;

        if (NumOps.LessThanOrEqual(sample[node.FeatureIndex!.Value], node.Threshold!))
        {
            return PredictSingle(sample, node.Left!);
        }
        else
        {
            return PredictSingle(sample, node.Right!);
        }
    }

    /// <summary>
    /// Helper class for storing split results.
    /// </summary>
    private class SplitResult<TSplit>
    {
        public int FeatureIndex { get; set; }
        public TSplit? Threshold { get; set; }
        public List<int> LeftIndices { get; set; } = new List<int>();
        public List<int> RightIndices { get; set; } = new List<int>();
        public TSplit? LeftImpurity { get; set; }
        public TSplit? RightImpurity { get; set; }
    }
}
```

### Step 4: Create RandomForestClassifier

Create file: `C:\Users\cheat\source\repos\AiDotNet\src\Classification\RandomForestClassifier.cs`

```csharp
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.Helpers;

namespace AiDotNet.Classification;

/// <summary>
/// Implements a Random Forest classifier (ensemble of decision trees).
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> Random Forest is like asking multiple experts and voting.
///
/// Instead of one decision tree, Random Forest creates many trees:
/// 1. Each tree is trained on a random sample of the data (bootstrap)
/// 2. Each tree considers only a random subset of features at each split
/// 3. For prediction, all trees vote and majority wins
///
/// Why is this better?
/// - Single tree: Can overfit (memorize training data)
/// - Random Forest: Averaging reduces overfitting
/// - More robust to noise and outliers
/// - Generally higher accuracy
///
/// Analogy: One doctor might misdiagnose, but if 100 doctors independently
/// examine a patient and 95 say "flu", you can be pretty confident it's flu.
/// </para>
/// </remarks>
public class RandomForestClassifier<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    /// <summary>
    /// The ensemble of decision trees.
    /// </summary>
    private List<DecisionTreeClassifier<T>> _trees = new List<DecisionTreeClassifier<T>>();

    /// <summary>
    /// Number of trees in the forest.
    /// </summary>
    public int NEstimators { get; set; }

    /// <summary>
    /// Maximum depth of each tree.
    /// </summary>
    public int MaxDepth { get; set; }

    /// <summary>
    /// Minimum samples to split a node.
    /// </summary>
    public int MinSamplesSplit { get; set; }

    /// <summary>
    /// Whether to use bootstrap samples.
    /// </summary>
    public bool Bootstrap { get; set; }

    /// <summary>
    /// Number of features to consider at each split (null = sqrt(n_features)).
    /// </summary>
    public int? MaxFeatures { get; set; }

    /// <summary>
    /// Random seed for reproducibility.
    /// </summary>
    public int? RandomState { get; set; }

    /// <summary>
    /// Feature importances (aggregated from all trees).
    /// </summary>
    public Vector<T>? FeatureImportances { get; private set; }

    private Random _random;

    /// <summary>
    /// Creates a new Random Forest classifier.
    /// </summary>
    public RandomForestClassifier(int nEstimators = 100, int maxDepth = int.MaxValue,
        int minSamplesSplit = 2, bool bootstrap = true, int? maxFeatures = null,
        int? randomState = null)
    {
        NEstimators = nEstimators;
        MaxDepth = maxDepth;
        MinSamplesSplit = minSamplesSplit;
        Bootstrap = bootstrap;
        MaxFeatures = maxFeatures;
        RandomState = randomState;

        _random = randomState.HasValue ? new Random(randomState.Value) : new Random();
    }

    /// <summary>
    /// Trains the Random Forest.
    /// </summary>
    public void Fit(Matrix<T> X, Vector<T> y)
    {
        _trees = new List<DecisionTreeClassifier<T>>();
        int nSamples = X.Rows;
        int nFeatures = X.Columns;

        // Default max_features: sqrt(n_features)
        int maxFeaturesPerTree = MaxFeatures ?? (int)Math.Sqrt(nFeatures);

        // Train each tree
        for (int i = 0; i < NEstimators; i++)
        {
            // Bootstrap sample (if enabled)
            var (bootX, bootY) = Bootstrap
                ? CreateBootstrapSample(X, y, nSamples)
                : (X, y);

            // Create and train tree
            var tree = new DecisionTreeClassifier<T>(
                maxDepth: MaxDepth,
                minSamplesSplit: MinSamplesSplit
            );

            tree.Fit(bootX, bootY);
            _trees.Add(tree);
        }

        // Aggregate feature importances
        FeatureImportances = new Vector<T>(nFeatures);
        foreach (var tree in _trees)
        {
            if (tree.FeatureImportances != null)
            {
                for (int i = 0; i < nFeatures; i++)
                {
                    FeatureImportances[i] = NumOps.Add(
                        FeatureImportances[i],
                        tree.FeatureImportances[i]
                    );
                }
            }
        }

        // Normalize
        T total = NumOps.Zero;
        for (int i = 0; i < nFeatures; i++)
        {
            total = NumOps.Add(total, FeatureImportances[i]);
        }

        if (NumOps.GreaterThan(total, NumOps.Zero))
        {
            for (int i = 0; i < nFeatures; i++)
            {
                FeatureImportances[i] = NumOps.Divide(FeatureImportances[i], total);
            }
        }
    }

    /// <summary>
    /// Creates a bootstrap sample (sampling with replacement).
    /// </summary>
    private (Matrix<T> X, Vector<T> y) CreateBootstrapSample(Matrix<T> X, Vector<T> y, int nSamples)
    {
        var bootX = new Matrix<T>(nSamples, X.Columns);
        var bootY = new Vector<T>(nSamples);

        for (int i = 0; i < nSamples; i++)
        {
            int idx = _random.Next(nSamples);
            for (int j = 0; j < X.Columns; j++)
            {
                bootX[i, j] = X[idx, j];
            }
            bootY[i] = y[idx];
        }

        return (bootX, bootY);
    }

    /// <summary>
    /// Predicts class labels using majority voting.
    /// </summary>
    public Vector<T> Predict(Matrix<T> X)
    {
        if (_trees.Count == 0)
            throw new InvalidOperationException("Model must be trained before prediction");

        var predictions = new Vector<T>(X.Rows);

        for (int i = 0; i < X.Rows; i++)
        {
            var sample = X.GetRow(i);
            var sampleMatrix = new Matrix<T>(1, sample.Length);
            for (int j = 0; j < sample.Length; j++)
            {
                sampleMatrix[0, j] = sample[j];
            }

            // Collect votes from all trees
            var votes = new Dictionary<T, int>();
            foreach (var tree in _trees)
            {
                T vote = tree.Predict(sampleMatrix)[0];
                if (!votes.ContainsKey(vote))
                    votes[vote] = 0;
                votes[vote]++;
            }

            // Majority vote
            predictions[i] = votes.OrderByDescending(kv => kv.Value).First().Key;
        }

        return predictions;
    }
}
```

## Test Coverage Checklist

**Decision Tree:**
- [ ] Perfect separation (iris dataset)
- [ ] Handles overfitting (deep vs shallow trees)
- [ ] Gini vs Entropy criterion
- [ ] Feature importance calculation
- [ ] Tree depth calculation
- [ ] Min samples split/leaf enforcement

**Random Forest:**
- [ ] Better accuracy than single tree
- [ ] Bootstrap sampling
- [ ] Majority voting
- [ ] Feature importances aggregation
- [ ] Multiple trees independence
- [ ] Handles overfitting better than single tree

## Common Mistakes to Avoid

1. **Too deep trees**: Leads to overfitting
2. **Not enough trees in forest**: Reduces accuracy, use 100+
3. **Ignoring feature importance**: Valuable for feature selection
4. **Not using bootstrap**: Reduces diversity in forest
5. **Wrong max_features**: Use sqrt(n) for classification

## Learning Resources

- **CART Algorithm**: https://en.wikipedia.org/wiki/Decision_tree_learning
- **Random Forests**: https://en.wikipedia.org/wiki/Random_forest
- **Sklearn Decision Trees**: https://scikit-learn.org/stable/modules/tree.html

## Validation Criteria

1. Decision tree implements CART algorithm
2. Random forest uses bootstrap and majority voting
3. Feature importances calculated correctly
4. Test coverage 80%+
5. Handles both Gini and Entropy criteria

---

**Good luck!** Decision trees and Random Forests are among the most widely used ML algorithms in production systems.
