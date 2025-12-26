using AiDotNet.Classification;
using AiDotNet.Models.Options;
using AiDotNet.Tensors.Helpers;

namespace AiDotNet.Classification.Trees;

/// <summary>
/// A decision tree classifier that learns a hierarchy of decision rules from training data.
/// </summary>
/// <typeparam name="T">The numeric data type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// Decision trees are non-parametric supervised learning algorithms that learn decision rules
/// inferred from data features. They partition the feature space into regions and assign
/// class labels to each region.
/// </para>
/// <para>
/// <b>For Beginners:</b>
/// Imagine playing a game of "20 Questions" to classify things. The decision tree learns
/// which questions (based on features) best separate the different classes.
///
/// Example: Classifying whether to play tennis
/// 1. Is it raining? -> No: Go to step 2, Yes: Don't play
/// 2. Is humidity > 75%? -> No: Play!, Yes: Don't play
///
/// Each question splits the data based on a feature value, and leaves contain the final decisions.
/// </para>
/// </remarks>
public class DecisionTreeClassifier<T> : ProbabilisticClassifierBase<T>, ITreeBasedClassifier<T>
{
    /// <summary>
    /// Gets the decision tree specific options.
    /// </summary>
    protected new DecisionTreeClassifierOptions<T> Options => (DecisionTreeClassifierOptions<T>)base.Options;

    /// <summary>
    /// The root node of the decision tree.
    /// </summary>
    private DecisionNode<T>? _root;

    /// <summary>
    /// Random number generator for feature selection when MaxFeatures is set.
    /// </summary>
    private Random? _random;

    /// <inheritdoc/>
    public int MaxDepth => Options.MaxDepth ?? CalculateActualDepth(_root);

    /// <inheritdoc/>
    public Vector<T>? FeatureImportances { get; private set; }

    /// <inheritdoc/>
    public int LeafCount => CountLeaves(_root);

    /// <inheritdoc/>
    public int NodeCount => CountNodes(_root);

    /// <summary>
    /// Initializes a new instance of the DecisionTreeClassifier class.
    /// </summary>
    /// <param name="options">Configuration options for the decision tree.</param>
    /// <param name="regularization">Optional regularization strategy.</param>
    public DecisionTreeClassifier(DecisionTreeClassifierOptions<T>? options = null,
        IRegularization<T, Matrix<T>, Vector<T>>? regularization = null)
        : base(options ?? new DecisionTreeClassifierOptions<T>(), regularization, new CrossEntropyLoss<T>())
    {
    }

    /// <summary>
    /// Returns the model type identifier for this classifier.
    /// </summary>
    protected override ModelType GetModelType() => ModelType.DecisionTreeClassifier;

    /// <summary>
    /// Trains the decision tree on the provided data.
    /// </summary>
    public override void Train(Matrix<T> x, Vector<T> y)
    {
        if (x.Rows != y.Length)
        {
            throw new ArgumentException("Number of samples in X must match length of y.");
        }

        NumFeatures = x.Columns;
        ClassLabels = ExtractClassLabels(y);
        NumClasses = ClassLabels.Length;
        TaskType = InferTaskType(y);

        // Initialize random number generator
        _random = Options.RandomState.HasValue
            ? RandomHelper.CreateSeededRandom(Options.RandomState.Value)
            : RandomHelper.CreateSeededRandom(42);

        // Initialize feature importances
        FeatureImportances = new Vector<T>(NumFeatures);

        // Create indices array
        var indices = Enumerable.Range(0, x.Rows).ToList();

        // Build the tree recursively
        _root = BuildTree(x, y, indices, 0);

        // Normalize feature importances
        NormalizeFeatureImportances();
    }

    /// <summary>
    /// Builds the decision tree recursively.
    /// </summary>
    private DecisionNode<T> BuildTree(Matrix<T> x, Vector<T> y, List<int> indices, int depth)
    {
        // Check stopping conditions
        if (ShouldStop(indices, y, depth))
        {
            return CreateLeafNode(y, indices);
        }

        // Find the best split
        var (bestFeature, bestThreshold, bestGain) = FindBestSplit(x, y, indices);

        // If no good split found, create leaf
        if (bestFeature < 0 || NumOps.Compare(bestGain, NumOps.FromDouble(Options.MinImpurityDecrease)) <= 0)
        {
            return CreateLeafNode(y, indices);
        }

        // Split the data
        var (leftIndices, rightIndices) = SplitData(x, indices, bestFeature, bestThreshold);

        // Check if split is valid
        if (leftIndices.Count < Options.MinSamplesLeaf || rightIndices.Count < Options.MinSamplesLeaf)
        {
            return CreateLeafNode(y, indices);
        }

        // Update feature importances
        UpdateFeatureImportances(bestFeature, bestGain, indices.Count);

        // Recursively build children
        var leftChild = BuildTree(x, y, leftIndices, depth + 1);
        var rightChild = BuildTree(x, y, rightIndices, depth + 1);

        return new DecisionNode<T>
        {
            FeatureIndex = bestFeature,
            Threshold = bestThreshold,
            Left = leftChild,
            Right = rightChild,
            IsLeaf = false,
            NumSamples = indices.Count
        };
    }

    /// <summary>
    /// Determines if we should stop splitting.
    /// </summary>
    private bool ShouldStop(List<int> indices, Vector<T> y, int depth)
    {
        // Check max depth
        if (Options.MaxDepth.HasValue && depth >= Options.MaxDepth.Value)
            return true;

        // Check minimum samples
        if (indices.Count < Options.MinSamplesSplit)
            return true;

        // Check if pure node (all same class)
        var firstClass = y[indices[0]];
        bool allSame = true;
        for (int i = 1; i < indices.Count && allSame; i++)
        {
            if (NumOps.Compare(y[indices[i]], firstClass) != 0)
            {
                allSame = false;
            }
        }
        if (allSame)
            return true;

        return false;
    }

    /// <summary>
    /// Creates a leaf node with class probabilities.
    /// </summary>
    private DecisionNode<T> CreateLeafNode(Vector<T> y, List<int> indices)
    {
        var classCounts = new int[NumClasses];
        foreach (var idx in indices)
        {
            int classIdx = GetClassIndex(y[idx]);
            classCounts[classIdx]++;
        }

        var probabilities = new Vector<T>(NumClasses);
        T total = NumOps.FromDouble(indices.Count);
        for (int c = 0; c < NumClasses; c++)
        {
            probabilities[c] = NumOps.Divide(NumOps.FromDouble(classCounts[c]), total);
        }

        // Find predicted class (argmax)
        int predictedClass = 0;
        for (int c = 1; c < NumClasses; c++)
        {
            if (classCounts[c] > classCounts[predictedClass])
            {
                predictedClass = c;
            }
        }

        return new DecisionNode<T>
        {
            IsLeaf = true,
            ClassProbabilities = probabilities,
            PredictedClass = predictedClass,
            NumSamples = indices.Count
        };
    }

    /// <summary>
    /// Finds the best split for the given indices.
    /// </summary>
    private (int feature, T threshold, T gain) FindBestSplit(Matrix<T> x, Vector<T> y, List<int> indices)
    {
        int bestFeature = -1;
        T bestThreshold = NumOps.Zero;
        T bestGain = NumOps.Zero;

        // Determine which features to consider
        var featuresToConsider = GetFeaturesToConsider();

        // Calculate current impurity
        T currentImpurity = CalculateImpurity(y, indices);

        foreach (int feature in featuresToConsider)
        {
            // Get sorted unique values for this feature
            var values = indices
                .Select(i => x[i, feature])
                .Distinct()
                .OrderBy(v => NumOps.ToDouble(v))
                .ToList();

            // Try midpoints between consecutive values as thresholds
            for (int i = 0; i < values.Count - 1; i++)
            {
                T threshold = NumOps.Divide(NumOps.Add(values[i], values[i + 1]), NumOps.FromDouble(2.0));

                // Calculate gain for this split
                var (leftIndices, rightIndices) = SplitData(x, indices, feature, threshold);

                if (leftIndices.Count < Options.MinSamplesLeaf || rightIndices.Count < Options.MinSamplesLeaf)
                    continue;

                T leftImpurity = CalculateImpurity(y, leftIndices);
                T rightImpurity = CalculateImpurity(y, rightIndices);

                T leftWeight = NumOps.Divide(NumOps.FromDouble(leftIndices.Count), NumOps.FromDouble(indices.Count));
                T rightWeight = NumOps.Divide(NumOps.FromDouble(rightIndices.Count), NumOps.FromDouble(indices.Count));

                T weightedImpurity = NumOps.Add(
                    NumOps.Multiply(leftWeight, leftImpurity),
                    NumOps.Multiply(rightWeight, rightImpurity)
                );

                T gain = NumOps.Subtract(currentImpurity, weightedImpurity);

                if (NumOps.Compare(gain, bestGain) > 0)
                {
                    bestGain = gain;
                    bestFeature = feature;
                    bestThreshold = threshold;
                }
            }
        }

        return (bestFeature, bestThreshold, bestGain);
    }

    /// <summary>
    /// Gets the features to consider for splitting based on MaxFeatures setting.
    /// </summary>
    private int[] GetFeaturesToConsider()
    {
        if (!Options.MaxFeatures.HasValue || Options.MaxFeatures.Value >= NumFeatures)
        {
            return Enumerable.Range(0, NumFeatures).ToArray();
        }

        // Randomly select MaxFeatures features
        var allFeatures = Enumerable.Range(0, NumFeatures).ToList();
        var selectedFeatures = new int[Options.MaxFeatures.Value];

        for (int i = 0; i < Options.MaxFeatures.Value; i++)
        {
            int idx = _random!.Next(allFeatures.Count);
            selectedFeatures[i] = allFeatures[idx];
            allFeatures.RemoveAt(idx);
        }

        return selectedFeatures;
    }

    /// <summary>
    /// Calculates impurity for a set of samples based on the configured criterion.
    /// </summary>
    private T CalculateImpurity(Vector<T> y, List<int> indices)
    {
        if (indices.Count == 0)
            return NumOps.Zero;

        var classCounts = new int[NumClasses];
        foreach (var idx in indices)
        {
            int classIdx = GetClassIndex(y[idx]);
            classCounts[classIdx]++;
        }

        return Options.Criterion switch
        {
            ClassificationSplitCriterion.Gini => CalculateGiniImpurity(classCounts, indices.Count),
            ClassificationSplitCriterion.Entropy => CalculateEntropy(classCounts, indices.Count),
            ClassificationSplitCriterion.LogLoss => CalculateEntropy(classCounts, indices.Count), // Same as entropy for splits
            _ => CalculateGiniImpurity(classCounts, indices.Count)
        };
    }

    /// <summary>
    /// Calculates Gini impurity.
    /// </summary>
    private T CalculateGiniImpurity(int[] classCounts, int total)
    {
        T gini = NumOps.One;
        T totalT = NumOps.FromDouble(total);

        for (int c = 0; c < NumClasses; c++)
        {
            T prob = NumOps.Divide(NumOps.FromDouble(classCounts[c]), totalT);
            gini = NumOps.Subtract(gini, NumOps.Multiply(prob, prob));
        }

        return gini;
    }

    /// <summary>
    /// Calculates entropy.
    /// </summary>
    private T CalculateEntropy(int[] classCounts, int total)
    {
        T entropy = NumOps.Zero;
        T totalT = NumOps.FromDouble(total);

        for (int c = 0; c < NumClasses; c++)
        {
            if (classCounts[c] > 0)
            {
                T prob = NumOps.Divide(NumOps.FromDouble(classCounts[c]), totalT);
                entropy = NumOps.Subtract(entropy, NumOps.Multiply(prob, NumOps.Log(prob)));
            }
        }

        return entropy;
    }

    /// <summary>
    /// Splits data based on a feature and threshold.
    /// </summary>
    private (List<int> left, List<int> right) SplitData(Matrix<T> x, List<int> indices, int feature, T threshold)
    {
        var left = new List<int>();
        var right = new List<int>();

        foreach (var idx in indices)
        {
            if (NumOps.Compare(x[idx, feature], threshold) <= 0)
            {
                left.Add(idx);
            }
            else
            {
                right.Add(idx);
            }
        }

        return (left, right);
    }

    /// <summary>
    /// Updates feature importances based on the gain from a split.
    /// </summary>
    private void UpdateFeatureImportances(int feature, T gain, int numSamples)
    {
        T weightedGain = NumOps.Multiply(gain, NumOps.FromDouble(numSamples));
        FeatureImportances![feature] = NumOps.Add(FeatureImportances[feature], weightedGain);
    }

    /// <summary>
    /// Normalizes feature importances to sum to 1.
    /// </summary>
    private void NormalizeFeatureImportances()
    {
        if (FeatureImportances == null)
            return;

        T sum = NumOps.Zero;
        for (int i = 0; i < FeatureImportances.Length; i++)
        {
            sum = NumOps.Add(sum, FeatureImportances[i]);
        }

        if (NumOps.Compare(sum, NumOps.Zero) > 0)
        {
            for (int i = 0; i < FeatureImportances.Length; i++)
            {
                FeatureImportances[i] = NumOps.Divide(FeatureImportances[i], sum);
            }
        }
    }

    /// <summary>
    /// Gets the class index for a label.
    /// </summary>
    private int GetClassIndex(T label)
    {
        if (ClassLabels == null)
        {
            throw new InvalidOperationException("Model must be trained before getting class index.");
        }

        double labelValue = NumOps.ToDouble(label);
        for (int i = 0; i < ClassLabels.Length; i++)
        {
            if (Math.Abs(NumOps.ToDouble(ClassLabels[i]) - labelValue) < 1e-10)
            {
                return i;
            }
        }

        throw new ArgumentException($"Label {label} not found in class labels.");
    }

    /// <inheritdoc/>
    public override Matrix<T> PredictProbabilities(Matrix<T> input)
    {
        if (_root == null)
        {
            throw new InvalidOperationException("Model must be trained before prediction.");
        }

        var probabilities = new Matrix<T>(input.Rows, NumClasses);

        for (int i = 0; i < input.Rows; i++)
        {
            var sample = new Vector<T>(input.Columns);
            for (int j = 0; j < input.Columns; j++)
            {
                sample[j] = input[i, j];
            }

            var leafNode = TraverseTree(sample, _root);

            for (int c = 0; c < NumClasses; c++)
            {
                probabilities[i, c] = leafNode.ClassProbabilities![c];
            }
        }

        return probabilities;
    }

    /// <summary>
    /// Traverses the tree to find the appropriate leaf node.
    /// </summary>
    private DecisionNode<T> TraverseTree(Vector<T> sample, DecisionNode<T> node)
    {
        if (node.IsLeaf)
        {
            return node;
        }

        if (NumOps.Compare(sample[node.FeatureIndex], node.Threshold) <= 0)
        {
            return TraverseTree(sample, node.Left!);
        }
        else
        {
            return TraverseTree(sample, node.Right!);
        }
    }

    /// <summary>
    /// Calculates the actual depth of the tree.
    /// </summary>
    private int CalculateActualDepth(DecisionNode<T>? node)
    {
        if (node == null || node.IsLeaf)
            return 0;

        return 1 + Math.Max(CalculateActualDepth(node.Left), CalculateActualDepth(node.Right));
    }

    /// <summary>
    /// Counts the number of leaf nodes.
    /// </summary>
    private int CountLeaves(DecisionNode<T>? node)
    {
        if (node == null)
            return 0;
        if (node.IsLeaf)
            return 1;
        return CountLeaves(node.Left) + CountLeaves(node.Right);
    }

    /// <summary>
    /// Counts the total number of nodes.
    /// </summary>
    private int CountNodes(DecisionNode<T>? node)
    {
        if (node == null)
            return 0;
        return 1 + CountNodes(node.Left) + CountNodes(node.Right);
    }

    /// <inheritdoc/>
    protected override IFullModel<T, Matrix<T>, Vector<T>> CreateNewInstance()
    {
        return new DecisionTreeClassifier<T>(new DecisionTreeClassifierOptions<T>
        {
            MaxDepth = Options.MaxDepth,
            MinSamplesSplit = Options.MinSamplesSplit,
            MinSamplesLeaf = Options.MinSamplesLeaf,
            MaxFeatures = Options.MaxFeatures,
            Criterion = Options.Criterion,
            RandomState = Options.RandomState,
            MinImpurityDecrease = Options.MinImpurityDecrease
        });
    }

    /// <inheritdoc/>
    public override IFullModel<T, Matrix<T>, Vector<T>> Clone()
    {
        var clone = new DecisionTreeClassifier<T>(new DecisionTreeClassifierOptions<T>
        {
            MaxDepth = Options.MaxDepth,
            MinSamplesSplit = Options.MinSamplesSplit,
            MinSamplesLeaf = Options.MinSamplesLeaf,
            MaxFeatures = Options.MaxFeatures,
            Criterion = Options.Criterion,
            RandomState = Options.RandomState,
            MinImpurityDecrease = Options.MinImpurityDecrease
        });

        clone.NumFeatures = NumFeatures;
        clone.NumClasses = NumClasses;
        clone.TaskType = TaskType;

        if (ClassLabels != null)
        {
            clone.ClassLabels = new Vector<T>(ClassLabels.Length);
            for (int i = 0; i < ClassLabels.Length; i++)
            {
                clone.ClassLabels[i] = ClassLabels[i];
            }
        }

        if (FeatureImportances != null)
        {
            clone.FeatureImportances = new Vector<T>(FeatureImportances.Length);
            for (int i = 0; i < FeatureImportances.Length; i++)
            {
                clone.FeatureImportances[i] = FeatureImportances[i];
            }
        }

        if (_root != null)
        {
            clone._root = CloneNode(_root);
        }

        return clone;
    }

    /// <summary>
    /// Deep clones a decision tree node.
    /// </summary>
    private DecisionNode<T> CloneNode(DecisionNode<T> node)
    {
        var cloned = new DecisionNode<T>
        {
            IsLeaf = node.IsLeaf,
            FeatureIndex = node.FeatureIndex,
            Threshold = node.Threshold,
            PredictedClass = node.PredictedClass,
            NumSamples = node.NumSamples
        };

        if (node.ClassProbabilities != null)
        {
            cloned.ClassProbabilities = new Vector<T>(node.ClassProbabilities.Length);
            for (int i = 0; i < node.ClassProbabilities.Length; i++)
            {
                cloned.ClassProbabilities[i] = node.ClassProbabilities[i];
            }
        }

        if (node.Left != null)
        {
            cloned.Left = CloneNode(node.Left);
        }

        if (node.Right != null)
        {
            cloned.Right = CloneNode(node.Right);
        }

        return cloned;
    }

    /// <inheritdoc/>
    public override Vector<T> GetParameters()
    {
        // Decision trees don't have traditional numeric parameters
        // Return feature importances as a representation
        return FeatureImportances ?? new Vector<T>(0);
    }

    /// <inheritdoc/>
    public override IFullModel<T, Matrix<T>, Vector<T>> WithParameters(Vector<T> parameters)
    {
        var newModel = (DecisionTreeClassifier<T>)Clone();
        newModel.SetParameters(parameters);
        return newModel;
    }

    /// <inheritdoc/>
    public override void SetParameters(Vector<T> parameters)
    {
        // Decision trees don't use traditional parameters
        // This is a no-op for compatibility
        if (parameters.Length == NumFeatures && FeatureImportances != null)
        {
            for (int i = 0; i < parameters.Length; i++)
            {
                FeatureImportances[i] = parameters[i];
            }
        }
    }

    /// <inheritdoc/>
    public override Vector<T> ComputeGradients(Matrix<T> input, Vector<T> target, ILossFunction<T>? lossFunction = null)
    {
        // Decision trees don't typically use gradient-based optimization
        // Return zero gradients for compatibility
        return new Vector<T>(NumFeatures);
    }

    /// <inheritdoc/>
    public override void ApplyGradients(Vector<T> gradients, T learningRate)
    {
        // Decision trees don't typically use gradient-based optimization
        // This is a no-op for compatibility
    }

    /// <inheritdoc/>
    public override ModelMetadata<T> GetModelMetadata()
    {
        var metadata = base.GetModelMetadata();
        metadata.AdditionalInfo["MaxDepth"] = Options.MaxDepth?.ToString() ?? "unlimited";
        metadata.AdditionalInfo["Criterion"] = Options.Criterion.ToString();
        metadata.AdditionalInfo["LeafCount"] = LeafCount;
        metadata.AdditionalInfo["NodeCount"] = NodeCount;
        return metadata;
    }
}

/// <summary>
/// Represents a node in the decision tree.
/// </summary>
/// <typeparam name="T">The numeric type.</typeparam>
internal class DecisionNode<T>
{
    /// <summary>
    /// Whether this is a leaf node.
    /// </summary>
    public bool IsLeaf { get; set; }

    /// <summary>
    /// The feature index used for splitting (internal nodes only).
    /// </summary>
    public int FeatureIndex { get; set; }

    /// <summary>
    /// The threshold value for splitting (internal nodes only).
    /// </summary>
    public T Threshold { get; set; } = default!;

    /// <summary>
    /// The left child node (values <= threshold).
    /// </summary>
    public DecisionNode<T>? Left { get; set; }

    /// <summary>
    /// The right child node (values > threshold).
    /// </summary>
    public DecisionNode<T>? Right { get; set; }

    /// <summary>
    /// Class probabilities (leaf nodes only).
    /// </summary>
    public Vector<T>? ClassProbabilities { get; set; }

    /// <summary>
    /// Predicted class (leaf nodes only).
    /// </summary>
    public int PredictedClass { get; set; }

    /// <summary>
    /// Number of samples that reached this node during training.
    /// </summary>
    public int NumSamples { get; set; }
}
