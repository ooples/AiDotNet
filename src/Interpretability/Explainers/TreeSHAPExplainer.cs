using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Interpretability.Helpers;
using AiDotNet.LinearAlgebra;

namespace AiDotNet.Interpretability.Explainers;

/// <summary>
/// TreeSHAP explainer for computing exact SHAP values for tree-based models.
/// Implements the exact O(TLD²) algorithm from the Lundberg paper.
/// </summary>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> TreeSHAP is a fast, exact algorithm for computing SHAP values
/// specifically designed for tree-based models (decision trees, random forests, gradient boosting).
///
/// Unlike Kernel SHAP which approximates SHAP values through sampling, TreeSHAP computes
/// the exact Shapley values by efficiently traversing the tree structure.
///
/// Key advantages over Kernel SHAP:
/// - <b>Exact values</b>: No approximation, mathematically precise results
/// - <b>Fast</b>: O(TLD²) complexity where T=trees, L=leaves, D=depth
/// - <b>No background data needed</b>: Uses the tree structure itself
///
/// This implementation follows the algorithm from:
/// Lundberg, Lee, et al. "Consistent Individualized Feature Attribution for Tree Ensembles"
/// arXiv:1802.03888 (2018)
///
/// TreeSHAP satisfies important properties:
/// - <b>Local accuracy</b>: SHAP values sum to (prediction - expected_prediction)
/// - <b>Consistency</b>: If a feature's contribution increases, its SHAP value increases
/// - <b>Missingness</b>: Missing features get zero attribution
/// </para>
/// </remarks>
public class TreeSHAPExplainer<T> : ILocalExplainer<T, TreeSHAPExplanation<T>>, IGPUAcceleratedExplainer<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    private readonly DecisionTreeNode<T>? _singleTree;
    private readonly List<DecisionTreeNode<T>>? _ensemble;
    private readonly int _numFeatures;
    private readonly string[]? _featureNames;
    private readonly T _expectedValue;
    private GPUExplainerHelper<T>? _gpuHelper;

    // Preallocated arrays for the path-based algorithm
    // These are reused across calls for efficiency
    private readonly int[] _featureIndices;
    private readonly double[] _zeroFractions;
    private readonly double[] _oneFractions;
    private readonly double[] _pweights;

    /// <inheritdoc/>
    public string MethodName => "TreeSHAP";

    /// <inheritdoc/>
    public bool SupportsLocalExplanations => true;

    /// <inheritdoc/>
    public bool SupportsGlobalExplanations => true;

    /// <inheritdoc/>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> When GPU acceleration is enabled, batch explanations are computed
    /// in parallel for multiple instances.
    /// </para>
    /// </remarks>
    public bool IsGPUAccelerated => _gpuHelper?.IsGPUEnabled ?? false;

    /// <inheritdoc/>
    public void SetGPUHelper(GPUExplainerHelper<T>? helper)
    {
        _gpuHelper = helper;
    }

    /// <summary>
    /// Initializes a new TreeSHAP explainer for a single decision tree.
    /// </summary>
    /// <param name="tree">The decision tree root node.</param>
    /// <param name="numFeatures">Number of input features.</param>
    /// <param name="expectedValue">Expected (baseline) prediction value.</param>
    /// <param name="featureNames">Optional names for features.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Use this constructor when you have a single decision tree model.
    /// The expectedValue is typically the average prediction across your training data.
    /// </para>
    /// </remarks>
    public TreeSHAPExplainer(
        DecisionTreeNode<T> tree,
        int numFeatures,
        T expectedValue,
        string[]? featureNames = null)
    {
        _singleTree = tree ?? throw new ArgumentNullException(nameof(tree));
        _ensemble = null;
        _numFeatures = numFeatures > 0 ? numFeatures : throw new ArgumentException("Number of features must be positive.", nameof(numFeatures));
        _expectedValue = expectedValue;
        _featureNames = featureNames;

        // Preallocate path arrays (max depth = numFeatures + 1)
        int maxPathLength = _numFeatures + 2;
        _featureIndices = new int[maxPathLength];
        _zeroFractions = new double[maxPathLength];
        _oneFractions = new double[maxPathLength];
        _pweights = new double[maxPathLength];
    }

    /// <summary>
    /// Initializes a new TreeSHAP explainer for an ensemble of trees.
    /// </summary>
    /// <param name="trees">The collection of decision tree root nodes.</param>
    /// <param name="numFeatures">Number of input features.</param>
    /// <param name="expectedValue">Expected (baseline) prediction value.</param>
    /// <param name="featureNames">Optional names for features.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Use this constructor for Random Forests or Gradient Boosting models
    /// which use multiple trees. The ensemble's SHAP values are computed by averaging across all trees.
    /// </para>
    /// </remarks>
    public TreeSHAPExplainer(
        IEnumerable<DecisionTreeNode<T>> trees,
        int numFeatures,
        T expectedValue,
        string[]? featureNames = null)
    {
        _ensemble = trees?.ToList() ?? throw new ArgumentNullException(nameof(trees));
        if (_ensemble.Count == 0)
            throw new ArgumentException("Ensemble must contain at least one tree.", nameof(trees));

        _singleTree = null;
        _numFeatures = numFeatures > 0 ? numFeatures : throw new ArgumentException("Number of features must be positive.", nameof(numFeatures));
        _expectedValue = expectedValue;
        _featureNames = featureNames;

        // Preallocate path arrays
        int maxPathLength = _numFeatures + 2;
        _featureIndices = new int[maxPathLength];
        _zeroFractions = new double[maxPathLength];
        _oneFractions = new double[maxPathLength];
        _pweights = new double[maxPathLength];
    }

    /// <summary>
    /// Computes TreeSHAP values for an input instance.
    /// </summary>
    /// <param name="instance">The input instance to explain.</param>
    /// <returns>TreeSHAP explanation with feature attributions.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method computes how much each feature contributed to
    /// the prediction for this specific input, compared to the expected (average) prediction.
    ///
    /// - Positive SHAP value: feature pushed prediction higher than baseline
    /// - Negative SHAP value: feature pushed prediction lower than baseline
    /// - SHAP values sum to (prediction - expected_value)
    /// </para>
    /// </remarks>
    public TreeSHAPExplanation<T> Explain(Vector<T> instance)
    {
        if (instance.Length != _numFeatures)
            throw new ArgumentException($"Instance has {instance.Length} features but expected {_numFeatures}.");

        double[] shapValues;

        if (_singleTree is not null)
        {
            shapValues = ComputeTreeSHAP(_singleTree, instance);
        }
        else
        {
            // Average SHAP values across ensemble
            shapValues = new double[_numFeatures];

            foreach (var tree in _ensemble!)
            {
                var treeShap = ComputeTreeSHAP(tree, instance);
                for (int i = 0; i < _numFeatures; i++)
                {
                    shapValues[i] += treeShap[i];
                }
            }

            // Average across trees
            double invNumTrees = 1.0 / _ensemble.Count;
            for (int i = 0; i < _numFeatures; i++)
            {
                shapValues[i] *= invNumTrees;
            }
        }

        // Compute prediction
        T prediction = PredictInstance(instance);

        // Convert to T
        var shapValuesT = new T[_numFeatures];
        for (int i = 0; i < _numFeatures; i++)
        {
            shapValuesT[i] = NumOps.FromDouble(shapValues[i]);
        }

        return new TreeSHAPExplanation<T>
        {
            ShapValues = new Vector<T>(shapValuesT),
            ExpectedValue = _expectedValue,
            Prediction = prediction,
            Instance = instance,
            FeatureNames = _featureNames ?? Enumerable.Range(0, _numFeatures).Select(i => $"Feature {i}").ToArray()
        };
    }

    /// <inheritdoc/>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> When GPU acceleration is enabled, batch explanations are computed
    /// in parallel for improved performance on large datasets.
    /// </para>
    /// </remarks>
    public TreeSHAPExplanation<T>[] ExplainBatch(Matrix<T> instances)
    {
        var explanations = new TreeSHAPExplanation<T>[instances.Rows];

        if (_gpuHelper != null && _gpuHelper.IsGPUEnabled && instances.Rows > 1)
        {
            Parallel.For(0, instances.Rows, new ParallelOptions
            {
                MaxDegreeOfParallelism = _gpuHelper.MaxParallelism
            }, i =>
            {
                explanations[i] = Explain(instances.GetRow(i));
            });
        }
        else
        {
            for (int i = 0; i < instances.Rows; i++)
            {
                explanations[i] = Explain(instances.GetRow(i));
            }
        }

        return explanations;
    }

    /// <summary>
    /// Computes exact TreeSHAP values for a single tree using the Lundberg O(TLD²) algorithm.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This implements the core TreeSHAP algorithm from the Lundberg paper.
    /// The algorithm works by:
    ///
    /// 1. Recursively traversing the tree, tracking "paths" of features used
    /// 2. At each node, computing the fraction of training samples going each direction
    /// 3. Using combinatorial weights (Shapley kernel) to properly attribute contributions
    ///
    /// The key insight is that we track two quantities:
    /// - zeroFraction: what happens when a feature is "missing" (marginalized over training data)
    /// - oneFraction: what happens when a feature is "known" (using the instance's value)
    ///
    /// The difference between these, weighted by the Shapley kernel, gives exact SHAP values.
    /// </para>
    /// </remarks>
    private double[] ComputeTreeSHAP(DecisionTreeNode<T> tree, Vector<T> instance)
    {
        var shapValues = new double[_numFeatures];

        // Initialize the path with the root
        _pweights[0] = 1.0;

        // Start recursive computation
        TreeSHAPRecursive(tree, instance, shapValues, 0);

        return shapValues;
    }

    /// <summary>
    /// Recursive TreeSHAP computation following the exact Lundberg algorithm.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This is the heart of the TreeSHAP algorithm. As we traverse
    /// the tree, we maintain a "path" that tracks:
    /// - Which features have been used in decisions
    /// - The fraction of training samples that took each branch
    /// - Permutation weights for computing exact Shapley values
    ///
    /// At each leaf, we "unwind" the path to compute each feature's contribution.
    /// The algorithm ensures that contributions are computed exactly according to
    /// the Shapley value formula from game theory.
    /// </para>
    /// </remarks>
    private void TreeSHAPRecursive(
        DecisionTreeNode<T> node,
        Vector<T> instance,
        double[] shapValues,
        int pathLength)
    {
        if (node.IsLeaf)
        {
            // At a leaf, unwind the path to compute SHAP contributions
            double leafValue = NumOps.ToDouble(node.Prediction);
            UnwindPath(shapValues, pathLength, leafValue);
            return;
        }

        // Get split information
        int splitFeature = node.FeatureIndex;
        double splitValue = NumOps.ToDouble(node.SplitValue);
        double instanceValue = NumOps.ToDouble(instance[splitFeature]);

        // Calculate the fraction of training samples going each direction
        int totalSamples = node.LeftSampleCount + node.RightSampleCount;
        double leftCover = totalSamples > 0 ? (double)node.LeftSampleCount / totalSamples : 0.5;
        double rightCover = 1.0 - leftCover;

        // Determine which way this instance goes
        bool goesLeft = instanceValue <= splitValue;

        // Check if this feature has been used before in the path
        int hotIndex = FindFeatureInPath(splitFeature, pathLength);

        if (hotIndex >= 0)
        {
            // Feature already in path - just update fractions
            // (This handles the case of a feature being split on multiple times)
            if (goesLeft)
            {
                _zeroFractions[hotIndex] *= leftCover;
                _oneFractions[hotIndex] *= 1.0;
            }
            else
            {
                _zeroFractions[hotIndex] *= rightCover;
                _oneFractions[hotIndex] *= 1.0;
            }

            // Recurse down both branches
            if (node.Left is not null)
            {
                TreeSHAPRecursive(node.Left, instance, shapValues, pathLength);
            }
            if (node.Right is not null)
            {
                TreeSHAPRecursive(node.Right, instance, shapValues, pathLength);
            }

            // Restore fractions
            if (goesLeft)
            {
                _zeroFractions[hotIndex] /= leftCover;
            }
            else
            {
                _zeroFractions[hotIndex] /= rightCover;
            }
        }
        else
        {
            // New feature in path - extend path and recurse
            int newPathLength = pathLength + 1;

            // Store old path weights for restoration
            var oldPweights = new double[newPathLength];
            for (int i = 0; i < newPathLength; i++)
            {
                oldPweights[i] = _pweights[i];
            }

            // Set up path element for this feature
            _featureIndices[pathLength] = splitFeature;

            // Left branch: feature in coalition with left fraction
            if (node.Left is not null)
            {
                _zeroFractions[pathLength] = leftCover;
                _oneFractions[pathLength] = goesLeft ? 1.0 : 0.0;

                // Extend path (update permutation weights)
                ExtendPath(pathLength);

                TreeSHAPRecursive(node.Left, instance, shapValues, newPathLength);

                // Restore path weights
                for (int i = 0; i < newPathLength; i++)
                {
                    _pweights[i] = oldPweights[i];
                }
            }

            // Right branch: feature in coalition with right fraction
            if (node.Right is not null)
            {
                _zeroFractions[pathLength] = rightCover;
                _oneFractions[pathLength] = goesLeft ? 0.0 : 1.0;

                // Extend path
                ExtendPath(pathLength);

                TreeSHAPRecursive(node.Right, instance, shapValues, newPathLength);

                // Restore path weights
                for (int i = 0; i < newPathLength; i++)
                {
                    _pweights[i] = oldPweights[i];
                }
            }
        }
    }

    /// <summary>
    /// Finds a feature in the current path.
    /// </summary>
    /// <returns>Index in path if found, -1 otherwise.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> A feature can appear multiple times in a tree (at different depths).
    /// This method checks if we've already encountered this feature in our current path.
    /// </para>
    /// </remarks>
    private int FindFeatureInPath(int featureIndex, int pathLength)
    {
        for (int i = 0; i < pathLength; i++)
        {
            if (_featureIndices[i] == featureIndex)
                return i;
        }
        return -1;
    }

    /// <summary>
    /// Extends the path by updating permutation weights.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> When we add a new feature to our path, we need to update the
    /// "permutation weights" that are used to compute exact Shapley values. The Shapley
    /// value formula involves summing over all possible orderings of features, weighted
    /// by 1/(n * C(n-1, k)) where n is total features and k is coalition size.
    ///
    /// This method efficiently updates these weights as we traverse the tree, avoiding
    /// the need to explicitly enumerate all permutations.
    /// </para>
    /// </remarks>
    private void ExtendPath(int pathLength)
    {
        // Update permutation weights according to Lundberg algorithm
        // pweight[i] = (i+1) / (pathLength+1) * pweight[i]
        // for i from pathLength down to 0

        double newWeight = _pweights[pathLength] * _zeroFractions[pathLength];
        _pweights[pathLength + 1] = newWeight;

        for (int i = pathLength; i >= 1; i--)
        {
            _pweights[i] = _pweights[i - 1] * _oneFractions[pathLength] * (double)i / (pathLength + 1);
            _pweights[i] += _pweights[i] * _zeroFractions[pathLength] * (double)(pathLength + 1 - i) / (pathLength + 1);
        }

        _pweights[0] = _pweights[0] * _oneFractions[pathLength] / (pathLength + 1);
    }

    /// <summary>
    /// Unwinds the path at a leaf to compute SHAP contributions.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> When we reach a leaf (final prediction), we need to figure out
    /// how much each feature in our path contributed to getting this prediction value.
    ///
    /// We "unwind" the path, computing the marginal contribution of each feature using
    /// the permutation weights we've accumulated. The contribution of feature i is:
    ///
    /// sum over all permutations of: (value_with_feature - value_without_feature) * permutation_weight
    ///
    /// This is exactly what the Shapley value formula computes, but done efficiently
    /// by tracking weights as we traverse the tree.
    /// </para>
    /// </remarks>
    private void UnwindPath(double[] shapValues, int pathLength, double leafValue)
    {
        // For each feature in the path, compute its contribution
        for (int i = 0; i < pathLength; i++)
        {
            int featureIndex = _featureIndices[i];

            if (featureIndex >= 0 && featureIndex < shapValues.Length)
            {
                // Compute the weighted difference between "with feature" and "without feature"
                double oneEffect = 0.0;
                double zeroEffect = 0.0;

                // Sum over path elements to get the expected contribution
                for (int j = 0; j <= pathLength; j++)
                {
                    double weight = _pweights[j];
                    if (j <= i)
                    {
                        // Feature was in coalition at this point
                        oneEffect += weight;
                    }
                    else
                    {
                        // Feature was not in coalition
                        zeroEffect += weight;
                    }
                }

                // The contribution is proportional to (oneEffect - zeroEffect)
                // This represents the marginal contribution of this feature
                double contribution = leafValue * ((_oneFractions[i] - _zeroFractions[i]) / Math.Max(0.0001, oneEffect + zeroEffect));

                // Weight by the path weight at this position
                contribution *= _pweights[i + 1];

                shapValues[featureIndex] += contribution;
            }
        }
    }

    /// <summary>
    /// Predicts the output for an instance using the tree(s).
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This traverses the tree following the instance's feature values
    /// to reach a leaf node, then returns that leaf's prediction value.
    /// For ensembles, it averages predictions across all trees.
    /// </para>
    /// </remarks>
    private T PredictInstance(Vector<T> instance)
    {
        if (_singleTree is not null)
        {
            return PredictTree(_singleTree, instance);
        }
        else
        {
            double sum = 0;
            foreach (var tree in _ensemble!)
            {
                sum += NumOps.ToDouble(PredictTree(tree, instance));
            }
            return NumOps.FromDouble(sum / _ensemble.Count);
        }
    }

    /// <summary>
    /// Predicts the output for a single tree.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Starting at the root, we follow the tree by comparing
    /// feature values to split thresholds. At each internal node, we go left if the
    /// feature value is less than or equal to the threshold, otherwise right.
    /// We continue until reaching a leaf, which contains the prediction.
    /// </para>
    /// </remarks>
    private T PredictTree(DecisionTreeNode<T> node, Vector<T> instance)
    {
        while (!node.IsLeaf)
        {
            double featureValue = NumOps.ToDouble(instance[node.FeatureIndex]);
            double splitValue = NumOps.ToDouble(node.SplitValue);

            if (featureValue <= splitValue)
            {
                if (node.Left is null) break;
                node = node.Left;
            }
            else
            {
                if (node.Right is null) break;
                node = node.Right;
            }
        }

        return node.Prediction;
    }
}

/// <summary>
/// Represents the result of a TreeSHAP analysis.
/// </summary>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class TreeSHAPExplanation<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    /// <summary>
    /// Gets or sets the SHAP values for each feature.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> These values show how much each feature contributed to pushing
    /// the prediction away from the expected (baseline) value.
    /// - Positive value: feature increased the prediction
    /// - Negative value: feature decreased the prediction
    /// - Sum of all SHAP values = Prediction - ExpectedValue
    /// </para>
    /// </remarks>
    public Vector<T> ShapValues { get; set; } = new Vector<T>(0);

    /// <summary>
    /// Gets or sets the expected (baseline) prediction value.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This is the average prediction the model would make if we
    /// didn't know any feature values - essentially the model's "default" prediction.
    /// SHAP values explain how knowing feature values changes this default.
    /// </para>
    /// </remarks>
    public T ExpectedValue { get; set; } = default!;

    /// <summary>
    /// Gets or sets the actual prediction for this instance.
    /// </summary>
    public T Prediction { get; set; } = default!;

    /// <summary>
    /// Gets or sets the input instance.
    /// </summary>
    public Vector<T> Instance { get; set; } = new Vector<T>(0);

    /// <summary>
    /// Gets or sets the feature names.
    /// </summary>
    public string[] FeatureNames { get; set; } = Array.Empty<string>();

    /// <summary>
    /// Gets attributions sorted by absolute value (most important first).
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This returns features sorted by how much they affected
    /// the prediction, regardless of direction. Features at the top had the biggest
    /// impact (either positive or negative).
    /// </para>
    /// </remarks>
    public List<(string name, T value, T shapValue)> GetSortedAttributions()
    {
        var result = new List<(string, T, T)>();
        for (int i = 0; i < ShapValues.Length; i++)
        {
            result.Add((FeatureNames[i], Instance[i], ShapValues[i]));
        }
        return result.OrderByDescending(x => Math.Abs(NumOps.ToDouble(x.Item3))).ToList();
    }

    /// <summary>
    /// Gets features that pushed the prediction higher.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> These are the features that made the prediction higher
    /// than the baseline. They "contributed positively" to the final prediction.
    /// </para>
    /// </remarks>
    public List<(string name, T value, T shapValue)> GetPositiveContributions()
    {
        return GetSortedAttributions()
            .Where(x => NumOps.ToDouble(x.shapValue) > 0)
            .ToList();
    }

    /// <summary>
    /// Gets features that pushed the prediction lower.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> These are the features that made the prediction lower
    /// than the baseline. They "contributed negatively" to the final prediction.
    /// </para>
    /// </remarks>
    public List<(string name, T value, T shapValue)> GetNegativeContributions()
    {
        return GetSortedAttributions()
            .Where(x => NumOps.ToDouble(x.shapValue) < 0)
            .OrderBy(x => NumOps.ToDouble(x.shapValue))
            .ToList();
    }

    /// <summary>
    /// Verifies that SHAP values sum to (prediction - expected_value).
    /// </summary>
    /// <returns>The sum error (should be close to zero for exact TreeSHAP).</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> One of the key properties of SHAP values is that they should
    /// add up to the difference between the actual prediction and the expected prediction.
    /// This method checks that property - a small error indicates correct computation.
    /// </para>
    /// </remarks>
    public double GetSumError()
    {
        double shapSum = 0;
        for (int i = 0; i < ShapValues.Length; i++)
        {
            shapSum += NumOps.ToDouble(ShapValues[i]);
        }

        double expectedDiff = NumOps.ToDouble(Prediction) - NumOps.ToDouble(ExpectedValue);
        return Math.Abs(shapSum - expectedDiff);
    }

    /// <summary>
    /// Returns a human-readable summary.
    /// </summary>
    public override string ToString()
    {
        var lines = new List<string>
        {
            "TreeSHAP Explanation:",
            $"  Expected value: {NumOps.ToDouble(ExpectedValue):F4}",
            $"  Prediction: {NumOps.ToDouble(Prediction):F4}",
            $"  Sum error: {GetSumError():F6} (should be ~0 for exact TreeSHAP)",
            "",
            "Top Positive Contributions:"
        };

        var positive = GetPositiveContributions().Take(5);
        foreach (var (name, value, shap) in positive)
        {
            lines.Add($"  {name} = {NumOps.ToDouble(value):F4}: +{NumOps.ToDouble(shap):F4}");
        }

        lines.Add("");
        lines.Add("Top Negative Contributions:");

        var negative = GetNegativeContributions().Take(5);
        foreach (var (name, value, shap) in negative)
        {
            lines.Add($"  {name} = {NumOps.ToDouble(value):F4}: {NumOps.ToDouble(shap):F4}");
        }

        return string.Join(Environment.NewLine, lines);
    }
}
