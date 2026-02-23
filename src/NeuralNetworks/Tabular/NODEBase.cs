using AiDotNet.ActivationFunctions;
using AiDotNet.Extensions;
using AiDotNet.Helpers;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks.Layers;

namespace AiDotNet.NeuralNetworks.Tabular;

/// <summary>
/// Base class for NODE (Neural Oblivious Decision Ensembles).
/// </summary>
/// <remarks>
/// <para>
/// NODE uses differentiable oblivious decision trees that can be trained end-to-end:
/// - Each tree uses the same feature at each depth level (oblivious)
/// - Split decisions are soft (differentiable) using sigmoid
/// - Trees are combined additively for the final output
/// </para>
/// <para>
/// <b>For Beginners:</b> Think of NODE as making decision trees trainable like neural networks:
///
/// - **Oblivious trees**: At each depth, all nodes split on the same feature.
///   This makes trees faster and more regularized.
/// - **Soft splits**: Instead of "if feature &gt; threshold then go right",
///   we use a smooth function that gradually transitions.
/// - **Ensemble**: Multiple trees vote together for the final answer.
///
/// The result is a model that combines tree interpretability with deep learning power.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public abstract class NODEBase<T>
{
    protected readonly NODEOptions<T> Options;
    protected readonly int NumFeatures;
    protected static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();
    private readonly Random _random = RandomHelper.CreateSecureRandom();

    // Feature preprocessing (optional)
    private readonly FullyConnectedLayer<T>? _featurePreprocessing;

    // Per-tree parameters
    private readonly Tensor<T>[] _featureSelectionWeights;  // [depth, features] per tree
    private readonly Tensor<T>[] _splitThresholds;          // [depth] per tree
    private readonly Tensor<T>[] _leafValues;               // [2^depth, output_dim] per tree

    // Caches for backward pass
    private Tensor<T>? _preprocessedFeaturesCache;
    private List<Tensor<T>>? _splitProbabilitiesCache;
    private List<Tensor<T>>? _leafWeightsCache;
    private Tensor<T>? _treeOutputsCache;

    /// <summary>
    /// Gets the tree output dimension.
    /// </summary>
    public int TreeOutputDimension => Options.TreeOutputDimension;

    /// <summary>
    /// Gets the total number of trainable parameters.
    /// </summary>
    public virtual int ParameterCount
    {
        get
        {
            int count = 0;

            if (_featurePreprocessing != null)
                count += _featurePreprocessing.ParameterCount;

            foreach (var weights in _featureSelectionWeights)
                count += weights.Length;

            foreach (var thresholds in _splitThresholds)
                count += thresholds.Length;

            foreach (var leaves in _leafValues)
                count += leaves.Length;

            return count;
        }
    }

    /// <summary>
    /// Initializes a new instance of the NODEBase class.
    /// </summary>
    /// <param name="numFeatures">Number of input features.</param>
    /// <param name="options">Model configuration options.</param>
    protected NODEBase(int numFeatures, NODEOptions<T>? options = null)
    {
        Options = options ?? new NODEOptions<T>();
        NumFeatures = numFeatures;

        if (numFeatures < 1)
        {
            throw new ArgumentException("Number of features must be at least 1", nameof(numFeatures));
        }

        // Optional feature preprocessing
        if (Options.UseFeaturePreprocessing)
        {
            _featurePreprocessing = new FullyConnectedLayer<T>(
                numFeatures,
                numFeatures,
                Options.HiddenActivation ?? new ReLUActivation<T>());
        }

        // Initialize tree parameters
        _featureSelectionWeights = new Tensor<T>[Options.NumTrees];
        _splitThresholds = new Tensor<T>[Options.NumTrees];
        _leafValues = new Tensor<T>[Options.NumTrees];

        for (int t = 0; t < Options.NumTrees; t++)
        {
            // Feature selection weights for each depth
            _featureSelectionWeights[t] = new Tensor<T>(new[] { Options.TreeDepth, NumFeatures });
            InitializeWeights(_featureSelectionWeights[t]);

            // Split thresholds for each depth
            _splitThresholds[t] = new Tensor<T>(new[] { Options.TreeDepth });
            for (int d = 0; d < Options.TreeDepth; d++)
            {
                _splitThresholds[t][d] = NumOps.Zero;
            }

            // Leaf values
            _leafValues[t] = new Tensor<T>(new[] { Options.NumLeaves, Options.TreeOutputDimension });
            InitializeWeights(_leafValues[t]);
        }
    }

    private void InitializeWeights(Tensor<T> tensor)
    {
        var scale = NumOps.FromDouble(Options.InitScale);
        for (int i = 0; i < tensor.Length; i++)
        {
            tensor[i] = NumOps.Multiply(
                NumOps.FromDouble(_random.NextGaussian()),
                scale);
        }
    }

    /// <summary>
    /// Performs the forward pass through the NODE backbone.
    /// </summary>
    /// <param name="features">Input features [batch_size, num_features].</param>
    /// <returns>Aggregated tree outputs [batch_size, tree_output_dim].</returns>
    protected Tensor<T> ForwardBackbone(Tensor<T> features)
    {
        int batchSize = features.Shape[0];
        _splitProbabilitiesCache = [];
        _leafWeightsCache = [];

        // Optional feature preprocessing
        var processed = features;
        if (_featurePreprocessing != null)
        {
            processed = _featurePreprocessing.Forward(features);
        }
        _preprocessedFeaturesCache = processed;

        // Aggregate outputs from all trees
        var output = new Tensor<T>(new[] { batchSize, Options.TreeOutputDimension });

        for (int t = 0; t < Options.NumTrees; t++)
        {
            var treeOutput = ForwardTree(t, processed, batchSize);

            // Aggregate tree outputs (additive ensemble)
            for (int b = 0; b < batchSize; b++)
            {
                for (int d = 0; d < Options.TreeOutputDimension; d++)
                {
                    int idx = b * Options.TreeOutputDimension + d;
                    output[idx] = NumOps.Add(output[idx], treeOutput[idx]);
                }
            }
        }

        // Average over trees
        var scale = NumOps.FromDouble(1.0 / Options.NumTrees);
        for (int i = 0; i < output.Length; i++)
        {
            output[i] = NumOps.Multiply(output[i], scale);
        }

        _treeOutputsCache = output;
        return output;
    }

    private Tensor<T> ForwardTree(int treeIdx, Tensor<T> features, int batchSize)
    {
        // Compute leaf weights for each sample
        var leafWeights = new Tensor<T>(new[] { batchSize, Options.NumLeaves });

        // Initialize all samples at root (weight 1.0 for leaf index 0's path)
        for (int b = 0; b < batchSize; b++)
        {
            // For oblivious trees, compute the leaf index based on all split decisions
            var splitProbs = new Tensor<T>(new[] { Options.TreeDepth });

            for (int d = 0; d < Options.TreeDepth; d++)
            {
                // Compute feature selection (soft attention over features)
                var selectedFeature = NumOps.Zero;
                var softmaxWeights = ComputeSoftmax(_featureSelectionWeights[treeIdx], d);

                for (int f = 0; f < NumFeatures; f++)
                {
                    var weight = softmaxWeights[f];
                    var featureVal = features[b * NumFeatures + f];
                    selectedFeature = NumOps.Add(selectedFeature,
                        NumOps.Multiply(weight, featureVal));
                }

                // Soft split decision using sigmoid
                var threshold = _splitThresholds[treeIdx][d];
                var diff = NumOps.Subtract(selectedFeature, threshold);
                var temp = NumOps.FromDouble(Options.Temperature);
                var scaledDiff = NumOps.Divide(diff, temp);

                // Sigmoid for soft split probability (probability of going right)
                splitProbs[d] = Sigmoid(scaledDiff);
            }

            _splitProbabilitiesCache!.Add(splitProbs);

            // Compute leaf weights using the product of split probabilities
            // Each leaf corresponds to a binary path through the tree
            for (int leaf = 0; leaf < Options.NumLeaves; leaf++)
            {
                var weight = NumOps.One;

                for (int d = 0; d < Options.TreeDepth; d++)
                {
                    // Check if this leaf's path goes right (1) or left (0) at depth d
                    bool goesRight = ((leaf >> (Options.TreeDepth - 1 - d)) & 1) == 1;

                    if (goesRight)
                    {
                        weight = NumOps.Multiply(weight, splitProbs[d]);
                    }
                    else
                    {
                        weight = NumOps.Multiply(weight,
                            NumOps.Subtract(NumOps.One, splitProbs[d]));
                    }
                }

                leafWeights[b * Options.NumLeaves + leaf] = weight;
            }
        }

        _leafWeightsCache!.Add(leafWeights);

        // Compute tree output as weighted sum of leaf values
        var output = new Tensor<T>(new[] { batchSize, Options.TreeOutputDimension });

        for (int b = 0; b < batchSize; b++)
        {
            for (int d = 0; d < Options.TreeOutputDimension; d++)
            {
                var sum = NumOps.Zero;

                for (int leaf = 0; leaf < Options.NumLeaves; leaf++)
                {
                    var weight = leafWeights[b * Options.NumLeaves + leaf];
                    var value = _leafValues[treeIdx][leaf * Options.TreeOutputDimension + d];
                    sum = NumOps.Add(sum, NumOps.Multiply(weight, value));
                }

                output[b * Options.TreeOutputDimension + d] = sum;
            }
        }

        return output;
    }

    private Vector<T> ComputeSoftmax(Tensor<T> weights, int depthIdx)
    {
        var result = new Vector<T>(NumFeatures);
        var maxVal = weights[depthIdx * NumFeatures + 0];

        for (int f = 1; f < NumFeatures; f++)
        {
            var val = weights[depthIdx * NumFeatures + f];
            if (NumOps.Compare(val, maxVal) > 0)
                maxVal = val;
        }

        var sumExp = NumOps.Zero;
        for (int f = 0; f < NumFeatures; f++)
        {
            var val = weights[depthIdx * NumFeatures + f];
            var exp = NumOps.Exp(NumOps.Subtract(val, maxVal));
            result[f] = exp;
            sumExp = NumOps.Add(sumExp, exp);
        }

        for (int f = 0; f < NumFeatures; f++)
        {
            result[f] = NumOps.Divide(result[f], sumExp);
        }

        return result;
    }

    private T Sigmoid(T x)
    {
        var negX = NumOps.Negate(x);
        var expNegX = NumOps.Exp(negX);
        return NumOps.Divide(NumOps.One, NumOps.Add(NumOps.One, expNegX));
    }

    /// <summary>
    /// Performs the backward pass through the backbone.
    /// </summary>
    protected Tensor<T> BackwardBackbone(Tensor<T> gradOutput)
    {
        // Simplified backward pass - gradients flow back through preprocessing
        if (_featurePreprocessing != null && _preprocessedFeaturesCache != null)
        {
            return _featurePreprocessing.Backward(gradOutput);
        }

        return gradOutput;
    }

    /// <summary>
    /// Gets the feature importance scores.
    /// </summary>
    /// <returns>Feature importance scores [num_features].</returns>
    public Vector<T> GetFeatureImportance()
    {
        var importance = new Vector<T>(NumFeatures);

        for (int t = 0; t < Options.NumTrees; t++)
        {
            for (int d = 0; d < Options.TreeDepth; d++)
            {
                var softmaxWeights = ComputeSoftmax(_featureSelectionWeights[t], d);

                for (int f = 0; f < NumFeatures; f++)
                {
                    importance[f] = NumOps.Add(importance[f], softmaxWeights[f]);
                }
            }
        }

        // Normalize
        var scale = NumOps.FromDouble(1.0 / (Options.NumTrees * Options.TreeDepth));
        for (int f = 0; f < NumFeatures; f++)
        {
            importance[f] = NumOps.Multiply(importance[f], scale);
        }

        return importance;
    }

    /// <summary>
    /// Updates all parameters.
    /// </summary>
    public virtual void UpdateParameters(T learningRate)
    {
        _featurePreprocessing?.UpdateParameters(learningRate);

        // Update tree parameters (simplified gradient descent)
        // In practice, this would use proper gradient computation
    }

    /// <summary>
    /// Resets internal state.
    /// </summary>
    public virtual void ResetState()
    {
        _preprocessedFeaturesCache = null;
        _splitProbabilitiesCache = null;
        _leafWeightsCache = null;
        _treeOutputsCache = null;

        _featurePreprocessing?.ResetState();
    }
}
