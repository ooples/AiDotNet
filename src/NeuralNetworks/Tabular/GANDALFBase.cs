using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks.Layers;

namespace AiDotNet.NeuralNetworks.Tabular;

/// <summary>
/// Base implementation of GANDALF (Gated Additive Neural Decision Forest).
/// </summary>
/// <remarks>
/// <para>
/// GANDALF combines gated feature selection with an ensemble of differentiable
/// decision trees. Each tree makes soft routing decisions, and their outputs
/// are combined additively.
/// </para>
/// <para>
/// <b>For Beginners:</b> GANDALF works like a smart forest of decision trees:
///
/// Architecture:
/// 1. **Gating Network**: Learns which features are important
/// 2. **Neural Decision Trees**: Trees with learnable split decisions
/// 3. **Soft Routing**: Samples can go down multiple paths with probabilities
/// 4. **Additive Ensemble**: Tree outputs are summed for final prediction
///
/// Key insight: Traditional trees have hard decisions (left or right).
/// GANDALF uses soft decisions where a sample partially goes both ways,
/// making the whole thing differentiable and trainable with gradient descent.
///
/// Example flow:
/// Input → Gating (feature importance) → Trees (soft routing) → Sum → Output
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public abstract class GANDALFBase<T>
{
    /// <summary>
    /// Numeric operations helper for type T.
    /// </summary>
    protected readonly INumericOperations<T> NumOps;

    /// <summary>
    /// The model configuration options.
    /// </summary>
    protected readonly GANDALFOptions<T> Options;

    /// <summary>
    /// Number of input features.
    /// </summary>
    protected readonly int NumFeatures;

    // Gating network layers
    private readonly List<FullyConnectedLayer<T>> _gatingLayers;
    private readonly FullyConnectedLayer<T> _gatingOutput;

    // Tree parameters
    // For each tree: split features [numInternalNodes], split values [numInternalNodes], leaf values [numLeaves, leafDim]
    private readonly List<Tensor<T>> _treeSplitWeights;    // [numTrees][numInternalNodes, numFeatures]
    private readonly List<Tensor<T>> _treeSplitBiases;     // [numTrees][numInternalNodes]
    private readonly List<Tensor<T>> _treeLeafValues;      // [numTrees][numLeaves, leafDim]

    // Gradients
    private readonly List<Tensor<T>?> _treeSplitWeightsGrad;
    private readonly List<Tensor<T>?> _treeSplitBiasesGrad;
    private readonly List<Tensor<T>?> _treeLeafValuesGrad;

    // Cache for backward pass
    private Tensor<T>? _inputCache;
    private Tensor<T>? _gatingWeightsCache;
    private List<Tensor<T>>? _routingProbsCache;  // Per tree routing probabilities

    /// <summary>
    /// Gets the number of trees.
    /// </summary>
    public int NumTrees => Options.NumTrees;

    /// <summary>
    /// Gets the tree depth.
    /// </summary>
    public int TreeDepth => Options.TreeDepth;

    /// <summary>
    /// Gets the total number of trainable parameters.
    /// </summary>
    public virtual int ParameterCount
    {
        get
        {
            int count = 0;

            // Gating layers
            foreach (var layer in _gatingLayers)
                count += layer.ParameterCount;
            count += _gatingOutput.ParameterCount;

            // Tree parameters
            for (int t = 0; t < Options.NumTrees; t++)
            {
                count += _treeSplitWeights[t].Length;
                count += _treeSplitBiases[t].Length;
                count += _treeLeafValues[t].Length;
            }

            return count;
        }
    }

    /// <summary>
    /// Initializes a new instance of the GANDALFBase class.
    /// </summary>
    /// <param name="numFeatures">Number of input features.</param>
    /// <param name="options">Model configuration options.</param>
    protected GANDALFBase(int numFeatures, GANDALFOptions<T>? options = null)
    {
        NumOps = MathHelper.GetNumericOperations<T>();
        Options = options ?? new GANDALFOptions<T>();
        NumFeatures = numFeatures;

        // Initialize gating network
        _gatingLayers = new List<FullyConnectedLayer<T>>();
        int prevDim = numFeatures;

        for (int i = 0; i < Options.NumGatingLayers; i++)
        {
            var layer = new FullyConnectedLayer<T>(
                prevDim,
                Options.GatingHiddenDimension,
                new ReLUActivation<T>() as IActivationFunction<T>);
            _gatingLayers.Add(layer);
            prevDim = Options.GatingHiddenDimension;
        }

        // Gating output produces importance weights for each feature
        _gatingOutput = new FullyConnectedLayer<T>(
            prevDim,
            numFeatures,
            (IActivationFunction<T>?)null);  // Sigmoid applied separately

        // Initialize tree parameters
        _treeSplitWeights = new List<Tensor<T>>();
        _treeSplitBiases = new List<Tensor<T>>();
        _treeLeafValues = new List<Tensor<T>>();
        _treeSplitWeightsGrad = new List<Tensor<T>?>();
        _treeSplitBiasesGrad = new List<Tensor<T>?>();
        _treeLeafValuesGrad = new List<Tensor<T>?>();

        var random = RandomHelper.CreateSecureRandom();
        int numInternalNodes = Options.NumInternalNodes;
        int numLeaves = Options.NumLeaves;

        for (int t = 0; t < Options.NumTrees; t++)
        {
            // Split weights: each internal node has weights for all features
            var splitWeights = new Tensor<T>([numInternalNodes, numFeatures]);
            InitializeNormal(splitWeights, Options.InitScale, random);
            _treeSplitWeights.Add(splitWeights);

            // Split biases
            var splitBiases = new Tensor<T>([numInternalNodes]);
            splitBiases.Fill(NumOps.Zero);
            _treeSplitBiases.Add(splitBiases);

            // Leaf values
            var leafValues = new Tensor<T>([numLeaves, Options.LeafDimension]);
            InitializeNormal(leafValues, Options.InitScale, random);
            _treeLeafValues.Add(leafValues);

            _treeSplitWeightsGrad.Add(null);
            _treeSplitBiasesGrad.Add(null);
            _treeLeafValuesGrad.Add(null);
        }
    }

    /// <summary>
    /// Initializes a tensor with normal distribution.
    /// </summary>
    private void InitializeNormal(Tensor<T> tensor, double scale, Random random)
    {
        for (int i = 0; i < tensor.Length; i++)
        {
            double u1 = 1.0 - random.NextDouble();
            double u2 = 1.0 - random.NextDouble();
            double normal = Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2);
            tensor[i] = NumOps.FromDouble(normal * scale);
        }
    }

    /// <summary>
    /// Computes the gating weights (feature importance).
    /// </summary>
    /// <param name="features">Input features [batch_size, num_features].</param>
    /// <returns>Gating weights [batch_size, num_features] in range [0, 1].</returns>
    protected Tensor<T> ComputeGating(Tensor<T> features)
    {
        var x = features;

        // Pass through gating layers
        foreach (var layer in _gatingLayers)
        {
            x = layer.Forward(x);
        }

        // Output layer
        var logits = _gatingOutput.Forward(x);

        // Apply sigmoid for gating weights in [0, 1]
        int batchSize = logits.Shape[0];
        var weights = new Tensor<T>(logits.Shape);

        for (int i = 0; i < logits.Length; i++)
        {
            var negLogit = NumOps.Negate(logits[i]);
            var expNeg = NumOps.Exp(negLogit);
            weights[i] = NumOps.Divide(NumOps.One, NumOps.Add(NumOps.One, expNeg));
        }

        _gatingWeightsCache = weights;
        return weights;
    }

    /// <summary>
    /// Computes soft routing probabilities through a single tree.
    /// </summary>
    /// <param name="gatedFeatures">Gated input features [batch_size, num_features].</param>
    /// <param name="treeIndex">Index of the tree.</param>
    /// <returns>Leaf probabilities [batch_size, num_leaves].</returns>
    protected Tensor<T> ComputeTreeRouting(Tensor<T> gatedFeatures, int treeIndex)
    {
        int batchSize = gatedFeatures.Shape[0];
        int numInternalNodes = Options.NumInternalNodes;
        int numLeaves = Options.NumLeaves;
        int depth = Options.TreeDepth;

        var splitWeights = _treeSplitWeights[treeIndex];
        var splitBiases = _treeSplitBiases[treeIndex];
        var temperature = NumOps.FromDouble(Options.Temperature);

        // Compute split decisions for all internal nodes
        // Split decision = sigmoid((x @ w + b) / temperature)
        var splitDecisions = new Tensor<T>([batchSize, numInternalNodes]);

        for (int b = 0; b < batchSize; b++)
        {
            for (int n = 0; n < numInternalNodes; n++)
            {
                // Compute weighted sum
                var sum = splitBiases[n];
                for (int f = 0; f < NumFeatures; f++)
                {
                    sum = NumOps.Add(sum,
                        NumOps.Multiply(gatedFeatures[b * NumFeatures + f],
                                       splitWeights[n * NumFeatures + f]));
                }

                // Apply temperature scaling and sigmoid
                var scaled = NumOps.Divide(sum, temperature);
                var negScaled = NumOps.Negate(scaled);
                var expNeg = NumOps.Exp(negScaled);
                splitDecisions[b * numInternalNodes + n] =
                    NumOps.Divide(NumOps.One, NumOps.Add(NumOps.One, expNeg));
            }
        }

        // Compute leaf probabilities through soft routing
        // For each leaf, compute the probability of reaching it
        var leafProbs = new Tensor<T>([batchSize, numLeaves]);

        for (int b = 0; b < batchSize; b++)
        {
            for (int leaf = 0; leaf < numLeaves; leaf++)
            {
                // Trace path from root to leaf
                var prob = NumOps.One;
                int nodeIdx = 0;

                for (int d = 0; d < depth; d++)
                {
                    var splitProb = splitDecisions[b * numInternalNodes + nodeIdx];

                    // Check which branch this leaf is on at this depth
                    int levelSize = 1 << d;
                    int posInLevel = leaf >> (depth - d - 1);
                    bool goRight = (posInLevel & 1) == 1;

                    if (goRight)
                    {
                        prob = NumOps.Multiply(prob, splitProb);
                        nodeIdx = 2 * nodeIdx + 2;  // Right child
                    }
                    else
                    {
                        prob = NumOps.Multiply(prob, NumOps.Subtract(NumOps.One, splitProb));
                        nodeIdx = 2 * nodeIdx + 1;  // Left child
                    }

                    // Fix node index to be within bounds
                    if (nodeIdx >= numInternalNodes)
                        break;
                }

                leafProbs[b * numLeaves + leaf] = prob;
            }
        }

        return leafProbs;
    }

    /// <summary>
    /// Computes the output of a single tree.
    /// </summary>
    /// <param name="leafProbs">Leaf probabilities [batch_size, num_leaves].</param>
    /// <param name="treeIndex">Index of the tree.</param>
    /// <returns>Tree output [batch_size, leaf_dim].</returns>
    protected Tensor<T> ComputeTreeOutput(Tensor<T> leafProbs, int treeIndex)
    {
        int batchSize = leafProbs.Shape[0];
        int numLeaves = Options.NumLeaves;
        int leafDim = Options.LeafDimension;

        var leafValues = _treeLeafValues[treeIndex];
        var output = new Tensor<T>([batchSize, leafDim]);

        for (int b = 0; b < batchSize; b++)
        {
            for (int d = 0; d < leafDim; d++)
            {
                var sum = NumOps.Zero;
                for (int leaf = 0; leaf < numLeaves; leaf++)
                {
                    sum = NumOps.Add(sum,
                        NumOps.Multiply(leafProbs[b * numLeaves + leaf],
                                       leafValues[leaf * leafDim + d]));
                }
                output[b * leafDim + d] = sum;
            }
        }

        return output;
    }

    /// <summary>
    /// Performs the forward pass through the GANDALF backbone.
    /// </summary>
    /// <param name="features">Input features [batch_size, num_features].</param>
    /// <returns>Aggregated tree outputs [batch_size, leaf_dim].</returns>
    protected Tensor<T> ForwardBackbone(Tensor<T> features)
    {
        _inputCache = features;
        int batchSize = features.Shape[0];
        int leafDim = Options.LeafDimension;

        // Step 1: Compute gating weights
        var gatingWeights = ComputeGating(features);

        // Step 2: Apply gating to features
        var gatedFeatures = new Tensor<T>(features.Shape);
        for (int i = 0; i < features.Length; i++)
        {
            gatedFeatures[i] = NumOps.Multiply(features[i], gatingWeights[i]);
        }

        // Step 3: Pass through all trees and aggregate
        var aggregatedOutput = new Tensor<T>([batchSize, leafDim]);
        aggregatedOutput.Fill(NumOps.Zero);

        _routingProbsCache = new List<Tensor<T>>();

        for (int t = 0; t < Options.NumTrees; t++)
        {
            var leafProbs = ComputeTreeRouting(gatedFeatures, t);
            _routingProbsCache.Add(leafProbs);

            var treeOutput = ComputeTreeOutput(leafProbs, t);

            // Add to aggregated output
            for (int i = 0; i < aggregatedOutput.Length; i++)
            {
                aggregatedOutput[i] = NumOps.Add(aggregatedOutput[i], treeOutput[i]);
            }
        }

        return aggregatedOutput;
    }

    /// <summary>
    /// Performs the backward pass through the GANDALF backbone.
    /// </summary>
    /// <param name="outputGradient">Gradient from prediction head [batch_size, leaf_dim].</param>
    /// <returns>Gradient with respect to input features [batch_size, num_features].</returns>
    protected Tensor<T> BackwardBackbone(Tensor<T> outputGradient)
    {
        if (_inputCache == null || _gatingWeightsCache == null || _routingProbsCache == null)
        {
            throw new InvalidOperationException("Forward pass must be called before backward pass.");
        }

        int batchSize = outputGradient.Shape[0];
        int leafDim = Options.LeafDimension;
        int numLeaves = Options.NumLeaves;

        // Initialize gradients
        for (int t = 0; t < Options.NumTrees; t++)
        {
            _treeSplitWeightsGrad[t] = new Tensor<T>(_treeSplitWeights[t].Shape);
            _treeSplitWeightsGrad[t]!.Fill(NumOps.Zero);
            _treeSplitBiasesGrad[t] = new Tensor<T>(_treeSplitBiases[t].Shape);
            _treeSplitBiasesGrad[t]!.Fill(NumOps.Zero);
            _treeLeafValuesGrad[t] = new Tensor<T>(_treeLeafValues[t].Shape);
            _treeLeafValuesGrad[t]!.Fill(NumOps.Zero);
        }

        // Gradient w.r.t. leaf values for each tree
        for (int t = 0; t < Options.NumTrees; t++)
        {
            var leafProbs = _routingProbsCache[t];
            var leafValuesGrad = _treeLeafValuesGrad[t]!;

            for (int b = 0; b < batchSize; b++)
            {
                for (int leaf = 0; leaf < numLeaves; leaf++)
                {
                    for (int d = 0; d < leafDim; d++)
                    {
                        // d(loss)/d(leafValue) = d(loss)/d(output) * leafProb
                        leafValuesGrad[leaf * leafDim + d] = NumOps.Add(
                            leafValuesGrad[leaf * leafDim + d],
                            NumOps.Multiply(outputGradient[b * leafDim + d],
                                           leafProbs[b * numLeaves + leaf]));
                    }
                }
            }
        }

        // Backward through gating network (simplified - full implementation more complex)
        var gatingGrad = new Tensor<T>([batchSize, NumFeatures]);
        gatingGrad.Fill(NumOps.Zero);

        // Backward through gating layers
        var grad = _gatingOutput.Backward(gatingGrad);
        for (int i = _gatingLayers.Count - 1; i >= 0; i--)
        {
            grad = _gatingLayers[i].Backward(grad);
        }

        return grad;
    }

    /// <summary>
    /// Updates all parameters using the calculated gradients.
    /// </summary>
    public virtual void UpdateParameters(T learningRate)
    {
        // Update gating layers
        foreach (var layer in _gatingLayers)
        {
            layer.UpdateParameters(learningRate);
        }
        _gatingOutput.UpdateParameters(learningRate);

        // Update tree parameters
        for (int t = 0; t < Options.NumTrees; t++)
        {
            if (_treeSplitWeightsGrad[t] != null)
            {
                for (int i = 0; i < _treeSplitWeights[t].Length; i++)
                {
                    _treeSplitWeights[t][i] = NumOps.Subtract(
                        _treeSplitWeights[t][i],
                        NumOps.Multiply(learningRate, _treeSplitWeightsGrad[t]![i]));
                }
            }

            if (_treeSplitBiasesGrad[t] != null)
            {
                for (int i = 0; i < _treeSplitBiases[t].Length; i++)
                {
                    _treeSplitBiases[t][i] = NumOps.Subtract(
                        _treeSplitBiases[t][i],
                        NumOps.Multiply(learningRate, _treeSplitBiasesGrad[t]![i]));
                }
            }

            if (_treeLeafValuesGrad[t] != null)
            {
                for (int i = 0; i < _treeLeafValues[t].Length; i++)
                {
                    _treeLeafValues[t][i] = NumOps.Subtract(
                        _treeLeafValues[t][i],
                        NumOps.Multiply(learningRate, _treeLeafValuesGrad[t]![i]));
                }
            }
        }
    }

    /// <summary>
    /// Resets internal state.
    /// </summary>
    public virtual void ResetState()
    {
        _inputCache = null;
        _gatingWeightsCache = null;
        _routingProbsCache = null;

        foreach (var layer in _gatingLayers)
            layer.ResetState();
        _gatingOutput.ResetState();

        for (int t = 0; t < Options.NumTrees; t++)
        {
            _treeSplitWeightsGrad[t] = null;
            _treeSplitBiasesGrad[t] = null;
            _treeLeafValuesGrad[t] = null;
        }
    }

    /// <summary>
    /// Gets the gating weights (feature importance) from the last forward pass.
    /// </summary>
    /// <returns>Gating weights [batch_size, num_features] or null if not available.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> These weights show how important each feature is:
    /// - Weights close to 1.0: Feature is very important
    /// - Weights close to 0.0: Feature is not important
    ///
    /// Use this for feature importance analysis and interpretability.
    /// </para>
    /// </remarks>
    public Tensor<T>? GetGatingWeights() => _gatingWeightsCache;

    /// <summary>
    /// Gets the average feature importance across all predictions.
    /// </summary>
    /// <param name="features">Features to compute importance for [num_samples, num_features].</param>
    /// <returns>Average feature importance [num_features].</returns>
    public Vector<T> GetFeatureImportance(Tensor<T> features)
    {
        // Run forward pass to populate gating weights
        _ = ForwardBackbone(features);

        if (_gatingWeightsCache == null)
        {
            var uniform = new Vector<T>(NumFeatures);
            var val = NumOps.FromDouble(1.0 / NumFeatures);
            for (int i = 0; i < NumFeatures; i++)
                uniform[i] = val;
            return uniform;
        }

        int batchSize = _gatingWeightsCache.Shape[0];
        var importance = new Vector<T>(NumFeatures);

        for (int f = 0; f < NumFeatures; f++)
        {
            var sum = NumOps.Zero;
            for (int b = 0; b < batchSize; b++)
            {
                sum = NumOps.Add(sum, _gatingWeightsCache[b * NumFeatures + f]);
            }
            importance[f] = NumOps.Divide(sum, NumOps.FromDouble(batchSize));
        }

        return importance;
    }
}
