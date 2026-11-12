using AiDotNet.Engines;
using AiDotNet.Attributes;
using System.Collections.Generic;
using AiDotNet.Models.Parameters;
using AiDotNet.LinearAlgebra;
using AiDotNet.Interfaces;
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
public abstract class GANDALFBase<T> : IParameterSource<T>
{
    /// <summary>
    /// Numeric operations helper for type T.
    /// </summary>
    protected readonly INumericOperations<T> NumOps;

    /// <summary>
    /// Hardware-accelerated engine for tensor operations.
    /// </summary>
    protected IEngine Engine => AiDotNetEngine.Current;

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
    [Scratch]
    private Tensor<T>? _gatingWeightsCache;

    /// <summary>
    /// Gets the number of trees.
    /// </summary>
    public int NumTrees => Options.NumTrees;

    /// <summary>
    /// Gets the tree depth.
    /// </summary>
    public int TreeDepth => Options.TreeDepth;

    /// <summary>Built once on first parameter access, then reused.</summary>
    private ParameterComponentRegistry<T>? _parameterRegistry;

    /// <summary>
    /// Extra trainable layers a subclass contributes, folded after the shared backbone.
    /// </summary>
    /// <remarks>
    /// The regression and classification variants share this whole backbone and differ only by a
    /// final projection. Each used to override <see cref="ParameterCount"/> purely to append that
    /// one layer -- and because this base had no GetParameters or SetParameters at all, the head was
    /// COUNTED and never read, never restored and never checkpointed. The count grew; the model that
    /// could be saved did not. Declaring the head here means the subclass states WHAT it adds and
    /// the registry decides where it goes, so count, vector and restore cannot disagree about it.
    /// </remarks>
    protected virtual IEnumerable<IParameterSource<T>> GetExtraTrainableLayers()
        => GeneratedParameterDiscovery.EnumerateDerivedSources<T>(this, typeof(GANDALFBase<T>));

    /// <summary>
    /// The single ordered traversal of this model's parameter-bearing components.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Count, read and restore all derive from THIS, rather than each restating the component list.
    /// Three parallel walks are how a count and a vector come to describe different models: they
    /// agree until someone adds a component to two of them, and nothing reports the disagreement
    /// because the lengths still look plausible. One enumeration makes that unrepresentable.
    /// </para>
    /// <para>
    /// The stable IDs carry a numeric prefix because the registry orders by identity rather than by
    /// the order Register happened to be called -- so the prefix, not the call order, is what pins
    /// serialization order, and it survives a component being added in the middle later.
    /// </para>
    /// </remarks>
    private ParameterComponentRegistry<T> ParameterRegistry
    {
        get
        {
            if (_parameterRegistry is not null) return _parameterRegistry;

            var registry = new ParameterComponentRegistry<T>();
            for (int i = 0; i < _gatingLayers.Count; i++)
                registry.Register($"00000000/{i:D8}", _gatingLayers[i]);

            registry.Register("00000001/output", _gatingOutput);

            registry.Register("00000002/splitWeights",
                new TensorCollectionParameterSource<T>(() => _treeSplitWeights));
            registry.Register("00000003/splitBiases",
                new TensorCollectionParameterSource<T>(() => _treeSplitBiases));
            registry.Register("00000004/leafValues",
                new TensorCollectionParameterSource<T>(() => _treeLeafValues));
            int extraIndex = 0;
            foreach (var extra in GetExtraTrainableLayers())
            {
                if (extra is not null) registry.Register($"00009000/{extraIndex++:D8}", extra);
            }

            _parameterRegistry = registry;
            return registry;
        }
    }

    /// <inheritdoc cref="GetParameters"/>
    public virtual long ParameterCount => ParameterRegistry.ParameterCount;

    /// <summary>
    /// Reads every parameter in traversal order. <see cref="SetParameters"/> reads it back in the
    /// same order, and <see cref="ParameterCount"/> is the length of what this returns.
    /// </summary>
    public virtual Vector<T> GetParameters() => ParameterRegistry.GetParameters();

    /// <summary>Restores every parameter, in the order <see cref="GetParameters"/> emitted them.</summary>
    public virtual void SetParameters(Vector<T> parameters) => ParameterRegistry.SetParameters(parameters);

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
        // N(0, scale^2) for every element in one engine call, seeded from the caller's (secure) Random.
        Engine.TensorNormalInPlace(tensor, 0.0, scale, random.Next());
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
        var weights = Engine.Sigmoid(logits);

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

        if (depth == 0)
        {
            // A stump has one leaf that every row reaches with probability 1.
            var certain = new Tensor<T>(new[] { batchSize, numLeaves });
            certain.Fill(NumOps.One);
            return certain;
        }

        var splitWeights = _treeSplitWeights[treeIndex];
        var splitBiases = _treeSplitBiases[treeIndex];
        var temperature = NumOps.FromDouble(Options.Temperature);

        // Compute split decisions for all internal nodes
        // Split decision = sigmoid((x @ w + b) / temperature)
        var logits = Engine.TensorMatMul(
            Engine.Reshape(gatedFeatures, new[] { batchSize, NumFeatures }),
            Engine.TensorTranspose(Engine.Reshape(splitWeights, new[] { numInternalNodes, NumFeatures })));
        logits = Engine.TensorAdd(logits, Engine.TensorBroadcastTo(
            Engine.Reshape(splitBiases, new[] { 1, numInternalNodes }), new[] { batchSize, numInternalNodes }));
        var splitDecisions = Engine.TensorSigmoid(Engine.TensorDivideScalar(logits, temperature));

        // Soft routing: the probability of reaching a leaf is the product, over the root-to-leaf
        // path, of p (right branch) or 1 - p (left branch). The path is fixed by the tree shape,
        // so per depth we gather each leaf's node and form c0 + p * c1 with (c0, c1) = (0, 1)
        // for right and (1, -1) for left.
        var pathNodes = new int[depth * numLeaves];
        var offsetTerms = new int[depth * numLeaves];
        var slopeTerms = new int[depth * numLeaves];
        for (int leaf = 0; leaf < numLeaves; leaf++)
        {
            int nodeIdx = 0;
            for (int d = 0; d < depth; d++)
            {
                bool goRight = ((leaf >> (depth - d - 1)) & 1) == 1;
                pathNodes[d * numLeaves + leaf] = nodeIdx;
                offsetTerms[d * numLeaves + leaf] = goRight ? 0 : 1;
                slopeTerms[d * numLeaves + leaf] = goRight ? 1 : -1;
                nodeIdx = goRight ? 2 * nodeIdx + 2 : 2 * nodeIdx + 1;
            }
        }

        var pathShape = new[] { depth, numLeaves };
        var gathered = Engine.Reshape(
            Engine.TensorIndexSelect(splitDecisions, Tensor<int>.FromVector(new Vector<int>(pathNodes)), 1),
            new[] { batchSize, depth, numLeaves });
        var factors = Engine.TensorAdd(
            Engine.TensorBroadcastTo(Tensor<int>.FromVector(new Vector<int>(offsetTerms), pathShape).Cast<T>(), new[] { batchSize, depth, numLeaves }),
            Engine.TensorMultiply(gathered,
                Engine.TensorBroadcastTo(Tensor<int>.FromVector(new Vector<int>(slopeTerms), pathShape).Cast<T>(), new[] { batchSize, depth, numLeaves })));

        var leafProbs = Engine.Reshape(
            Engine.TensorSlice(factors, new[] { 0, 0, 0 }, new[] { batchSize, 1, numLeaves }), new[] { batchSize, numLeaves });
        for (int d = 1; d < depth; d++)
        {
            leafProbs = Engine.TensorMultiply(leafProbs, Engine.Reshape(
                Engine.TensorSlice(factors, new[] { 0, d, 0 }, new[] { batchSize, 1, numLeaves }), new[] { batchSize, numLeaves }));
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

        // output[b, d] = sum_leaf leafProbs[b, leaf] * leafValues[leaf, d]
        return Engine.TensorMatMul(
            Engine.Reshape(leafProbs, new[] { batchSize, numLeaves }),
            Engine.Reshape(leafValues, new[] { numLeaves, leafDim }));
    }

    /// <summary>
    /// Performs the forward pass through the GANDALF backbone.
    /// </summary>
    /// <param name="features">Input features [batch_size, num_features].</param>
    /// <returns>Aggregated tree outputs [batch_size, leaf_dim].</returns>
    protected Tensor<T> ForwardBackbone(Tensor<T> features)
    {
        int batchSize = features.Shape[0];
        int leafDim = Options.LeafDimension;

        // Step 1: Compute gating weights
        var gatingWeights = ComputeGating(features);

        // Step 2: Apply gating to features
        var gatedFeatures = Engine.TensorMultiply(features, gatingWeights);

        // Step 3: Pass through all trees and aggregate
        var aggregatedOutput = new Tensor<T>([batchSize, leafDim]);
        aggregatedOutput.Fill(NumOps.Zero);

        for (int t = 0; t < Options.NumTrees; t++)
        {
            var leafProbs = ComputeTreeRouting(gatedFeatures, t);

            var treeOutput = ComputeTreeOutput(leafProbs, t);

            // Add to aggregated output
            aggregatedOutput = Engine.TensorAdd(aggregatedOutput, treeOutput);
        }

        return aggregatedOutput;
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
            if (_treeSplitWeightsGrad[t] is { } swGrad)
            {
                _treeSplitWeights[t] = Engine.TensorSubtract(_treeSplitWeights[t],
                    Engine.TensorMultiplyScalar(swGrad, learningRate));
            }

            if (_treeSplitBiasesGrad[t] is { } sbGrad)
            {
                _treeSplitBiases[t] = Engine.TensorSubtract(_treeSplitBiases[t],
                    Engine.TensorMultiplyScalar(sbGrad, learningRate));
            }

            if (_treeLeafValuesGrad[t] is { } lvGrad)
            {
                _treeLeafValues[t] = Engine.TensorSubtract(_treeLeafValues[t],
                    Engine.TensorMultiplyScalar(lvGrad, learningRate));
            }
        }
    }

    /// <summary>
    /// Resets internal state.
    /// </summary>
    public virtual void ResetState()
    {
        _gatingWeightsCache = null;

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
            uniform.Fill(NumOps.FromDouble(1.0 / NumFeatures));
            return uniform;
        }

        int batchSize = _gatingWeightsCache.Shape[0];

        // Mean gating weight per feature over the batch.
        var summed = Engine.ReduceSum(
            Engine.Reshape(_gatingWeightsCache, new[] { batchSize, NumFeatures }), new[] { 0 }, false);
        return Engine.TensorDivideScalar(summed, NumOps.FromDouble(batchSize)).ToVector();
    }
}
