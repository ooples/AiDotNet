#pragma warning disable CS0649, CS0414, CS0169
using AiDotNet.LearningRateSchedulers;
using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.NeuralNetworks.Options;
using AiDotNet.Optimizers;

namespace AiDotNet.NeuralNetworks;

/// <summary>
/// Represents a Graph Neural Network that can process data represented as graphs.
/// </summary>
/// <remarks>
/// <para>
/// A Graph Neural Network (GNN) is designed to work with data structured as graphs, 
/// where nodes represent entities and edges represent relationships between these entities.
/// This implementation supports various activation functions for different layers and
/// provides methods for predicting outputs from both vector inputs and graph inputs.
/// </para>
/// <para><b>For Beginners:</b> A Graph Neural Network is a type of neural network that works with connected data.
/// 
/// Think of it like analyzing a social network:
/// - Each person is a "node" in the graph
/// - Friendships between people are "edges" connecting the nodes
/// - People have attributes (like age, location, interests) which are "node features"
/// 
/// GNNs are useful when:
/// - The relationships between items are as important as the items themselves
/// - You're working with network-like data (social networks, molecules, road systems)
/// - You need to make predictions about how nodes influence each other
/// 
/// For example, GNNs can help predict which products a customer might like based on
/// what similar customers have purchased, by analyzing the connections between customers and products.
/// </para>
/// </remarks>
/// <example>
/// <code>
/// var options = new GraphNeuralNetworkOptions { NodeFeatureSize = 16, HiddenSize = 64, NumLayers = 3 };
/// var nodeFeatures = Tensor&lt;float&gt;.CreateRandom(new[] { 10, 16 });
/// var trainX = Tensor&lt;float&gt;.CreateRandom(4, 8);
/// var trainY = Tensor&lt;float&gt;.CreateRandom(4, 2);
/// var result = new AiModelBuilder&lt;float, Tensor&lt;float&gt;, Tensor&lt;float&gt;&gt;()
///     .ConfigureModel(new GraphNeuralNetwork&lt;float&gt;(options))
///     .Build(trainX, trainY);
/// var output = result.Predict(nodeFeatures);
/// </code>
/// </example>
/// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
[ModelDomain(ModelDomain.GraphAnalysis)]
[ModelCategory(ModelCategory.NeuralNetwork)]
[ModelCategory(ModelCategory.GraphNetwork)]
[ModelTask(ModelTask.Classification)]
[ModelTask(ModelTask.Regression)]
[ModelComplexity(ModelComplexity.Medium)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("Semi-Supervised Classification with Graph Convolutional Networks", "https://arxiv.org/abs/1609.02907", Year = 2017, Authors = "Thomas N. Kipf, Max Welling")]
[PaperOptimizer(OptimizerKind.Adam, LearningRate = 0.01, EarlyStoppingPatience = 10,
                Source = "Kipf and Welling 2017, Sec. 5: all models trained for a maximum of 200 epochs "
                        + "using Adam with a learning rate of 0.01 and early stopping with a window size "
                        + "of 10.")]
public partial class GraphNeuralNetwork<T> : GraphModelLayoutBase<T>, IAuxiliaryLossLayer<T>
{
    private readonly GraphNeuralNetworkOptions _options;
    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> _optimizer;

    /// <summary>
    /// Gets or sets whether auxiliary loss (graph smoothness regularization) should be used during training.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Graph smoothness regularization encourages connected nodes to have similar representations.
    /// This is based on the principle that nodes with edges between them should have similar features,
    /// which is a common assumption in many graph-based learning tasks.
    /// </para>
    /// <para><b>For Beginners:</b> Graph smoothness is like encouraging friends to be similar.
    ///
    /// In a graph:
    /// - Nodes that are connected (like friends in a social network) should have similar features
    /// - This auxiliary loss penalizes the network when connected nodes have very different representations
    /// - It helps the network learn more meaningful patterns that respect the graph structure
    ///
    /// For example:
    /// - In a social network, friends often have similar interests
    /// - In a molecule, bonded atoms influence each other's properties
    /// - In a citation network, papers that cite each other often cover similar topics
    ///
    /// Enabling this helps the network learn representations that are consistent with the graph structure.
    /// </para>
    /// </remarks>
    public bool UseAuxiliaryLoss { get; set; } = false;

    /// <summary>
    /// Gets or sets the weight for the graph smoothness auxiliary loss.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This weight controls how much the graph smoothness regularization contributes to the total loss.
    /// The total loss is: main_loss + (auxiliary_weight * smoothness_loss).
    /// Typical values range from 0.01 to 0.1.
    /// </para>
    /// <para><b>For Beginners:</b> This controls how much the network should enforce similarity between connected nodes.
    ///
    /// The weight determines the balance between:
    /// - Task accuracy (main loss) - making correct predictions
    /// - Graph smoothness (auxiliary loss) - keeping connected nodes similar
    ///
    /// Common values:
    /// - 0.05 (default): Balanced smoothness regularization
    /// - 0.01-0.03: Light smoothness enforcement
    /// - 0.08-0.1: Strong smoothness enforcement
    ///
    /// Higher values make the network focus more on keeping connected nodes similar,
    /// which can help with generalization but may reduce flexibility.
    /// </para>
    /// </remarks>
    public T AuxiliaryLossWeight { get; set; }

    private T _lastGraphSmoothnessLoss;
    [Scratch]
    private Tensor<T>? _lastNodeRepresentations = null;
    [Scratch]
    private Tensor<T>? _lastAdjacencyMatrix = null;
    private IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _trainOptimizer;
    /// <summary>
    /// Gets or sets the vector activation function used in graph convolutional layers.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This activation function is applied to the output of graph convolutional layers and operates on entire vectors.
    /// Vector activation functions transform multiple values at once, potentially capturing relationships between them.
    /// </para>
    /// <para><b>For Beginners:</b> This determines how signals are processed in the graph layers.
    /// 
    /// An activation function:
    /// - Adds non-linearity to the network (helps it learn complex patterns)
    /// - Decides whether a node should pass information to the next layer
    /// - Usually transforms values to be within a certain range (like 0 to 1)
    /// 
    /// This specific activation works on groups of numbers together, rather than individually.
    /// </para>
    /// </remarks>
    private IVectorActivationFunction<T>? _graphConvolutionalVectorActivation { get; set; }

    /// <summary>
    /// Gets or sets the vector activation function used in standard activation layers.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This activation function is applied in dedicated activation layers and operates on entire vectors.
    /// These layers typically follow other types of layers to introduce non-linearity.
    /// </para>
    /// <para><b>For Beginners:</b> This activation function is used in specific layers
    /// dedicated just to adding non-linearity to the network.
    /// 
    /// Think of it like a filter that processes multiple values at once to decide
    /// which information should continue through the network.
    /// </para>
    /// </remarks>
    private IVectorActivationFunction<T>? _activationLayerVectorActivation { get; set; }

    /// <summary>
    /// Gets or sets the vector activation function used in the final dense layer.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This activation function is applied to the output of the final dense layer and operates on entire vectors.
    /// The final dense layer typically maps features to output classes or values.
    /// </para>
    /// <para><b>For Beginners:</b> This activation function processes the information 
    /// in the last main layer of the network.
    /// 
    /// It helps transform the final calculations into the format needed for your prediction.
    /// </para>
    /// </remarks>
    private IVectorActivationFunction<T>? _finalDenseLayerVectorActivation { get; set; }

    /// <summary>
    /// Gets or sets the vector activation function used in the final activation layer.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This activation function is applied in the final dedicated activation layer and operates on entire vectors.
    /// The final activation layer often determines the range and characteristics of the network's output.
    /// </para>
    /// <para><b>For Beginners:</b> This is the very last processing step that determines
    /// the final output format of your network.
    /// 
    /// For example:
    /// - If you're classifying items into categories, this might ensure outputs are probabilities (0-1)
    /// - If you're predicting a continuous value, this might allow outputs to be any number
    /// </para>
    /// </remarks>
    private IVectorActivationFunction<T>? _finalActivationLayerVectorActivation { get; set; }

    /// <summary>
    /// Gets or sets the scalar activation function used in graph convolutional layers.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This activation function is applied to individual scalars in graph convolutional layers.
    /// Scalar activation functions transform each value independently.
    /// </para>
    /// <para><b>For Beginners:</b> This processes each individual number in graph layers separately.
    /// 
    /// Unlike the vector activation which works on groups of numbers together,
    /// this one applies the same transformation to each number independently.
    /// </para>
    /// </remarks>
    private IActivationFunction<T>? _graphConvolutionalScalarActivation { get; set; }

    /// <summary>
    /// Gets or sets the scalar activation function used in standard activation layers.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This activation function is applied to individual scalars in dedicated activation layers.
    /// Scalar activation functions transform each value independently.
    /// </para>
    /// <para><b>For Beginners:</b> This processes each value individually in layers
    /// dedicated to adding non-linearity.
    /// 
    /// It's like having a separate filter for each piece of information.
    /// </para>
    /// </remarks>
    private IActivationFunction<T>? _activationLayerScalarActivation { get; set; }

    /// <summary>
    /// Gets or sets the scalar activation function used in the final dense layer.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This activation function is applied to individual scalars in the final dense layer.
    /// Scalar activation functions transform each value independently.
    /// </para>
    /// <para><b>For Beginners:</b> This processes each number individually in the 
    /// last main layer of the network.
    /// 
    /// Each output gets its own separate transformation.
    /// </para>
    /// </remarks>
    private IActivationFunction<T>? _finalDenseLayerScalarActivation { get; set; }

    /// <summary>
    /// Gets or sets the scalar activation function used in the final activation layer.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This activation function is applied to individual scalars in the final activation layer.
    /// Scalar activation functions transform each value independently.
    /// </para>
    /// <para><b>For Beginners:</b> This is the very last processing step applied
    /// individually to each output value.
    /// 
    /// It determines the final form of each individual output number.
    /// </para>
    /// </remarks>
    private IActivationFunction<T>? _finalActivationLayerScalarActivation { get; set; }

    /// <summary>
    /// Initializes a new instance of the <see cref="GraphNeuralNetwork{T}"/> class with vector activation functions.
    /// </summary>
    /// <param name="architecture">The neural network architecture defining the structure of the network.</param>
    /// <param name="graphConvolutionalVectorActivation">The vector activation function for graph convolutional layers.</param>
    /// <param name="activationLayerVectorActivation">The vector activation function for standard activation layers.</param>
    /// <param name="finalDenseLayerVectorActivation">The vector activation function for the final dense layer.</param>
    /// <param name="finalActivationLayerVectorActivation">The vector activation function for the final activation layer.</param>
    /// <remarks>
    /// <para>
    /// This constructor creates a graph neural network with the specified architecture and vector activation functions.
    /// Vector activation functions operate on entire vectors rather than individual elements.
    /// </para>
    /// <para><b>For Beginners:</b> This creates a new graph neural network where the 
    /// activation functions work on groups of numbers together.
    /// 
    /// When creating your network, you specify:
    /// - The overall structure (architecture)
    /// - Which activation functions to use at different stages
    /// 
    /// Vector activation functions process multiple values as a group, which can help
    /// capture relationships between different values.
    /// </para>
    /// </remarks>
    /// <summary>
    /// Initializes a new instance with default architecture settings.
    /// </summary>
    public GraphNeuralNetwork()
        : this((GraphNeuralNetworkOptions?)null)
    {
    }

    /// <summary>
    /// Initializes a new instance from fully customizable GCN options.
    /// </summary>
    public GraphNeuralNetwork(GraphNeuralNetworkOptions? options)
        : this(CreateDefaultArchitecture(options),
            graphConvolutionalVectorActivation: (IVectorActivationFunction<T>?)null,
            options: options)
    {
    }

    private static NeuralNetworkArchitecture<T> CreateDefaultArchitecture(GraphNeuralNetworkOptions? options)
    {
        var resolvedOptions = options ?? new GraphNeuralNetworkOptions();
        return new NeuralNetworkArchitecture<T>(
            inputType: Enums.InputType.OneDimensional,
            taskType: Enums.NeuralNetworkTaskType.MultiClassClassification,
            inputSize: resolvedOptions.NodeFeatureSize,
            outputSize: resolvedOptions.NumClasses);
    }

    public GraphNeuralNetwork(NeuralNetworkArchitecture<T> architecture,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
        ILossFunction<T>? lossFunction = null,
        IVectorActivationFunction<T>? graphConvolutionalVectorActivation = null,
        IVectorActivationFunction<T>? activationLayerVectorActivation = null, IVectorActivationFunction<T>? finalDenseLayerVectorActivation = null,
        IVectorActivationFunction<T>? finalActivationLayerVectorActivation = null,
        GraphNeuralNetworkOptions? options = null) :
        base(architecture, lossFunction ?? NeuralNetworkHelper<T>.GetDefaultLossFunction(architecture.TaskType))
    {
        _options = options ?? new GraphNeuralNetworkOptions();
        _optimizer = optimizer ?? CreateDefaultOptimizer(this, _options);
        Options = _options;
        UseAuxiliaryLoss = _options.UseAuxiliaryLoss;
        AuxiliaryLossWeight = NumOps.FromDouble(_options.AuxiliaryLossWeight);
        _lastGraphSmoothnessLoss = NumOps.Zero;

        _graphConvolutionalVectorActivation = graphConvolutionalVectorActivation;
        _activationLayerVectorActivation = activationLayerVectorActivation;
        _finalDenseLayerVectorActivation = finalDenseLayerVectorActivation;
        _finalActivationLayerVectorActivation = finalActivationLayerVectorActivation;

        InitializeLayers();
    }

    /// <summary>
    /// Initializes a new instance of the <see cref="GraphNeuralNetwork{T}"/> class with scalar activation functions.
    /// </summary>
    /// <param name="architecture">The neural network architecture defining the structure of the network.</param>
    /// <param name="graphConvolutionalActivation">The scalar activation function for graph convolutional layers.</param>
    /// <param name="activationLayerActivation">The scalar activation function for standard activation layers.</param>
    /// <param name="finalDenseLayerActivation">The scalar activation function for the final dense layer.</param>
    /// <param name="finalActivationLayerActivation">The scalar activation function for the final activation layer.</param>
    /// <remarks>
    /// <para>
    /// This constructor creates a graph neural network with the specified architecture and scalar activation functions.
    /// Scalar activation functions operate on individual elements rather than entire vectors.
    /// </para>
    /// <para><b>For Beginners:</b> This creates a new graph neural network where the 
    /// activation functions work on individual numbers separately.
    /// 
    /// When creating your network, you specify:
    /// - The overall structure (architecture)
    /// - Which activation functions to use at different stages
    /// 
    /// Scalar activation functions process each value independently, applying the same
    /// transformation to each number without considering relationships between values.
    /// </para>
    /// </remarks>
    public GraphNeuralNetwork(NeuralNetworkArchitecture<T> architecture,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
        ILossFunction<T>? lossFunction = null,
        IActivationFunction<T>? graphConvolutionalActivation = null,
        IActivationFunction<T>? activationLayerActivation = null, IActivationFunction<T>? finalDenseLayerActivation = null,
        IActivationFunction<T>? finalActivationLayerActivation = null,
        GraphNeuralNetworkOptions? options = null) :
        base(architecture, lossFunction ?? NeuralNetworkHelper<T>.GetDefaultLossFunction(architecture.TaskType))
    {
        _options = options ?? new GraphNeuralNetworkOptions();
        _optimizer = optimizer ?? CreateDefaultOptimizer(this, _options);
        Options = _options;
        UseAuxiliaryLoss = _options.UseAuxiliaryLoss;
        AuxiliaryLossWeight = NumOps.FromDouble(_options.AuxiliaryLossWeight);
        _lastGraphSmoothnessLoss = NumOps.Zero;

        _graphConvolutionalScalarActivation = graphConvolutionalActivation;
        _activationLayerScalarActivation = activationLayerActivation;
        _finalDenseLayerScalarActivation = finalDenseLayerActivation;
        _finalActivationLayerScalarActivation = finalActivationLayerActivation;

        InitializeLayers();
    }

    private static AdamOptimizer<T, Tensor<T>, Tensor<T>> CreateDefaultOptimizer(
        GraphNeuralNetwork<T> model,
        GraphNeuralNetworkOptions options)
    {
        return new AdamOptimizer<T, Tensor<T>, Tensor<T>>(
            model,
            new AdamOptimizerOptions<T, Tensor<T>, Tensor<T>>
            {
                InitialLearningRate = options.LearningRate,
                // Kipf & Welling regularize only the first graph-convolution
                // weight matrix, not later weights or any biases.
                //
                // The range is resolved from the CONSTRUCTED layers, not computed as
                // CalculatedInputSize * HiddenSize. That product assumed two things a custom
                // Architecture.Layers can break: that the first graph convolution is the first
                // trainable block in the flat parameter vector (so its weights start at index 0),
                // and that its shape matches the option-derived one. Either being false silently
                // shrinks a different slice of parameters than the paper's rule names. It has to
                // be lazy because the constructor calls this BEFORE InitializeLayers.
                Regularization = new FirstLayerWeightsL2Regularization(
                    options.L2Regularization,
                    () => ResolveFirstGraphConvWeightRange(model))
            });
    }

    /// <summary>
    /// Locates the first graph-convolution weight matrix inside the flat parameter vector, as the
    /// half-open range <c>[start, start + count)</c>.
    /// </summary>
    /// <remarks>
    /// <c>GraphConvolutionalLayer</c> lays its parameters out as weights then bias, and
    /// <c>InputFeatures * OutputFeatures</c> is the weight block, so the bias is excluded by taking
    /// only that prefix of the layer's own slice. An empty range means no graph convolution is
    /// present -- possible with a fully custom layer list -- and regularizing nothing is the correct
    /// reading of a rule that names a layer this network does not have.
    /// </remarks>
    private static (int Start, int Count) ResolveFirstGraphConvWeightRange(GraphNeuralNetwork<T> model)
    {
        int start = 0;
        foreach (var layer in model.Layers)
        {
            if (layer is GraphConvolutionalLayer<T> graphConv)
            {
                return (start, checked(graphConv.InputFeatures * graphConv.OutputFeatures));
            }

            start = checked(start + (int)layer.ParameterCount);
        }

        return (0, 0);
    }

    private sealed class FirstLayerWeightsL2Regularization
        : AiDotNet.Regularization.RegularizationBase<T, Tensor<T>, Tensor<T>>
    {
        private readonly Func<(int Start, int Count)> _resolveRange;
        private bool _rangeResolved;
        private int _start;
        private int _weightCount;

        public FirstLayerWeightsL2Regularization(double strength, Func<(int Start, int Count)> resolveRange)
            : base(new RegularizationOptions
            {
                Type = RegularizationType.L2,
                Strength = strength
            })
        {
            _resolveRange = resolveRange ?? throw new ArgumentNullException(nameof(resolveRange));

            // STRENGTH IS A SHRINKAGE FACTOR, so it has to land in [0, 1]. Regularize computes
            // 1 - strength: above 1 the shrinkage goes negative and every first-layer weight FLIPS
            // SIGN on each application, and below 0 it exceeds 1 and amplifies the weights it is
            // supposed to shrink. Neither fails anywhere, they just train something else.
            if (strength < 0.0 || strength > 1.0 || double.IsNaN(strength))
            {
                throw new ArgumentOutOfRangeException(
                    nameof(strength), strength,
                    "L2 shrinkage strength must be in [0, 1]; 1 - strength is applied as a multiplier.");
            }
        }

        /// <summary>
        /// Resolves and caches the regularized parameter range on first use.
        /// </summary>
        /// <remarks>
        /// Deferred because the network constructs its optimizer before its layers, so the range does
        /// not exist yet at construction. Cached because the layer shapes are fixed once built, and
        /// re-walking them on every gradient step would put a layer scan on the training path.
        /// </remarks>
        private (int Start, int Count) Range()
        {
            if (!_rangeResolved)
            {
                (_start, _weightCount) = _resolveRange();
                if (_start < 0 || _weightCount < 0)
                {
                    throw new InvalidOperationException(
                        $"The first-layer weight range resolved to start {_start}, count {_weightCount}; " +
                        "both must be non-negative.");
                }

                _rangeResolved = true;
            }

            return (_start, _weightCount);
        }

        public override Vector<T> Regularize(Vector<T> gradient, Vector<T> coefficients)
        {
            if (gradient.Length != coefficients.Length)
                throw new ArgumentException("Gradient and coefficient vectors must have the same length.");

            var result = new Vector<T>(gradient.Length);
            var strength = NumOps.FromDouble(Options.Strength);
            var (start, end) = ClampedRange(gradient.Length);

            // gradient + strength * coefficients inside [start, end), the gradient unchanged elsewhere
            gradient.AsSpan().CopyTo(result.AsWritableSpan());
            if (end > start)
            {
                var regularized = (Vector<T>)AiDotNetEngine.Current.Add(
                    gradient.GetSubVector(start, end - start),
                    (Vector<T>)AiDotNetEngine.Current.Multiply(coefficients.GetSubVector(start, end - start), strength));
                regularized.AsSpan().CopyTo(result.AsWritableSpan().Slice(start));
            }

            return result;
        }

        /// <summary>The regularized half-open index range, clipped to a vector of <paramref name="length"/>.</summary>
        private (int Start, int End) ClampedRange(int length)
        {
            var (start, count) = Range();
            int clampedStart = Math.Min(start, length);
            return (clampedStart, Math.Min(clampedStart + count, length));
        }

        public override Tensor<T> Regularize(Tensor<T> gradient, Tensor<T> coefficients)
        {
            var result = Regularize(gradient.ToVector(), coefficients.ToVector());
            return Tensor<T>.FromVector(result).Reshape(gradient.Shape.ToArray());
        }

        public override Vector<T> Regularize(Vector<T> data)
        {
            var result = new Vector<T>(data.Length);
            var shrinkage = NumOps.Subtract(NumOps.One, NumOps.FromDouble(Options.Strength));
            var (start, end) = ClampedRange(data.Length);

            // data * shrinkage inside [start, end), unchanged elsewhere
            data.AsSpan().CopyTo(result.AsWritableSpan());
            if (end > start)
            {
                var shrunk = (Vector<T>)AiDotNetEngine.Current.Multiply(data.GetSubVector(start, end - start), shrinkage);
                shrunk.AsSpan().CopyTo(result.AsWritableSpan().Slice(start));
            }

            return result;
        }

        public override Matrix<T> Regularize(Matrix<T> data)
        {
            var result = new Matrix<T>(data.Rows, data.Columns);
            var shrinkage = NumOps.Subtract(NumOps.One, NumOps.FromDouble(Options.Strength));
            var (start, end) = ClampedRange(data.Rows * data.Columns);
            // Row-major flat range [start, end) is scaled by shrinkage, the rest copied unchanged
            data.AsSpan().CopyTo(result.AsWritableSpan());
            if (end > start)
            {
                var shrunk = (Vector<T>)AiDotNetEngine.Current.Multiply(new Vector<T>(data.AsSpan().Slice(start, end - start).ToArray()), shrinkage);
                shrunk.AsSpan().CopyTo(result.AsWritableSpan().Slice(start));
            }

            return result;
        }
    }

    /// <summary>
    /// Initializes the layers of the neural network based on the provided architecture.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This method sets up the layers of the graph neural network. If the architecture provides specific layers,
    /// those are used directly. Otherwise, default layers appropriate for a graph neural network are created.
    /// </para>
    /// <para><b>For Beginners:</b> This method sets up the building blocks of your neural network.
    /// 
    /// When initializing layers:
    /// - If you provided specific layers in the architecture, those are used
    /// - If not, the network creates a standard set of layers for graph processing
    /// 
    /// Think of this like assembling the components of the network before training begins.
    /// </para>
    /// </remarks>
    protected override void InitializeLayers()
    {
        if (Architecture.Layers != null && Architecture.Layers.Count > 0)
        {
            Layers.AddRange(Architecture.Layers);
            ValidateCustomLayers(Layers);
        }
        else
        {
            // Per Kipf & Welling 2017, GCN uses softmax output + cross-entropy loss.
            // The default layers include softmax which is needed for the loss to work correctly.
            Layers.AddRange(LayerHelper<T>.CreateDefaultGNNLayers(
                Architecture,
                _options.HiddenSize,
                _options.NumLayers,
                _options.DropoutRate,
                _options.UseBias));
        }
    }

    /// <summary>
    /// Computes the auxiliary loss for graph smoothness regularization.
    /// </summary>
    /// <returns>The computed graph smoothness auxiliary loss.</returns>
    /// <remarks>
    /// <para>
    /// This method computes the graph smoothness loss, which encourages connected nodes
    /// to have similar representations. The loss is computed as the sum of squared differences
    /// between representations of connected nodes, weighted by the adjacency matrix.
    /// Formula: L_smooth = Σ_edges ||h_i - h_j||² * A_{ij}
    /// </para>
    /// <para><b>For Beginners:</b> This calculates how different connected nodes are from each other.
    ///
    /// Graph smoothness works by:
    /// 1. Looking at each pair of connected nodes in the graph
    /// 2. Measuring how different their learned representations are
    /// 3. Penalizing large differences between connected nodes
    /// 4. Summing up these penalties across all edges
    ///
    /// This helps because:
    /// - It encourages the network to respect the graph structure
    /// - Connected nodes learn similar representations
    /// - The network generalizes better to new graph data
    ///
    /// For example, in a social network, friends (connected nodes) will have
    /// similar learned features, which makes sense since friends often share interests.
    /// </para>
    /// </remarks>
    public T ComputeAuxiliaryLoss()
    {
        if (!UseAuxiliaryLoss || _lastNodeRepresentations == null || _lastAdjacencyMatrix == null)
        {
            _lastGraphSmoothnessLoss = NumOps.Zero;
            return NumOps.Zero;
        }

        // Compute graph smoothness loss: Σ_ij A_ij * ||h_i - h_j||²
        // where A is the adjacency matrix and h are node representations
        T smoothnessLoss = NumOps.Zero;
        int numNodes = _lastNodeRepresentations.Shape[0];
        int edgeCount = 0;

        if (numNodes > 0)
        {
            using (new AiDotNet.Tensors.Engines.Autodiff.NoGradScope<T>())
            {
                // Edges are the nonzero adjacency entries (NaN counts as an edge), in row-major order
                var adjacency = Engine.TensorSlice(_lastAdjacencyMatrix, [0, 0], [numNodes, numNodes]);
                var edgeList = Engine.TensorNonzero(adjacency).ToArray();
                edgeCount = edgeList.Length / 2;

                if (edgeCount > 0)
                {
                    var sources = new int[edgeCount];
                    var targets = new int[edgeCount];
                    var flatIndex = new int[edgeCount];
                    for (int e = 0; e < edgeCount; e++)
                    {
                        sources[e] = edgeList[2 * e];
                        targets[e] = edgeList[2 * e + 1];
                        flatIndex[e] = sources[e] * numNodes + targets[e];
                    }

                    // A_ij * ||h_i - h_j||^2 per edge, gathered without forming all pairwise distances
                    var nodes = Engine.Reshape(_lastNodeRepresentations,
                        [numNodes, _lastNodeRepresentations.Length / numNodes]);
                    var diff = Engine.TensorSubtract(
                        Engine.TensorGather(nodes, new Tensor<int>(sources, [edgeCount]), 0),
                        Engine.TensorGather(nodes, new Tensor<int>(targets, [edgeCount]), 0));
                    var squaredDiff = Engine.ReduceSum(Engine.TensorMultiply(diff, diff), new[] { 1 }, keepDims: false);
                    var edgeWeights = Engine.TensorGather(
                        Engine.Reshape(adjacency, [numNodes * numNodes]), new Tensor<int>(flatIndex, [edgeCount]), 0);
                    smoothnessLoss = Engine.TensorSum(Engine.TensorMultiply(edgeWeights, squaredDiff));
                }
            }
        }

        if (edgeCount > 0)
        {
            smoothnessLoss = NumOps.Divide(smoothnessLoss, NumOps.FromDouble(edgeCount));
        }

        _lastGraphSmoothnessLoss = smoothnessLoss;
        return smoothnessLoss;
    }

    /// <summary>
    /// Gets diagnostic information about the graph smoothness auxiliary loss.
    /// </summary>
    /// <returns>A dictionary containing diagnostic information about auxiliary losses.</returns>
    /// <remarks>
    /// <para>
    /// This method returns detailed diagnostics about the graph smoothness regularization, including
    /// the computed smoothness loss, weight applied, and whether the feature is enabled.
    /// This information is useful for monitoring training progress and debugging.
    /// </para>
    /// <para><b>For Beginners:</b> This provides information about how graph smoothness regularization is working.
    ///
    /// The diagnostics include:
    /// - Total smoothness loss (how different connected nodes are)
    /// - Weight applied to the smoothness loss
    /// - Whether smoothness regularization is enabled
    /// - Whether node representations are being tracked
    ///
    /// This helps you:
    /// - Monitor if smoothness regularization is helping training
    /// - Debug issues with graph structure learning
    /// - Understand the impact of smoothness enforcement on learning
    ///
    /// You can use this information to adjust the auxiliary loss weight for better results.
    /// </para>
    /// </remarks>
    public Dictionary<string, string> GetAuxiliaryLossDiagnostics()
    {
        return new Dictionary<string, string>
        {
            { "TotalGraphSmoothnessLoss", _lastGraphSmoothnessLoss?.ToString() ?? "0" },
            { "SmoothnessWeight", AuxiliaryLossWeight?.ToString() ?? "0.05" },
            { "UseGraphSmoothness", UseAuxiliaryLoss.ToString() },
            { "NodeRepresentationsCached", (_lastNodeRepresentations != null).ToString() },
            { "AdjacencyMatrixCached", (_lastAdjacencyMatrix != null).ToString() }
        };
    }

    /// <summary>
    /// Gets diagnostic information about this component's state and behavior.
    /// Overrides <see cref="LayerBase{T}.GetDiagnostics"/> to include auxiliary loss diagnostics.
    /// </summary>
    /// <returns>
    /// A dictionary containing diagnostic metrics including both base layer diagnostics and
    /// auxiliary loss diagnostics from <see cref="GetAuxiliaryLossDiagnostics"/>.
    /// </returns>
    public Dictionary<string, string> GetDiagnostics()
    {
        var diagnostics = new Dictionary<string, string>();

        // Merge auxiliary loss diagnostics
        var auxDiagnostics = GetAuxiliaryLossDiagnostics();
        foreach (var kvp in auxDiagnostics)
        {
            diagnostics[kvp.Key] = kvp.Value;
        }

        return diagnostics;
    }

    /// <summary>
    /// Predicts over the given graph: <see cref="NeuralNetworkBase{T}.Predict"/> with <paramref name="adjacencyMatrix"/>
    /// in place of the graph set through <see cref="SetAdjacencyMatrix"/>, for this call only.
    /// </summary>
    /// <param name="nodeFeatures">Node features, <c>[nodes, features]</c> or <c>[batch, nodes, features]</c>.</param>
    /// <param name="adjacencyMatrix">The <c>[nodes, nodes]</c> matrix every graph layer multiplies by, used as given.</param>
    /// <returns>The network's output: per node, or per graph when the layer stack ends in a readout.</returns>
    /// <exception cref="ArgumentNullException">Thrown when either argument is null.</exception>
    /// <exception cref="ArgumentException">Thrown when the adjacency is not square over the features' node axis.</exception>
    /// <remarks>
    /// <para>
    /// The output comes from the network's own trained layers. This method used to end in a "hybrid pooling" that
    /// built two new, randomly initialized dense layers on every call, so its result was untrained and changed from
    /// call to call; with <c>[nodes, features]</c> input it pooled over the feature axis and threw. A graph-level
    /// readout belongs in the layer stack (a <c>MeanLayer</c> over the node axis, say), where it is trained.
    /// </para>
    /// <para><b>For Beginners:</b> Use this to run the trained network on a graph you have at hand without changing
    /// the graph the network trains on. The adjacency matrix says which nodes are connected; for a Kipf-Welling GCN,
    /// pass its normalized form <c>D^-1/2 (A + I) D^-1/2</c>.
    /// </para>
    /// </remarks>
    public Tensor<T> PredictGraph(Tensor<T> nodeFeatures, Tensor<T> adjacencyMatrix)
    {
        if (nodeFeatures == null || adjacencyMatrix == null)
            throw new ArgumentNullException(nodeFeatures == null ? nameof(nodeFeatures) : nameof(adjacencyMatrix));

        // The adjacency is [nodes, nodes] and the features [nodes, features] or [batch, nodes, features]. This compared
        // the feature width with the node count, so it rejected every graph whose nodes and features differ in number.
        if (adjacencyMatrix.Rank != 2 || adjacencyMatrix.Shape[0] != adjacencyMatrix.Shape[1]
            || nodeFeatures.Rank < 2 || NodeCount(nodeFeatures) != adjacencyMatrix.Shape[0])
            throw new ArgumentException(
                $"Node features [{string.Join(", ", nodeFeatures.Shape.ToArray())}] and adjacency matrix "
                + $"[{string.Join(", ", adjacencyMatrix.Shape.ToArray())}] dimensions are incompatible: the adjacency must be "
                + "[nodes, nodes] over the features' node axis.");

        var trainingGraph = CurrentGraph();
        InstallGraph(adjacencyMatrix);
        try
        {
            var output = Predict(nodeFeatures);

            // Cached for the graph-smoothness diagnostics.
            if (UseAuxiliaryLoss)
            {
                _lastAdjacencyMatrix = adjacencyMatrix;
                _lastNodeRepresentations = output;
            }

            return output;
        }
        finally
        {
            // Back to the graph set through SetAdjacencyMatrix, or to the identity graph when none was set.
            InstallGraph(trainingGraph ?? IdentityGraph(adjacencyMatrix.Shape[0]));
        }
    }

    // UpdateParameters re-sliced the flat vector across Layers by hand -- the base walks
    // exactly the same enumeration, so this said nothing the base does not already say.
    /// <summary>
    /// Performs a forward pass through the network to make a prediction using a standard input tensor.
    /// </summary>
    /// <param name="input">The input tensor to process.</param>
    /// <returns>The output tensor containing the prediction.</returns>
    /// <remarks>
    /// <para>
    /// This method passes the input data through each layer of the network sequentially.
    /// For a GraphNeuralNetwork, this method assumes the input tensor already contains the necessary
    /// node features and adjacency information in a preprocessed format, or that the input is
    /// for a portion of the network that uses standard layers rather than graph-specific ones.
    /// </para>
    /// <para><b>For Beginners:</b> While GNNs work best with explicit graph data
    /// (provided through the PredictGraph method), this method allows the network to
    /// process pre-processed graph data or operate on non-graph portions of the network.
    /// 
    /// It works by:
    /// 1. Starting with your input data
    /// 2. Passing it through each layer of the network in sequence
    /// 3. Letting each layer transform the data based on its specific function
    /// 
    /// The result is a prediction based on the trained network's understanding of graph patterns.
    /// </para>
    /// </remarks>
    protected override Tensor<T> PredictCore(Tensor<T> input)
    {

        // Ensure 2D input [numNodes, features]
        if (input.Rank == 1)
            input = input.Reshape([1, input.Shape[0]]);

        var adjacencyMatrix = EnsureAdjacencyMatrix(NodeCount(input));

        // Set all layers to inference mode
        foreach (var layer in Layers)
        {
            layer.SetTrainingMode(false);
        }

        // Forward pass through all layers with adjacency
        Tensor<T> current = input;
        foreach (var layer in Layers)
        {
            if (layer is IGraphConvolutionLayer<T> graphLayer)
            {
                graphLayer.SetAdjacencyMatrix(adjacencyMatrix);
            }
            current = layer.Forward(current);
        }

        return current;
    }

    /// <summary>
    /// Cached adjacency matrix for forward/backward passes.
    /// </summary>
    [Scratch]
    private Tensor<T>? _autoAdjacencyMatrix;

    /// <summary>
    /// The last graph <see cref="CurrentGraph"/> confirmed is not an identity matrix, so the check runs once per graph.
    /// </summary>
    [Scratch]
    private Tensor<T>? _confirmedGraph;

    /// <summary>
    /// Sets the graph that <see cref="NeuralNetworkBase{T}.Predict"/> and <see cref="NeuralNetworkBase{T}.Train"/>
    /// propagate over, or clears it with <c>null</c>.
    /// </summary>
    /// <param name="adjacencyMatrix">
    /// A square <c>[nodes, nodes]</c> matrix that every graph layer multiplies by, used as given: for a Kipf-Welling
    /// GCN pass the normalized <c>D^-1/2 (A + I) D^-1/2</c>. <c>null</c> returns to the identity graph.
    /// </param>
    /// <exception cref="ArgumentException">The matrix is not square and rank 2.</exception>
    /// <remarks>
    /// <para>
    /// Without a graph, Predict and Train use the identity adjacency, under which a graph layer sees only each
    /// node's own features. <see cref="PredictGraph"/> takes a graph per call, but it is inference only; this is
    /// how the graph reaches training.
    /// </para>
    /// <para>
    /// A clone or a saved and loaded copy keeps the graph.
    /// </para>
    /// <para><b>For Beginners:</b> The adjacency matrix says which nodes are connected. Set it once before training
    /// so that each node learns from its neighbours, not only from itself. Inputs are then <c>[nodes, features]</c>,
    /// or <c>[batch, nodes, features]</c> for several feature sets over the same graph.
    /// </para>
    /// </remarks>
    public void SetAdjacencyMatrix(Tensor<T>? adjacencyMatrix)
    {
        if (adjacencyMatrix is not null
            && (adjacencyMatrix.Rank != 2 || adjacencyMatrix.Shape[0] != adjacencyMatrix.Shape[1]))
        {
            throw new ArgumentException(
                $"The adjacency matrix must be square [nodes, nodes]; got [{string.Join(", ", adjacencyMatrix.Shape)}].",
                nameof(adjacencyMatrix));
        }

        if (adjacencyMatrix is not null)
        {
            InstallGraph(adjacencyMatrix);
            return;
        }

        var current = CurrentGraph();
        if (current is not null)
            InstallGraph(IdentityGraph(current.Shape[0]));
    }

    /// <summary>
    /// The graph the layers hold, or null when they hold none or only the identity graph this network supplies.
    /// </summary>
    /// <remarks>
    /// The graph lives in the graph layers, not in a field here: every graph layer persists its adjacency as
    /// input-sized state, outside the parameter vector, so a clone or a saved and loaded copy carries it. A model
    /// field would have to join the parameter vector to be persisted, which would make ParameterCount depend on
    /// the graph. An identity matrix counts as no graph, since that is what no graph means, which also covers the
    /// fallback identity a clone copies from its source's layers.
    /// </remarks>
    private Tensor<T>? CurrentGraph()
    {
        Tensor<T>? graph = null;
        foreach (var layer in Layers)
        {
            if (layer is IGraphConvolutionLayer<T> graphLayer && graphLayer.GetAdjacencyMatrix() is { } held)
            {
                graph = held;
                break;
            }
        }

        if (graph is null || ReferenceEquals(graph, _autoAdjacencyMatrix)) return null;
        if (ReferenceEquals(graph, _confirmedGraph)) return graph;
        if (IsIdentity(graph)) return null;

        _confirmedGraph = graph;
        return graph;
    }

    private bool IsIdentity(Tensor<T> graph)
    {
        if (graph.Rank != 2 || graph.Shape[0] != graph.Shape[1]) return false;
        int nodes = graph.Shape[0];
        for (int i = 0; i < nodes; i++)
        {
            for (int j = 0; j < nodes; j++)
            {
                if (!NumOps.Equals(graph[i, j], i == j ? NumOps.One : NumOps.Zero)) return false;
            }
        }

        return true;
    }

    private void InstallGraph(Tensor<T> graph)
    {
        foreach (var layer in Layers)
        {
            if (layer is IGraphConvolutionLayer<T> graphLayer)
                graphLayer.SetAdjacencyMatrix(graph);
        }
    }

    // The node axis: [nodes, features] or [batch, nodes, features].
    private static int NodeCount(Tensor<T> input) => input.Rank >= 3 ? input.Shape[input.Rank - 2] : input.Shape[0];

    private Tensor<T> EnsureAdjacencyMatrix(int numNodes)
    {
        var graph = CurrentGraph();
        if (graph is not null)
        {
            if (graph.Shape[0] != numNodes)
            {
                throw new ArgumentException(
                    $"The input has {numNodes} nodes but the adjacency matrix set through SetAdjacencyMatrix covers "
                    + $"{graph.Shape[0]}.");
            }

            return graph;
        }

        return IdentityGraph(numNodes);
    }

    private Tensor<T> IdentityGraph(int numNodes)
    {
        if (_autoAdjacencyMatrix != null && _autoAdjacencyMatrix.Shape[0] == numNodes)
            return _autoAdjacencyMatrix;

        if (numNodes <= 0)
        {
            throw new ArgumentException(
                $"Cannot create adjacency matrix for {numNodes} nodes. Graph must have at least 1 node.",
                nameof(numNodes));
        }

        // No explicit graph was supplied, so use the neutral Kipf-Welling case:
        // A has no off-diagonal edges, A + I is identity, and symmetric
        // normalization leaves identity unchanged. Real graph structure should
        // be provided through SetAdjacencyMatrix (Predict and Train) or PredictGraph.
        var adj = new Tensor<T>([numNodes, numNodes]);
        for (int i = 0; i < numNodes; i++)
            adj[i, i] = NumOps.One;

        _autoAdjacencyMatrix = adj;
        return adj;
    }

    /// <inheritdoc/>
    /// <remarks>A single feature vector is read as a one-node graph.</remarks>
    protected override (Tensor<T> Input, Tensor<T>? Target) PrepareTrainingPair(Tensor<T> input, Tensor<T>? target, long draw)
        => (input.Rank == 1 ? input.Reshape([1, input.Shape[0]]) : input, target);

    /// <inheritdoc/>
    /// <remarks>The graph layers propagate over the adjacency of the step's node count.</remarks>
    protected override void OnTrainingStepStarting(Tensor<T> input, Tensor<T> target)
    {
        base.OnTrainingStepStarting(input, target);
        var adjacencyMatrix = EnsureAdjacencyMatrix(input.Rank == 1 ? 1 : NodeCount(input));
        foreach (var layer in Layers)
        {
            if (layer is IGraphConvolutionLayer<T> graphLayer)
                graphLayer.SetAdjacencyMatrix(adjacencyMatrix);
        }
    }

    public override Tensor<T> ForwardForTraining(Tensor<T> input)
    {
        if (input.Rank == 1)
            input = Engine.Reshape(input, [1, input.Shape[0]]);

        var adjacencyMatrix = EnsureAdjacencyMatrix(NodeCount(input));

        foreach (var layer in Layers)
        {
            if (layer is IGraphConvolutionLayer<T> graphLayer)
                graphLayer.SetAdjacencyMatrix(adjacencyMatrix);
        }

        return base.ForwardForTraining(input);
    }


    /// <summary>
    /// Gets metadata about the Graph Neural Network model.
    /// </summary>
    /// <returns>A ModelMetaData object containing information about the model.</returns>
    /// <remarks>
    /// <para>
    /// This method returns metadata about the Graph Neural Network, including its model type,
    /// number of layers, types of activation functions used, and serialized model data. This information
    /// is useful for model management and serialization.
    /// </para>
    /// <para><b>For Beginners:</b> This method provides a summary of your GNN's configuration.
    /// 
    /// The metadata includes:
    /// - The type of model (GraphNeuralNetwork)
    /// - Details about the network structure (layers, activations)
    /// - Performance metrics
    /// - Data needed to save and load the model
    /// 
    /// This is useful for:
    /// - Keeping track of different models you've created
    /// - Understanding a model's properties
    /// - Saving the model for later use
    /// </para>
    /// </remarks>
    /// <summary>
    /// Gets the intermediate activations from each layer, ensuring adjacency is set for graph layers.
    /// </summary>
    public override Dictionary<string, Tensor<T>> GetNamedLayerActivations(Tensor<T> input)
    {
        if (input.Rank == 1)
            input = input.Reshape([1, input.Shape[0]]);

        var adjacencyMatrix = EnsureAdjacencyMatrix(NodeCount(input));

        foreach (var layer in Layers)
        {
            if (layer is IGraphConvolutionLayer<T> graphLayer)
                graphLayer.SetAdjacencyMatrix(adjacencyMatrix);
        }

        var activations = new Dictionary<string, Tensor<T>>();
        var current = input;
        for (int i = 0; i < Layers.Count; i++)
        {
            current = Layers[i].Forward(current);
            activations[$"Layer_{i}_{Layers[i].GetType().Name}"] = current.Clone();
        }

        return activations;
    }

    /// <inheritdoc />
    public override ModelMetadata<T> GetModelMetadata()
    {
        return new ModelMetadata<T>
        {
            AdditionalInfo = new Dictionary<string, object>
            {
                { "LayerCount", Layers.Count },
                { "GraphLayerCount", Layers.Count(l => l is IGraphConvolutionLayer<T>) },
                { "StandardLayerCount", Layers.Count(l => !(l is IGraphConvolutionLayer<T>)) },
                { "ParameterCount", GetParameterCount() },
                { "ActivationTypes", string.Join(", ", GetActivationTypes()) }
            },
            ModelDataProvider = () => SerializeForMetadata()
        };
    }

    /// <summary>
    /// Gets the types of activation functions used in the network.
    /// </summary>
    /// <returns>A collection of activation function types used in the network.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method identifies what kinds of activation functions
    /// your neural network is using at different stages.
    /// 
    /// Activation functions determine how signals flow through the network and
    /// are a key part of how neural networks learn complex patterns.
    /// </para>
    /// </remarks>
    private IEnumerable<string> GetActivationTypes()
    {
        var activationTypes = new List<string>();

        // Add scalar activations if they exist
        if (_graphConvolutionalScalarActivation != null)
            activationTypes.Add(_graphConvolutionalScalarActivation.GetType().Name);

        if (_activationLayerScalarActivation != null)
            activationTypes.Add(_activationLayerScalarActivation.GetType().Name);

        if (_finalDenseLayerScalarActivation != null)
            activationTypes.Add(_finalDenseLayerScalarActivation.GetType().Name);

        if (_finalActivationLayerScalarActivation != null)
            activationTypes.Add(_finalActivationLayerScalarActivation.GetType().Name);

        // Add vector activations if they exist
        if (_graphConvolutionalVectorActivation != null)
            activationTypes.Add(_graphConvolutionalVectorActivation.GetType().Name);

        if (_activationLayerVectorActivation != null)
            activationTypes.Add(_activationLayerVectorActivation.GetType().Name);

        if (_finalDenseLayerVectorActivation != null)
            activationTypes.Add(_finalDenseLayerVectorActivation.GetType().Name);

        if (_finalActivationLayerVectorActivation != null)
            activationTypes.Add(_finalActivationLayerVectorActivation.GetType().Name);

        return activationTypes.Distinct();
    }
}
