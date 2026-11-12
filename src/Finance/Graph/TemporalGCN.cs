using AiDotNet.LearningRateSchedulers;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Finance.Interfaces;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LossFunctions;
using AiDotNet.Models;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Optimizers;
using AiDotNet.Tensors;
using AiDotNet.Tensors.Helpers;
using Microsoft.ML.OnnxRuntime;
using OnnxTensors = Microsoft.ML.OnnxRuntime.Tensors;

using AiDotNet.Finance.Base;
namespace AiDotNet.Finance.Graph;

/// <summary>
/// TemporalGCN (Temporal Graph Convolutional Network) for time series forecasting on graph-structured data.
/// </summary>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
/// <remarks>
/// <para>
/// TemporalGCN combines graph convolutional networks with recurrent neural networks (GRU/LSTM)
/// to capture both spatial dependencies and temporal dynamics in graph-structured time series.
/// </para>
/// <para><b>For Beginners:</b> TemporalGCN learns two types of patterns simultaneously:
///
/// <b>The Key Insight:</b>
/// Many time series exist on networks: traffic sensors on roads, users in social networks,
/// weather stations across regions. TemporalGCN captures both HOW these entities are connected
/// (spatial) and HOW they change over time (temporal).
///
/// <b>What Problems Does TemporalGCN Solve?</b>
/// - Traffic flow prediction (sensors connected by roads)
/// - Social network activity forecasting (users connected by friendships)
/// - Epidemiological prediction (regions connected geographically)
/// - Financial network analysis (assets connected by correlations)
///
/// <b>How TemporalGCN Works:</b>
/// 1. <b>Graph Convolution:</b> Aggregate information from neighboring nodes
/// 2. <b>Temporal Recurrence:</b> Process time sequences with GRU cells
/// 3. <b>Interleaved Processing:</b> Alternate between spatial and temporal layers
/// 4. <b>Prediction:</b> Output forecasts for all nodes simultaneously
///
/// <b>TemporalGCN Architecture:</b>
/// - GCN Layers: Chebyshev spectral graph convolution for neighbor aggregation
/// - GRU Layers: Gated recurrent units for temporal sequence modeling
/// - Batch Normalization: Stabilizes training across layers
/// - Residual Connections: Helps gradients flow through deep networks
///
/// <b>Key Benefits:</b>
/// - Jointly learns spatial and temporal dependencies
/// - Handles variable graph structures (nodes can have different numbers of neighbors)
/// - Computationally efficient with Chebyshev polynomial approximation
/// - Works with both static and dynamic graphs
/// </para>
/// <para>
/// <b>Reference:</b> Zhao et al., "T-GCN: A Temporal Graph Convolutional Network for Traffic Prediction", 2019.
/// https://arxiv.org/abs/1811.05320
/// </para>
/// </remarks>
/// <example>
/// <code>
/// // Define architecture for temporal graph-based traffic prediction (156 sensors, 12-step horizon)
/// var architecture = new NeuralNetworkArchitecture&lt;double&gt;(
///     inputType: InputType.OneDimensional,
///     taskType: NeuralNetworkTaskType.Regression,
///     inputHeight: 12, inputWidth: 156, inputDepth: 1, outputSize: 12);
///
/// // Training mode: combines GCN spatial aggregation with GRU temporal modeling
/// var model = new TemporalGCN&lt;double&gt;(architecture);
///
/// // ONNX inference mode: load pre-trained T-GCN model
/// var onnxModel = new TemporalGCN&lt;double&gt;(architecture, "tgcn_traffic.onnx");
/// </code>
/// </example>
[ModelDomain(ModelDomain.Finance)]
[ModelDomain(ModelDomain.GraphAnalysis)]
[ModelCategory(ModelCategory.GraphNetwork)]
[ModelCategory(ModelCategory.RecurrentNetwork)]
[ModelTask(ModelTask.Forecasting)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("T-GCN: A Temporal Graph Convolutional Network for Traffic Prediction", "https://arxiv.org/abs/1811.05320", Year = 2020, Authors = "Ling Zhao, Yujiao Song, Chao Zhang, Yu Liu, Pu Wang, Tao Lin, Min Deng, Haifeng Li")]
[PaperOptimizer(OptimizerKind.Adam, LearningRate = 0.001, ReferenceBatchSize = 64,
                Source = "Zhao et al. 2020, Sec. 4: the T-GCN model is trained using the Adam "
                        + "optimizer, with the learning rate set to 0.001 and the batch size to 64.")]
public partial class TemporalGCN<T> : ForecastingModelBase<T>
{
    #region Execution Mode
    private readonly bool _useNativeMode;
    #endregion

    
    #region Native Mode Fields
    private DenseLayer<T>? _inputProjection;
    private List<DenseLayer<T>>? _gcnLayers;
    private List<DenseLayer<T>>? _gruLayers;
    private List<LayerNormalizationLayer<T>>? _layerNorms;
    private List<DropoutLayer<T>>? _dropoutLayers;
    private DenseLayer<T>? _outputLayer;
    #endregion

    #region Graph Fields
    private readonly Tensor<T>? _adjacencyMatrix;
    private readonly Tensor<T>? _normalizedLaplacian;
    private readonly Random _random;
    #endregion

    #region Shared Fields
    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> _optimizer;
    private readonly ILossFunction<T> _lossFunction;
    private readonly TemporalGCNOptions<T> _options;

    private int _sequenceLength;
    private int _forecastHorizon;
    private int _numNodes;
    private int _numFeatures;
    private int _hiddenDimension;
    private int _numGCNLayers;
    private int _numTemporalLayers;
    private int _chebyshevOrder;
    private string _temporalCellType;
    private int _numSamples;
    private bool _useResidualConnections;
    private bool _useBatchNormalization;
    #endregion

    #region IForecastingModel Properties

    /// <inheritdoc/>
    public override int SequenceLength => _sequenceLength;

    /// <inheritdoc/>
    public override int PredictionHorizon => _forecastHorizon;

    /// <inheritdoc/>
    public override int NumFeatures => _numFeatures;

    /// <inheritdoc/>
    public override int PatchSize => 1;

    /// <inheritdoc/>
    public override int Stride => 1;

    /// <inheritdoc/>
    public override bool IsChannelIndependent => false;

    /// <inheritdoc/>
    public override bool UseNativeMode => _useNativeMode;

    /// <summary>
    /// Gets the number of nodes in the graph.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> How many entities/locations are in the spatial network.
    /// Each node has its own time series that gets forecast.
    /// </para>
    /// </remarks>
    public int NumNodes => _numNodes;

    /// <summary>
    /// Gets the forecast horizon.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> How many future time steps to predict for each node.
    /// </para>
    /// </remarks>
    public int ForecastHorizon => _forecastHorizon;

    /// <summary>
    /// Gets whether the model supports training (native mode only).
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> ONNX mode is inference-only.
    /// Native mode supports both training and inference.
    /// </para>
    /// </remarks>
    public override bool SupportsTraining => _useNativeMode;

    /// <summary>
    /// Gets the Chebyshev polynomial order for graph convolution.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Controls how many hops of neighbors are considered.
    /// Higher order means aggregating from farther neighbors.
    /// </para>
    /// </remarks>
    public int ChebyshevOrder => _chebyshevOrder;

    /// <summary>
    /// Gets the type of temporal recurrent cell used.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Either "gru" (simpler) or "lstm" (more capacity).
    /// </para>
    /// </remarks>
    public string TemporalCellType => _temporalCellType;

    /// <summary>
    /// Gets the number of samples for uncertainty estimation.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> For probabilistic forecasting with MC Dropout.
    /// </para>
    /// </remarks>
    public int NumSamples => _numSamples;

    #endregion

    #region Constructors

    /// <summary>
    /// Initializes a new instance of the TemporalGCN model in ONNX mode for inference.
    /// </summary>
    /// <param name="architecture">The neural network architecture configuration.</param>
    /// <param name="onnxModelPath">Path to a pretrained ONNX model file.</param>
    /// <param name="options">TemporalGCN-specific options.</param>
    /// <param name="adjacencyMatrix">Optional pre-defined adjacency matrix for the graph.</param>
    /// <param name="optimizer">Optional optimizer for fine-tuning.</param>
    /// <param name="lossFunction">Optional loss function.</param>
    /// <remarks>
    /// <para><b>For Beginners:</b> Use this constructor to load a pretrained TemporalGCN model.
    /// The ONNX model contains the trained graph convolution and temporal processing layers.
    /// </para>
    /// </remarks>
    public TemporalGCN(
        NeuralNetworkArchitecture<T> architecture,
        string onnxModelPath,
        TemporalGCNOptions<T>? options = null,
        Tensor<T>? adjacencyMatrix = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
        ILossFunction<T>? lossFunction = null)
        : base(architecture, lossFunction ?? new MeanSquaredErrorLoss<T>(), 1.0)
    {
        if (string.IsNullOrWhiteSpace(onnxModelPath))
            throw new ArgumentNullException(nameof(onnxModelPath));
        if (!System.IO.File.Exists(onnxModelPath))
            throw new System.IO.FileNotFoundException($"ONNX model not found: {onnxModelPath}");

        _useNativeMode = false;
        OnnxModelPath = onnxModelPath;
        OnnxSession = new InferenceSession(onnxModelPath);
        _options = options ?? new TemporalGCNOptions<T>();
        Options = _options;
        _lossFunction = lossFunction ?? new MeanSquaredErrorLoss<T>();
        _optimizer = optimizer
    ?? PaperOptimizerFactory.CreateFor<T, Tensor<T>, Tensor<T>>(this)
    ?? new AdamOptimizer<T, Tensor<T>, Tensor<T>>(this);

        _sequenceLength = _options.SequenceLength;
        _forecastHorizon = _options.ForecastHorizon;
        _numNodes = _options.NumNodes;
        _numFeatures = _options.NumFeatures;
        _hiddenDimension = _options.HiddenDimension;
        _numGCNLayers = _options.NumGCNLayers;
        _numTemporalLayers = _options.NumTemporalLayers;
        _chebyshevOrder = _options.ChebyshevOrder;
        _temporalCellType = _options.TemporalCellType;
        _numSamples = _options.NumSamples;
        _useResidualConnections = _options.UseResidualConnections;
        _useBatchNormalization = _options.UseBatchNormalization;

        RequireNodeCount(adjacencyMatrix, _numNodes, nameof(adjacencyMatrix));
        _adjacencyMatrix = adjacencyMatrix ?? CreateDefaultAdjacencyMatrix(_numNodes);
        _normalizedLaplacian = ComputeNormalizedLaplacian(_adjacencyMatrix);
        _random = LayerInitializationSeedScope.CreateComponentRandom(_options.Seed);
    }

    /// <summary>
    /// Initializes a new instance of the TemporalGCN model in native mode for training.
    /// </summary>
    /// <param name="architecture">The neural network architecture configuration.</param>
    /// <param name="options">TemporalGCN-specific options.</param>
    /// <param name="adjacencyMatrix">Optional pre-defined adjacency matrix for the graph.</param>
    /// <param name="optimizer">Optional optimizer for training.</param>
    /// <param name="lossFunction">Optional loss function.</param>
    /// <remarks>
    /// <para><b>For Beginners:</b> Use this constructor to create a TemporalGCN model for training.
    /// Provide an adjacency matrix that defines how nodes are connected. If not provided,
    /// a default graph with nearby connections is used.
    /// </para>
    /// </remarks>
    public TemporalGCN(
        NeuralNetworkArchitecture<T> architecture,
        TemporalGCNOptions<T>? options = null,
        Tensor<T>? adjacencyMatrix = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
        ILossFunction<T>? lossFunction = null)
        : base(architecture, lossFunction ?? new MeanSquaredErrorLoss<T>(), 1.0)
    {
        _useNativeMode = true;
        _options = options ?? new TemporalGCNOptions<T>();
        Options = _options;
        _lossFunction = lossFunction ?? new MeanSquaredErrorLoss<T>();
        _optimizer = optimizer
    ?? PaperOptimizerFactory.CreateFor<T, Tensor<T>, Tensor<T>>(this)
    ?? new AdamOptimizer<T, Tensor<T>, Tensor<T>>(this);

        _sequenceLength = _options.SequenceLength;
        _forecastHorizon = _options.ForecastHorizon;
        _numNodes = _options.NumNodes;
        _numFeatures = _options.NumFeatures;
        _hiddenDimension = _options.HiddenDimension;
        _numGCNLayers = _options.NumGCNLayers;
        _numTemporalLayers = _options.NumTemporalLayers;
        _chebyshevOrder = _options.ChebyshevOrder;
        _temporalCellType = _options.TemporalCellType;
        _numSamples = _options.NumSamples;
        _useResidualConnections = _options.UseResidualConnections;
        _useBatchNormalization = _options.UseBatchNormalization;

        RequireNodeCount(adjacencyMatrix, _numNodes, nameof(adjacencyMatrix));
        _adjacencyMatrix = adjacencyMatrix ?? CreateDefaultAdjacencyMatrix(_numNodes);
        _normalizedLaplacian = ComputeNormalizedLaplacian(_adjacencyMatrix);
        _random = LayerInitializationSeedScope.CreateComponentRandom(_options.Seed);

        InitializeLayers();
    }

    #endregion

    #region Initialization

    /// <summary>
    /// Initializes all layers for the TemporalGCN model.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> This sets up the graph-temporal processing architecture:
    ///
    /// <b>Layer Structure:</b>
    /// 1. Input projection: Maps node features to hidden dimension
    /// 2. GCN layers: Aggregate information from neighboring nodes
    /// 3. GRU layers: Process temporal sequences at each node
    /// 4. Output projection: Maps to forecast dimension
    ///
    /// The key is interleaving spatial (GCN) and temporal (GRU) processing so each
    /// can inform the other.
    /// </para>
    /// </remarks>
    protected override void InitializeLayers()
    {
        if (Architecture.Layers is not null && Architecture.Layers.Count > 0)
        {
            Layers.AddRange(Architecture.Layers);
            ValidateCustomLayersWithOwnershipRollback(Layers);
        }
        else if (_useNativeMode)
        {
            Layers.AddRange(LayerHelper<T>.CreateDefaultTemporalGCNLayers(
                _sequenceLength,
                _forecastHorizon,
                _numNodes,
                _numFeatures,
                _hiddenDimension,
                _numGCNLayers,
                _numTemporalLayers));

            ExtractLayerReferences();
        }
    }

    /// <summary>
    /// Extracts references to key layers for efficient access.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Keeps direct references to the GCN layers, GRU layers,
    /// and output layer for organized processing during forward/backward passes.
    /// </para>
    /// </remarks>
    private void ExtractLayerReferences()
    {
        var allDense = Layers.OfType<DenseLayer<T>>().ToList();
        _layerNorms = Layers.OfType<LayerNormalizationLayer<T>>().ToList();
        _dropoutLayers = Layers.OfType<DropoutLayer<T>>().ToList();

        if (allDense.Count >= 2)
        {
            _inputProjection = allDense[0];

            // GCN layers are the first group after input
            int gcnStart = 1;
            int gcnEnd = gcnStart + _numGCNLayers;
            _gcnLayers = allDense.Skip(gcnStart).Take(_numGCNLayers).ToList();

            // GRU layers follow GCN layers
            int gruStart = gcnEnd;
            _gruLayers = allDense.Skip(gruStart).Take(_numTemporalLayers * 2).ToList();

            _outputLayer = allDense[allDense.Count - 1];
        }
    }

    /// <summary>
    /// Creates a default adjacency matrix for the graph.
    /// </summary>
    /// <param name="numNodes">Number of nodes in the graph.</param>
    /// <returns>Adjacency matrix with proximity-based weights.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> When no graph structure is provided, creates a
    /// default matrix where nearby nodes (by index) are connected. In practice,
    /// you should provide a real adjacency matrix based on actual relationships.
    /// </para>
    /// </remarks>
    private Tensor<T> CreateDefaultAdjacencyMatrix(int numNodes)
    {
        // Self-connections weigh 1; nodes within range = max(1, n / 10) of each other weigh
        // exp(-|i - j| / range); everything else is unconnected.
        using var noGrad = new AiDotNet.Tensors.Engines.Autodiff.NoGradScope<T>();
        int range = Math.Max(1, numNodes / 10);
        var distance = new int[numNodes * numNodes];
        var near = new int[numNodes * numNodes];
        for (int i = 0; i < numNodes; i++)
            for (int j = 0; j < numNodes; j++)
            {
                int d = Math.Abs(i - j);
                distance[i * numNodes + j] = d;
                near[i * numNodes + j] = d <= range ? 1 : 0;
            }
        var square = new[] { numNodes, numNodes };
        var distances = new Tensor<int>(square, new Vector<int>(distance)).Cast<T>();
        var weights = Engine.TensorExp(Engine.TensorDivideScalar(Engine.TensorNegate(distances), NumOps.FromDouble(range)));
        var connected = Engine.TensorWhere(
            new Tensor<int>(square, new Vector<int>(near)).Cast<T>(), weights, Engine.TensorZerosLike(weights));
        return Engine.TensorWhere(Engine.TensorEquals(distances, NumOps.Zero), Engine.TensorFullLike(weights, NumOps.One), connected);
    }

    /// <summary>Requires a numNodes x numNodes adjacency matrix when one is given.</summary>
    private static void RequireNodeCount(Tensor<T>? adjacency, int numNodes, string paramName)
    {
        if (adjacency is null) return;
        int rows = adjacency.Rank > 0 ? adjacency.Shape[0] : 0;
        int columns = adjacency.Rank > 1 ? adjacency.Shape[1] : 0;
        if (adjacency.Rank != 2 || rows != numNodes || columns != numNodes)
        {
            throw new ArgumentException(
                $"The adjacency matrix is {rows} x {columns}, but the model has {numNodes} " +
                $"nodes (NumNodes). Pass a {numNodes} x {numNodes} matrix, or set NumNodes to the node count of the graph.",
                paramName);
        }
    }

    /// <summary>
    /// Computes the normalized graph Laplacian for Chebyshev convolution.
    /// </summary>
    /// <param name="adjacencyMatrix">The adjacency matrix.</param>
    /// <returns>The normalized Laplacian matrix.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> The Laplacian L = D - A is transformed to a normalized
    /// form that enables efficient spectral graph convolution. The eigenvalues of the
    /// normalized Laplacian are bounded in [-1, 1], which is ideal for Chebyshev polynomials.
    ///
    /// Formula: L_norm = 2 * (D^(-1/2) * L * D^(-1/2)) / lambda_max - I
    /// </para>
    /// </remarks>
    private Tensor<T> ComputeNormalizedLaplacian(Tensor<T> adjacencyMatrix)
    {
        // I - D^(-1/2) A D^(-1/2), then minus I (lambda_max taken as 2): the diagonal is 0 and an
        // off-diagonal entry is -a_ij / sqrt(d_i) / sqrt(d_j) where both degrees are positive, else 0.
        using var noGrad = new AiDotNet.Tensors.Engines.Autodiff.NoGradScope<T>();
        int n = adjacencyMatrix.Shape[0];
        var square = new[] { n, n };
        var adjacency = Engine.Reshape(adjacencyMatrix.Contiguous(), square);
        var degree = Engine.ReduceSum(adjacency, new[] { 1 }, false);
        var positive = Engine.TensorGreaterThan(degree, NumOps.Zero);
        var safeDegree = Engine.TensorWhere(positive, degree, Engine.TensorFullLike(degree, NumOps.One));
        var invSqrt = Engine.TensorDivide(Engine.TensorFullLike(degree, NumOps.One), Engine.TensorSqrt(safeDegree));
        var rowScale = Engine.TensorBroadcastTo(Engine.Reshape(invSqrt, new[] { n, 1 }), square);
        var columnScale = Engine.TensorBroadcastTo(Engine.Reshape(invSqrt, new[] { 1, n }), square);
        var scaled = Engine.TensorMultiply(Engine.TensorMultiply(Engine.TensorNegate(adjacency), rowScale), columnScale);
        var rowPositive = Engine.TensorBroadcastTo(Engine.Reshape(positive, new[] { n, 1 }), square);
        var columnPositive = Engine.TensorBroadcastTo(Engine.Reshape(positive, new[] { 1, n }), square);
        var bothPositive = Engine.TensorMultiply(rowPositive, columnPositive);
        var offDiagonal = Engine.TensorSubtract(Engine.TensorFullLike(scaled, NumOps.One), Engine.TensorEye<T>(n));
        return Engine.TensorWhere(Engine.TensorMultiply(bothPositive, offDiagonal), scaled, Engine.TensorZerosLike(scaled));
    }

    /// <summary>
    /// Validates custom layers provided by the user.
    /// </summary>
    /// <param name="layers">The list of custom layers.</param>
    /// <remarks>
    /// <para><b>For Beginners:</b> Ensures custom layers form a valid TemporalGCN architecture
    /// with sufficient depth for spatial and temporal processing.
    /// </para>
    /// </remarks>
    protected override void ValidateCustomLayers(List<ILayer<T>> layers)
    {
        base.ValidateCustomLayers(layers);

        if (layers.Count < 4)
            throw new ArgumentException("TemporalGCN requires at least 4 layers (input, GCN, GRU, output).");
    }

    #endregion

    #region NeuralNetworkBase Overrides

    /// <summary>
    /// Performs forward prediction on the input tensor.
    /// </summary>
    /// <param name="input">Input tensor with shape [nodes, sequence, features].</param>
    /// <returns>Output tensor with forecasts for all nodes.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Processes graph-structured input through:
    /// 1. Project input features to hidden dimension
    /// 2. Apply GCN layers for spatial aggregation
    /// 3. Apply GRU layers for temporal processing
    /// 4. Project to forecast dimension
    /// </para>
    /// </remarks>
    protected override Tensor<T> PredictCore(Tensor<T> input)
    {
        return _useNativeMode ? ForecastNative(input) : ForecastOnnx(input);
    }

    /// <summary>
    /// Trains the TemporalGCN model on a batch of input-target pairs.
    /// </summary>
    /// <param name="input">Input tensor with historical data for all nodes.</param>
    /// <param name="target">Target tensor with future values for all nodes.</param>
    /// <remarks>
    /// <para><b>For Beginners:</b> Training minimizes prediction error across all nodes:
    /// 1. Forward pass through GCN and GRU layers
    /// 2. Compute loss between predicted and actual future values
    /// 3. Backpropagate through both spatial and temporal paths
    /// 4. Update network parameters
    ///
    /// The graph structure guides how information flows spatially during training.
    /// </para>
    /// </remarks>
    /// <summary>
    /// Training-mode forward: replicates <see cref="Forward"/> but
    /// routes through a tape-aware Chebyshev convolution so gradients
    /// can propagate back through the dense layers. The default
    /// <see cref="ApplyChebyshevConvolution"/> serializes features via
    /// <c>ToVector()</c> and rebuilds a new tensor from a
    /// <c>Vector&lt;T&gt;</c>, which severs the gradient tape.
    /// </summary>
    protected override Tensor<T> ForwardNativeForTraining(Tensor<T> input)
    {
        var current = ReshapeToNodes(input);
        foreach (var layer in Layers)
            current = layer.Forward(current);
        if (_normalizedLaplacian is not null && _useNativeMode)
            current = ApplyChebyshevConvolutionTape(current);
        return current;
    }

    /// <summary>
    /// Reshapes the input to [numNodes, featuresPerNode] (numNodes as the batch
    /// dimension) via the tape-aware Engine.Reshape so the per-node MLP layers
    /// apply SHARED weights across nodes and gradients flow back to the input.
    /// </summary>
    private Tensor<T> ReshapeToNodes(Tensor<T> input)
    {
        int totalSize = input.Length;
        int featuresPerNode = totalSize / _numNodes;
        return featuresPerNode * _numNodes == totalSize
            ? Engine.Reshape(input, new[] { _numNodes, featuresPerNode })
            : FlattenInput(input);
    }

    /// <summary>
    /// Tape-aware Chebyshev (K=2) graph filter:
    /// <c>y = 0.5 * x + 0.5 * L x</c>. Uses
    /// <see cref="IEngine.TensorMatMul"/> instead of the manual
    /// double-accumulator loop in
    /// <see cref="ApplyChebyshevConvolution"/>, so the gradient tape
    /// sees every operation and can flow gradients back through the
    /// preceding dense layers.
    /// </summary>
    private Tensor<T> ApplyChebyshevConvolutionTape(Tensor<T> nodeFeatures)
    {
        if (_normalizedLaplacian is null)
            return nodeFeatures;

        int totalSize = nodeFeatures.Length;
        int featuresPerNode = totalSize / _numNodes;
        if (featuresPerNode * _numNodes != totalSize)
            return nodeFeatures; // graceful no-op on shape mismatch

        // Build the Laplacian as a Tensor<T> on demand. This is a
        // constant (non-trainable) tensor; we reconstruct it every
        // training step because _normalizedLaplacian is double[,] and
        // caching a Tensor<T> copy would need threading a mutable
        // field. The cost is O(numNodes^2) per step, which is small
        // relative to a transformer-block forward.
        var laplacian = _normalizedLaplacian;

        // Reshape features to [numNodes, featuresPerNode] via the
        // tape-aware Engine.Reshape so the eventual output is still
        // connected to the layer stack on backward.
        var xMat = Engine.Reshape(nodeFeatures, new[] { _numNodes, featuresPerNode });
        var t1 = Engine.TensorMatMul(laplacian, xMat);
        var combined = Engine.TensorAdd(xMat, t1);
        combined = Engine.TensorMultiplyScalar(combined, NumOps.FromDouble(0.5));
        // Reshape back to the input's original shape so downstream
        // layout expectations (e.g. rank-1 per-step output) are
        // preserved.
        return Engine.Reshape(combined, nodeFeatures._shape);
    }

    // UpdateParameters was an empty override, silently dropping every restore. The base
    // distributes the vector over the declared enumeration.
    /// <summary>
    /// Gets metadata about the TemporalGCN model.
    /// </summary>
    /// <returns>ModelMetadata containing model information.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the TemporalGCN model, GetModelMetadata performs a supporting step in the workflow. It keeps the TemporalGCN architecture pipeline consistent.
    /// </para>
    /// </remarks>
    public override ModelMetadata<T> GetModelMetadata()
    {
        return new ModelMetadata<T>
        {
            AdditionalInfo = new Dictionary<string, object>
            {
                { "NetworkType", "TemporalGCN" },
                { "SequenceLength", _sequenceLength },
                { "ForecastHorizon", _forecastHorizon },
                { "NumNodes", _numNodes },
                { "NumFeatures", _numFeatures },
                { "HiddenDimension", _hiddenDimension },
                { "NumGCNLayers", _numGCNLayers },
                { "NumTemporalLayers", _numTemporalLayers },
                { "ChebyshevOrder", _chebyshevOrder },
                { "TemporalCellType", _temporalCellType },
                { "UseResidualConnections", _useResidualConnections },
                { "UseBatchNormalization", _useBatchNormalization },
                { "NumSamples", _numSamples },
                { "UseNativeMode", _useNativeMode }
            }
        };
    }

    /// <summary>
    /// Serializes TemporalGCN-specific data.
    /// </summary>
    /// <param name="writer">The binary writer.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the TemporalGCN model, SerializeNetworkSpecificData saves or restores model-specific settings. This lets the TemporalGCN architecture be reused later.
    /// </para>
    /// </remarks>


    /// <summary>
    /// Deserializes TemporalGCN-specific data.
    /// </summary>
    /// <param name="reader">The binary reader.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the TemporalGCN model, DeserializeNetworkSpecificData saves or restores model-specific settings. This lets the TemporalGCN architecture be reused later.
    /// </para>
    /// </remarks>


    #endregion

    #region IForecastingModel Implementation

    /// <summary>
    /// Generates forecasts for all nodes in the graph.
    /// </summary>
    /// <param name="historicalData">Input tensor with historical data for all nodes.</param>
    /// <param name="quantiles">Optional quantile levels for probabilistic forecasting.</param>
    /// <returns>Forecast tensor with predictions for all nodes.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Predicts future values for every node simultaneously.
    /// The GCN layers allow nodes to share information with their neighbors, improving
    /// predictions especially when some nodes have sparse or noisy data.
    /// </para>
    /// </remarks>
    protected override Tensor<T> ForecastCore(Tensor<T> historicalData, double[]? quantiles)
    {
        if (quantiles is not null && quantiles.Length > 0)
        {
            var samples = GenerateSamples(historicalData, _numSamples);
            return ComputeQuantiles(samples, quantiles);
        }

        return _useNativeMode ? ForecastNative(historicalData) : ForecastOnnx(historicalData);
    }

    /// <summary>
    /// Generates forecasts with prediction intervals.
    /// </summary>
    /// <param name="input">Input tensor.</param>
    /// <param name="confidenceLevel">Confidence level (e.g., 0.95).</param>
    /// <returns>Tuple of (forecast, lower bound, upper bound).</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the TemporalGCN model, public performs a supporting step in the workflow. It keeps the TemporalGCN architecture pipeline consistent.
    /// </para>
    /// </remarks>
    public (Tensor<T> Forecast, Tensor<T> Lower, Tensor<T> Upper) ForecastWithIntervals(
        Tensor<T> input,
        double confidenceLevel = 0.95)
    {
        var samples = GenerateSamples(input, _numSamples);
        return ComputePredictionIntervals(samples, confidenceLevel);
    }

    /// <summary>
    /// Performs autoregressive forecasting for extended horizons.
    /// </summary>
    /// <param name="input">Initial input.</param>
    /// <param name="steps">Number of forecast steps.</param>
    /// <returns>Extended forecast tensor.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the TemporalGCN model, AutoregressiveForecast produces predictions from input data. This is the main inference step of the TemporalGCN architecture.
    /// </para>
    /// </remarks>
    public override Tensor<T> AutoregressiveForecast(Tensor<T> input, int steps)
    {
        var predictions = new List<Tensor<T>>();
        var currentInput = input;

        for (int i = 0; i < steps; i++)
        {
            var prediction = Forecast(currentInput, null);
            predictions.Add(prediction);
            currentInput = ShiftInputWindow(currentInput, prediction);
        }

        return ConcatenatePredictions(predictions);
    }

    /// <summary>
    /// Evaluates forecast quality against actual values.
    /// </summary>
    /// <param name="predictions">Predicted values for all nodes.</param>
    /// <param name="actuals">Actual values for all nodes.</param>
    /// <returns>Dictionary of evaluation metrics.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the TemporalGCN model, Evaluate performs a supporting step in the workflow. It keeps the TemporalGCN architecture pipeline consistent.
    /// </para>
    /// </remarks>
    public override Dictionary<string, T> Evaluate(Tensor<T> predictions, Tensor<T> actuals)
    {
        var metrics = new Dictionary<string, T>();

        var (mse, mae, count) = SumSquaredAndAbsoluteError(predictions, actuals);

        if (count > 0)
        {
            mse = NumOps.Divide(mse, NumOps.FromDouble(count));
            mae = NumOps.Divide(mae, NumOps.FromDouble(count));
        }

        T rmse = NumOps.Sqrt(mse);

        metrics["MSE"] = mse;
        metrics["MAE"] = mae;
        metrics["RMSE"] = rmse;

        return metrics;
    }

    /// <summary>
    /// Applies instance normalization (identity for TemporalGCN).
    /// </summary>
    /// <param name="input">Input tensor.</param>
    /// <returns>The input tensor unchanged.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the TemporalGCN model, ApplyInstanceNormalization performs a supporting step in the workflow. It keeps the TemporalGCN architecture pipeline consistent.
    /// </para>
    /// </remarks>
    public override Tensor<T> ApplyInstanceNormalization(Tensor<T> input)
    {
        return input;
    }

    /// <summary>
    /// Gets financial-specific metrics.
    /// </summary>
    /// <returns>Dictionary of financial metrics.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the TemporalGCN model, GetFinancialMetrics calculates evaluation metrics. This summarizes how the TemporalGCN architecture is performing.
    /// </para>
    /// </remarks>
    public override Dictionary<string, T> GetFinancialMetrics()
    {
        T lastLoss = LastLoss is not null ? LastLoss : NumOps.Zero;

        return new Dictionary<string, T>
        {
            ["LastLoss"] = lastLoss,
            ["NumNodes"] = NumOps.FromDouble(_numNodes),
            ["SequenceLength"] = NumOps.FromDouble(_sequenceLength),
            ["ForecastHorizon"] = NumOps.FromDouble(_forecastHorizon),
            ["NumGCNLayers"] = NumOps.FromDouble(_numGCNLayers),
            ["NumTemporalLayers"] = NumOps.FromDouble(_numTemporalLayers),
            ["ChebyshevOrder"] = NumOps.FromDouble(_chebyshevOrder)
        };
    }

    #endregion

    #region Forward/Backward Pass

    /// <summary>
    /// Performs the forward pass through all layers.
    /// </summary>
    /// <param name="input">Input tensor.</param>
    /// <returns>Output tensor.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> The forward pass:
    /// 1. Projects input to hidden dimension
    /// 2. Applies GCN layers with Chebyshev convolution
    /// 3. Applies GRU layers for temporal modeling
    /// 4. Projects to output dimension
    /// </para>
    /// </remarks>
    public Tensor<T> Forward(Tensor<T> input)
    {
        var current = ReshapeToNodes(input);

        // Per-node MLP stack (shared weights across nodes).
        foreach (var layer in Layers)
        {
            current = layer.Forward(current);
        }

        // Chebyshev graph convolution (spatial aggregation). Use the tape-aware
        // variant so inference matches the training forward exactly (determinism
        // and Clone-parity invariants).
        if (_normalizedLaplacian is not null && _useNativeMode)
        {
            current = ApplyChebyshevConvolutionTape(current);
        }

        return current;
    }

    /// <summary>
    /// Flattens input tensor for dense layer processing.
    /// </summary>
    /// <param name="input">Input tensor.</param>
    /// <returns>Flattened tensor.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the TemporalGCN model, FlattenInput performs a supporting step in the workflow. It keeps the TemporalGCN architecture pipeline consistent.
    /// </para>
    /// </remarks>
    private Tensor<T> FlattenInput(Tensor<T> input)
    {
        int totalSize = 1;
        foreach (var dim in input._shape)
        {
            totalSize *= dim;
        }

        var flattened = new Tensor<T>(new[] { totalSize });
        input.Data.Span.Slice(0, totalSize).CopyTo(flattened.Data.Span);

        return flattened;
    }

    /// <summary>
    /// Applies Chebyshev spectral graph convolution.
    /// </summary>
    /// <param name="nodeFeatures">Node feature tensor.</param>
    /// <returns>Convolved features after spectral graph filtering.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Chebyshev convolution approximates spectral graph filtering
    /// using Chebyshev polynomials. Instead of computing expensive eigendecomposition,
    /// it uses recursive polynomial terms:
    ///
    /// T_0(L) = I (identity)
    /// T_1(L) = L (Laplacian)
    /// T_k(L) = 2*L*T_{k-1}(L) - T_{k-2}(L)
    ///
    /// The output is a weighted sum of these polynomial terms, where weights are learned.
    /// Higher-order polynomials capture longer-range spatial dependencies.
    /// </para>
    /// </remarks>
    private Tensor<T> ApplyChebyshevConvolution(Tensor<T> nodeFeatures)
    {
        if (_normalizedLaplacian is null)
            return nodeFeatures;

        int totalSize = nodeFeatures.Length;
        int featuresPerNode = totalSize / _numNodes;

        if (featuresPerNode * _numNodes != totalSize)
            return nodeFeatures;

        // Simplified Chebyshev convolution (K=2): 0.5 * x + 0.5 * (L x), detached as before.
        using var noGrad = new AiDotNet.Tensors.Engines.Autodiff.NoGradScope<T>();
        var half = NumOps.FromDouble(0.5);
        var x = Engine.Reshape(nodeFeatures.Contiguous(), new[] { _numNodes, featuresPerNode });
        var combined = Engine.TensorAdd(
            Engine.TensorMultiplyScalar(x, half), Engine.TensorMultiplyScalar(Engine.TensorMatMul(_normalizedLaplacian, x), half));
        return Engine.Reshape(combined, nodeFeatures._shape);
    }

    #endregion

    #region Forecasting Methods

    /// <summary>
    /// Performs native mode forecasting.
    /// </summary>
    /// <param name="context">Input context tensor.</param>
    /// <returns>Forecast tensor.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the TemporalGCN model, ForecastNative produces predictions from input data. This is the main inference step of the TemporalGCN architecture.
    /// </para>
    /// </remarks>
    private Tensor<T> ForecastNative(Tensor<T> context)
    {
        SetTrainingMode(false);
        return Forward(context);
    }

    /// <summary>
    /// Performs ONNX mode forecasting.
    /// </summary>
    /// <param name="input">Input tensor.</param>
    /// <returns>Forecast tensor.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the TemporalGCN model, ForecastOnnx produces predictions from input data. This is the main inference step of the TemporalGCN architecture.
    /// </para>
    /// </remarks>
    protected override Tensor<T> ForecastOnnx(Tensor<T> input)
    {
        if (OnnxSession is null)
            throw new InvalidOperationException("ONNX session not initialized.");

        var flatInput = FlattenInput(input);
        var inputData = new float[flatInput.Data.Length];
        NumOps.ToFloatSpan(flatInput.Data.Span.Slice(0, flatInput.Data.Length), inputData.AsSpan(0, flatInput.Data.Length));

        var inputTensor = new OnnxTensors.DenseTensor<float>(
            inputData,
            new[] { 1, _numNodes, _sequenceLength, _numFeatures });

        var inputs = new List<NamedOnnxValue>
        {
            NamedOnnxValue.CreateFromTensor("input", inputTensor)
        };

        using var results = OnnxSession.Run(inputs);
        var outputTensor = results.First().AsTensor<float>();

        var outputData = new T[outputTensor.Length];
        NumOps.FromFloatSpan(outputTensor.ToDenseTensor().Buffer.Span.Slice(0, (int)outputTensor.Length), outputData.AsSpan(0, (int)outputTensor.Length));

        return new Tensor<T>(new[] { _numNodes * _forecastHorizon }, new Vector<T>(outputData));
    }

    /// <summary>
    /// Generates multiple forecast samples using MC Dropout.
    /// </summary>
    /// <param name="context">Input context.</param>
    /// <param name="numSamples">Number of samples to generate.</param>
    /// <returns>List of forecast samples.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Monte Carlo Dropout keeps dropout active during inference.
    /// Each forward pass gives a slightly different result, capturing model uncertainty.
    /// </para>
    /// </remarks>
    private List<Tensor<T>> GenerateSamples(Tensor<T> context, int numSamples)
    {
        var samples = new List<Tensor<T>>();

        for (int i = 0; i < numSamples; i++)
        {
            var perturbedInput = AddSmallPerturbation(context);
            samples.Add(ForecastNative(perturbedInput));
        }

        return samples;
    }

    /// <summary>
    /// Adds small random perturbation to input for sample diversity.
    /// </summary>
    /// <param name="input">Input tensor.</param>
    /// <returns>Perturbed tensor.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the TemporalGCN model, AddSmallPerturbation performs a supporting step in the workflow. It keeps the TemporalGCN architecture pipeline consistent.
    /// </para>
    /// </remarks>
    private Tensor<T> AddSmallPerturbation(Tensor<T> input)
    {
        var inputVec = input.ToVector();
        var perturbed = new T[inputVec.Length];

        for (int i = 0; i < inputVec.Length; i++)
        {
            double val = NumOps.ToDouble(inputVec[i]);
            double noise = (_random.NextDouble() - 0.5) * 0.01 * Math.Abs(val + 1e-6);
            perturbed[i] = NumOps.FromDouble(val + noise);
        }

        return new Tensor<T>(input._shape, new Vector<T>(perturbed));
    }

    #endregion

    #region Helper Methods

    /// <summary>
    /// Shifts input window by appending new prediction.
    /// </summary>
    /// <param name="input">Current input.</param>
    /// <param name="prediction">New prediction to append.</param>
    /// <returns>Shifted input.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the TemporalGCN model, ShiftInputWindow performs a supporting step in the workflow. It keeps the TemporalGCN architecture pipeline consistent.
    /// </para>
    /// </remarks>
    private Tensor<T> ShiftInputWindow(Tensor<T> input, Tensor<T> prediction)
    {
        // [input after the first shiftSize values | leading predictions | zeros], detached.
        using var noGrad = new AiDotNet.Tensors.Engines.Autodiff.NoGradScope<T>();
        int inputSize = input.Length;
        int shiftSize = Math.Min(_numNodes * _numFeatures, inputSize);
        int copyLen = Math.Max(0, inputSize - shiftSize);
        int fill = Math.Max(0, Math.Min(shiftSize, prediction.Length));
        var parts = new List<Tensor<T>>(3);
        if (copyLen > 0)
            parts.Add(Engine.TensorNarrow(Engine.Reshape(input.Contiguous(), new[] { inputSize }), 0, inputSize - copyLen, copyLen));
        if (fill > 0)
            parts.Add(Engine.TensorNarrow(Engine.Reshape(prediction.Contiguous(), new[] { prediction.Length }), 0, 0, fill));
        if (inputSize - copyLen - fill > 0)
            parts.Add(new Tensor<T>(new[] { inputSize - copyLen - fill }));
        if (parts.Count == 0)
            return new Tensor<T>(new[] { inputSize });
        return parts.Count == 1 ? parts[0].Contiguous() : Engine.TensorConcatenate(parts.Select(p => p.Contiguous()).ToArray(), 0);
    }

    /// <summary>
    /// Concatenates multiple predictions.
    /// </summary>
    /// <param name="predictions">List of predictions.</param>
    /// <returns>Concatenated tensor.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the TemporalGCN model, ConcatenatePredictions produces predictions from input data. This is the main inference step of the TemporalGCN architecture.
    /// </para>
    /// </remarks>
        protected Tensor<T> ConcatenatePredictions(List<Tensor<T>> predictions)
    {
        int totalLen = predictions.Sum(p => p.ToVector().Length);
        var result = new T[totalLen];
        int offset = 0;

        foreach (var pred in predictions)
        {
            var predVec = pred.ToVector();
            for (int i = 0; i < predVec.Length; i++)
            {
                result[offset + i] = predVec[i];
            }
            offset += predVec.Length;
        }

        return new Tensor<T>(new[] { totalLen }, new Vector<T>(result));
    }

    /// <summary>
    /// Computes quantiles from samples.
    /// </summary>
    /// <param name="samples">List of samples.</param>
    /// <param name="quantiles">Quantile levels.</param>
    /// <returns>Quantile tensor.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the TemporalGCN model, ComputeQuantiles performs a supporting step in the workflow. It keeps the TemporalGCN architecture pipeline consistent.
    /// </para>
    /// </remarks>
    private Tensor<T> ComputeQuantiles(List<Tensor<T>> samples, double[] quantiles)
    {
        if (samples.Count == 0)
            return new Tensor<T>(new[] { 0 });

        int len = samples[0].ToVector().Length;
        using var noGrad = new AiDotNet.Tensors.Engines.Autodiff.NoGradScope<T>();

        // One row per sample, sorted down each position; quantile q reads row (int)(q * (n - 1)).
        var rows = new Tensor<T>[samples.Count];
        for (int s = 0; s < samples.Count; s++)
            rows[s] = Engine.TensorNarrow(Engine.Reshape(samples[s], new[] { samples[s].Length }), 0, 0, len);
        var grid = Engine.TensorStack(rows, 0);

        if (Engine.TensorCountNonzero(Engine.TensorIsNan(grid)) > 0)
        {
            // NaN sorts first under the default double comparer; keep that exact ordering.
            var result = new T[len * quantiles.Length];
            for (int pos = 0; pos < len; pos++)
            {
                var values = samples.Select(s => NumOps.ToDouble(s.ToVector()[pos])).OrderBy(v => v).ToList();

                for (int q = 0; q < quantiles.Length; q++)
                {
                    int idx = (int)(quantiles[q] * (values.Count - 1));
                    result[q * len + pos] = NumOps.FromDouble(values[idx]);
                }
            }

            return new Tensor<T>(new[] { quantiles.Length, len }, new Vector<T>(result));
        }

        var sorted = Engine.TensorSort(grid, 0, false).Values;
        var picks = new int[quantiles.Length];
        for (int q = 0; q < quantiles.Length; q++)
            picks[q] = (int)(quantiles[q] * (samples.Count - 1));

        return Engine.TensorIndexSelect(sorted, new Tensor<int>(new[] { picks.Length }, new Vector<int>(picks)), 0);
    }

    /// <summary>
    /// Computes prediction intervals from samples.
    /// </summary>
    /// <param name="samples">List of samples.</param>
    /// <param name="confidenceLevel">Confidence level.</param>
    /// <returns>Tuple of (median, lower, upper).</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the TemporalGCN model, private performs a supporting step in the workflow. It keeps the TemporalGCN architecture pipeline consistent.
    /// </para>
    /// </remarks>
    private (Tensor<T> Forecast, Tensor<T> Lower, Tensor<T> Upper) ComputePredictionIntervals(
        List<Tensor<T>> samples,
        double confidenceLevel)
    {
        double alpha = 1 - confidenceLevel;
        double[] quantiles = { alpha / 2, 0.5, 1 - alpha / 2 };

        var quantileResult = ComputeQuantiles(samples, quantiles);
        var resultVec = quantileResult.ToVector();

        int len = samples[0].ToVector().Length;
        var flat = Tensor<T>.FromVector(resultVec);

        // Rows 0, 1 and 2 of the [quantiles, len] block are the lower, median and upper bounds.
        return (
            Engine.TensorNarrow(flat, 0, len, len).Contiguous(),
            Engine.TensorNarrow(flat, 0, 0, len).Contiguous(),
            Engine.TensorNarrow(flat, 0, 2 * len, len).Contiguous()
        );
    }

    #endregion

    #region IDisposable

    /// <summary>
    /// Disposes of resources used by the model.
    /// </summary>
    /// <param name="disposing">True if disposing managed resources.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the TemporalGCN model, Dispose performs a supporting step in the workflow. It keeps the TemporalGCN architecture pipeline consistent.
    /// </para>
    /// </remarks>
    protected override void Dispose(bool disposing)
    {
        if (disposing)
        {
            OnnxSession?.Dispose();
        }
        base.Dispose(disposing);
    }

    #endregion
}



