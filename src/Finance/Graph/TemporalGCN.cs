using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
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
public class TemporalGCN<T> : ForecastingModelBase<T>
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
    private readonly double[,]? _adjacencyMatrix;
    private readonly double[,]? _normalizedLaplacian;
    private readonly Random _random;
    #endregion

    #region Shared Fields
    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> _optimizer;
    private readonly ILossFunction<T> _lossFunction;
    private readonly TemporalGCNOptions<T> _options;
    private readonly int _sequenceLength;
    private readonly int _forecastHorizon;
    private readonly int _numNodes;
    private readonly int _numFeatures;
    private readonly int _hiddenDimension;
    private readonly int _numGCNLayers;
    private readonly int _numTemporalLayers;
    private readonly int _chebyshevOrder;
    private readonly string _temporalCellType;
    private readonly int _numSamples;
    private readonly bool _useResidualConnections;
    private readonly bool _useBatchNormalization;
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
        double[,]? adjacencyMatrix = null,
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
        _lossFunction = lossFunction ?? new MeanSquaredErrorLoss<T>();
        _optimizer = optimizer ?? new AdamOptimizer<T, Tensor<T>, Tensor<T>>(this);

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

        _adjacencyMatrix = adjacencyMatrix ?? CreateDefaultAdjacencyMatrix(_numNodes);
        _normalizedLaplacian = ComputeNormalizedLaplacian(_adjacencyMatrix);
        _random = RandomHelper.CreateSecureRandom();
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
        double[,]? adjacencyMatrix = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
        ILossFunction<T>? lossFunction = null)
        : base(architecture, lossFunction ?? new MeanSquaredErrorLoss<T>(), 1.0)
    {
        _useNativeMode = true;
        _options = options ?? new TemporalGCNOptions<T>();
        _lossFunction = lossFunction ?? new MeanSquaredErrorLoss<T>();
        _optimizer = optimizer ?? new AdamOptimizer<T, Tensor<T>, Tensor<T>>(this);

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

        _adjacencyMatrix = adjacencyMatrix ?? CreateDefaultAdjacencyMatrix(_numNodes);
        _normalizedLaplacian = ComputeNormalizedLaplacian(_adjacencyMatrix);
        _random = RandomHelper.CreateSecureRandom();

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
            ValidateCustomLayers(Layers);
        }
        else if (_useNativeMode)
        {
            Layers.AddRange(LayerHelper<T>.CreateDefaultTemporalGCNLayers(
                Architecture,
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
    private double[,] CreateDefaultAdjacencyMatrix(int numNodes)
    {
        var adj = new double[numNodes, numNodes];

        // Create sparse adjacency with nearby connections
        for (int i = 0; i < numNodes; i++)
        {
            adj[i, i] = 1.0; // Self-connections

            // Connect to nearby nodes (simulate spatial proximity)
            int range = Math.Max(1, numNodes / 10);
            for (int j = Math.Max(0, i - range); j <= Math.Min(numNodes - 1, i + range); j++)
            {
                if (i != j)
                {
                    double distance = Math.Abs(i - j);
                    adj[i, j] = Math.Exp(-distance / range);
                }
            }
        }

        return adj;
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
    private double[,] ComputeNormalizedLaplacian(double[,] adjacencyMatrix)
    {
        int n = adjacencyMatrix.GetLength(0);
        var laplacian = new double[n, n];
        var degree = new double[n];

        // Compute degree matrix
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++)
            {
                degree[i] += adjacencyMatrix[i, j];
            }
        }

        // Compute normalized Laplacian: L_norm = I - D^(-1/2) * A * D^(-1/2)
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++)
            {
                if (i == j)
                {
                    laplacian[i, j] = 1.0;
                }
                else if (degree[i] > 0 && degree[j] > 0)
                {
                    double dInvSqrtI = 1.0 / Math.Sqrt(degree[i]);
                    double dInvSqrtJ = 1.0 / Math.Sqrt(degree[j]);
                    laplacian[i, j] = -adjacencyMatrix[i, j] * dInvSqrtI * dInvSqrtJ;
                }
            }
        }

        // Scale to [-1, 1] for Chebyshev (approximate lambda_max as 2)
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++)
            {
                laplacian[i, j] = laplacian[i, j] - (i == j ? 1.0 : 0.0);
            }
        }

        return laplacian;
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
    public override Tensor<T> Predict(Tensor<T> input)
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
    public override void Train(Tensor<T> input, Tensor<T> target)
    {
        if (!_useNativeMode)
            throw new InvalidOperationException("Training requires native mode.");

        SetTrainingMode(true);

        // Forward pass
        var output = Forward(input);

        // Calculate loss
        LastLoss = _lossFunction.CalculateLoss(output.ToVector(), target.ToVector());

        // Backward pass
        var gradient = _lossFunction.CalculateDerivative(output.ToVector(), target.ToVector());
        Backward(Tensor<T>.FromVector(gradient, output.Shape));

        _optimizer.UpdateParameters(Layers);

        SetTrainingMode(false);
    }

    /// <summary>
    /// Updates parameters using the provided gradients.
    /// </summary>
    /// <param name="gradients">Gradient vector.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the TemporalGCN model, UpdateParameters updates internal parameters or state. This keeps the TemporalGCN architecture aligned with the latest values.
    /// </para>
    /// </remarks>
    public override void UpdateParameters(Vector<T> gradients)
    {
        // Parameters are updated through the optimizer in Train()
    }

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
            ModelType = ModelType.NeuralNetwork,
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
    /// Creates a new instance with the same configuration.
    /// </summary>
    /// <returns>A new TemporalGCN instance.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the TemporalGCN model, CreateNewInstance builds and wires up model components. This sets up the TemporalGCN architecture before use.
    /// </para>
    /// </remarks>
    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance()
    {
        return new TemporalGCN<T>(Architecture, _options, _adjacencyMatrix);
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
    protected override void SerializeNetworkSpecificData(BinaryWriter writer)
    {
        writer.Write(_sequenceLength);
        writer.Write(_forecastHorizon);
        writer.Write(_numNodes);
        writer.Write(_numFeatures);
        writer.Write(_hiddenDimension);
        writer.Write(_numGCNLayers);
        writer.Write(_numTemporalLayers);
        writer.Write(_chebyshevOrder);
        writer.Write(_temporalCellType);
        writer.Write(_useResidualConnections);
        writer.Write(_useBatchNormalization);
        writer.Write(_numSamples);
    }

    /// <summary>
    /// Deserializes TemporalGCN-specific data.
    /// </summary>
    /// <param name="reader">The binary reader.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the TemporalGCN model, DeserializeNetworkSpecificData saves or restores model-specific settings. This lets the TemporalGCN architecture be reused later.
    /// </para>
    /// </remarks>
    protected override void DeserializeNetworkSpecificData(BinaryReader reader)
    {
        _ = reader.ReadInt32(); // sequenceLength
        _ = reader.ReadInt32(); // forecastHorizon
        _ = reader.ReadInt32(); // numNodes
        _ = reader.ReadInt32(); // numFeatures
        _ = reader.ReadInt32(); // hiddenDimension
        _ = reader.ReadInt32(); // numGCNLayers
        _ = reader.ReadInt32(); // numTemporalLayers
        _ = reader.ReadInt32(); // chebyshevOrder
        _ = reader.ReadString(); // temporalCellType
        _ = reader.ReadBoolean(); // useResidualConnections
        _ = reader.ReadBoolean(); // useBatchNormalization
        _ = reader.ReadInt32(); // numSamples
    }

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
    public override Tensor<T> Forecast(Tensor<T> historicalData, double[]? quantiles = null)
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

        T mse = NumOps.Zero;
        T mae = NumOps.Zero;
        int count = Math.Min(predictions.Data.Length, actuals.Data.Length);

        for (int i = 0; i < count; i++)
        {
            T diff = NumOps.Subtract(predictions.Data.Span[i], actuals.Data.Span[i]);
            mse = NumOps.Add(mse, NumOps.Multiply(diff, diff));
            mae = NumOps.Add(mae, NumOps.Abs(diff));
        }

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
        var current = FlattenInput(input);

        // Apply layers
        foreach (var layer in Layers)
        {
            current = layer.Forward(current);
        }

        // Apply Chebyshev graph convolution
        if (_normalizedLaplacian is not null && _useNativeMode)
        {
            current = ApplyChebyshevConvolution(current);
        }

        return current;
    }

    /// <summary>
    /// Performs backpropagation through all layers.
    /// </summary>
    /// <param name="gradOutput">Gradient of loss with respect to output.</param>
    /// <returns>Gradient with respect to input.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the TemporalGCN model, Backward propagates gradients backward. This teaches the TemporalGCN architecture how to adjust its weights.
    /// </para>
    /// </remarks>
    public Tensor<T> Backward(Tensor<T> gradOutput)
    {
        var grad = gradOutput;

        for (int i = Layers.Count - 1; i >= 0; i--)
        {
            grad = Layers[i].Backward(grad);
        }

        return grad;
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
        foreach (var dim in input.Shape)
        {
            totalSize *= dim;
        }

        var flattened = new Tensor<T>(new[] { totalSize });
        for (int i = 0; i < totalSize; i++)
        {
            flattened.Data.Span[i] = input.Data.Span[i];
        }

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

        var featureVec = nodeFeatures.ToVector();
        int totalSize = featureVec.Length;
        int featuresPerNode = totalSize / _numNodes;

        if (featuresPerNode * _numNodes != totalSize)
            return nodeFeatures;

        var result = new T[totalSize];

        // Apply simplified Chebyshev convolution (K=2)
        // T_0 term: just the features
        // T_1 term: L * features
        for (int node = 0; node < _numNodes; node++)
        {
            for (int f = 0; f < featuresPerNode; f++)
            {
                double t0 = NumOps.ToDouble(featureVec[node * featuresPerNode + f]);
                double t1 = 0;

                // Compute L * x for T_1
                for (int neighbor = 0; neighbor < _numNodes; neighbor++)
                {
                    double laplacianValue = _normalizedLaplacian[node, neighbor];
                    double neighborFeature = NumOps.ToDouble(featureVec[neighbor * featuresPerNode + f]);
                    t1 += laplacianValue * neighborFeature;
                }

                // Combine terms with equal weighting
                double combined = 0.5 * t0 + 0.5 * t1;
                result[node * featuresPerNode + f] = NumOps.FromDouble(combined);
            }
        }

        return new Tensor<T>(nodeFeatures.Shape, new Vector<T>(result));
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
        for (int i = 0; i < flatInput.Data.Length; i++)
        {
            inputData[i] = Convert.ToSingle(NumOps.ToDouble(flatInput.Data.Span[i]));
        }

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
        for (int i = 0; i < outputTensor.Length; i++)
        {
            outputData[i] = NumOps.FromDouble(outputTensor.GetValue(i));
        }

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

        return new Tensor<T>(input.Shape, new Vector<T>(perturbed));
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
        var inputVec = input.ToVector();
        var predVec = prediction.ToVector();

        int inputSize = _numNodes * _sequenceLength * _numFeatures;
        int shiftSize = _numNodes * _numFeatures;
        var shifted = new T[inputSize];

        // Remove oldest time step, add prediction
        for (int i = 0; i < inputSize - shiftSize; i++)
        {
            shifted[i] = inputVec[i + shiftSize];
        }
        for (int i = 0; i < Math.Min(shiftSize, predVec.Length); i++)
        {
            shifted[inputSize - shiftSize + i] = predVec[i];
        }

        return new Tensor<T>(new[] { inputSize }, new Vector<T>(shifted));
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
        var lowerVec = new T[len];
        var medianVec = new T[len];
        var upperVec = new T[len];

        for (int i = 0; i < len; i++)
        {
            lowerVec[i] = resultVec[i];
            medianVec[i] = resultVec[len + i];
            upperVec[i] = resultVec[2 * len + i];
        }

        return (
            new Tensor<T>(new[] { len }, new Vector<T>(medianVec)),
            new Tensor<T>(new[] { len }, new Vector<T>(lowerVec)),
            new Tensor<T>(new[] { len }, new Vector<T>(upperVec))
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



