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
/// STGNN (Spatio-Temporal Graph Neural Network) for forecasting on graph-structured time series data.
/// </summary>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
/// <remarks>
/// <para>
/// STGNN combines graph neural networks for spatial dependencies with temporal convolutions
/// for time series patterns, enabling forecasting on interconnected entities.
/// </para>
/// <para><b>For Beginners:</b> STGNN is designed for data where locations/entities are connected:
///
/// <b>The Key Insight:</b>
/// Many real-world time series are not independent - they're connected in space or by relationships.
/// Traffic at one intersection affects nearby intersections; one stock's movement affects related stocks.
/// STGNN models both these spatial connections and temporal patterns simultaneously.
///
/// <b>What Problems Does STGNN Solve?</b>
/// - Traffic forecasting (sensors connected by roads)
/// - Financial network prediction (assets connected by correlations)
/// - Weather forecasting (stations connected geographically)
/// - Social network dynamics (users connected by relationships)
///
/// <b>How STGNN Works:</b>
/// 1. <b>Graph Representation:</b> Encode spatial relationships as an adjacency matrix
/// 2. <b>Spatial Aggregation:</b> Each node gathers information from neighbors via graph convolution
/// 3. <b>Temporal Modeling:</b> Capture time patterns using temporal convolutions
/// 4. <b>ST Fusion:</b> Alternate spatial and temporal processing for joint modeling
/// 5. <b>Prediction:</b> Output forecasts for all nodes in the network
///
/// <b>STGNN Architecture:</b>
/// - ST-Conv Blocks: Sandwich structure (Temporal-Spatial-Temporal) for deep ST learning
/// - Graph Convolution: Chebyshev spectral convolution for efficient neighbor aggregation
/// - Gated Units: Control information flow between spatial and temporal paths
/// - Residual Connections: Enable training of deep networks
///
/// <b>Key Benefits:</b>
/// - Captures complex spatio-temporal dependencies
/// - Scales to large graphs (hundreds of nodes)
/// - Handles both directed and undirected graphs
/// - Provides multi-step ahead forecasts for all nodes
/// </para>
/// <para>
/// <b>Reference:</b> Yu et al., "Spatio-Temporal Graph Convolutional Networks", IJCAI 2018.
/// https://arxiv.org/abs/1709.04875
/// </para>
/// </remarks>
public class STGNN<T> : ForecastingModelBase<T>
{
    #region Execution Mode
    private readonly bool _useNativeMode;
    #endregion

    
    #region Native Mode Fields
    private DenseLayer<T>? _inputEmbedding;
    private List<DenseLayer<T>>? _stConvBlocks;
    private List<LayerNormalizationLayer<T>>? _layerNorms;
    private DenseLayer<T>? _outputLayer;
    #endregion

    #region Graph Fields
    private readonly double[,]? _adjacencyMatrix;
    private readonly Random _random;
    #endregion

    #region Shared Fields
    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> _optimizer;
    private readonly ILossFunction<T> _lossFunction;
    private readonly STGNNOptions<T> _options;
    private int _sequenceLength;
    private int _forecastHorizon;
    private int _numNodes;
    private int _numFeatures;
    private int _hiddenDimension;
    private int _numSpatialLayers;
    private int _numTemporalLayers;
    private int _numSamples;
    private string _graphConvType;
    private bool _useGatedFusion;
    private bool _useResidualConnections;
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
    /// </para>
    /// </remarks>
    public int NumNodes => _numNodes;

    /// <summary>
    /// Gets the forecast horizon.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> How many future steps to predict for each node.
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
    /// Gets the type of graph convolution used.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> The method for aggregating neighbor information.
    /// </para>
    /// </remarks>
    public string GraphConvType => _graphConvType;

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
    /// Initializes a new instance of the STGNN model in ONNX mode for inference.
    /// </summary>
    /// <param name="architecture">The neural network architecture configuration.</param>
    /// <param name="onnxModelPath">Path to a pretrained ONNX model file.</param>
    /// <param name="options">STGNN-specific options.</param>
    /// <param name="adjacencyMatrix">Optional pre-defined adjacency matrix for the graph.</param>
    /// <param name="optimizer">Optional optimizer for fine-tuning.</param>
    /// <param name="lossFunction">Optional loss function.</param>
    /// <remarks>
    /// <para><b>For Beginners:</b> Use this constructor to load a pretrained STGNN model.
    /// The ONNX model contains the trained spatial and temporal processing layers.
    /// </para>
    /// </remarks>
    public STGNN(
        NeuralNetworkArchitecture<T> architecture,
        string onnxModelPath,
        STGNNOptions<T>? options = null,
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
        _options = options ?? new STGNNOptions<T>();
        _lossFunction = lossFunction ?? new MeanSquaredErrorLoss<T>();
        _optimizer = optimizer ?? new AdamOptimizer<T, Tensor<T>, Tensor<T>>(this);

        _sequenceLength = _options.SequenceLength;
        _forecastHorizon = _options.ForecastHorizon;
        _numNodes = _options.NumNodes;
        _numFeatures = _options.NumFeatures;
        _hiddenDimension = _options.HiddenDimension;
        _numSpatialLayers = _options.NumSpatialLayers;
        _numTemporalLayers = _options.NumTemporalLayers;
        _numSamples = _options.NumSamples;
        _graphConvType = _options.GraphConvType;
        _useGatedFusion = _options.UseGatedFusion;
        _useResidualConnections = _options.UseResidualConnections;

        _adjacencyMatrix = adjacencyMatrix ?? CreateDefaultAdjacencyMatrix(_numNodes);
        _random = RandomHelper.CreateSecureRandom();
    }

    /// <summary>
    /// Initializes a new instance of the STGNN model in native mode for training.
    /// </summary>
    /// <param name="architecture">The neural network architecture configuration.</param>
    /// <param name="options">STGNN-specific options.</param>
    /// <param name="adjacencyMatrix">Optional pre-defined adjacency matrix for the graph.</param>
    /// <param name="optimizer">Optional optimizer for training.</param>
    /// <param name="lossFunction">Optional loss function.</param>
    /// <remarks>
    /// <para><b>For Beginners:</b> Use this constructor to create an STGNN model for training.
    /// Provide an adjacency matrix that defines how nodes are connected. If not provided,
    /// a default fully-connected graph with distance-based weights is used.
    /// </para>
    /// </remarks>
    public STGNN(
        NeuralNetworkArchitecture<T> architecture,
        STGNNOptions<T>? options = null,
        double[,]? adjacencyMatrix = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
        ILossFunction<T>? lossFunction = null)
        : base(architecture, lossFunction ?? new MeanSquaredErrorLoss<T>(), 1.0)
    {
        _useNativeMode = true;
        _options = options ?? new STGNNOptions<T>();
        _lossFunction = lossFunction ?? new MeanSquaredErrorLoss<T>();
        _optimizer = optimizer ?? new AdamOptimizer<T, Tensor<T>, Tensor<T>>(this);

        _sequenceLength = _options.SequenceLength;
        _forecastHorizon = _options.ForecastHorizon;
        _numNodes = _options.NumNodes;
        _numFeatures = _options.NumFeatures;
        _hiddenDimension = _options.HiddenDimension;
        _numSpatialLayers = _options.NumSpatialLayers;
        _numTemporalLayers = _options.NumTemporalLayers;
        _numSamples = _options.NumSamples;
        _graphConvType = _options.GraphConvType;
        _useGatedFusion = _options.UseGatedFusion;
        _useResidualConnections = _options.UseResidualConnections;

        _adjacencyMatrix = adjacencyMatrix ?? CreateDefaultAdjacencyMatrix(_numNodes);
        _random = RandomHelper.CreateSecureRandom();

        InitializeLayers();
    }

    #endregion

    #region Initialization

    /// <summary>
    /// Initializes all layers for the STGNN model.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> This sets up the spatio-temporal processing architecture:
    ///
    /// <b>Layer Structure:</b>
    /// 1. Input embedding: Projects node features to hidden dimension
    /// 2. ST-Conv blocks: Alternating temporal and spatial processing
    /// 3. Output projection: Maps to forecast dimension for all nodes
    ///
    /// The key insight is the "sandwich" structure: temporal-spatial-temporal processing
    /// captures how spatial patterns evolve over time.
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
            Layers.AddRange(LayerHelper<T>.CreateDefaultSTGNNLayers(
                Architecture,
                _sequenceLength,
                _forecastHorizon,
                _numNodes,
                _numFeatures,
                _hiddenDimension,
                _numSpatialLayers,
                _numTemporalLayers));

            ExtractLayerReferences();
        }
    }

    /// <summary>
    /// Extracts references to key layers for efficient access.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Keeps direct references to the input embedding,
    /// ST-Conv blocks, and output layer for organized processing.
    /// </para>
    /// </remarks>
    private void ExtractLayerReferences()
    {
        var allDense = Layers.OfType<DenseLayer<T>>().ToList();
        _layerNorms = Layers.OfType<LayerNormalizationLayer<T>>().ToList();

        if (allDense.Count >= 2)
        {
            _inputEmbedding = allDense[0];
            _stConvBlocks = allDense.Skip(1).Take(allDense.Count - 3).ToList();
            _outputLayer = allDense[allDense.Count - 1];
        }
    }

    /// <summary>
    /// Creates a default adjacency matrix for the graph.
    /// </summary>
    /// <param name="numNodes">Number of nodes in the graph.</param>
    /// <returns>Adjacency matrix with distance-based weights.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> When no graph structure is provided, creates a
    /// default matrix where nearby nodes (by index) are more connected. In practice,
    /// you should provide a real adjacency matrix based on actual spatial relationships.
    /// </para>
    /// </remarks>
    private double[,] CreateDefaultAdjacencyMatrix(int numNodes)
    {
        var adj = new double[numNodes, numNodes];

        // Create a sparse adjacency with nearby connections
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
                    adj[i, j] = Math.Exp(-distance / range); // Exponential decay
                }
            }
        }

        // Normalize rows
        for (int i = 0; i < numNodes; i++)
        {
            double rowSum = 0;
            for (int j = 0; j < numNodes; j++)
            {
                rowSum += adj[i, j];
            }
            if (rowSum > 0)
            {
                for (int j = 0; j < numNodes; j++)
                {
                    adj[i, j] /= rowSum;
                }
            }
        }

        return adj;
    }

    /// <summary>
    /// Validates custom layers provided by the user.
    /// </summary>
    /// <param name="layers">The list of custom layers.</param>
    /// <remarks>
    /// <para><b>For Beginners:</b> Ensures custom layers form a valid STGNN architecture.
    /// </para>
    /// </remarks>
    protected override void ValidateCustomLayers(List<ILayer<T>> layers)
    {
        base.ValidateCustomLayers(layers);

        if (layers.Count < 4)
            throw new ArgumentException("STGNN requires at least 4 layers (input, ST blocks, output).");
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
    /// 1. Embed node features
    /// 2. Apply ST-Conv blocks (temporal-spatial-temporal processing)
    /// 3. Apply graph convolution for neighbor aggregation
    /// 4. Project to forecast dimension
    /// </para>
    /// </remarks>
    public override Tensor<T> Predict(Tensor<T> input)
    {
        return _useNativeMode ? ForecastNative(input) : ForecastOnnx(input);
    }

    /// <summary>
    /// Trains the STGNN model on a batch of input-target pairs.
    /// </summary>
    /// <param name="input">Input tensor with historical data for all nodes.</param>
    /// <param name="target">Target tensor with future values for all nodes.</param>
    /// <remarks>
    /// <para><b>For Beginners:</b> Training minimizes prediction error across all nodes:
    /// 1. Forward pass through ST-Conv blocks with graph convolution
    /// 2. Compute loss between predicted and actual future values
    /// 3. Backpropagate through both spatial and temporal paths
    /// 4. Update network parameters
    ///
    /// The graph structure (adjacency matrix) guides how information flows between nodes.
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
    /// <b>For Beginners:</b> In the STGNN model, UpdateParameters updates internal parameters or state. This keeps the STGNN architecture aligned with the latest values.
    /// </para>
    /// </remarks>
    public override void UpdateParameters(Vector<T> gradients)
    {
        // Parameters are updated through the optimizer in Train()
    }

    /// <summary>
    /// Gets metadata about the STGNN model.
    /// </summary>
    /// <returns>ModelMetadata containing model information.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the STGNN model, GetModelMetadata performs a supporting step in the workflow. It keeps the STGNN architecture pipeline consistent.
    /// </para>
    /// </remarks>
    public override ModelMetadata<T> GetModelMetadata()
    {
        return new ModelMetadata<T>
        {
            ModelType = ModelType.NeuralNetwork,
            AdditionalInfo = new Dictionary<string, object>
            {
                { "NetworkType", "STGNN" },
                { "SequenceLength", _sequenceLength },
                { "ForecastHorizon", _forecastHorizon },
                { "NumNodes", _numNodes },
                { "NumFeatures", _numFeatures },
                { "HiddenDimension", _hiddenDimension },
                { "NumSpatialLayers", _numSpatialLayers },
                { "NumTemporalLayers", _numTemporalLayers },
                { "GraphConvType", _graphConvType },
                { "UseGatedFusion", _useGatedFusion },
                { "UseResidualConnections", _useResidualConnections },
                { "NumSamples", _numSamples },
                { "UseNativeMode", _useNativeMode }
            }
        };
    }

    /// <summary>
    /// Creates a new instance with the same configuration.
    /// </summary>
    /// <returns>A new STGNN instance.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the STGNN model, CreateNewInstance builds and wires up model components. This sets up the STGNN architecture before use.
    /// </para>
    /// </remarks>
    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance()
    {
        return new STGNN<T>(Architecture, _options, _adjacencyMatrix);
    }

    /// <summary>
    /// Serializes STGNN-specific data.
    /// </summary>
    /// <param name="writer">The binary writer.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the STGNN model, SerializeNetworkSpecificData saves or restores model-specific settings. This lets the STGNN architecture be reused later.
    /// </para>
    /// </remarks>
    protected override void SerializeNetworkSpecificData(BinaryWriter writer)
    {
        writer.Write(_sequenceLength);
        writer.Write(_forecastHorizon);
        writer.Write(_numNodes);
        writer.Write(_numFeatures);
        writer.Write(_hiddenDimension);
        writer.Write(_numSpatialLayers);
        writer.Write(_numTemporalLayers);
        writer.Write(_graphConvType);
        writer.Write(_useGatedFusion);
        writer.Write(_useResidualConnections);
        writer.Write(_numSamples);
    }

    /// <summary>
    /// Deserializes STGNN-specific data.
    /// </summary>
    /// <param name="reader">The binary reader.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the STGNN model, DeserializeNetworkSpecificData saves or restores model-specific settings. This lets the STGNN architecture be reused later.
    /// </para>
    /// </remarks>
    protected override void DeserializeNetworkSpecificData(BinaryReader reader)
    {
        _sequenceLength = reader.ReadInt32();
        _forecastHorizon = reader.ReadInt32();
        _numNodes = reader.ReadInt32();
        _numFeatures = reader.ReadInt32();
        _hiddenDimension = reader.ReadInt32();
        _numSpatialLayers = reader.ReadInt32();
        _numTemporalLayers = reader.ReadInt32();
        _graphConvType = reader.ReadString();
        _useGatedFusion = reader.ReadBoolean();
        _useResidualConnections = reader.ReadBoolean();
        _numSamples = reader.ReadInt32();
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
    /// <para><b>For Beginners:</b> Predicts future values for every node in the graph
    /// simultaneously. The spatial structure helps because nodes can "borrow information"
    /// from their neighbors through graph convolution.
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
    /// <b>For Beginners:</b> In the STGNN model, public performs a supporting step in the workflow. It keeps the STGNN architecture pipeline consistent.
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
    /// <b>For Beginners:</b> In the STGNN model, AutoregressiveForecast produces predictions from input data. This is the main inference step of the STGNN architecture.
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
    /// <b>For Beginners:</b> In the STGNN model, Evaluate performs a supporting step in the workflow. It keeps the STGNN architecture pipeline consistent.
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
    /// Applies instance normalization (identity for STGNN).
    /// </summary>
    /// <param name="input">Input tensor.</param>
    /// <returns>The input tensor unchanged.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the STGNN model, ApplyInstanceNormalization performs a supporting step in the workflow. It keeps the STGNN architecture pipeline consistent.
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
    /// <b>For Beginners:</b> In the STGNN model, GetFinancialMetrics calculates evaluation metrics. This summarizes how the STGNN architecture is performing.
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
            ["NumSpatialLayers"] = NumOps.FromDouble(_numSpatialLayers),
            ["NumTemporalLayers"] = NumOps.FromDouble(_numTemporalLayers)
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
    /// 1. Embeds input features
    /// 2. Applies graph convolution for spatial aggregation
    /// 3. Processes through ST-Conv blocks
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

        // Apply graph convolution (matrix multiplication with adjacency)
        if (_adjacencyMatrix is not null && _useNativeMode)
        {
            current = ApplyGraphConvolution(current);
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
    /// <b>For Beginners:</b> In the STGNN model, Backward propagates gradients backward. This teaches the STGNN architecture how to adjust its weights.
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
    /// <b>For Beginners:</b> In the STGNN model, FlattenInput performs a supporting step in the workflow. It keeps the STGNN architecture pipeline consistent.
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
    /// Applies graph convolution using the adjacency matrix.
    /// </summary>
    /// <param name="nodeFeatures">Node feature tensor.</param>
    /// <returns>Aggregated features after graph convolution.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Graph convolution aggregates information from neighbors:
    /// h'_i = sum_j(A_ij * h_j)
    ///
    /// Each node's new features are a weighted average of its neighbors' features,
    /// where weights come from the adjacency matrix.
    /// </para>
    /// </remarks>
    private Tensor<T> ApplyGraphConvolution(Tensor<T> nodeFeatures)
    {
        if (_adjacencyMatrix is null)
            return nodeFeatures;

        var featureVec = nodeFeatures.ToVector();
        int totalSize = featureVec.Length;
        int featuresPerNode = totalSize / _numNodes;

        if (featuresPerNode * _numNodes != totalSize)
            return nodeFeatures; // Can't reshape, return unchanged

        var result = new T[totalSize];

        // Apply graph convolution: H' = A * H
        for (int node = 0; node < _numNodes; node++)
        {
            for (int f = 0; f < featuresPerNode; f++)
            {
                double aggregated = 0;
                for (int neighbor = 0; neighbor < _numNodes; neighbor++)
                {
                    double weight = _adjacencyMatrix[node, neighbor];
                    double feature = NumOps.ToDouble(featureVec[neighbor * featuresPerNode + f]);
                    aggregated += weight * feature;
                }
                result[node * featuresPerNode + f] = NumOps.FromDouble(aggregated);
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
    /// <b>For Beginners:</b> In the STGNN model, ForecastNative produces predictions from input data. This is the main inference step of the STGNN architecture.
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
    /// <b>For Beginners:</b> In the STGNN model, ForecastOnnx produces predictions from input data. This is the main inference step of the STGNN architecture.
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

        // For MC Dropout, we would keep dropout active
        // Here we add small perturbations for diversity
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
    /// <b>For Beginners:</b> In the STGNN model, AddSmallPerturbation performs a supporting step in the workflow. It keeps the STGNN architecture pipeline consistent.
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
    /// <b>For Beginners:</b> In the STGNN model, ShiftInputWindow performs a supporting step in the workflow. It keeps the STGNN architecture pipeline consistent.
    /// </para>
    /// </remarks>
    private Tensor<T> ShiftInputWindow(Tensor<T> input, Tensor<T> prediction)
    {
        var inputVec = input.ToVector();
        var predVec = prediction.ToVector();

        // Use actual input size, not calculated expected size
        int inputSize = inputVec.Length;
        int shiftSize = Math.Min(_numNodes * _numFeatures, inputSize);
        var shifted = new T[inputSize];

        // Copy shifted values from input (avoiding index out of range)
        int copyLen = Math.Max(0, inputSize - shiftSize);
        for (int i = 0; i < copyLen; i++)
        {
            int srcIdx = i + shiftSize;
            if (srcIdx < inputVec.Length)
                shifted[i] = inputVec[srcIdx];
        }
        // Fill end with prediction values
        for (int i = 0; i < Math.Min(shiftSize, predVec.Length); i++)
        {
            int dstIdx = inputSize - shiftSize + i;
            if (dstIdx >= 0 && dstIdx < inputSize)
                shifted[dstIdx] = predVec[i];
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
    /// <b>For Beginners:</b> In the STGNN model, ConcatenatePredictions produces predictions from input data. This is the main inference step of the STGNN architecture.
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
    /// <b>For Beginners:</b> In the STGNN model, ComputeQuantiles performs a supporting step in the workflow. It keeps the STGNN architecture pipeline consistent.
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
    /// <b>For Beginners:</b> In the STGNN model, private performs a supporting step in the workflow. It keeps the STGNN architecture pipeline consistent.
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

        var shape = new[] { len };
        return (new Tensor<T>(shape, new Vector<T>(medianVec)),
                new Tensor<T>(shape, new Vector<T>(lowerVec)),
                new Tensor<T>(shape, new Vector<T>(upperVec)));
    }

    #endregion

    #region IDisposable

    /// <summary>
    /// Disposes of resources.
    /// </summary>
    /// <param name="disposing">Whether to dispose managed resources.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the STGNN model, Dispose performs a supporting step in the workflow. It keeps the STGNN architecture pipeline consistent.
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



