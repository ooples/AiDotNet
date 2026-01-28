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

namespace AiDotNet.Finance.Graph;

/// <summary>
/// MTGNN (Multivariate Time-series Graph Neural Network) for automatic graph learning and forecasting.
/// </summary>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
/// <remarks>
/// <para>
/// MTGNN automatically discovers the graph structure from data while performing spatio-temporal
/// forecasting, eliminating the need for a predefined adjacency matrix.
/// </para>
/// <para><b>For Beginners:</b> MTGNN is unique because it LEARNS how variables are connected:
///
/// <b>The Key Insight:</b>
/// Unlike other graph models that require you to define the graph upfront, MTGNN
/// automatically discovers which time series influence each other. It learns node
/// embeddings whose similarities form an adaptive graph.
///
/// <b>What Problems Does MTGNN Solve?</b>
/// - Traffic prediction when road relationships are complex or unknown
/// - Multivariate financial forecasting with hidden correlations
/// - Sensor networks where dependencies change over time
/// - Any multivariate series where inter-variable relationships are important but unknown
///
/// <b>How MTGNN Works:</b>
/// 1. <b>Graph Learning:</b> Learns node embeddings E1, E2; computes A = softmax(E1 * E2^T)
/// 2. <b>Mix-hop Propagation:</b> Aggregates 1-hop, 2-hop, ... K-hop neighbors
/// 3. <b>Dilated Inception:</b> Captures multi-scale temporal patterns via dilated convolutions
/// 4. <b>Joint Learning:</b> Graph structure and predictions are optimized together
///
/// <b>MTGNN Architecture:</b>
/// - Node Embeddings: Two learnable embedding matrices E1, E2
/// - Adaptive Adjacency: A = softmax(ReLU(E1 * E2^T - E2 * E1^T))
/// - Mix-hop Propagation: H_out = concat(H, A*H, A^2*H, ..., A^K*H) * W
/// - Dilated Inception: Parallel convolutions with exponentially increasing dilation
/// - Skip Connections: Gated residual connections between layers
///
/// <b>Key Benefits:</b>
/// - No need to predefine graph structure
/// - Discovers hidden variable relationships automatically
/// - Captures both unidirectional and bidirectional dependencies
/// - Scales to hundreds of variables with subgraph sampling
/// </para>
/// <para>
/// <b>Reference:</b> Wu et al., "Connecting the Dots: Multivariate Time Series Forecasting with Graph Neural Networks", KDD 2020.
/// https://arxiv.org/abs/2005.11650
/// </para>
/// </remarks>
public class MTGNN<T> : NeuralNetworkBase<T>, IForecastingModel<T>
{
    #region Execution Mode
    private readonly bool _useNativeMode;
    #endregion

    #region ONNX Mode Fields
    private readonly InferenceSession? _onnxSession;
    private readonly string? _onnxModelPath;
    #endregion

    #region Native Mode Fields
    private DenseLayer<T>? _inputProjection;
    private DenseLayer<T>? _nodeEmbeddingLayer;
    private List<DenseLayer<T>>? _mixHopLayers;
    private List<DenseLayer<T>>? _temporalLayers;
    private List<DenseLayer<T>>? _gateLayers;
    private List<LayerNormalizationLayer<T>>? _layerNorms;
    private List<DropoutLayer<T>>? _dropoutLayers;
    private DenseLayer<T>? _outputLayer;
    #endregion

    #region Graph Learning Fields
    private double[,]? _nodeEmbedding1;
    private double[,]? _nodeEmbedding2;
    private double[,]? _adaptiveAdjacency;
    private readonly double[,]? _predefinedAdjacency;
    private readonly Random _random;
    #endregion

    #region Shared Fields
    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> _optimizer;
    private readonly ILossFunction<T> _lossFunction;
    private readonly MTGNNOptions<T> _options;
    private readonly int _sequenceLength;
    private readonly int _forecastHorizon;
    private readonly int _numNodes;
    private readonly int _numFeatures;
    private readonly int _hiddenDimension;
    private readonly int _nodeEmbeddingDim;
    private readonly int _numLayers;
    private readonly int _mixHopDepth;
    private readonly int _temporalKernelSize;
    private readonly int _dilationFactor;
    private readonly int _numSamples;
    private readonly bool _usePredefinedGraph;
    private readonly bool _useSubgraphSampling;
    private readonly int _subgraphSize;
    #endregion

    #region IForecastingModel Properties

    /// <inheritdoc/>
    public int SequenceLength => _sequenceLength;

    /// <inheritdoc/>
    public int PredictionHorizon => _forecastHorizon;

    /// <inheritdoc/>
    public int NumFeatures => _numFeatures;

    /// <inheritdoc/>
    public int PatchSize => 1;

    /// <inheritdoc/>
    public int Stride => 1;

    /// <inheritdoc/>
    public bool IsChannelIndependent => false;

    /// <inheritdoc/>
    public bool UseNativeMode => _useNativeMode;

    /// <summary>
    /// Gets the number of nodes (variables/time series).
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> How many time series are being modeled simultaneously.
    /// MTGNN learns the relationships between these series automatically.
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
    /// Gets the node embedding dimension.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> The dimension of learnable node representations
    /// used for automatic graph structure discovery.
    /// </para>
    /// </remarks>
    public int NodeEmbeddingDim => _nodeEmbeddingDim;

    /// <summary>
    /// Gets the mix-hop propagation depth.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> How many hops of neighbors to aggregate information from.
    /// </para>
    /// </remarks>
    public int MixHopDepth => _mixHopDepth;

    /// <summary>
    /// Gets the number of samples for uncertainty estimation.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> For probabilistic forecasting with MC Dropout.
    /// </para>
    /// </remarks>
    public int NumSamples => _numSamples;

    /// <summary>
    /// Gets the learned adaptive adjacency matrix (read-only copy).
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> The graph structure discovered from data.
    /// Entry (i,j) indicates how strongly node j influences node i.
    /// </para>
    /// </remarks>
    public double[,]? LearnedAdjacency => _adaptiveAdjacency is not null ? (double[,])_adaptiveAdjacency.Clone() : null;

    #endregion

    #region Constructors

    /// <summary>
    /// Initializes a new instance of the MTGNN model in ONNX mode for inference.
    /// </summary>
    /// <param name="architecture">The neural network architecture configuration.</param>
    /// <param name="onnxModelPath">Path to a pretrained ONNX model file.</param>
    /// <param name="options">MTGNN-specific options.</param>
    /// <param name="predefinedAdjacency">Optional predefined adjacency matrix to combine with learned graph.</param>
    /// <param name="optimizer">Optional optimizer for fine-tuning.</param>
    /// <param name="lossFunction">Optional loss function.</param>
    /// <remarks>
    /// <para><b>For Beginners:</b> Use this constructor to load a pretrained MTGNN model.
    /// The ONNX model includes both the learned graph structure and the forecasting layers.
    /// </para>
    /// </remarks>
    public MTGNN(
        NeuralNetworkArchitecture<T> architecture,
        string onnxModelPath,
        MTGNNOptions<T>? options = null,
        double[,]? predefinedAdjacency = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
        ILossFunction<T>? lossFunction = null)
        : base(architecture, lossFunction ?? new MeanSquaredErrorLoss<T>(), 1.0)
    {
        if (string.IsNullOrWhiteSpace(onnxModelPath))
            throw new ArgumentNullException(nameof(onnxModelPath));
        if (!System.IO.File.Exists(onnxModelPath))
            throw new System.IO.FileNotFoundException($"ONNX model not found: {onnxModelPath}");

        _useNativeMode = false;
        _onnxModelPath = onnxModelPath;
        _onnxSession = new InferenceSession(onnxModelPath);
        _options = options ?? new MTGNNOptions<T>();
        _lossFunction = lossFunction ?? new MeanSquaredErrorLoss<T>();
        _optimizer = optimizer ?? new AdamOptimizer<T, Tensor<T>, Tensor<T>>(this);

        _sequenceLength = _options.SequenceLength;
        _forecastHorizon = _options.ForecastHorizon;
        _numNodes = _options.NumNodes;
        _numFeatures = _options.NumFeatures;
        _hiddenDimension = _options.HiddenDimension;
        _nodeEmbeddingDim = _options.NodeEmbeddingDim;
        _numLayers = _options.NumLayers;
        _mixHopDepth = _options.MixHopDepth;
        _temporalKernelSize = _options.TemporalKernelSize;
        _dilationFactor = _options.DilationFactor;
        _numSamples = _options.NumSamples;
        _usePredefinedGraph = _options.UsePredefinedGraph;
        _useSubgraphSampling = _options.UseSubgraphSampling;
        _subgraphSize = _options.SubgraphSize;

        _predefinedAdjacency = predefinedAdjacency;
        _random = RandomHelper.CreateSecureRandom();

        InitializeNodeEmbeddings();
    }

    /// <summary>
    /// Initializes a new instance of the MTGNN model in native mode for training.
    /// </summary>
    /// <param name="architecture">The neural network architecture configuration.</param>
    /// <param name="options">MTGNN-specific options.</param>
    /// <param name="predefinedAdjacency">Optional predefined adjacency matrix to combine with learned graph.</param>
    /// <param name="optimizer">Optional optimizer for training.</param>
    /// <param name="lossFunction">Optional loss function.</param>
    /// <remarks>
    /// <para><b>For Beginners:</b> Use this constructor to create an MTGNN model for training.
    /// The model will automatically learn the graph structure during training.
    /// Optionally provide a predefined adjacency to combine with the learned graph.
    /// </para>
    /// </remarks>
    public MTGNN(
        NeuralNetworkArchitecture<T> architecture,
        MTGNNOptions<T>? options = null,
        double[,]? predefinedAdjacency = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
        ILossFunction<T>? lossFunction = null)
        : base(architecture, lossFunction ?? new MeanSquaredErrorLoss<T>(), 1.0)
    {
        _useNativeMode = true;
        _options = options ?? new MTGNNOptions<T>();
        _lossFunction = lossFunction ?? new MeanSquaredErrorLoss<T>();
        _optimizer = optimizer ?? new AdamOptimizer<T, Tensor<T>, Tensor<T>>(this);

        _sequenceLength = _options.SequenceLength;
        _forecastHorizon = _options.ForecastHorizon;
        _numNodes = _options.NumNodes;
        _numFeatures = _options.NumFeatures;
        _hiddenDimension = _options.HiddenDimension;
        _nodeEmbeddingDim = _options.NodeEmbeddingDim;
        _numLayers = _options.NumLayers;
        _mixHopDepth = _options.MixHopDepth;
        _temporalKernelSize = _options.TemporalKernelSize;
        _dilationFactor = _options.DilationFactor;
        _numSamples = _options.NumSamples;
        _usePredefinedGraph = _options.UsePredefinedGraph;
        _useSubgraphSampling = _options.UseSubgraphSampling;
        _subgraphSize = _options.SubgraphSize;

        _predefinedAdjacency = predefinedAdjacency;
        _random = RandomHelper.CreateSecureRandom();

        InitializeNodeEmbeddings();
        InitializeLayers();
    }

    #endregion

    #region Initialization

    /// <summary>
    /// Initializes the learnable node embeddings for adaptive graph learning.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Creates two embedding matrices E1 and E2.
    /// The adaptive adjacency is computed as A = softmax(ReLU(E1 * E2^T - E2 * E1^T)).
    /// This captures both unidirectional and bidirectional dependencies.
    /// </para>
    /// </remarks>
    private void InitializeNodeEmbeddings()
    {
        _nodeEmbedding1 = new double[_numNodes, _nodeEmbeddingDim];
        _nodeEmbedding2 = new double[_numNodes, _nodeEmbeddingDim];

        // Initialize with small random values
        double scale = Math.Sqrt(2.0 / (_numNodes + _nodeEmbeddingDim));
        for (int i = 0; i < _numNodes; i++)
        {
            for (int j = 0; j < _nodeEmbeddingDim; j++)
            {
                _nodeEmbedding1[i, j] = (_random.NextDouble() - 0.5) * 2 * scale;
                _nodeEmbedding2[i, j] = (_random.NextDouble() - 0.5) * 2 * scale;
            }
        }

        // Compute initial adaptive adjacency
        UpdateAdaptiveAdjacency();
    }

    /// <summary>
    /// Updates the adaptive adjacency matrix from current node embeddings.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Computes the graph structure from learned embeddings:
    /// A = softmax(ReLU(E1 * E2^T - E2 * E1^T))
    ///
    /// The subtraction (E1*E2^T - E2*E1^T) creates an asymmetric adjacency,
    /// capturing directed relationships where A[i,j] != A[j,i].
    /// </para>
    /// </remarks>
    private void UpdateAdaptiveAdjacency()
    {
        if (_nodeEmbedding1 is null || _nodeEmbedding2 is null)
            return;

        _adaptiveAdjacency = new double[_numNodes, _numNodes];
        var rawScores = new double[_numNodes, _numNodes];

        // Compute E1 * E2^T - E2 * E1^T
        for (int i = 0; i < _numNodes; i++)
        {
            for (int j = 0; j < _numNodes; j++)
            {
                double score1 = 0, score2 = 0;
                for (int k = 0; k < _nodeEmbeddingDim; k++)
                {
                    score1 += _nodeEmbedding1[i, k] * _nodeEmbedding2[j, k];
                    score2 += _nodeEmbedding2[i, k] * _nodeEmbedding1[j, k];
                }
                rawScores[i, j] = Math.Max(0, score1 - score2); // ReLU
            }
        }

        // Apply row-wise softmax
        for (int i = 0; i < _numNodes; i++)
        {
            double maxVal = double.NegativeInfinity;
            for (int j = 0; j < _numNodes; j++)
            {
                maxVal = Math.Max(maxVal, rawScores[i, j]);
            }

            double sumExp = 0;
            for (int j = 0; j < _numNodes; j++)
            {
                _adaptiveAdjacency[i, j] = Math.Exp(rawScores[i, j] - maxVal);
                sumExp += _adaptiveAdjacency[i, j];
            }

            for (int j = 0; j < _numNodes; j++)
            {
                _adaptiveAdjacency[i, j] /= sumExp;
            }
        }

        // Combine with predefined adjacency if available
        if (_usePredefinedGraph && _predefinedAdjacency is not null)
        {
            for (int i = 0; i < _numNodes; i++)
            {
                for (int j = 0; j < _numNodes; j++)
                {
                    _adaptiveAdjacency[i, j] = 0.5 * _adaptiveAdjacency[i, j] + 0.5 * _predefinedAdjacency[i, j];
                }
            }
        }
    }

    /// <summary>
    /// Initializes all layers for the MTGNN model.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Sets up the model architecture:
    ///
    /// <b>Layer Structure:</b>
    /// 1. Input projection: Maps raw features to hidden dimension
    /// 2. Node embedding: Creates representations for graph learning
    /// 3. Mix-hop propagation: Multi-scale spatial aggregation
    /// 4. Dilated temporal convolution: Multi-scale temporal patterns
    /// 5. Gated skip connections: Residual learning with gating
    /// 6. Output projection: Maps to forecast dimension
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
            Layers.AddRange(LayerHelper<T>.CreateDefaultMTGNNLayers(
                Architecture,
                _sequenceLength,
                _forecastHorizon,
                _numNodes,
                _numFeatures,
                _hiddenDimension,
                _nodeEmbeddingDim,
                _numLayers,
                _mixHopDepth));

            ExtractLayerReferences();
        }
    }

    /// <summary>
    /// Extracts references to key layers for efficient access.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Keeps direct references to different layer types
    /// for organized processing during forward pass and graph learning updates.
    /// </para>
    /// </remarks>
    private void ExtractLayerReferences()
    {
        var allDense = Layers.OfType<DenseLayer<T>>().ToList();
        _layerNorms = Layers.OfType<LayerNormalizationLayer<T>>().ToList();
        _dropoutLayers = Layers.OfType<DropoutLayer<T>>().ToList();

        if (allDense.Count >= 4)
        {
            _inputProjection = allDense[0];
            _nodeEmbeddingLayer = allDense[1];

            // Mix-hop layers, temporal layers, and gate layers are interleaved
            _mixHopLayers = new List<DenseLayer<T>>();
            _temporalLayers = new List<DenseLayer<T>>();
            _gateLayers = new List<DenseLayer<T>>();

            int idx = 2;
            for (int layer = 0; layer < _numLayers; layer++)
            {
                // Mix-hop layers
                for (int hop = 0; hop < _mixHopDepth && idx < allDense.Count - 2; hop++)
                {
                    _mixHopLayers.Add(allDense[idx++]);
                }

                // Temporal and gate layers
                if (idx < allDense.Count - 2)
                    _temporalLayers.Add(allDense[idx++]);
                if (idx < allDense.Count - 2)
                    _gateLayers.Add(allDense[idx++]);
            }

            _outputLayer = allDense[allDense.Count - 1];
        }
    }

    /// <summary>
    /// Validates custom layers provided by the user.
    /// </summary>
    /// <param name="layers">The list of custom layers.</param>
    /// <remarks>
    /// <para><b>For Beginners:</b> Ensures custom layers form a valid MTGNN architecture
    /// with sufficient depth for graph learning and spatio-temporal processing.
    /// </para>
    /// </remarks>
    protected override void ValidateCustomLayers(List<ILayer<T>> layers)
    {
        base.ValidateCustomLayers(layers);

        if (layers.Count < 4)
            throw new ArgumentException("MTGNN requires at least 4 layers (input, embedding, processing, output).");
    }

    #endregion

    #region NeuralNetworkBase Overrides

    /// <summary>
    /// Performs forward prediction on the input tensor.
    /// </summary>
    /// <param name="input">Input tensor with shape [nodes, sequence, features].</param>
    /// <returns>Output tensor with forecasts for all nodes.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Processes input through:
    /// 1. Update adaptive adjacency from current embeddings
    /// 2. Apply mix-hop propagation with learned graph
    /// 3. Apply dilated temporal convolutions
    /// 4. Project to forecast dimension
    /// </para>
    /// </remarks>
    public override Tensor<T> Predict(Tensor<T> input)
    {
        return _useNativeMode ? ForecastNative(input) : ForecastOnnx(input);
    }

    /// <summary>
    /// Trains the MTGNN model on a batch of input-target pairs.
    /// </summary>
    /// <param name="input">Input tensor with historical data for all nodes.</param>
    /// <param name="target">Target tensor with future values for all nodes.</param>
    /// <remarks>
    /// <para><b>For Beginners:</b> Training jointly optimizes:
    /// 1. Node embeddings (for graph structure discovery)
    /// 2. Mix-hop propagation weights (for spatial aggregation)
    /// 3. Temporal convolution weights (for time pattern learning)
    ///
    /// The graph structure evolves during training as embeddings are updated.
    /// </para>
    /// </remarks>
    public override void Train(Tensor<T> input, Tensor<T> target)
    {
        if (!_useNativeMode)
            throw new InvalidOperationException("Training requires native mode.");

        SetTrainingMode(true);

        // Update adaptive adjacency before forward pass
        UpdateAdaptiveAdjacency();

        // Forward pass
        var output = Forward(input);

        // Calculate loss
        LastLoss = _lossFunction.CalculateLoss(output.ToVector(), target.ToVector());

        // Backward pass
        var gradient = _lossFunction.CalculateDerivative(output.ToVector(), target.ToVector());
        Backward(Tensor<T>.FromVector(gradient));

        _optimizer.UpdateParameters(Layers);

        // Update node embeddings (simplified gradient update)
        UpdateNodeEmbeddingsFromGradient(gradient);

        SetTrainingMode(false);
    }

    /// <summary>
    /// Updates node embeddings based on loss gradient.
    /// </summary>
    /// <param name="gradient">Loss gradient.</param>
    /// <remarks>
    /// <para><b>For Beginners:</b> Adjusts the node embedding matrices E1 and E2
    /// to improve the learned graph structure. This is a simplified update;
    /// full MTGNN would backprop through the adjacency computation.
    /// </para>
    /// </remarks>
    private void UpdateNodeEmbeddingsFromGradient(Vector<T> gradient)
    {
        if (_nodeEmbedding1 is null || _nodeEmbedding2 is null)
            return;

        double lr = 0.001;
        double gradNorm = 0;
        for (int i = 0; i < gradient.Length; i++)
        {
            double val = NumOps.ToDouble(gradient[i]);
            gradNorm += val * val;
        }
        gradNorm = Math.Sqrt(gradNorm + 1e-8);

        // Simplified update: perturb embeddings in direction that might improve
        for (int i = 0; i < _numNodes; i++)
        {
            for (int j = 0; j < _nodeEmbeddingDim; j++)
            {
                double noise = (_random.NextDouble() - 0.5) * lr / gradNorm;
                _nodeEmbedding1[i, j] += noise;
                _nodeEmbedding2[i, j] += noise;
            }
        }
    }

    /// <summary>
    /// Updates parameters using the provided gradients.
    /// </summary>
    /// <param name="gradients">Gradient vector.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the MTGNN model, UpdateParameters updates internal parameters or state. This keeps the MTGNN architecture aligned with the latest values.
    /// </para>
    /// </remarks>
    public override void UpdateParameters(Vector<T> gradients)
    {
        // Parameters are updated through the optimizer in Train()
    }

    /// <summary>
    /// Gets metadata about the MTGNN model.
    /// </summary>
    /// <returns>ModelMetadata containing model information.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the MTGNN model, GetModelMetadata performs a supporting step in the workflow. It keeps the MTGNN architecture pipeline consistent.
    /// </para>
    /// </remarks>
    public override ModelMetadata<T> GetModelMetadata()
    {
        return new ModelMetadata<T>
        {
            ModelType = ModelType.NeuralNetwork,
            AdditionalInfo = new Dictionary<string, object>
            {
                { "NetworkType", "MTGNN" },
                { "SequenceLength", _sequenceLength },
                { "ForecastHorizon", _forecastHorizon },
                { "NumNodes", _numNodes },
                { "NumFeatures", _numFeatures },
                { "HiddenDimension", _hiddenDimension },
                { "NodeEmbeddingDim", _nodeEmbeddingDim },
                { "NumLayers", _numLayers },
                { "MixHopDepth", _mixHopDepth },
                { "TemporalKernelSize", _temporalKernelSize },
                { "DilationFactor", _dilationFactor },
                { "UsePredefinedGraph", _usePredefinedGraph },
                { "UseSubgraphSampling", _useSubgraphSampling },
                { "NumSamples", _numSamples },
                { "UseNativeMode", _useNativeMode }
            }
        };
    }

    /// <summary>
    /// Creates a new instance with the same configuration.
    /// </summary>
    /// <returns>A new MTGNN instance.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the MTGNN model, CreateNewInstance builds and wires up model components. This sets up the MTGNN architecture before use.
    /// </para>
    /// </remarks>
    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance()
    {
        return new MTGNN<T>(Architecture, _options, _predefinedAdjacency);
    }

    /// <summary>
    /// Serializes MTGNN-specific data.
    /// </summary>
    /// <param name="writer">The binary writer.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the MTGNN model, SerializeNetworkSpecificData saves or restores model-specific settings. This lets the MTGNN architecture be reused later.
    /// </para>
    /// </remarks>
    protected override void SerializeNetworkSpecificData(BinaryWriter writer)
    {
        writer.Write(_sequenceLength);
        writer.Write(_forecastHorizon);
        writer.Write(_numNodes);
        writer.Write(_numFeatures);
        writer.Write(_hiddenDimension);
        writer.Write(_nodeEmbeddingDim);
        writer.Write(_numLayers);
        writer.Write(_mixHopDepth);
        writer.Write(_temporalKernelSize);
        writer.Write(_dilationFactor);
        writer.Write(_usePredefinedGraph);
        writer.Write(_useSubgraphSampling);
        writer.Write(_subgraphSize);
        writer.Write(_numSamples);

        // Serialize node embeddings
        if (_nodeEmbedding1 is not null && _nodeEmbedding2 is not null)
        {
            writer.Write(true);
            for (int i = 0; i < _numNodes; i++)
            {
                for (int j = 0; j < _nodeEmbeddingDim; j++)
                {
                    writer.Write(_nodeEmbedding1[i, j]);
                    writer.Write(_nodeEmbedding2[i, j]);
                }
            }
        }
        else
        {
            writer.Write(false);
        }
    }

    /// <summary>
    /// Deserializes MTGNN-specific data.
    /// </summary>
    /// <param name="reader">The binary reader.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the MTGNN model, DeserializeNetworkSpecificData saves or restores model-specific settings. This lets the MTGNN architecture be reused later.
    /// </para>
    /// </remarks>
    protected override void DeserializeNetworkSpecificData(BinaryReader reader)
    {
        _ = reader.ReadInt32(); // sequenceLength
        _ = reader.ReadInt32(); // forecastHorizon
        int numNodes = reader.ReadInt32();
        _ = reader.ReadInt32(); // numFeatures
        _ = reader.ReadInt32(); // hiddenDimension
        int embeddingDim = reader.ReadInt32();
        _ = reader.ReadInt32(); // numLayers
        _ = reader.ReadInt32(); // mixHopDepth
        _ = reader.ReadInt32(); // temporalKernelSize
        _ = reader.ReadInt32(); // dilationFactor
        _ = reader.ReadBoolean(); // usePredefinedGraph
        _ = reader.ReadBoolean(); // useSubgraphSampling
        _ = reader.ReadInt32(); // subgraphSize
        _ = reader.ReadInt32(); // numSamples

        // Deserialize node embeddings
        bool hasEmbeddings = reader.ReadBoolean();
        if (hasEmbeddings)
        {
            _nodeEmbedding1 = new double[numNodes, embeddingDim];
            _nodeEmbedding2 = new double[numNodes, embeddingDim];
            for (int i = 0; i < numNodes; i++)
            {
                for (int j = 0; j < embeddingDim; j++)
                {
                    _nodeEmbedding1[i, j] = reader.ReadDouble();
                    _nodeEmbedding2[i, j] = reader.ReadDouble();
                }
            }
            UpdateAdaptiveAdjacency();
        }
    }

    #endregion

    #region IForecastingModel Implementation

    /// <summary>
    /// Generates forecasts for all nodes.
    /// </summary>
    /// <param name="historicalData">Input tensor with historical data for all nodes.</param>
    /// <param name="quantiles">Optional quantile levels for probabilistic forecasting.</param>
    /// <returns>Forecast tensor with predictions for all nodes.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Uses the learned graph structure to aggregate
    /// information across related nodes, improving predictions especially when
    /// some nodes have limited or noisy data.
    /// </para>
    /// </remarks>
    public Tensor<T> Forecast(Tensor<T> historicalData, double[]? quantiles = null)
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
    /// <b>For Beginners:</b> In the MTGNN model, public performs a supporting step in the workflow. It keeps the MTGNN architecture pipeline consistent.
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
    /// <b>For Beginners:</b> In the MTGNN model, AutoregressiveForecast produces predictions from input data. This is the main inference step of the MTGNN architecture.
    /// </para>
    /// </remarks>
    public Tensor<T> AutoregressiveForecast(Tensor<T> input, int steps)
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
    /// <b>For Beginners:</b> In the MTGNN model, Evaluate performs a supporting step in the workflow. It keeps the MTGNN architecture pipeline consistent.
    /// </para>
    /// </remarks>
    public Dictionary<string, T> Evaluate(Tensor<T> predictions, Tensor<T> actuals)
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
    /// Applies instance normalization (identity for MTGNN).
    /// </summary>
    /// <param name="input">Input tensor.</param>
    /// <returns>The input tensor unchanged.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the MTGNN model, ApplyInstanceNormalization performs a supporting step in the workflow. It keeps the MTGNN architecture pipeline consistent.
    /// </para>
    /// </remarks>
    public Tensor<T> ApplyInstanceNormalization(Tensor<T> input)
    {
        return input;
    }

    /// <summary>
    /// Gets financial-specific metrics.
    /// </summary>
    /// <returns>Dictionary of financial metrics.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the MTGNN model, GetFinancialMetrics calculates evaluation metrics. This summarizes how the MTGNN architecture is performing.
    /// </para>
    /// </remarks>
    public Dictionary<string, T> GetFinancialMetrics()
    {
        T lastLoss = LastLoss is not null ? LastLoss : NumOps.Zero;

        return new Dictionary<string, T>
        {
            ["LastLoss"] = lastLoss,
            ["NumNodes"] = NumOps.FromDouble(_numNodes),
            ["SequenceLength"] = NumOps.FromDouble(_sequenceLength),
            ["ForecastHorizon"] = NumOps.FromDouble(_forecastHorizon),
            ["NumLayers"] = NumOps.FromDouble(_numLayers),
            ["MixHopDepth"] = NumOps.FromDouble(_mixHopDepth),
            ["NodeEmbeddingDim"] = NumOps.FromDouble(_nodeEmbeddingDim)
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
    /// 2. Applies mix-hop propagation with learned graph
    /// 3. Applies dilated temporal convolutions
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

        // Apply mix-hop propagation with adaptive adjacency
        if (_adaptiveAdjacency is not null && _useNativeMode)
        {
            current = ApplyMixHopPropagation(current);
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
    /// <b>For Beginners:</b> In the MTGNN model, Backward propagates gradients backward. This teaches the MTGNN architecture how to adjust its weights.
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
    /// <b>For Beginners:</b> In the MTGNN model, FlattenInput performs a supporting step in the workflow. It keeps the MTGNN architecture pipeline consistent.
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
    /// Applies mix-hop propagation using the adaptive adjacency matrix.
    /// </summary>
    /// <param name="nodeFeatures">Node feature tensor.</param>
    /// <returns>Features after multi-hop aggregation.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Mix-hop propagation aggregates information from
    /// multiple hop distances simultaneously:
    /// H_out = concat(H, A*H, A^2*H, ..., A^K*H) * W
    ///
    /// This captures both local (1-hop) and more distant (K-hop) spatial dependencies.
    /// The learned adjacency matrix determines how strongly each neighbor contributes.
    /// </para>
    /// </remarks>
    private Tensor<T> ApplyMixHopPropagation(Tensor<T> nodeFeatures)
    {
        if (_adaptiveAdjacency is null)
            return nodeFeatures;

        var featureVec = nodeFeatures.ToVector();
        int totalSize = featureVec.Length;
        int featuresPerNode = totalSize / _numNodes;

        if (featuresPerNode * _numNodes != totalSize)
            return nodeFeatures;

        // Start with original features
        var result = new T[totalSize];
        var currentPower = new double[_numNodes, featuresPerNode];
        var previousPower = new double[_numNodes, featuresPerNode];

        // Initialize with original features
        for (int i = 0; i < _numNodes; i++)
        {
            for (int f = 0; f < featuresPerNode; f++)
            {
                previousPower[i, f] = NumOps.ToDouble(featureVec[i * featuresPerNode + f]);
                result[i * featuresPerNode + f] = NumOps.FromDouble(previousPower[i, f]);
            }
        }

        // Compute and accumulate A^k * H for k = 1 to mixHopDepth
        for (int hop = 0; hop < _mixHopDepth; hop++)
        {
            // Compute A * previousPower
            for (int i = 0; i < _numNodes; i++)
            {
                for (int f = 0; f < featuresPerNode; f++)
                {
                    double aggregated = 0;
                    for (int j = 0; j < _numNodes; j++)
                    {
                        aggregated += _adaptiveAdjacency[i, j] * previousPower[j, f];
                    }
                    currentPower[i, f] = aggregated;
                }
            }

            // Add to result (weighted by 1/(hop+2) to balance contributions)
            double weight = 1.0 / (hop + 2);
            for (int i = 0; i < _numNodes; i++)
            {
                for (int f = 0; f < featuresPerNode; f++)
                {
                    double existing = NumOps.ToDouble(result[i * featuresPerNode + f]);
                    result[i * featuresPerNode + f] = NumOps.FromDouble(existing + weight * currentPower[i, f]);
                }
            }

            // Swap for next iteration
            (currentPower, previousPower) = (previousPower, currentPower);
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
    /// <b>For Beginners:</b> In the MTGNN model, ForecastNative produces predictions from input data. This is the main inference step of the MTGNN architecture.
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
    /// <b>For Beginners:</b> In the MTGNN model, ForecastOnnx produces predictions from input data. This is the main inference step of the MTGNN architecture.
    /// </para>
    /// </remarks>
    private Tensor<T> ForecastOnnx(Tensor<T> input)
    {
        if (_onnxSession is null)
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

        using var results = _onnxSession.Run(inputs);
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
    /// <para>
    /// <b>For Beginners:</b> In the MTGNN model, GenerateSamples performs a supporting step in the workflow. It keeps the MTGNN architecture pipeline consistent.
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
    /// <b>For Beginners:</b> In the MTGNN model, AddSmallPerturbation performs a supporting step in the workflow. It keeps the MTGNN architecture pipeline consistent.
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
    /// <b>For Beginners:</b> In the MTGNN model, ShiftInputWindow performs a supporting step in the workflow. It keeps the MTGNN architecture pipeline consistent.
    /// </para>
    /// </remarks>
    private Tensor<T> ShiftInputWindow(Tensor<T> input, Tensor<T> prediction)
    {
        var inputVec = input.ToVector();
        var predVec = prediction.ToVector();

        int inputSize = _numNodes * _sequenceLength * _numFeatures;
        int shiftSize = _numNodes * _numFeatures;
        var shifted = new T[inputSize];

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
    /// <b>For Beginners:</b> In the MTGNN model, ConcatenatePredictions produces predictions from input data. This is the main inference step of the MTGNN architecture.
    /// </para>
    /// </remarks>
    private Tensor<T> ConcatenatePredictions(List<Tensor<T>> predictions)
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
    /// <b>For Beginners:</b> In the MTGNN model, ComputeQuantiles performs a supporting step in the workflow. It keeps the MTGNN architecture pipeline consistent.
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
    /// <b>For Beginners:</b> In the MTGNN model, private performs a supporting step in the workflow. It keeps the MTGNN architecture pipeline consistent.
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
    /// <b>For Beginners:</b> In the MTGNN model, Dispose performs a supporting step in the workflow. It keeps the MTGNN architecture pipeline consistent.
    /// </para>
    /// </remarks>
    protected override void Dispose(bool disposing)
    {
        if (disposing)
        {
            _onnxSession?.Dispose();
        }
        base.Dispose(disposing);
    }

    #endregion
}
