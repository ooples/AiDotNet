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
/// <example>
/// <code>
/// // Define architecture for multivariate time series with auto-discovered graph (100 variables, 24-step forecast)
/// var architecture = new NeuralNetworkArchitecture&lt;double&gt;(
///     inputType: InputType.OneDimensional,
///     taskType: NeuralNetworkTaskType.Regression,
///     inputHeight: 60, inputWidth: 100, inputDepth: 1, outputSize: 24);
///
/// // Training mode: automatically discovers inter-variable relationships
/// var model = new MTGNN&lt;double&gt;(architecture);
///
/// // ONNX inference mode: load pre-trained MTGNN model
/// var onnxModel = new MTGNN&lt;double&gt;(architecture, "mtgnn_forecast.onnx");
/// </code>
/// </example>
[ModelDomain(ModelDomain.Finance)]
[ModelDomain(ModelDomain.GraphAnalysis)]
[ModelCategory(ModelCategory.GraphNetwork)]
[ModelCategory(ModelCategory.ConvolutionalNetwork)]
[ModelTask(ModelTask.Forecasting)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("Connecting the Dots: Multivariate Time Series Forecasting with Graph Neural Networks", "https://arxiv.org/abs/2005.11650", Year = 2020, Authors = "Zonghan Wu, Shirui Pan, Guodong Long, Jing Jiang, Xiaojun Chang, Chengqi Zhang")]
[PaperOptimizer(OptimizerKind.Adam, MaxGradientNorm = 5.0,
                Source = "Wu et al. 2020, Sec. 4: the model is trained by the Adam optimizer with a "
                        + "gradient clip of 5. The paper states no learning rate or batch size in its "
                        + "training description, so neither is declared.")]
public partial class MTGNN<T> : ForecastingModelBase<T>
{
    #region Execution Mode
    private readonly bool _useNativeMode;
    #endregion

    
    #region Native Mode Fields
    private DenseLayer<T>? _inputProjection;
    private DenseLayer<T>? _nodeEmbeddingLayer;
    private List<LayerNormalizationLayer<T>>? _layerNorms;
    private List<DropoutLayer<T>>? _dropoutLayers;
    private DenseLayer<T>? _outputLayer;
    #endregion

    #region Graph Learning Fields
    /// <summary>Node embedding E1, trained through the adaptive adjacency.</summary>
    [TrainableParameter]
    private Tensor<T>? _nodeEmbedding1;
    /// <summary>Node embedding E2, trained through the adaptive adjacency.</summary>
    [TrainableParameter]
    private Tensor<T>? _nodeEmbedding2;
    // Not readonly: a checkpoint restore replaces the graph (and what is derived from it) rather than
    // copying into a constructor-built tensor that may be absent or shaped for a different NumNodes.
    private Tensor<T>? _predefinedAdjacency;
    private readonly Random _random;
    #endregion

    #region Shared Fields
    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> _optimizer;
    private readonly ILossFunction<T> _lossFunction;
    private readonly MTGNNOptions<T> _options;

    private int _sequenceLength;
    private int _forecastHorizon;
    private int _numNodes;
    private int _numFeatures;
    private int _hiddenDimension;
    private int _nodeEmbeddingDim;
    private int _numLayers;
    private int _mixHopDepth;
    private int _temporalKernelSize;
    private int _dilationFactor;
    private int _numSamples;
    private bool _usePredefinedGraph;
    private bool _useSubgraphSampling;
    private int _subgraphSize;
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
    public Tensor<T>? LearnedAdjacency => CurrentAdaptiveAdjacency();

    #endregion

    #region Constructors

    /// <summary>
    /// Initializes a new instance of the MTGNN model in ONNX mode for inference.
    /// </summary>
    /// <param name="architecture">The neural network architecture configuration.</param>
    /// <param name="onnxModelPath">Path to a pretrained ONNX model file.</param>
    /// <param name="options">MTGNN-specific options.</param>
    /// <param name="predefinedAdjacency">Optional predefined adjacency matrix to combine with learned graph.</param>
    /// <remarks>
    /// <para>Every parameter is required: an ONNX model is inference-only, so it takes no optimizer or loss
    /// function, and the native constructor is the one overload with optional parameters.</para>
    /// <para><b>For Beginners:</b> Use this constructor to load a pretrained MTGNN model.
    /// The ONNX model includes both the learned graph structure and the forecasting layers.
    /// </para>
    /// </remarks>
    public MTGNN(
        NeuralNetworkArchitecture<T> architecture,
        string onnxModelPath,
        MTGNNOptions<T>? options,
        Tensor<T>? predefinedAdjacency)
        : base(architecture, new MeanSquaredErrorLoss<T>(), 1.0)
    {
        if (string.IsNullOrWhiteSpace(onnxModelPath))
            throw new ArgumentNullException(nameof(onnxModelPath));
        if (!System.IO.File.Exists(onnxModelPath))
            throw new System.IO.FileNotFoundException($"ONNX model not found: {onnxModelPath}");

        _useNativeMode = false;
        OnnxModelPath = onnxModelPath;
        OnnxSession = new InferenceSession(onnxModelPath);
        _options = options ?? new MTGNNOptions<T>();
        Options = _options;
        _lossFunction = new MeanSquaredErrorLoss<T>();
        _optimizer = PaperOptimizerFactory.CreateFor<T, Tensor<T>, Tensor<T>>(this)
    ?? new AdamOptimizer<T, Tensor<T>, Tensor<T>>(this);

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

        RequireAdjacencyNodeCount(predefinedAdjacency, _numNodes, nameof(predefinedAdjacency));
        _predefinedAdjacency = predefinedAdjacency;
        _random = LayerInitializationSeedScope.CreateComponentRandom(_options.Seed);

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
        Tensor<T>? predefinedAdjacency = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
        ILossFunction<T>? lossFunction = null)
        : base(architecture, lossFunction ?? new MeanSquaredErrorLoss<T>(), 1.0)
    {
        _useNativeMode = true;
        _options = options ?? new MTGNNOptions<T>();
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
        _nodeEmbeddingDim = _options.NodeEmbeddingDim;
        _numLayers = _options.NumLayers;
        _mixHopDepth = _options.MixHopDepth;
        _temporalKernelSize = _options.TemporalKernelSize;
        _dilationFactor = _options.DilationFactor;
        _numSamples = _options.NumSamples;
        _usePredefinedGraph = _options.UsePredefinedGraph;
        _useSubgraphSampling = _options.UseSubgraphSampling;
        _subgraphSize = _options.SubgraphSize;

        RequireAdjacencyNodeCount(predefinedAdjacency, _numNodes, nameof(predefinedAdjacency));
        _predefinedAdjacency = predefinedAdjacency;
        _random = LayerInitializationSeedScope.CreateComponentRandom(_options.Seed);

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
        // Uniform on [-scale, scale], as master's (u - 0.5) * 2 * scale draws were, sampled by the engine in T.
        // Each embedding's stream is seeded from the model's Random, so a fixed Seed reproduces both.
        double scale = Math.Sqrt(2.0 / (_numNodes + _nodeEmbeddingDim));
        var shape = new[] { _numNodes, _nodeEmbeddingDim };
        T low = NumOps.FromDouble(-scale);
        T high = NumOps.FromDouble(scale);
        _nodeEmbedding1 = Engine.TensorRandomUniformRange(shape, low, high, _random.Next());
        _nodeEmbedding2 = Engine.TensorRandomUniformRange(shape, low, high, _random.Next());
    }

    /// <summary>
    /// The learned graph built from the current node embeddings, as a detached constant, or null when the model
    /// has no embeddings. Computed on demand so it always reflects the embeddings, including after the
    /// parameters are restored or copied into a clone.
    /// </summary>
    private Tensor<T>? CurrentAdaptiveAdjacency()
    {
        if (_nodeEmbedding1 is null || _nodeEmbedding2 is null)
            return null;

        using var noGrad = new AiDotNet.Tensors.Engines.Autodiff.NoGradScope<T>();
        return ComputeAdaptiveAdjacency(_nodeEmbedding1, _nodeEmbedding2);
    }

    /// <summary>
    /// softmax(ReLU(E1 E2^T - E2 E1^T)) over each row, averaged with the predefined graph when one is used.
    /// Built from engine ops only, so under the gradient tape the embeddings receive gradients.
    /// </summary>
    private Tensor<T> ComputeAdaptiveAdjacency(Tensor<T> embedding1, Tensor<T> embedding2)
    {
        var forward = Engine.TensorMatMul(embedding1, Engine.TensorTranspose(embedding2));
        var backward = Engine.TensorMatMul(embedding2, Engine.TensorTranspose(embedding1));
        var adjacency = Engine.TensorSoftmax(Engine.TensorReLU(Engine.TensorSubtract(forward, backward)), 1);

        // Combine with predefined adjacency if available
        if (_usePredefinedGraph && _predefinedAdjacency is not null)
        {
            var half = NumOps.FromDouble(0.5);
            adjacency = Engine.TensorAdd(
                Engine.TensorMultiplyScalar(adjacency, half),
                Engine.TensorMultiplyScalar(Engine.Reshape(_predefinedAdjacency.Contiguous(), new[] { _numNodes, _numNodes }), half));
        }

        return adjacency;
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
            ValidateCustomLayersWithOwnershipRollback(Layers);
        }
        else if (_useNativeMode)
        {
            Layers.AddRange(LayerHelper<T>.CreateDefaultMTGNNLayers(
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

            // The interleaved mix-hop, temporal and gate layers run through Layers; no separate list reads them.

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
    protected override Tensor<T> PredictCore(Tensor<T> input)
    {
        return _useNativeMode ? ForecastNative(input) : ForecastOnnx(input);
    }

    /// <summary>
    /// Training-mode forward: mirrors <see cref="Forward"/>, but rebuilds the adaptive adjacency from the
    /// node embeddings on the tape and propagates through <see cref="ApplyMixHopPropagationTape"/>, so
    /// gradients reach both the dense layers below the mix-hop step and the embeddings E1 and E2.
    /// </summary>
    protected override Tensor<T> ForwardNativeForTraining(Tensor<T> input)
    {
        var current = ReshapeToNodes(input);
        foreach (var layer in Layers)
            current = layer.Forward(current);
        if (_useNativeMode && _nodeEmbedding1 is not null && _nodeEmbedding2 is not null)
            current = ApplyMixHopPropagationTape(current, ComputeAdaptiveAdjacency(_nodeEmbedding1, _nodeEmbedding2));
        return current;
    }

    /// <summary>
    /// Reshapes the input to [numNodes, featuresPerNode] (numNodes as the batch
    /// dimension) via the tape-aware Engine.Reshape so the per-node MLP layers apply
    /// SHARED weights across nodes and gradients flow back to the input.
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
    /// Tape-aware MixHop propagation: accumulates weighted powers of
    /// the adaptive adjacency matrix times the node-feature matrix
    /// (<c>H + sum_k w_k * A^k * H</c>) via
    /// <see cref="IEngine.TensorMatMul"/>, preserving the gradient tape so backward can
    /// propagate through the whole expression. Both the inference and training forwards use it.
    /// </summary>
    private Tensor<T> ApplyMixHopPropagationTape(Tensor<T> nodeFeatures, Tensor<T> adjacency)
    {
        int totalSize = nodeFeatures.Length;
        int featuresPerNode = totalSize / _numNodes;
        if (featuresPerNode * _numNodes != totalSize)
            return nodeFeatures;

        var adj = adjacency;

        var hMat = Engine.Reshape(nodeFeatures, new[] { _numNodes, featuresPerNode });
        var acc = hMat; // k=0 term (identity)
        var currentPower = hMat;
        for (int hop = 0; hop < _mixHopDepth; hop++)
        {
            currentPower = Engine.TensorMatMul(adj, currentPower);
            T weight = NumOps.FromDouble(1.0 / (hop + 2));
            var weighted = Engine.TensorMultiplyScalar(currentPower, weight);
            acc = Engine.TensorAdd(acc, weighted);
        }
        return Engine.Reshape(acc, nodeFeatures._shape);
    }


    // UpdateParameters was an empty override, silently dropping every restore. The base
    // distributes the vector over the declared enumeration.
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
    /// Serializes MTGNN-specific data.
    /// </summary>
    /// <param name="writer">The binary writer.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the MTGNN model, SerializeNetworkSpecificData saves or restores model-specific settings. This lets the MTGNN architecture be reused later.
    /// </para>
    /// </remarks>


    /// <summary>
    /// Deserializes MTGNN-specific data.
    /// </summary>
    /// <param name="reader">The binary reader.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the MTGNN model, DeserializeNetworkSpecificData saves or restores model-specific settings. This lets the MTGNN architecture be reused later.
    /// </para>
    /// </remarks>


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
    /// <b>For Beginners:</b> In the MTGNN model, Evaluate performs a supporting step in the workflow. It keeps the MTGNN architecture pipeline consistent.
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
    /// Applies instance normalization (identity for MTGNN).
    /// </summary>
    /// <param name="input">Input tensor.</param>
    /// <returns>The input tensor unchanged.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the MTGNN model, ApplyInstanceNormalization performs a supporting step in the workflow. It keeps the MTGNN architecture pipeline consistent.
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
    /// <b>For Beginners:</b> In the MTGNN model, GetFinancialMetrics calculates evaluation metrics. This summarizes how the MTGNN architecture is performing.
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
        var current = ReshapeToNodes(input);

        // Per-node temporal stack (shared weights across nodes).
        foreach (var layer in Layers)
        {
            current = layer.Forward(current);
        }

        // Mix-hop graph propagation (spatial aggregation). Use the tape-aware variant
        // so inference matches the training forward (determinism / Clone parity).
        var adaptiveAdjacency = _useNativeMode ? CurrentAdaptiveAdjacency() : null;
        if (adaptiveAdjacency is not null)
        {
            current = ApplyMixHopPropagationTape(current, adaptiveAdjacency);
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
    /// <b>For Beginners:</b> In the MTGNN model, FlattenInput performs a supporting step in the workflow. It keeps the MTGNN architecture pipeline consistent.
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
    /// <b>For Beginners:</b> In the MTGNN model, ShiftInputWindow performs a supporting step in the workflow. It keeps the MTGNN architecture pipeline consistent.
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
    /// <b>For Beginners:</b> In the MTGNN model, ConcatenatePredictions produces predictions from input data. This is the main inference step of the MTGNN architecture.
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
    /// <b>For Beginners:</b> In the MTGNN model, ComputeQuantiles performs a supporting step in the workflow. It keeps the MTGNN architecture pipeline consistent.
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
    /// <b>For Beginners:</b> In the MTGNN model, Dispose performs a supporting step in the workflow. It keeps the MTGNN architecture pipeline consistent.
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



