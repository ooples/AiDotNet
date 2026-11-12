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
/// GraphWaveNet (Graph WaveNet) for deep spatial-temporal graph modeling.
/// </summary>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
/// <remarks>
/// <para>
/// GraphWaveNet combines adaptive graph learning with WaveNet-style dilated causal
/// convolutions for state-of-the-art traffic and time series forecasting.
/// </para>
/// <para><b>For Beginners:</b> GraphWaveNet achieves top performance on traffic forecasting by combining:
///
/// <b>The Key Insight:</b>
/// Traditional methods either use a fixed graph structure OR learn it separately from forecasting.
/// GraphWaveNet jointly learns the graph structure AND the forecasting model, allowing them to
/// inform each other during training.
///
/// <b>What Problems Does GraphWaveNet Solve?</b>
/// - Traffic speed/flow prediction on road networks
/// - Air quality forecasting across sensor networks
/// - Electricity demand prediction across power grids
/// - Any spatial-temporal forecasting with underlying graph structure
///
/// <b>How GraphWaveNet Works:</b>
/// 1. <b>Adaptive Graph:</b> Learns A = softmax(ReLU(E1 * E2^T)) from node embeddings
/// 2. <b>Diffusion Conv:</b> H' = sum_k(P_f^k * H * W_k + P_b^k * H * V_k) for bidirectional propagation
/// 3. <b>Gated TCN:</b> tanh(conv_f) ⊙ sigmoid(conv_g) with exponentially increasing dilation
/// 4. <b>Skip Connections:</b> Sum outputs from all layers for multi-scale features
///
/// <b>GraphWaveNet Architecture:</b>
/// - Node Embeddings E1, E2: Learnable [num_nodes, embedding_dim] matrices
/// - Diffusion Convolution: Forward (A) and backward (A^T) random walk diffusion
/// - Gated TCN: Filter-Gate mechanism with dilated causal convolutions
/// - Skip Connections: Residual learning + skip from each layer to output
/// - Output: ReLU + Linear projection to forecast dimension
///
/// <b>Key Benefits:</b>
/// - Learns graph structure without prior knowledge
/// - Captures complex spatial dependencies via bidirectional diffusion
/// - Efficient training via parallel dilated convolutions
/// - State-of-the-art on METR-LA and PEMS-BAY traffic datasets
/// </para>
/// <para>
/// <b>Reference:</b> Wu et al., "Graph WaveNet for Deep Spatial-Temporal Graph Modeling", IJCAI 2019.
/// https://arxiv.org/abs/1906.00121
/// </para>
/// </remarks>
/// <example>
/// <code>
/// // Define architecture for adaptive graph traffic forecasting (207 sensors, 12-step horizon)
/// var architecture = new NeuralNetworkArchitecture&lt;double&gt;(
///     inputType: InputType.OneDimensional,
///     taskType: NeuralNetworkTaskType.Regression,
///     inputHeight: 12, inputWidth: 207, inputDepth: 2, outputSize: 12);
///
/// // Training mode: learns graph structure adaptively via node embeddings
/// var model = new GraphWaveNet&lt;double&gt;(architecture);
///
/// // ONNX inference mode: load pre-trained Graph WaveNet model
/// var onnxModel = new GraphWaveNet&lt;double&gt;(architecture, "graphwavenet.onnx");
/// </code>
/// </example>
[ModelDomain(ModelDomain.Finance)]
[ModelDomain(ModelDomain.GraphAnalysis)]
[ModelCategory(ModelCategory.GraphNetwork)]
[ModelCategory(ModelCategory.ConvolutionalNetwork)]
[ModelTask(ModelTask.Forecasting)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("Graph WaveNet for Deep Spatial-Temporal Graph Modeling", "https://arxiv.org/abs/1906.00121", Year = 2019, Authors = "Zonghan Wu, Shirui Pan, Guodong Long, Jing Jiang, Chengqi Zhang")]
[PaperOptimizer(OptimizerKind.Adam, LearningRate = 0.001,
                Source = "Wu et al. 2019, Sec. 4: Adam with an initial learning rate of 0.001.")]
public partial class GraphWaveNet<T> : ForecastingModelBase<T>
{
    #region Execution Mode
    private readonly bool _useNativeMode;
    #endregion

    
    #region Native Mode Fields
    private DenseLayer<T>? _inputProjection;
    private List<LayerNormalizationLayer<T>>? _layerNorms;
    private List<DropoutLayer<T>>? _dropoutLayers;
    private DenseLayer<T>? _endLayer1;
    private DenseLayer<T>? _outputLayer;
    #endregion

    #region Graph Fields
    /// <summary>Node embedding E1, trained through the adaptive adjacency.</summary>
    [TrainableParameter]
    private Tensor<T>? _nodeEmbedding1;
    /// <summary>Node embedding E2, trained through the adaptive adjacency.</summary>
    [TrainableParameter]
    private Tensor<T>? _nodeEmbedding2;
    private readonly Tensor<T>? _predefinedAdjacency;
    private readonly Tensor<T>? _predefinedAdjacencyTranspose;
    private readonly Random _random;
    #endregion

    #region Shared Fields
    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> _optimizer;
    private readonly ILossFunction<T> _lossFunction;
    private readonly GraphWaveNetOptions<T> _options;

    private int _sequenceLength;
    private int _forecastHorizon;
    private int _numNodes;
    private int _numFeatures;
    private int _residualChannels;
    private int _dilationChannels;
    private int _skipChannels;
    private int _endChannels;
    private int _nodeEmbeddingDim;
    private int _numBlocks;
    private int _layersPerBlock;
    private int _diffusionSteps;
    private int _numSamples;
    private bool _useAdaptiveGraph;
    private bool _usePredefinedGraph;
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
    public int NumNodes => _numNodes;

    /// <summary>
    /// Gets the forecast horizon.
    /// </summary>
    public int ForecastHorizon => _forecastHorizon;

    /// <summary>
    /// Gets whether the model supports training.
    /// </summary>
    public override bool SupportsTraining => _useNativeMode;

    /// <summary>
    /// Gets the number of WaveNet blocks.
    /// </summary>
    public int NumBlocks => _numBlocks;

    /// <summary>
    /// Gets the number of diffusion steps (K).
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Diffusion steps control how far information can travel
    /// across the graph. More steps let the model capture longer-range relationships.
    /// </para>
    /// </remarks>
    public int DiffusionSteps => _diffusionSteps;

    /// <summary>
    /// Gets whether adaptive graph learning is enabled.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> When enabled, the model learns the graph connections
    /// directly from data instead of relying only on a predefined graph.
    /// </para>
    /// </remarks>
    public bool UseAdaptiveGraph => _useAdaptiveGraph;

    /// <summary>
    /// Gets the number of samples for uncertainty estimation.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This is how many times the model samples predictions
    /// to estimate uncertainty around its forecasts.
    /// </para>
    /// </remarks>
    public int NumSamples => _numSamples;

    /// <summary>
    /// Gets the learned adaptive adjacency matrix.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This is the graph the model has learned, showing how
    /// strongly each node influences others.
    /// </para>
    /// </remarks>
    public Tensor<T>? LearnedAdjacency => CurrentAdaptiveAdjacency();

    #endregion

    #region Constructors

    /// <summary>
    /// Initializes a new instance of the GraphWaveNet model in ONNX mode for inference.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the GraphWaveNet model, GraphWaveNet sets up the architecture and options. This prepares the model for training or inference.
    /// </para>
    /// </remarks>
    [System.Diagnostics.CodeAnalysis.SuppressMessage("ApiDesign", "RS0026:Do not add multiple public overloads with optional parameters", Justification = "Both constructors replace removed overloads whose adjacency parameter changed type; no old overload remains for a call to become ambiguous with.")]
    public GraphWaveNet(
        NeuralNetworkArchitecture<T> architecture,
        string onnxModelPath,
        GraphWaveNetOptions<T>? options = null,
        Tensor<T>? predefinedAdjacency = null,
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
        _options = options ?? new GraphWaveNetOptions<T>();
        Options = _options;
        _lossFunction = lossFunction ?? new MeanSquaredErrorLoss<T>();
        _optimizer = optimizer
    ?? PaperOptimizerFactory.CreateFor<T, Tensor<T>, Tensor<T>>(this)
    ?? new AdamOptimizer<T, Tensor<T>, Tensor<T>>(this);

        _sequenceLength = _options.SequenceLength;
        _forecastHorizon = _options.ForecastHorizon;
        _numNodes = _options.NumNodes;
        _numFeatures = _options.NumFeatures;
        _residualChannels = _options.ResidualChannels;
        _dilationChannels = _options.DilationChannels;
        _skipChannels = _options.SkipChannels;
        _endChannels = _options.EndChannels;
        _nodeEmbeddingDim = _options.NodeEmbeddingDim;
        _numBlocks = _options.NumBlocks;
        _layersPerBlock = _options.LayersPerBlock;
        _diffusionSteps = _options.DiffusionSteps;
        _numSamples = _options.NumSamples;
        _useAdaptiveGraph = _options.UseAdaptiveGraph;
        _usePredefinedGraph = _options.UsePredefinedGraph;

        RequireAdjacencyNodeCount(predefinedAdjacency, _numNodes, nameof(predefinedAdjacency));
        _predefinedAdjacency = predefinedAdjacency;
        _predefinedAdjacencyTranspose = predefinedAdjacency is not null ? Engine.TensorTranspose(predefinedAdjacency).Contiguous() : null;
        _random = LayerInitializationSeedScope.CreateComponentRandom(_options.Seed);

        if (_useAdaptiveGraph)
            InitializeNodeEmbeddings();
    }

    /// <summary>
    /// Initializes a new instance of the GraphWaveNet model in native mode for training.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the GraphWaveNet model, GraphWaveNet sets up the architecture and options. This prepares the model for training or inference.
    /// </para>
    /// </remarks>
    [System.Diagnostics.CodeAnalysis.SuppressMessage("ApiDesign", "RS0026:Do not add multiple public overloads with optional parameters", Justification = "Both constructors replace removed overloads whose adjacency parameter changed type; no old overload remains for a call to become ambiguous with.")]
    public GraphWaveNet(
        NeuralNetworkArchitecture<T> architecture,
        GraphWaveNetOptions<T>? options = null,
        Tensor<T>? predefinedAdjacency = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
        ILossFunction<T>? lossFunction = null)
        : base(architecture, lossFunction ?? new MeanSquaredErrorLoss<T>(), 1.0)
    {
        _useNativeMode = true;
        _options = options ?? new GraphWaveNetOptions<T>();
        Options = _options;
        _lossFunction = lossFunction ?? new MeanSquaredErrorLoss<T>();
        _optimizer = optimizer
    ?? PaperOptimizerFactory.CreateFor<T, Tensor<T>, Tensor<T>>(this)
    ?? new AdamOptimizer<T, Tensor<T>, Tensor<T>>(this);

        _sequenceLength = _options.SequenceLength;
        _forecastHorizon = _options.ForecastHorizon;
        _numNodes = _options.NumNodes;
        _numFeatures = _options.NumFeatures;
        _residualChannels = _options.ResidualChannels;
        _dilationChannels = _options.DilationChannels;
        _skipChannels = _options.SkipChannels;
        _endChannels = _options.EndChannels;
        _nodeEmbeddingDim = _options.NodeEmbeddingDim;
        _numBlocks = _options.NumBlocks;
        _layersPerBlock = _options.LayersPerBlock;
        _diffusionSteps = _options.DiffusionSteps;
        _numSamples = _options.NumSamples;
        _useAdaptiveGraph = _options.UseAdaptiveGraph;
        _usePredefinedGraph = _options.UsePredefinedGraph;

        RequireAdjacencyNodeCount(predefinedAdjacency, _numNodes, nameof(predefinedAdjacency));
        _predefinedAdjacency = predefinedAdjacency ?? ProximityAdjacency(_numNodes, rowNormalize: true);
        _predefinedAdjacencyTranspose = Engine.TensorTranspose(_predefinedAdjacency).Contiguous();
        _random = LayerInitializationSeedScope.CreateComponentRandom(_options.Seed);

        if (_useAdaptiveGraph)
            InitializeNodeEmbeddings();

        InitializeLayers();
    }

    #endregion

    #region Initialization

    /// <summary>
    /// Initializes the learnable node embeddings for adaptive graph learning.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the GraphWaveNet model, InitializeNodeEmbeddings builds and wires up model components. This sets up the GraphWaveNet architecture before use.
    /// </para>
    /// </remarks>
    private void InitializeNodeEmbeddings()
    {
        // (u - 0.5) * 2 * scale, drawn E1 then E2 for each entry.
        double scale = Math.Sqrt(2.0 / (_numNodes + _nodeEmbeddingDim));
        int entries = _numNodes * _nodeEmbeddingDim;
        var uniforms = new double[2 * entries];
        for (int i = 0; i < uniforms.Length; i++)
            uniforms[i] = _random.NextDouble();

        // Draws alternate E1, E2 per entry: column 0 of the [entries, 2] view is E1, column 1 is E2.
        var shape = new[] { _numNodes, _nodeEmbeddingDim };
        var pairs = Engine.TensorMultiplyScalar(
            Engine.TensorAddScalar(
                new Tensor<double>(new[] { entries, 2 }, new Vector<double>(uniforms)).Cast<T>(), NumOps.FromDouble(-0.5)),
            NumOps.FromDouble(2 * scale));
        _nodeEmbedding1 = Engine.Reshape(Engine.TensorNarrow(pairs, 1, 0, 1).Contiguous(), shape);
        _nodeEmbedding2 = Engine.Reshape(Engine.TensorNarrow(pairs, 1, 1, 1).Contiguous(), shape);
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
    /// softmax(ReLU(E1 E2^T)) over each row (Wu et al. 2019). Built from engine ops only, so under the
    /// gradient tape the node embeddings receive gradients.
    /// </summary>
    private Tensor<T> ComputeAdaptiveAdjacency(Tensor<T> embedding1, Tensor<T> embedding2)
        => Engine.TensorSoftmax(Engine.TensorReLU(Engine.TensorMatMul(embedding1, Engine.TensorTranspose(embedding2))), 1);

    /// <summary>
    /// Initializes all layers for the GraphWaveNet model.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the GraphWaveNet model, InitializeLayers builds and wires up model components. This sets up the GraphWaveNet architecture before use.
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
            Layers.AddRange(LayerHelper<T>.CreateDefaultGraphWaveNetLayers(
                _sequenceLength,
                _forecastHorizon,
                _numNodes,
                _numFeatures,
                _residualChannels,
                _skipChannels,
                _endChannels,
                _numBlocks,
                _layersPerBlock));

            ExtractLayerReferences();
        }
    }

    /// <summary>
    /// Extracts references to key layers for efficient access.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the GraphWaveNet model, ExtractLayerReferences performs a supporting step in the workflow. It keeps the GraphWaveNet architecture pipeline consistent.
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
            // The per-block filter/gate/diffusion/skip layers run through Layers; no separate list reads them.
            _endLayer1 = allDense[allDense.Count - 2];
            _outputLayer = allDense[allDense.Count - 1];
        }
    }

    /// <summary>
    /// Validates custom layers.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the GraphWaveNet model, ValidateCustomLayers checks inputs and configuration. This protects the GraphWaveNet architecture from mismatches and errors.
    /// </para>
    /// </remarks>
    protected override void ValidateCustomLayers(List<ILayer<T>> layers)
    {
        base.ValidateCustomLayers(layers);
        if (layers.Count < 4)
            throw new ArgumentException("GraphWaveNet requires at least 4 layers.");
    }

    #endregion

    #region NeuralNetworkBase Overrides

    /// <summary>
    /// Performs forward prediction.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the GraphWaveNet model, Predict produces predictions from input data. This is the main inference step of the GraphWaveNet architecture.
    /// </para>
    /// </remarks>
    protected override Tensor<T> PredictCore(Tensor<T> input)
    {
        return _useNativeMode ? ForecastNative(input) : ForecastOnnx(input);
    }

    /// <summary>
    /// Training-mode forward: calls <see cref="Forward"/> directly so
    /// dropout and the adaptive-graph conv stay in training mode under
    /// the tape. Default routes through <c>ForecastNative</c>, which
    /// forces inference mode.
    /// </summary>
    protected override Tensor<T> ForwardNativeForTraining(Tensor<T> input)
    {
        return Forward(input);
    }
    // UpdateParameters was an empty override, silently dropping every restore. The base
    // distributes the vector over the declared enumeration.
    /// <summary>
    /// Gets model metadata.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the GraphWaveNet model, GetModelMetadata performs a supporting step in the workflow. It keeps the GraphWaveNet architecture pipeline consistent.
    /// </para>
    /// </remarks>
    public override ModelMetadata<T> GetModelMetadata()
    {
        return new ModelMetadata<T>
        {
            AdditionalInfo = new Dictionary<string, object>
            {
                { "NetworkType", "GraphWaveNet" },
                { "SequenceLength", _sequenceLength },
                { "ForecastHorizon", _forecastHorizon },
                { "NumNodes", _numNodes },
                { "NumBlocks", _numBlocks },
                { "LayersPerBlock", _layersPerBlock },
                { "DiffusionSteps", _diffusionSteps },
                { "UseAdaptiveGraph", _useAdaptiveGraph },
                { "UsePredefinedGraph", _usePredefinedGraph },
                { "UseNativeMode", _useNativeMode }
            }
        };
    }

    /// <summary>
    /// Serializes model-specific data.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the GraphWaveNet model, SerializeNetworkSpecificData saves or restores model-specific settings. This lets the GraphWaveNet architecture be reused later.
    /// </para>
    /// </remarks>


    /// <summary>
    /// Deserializes model-specific data.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the GraphWaveNet model, DeserializeNetworkSpecificData saves or restores model-specific settings. This lets the GraphWaveNet architecture be reused later.
    /// </para>
    /// </remarks>


    #endregion

    #region IForecastingModel Implementation

    /// <summary>
    /// Generates forecasts.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the GraphWaveNet model, Forecast produces predictions from input data. This is the main inference step of the GraphWaveNet architecture.
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
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the GraphWaveNet model, public performs a supporting step in the workflow. It keeps the GraphWaveNet architecture pipeline consistent.
    /// </para>
    /// </remarks>
    public (Tensor<T> Forecast, Tensor<T> Lower, Tensor<T> Upper) ForecastWithIntervals(
        Tensor<T> input, double confidenceLevel = 0.95)
    {
        var samples = GenerateSamples(input, _numSamples);
        return ComputePredictionIntervals(samples, confidenceLevel);
    }

    /// <summary>
    /// Performs autoregressive forecasting.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the GraphWaveNet model, AutoregressiveForecast produces predictions from input data. This is the main inference step of the GraphWaveNet architecture.
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
    /// Evaluates forecast quality.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the GraphWaveNet model, Evaluate performs a supporting step in the workflow. It keeps the GraphWaveNet architecture pipeline consistent.
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

        metrics["MSE"] = mse;
        metrics["MAE"] = mae;
        metrics["RMSE"] = NumOps.Sqrt(mse);
        return metrics;
    }

    /// <summary>
    /// Applies instance normalization (identity for GraphWaveNet).
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the GraphWaveNet model, ApplyInstanceNormalization performs a supporting step in the workflow. It keeps the GraphWaveNet architecture pipeline consistent.
    /// </para>
    /// </remarks>
    public override Tensor<T> ApplyInstanceNormalization(Tensor<T> input) => input;

    /// <summary>
    /// Gets financial metrics.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the GraphWaveNet model, GetFinancialMetrics calculates evaluation metrics. This summarizes how the GraphWaveNet architecture is performing.
    /// </para>
    /// </remarks>
    public override Dictionary<string, T> GetFinancialMetrics()
    {
        return new Dictionary<string, T>
        {
            ["LastLoss"] = LastLoss is not null ? LastLoss : NumOps.Zero,
            ["NumNodes"] = NumOps.FromDouble(_numNodes),
            ["NumBlocks"] = NumOps.FromDouble(_numBlocks),
            ["DiffusionSteps"] = NumOps.FromDouble(_diffusionSteps)
        };
    }

    #endregion

    #region Forward/Backward Pass

    /// <summary>
    /// Performs the forward pass.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the GraphWaveNet model, Forward runs the forward pass through the layers. This moves data through the GraphWaveNet architecture to compute outputs.
    /// </para>
    /// </remarks>
    public Tensor<T> Forward(Tensor<T> input)
    {
        // Organize the input as [numNodes, featuresPerNode] (numNodes as the batch
        // dimension) so every DenseLayer applies SHARED per-node weights — the
        // paper-faithful design with O(channels^2) parameters instead of the old
        // numNodes-wide flattened denses that OOM-crashed the optimizer. The
        // tape-aware Engine.Reshape keeps gradients flowing back to the input.
        int totalSize = input.Length;
        int featuresPerNode = totalSize / _numNodes;
        var current = featuresPerNode * _numNodes == totalSize
            ? Engine.Reshape(input, new[] { _numNodes, featuresPerNode })
            : FlattenInput(input);

        // Per-node WaveNet-style temporal stack (shared weights across nodes).
        foreach (var layer in Layers)
        {
            current = layer.Forward(current);
        }

        // Diffusion convolution: adaptive + predefined graph mixing across nodes.
        if (_useNativeMode)
        {
            current = ApplyDiffusionConvolution(current);
        }

        return current;
    }

    /// <summary>
    /// Flattens input tensor.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the GraphWaveNet model, FlattenInput performs a supporting step in the workflow. It keeps the GraphWaveNet architecture pipeline consistent.
    /// </para>
    /// </remarks>
    private Tensor<T> FlattenInput(Tensor<T> input)
    {
        int totalSize = 1;
        foreach (var dim in input._shape)
            totalSize *= dim;

        var flattened = new Tensor<T>(new[] { totalSize });
        input.Data.Span.Slice(0, totalSize).CopyTo(flattened.Data.Span);

        return flattened;
    }

    /// <summary>
    /// Applies diffusion convolution with forward and backward random walks.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the GraphWaveNet model, ApplyDiffusionConvolution performs a supporting step in the workflow. It keeps the GraphWaveNet architecture pipeline consistent.
    /// </para>
    /// </remarks>
    private Tensor<T> ApplyDiffusionConvolution(Tensor<T> nodeFeatures)
    {
        int totalSize = nodeFeatures.Length;
        int featuresPerNode = totalSize / _numNodes;

        if (featuresPerNode * _numNodes != totalSize)
            return nodeFeatures;

        // Collect the graph supports (constant, non-trainable adjacency matrices).
        var supports = new List<Tensor<T>>();
        if (_usePredefinedGraph && _predefinedAdjacency is not null)
        {
            supports.Add(_predefinedAdjacency);
            if (_predefinedAdjacencyTranspose is not null)
                supports.Add(_predefinedAdjacencyTranspose);
        }
        if (_useAdaptiveGraph && _nodeEmbedding1 is not null && _nodeEmbedding2 is not null)
        {
            // Rebuilt from the embeddings on every call: under the training tape the embeddings get gradients,
            // and at inference it equals the cached LearnedAdjacency.
            supports.Add(ComputeAdaptiveAdjacency(_nodeEmbedding1, _nodeEmbedding2));
        }

        // Reshape features to [numNodes, featuresPerNode] via the tape-aware
        // Engine.Reshape so the diffusion output stays connected to the preceding
        // layer stack on backward (the old manual double-loop implementation
        // detached the graph and zeroed every gradient).
        var xMat = Engine.Reshape(nodeFeatures, new[] { _numNodes, featuresPerNode });

        // Diffusion convolution (Wu et al. 2019, "Graph WaveNet"):
        //   result = X + sum_supports sum_{k=1..K} (1/(k+1)) * P^k X
        // The supports P are constants, so P^k X stays differentiable w.r.t. X
        // (and thus the upstream layer weights) through Engine.TensorMatMul.
        var result = xMat;
        foreach (var support in supports)
        {
            var supportTensor = Engine.Reshape(support.Contiguous(), new[] { _numNodes, _numNodes });
            var currentPower = xMat;
            for (int k = 0; k < _diffusionSteps; k++)
            {
                currentPower = Engine.TensorMatMul(supportTensor, currentPower);  // P^(k+1) X
                double weight = 1.0 / (k + 2);
                result = Engine.TensorAdd(result,
                    Engine.TensorMultiplyScalar(currentPower, NumOps.FromDouble(weight)));
            }
        }

        return Engine.Reshape(result, nodeFeatures._shape);
    }
    #endregion

    #region Forecasting Methods

    /// <summary>
    /// Executes ForecastNative for the GraphWaveNet.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the GraphWaveNet model, ForecastNative produces predictions from input data. This is the main inference step of the GraphWaveNet architecture.
    /// </para>
    /// </remarks>
    private Tensor<T> ForecastNative(Tensor<T> context)
    {
        SetTrainingMode(false);
        return Forward(context);
    }

    /// <summary>
    /// Executes ForecastOnnx for the GraphWaveNet.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the GraphWaveNet model, ForecastOnnx produces predictions from input data. This is the main inference step of the GraphWaveNet architecture.
    /// </para>
    /// </remarks>
    protected override Tensor<T> ForecastOnnx(Tensor<T> input)
    {
        if (OnnxSession is null)
            throw new InvalidOperationException("ONNX session not initialized.");

        var flatInput = FlattenInput(input);
        var inputData = new float[flatInput.Data.Length];
        NumOps.ToFloatSpan(flatInput.Data.Span.Slice(0, flatInput.Data.Length), inputData.AsSpan(0, flatInput.Data.Length));

        var inputTensor = new OnnxTensors.DenseTensor<float>(inputData,
            new[] { 1, _numNodes, _sequenceLength, _numFeatures });

        var inputs = new List<NamedOnnxValue> { NamedOnnxValue.CreateFromTensor("input", inputTensor) };

        using var results = OnnxSession.Run(inputs);
        var outputTensor = results.First().AsTensor<float>();

        var outputData = new T[outputTensor.Length];
        NumOps.FromFloatSpan(outputTensor.ToDenseTensor().Buffer.Span.Slice(0, (int)outputTensor.Length), outputData.AsSpan(0, (int)outputTensor.Length));

        return new Tensor<T>(new[] { _numNodes * _forecastHorizon }, new Vector<T>(outputData));
    }

    /// <summary>
    /// Executes GenerateSamples for the GraphWaveNet.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the GraphWaveNet model, GenerateSamples performs a supporting step in the workflow. It keeps the GraphWaveNet architecture pipeline consistent.
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
    /// Executes AddSmallPerturbation for the GraphWaveNet.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the GraphWaveNet model, AddSmallPerturbation performs a supporting step in the workflow. It keeps the GraphWaveNet architecture pipeline consistent.
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
    /// Executes ShiftInputWindow for the GraphWaveNet.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the GraphWaveNet model, ShiftInputWindow performs a supporting step in the workflow. It keeps the GraphWaveNet architecture pipeline consistent.
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
    /// Executes ConcatenatePredictions for the GraphWaveNet.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the GraphWaveNet model, ConcatenatePredictions produces predictions from input data. This is the main inference step of the GraphWaveNet architecture.
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
                result[offset + i] = predVec[i];
            offset += predVec.Length;
        }

        return new Tensor<T>(new[] { totalLen }, new Vector<T>(result));
    }

    /// <summary>
    /// Executes ComputeQuantiles for the GraphWaveNet.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the GraphWaveNet model, ComputeQuantiles performs a supporting step in the workflow. It keeps the GraphWaveNet architecture pipeline consistent.
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
    /// Executes private for the GraphWaveNet.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the GraphWaveNet model, private performs a supporting step in the workflow. It keeps the GraphWaveNet architecture pipeline consistent.
    /// </para>
    /// </remarks>
    private (Tensor<T> Forecast, Tensor<T> Lower, Tensor<T> Upper) ComputePredictionIntervals(
        List<Tensor<T>> samples, double confidenceLevel)
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
    /// Executes Dispose for the GraphWaveNet.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the GraphWaveNet model, Dispose performs a supporting step in the workflow. It keeps the GraphWaveNet architecture pipeline consistent.
    /// </para>
    /// </remarks>
    protected override void Dispose(bool disposing)
    {
        if (disposing)
            OnnxSession?.Dispose();
        base.Dispose(disposing);
    }

    #endregion
}



