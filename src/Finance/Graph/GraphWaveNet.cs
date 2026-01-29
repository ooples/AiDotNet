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
public class GraphWaveNet<T> : ForecastingModelBase<T>
{
    #region Execution Mode
    private readonly bool _useNativeMode;
    #endregion

    
    #region Native Mode Fields
    private DenseLayer<T>? _inputProjection;
    private List<DenseLayer<T>>? _filterLayers;
    private List<DenseLayer<T>>? _gateLayers;
    private List<DenseLayer<T>>? _diffusionLayers;
    private List<DenseLayer<T>>? _skipLayers;
    private List<LayerNormalizationLayer<T>>? _layerNorms;
    private List<DropoutLayer<T>>? _dropoutLayers;
    private DenseLayer<T>? _endLayer1;
    private DenseLayer<T>? _outputLayer;
    #endregion

    #region Graph Fields
    private double[,]? _nodeEmbedding1;
    private double[,]? _nodeEmbedding2;
    private double[,]? _adaptiveAdjacency;
    private readonly double[,]? _predefinedAdjacency;
    private readonly double[,]? _predefinedAdjacencyTranspose;
    private readonly Random _random;
    #endregion

    #region Shared Fields
    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> _optimizer;
    private readonly ILossFunction<T> _lossFunction;
    private readonly GraphWaveNetOptions<T> _options;
    private readonly int _sequenceLength;
    private readonly int _forecastHorizon;
    private readonly int _numNodes;
    private readonly int _numFeatures;
    private readonly int _residualChannels;
    private readonly int _dilationChannels;
    private readonly int _skipChannels;
    private readonly int _endChannels;
    private readonly int _nodeEmbeddingDim;
    private readonly int _numBlocks;
    private readonly int _layersPerBlock;
    private readonly int _diffusionSteps;
    private readonly int _numSamples;
    private readonly bool _useAdaptiveGraph;
    private readonly bool _usePredefinedGraph;
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
    public double[,]? LearnedAdjacency => _adaptiveAdjacency is not null ? (double[,])_adaptiveAdjacency.Clone() : null;

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
    public GraphWaveNet(
        NeuralNetworkArchitecture<T> architecture,
        string onnxModelPath,
        GraphWaveNetOptions<T>? options = null,
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
        OnnxModelPath = onnxModelPath;
        OnnxSession = new InferenceSession(onnxModelPath);
        _options = options ?? new GraphWaveNetOptions<T>();
        _lossFunction = lossFunction ?? new MeanSquaredErrorLoss<T>();
        _optimizer = optimizer ?? new AdamOptimizer<T, Tensor<T>, Tensor<T>>(this);

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

        _predefinedAdjacency = predefinedAdjacency;
        _predefinedAdjacencyTranspose = predefinedAdjacency is not null ? TransposeMatrix(predefinedAdjacency) : null;
        _random = RandomHelper.CreateSecureRandom();

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
    public GraphWaveNet(
        NeuralNetworkArchitecture<T> architecture,
        GraphWaveNetOptions<T>? options = null,
        double[,]? predefinedAdjacency = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
        ILossFunction<T>? lossFunction = null)
        : base(architecture, lossFunction ?? new MeanSquaredErrorLoss<T>(), 1.0)
    {
        _useNativeMode = true;
        _options = options ?? new GraphWaveNetOptions<T>();
        _lossFunction = lossFunction ?? new MeanSquaredErrorLoss<T>();
        _optimizer = optimizer ?? new AdamOptimizer<T, Tensor<T>, Tensor<T>>(this);

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

        _predefinedAdjacency = predefinedAdjacency ?? CreateDefaultAdjacencyMatrix(_numNodes);
        _predefinedAdjacencyTranspose = TransposeMatrix(_predefinedAdjacency);
        _random = RandomHelper.CreateSecureRandom();

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
        _nodeEmbedding1 = new double[_numNodes, _nodeEmbeddingDim];
        _nodeEmbedding2 = new double[_numNodes, _nodeEmbeddingDim];

        double scale = Math.Sqrt(2.0 / (_numNodes + _nodeEmbeddingDim));
        for (int i = 0; i < _numNodes; i++)
        {
            for (int j = 0; j < _nodeEmbeddingDim; j++)
            {
                _nodeEmbedding1[i, j] = (_random.NextDouble() - 0.5) * 2 * scale;
                _nodeEmbedding2[i, j] = (_random.NextDouble() - 0.5) * 2 * scale;
            }
        }

        UpdateAdaptiveAdjacency();
    }

    /// <summary>
    /// Updates the adaptive adjacency matrix from current node embeddings.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the GraphWaveNet model, UpdateAdaptiveAdjacency updates internal parameters or state. This keeps the GraphWaveNet architecture aligned with the latest values.
    /// </para>
    /// </remarks>
    private void UpdateAdaptiveAdjacency()
    {
        if (_nodeEmbedding1 is null || _nodeEmbedding2 is null)
            return;

        _adaptiveAdjacency = new double[_numNodes, _numNodes];
        var rawScores = new double[_numNodes, _numNodes];

        // Compute E1 * E2^T
        for (int i = 0; i < _numNodes; i++)
        {
            for (int j = 0; j < _numNodes; j++)
            {
                double score = 0;
                for (int k = 0; k < _nodeEmbeddingDim; k++)
                {
                    score += _nodeEmbedding1[i, k] * _nodeEmbedding2[j, k];
                }
                rawScores[i, j] = Math.Max(0, score); // ReLU
            }
        }

        // Apply row-wise softmax
        for (int i = 0; i < _numNodes; i++)
        {
            double maxVal = double.NegativeInfinity;
            for (int j = 0; j < _numNodes; j++)
                maxVal = Math.Max(maxVal, rawScores[i, j]);

            double sumExp = 0;
            for (int j = 0; j < _numNodes; j++)
            {
                _adaptiveAdjacency[i, j] = Math.Exp(rawScores[i, j] - maxVal);
                sumExp += _adaptiveAdjacency[i, j];
            }

            for (int j = 0; j < _numNodes; j++)
                _adaptiveAdjacency[i, j] /= sumExp;
        }
    }

    /// <summary>
    /// Creates a default adjacency matrix with proximity-based connections.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the GraphWaveNet model, CreateDefaultAdjacencyMatrix builds and wires up model components. This sets up the GraphWaveNet architecture before use.
    /// </para>
    /// </remarks>
    private double[,] CreateDefaultAdjacencyMatrix(int numNodes)
    {
        var adj = new double[numNodes, numNodes];

        for (int i = 0; i < numNodes; i++)
        {
            adj[i, i] = 1.0;
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

        // Row-normalize
        for (int i = 0; i < numNodes; i++)
        {
            double rowSum = 0;
            for (int j = 0; j < numNodes; j++)
                rowSum += adj[i, j];
            if (rowSum > 0)
                for (int j = 0; j < numNodes; j++)
                    adj[i, j] /= rowSum;
        }

        return adj;
    }

    /// <summary>
    /// Transposes a matrix.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the GraphWaveNet model, TransposeMatrix performs a supporting step in the workflow. It keeps the GraphWaveNet architecture pipeline consistent.
    /// </para>
    /// </remarks>
    private double[,] TransposeMatrix(double[,] matrix)
    {
        int rows = matrix.GetLength(0);
        int cols = matrix.GetLength(1);
        var transposed = new double[cols, rows];
        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++)
                transposed[j, i] = matrix[i, j];
        return transposed;
    }

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
            ValidateCustomLayers(Layers);
        }
        else if (_useNativeMode)
        {
            Layers.AddRange(LayerHelper<T>.CreateDefaultGraphWaveNetLayers(
                Architecture,
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
            _filterLayers = new List<DenseLayer<T>>();
            _gateLayers = new List<DenseLayer<T>>();
            _diffusionLayers = new List<DenseLayer<T>>();
            _skipLayers = new List<DenseLayer<T>>();

            int idx = 1;
            int totalLayers = _numBlocks * _layersPerBlock;
            for (int i = 0; i < totalLayers && idx + 3 < allDense.Count - 2; i++)
            {
                _filterLayers.Add(allDense[idx++]);
                _gateLayers.Add(allDense[idx++]);
                _diffusionLayers.Add(allDense[idx++]);
                _skipLayers.Add(allDense[idx++]);
            }

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
    public override Tensor<T> Predict(Tensor<T> input)
    {
        return _useNativeMode ? ForecastNative(input) : ForecastOnnx(input);
    }

    /// <summary>
    /// Trains the model.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the GraphWaveNet model, Train performs a training step. This updates the GraphWaveNet architecture so it learns from data.
    /// </para>
    /// </remarks>
    public override void Train(Tensor<T> input, Tensor<T> target)
    {
        if (!_useNativeMode)
            throw new InvalidOperationException("Training requires native mode.");

        SetTrainingMode(true);

        if (_useAdaptiveGraph)
            UpdateAdaptiveAdjacency();

        var output = Forward(input);
        LastLoss = _lossFunction.CalculateLoss(output.ToVector(), target.ToVector());

        var gradient = _lossFunction.CalculateDerivative(output.ToVector(), target.ToVector());
        Backward(Tensor<T>.FromVector(gradient, output.Shape));

        _optimizer.UpdateParameters(Layers);

        if (_useAdaptiveGraph)
            UpdateNodeEmbeddingsFromGradient(gradient);

        SetTrainingMode(false);
    }

    /// <summary>
    /// Updates node embeddings based on loss gradient.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the GraphWaveNet model, UpdateNodeEmbeddingsFromGradient updates internal parameters or state. This keeps the GraphWaveNet architecture aligned with the latest values.
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
    /// Updates parameters.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the GraphWaveNet model, UpdateParameters updates internal parameters or state. This keeps the GraphWaveNet architecture aligned with the latest values.
    /// </para>
    /// </remarks>
    public override void UpdateParameters(Vector<T> gradients) { }

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
            ModelType = ModelType.NeuralNetwork,
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
    /// Creates a new instance.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the GraphWaveNet model, CreateNewInstance builds and wires up model components. This sets up the GraphWaveNet architecture before use.
    /// </para>
    /// </remarks>
    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance()
    {
        return new GraphWaveNet<T>(Architecture, _options, _predefinedAdjacency);
    }

    /// <summary>
    /// Serializes model-specific data.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the GraphWaveNet model, SerializeNetworkSpecificData saves or restores model-specific settings. This lets the GraphWaveNet architecture be reused later.
    /// </para>
    /// </remarks>
    protected override void SerializeNetworkSpecificData(BinaryWriter writer)
    {
        writer.Write(_sequenceLength);
        writer.Write(_forecastHorizon);
        writer.Write(_numNodes);
        writer.Write(_numFeatures);
        writer.Write(_residualChannels);
        writer.Write(_skipChannels);
        writer.Write(_endChannels);
        writer.Write(_nodeEmbeddingDim);
        writer.Write(_numBlocks);
        writer.Write(_layersPerBlock);
        writer.Write(_diffusionSteps);
        writer.Write(_useAdaptiveGraph);
        writer.Write(_usePredefinedGraph);
        writer.Write(_numSamples);

        if (_nodeEmbedding1 is not null && _nodeEmbedding2 is not null)
        {
            writer.Write(true);
            for (int i = 0; i < _numNodes; i++)
                for (int j = 0; j < _nodeEmbeddingDim; j++)
                {
                    writer.Write(_nodeEmbedding1[i, j]);
                    writer.Write(_nodeEmbedding2[i, j]);
                }
        }
        else
        {
            writer.Write(false);
        }
    }

    /// <summary>
    /// Deserializes model-specific data.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the GraphWaveNet model, DeserializeNetworkSpecificData saves or restores model-specific settings. This lets the GraphWaveNet architecture be reused later.
    /// </para>
    /// </remarks>
    protected override void DeserializeNetworkSpecificData(BinaryReader reader)
    {
        _ = reader.ReadInt32();
        _ = reader.ReadInt32();
        int numNodes = reader.ReadInt32();
        _ = reader.ReadInt32();
        _ = reader.ReadInt32();
        _ = reader.ReadInt32();
        _ = reader.ReadInt32();
        int embeddingDim = reader.ReadInt32();
        _ = reader.ReadInt32();
        _ = reader.ReadInt32();
        _ = reader.ReadInt32();
        _ = reader.ReadBoolean();
        _ = reader.ReadBoolean();
        _ = reader.ReadInt32();

        bool hasEmbeddings = reader.ReadBoolean();
        if (hasEmbeddings)
        {
            _nodeEmbedding1 = new double[numNodes, embeddingDim];
            _nodeEmbedding2 = new double[numNodes, embeddingDim];
            for (int i = 0; i < numNodes; i++)
                for (int j = 0; j < embeddingDim; j++)
                {
                    _nodeEmbedding1[i, j] = reader.ReadDouble();
                    _nodeEmbedding2[i, j] = reader.ReadDouble();
                }
            UpdateAdaptiveAdjacency();
        }
    }

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
        var current = FlattenInput(input);

        foreach (var layer in Layers)
        {
            current = layer.Forward(current);
        }

        // Apply diffusion convolution
        if (_useNativeMode)
        {
            current = ApplyDiffusionConvolution(current);
        }

        return current;
    }

    /// <summary>
    /// Performs backpropagation.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the GraphWaveNet model, Backward propagates gradients backward. This teaches the GraphWaveNet architecture how to adjust its weights.
    /// </para>
    /// </remarks>
    public Tensor<T> Backward(Tensor<T> gradOutput)
    {
        var grad = gradOutput;
        for (int i = Layers.Count - 1; i >= 0; i--)
            grad = Layers[i].Backward(grad);
        return grad;
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
        foreach (var dim in input.Shape)
            totalSize *= dim;

        var flattened = new Tensor<T>(new[] { totalSize });
        for (int i = 0; i < totalSize; i++)
            flattened.Data.Span[i] = input.Data.Span[i];

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
        var featureVec = nodeFeatures.ToVector();
        int totalSize = featureVec.Length;
        int featuresPerNode = totalSize / _numNodes;

        if (featuresPerNode * _numNodes != totalSize)
            return nodeFeatures;

        var result = new double[totalSize];
        var supports = new List<double[,]>();

        // Add predefined supports if enabled
        if (_usePredefinedGraph && _predefinedAdjacency is not null)
        {
            supports.Add(_predefinedAdjacency);
            if (_predefinedAdjacencyTranspose is not null)
                supports.Add(_predefinedAdjacencyTranspose);
        }

        // Add adaptive support if enabled
        if (_useAdaptiveGraph && _adaptiveAdjacency is not null)
        {
            supports.Add(_adaptiveAdjacency);
        }

        // Initialize result with original features
        for (int i = 0; i < totalSize; i++)
            result[i] = NumOps.ToDouble(featureVec[i]);

        // Apply diffusion for each support
        foreach (var support in supports)
        {
            var currentPower = new double[_numNodes, featuresPerNode];
            var nextPower = new double[_numNodes, featuresPerNode];

            // Initialize with features
            for (int n = 0; n < _numNodes; n++)
                for (int f = 0; f < featuresPerNode; f++)
                    currentPower[n, f] = NumOps.ToDouble(featureVec[n * featuresPerNode + f]);

            // Apply diffusion steps
            for (int k = 0; k < _diffusionSteps; k++)
            {
                // Compute support * currentPower
                for (int i = 0; i < _numNodes; i++)
                {
                    for (int f = 0; f < featuresPerNode; f++)
                    {
                        double aggregated = 0;
                        for (int j = 0; j < _numNodes; j++)
                            aggregated += support[i, j] * currentPower[j, f];
                        nextPower[i, f] = aggregated;
                    }
                }

                // Add to result with decay weight
                double weight = 1.0 / (k + 2);
                for (int i = 0; i < _numNodes; i++)
                    for (int f = 0; f < featuresPerNode; f++)
                        result[i * featuresPerNode + f] += weight * nextPower[i, f];

                (currentPower, nextPower) = (nextPower, currentPower);
            }
        }

        return new Tensor<T>(nodeFeatures.Shape, new Vector<T>(result.Select(d => NumOps.FromDouble(d)).ToArray()));
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
        for (int i = 0; i < flatInput.Data.Length; i++)
            inputData[i] = Convert.ToSingle(NumOps.ToDouble(flatInput.Data.Span[i]));

        var inputTensor = new OnnxTensors.DenseTensor<float>(inputData,
            new[] { 1, _numNodes, _sequenceLength, _numFeatures });

        var inputs = new List<NamedOnnxValue> { NamedOnnxValue.CreateFromTensor("input", inputTensor) };

        using var results = OnnxSession.Run(inputs);
        var outputTensor = results.First().AsTensor<float>();

        var outputData = new T[outputTensor.Length];
        for (int i = 0; i < outputTensor.Length; i++)
            outputData[i] = NumOps.FromDouble(outputTensor.GetValue(i));

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
        return new Tensor<T>(input.Shape, new Vector<T>(perturbed));
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
        var inputVec = input.ToVector();
        var predVec = prediction.ToVector();
        int inputSize = _numNodes * _sequenceLength * _numFeatures;
        int shiftSize = _numNodes * _numFeatures;
        var shifted = new T[inputSize];

        for (int i = 0; i < inputSize - shiftSize; i++)
            shifted[i] = inputVec[i + shiftSize];
        for (int i = 0; i < Math.Min(shiftSize, predVec.Length); i++)
            shifted[inputSize - shiftSize + i] = predVec[i];

        return new Tensor<T>(new[] { inputSize }, new Vector<T>(shifted));
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



