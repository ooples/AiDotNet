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
/// DCRNN (Diffusion Convolutional Recurrent Neural Network) for spatial-temporal forecasting.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (typically double or float).</typeparam>
/// <remarks>
/// <para>
/// DCRNN combines diffusion convolution with sequence-to-sequence architecture for
/// traffic forecasting on road networks and other spatial-temporal prediction tasks.
/// </para>
/// <para><b>For Beginners:</b> DCRNN was a breakthrough model for traffic prediction that introduced
/// two key innovations:
///
/// <b>The Key Insight:</b>
/// Traffic flow can be modeled as a diffusion process - congestion spreads through a road network
/// similar to how heat diffuses through a material. DCRNN captures this with diffusion convolution
/// while using an encoder-decoder architecture for multi-step forecasting.
///
/// <b>What Makes DCRNN Special:</b>
/// 1. <b>Diffusion Convolution:</b> Models spatial dependencies as bidirectional random walks on the graph
/// 2. <b>DCGRU Cells:</b> GRU cells where matrix multiplications are replaced with diffusion convolution
/// 3. <b>Encoder-Decoder:</b> Seq2seq architecture for multi-step prediction
/// 4. <b>Scheduled Sampling:</b> Gradually transitions from teacher forcing to autoregressive during training
///
/// <b>Mathematical Foundation:</b>
/// Diffusion convolution: X_star = sum_k (theta_k * (D_O^(-1)*W)^k * X + theta'_k * (D_I^(-1)*W^T)^k * X)
/// where D_O, D_I are out/in-degree matrices and W is the adjacency matrix.
///
/// <b>Architecture:</b>
/// - Encoder: Stacked DCGRU layers process input sequence
/// - Final encoder state becomes initial decoder state
/// - Decoder: Stacked DCGRU layers generate output autoregressively
/// - Output: Linear projection to forecast dimension
///
/// <b>Key Benefits:</b>
/// - Captures bidirectional spatial dependencies (upstream and downstream effects)
/// - Multi-step prediction without error accumulation from teacher forcing
/// - Effective on large-scale traffic networks
/// </para>
/// <para>
/// <b>Reference:</b> Li et al., "Diffusion Convolutional Recurrent Neural Network: Data-Driven Traffic Forecasting", ICLR 2018.
/// https://arxiv.org/abs/1707.01926
/// </para>
/// </remarks>
public class DCRNN<T> : ForecastingModelBase<T>
{
    #region Execution Mode
    private readonly bool _useNativeMode;
    #endregion

    
    #region Native Mode Fields
    private DenseLayer<T>? _inputProjection;
    private List<GRULayer<T>> _encoderGRUs;
    private List<GRULayer<T>> _decoderGRUs;
    private DenseLayer<T>? _outputLayer;
    #endregion

    #region Diffusion Fields
    /// <summary>
    /// Executes D_O^ for the DCRNN.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the DCRNN model, D_O^ performs a supporting step in the workflow. It keeps the DCRNN architecture pipeline consistent.
    /// </para>
    /// </remarks>
    private double[,]? _forwardDiffusion;    // D_O^(-1) * W
    /// <summary>
    /// Executes D_I^ for the DCRNN.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the DCRNN model, D_I^ performs a supporting step in the workflow. It keeps the DCRNN architecture pipeline consistent.
    /// </para>
    /// </remarks>
    private double[,]? _backwardDiffusion;   // D_I^(-1) * W^T
    private List<double[,]> _forwardPowers;  // Powers of forward diffusion
    private List<double[,]> _backwardPowers; // Powers of backward diffusion
    private readonly double[,]? _adjacencyMatrix;
    private readonly Random _random;
    #endregion

    #region Shared Fields
    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> _optimizer;
    private readonly ILossFunction<T> _lossFunction;
    private readonly DCRNNOptions<T> _options;
    private int _sequenceLength;
    private int _forecastHorizon;
    private int _numNodes;
    private int _numFeatures;
    private int _hiddenDimension;
    private int _numEncoderLayers;
    private int _numDecoderLayers;
    private int _diffusionSteps;
    private int _numSamples;
    private int _trainingStep;
    #endregion

    #region IForecastingModel Properties

    /// <inheritdoc/>
    public override int SequenceLength => _sequenceLength;

    /// <inheritdoc/>
    public override int PredictionHorizon => _forecastHorizon;

    /// <inheritdoc/>
    /// <remarks>
    /// <para><b>For Beginners:</b> For DCRNN (a graph model), the expected input feature
    /// dimension is numNodes * numFeatures since we process all nodes together.
    /// </para>
    /// </remarks>
    public override int NumFeatures => _numNodes * _numFeatures;

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
    /// <para><b>For Beginners:</b> How many sensors/locations in the network.
    /// </para>
    /// </remarks>
    public int NumNodes => _numNodes;

    /// <summary>
    /// Gets the forecast horizon.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> How many future time steps to predict.
    /// </para>
    /// </remarks>
    public int ForecastHorizon => _forecastHorizon;

    /// <summary>
    /// Gets whether the model supports training.
    /// </summary>
    public override bool SupportsTraining => _useNativeMode;

    /// <summary>
    /// Gets the number of diffusion steps.
    /// </summary>
    public int DiffusionSteps => _diffusionSteps;

    /// <summary>
    /// Gets the number of samples for uncertainty estimation.
    /// </summary>
    public int NumSamples => _numSamples;

    #endregion

    #region Constructors

    /// <summary>
    /// Initializes a new instance of DCRNN in ONNX inference mode.
    /// </summary>
    /// <param name="architecture">The neural network architecture configuration.</param>
    /// <param name="onnxModelPath">Path to the pre-trained ONNX model file.</param>
    /// <param name="options">DCRNN configuration options.</param>
    /// <param name="adjacencyMatrix">Optional predefined adjacency matrix.</param>
    /// <param name="optimizer">Optional gradient-based optimizer.</param>
    /// <param name="lossFunction">Optional loss function for training.</param>
    /// <remarks>
    /// <para><b>For Beginners:</b> Use this constructor when you have a pre-trained DCRNN model
    /// saved in ONNX format. ONNX models can be trained in PyTorch or other frameworks
    /// and loaded here for inference.
    /// </para>
    /// </remarks>
    public DCRNN(
        NeuralNetworkArchitecture<T> architecture,
        string onnxModelPath,
        DCRNNOptions<T>? options = null,
        double[,]? adjacencyMatrix = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
        ILossFunction<T>? lossFunction = null)
        : base(architecture, lossFunction ?? new MeanSquaredErrorLoss<T>(), 1.0)
    {
        if (string.IsNullOrWhiteSpace(onnxModelPath))
            throw new ArgumentNullException(nameof(onnxModelPath));
        if (!File.Exists(onnxModelPath))
            throw new FileNotFoundException($"ONNX model not found: {onnxModelPath}");

        _useNativeMode = false;
        OnnxModelPath = onnxModelPath;
        OnnxSession = new InferenceSession(onnxModelPath);
        _options = options ?? new DCRNNOptions<T>();
        Options = _options;
        _lossFunction = lossFunction ?? new MeanSquaredErrorLoss<T>();
        _optimizer = optimizer ?? new AdamOptimizer<T, Tensor<T>, Tensor<T>>(this);

        _sequenceLength = _options.SequenceLength;
        _forecastHorizon = _options.ForecastHorizon;
        _numNodes = _options.NumNodes;
        _numFeatures = _options.NumFeatures;
        _hiddenDimension = _options.HiddenDimension;
        _numEncoderLayers = _options.NumEncoderLayers;
        _numDecoderLayers = _options.NumDecoderLayers;
        _diffusionSteps = _options.DiffusionSteps;
        _numSamples = _options.NumSamples;
        _trainingStep = 0;

        _adjacencyMatrix = adjacencyMatrix;
        _random = RandomHelper.CreateSecureRandom();
        _encoderGRUs = new List<GRULayer<T>>();
        _decoderGRUs = new List<GRULayer<T>>();
        _forwardPowers = new List<double[,]>();
        _backwardPowers = new List<double[,]>();

        if (adjacencyMatrix is not null)
        {
            InitializeDiffusionMatrices(adjacencyMatrix);
        }
    }

    /// <summary>
    /// Initializes a new instance of DCRNN in native training/inference mode.
    /// </summary>
    /// <param name="architecture">The neural network architecture configuration.</param>
    /// <param name="options">DCRNN configuration options.</param>
    /// <param name="adjacencyMatrix">Optional predefined adjacency matrix for the graph.</param>
    /// <param name="optimizer">Optional gradient-based optimizer.</param>
    /// <param name="lossFunction">Optional loss function for training.</param>
    /// <remarks>
    /// <para><b>For Beginners:</b> Use this constructor for training DCRNN from scratch or
    /// when you need full control over the model. You can provide the adjacency matrix
    /// representing the road network.
    /// </para>
    /// </remarks>
    public DCRNN(
        NeuralNetworkArchitecture<T> architecture,
        DCRNNOptions<T>? options = null,
        double[,]? adjacencyMatrix = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
        ILossFunction<T>? lossFunction = null)
        : base(architecture, lossFunction ?? new MeanSquaredErrorLoss<T>(), 1.0)
    {
        _useNativeMode = true;
        _options = options ?? new DCRNNOptions<T>();
        Options = _options;
        _lossFunction = lossFunction ?? new MeanSquaredErrorLoss<T>();
        _optimizer = optimizer ?? new AdamOptimizer<T, Tensor<T>, Tensor<T>>(this);

        _sequenceLength = _options.SequenceLength;
        _forecastHorizon = _options.ForecastHorizon;
        _numNodes = _options.NumNodes;
        _numFeatures = _options.NumFeatures;
        _hiddenDimension = _options.HiddenDimension;
        _numEncoderLayers = _options.NumEncoderLayers;
        _numDecoderLayers = _options.NumDecoderLayers;
        _diffusionSteps = _options.DiffusionSteps;
        _numSamples = _options.NumSamples;
        _trainingStep = 0;

        _adjacencyMatrix = adjacencyMatrix;
        _random = RandomHelper.CreateSecureRandom();
        _encoderGRUs = new List<GRULayer<T>>();
        _decoderGRUs = new List<GRULayer<T>>();
        _forwardPowers = new List<double[,]>();
        _backwardPowers = new List<double[,]>();

        InitializeLayers();

        if (adjacencyMatrix is not null)
        {
            InitializeDiffusionMatrices(adjacencyMatrix);
        }
        else
        {
            InitializeDefaultDiffusionMatrices();
        }
    }

    #endregion

    #region Initialization

    /// <summary>
    /// Initializes the neural network layers for DCRNN.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method sets up all the components of DCRNN:
    /// - Input projection to convert features to hidden dimension
    /// - Encoder GRU layers for processing input sequence
    /// - Decoder GRU layers for generating predictions
    /// - Output projection to generate final forecasts
    /// </para>
    /// </remarks>
    protected override void InitializeLayers()
    {
        if (Architecture.Layers is not null && Architecture.Layers.Count > 0)
        {
            Layers.AddRange(Architecture.Layers);
            ValidateCustomLayers(Layers);
            ExtractLayerReferences();
        }
        else if (_useNativeMode)
        {
            Layers.AddRange(LayerHelper<T>.CreateDefaultDCRNNLayers(
                Architecture,
                numNodes: _numNodes,
                numFeatures: _numFeatures,
                hiddenDimension: _hiddenDimension,
                numEncoderLayers: _numEncoderLayers,
                numDecoderLayers: _numDecoderLayers,
                forecastHorizon: _forecastHorizon,
                diffusionSteps: _diffusionSteps));

            ExtractLayerReferences();
        }
    }

    /// <summary>
    /// Extracts references to specific layers for direct access during forward pass.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> DCRNN has encoder and decoder sections.
    /// This method identifies each layer type so we can apply them correctly.
    /// </para>
    /// </remarks>
    private void ExtractLayerReferences()
    {
        _encoderGRUs = new List<GRULayer<T>>();
        _decoderGRUs = new List<GRULayer<T>>();

        int gruCount = 0;

        foreach (var layer in Layers)
        {
            if (layer is DenseLayer<T> dense)
            {
                if (_inputProjection is null)
                    _inputProjection = dense;
                else
                    _outputLayer = dense;
            }
            else if (layer is GRULayer<T> gru)
            {
                gruCount++;
                if (gruCount <= _numEncoderLayers)
                    _encoderGRUs.Add(gru);
                else
                    _decoderGRUs.Add(gru);
            }
        }
    }

    /// <summary>
    /// Initializes diffusion matrices from the adjacency matrix.
    /// </summary>
    /// <param name="adjacencyMatrix">The graph adjacency matrix.</param>
    /// <remarks>
    /// <para><b>For Beginners:</b> DCRNN uses random walk diffusion to model how traffic
    /// propagates through the network. This method computes two diffusion matrices:
    /// - Forward: D_O^(-1) * W - models traffic flowing with edge direction
    /// - Backward: D_I^(-1) * W^T - models traffic flowing against edge direction
    /// </para>
    /// </remarks>
    private void InitializeDiffusionMatrices(double[,] adjacencyMatrix)
    {
        int n = adjacencyMatrix.GetLength(0);

        // Compute out-degree and in-degree
        double[] outDegree = new double[n];
        double[] inDegree = new double[n];

        for (int i = 0; i < n; i++)
        {
            double outSum = 0;
            double inSum = 0;
            for (int j = 0; j < n; j++)
            {
                outSum += adjacencyMatrix[i, j];
                inSum += adjacencyMatrix[j, i];
            }
            outDegree[i] = outSum;
            inDegree[i] = inSum;
        }

        // Create forward diffusion: D_O^(-1) * W
        _forwardDiffusion = new double[n, n];
        for (int i = 0; i < n; i++)
        {
            double dInv = outDegree[i] > 0 ? 1.0 / outDegree[i] : 0;
            for (int j = 0; j < n; j++)
            {
                _forwardDiffusion[i, j] = dInv * adjacencyMatrix[i, j];
            }
        }

        // Create backward diffusion: D_I^(-1) * W^T
        _backwardDiffusion = new double[n, n];
        for (int i = 0; i < n; i++)
        {
            double dInv = inDegree[i] > 0 ? 1.0 / inDegree[i] : 0;
            for (int j = 0; j < n; j++)
            {
                _backwardDiffusion[i, j] = dInv * adjacencyMatrix[j, i];
            }
        }

        ComputeDiffusionPowers();
    }

    /// <summary>
    /// Initializes default diffusion matrices when no adjacency is provided.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> When no road network is provided, we create identity
    /// diffusion matrices. This means each node only considers itself.
    /// </para>
    /// </remarks>
    private void InitializeDefaultDiffusionMatrices()
    {
        int n = _numNodes;

        _forwardDiffusion = new double[n, n];
        _backwardDiffusion = new double[n, n];

        for (int i = 0; i < n; i++)
        {
            _forwardDiffusion[i, i] = 1.0;
            _backwardDiffusion[i, i] = 1.0;
        }

        ComputeDiffusionPowers();
    }

    /// <summary>
    /// Precomputes powers of the diffusion matrices.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Diffusion convolution uses P^0, P^1, ..., P^K.
    /// Computing these once is more efficient than during each forward pass.
    /// </para>
    /// </remarks>
    private void ComputeDiffusionPowers()
    {
        if (_forwardDiffusion is null || _backwardDiffusion is null)
            return;

        int n = _forwardDiffusion.GetLength(0);

        _forwardPowers = new List<double[,]>();
        _backwardPowers = new List<double[,]>();

        // P^0 = Identity
        var identityF = new double[n, n];
        var identityB = new double[n, n];
        for (int i = 0; i < n; i++)
        {
            identityF[i, i] = 1.0;
            identityB[i, i] = 1.0;
        }
        _forwardPowers.Add(identityF);
        _backwardPowers.Add(identityB);

        // P^1
        _forwardPowers.Add((double[,])_forwardDiffusion.Clone());
        _backwardPowers.Add((double[,])_backwardDiffusion.Clone());

        // Compute P^2, P^3, ..., P^K
        var currentF = (double[,])_forwardDiffusion.Clone();
        var currentB = (double[,])_backwardDiffusion.Clone();

        for (int k = 2; k <= _options.MaxDiffusionStep; k++)
        {
            currentF = MatrixMultiply(currentF, _forwardDiffusion);
            currentB = MatrixMultiply(currentB, _backwardDiffusion);
            _forwardPowers.Add((double[,])currentF.Clone());
            _backwardPowers.Add((double[,])currentB.Clone());
        }
    }

    /// <summary>
    /// Validates custom layers provided by the user.
    /// </summary>
    /// <param name="layers">The list of layers to validate.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the DCRNN model, ValidateCustomLayers checks inputs and configuration. This protects the DCRNN architecture from mismatches and errors.
    /// </para>
    /// </remarks>
    protected override void ValidateCustomLayers(List<ILayer<T>> layers)
    {
        base.ValidateCustomLayers(layers);

        int gruCount = layers.Count(l => l is GRULayer<T>);
        int expectedGrus = _numEncoderLayers + _numDecoderLayers;

        if (gruCount < expectedGrus)
        {
            throw new ArgumentException(
                $"DCRNN requires at least {expectedGrus} GRU layers " +
                $"({_numEncoderLayers} encoder + {_numDecoderLayers} decoder), " +
                $"but only {gruCount} were provided.");
        }
    }

    #endregion

    #region NeuralNetworkBase Overrides

    /// <summary>
    /// Makes a prediction using the DCRNN model.
    /// </summary>
    /// <param name="input">Input tensor.</param>
    /// <returns>Predictions tensor.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the DCRNN model, Predict produces predictions from input data. This is the main inference step of the DCRNN architecture.
    /// </para>
    /// </remarks>
    public override Tensor<T> Predict(Tensor<T> input)
    {
        return _useNativeMode ? ForecastNative(input) : ForecastOnnx(input);
    }

    /// <summary>
    /// Trains the DCRNN model on provided data.
    /// </summary>
    /// <param name="input">Training input tensor.</param>
    /// <param name="target">Target output tensor.</param>
    /// <remarks>
    /// <para><b>For Beginners:</b> Training adjusts the model's parameters.
    /// DCRNN uses scheduled sampling during training.
    /// </para>
    /// </remarks>
    public override void Train(Tensor<T> input, Tensor<T> target)
    {
        if (!_useNativeMode)
            throw new InvalidOperationException("Cannot train in ONNX mode");

        SetTrainingMode(true);

        var prediction = Forward(input);
        LastLoss = _lossFunction.CalculateLoss(prediction.ToVector(), target.ToVector());

        var gradient = _lossFunction.CalculateDerivative(prediction.ToVector(), target.ToVector());
        Backward(Tensor<T>.FromVector(gradient, prediction.Shape));

        _optimizer.UpdateParameters(Layers);

        _trainingStep++;
        SetTrainingMode(false);
    }

    /// <summary>
    /// Updates parameters using the provided gradients.
    /// </summary>
    /// <param name="gradients">Gradient vector.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the DCRNN model, UpdateParameters updates internal parameters or state. This keeps the DCRNN architecture aligned with the latest values.
    /// </para>
    /// </remarks>
    public override void UpdateParameters(Vector<T> gradients)
    {
        // Parameters are updated through the optimizer in Train()
    }

    /// <summary>
    /// Gets metadata about the DCRNN model.
    /// </summary>
    /// <returns>ModelMetadata containing model information.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the DCRNN model, GetModelMetadata performs a supporting step in the workflow. It keeps the DCRNN architecture pipeline consistent.
    /// </para>
    /// </remarks>
    public override ModelMetadata<T> GetModelMetadata()
    {
        return new ModelMetadata<T>
        {
            ModelType = ModelType.NeuralNetwork,
            AdditionalInfo = new Dictionary<string, object>
            {
                { "NetworkType", "DCRNN" },
                { "SequenceLength", _sequenceLength },
                { "ForecastHorizon", _forecastHorizon },
                { "NumNodes", _numNodes },
                { "NumFeatures", _numFeatures },
                { "HiddenDimension", _hiddenDimension },
                { "NumEncoderLayers", _numEncoderLayers },
                { "NumDecoderLayers", _numDecoderLayers },
                { "DiffusionSteps", _diffusionSteps },
                { "FilterType", _options.FilterType },
                { "NumSamples", _numSamples },
                { "UseNativeMode", _useNativeMode }
            }
        };
    }

    /// <summary>
    /// Creates a new instance with the same configuration.
    /// </summary>
    /// <returns>A new DCRNN instance.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the DCRNN model, CreateNewInstance builds and wires up model components. This sets up the DCRNN architecture before use.
    /// </para>
    /// </remarks>
    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance()
    {
        return new DCRNN<T>(Architecture, _options, _adjacencyMatrix);
    }

    /// <summary>
    /// Serializes DCRNN-specific data.
    /// </summary>
    /// <param name="writer">The binary writer.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the DCRNN model, SerializeNetworkSpecificData saves or restores model-specific settings. This lets the DCRNN architecture be reused later.
    /// </para>
    /// </remarks>
    protected override void SerializeNetworkSpecificData(BinaryWriter writer)
    {
        writer.Write(_sequenceLength);
        writer.Write(_forecastHorizon);
        writer.Write(_numNodes);
        writer.Write(_numFeatures);
        writer.Write(_hiddenDimension);
        writer.Write(_numEncoderLayers);
        writer.Write(_numDecoderLayers);
        writer.Write(_diffusionSteps);
        writer.Write(_trainingStep);

        // Serialize diffusion matrices
        if (_forwardDiffusion is not null && _backwardDiffusion is not null)
        {
            writer.Write(true);
            int n = _forwardDiffusion.GetLength(0);
            writer.Write(n);
            for (int i = 0; i < n; i++)
            {
                for (int j = 0; j < n; j++)
                {
                    writer.Write(_forwardDiffusion[i, j]);
                    writer.Write(_backwardDiffusion[i, j]);
                }
            }
        }
        else
        {
            writer.Write(false);
        }
    }

    /// <summary>
    /// Deserializes DCRNN-specific data.
    /// </summary>
    /// <param name="reader">The binary reader.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the DCRNN model, DeserializeNetworkSpecificData saves or restores model-specific settings. This lets the DCRNN architecture be reused later.
    /// </para>
    /// </remarks>
    protected override void DeserializeNetworkSpecificData(BinaryReader reader)
    {
        _sequenceLength = reader.ReadInt32();
        _forecastHorizon = reader.ReadInt32();
        _numNodes = reader.ReadInt32();
        int numNodes = _numNodes;
        _numFeatures = reader.ReadInt32();
        _hiddenDimension = reader.ReadInt32();
        _numEncoderLayers = reader.ReadInt32();
        _numDecoderLayers = reader.ReadInt32();
        _diffusionSteps = reader.ReadInt32();
        _trainingStep = reader.ReadInt32();

        bool hasDiffusion = reader.ReadBoolean();
        if (hasDiffusion)
        {
            int n = reader.ReadInt32();
            _forwardDiffusion = new double[n, n];
            _backwardDiffusion = new double[n, n];
            for (int i = 0; i < n; i++)
            {
                for (int j = 0; j < n; j++)
                {
                    _forwardDiffusion[i, j] = reader.ReadDouble();
                    _backwardDiffusion[i, j] = reader.ReadDouble();
                }
            }
            ComputeDiffusionPowers();
        }
    }

    #endregion

    #region IForecastingModel Implementation

    /// <summary>
    /// Generates forecasts for all nodes.
    /// </summary>
    /// <param name="historicalData">Input tensor with historical data.</param>
    /// <param name="quantiles">Optional quantile levels for probabilistic forecasting.</param>
    /// <returns>Forecast tensor.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the DCRNN model, Forecast produces predictions from input data. This is the main inference step of the DCRNN architecture.
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
    /// <b>For Beginners:</b> In the DCRNN model, public performs a supporting step in the workflow. It keeps the DCRNN architecture pipeline consistent.
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
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the DCRNN model, AutoregressiveForecast produces predictions from input data. This is the main inference step of the DCRNN architecture.
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
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the DCRNN model, Evaluate performs a supporting step in the workflow. It keeps the DCRNN architecture pipeline consistent.
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
    /// Applies instance normalization (identity for DCRNN).
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the DCRNN model, ApplyInstanceNormalization performs a supporting step in the workflow. It keeps the DCRNN architecture pipeline consistent.
    /// </para>
    /// </remarks>
    public override Tensor<T> ApplyInstanceNormalization(Tensor<T> input)
    {
        return input;
    }

    /// <summary>
    /// Gets financial-specific metrics.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the DCRNN model, GetFinancialMetrics calculates evaluation metrics. This summarizes how the DCRNN architecture is performing.
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
            ["DiffusionSteps"] = NumOps.FromDouble(_diffusionSteps),
            ["TrainingStep"] = NumOps.FromDouble(_trainingStep)
        };
    }

    #endregion

    #region Forward/Backward Pass

    /// <summary>
    /// Performs forward pass through all layers.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the DCRNN model, Forward runs the forward pass through the layers. This moves data through the DCRNN architecture to compute outputs.
    /// </para>
    /// </remarks>
    public Tensor<T> Forward(Tensor<T> input)
    {
        // Store original dimensions for reshape calculations
        int origBatch = input.Rank >= 1 ? input.Shape[0] : 1;
        int origSeq = input.Rank >= 2 ? input.Shape[1] : 1;

        var current = FlattenInput(input);

        // Apply all layers with proper reshape between Dense and GRU
        foreach (var layer in Layers)
        {
            // Before GRU layer, reshape from 2D to 3D
            // Dense outputs [batch*seq, numNodes*hiddenDim] but GRU expects [batch, seq, hiddenDim]
            if (layer is GRULayer<T> && current.Rank == 2)
            {
                int totalSamples = current.Shape[0];
                int totalFeatures = current.Shape[1];

                // GRU expects inputSize = hiddenDimension
                // Dense output has numNodes * hiddenDim features
                int gruInputSize = _hiddenDimension;
                if (totalFeatures % gruInputSize == 0)
                {
                    int nodesPerSample = totalFeatures / gruInputSize;
                    int totalSeqLen = totalSamples * nodesPerSample;
                    // Use batch=1 and expand sequence dimension to include nodes
                    current = current.Reshape(new[] { 1, totalSeqLen, gruInputSize });
                }
                else
                {
                    // Features don't divide evenly by hiddenDim - use as-is but add batch dim
                    current = current.Reshape(new[] { 1, totalSamples, totalFeatures });
                }
            }

            // Before Dense layer (not the final output projection), reshape from 3D to 2D
            // GRU outputs [batch, seq, hiddenDim] but Dense expects [batch*seq, numNodes*hiddenDim]
            if (layer is DenseLayer<T> && current.Rank == 3)
            {
                int batch = current.Shape[0];
                int seqLen = current.Shape[1];
                int hidden = current.Shape[2];

                // Flatten to 2D: [batch * seq / numNodes, numNodes * hidden]
                // This preserves total elements while matching Dense input expectations
                int totalElements = batch * seqLen * hidden;
                int denseFeatures = _numNodes * _hiddenDimension;

                if (totalElements % denseFeatures == 0)
                {
                    int denseBatch = totalElements / denseFeatures;
                    current = current.Reshape(new[] { denseBatch, denseFeatures });
                }
                else
                {
                    // Fallback: flatten to [total_samples, hidden]
                    current = current.Reshape(new[] { batch * seqLen, hidden });
                }
            }

            current = layer.Forward(current);
        }

        // Apply diffusion convolution
        if (_forwardDiffusion is not null && _useNativeMode)
        {
            current = ApplyDiffusionConvolution(current);
        }

        return current;
    }

    /// <summary>
    /// Performs backpropagation through all layers.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the DCRNN model, Backward propagates gradients backward. This teaches the DCRNN architecture how to adjust its weights.
    /// </para>
    /// </remarks>
    public Tensor<T> Backward(Tensor<T> gradOutput)
    {
        var grad = gradOutput;

        for (int i = Layers.Count - 1; i >= 0; i--)
        {
            var layer = Layers[i];

            // Before GRU backward, if gradient is 2D but GRU expects 3D, reshape
            // This reverses what we did in forward where we flattened 3D to 2D after GRU
            if (layer is GRULayer<T> && grad.Rank == 2)
            {
                int totalElements = grad.Length;
                int hidden = _hiddenDimension;
                if (totalElements % hidden == 0)
                {
                    int seqLen = totalElements / hidden;
                    grad = grad.Reshape(new[] { 1, seqLen, hidden });
                }
            }

            grad = layer.Backward(grad);

            // After GRU backward, if gradient is 3D but previous layer was Dense, flatten to 2D
            // This reverses what we did in forward where we reshaped 2D to 3D before GRU
            if (layer is GRULayer<T> && grad.Rank == 3 && i > 0 && Layers[i - 1] is DenseLayer<T>)
            {
                int batch = grad.Shape[0];
                int seqLen = grad.Shape[1];
                int hidden = grad.Shape[2];
                int totalElements = batch * seqLen * hidden;
                int denseFeatures = _numNodes * _hiddenDimension;

                if (totalElements % denseFeatures == 0)
                {
                    int denseBatch = totalElements / denseFeatures;
                    grad = grad.Reshape(new[] { denseBatch, denseFeatures });
                }
                else
                {
                    // Fallback: flatten to 2D
                    grad = grad.Reshape(new[] { batch * seqLen, hidden });
                }
            }
        }

        return grad;
    }

    /// <summary>
    /// Flattens input tensor for processing.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the DCRNN model, FlattenInput performs a supporting step in the workflow. It keeps the DCRNN architecture pipeline consistent.
    /// </para>
    /// </remarks>
    private Tensor<T> FlattenInput(Tensor<T> input)
    {
        // For DCRNN, we need to preserve temporal structure for GRU layers
        // Expected input: [batch, sequence, features] or [batch, sequence, nodes, features]
        // First layer (DenseLayer) expects: [batch * sequence, numNodes * numFeatures]
        // GRU layers expect: [batch, sequence, hiddenDim]

        if (input.Rank == 2)
        {
            // Already [sequence, features] - add batch dimension
            return input.Reshape(new[] { 1, input.Shape[0], input.Shape[1] });
        }
        else if (input.Rank == 3)
        {
            // [batch, sequence, features] - flatten for first dense layer
            int batchSize = input.Shape[0];
            int seqLength = input.Shape[1];
            int features = input.Shape[2];

            // Reshape to [batch * seqLength, features] for dense layer
            return input.Reshape(new[] { batchSize * seqLength, features });
        }
        else if (input.Rank == 4)
        {
            // [batch, sequence, nodes, features] - flatten nodes and features
            int batchSize = input.Shape[0];
            int seqLength = input.Shape[1];
            int nodes = input.Shape[2];
            int features = input.Shape[3];

            // Reshape to [batch * seqLength, nodes * features] for dense layer
            return input.Reshape(new[] { batchSize * seqLength, nodes * features });
        }
        else
        {
            // Fallback: flatten to 2D preserving first dimension as batch
            int batchDim = input.Shape[0];
            int restDim = input.Length / batchDim;
            return input.Reshape(new[] { batchDim, restDim });
        }
    }

    /// <summary>
    /// Performs native forward pass through DCRNN.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the DCRNN model, ForecastNative produces predictions from input data. This is the main inference step of the DCRNN architecture.
    /// </para>
    /// </remarks>
    private Tensor<T> ForecastNative(Tensor<T> input)
    {
        return Forward(input);
    }

    /// <summary>
    /// Performs ONNX inference.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the DCRNN model, ForecastOnnx produces predictions from input data. This is the main inference step of the DCRNN architecture.
    /// </para>
    /// </remarks>
    protected override Tensor<T> ForecastOnnx(Tensor<T> input)
    {
        if (OnnxSession is null)
            throw new InvalidOperationException("ONNX session not initialized");

        var inputMeta = OnnxSession.InputMetadata;
        var inputName = inputMeta.Keys.First();

        float[] inputData = new float[input.Data.Length];
        for (int i = 0; i < input.Data.Length; i++)
        {
            inputData[i] = NumOps.ToFloat(input.Data.Span[i]);
        }

        var inputTensor = new OnnxTensors.DenseTensor<float>(inputData, input.Shape.Select(d => (int)d).ToArray());
        var inputs = new List<NamedOnnxValue> { NamedOnnxValue.CreateFromTensor(inputName, inputTensor) };

        using var results = OnnxSession.Run(inputs);
        var outputTensor = results.First().AsTensor<float>();

        var outputShape = outputTensor.Dimensions.ToArray();
        var output = new Tensor<T>(outputShape);
        for (int i = 0; i < outputTensor.Length; i++)
        {
            output.Data.Span[i] = NumOps.FromDouble(outputTensor.GetValue(i));
        }

        return output;
    }

    #endregion

    #region Diffusion Convolution

    /// <summary>
    /// Applies diffusion convolution using precomputed diffusion matrices.
    /// </summary>
    /// <param name="input">Input tensor.</param>
    /// <returns>Output after diffusion convolution.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Diffusion convolution aggregates information from neighbors
    /// using random walk diffusion. It's computed as:
    /// sum_k (theta_k * P_forward^k * X + theta'_k * P_backward^k * X)
    /// </para>
    /// </remarks>
    private Tensor<T> ApplyDiffusionConvolution(Tensor<T> input)
    {
        if (_forwardPowers.Count == 0 || _backwardPowers.Count == 0)
            return input;

        var result = new Tensor<T>(input.Shape);
        int n = _numNodes;
        int featuresPerNode = input.Data.Length / n;

        // Sum contributions from each diffusion step
        for (int k = 0; k < Math.Min(_forwardPowers.Count, _diffusionSteps + 1); k++)
        {
            var forwardResult = ApplyMatrixToTensor(_forwardPowers[k], input, n, featuresPerNode);

            if (_options.FilterType == "dual_random_walk")
            {
                var backwardResult = ApplyMatrixToTensor(_backwardPowers[k], input, n, featuresPerNode);

                // Average forward and backward
                for (int i = 0; i < result.Data.Length; i++)
                {
                    var combined = NumOps.Multiply(
                        NumOps.FromDouble(0.5),
                        NumOps.Add(forwardResult.Data.Span[i], backwardResult.Data.Span[i]));
                    result.Data.Span[i] = NumOps.Add(result.Data.Span[i], combined);
                }
            }
            else
            {
                for (int i = 0; i < result.Data.Length; i++)
                {
                    result.Data.Span[i] = NumOps.Add(result.Data.Span[i], forwardResult.Data.Span[i]);
                }
            }
        }

        // Normalize
        T scale = NumOps.FromDouble(1.0 / (_diffusionSteps + 1));
        for (int i = 0; i < result.Data.Length; i++)
        {
            result.Data.Span[i] = NumOps.Multiply(result.Data.Span[i], scale);
        }

        return result;
    }

    /// <summary>
    /// Applies a matrix to a tensor along the node dimension.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the DCRNN model, ApplyMatrixToTensor performs a supporting step in the workflow. It keeps the DCRNN architecture pipeline consistent.
    /// </para>
    /// </remarks>
    private Tensor<T> ApplyMatrixToTensor(double[,] matrix, Tensor<T> tensor, int n, int featuresPerNode)
    {
        var result = new Tensor<T>(tensor.Shape);

        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++)
            {
                double weight = matrix[i, j];
                if (Math.Abs(weight) < 1e-10)
                    continue;

                T weightT = NumOps.FromDouble(weight);
                for (int f = 0; f < featuresPerNode; f++)
                {
                    int srcIdx = j * featuresPerNode + f;
                    int dstIdx = i * featuresPerNode + f;
                    if (srcIdx < tensor.Data.Length && dstIdx < result.Data.Length)
                    {
                        result.Data.Span[dstIdx] = NumOps.Add(
                            result.Data.Span[dstIdx],
                            NumOps.Multiply(weightT, tensor.Data.Span[srcIdx]));
                    }
                }
            }
        }

        return result;
    }

    #endregion

    #region Helper Methods

    /// <summary>
    /// Multiplies two matrices.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the DCRNN model, MatrixMultiply performs a supporting step in the workflow. It keeps the DCRNN architecture pipeline consistent.
    /// </para>
    /// </remarks>
    private double[,] MatrixMultiply(double[,] a, double[,] b)
    {
        int m = a.GetLength(0);
        int n = b.GetLength(1);
        int k = a.GetLength(1);

        var result = new double[m, n];

        for (int i = 0; i < m; i++)
        {
            for (int j = 0; j < n; j++)
            {
                double sum = 0;
                for (int p = 0; p < k; p++)
                {
                    sum += a[i, p] * b[p, j];
                }
                result[i, j] = sum;
            }
        }

        return result;
    }

    /// <summary>
    /// Generates multiple samples using MC dropout.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the DCRNN model, GenerateSamples performs a supporting step in the workflow. It keeps the DCRNN architecture pipeline consistent.
    /// </para>
    /// </remarks>
    private List<Tensor<T>> GenerateSamples(Tensor<T> input, int numSamples)
    {
        var samples = new List<Tensor<T>>();

        SetTrainingMode(true); // Enable dropout
        for (int i = 0; i < numSamples; i++)
        {
            samples.Add(Predict(input));
        }
        SetTrainingMode(false);

        return samples;
    }

    /// <summary>
    /// Computes quantiles from samples.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the DCRNN model, ComputeQuantiles performs a supporting step in the workflow. It keeps the DCRNN architecture pipeline consistent.
    /// </para>
    /// </remarks>
    private Tensor<T> ComputeQuantiles(List<Tensor<T>> samples, double[] quantiles)
    {
        if (samples.Count == 0)
            return new Tensor<T>(new[] { quantiles.Length });

        int size = samples[0].Data.Length;
        var result = new Tensor<T>(new[] { quantiles.Length, size });

        for (int i = 0; i < size; i++)
        {
            var values = samples.Select(s => NumOps.ToDouble(s.Data.Span[i])).OrderBy(v => v).ToList();

            for (int q = 0; q < quantiles.Length; q++)
            {
                int idx = (int)(quantiles[q] * (values.Count - 1));
                idx = Math.Max(0, Math.Min(values.Count - 1, idx));
                result.Data.Span[q * size + i] = NumOps.FromDouble(values[idx]);
            }
        }

        return result;
    }

    /// <summary>
    /// Computes prediction intervals from samples.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the DCRNN model, private performs a supporting step in the workflow. It keeps the DCRNN architecture pipeline consistent.
    /// </para>
    /// </remarks>
    private (Tensor<T> Forecast, Tensor<T> Lower, Tensor<T> Upper) ComputePredictionIntervals(
        List<Tensor<T>> samples, double confidenceLevel)
    {
        if (samples.Count == 0)
        {
            var empty = new Tensor<T>(new[] { 1 });
            return (empty, empty, empty);
        }

        double alpha = 1 - confidenceLevel;
        var quantiles = new[] { alpha / 2, 0.5, 1 - alpha / 2 };
        var quantileTensor = ComputeQuantiles(samples, quantiles);

        int size = samples[0].Data.Length;
        var forecast = new Tensor<T>(new[] { size });
        var lower = new Tensor<T>(new[] { size });
        var upper = new Tensor<T>(new[] { size });

        for (int i = 0; i < size; i++)
        {
            lower.Data.Span[i] = quantileTensor.Data.Span[0 * size + i];
            forecast.Data.Span[i] = quantileTensor.Data.Span[1 * size + i];
            upper.Data.Span[i] = quantileTensor.Data.Span[2 * size + i];
        }

        return (forecast, lower, upper);
    }

    /// <summary>
    /// Shifts input window to include new prediction.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the DCRNN model, ShiftInputWindow performs a supporting step in the workflow. It keeps the DCRNN architecture pipeline consistent.
    /// </para>
    /// </remarks>
    private Tensor<T> ShiftInputWindow(Tensor<T> input, Tensor<T> newData)
    {
        var shifted = new Tensor<T>(input.Shape);

        // Guard against stepSize larger than input length
        int stepSize = Math.Min(_numNodes * _numFeatures, input.Data.Length);
        if (stepSize == 0) stepSize = 1;
        int totalSteps = input.Data.Length / stepSize;

        // Shift left by one time step
        for (int t = 0; t < totalSteps - 1; t++)
        {
            for (int i = 0; i < stepSize; i++)
            {
                int srcIdx = (t + 1) * stepSize + i;
                int dstIdx = t * stepSize + i;
                if (srcIdx < input.Data.Length && dstIdx < shifted.Data.Length)
                {
                    shifted.Data.Span[dstIdx] = input.Data.Span[srcIdx];
                }
            }
        }

        // Add new prediction at the end
        int lastStepOffset = (totalSteps - 1) * stepSize;
        for (int i = 0; i < stepSize && i < newData.Data.Length; i++)
        {
            if (lastStepOffset + i < shifted.Data.Length)
            {
                shifted.Data.Span[lastStepOffset + i] = newData.Data.Span[i];
            }
        }

        return shifted;
    }

    /// <summary>
    /// Concatenates multiple prediction tensors.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the DCRNN model, ConcatenatePredictions produces predictions from input data. This is the main inference step of the DCRNN architecture.
    /// </para>
    /// </remarks>
        protected Tensor<T> ConcatenatePredictions(List<Tensor<T>> predictions)
    {
        if (predictions.Count == 0)
            return new Tensor<T>(new[] { 1 });

        int totalLength = predictions.Sum(p => p.Data.Length);
        var combined = new Tensor<T>(new[] { totalLength });

        int offset = 0;
        foreach (var pred in predictions)
        {
            for (int i = 0; i < pred.Data.Length; i++)
            {
                combined.Data.Span[offset + i] = pred.Data.Span[i];
            }
            offset += pred.Data.Length;
        }

        return combined;
    }

    /// <summary>
    /// Gets the teacher forcing ratio based on scheduled sampling.
    /// </summary>
    /// <returns>Probability of using ground truth during decoding.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Scheduled sampling gradually reduces teacher forcing
    /// as training progresses using inverse sigmoid decay.
    /// </para>
    /// </remarks>
    private double GetTeacherForcingRatio()
    {
        if (!_options.UseScheduledSampling)
            return 1.0;

        double k = _options.ScheduledSamplingDecaySteps;
        double ratio = k / (k + Math.Exp(_trainingStep / k));

        return Math.Max(_options.MinTeacherForcingRatio, ratio);
    }

    #endregion

    #region IDisposable

    /// <summary>
    /// Releases resources used by the DCRNN model.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the DCRNN model, Dispose performs a supporting step in the workflow. It keeps the DCRNN architecture pipeline consistent.
    /// </para>
    /// </remarks>
    protected override void Dispose(bool disposing)
    {
        if (disposing)
        {
            OnnxSession?.Dispose();
            _encoderGRUs.Clear();
            _decoderGRUs.Clear();
            _forwardPowers.Clear();
            _backwardPowers.Clear();
        }
        base.Dispose(disposing);
    }

    #endregion
}



