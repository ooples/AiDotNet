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
using AiDotNet.NeuralNetworks.Layers.SSM;
using AiDotNet.Optimizers;
using AiDotNet.Tensors;
using Microsoft.ML.OnnxRuntime;
using OnnxTensors = Microsoft.ML.OnnxRuntime.Tensors;

using AiDotNet.Finance.Base;
namespace AiDotNet.Finance.Forecasting.StateSpace;

/// <summary>
/// HiPPO (High-order Polynomial Projection Operators) for time series forecasting.
/// </summary>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
/// <remarks>
/// <para>
/// HiPPO provides the theoretical foundation for efficient state space models like S4 and Mamba.
/// It defines optimal state matrices for compressing sequential input history into a fixed-size state.
/// </para>
/// <para><b>For Beginners:</b> HiPPO answers a fundamental question in sequence modeling:
/// "How do we optimally remember a continuous history in a fixed-size memory?"
///
/// <b>The Core Problem:</b>
/// When processing a sequence (like a time series), we need to "remember" the past.
/// But we can't store infinite history - we need a fixed-size "state".
/// HiPPO shows how to create a state that is the OPTIMAL approximation of history.
///
/// <b>How It Works:</b>
/// 1. <b>Polynomial Basis:</b> Approximate history using polynomials (like Legendre)
/// 2. <b>Optimal Projection:</b> State x(t) = coefficients of best polynomial fit to history
/// 3. <b>Online Update:</b> Update state efficiently as new inputs arrive
/// 4. <b>Memory Matrix A:</b> Derived mathematically to ensure optimal approximation
///
/// <b>The Math (simplified):</b>
/// - State Space Model: dx/dt = Ax + Bu
/// - A is the "HiPPO matrix" for your chosen polynomial basis
/// - x(t) contains polynomial coefficients: history(s) ≈ Σ x_i(t) * P_i(s)
/// - Different A matrices give different memory properties
///
/// <b>Available Methods:</b>
/// - HiPPO-LegS: Sliding window memory (recent history weighted equally)
/// - HiPPO-LegT: Fixed window [0,t] (entire history weighted equally)
/// - HiPPO-LagT: Exponential decay (recent > distant)
///
/// <b>Why HiPPO Matters:</b>
/// - Mathematically principled initialization for SSMs
/// - Provably optimal history compression
/// - Foundation for S4, Mamba, and modern sequence models
/// - Enables models to handle very long sequences efficiently
/// </para>
/// <para>
/// <b>Reference:</b> Gu et al., "HiPPO: Recurrent Memory with Optimal Polynomial Projections", 2020.
/// https://arxiv.org/abs/2008.07669
/// </para>
/// </remarks>
/// <example>
/// <code>
/// var architecture = new NeuralNetworkArchitecture&lt;double&gt;(
///     inputType: InputType.OneDimensional,
///     taskType: NeuralNetworkTaskType.Regression,
///     inputHeight: 256, inputWidth: 1, inputDepth: 1, outputSize: 24);
/// var model = new Hippo&lt;double&gt;(architecture);
/// var onnxModel = new Hippo&lt;double&gt;(architecture, "hippo.onnx");
/// </code>
/// </example>
[ModelDomain(ModelDomain.Finance)]
[ModelDomain(ModelDomain.TimeSeries)]
[ModelCategory(ModelCategory.NeuralNetwork)]
[ModelTask(ModelTask.Forecasting)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("HiPPO: Recurrent Memory with Optimal Polynomial Projections", "https://arxiv.org/abs/2008.07669", Year = 2020, Authors = "Albert Gu, Tri Dao, Stefano Ermon, Atri Rudra, Christopher Re")]
public class Hippo<T> : ForecastingModelBase<T>
{
    #region Execution Mode
    private bool _useNativeMode;
    #endregion


    #region Native Mode Fields
    // Original-paper HiPPO-RNN cells followed by the end-state decoder.
    private List<HippoMemoryCellLayer<T>> _memoryLayers = new();
    private List<LayerNormalizationLayer<T>> _blockNorms = new();
    private List<DropoutLayer<T>> _dropouts = new();
    private DenseLayer<T>? _outputProjection;
    private bool _useCustomLayerStack;
    #endregion

    #region Shared Fields
    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> _optimizer;
    private readonly ILossFunction<T> _lossFunction;
    private readonly HippoOptions<T> _options;

    /// <inheritdoc/>
    public override ModelOptions GetOptions() => _options;

    private int _contextLength;
    private int _forecastHorizon;
    private int _modelDimension;
    private int _stateDimension;
    private int _numLayers;
    private int _memorySize;
    private string _hippoMethod;
    private string _discretizationMethod;
    private int _initialTime;
    private double _timeStep;
    private double _timescaleMin;
    private double _timescaleMax;
    private bool _useGate;
    private bool _useNormalization;
    private int _numFeatures;
    #endregion

    #region IForecastingModel Properties

    /// <inheritdoc/>
    public override int SequenceLength => _contextLength;

    /// <inheritdoc/>
    public override int PredictionHorizon => _forecastHorizon;

    /// <inheritdoc/>
    public override int NumFeatures => _numFeatures;

    /// <inheritdoc/>
    public override int PatchSize => 1; // HiPPO operates on individual time steps

    /// <inheritdoc/>
    public override int Stride => 1;

    /// <inheritdoc/>
    public override bool IsChannelIndependent => true;

    /// <inheritdoc/>
    public override bool UseNativeMode => _useNativeMode;

    /// <summary>
    /// Gets the input context length for the model.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> This is how many past time steps HiPPO looks at.
    /// The polynomial projection enables efficient handling of long sequences.
    /// </para>
    /// </remarks>
    public int ContextLength => _contextLength;

    /// <summary>
    /// Gets the forecast horizon (number of future steps to predict).
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> This is how many steps into the future
    /// the model predicts in one forward pass.
    /// </para>
    /// </remarks>
    public int ForecastHorizon => _forecastHorizon;

    /// <summary>
    /// Gets whether the model supports training (native mode only).
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> ONNX mode is inference-only (pretrained models).
    /// Native mode supports both training and inference.
    /// </para>
    /// </remarks>
    public override bool SupportsTraining => _useNativeMode;

    /// <summary>
    /// Gets the state dimension (polynomial order) of the HiPPO model.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> The size of the state that stores polynomial coefficients.
    /// Larger = more accurate history approximation, but more computation.
    /// </para>
    /// </remarks>
    public int StateDimension => _stateDimension;

    /// <summary>
    /// Gets the HiPPO method used for state initialization.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Different HiPPO methods give different memory properties:
    /// - "legs": Sliding window (recent history weighted equally)
    /// - "legt": Fixed window (entire history weighted equally)
    /// - "lagt": Exponential decay (recent > distant)
    /// </para>
    /// </remarks>
    public string HippoMethod => _hippoMethod;

    #endregion

    #region Constructors

    /// <summary>
    /// Initializes a new instance of the HiPPO model in ONNX mode for inference.
    /// </summary>
    /// <param name="architecture">The neural network architecture configuration.</param>
    /// <param name="onnxModelPath">Path to a pretrained ONNX model file.</param>
    /// <param name="options">HiPPO-specific options.</param>
    /// <param name="optimizer">Optional optimizer for fine-tuning.</param>
    /// <param name="lossFunction">Optional loss function.</param>
    /// <remarks>
    /// <para><b>For Beginners:</b> Use this constructor to load a pretrained HiPPO model
    /// for fast inference. ONNX models are optimized for deployment.
    /// </para>
    /// </remarks>
    public Hippo(
        NeuralNetworkArchitecture<T> architecture,
        string onnxModelPath,
        HippoOptions<T>? options = null,
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
        _options = options ?? new HippoOptions<T>();
        Options = _options;
        _lossFunction = lossFunction ?? new MeanSquaredErrorLoss<T>();
        _optimizer = optimizer ?? new AdamOptimizer<T, Tensor<T>, Tensor<T>>(this);
        SetBaseTrainOptimizer(_optimizer);

        _contextLength = _options.ContextLength;
        _forecastHorizon = _options.ForecastHorizon;
        _modelDimension = _options.ModelDimension;
        _stateDimension = _options.StateDimension < 0
            ? _modelDimension
            : _options.StateDimension;
        _numLayers = _options.NumLayers;
        _memorySize = _options.MemorySize;
        _hippoMethod = _options.HippoMethod;
        _discretizationMethod = _options.DiscretizationMethod;
        _initialTime = _options.InitialTime;
        _timeStep = _options.TimeStep;
        _timescaleMin = _options.TimescaleMin;
        _timescaleMax = _options.TimescaleMax;
        _useGate = _options.UseGate;
        _useNormalization = _options.UseNormalization;
        _numFeatures = 1;
    }

    /// <summary>
    /// Initializes a new instance of the HiPPO model in native mode for training.
    /// </summary>
    /// <param name="architecture">The neural network architecture configuration.</param>
    /// <param name="options">HiPPO-specific options.</param>
    /// <param name="numFeatures">Number of input features.</param>
    /// <param name="optimizer">Optional optimizer for training.</param>
    /// <param name="lossFunction">Optional loss function.</param>
    /// <remarks>
    /// <para><b>For Beginners:</b> Use this constructor to create a HiPPO model
    /// that can be trained on your data. The model uses optimal polynomial
    /// projection for memory compression.
    /// </para>
    /// </remarks>
    public Hippo(
        NeuralNetworkArchitecture<T> architecture,
        HippoOptions<T>? options = null,
        int numFeatures = 1,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
        ILossFunction<T>? lossFunction = null)
        : base(architecture, lossFunction ?? new MeanSquaredErrorLoss<T>(), 1.0)
    {
        _useNativeMode = true;
        _options = options ?? new HippoOptions<T>();
        Options = _options;
        _lossFunction = lossFunction ?? new MeanSquaredErrorLoss<T>();
        _optimizer = optimizer ?? new AdamOptimizer<T, Tensor<T>, Tensor<T>>(this);
        SetBaseTrainOptimizer(_optimizer);

        _contextLength = _options.ContextLength;
        _forecastHorizon = _options.ForecastHorizon;
        _modelDimension = _options.ModelDimension;
        _stateDimension = _options.StateDimension < 0
            ? _modelDimension
            : _options.StateDimension;
        _numLayers = _options.NumLayers;
        _memorySize = _options.MemorySize;
        _hippoMethod = _options.HippoMethod;
        _discretizationMethod = _options.DiscretizationMethod;
        _initialTime = _options.InitialTime;
        _timeStep = _options.TimeStep;
        _timescaleMin = _options.TimescaleMin;
        _timescaleMax = _options.TimescaleMax;
        _useGate = _options.UseGate;
        _useNormalization = _options.UseNormalization;
        _numFeatures = numFeatures;

        InitializeLayers();
    }

    #endregion

    #region Initialization

    /// <summary>
    /// Initializes all layers for the HiPPO model.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method sets up the neural network layers
    /// that implement HiPPO's polynomial projection computation:
    ///
    /// <b>Layer Structure:</b>
    /// 1. Input embedding to model dimension
    /// 2. For each HiPPO layer:
    ///    - B projection (input to polynomial state)
    ///    - A application (state evolution via HiPPO matrix)
    ///    - C projection (polynomial state to output)
    ///    - D feedthrough (skip connection)
    ///    - Normalization and dropout
    /// 3. FFN block for post-processing
    /// 4. Output projection to forecast horizon
    /// </para>
    /// </remarks>
    protected override void InitializeLayers()
    {
        if (Architecture.Layers is not null && Architecture.Layers.Count > 0)
        {
            _useCustomLayerStack = true;
            Layers.AddRange(Architecture.Layers);
            ValidateCustomLayers(Layers);
        }
        else if (_useNativeMode)
        {
            _useCustomLayerStack = false;
            Layers.AddRange(LayerHelper<T>.CreateDefaultHippoLayers(
                Architecture,
                _contextLength,
                _forecastHorizon,
                _modelDimension,
                _stateDimension,
                _numLayers,
                _options.DropoutRate,
                _useNormalization,
                _numFeatures,
                _memorySize,
                _hippoMethod,
                _discretizationMethod,
                _initialTime,
                _timeStep,
                _timescaleMin,
                _timescaleMax,
                _useGate));

            ExtractLayerReferences();
        }
    }

    /// <summary>
    /// Extracts references to key layers for efficient access.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> After creating all layers, we keep direct references
    /// to important ones (B, A, C, D projections) for quick access during computation.
    /// </para>
    /// </remarks>
    private void ExtractLayerReferences()
    {
        // Idempotent: clear list-typed refs so a re-extract (after deserialize)
        // rebinds rather than appending a doubled stack.
        _memoryLayers.Clear();
        _blockNorms.Clear();
        _dropouts.Clear();

        // Layer order produced by CreateDefaultHippoLayers:
        //   { HippoMemoryCell ([BlockNorm]) [Dropout] }×N [OutputHead Dense]
        var dense = Layers.OfType<DenseLayer<T>>().ToList();
        _outputProjection = dense.LastOrDefault();

        _memoryLayers = Layers.OfType<HippoMemoryCellLayer<T>>().ToList();
        _dropouts = Layers.OfType<DropoutLayer<T>>().ToList();

        var norms = Layers.OfType<LayerNormalizationLayer<T>>().ToList();
        _blockNorms = _useNormalization ? norms : new List<LayerNormalizationLayer<T>>();
    }

    /// <summary>
    /// Validates custom layers provided through the architecture.
    /// </summary>
    /// <param name="layers">The list of custom layers to validate.</param>
    /// <remarks>
    /// <para><b>For Beginners:</b> When users provide custom layers, this method
    /// ensures they form a valid HiPPO architecture with proper input/output dimensions.
    /// </para>
    /// </remarks>
    protected override void ValidateCustomLayers(List<ILayer<T>> layers)
    {
        base.ValidateCustomLayers(layers);

        if (layers.Count == 0)
            throw new ArgumentException("A custom HiPPO architecture must contain at least one layer.");
    }

    #endregion

    #region NeuralNetworkBase Overrides

    /// <summary>
    /// Performs forward prediction on the input tensor.
    /// </summary>
    /// <param name="input">Input tensor of shape [batch, context, features].</param>
    /// <returns>Output tensor of shape [batch, forecast_horizon].</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This is the main prediction method that runs
    /// input data through the HiPPO model to generate forecasts.
    ///
    /// In ONNX mode, it uses the optimized pretrained model.
    /// In native mode, it runs through our custom layer implementation.
    /// </para>
    /// </remarks>
    protected override Tensor<T> PredictCore(Tensor<T> input)
    {
        return _useNativeMode ? ForecastNative(input) : ForecastOnnx(input);
    }

    /// <summary>
    /// Trains the HiPPO model on a batch of input-target pairs.
    /// </summary>
    /// <param name="input">Input tensor of shape [batch, context, features].</param>
    /// <param name="target">Target tensor of shape [batch, forecast_horizon].</param>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method trains HiPPO using standard
    /// backpropagation. The HiPPO layers learn to:
    /// 1. Project input into optimal polynomial state space
    /// 2. Evolve the state using HiPPO-like dynamics
    /// 3. Produce accurate forecasts from the polynomial approximation
    ///
    /// Only available in native mode (not ONNX).
    /// </para>
    /// </remarks>
    public override void Train(Tensor<T> input, Tensor<T> target)
    {
        if (!_useNativeMode)
            throw new InvalidOperationException("Training is only supported in native mode.");

        base.Train(input, target);
    }

    /// <summary>
    /// Training-mode forward: calls <see cref="Forward"/> directly so
    /// the tape sees the Hippo SSM layers in training mode. The default
    /// path uses <c>ForecastNative</c>, which sets inference mode.
    /// </summary>
    protected override Tensor<T> ForwardNativeForTraining(Tensor<T> input)
    {
        return Forward(input);
    }

    /// <summary>
    /// HiPPO's recurrent state update depends on the current time step and is not a
    /// static operation graph that can be compiled once and replayed safely.
    /// </summary>
    protected override bool SupportsFusedCompiledTraining => false;

    /// <summary>
    /// Updates the model parameters using the optimizer (required override).
    /// </summary>
    /// <param name="gradients">Gradient vector (not used - layers handle gradients internally).</param>
    /// <remarks>
    /// <para><b>For Beginners:</b> This override is required by the base class.
    /// Actual parameter updates happen through the optimizer in the Train method.
    /// </para>
    /// </remarks>
    public override void UpdateParameters(Vector<T> gradients)
    {
        // Parameters are updated through the optimizer in Train()
    }

    /// <summary>
    /// Gets metadata about the HiPPO model.
    /// </summary>
    /// <returns>ModelMetadata containing model information.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Returns information about the model architecture
    /// and configuration, useful for logging and debugging.
    /// </para>
    /// </remarks>
    public override ModelMetadata<T> GetModelMetadata()
    {
        return new ModelMetadata<T>
        {
            AdditionalInfo = new Dictionary<string, object>
            {
                { "NetworkType", "HiPPO" },
                { "ContextLength", _contextLength },
                { "ForecastHorizon", _forecastHorizon },
                { "ModelDimension", _modelDimension },
                { "StateDimension", _stateDimension },
                { "NumLayers", _numLayers },
                { "MemorySize", _memorySize },
                { "HippoMethod", _hippoMethod },
                { "DiscretizationMethod", _discretizationMethod },
                { "InitialTime", _initialTime },
                { "TimeStep", _timeStep },
                { "TimescaleMin", _timescaleMin },
                { "TimescaleMax", _timescaleMax },
                { "UseGate", _useGate },
                { "UseNormalization", _useNormalization },
                { "UseNativeMode", _useNativeMode },
                { "SupportsTraining", SupportsTraining }
            }
        };
    }

    /// <summary>
    /// Creates a new instance of the HiPPO model with the same configuration.
    /// </summary>
    /// <returns>A new HiPPO instance.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Creates a fresh copy of the model with
    /// randomly initialized weights but the same architecture.
    /// </para>
    /// </remarks>
    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance()
    {
        return new Hippo<T>(Architecture, new HippoOptions<T>(_options), _numFeatures);
    }

    /// <summary>
    /// Serializes HiPPO-specific data for model persistence.
    /// </summary>
    /// <param name="writer">The binary writer to serialize data to.</param>
    /// <remarks>
    /// <para><b>For Beginners:</b> Saves HiPPO-specific configuration so the model
    /// can be reconstructed later.
    /// </para>
    /// </remarks>
    protected override void SerializeNetworkSpecificData(BinaryWriter writer)
    {
        writer.Write(_contextLength);
        writer.Write(_forecastHorizon);
        writer.Write(_modelDimension);
        writer.Write(_stateDimension);
        writer.Write(_numLayers);
        writer.Write(_hippoMethod);
        writer.Write(_discretizationMethod);
        writer.Write(_timescaleMin);
        writer.Write(_timescaleMax);
        writer.Write(_useNormalization);

        // Keep the original HiPPO payload above byte-for-byte compatible.
        // New recurrent-cell settings are append-only so older payloads can
        // still be loaded with the paper defaults below.
        writer.Write(_memorySize);
        writer.Write(_initialTime);
        writer.Write(_timeStep);
        writer.Write(_useGate);
    }

    /// <summary>
    /// Deserializes HiPPO-specific data when loading a saved model.
    /// </summary>
    /// <param name="reader">The binary reader to deserialize data from.</param>
    /// <remarks>
    /// <para><b>For Beginners:</b> Restores HiPPO-specific configuration when
    /// loading a previously saved model.
    /// </para>
    /// </remarks>
    protected override void DeserializeNetworkSpecificData(BinaryReader reader)
    {
        _contextLength = reader.ReadInt32();
        _forecastHorizon = reader.ReadInt32();
        _modelDimension = reader.ReadInt32();
        _stateDimension = reader.ReadInt32();
        _numLayers = reader.ReadInt32();
        _hippoMethod = reader.ReadString();
        _discretizationMethod = reader.ReadString();
        _timescaleMin = reader.ReadDouble();
        _timescaleMax = reader.ReadDouble();
        _useNormalization = reader.ReadBoolean();

        // These fields were appended after the original payload. Preserve
        // the paper configuration when loading a model serialized before
        // they existed.
        _memorySize = 1;
        _initialTime = 0;
        _timeStep = 0.0;
        _useGate = true;
        if (reader.BaseStream.Position < reader.BaseStream.Length)
            _memorySize = reader.ReadInt32();
        if (reader.BaseStream.Position < reader.BaseStream.Length)
            _initialTime = reader.ReadInt32();
        if (reader.BaseStream.Position < reader.BaseStream.Length)
            _timeStep = reader.ReadDouble();
        if (reader.BaseStream.Position < reader.BaseStream.Length)
            _useGate = reader.ReadBoolean();

        // Re-bind cached layer references so a deserialized/cloned model runs on
        // the restored layers, not the construction-time random ones.
        ExtractLayerReferences();
    }

    #endregion

    #region IForecastingModel Implementation

    /// <summary>
    /// Generates forecasts for the given input time series.
    /// </summary>
    /// <param name="historicalData">Input tensor of shape [batch, context, features].</param>
    /// <param name="quantiles">Optional quantile levels for probabilistic forecasting.</param>
    /// <returns>Forecast tensor of shape [batch, forecast_horizon].</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This is the main forecasting interface.
    /// Given historical data, HiPPO compresses it into an optimal polynomial
    /// representation and produces future predictions.
    /// </para>
    /// </remarks>
    public override Tensor<T> Forecast(Tensor<T> historicalData, double[]? quantiles = null)
    {
        var output = _useNativeMode ? Forward(historicalData) : ForecastOnnx(historicalData);

        // If quantiles are requested, return the point forecast
        // (HiPPO doesn't natively support quantile forecasting)
        return output;
    }

    /// <summary>
    /// Generates forecasts with prediction intervals for uncertainty quantification.
    /// </summary>
    /// <param name="input">Input tensor of shape [batch, context, features].</param>
    /// <param name="confidenceLevel">Confidence level for intervals (e.g., 0.95).</param>
    /// <returns>Tuple of (point forecast, lower bound, upper bound).</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> In addition to point predictions, this method
    /// provides uncertainty bounds. HiPPO uses Monte Carlo dropout to estimate
    /// prediction uncertainty by running multiple forward passes with different
    /// dropout masks.
    /// </para>
    /// </remarks>
    public (Tensor<T> Forecast, Tensor<T> Lower, Tensor<T> Upper) ForecastWithIntervals(
        Tensor<T> input,
        double confidenceLevel = 0.95)
    {
        if (!_useNativeMode)
        {
            var forecast = ForecastOnnx(input);
            return (forecast, forecast, forecast);
        }

        // Use Monte Carlo dropout for uncertainty estimation
        const int numSamples = 30;
        var samples = new List<Tensor<T>>();

        SetTrainingMode(true); // Enable dropout
        for (int i = 0; i < numSamples; i++)
        {
            samples.Add(Forward(input));
        }
        SetTrainingMode(false);

        return ComputePredictionIntervals(samples, confidenceLevel);
    }

    /// <summary>
    /// Performs autoregressive forecasting step by step.
    /// </summary>
    /// <param name="input">Initial input tensor.</param>
    /// <param name="steps">Number of autoregressive steps to perform.</param>
    /// <returns>Forecast tensor containing all predicted steps.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Autoregressive forecasting predicts one step,
    /// then uses that prediction as input for the next step. HiPPO's polynomial
    /// state naturally captures the history needed for each step.
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

            // Shift input window and append prediction for next step
            currentInput = ShiftInputWindow(currentInput, prediction);
        }

        return ConcatenatePredictions(predictions);
    }

    /// <summary>
    /// Evaluates forecast quality against actual values.
    /// </summary>
    /// <param name="predictions">Predicted values.</param>
    /// <param name="actuals">Actual observed values.</param>
    /// <returns>Dictionary of evaluation metrics (MSE, MAE, RMSE, etc.).</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Compares predictions to actual values using
    /// standard forecasting metrics to measure how well the model performed.
    /// </para>
    /// </remarks>
    public override Dictionary<string, T> Evaluate(Tensor<T> predictions, Tensor<T> actuals)
    {
        var metrics = new Dictionary<string, T>();

        // Calculate MSE
        T mse = NumOps.Zero;
        T mae = NumOps.Zero;
        int count = Math.Min(predictions.Data.Length, actuals.Data.Length);

        for (int i = 0; i < count; i++)
        {
            T diff = NumOps.Subtract(predictions.Data.Span[i], actuals.Data.Span[i]);
            mse = NumOps.Add(mse, NumOps.Multiply(diff, diff));
            mae = NumOps.Add(mae, NumOps.Abs(diff));
        }

        mse = NumOps.Divide(mse, NumOps.FromDouble(count));
        mae = NumOps.Divide(mae, NumOps.FromDouble(count));
        T rmse = NumOps.FromDouble(Math.Sqrt(NumOps.ToDouble(mse)));

        metrics["MSE"] = mse;
        metrics["MAE"] = mae;
        metrics["RMSE"] = rmse;

        return metrics;
    }

    /// <summary>
    /// Applies instance normalization to the input.
    /// </summary>
    /// <param name="input">Input tensor to normalize.</param>
    /// <returns>Normalized tensor.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> HiPPO uses layer normalization internally,
    /// so this returns the input unchanged.
    /// </para>
    /// </remarks>
    public override Tensor<T> ApplyInstanceNormalization(Tensor<T> input)
    {
        // HiPPO uses layer normalization internally
        return input;
    }

    /// <summary>
    /// Gets financial-specific metrics about the model.
    /// </summary>
    /// <returns>Dictionary of financial metrics.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Returns metrics relevant for financial forecasting
    /// applications, such as the last training loss and model configuration info.
    /// </para>
    /// </remarks>
    public override Dictionary<string, T> GetFinancialMetrics()
    {
        T lastLoss = LastLoss is not null ? LastLoss : NumOps.Zero;

        return new Dictionary<string, T>
        {
            ["LastLoss"] = lastLoss,
            ["ContextLength"] = NumOps.FromDouble(_contextLength),
            ["ForecastHorizon"] = NumOps.FromDouble(_forecastHorizon),
            ["ModelDimension"] = NumOps.FromDouble(_modelDimension),
            ["StateDimension"] = NumOps.FromDouble(_stateDimension),
            ["NumLayers"] = NumOps.FromDouble(_numLayers)
        };
    }

    #endregion

    #region Forward/Backward Pass

    /// <summary>
    /// Performs the forward pass through all layers.
    /// </summary>
    /// <param name="input">Input tensor.</param>
    /// <returns>Output tensor after processing through all layers.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> The forward pass runs the input through
    /// the HiPPO architecture:
    /// 1. Embed input to model dimension
    /// 2. For each HiPPO layer: project to polynomial state, apply A matrix, read out
    /// 3. Apply FFN processing
    /// 4. Project to forecast horizon
    /// </para>
    /// </remarks>
    public Tensor<T> Forward(Tensor<T> input)
    {
        if (_useCustomLayerStack)
        {
            var custom = input;
            foreach (var layer in Layers) custom = layer.Forward(custom);
            return custom;
        }

        var x = input;
        for (int i = 0; i < _memoryLayers.Count; i++)
        {
            x = _memoryLayers[i].Forward(x);
            if (i < _blockNorms.Count)
                x = _blockNorms[i].Forward(x);
            if (i < _dropouts.Count)
                x = _dropouts[i].Forward(x);
        }

        int sequenceAxis = x.Rank - 2;
        var finalState = Engine.TensorSliceAxis(x, sequenceAxis, x.Shape[sequenceAxis] - 1);
        return _outputProjection is not null
            ? _outputProjection.Forward(finalState)
            : finalState;
    }

    /// <summary>
    /// Performs native mode forecasting through the layer stack.
    /// </summary>
    /// <param name="input">Input tensor.</param>
    /// <returns>Forecast tensor.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Native mode runs our custom HiPPO implementation
    /// which uses optimal polynomial projection for memory compression.
    /// </para>
    /// </remarks>
    private Tensor<T> ForecastNative(Tensor<T> input)
    {
        SetTrainingMode(false);
        return Forward(input);
    }

    /// <summary>
    /// Performs ONNX mode forecasting using the pretrained model.
    /// </summary>
    /// <param name="input">Input tensor.</param>
    /// <returns>Forecast tensor.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> ONNX mode uses a pretrained HiPPO model
    /// optimized for fast inference. This is useful when you have a model
    /// trained elsewhere or want maximum inference speed.
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
            inputData[i] = Convert.ToSingle(flatInput.Data.Span[i]);
        }

        var inputTensor = new OnnxTensors.DenseTensor<float>(
            inputData,
            new[] { 1, _contextLength, _numFeatures });

        var inputs = new List<NamedOnnxValue>
        {
            NamedOnnxValue.CreateFromTensor("input", inputTensor)
        };

        using var results = OnnxSession.Run(inputs);
        var outputTensor = results[0].AsTensor<float>();

        var output = new Tensor<T>(new[] { _forecastHorizon });
        for (int i = 0; i < _forecastHorizon; i++)
        {
            output.Data.Span[i] = NumOps.FromDouble(outputTensor.GetValue(i));
        }

        return output;
    }

    #endregion

    #region Model-Specific Processing

    /// <summary>
    /// Flattens the input tensor for processing through dense layers.
    /// </summary>
    /// <param name="input">Input tensor of shape [batch, context, features].</param>
    /// <returns>Flattened tensor of shape [batch, context * features].</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> HiPPO operates on the entire input sequence,
    /// so we flatten it for processing through our dense layer approximation.
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
        for (int i = 0; i < totalSize; i++)
        {
            flattened.Data.Span[i] = input.Data.Span[i];
        }

        return flattened;
    }

    /// <summary>
    /// Computes prediction intervals from Monte Carlo samples.
    /// </summary>
    /// <param name="samples">List of forecast samples from MC dropout.</param>
    /// <param name="confidenceLevel">Confidence level for intervals.</param>
    /// <returns>Tuple of (mean forecast, lower bound, upper bound).</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> By running the model multiple times with
    /// dropout enabled, we get different predictions. The spread of these
    /// predictions indicates uncertainty.
    /// </para>
    /// </remarks>
    private (Tensor<T> Forecast, Tensor<T> Lower, Tensor<T> Upper) ComputePredictionIntervals(
        List<Tensor<T>> samples,
        double confidenceLevel)
    {
        int horizonLength = samples[0].Data.Length;
        var mean = new Tensor<T>(new[] { horizonLength });
        var lower = new Tensor<T>(new[] { horizonLength });
        var upper = new Tensor<T>(new[] { horizonLength });

        double alpha = 1.0 - confidenceLevel;
        int lowerIdx = (int)(samples.Count * alpha / 2);
        int upperIdx = samples.Count - 1 - lowerIdx;

        for (int t = 0; t < horizonLength; t++)
        {
            var values = new List<double>();
            double sum = 0;

            foreach (var sample in samples)
            {
                double val = NumOps.ToDouble(sample.Data.Span[t]);
                values.Add(val);
                sum += val;
            }

            values.Sort();
            mean.Data.Span[t] = NumOps.FromDouble(sum / samples.Count);
            lower.Data.Span[t] = NumOps.FromDouble(values[lowerIdx]);
            upper.Data.Span[t] = NumOps.FromDouble(values[upperIdx]);
        }

        return (mean, lower, upper);
    }

    /// <summary>
    /// Shifts the input window by removing oldest values and appending new prediction.
    /// </summary>
    /// <param name="input">Current input tensor.</param>
    /// <param name="prediction">New prediction to append.</param>
    /// <returns>Shifted input tensor for next autoregressive step.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> For autoregressive forecasting, we shift
    /// the input window forward by removing old values and adding the prediction.
    /// </para>
    /// </remarks>
    private Tensor<T> ShiftInputWindow(Tensor<T> input, Tensor<T> prediction)
    {
        int inputLength = input.Data.Length;
        int predLength = Math.Min(prediction.Data.Length, inputLength);

        var shifted = new Tensor<T>(input._shape);

        // Copy shifted values (skip first predLength values)
        for (int i = predLength; i < inputLength; i++)
        {
            shifted.Data.Span[i - predLength] = input.Data.Span[i];
        }

        // Append prediction values at the end
        for (int i = 0; i < predLength; i++)
        {
            shifted.Data.Span[inputLength - predLength + i] = prediction.Data.Span[i];
        }

        return shifted;
    }

    /// <summary>
    /// Concatenates multiple prediction tensors into a single tensor.
    /// </summary>
    /// <param name="predictions">List of prediction tensors.</param>
    /// <returns>Concatenated tensor containing all predictions.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> After autoregressive forecasting produces multiple
    /// prediction tensors, this combines them into a single tensor.
    /// </para>
    /// </remarks>
        protected Tensor<T> ConcatenatePredictions(List<Tensor<T>> predictions)
    {
        if (predictions.Count == 0)
            return new Tensor<T>(new[] { 0 });

        int totalLength = 0;
        foreach (var pred in predictions)
        {
            totalLength += pred.Data.Length;
        }

        var result = new Tensor<T>(new[] { totalLength });
        int offset = 0;

        foreach (var pred in predictions)
        {
            for (int i = 0; i < pred.Data.Length; i++)
            {
                result.Data.Span[offset + i] = pred.Data.Span[i];
            }
            offset += pred.Data.Length;
        }

        return result;
    }

    #endregion

    #region IDisposable

    /// <summary>
    /// Disposes of managed and unmanaged resources.
    /// </summary>
    /// <param name="disposing">True if called from Dispose(), false if from finalizer.</param>
    /// <remarks>
    /// <para><b>For Beginners:</b> Proper cleanup ensures ONNX sessions and other
    /// resources are released when the model is no longer needed.
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



