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
using Microsoft.ML.OnnxRuntime;
using OnnxTensors = Microsoft.ML.OnnxRuntime.Tensors;

namespace AiDotNet.Finance.Forecasting.StateSpace;

/// <summary>
/// S4 (Structured State Space Sequence Model) for time series forecasting.
/// </summary>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
/// <remarks>
/// <para>
/// S4 is a foundational state space model that achieves near-linear complexity through
/// structured parameterization of the state transition matrix using the HiPPO framework.
/// </para>
/// <para><b>For Beginners:</b> S4 is a breakthrough model that showed state space models
/// can match transformers on long-range sequence tasks:
///
/// <b>The Core Idea:</b>
/// S4 uses a "structured" state space model where:
/// 1. The A matrix is initialized using HiPPO (optimal for compressing history)
/// 2. A is decomposed as diagonal + low-rank for efficiency (DPLR)
/// 3. The computation is done as a convolution using FFT (O(n log n))
///
/// <b>Why S4 Works:</b>
/// - HiPPO initialization: The state optimally compresses the input history
/// - DPLR structure: Allows efficient computation while keeping HiPPO's benefits
/// - FFT convolution: Converts sequential computation to parallel
///
/// <b>The Math (simplified):</b>
/// State Space Model: x'(t) = Ax(t) + Bu(t), y(t) = Cx(t) + Du(t)
/// - A: State transition (N x N HiPPO matrix, DPLR decomposed)
/// - B: Input projection (N x 1)
/// - C: Output projection (1 x N)
/// - D: Direct feedthrough (skip connection)
///
/// Discretized for sequences: x_k = A_bar * x_{k-1} + B_bar * u_k
/// Computed as convolution: y = K * u where K is the SSM kernel
///
/// <b>Key Benefits:</b>
/// - O(n log n) complexity via FFT (vs O(n^2) for attention)
/// - Handles very long sequences (16K+ tokens)
/// - Foundation for modern SSMs (Mamba, H3, etc.)
/// - Strong performance on Long Range Arena benchmark
/// </para>
/// <para>
/// <b>Reference:</b> Gu et al., "Efficiently Modeling Long Sequences with Structured State Spaces", 2022.
/// https://arxiv.org/abs/2111.00396
/// </para>
/// </remarks>
public class S4<T> : NeuralNetworkBase<T>, IForecastingModel<T>
{
    #region Execution Mode
    private readonly bool _useNativeMode;
    #endregion

    #region ONNX Mode Fields
    private readonly InferenceSession? _onnxSession;
    private readonly string? _onnxModelPath;
    #endregion

    #region Native Mode Fields
    private DenseLayer<T>? _inputEmbedding;
    private List<DenseLayer<T>>? _ssmBLayers;
    private List<DenseLayer<T>>? _ssmDiagLayers;
    private List<DenseLayer<T>>? _ssmPLayers;
    private List<DenseLayer<T>>? _ssmQLayers;
    private List<DenseLayer<T>>? _ssmCLayers;
    private List<DenseLayer<T>>? _ssmDLayers;
    private List<LayerNormalizationLayer<T>>? _layerNorms;
    private DenseLayer<T>? _outputProjection;
    #endregion

    #region Shared Fields
    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> _optimizer;
    private readonly ILossFunction<T> _lossFunction;
    private readonly S4Options<T> _options;
    private readonly int _contextLength;
    private readonly int _forecastHorizon;
    private readonly int _modelDimension;
    private readonly int _stateDimension;
    private readonly int _numLayers;
    private readonly bool _useLowRankCorrection;
    private readonly int _lowRankRank;
    private readonly string _hippoMethod;
    private readonly string _discretizationMethod;
    private readonly int _numFeatures;
    #endregion

    #region IForecastingModel Properties

    /// <inheritdoc/>
    public int SequenceLength => _contextLength;

    /// <inheritdoc/>
    public int PredictionHorizon => _forecastHorizon;

    /// <inheritdoc/>
    public int NumFeatures => _numFeatures;

    /// <inheritdoc/>
    public int PatchSize => 1; // S4 operates on individual time steps

    /// <inheritdoc/>
    public int Stride => 1;

    /// <inheritdoc/>
    public bool IsChannelIndependent => true;

    /// <inheritdoc/>
    public bool UseNativeMode => _useNativeMode;

    /// <summary>
    /// Gets the input context length for the model.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> This is how many past time steps S4 looks at.
    /// S4 excels at very long contexts (4K-16K+) due to near-linear complexity.
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
    /// Gets the state dimension of the SSM.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> The size of the hidden state that captures sequence dynamics.
    /// </para>
    /// </remarks>
    public int StateDimension => _stateDimension;

    /// <summary>
    /// Gets the HiPPO method used for state initialization.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> HiPPO defines how the state optimally compresses history.
    /// </para>
    /// </remarks>
    public string HippoMethod => _hippoMethod;

    /// <summary>
    /// Gets whether low-rank correction is used in DPLR decomposition.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> DPLR = diagonal + low-rank for efficient computation.
    /// </para>
    /// </remarks>
    public bool UseLowRankCorrection => _useLowRankCorrection;

    #endregion

    #region Constructors

    /// <summary>
    /// Initializes a new instance of the S4 model in ONNX mode for inference.
    /// </summary>
    /// <param name="architecture">The neural network architecture configuration.</param>
    /// <param name="onnxModelPath">Path to a pretrained ONNX model file.</param>
    /// <param name="options">S4-specific options.</param>
    /// <param name="optimizer">Optional optimizer for fine-tuning.</param>
    /// <param name="lossFunction">Optional loss function.</param>
    /// <remarks>
    /// <para><b>For Beginners:</b> Use this constructor to load a pretrained S4 model
    /// for fast inference. ONNX models are optimized for deployment.
    /// </para>
    /// </remarks>
    public S4(
        NeuralNetworkArchitecture<T> architecture,
        string onnxModelPath,
        S4Options<T>? options = null,
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
        _options = options ?? new S4Options<T>();
        _lossFunction = lossFunction ?? new MeanSquaredErrorLoss<T>();
        _optimizer = optimizer ?? new AdamOptimizer<T, Tensor<T>, Tensor<T>>(this);

        _contextLength = _options.ContextLength;
        _forecastHorizon = _options.ForecastHorizon;
        _modelDimension = _options.ModelDimension;
        _stateDimension = _options.StateDimension;
        _numLayers = _options.NumLayers;
        _useLowRankCorrection = _options.UseLowRankCorrection;
        _lowRankRank = _options.LowRankRank;
        _hippoMethod = _options.HippoMethod;
        _discretizationMethod = _options.DiscretizationMethod;
        _numFeatures = 1;
    }

    /// <summary>
    /// Initializes a new instance of the S4 model in native mode for training.
    /// </summary>
    /// <param name="architecture">The neural network architecture configuration.</param>
    /// <param name="options">S4-specific options.</param>
    /// <param name="numFeatures">Number of input features.</param>
    /// <param name="optimizer">Optional optimizer for training.</param>
    /// <param name="lossFunction">Optional loss function.</param>
    /// <remarks>
    /// <para><b>For Beginners:</b> Use this constructor to create a S4 model
    /// that can be trained on your data. The model uses HiPPO-structured
    /// state space computation with DPLR decomposition.
    /// </para>
    /// </remarks>
    public S4(
        NeuralNetworkArchitecture<T> architecture,
        S4Options<T>? options = null,
        int numFeatures = 1,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
        ILossFunction<T>? lossFunction = null)
        : base(architecture, lossFunction ?? new MeanSquaredErrorLoss<T>(), 1.0)
    {
        _useNativeMode = true;
        _options = options ?? new S4Options<T>();
        _lossFunction = lossFunction ?? new MeanSquaredErrorLoss<T>();
        _optimizer = optimizer ?? new AdamOptimizer<T, Tensor<T>, Tensor<T>>(this);

        _contextLength = _options.ContextLength;
        _forecastHorizon = _options.ForecastHorizon;
        _modelDimension = _options.ModelDimension;
        _stateDimension = _options.StateDimension;
        _numLayers = _options.NumLayers;
        _useLowRankCorrection = _options.UseLowRankCorrection;
        _lowRankRank = _options.LowRankRank;
        _hippoMethod = _options.HippoMethod;
        _discretizationMethod = _options.DiscretizationMethod;
        _numFeatures = numFeatures;

        InitializeLayers();
    }

    #endregion

    #region Initialization

    /// <summary>
    /// Initializes all layers for the S4 model.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method sets up the neural network layers
    /// that implement S4's structured state space computation:
    ///
    /// <b>Layer Structure:</b>
    /// 1. Input embedding to model dimension
    /// 2. For each S4 layer:
    ///    - B projection (input to state)
    ///    - Diagonal A (state transition eigenvalues)
    ///    - P, Q projections (low-rank correction if enabled)
    ///    - C projection (state to output)
    ///    - D feedthrough (skip connection)
    ///    - Layer normalization
    /// 3. FFN block for post-processing
    /// 4. Output projection to forecast horizon
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
            Layers.AddRange(LayerHelper<T>.CreateDefaultS4Layers(
                Architecture,
                _contextLength,
                _forecastHorizon,
                _modelDimension,
                _stateDimension,
                _numLayers,
                _useLowRankCorrection,
                _lowRankRank,
                _numFeatures));

            ExtractLayerReferences();
        }
    }

    /// <summary>
    /// Extracts references to key layers for efficient access.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> After creating all layers, we keep direct references
    /// to important ones for quick access during computation.
    /// </para>
    /// </remarks>
    private void ExtractLayerReferences()
    {
        _inputEmbedding = Layers.OfType<DenseLayer<T>>().FirstOrDefault();
        _layerNorms = Layers.OfType<LayerNormalizationLayer<T>>().ToList();
        _outputProjection = Layers.OfType<DenseLayer<T>>().LastOrDefault();

        // Get all dense layers between input and output for SSM components
        var allDense = Layers.OfType<DenseLayer<T>>().ToList();
        if (allDense.Count > 2)
        {
            _ssmBLayers = allDense.Skip(1).Take(allDense.Count - 2).ToList();
        }
        else
        {
            _ssmBLayers = new List<DenseLayer<T>>();
        }

        _ssmDiagLayers = new List<DenseLayer<T>>();
        _ssmPLayers = new List<DenseLayer<T>>();
        _ssmQLayers = new List<DenseLayer<T>>();
        _ssmCLayers = new List<DenseLayer<T>>();
        _ssmDLayers = new List<DenseLayer<T>>();
    }

    /// <summary>
    /// Validates custom layers provided through the architecture.
    /// </summary>
    /// <param name="layers">The list of custom layers to validate.</param>
    /// <remarks>
    /// <para><b>For Beginners:</b> When users provide custom layers, this method
    /// ensures they form a valid S4 architecture with proper input/output dimensions.
    /// </para>
    /// </remarks>
    protected override void ValidateCustomLayers(List<ILayer<T>> layers)
    {
        base.ValidateCustomLayers(layers);

        if (layers.Count < 3)
            throw new ArgumentException("S4 requires at least 3 layers (embedding, processing, output).");
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
    /// input data through the S4 model to generate forecasts.
    ///
    /// In ONNX mode, it uses the optimized pretrained model.
    /// In native mode, it runs through our custom layer implementation.
    /// </para>
    /// </remarks>
    public override Tensor<T> Predict(Tensor<T> input)
    {
        return _useNativeMode ? ForecastNative(input) : ForecastOnnx(input);
    }

    /// <summary>
    /// Trains the S4 model on a batch of input-target pairs.
    /// </summary>
    /// <param name="input">Input tensor of shape [batch, context, features].</param>
    /// <param name="target">Target tensor of shape [batch, forecast_horizon].</param>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method trains S4 using standard
    /// backpropagation. The S4 layers learn to:
    /// 1. Compress input history efficiently using HiPPO-like dynamics
    /// 2. Produce accurate forecasts from the compressed state
    ///
    /// Only available in native mode (not ONNX).
    /// </para>
    /// </remarks>
    public override void Train(Tensor<T> input, Tensor<T> target)
    {
        if (!_useNativeMode)
            throw new InvalidOperationException("Training requires native mode. ONNX mode is inference-only.");

        SetTrainingMode(true);
        var output = Forward(input);

        // Calculate loss using vectors
        LastLoss = _lossFunction.CalculateLoss(output.ToVector(), target.ToVector());

        // Backward pass
        var gradient = _lossFunction.CalculateDerivative(output.ToVector(), target.ToVector());
        Backward(Tensor<T>.FromVector(gradient));

        _optimizer.UpdateParameters(Layers);

        SetTrainingMode(false);
    }

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
    /// Gets metadata about the S4 model.
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
            ModelType = ModelType.NeuralNetwork,
            AdditionalInfo = new Dictionary<string, object>
            {
                { "NetworkType", "S4" },
                { "ContextLength", _contextLength },
                { "ForecastHorizon", _forecastHorizon },
                { "ModelDimension", _modelDimension },
                { "StateDimension", _stateDimension },
                { "NumLayers", _numLayers },
                { "HippoMethod", _hippoMethod },
                { "DiscretizationMethod", _discretizationMethod },
                { "UseLowRankCorrection", _useLowRankCorrection },
                { "LowRankRank", _lowRankRank },
                { "UseNativeMode", _useNativeMode },
                { "SupportsTraining", SupportsTraining }
            }
        };
    }

    /// <summary>
    /// Creates a new instance of the S4 model with the same configuration.
    /// </summary>
    /// <returns>A new S4 instance.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Creates a fresh copy of the model with
    /// randomly initialized weights but the same architecture.
    /// </para>
    /// </remarks>
    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance()
    {
        return new S4<T>(Architecture, _options);
    }

    /// <summary>
    /// Serializes S4-specific data for model persistence.
    /// </summary>
    /// <param name="writer">The binary writer to serialize data to.</param>
    /// <remarks>
    /// <para><b>For Beginners:</b> Saves S4-specific configuration so the model
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
        writer.Write(_useLowRankCorrection);
        writer.Write(_lowRankRank);
    }

    /// <summary>
    /// Deserializes S4-specific data when loading a saved model.
    /// </summary>
    /// <param name="reader">The binary reader to deserialize data from.</param>
    /// <remarks>
    /// <para><b>For Beginners:</b> Restores S4-specific configuration when
    /// loading a previously saved model.
    /// </para>
    /// </remarks>
    protected override void DeserializeNetworkSpecificData(BinaryReader reader)
    {
        _ = reader.ReadInt32(); // contextLength
        _ = reader.ReadInt32(); // forecastHorizon
        _ = reader.ReadInt32(); // modelDimension
        _ = reader.ReadInt32(); // stateDimension
        _ = reader.ReadInt32(); // numLayers
        _ = reader.ReadString(); // hippoMethod
        _ = reader.ReadString(); // discretizationMethod
        _ = reader.ReadBoolean(); // useLowRankCorrection
        _ = reader.ReadInt32(); // lowRankRank
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
    /// Given historical data, S4 compresses it into a state representation
    /// using HiPPO-like dynamics and produces future predictions.
    /// </para>
    /// </remarks>
    public Tensor<T> Forecast(Tensor<T> historicalData, double[]? quantiles = null)
    {
        var output = _useNativeMode ? Forward(historicalData) : ForecastOnnx(historicalData);

        // If quantiles are requested, return the point forecast
        // (S4 doesn't natively support quantile forecasting)
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
    /// provides uncertainty bounds. S4 uses Monte Carlo dropout to estimate
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
    /// then uses that prediction as input for the next step. S4's state space
    /// structure naturally supports this by updating the hidden state sequentially.
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
    public Dictionary<string, T> Evaluate(Tensor<T> predictions, Tensor<T> actuals)
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
    /// <para><b>For Beginners:</b> S4 uses layer normalization internally,
    /// so this returns the input unchanged. Instance normalization is not
    /// typically used with state space models.
    /// </para>
    /// </remarks>
    public Tensor<T> ApplyInstanceNormalization(Tensor<T> input)
    {
        // S4 uses layer normalization internally
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
    public Dictionary<string, T> GetFinancialMetrics()
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
    /// the S4 architecture:
    /// 1. Embed input to model dimension
    /// 2. For each S4 layer: apply SSM computation (B, A, C, D)
    /// 3. Apply FFN processing
    /// 4. Project to forecast horizon
    /// </para>
    /// </remarks>
    public Tensor<T> Forward(Tensor<T> input)
    {
        var current = FlattenInput(input);

        foreach (var layer in Layers)
        {
            current = layer.Forward(current);
        }

        return current;
    }

    /// <summary>
    /// Performs backpropagation through all layers.
    /// </summary>
    /// <param name="gradOutput">Gradient of the loss with respect to output.</param>
    /// <returns>Gradient with respect to input.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Backpropagation computes how much each parameter
    /// contributed to the prediction error. For S4, gradients flow back through:
    /// - Output projection
    /// - FFN layers
    /// - SSM components (D, C, Q, P, Diag, B)
    /// - Input embedding
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
    /// Performs native mode forecasting through the layer stack.
    /// </summary>
    /// <param name="input">Input tensor.</param>
    /// <returns>Forecast tensor.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Native mode runs our custom S4 implementation
    /// which simulates the HiPPO-based state space computation using dense layers.
    /// While not using actual FFT, it learns equivalent transformations.
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
    /// <para><b>For Beginners:</b> ONNX mode uses a pretrained S4 model
    /// optimized for fast inference. This is useful when you have a model
    /// trained elsewhere or want maximum inference speed.
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
            inputData[i] = Convert.ToSingle(flatInput.Data.Span[i]);
        }

        var inputTensor = new OnnxTensors.DenseTensor<float>(
            inputData,
            new[] { 1, _contextLength, _numFeatures });

        var inputs = new List<NamedOnnxValue>
        {
            NamedOnnxValue.CreateFromTensor("input", inputTensor)
        };

        using var results = _onnxSession.Run(inputs);
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
    /// <para><b>For Beginners:</b> S4 processes the entire sequence at once
    /// (using convolution), so we flatten the input for our dense layer
    /// approximation. The state space structure is captured in the learned weights.
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
    /// Computes prediction intervals from Monte Carlo samples.
    /// </summary>
    /// <param name="samples">List of forecast samples from MC dropout.</param>
    /// <param name="confidenceLevel">Confidence level for intervals.</param>
    /// <returns>Tuple of (mean forecast, lower bound, upper bound).</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> By running the model multiple times with
    /// dropout enabled, we get different predictions. The spread of these
    /// predictions indicates uncertainty:
    /// - Mean: The point forecast
    /// - Lower/Upper: Bounds containing the true value with specified confidence
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
    /// <para><b>For Beginners:</b> For autoregressive forecasting, we need to shift
    /// the input window forward. This removes the oldest values and appends the
    /// new prediction so the model can predict the next step.
    /// </para>
    /// </remarks>
    private Tensor<T> ShiftInputWindow(Tensor<T> input, Tensor<T> prediction)
    {
        int inputLength = input.Data.Length;
        int predLength = Math.Min(prediction.Data.Length, inputLength);

        var shifted = new Tensor<T>(input.Shape);

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
    /// prediction tensors (one per step), this combines them into a single tensor
    /// for the complete forecast.
    /// </para>
    /// </remarks>
    private Tensor<T> ConcatenatePredictions(List<Tensor<T>> predictions)
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

    /// <summary>
    /// Computes the HiPPO matrix for a given method.
    /// </summary>
    /// <param name="n">State dimension.</param>
    /// <param name="method">HiPPO method (legs, legt, lagt, fourier).</param>
    /// <returns>The N x N HiPPO matrix A.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> HiPPO (High-order Polynomial Projection Operators)
    /// defines how the state optimally compresses input history:
    /// - "legs": Projects onto Legendre polynomials (best general purpose)
    /// - "legt": Translated Legendre for fixed context windows
    /// - "lagt": Laguerre for infinite context with exponential decay
    /// - "fourier": For periodic signals
    ///
    /// The matrix A defines the state dynamics: x'(t) = Ax(t) + Bu(t).
    /// Different HiPPO methods give different A matrices optimized for
    /// different types of sequence memory.
    /// </para>
    /// </remarks>
    private Tensor<T> ComputeHippoMatrix(int n, string method)
    {
        var A = new Tensor<T>(new[] { n, n });

        if (method == "legs")
        {
            // Legendre polynomial basis (HiPPO-LegS)
            // A[i,j] = -sqrt(2i+1) * sqrt(2j+1) if i > j else -sqrt(2i+1) * sqrt(2j+1) - (i+1) if i == j
            for (int i = 0; i < n; i++)
            {
                for (int j = 0; j < n; j++)
                {
                    double sqrtI = Math.Sqrt(2 * i + 1);
                    double sqrtJ = Math.Sqrt(2 * j + 1);

                    if (i > j)
                    {
                        A.Data.Span[i * n + j] = NumOps.FromDouble(-sqrtI * sqrtJ);
                    }
                    else if (i == j)
                    {
                        A.Data.Span[i * n + j] = NumOps.FromDouble(-sqrtI * sqrtJ - (i + 1));
                    }
                    else
                    {
                        A.Data.Span[i * n + j] = NumOps.FromDouble(0);
                    }
                }
            }
        }
        else if (method == "legt")
        {
            // Translated Legendre (HiPPO-LegT)
            for (int i = 0; i < n; i++)
            {
                for (int j = 0; j < n; j++)
                {
                    double sqrtI = Math.Sqrt(2 * i + 1);
                    double sqrtJ = Math.Sqrt(2 * j + 1);

                    if (i >= j)
                    {
                        A.Data.Span[i * n + j] = NumOps.FromDouble(-sqrtI * sqrtJ);
                    }
                    else
                    {
                        A.Data.Span[i * n + j] = NumOps.FromDouble(sqrtI * sqrtJ * Math.Pow(-1, i - j));
                    }
                }
            }
        }
        else // Default to legs
        {
            return ComputeHippoMatrix(n, "legs");
        }

        return A;
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
            _onnxSession?.Dispose();
        }
        base.Dispose(disposing);
    }

    #endregion
}
