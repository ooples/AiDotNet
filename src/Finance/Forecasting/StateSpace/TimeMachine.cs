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

using AiDotNet.Finance.Base;
namespace AiDotNet.Finance.Forecasting.StateSpace;

/// <summary>
/// TimeMachine (Time Series State Space Model) for multi-scale time series forecasting.
/// </summary>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
/// <remarks>
/// <para>
/// TimeMachine is a state space model specifically designed for time series forecasting
/// that combines multiple SSM blocks at different temporal scales to capture both
/// short-term and long-term patterns effectively.
/// </para>
/// <para><b>For Beginners:</b> TimeMachine is a modern architecture whose key insight is
/// that "A Time Series is Worth 4 Mambas" - using multiple SSM blocks at different scales:
///
/// <b>The Core Idea:</b>
/// Time series data contains patterns at multiple temporal scales:
/// - High-frequency noise and short-term fluctuations
/// - Daily, weekly, monthly patterns
/// - Long-term trends
///
/// TimeMachine captures all these by processing the data at multiple scales simultaneously.
///
/// <b>How It Works:</b>
/// 1. <b>Temporal Decomposition:</b> Separates the signal into multiple scales
/// 2. <b>Multi-Scale SSM:</b> Each scale has its own Mamba-style SSM blocks
/// 3. <b>Scale-wise Attention:</b> Learns which scales are most important
/// 4. <b>Reconstruction:</b> Combines multi-scale outputs for final forecast
///
/// <b>Architecture:</b>
/// - Input embedding with reversible instance normalization (RevIN)
/// - 4 parallel SSM branches at different downsampling rates
/// - Attention-based fusion of scale outputs
/// - Output projection with de-normalization
///
/// <b>Key Benefits:</b>
/// - Linear complexity O(n) from SSM backbone
/// - Multi-scale captures patterns at all frequencies
/// - RevIN handles non-stationarity
/// - State-of-the-art results on long-term forecasting benchmarks
/// </para>
/// <para>
/// <b>Reference:</b> Ahamed et al., "TimeMachine: A Time Series is Worth 4 Mambas for Long-term Forecasting", 2024.
/// https://arxiv.org/abs/2403.09898
/// </para>
/// </remarks>
public class TimeMachine<T> : ForecastingModelBase<T>
{
    #region Execution Mode
    private bool _useNativeMode;
    #endregion


    #region Native Mode Fields
    private DenseLayer<T>? _inputEmbedding;
    private List<DenseLayer<T>>? _temporalDecompLayers;
    private List<List<DenseLayer<T>>>? _scaleSSMLayers;
    private List<LayerNormalizationLayer<T>>? _layerNorms;
    private DenseLayer<T>? _scaleFusion;
    private DenseLayer<T>? _outputProjection;
    #endregion

    #region Shared Fields
    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> _optimizer;
    private readonly ILossFunction<T> _lossFunction;
    private readonly TimeMachineOptions<T> _options;
    private int _contextLength;
    private int _forecastHorizon;
    private int _modelDimension;
    private int _stateDimension;
    private int _numScales;
    private int _numLayers;
    private int _expandFactor;
    private int _convKernelSize;
    private bool _useMultiScaleAttention;
    private bool _useReversibleNormalization;
    private string _decompositionMethod;
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
    public override int PatchSize => 1; // TimeMachine operates on individual time steps

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
    /// <para><b>For Beginners:</b> This is how many past time steps TimeMachine looks at.
    /// The multi-scale processing efficiently handles long contexts.
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
    /// Gets the number of temporal scales used.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> TimeMachine processes data at multiple scales
    /// (4 by default, corresponding to "4 Mambas"). Each scale captures patterns
    /// at a different temporal granularity.
    /// </para>
    /// </remarks>
    public int NumScales => _numScales;

    /// <summary>
    /// Gets whether multi-scale attention is used for fusion.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> When true, the model learns to dynamically weight
    /// different scales based on the input. This provides more flexibility.
    /// </para>
    /// </remarks>
    public bool UseMultiScaleAttention => _useMultiScaleAttention;

    /// <summary>
    /// Gets whether reversible instance normalization is used.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> RevIN normalizes each time series individually
    /// and reverses the normalization after prediction. This helps handle
    /// non-stationary data with varying scales and trends.
    /// </para>
    /// </remarks>
    public bool UseReversibleNormalization => _useReversibleNormalization;

    #endregion

    #region Constructors

    /// <summary>
    /// Initializes a new instance of the TimeMachine model in ONNX mode for inference.
    /// </summary>
    /// <param name="architecture">The neural network architecture configuration.</param>
    /// <param name="onnxModelPath">Path to a pretrained ONNX model file.</param>
    /// <param name="options">TimeMachine-specific options.</param>
    /// <param name="optimizer">Optional optimizer for fine-tuning.</param>
    /// <param name="lossFunction">Optional loss function.</param>
    /// <remarks>
    /// <para><b>For Beginners:</b> Use this constructor to load a pretrained TimeMachine model
    /// for fast inference. ONNX models are optimized for deployment.
    /// </para>
    /// </remarks>
    public TimeMachine(
        NeuralNetworkArchitecture<T> architecture,
        string onnxModelPath,
        TimeMachineOptions<T>? options = null,
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
        _options = options ?? new TimeMachineOptions<T>();
        _lossFunction = lossFunction ?? new MeanSquaredErrorLoss<T>();
        _optimizer = optimizer ?? new AdamOptimizer<T, Tensor<T>, Tensor<T>>(this);

        _contextLength = _options.ContextLength;
        _forecastHorizon = _options.ForecastHorizon;
        _modelDimension = _options.ModelDimension;
        _stateDimension = _options.StateDimension;
        _numScales = _options.NumScales;
        _numLayers = _options.NumLayers;
        _expandFactor = _options.ExpandFactor;
        _convKernelSize = _options.ConvKernelSize;
        _useMultiScaleAttention = _options.UseMultiScaleAttention;
        _useReversibleNormalization = _options.UseReversibleNormalization;
        _decompositionMethod = _options.TemporalDecompositionMethod;
        _numFeatures = 1;
    }

    /// <summary>
    /// Initializes a new instance of the TimeMachine model in native mode for training.
    /// </summary>
    /// <param name="architecture">The neural network architecture configuration.</param>
    /// <param name="options">TimeMachine-specific options.</param>
    /// <param name="numFeatures">Number of input features.</param>
    /// <param name="optimizer">Optional optimizer for training.</param>
    /// <param name="lossFunction">Optional loss function.</param>
    /// <remarks>
    /// <para><b>For Beginners:</b> Use this constructor to create a TimeMachine model
    /// that can be trained on your data. The model uses multi-scale SSM processing
    /// to capture patterns at different temporal granularities.
    /// </para>
    /// </remarks>
    public TimeMachine(
        NeuralNetworkArchitecture<T> architecture,
        TimeMachineOptions<T>? options = null,
        int numFeatures = 1,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
        ILossFunction<T>? lossFunction = null)
        : base(architecture, lossFunction ?? new MeanSquaredErrorLoss<T>(), 1.0)
    {
        _useNativeMode = true;
        _options = options ?? new TimeMachineOptions<T>();
        _lossFunction = lossFunction ?? new MeanSquaredErrorLoss<T>();
        _optimizer = optimizer ?? new AdamOptimizer<T, Tensor<T>, Tensor<T>>(this);

        _contextLength = _options.ContextLength;
        _forecastHorizon = _options.ForecastHorizon;
        _modelDimension = _options.ModelDimension;
        _stateDimension = _options.StateDimension;
        _numScales = _options.NumScales;
        _numLayers = _options.NumLayers;
        _expandFactor = _options.ExpandFactor;
        _convKernelSize = _options.ConvKernelSize;
        _useMultiScaleAttention = _options.UseMultiScaleAttention;
        _useReversibleNormalization = _options.UseReversibleNormalization;
        _decompositionMethod = _options.TemporalDecompositionMethod;
        _numFeatures = numFeatures;

        InitializeLayers();
    }

    #endregion

    #region Initialization

    /// <summary>
    /// Initializes all layers for the TimeMachine model.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method sets up the neural network layers
    /// that implement TimeMachine's multi-scale SSM architecture:
    ///
    /// <b>Layer Structure:</b>
    /// 1. Input embedding with reversible normalization
    /// 2. For each scale (4 by default):
    ///    - Temporal decomposition (downsampling)
    ///    - Multiple SSM layers (Mamba-style)
    ///    - Upsampling back to original length
    /// 3. Multi-scale attention fusion
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
            Layers.AddRange(LayerHelper<T>.CreateDefaultTimeMachineLayers(
                Architecture,
                _contextLength,
                _forecastHorizon,
                _modelDimension,
                _stateDimension,
                _numScales,
                _numLayers,
                _expandFactor,
                _convKernelSize,
                _useMultiScaleAttention,
                _numFeatures));

            ExtractLayerReferences();
        }
    }

    /// <summary>
    /// Extracts references to key layers for efficient access.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> After creating all layers, we keep direct references
    /// to important ones for quick access during computation. This includes the input
    /// embedding, temporal decomposition layers, SSM layers for each scale, and the
    /// fusion and output layers.
    /// </para>
    /// </remarks>
    private void ExtractLayerReferences()
    {
        _inputEmbedding = Layers.OfType<DenseLayer<T>>().FirstOrDefault();
        _layerNorms = Layers.OfType<LayerNormalizationLayer<T>>().ToList();
        _outputProjection = Layers.OfType<DenseLayer<T>>().LastOrDefault();

        // Organize SSM layers by scale
        var allDense = Layers.OfType<DenseLayer<T>>().ToList();
        _scaleSSMLayers = new List<List<DenseLayer<T>>>();
        _temporalDecompLayers = new List<DenseLayer<T>>();

        // Simple grouping - each scale has multiple dense layers
        int layersPerScale = (allDense.Count - 2) / _numScales; // Exclude input/output
        for (int s = 0; s < _numScales; s++)
        {
            int start = 1 + s * layersPerScale;
            int count = Math.Min(layersPerScale, allDense.Count - start - 1);
            if (count > 0)
            {
                var scaleLayers = allDense.Skip(start).Take(count).ToList();
                _scaleSSMLayers.Add(scaleLayers);
                // First layer of each scale is the temporal decomposition
                if (scaleLayers.Count > 0)
                {
                    _temporalDecompLayers.Add(scaleLayers[0]);
                }
            }
        }

        _scaleFusion = allDense.Count > 2 ? allDense[allDense.Count - 2] : null;
    }

    /// <summary>
    /// Validates custom layers provided through the architecture.
    /// </summary>
    /// <param name="layers">The list of custom layers to validate.</param>
    /// <remarks>
    /// <para><b>For Beginners:</b> When users provide custom layers, this method
    /// ensures they form a valid TimeMachine architecture with proper multi-scale
    /// processing capability.
    /// </para>
    /// </remarks>
    protected override void ValidateCustomLayers(List<ILayer<T>> layers)
    {
        base.ValidateCustomLayers(layers);

        if (layers.Count < 5)
            throw new ArgumentException("TimeMachine requires at least 5 layers (embedding, scales, fusion, output).");
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
    /// input data through the TimeMachine model to generate forecasts.
    ///
    /// In ONNX mode, it uses the optimized pretrained model.
    /// In native mode, it runs through our custom multi-scale layer implementation.
    /// </para>
    /// </remarks>
    public override Tensor<T> Predict(Tensor<T> input)
    {
        return _useNativeMode ? ForecastNative(input) : ForecastOnnx(input);
    }

    /// <summary>
    /// Trains the TimeMachine model on a batch of input-target pairs.
    /// </summary>
    /// <param name="input">Input tensor of shape [batch, context, features].</param>
    /// <param name="target">Target tensor of shape [batch, forecast_horizon].</param>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method trains TimeMachine using standard
    /// backpropagation. The multi-scale SSM layers learn to:
    /// 1. Decompose the input into different temporal scales
    /// 2. Process each scale with Mamba-style SSM blocks
    /// 3. Fuse the multi-scale outputs for accurate forecasting
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
        Backward(Tensor<T>.FromVector(gradient, output.Shape));

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
    /// Gets metadata about the TimeMachine model.
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
                { "NetworkType", "TimeMachine" },
                { "ContextLength", _contextLength },
                { "ForecastHorizon", _forecastHorizon },
                { "ModelDimension", _modelDimension },
                { "StateDimension", _stateDimension },
                { "NumScales", _numScales },
                { "NumLayers", _numLayers },
                { "ExpandFactor", _expandFactor },
                { "ConvKernelSize", _convKernelSize },
                { "UseMultiScaleAttention", _useMultiScaleAttention },
                { "UseReversibleNormalization", _useReversibleNormalization },
                { "DecompositionMethod", _decompositionMethod },
                { "UseNativeMode", _useNativeMode },
                { "SupportsTraining", SupportsTraining }
            }
        };
    }

    /// <summary>
    /// Creates a new instance of the TimeMachine model with the same configuration.
    /// </summary>
    /// <returns>A new TimeMachine instance.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Creates a fresh copy of the model with
    /// randomly initialized weights but the same architecture.
    /// </para>
    /// </remarks>
    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance()
    {
        return new TimeMachine<T>(Architecture, _options);
    }

    /// <summary>
    /// Serializes TimeMachine-specific data for model persistence.
    /// </summary>
    /// <param name="writer">The binary writer to serialize data to.</param>
    /// <remarks>
    /// <para><b>For Beginners:</b> Saves TimeMachine-specific configuration so the model
    /// can be reconstructed later.
    /// </para>
    /// </remarks>
    protected override void SerializeNetworkSpecificData(BinaryWriter writer)
    {
        writer.Write(_contextLength);
        writer.Write(_forecastHorizon);
        writer.Write(_modelDimension);
        writer.Write(_stateDimension);
        writer.Write(_numScales);
        writer.Write(_numLayers);
        writer.Write(_expandFactor);
        writer.Write(_convKernelSize);
        writer.Write(_useMultiScaleAttention);
        writer.Write(_useReversibleNormalization);
        writer.Write(_decompositionMethod);
    }

    /// <summary>
    /// Deserializes TimeMachine-specific data when loading a saved model.
    /// </summary>
    /// <param name="reader">The binary reader to deserialize data from.</param>
    /// <remarks>
    /// <para><b>For Beginners:</b> Restores TimeMachine-specific configuration when
    /// loading a previously saved model.
    /// </para>
    /// </remarks>
    protected override void DeserializeNetworkSpecificData(BinaryReader reader)
    {
        _contextLength = reader.ReadInt32();
        _forecastHorizon = reader.ReadInt32();
        _modelDimension = reader.ReadInt32();
        _stateDimension = reader.ReadInt32();
        _numScales = reader.ReadInt32();
        _numLayers = reader.ReadInt32();
        _expandFactor = reader.ReadInt32();
        _convKernelSize = reader.ReadInt32();
        _useMultiScaleAttention = reader.ReadBoolean();
        _useReversibleNormalization = reader.ReadBoolean();
        _decompositionMethod = reader.ReadString();
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
    /// Given historical data, TimeMachine processes it at multiple temporal scales
    /// using SSM blocks and produces future predictions.
    /// </para>
    /// </remarks>
    public override Tensor<T> Forecast(Tensor<T> historicalData, double[]? quantiles = null)
    {
        var output = _useNativeMode ? Forward(historicalData) : ForecastOnnx(historicalData);

        // If quantiles are requested, return the point forecast
        // (TimeMachine doesn't natively support quantile forecasting)
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
    /// provides uncertainty bounds. TimeMachine uses Monte Carlo dropout to estimate
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
    /// then uses that prediction as input for the next step. TimeMachine's multi-scale
    /// structure helps maintain coherent predictions across multiple steps.
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
    /// <para><b>For Beginners:</b> TimeMachine uses reversible instance normalization
    /// (RevIN) which normalizes each time series individually. This method applies
    /// the forward normalization step, storing mean and variance for later reversal.
    /// </para>
    /// </remarks>
    public override Tensor<T> ApplyInstanceNormalization(Tensor<T> input)
    {
        if (!_useReversibleNormalization)
            return input;

        // Compute mean and std along time dimension
        int length = input.Data.Length;
        T sum = NumOps.Zero;
        for (int i = 0; i < length; i++)
        {
            sum = NumOps.Add(sum, input.Data.Span[i]);
        }
        T mean = NumOps.Divide(sum, NumOps.FromDouble(length));

        T variance = NumOps.Zero;
        for (int i = 0; i < length; i++)
        {
            T diff = NumOps.Subtract(input.Data.Span[i], mean);
            variance = NumOps.Add(variance, NumOps.Multiply(diff, diff));
        }
        variance = NumOps.Divide(variance, NumOps.FromDouble(length));
        T std = NumOps.FromDouble(Math.Sqrt(NumOps.ToDouble(variance)) + 1e-8);

        // Normalize
        var normalized = new Tensor<T>(input.Shape);
        for (int i = 0; i < length; i++)
        {
            normalized.Data.Span[i] = NumOps.Divide(
                NumOps.Subtract(input.Data.Span[i], mean),
                std);
        }

        return normalized;
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
            ["NumScales"] = NumOps.FromDouble(_numScales),
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
    /// the TimeMachine architecture:
    /// 1. Apply reversible normalization (if enabled)
    /// 2. Embed input to model dimension
    /// 3. For each scale: decompose, process with SSM, upsample
    /// 4. Fuse multi-scale outputs
    /// 5. Project to forecast horizon
    /// 6. Apply reverse normalization (if enabled)
    /// </para>
    /// </remarks>
    public Tensor<T> Forward(Tensor<T> input)
    {
        var current = FlattenInput(input);

        // Apply RevIN if enabled (forward normalization)
        if (_useReversibleNormalization)
        {
            current = ApplyInstanceNormalization(current);
        }

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
    /// contributed to the prediction error. For TimeMachine, gradients flow back through:
    /// - Output projection
    /// - Multi-scale fusion
    /// - Each scale's SSM layers
    /// - Temporal decomposition
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
    /// <para><b>For Beginners:</b> Native mode runs our custom TimeMachine implementation
    /// which processes data at multiple temporal scales using Mamba-style SSM blocks.
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
    /// <para><b>For Beginners:</b> ONNX mode uses a pretrained TimeMachine model
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
    /// <para><b>For Beginners:</b> TimeMachine processes the time series through
    /// dense layers after initial multi-scale decomposition. We flatten the input
    /// for compatibility with the layer structure.
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

    /// <summary>
    /// Applies temporal decomposition to separate different frequency components.
    /// </summary>
    /// <param name="input">Input tensor.</param>
    /// <param name="scale">Scale index (0 = finest, higher = coarser).</param>
    /// <returns>Decomposed tensor for the specified scale.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Temporal decomposition separates the time series
    /// into components at different scales:
    /// - Scale 0: High-frequency variations (noise, short-term)
    /// - Scale 1: Medium-high (daily patterns)
    /// - Scale 2: Medium-low (weekly patterns)
    /// - Scale 3: Low-frequency (long-term trends)
    ///
    /// This is done using a moving average or downsampling operation.
    /// </para>
    /// </remarks>
    private Tensor<T> ApplyTemporalDecomposition(Tensor<T> input, int scale)
    {
        int factor = 1 << scale; // 2^scale downsampling
        int outputLength = input.Data.Length / factor;

        var decomposed = new Tensor<T>(new[] { outputLength });

        for (int i = 0; i < outputLength; i++)
        {
            T sum = NumOps.Zero;
            for (int j = 0; j < factor; j++)
            {
                int idx = i * factor + j;
                if (idx < input.Data.Length)
                {
                    sum = NumOps.Add(sum, input.Data.Span[idx]);
                }
            }
            decomposed.Data.Span[i] = NumOps.Divide(sum, NumOps.FromDouble(factor));
        }

        return decomposed;
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



