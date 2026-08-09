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
/// TimeMachine's four-Mamba architecture for long-term time series forecasting.
/// </summary>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
/// <remarks>
/// <para>
/// TimeMachine embeds the input length twice and uses two outer and two inner Mamba
/// branches in complementary channel/embedding orientations.
/// </para>
/// <para><b>For Beginners:</b> TimeMachine is a modern architecture whose key insight is
/// that "A Time Series is Worth 4 Mambas." After RevIN, E1 embeds the history length.
/// Two outer Mambas process that representation in complementary orientations. E2 then
/// creates a smaller representation for two inner Mambas. Residual projection P1 joins
/// the inner path, its result is concatenated with the outer path, and P2 produces the forecast.
///
/// <b>Key Benefits:</b>
/// - Linear complexity O(n) from SSM backbone
/// - Complementary channel/embedding scans capture cross-variate and temporal structure
/// - RevIN handles non-stationarity
/// - State-of-the-art results on long-term forecasting benchmarks
/// </para>
/// <para>
/// <b>Reference:</b> Ahamed et al., "TimeMachine: A Time Series is Worth 4 Mambas for Long-term Forecasting", 2024.
/// https://arxiv.org/abs/2403.09898
/// </para>
/// </remarks>
/// <example>
/// <code>
/// var architecture = new NeuralNetworkArchitecture&lt;double&gt;(
///     inputType: InputType.OneDimensional,
///     taskType: NeuralNetworkTaskType.Regression,
///     inputHeight: 512, inputWidth: 7, inputDepth: 1, outputSize: 96);
/// var model = new TimeMachine&lt;double&gt;(architecture);
/// var onnxModel = new TimeMachine&lt;double&gt;(architecture, "timemachine.onnx");
/// </code>
/// </example>
[ModelDomain(ModelDomain.Finance)]
[ModelDomain(ModelDomain.TimeSeries)]
[ModelCategory(ModelCategory.NeuralNetwork)]
[ModelCategory(ModelCategory.RecurrentNetwork)]
[ModelTask(ModelTask.Forecasting)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("TimeMachine: A Time Series is Worth 4 Mambas for Long-term Forecasting", "https://arxiv.org/abs/2403.09898")]
public class TimeMachine<T> : ForecastingModelBase<T>
{
    #region Execution Mode
    private bool _useNativeMode;
    #endregion


    #region Native Mode Fields
    private DenseLayer<T>? _firstEmbedding;
    private DropoutLayer<T>? _firstDropout;
    private MambaBlock<T>? _outerChannelMamba;
    private MambaBlock<T>? _outerEmbeddingMamba;
    private DenseLayer<T>? _secondEmbedding;
    private DropoutLayer<T>? _secondDropout;
    private MambaBlock<T>? _innerEmbeddingMamba;
    private MambaBlock<T>? _innerChannelMamba;
    private DenseLayer<T>? _residualProjection;
    private DenseLayer<T>? _outputProjection;
    private bool _usesDefaultArchitecture;
    #endregion

    #region Shared Fields
    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> _optimizer;
    private readonly ILossFunction<T> _lossFunction;
    private readonly TimeMachineOptions<T> _options;

    /// <inheritdoc/>
    protected override IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? TrainingOptimizer => _optimizer;

    /// <inheritdoc/>
    public override ModelOptions GetOptions() => _options;

    private int _contextLength;
    private int _forecastHorizon;
    private int _modelDimension;
    private int _stateDimension;
    private int _expandFactor;
    private int _convKernelSize;
    private bool _useReversibleNormalization;

    /// <summary>
    /// Variance floor for TimeMachine's reversible instance normalization.
    /// </summary>
    /// <remarks>
    /// Kept at the 1e-8 the original hand-rolled implementation used, so the floor's magnitude is
    /// unchanged. It is now applied as sqrt(var + eps) rather than sqrt(var) + eps -- the standard
    /// RevIN form (Kim et al. 2022), which is better conditioned on a constant series.
    /// </remarks>
    private const double TimeMachineRevInEpsilon = 1e-8;
    private int _numFeatures;

    // RevIN reverse-step statistics (Kim et al. 2022). ApplyInstanceNormalization
    // stores the instance mean/std here so the forecast can be restored to the
    // input's original scale; FlattenInput collapses the input to a single
    // instance, so one (mean, std) pair suffices.
    private Vector<T> _revinMean = new Vector<T>(0);
    private Vector<T> _revinStd = new Vector<T>(0);
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
    /// The linear-time Mamba branches efficiently handle long contexts.
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

    // REMOVED: NumScales, NumLayers, UseMultiScaleAttention and the decomposition-method label.
    //
    // All four were public, settable, documented and read by nothing. TimeMachine's graph is fixed by
    // the paper -- exactly four Mambas in a two-outer/two-inner arrangement, combined by addition and
    // concatenation rather than attention -- so a caller who set any of them configured a model and
    // got the same one back, with no error and no way to notice. Keeping a mutable setting that
    // silently does nothing is worse than a compile error telling the caller it is gone, which is what
    // they now get. See the migration note on TimeMachineOptions<T>.

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
        Options = _options;
        _lossFunction = lossFunction ?? new MeanSquaredErrorLoss<T>();
        _optimizer = optimizer ?? new AdamOptimizer<T, Tensor<T>, Tensor<T>>(this);

        _contextLength = _options.ContextLength;
        _forecastHorizon = _options.ForecastHorizon;
        _modelDimension = _options.ModelDimension;
        _stateDimension = _options.StateDimension;
        _expandFactor = _options.ExpandFactor;
        _convKernelSize = _options.ConvKernelSize;
        _useReversibleNormalization = _options.UseReversibleNormalization;
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
    /// that can be trained on your data. The model uses four Mamba branches in the
    /// paper's outer/inner arrangement.
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
        Options = _options;
        _lossFunction = lossFunction ?? new MeanSquaredErrorLoss<T>();
        _optimizer = optimizer ?? new AdamOptimizer<T, Tensor<T>, Tensor<T>>(this);

        _contextLength = _options.ContextLength;
        _forecastHorizon = _options.ForecastHorizon;
        _modelDimension = _options.ModelDimension;
        _stateDimension = _options.StateDimension;
        _expandFactor = _options.ExpandFactor;
        _convKernelSize = _options.ConvKernelSize;
        _useReversibleNormalization = _options.UseReversibleNormalization;
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
    /// that implement TimeMachine's four-Mamba architecture:
    ///
    /// <b>Layer Structure:</b>
    /// 1. E1 and dropout
    /// 2. Two outer Mamba branches
    /// 3. E2, dropout, and two inner Mamba branches
    /// 4. P1 residual, branch concatenation, and P2 output projection
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
                _expandFactor,
                _convKernelSize,
                _numFeatures,
                _options.DropoutRate));

            ExtractLayerReferences();
        }
    }

    /// <summary>
    /// Extracts references to key layers for efficient access.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> After creating all layers, we keep direct references
    /// to important ones for quick access during computation. This includes the input
    /// two embeddings, four Mamba branches, and the two output projections.
    /// </para>
    /// </remarks>
    private void ExtractLayerReferences()
    {
        // Ahamed et al. §3 / the official implementation use this exact graph:
        // E1 -> two outer Mambas -> E2 -> two inner Mambas -> P1/residual -> concat -> P2.
        // Keep the assertion structural so a flat Dense substitute cannot silently satisfy
        // the model contract again.
        if (Layers.Count != 10
            || Layers[0] is not DenseLayer<T> firstEmbedding
            || Layers[1] is not DropoutLayer<T> firstDropout
            || Layers[2] is not MambaBlock<T> outerChannel
            || Layers[3] is not MambaBlock<T> outerEmbedding
            || Layers[4] is not DenseLayer<T> secondEmbedding
            || Layers[5] is not DropoutLayer<T> secondDropout
            || Layers[6] is not MambaBlock<T> innerEmbedding
            || Layers[7] is not MambaBlock<T> innerChannel
            || Layers[8] is not DenseLayer<T> residualProjection
            || Layers[9] is not DenseLayer<T> outputProjection)
        {
            throw new InvalidOperationException(
                "The default TimeMachine architecture must contain E1, dropout, four Mamba blocks, " +
                "E2, dropout, P1, and P2 in the paper-defined order.");
        }

        int secondEmbeddingDimension = Math.Max(1, _modelDimension / 2);
        if (firstEmbedding.GetOutputShape()[0] != _modelDimension
            || secondEmbedding.GetOutputShape()[0] != secondEmbeddingDimension
            || residualProjection.GetOutputShape()[0] != _modelDimension
            || outputProjection.GetOutputShape()[0] != _forecastHorizon
            || outerChannel.ModelDimension != _modelDimension
            || outerEmbedding.ModelDimension != 1
            || innerEmbedding.ModelDimension != 1
            || innerChannel.ModelDimension != secondEmbeddingDimension)
        {
            throw new InvalidOperationException(
                "The default TimeMachine layer dimensions do not match the paper's four-branch architecture.");
        }

        _firstEmbedding = firstEmbedding;
        _firstDropout = firstDropout;
        _outerChannelMamba = outerChannel;
        _outerEmbeddingMamba = outerEmbedding;
        _secondEmbedding = secondEmbedding;
        _secondDropout = secondDropout;
        _innerEmbeddingMamba = innerEmbedding;
        _innerChannelMamba = innerChannel;
        _residualProjection = residualProjection;
        _outputProjection = outputProjection;
        _usesDefaultArchitecture = true;
    }

    /// <summary>
    /// Validates custom layers provided through the architecture.
    /// </summary>
    /// <param name="layers">The list of custom layers to validate.</param>
    /// <remarks>
    /// <para><b>For Beginners:</b> When users provide custom layers, this method
    /// ensures the supplied sequential architecture has enough layers to be useful.
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
    /// In native mode, it runs the four-Mamba graph or the user's custom layer stack.
    /// </para>
    /// </remarks>
    protected override Tensor<T> PredictCore(Tensor<T> input)
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
    /// backpropagation. The outer and inner Mamba branches learn complementary
    /// channel-axis and embedding-axis dynamics before their residual/concatenation head.
    ///
    /// Only available in native mode (not ONNX).
    /// </para>
    /// </remarks>
    public override void Train(Tensor<T> input, Tensor<T> target)
    {
        if (!_useNativeMode)
            throw new InvalidOperationException("Training is only supported in native mode.");

        // Issue #1166: the old body computed a loss + gradient and then
        // called _optimizer.UpdateParameters(Layers) without a backward
        // pass, so every layer's UpdateParameters threw "Backward pass
        // must be called before updating parameters." Delegate to
        // FinancialModelBase.Train — it routes through the tape-based
        // NeuralNetworkBase.TrainWithTape flow (GradientTape forward +
        // tape.ComputeGradients + optimizer.Step) that every other
        // NeuralNetworkBase subclass uses.
        base.Train(input, target);
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
            AdditionalInfo = new Dictionary<string, object>
            {
                { "NetworkType", "TimeMachine" },
                { "ContextLength", _contextLength },
                { "ForecastHorizon", _forecastHorizon },
                { "ModelDimension", _modelDimension },
                { "StateDimension", _stateDimension },
                { "ExpandFactor", _expandFactor },
                { "ConvKernelSize", _convKernelSize },
                { "UseReversibleNormalization", _useReversibleNormalization },
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
        return new TimeMachine<T>(Architecture, new TimeMachineOptions<T>(_options), _numFeatures);
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
        writer.Write(_expandFactor);
        writer.Write(_convKernelSize);
        writer.Write(_useReversibleNormalization);
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
        _expandFactor = reader.ReadInt32();
        _convKernelSize = reader.ReadInt32();
        _useReversibleNormalization = reader.ReadBoolean();
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
    /// Given historical data, TimeMachine processes it through its four Mamba branches
    /// and produces future predictions.
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
    /// then uses that prediction as input for the next step. TimeMachine's residual
    /// branch structure helps maintain coherent predictions across multiple steps.
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

        // The TimeMachine reference applies RevIN independently to each [batch, channel]
        // series over the time axis. For its native [B, L, M] layout, transpose to
        // [B, M, L], collapse B*M to rows, and use the shared tape-aware normalization.
        if (_usesDefaultArchitecture && input.Rank == 3)
        {
            int batch = input.Shape[0];
            int context = input.Shape[1];
            int features = input.Shape[2];
            var channelMajor = Engine.TensorPermute(input, new[] { 0, 2, 1 });
            var rows = Engine.Reshape(channelMajor, new[] { batch * features, context });
            var normalizedRows = NormalizeInstanceOnTape(
                rows, TimeMachineRevInEpsilon, out _revinMean, out _revinStd);
            var normalizedChannelMajor = Engine.Reshape(
                normalizedRows, new[] { batch, features, context });
            return Engine.TensorPermute(normalizedChannelMajor, new[] { 0, 2, 1 });
        }

        // Custom layer stacks retain the historical single-instance convention.
        var oneRow = Engine.Reshape(input, new[] { 1, input.Length });
        var normalized = NormalizeInstanceOnTape(
            oneRow, TimeMachineRevInEpsilon, out _revinMean, out _revinStd);
        return Engine.Reshape(normalized, (int[])input._shape.Clone());
    }

    /// <summary>
    /// RevIN reverse step (Kim et al. 2022): restores the instance mean/std stored
    /// by <see cref="ApplyInstanceNormalization"/> to the forecast so it is
    /// expressed on the input's original scale. The multiply/add go through the
    /// Engine so the forecast stays on the autodiff tape.
    /// </summary>
    private Tensor<T> DenormalizeForecast(Tensor<T> forecast)
    {
        if (_usesDefaultArchitecture && forecast.Rank == 3)
        {
            int batch = forecast.Shape[0];
            int horizon = forecast.Shape[1];
            int features = forecast.Shape[2];
            int rows = batch * features;
            if (_revinMean.Length != rows || _revinStd.Length != rows)
                return forecast;

            var channelMean = new Tensor<T>(new[] { rows, 1 });
            var channelStd = new Tensor<T>(new[] { rows, 1 });
            for (int row = 0; row < rows; row++)
            {
                channelMean.Data.Span[row] = _revinMean[row];
                channelStd.Data.Span[row] = _revinStd[row];
            }

            var channelMajor = Engine.TensorPermute(forecast, new[] { 0, 2, 1 });
            var flat = Engine.Reshape(channelMajor, new[] { rows, horizon });
            var channelScaled = Engine.TensorBroadcastMultiply(flat, channelStd);
            var channelShifted = Engine.TensorBroadcastAdd(channelScaled, channelMean);
            var restoredChannelMajor = Engine.Reshape(channelShifted, new[] { batch, features, horizon });
            return Engine.TensorPermute(restoredChannelMajor, new[] { 0, 2, 1 });
        }

        if (_revinMean.Length != 1 || _revinStd.Length != 1)
            return forecast;

        var meanT = new Tensor<T>(new[] { 1, 1 }) ;
        var stdT = new Tensor<T>(new[] { 1, 1 });
        meanT.Data.Span[0] = _revinMean[0];
        stdT.Data.Span[0] = _revinStd[0];

        bool reshaped = forecast.Rank != 2;
        var work = reshaped ? Engine.Reshape(forecast, new[] { 1, forecast.Length }) : forecast;
        var scaled = Engine.TensorBroadcastMultiply(work, stdT);
        var shifted = Engine.TensorBroadcastAdd(scaled, meanT);
        return reshaped ? Engine.Reshape(shifted, forecast._shape) : shifted;
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
            ["ExpandFactor"] = NumOps.FromDouble(_expandFactor)
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
    /// 3. Run the outer and inner Mamba pairs with their residuals
    /// 4. Concatenate the branch results and project to the forecast horizon
    /// 5. Apply reverse normalization (if enabled)
    /// </para>
    /// </remarks>
    public Tensor<T> Forward(Tensor<T> input)
    {
        return _usesDefaultArchitecture
            ? ForwardDefaultArchitecture(input, null)
            : ForwardCustomArchitecture(input);
    }

    /// <summary>
    /// Runs a user-provided layer list as the sequential graph supplied by the user.
    /// </summary>
    private Tensor<T> ForwardCustomArchitecture(Tensor<T> input)
    {
        var current = FlattenInput(input);

        // Apply RevIN if enabled (forward normalization)
        if (_useReversibleNormalization)
        {
            current = ApplyInstanceNormalization(current);
        }

        // Add a leading batch axis so the embedding ReshapeLayer tokenizes the
        // flattened context into per-timestep tokens [1, contextLength,
        // numFeatures] instead of misreading the contextLength vector as a batch.
        bool addedBatchDim = current.Rank == 1;
        if (addedBatchDim)
        {
            // Issue #1670: use Engine.Reshape, NOT tensor.Reshape. The raw
            // Tensor<T>.Reshape allocates a fresh tensor with no autodiff-tape
            // link, severing the gradient path. During training this Forward is
            // driven under a tape (ForwardNativeForTraining), so a raw reshape
            // here — and especially the output reshape below — disconnects every
            // layer parameter from the loss, making all gradients zero and
            // training a no-op. Engine.Reshape records the op on the tape.
            current = Engine.Reshape(current, new[] { 1, current.Length });
        }

        foreach (var layer in Layers)
        {
            current = layer.Forward(current);
        }

        // Apply RevIN reverse step so the forecast is on the input's scale.
        if (_useReversibleNormalization)
        {
            current = DenormalizeForecast(current);
        }

        if (addedBatchDim && current.Rank == 2 && current.Shape[0] == 1)
        {
            // Tape-safe reshape (see note above) — this runs on the final output
            // after all layers, so a raw reshape here is the primary tape-sever
            // that zeroed gradients for #1670.
            current = Engine.Reshape(current, new[] { current.Shape[1] });
        }

        return current;
    }

    /// <summary>
    /// Runs the four-Mamba graph from Ahamed et al. (2024) and the official implementation.
    /// </summary>
    private Tensor<T> ForwardDefaultArchitecture(
        Tensor<T> input,
        Dictionary<string, Tensor<T>>? activations)
    {
        if (_firstEmbedding is null || _firstDropout is null
            || _outerChannelMamba is null || _outerEmbeddingMamba is null
            || _secondEmbedding is null || _secondDropout is null
            || _innerEmbeddingMamba is null || _innerChannelMamba is null
            || _residualProjection is null || _outputProjection is null)
        {
            throw new InvalidOperationException("The default TimeMachine layers have not been initialized.");
        }

        var paperInput = PreparePaperInput(input, out var inputLayout);
        int batch = paperInput.Shape[0];
        int context = paperInput.Shape[1];
        int features = paperInput.Shape[2];

        if (_useReversibleNormalization)
            paperInput = ApplyInstanceNormalization(paperInput);

        // [B, L, M] -> [B*M, 1, L], matching ch_ind=1 in official TimeMachine.py.
        var channelMajor = Engine.TensorPermute(paperInput, new[] { 0, 2, 1 });
        var current = Engine.Reshape(channelMajor, new[] { batch * features, 1, context });

        var firstEmbedding = _firstEmbedding.Forward(current);                  // E1: L -> n1
        CaptureActivation(activations, 0, firstEmbedding);
        var firstResidual = firstEmbedding;
        current = _firstDropout.Forward(firstEmbedding);
        CaptureActivation(activations, 1, current);

        // Outer pair. Mamba 3 consumes [B*M, 1, n1]; Mamba 4 consumes its
        // transposition [B*M, n1, 1]. Their outputs are restored and summed.
        // AiDotNet's reusable MambaBlock wraps its mixer in an internal residual;
        // mamba_ssm.Mamba in the TimeMachine reference is the bare mixer. Remove that
        // wrapper residual here, then apply only the paper's explicit graph residuals.
        var outerChannel = Engine.TensorSubtract(_outerChannelMamba.Forward(current), current);
        CaptureActivation(activations, 2, outerChannel);
        var outerEmbeddingInput = Engine.TensorPermute(current, new[] { 0, 2, 1 });
        var outerEmbedding = Engine.TensorSubtract(
            _outerEmbeddingMamba.Forward(outerEmbeddingInput), outerEmbeddingInput);
        CaptureActivation(activations, 3, outerEmbedding);
        outerEmbedding = Engine.TensorPermute(outerEmbedding, new[] { 0, 2, 1 });
        var outerCombined = Engine.TensorAdd(outerChannel, outerEmbedding);

        current = _secondEmbedding.Forward(current);                            // E2: n1 -> n2
        CaptureActivation(activations, 4, current);
        var secondResidual = current;
        current = _secondDropout.Forward(current);
        CaptureActivation(activations, 5, current);

        // Inner pair, again using complementary channel/embedding orientations.
        var innerEmbeddingInput = Engine.TensorPermute(current, new[] { 0, 2, 1 });
        var innerEmbedding = Engine.TensorSubtract(
            _innerEmbeddingMamba.Forward(innerEmbeddingInput), innerEmbeddingInput);
        CaptureActivation(activations, 6, innerEmbedding);
        innerEmbedding = Engine.TensorPermute(innerEmbedding, new[] { 0, 2, 1 });
        var innerChannel = Engine.TensorSubtract(_innerChannelMamba.Forward(current), current);
        CaptureActivation(activations, 7, innerChannel);
        current = Engine.TensorAdd(Engine.TensorAdd(innerEmbedding, secondResidual), innerChannel);

        current = _residualProjection.Forward(current);                         // P1: n2 -> n1
        CaptureActivation(activations, 8, current);
        current = Engine.TensorAdd(current, firstResidual);

        // Concatenate the inner/residual path with the outer path, then P2 maps
        // 2*n1 directly to the prediction horizon.
        current = Engine.TensorConcatenate(new[] { current, outerCombined }, axis: 2);
        current = _outputProjection.Forward(current);
        CaptureActivation(activations, 9, current);

        var forecastChannelMajor = Engine.Reshape(
            current, new[] { batch, features, _forecastHorizon });
        var forecast = Engine.TensorPermute(forecastChannelMajor, new[] { 0, 2, 1 });
        if (_useReversibleNormalization)
            forecast = DenormalizeForecast(forecast);

        return RestorePaperOutputLayout(forecast, inputLayout);
    }

    private void CaptureActivation(
        Dictionary<string, Tensor<T>>? activations,
        int layerIndex,
        Tensor<T> value)
    {
        if (activations is not null)
            activations[$"Layer_{layerIndex}_{Layers[layerIndex].GetType().Name}"] = value.Clone();
    }

    private enum PaperInputLayout
    {
        Flat,
        SequenceFeatures,
        BatchSequence,
        BatchSequenceFeatures
    }

    /// <summary>Converts supported public layouts to the paper's [B, L, M] layout.</summary>
    private Tensor<T> PreparePaperInput(Tensor<T> input, out PaperInputLayout layout)
    {
        if (input.Rank == 1)
        {
            int expected = _contextLength * _numFeatures;
            if (input.Length != expected)
                throw new ArgumentException(
                    $"TimeMachine expected {expected} input values ({_contextLength} steps x {_numFeatures} features), " +
                    $"but received {input.Length}.", nameof(input));
            layout = PaperInputLayout.Flat;
            return Engine.Reshape(input, new[] { 1, _contextLength, _numFeatures });
        }

        if (input.Rank == 2)
        {
            if (input.Shape[0] == _contextLength && input.Shape[1] == _numFeatures)
            {
                layout = PaperInputLayout.SequenceFeatures;
                return Engine.Reshape(input, new[] { 1, _contextLength, _numFeatures });
            }

            if (_numFeatures == 1 && input.Shape[1] == _contextLength)
            {
                layout = PaperInputLayout.BatchSequence;
                return Engine.Reshape(input, new[] { input.Shape[0], _contextLength, 1 });
            }
        }

        if (input.Rank == 3
            && input.Shape[1] == _contextLength
            && input.Shape[2] == _numFeatures)
        {
            layout = PaperInputLayout.BatchSequenceFeatures;
            return input;
        }

        throw new ArgumentException(
            $"TimeMachine expects [L*M], [L,M], [B,L] for univariate data, or [B,L,M] with " +
            $"L={_contextLength} and M={_numFeatures}; received [{string.Join(", ", input.Shape)}].",
            nameof(input));
    }

    /// <summary>Restores the rank convention used by the caller.</summary>
    private Tensor<T> RestorePaperOutputLayout(Tensor<T> forecast, PaperInputLayout layout)
    {
        return layout switch
        {
            PaperInputLayout.Flat => Engine.Reshape(forecast, new[] { forecast.Length }),
            PaperInputLayout.SequenceFeatures => Engine.Reshape(
                forecast, new[] { _forecastHorizon, _numFeatures }),
            PaperInputLayout.BatchSequence => Engine.Reshape(
                forecast, new[] { forecast.Shape[0], _forecastHorizon }),
            _ => forecast
        };
    }

    /// <summary>
    /// Training-mode forward. Routes through <see cref="Forward"/> so training uses
    /// the same RevIN normalize/denormalize as inference (and keeps training mode
    /// active for dropout), instead of the base default that flips to inference.
    /// </summary>
    protected override Tensor<T> ForwardNativeForTraining(Tensor<T> input)
    {
        return Forward(input);
    }

    /// <inheritdoc/>
    /// <remarks>
    /// Mirrors <see cref="Forward"/>'s preprocessing so the captured activations match the
    /// real forward pass: the input is flattened, RevIN-normalized and given a leading batch
    /// axis before it reaches the embedding <c>ReshapeLayer</c> (which would otherwise misread
    /// the flattened context vector as a multi-row batch).
    /// </remarks>
    public override Dictionary<string, Tensor<T>> GetNamedLayerActivations(Tensor<T> input)
    {
        if (!_useNativeMode)
            return base.GetNamedLayerActivations(input);

        var activations = new Dictionary<string, Tensor<T>>();

        if (_usesDefaultArchitecture)
        {
            ForwardDefaultArchitecture(input, activations);
            return activations;
        }

        var current = FlattenInput(input);
        if (_useReversibleNormalization)
            current = ApplyInstanceNormalization(current);
        if (current.Rank == 1)
            current = current.Reshape(new[] { 1, current.Length });

        for (int i = 0; i < Layers.Count; i++)
        {
            current = Layers[i].Forward(current);
            activations[$"Layer_{i}_{Layers[i].GetType().Name}"] = current.Clone();
        }

        return activations;
    }

    /// <summary>
    /// Performs native mode forecasting through the layer stack.
    /// </summary>
    /// <param name="input">Input tensor.</param>
    /// <returns>Forecast tensor.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Native mode runs our custom TimeMachine implementation
    /// which processes data through the paper's four Mamba branches.
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
    /// <para><b>For Beginners:</b> User-provided layer stacks retain the historical
    /// flat sequential input convention. The paper-faithful default path does not use this helper.
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
        // .Data demands a contiguous buffer, and neither argument is guaranteed to be
        // one: AutoregressiveForecast feeds this the output of the previous step, which
        // by then can be a sliced or transposed VIEW. Reading .Data on such a view throws
        // "Cannot get contiguous Memory from a non-contiguous tensor view" -- the failure
        // that ended TimeMachine quantile forecasting for both precisions. Materialise
        // once, up front, rather than at each of the four .Data reads below.
        var source = input.IsContiguous ? input : input.Contiguous();
        var predicted = prediction.IsContiguous ? prediction : prediction.Contiguous();

        int inputLength = source.Data.Length;
        int predLength = Math.Min(predicted.Data.Length, inputLength);

        var shifted = new Tensor<T>(source._shape);

        // Copy shifted values (skip first predLength values)
        for (int i = predLength; i < inputLength; i++)
        {
            shifted.Data.Span[i - predLength] = source.Data.Span[i];
        }

        // Append prediction values at the end
        for (int i = 0; i < predLength; i++)
        {
            shifted.Data.Span[inputLength - predLength + i] = predicted.Data.Span[i];
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

        // Same contiguity requirement as ShiftInputWindow above: these predictions come
        // straight from the forecast loop and can be views, and .Data throws on a view.
        // Materialise each ONCE here rather than at the three reads below, which would
        // otherwise rebuild the same buffer per element.
        var materialised = new List<Tensor<T>>(predictions.Count);
        foreach (var pred in predictions)
        {
            materialised.Add(pred.IsContiguous ? pred : pred.Contiguous());
        }

        int totalLength = 0;
        foreach (var pred in materialised)
        {
            totalLength += pred.Data.Length;
        }

        var result = new Tensor<T>(new[] { totalLength });
        int offset = 0;

        foreach (var pred in materialised)
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



