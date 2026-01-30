using System.IO;
using AiDotNet.Enums;
using AiDotNet.Finance.Base;
using AiDotNet.Finance.Interfaces;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LossFunctions;
using AiDotNet.Models;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Optimizers;
using AiDotNet.Tensors.Helpers;
using Microsoft.ML.OnnxRuntime;
using OnnxTensors = Microsoft.ML.OnnxRuntime.Tensors;

namespace AiDotNet.Finance.Forecasting.Neural;

/// <summary>
/// DeepAR probabilistic autoregressive forecasting model using LSTM networks.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (typically float or double).</typeparam>
/// <remarks>
/// <para>
/// DeepAR is a probabilistic forecasting model that produces forecast distributions rather than
/// point predictions. It uses autoregressive recurrent neural networks to learn temporal patterns
/// and outputs distribution parameters (e.g., mean and standard deviation for Gaussian).
/// </para>
/// <para>
/// <b>For Beginners:</b> DeepAR is special because it doesn't just predict a single value - it
/// predicts a probability distribution. This means you get:
/// - A most likely value (the mean)
/// - A measure of confidence (the standard deviation)
/// - The ability to generate prediction intervals (e.g., 95% confidence bounds)
///
/// Key features:
/// - <b>Autoregressive:</b> Each prediction depends on previous predictions
/// - <b>Probabilistic:</b> Outputs full distributions, not just point forecasts
/// - <b>Multi-series:</b> Can learn patterns across many related time series
/// - <b>Covariates:</b> Can include additional features like holidays or promotions
/// </para>
/// <para>
/// <b>Reference:</b> Salinas et al., "DeepAR: Probabilistic Forecasting with Autoregressive
/// Recurrent Networks", International Journal of Forecasting 2020.
/// https://arxiv.org/abs/1704.04110
/// </para>
/// </remarks>
public class DeepAR<T> : ForecastingModelBase<T>
{
    #region Native Mode Fields

    /// <summary>
    /// Input projection layer to prepare features for LSTM processing.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This layer transforms input features into a format
    /// suitable for the LSTM layers to process effectively.
    /// </para>
    /// </remarks>
    private ILayer<T>? _inputProjection;

    /// <summary>
    /// Stacked LSTM layers for sequence modeling.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> LSTM (Long Short-Term Memory) layers are recurrent neural networks
    /// that can learn patterns across time. They're good at remembering important information
    /// from the past while forgetting irrelevant details.
    /// </para>
    /// </remarks>
    private readonly List<ILayer<T>> _lstmLayers = [];

    /// <summary>
    /// Layer normalization for stable training.
    /// </summary>
    private ILayer<T>? _layerNorm;

    /// <summary>
    /// Output layer for distribution mean (mu).
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This layer outputs the predicted mean of the distribution.
    /// In a Gaussian distribution, this is the most likely value.
    /// </para>
    /// </remarks>
    private ILayer<T>? _muProjection;

    /// <summary>
    /// Output layer for distribution scale (sigma).
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This layer outputs the predicted standard deviation.
    /// Larger values mean more uncertainty in the prediction.
    /// </para>
    /// </remarks>
    private ILayer<T>? _sigmaProjection;

    /// <summary>
    /// The last computed sigma from Forward (used for quantile sampling).
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This stores the learned uncertainty from the most recent
    /// forward pass so SampleQuantiles can use the model's actual learned sigma
    /// rather than a hardcoded estimate.
    /// </para>
    /// </remarks>
    private Tensor<T>? _lastSigma;

    /// <summary>
    /// Instance normalization scale for denormalization.
    /// </summary>
    private Tensor<T>? _scaleStd;

    #endregion

    #region Shared Fields

    /// <summary>
    /// The optimizer for training.
    /// </summary>
    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> _optimizer;

    /// <summary>
    /// The hidden size of LSTM cells.
    /// </summary>
    private int _hiddenSize;

    /// <summary>
    /// The number of stacked LSTM layers.
    /// </summary>
    private int _numLstmLayers;

    /// <summary>
    /// Embedding dimension for categorical features.
    /// </summary>
    private int _embeddingDim;

    /// <summary>
    /// The dropout rate for regularization.
    /// </summary>
    private double _dropout;

    /// <summary>
    /// The output distribution type (gaussian, negative_binomial, student_t).
    /// </summary>
    private string _distributionType;

    /// <summary>
    /// Number of samples for Monte Carlo estimation.
    /// </summary>
    private int _numSamples;

    /// <summary>
    /// Whether to use scaling (dividing by mean absolute value).
    /// </summary>
    private bool _useScaling;

    /// <summary>
    /// Random number generator for sampling.
    /// </summary>
    private readonly Random _random;

    #endregion

    #region IForecastingModel Properties

    /// <summary>
    /// Gets the patch size for the model. DeepAR processes time steps sequentially via LSTM, so this is always 1.
    /// </summary>
    public override int PatchSize => 1;

    /// <summary>
    /// Gets the stride for the model. DeepAR processes every time step, so this is always 1.
    /// </summary>
    public override int Stride => 1;

    /// <summary>
    /// Gets whether the model processes channels independently. DeepAR processes all features together to learn correlations.
    /// </summary>
    public override bool IsChannelIndependent => false;

    #endregion

    #region Constructors

    /// <summary>
    /// Creates a DeepAR network using pretrained ONNX model.
    /// </summary>
    /// <param name="architecture">The neural network architecture configuration.</param>
    /// <param name="onnxModelPath">Path to the ONNX model file.</param>
    /// <param name="options">Configuration options for the model.</param>
    /// <param name="optimizer">Optional optimizer for fine-tuning.</param>
    /// <param name="lossFunction">Optional loss function.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Use this constructor when you have a pretrained ONNX model.
    /// ONNX models are pre-trained and ready to use for predictions immediately.
    /// </para>
    /// </remarks>
    public DeepAR(
        NeuralNetworkArchitecture<T> architecture,
        string onnxModelPath,
        DeepAROptions<T>? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
        ILossFunction<T>? lossFunction = null)
        : base(architecture, onnxModelPath, 
               options?.LookbackWindow ?? 96, 
               options?.ForecastHorizon ?? 24, 
               architecture.InputSize)
    {
        options ??= new DeepAROptions<T>();
        ValidateOptions(options);

        _optimizer = optimizer ?? new AdamOptimizer<T, Tensor<T>, Tensor<T>>(this);

        _hiddenSize = options.HiddenSize;
        _numLstmLayers = options.NumLayers;
        _embeddingDim = options.EmbeddingDimension;
        _dropout = options.DropoutRate;
        _distributionType = options.LikelihoodType;
        _numSamples = options.NumSamples;
        _useScaling = true;

        _random = RandomHelper.CreateSecureRandom();

        InitializeLayers();
    }

    /// <summary>
    /// Creates a DeepAR network in native mode for training from scratch.
    /// </summary>
    /// <param name="architecture">The neural network architecture configuration.</param>
    /// <param name="options">Configuration options for the model.</param>
    /// <param name="optimizer">Optional optimizer.</param>
    /// <param name="lossFunction">Optional loss function.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Use this constructor to train a new DeepAR model from scratch.
    /// DeepAR excels at:
    /// - Probabilistic forecasting (predicting uncertainty)
    /// - Multiple related time series (learning shared patterns)
    /// - Handling covariates (additional features like day of week)
    /// - Producing prediction intervals
    /// </para>
    /// </remarks>
    public DeepAR(
        NeuralNetworkArchitecture<T> architecture,
        DeepAROptions<T>? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
        ILossFunction<T>? lossFunction = null)
        : base(architecture, 
               options?.LookbackWindow ?? 96, 
               options?.ForecastHorizon ?? 24, 
               architecture.InputSize, 
               lossFunction)
    {
        options ??= new DeepAROptions<T>();
        ValidateOptions(options);

        _optimizer = optimizer ?? new AdamOptimizer<T, Tensor<T>, Tensor<T>>(this);

        _hiddenSize = options.HiddenSize;
        _numLstmLayers = options.NumLayers;
        _embeddingDim = options.EmbeddingDimension;
        _dropout = options.DropoutRate;
        _distributionType = options.LikelihoodType;
        _numSamples = options.NumSamples;
        _useScaling = true;

        _random = RandomHelper.CreateSecureRandom();

        InitializeLayers();
    }

    #endregion

    #region Initialization

    /// <summary>
    /// Initializes the neural network layers for DeepAR.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> DeepAR has several specialized components:
    /// </para>
    /// <para>
    /// <list type="number">
    /// <item><b>Input Projection:</b> Prepares input features for LSTM processing</item>
    /// <item><b>Stacked LSTM Layers:</b> Learn temporal dependencies in the data</item>
    /// <item><b>Layer Normalization:</b> Stabilizes training by normalizing activations</item>
    /// <item><b>Distribution Heads:</b> Output mu (mean) and sigma (std deviation) for the forecast distribution</item>
    /// </list>
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
        else if (UseNativeMode)
        {
            Layers.AddRange(LayerHelper<T>.CreateDefaultDeepARLayers(
                Architecture, SequenceLength, PredictionHorizon, NumFeatures,
                _hiddenSize, _numLstmLayers, _embeddingDim, _dropout));

            ExtractLayerReferences();
        }
    }

    /// <summary>
    /// Extracts references to specific layers from the layer collection.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> DeepAR has multiple layer types that need to be accessed
    /// during the forward pass. This organizes them for efficient access.
    /// </para>
    /// </remarks>
    private void ExtractLayerReferences()
    {
        int idx = 0;

        // Input projection
        if (Layers.Count > idx)
            _inputProjection = Layers[idx++];

        // LSTM layers
        _lstmLayers.Clear();
        for (int i = 0; i < _numLstmLayers && idx < Layers.Count; i++)
        {
            _lstmLayers.Add(Layers[idx++]);

            // Skip dropout layers between LSTMs
            if (i < _numLstmLayers - 1 && idx < Layers.Count &&
                Layers[idx] is DropoutLayer<T>)
            {
                _lstmLayers.Add(Layers[idx++]);
            }
        }

        // Layer normalization
        if (idx < Layers.Count)
            _layerNorm = Layers[idx++];

        // Distribution heads (mu and sigma)
        if (idx < Layers.Count)
            _muProjection = Layers[idx++];
        if (idx < Layers.Count)
            _sigmaProjection = Layers[idx++];
    }

    /// <summary>
    /// Validates that custom layers meet DeepAR's architectural requirements.
    /// </summary>
    /// <param name="layers">The list of custom layers to validate.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> DeepAR requires at least an LSTM layer and distribution
    /// output heads for proper probabilistic forecasting.
    /// </para>
    /// </remarks>
    protected override void ValidateCustomLayers(List<ILayer<T>> layers)
    {
        base.ValidateCustomLayers(layers);
        if (layers.Count < 3)
        {
            throw new ArgumentException(
                "DeepAR requires at least 3 layers: input projection, LSTM, and distribution output.",
                nameof(layers));
        }
    }

    /// <summary>
    /// Validates the DeepAR options.
    /// </summary>
    /// <param name="options">The options to validate.</param>
    /// <exception cref="ArgumentException">Thrown when options are invalid.</exception>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This ensures all configuration values are reasonable
    /// before attempting to build the model.
    /// </para>
    /// </remarks>
    private static void ValidateOptions(DeepAROptions<T> options)
    {
        var errors = new List<string>();

        if (options.LookbackWindow < 1)
            errors.Add("LookbackWindow must be at least 1.");
        if (options.ForecastHorizon < 1)
            errors.Add("ForecastHorizon must be at least 1.");
        if (options.HiddenSize < 1)
            errors.Add("HiddenSize must be at least 1.");
        if (options.NumLayers < 1)
            errors.Add("NumLayers must be at least 1.");
        if (options.DropoutRate < 0 || options.DropoutRate >= 1)
            errors.Add("DropoutRate must be in [0, 1).");
        if (options.NumSamples < 1)
            errors.Add("NumSamples must be at least 1.");

        var validDistributions = new[] { "Gaussian", "gaussian", "StudentT", "student_t", "NegativeBinomial", "negative_binomial" };
        if (!validDistributions.Contains(options.LikelihoodType, StringComparer.OrdinalIgnoreCase))
            errors.Add($"LikelihoodType must be one of: Gaussian, StudentT, NegativeBinomial.");

        if (errors.Count > 0)
            throw new ArgumentException($"Invalid options: {string.Join(", ", errors)}");
    }

    #endregion

    #region NeuralNetworkBase Overrides

    /// <summary>
    /// Trains the model on a single batch of input-output pairs.
    /// </summary>
    /// <param name="input">Input tensor.</param>
    /// <param name="target">Target tensor.</param>
    /// <param name="output">Model output from forward pass.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Training teaches the model to make better predictions by
    /// showing it examples of historical data and what actually happened next.
    /// </para>
    /// </remarks>
    protected override void TrainCore(Tensor<T> input, Tensor<T> target, Tensor<T> output)
    {
        SetTrainingMode(true);
        try
        {
            // Backward pass
            var gradient = ComputeGradient(output, target);
            Backward(gradient);

            // Update weights via optimizer
            _optimizer.UpdateParameters(Layers);
        }
        finally
        {
            SetTrainingMode(false);
        }
    }

    /// <summary>
    /// Updates the model's parameters using the provided gradients.
    /// </summary>
    /// <param name="gradients">Vector of parameter gradients.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method applies the calculated adjustments (gradients)
    /// to the model's weights. In DeepAR, this update is handled by the optimizer
    /// during the training step, so this specific override is a placeholder or used
    /// for manual parameter manipulation.
    /// </para>
    /// </remarks>
    public override void UpdateParameters(Vector<T> gradients)
    {
        // Parameters are updated through the optimizer in Train method
    }

    /// <summary>
    /// Gets metadata about the model for serialization and inspection.
    /// </summary>
    /// <returns>A ModelMetadata object containing model information.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This saves important details about the DeepAR model,
    /// such as the number of LSTM layers and the distribution type, so it can
    /// be correctly reloaded later.
    /// </para>
    /// </remarks>
    public override ModelMetadata<T> GetModelMetadata()
    {
        var metadata = base.GetModelMetadata();
        metadata.AdditionalInfo["NetworkType"] = "DeepAR";
        metadata.AdditionalInfo["HiddenSize"] = _hiddenSize;
        metadata.AdditionalInfo["NumLSTMLayers"] = _numLstmLayers;
        metadata.AdditionalInfo["DistributionType"] = _distributionType;
        metadata.AdditionalInfo["NumSamples"] = _numSamples;
        metadata.AdditionalInfo["UseScaling"] = _useScaling;
        metadata.AdditionalInfo["ParameterCount"] = GetParameterCount();

        return metadata;
    }

    /// <summary>
    /// Creates a new instance of this model with the same configuration.
    /// </summary>
    /// <returns>A new DeepAR model instance.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This creates a fresh copy of the model with the same settings
    /// but new (randomly initialized) weights. Useful for ensemble training or cross-validation.
    /// </para>
    /// </remarks>
    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance()
    {
        var options = new DeepAROptions<T>
        {
            LookbackWindow = SequenceLength,
            ForecastHorizon = PredictionHorizon,
            HiddenSize = _hiddenSize,
            NumLayers = _numLstmLayers,
            EmbeddingDimension = _embeddingDim,
            DropoutRate = _dropout,
            LikelihoodType = _distributionType,
            NumSamples = _numSamples
        };

        if (UseNativeMode)
        {
            return new DeepAR<T>(Architecture, options, _optimizer, LossFunction);
        }
        else
        {
            // Use null-coalescing throw to satisfy null analysis across all framework targets
            string onnxPath = OnnxModelPath ?? throw new InvalidOperationException(
                "Cannot create new instance from ONNX mode when OnnxModelPath is not available.");
            if (onnxPath.Length == 0)
            {
                throw new InvalidOperationException(
                    "Cannot create new instance from ONNX mode when OnnxModelPath is empty.");
            }
            return new DeepAR<T>(Architecture, onnxPath, options, _optimizer, LossFunction);
        }
    }

    /// <summary>
    /// Writes DeepAR-specific configuration during serialization.
    /// </summary>
    /// <param name="writer">Binary writer for output.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This saves DeepAR settings like hidden size and distribution type
    /// to a file so the model can be loaded later with the same configuration.
    /// </para>
    /// </remarks>
    protected override void SerializeModelSpecificData(BinaryWriter writer)
    {
        writer.Write(_hiddenSize);
        writer.Write(_numLstmLayers);
        writer.Write(_embeddingDim);
        writer.Write(_dropout);
        writer.Write(_distributionType);
        writer.Write(_numSamples);
        writer.Write(_useScaling);
    }

    /// <summary>
    /// Reads DeepAR-specific configuration during deserialization.
    /// </summary>
    /// <param name="reader">Binary reader for input.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This reads back DeepAR settings when loading a saved model.
    /// The values advance the reader but aren't used since constructor sets them.
    /// </para>
    /// </remarks>
    protected override void DeserializeModelSpecificData(BinaryReader reader)
    {
        _hiddenSize = reader.ReadInt32();
        _numLstmLayers = reader.ReadInt32();
        _embeddingDim = reader.ReadInt32();
        _dropout = reader.ReadDouble();
        _distributionType = reader.ReadString();
        _numSamples = reader.ReadInt32();
        _useScaling = reader.ReadBoolean();
    }

    #endregion

    #region IForecastingModel Implementation

    /// <summary>
    /// Generates probabilistic forecasts for the given input data.
    /// </summary>
    /// <param name="historicalData">The historical time series data to forecast from.</param>
    /// <param name="quantiles">Optional quantiles to estimate (e.g. 0.1, 0.5, 0.9).</param>
    /// <returns>Forecast tensor containing predicted values or distribution quantiles.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> DeepAR predicts future values by learning the probability distribution
    /// of the next time step. Instead of just a single number, it estimates the range of possible outcomes.
    /// </para>
    /// <para>
    /// If you provide quantiles, the model will return specific points in that probability distribution.
    /// For example, the 0.5 quantile is the median prediction, while 0.1 and 0.9 give you an 80% confidence interval.
    /// </para>
    /// </remarks>
    public override Tensor<T> Forecast(Tensor<T> historicalData, double[]? quantiles = null)
    {
        // Apply scaling if enabled
        var dataToProcess = _useScaling ? ApplyScaling(historicalData) : historicalData;

        // Get forecast distribution parameters
        var forecast = base.Forecast(dataToProcess, quantiles);

        // Reverse scaling
        if (_useScaling)
            forecast = ReverseScaling(forecast);

        // If quantiles requested, sample from distribution
        if (quantiles is not null && quantiles.Length > 0)
        {
            return SampleQuantiles(forecast, quantiles);
        }

        return forecast;
    }

    /// <summary>
    /// Generates multi-step forecasts by feeding predictions back into the model.
    /// </summary>
    /// <param name="input">Input tensor containing historical data.</param>
    /// <param name="steps">Number of future steps to predict.</param>
    /// <returns>Tensor containing the concatenated multi-step forecast.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> DeepAR can predict multiple steps ahead by repeatedly
    /// using its own predictions as new input. This "rolling forward" approach
    /// lets it forecast farther into the future than a single step.
    /// </para>
    /// </remarks>
    public override Tensor<T> AutoregressiveForecast(Tensor<T> input, int steps)
    {
        var predictions = new List<Tensor<T>>();
        var currentInput = input;

        int stepsRemaining = steps;
        while (stepsRemaining > 0)
        {
            var prediction = Forecast(currentInput, null);
            predictions.Add(prediction);

            int stepsUsed = Math.Min(PredictionHorizon, stepsRemaining);
            stepsRemaining -= stepsUsed;

            if (stepsRemaining > 0)
            {
                currentInput = ShiftInputWithPredictions(currentInput, prediction, stepsUsed);
            }
        }

        return ConcatenatePredictions(predictions, steps);
    }

    /// <summary>
    /// Evaluates forecast accuracy using common error metrics.
    /// </summary>
    /// <param name="predictions">Predicted values.</param>
    /// <param name="actuals">Actual ground-truth values.</param>
    /// <returns>Dictionary of metric names and values.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method reports how far the predictions are from
    /// the true values using familiar statistics like MAE and RMSE.
    /// Lower values mean better forecasts.
    /// </para>
    /// </remarks>
    public override Dictionary<string, T> Evaluate(Tensor<T> predictions, Tensor<T> actuals)
    {
        var metrics = new Dictionary<string, T>();

        T mse = NumOps.Zero;
        T mae = NumOps.Zero;
        int count = 0;

        for (int i = 0; i < predictions.Length && i < actuals.Length; i++)
        {
            var diff = NumOps.Subtract(predictions[i], actuals[i]);
            mse = NumOps.Add(mse, NumOps.Multiply(diff, diff));
            mae = NumOps.Add(mae, NumOps.Abs(diff));
            count++;
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
    /// Applies scaling to the input tensor for DeepAR processing.
    /// </summary>
    /// <param name="input">Input tensor to scale.</param>
    /// <returns>Scaled tensor.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> DeepAR works best when the data is scaled (normalized) so that values
    /// are roughly around 1.0. This method divides the input by its mean absolute value to achieve this scaling.
    /// </para>
    /// <para>
    /// This helps the model learn patterns across time series with very different magnitudes (e.g., one stock
    /// priced at $10 and another at $1000).
    /// </para>
    /// </remarks>
    public override Tensor<T> ApplyInstanceNormalization(Tensor<T> input)
    {
        return ApplyScaling(input);
    }

    /// <summary>
    /// Gets metrics specific to the DeepAR model configuration.
    /// </summary>
    /// <returns>Dictionary of metric names and values.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This returns details about the DeepAR model's internal setup,
    /// such as the size of the hidden layers (LSTM memory), the number of layers, and the number
    /// of samples used for probabilistic estimation.
    /// </para>
    /// </remarks>
    public override Dictionary<string, T> GetFinancialMetrics()
    {
        var metrics = base.GetFinancialMetrics();
        metrics["HiddenSize"] = NumOps.FromDouble(_hiddenSize);
        metrics["NumLSTMLayers"] = NumOps.FromDouble(_numLstmLayers);
        metrics["NumSamples"] = NumOps.FromDouble(_numSamples);
        
        return metrics;
    }

    #endregion

    #region Forward/Backward Pass

    /// <summary>
    /// Performs the forward pass through the DeepAR network.
    /// </summary>
    /// <param name="input">Input tensor of shape [batch, context_length, features].</param>
    /// <returns>Output tensor containing distribution parameters.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The DeepAR forward pass processes data through:
    /// 1. Input projection: Prepare features for LSTM
    /// 2. LSTM stack: Learn temporal patterns
    /// 3. Distribution heads: Output mu (mean) and sigma (std deviation)
    /// The output represents parameters of a probability distribution.
    /// </para>
    /// </remarks>
    private Tensor<T> Forward(Tensor<T> input)
    {
        var current = input;

        // Input projection
        if (_inputProjection is not null)
        {
            current = _inputProjection.Forward(current);
        }

        // LSTM layers
        foreach (var layer in _lstmLayers)
        {
            current = layer.Forward(current);
        }

        // Layer normalization
        if (_layerNorm is not null)
        {
            current = _layerNorm.Forward(current);
        }

        // Get distribution parameters
        Tensor<T> mu = current;
        Tensor<T> sigma = current;

        if (_muProjection is not null)
        {
            mu = _muProjection.Forward(current);
        }

        if (_sigmaProjection is not null)
        {
            sigma = _sigmaProjection.Forward(current);
            // Store sigma for use in SampleQuantiles
            _lastSigma = sigma;
        }

        // Combine mu and sigma into output
        // For simplicity, return mu as point forecast; sigma used in sampling
        return mu;
    }

    /// <summary>
    /// Performs the backward pass through the DeepAR network.
    /// </summary>
    /// <param name="gradOutput">Gradient from the loss function.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Backward pass computes gradients for all learnable
    /// parameters by propagating error signals backwards through each layer.
    /// This enables the model to learn from its mistakes.
    /// </para>
    /// </remarks>
    private void Backward(Tensor<T> gradOutput)
    {
        var grad = gradOutput;

        // Backward through distribution heads
        // Forward returns mu only, so propagate gradients through mu projection.
        if (_muProjection is not null)
        {
            grad = _muProjection.Backward(grad);
        }

        // Backward through layer normalization
        if (_layerNorm is not null)
        {
            grad = _layerNorm.Backward(grad);
        }

        // Backward through LSTM layers (in reverse order)
        for (int i = _lstmLayers.Count - 1; i >= 0; i--)
        {
            grad = _lstmLayers[i].Backward(grad);
        }

        // Backward through input projection
        if (_inputProjection is not null)
        {
            _inputProjection.Backward(grad);
        }
    }

    /// <summary>
    /// Performs native mode forecasting.
    /// </summary>
    /// <param name="input">Input historical data.</param>
    /// <param name="quantiles">Optional quantiles for uncertainty estimation.</param>
    /// <returns>Forecasted values.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Native mode uses the layers defined in this library
    /// for inference. This allows full control and training capability.
    /// </para>
    /// </remarks>
    protected override Tensor<T> ForecastNative(Tensor<T> input, double[]? quantiles)
    {
        SetTrainingMode(false);
        return Forward(input);
    }

    /// <summary>
    /// Validates the input tensor shape for DeepAR.
    /// </summary>
    /// <param name="input">The input tensor to validate.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the DeepAR model, ValidateInputShape checks inputs and configuration. This protects the DeepAR architecture from mismatches and errors.
    /// </para>
    /// </remarks>
    protected override void ValidateInputShape(Tensor<T> input)
    {
        // Currently only rank-3 is properly supported by ApplyScaling, ReverseScaling, and ShiftInputWithPredictions
        // TODO: Add rank-2 support to helper methods if unbatched input is needed
        if (input.Rank != 3)
            throw new ArgumentException("Input tensor must be 3D [batch_size, context_length, num_features].", nameof(input));

        int actualSeqLen = input.Shape[1];
        int actualNumFeatures = input.Shape[2];

        if (actualSeqLen != SequenceLength)
            throw new ArgumentException($"Input sequence length {actualSeqLen} does not match expected {SequenceLength}.", nameof(input));
        if (actualNumFeatures != NumFeatures)
            throw new ArgumentException($"Input number of features {actualNumFeatures} does not match expected {NumFeatures}.", nameof(input));
    }

    #endregion

    #region Model-Specific Processing

    /// <summary>
    /// Applies scaling by dividing by mean absolute value.
    /// </summary>
    /// <param name="input">Input tensor.</param>
    /// <returns>Scaled tensor.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Scaling helps DeepAR handle time series with different
    /// magnitudes. Each series is divided by its mean absolute value, bringing
    /// everything to a similar scale. This makes training more stable.
    /// </para>
    /// </remarks>
    private Tensor<T> ApplyScaling(Tensor<T> input)
    {
        int batchSize = input.Shape[0];
        int seqLen = input.Shape[1];
        int features = input.Shape.Length > 2 ? input.Shape[2] : 1;

        // Only _scaleStd is used for denormalization; _scaleMean is not needed for this scaling approach
        _scaleStd = new Tensor<T>(new[] { batchSize, 1, features });

        var scaled = new Tensor<T>(input.Shape);
        T epsilon = NumOps.FromDouble(1e-5);

        for (int b = 0; b < batchSize; b++)
        {
            for (int f = 0; f < features; f++)
            {
                // Compute mean absolute value
                T sumAbs = NumOps.Zero;
                for (int t = 0; t < seqLen; t++)
                {
                    int idx = b * seqLen * features + t * features + f;
                    if (idx < input.Length)
                        sumAbs = NumOps.Add(sumAbs, NumOps.Abs(input.Data.Span[idx]));
                }
                T scale = NumOps.Divide(sumAbs, NumOps.FromDouble(seqLen));
                scale = NumOps.Add(scale, epsilon); // Avoid division by zero

                // Store scale for reverse
                int scaleIdx = b * features + f;
                if (scaleIdx < _scaleStd.Length)
                    _scaleStd.Data.Span[scaleIdx] = scale;

                // Apply scaling
                for (int t = 0; t < seqLen; t++)
                {
                    int idx = b * seqLen * features + t * features + f;
                    if (idx < input.Length && idx < scaled.Length)
                        scaled.Data.Span[idx] = NumOps.Divide(input.Data.Span[idx], scale);
                }
            }
        }

        return scaled;
    }

    /// <summary>
    /// Reverses the scaling applied during preprocessing.
    /// </summary>
    /// <param name="output">Scaled output tensor.</param>
    /// <returns>Unscaled tensor.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> After making predictions on scaled data, we need to
    /// multiply by the original scale to get predictions in the original units.
    /// </para>
    /// </remarks>
    private Tensor<T> ReverseScaling(Tensor<T> output)
    {
        if (_scaleStd is null)
            return output;

        int batchSize = output.Shape[0];
        int seqLen = output.Shape.Length > 1 ? output.Shape[1] : 1;
        int features = output.Shape.Length > 2 ? output.Shape[2] : 1;

        var unscaled = new Tensor<T>(output.Shape);

        for (int b = 0; b < batchSize; b++)
        {
            for (int f = 0; f < features; f++)
            {
                int scaleIdx = b * features + f;
                T scale = scaleIdx < _scaleStd.Length ? _scaleStd.Data.Span[scaleIdx] : NumOps.One;

                for (int t = 0; t < seqLen; t++)
                {
                    int idx = b * seqLen * features + t * features + f;
                    if (idx < output.Length && idx < unscaled.Length)
                        unscaled.Data.Span[idx] = NumOps.Multiply(output.Data.Span[idx], scale);
                }
            }
        }

        return unscaled;
    }

    /// <summary>
    /// Samples quantiles from the forecast distribution.
    /// </summary>
    /// <param name="forecast">Forecast tensor (mu values).</param>
    /// <param name="quantiles">Quantile levels to sample (e.g., [0.1, 0.5, 0.9]).</param>
    /// <returns>Tensor with quantile forecasts.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Quantiles help express uncertainty. For example:
    /// - 0.1 quantile means 10% of values should be below this
    /// - 0.5 quantile is the median (middle value)
    /// - 0.9 quantile means 90% of values should be below this
    /// Together, the 0.1 and 0.9 quantiles form an 80% prediction interval.
    /// </para>
    /// </remarks>
    private Tensor<T> SampleQuantiles(Tensor<T> forecast, double[] quantiles)
    {
        // For Gaussian distribution, quantiles can be computed analytically
        // For simplicity, we use a standard deviation estimate
        int batchSize = forecast.Shape[0];
        int seqLen = forecast.Shape.Length > 1 ? forecast.Shape[1] : 1;
        int numQuantiles = quantiles.Length;

        var quantileForecast = new Tensor<T>(new[] { batchSize, seqLen, numQuantiles });

        for (int b = 0; b < batchSize; b++)
        {
            for (int t = 0; t < seqLen; t++)
            {
                int muIdx = b * seqLen + t;
                T mu = muIdx < forecast.Length ? forecast.Data.Span[muIdx] : NumOps.Zero;

                // Use learned sigma from Forward if available, otherwise estimate
                T sigma;
                if (_lastSigma is not null && muIdx < _lastSigma.Length)
                {
                    sigma = _lastSigma.Data.Span[muIdx];
                    // Ensure sigma is positive with a minimum floor
                    sigma = NumOps.Add(NumOps.Abs(sigma), NumOps.FromDouble(0.01));
                }
                else
                {
                    // Fallback: estimate sigma as 10% of mu
                    sigma = NumOps.Multiply(NumOps.Abs(mu), NumOps.FromDouble(0.1));
                    sigma = NumOps.Add(sigma, NumOps.FromDouble(0.01)); // Minimum sigma
                }

                for (int q = 0; q < numQuantiles; q++)
                {
                    // Standard normal quantile (approximation)
                    double z = GetStandardNormalQuantile(quantiles[q]);
                    T quantileValue = NumOps.Add(mu, NumOps.Multiply(sigma, NumOps.FromDouble(z)));

                    int outIdx = b * seqLen * numQuantiles + t * numQuantiles + q;
                    if (outIdx < quantileForecast.Length)
                        quantileForecast.Data.Span[outIdx] = quantileValue;
                }
            }
        }

        return quantileForecast;
    }

    /// <summary>
    /// Computes the standard normal quantile (inverse CDF).
    /// </summary>
    /// <param name="p">Probability (0 to 1).</param>
    /// <returns>The z-score corresponding to the quantile.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This converts a probability (like 0.95) to a z-score
    /// (like 1.645) that can be used to compute prediction intervals.
    /// </para>
    /// </remarks>
    private static double GetStandardNormalQuantile(double p)
    {
        // Approximation using Abramowitz and Stegun formula 26.2.23
        if (p <= 0) return double.NegativeInfinity;
        if (p >= 1) return double.PositiveInfinity;
        if (Math.Abs(p - 0.5) < 1e-10) return 0;

        double sign = p < 0.5 ? -1 : 1;
        double pp = p < 0.5 ? p : 1 - p;

        double t = Math.Sqrt(-2 * Math.Log(pp));
        double c0 = 2.515517;
        double c1 = 0.802853;
        double c2 = 0.010328;
        double d1 = 1.432788;
        double d2 = 0.189269;
        double d3 = 0.001308;

        double z = t - (c0 + c1 * t + c2 * t * t) / (1 + d1 * t + d2 * t * t + d3 * t * t * t);

        return sign * z;
    }

    /// <summary>
    /// Computes negative log-likelihood loss for the distribution.
    /// </summary>
    /// <param name="predictions">Predicted distribution parameters.</param>
    /// <param name="targets">Actual target values.</param>
    /// <returns>The negative log-likelihood loss.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Negative log-likelihood measures how well the predicted
    /// distribution explains the actual data. Lower values mean the model assigns
    /// higher probability to the correct outcomes.
    /// </para>
    /// </remarks>
    private T ComputeNegativeLogLikelihood(Tensor<T> predictions, Tensor<T> targets)
    {
        // For Gaussian distribution: -log(p(y|mu,sigma)) = 0.5*log(2*pi*sigma^2) + (y-mu)^2/(2*sigma^2)
        // Simplified version using MSE as proxy
        return LossFunction.CalculateLoss(predictions.ToVector(), targets.ToVector());
    }

    /// <summary>
    /// Computes gradient for backpropagation.
    /// </summary>
    /// <param name="predictions">Predicted values.</param>
    /// <param name="targets">Target values.</param>
    /// <returns>Gradient tensor.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The gradient tells us how much each prediction error
    /// contributes to the overall loss. This guides the model on how to adjust
    /// its weights to make better predictions.
    /// </para>
    /// </remarks>
    private Tensor<T> ComputeGradient(Tensor<T> predictions, Tensor<T> targets)
    {
        var gradVector = LossFunction.CalculateDerivative(predictions.ToVector(), targets.ToVector());
        return Tensor<T>.FromVector(gradVector, predictions.Shape);
    }

    /// <summary>
    /// Computes CRPS (Continuous Ranked Probability Score) for probabilistic evaluation.
    /// </summary>
    /// <param name="predictions">Predicted values.</param>
    /// <param name="actuals">Actual values.</param>
    /// <returns>CRPS score.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> CRPS is a proper scoring rule for probabilistic forecasts.
    /// It measures how well the predicted distribution matches reality. Lower is better.
    /// Unlike MSE, it rewards both accuracy and calibration.
    /// </para>
    /// </remarks>
    private T ComputeCRPS(Tensor<T> predictions, Tensor<T> actuals)
    {
        // Simplified CRPS approximation
        T crps = NumOps.Zero;
        int count = 0;

        for (int i = 0; i < predictions.Length && i < actuals.Length; i++)
        {
            var diff = NumOps.Abs(NumOps.Subtract(predictions[i], actuals[i]));
            crps = NumOps.Add(crps, diff);
            count++;
        }

        return count > 0 ? NumOps.Divide(crps, NumOps.FromDouble(count)) : NumOps.Zero;
    }

    #endregion

    #region Helper Methods

    /// <summary>
    /// Shifts input by incorporating recent predictions.
    /// </summary>
    /// <param name="input">Current input tensor.</param>
    /// <param name="prediction">Recent prediction tensor.</param>
    /// <param name="stepsUsed">Number of prediction steps to incorporate.</param>
    /// <returns>Shifted input tensor.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> For autoregressive forecasting, we need to feed predictions
    /// back as input to generate longer forecasts. This method shifts the input window
    /// forward and appends recent predictions.
    /// </para>
    /// </remarks>
    protected override Tensor<T> ShiftInputWithPredictions(Tensor<T> input, Tensor<T> prediction, int stepsUsed)
    {
        int batchSize = input.Shape[0];
        int seqLen = input.Shape[1];
        int features = input.Shape.Length > 2 ? input.Shape[2] : 1;

        var shifted = new Tensor<T>(input.Shape);

        for (int b = 0; b < batchSize; b++)
        {
            for (int f = 0; f < features; f++)
            {
                // Shift old values
                for (int t = 0; t < seqLen - stepsUsed; t++)
                {
                    int srcIdx = b * seqLen * features + (t + stepsUsed) * features + f;
                    int dstIdx = b * seqLen * features + t * features + f;
                    if (srcIdx < input.Length && dstIdx < shifted.Length)
                        shifted.Data.Span[dstIdx] = input.Data.Span[srcIdx];
                }

                // Add new predictions
                for (int t = seqLen - stepsUsed; t < seqLen; t++)
                {
                    int predIdx = b * stepsUsed * features + (t - (seqLen - stepsUsed)) * features + f;
                    int dstIdx = b * seqLen * features + t * features + f;
                    if (predIdx < prediction.Length && dstIdx < shifted.Length)
                        shifted.Data.Span[dstIdx] = prediction.Data.Span[predIdx];
                }
            }
        }

        return shifted;
    }

    /// <summary>
    /// Concatenates multiple predictions into a single tensor.
    /// </summary>
    /// <param name="predictions">List of prediction tensors.</param>
    /// <param name="totalSteps">Total number of steps to include.</param>
    /// <returns>Concatenated prediction tensor.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> When making long forecasts that exceed the model's
    /// prediction horizon, we make multiple predictions and combine them here.
    /// </para>
    /// </remarks>
    protected override Tensor<T> ConcatenatePredictions(List<Tensor<T>> predictions, int totalSteps)
    {
        if (predictions.Count == 0)
            return new Tensor<T>(new[] { 1, totalSteps, NumFeatures });

        int batchSize = predictions[0].Shape[0];
        int features = predictions[0].Shape.Length > 2 ? predictions[0].Shape[2] : NumFeatures;

        var result = new Tensor<T>(new[] { batchSize, totalSteps, features });
        int currentStep = 0;

        foreach (var pred in predictions)
        {
            int predSteps = pred.Shape.Length > 1 ? pred.Shape[1] : 1;
            int stepsToCopy = Math.Min(predSteps, totalSteps - currentStep);

            for (int b = 0; b < batchSize; b++)
            {
                for (int t = 0; t < stepsToCopy; t++)
                {
                    for (int f = 0; f < features; f++)
                    {
                        int srcIdx = b * predSteps * features + t * features + f;
                        int dstIdx = b * totalSteps * features + (currentStep + t) * features + f;
                        if (srcIdx < pred.Length && dstIdx < result.Length)
                            result.Data.Span[dstIdx] = pred.Data.Span[srcIdx];
                    }
                }
            }

            currentStep += stepsToCopy;
            if (currentStep >= totalSteps)
                break;
        }

        return result;
    }

    #endregion

    #region IDisposable

    /// <summary>
    /// Releases resources used by the DeepAR model.
    /// </summary>
    /// <param name="disposing">True if called from Dispose(), false if from finalizer.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This cleans up resources like the ONNX session when the
    /// model is no longer needed. Always dispose models properly to free memory.
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
