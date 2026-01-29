using System.IO;
using AiDotNet.Enums;
using AiDotNet.Finance.Interfaces;
using AiDotNet.Models.Options;
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

using AiDotNet.Finance.Base;
namespace AiDotNet.Finance.Forecasting.Transformers;

/// <summary>
/// TSMixer: An all-MLP architecture for time series forecasting.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (typically float or double).</typeparam>
/// <remarks>
/// <para>
/// TSMixer achieves state-of-the-art results using only multilayer perceptrons (MLPs),
/// without attention mechanisms. It alternates between time-mixing and feature-mixing
/// operations to capture both temporal patterns and cross-variable relationships.
/// </para>
/// <para>
/// <b>For Beginners:</b> TSMixer is a simpler alternative to transformer-based models that:
/// - Uses only MLPs (fully connected layers)
/// - Alternates between mixing information across time and across features
/// - Is faster to train and more memory-efficient than attention-based models
///
/// The core idea is that mixing time and feature information separately but alternately
/// is sufficient for capturing complex patterns in multivariate time series.
/// </para>
/// <para>
/// <b>Reference:</b> Chen et al., "TSMixer: An All-MLP Architecture for Time Series
/// Forecasting", TMLR 2023. https://arxiv.org/abs/2303.06053
/// </para>
/// </remarks>
public class TSMixer<T> : ForecastingModelBase<T>
{
    #region Execution Mode

    /// <summary>
    /// Indicates whether this network uses native layers (true) or ONNX model (false).
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The model can run in two modes:
    /// - Native mode: Uses built-in layers for full training support
    /// - ONNX mode: Uses a pre-trained model for inference only
    /// </para>
    /// </remarks>
    private readonly bool _useNativeMode;

    #endregion

    
    #region Native Mode Fields

    /// <summary>
    /// Input projection layer.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This layer transforms the input features into a hidden
    /// representation suitable for the mixer blocks to process.
    /// </para>
    /// </remarks>
    private ILayer<T>? _inputProjection;

    /// <summary>
    /// Mixer blocks containing time-mixing and feature-mixing MLPs.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Each mixer block contains:
    /// - Time-mixing: MLPs that learn patterns across the time dimension
    /// - Feature-mixing: MLPs that learn relationships between variables
    /// </para>
    /// </remarks>
    private readonly List<ILayer<T>> _mixerBlocks = [];

    /// <summary>
    /// Temporal projection layer.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This layer maps from the input sequence length
    /// to the desired prediction horizon.
    /// </para>
    /// </remarks>
    private ILayer<T>? _temporalProjection;

    /// <summary>
    /// Output projection layer.
    /// </summary>
    private ILayer<T>? _outputProjection;

    /// <summary>
    /// Instance mean for RevIN normalization.
    /// </summary>
    private Tensor<T>? _instanceMean;

    /// <summary>
    /// Instance standard deviation for RevIN normalization.
    /// </summary>
    private Tensor<T>? _instanceStd;

    #endregion

    #region Shared Fields

    /// <summary>
    /// The optimizer for training.
    /// </summary>
    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> _optimizer;

    /// <summary>
    /// The loss function for training.
    /// </summary>
    private readonly ILossFunction<T> _lossFunction;

    /// <summary>
    /// The input sequence length (lookback window).
    /// </summary>
    private readonly int _sequenceLength;

    /// <summary>
    /// The prediction horizon.
    /// </summary>
    private readonly int _predictionHorizon;

    /// <summary>
    /// Number of input features.
    /// </summary>
    private readonly int _numFeatures;

    /// <summary>
    /// Hidden dimension for MLP layers.
    /// </summary>
    private readonly int _hiddenDim;

    /// <summary>
    /// Number of mixer blocks.
    /// </summary>
    private readonly int _numBlocks;

    /// <summary>
    /// Feedforward expansion factor.
    /// </summary>
    private readonly double _feedForwardExpansion;

    /// <summary>
    /// Whether to mix features before time.
    /// </summary>
    private readonly bool _featuresFirst;

    /// <summary>
    /// Whether to use RevIN normalization.
    /// </summary>
    private readonly bool _useRevIN;

    /// <summary>
    /// Dropout rate.
    /// </summary>
    private readonly double _dropout;

    #endregion

    #region Interface Properties

    /// <summary>
    /// Gets the patch size. TSMixer does not use patches (returns 1).
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Unlike PatchTST, TSMixer processes the entire sequence
    /// without dividing it into patches.
    /// </para>
    /// </remarks>
    public override int PatchSize => 1;

    /// <summary>
    /// Gets the stride. TSMixer does not use patches (returns 1).
    /// </summary>
    public override int Stride => 1;

    /// <summary>
    /// Gets whether the model processes channels independently.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> TSMixer processes all features together through
    /// feature-mixing operations to capture cross-variable relationships.
    /// </para>
    /// </remarks>
    public override bool IsChannelIndependent => false;

    /// <summary>
    /// Gets whether the model uses native mode (true) or ONNX mode (false).
    /// </summary>
    public override bool UseNativeMode => _useNativeMode;

    /// <summary>
    /// Gets the input sequence length.
    /// </summary>
    public override int SequenceLength => _sequenceLength;

    /// <summary>
    /// Gets the prediction horizon.
    /// </summary>
    public override int PredictionHorizon => _predictionHorizon;

    /// <summary>
    /// Gets the number of input features.
    /// </summary>
    public override int NumFeatures => _numFeatures;

    #endregion

    #region Constructors

    /// <summary>
    /// Creates a new TSMixer instance for ONNX inference mode.
    /// </summary>
    /// <param name="architecture">Neural network architecture configuration.</param>
    /// <param name="onnxModelPath">Path to the ONNX model file.</param>
    /// <param name="options">Model options for configuration.</param>
    /// <param name="optimizer">Optional optimizer for fine-tuning.</param>
    /// <param name="lossFunction">Optional loss function.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Use this constructor when you have a pre-trained ONNX model.
    /// This mode is for inference only - you cannot train the model.
    /// </para>
    /// </remarks>
    public TSMixer(
        NeuralNetworkArchitecture<T> architecture,
        string onnxModelPath,
        TSMixerOptions<T>? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
        ILossFunction<T>? lossFunction = null)
        : base(architecture, lossFunction ?? new MeanSquaredErrorLoss<T>(), 1.0)
    {
        if (string.IsNullOrWhiteSpace(onnxModelPath))
            throw new ArgumentNullException(nameof(onnxModelPath));
        if (!File.Exists(onnxModelPath))
            throw new FileNotFoundException($"ONNX model not found: {onnxModelPath}");

        var opts = options ?? new TSMixerOptions<T>();
        var errors = opts.Validate();
        if (errors.Count > 0)
            throw new ArgumentException($"Invalid options: {string.Join(", ", errors)}");

        _useNativeMode = false;
        OnnxSession = new InferenceSession(onnxModelPath);
        OnnxModelPath = onnxModelPath;

        _sequenceLength = opts.SequenceLength;
        _predictionHorizon = opts.PredictionHorizon;
        _numFeatures = opts.NumFeatures;
        _hiddenDim = opts.HiddenDimension;
        _numBlocks = opts.NumBlocks;
        _feedForwardExpansion = opts.FeedForwardExpansion;
        _featuresFirst = opts.FeaturesFirst;
        _useRevIN = opts.UseRevIN;
        _dropout = opts.Dropout;

        _optimizer = optimizer ?? new AdamOptimizer<T, Tensor<T>, Tensor<T>>(this);
        _lossFunction = lossFunction ?? new MeanSquaredErrorLoss<T>();
    }

    /// <summary>
    /// Creates a new TSMixer instance for native training mode.
    /// </summary>
    /// <param name="architecture">Neural network architecture configuration.</param>
    /// <param name="options">Model options for configuration.</param>
    /// <param name="optimizer">Optional optimizer for training.</param>
    /// <param name="lossFunction">Optional loss function.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Use this constructor when you want to train a model from scratch.
    /// This mode supports full training and inference.
    /// </para>
    /// </remarks>
    public TSMixer(
        NeuralNetworkArchitecture<T> architecture,
        TSMixerOptions<T>? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
        ILossFunction<T>? lossFunction = null)
        : base(architecture, lossFunction ?? new MeanSquaredErrorLoss<T>(), 1.0)
    {
        var opts = options ?? new TSMixerOptions<T>();
        var errors = opts.Validate();
        if (errors.Count > 0)
            throw new ArgumentException($"Invalid options: {string.Join(", ", errors)}");

        _useNativeMode = true;

        _sequenceLength = opts.SequenceLength;
        _predictionHorizon = opts.PredictionHorizon;
        _numFeatures = opts.NumFeatures;
        _hiddenDim = opts.HiddenDimension;
        _numBlocks = opts.NumBlocks;
        _feedForwardExpansion = opts.FeedForwardExpansion;
        _featuresFirst = opts.FeaturesFirst;
        _useRevIN = opts.UseRevIN;
        _dropout = opts.Dropout;

        _optimizer = optimizer ?? new AdamOptimizer<T, Tensor<T>, Tensor<T>>(this);
        _lossFunction = lossFunction ?? new MeanSquaredErrorLoss<T>();

        InitializeLayers();
    }

    #endregion

    #region Initialization

    /// <summary>
    /// Initializes the neural network layers.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method sets up all the layers that make up the
    /// TSMixer model. It either uses custom layers provided in the architecture
    /// or creates default layers using LayerHelper.
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
            Layers.AddRange(LayerHelper<T>.CreateDefaultTSMixerLayers(
                Architecture,
                _sequenceLength,
                _predictionHorizon,
                _numFeatures,
                _hiddenDim,
                _numBlocks,
                _feedForwardExpansion,
                _dropout));
            ExtractLayerReferences();
        }
    }

    /// <summary>
    /// Extracts references to specific layers for the TSMixer architecture.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> After creating all layers, we identify which layer
    /// serves which purpose (input, mixing, output) so we can use them correctly
    /// during forward and backward passes.
    /// </para>
    /// </remarks>
    private void ExtractLayerReferences()
    {
        int layerIndex = 0;

        // Input projection (first dense layer)
        if (layerIndex < Layers.Count && Layers[layerIndex] is DenseLayer<T>)
        {
            _inputProjection = Layers[layerIndex++];
        }

        // Mixer blocks
        // Each block has: LayerNorm, Dense (expand), Dense (project), Dropout (time),
        //                 LayerNorm, Dense (expand), Dense (project), Dropout (feature)
        int layersPerBlock = 8;
        for (int i = 0; i < _numBlocks && layerIndex + layersPerBlock <= Layers.Count; i++)
        {
            for (int j = 0; j < layersPerBlock; j++)
            {
                _mixerBlocks.Add(Layers[layerIndex++]);
            }
        }

        // Final layer normalization
        if (layerIndex < Layers.Count && Layers[layerIndex] is LayerNormalizationLayer<T>)
        {
            layerIndex++;
        }

        // Temporal projection
        if (layerIndex < Layers.Count && Layers[layerIndex] is DenseLayer<T>)
        {
            _temporalProjection = Layers[layerIndex++];
        }

        // Output projection
        if (layerIndex < Layers.Count && Layers[layerIndex] is DenseLayer<T>)
        {
            _outputProjection = Layers[layerIndex];
        }
    }

    /// <summary>
    /// Validates custom layers provided through the architecture.
    /// </summary>
    /// <param name="layers">List of custom layers to validate.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> When you provide custom layers instead of using defaults,
    /// this method ensures they are compatible with TSMixer requirements.
    /// </para>
    /// </remarks>
    protected override void ValidateCustomLayers(List<ILayer<T>> layers)
    {
        base.ValidateCustomLayers(layers);

        if (layers.Count < 3)
        {
            throw new ArgumentException(
                "TSMixer requires at least 3 layers: input projection, mixer block(s), and output projection.");
        }
    }

    #endregion

    #region NeuralNetworkBase Overrides

    /// <summary>
    /// Gets whether this network supports training.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Training is only supported in native mode.
    /// ONNX mode is for inference (predictions) only.
    /// </para>
    /// </remarks>
    public override bool SupportsTraining => _useNativeMode;

    /// <summary>
    /// Performs prediction on the given input.
    /// </summary>
    /// <param name="input">Input tensor with shape [batch, sequence, features].</param>
    /// <returns>Prediction tensor with shape [batch, horizon, features].</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method takes your historical data and produces predictions.
    /// It automatically chooses between native and ONNX execution based on how the model was created.
    /// </para>
    /// </remarks>
    public override Tensor<T> Predict(Tensor<T> input)
    {
        return _useNativeMode ? Forward(input) : ForecastOnnx(input);
    }

    /// <summary>
    /// Trains the network on a single batch.
    /// </summary>
    /// <param name="input">Input tensor.</param>
    /// <param name="target">Target tensor.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Training adjusts the model's internal parameters to make better predictions.
    /// This method:
    /// 1. Makes a prediction (forward pass)
    /// 2. Calculates how wrong it was (loss)
    /// 3. Figures out how to improve (backward pass)
    /// 4. Updates the model (optimizer step)
    /// </para>
    /// </remarks>
    public override void Train(Tensor<T> input, Tensor<T> target)
    {
        if (!_useNativeMode)
            throw new InvalidOperationException("Training is only supported in native mode.");

        SetTrainingMode(true);
        var predictions = Forward(input);
        LastLoss = _lossFunction.CalculateLoss(predictions.ToVector(), target.ToVector());
        var gradient = _lossFunction.CalculateDerivative(predictions.ToVector(), target.ToVector());
        Backward(Tensor<T>.FromVector(gradient, predictions.Shape));
        _optimizer.UpdateParameters(Layers);
        SetTrainingMode(false);
    }

    /// <summary>
    /// Updates network parameters based on gradients.
    /// </summary>
    /// <param name="gradients">Gradient vector for parameter updates.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Parameter updates are handled by the optimizer in the Train method.
    /// This method exists for interface compatibility.
    /// </para>
    /// </remarks>
    public override void UpdateParameters(Vector<T> gradients)
    {
        // Parameters are updated via the optimizer in Train()
    }

    /// <summary>
    /// Gets metadata about the model.
    /// </summary>
    /// <returns>Model metadata including architecture details.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Metadata provides information about the model's configuration
    /// that can be useful for saving, loading, or inspecting the model.
    /// </para>
    /// </remarks>
    public override ModelMetadata<T> GetModelMetadata()
    {
        return new ModelMetadata<T>
        {
            ModelType = ModelType.NeuralNetwork,
            AdditionalInfo = new Dictionary<string, object>
            {
                ["NetworkType"] = "TSMixer",
                ["SequenceLength"] = _sequenceLength,
                ["PredictionHorizon"] = _predictionHorizon,
                ["NumFeatures"] = _numFeatures,
                ["HiddenDimension"] = _hiddenDim,
                ["NumBlocks"] = _numBlocks,
                ["FeaturesFirst"] = _featuresFirst,
                ["UseRevIN"] = _useRevIN,
                ["UseNativeMode"] = _useNativeMode
            }
        };
    }

    /// <summary>
    /// Creates a new instance of this network type.
    /// </summary>
    /// <returns>A new TSMixer instance.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This factory method creates a copy of the model structure,
    /// useful for ensemble methods or hyperparameter search.
    /// </para>
    /// </remarks>
    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance()
    {
        var options = new TSMixerOptions<T>
        {
            SequenceLength = _sequenceLength,
            PredictionHorizon = _predictionHorizon,
            NumFeatures = _numFeatures,
            HiddenDimension = _hiddenDim,
            NumBlocks = _numBlocks,
            FeedForwardExpansion = _feedForwardExpansion,
            FeaturesFirst = _featuresFirst,
            UseRevIN = _useRevIN,
            Dropout = _dropout
        };

        return new TSMixer<T>(Architecture, options);
    }

    /// <summary>
    /// Serializes network-specific data for persistence.
    /// </summary>
    /// <param name="writer">Binary writer for output.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method packages the model's learned parameters
    /// into a format that can be saved to disk and loaded later.
    /// </para>
    /// </remarks>
    protected override void SerializeNetworkSpecificData(BinaryWriter writer)
    {
        writer.Write(_sequenceLength);
        writer.Write(_predictionHorizon);
        writer.Write(_numFeatures);
        writer.Write(_hiddenDim);
        writer.Write(_numBlocks);
        writer.Write(_feedForwardExpansion);
        writer.Write(_featuresFirst);
        writer.Write(_useRevIN);
        writer.Write(_dropout);
        writer.Write(_useNativeMode);
    }

    /// <summary>
    /// Deserializes network-specific data from persistence.
    /// </summary>
    /// <param name="reader">Binary reader for input.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method restores the model's learned parameters
    /// from a saved format. Called when loading a model from disk.
    /// </para>
    /// </remarks>
    protected override void DeserializeNetworkSpecificData(BinaryReader reader)
    {
        _ = reader.ReadInt32(); // sequenceLength
        _ = reader.ReadInt32(); // predictionHorizon
        _ = reader.ReadInt32(); // numFeatures
        _ = reader.ReadInt32(); // hiddenDim
        _ = reader.ReadInt32(); // numBlocks
        _ = reader.ReadDouble(); // feedForwardExpansion
        _ = reader.ReadBoolean(); // featuresFirst
        _ = reader.ReadBoolean(); // useRevIN
        _ = reader.ReadDouble(); // dropout
        _ = reader.ReadBoolean(); // useNativeMode
    }

    #endregion

    #region IForecastingModel Implementation

    /// <summary>
    /// Generates forecasts for the given historical data.
    /// </summary>
    /// <param name="historicalData">Historical time series data.</param>
    /// <param name="quantiles">Optional quantile levels for probabilistic forecasting.</param>
    /// <returns>Forecasted values.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This is the main method for making predictions.
    /// You provide historical data, and it returns predictions for the future.
    /// </para>
    /// </remarks>
    public override Tensor<T> Forecast(Tensor<T> historicalData, double[]? quantiles = null)
    {
        // Apply RevIN normalization if enabled
        if (_useRevIN)
            historicalData = ApplyRevIN(historicalData, normalize: true);

        var forecast = _useNativeMode ? ForecastNative(historicalData, _predictionHorizon) : ForecastOnnx(historicalData);

        // Apply RevIN de-normalization if enabled
        if (_useRevIN)
            forecast = ApplyRevIN(forecast, normalize: false);

        return forecast;
    }

    /// <summary>
    /// Generates multi-step forecasts using autoregressive prediction.
    /// </summary>
    /// <param name="input">Input tensor.</param>
    /// <param name="steps">Number of steps to predict.</param>
    /// <returns>Autoregressive forecast tensor.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Autoregressive forecasting makes predictions step by step,
    /// feeding each prediction back as input for the next. This can be more accurate
    /// for long horizons but is slower.
    /// </para>
    /// </remarks>
    public override Tensor<T> AutoregressiveForecast(Tensor<T> input, int steps)
    {
        if (steps <= 0)
            throw new ArgumentOutOfRangeException(nameof(steps), "Steps must be positive.");

        var result = new Tensor<T>(new[] { 1, steps, _numFeatures });
        var currentInput = input;

        for (int i = 0; i < steps; i += _predictionHorizon)
        {
            var prediction = Predict(currentInput);
            int stepsToCopy = Math.Min(_predictionHorizon, steps - i);

            // Copy predictions to result
            for (int j = 0; j < stepsToCopy; j++)
            {
                for (int f = 0; f < _numFeatures; f++)
                {
                    int srcIdx = j * _numFeatures + f;
                    int dstIdx = (i + j) * _numFeatures + f;
                    if (srcIdx < prediction.Length && dstIdx < result.Length)
                    {
                        result.Data.Span[dstIdx] = prediction.Data.Span[srcIdx];
                    }
                }
            }

            // Update input for next iteration
            if (i + _predictionHorizon < steps)
            {
                currentInput = ShiftAndAppend(currentInput, prediction);
            }
        }

        return result;
    }

    /// <summary>
    /// Evaluates the model on a test dataset.
    /// </summary>
    /// <param name="testData">Test input data.</param>
    /// <param name="testLabels">Test target labels.</param>
    /// <returns>Dictionary of evaluation metrics.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Evaluation tells you how well your model performs on data
    /// it hasn't seen before. Common metrics include MSE (mean squared error) and MAE (mean absolute error).
    /// </para>
    /// </remarks>
    public override Dictionary<string, T> Evaluate(Tensor<T> testData, Tensor<T> testLabels)
    {
        var predictions = Predict(testData);
        var mse = _lossFunction.CalculateLoss(predictions.ToVector(), testLabels.ToVector());

        // Calculate MAE
        T mae = NumOps.Zero;
        for (int i = 0; i < predictions.Length && i < testLabels.Length; i++)
        {
            var diff = NumOps.Subtract(predictions.Data.Span[i], testLabels.Data.Span[i]);
            mae = NumOps.Add(mae, NumOps.Abs(diff));
        }
        mae = NumOps.Divide(mae, NumOps.FromDouble(predictions.Length));

        return new Dictionary<string, T>
        {
            ["MSE"] = mse,
            ["MAE"] = mae,
            ["RMSE"] = NumOps.Sqrt(mse)
        };
    }

    /// <summary>
    /// Applies instance normalization (RevIN) to the input.
    /// </summary>
    /// <param name="input">Input tensor to normalize.</param>
    /// <returns>Normalized tensor.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Instance normalization adjusts each sample to have zero mean
    /// and unit variance. This helps the model handle different scales in the data.
    /// </para>
    /// </remarks>
    public override Tensor<T> ApplyInstanceNormalization(Tensor<T> input)
    {
        return ApplyRevIN(input, normalize: true);
    }

    /// <summary>
    /// Gets financial metrics specific to the model.
    /// </summary>
    /// <returns>Dictionary of financial metrics.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Financial metrics provide additional information about
    /// the model's performance that's relevant to trading and forecasting applications.
    /// </para>
    /// </remarks>
    public override Dictionary<string, T> GetFinancialMetrics()
    {
        return new Dictionary<string, T>
        {
            ["LastTrainingLoss"] = LastLoss ?? NumOps.Zero,
            ["SequenceLength"] = NumOps.FromDouble(_sequenceLength),
            ["PredictionHorizon"] = NumOps.FromDouble(_predictionHorizon),
            ["HiddenDimension"] = NumOps.FromDouble(_hiddenDim),
            ["NumBlocks"] = NumOps.FromDouble(_numBlocks)
        };
    }

    #endregion

    #region Forward/Backward Pass

    /// <summary>
    /// Performs the forward pass through the network.
    /// </summary>
    /// <param name="input">Input tensor with shape [batch, sequence, features].</param>
    /// <returns>Output tensor with predictions.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The forward pass processes input data through all layers
    /// to produce predictions. For TSMixer, this includes:
    /// 1. Input projection
    /// 2. Alternating time-mixing and feature-mixing in each block
    /// 3. Temporal and output projection
    /// </para>
    /// </remarks>
    private Tensor<T> Forward(Tensor<T> input)
    {
        // Apply RevIN if enabled
        var normalized = _useRevIN ? ApplyRevIN(input, normalize: true) : input;

        // Input projection
        var current = _inputProjection?.Forward(normalized) ?? normalized;

        // Mixer blocks
        foreach (var layer in _mixerBlocks)
        {
            current = layer.Forward(current);
        }

        // Process remaining layers (final norm, projections)
        int mixerEndIndex = 1 + _mixerBlocks.Count; // 1 for input projection
        for (int i = mixerEndIndex; i < Layers.Count; i++)
        {
            current = Layers[i].Forward(current);
        }

        // Apply RevIN de-normalization if enabled
        if (_useRevIN)
        {
            current = ApplyRevIN(current, normalize: false);
        }

        return current;
    }

    /// <summary>
    /// Performs the backward pass for gradient computation.
    /// </summary>
    /// <param name="outputGradient">Gradient from the loss function.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The backward pass calculates how each parameter contributed
    /// to the error. This information is then used by the optimizer to update the parameters.
    /// </para>
    /// </remarks>
    private void Backward(Tensor<T> outputGradient)
    {
        var gradient = outputGradient;

        // Backward through all layers in reverse order
        for (int i = Layers.Count - 1; i >= 0; i--)
        {
            gradient = Layers[i].Backward(gradient);
        }
    }

    /// <summary>
    /// Performs native forecasting using the built-in layers.
    /// </summary>
    /// <param name="historicalData">Historical data for context.</param>
    /// <param name="horizon">Number of steps to forecast.</param>
    /// <returns>Forecast tensor.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method handles the actual prediction when running
    /// in native mode. It processes the input through all layers and formats the output.
    /// </para>
    /// </remarks>
    private Tensor<T> ForecastNative(Tensor<T> historicalData, int horizon)
    {
        SetTrainingMode(false);
        var output = Forward(historicalData);

        // Reshape output if needed
        int totalSteps = Math.Min(horizon, _predictionHorizon);
        var result = new Tensor<T>(new[] { 1, totalSteps, _numFeatures });

        for (int i = 0; i < totalSteps * _numFeatures && i < output.Length; i++)
        {
            result.Data.Span[i] = output.Data.Span[i];
        }

        return result;
    }

    /// <summary>
    /// Performs ONNX-based forecasting using the loaded model.
    /// </summary>
    /// <param name="input">Input tensor.</param>
    /// <returns>Forecast tensor.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> ONNX mode runs a pre-trained model for inference.
    /// This is typically faster than native mode but doesn't support training.
    /// </para>
    /// </remarks>
    protected override Tensor<T> ForecastOnnx(Tensor<T> input)
    {
        if (OnnxSession is null)
            throw new InvalidOperationException("ONNX session is not initialized.");

        var inputData = ConvertToFloatArray(input);
        var inputTensor = new OnnxTensors.DenseTensor<float>(
            inputData,
            new[] { 1, _sequenceLength, _numFeatures });

        var inputs = new List<NamedOnnxValue>
        {
            NamedOnnxValue.CreateFromTensor("input", inputTensor)
        };

        using var results = OnnxSession.Run(inputs);
        var outputTensor = results.First().AsTensor<float>();
        return ConvertFromOnnxTensor(outputTensor);
    }

    #endregion

    #region Model-Specific Processing

    /// <summary>
    /// Applies RevIN (Reversible Instance Normalization).
    /// </summary>
    /// <param name="input">Input tensor.</param>
    /// <param name="normalize">True to normalize, false to de-normalize.</param>
    /// <returns>Transformed tensor.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> RevIN normalizes each instance (sample) independently
    /// by subtracting mean and dividing by std. De-normalization reverses this
    /// to restore the original scale of predictions.
    /// </para>
    /// </remarks>
    private Tensor<T> ApplyRevIN(Tensor<T> input, bool normalize)
    {
        var result = new Tensor<T>(input.Shape);
        var eps = NumOps.FromDouble(1e-5);

        if (normalize)
        {
            // Compute mean and std for this instance
            T sum = NumOps.Zero;
            for (int i = 0; i < input.Length; i++)
            {
                sum = NumOps.Add(sum, input.Data.Span[i]);
            }
            var mean = NumOps.Divide(sum, NumOps.FromDouble(input.Length));

            T varSum = NumOps.Zero;
            for (int i = 0; i < input.Length; i++)
            {
                var diff = NumOps.Subtract(input.Data.Span[i], mean);
                varSum = NumOps.Add(varSum, NumOps.Multiply(diff, diff));
            }
            var std = NumOps.Sqrt(NumOps.Divide(varSum, NumOps.FromDouble(input.Length)));
            std = NumOps.Add(std, eps);

            // Store for de-normalization
            _instanceMean = new Tensor<T>(new[] { 1 });
            _instanceMean.Data.Span[0] = mean;
            _instanceStd = new Tensor<T>(new[] { 1 });
            _instanceStd.Data.Span[0] = std;

            // Normalize
            for (int i = 0; i < input.Length; i++)
            {
                var normalized = NumOps.Divide(
                    NumOps.Subtract(input.Data.Span[i], mean),
                    std);
                result.Data.Span[i] = normalized;
            }
        }
        else
        {
            // De-normalize using stored statistics
            if (_instanceMean is null || _instanceStd is null)
            {
                return input;
            }

            var mean = _instanceMean.Data.Span[0];
            var std = _instanceStd.Data.Span[0];

            for (int i = 0; i < input.Length; i++)
            {
                var denormalized = NumOps.Add(
                    NumOps.Multiply(input.Data.Span[i], std),
                    mean);
                result.Data.Span[i] = denormalized;
            }
        }

        return result;
    }

    /// <summary>
    /// Shifts the input window and appends new predictions for autoregressive forecasting.
    /// </summary>
    /// <param name="input">Current input tensor.</param>
    /// <param name="prediction">New prediction to append.</param>
    /// <returns>Updated input tensor for next iteration.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In autoregressive forecasting, we slide the input window
    /// forward in time, dropping old data and adding our predictions as new input.
    /// </para>
    /// </remarks>
    private Tensor<T> ShiftAndAppend(Tensor<T> input, Tensor<T> prediction)
    {
        var result = new Tensor<T>(input.Shape);
        int seqLen = _sequenceLength;
        int predLen = Math.Min(_predictionHorizon, seqLen);

        // Shift existing data left
        for (int t = 0; t < seqLen - predLen; t++)
        {
            for (int f = 0; f < _numFeatures; f++)
            {
                int srcIdx = (t + predLen) * _numFeatures + f;
                int dstIdx = t * _numFeatures + f;
                if (srcIdx < input.Length && dstIdx < result.Length)
                {
                    result.Data.Span[dstIdx] = input.Data.Span[srcIdx];
                }
            }
        }

        // Append prediction
        for (int t = 0; t < predLen; t++)
        {
            for (int f = 0; f < _numFeatures; f++)
            {
                int srcIdx = t * _numFeatures + f;
                int dstIdx = (seqLen - predLen + t) * _numFeatures + f;
                if (srcIdx < prediction.Length && dstIdx < result.Length)
                {
                    result.Data.Span[dstIdx] = prediction.Data.Span[srcIdx];
                }
            }
        }

        return result;
    }

    #endregion

    #region Helper Methods

    /// <summary>
    /// Converts our tensor to a float array for ONNX.
    /// </summary>
    /// <param name="tensor">Input tensor.</param>
    /// <returns>Float array.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> ONNX models expect data in a specific format (float arrays).
    /// This method converts our generic tensor to that format.
    /// </para>
    /// </remarks>
    private float[] ConvertToFloatArray(Tensor<T> tensor)
    {
        var result = new float[tensor.Length];
        for (int i = 0; i < tensor.Length; i++)
        {
            result[i] = (float)NumOps.ToDouble(tensor.Data.Span[i]);
        }
        return result;
    }

    /// <summary>
    /// Converts an ONNX tensor back to our tensor type.
    /// </summary>
    /// <param name="onnxTensor">ONNX tensor.</param>
    /// <returns>Converted tensor.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> After ONNX inference, we need to convert the output
    /// back to our tensor format for further processing or returning to the user.
    /// </para>
    /// </remarks>
    private Tensor<T> ConvertFromOnnxTensor(OnnxTensors.Tensor<float> onnxTensor)
    {
        var shape = onnxTensor.Dimensions.ToArray();
        var result = new Tensor<T>(shape);
        var span = onnxTensor.ToArray();

        for (int i = 0; i < span.Length; i++)
        {
            result.Data.Span[i] = NumOps.FromDouble(span[i]);
        }

        return result;
    }

    #endregion

    #region IDisposable

    /// <summary>
    /// Disposes of managed resources.
    /// </summary>
    /// <param name="disposing">Whether this is a managed dispose.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Proper disposal frees up memory and resources.
    /// The ONNX session in particular holds unmanaged resources that need cleanup.
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

