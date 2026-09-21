using System.IO;
using AiDotNet.Attributes;
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
/// <example>
/// <code>
/// var architecture = new NeuralNetworkArchitecture&lt;double&gt;(
///     inputType: InputType.OneDimensional,
///     taskType: NeuralNetworkTaskType.Regression,
///     inputHeight: 60, inputWidth: 1, inputDepth: 1, outputSize: 24);
/// var model = new DeepAR&lt;double&gt;(architecture);
/// var onnxModel = new DeepAR&lt;double&gt;(architecture, "deepar.onnx");
/// </code>
/// </example>
[ModelDomain(ModelDomain.Finance)]
[ModelDomain(ModelDomain.TimeSeries)]
[ModelCategory(ModelCategory.NeuralNetwork)]
[ModelCategory(ModelCategory.RecurrentNetwork)]
[ModelTask(ModelTask.Forecasting)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("DeepAR: Probabilistic Forecasting with Autoregressive Recurrent Networks", "https://arxiv.org/abs/1704.04110", Year = 2020, Authors = "David Salinas, Valentin Flunkert, Jan Gasthaus, Tim Januschowski")]
public partial class DeepAR<T> : ForecastingModelBase<T>, ITrainingObjectiveProvider<T>
{
    #region Native Mode Fields

    /// <summary>
    /// The recurrent trunk: every layer ahead of the two distribution heads, applied in order.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> LSTM (Long Short-Term Memory) layers are recurrent neural networks
    /// that can learn patterns across time. They are good at remembering important information
    /// from the past while forgetting irrelevant details. Any dropout layers sitting between them
    /// live in this list too, so the trunk is simply run front to back.
    /// </para>
    /// <para>
    /// This is a single ordered list rather than named input-projection / LSTM / layer-norm fields
    /// because the previous split assigned layers by fixed INDEX: layer 0 was taken to be an input
    /// projection and the layer before the heads to be a normalization, whatever a caller-supplied
    /// custom stack actually contained. Running the trunk in order is both correct for a custom
    /// stack and an exact match for DeepAR (Salinas et al. 2020) section 3.1, where the hidden state
    /// of a plain stacked LSTM feeds the heads with nothing in between.
    /// </para>
    /// </remarks>
    private readonly List<ILayer<T>> _trunkLayers = [];

    /// <summary>
    /// Whether training maximizes the paper's Gaussian log-likelihood rather than a
    /// caller-supplied loss on the mean head.
    /// </summary>
    /// <remarks>
    /// DeepAR (Salinas et al. 2020) eq. 2 fits the network by maximizing
    /// sum_i sum_t log l(z_i,t | theta(h_i,t)), where theta is the (mu, sigma) pair of eq. 3.
    /// That is the default here. A caller who passes an explicit loss function gets that loss
    /// on the mean head instead, so the paper's objective is the default rather than a
    /// hardcoded one.
    /// </remarks>
    private readonly bool _usePaperLikelihood;

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
    [Scratch]
    private Tensor<T>? _lastSigma;

    /// <summary>
    /// Instance normalization scale for denormalization.
    /// </summary>
    [Scratch]
    private Tensor<T>? _scaleStd;

    #endregion

    #region Shared Fields

    /// <summary>
    /// The optimizer for training.
    /// </summary>
    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> _optimizer;
    private readonly DeepAROptions<T> _options;

    /// <inheritdoc/>
    public override ModelOptions GetOptions() => _options;

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
    /// Creates a DeepAR model with default configuration for native training.
    /// </summary>
    public DeepAR()
        : this(new NeuralNetworkArchitecture<T>(
            inputType: InputType.OneDimensional,
            taskType: NeuralNetworkTaskType.Regression,
            inputSize: 1,
            outputSize: 1))
    {
    }

    /// <summary>
    /// Returns the caller's options, or a default instance, so every constructor reads its
    /// lookback and horizon from one source.
    /// </summary>
    /// <remarks>
    /// The base initializers previously carried their own literals - <c>options?.LookbackWindow ?? 96</c>
    /// and <c>options?.ForecastHorizon ?? 24</c> - while the body fell back to <c>new DeepAROptions&lt;T&gt;()</c>,
    /// whose own defaults are 30 and 7. A caller passing no options therefore got a base configured for a
    /// 96-step lookback and a 24-step horizon wrapped around an options object that reported 30 and 7.
    /// </remarks>
    private static DeepAROptions<T> OptionsOrDefault(DeepAROptions<T>? options) => options ?? new DeepAROptions<T>();

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
               OptionsOrDefault(options).LookbackWindow,
               OptionsOrDefault(options).ForecastHorizon,
               architecture.InputSize)
    {
        options ??= new DeepAROptions<T>();
        _options = options;
        Options = _options;
        ValidateOptions(options);

        // The caller's optimizer has to reach the tape trainer, not just a private field.
        // TrainWithTape resolves _baseTrainOptimizer; without SetBaseTrainOptimizer an
        // optimizer passed here is silently discarded and training runs on the base Adam
        // default instead - measured at lr 1e-5 and lr 0, where the trajectory was
        // unchanged. The self-adopting `new AdamOptimizer(this)` masked it whenever the
        // caller passed nothing. Same wiring Chronos, MOIRAI, SimMTM, TOTEM and TOTO use.
        // DeepAR (Salinas et al. 2020) section 4: Adam at 1e-3, which is DeepAROptions.LearningRate's
        // default - but a caller who changes it must actually get the rate they asked for.
        _optimizer = optimizer ?? new AdamOptimizer<T, Tensor<T>, Tensor<T>>(
            this,
            new AdamOptimizerOptions<T, Tensor<T>, Tensor<T>> { InitialLearningRate = options.LearningRate });
        SetBaseTrainOptimizer(_optimizer);

        _hiddenSize = options.HiddenSize;
        _numLstmLayers = options.NumLayers;
        _embeddingDim = options.EmbeddingDimension;
        _dropout = options.DropoutRate;
        _distributionType = options.LikelihoodType;
        _numSamples = options.NumSamples;
        _useScaling = true;
        _usePaperLikelihood = lossFunction is null;

        // Honour the configured seed. DeepAR (Salinas et al. 2020) section 4 draws 200 samples from the
        // decoder to form its forecast, so this RNG decides the prediction, not just an initialization
        // detail: an unconditionally secure RNG made Predict irreproducible for a caller who set Seed.
        _random = options.Seed.HasValue
            ? RandomHelper.CreateSeededRandom(options.Seed.Value)
            : RandomHelper.CreateSecureRandom();

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
               OptionsOrDefault(options).LookbackWindow,
               OptionsOrDefault(options).ForecastHorizon,
               architecture.InputSize,
               lossFunction)
    {
        options ??= new DeepAROptions<T>();
        _options = options;
        Options = _options;
        ValidateOptions(options);

        // The caller's optimizer has to reach the tape trainer, not just a private field.
        // TrainWithTape resolves _baseTrainOptimizer; without SetBaseTrainOptimizer an
        // optimizer passed here is silently discarded and training runs on the base Adam
        // default instead - measured at lr 1e-5 and lr 0, where the trajectory was
        // unchanged. The self-adopting `new AdamOptimizer(this)` masked it whenever the
        // caller passed nothing. Same wiring Chronos, MOIRAI, SimMTM, TOTEM and TOTO use.
        // DeepAR (Salinas et al. 2020) section 4: Adam at 1e-3, which is DeepAROptions.LearningRate's
        // default - but a caller who changes it must actually get the rate they asked for.
        _optimizer = optimizer ?? new AdamOptimizer<T, Tensor<T>, Tensor<T>>(
            this,
            new AdamOptimizerOptions<T, Tensor<T>, Tensor<T>> { InitialLearningRate = options.LearningRate });
        SetBaseTrainOptimizer(_optimizer);

        _hiddenSize = options.HiddenSize;
        _numLstmLayers = options.NumLayers;
        _embeddingDim = options.EmbeddingDimension;
        _dropout = options.DropoutRate;
        _distributionType = options.LikelihoodType;
        _numSamples = options.NumSamples;
        _useScaling = true;
        _usePaperLikelihood = lossFunction is null;

        // Honour the configured seed. DeepAR (Salinas et al. 2020) section 4 draws 200 samples from the
        // decoder to form its forecast, so this RNG decides the prediction, not just an initialization
        // detail: an unconditionally secure RNG made Predict irreproducible for a caller who set Seed.
        _random = options.Seed.HasValue
            ? RandomHelper.CreateSeededRandom(options.Seed.Value)
            : RandomHelper.CreateSecureRandom();

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
            ValidateCustomLayersWithOwnershipRollback(Layers);
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
        _trunkLayers.Clear();
        _muProjection = null;
        _sigmaProjection = null;

        // The last two layers are always the mean and standard-deviation heads (paper eq. 3);
        // everything ahead of them is the recurrent trunk, applied in order. Walking back from the
        // END rather than counting forward from index 0 keeps a caller-supplied custom stack intact:
        // the previous forward walk consumed a fixed "input projection" slot and a fixed
        // "layer normalization" slot whether or not the stack held such layers, so a custom stack of
        // a different shape had its layers silently bound to the wrong roles.
        int headCount = System.Math.Min(2, Layers.Count);
        int trunkCount = Layers.Count - headCount;

        for (int i = 0; i < trunkCount; i++)
        {
            _trunkLayers.Add(Layers[i]);
        }

        if (headCount == 2)
        {
            _muProjection = Layers[trunkCount];
            _sigmaProjection = Layers[trunkCount + 1];
        }
        else if (headCount == 1)
        {
            _muProjection = Layers[0];
        }
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

        // Only Gaussian is implemented in native mode (mu/sigma heads only)
        var validDistributions = new[] { "Gaussian", "gaussian" };
        if (!validDistributions.Contains(options.LikelihoodType, StringComparer.OrdinalIgnoreCase))
            errors.Add("LikelihoodType must be Gaussian in native mode (StudentT and NegativeBinomial not yet implemented).");

        if (errors.Count > 0)
            throw new ArgumentException($"Invalid options: {string.Join(", ", errors)}");
    }

    #endregion

    #region NeuralNetworkBase Overrides


    // UpdateParameters was an empty override, silently dropping every restore. The base
    // distributes the vector over the declared enumeration.
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
    /// Writes DeepAR-specific configuration during serialization.
    /// </summary>
    /// <param name="writer">Binary writer for output.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This saves DeepAR settings like hidden size and distribution type
    /// to a file so the model can be loaded later with the same configuration.
    /// </para>
    /// </remarks>


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
        // Point forecast = the distribution mean from the layer stack, computed
        // identically to the training-path forward (ForwardNativeForTraining also
        // returns Forward(input)). DeepAR (Salinas et al. 2020): the point
        // forecast is the predicted mean.
        //
        // The previous ApplyScaling → base.Forecast → ReverseScaling round-trip
        // is intentionally NOT used on the point-forecast path: it mutated the
        // per-instance _scaleStd state, so a model and its clone diverged on the
        // SAME input (Clone_ShouldProduceIdenticalOutput: original=0 vs
        // clone=0.16), and it was applied on inference but not on training — an
        // inconsistency. The mean here is fully deterministic and depends only on
        // the (cloned-faithfully) layer weights.
        var mean = Forward(historicalData);

        // Probabilistic forecast: sample the requested quantiles from the
        // predicted Gaussian (mean + the sigma head cached during Forward).
        if (quantiles is not null && quantiles.Length > 0)
        {
            return SampleQuantiles(mean, quantiles);
        }

        return mean;
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

        // Recurrent trunk, front to back (paper section 3.1: h_{i,t} = RNN(h_{i,t-1}, z_{i,t-1}, x_{i,t})).
        foreach (var layer in _trunkLayers)
        {
            current = layer.Forward(current);
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
    /// Tape-connected forward used by <see cref="NeuralNetworks.NeuralNetworkBase{T}.TrainWithTape"/>.
    /// </summary>
    /// <remarks>
    /// The base <c>ForwardNativeForTraining</c> routes through <see cref="Forecast"/>, whose
    /// <c>ApplyScaling</c> / sampling steps build new tensors by manual indexing — that detaches
    /// the autodiff graph, so the gradient tape saw a constant and no weight gradients ever flowed
    /// (params never changed, loss never moved). Training instead runs the differentiable layer
    /// stack straight to the distribution mean head (<see cref="Forward"/>): per DeepAR (Salinas
    /// et al. 2020) the network is fit by maximizing the observed series' likelihood under the
    /// predicted distribution, and the mean head is the tape-connected quantity the optimizer
    /// backpropagates through.
    /// </remarks>
    protected override Tensor<T> ForwardNativeForTraining(Tensor<T> input) => Forward(input);

    /// <summary>
    /// Runs one gradient step against DeepAR's paper objective.
    /// </summary>
    /// <param name="input">The conditioning range.</param>
    /// <param name="expectedOutput">The observed series the likelihood is evaluated on.</param>
    /// <remarks>
    /// <para>
    /// Paper eq. 2 maximizes the log-likelihood of the observed series under the predicted
    /// distribution, so the gradient reaches BOTH heads of eq. 3. The default supervised step
    /// backpropagated a squared error on the mean alone: the sigma head ran on every forward
    /// pass, was counted by ParameterCount and was serialized, yet received exactly zero
    /// gradient - a registered head that never trained, leaving the predictive interval at
    /// whatever initialization produced.
    /// </para>
    /// <para>
    /// A caller who supplied an explicit loss function keeps it, on the mean head, through the
    /// ordinary supervised path.
    /// </para>
    /// </remarks>
    protected override void RunTrainingStep(Tensor<T> input, Tensor<T> expectedOutput)
    {
        if (!_usePaperLikelihood)
        {
            base.RunTrainingStep(input, expectedOutput);
            return;
        }

        // _lastSigma is written by Forward, which TrainWithCustomLoss has already run on the tape
        // by the time this callback is invoked, so the sigma read here is the tape-connected one.
        TrainWithCustomLoss(
            input,
            mu => GaussianNegativeLogLikelihood(mu, _lastSigma, expectedOutput),
            _optimizer);
    }

    /// <summary>
    /// Mean Gaussian negative log-likelihood of <paramref name="target"/> under N(mu, sigma^2),
    /// as a recorded scalar the tape can follow.
    /// </summary>
    /// <param name="mu">The mean head's output (paper eq. 3, affine).</param>
    /// <param name="sigma">The standard-deviation head's output (paper eq. 3, softplus).</param>
    /// <param name="target">The observed series.</param>
    /// <remarks>
    /// -log l(z | mu, sigma) = log(sigma) + 0.5*log(2*pi) + (z - mu)^2 / (2*sigma^2).
    /// Softplus keeps sigma positive but can still underflow to zero in float, so a small floor
    /// is added before the division and the logarithm. When the sigma head is absent or shaped
    /// differently from the mean - a caller-supplied custom stack may emit either - this falls
    /// back to the mean squared error, which is the sigma-constant special case of the same
    /// objective up to an additive constant.
    /// </remarks>
    private Tensor<T> GaussianNegativeLogLikelihood(Tensor<T> mu, Tensor<T>? sigma, Tensor<T> target)
    {
        if (sigma is null || sigma.Length != mu.Length)
        {
            return MeanSquaredDifference(mu, target);
        }

        var safeSigma = Engine.TensorAddScalar(sigma, NumOps.FromDouble(SigmaFloor));
        var standardized = Engine.TensorDivide(Engine.TensorSubtract(mu, target), safeSigma);
        var quadratic = Engine.TensorMultiplyScalar(
            Engine.TensorMultiply(standardized, standardized), NumOps.FromDouble(0.5));
        var perElement = Engine.TensorAddScalar(
            Engine.TensorAdd(quadratic, Engine.TensorLog(safeSigma)),
            NumOps.FromDouble(0.5 * System.Math.Log(2.0 * System.Math.PI)));

        return MeanOverAllAxes(perElement);
    }

    /// <summary>Mean squared difference, as a recorded scalar the tape can follow.</summary>
    private Tensor<T> MeanSquaredDifference(Tensor<T> predicted, Tensor<T> target)
    {
        var difference = Engine.TensorSubtract(predicted, target);
        return MeanOverAllAxes(Engine.TensorMultiply(difference, difference));
    }

    /// <summary>Reduces every axis of <paramref name="tensor"/> to a scalar mean.</summary>
    private Tensor<T> MeanOverAllAxes(Tensor<T> tensor)
    {
        var allAxes = System.Linq.Enumerable.Range(0, tensor.Shape.Length).ToArray();
        return Engine.ReduceMean(tensor, allAxes, keepDims: false);
    }

    /// <summary>
    /// Floor added to the softplus standard deviation before it is divided by or logged.
    /// </summary>
    private const double SigmaFloor = 1e-6;

    /// <inheritdoc/>
    /// <remarks>
    /// DeepAR's target is the caller's observed series; only the objective differs from ordinary
    /// squared error, so the proposed target passes through unchanged.
    /// </remarks>
    TrainingObjectiveKind ITrainingObjectiveProvider<T>.TrainingObjectiveKind
        => TrainingObjectiveKind.Supervised;

    /// <inheritdoc/>
    Tensor<T> ITrainingObjectiveProvider<T>.ResolveTrainingTarget(Tensor<T> input, Tensor<T> proposedTarget)
        => proposedTarget;

    /// <inheritdoc/>
    /// <remarks>
    /// Reports the same Gaussian negative log-likelihood the optimizer descends, so a caller
    /// judging whether training improved measures the objective training actually minimized
    /// rather than a squared error on the mean that the sigma head does not appear in.
    /// </remarks>
    T ITrainingObjectiveProvider<T>.EvaluateTrainingObjective(Tensor<T> input, Tensor<T> target)
    {
        var mu = Forward(input);
        var objective = _usePaperLikelihood
            ? GaussianNegativeLogLikelihood(mu, _lastSigma, target)
            : MeanSquaredDifference(mu, target);
        return objective.Length > 0 ? objective[0] : NumOps.Zero;
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

        var scaled = new Tensor<T>(input._shape);
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

        var unscaled = new Tensor<T>(output._shape);

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
        int batchSize = forecast.Shape[0];
        int seqLen = forecast.Shape.Length > 1 ? forecast.Shape[1] : 1;
        int features = forecast.Shape.Length > 2 ? forecast.Shape[2] : 1;
        int numQuantiles = quantiles.Length;

        // Output shape includes features dimension to preserve multivariate outputs
        var quantileForecast = new Tensor<T>(new[] { batchSize, seqLen, features * numQuantiles });

        for (int b = 0; b < batchSize; b++)
        {
            for (int t = 0; t < seqLen; t++)
            {
                for (int f = 0; f < features; f++)
                {
                    // Feature-aware indexing
                    int muIdx = b * seqLen * features + t * features + f;
                    T mu = muIdx < forecast.Length ? forecast.Data.Span[muIdx] : NumOps.Zero;

                    // Use learned sigma from Forward if available, otherwise estimate
                    T sigma;
                    if (_lastSigma is not null && muIdx < _lastSigma.Length)
                    {
                        sigma = _lastSigma.Data.Span[muIdx];
                        // If scaling was applied, unscale sigma to match unscaled mu
                        if (_useScaling && _scaleStd is not null)
                        {
                            int scaleIdx = b * features + f;
                            if (scaleIdx < _scaleStd.Length)
                            {
                                sigma = NumOps.Multiply(sigma, _scaleStd.Data.Span[scaleIdx]);
                            }
                        }
                    }
                    else
                    {
                        // Fallback: estimate sigma as 10% of mu
                        sigma = NumOps.Multiply(NumOps.Abs(mu), NumOps.FromDouble(0.1));
                    }

                    // Ensure sigma is positive with a minimum floor
                    sigma = NumOps.Add(NumOps.Abs(sigma), NumOps.FromDouble(0.01));

                    for (int q = 0; q < numQuantiles; q++)
                    {
                        // Standard normal quantile (approximation)
                        double z = GetStandardNormalQuantile(quantiles[q]);
                        T quantileValue = NumOps.Add(mu, NumOps.Multiply(sigma, NumOps.FromDouble(z)));

                        int outIdx = b * seqLen * features * numQuantiles + t * features * numQuantiles + f * numQuantiles + q;
                        if (outIdx < quantileForecast.Length)
                            quantileForecast.Data.Span[outIdx] = quantileValue;
                    }
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

        var shifted = new Tensor<T>(input._shape);

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

                // Add new predictions - use prediction's actual sequence length, not stepsUsed
                int predSeqLen = prediction.Shape.Length > 1 ? prediction.Shape[1] : 1;
                for (int t = seqLen - stepsUsed; t < seqLen; t++)
                {
                    int predT = t - (seqLen - stepsUsed);
                    int predIdx = b * predSeqLen * features + predT * features + f;
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
