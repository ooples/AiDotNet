using AiDotNet.AutoML;
using AiDotNet.Enums;
using AiDotNet.Finance.Forecasting.Neural;
using AiDotNet.Finance.Forecasting.Transformers;
using AiDotNet.Finance.Risk;
using AiDotNet.Interfaces;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tensors;
using AiDotNet.Validation;

namespace AiDotNet.Finance.AutoML;

/// <summary>
/// Creates finance models for AutoML trials.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// This factory centralizes finance model construction so AutoML can create candidates
/// consistently without exposing internal configuration details.
/// </para>
/// <para>
/// <b>For Beginners:</b> AutoML needs a way to build many models quickly.
/// This class is the "model builder" for finance-specific candidates.
/// </para>
/// </remarks>
internal sealed class FinancialModelFactory<T>
{
    private readonly NeuralNetworkArchitecture<T> _architecture;

    // What every forecasting candidate is built from: the given architecture when it states a time x features
    // grid, otherwise a grid of InputSize time steps by one feature. The candidates read a flat InputSize two ways
    // (PatchTST, iTransformer and N-BEATS as the window, DeepAR and TFT as the feature count), and one search
    // feeds them all the same tensors, so they must all see the same [batch, time steps, features] layout.
    private readonly NeuralNetworkArchitecture<T> _sequenceArchitecture;
    private readonly int _sequenceLength;
    private readonly int _sequenceFeatures;

    /// <summary>
    /// Initializes the factory with a user-provided architecture.
    /// </summary>
    /// <param name="architecture">The architecture to reuse across models.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The architecture is provided by you so the library does not
    /// leak internal design details.
    /// </para>
    /// </remarks>
    public FinancialModelFactory(NeuralNetworkArchitecture<T> architecture)
    {
        Guard.NotNull(architecture);
        _architecture = architecture;

        int height = architecture.InputHeight, width = architecture.InputWidth;
        if (height > 1 && width > 0 && height != width)
        {
            _sequenceArchitecture = architecture;
            _sequenceLength = height;
            _sequenceFeatures = width;
        }
        else
        {
            _sequenceLength = architecture.InputSize;
            _sequenceFeatures = 1;
            _sequenceArchitecture = _sequenceLength > 1
                ? new NeuralNetworkArchitecture<T>(InputType.TwoDimensional, architecture.TaskType,
                    architecture.Complexity, inputHeight: _sequenceLength, inputWidth: 1,
                    outputSize: architecture.OutputSize)
                : architecture;
        }
    }

    /// <summary>
    /// Creates a finance model instance for the requested model type.
    /// </summary>
    /// <param name="modelType">The candidate model type.</param>
    /// <param name="parameters">AutoML parameters for the trial.</param>
    /// <returns>The created model.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> AutoML picks a model type and optional settings,
    /// then this method builds that model for training and evaluation.
    /// </para>
    /// </remarks>
    public IFullModel<T, Tensor<T>, Tensor<T>> Create(Type modelType, IReadOnlyDictionary<string, object> parameters)
    {
        var lookupType = modelType.IsGenericType ? modelType.GetGenericTypeDefinition() : modelType;

        if (lookupType == typeof(PatchTST<>)) return CreateWithOptions(
            (PatchTSTOptions<T> options) => new PatchTST<T>(_sequenceArchitecture, options: options), new PatchTSTOptions<T>(), parameters);
        if (lookupType == typeof(ITransformer<>)) return CreateWithOptions(
            (ITransformerOptions<T> options) => new ITransformer<T>(_sequenceArchitecture, options: options), new ITransformerOptions<T>(), parameters);
        if (lookupType == typeof(DeepAR<>)) return CreateWithOptions(
            (DeepAROptions<T> options) => new DeepAR<T>(_sequenceArchitecture, options), new DeepAROptions<T>(), parameters);
        if (lookupType == typeof(NBEATSFinance<>)) return CreateWithOptions(
            (NBEATSModelOptions<T> options) => new NBEATSFinance<T>(_sequenceArchitecture, options), new NBEATSModelOptions<T>(), parameters);
        if (lookupType == typeof(TFT<>)) return CreateWithOptions(
            (TemporalFusionTransformerOptions<T> options) => new TFT<T>(_sequenceArchitecture, options), new TemporalFusionTransformerOptions<T>(), parameters);
        if (lookupType == typeof(NeuralVaR<>)) return CreateWithOptions(
            (NeuralVaROptions<T> options) => new NeuralVaR<T>(_architecture, options), new NeuralVaROptions<T>(), parameters);
        if (lookupType == typeof(TabNet<>)) return CreateWithOptions(
            (TabNetOptions<T> options) => new TabNet<T>(_architecture, options), new TabNetOptions<T>(), parameters);
        if (lookupType == typeof(TabTransformer<>)) return CreateWithOptions(
            (TabTransformerOptions<T> options) => new TabTransformer<T>(_architecture, options), new TabTransformerOptions<T>(), parameters);

        throw new NotSupportedException(
            $"Finance AutoML does not support model type '{modelType.Name}'.");
    }

    /// <summary>
    /// Applies AutoML parameters to options and builds the model.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> AutoML samples settings (like hidden size),
    /// applies them to the options object, and then constructs the model.
    /// </para>
    /// </remarks>
    private TModel CreateWithOptions<TOptions, TModel>(
        Func<TOptions, TModel> factory,
        TOptions options,
        IReadOnlyDictionary<string, object> parameters)
    {
        if (factory is null)
            throw new ArgumentNullException(nameof(factory));
        if (options is null)
            throw new ArgumentNullException(nameof(options));

        SeedShapeFromArchitecture(options);
        AutoMLHyperparameterApplicator.ApplyToOptions(options, ToOptionNames(typeof(TModel), parameters));
        FitSampledSizesToShape(options);
        return factory(options);
    }

    // The search spaces use one vocabulary for every model (HiddenSize, DropoutRate, NumAttentionHeads, ...), but
    // the applicator sets an option only when a property has exactly that name and skips the rest silently, so most
    // sampled values never reached the model. Each entry renames a search key to the model's own option, or drops
    // it (null) where the model has no such option (training-loop settings like Epochs live outside these options).
    private static readonly Dictionary<Type, Dictionary<string, string?>> OptionNames = new()
    {
        [typeof(PatchTST<>)] = new(StringComparer.Ordinal)
        {
            ["HiddenSize"] = "ModelDimension", ["DropoutRate"] = "Dropout", ["PatchLength"] = "PatchSize",
            ["NumAttentionHeads"] = "NumHeads", ["Epochs"] = null, ["BatchSize"] = null,
        },
        [typeof(ITransformer<>)] = new(StringComparer.Ordinal)
        {
            ["HiddenSize"] = "ModelDimension", ["DropoutRate"] = "Dropout", ["NumAttentionHeads"] = "NumHeads",
            ["FeedForwardDim"] = "FeedForwardDimension", ["Epochs"] = null, ["BatchSize"] = null,
        },
        [typeof(NBEATSFinance<>)] = new(StringComparer.Ordinal)
        {
            ["HiddenSize"] = "HiddenLayerSize", ["NumLayers"] = "NumHiddenLayers", ["DropoutRate"] = null,
        },
        [typeof(NeuralVaR<>)] = new(StringComparer.Ordinal)
        {
            ["HiddenSize"] = "HiddenDimension", ["NumLayers"] = "HiddenLayers", ["LearningRate"] = null,
            ["DropoutRate"] = null, ["Epochs"] = null, ["BatchSize"] = null,
        },
        [typeof(TabNet<>)] = new(StringComparer.Ordinal)
        {
            ["HiddenSize"] = "FeatureDimension", ["NumLayers"] = "NumSharedLayers", ["Epochs"] = null, ["BatchSize"] = null,
        },
        [typeof(TabTransformer<>)] = new(StringComparer.Ordinal)
        {
            ["HiddenSize"] = "HiddenDimension", ["EmbeddingDim"] = "EmbeddingDimension", ["NumAttentionHeads"] = "NumHeads",
            ["Epochs"] = null, ["BatchSize"] = null,
        },
    };

    /// <summary>The option a search key sets on a model, or null when the model has no such option.</summary>
    internal static string? OptionName(Type modelType, string key)
    {
        var lookupType = modelType.IsGenericType ? modelType.GetGenericTypeDefinition() : modelType;
        return OptionNames.TryGetValue(lookupType, out var names) && names.TryGetValue(key, out var name) ? name : key;
    }

    private static IReadOnlyDictionary<string, object> ToOptionNames(Type modelType, IReadOnlyDictionary<string, object> parameters)
    {
        var renamed = new Dictionary<string, object>(StringComparer.Ordinal);
        foreach (var (key, value) in parameters)
        {
            if (string.Equals(key, "ModelType", StringComparison.Ordinal))
            {
                renamed[key] = value;
                continue;
            }

            if (OptionName(modelType, key) is { } name)
                renamed[name] = value;
        }

        return renamed;
    }

    /// <summary>
    /// Sizes the options from the architecture the search was given. Every candidate validates that its options
    /// agree with the architecture, and the options' own defaults (10 risk features, a 96 x 7 forecasting window)
    /// never matched the architecture, so every trial used to fail before training.
    /// </summary>
    /// <remarks>
    /// Forecasting candidates take the window from the time x features grid (see the sequence-architecture field);
    /// a flat one-dimensional input is a univariate window of InputSize past values, forecasting OutputSize steps.
    /// </remarks>
    private void SeedShapeFromArchitecture(object options)
    {
        int output = Math.Max(1, _architecture.OutputSize);
        switch (options)
        {
            case RiskModelOptions<T> risk when _architecture.InputSize > 0:
                risk.NumFeatures = _architecture.InputSize;
                break;
            case PatchTSTOptions<T> patchTst when _sequenceLength > 0:
                patchTst.SequenceLength = _sequenceLength;
                patchTst.NumFeatures = _sequenceFeatures;
                patchTst.PredictionHorizon = output;
                break;
            case ITransformerOptions<T> iTransformer when _sequenceLength > 0:
                iTransformer.SequenceLength = _sequenceLength;
                iTransformer.NumFeatures = _sequenceFeatures;
                iTransformer.PredictionHorizon = output;
                break;
            case NBEATSModelOptions<T> nbeats when _sequenceLength > 0:
                nbeats.LookbackWindow = _sequenceLength;
                nbeats.ForecastHorizon = output;
                break;
            case DeepAROptions<T> deepAr when _sequenceLength > 0:
                deepAr.LookbackWindow = _sequenceLength;
                deepAr.ForecastHorizon = output;
                break;
            case TemporalFusionTransformerOptions<T> tft when _sequenceLength > 0:
                tft.LookbackWindow = _sequenceLength;
                tft.ForecastHorizon = output;
                break;
        }
    }

    /// <summary>
    /// Keeps a sampled size inside the data's shape where the search space cannot know it: a PatchTST patch
    /// (and its stride) cannot be longer than the lookback window.
    /// </summary>
    private static void FitSampledSizesToShape(object options)
    {
        switch (options)
        {
            case PatchTSTOptions<T> patchTst:
                patchTst.PatchSize = Math.Min(patchTst.PatchSize, patchTst.SequenceLength);
                patchTst.Stride = Math.Max(1, Math.Min(patchTst.Stride, patchTst.PatchSize));
                patchTst.ModelDimension = SplitEvenly(patchTst.ModelDimension, patchTst.NumHeads);
                break;
            case ITransformerOptions<T> iTransformer:
                iTransformer.ModelDimension = SplitEvenly(iTransformer.ModelDimension, iTransformer.NumHeads);
                break;
            case TemporalFusionTransformerOptions<T> tft:
                tft.HiddenSize = SplitEvenly(tft.HiddenSize, tft.NumAttentionHeads);
                break;
            case TabTransformerOptions<T> tabTransformer:
                tabTransformer.HiddenDimension = SplitEvenly(tabTransformer.HiddenDimension, tabTransformer.NumHeads);
                tabTransformer.EmbeddingDimension = SplitEvenly(tabTransformer.EmbeddingDimension, tabTransformer.NumHeads);
                break;
        }
    }

    // Width and head count are sampled independently (an integer range drew 73 with 8 heads), and attention splits
    // the width evenly across the heads, so the sampled width is rounded up to the next multiple of the heads.
    private static int SplitEvenly(int width, int heads) =>
        heads <= 0 || width % heads == 0 ? width : (width / heads + 1) * heads;
}
