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
    public IFullModel<T, Tensor<T>, Tensor<T>> Create(ModelType modelType, IReadOnlyDictionary<string, object> parameters)
    {
        return modelType switch
        {
            ModelType.PatchTST => new PatchTST<T>(_architecture),
            ModelType.ITransformer => new ITransformer<T>(_architecture),
            ModelType.DeepAR => CreateWithOptions(
                (DeepAROptions<T> options) => new DeepAR<T>(_architecture, options),
                new DeepAROptions<T>(),
                parameters),
            ModelType.NBEATS => CreateWithOptions(
                (NBEATSModelOptions<T> options) => new NBEATSFinance<T>(_architecture, options),
                new NBEATSModelOptions<T>(),
                parameters),
            ModelType.TFT => CreateWithOptions(
                (TemporalFusionTransformerOptions<T> options) => new TFT<T>(_architecture, options),
                new TemporalFusionTransformerOptions<T>(),
                parameters),
            ModelType.NeuralVaR => CreateWithOptions(
                (NeuralVaROptions<T> options) => new NeuralVaR<T>(_architecture, options),
                new NeuralVaROptions<T>(),
                parameters),
            ModelType.TabNet => CreateWithOptions(
                (TabNetOptions<T> options) => new TabNet<T>(_architecture, options),
                new TabNetOptions<T>(),
                parameters),
            ModelType.TabTransformer => CreateWithOptions(
                (TabTransformerOptions<T> options) => new TabTransformer<T>(_architecture, options),
                new TabTransformerOptions<T>(),
                parameters),
            _ => throw new NotSupportedException(
                $"Finance AutoML does not support model type '{modelType}'.")
        };
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
    private static TModel CreateWithOptions<TOptions, TModel>(
        Func<TOptions, TModel> factory,
        TOptions options,
        IReadOnlyDictionary<string, object> parameters)
    {
        if (factory is null)
            throw new ArgumentNullException(nameof(factory));
        if (options is null)
            throw new ArgumentNullException(nameof(options));

        AutoMLHyperparameterApplicator.ApplyToOptions(options, parameters);
        return factory(options);
    }
}
