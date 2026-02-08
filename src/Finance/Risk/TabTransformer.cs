using AiDotNet.Finance.Base;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LossFunctions;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Optimizers;

namespace AiDotNet.Finance.Risk;

/// <summary>
/// TabTransformer model for tabular data, using transformers for categorical features.
/// </summary>
/// <typeparam name="T">The numeric type.</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> TabTransformer applies the power of Transformers (like BERT/GPT) specifically
/// to the categorical parts of your data (e.g., "Sector", "Country", "Rating"). It learns deep
/// contextual embeddings for these categories before combining them with numerical data.
/// </para>
/// </remarks>
public class TabTransformer<T> : RiskModelBase<T>
{
    #region Shared Fields

    private readonly TabTransformerOptions<T> _options;

    /// <inheritdoc/>
    public override ModelOptions GetOptions() => _options;

    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> _optimizer;
    private readonly ILossFunction<T> _lossFunction;

    #endregion

    /// <summary>
    /// Initializes a new instance of the TabTransformer model.
    /// </summary>
    /// <param name="architecture">The neural network architecture configuration.</param>
    /// <param name="options">Configuration options for the model.</param>
    /// <param name="optimizer">Optional optimizer for training.</param>
    /// <param name="lossFunction">Optional loss function.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Use this constructor when training TabTransformer from scratch.
    /// It sets up the transformer layers that specialize in understanding categorical features.
    /// </para>
    /// <para>
    /// If you already defined custom layers in <paramref name="architecture"/>,
    /// those layers will be used instead of the default TabTransformer layers.
    /// </para>
    /// </remarks>
    public TabTransformer(
        NeuralNetworkArchitecture<T> architecture,
        TabTransformerOptions<T>? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
        ILossFunction<T>? lossFunction = null)
        : base(
            architecture,
            options?.NumFeatures ?? 50,
            options?.ConfidenceLevel ?? 0.95,
            options?.TimeHorizon ?? 1,
            lossFunction)
    {
        _options = options ?? new TabTransformerOptions<T>();
        Options = _options;
        _lossFunction = lossFunction ?? new MeanSquaredErrorLoss<T>();
        _optimizer = optimizer ?? new AdamOptimizer<T, Tensor<T>, Tensor<T>>(this);

        InitializeLayers();
    }

    /// <summary>
    /// Initializes a new instance from ONNX.
    /// </summary>
    /// <param name="architecture">The neural network architecture configuration.</param>
    /// <param name="onnxModelPath">Path to the ONNX model file.</param>
    /// <param name="options">Configuration options for the model.</param>
    /// <param name="optimizer">Optional optimizer for fine-tuning.</param>
    /// <param name="lossFunction">Optional loss function.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Use this constructor to load a pretrained TabTransformer
    /// from an ONNX file. This is best for inference (predictions) rather than training.
    /// </para>
    /// </remarks>
    public TabTransformer(
        NeuralNetworkArchitecture<T> architecture,
        string onnxModelPath,
        TabTransformerOptions<T>? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
        ILossFunction<T>? lossFunction = null)
        : base(
            architecture,
            onnxModelPath,
            options?.NumFeatures ?? 50,
            options?.ConfidenceLevel ?? 0.95,
            options?.TimeHorizon ?? 1)
    {
        _options = options ?? new TabTransformerOptions<T>();
        Options = _options;
        _lossFunction = lossFunction ?? new MeanSquaredErrorLoss<T>();
        _optimizer = optimizer ?? new AdamOptimizer<T, Tensor<T>, Tensor<T>>(this);

        InitializeLayers();
    }

    #region Initialization

    /// <summary>
    /// Initializes the neural network layers for TabTransformer.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> TabTransformer uses attention to learn how categories
    /// relate to each other (like sector and country). This method builds that
    /// attention pipeline using default layers unless you provided custom ones.
    /// </para>
    /// </remarks>
    protected override void InitializeLayers()
    {
        if (Architecture.Layers is not null && Architecture.Layers.Count > 0)
        {
            Layers.AddRange(Architecture.Layers);
            ValidateCustomLayers(Layers);
        }
        else if (UseNativeMode)
        {
            Layers.AddRange(LayerHelper<T>.CreateDefaultTabTransformerLayers(
                Architecture,
                _options.NumFeatures,
                _options.HiddenDimension,
                _options.NumHeads,
                _options.NumLayers,
                _options.NumCategoricalFeatures,
                1,
                _options.DropoutRate));
        }
    }

    #endregion

    /// <summary>
    /// Calculates risk using TabTransformer.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This runs the model forward and extracts a single
    /// risk number. A higher value means higher risk.
    /// </para>
    /// </remarks>
    public override T CalculateRisk(Tensor<T> input)
    {
        var prediction = Predict(input);
        return NumOps.Abs(prediction.ToVector()[0]);
    }

    /// <summary>
    /// Adjusts action for risk.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> If the risk is too high, this scales the action down
    /// so it fits within your allowed budget.
    /// </para>
    /// </remarks>
    public override Tensor<T> AdjustForRisk(Tensor<T> action, T riskBudget)
    {
        var risk = CalculateRisk(action);
        if (NumOps.GreaterThan(risk, riskBudget))
        {
            return action.Multiply(NumOps.Divide(riskBudget, risk));
        }
        return action;
    }

    #region NeuralNetworkBase Overrides

    /// <summary>
    /// Updates the model parameters from a flat parameter vector.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This lets you load or apply all weights at once,
    /// which is useful when cloning or restoring the model.
    /// </para>
    /// </remarks>
    public override void UpdateParameters(Vector<T> parameters)
    {
        int offset = 0;
        foreach (var layer in Layers)
        {
            var layerParams = layer.GetParameters();
            layer.SetParameters(parameters.Slice(offset, layerParams.Length));
            offset += layerParams.Length;
        }
    }

    /// <summary>
    /// Creates a new instance of the TabTransformer model with the same configuration.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This is used by the framework to clone the model setup
    /// so it can create a fresh instance with identical settings.
    /// </para>
    /// </remarks>
    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance()
    {
        var options = new TabTransformerOptions<T>
        {
            NumFeatures = _options.NumFeatures,
            ConfidenceLevel = _options.ConfidenceLevel,
            TimeHorizon = _options.TimeHorizon,
            HiddenDimension = _options.HiddenDimension,
            NumHeads = _options.NumHeads,
            NumLayers = _options.NumLayers,
            NumCategoricalFeatures = _options.NumCategoricalFeatures,
            DropoutRate = _options.DropoutRate
        };

        return new TabTransformer<T>(Architecture, options, _optimizer, LossFunction);
    }

    #endregion
}
