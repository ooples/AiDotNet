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
/// SAINT (Self-Attention and Intersample Attention Transformer) for tabular data.
/// </summary>
/// <typeparam name="T">The numeric type.</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> SAINT improves on standard transformers for tabular data by paying attention
/// not just to the relationships between columns (features) but also between rows (samples).
/// This allows it to learn from similar examples in the batch, much like a nearest-neighbor approach
/// but fully learnable.
/// </para>
/// </remarks>
public class SAINT<T> : RiskModelBase<T>
{
    #region Shared Fields

    private readonly SAINTOptions<T> _options;
    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> _optimizer;
    private readonly ILossFunction<T> _lossFunction;

    #endregion

    /// <summary>
    /// Initializes a new instance of the SAINT model.
    /// </summary>
    /// <param name="architecture">The neural network architecture configuration.</param>
    /// <param name="options">Configuration options for the model.</param>
    /// <param name="optimizer">Optional optimizer for training.</param>
    /// <param name="lossFunction">Optional loss function.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Use this constructor when training SAINT from scratch.
    /// SAINT learns both column relationships (features) and row relationships (samples).
    /// </para>
    /// </remarks>
    public SAINT(
        NeuralNetworkArchitecture<T> architecture,
        SAINTOptions<T>? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
        ILossFunction<T>? lossFunction = null)
        : base(
            architecture,
            options?.NumFeatures ?? 50,
            options?.ConfidenceLevel ?? 0.95,
            options?.TimeHorizon ?? 1,
            lossFunction)
    {
        _options = options ?? new SAINTOptions<T>();
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
    /// <b>For Beginners:</b> Use this constructor when you already have a pretrained
    /// SAINT model saved as an ONNX file. This is best for inference.
    /// </para>
    /// </remarks>
    public SAINT(
        NeuralNetworkArchitecture<T> architecture,
        string onnxModelPath,
        SAINTOptions<T>? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
        ILossFunction<T>? lossFunction = null)
        : base(
            architecture,
            onnxModelPath,
            options?.NumFeatures ?? 50,
            options?.ConfidenceLevel ?? 0.95,
            options?.TimeHorizon ?? 1)
    {
        _options = options ?? new SAINTOptions<T>();
        _lossFunction = lossFunction ?? new MeanSquaredErrorLoss<T>();
        _optimizer = optimizer ?? new AdamOptimizer<T, Tensor<T>, Tensor<T>>(this);

        InitializeLayers();
    }

    #region Initialization

    /// <summary>
    /// Initializes the neural network layers for SAINT.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> SAINT combines attention over features and over samples.
    /// This method builds those layers using defaults unless you provide custom layers.
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
            Layers.AddRange(LayerHelper<T>.CreateDefaultSAINTLayers(
                Architecture,
                _options.NumFeatures,
                _options.HiddenDimension,
                _options.NumHeads,
                _options.NumLayers,
                _options.BatchSize,
                1,
                _options.DropoutRate));
        }
    }

    #endregion

    /// <summary>
    /// Calculates risk using SAINT.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Produces a single risk score based on the model’s
    /// understanding of your tabular data.
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
    /// <b>For Beginners:</b> If the predicted risk is too high, this shrinks
    /// the action until it fits the allowed risk budget.
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
}
