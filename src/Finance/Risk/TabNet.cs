using AiDotNet.Finance.Base;
using AiDotNet.Models.Options;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LossFunctions;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Optimizers;

namespace AiDotNet.Finance.Risk;

/// <summary>
/// TabNet model for tabular data learning, combining tree-based interpretability with deep learning performance.
/// </summary>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> TabNet is designed specifically for data that comes in tables (rows and columns),
/// like spreadsheets or databases. Traditional neural networks struggle with this kind of data compared to
/// decision trees (like XGBoost). TabNet tries to get the best of both worlds: the accuracy of trees
/// and the end-to-end learning of neural networks.
/// </para>
/// </remarks>
public class TabNet<T> : RiskModelBase<T>
{
    #region Shared Fields

    private readonly TabNetOptions<T> _options;
    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> _optimizer;
    private readonly ILossFunction<T> _lossFunction;

    #endregion

    /// <summary>
    /// Initializes a new instance of the TabNet model.
    /// </summary>
    /// <param name="architecture">The neural network architecture configuration.</param>
    /// <param name="options">Configuration options for the model.</param>
    /// <param name="optimizer">Optional optimizer for training.</param>
    /// <param name="lossFunction">Optional loss function.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Use this constructor when you want to TRAIN TabNet from scratch.
    /// You provide a network blueprint (<paramref name="architecture"/>) and a set of options
    /// that describe how TabNet should behave (like how wide the hidden layers are and how
    /// many decision steps it takes).
    /// </para>
    /// <para>
    /// In simple terms, this sets up a TabNet model that can learn from your data:
    /// <list type="number">
    /// <item>Create the default TabNet layers (if you didn’t provide your own)</item>
    /// <item>Attach an optimizer so the model can update its weights</item>
    /// <item>Attach a loss function so the model knows how "wrong" it is</item>
    /// </list>
    /// </para>
    /// <para>
    /// If you already have custom layers in the architecture, those layers are used instead.
    /// This keeps the model flexible while still following the golden pattern.
    /// </para>
    /// </remarks>
    public TabNet(
        NeuralNetworkArchitecture<T> architecture,
        TabNetOptions<T>? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
        ILossFunction<T>? lossFunction = null)
        : base(
            architecture,
            options?.NumFeatures ?? 50,
            options?.ConfidenceLevel ?? 0.95,
            options?.TimeHorizon ?? 1,
            lossFunction)
    {
        _options = options ?? new TabNetOptions<T>();
        Options = _options;
        _lossFunction = lossFunction ?? new MeanSquaredErrorLoss<T>();
        _optimizer = optimizer ?? new AdamOptimizer<T, Tensor<T>, Tensor<T>>(this);

        InitializeLayers();
    }

    /// <summary>
    /// Initializes a new instance of the TabNet model from a pretrained ONNX model.
    /// </summary>
    /// <param name="architecture">The neural network architecture configuration.</param>
    /// <param name="onnxModelPath">Path to the ONNX model file.</param>
    /// <param name="options">Configuration options for the model.</param>
    /// <param name="optimizer">Optional optimizer for fine-tuning.</param>
    /// <param name="lossFunction">Optional loss function.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Use this constructor when you ALREADY have a trained TabNet
    /// model saved as an ONNX file. ONNX is a standard file format that lets you load
    /// pretrained models quickly for inference (predictions).
    /// </para>
    /// <para>
    /// This mode is meant for running predictions, not training. The model weights are
    /// loaded from the ONNX file, and the architecture + options are used to keep input
    /// and output shapes consistent with the training setup.
    /// </para>
    /// <para>
    /// If you want to train a model, use the other constructor instead.
    /// </para>
    /// </remarks>
    public TabNet(
        NeuralNetworkArchitecture<T> architecture,
        string onnxModelPath,
        TabNetOptions<T>? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
        ILossFunction<T>? lossFunction = null)
        : base(
            architecture,
            onnxModelPath,
            options?.NumFeatures ?? 50,
            options?.ConfidenceLevel ?? 0.95,
            options?.TimeHorizon ?? 1)
    {
        _options = options ?? new TabNetOptions<T>();
        Options = _options;
        _lossFunction = lossFunction ?? new MeanSquaredErrorLoss<T>();
        _optimizer = optimizer ?? new AdamOptimizer<T, Tensor<T>, Tensor<T>>(this);

        InitializeLayers();
    }

    #region Initialization

    /// <summary>
    /// Initializes the neural network layers for TabNet.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method builds the "assembly line" of TabNet.
    /// TabNet works by repeatedly selecting which features matter most, step by step.
    /// That’s why it uses multiple decision steps instead of a single monolithic layer.
    /// </para>
    /// <para>
    /// The default setup creates:
    /// <list type="number">
    /// <item><b>Feature Transformer:</b> Turns raw feature values into richer hidden representations.</item>
    /// <item><b>Decision Steps:</b> Repeats the "feature selection + processing" cycle to refine the answer.</item>
    /// <item><b>Output Layer:</b> Produces the final risk score.</item>
    /// </list>
    /// </para>
    /// <para>
    /// If you already supplied layers in the architecture, this method just uses them
    /// and validates they are compatible.
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
            Layers.AddRange(LayerHelper<T>.CreateDefaultTabNetLayers(
                Architecture,
                _options.NumFeatures,
                _options.HiddenDimension,
                _options.NumDecisionSteps,
                1,
                _options.DropoutRate));
        }
    }

    #endregion

    /// <summary>
    /// Calculates risk using the TabNet model.
    /// </summary>
    /// <param name="input">Input tensor.</param>
    /// <returns>Risk metric.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method runs the model forward and turns its output
    /// into a single "risk number." Higher values mean more risk. The absolute value
    /// is used so the score is always positive and easier to interpret.
    /// </para>
    /// <para>
    /// TabNet is especially useful here because it focuses on the most important
    /// columns in your data (for example, leverage ratios or volatility measures),
    /// which makes the risk estimate more interpretable than a generic black box.
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
    /// <param name="action">Proposed action.</param>
    /// <param name="riskBudget">Risk budget.</param>
    /// <returns>Adjusted action.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Think of this like a safety governor on a car engine.
    /// If the model predicts that the action is too risky (above the budget),
    /// this method scales the action down to stay within limits.
    /// </para>
    /// <para>
    /// Example: If the budget is "risk = 1.0" but the prediction says the action
    /// has risk "2.0", the action is scaled by 1.0 / 2.0 = 0.5.
    /// This keeps your system from taking oversized or dangerous positions.
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
    /// which is handy for cloning or restoring a trained model.
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
    /// Creates a new instance of the TabNet model with the same configuration.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This is used by the framework to clone the model setup
    /// so it can create a fresh instance with identical settings.
    /// </para>
    /// </remarks>
    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance()
    {
        var options = new TabNetOptions<T>
        {
            NumFeatures = _options.NumFeatures,
            ConfidenceLevel = _options.ConfidenceLevel,
            TimeHorizon = _options.TimeHorizon,
            HiddenDimension = _options.HiddenDimension,
            NumDecisionSteps = _options.NumDecisionSteps,
            DropoutRate = _options.DropoutRate
        };

        return new TabNet<T>(Architecture, options, _optimizer, LossFunction);
    }

    #endregion
}
