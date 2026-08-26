using System.IO;
using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Finance.Interfaces;
using AiDotNet.Models.Options;
using AiDotNet.Finance.Base;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LossFunctions;
using AiDotNet.Models;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Optimizers;

namespace AiDotNet.Finance.Risk;

/// <summary>
/// Neural Value-at-Risk (VaR) model for non-linear market risk assessment.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// NeuralVaR uses deep neural networks to estimate the potential loss of a portfolio
/// under various market conditions, accounting for complex non-linear dependencies.
/// </para>
/// <para><b>For Beginners:</b> Value-at-Risk (VaR) is a way to answer the question:
/// "What is the most I could lose on this investment tomorrow with 95% confidence?"
/// Traditional methods often assume simple patterns, but this AI model "learns"
/// from historical market crashes and complex trends to give a more realistic
/// estimate of risk.
/// </para>
/// <para>
/// Reference: Barrera et al., "Statistical Learning of Value-at-Risk and Expected Shortfall", 2026.
/// </para>
/// </remarks>
/// <example>
/// <code>
/// // Define architecture for Value-at-Risk estimation (10 market risk factors)
/// var architecture = new NeuralNetworkArchitecture&lt;double&gt;(
///     inputType: InputType.OneDimensional,
///     taskType: NeuralNetworkTaskType.Regression,
///     inputSize: 10, outputSize: 1);
///
/// // Training mode: neural network learns non-linear VaR from historical data
/// var model = new NeuralVaR&lt;double&gt;(architecture);
///
/// // ONNX inference mode: load pre-trained VaR model
/// var onnxModel = new NeuralVaR&lt;double&gt;(architecture, "neural_var.onnx");
/// </code>
/// </example>
[ModelDomain(ModelDomain.Finance)]
[ModelDomain(ModelDomain.MachineLearning)]
[ModelCategory(ModelCategory.NeuralNetwork)]
[ModelTask(ModelTask.Regression)]
[ModelComplexity(ModelComplexity.Medium)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper(
    "Statistical Learning of Value-at-Risk and Expected Shortfall",
    "https://doi.org/10.1111/mafi.70000",
    Year = 2026,
    Authors = "Barrera, Crépey, Gobet, Nguyen, and Saadeddine")]
public partial class NeuralVaR<T> : RiskModelBase<T>
{
    #region Shared Fields

    private readonly NeuralVaROptions<T> _options;
    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> _optimizer;
    private readonly ILossFunction<T> _lossFunction;

    /// <summary>
    /// Routes the optimizer selected for NeuralVaR through the shared tape-training path.
    /// </summary>
    protected override IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? TrainingOptimizer => _optimizer;

    /// <inheritdoc/>
    public override ModelOptions GetOptions() => _options;

    #endregion

    #region Constructors

    private static ILossFunction<T> ResolveTrainingLoss(
        NeuralVaROptions<T>? options,
        ILossFunction<T>? lossFunction)
    {
        // A VaR forecast is a conditional quantile, not a conditional mean.
        // Barrera et al. learn VaR with neural-network quantile regression;
        // MSE would instead estimate E[Y|X] and is therefore the wrong objective.
        return lossFunction
            ?? options?.LossFunction
            ?? new QuantileLoss<T>(options?.ConfidenceLevel ?? 0.95);
    }

    /// <summary>
    /// Creates a NeuralVaR model using a pretrained ONNX model.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Use this when you already have a trained NeuralVaR
    /// model saved as an ONNX file and want to run predictions quickly.
    /// </para>
    /// </remarks>
    public NeuralVaR(
        NeuralNetworkArchitecture<T> architecture,
        string onnxModelPath,
        NeuralVaROptions<T>? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
        ILossFunction<T>? lossFunction = null)
        : base(architecture, onnxModelPath, options?.NumFeatures ?? 10,
               options?.ConfidenceLevel ?? 0.95, options?.TimeHorizon ?? 1)
    {
        options ??= new NeuralVaROptions<T>();
        _options = options;
        Options = _options;
        options.Validate();

        _lossFunction = ResolveTrainingLoss(_options, lossFunction);
        _options.LossFunction = _lossFunction;
        _optimizer = optimizer ?? new AdamOptimizer<T, Tensor<T>, Tensor<T>>(this);

        InitializeLayers();
    }

    /// <summary>
    /// Creates a NeuralVaR model in native mode for training.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Use this constructor when you want to train NeuralVaR
    /// from scratch on your own data.
    /// </para>
    /// </remarks>
    public NeuralVaR(
        NeuralNetworkArchitecture<T> architecture,
        NeuralVaROptions<T>? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
        ILossFunction<T>? lossFunction = null)
        : base(architecture, options?.NumFeatures ?? 10,
               options?.ConfidenceLevel ?? 0.95, options?.TimeHorizon ?? 1,
               ResolveTrainingLoss(options, lossFunction))
    {
        options ??= new NeuralVaROptions<T>();
        _options = options;
        Options = _options;
        options.Validate();

        _lossFunction = LossFunction;
        _options.LossFunction = _lossFunction;
        _optimizer = optimizer ?? new AdamOptimizer<T, Tensor<T>, Tensor<T>>(this);

        InitializeLayers();
    }

    #endregion

    #region Initialization

    /// <summary>
    /// Executes InitializeLayers for the NeuralVaR.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the NeuralVaR model, InitializeLayers builds and wires up model components. This sets up the NeuralVaR architecture before use.
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
            Layers.AddRange(LayerHelper<T>.CreateDefaultNeuralVaRLayers(
                Architecture,
                NumFeatures,
                _options.HiddenLayers,
                _options.HiddenDimension));
        }
    }

    #endregion

    #region Risk Calculation

    /// <inheritdoc/>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the NeuralVaR model, CalculateRisk estimates risk values. This is the core signal the NeuralVaR architecture focuses on.
    /// </para>
    /// </remarks>
    public override T CalculateRisk(Tensor<T> input)
    {
        var output = Predict(input);
        return output.Data.Span[0];
    }

    /// <inheritdoc/>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the NeuralVaR model, AdjustForRisk estimates risk values. This is the core signal the NeuralVaR architecture focuses on.
    /// </para>
    /// </remarks>
    public override Tensor<T> AdjustForRisk(Tensor<T> action, T riskBudget)
    {
        // For Beginners: If the calculated risk exceeds the budget, scale down the action.
        T currentRisk = CalculateRisk(action);
        if (NumOps.GreaterThan(currentRisk, riskBudget))
        {
            T scale = NumOps.Divide(riskBudget, currentRisk);
            return Engine.TensorMultiplyScalar(action, scale);
        }

        return action;
    }

    #endregion

    #region NeuralNetworkBase Overrides

    /// <summary>
    /// Executes Predict for the NeuralVaR.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the NeuralVaR model, Predict produces predictions from input data. This is the main inference step of the NeuralVaR architecture.
    /// </para>
    /// </remarks>
    protected override Tensor<T> PredictCore(Tensor<T> input)
    {
        // Inference mode is REQUIRED here: the stack contains
        // BatchNormalizationLayers, which in training mode normalize across the
        // batch axis. A single-instance prediction (batch = 1) then has each
        // feature's batch-mean equal to its own value, so the normalized output
        // is ~0 regardless of input and every constant input collapses to the
        // same VaR estimate. Inference mode uses the running statistics instead.
        SetTrainingMode(false);

        var current = input;
        foreach (var layer in Layers) current = layer.Forward(current);
        return current;
    }

    /// <summary>
    /// Executes Train for the NeuralVaR.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the NeuralVaR model, Train performs a training step. This updates the NeuralVaR architecture so it learns from data.
    /// </para>
    /// </remarks>
    public override void Train(Tensor<T> input, Tensor<T> target)
    {
        if (!UseNativeMode)
            throw new InvalidOperationException("Training is only supported in native mode.");

        // The paper's reference training restores the iterate with the best
        // validation quantile loss. Preserve that property at the incremental
        // Train API boundary: a single-observation pinball gradient is constant
        // on either side of its kink, so Adam can otherwise step across the
        // optimum and finish a short training budget worse than its best iterate.
        // This is parameter selection, not a replacement objective: the candidate
        // step is still computed by the configured optimizer and quantile loss.
        var parametersBefore = GetParameters();
        double lossBefore = NumOps.ToDouble(
            _lossFunction.CalculateLoss(Predict(input).ToVector(), target.ToVector()));

        base.Train(input, target);

        double lossAfter = NumOps.ToDouble(
            _lossFunction.CalculateLoss(Predict(input).ToVector(), target.ToVector()));
        if (double.IsNaN(lossAfter) || double.IsInfinity(lossAfter) || lossAfter > lossBefore)
        {
            var candidateParameters = GetParameters();
            bool accepted = false;

            // Backtrack along the optimizer's proposed direction. Pinball loss is
            // locally affine away from its kink, so a sufficiently small step in
            // the computed descent direction must improve the objective; retaining
            // a smaller improving update is preferable to turning the call into a
            // no-op by restoring the old iterate outright.
            for (double scale = 0.5; scale >= 1.0 / 65536.0; scale *= 0.5)
            {
                var trialParameters = new Vector<T>(parametersBefore.Length);
                T trialScale = NumOps.FromDouble(scale);
                for (int i = 0; i < trialParameters.Length; i++)
                {
                    T direction = NumOps.Subtract(candidateParameters[i], parametersBefore[i]);
                    trialParameters[i] = NumOps.Add(
                        parametersBefore[i],
                        NumOps.Multiply(direction, trialScale));
                }

                UpdateParameters(trialParameters);
                double trialLoss = NumOps.ToDouble(
                    _lossFunction.CalculateLoss(Predict(input).ToVector(), target.ToVector()));
                if (!double.IsNaN(trialLoss) && !double.IsInfinity(trialLoss) && trialLoss < lossBefore)
                {
                    accepted = true;
                    break;
                }
            }

            if (!accepted)
                UpdateParameters(parametersBefore);
        }
    }

    // UpdateParameters re-sliced the flat vector across Layers by hand -- the base walks
    // exactly the same enumeration, so this said nothing the base does not already say.
    /// <summary>
    /// Executes GetModelMetadata for the NeuralVaR.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the NeuralVaR model, GetModelMetadata performs a supporting step in the workflow. It keeps the NeuralVaR architecture pipeline consistent.
    /// </para>
    /// </remarks>
    public override ModelMetadata<T> GetModelMetadata()
    {
        return new ModelMetadata<T>
        {
            AdditionalInfo = new Dictionary<string, object>
            {
                { "ModelType", "NeuralVaR" },
                { "ConfidenceLevel", _confidenceLevel },
                { "TimeHorizon", _timeHorizon },
                { "ParameterCount", GetParameterCount() }
            }
        };
    }

    #endregion
}
