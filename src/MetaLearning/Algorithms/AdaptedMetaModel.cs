using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.MetaLearning.Models;
using AiDotNet.Models;

using AiDotNet.Models.Parameters;

namespace AiDotNet.MetaLearning.Algorithms;

/// <summary>
/// Generic adapted model wrapper for meta-learning algorithms that use gradient-based inner-loop adaptation.
/// After adaptation, the model uses the adapted parameters for prediction.
/// </summary>
/// <typeparam name="T">The numeric type.</typeparam>
/// <typeparam name="TInput">The input data type.</typeparam>
/// <typeparam name="TOutput">The output data type.</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> An adapted meta-model is the result of a meta-learning
/// algorithm (like MAML) adapting to a new task. After seeing a few examples of a new task,
/// the meta-learner produces this adapted model with task-specific parameters. Think of it
/// like a student who has learned general problem-solving skills and then quickly adapts
/// to a specific exam topic after seeing just a few practice questions.</para>
/// </remarks>
/// <example>
/// <code>
/// // Create an adapted model after MAML inner-loop adaptation
/// var adaptedParams = new Vector&lt;float&gt;(baseModel.GetParameters().Length);
/// // ... inner-loop gradient updates fill adaptedParams ...
/// var adapted = new AdaptedMetaModel&lt;float, Tensor&lt;float&gt;, Tensor&lt;float&gt;&gt;(
///     baseModel, adaptedParams);
/// Tensor&lt;float&gt; prediction = adapted.Predict(queryInput);
/// </code>
/// </example>
[ModelDomain(ModelDomain.MachineLearning)]
[ModelCategory(ModelCategory.MetaLearning)]
[ModelTask(ModelTask.Classification)]
[ModelTask(ModelTask.Regression)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks", "https://arxiv.org/abs/1703.03400")]
[ComponentType(ComponentType.MetaLearner)]
[PipelineStage(PipelineStage.Training)]
public partial class AdaptedMetaModel<T, TInput, TOutput> : MetaLearningModelBase<T, TInput, TOutput>, IAdaptedMetaModel<T>
{

    /// <inheritdoc />
    /// <remarks>The adapted parameter vector this model carries INSTEAD of the wrapped model's. It is replaced wholesale on restore, which is why the source takes a setter.</remarks>
    protected override void RegisterComponents()
    {
        RegisterParameterComponent(new VectorFieldParameterSource<T>(
            () => _adaptedParams,
            value => _adaptedParams = value));
    }
    [AiDotNet.Attributes.TrainableParameter]
    private Vector<T> _adaptedParams;
    // Task examples are inputs to adaptation, not optimizer-owned model state. Keeping this
    // declaration explicit prevents the generated parameter graph from treating an optional
    // support vector as a shape-deferred weight slot.
    [ExternalState]
    private readonly Vector<T>? _supportFeatures;
    private readonly double[]? _modulationFactors;

    /// <inheritdoc/>
    public Vector<T>? AdaptedSupportFeatures => _supportFeatures;

    /// <inheritdoc/>
    public double[]? ParameterModulationFactors => _modulationFactors;

    public AdaptedMetaModel(
        IFullModel<T, TInput, TOutput> model,
        Vector<T> adaptedParams,
        Vector<T>? supportFeatures = null,
        double[]? modulationFactors = null)
        : base(model)
    {
        _adaptedParams = adaptedParams ?? throw new ArgumentNullException(nameof(adaptedParams));
        _supportFeatures = supportFeatures;
        _modulationFactors = modulationFactors;
    }

    /// <inheritdoc/>
    /// <remarks>
    /// This method is not thread-safe. The shared model's parameters are temporarily
    /// replaced for prediction and restored afterward. External synchronization is
    /// required if multiple AdaptedMetaModel instances share the same underlying model.
    /// </remarks>
    public override TOutput Predict(TInput input)
    {
        var originalParams = InterfaceGuard.Parameterizable(BaseModel).GetParameters();
        try
        {
            if (_modulationFactors is not null && _modulationFactors.Length > 0)
            {
                var modulated = new Vector<T>(_adaptedParams.Length);
                for (int i = 0; i < _adaptedParams.Length; i++)
                    modulated[i] = NumOps.Multiply(_adaptedParams[i],
                        NumOps.FromDouble(_modulationFactors[i % _modulationFactors.Length]));
                InterfaceGuard.Parameterizable(BaseModel).SetParameters(modulated);
            }
            else
            {
                InterfaceGuard.Parameterizable(BaseModel).SetParameters(_adaptedParams);
            }
            return BaseModel.Predict(input);
        }
        finally
        {
            InterfaceGuard.Parameterizable(BaseModel).SetParameters(originalParams);
        }
    }

    /// <inheritdoc/>
    public override IFullModel<T, TInput, TOutput> WithParameters(Vector<T> parameters)
    {
        return new AdaptedMetaModel<T, TInput, TOutput>(BaseModel, parameters, _supportFeatures, _modulationFactors);
    }
}
