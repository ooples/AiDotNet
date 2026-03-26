using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.MetaLearning.Models;
using AiDotNet.Models;

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
public class AdaptedMetaModel<T, TInput, TOutput> : MetaLearningModelBase<T, TInput, TOutput>, IAdaptedMetaModel<T>
{
    private Vector<T> _adaptedParams;
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
        var originalParams = BaseModel.GetParameters();
        try
        {
            if (_modulationFactors is not null && _modulationFactors.Length > 0)
            {
                var modulated = new Vector<T>(_adaptedParams.Length);
                for (int i = 0; i < _adaptedParams.Length; i++)
                    modulated[i] = NumOps.Multiply(_adaptedParams[i],
                        NumOps.FromDouble(_modulationFactors[i % _modulationFactors.Length]));
                BaseModel.SetParameters(modulated);
            }
            else
            {
                BaseModel.SetParameters(_adaptedParams);
            }
            return BaseModel.Predict(input);
        }
        finally
        {
            BaseModel.SetParameters(originalParams);
        }
    }

    /// <inheritdoc/>
    public override Vector<T> GetParameters() => _adaptedParams;

    /// <inheritdoc/>
    public override void SetParameters(Vector<T> parameters)
    {
        if (parameters is null)
            throw new ArgumentNullException(nameof(parameters));
        _adaptedParams = parameters;
    }

    /// <inheritdoc/>
    public override IFullModel<T, TInput, TOutput> WithParameters(Vector<T> parameters)
    {
        return new AdaptedMetaModel<T, TInput, TOutput>(BaseModel, parameters, _supportFeatures, _modulationFactors);
    }

    /// <inheritdoc/>
    public override IFullModel<T, TInput, TOutput> DeepCopy()
    {
        var clonedModel = BaseModel.DeepCopy();
        var clonedParams = _adaptedParams.Clone();
        var clonedFeatures = _supportFeatures?.Clone();
        var clonedModulation = _modulationFactors is not null ? (double[])_modulationFactors.Clone() : null;
        return new AdaptedMetaModel<T, TInput, TOutput>(clonedModel, clonedParams, clonedFeatures, clonedModulation);
    }
}
