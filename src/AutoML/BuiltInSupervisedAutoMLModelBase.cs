using AiDotNet.AutoML.Policies;
using AiDotNet.AutoML.Registry;
using AiDotNet.Enums;
using AiDotNet.Interfaces;

namespace AiDotNet.AutoML;

/// <summary>
/// Base class for built-in supervised AutoML strategies that operate on tabular Matrix/Vector tasks.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <typeparam name="TInput">The input data type.</typeparam>
/// <typeparam name="TOutput">The output data type.</typeparam>
/// <remarks>
/// <para>
/// This class centralizes built-in model construction and default candidate selection so different search
/// strategies (random, Bayesian, evolutionary, etc.) can focus on "how to propose the next trial".
/// </para>
/// <para>
/// <b>For Beginners:</b> AutoML tries many "trials". Each trial picks a model family and some settings, then trains
/// and scores it. Different strategies decide which settings to try next.
/// </para>
/// </remarks>
public abstract class BuiltInSupervisedAutoMLModelBase<T, TInput, TOutput> : SupervisedAutoMLModelBase<T, TInput, TOutput>
{
    protected BuiltInSupervisedAutoMLModelBase(IModelEvaluator<T, TInput, TOutput>? modelEvaluator = null, Random? random = null)
        : base(modelEvaluator, random)
    {
    }

    protected override Task<IFullModel<T, TInput, TOutput>> CreateModelAsync(ModelType modelType, Dictionary<string, object> parameters)
    {
        if (typeof(TInput) != typeof(Matrix<T>) || typeof(TOutput) != typeof(Vector<T>))
        {
            throw new NotSupportedException(
                $"Built-in supervised AutoML currently supports Matrix<T>/Vector<T> tasks. Received {typeof(TInput).Name}/{typeof(TOutput).Name}.");
        }

        var model = AutoMLTabularModelFactory<T>.Create(modelType, parameters);
        return Task.FromResult((IFullModel<T, TInput, TOutput>)model);
    }

    protected override Dictionary<string, ParameterRange> GetDefaultSearchSpace(ModelType modelType)
    {
        return AutoMLTabularSearchSpaceRegistry.GetDefaultSearchSpace(modelType);
    }

    /// <summary>
    /// Applies built-in default candidate models when the user has not configured candidates explicitly.
    /// </summary>
    protected void EnsureDefaultCandidateModels(TInput inputs, TOutput targets)
    {
        lock (_lock)
        {
            if (_candidateModels.Count != 0)
            {
                return;
            }

            if (typeof(TInput) != typeof(Matrix<T>) || typeof(TOutput) != typeof(Vector<T>))
            {
                return;
            }

            int featureCount = InputHelper<T, TInput>.GetInputSize(inputs);
            var taskFamily = AutoMLTaskFamilyInference.InferFromTargets<T, TOutput>(targets);
            foreach (var candidate in AutoMLDefaultCandidateModelsPolicy.GetDefaultCandidates(taskFamily, featureCount, BudgetPreset))
            {
                _candidateModels.Add(candidate);
            }
        }
    }
}
