using AiDotNet.ContinualLearning.Interfaces;
using AiDotNet.ContinualLearning.Strategies;
using AiDotNet.ContinualLearning.Trainers;
using AiDotNet.Interfaces;
using AiDotNet.LossFunctions;

namespace AiDotNet.ContinualLearning;

/// <summary>
/// Builds the continual-learning trainer that matches a strategy, around a given model — so continual
/// learning operates on the single configured model rather than a separately-wrapped one.
/// </summary>
internal static class ContinualLearningTrainerFactory
{
    /// <summary>
    /// Creates the trainer for a strategy. A <c>null</c> strategy yields the industry-standard default:
    /// Elastic Weight Consolidation with experience replay (on by default in <see cref="EWCTrainer{T,
    /// TInput, TOutput}"/>) — a regularization+rehearsal hybrid that outperforms either alone.
    /// </summary>
    public static IContinualLearner<T, TInput, TOutput> Create<T, TInput, TOutput>(
        IContinualLearningStrategy<T, TInput, TOutput>? strategy,
        IFullModel<T, TInput, TOutput> model,
        ILossFunction<T> lossFunction,
        IContinualLearnerConfig<T> config)
    {
        if (strategy is null)
        {
            var ewc = new ElasticWeightConsolidation<T, TInput, TOutput>(lossFunction);
            return new EWCTrainer<T, TInput, TOutput>(model, lossFunction, config, ewc);
        }

        return strategy switch
        {
            ElasticWeightConsolidation<T, TInput, TOutput> => new EWCTrainer<T, TInput, TOutput>(model, lossFunction, config, strategy),
            GradientEpisodicMemory<T, TInput, TOutput> => new GEMTrainer<T, TInput, TOutput>(model, lossFunction, config, strategy),
            LearningWithoutForgetting<T, TInput, TOutput> => new LwFTrainer<T, TInput, TOutput>(model, lossFunction, config, strategy),
            MemoryAwareSynapses<T, TInput, TOutput> => new MASTrainer<T, TInput, TOutput>(model, lossFunction, config, strategy),
            SynapticIntelligence<T, TInput, TOutput> => new SITrainer<T, TInput, TOutput>(model, lossFunction, config, strategy),
            _ => throw new System.NotSupportedException(
                $"No continual-learning trainer is registered for strategy '{strategy.GetType().Name}'. " +
                "Supported strategies: ElasticWeightConsolidation, GradientEpisodicMemory, " +
                "LearningWithoutForgetting, MemoryAwareSynapses, SynapticIntelligence."),
        };
    }
}
