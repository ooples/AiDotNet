using AiDotNet.ReinforcementLearning.Tournament.Results;

namespace AiDotNet.Interfaces
{
    /// <summary>
    /// Interface for evaluation metrics used in tournaments.
    /// </summary>
    /// <typeparam name="T">The numeric type used for calculations.</typeparam>
    public interface IEvaluationMetric<T>
    {
        /// <summary>
        /// Gets the name of the metric.
        /// </summary>
        string Name { get; }
        
        /// <summary>
        /// Gets the description of the metric.
        /// </summary>
        string Description { get; }
        
        /// <summary>
        /// Gets a value indicating whether higher values of this metric are better.
        /// </summary>
        bool HigherIsBetter { get; }
        
        /// <summary>
        /// Calculates the metric value for an episode.
        /// </summary>
        /// <param name="episodeResult">The episode result to calculate the metric for.</param>
        /// <returns>The calculated metric value.</returns>
        T Calculate(ModelEpisodeResult<T> episodeResult);
    }
}