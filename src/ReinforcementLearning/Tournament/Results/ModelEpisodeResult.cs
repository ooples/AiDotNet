using AiDotNet.Helpers;
using AiDotNet.Interfaces;

namespace AiDotNet.ReinforcementLearning.Tournament.Results
{
    /// <summary>
    /// Represents the results of a single episode for a model.
    /// </summary>
    /// <typeparam name="T">The numeric type used for calculations.</typeparam>
    public class ModelEpisodeResult<T>
    {
        /// <summary>
        /// Gets the total reward obtained during the episode.
        /// </summary>
        public T TotalReward { get; }
        
        /// <summary>
        /// Gets the list of states visited during the episode.
        /// </summary>
        public IReadOnlyList<object> States { get; }
        
        /// <summary>
        /// Gets the list of actions taken during the episode.
        /// </summary>
        public IReadOnlyList<object> Actions { get; }
        
        /// <summary>
        /// Gets the list of rewards received during the episode.
        /// </summary>
        public IReadOnlyList<T> Rewards { get; }
        
        /// <summary>
        /// Gets the number of steps taken during the episode.
        /// </summary>
        public int Steps => Rewards.Count;

        /// <summary>
        /// Initializes a new instance of the <see cref="ModelEpisodeResult{T}"/> class.
        /// </summary>
        /// <param name="totalReward">The total reward obtained during the episode.</param>
        /// <param name="states">The list of states visited during the episode.</param>
        /// <param name="actions">The list of actions taken during the episode.</param>
        /// <param name="rewards">The list of rewards received during the episode.</param>
        public ModelEpisodeResult(
            T totalReward,
            IEnumerable<object> states,
            IEnumerable<object> actions,
            IEnumerable<T> rewards)
        {
            TotalReward = totalReward;
            States = states?.ToList() ?? throw new ArgumentNullException(nameof(states));
            Actions = actions?.ToList() ?? throw new ArgumentNullException(nameof(actions));
            Rewards = rewards?.ToList() ?? throw new ArgumentNullException(nameof(rewards));
            
            if (States.Count != Actions.Count || Actions.Count != Rewards.Count)
            {
                throw new ArgumentException("States, actions, and rewards must have the same length.");
            }
        }
    }
}