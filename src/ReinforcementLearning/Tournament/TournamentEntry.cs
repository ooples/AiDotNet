using AiDotNet.Interfaces;
using AiDotNet.Interfaces;

namespace AiDotNet.ReinforcementLearning.Tournament
{
    /// <summary>
    /// Represents a model entry in a tournament.
    /// </summary>
    /// <typeparam name="TState">The type of environment state.</typeparam>
    /// <typeparam name="TAction">The type of action.</typeparam>
    /// <typeparam name="T">The numeric type used for calculations.</typeparam>
    public class TournamentEntry<TState, TAction, T>
    {
        /// <summary>
        /// Gets the model.
        /// </summary>
        public IReinforcementLearningModel<T> Model { get; }
        
        /// <summary>
        /// Gets the name of the model entry.
        /// </summary>
        public string Name { get; }
        
        /// <summary>
        /// Gets the description of the model entry.
        /// </summary>
        public string Description { get; }
        
        /// <summary>
        /// Gets additional metadata about the model entry.
        /// </summary>
        public Dictionary<string, object> Metadata { get; }

        /// <summary>
        /// Initializes a new instance of the <see cref="TournamentEntry{TState, TAction, T}"/> class.
        /// </summary>
        /// <param name="model">The model.</param>
        /// <param name="name">The name of the model entry.</param>
        /// <param name="description">The description of the model entry.</param>
        /// <param name="metadata">Additional metadata about the model entry.</param>
        public TournamentEntry(
            IReinforcementLearningModel<T> model,
            string name,
            string description,
            Dictionary<string, object> metadata)
        {
            Model = model ?? throw new ArgumentNullException(nameof(model));
            Name = name ?? throw new ArgumentNullException(nameof(name));
            Description = description ?? string.Empty;
            Metadata = metadata ?? new Dictionary<string, object>();
        }
    }
}