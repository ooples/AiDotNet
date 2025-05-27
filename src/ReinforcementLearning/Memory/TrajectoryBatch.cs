using System;

namespace AiDotNet.ReinforcementLearning.Memory
{
    /// <summary>
    /// A container for a batch of trajectories used in sequential reinforcement learning.
    /// </summary>
    /// <typeparam name="TState">The type of state observation.</typeparam>
    /// <typeparam name="TAction">The type of action.</typeparam>
    /// <typeparam name="T">The numeric type used for calculations.</typeparam>
    /// <remarks>
    /// <para>
    /// TrajectoryBatch represents a collection of sequential experiences that maintain temporal order.
    /// This is essential for algorithms like Decision Transformer that need to understand the sequence
    /// of states, actions, and rewards over time.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b>
    /// Think of this as a collection of "stories" about what happened in past trading sessions.
    /// Each story contains:
    /// - A sequence of market conditions (states)
    /// - The trading decisions made (actions)
    /// - The profits or losses that resulted (rewards)
    /// - Whether each session ended (done flags)
    /// 
    /// The Decision Transformer learns from these complete stories to make better trading decisions.
    /// </para>
    /// </remarks>
    public class TrajectoryBatch<TState, TAction, T>
    {
        /// <summary>
        /// Gets the array of states in the batch.
        /// </summary>
        public TState[] States { get; }
        
        /// <summary>
        /// Gets the array of actions in the batch.
        /// </summary>
        public TAction[] Actions { get; }
        
        /// <summary>
        /// Gets the array of rewards in the batch.
        /// </summary>
        public T[] Rewards { get; }
        
        /// <summary>
        /// Gets the array of next states in the batch.
        /// </summary>
        public TState[] NextStates { get; }
        
        /// <summary>
        /// Gets the array of done flags in the batch.
        /// </summary>
        public bool[] Dones { get; }
        
        /// <summary>
        /// Gets the number of experiences in this batch.
        /// </summary>
        public int BatchSize => States?.Length ?? 0;
        
        /// <summary>
        /// Gets a value indicating whether the batch is empty.
        /// </summary>
        public bool IsEmpty => BatchSize == 0;
        
        /// <summary>
        /// Initializes a new instance of the <see cref="TrajectoryBatch{TState, TAction, T}"/> class.
        /// </summary>
        /// <param name="states">The array of states.</param>
        /// <param name="actions">The array of actions.</param>
        /// <param name="rewards">The array of rewards.</param>
        /// <param name="nextStates">The array of next states.</param>
        /// <param name="dones">The array of done flags.</param>
        /// <exception cref="ArgumentException">Thrown when array lengths don't match.</exception>
        public TrajectoryBatch(TState[] states, TAction[] actions, T[] rewards, TState[] nextStates, bool[] dones)
        {
            // Validate input arrays
            if (states == null) throw new ArgumentNullException(nameof(states));
            if (actions == null) throw new ArgumentNullException(nameof(actions));
            if (rewards == null) throw new ArgumentNullException(nameof(rewards));
            if (nextStates == null) throw new ArgumentNullException(nameof(nextStates));
            if (dones == null) throw new ArgumentNullException(nameof(dones));
            
            // Validate array lengths match
            int length = states.Length;
            if (actions.Length != length || rewards.Length != length || 
                nextStates.Length != length || dones.Length != length)
            {
                throw new ArgumentException("All arrays must have the same length");
            }
            
            States = states;
            Actions = actions;
            Rewards = rewards;
            NextStates = nextStates;
            Dones = dones;
        }
        
        /// <summary>
        /// Creates an empty trajectory batch.
        /// </summary>
        /// <returns>An empty TrajectoryBatch instance.</returns>
        public static TrajectoryBatch<TState, TAction, T> Empty()
        {
            return new TrajectoryBatch<TState, TAction, T>(
                Array.Empty<TState>(),
                Array.Empty<TAction>(),
                Array.Empty<T>(),
                Array.Empty<TState>(),
                Array.Empty<bool>()
            );
        }
        
        /// <summary>
        /// Gets the experience at the specified index.
        /// </summary>
        /// <param name="index">The index of the experience to retrieve.</param>
        /// <returns>An Experience object containing the data at the specified index.</returns>
        /// <exception cref="IndexOutOfRangeException">Thrown when index is out of range.</exception>
        public Experience<TState, TAction, T> GetExperience(int index)
        {
            if (index < 0 || index >= BatchSize)
            {
                throw new IndexOutOfRangeException($"Index {index} is out of range for batch of size {BatchSize}");
            }
            
            return new Experience<TState, TAction, T>(
                States[index],
                Actions[index],
                Rewards[index],
                NextStates[index],
                Dones[index]
            );
        }
        
        /// <summary>
        /// Creates a subset of this batch containing only the specified range of experiences.
        /// </summary>
        /// <param name="startIndex">The starting index (inclusive).</param>
        /// <param name="count">The number of experiences to include.</param>
        /// <returns>A new TrajectoryBatch containing the specified subset.</returns>
        /// <exception cref="ArgumentOutOfRangeException">Thrown when parameters are out of range.</exception>
        public TrajectoryBatch<TState, TAction, T> Slice(int startIndex, int count)
        {
            if (startIndex < 0 || startIndex >= BatchSize)
            {
                throw new ArgumentOutOfRangeException(nameof(startIndex));
            }
            
            if (count < 0 || startIndex + count > BatchSize)
            {
                throw new ArgumentOutOfRangeException(nameof(count));
            }
            
            if (count == 0)
            {
                return Empty();
            }
            
            // Create arrays for the subset
            var subsetStates = new TState[count];
            var subsetActions = new TAction[count];
            var subsetRewards = new T[count];
            var subsetNextStates = new TState[count];
            var subsetDones = new bool[count];
            
            // Copy the specified range
            Array.Copy(States, startIndex, subsetStates, 0, count);
            Array.Copy(Actions, startIndex, subsetActions, 0, count);
            Array.Copy(Rewards, startIndex, subsetRewards, 0, count);
            Array.Copy(NextStates, startIndex, subsetNextStates, 0, count);
            Array.Copy(Dones, startIndex, subsetDones, 0, count);
            
            return new TrajectoryBatch<TState, TAction, T>(
                subsetStates, subsetActions, subsetRewards, subsetNextStates, subsetDones);
        }
        
        /// <summary>
        /// Validates the integrity of the trajectory batch data.
        /// </summary>
        /// <returns>True if the batch is valid, false otherwise.</returns>
        public bool Validate()
        {
            try
            {
                // Check for null arrays
                if (States == null || Actions == null || Rewards == null || 
                    NextStates == null || Dones == null)
                {
                    return false;
                }
                
                // Check array length consistency
                int length = States.Length;
                if (Actions.Length != length || Rewards.Length != length || 
                    NextStates.Length != length || Dones.Length != length)
                {
                    return false;
                }
                
                // Check for null elements in state arrays
                for (int i = 0; i < length; i++)
                {
                    if (States[i] == null || NextStates[i] == null || Actions[i] == null)
                    {
                        return false;
                    }
                }
                
                return true;
            }
            catch
            {
                return false;
            }
        }
    }
}