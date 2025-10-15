using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.Interfaces;
using AiDotNet.ReinforcementLearning.Memory;
using System;
using System.Collections.Generic;
using System.Linq;

namespace AiDotNet.ReinforcementLearning.ReplayBuffers
{
    /// <summary>
    /// A replay buffer for storing and sampling sequential trajectories of experiences.
    /// </summary>
    /// <typeparam name="TState">The type of state observation.</typeparam>
    /// <typeparam name="TAction">The type of action.</typeparam>
    /// <typeparam name="T">The numeric type used for calculations.</typeparam>
    /// <remarks>
    /// <para>
    /// This buffer is designed for sequential models like Decision Transformer that process
    /// entire trajectories rather than individual transitions. It maintains the sequential
    /// nature of experiences and samples complete trajectory segments.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b>
    /// This is like a database of past trading experiences that keeps track of:
    /// - The sequence of market states (price movements, indicators, etc.)
    /// - What actions were taken at each step
    /// - What rewards (profits/losses) resulted
    /// 
    /// Unlike regular reinforcement learning that looks at individual experiences,
    /// the Sequential Replay Buffer maintains the order of experiences, which is
    /// crucial for learning temporal patterns in financial markets.
    /// </para>
    /// </remarks>
    public class SequentialReplayBuffer<TState, TAction, T> : ISequentialReplayBuffer<TState, TAction, T>
    {
        private readonly int _capacity;
        private readonly int _maxTrajectoryLength;
        private readonly Random _random = default!;
        private readonly INumericOperations<T> _numOps = default!;
        
        // Lists to store experiences by episode
        private readonly List<List<(TState state, TAction action, T reward, TState nextState, bool done)>> _episodes;
        
        // Current episode being built
        private List<(TState state, TAction action, T reward, TState nextState, bool done)> _currentEpisode;
        
        /// <summary>
        /// Gets the current number of experiences in the buffer.
        /// </summary>
        public int Size { get; private set; }
        
        /// <summary>
        /// Gets the maximum capacity of the buffer.
        /// </summary>
        public int Capacity => _capacity;
        
        /// <summary>
        /// Initializes a new instance of the <see cref="SequentialReplayBuffer{TState, TAction, T}"/> class.
        /// </summary>
        /// <param name="capacity">The maximum number of experiences to store.</param>
        /// <param name="maxTrajectoryLength">The maximum length of a trajectory to sample.</param>
        /// <param name="seed">An optional seed for the random number generator.</param>
        public SequentialReplayBuffer(int capacity, int maxTrajectoryLength, int? seed = null)
        {
            _capacity = capacity;
            _maxTrajectoryLength = maxTrajectoryLength;
            _random = seed.HasValue ? new Random(seed.Value) : new Random();
            _numOps = MathHelper.GetNumericOperations<T>();
            
            _episodes = new List<List<(TState, TAction, T, TState, bool)>>();
            _currentEpisode = new List<(TState, TAction, T, TState, bool)>();
            Size = 0;
        }
        
        /// <summary>
        /// Adds an experience to the buffer.
        /// </summary>
        /// <param name="state">The state observation.</param>
        /// <param name="action">The action taken.</param>
        /// <param name="reward">The reward received.</param>
        /// <param name="nextState">The next state observation.</param>
        /// <param name="done">Whether the episode is done.</param>
        public void Add(TState state, TAction action, T reward, TState nextState, bool done)
        {
            // Add experience to current episode
            _currentEpisode.Add((state, action, reward, nextState, done));
            
            // Increment size counter
            Size++;
            
            // If episode is done, store it and start a new one
            if (done)
            {
                _episodes.Add(_currentEpisode);
                _currentEpisode = new List<(TState, TAction, T, TState, bool)>();
                
                // If buffer is over capacity, remove oldest episodes
                while (Size > _capacity)
                {
                    Size -= _episodes[0].Count;
                    _episodes.RemoveAt(0);
                }
            }
        }
        
        /// <summary>
        /// Samples a batch of trajectories from the buffer.
        /// </summary>
        /// <param name="batchSize">The number of trajectories to sample.</param>
        /// <returns>A batch of sampled trajectories.</returns>
        public TrajectoryBatch<TState, TAction, T> SampleBatch(int batchSize)
        {
            if (_episodes.Count == 0 && _currentEpisode.Count == 0)
            {
                return TrajectoryBatch<TState, TAction, T>.Empty();
            }
            
            // Prepare lists for batch data
            var states = new List<TState>();
            var actions = new List<TAction>();
            var rewards = new List<T>();
            var nextStates = new List<TState>();
            var dones = new List<bool>();
            
            // Sample episodes and trajectory segments
            for (int b = 0; b < batchSize; b++)
            {
                // Choose a random episode (including current if not empty)
                var validEpisodes = _episodes.Count;
                if (_currentEpisode.Count > 0)
                {
                    validEpisodes++;
                }
                
                if (validEpisodes == 0)
                {
                    continue;
                }
                
                int episodeIndex = _random.Next(validEpisodes);
                var episode = episodeIndex < _episodes.Count 
                    ? _episodes[episodeIndex] 
                    : _currentEpisode;
                
                if (episode.Count == 0)
                {
                    continue;
                }
                
                // Choose a random starting point within the episode
                int maxStartIndex = Math.Max(0, episode.Count - _maxTrajectoryLength);
                int startIndex = maxStartIndex > 0 ? _random.Next(maxStartIndex) : 0;
                
                // Determine trajectory length
                int trajectoryLength = Math.Min(_maxTrajectoryLength, episode.Count - startIndex);
                
                // Extract trajectory segment
                for (int i = 0; i < trajectoryLength; i++)
                {
                    var experience = episode[startIndex + i];
                    states.Add(experience.state);
                    actions.Add(experience.action);
                    rewards.Add(experience.reward);
                    nextStates.Add(experience.nextState);
                    dones.Add(experience.done);
                }
            }
            
            // Convert lists to arrays
            return new TrajectoryBatch<TState, TAction, T>(
                states.ToArray(),
                actions.ToArray(),
                rewards.ToArray(),
                nextStates.ToArray(),
                dones.ToArray()
            );
        }
        
        /// <summary>
        /// Clears all experiences from the buffer.
        /// </summary>
        public void Clear()
        {
            _episodes.Clear();
            _currentEpisode.Clear();
            Size = 0;
        }
    }
}