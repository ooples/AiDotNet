using AiDotNet.Interfaces;
using AiDotNet.ReinforcementLearning.Interfaces;
using System.Collections.Concurrent;

namespace AiDotNet.ReinforcementLearning.ReplayBuffers
{
    /// <summary>
    /// Base class for reinforcement learning replay buffers.
    /// </summary>
    /// <typeparam name="TState">The type used to represent the environment state.</typeparam>
    /// <typeparam name="TAction">The type used to represent actions.</typeparam>
    /// <typeparam name="T">The numeric type used for calculations.</typeparam>
    public class ReplayBufferBase<TState, TAction, T> : IReplayBuffer<TState, TAction, T> 
       
    {
        /// <summary>
        /// The internal buffer storing experience tuples.
        /// </summary>
        protected readonly ConcurrentQueue<(TState state, TAction action, T reward, TState nextState, bool done)> _buffer;

        /// <summary>
        /// The maximum capacity of the buffer.
        /// </summary>
        protected readonly int _capacity;

        /// <summary>
        /// The random number generator for sampling.
        /// </summary>
        protected readonly Random _random;

        /// <summary>
        /// Gets the current number of experiences in the buffer.
        /// </summary>
        public int Count => _buffer.Count;

        /// <summary>
        /// Gets the current number of experiences in the buffer (alias for Count).
        /// </summary>
        public int Size => Count;

        /// <summary>
        /// Gets the maximum capacity of the buffer.
        /// </summary>
        public int Capacity => _capacity;

        /// <summary>
        /// Gets a value indicating whether the buffer is full.
        /// </summary>
        public bool IsFull => Count >= Capacity;

        /// <summary>
        /// Gets a value indicating whether the buffer supports prioritization.
        /// </summary>
        public virtual bool SupportsPrioritization => false;

        /// <summary>
        /// Initializes a new instance of the <see cref="ReplayBufferBase{TState, TAction, T}"/> class.
        /// </summary>
        /// <param name="capacity">The maximum capacity of the buffer.</param>
        /// <param name="seed">The random seed for sampling.</param>
        public ReplayBufferBase(int capacity, int? seed = null)
        {
            _capacity = capacity;
            _buffer = new ConcurrentQueue<(TState state, TAction action, T reward, TState nextState, bool done)>();
            _random = seed.HasValue ? new Random(seed.Value) : new Random();
        }

        /// <summary>
        /// Adds a new experience to the buffer.
        /// </summary>
        /// <param name="state">The state before the action was taken.</param>
        /// <param name="action">The action that was taken.</param>
        /// <param name="reward">The reward received after taking the action.</param>
        /// <param name="nextState">The state after the action was taken.</param>
        /// <param name="done">A flag indicating whether the episode ended after this action.</param>
        public virtual void Add(TState state, TAction action, T reward, TState nextState, bool done)
        {
            _buffer.Enqueue((state, action, reward, nextState, done));

            // If buffer exceeds capacity, remove oldest experiences
            while (_buffer.Count > _capacity && _buffer.TryDequeue(out _)) { }
        }

        /// <summary>
        /// Samples a batch of experiences from the buffer.
        /// </summary>
        /// <param name="batchSize">The number of experiences to sample.</param>
        /// <returns>A tuple containing arrays of states, actions, rewards, next states, and done flags.</returns>
        public virtual (TState[] states, TAction[] actions, T[] rewards, TState[] nextStates, bool[] dones) Sample(int batchSize)
        {
            // Make sure we don't sample more than the buffer contains
            batchSize = Math.Min(batchSize, Count);
            if (batchSize == 0)
            {
                throw new InvalidOperationException("Cannot sample from an empty buffer");
            }

            // Create arrays to hold the sampled data
            var states = new TState[batchSize];
            var actions = new TAction[batchSize];
            var rewards = new T[batchSize];
            var nextStates = new TState[batchSize];
            var dones = new bool[batchSize];

            // Convert buffer to array for random sampling
            var experiences = _buffer.ToArray();

            // Sample randomly
            for (int i = 0; i < batchSize; i++)
            {
                int index = _random.Next(experiences.Length);
                var experience = experiences[index];

                states[i] = experience.state;
                actions[i] = experience.action;
                rewards[i] = experience.reward;
                nextStates[i] = experience.nextState;
                dones[i] = experience.done;
            }

            return (states, actions, rewards, nextStates, dones);
        }

        /// <summary>
        /// Clears all experiences from the buffer.
        /// </summary>
        public virtual void Clear()
        {
            while (_buffer.TryDequeue(out _)) { }
        }

        /// <summary>
        /// Updates the priorities for specific experiences in the buffer.
        /// </summary>
        /// <param name="indices">The indices of the experiences to update.</param>
        /// <param name="priorities">The new priority values.</param>
        public virtual void UpdatePriorities(int[] indices, T[] priorities)
        {
            // Base implementation doesn't support prioritization
            throw new NotSupportedException("This replay buffer does not support prioritization");
        }

        /// <summary>
        /// Samples a batch of experiences from the buffer.
        /// </summary>
        /// <param name="batchSize">The number of experiences to sample.</param>
        /// <returns>A batch of experiences.</returns>
        public virtual ReplayBatch<TState, TAction, T> SampleBatch(int batchSize)
        {
            var (states, actions, rewards, nextStates, dones) = Sample(batchSize);
            return new ReplayBatch<TState, TAction, T>(states, actions, rewards, nextStates, dones);
        }
    }
}