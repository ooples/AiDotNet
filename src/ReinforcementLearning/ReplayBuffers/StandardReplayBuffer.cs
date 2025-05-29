using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Interfaces;
using System;
using System.Collections.Generic;

namespace AiDotNet.ReinforcementLearning.ReplayBuffers
{
    /// <summary>
    /// A standard replay buffer implementation for reinforcement learning with generic action types.
    /// </summary>
    /// <typeparam name="TState">The type used to represent the environment state.</typeparam>
    /// <typeparam name="TAction">The type used to represent actions.</typeparam>
    /// <typeparam name="T">The numeric type used for calculations.</typeparam>
    public class StandardReplayBuffer<TState, TAction, T> : IReplayBuffer<TState, TAction, T>
    {
        /// <summary>
        /// Gets the numeric operations for type T.
        /// </summary>
        protected INumericOperations<T> NumOps => MathHelper.GetNumericOperations<T>();
        
        private readonly int _capacity;
        private readonly TState[] _states;
        private readonly TAction[] _actions;
        private readonly T[] _rewards;
        private readonly TState[] _nextStates;
        private readonly bool[] _dones;
        protected int _position;
        private int _size;
        protected readonly Random _random;

        /// <summary>
        /// Initializes a new instance of the <see cref="StandardReplayBuffer{TState, TAction, T}"/> class.
        /// </summary>
        /// <param name="capacity">The maximum capacity of the buffer.</param>
        /// <param name="seed">Optional seed for the random number generator.</param>
        public StandardReplayBuffer(int capacity, int? seed = null)
        {
            _capacity = capacity;
            _states = new TState[capacity];
            _actions = new TAction[capacity];
            _rewards = new T[capacity];
            _nextStates = new TState[capacity];
            _dones = new bool[capacity];
            _position = 0;
            _size = 0;
            _random = seed.HasValue ? new Random(seed.Value) : new Random();
        }

        /// <summary>
        /// Adds an experience tuple to the buffer.
        /// </summary>
        /// <param name="state">The state before the action was taken.</param>
        /// <param name="action">The action that was taken.</param>
        /// <param name="reward">The reward received after taking the action.</param>
        /// <param name="nextState">The state after the action was taken.</param>
        /// <param name="done">A flag indicating whether the episode ended after this action.</param>
        public virtual void Add(TState state, TAction action, T reward, TState nextState, bool done)
        {
            _states[_position] = state;
            _actions[_position] = action;
            _rewards[_position] = reward;
            _nextStates[_position] = nextState;
            _dones[_position] = done;
            
            _position = (_position + 1) % _capacity;
            _size = Math.Min(_size + 1, _capacity);
        }

        /// <summary>
        /// Samples a batch of experiences from the buffer.
        /// </summary>
        /// <param name="batchSize">The size of the batch to sample.</param>
        /// <returns>A batch of experiences.</returns>
        public virtual ReplayBatch<TState, TAction, T> SampleBatch(int batchSize)
        {
            if (batchSize > _size)
            {
                throw new ArgumentException($"Batch size ({batchSize}) is larger than buffer size ({_size})");
            }
            
            var states = new TState[batchSize];
            var actions = new TAction[batchSize];
            var rewards = new T[batchSize];
            var nextStates = new TState[batchSize];
            var dones = new bool[batchSize];
            var indices = new int[batchSize];
            
            // Randomly sample batchSize experiences
            for (int i = 0; i < batchSize; i++)
            {
                var index = _random.Next(_size);
                indices[i] = index;
                
                // Convert from logical index to actual index in circular buffer
                int actualIdx = (_position - _size + index + _capacity) % _capacity;
                
                states[i] = _states[actualIdx];
                actions[i] = _actions[actualIdx];
                rewards[i] = _rewards[actualIdx];
                nextStates[i] = _nextStates[actualIdx];
                dones[i] = _dones[actualIdx];
            }
            
            return new ReplayBatch<TState, TAction, T>(states, actions, rewards, nextStates, dones, indices);
        }

        /// <summary>
        /// Gets the size of the buffer.
        /// </summary>
        public int Size => _size;

        /// <summary>
        /// Gets the capacity of the buffer.
        /// </summary>
        public int Capacity => _capacity;

        /// <summary>
        /// Clears all experiences from the buffer.
        /// </summary>
        public virtual void Clear()
        {
            Array.Clear(_states, 0, _states.Length);
            Array.Clear(_actions, 0, _actions.Length);
            Array.Clear(_rewards, 0, _rewards.Length);
            Array.Clear(_nextStates, 0, _nextStates.Length);
            Array.Clear(_dones, 0, _dones.Length);
            _position = 0;
            _size = 0;
        }
    }
}