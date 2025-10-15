using AiDotNet.Interfaces;
using AiDotNet.Interfaces;
using AiDotNet.Helpers;
using AiDotNet.ReinforcementLearning.Memory;

namespace AiDotNet.ReinforcementLearning.ReplayBuffers
{
    /// <summary>
    /// Implements a prioritized experience replay buffer for reinforcement learning.
    /// </summary>
    /// <typeparam name="TState">The type used to represent the environment state.</typeparam>
    /// <typeparam name="TAction">The type used to represent actions.</typeparam>
    /// <typeparam name="T">The numeric type used for calculations.</typeparam>
    /// <remarks>
    /// <para>
    /// Prioritized Experience Replay (PER) improves upon standard experience replay by sampling
    /// experiences with higher expected learning progress more frequently. This implementation
    /// uses a sum tree data structure for efficient sampling based on priorities.
    /// </para>
    /// </remarks>
    public class PrioritizedReplayBuffer<TState, TAction, T> : IPrioritizedReplayBuffer<TState, TAction, T>
    {
        /// <summary>
        /// Gets the numeric operations for type T.
        /// </summary>
        protected INumericOperations<T> NumOps => MathHelper.GetNumericOperations<T>();
        
        private readonly SumTree<T> _sumTree = default!;
        private readonly Experience<TState, TAction, T>[] _experiences;
        private readonly T _alpha = default!;
        private readonly T _betaIncrement = default!;
        private readonly T _epsilonPriority = default!;
        private readonly T _maxPriority = default!;
        private readonly Random _random = default!;
        private T _currentBeta = default!;
        private int _currentIndex;
        private bool _isFull;

        /// <summary>
        /// Gets the current number of experiences in the buffer.
        /// </summary>
        public int Count => _isFull ? _experiences.Length : _currentIndex;

        /// <summary>
        /// Gets the maximum capacity of the buffer.
        /// </summary>
        public int Capacity { get; }

        /// <summary>
        /// Gets a value indicating whether the buffer is full.
        /// </summary>
        public bool IsFull => _isFull;

        /// <summary>
        /// Gets a value indicating whether the buffer supports prioritization.
        /// </summary>
        public bool SupportsPrioritization => true;

        /// <summary>
        /// Gets or sets the current beta value for importance sampling weights.
        /// </summary>
        public T Beta
        {
            get => _currentBeta;
            set => _currentBeta = value;
        }

        /// <summary>
        /// Gets the alpha parameter which controls how much prioritization is used.
        /// </summary>
        public double Alpha { get; }

        /// <summary>
        /// Initializes a new instance of the <see cref="PrioritizedReplayBuffer{TState, TAction, T}"/> class.
        /// </summary>
        /// <param name="capacity">Maximum capacity of the buffer.</param>
        /// <param name="alpha">Exponent determining how much prioritization is used (0: no prioritization, 1: full prioritization).</param>
        /// <param name="beta">Importance sampling weight correction exponent.</param>
        /// <param name="betaIncrement">Value to increment beta by during learning.</param>
        /// <param name="epsilonPriority">Small constant added to priorities to ensure non-zero sampling probability.</param>
        /// <param name="seed">Random seed for reproducibility.</param>
        public PrioritizedReplayBuffer(
            int capacity,
            double alpha = 0.6,
            double beta = 0.4,
            double betaIncrement = 0.001,
            double epsilonPriority = 1e-5,
            int? seed = null)
        {
            Capacity = capacity;
            Alpha = alpha;
            _experiences = new Experience<TState, TAction, T>[capacity];
            _sumTree = new SumTree<T>(capacity, NumOps);
            _alpha = NumOps.FromDouble(alpha);
            _currentBeta = NumOps.FromDouble(beta);
            _betaIncrement = NumOps.FromDouble(betaIncrement);
            _epsilonPriority = NumOps.FromDouble(epsilonPriority);
            _maxPriority = NumOps.One;
            _random = seed.HasValue ? new Random(seed.Value) : new Random();
            _currentIndex = 0;
            _isFull = false;
        }

        /// <summary>
        /// Adds a new experience to the buffer with maximum priority.
        /// </summary>
        /// <param name="state">The state before the action was taken.</param>
        /// <param name="action">The action that was taken.</param>
        /// <param name="reward">The reward received after taking the action.</param>
        /// <param name="nextState">The state after the action was taken.</param>
        /// <param name="done">A flag indicating whether the episode ended after this action.</param>
        public virtual void Add(TState state, TAction action, T reward, TState nextState, bool done)
        {
            Add(state, action, reward, nextState, done, _maxPriority);
        }

        /// <summary>
        /// Adds a new experience to the buffer with the specified priority.
        /// </summary>
        /// <param name="state">The state before the action was taken.</param>
        /// <param name="action">The action that was taken.</param>
        /// <param name="reward">The reward received after taking the action.</param>
        /// <param name="nextState">The state after the action was taken.</param>
        /// <param name="done">A flag indicating whether the episode ended after this action.</param>
        /// <param name="priority">The priority of the experience.</param>
        public void Add(TState state, TAction action, T reward, TState nextState, bool done, T priority)
        {
            // Create new experience
            var experience = new Experience<TState, TAction, T>
            {
                State = state,
                Action = action,
                Reward = reward,
                NextState = nextState,
                Done = done
            };

            _experiences[_currentIndex] = experience;

            // Calculate priority value for sum tree (apply alpha)
            T priorityAlpha = NumOps.Power(priority, _alpha);

            // Update sum tree
            _sumTree.Update(_currentIndex, priorityAlpha);

            // Update index
            _currentIndex = (_currentIndex + 1) % _experiences.Length;
            if (_currentIndex == 0)
            {
                _isFull = true;
            }
        }

        /// <summary>
        /// Samples a batch of experiences from the buffer based on their priorities.
        /// </summary>
        /// <param name="batchSize">The number of experiences to sample.</param>
        /// <returns>A tuple containing arrays of states, actions, rewards, next states, done flags, and importance sampling weights.</returns>
        public (TState[] states, TAction[] actions, T[] rewards, TState[] nextStates, bool[] dones) Sample(int batchSize)
        {
            var batch = SampleBatch(batchSize);
            return (batch.States, batch.Actions, batch.Rewards, batch.NextStates, batch.Dones);
        }

        /// <summary>
        /// Samples a batch of experiences from the buffer with additional information.
        /// </summary>
        /// <param name="batchSize">The number of experiences to sample.</param>
        /// <returns>A batch with states, actions, rewards, next states, done flags, indices, and weights.</returns>
        public PrioritizedReplayBatch<TState, TAction, T> SampleBatch(int batchSize)
        {
            return SamplePrioritized(batchSize, _currentBeta);
        }

        /// <summary>
        /// Samples a batch of experiences from the buffer using prioritized sampling.
        /// </summary>
        /// <param name="batchSize">The number of experiences to sample.</param>
        /// <param name="beta">The beta parameter for importance sampling weights.</param>
        /// <returns>A batch of experiences with importance sampling weights.</returns>
        public PrioritizedReplayBatch<TState, TAction, T> SamplePrioritized(int batchSize, T beta)
        {
            // Validate batch size
            if (batchSize > Count)
            {
                throw new ArgumentException($"Requested batch size {batchSize} exceeds buffer size {Count}.");
            }

            int actualSize = Count;
            var states = new TState[batchSize];
            var actions = new TAction[batchSize];
            var rewards = new T[batchSize];
            var nextStates = new TState[batchSize];
            var dones = new bool[batchSize];
            var indices = new int[batchSize];
            var weights = new T[batchSize];

            // Calculate segment size for priority-based sampling
            T segment = NumOps.Divide(_sumTree.Total, NumOps.FromDouble(batchSize));
            
            // Incrementally increase beta to 1 for more accurate bias correction
            _currentBeta = MathHelper.Min(NumOps.Add(beta, _betaIncrement), NumOps.One);

            T maxWeight = NumOps.Zero;

            // Sample experiences based on priorities
            for (int i = 0; i < batchSize; i++)
            {
                // Calculate the priority segment range
                T a = NumOps.Multiply(NumOps.FromDouble(i), segment);
                T b = NumOps.Multiply(NumOps.FromDouble(i + 1), segment);

                // Sample uniformly from the segment
                T value = NumOps.Add(a, 
                    NumOps.Multiply(NumOps.FromDouble(_random.NextDouble()), 
                        NumOps.Subtract(b, a)));

                // Retrieve experience from the sum tree
                (int idx, T priority) = _sumTree.Get(value);
                indices[i] = idx;

                // Get experience from buffer
                Experience<TState, TAction, T> experience = _experiences[idx];
                
                // Store the sampled experience
                states[i] = experience.State;
                actions[i] = experience.Action;
                rewards[i] = experience.Reward;
                nextStates[i] = experience.NextState;
                dones[i] = experience.Done;

                // Calculate importance sampling weights
                T probability = NumOps.Divide(priority, _sumTree.Total);
                weights[i] = NumOps.Power(
                    NumOps.Multiply(NumOps.FromDouble(actualSize), probability), 
                    NumOps.Negate(_currentBeta));

                maxWeight = MathHelper.Max(maxWeight, weights[i]);
            }

            // Normalize weights to stabilize updates
            if (!NumOps.Equals(maxWeight, NumOps.Zero))
            {
                for (int i = 0; i < batchSize; i++)
                {
                    weights[i] = NumOps.Divide(weights[i], maxWeight);
                }
            }

            return new PrioritizedReplayBatch<TState, TAction, T>(
                states, actions, rewards, nextStates, dones, indices, weights);
        }

        /// <summary>
        /// Updates the priorities for specific experiences in the buffer.
        /// </summary>
        /// <param name="indices">The indices of the experiences to update.</param>
        /// <param name="priorities">The new priority values.</param>
        public void UpdatePriorities(int[] indices, T[] priorities)
        {
            if (indices.Length != priorities.Length)
            {
                throw new ArgumentException("Indices and priorities arrays must have the same length.");
            }

            for (int i = 0; i < indices.Length; i++)
            {
                int idx = indices[i];
                if (idx >= 0 && idx < _experiences.Length && _experiences[idx] != null)
                {
                    // Add small epsilon to priorities to avoid zero probability
                    T priority = NumOps.Add(priorities[i], _epsilonPriority);
                    
                    // Clip priority to max value
                    priority = MathHelper.Min(priority, _maxPriority);
                    
                    // Apply alpha exponent for prioritization
                    T priorityAlpha = NumOps.Power(priority, _alpha);
                    
                    // Update the priority in the sum tree
                    _sumTree.Update(idx, priorityAlpha);
                }
            }
        }

        /// <summary>
        /// Clears all experiences from the buffer.
        /// </summary>
        public virtual void Clear()
        {
            Array.Clear(_experiences, 0, _experiences.Length);
            _sumTree.Clear();
            _currentIndex = 0;
            _isFull = false;
        }

        /// <summary>
        /// Gets the current size of the buffer.
        /// </summary>
        public int Size => Count;

        /// <summary>
        /// Samples a batch of experiences from the buffer.
        /// </summary>
        /// <param name="batchSize">The number of experiences to sample.</param>
        /// <returns>A batch of sampled experiences.</returns>
        ReplayBatch<TState, TAction, T> IReplayBuffer<TState, TAction, T>.SampleBatch(int batchSize)
        {
            var prioritizedBatch = SampleBatch(batchSize);
            // Convert PrioritizedReplayBatch to ReplayBatch
            return new ReplayBatch<TState, TAction, T>(
                prioritizedBatch.States,
                prioritizedBatch.Actions,
                prioritizedBatch.Rewards,
                prioritizedBatch.NextStates,
                prioritizedBatch.Dones);
        }

        /// <summary>
        /// Updates the priority of an experience at the specified index.
        /// </summary>
        /// <param name="index">The index of the experience to update.</param>
        /// <param name="priority">The new priority value.</param>
        public void UpdatePriority(int index, T priority)
        {
            UpdatePriorities(new[] { index }, new[] { priority });
        }

        /// <summary>
        /// Samples a prioritized batch of experiences from the buffer.
        /// </summary>
        /// <param name="batchSize">The number of experiences to sample.</param>
        /// <param name="beta">The importance sampling weight parameter.</param>
        /// <returns>A prioritized batch of sampled experiences.</returns>
        public ReplayBatch<TState, TAction, T> SamplePrioritizedBatch(int batchSize, T beta)
        {
            return SamplePrioritized(batchSize, beta);
        }
    }
}