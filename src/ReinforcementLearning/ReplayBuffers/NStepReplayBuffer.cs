using AiDotNet.Interfaces;
using AiDotNet.Interfaces;
using AiDotNet.Helpers;
using System.Collections.Generic;
using System.Diagnostics;

namespace AiDotNet.ReinforcementLearning.ReplayBuffers
{
    /// <summary>
    /// Implements an n-step experience replay buffer for reinforcement learning algorithms like Rainbow DQN.
    /// </summary>
    /// <typeparam name="TState">The type used to represent the environment state.</typeparam>
    /// <typeparam name="TAction">The type used to represent actions.</typeparam>
    /// <typeparam name="T">The numeric type used for calculations.</typeparam>
    /// <remarks>
    /// <para>
    /// N-step replay buffers enhance learning by using n-step returns instead of single-step returns.
    /// This implements the multi-step learning component used in Rainbow DQN, which helps propagate
    /// reward signals more efficiently through the value function estimation.
    /// </para>
    /// <para>
    /// The n-step return is calculated as the discounted sum of rewards over n steps plus the 
    /// bootstrapped value of the state reached after n steps:
    /// R_t + γR_{t+1} + γ²R_{t+2} + ... + γ^{n-1}R_{t+n-1} + γ^n Q(S_{t+n})
    /// </para>
    /// </remarks>
    public class NStepReplayBuffer<TState, TAction, T> : ReplayBufferBase<TState, TAction, T>
    {
        /// <summary>
        /// Gets the numeric operations for type T.
        /// </summary>
        protected INumericOperations<T> NumOps => MathHelper.GetNumericOperations<T>();
        private readonly int _nSteps;
        private readonly T _gamma = default!;
        private readonly Queue<(TState state, TAction action, T reward, TState nextState, bool done)> _nStepBuffer;
        private readonly LinkedList<(TState state, TAction action, T reward, bool done)> _recentExperiences;

        /// <summary>
        /// Initializes a new instance of the <see cref="NStepReplayBuffer{TState, TAction, T}"/> class.
        /// </summary>
        /// <param name="capacity">The maximum capacity of the buffer.</param>
        /// <param name="nSteps">Number of steps to accumulate rewards for n-step returns.</param>
        /// <param name="gamma">Discount factor for future rewards.</param>
        /// <param name="seed">Optional random seed for reproducibility.</param>
        public NStepReplayBuffer(int capacity, int nSteps, double gamma, int? seed = null)
            : base(capacity, seed)
        {
            Debug.Assert(nSteps > 0, "N-steps must be a positive number");
            
            _nSteps = nSteps;
            _gamma = NumOps.FromDouble(gamma);
            _nStepBuffer = new Queue<(TState state, TAction action, T reward, TState nextState, bool done)>();
            _recentExperiences = new LinkedList<(TState state, TAction action, T reward, bool done)>();
        }

        /// <summary>
        /// Adds a new experience to the buffer with n-step return calculation.
        /// </summary>
        /// <param name="state">The state before the action was taken.</param>
        /// <param name="action">The action that was taken.</param>
        /// <param name="reward">The reward received after taking the action.</param>
        /// <param name="nextState">The state after the action was taken.</param>
        /// <param name="done">A flag indicating whether the episode ended after this action.</param>
        public override void Add(TState state, TAction action, T reward, TState nextState, bool done)
        {
            // Store experience in n-step buffer
            _nStepBuffer.Enqueue((state, action, reward, nextState, done));
            _recentExperiences.AddLast((state, action, reward, done));

            // If we don't have enough experiences for n-step return, just return
            if (_nStepBuffer.Count < _nSteps && !done)
            {
                return;
            }

            // Process the oldest experience with its n-step return
            var (oldestState, oldestAction, oldestReward, oldestNextState, oldestDone) = _nStepBuffer.Dequeue();
            
            // Calculate n-step return with discounted rewards
            T nStepReward = oldestReward;
            T discountFactor = _gamma;
            
            var node = _recentExperiences.First;
            _recentExperiences.RemoveFirst();  // Remove the oldest experience

            // Skip the oldest experience which we've already processed
            node = node?.Next;
            
            // Accumulate discounted rewards for the n steps
            int stepsAccumulated = 1;
            bool foundTerminal = oldestDone;
            
            while (node != null && stepsAccumulated < _nSteps && !foundTerminal)
            {
                var experience = node.Value;
                nStepReward = NumOps.Add(nStepReward, NumOps.Multiply(discountFactor, experience.reward));
                discountFactor = NumOps.Multiply(discountFactor, _gamma);
                foundTerminal = experience.done;
                stepsAccumulated++;
                node = node.Next;
            }
            
            // Add the experience to the underlying replay buffer with the n-step return
            // The next state is either the terminal state or the state after n steps
            TState effectiveNextState = oldestDone ? oldestNextState : (stepsAccumulated < _nSteps ? nextState : GetNthState(stepsAccumulated));
            bool effectiveDone = foundTerminal;

            // Add to parent buffer with n-step return
            base.Add(oldestState, oldestAction, nStepReward, effectiveNextState, effectiveDone);
            
            // If this was a terminal state, clear all n-step buffers
            if (done)
            {
                // Process all remaining experiences in the buffer with appropriate n-step returns
                while (_nStepBuffer.Count > 0)
                {
                    ProcessRemainingExperience();
                }
                
                _recentExperiences.Clear();
            }
        }

        /// <summary>
        /// Processes remaining experiences in the buffer after a terminal state.
        /// </summary>
        private void ProcessRemainingExperience()
        {
            if (_nStepBuffer.Count == 0) return;
            
            var (state, action, reward, nextState, done) = _nStepBuffer.Dequeue();
            _recentExperiences.RemoveFirst();
            
            // Calculate a shorter n-step return since we hit a terminal state
            T nStepReward = reward;
            T discountFactor = _gamma;
            bool foundTerminal = done;
            
            var node = _recentExperiences.First;
            int stepsAccumulated = 1;
            
            while (node != null && !foundTerminal)
            {
                var experience = node.Value;
                nStepReward = NumOps.Add(nStepReward, NumOps.Multiply(discountFactor, experience.reward));
                discountFactor = NumOps.Multiply(discountFactor, _gamma);
                foundTerminal = experience.done;
                stepsAccumulated++;
                node = node.Next;
            }
            
            // The effective next state is the last state before the terminal state
            TState effectiveNextState = foundTerminal ? GetStateAtIndex(_nStepBuffer.Count - 1) : nextState;
            
            // Add to parent buffer with shortened n-step return
            base.Add(state, action, nStepReward, effectiveNextState, foundTerminal);
        }

        /// <summary>
        /// Gets the state after n steps.
        /// </summary>
        private TState GetNthState(int stepsForward)
        {
            if (stepsForward <= 0 || _nStepBuffer.Count == 0)
            {
                throw new InvalidOperationException("Cannot get state with invalid steps forward value");
            }
            
            // If we're asking for a state beyond what we have, return the last available state
            if (stepsForward >= _nStepBuffer.Count)
            {
                return _nStepBuffer.Last().nextState;
            }
            
            // Return the state at the specified index
            return GetStateAtIndex(stepsForward - 1);
        }

        /// <summary>
        /// Gets the state at the specified index in the n-step buffer.
        /// </summary>
        private TState GetStateAtIndex(int index)
        {
            if (index < 0 || index >= _nStepBuffer.Count)
            {
                throw new IndexOutOfRangeException("Index out of range for n-step buffer");
            }
            
            // Have to convert queue to array to access by index
            var array = _nStepBuffer.ToArray();
            return array[index].nextState;
        }

        /// <summary>
        /// Clears all experiences from the buffer.
        /// </summary>
        public override void Clear()
        {
            base.Clear();
            _nStepBuffer.Clear();
            _recentExperiences.Clear();
        }
    }
}