using AiDotNet.Interfaces;
using AiDotNet.ReinforcementLearning.Memory;
using AiDotNet.Interfaces;
using AiDotNet.Helpers;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Diagnostics;

namespace AiDotNet.ReinforcementLearning.ReplayBuffers
{
    /// <summary>
    /// Implements a prioritized n-step experience replay buffer that combines both prioritized
    /// experience replay and n-step returns for reinforcement learning algorithms like Rainbow DQN.
    /// </summary>
    /// <typeparam name="TState">The type used to represent the environment state.</typeparam>
    /// <typeparam name="TAction">The type used to represent actions.</typeparam>
    /// <typeparam name="T">The numeric type used for calculations.</typeparam>
    /// <remarks>
    /// <para>
    /// This replay buffer combines two important improvements to DQN:
    /// 1. Prioritized Experience Replay (PER) - samples important transitions more frequently
    /// 2. N-step returns - uses multi-step bootstrapping to accelerate learning
    /// </para>
    /// <para>
    /// The n-step return is calculated as the discounted sum of rewards over n steps plus the 
    /// bootstrapped value of the state reached after n steps:
    /// R_t + γR_{t+1} + γ²R_{t+2} + ... + γ^{n-1}R_{t+n-1} + γ^n Q(S_{t+n})
    /// </para>
    /// </remarks>
    public class PrioritizedNStepReplayBuffer<TState, TAction, T> : PrioritizedReplayBuffer<TState, TAction, T>
    {
        private readonly int _nSteps;
        private readonly T _gamma = default!;
        private readonly Queue<(TState state, TAction action, T reward, TState nextState, bool done)> _nStepBuffer;
        private readonly LinkedList<(TState state, TAction action, T reward, bool done)> _recentExperiences;
        private readonly Dictionary<int, int> _experienceIdToIndex = default!;
        private int _experienceCounter;

        /// <summary>
        /// Initializes a new instance of the <see cref="PrioritizedNStepReplayBuffer{TState, TAction, T}"/> class.
        /// </summary>
        /// <param name="capacity">Maximum capacity of the buffer.</param>
        /// <param name="nSteps">Number of steps to accumulate rewards for n-step returns.</param>
        /// <param name="gamma">Discount factor for future rewards.</param>
        /// <param name="alpha">Exponent determining how much prioritization is used (0: no prioritization, 1: full prioritization).</param>
        /// <param name="beta">Importance sampling weight correction exponent.</param>
        /// <param name="betaIncrement">Value to increment beta by during learning.</param>
        /// <param name="epsilonPriority">Small constant added to priorities to ensure non-zero sampling probability.</param>
        /// <param name="seed">Random seed for reproducibility.</param>
        public PrioritizedNStepReplayBuffer(
            int capacity,
            int nSteps, 
            double gamma,
            double alpha = 0.6,
            double beta = 0.4,
            double betaIncrement = 0.001,
            double epsilonPriority = 1e-5,
            int? seed = null)
            : base(capacity, alpha, beta, betaIncrement, epsilonPriority, seed)
        {
            Debug.Assert(nSteps > 0, "N-steps must be a positive number");
            
            _nSteps = nSteps;
            _gamma = NumOps.FromDouble(gamma);
            _nStepBuffer = new Queue<(TState state, TAction action, T reward, TState nextState, bool done)>();
            _recentExperiences = new LinkedList<(TState state, TAction action, T reward, bool done)>();
            _experienceIdToIndex = new Dictionary<int, int>();
            _experienceCounter = 0;
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
            
            // The next state is either the terminal state or the state after n steps
            TState effectiveNextState = oldestDone ? oldestNextState : (stepsAccumulated < _nSteps ? nextState : GetNthState(stepsAccumulated));
            bool effectiveDone = foundTerminal;

            // Track the experience id for priority updates
            int experienceId = _experienceCounter++;
            
            // Add to parent prioritized buffer with n-step return
            base.Add(oldestState, oldestAction, nStepReward, effectiveNextState, effectiveDone, NumOps.One);

            // Map experience ID to its index in the buffer
            _experienceIdToIndex[experienceId] = Count - 1;
            
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

            // Track the experience id for priority updates
            int experienceId = _experienceCounter++;
            
            // Add to parent buffer with shortened n-step return
            base.Add(state, action, nStepReward, effectiveNextState, foundTerminal, NumOps.One);

            // Map experience ID to its index in the buffer
            _experienceIdToIndex[experienceId] = Count - 1;
        }

        /// <summary>
        /// Gets the state after n steps.
        /// </summary>
        private TState GetNthState(int stepsForward)
        {
            if (stepsForward <= 0 || _nStepBuffer.Count == 0)
            {
                throw new ArgumentException("Cannot get state with invalid steps forward value", nameof(stepsForward));
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
            _experienceIdToIndex.Clear();
            _experienceCounter = 0;
        }

        /// <summary>
        /// Updates priorities for specific experiences after TD error calculation.
        /// </summary>
        /// <param name="experienceIds">The IDs of experiences to update.</param>
        /// <param name="priorities">The new priority values based on TD errors.</param>
        public void UpdatePrioritiesByExperienceId(int[] experienceIds, T[] priorities)
        {
            if (experienceIds.Length != priorities.Length)
            {
                throw new ArgumentException("Experience IDs and priorities arrays must have the same length.");
            }

            var indices = new int[experienceIds.Length];
            for (int i = 0; i < experienceIds.Length; i++)
            {
                if (_experienceIdToIndex.TryGetValue(experienceIds[i], out int index))
                {
                    indices[i] = index;
                }
                else
                {
                    throw new KeyNotFoundException($"Experience ID {experienceIds[i]} not found in the buffer.");
                }
            }

            // Update priorities in the base prioritized buffer
            UpdatePriorities(indices, priorities);
        }
    }
}