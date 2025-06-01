using AiDotNet.LinearAlgebra;
using AiDotNet.Interfaces;
using AiDotNet.ReinforcementLearning.Interfaces;
using AiDotNet.ReinforcementLearning.ReplayBuffers;
using AiDotNet.ReinforcementLearning.Exploration;
using System;

namespace AiDotNet.ReinforcementLearning.Agents
{
    /// <summary>
    /// Base class for actor-critic reinforcement learning agents (DDPG, TD3, SAC, etc.).
    /// </summary>
    /// <typeparam name="TState">The type used to represent the environment state, typically Tensor<double>&lt;T&gt;.</typeparam>
    /// <typeparam name="TAction">The type used to represent actions, typically Vector<double>&lt;T&gt; for continuous action spaces.</typeparam>
    /// <typeparam name="T">The numeric type used for calculations (float, double, etc.).</typeparam>
    /// <typeparam name="TActor">The type of the actor (policy) network.</typeparam>
    /// <typeparam name="TCritic">The type of the critic (value function) network.</typeparam>
    public abstract class ActorCriticAgentBase<TState, TAction, T, TActor, TCritic> 
        : AgentBase<TState, TAction, T>
        where TState : Tensor<T>
        where TActor : class
        where TCritic : class
    {
        /// <summary>
        /// Gets the actor network, which maps states to actions.
        /// </summary>
        protected TActor Actor { get; }

        /// <summary>
        /// Gets the target actor network, used for stable learning.
        /// </summary>
        protected TActor ActorTarget { get; }

        /// <summary>
        /// Gets the critic network, which maps state-action pairs to Q-values.
        /// </summary>
        protected TCritic Critic { get; }

        /// <summary>
        /// Gets the target critic network, used for stable learning.
        /// </summary>
        protected TCritic CriticTarget { get; }

        /// <summary>
        /// Gets the replay buffer used for storing and sampling experiences.
        /// </summary>
        protected IReplayBuffer<TState, TAction, T> ReplayBuffer { get; }

        /// <summary>
        /// Gets the exploration strategy used for action selection.
        /// </summary>
        protected IExplorationStrategy<TAction, T> ExplorationStrategy { get; }

        /// <summary>
        /// Gets the warm-up steps before learning begins.
        /// </summary>
        protected int WarmUpSteps { get; }

        /// <summary>
        /// Gets a value indicating whether to use gradient clipping.
        /// </summary>
        protected bool UseGradientClipping { get; }

        /// <summary>
        /// Gets the maximum gradient norm for clipping.
        /// </summary>
        protected T MaxGradientNorm { get; }

        /// <summary>
        /// Initializes a new instance of the <see cref="ActorCriticAgentBase{TState, TAction, T, TActor, TCritic}"/> class.
        /// </summary>
        /// <param name="actor">The actor network.</param>
        /// <param name="actorTarget">The target actor network.</param>
        /// <param name="critic">The critic network.</param>
        /// <param name="criticTarget">The target critic network.</param>
        /// <param name="replayBuffer">The replay buffer for storing experiences.</param>
        /// <param name="explorationStrategy">The exploration strategy for action selection.</param>
        /// <param name="gamma">The discount factor for future rewards.</param>
        /// <param name="tau">The soft update factor for target networks.</param>
        /// <param name="batchSize">The batch size for training.</param>
        /// <param name="warmUpSteps">The number of warm-up steps before learning begins.</param>
        /// <param name="useGradientClipping">Whether to use gradient clipping.</param>
        /// <param name="maxGradientNorm">The maximum gradient norm for clipping.</param>
        /// <param name="seed">Optional seed for the random number generator.</param>
        protected ActorCriticAgentBase(
            TActor actor,
            TActor actorTarget,
            TCritic critic,
            TCritic criticTarget,
            IReplayBuffer<TState, TAction, T> replayBuffer,
            IExplorationStrategy<TAction, T> explorationStrategy,
            double gamma,
            double tau,
            int batchSize,
            int warmUpSteps,
            bool useGradientClipping,
            double maxGradientNorm,
            int? seed = null)
            : base(gamma, tau, batchSize, seed)
        {
            Actor = actor ?? throw new ArgumentNullException(nameof(actor));
            ActorTarget = actorTarget ?? throw new ArgumentNullException(nameof(actorTarget));
            Critic = critic ?? throw new ArgumentNullException(nameof(critic));
            CriticTarget = criticTarget ?? throw new ArgumentNullException(nameof(criticTarget));
            ReplayBuffer = replayBuffer ?? throw new ArgumentNullException(nameof(replayBuffer));
            ExplorationStrategy = explorationStrategy ?? throw new ArgumentNullException(nameof(explorationStrategy));
            WarmUpSteps = warmUpSteps;
            UseGradientClipping = useGradientClipping;
            MaxGradientNorm = NumOps.FromDouble(maxGradientNorm);
            
            // Initialize target networks
            InitializeTargetNetworks();
        }

        /// <summary>
        /// Updates the agent's knowledge based on an experience tuple.
        /// </summary>
        /// <param name="state">The state before the action was taken.</param>
        /// <param name="action">The action that was taken.</param>
        /// <param name="reward">The reward received after taking the action.</param>
        /// <param name="nextState">The state after the action was taken.</param>
        /// <param name="done">A flag indicating whether the episode ended after this action.</param>
        public override void Learn(TState state, TAction action, T reward, TState nextState, bool done)
        {
            if (!IsTraining)
                return;

            // Increment step counter
            IncrementStepCounter();

            // Store experience in replay buffer
            ReplayBuffer.Add(state, action, reward, nextState, done);

            // Only start training after collecting enough samples
            if (TotalSteps < WarmUpSteps || ReplayBuffer.Size < BatchSize)
                return;

            // Sample a batch of experiences and learn from it
            var batch = ReplayBuffer.SampleBatch(BatchSize);
            var states = batch.States;
            var actions = batch.Actions;
            var rewards = batch.Rewards;
            var nextStates = batch.NextStates;
            var dones = batch.Dones;
            var weights = batch.Weights;
            var indices = batch.Indices;

            // Update critic and actor networks
            UpdateNetworks(states, actions, rewards, nextStates, dones, weights, indices);

            // Update target networks
            if (ShouldUpdateTargets())
            {
                UpdateTargetNetworks();
            }
        }

        /// <summary>
        /// Initializes the target networks by copying parameters from the main networks.
        /// </summary>
        protected abstract void InitializeTargetNetworks();

        /// <summary>
        /// Updates the critic and actor networks based on a batch of experiences.
        /// </summary>
        /// <param name="states">Batch of states.</param>
        /// <param name="actions">Batch of actions.</param>
        /// <param name="rewards">Batch of rewards.</param>
        /// <param name="nextStates">Batch of next states.</param>
        /// <param name="dones">Batch of episode termination flags.</param>
        /// <param name="weights">Importance sampling weights (for prioritized replay).</param>
        /// <param name="indices">Indices of the sampled experiences (for prioritized replay).</param>
        protected abstract void UpdateNetworks(
            TState[] states,
            TAction[] actions,
            T[] rewards,
            TState[] nextStates,
            bool[] dones,
            T[] weights,
            int[] indices);

        /// <summary>
        /// Updates the target networks using soft update.
        /// </summary>
        protected abstract void UpdateTargetNetworks();

        /// <summary>
        /// Determines whether target networks should be updated in the current step.
        /// </summary>
        /// <returns>True if target networks should be updated, otherwise false.</returns>
        protected virtual bool ShouldUpdateTargets()
        {
            // By default, update every step
            return true;
        }

        /// <summary>
        /// Generates a random action for exploration, especially during warm-up.
        /// </summary>
        /// <returns>A random action.</returns>
        protected abstract TAction GenerateRandomAction();
    }
}