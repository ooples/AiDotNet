using AiDotNet.LinearAlgebra;
using AiDotNet.Interfaces;
using AiDotNet.ReinforcementLearning.Interfaces;
using AiDotNet.ReinforcementLearning.ReplayBuffers;
using AiDotNet.ReinforcementLearning.Exploration;
using System;

namespace AiDotNet.ReinforcementLearning.Agents
{
    /// <summary>
    /// Base class for actor-critic reinforcement learning agents with dual critics (TD3, SAC, etc.).
    /// </summary>
    /// <typeparam name="TState">The type used to represent the environment state, typically Tensor<double>&lt;T&gt;.</typeparam>
    /// <typeparam name="TAction">The type used to represent actions, typically Vector<double>&lt;T&gt; for continuous action spaces.</typeparam>
    /// <typeparam name="T">The numeric type used for calculations (float, double, etc.).</typeparam>
    /// <typeparam name="TActor">The type of the actor (policy) network.</typeparam>
    public abstract class DualCriticAgentBase<TState, TAction, T, TActor> 
        : AgentBase<TState, TAction, T>
        where TState : Tensor<T>
        where TActor : class
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
        /// Gets the first critic network, which maps state-action pairs to Q-values.
        /// </summary>
        protected IActionValueFunction<TState, TAction, T> Critic1 { get; }

        /// <summary>
        /// Gets the target first critic network, used for stable learning.
        /// </summary>
        protected IActionValueFunction<TState, TAction, T> Critic1Target { get; }

        /// <summary>
        /// Gets the second critic network, which maps state-action pairs to Q-values.
        /// </summary>
        protected IActionValueFunction<TState, TAction, T> Critic2 { get; }

        /// <summary>
        /// Gets the target second critic network, used for stable learning.
        /// </summary>
        protected IActionValueFunction<TState, TAction, T> Critic2Target { get; }

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
        /// Gets the policy update frequency.
        /// </summary>
        protected int PolicyUpdateFrequency { get; }

        /// <summary>
        /// Gets a value indicating whether to use the minimum Q-value from the two critics.
        /// </summary>
        protected bool UseMinimumQValue { get; }

        /// <summary>
        /// Initializes a new instance of the <see cref="DualCriticAgentBase{TState, TAction, T, TActor}"/> class.
        /// </summary>
        /// <param name="actor">The actor network.</param>
        /// <param name="actorTarget">The target actor network.</param>
        /// <param name="critic1">The first critic network.</param>
        /// <param name="critic1Target">The target first critic network.</param>
        /// <param name="critic2">The second critic network.</param>
        /// <param name="critic2Target">The target second critic network.</param>
        /// <param name="replayBuffer">The replay buffer for storing experiences.</param>
        /// <param name="explorationStrategy">The exploration strategy for action selection.</param>
        /// <param name="gamma">The discount factor for future rewards.</param>
        /// <param name="tau">The soft update factor for target networks.</param>
        /// <param name="batchSize">The batch size for training.</param>
        /// <param name="warmUpSteps">The number of warm-up steps before learning begins.</param>
        /// <param name="useGradientClipping">Whether to use gradient clipping.</param>
        /// <param name="maxGradientNorm">The maximum gradient norm for clipping.</param>
        /// <param name="policyUpdateFrequency">The frequency of policy updates.</param>
        /// <param name="useMinimumQValue">Whether to use the minimum Q-value from the two critics.</param>
        /// <param name="seed">Optional seed for the random number generator.</param>
        protected DualCriticAgentBase(
            TActor actor,
            TActor actorTarget,
            IActionValueFunction<TState, TAction, T> critic1,
            IActionValueFunction<TState, TAction, T> critic1Target,
            IActionValueFunction<TState, TAction, T> critic2,
            IActionValueFunction<TState, TAction, T> critic2Target,
            IReplayBuffer<TState, TAction, T> replayBuffer,
            IExplorationStrategy<TAction, T> explorationStrategy,
            double gamma,
            double tau,
            int batchSize,
            int warmUpSteps,
            bool useGradientClipping,
            double maxGradientNorm,
            int policyUpdateFrequency,
            bool useMinimumQValue,
            int? seed = null)
            : base(gamma, tau, batchSize, seed)
        {
            Actor = actor ?? throw new ArgumentNullException(nameof(actor));
            ActorTarget = actorTarget ?? throw new ArgumentNullException(nameof(actorTarget));
            Critic1 = critic1 ?? throw new ArgumentNullException(nameof(critic1));
            Critic1Target = critic1Target ?? throw new ArgumentNullException(nameof(critic1Target));
            Critic2 = critic2 ?? throw new ArgumentNullException(nameof(critic2));
            Critic2Target = critic2Target ?? throw new ArgumentNullException(nameof(critic2Target));
            ReplayBuffer = replayBuffer ?? throw new ArgumentNullException(nameof(replayBuffer));
            ExplorationStrategy = explorationStrategy ?? throw new ArgumentNullException(nameof(explorationStrategy));
            WarmUpSteps = warmUpSteps;
            UseGradientClipping = useGradientClipping;
            MaxGradientNorm = NumOps.FromDouble(maxGradientNorm);
            PolicyUpdateFrequency = policyUpdateFrequency;
            UseMinimumQValue = useMinimumQValue;
            
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
            // Sample batch from replay buffer
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
            // By default, update based on policy update frequency
            return TotalSteps % PolicyUpdateFrequency == 0;
        }

        /// <summary>
        /// Should the policy be updated in the current step?
        /// </summary>
        /// <returns>True if the policy should be updated, otherwise false.</returns>
        protected virtual bool ShouldUpdatePolicy()
        {
            // By default, update based on policy update frequency
            return TotalSteps % PolicyUpdateFrequency == 0;
        }

        /// <summary>
        /// Generates a random action for exploration, especially during warm-up.
        /// </summary>
        /// <returns>A random action.</returns>
        protected abstract TAction GenerateRandomAction();

        /// <summary>
        /// Gets the minimum or average of the two critic values, depending on UseMinimumQValue.
        /// </summary>
        /// <param name="q1">The Q-value from the first critic.</param>
        /// <param name="q2">The Q-value from the second critic.</param>
        /// <returns>The minimum or average of the two critic values.</returns>
        protected T GetCriticValue(T q1, T q2)
        {
            if (UseMinimumQValue)
            {
                return NumOps.LessThan(q1, q2) ? q1 : q2;
            }
            else
            {
                return NumOps.Divide(NumOps.Add(q1, q2), NumOps.FromDouble(2.0));
            }
        }
        
        /// <summary>
        /// Gets the parameters of the agent (actor and critics).
        /// </summary>
        /// <returns>A vector containing all parameters.</returns>
        public virtual Vector<T> GetParameters()
        {
            var allParameters = new List<T>();
            
            // Get actor parameters if it supports IParameterizable or IStochasticPolicy
            if (Actor is IParameterizable<T, TState, TAction> parameterizableActor)
            {
                var actorParams = parameterizableActor.GetParameters();
                for (int i = 0; i < actorParams.Length; i++)
                {
                    allParameters.Add(actorParams[i]);
                }
            }
            else if (Actor is IStochasticPolicy<TState, TAction, T> stochasticActor)
            {
                var actorParams = stochasticActor.GetParameters();
                for (int i = 0; i < actorParams.Length; i++)
                {
                    allParameters.Add(actorParams[i]);
                }
            }
            
            // Get critic1 parameters
            if (Critic1 != null)
            {
                var critic1Params = Critic1.GetParameters();
                for (int i = 0; i < critic1Params.Length; i++)
                {
                    allParameters.Add(critic1Params[i]);
                }
            }
            
            // Get critic2 parameters
            if (Critic2 != null)
            {
                var critic2Params = Critic2.GetParameters();
                for (int i = 0; i < critic2Params.Length; i++)
                {
                    allParameters.Add(critic2Params[i]);
                }
            }
            
            // Get target network parameters if they exist
            if (Critic1Target != null)
            {
                var critic1TargetParams = Critic1Target.GetParameters();
                for (int i = 0; i < critic1TargetParams.Length; i++)
                {
                    allParameters.Add(critic1TargetParams[i]);
                }
            }
            
            if (Critic2Target != null)
            {
                var critic2TargetParams = Critic2Target.GetParameters();
                for (int i = 0; i < critic2TargetParams.Length; i++)
                {
                    allParameters.Add(critic2TargetParams[i]);
                }
            }
            
            return new Vector<T>([.. allParameters]);
        }
        
        /// <summary>
        /// Sets the parameters of the agent (actor and critics).
        /// </summary>
        /// <param name="parameters">A vector containing all parameters.</param>
        public virtual void SetParameters(Vector<T> parameters)
        {
            int index = 0;
            
            // Set actor parameters if it supports IParameterizable or IStochasticPolicy
            if (Actor is IParameterizable<T, TState, TAction> parameterizableActor)
            {
                var actorParams = parameterizableActor.GetParameters();
                var newActorParams = new Vector<T>(actorParams.Length);
                for (int i = 0; i < actorParams.Length; i++)
                {
                    newActorParams[i] = parameters[index++];
                }
                parameterizableActor.SetParameters(newActorParams);
            }
            else if (Actor is IStochasticPolicy<TState, TAction, T> stochasticActor)
            {
                var actorParams = stochasticActor.GetParameters();
                var newActorParams = new Vector<T>(actorParams.Length);
                for (int i = 0; i < actorParams.Length; i++)
                {
                    newActorParams[i] = parameters[index++];
                }
                stochasticActor.SetParameters(newActorParams);
            }
            
            // Set critic1 parameters
            if (Critic1 != null)
            {
                var critic1Params = Critic1.GetParameters();
                var newCritic1Params = new Vector<T>(critic1Params.Length);
                for (int i = 0; i < critic1Params.Length; i++)
                {
                    newCritic1Params[i] = parameters[index++];
                }
                Critic1.SetParameters(newCritic1Params);
            }
            
            // Set critic2 parameters
            if (Critic2 != null)
            {
                var critic2Params = Critic2.GetParameters();
                var newCritic2Params = new Vector<T>(critic2Params.Length);
                for (int i = 0; i < critic2Params.Length; i++)
                {
                    newCritic2Params[i] = parameters[index++];
                }
                Critic2.SetParameters(newCritic2Params);
            }
            
            // Set target network parameters if they exist
            if (Critic1Target != null)
            {
                var critic1TargetParams = Critic1Target.GetParameters();
                var newCritic1TargetParams = new Vector<T>(critic1TargetParams.Length);
                for (int i = 0; i < critic1TargetParams.Length; i++)
                {
                    newCritic1TargetParams[i] = parameters[index++];
                }
                Critic1Target.SetParameters(newCritic1TargetParams);
            }
            
            if (Critic2Target != null)
            {
                var critic2TargetParams = Critic2Target.GetParameters();
                var newCritic2TargetParams = new Vector<T>(critic2TargetParams.Length);
                for (int i = 0; i < critic2TargetParams.Length; i++)
                {
                    newCritic2TargetParams[i] = parameters[index++];
                }
                Critic2Target.SetParameters(newCritic2TargetParams);
            }
        }
    }
}