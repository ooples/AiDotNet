---
title: "Reinforcement Learning"
description: "All 88 public types in the AiDotNet.reinforcementlearning namespace, organized by kind."
section: "API Reference"
---

**88** public types in this namespace, organized by kind.

## Models & Types (73)

| Type | Summary |
|:-----|:--------|
| [`A2CAgent<T>`](/docs/reference/wiki/reinforcementlearning/a2cagent/) | Advantage Actor-Critic (A2C) agent for reinforcement learning. |
| [`A3CAgent<T>`](/docs/reference/wiki/reinforcementlearning/a3cagent/) | Asynchronous Advantage Actor-Critic (A3C) agent for reinforcement learning. |
| [`BetaPolicy<T>`](/docs/reference/wiki/reinforcementlearning/betapolicy/) | Policy using Beta distribution for bounded continuous action spaces. |
| [`BoltzmannExploration<T>`](/docs/reference/wiki/reinforcementlearning/boltzmannexploration/) | Boltzmann (softmax) exploration with temperature-based action selection. |
| [`CQLAgent<T>`](/docs/reference/wiki/reinforcementlearning/cqlagent/) | Conservative Q-Learning (CQL) agent for offline reinforcement learning. |
| [`CartPoleEnvironment<T>`](/docs/reference/wiki/reinforcementlearning/cartpoleenvironment/) | Classic CartPole-v1 environment for reinforcement learning. |
| [`ContinuousPolicy<T>`](/docs/reference/wiki/reinforcementlearning/continuouspolicy/) | Policy for continuous action spaces using a neural network to output Gaussian parameters. |
| [`DDPGAgent<T>`](/docs/reference/wiki/reinforcementlearning/ddpgagent/) | Deep Deterministic Policy Gradient (DDPG) agent for continuous control. |
| [`DQNAgent<T>`](/docs/reference/wiki/reinforcementlearning/dqnagent/) | Deep Q-Network (DQN) agent for reinforcement learning. |
| [`DecisionTransformerAgent<T>`](/docs/reference/wiki/reinforcementlearning/decisiontransformeragent/) | Decision Transformer agent for offline reinforcement learning. |
| [`DeterministicBanditEnvironment<T>`](/docs/reference/wiki/reinforcementlearning/deterministicbanditenvironment/) | A deterministic multi-armed bandit environment for testing purposes. |
| [`DeterministicPolicy<T>`](/docs/reference/wiki/reinforcementlearning/deterministicpolicy/) | Deterministic policy for continuous action spaces. |
| [`DiscretePolicy<T>`](/docs/reference/wiki/reinforcementlearning/discretepolicy/) | Policy for discrete action spaces using a neural network to output action logits. |
| [`DoubleDQNAgent<T>`](/docs/reference/wiki/reinforcementlearning/doubledqnagent/) | Double Deep Q-Network (Double DQN) agent for reinforcement learning. |
| [`DoubleQLearningAgent<T>`](/docs/reference/wiki/reinforcementlearning/doubleqlearningagent/) | Double Q-Learning agent using two Q-tables to reduce overestimation bias. |
| [`DreamerAgent<T>`](/docs/reference/wiki/reinforcementlearning/dreameragent/) | Dreamer agent for model-based reinforcement learning. |
| [`DuelingDQNAgent<T>`](/docs/reference/wiki/reinforcementlearning/duelingdqnagent/) | Dueling Deep Q-Network agent for reinforcement learning. |
| [`DynaQAgent<T>`](/docs/reference/wiki/reinforcementlearning/dynaqagent/) | Dyna-Q agent combining learning and planning using a learned model. |
| [`DynaQPlusAgent<T>`](/docs/reference/wiki/reinforcementlearning/dynaqplusagent/) | Dyna-Q+ agent with exploration bonus for handling changing environments. |
| [`EpsilonGreedyBanditAgent<T>`](/docs/reference/wiki/reinforcementlearning/epsilongreedybanditagent/) | Epsilon-Greedy Multi-Armed Bandit agent. |
| [`EpsilonGreedyExploration<T>`](/docs/reference/wiki/reinforcementlearning/epsilongreedyexploration/) | Epsilon-greedy exploration: with probability epsilon, select random action. |
| [`EveryVisitMonteCarloAgent<T>`](/docs/reference/wiki/reinforcementlearning/everyvisitmontecarloagent/) | Every-Visit Monte Carlo agent that updates all visits to states in an episode. |
| [`ExpectedSARSAAgent<T>`](/docs/reference/wiki/reinforcementlearning/expectedsarsaagent/) | Expected SARSA agent using tabular methods. |
| [`Experience<T>`](/docs/reference/wiki/reinforcementlearning/experience/) | Simplified Experience record for Vector-based states and actions. |
| [`Experience<T, TState, TAction>`](/docs/reference/wiki/reinforcementlearning/experience-2/) | Represents a single experience tuple (s, a, r, s', done) for reinforcement learning. |
| [`FirstVisitMonteCarloAgent<T>`](/docs/reference/wiki/reinforcementlearning/firstvisitmontecarloagent/) | First-Visit Monte Carlo agent for episodic tasks. |
| [`GaussianNoiseExploration<T>`](/docs/reference/wiki/reinforcementlearning/gaussiannoiseexploration/) | Gaussian noise exploration for continuous action spaces. |
| [`GradientBanditAgent<T>`](/docs/reference/wiki/reinforcementlearning/gradientbanditagent/) | Gradient Bandit agent using softmax action preferences. |
| [`IQLAgent<T>`](/docs/reference/wiki/reinforcementlearning/iqlagent/) | Implicit Q-Learning (IQL) agent for offline reinforcement learning. |
| [`LSPIAgent<T>`](/docs/reference/wiki/reinforcementlearning/lspiagent/) | LSPI (Least-Squares Policy Iteration) agent using iterative policy improvement with LSTDQ. |
| [`LSTDAgent<T>`](/docs/reference/wiki/reinforcementlearning/lstdagent/) | LSTD (Least-Squares Temporal Difference) agent using direct solution for value function weights. |
| [`LinearQLearningAgent<T>`](/docs/reference/wiki/reinforcementlearning/linearqlearningagent/) | Linear Q-Learning agent using linear function approximation. |
| [`LinearSARSAAgent<T>`](/docs/reference/wiki/reinforcementlearning/linearsarsaagent/) | Linear SARSA agent using linear function approximation with on-policy learning. |
| [`MADDPGAgent<T>`](/docs/reference/wiki/reinforcementlearning/maddpgagent/) | Multi-Agent Deep Deterministic Policy Gradient (MADDPG) agent for cooperative and competitive multi-agent reinforcement learning with continuous action spaces. |
| [`MixedPolicy<T>`](/docs/reference/wiki/reinforcementlearning/mixedpolicy/) | Policy for environments with both discrete and continuous action spaces. |
| [`ModifiedPolicyIterationAgent<T>`](/docs/reference/wiki/reinforcementlearning/modifiedpolicyiterationagent/) | Modified Policy Iteration agent - hybrid of Policy Iteration and Value Iteration. |
| [`MonteCarloExploringStartsAgent<T>`](/docs/reference/wiki/reinforcementlearning/montecarloexploringstartsagent/) | Monte Carlo Exploring Starts agent for reinforcement learning. |
| [`MuZeroAgent<T>`](/docs/reference/wiki/reinforcementlearning/muzeroagent/) | MuZero agent combining tree search with learned models. |
| [`MultiModalPolicy<T>`](/docs/reference/wiki/reinforcementlearning/multimodalpolicy/) | Multi-modal policy using mixture of Gaussians for complex action distributions. |
| [`NStepQLearningAgent<T>`](/docs/reference/wiki/reinforcementlearning/nstepqlearningagent/) | N-step Q-Learning agent using multi-step off-policy returns. |
| [`NStepSARSAAgent<T>`](/docs/reference/wiki/reinforcementlearning/nstepsarsaagent/) | N-step SARSA agent using multi-step bootstrapping. |
| [`NoExploration<T>`](/docs/reference/wiki/reinforcementlearning/noexploration/) | No exploration - always use the policy's action directly (greedy). |
| [`OffPolicyMonteCarloAgent<T>`](/docs/reference/wiki/reinforcementlearning/offpolicymontecarloagent/) | Off-Policy Monte Carlo Control agent with weighted importance sampling. |
| [`OnPolicyMonteCarloAgent<T>`](/docs/reference/wiki/reinforcementlearning/onpolicymontecarloagent/) | On-Policy Monte Carlo Control agent with epsilon-greedy exploration. |
| [`OrnsteinUhlenbeckNoise<T>`](/docs/reference/wiki/reinforcementlearning/ornsteinuhlenbecknoise/) | Ornstein-Uhlenbeck process noise for temporally correlated exploration. |
| [`PPOAgent<T>`](/docs/reference/wiki/reinforcementlearning/ppoagent/) | Proximal Policy Optimization (PPO) agent for reinforcement learning. |
| [`PolicyIterationAgent<T>`](/docs/reference/wiki/reinforcementlearning/policyiterationagent/) | Policy Iteration agent for reinforcement learning using dynamic programming. |
| [`PrioritizedReplayBuffer<T>`](/docs/reference/wiki/reinforcementlearning/prioritizedreplaybuffer/) | Prioritized experience replay buffer for reinforcement learning. |
| [`PrioritizedSweepingAgent<T>`](/docs/reference/wiki/reinforcementlearning/prioritizedsweepingagent/) | Prioritized Sweeping agent that focuses planning on high-priority state-actions. |
| [`QLambdaAgent<T>`](/docs/reference/wiki/reinforcementlearning/qlambdaagent/) | Q(lambda) agent that combines Q-learning with eligibility traces for faster credit assignment in tabular reinforcement learning environments. |
| [`QMIXAgent<T>`](/docs/reference/wiki/reinforcementlearning/qmixagent/) | QMIX agent for multi-agent value-based reinforcement learning. |
| [`REINFORCEAgent<T>`](/docs/reference/wiki/reinforcementlearning/reinforceagent/) | REINFORCE (Monte Carlo Policy Gradient) agent for reinforcement learning. |
| [`RainbowDQNAgent<T>`](/docs/reference/wiki/reinforcementlearning/rainbowdqnagent/) | Rainbow DQN agent combining six extensions to DQN. |
| [`ReplayBuffer<T>`](/docs/reference/wiki/reinforcementlearning/replaybuffer/) | A buffer for storing and replaying experiences in reinforcement learning. |
| [`SACAgent<T>`](/docs/reference/wiki/reinforcementlearning/sacagent/) | Soft Actor-Critic (SAC) agent for continuous control reinforcement learning. |
| [`SARSAAgent<T>`](/docs/reference/wiki/reinforcementlearning/sarsaagent/) | SARSA (State-Action-Reward-State-Action) agent using tabular methods. |
| [`SARSALambdaAgent<T>`](/docs/reference/wiki/reinforcementlearning/sarsalambdaagent/) | SARSA(lambda) agent that combines on-policy SARSA control with eligibility traces for faster credit assignment while respecting the current exploration policy. |
| [`SequenceContext<T>`](/docs/reference/wiki/reinforcementlearning/sequencecontext/) | Context window for sequence modeling in Decision Transformer. |
| [`TD3Agent<T>`](/docs/reference/wiki/reinforcementlearning/td3agent/) | Twin Delayed Deep Deterministic Policy Gradient (TD3) agent for continuous control. |
| [`TRPOAgent<T>`](/docs/reference/wiki/reinforcementlearning/trpoagent/) | Trust Region Policy Optimization (TRPO) agent for reinforcement learning. |
| [`TabularActorCriticAgent<T>`](/docs/reference/wiki/reinforcementlearning/tabularactorcriticagent/) | Tabular Actor-Critic agent combining policy and value learning. |
| [`TabularQLearningAgent<T>`](/docs/reference/wiki/reinforcementlearning/tabularqlearningagent/) | Tabular Q-Learning agent using lookup table for Q-values. |
| [`ThompsonSamplingAgent<T>`](/docs/reference/wiki/reinforcementlearning/thompsonsamplingagent/) | Thompson Sampling (Bayesian) Multi-Armed Bandit agent. |
| [`ThompsonSamplingExploration<T>`](/docs/reference/wiki/reinforcementlearning/thompsonsamplingexploration/) | Thompson Sampling (Bayesian) exploration for discrete action spaces. |
| [`Trajectory<T>`](/docs/reference/wiki/reinforcementlearning/trajectory/) | Represents a trajectory of experience for on-policy RL algorithms (PPO, A2C, etc.). |
| [`TransitionData<T>`](/docs/reference/wiki/reinforcementlearning/transitiondata/) | Helper class for serializing model transition data. |
| [`UCBBanditAgent<T>`](/docs/reference/wiki/reinforcementlearning/ucbbanditagent/) | Upper Confidence Bound (UCB) Multi-Armed Bandit agent. |
| [`UniformReplayBuffer<T, TState, TAction>`](/docs/reference/wiki/reinforcementlearning/uniformreplaybuffer/) | A replay buffer that samples experiences uniformly at random. |
| [`UpperConfidenceBoundExploration<T>`](/docs/reference/wiki/reinforcementlearning/upperconfidenceboundexploration/) | Upper Confidence Bound (UCB) exploration for discrete action spaces. |
| [`ValueIterationAgent<T>`](/docs/reference/wiki/reinforcementlearning/valueiterationagent/) | Value Iteration agent for reinforcement learning using dynamic programming. |
| [`WatkinsQLambdaAgent<T>`](/docs/reference/wiki/reinforcementlearning/watkinsqlambdaagent/) | Watkins's Q(lambda) agent that combines Q-learning with eligibility traces but cuts traces when an exploratory (non-greedy) action is taken, ensuring convergence to the optimal policy. |
| [`WorkerNetworks<T>`](/docs/reference/wiki/reinforcementlearning/workernetworks/) | Worker-local networks for A3C agent. |
| [`WorldModelsAgent<T>`](/docs/reference/wiki/reinforcementlearning/worldmodelsagent/) | World Models agent learning compact representations with VAE and RNN. |

## Base Classes (4)

| Type | Summary |
|:-----|:--------|
| [`DeepReinforcementLearningAgentBase<T>`](/docs/reference/wiki/reinforcementlearning/deepreinforcementlearningagentbase/) | Base class for deep reinforcement learning agents that use neural networks as function approximators. |
| [`ExplorationStrategyBase<T>`](/docs/reference/wiki/reinforcementlearning/explorationstrategybase/) | Abstract base class for exploration strategy implementations. |
| [`PolicyBase<T>`](/docs/reference/wiki/reinforcementlearning/policybase/) | Abstract base class for policy implementations. |
| [`ReinforcementLearningAgentBase<T>`](/docs/reference/wiki/reinforcementlearning/reinforcementlearningagentbase/) | Base class for all reinforcement learning agents, providing common functionality and structure. |

## Interfaces (3)

| Type | Summary |
|:-----|:--------|
| [`IExplorationStrategy<T>`](/docs/reference/wiki/reinforcementlearning/iexplorationstrategy/) | Interface for exploration strategies used by policies. |
| [`IPolicy<T>`](/docs/reference/wiki/reinforcementlearning/ipolicy/) | Core interface for RL policies - defines how to select actions. |
| [`IReplayBuffer<T, TState, TAction>`](/docs/reference/wiki/reinforcementlearning/ireplaybuffer/) | Interface for experience replay buffers used in reinforcement learning. |

## Options & Configuration (7)

| Type | Summary |
|:-----|:--------|
| [`BetaPolicyOptions<T>`](/docs/reference/wiki/reinforcementlearning/betapolicyoptions/) | Configuration options for Beta distribution policies. |
| [`ContinuousPolicyOptions<T>`](/docs/reference/wiki/reinforcementlearning/continuouspolicyoptions/) | Configuration options for continuous action space policies in reinforcement learning. |
| [`DeterministicPolicyOptions<T>`](/docs/reference/wiki/reinforcementlearning/deterministicpolicyoptions/) | Configuration options for deterministic policies. |
| [`DiscretePolicyOptions<T>`](/docs/reference/wiki/reinforcementlearning/discretepolicyoptions/) | Configuration options for discrete action space policies in reinforcement learning. |
| [`MixedPolicyOptions<T>`](/docs/reference/wiki/reinforcementlearning/mixedpolicyoptions/) | Configuration options for mixed discrete and continuous policies. |
| [`MultiModalPolicyOptions<T>`](/docs/reference/wiki/reinforcementlearning/multimodalpolicyoptions/) | Configuration options for multi-modal mixture of Gaussians policies. |
| [`ReinforcementLearningOptions<T>`](/docs/reference/wiki/reinforcementlearning/reinforcementlearningoptions/) | Configuration options for reinforcement learning agents. |

## Helpers & Utilities (1)

| Type | Summary |
|:-----|:--------|
| [`MCTSNode<T>`](/docs/reference/wiki/reinforcementlearning/mctsnode/) | Monte Carlo Tree Search (MCTS) node for MuZero agent. |

