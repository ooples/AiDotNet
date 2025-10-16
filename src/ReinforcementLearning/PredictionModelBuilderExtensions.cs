using AiDotNet;
using AiDotNet.Models.Options;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.ReinforcementLearning.Models;
using AiDotNet.ReinforcementLearning.Models.Options;
using AiDotNet.Helpers;
// Use the simplified PredictionModelBuilder for reinforcement learning

namespace AiDotNet.ReinforcementLearning
{
    /// <summary>
    /// Extension methods for PredictionModelBuilder to support reinforcement learning models.
    /// </summary>
    public static class PredictionModelBuilderExtensions
    {
        /// <summary>
        /// Creates a REINFORCE model with policy gradients.
        /// </summary>
        /// <typeparam name="T">The numeric type used for calculations.</typeparam>
        /// <param name="builder">The prediction model builder.</param>
        /// <param name="options">The options for the REINFORCE algorithm.</param>
        /// <returns>The prediction model builder for method chaining.</returns>
        public static PredictionModelBuilder<T> UseREINFORCE<T>(
            this PredictionModelBuilder<T> builder,
            PolicyGradientOptions options)
        {
            // Create the model and set it in the builder
            var model = new REINFORCEModel<T>(options);
            builder.SetModel(model);
            return builder;
        }

        /// <summary>
        /// Creates a REINFORCE model with policy gradients using default options.
        /// </summary>
        /// <typeparam name="T">The numeric type used for calculations.</typeparam>
        /// <param name="builder">The prediction model builder.</param>
        /// <param name="stateSize">The size of the state space.</param>
        /// <param name="actionSize">The size of the action space.</param>
        /// <param name="isContinuous">Whether the action space is continuous.</param>
        /// <param name="useBaseline">Whether to use a baseline for variance reduction.</param>
        /// <returns>The prediction model builder for method chaining.</returns>
        public static PredictionModelBuilder<T> UseREINFORCE<T>(
            this PredictionModelBuilder<T> builder,
            int stateSize,
            int actionSize,
            bool isContinuous = false,
            bool useBaseline = true)
        {
            // Create options with default values
            var options = new PolicyGradientOptions
            {
                StateSize = stateSize,
                ActionSize = actionSize,
                IsContinuous = isContinuous,
                UseBaseline = useBaseline
            };

            return builder.UseREINFORCE(options);
        }

        /// <summary>
        /// Creates an actor-critic model (A2C).
        /// </summary>
        /// <typeparam name="T">The numeric type used for calculations.</typeparam>
        /// <param name="builder">The prediction model builder.</param>
        /// <param name="options">The options for the actor-critic algorithm.</param>
        /// <returns>The prediction model builder for method chaining.</returns>
        public static PredictionModelBuilder<T> UseActorCritic<T>(
            this PredictionModelBuilder<T> builder,
            ActorCriticOptions<T> options)
        {
            // Create the model and set it in the builder
            var model = new ActorCriticModel<T>(options);
            builder.SetModel(model);
            return builder;
        }

        /// <summary>
        /// Creates an actor-critic model (A2C) using default options.
        /// </summary>
        /// <typeparam name="T">The numeric type used for calculations.</typeparam>
        /// <param name="builder">The prediction model builder.</param>
        /// <param name="stateSize">The size of the state space.</param>
        /// <param name="actionSize">The size of the action space.</param>
        /// <param name="isContinuous">Whether the action space is continuous.</param>
        /// <param name="useGAE">Whether to use Generalized Advantage Estimation.</param>
        /// <returns>The prediction model builder for method chaining.</returns>
        public static PredictionModelBuilder<T> UseActorCritic<T>(
            this PredictionModelBuilder<T> builder,
            int stateSize,
            int actionSize,
            bool isContinuous = false,
            bool useGAE = true)
        {
            // Create options with default values
            var options = new ActorCriticOptions<T>
            {
                StateSize = stateSize,
                ActionSize = actionSize,
                IsContinuous = isContinuous,
                UseGAE = useGAE
            };

            return builder.UseActorCritic(options);
        }

        /// <summary>
        /// Creates a PPO model.
        /// </summary>
        /// <typeparam name="T">The numeric type used for calculations.</typeparam>
        /// <param name="builder">The prediction model builder.</param>
        /// <param name="options">The options for the PPO algorithm.</param>
        /// <returns>The prediction model builder for method chaining.</returns>
        public static PredictionModelBuilder<T> UsePPO<T>(
            this PredictionModelBuilder<T> builder,
            PPOOptions<T> options)
        {
            // Create the model and set it in the builder
            var model = new PPOModel<T>(options);
            builder.SetModel(model);
            return builder;
        }

        /// <summary>
        /// Creates a PPO model using default options.
        /// </summary>
        /// <typeparam name="T">The numeric type used for calculations.</typeparam>
        /// <param name="builder">The prediction model builder.</param>
        /// <param name="stateSize">The size of the state space.</param>
        /// <param name="actionSize">The size of the action space.</param>
        /// <param name="isContinuous">Whether the action space is continuous.</param>
        /// <returns>The prediction model builder for method chaining.</returns>
        public static PredictionModelBuilder<T> UsePPO<T>(
            this PredictionModelBuilder<T> builder,
            int stateSize,
            int actionSize,
            bool isContinuous = false)
        {
            // Create options with default values
            var options = new PPOOptions<T>
            {
                StateSize = stateSize,
                ActionSize = actionSize,
                IsContinuous = isContinuous
            };

            return builder.UsePPO(options);
        }

        /// <summary>
        /// Creates a DDPG model for continuous control.
        /// </summary>
        /// <typeparam name="T">The numeric type used for calculations.</typeparam>
        /// <param name="builder">The prediction model builder.</param>
        /// <param name="options">The options for the DDPG algorithm.</param>
        /// <returns>The prediction model builder for method chaining.</returns>
        public static PredictionModelBuilder<T> UseDDPG<T>(
            this PredictionModelBuilder<T> builder,
            DDPGOptions options)
        {
            // Create the model and set it in the builder
            var model = new DDPGModel<T>(options);
            builder.SetModel(model);
            return builder;
        }
        
        /// <summary>
        /// Creates a DDPG model for continuous control using default options.
        /// </summary>
        /// <typeparam name="T">The numeric type used for calculations.</typeparam>
        /// <param name="builder">The prediction model builder.</param>
        /// <param name="stateSize">The size of the state space.</param>
        /// <param name="actionSize">The size of the action space.</param>
        /// <param name="usePrioritizedReplay">Whether to use prioritized experience replay.</param>
        /// <returns>The prediction model builder for method chaining.</returns>
        public static PredictionModelBuilder<T> UseDDPG<T>(
            this PredictionModelBuilder<T> builder,
            int stateSize,
            int actionSize,
            bool usePrioritizedReplay = true)
        {
            // Create options with default values
            var options = new DDPGOptions
            {
                StateSize = stateSize,
                ActionSize = actionSize,
                UsePrioritizedReplay = usePrioritizedReplay,
                // DDPG is for continuous action spaces only
                IsContinuous = true
            };

            return builder.UseDDPG(options);
        }

        /// <summary>
        /// Creates a TD3 model for continuous control.
        /// </summary>
        /// <typeparam name="T">The numeric type used for calculations.</typeparam>
        /// <param name="builder">The prediction model builder.</param>
        /// <param name="options">The options for the TD3 algorithm.</param>
        /// <returns>The prediction model builder for method chaining.</returns>
        public static PredictionModelBuilder<T> UseTD3<T>(
            this PredictionModelBuilder<T> builder,
            TD3Options options)
        {
            // Create the model and set it in the builder
            var model = new TD3Model<T>(options);
            builder.SetModel(model);
            return builder;
        }
        
        /// <summary>
        /// Creates a TD3 model for continuous control using default options.
        /// </summary>
        /// <typeparam name="T">The numeric type used for calculations.</typeparam>
        /// <param name="builder">The prediction model builder.</param>
        /// <param name="stateSize">The size of the state space.</param>
        /// <param name="actionSize">The size of the action space.</param>
        /// <param name="policyUpdateFrequency">How often to update the policy network relative to critic updates.</param>
        /// <returns>The prediction model builder for method chaining.</returns>
        public static PredictionModelBuilder<T> UseTD3<T>(
            this PredictionModelBuilder<T> builder,
            int stateSize,
            int actionSize,
            int policyUpdateFrequency = 2)
        {
            // Create options with default values
            var options = new TD3Options
            {
                StateSize = stateSize,
                ActionSize = actionSize,
                PolicyUpdateFrequency = policyUpdateFrequency,
                // TD3 is for continuous action spaces only
                IsContinuous = true
            };

            return builder.UseTD3(options);
        }

        /// <summary>
        /// Creates a SAC model for continuous control.
        /// </summary>
        /// <typeparam name="T">The numeric type used for calculations.</typeparam>
        /// <param name="builder">The prediction model builder.</param>
        /// <param name="options">The options for the SAC algorithm.</param>
        /// <returns>The prediction model builder for method chaining.</returns>
        public static PredictionModelBuilder<T> UseSAC<T>(
            this PredictionModelBuilder<T> builder,
            SACOptions options)
        {
            // Create the model and set it in the builder
            var model = new SACModel<T>(options);
            builder.SetModel(model);
            return builder;
        }
        
        /// <summary>
        /// Creates a SAC model for continuous control using default options.
        /// </summary>
        /// <typeparam name="T">The numeric type used for calculations.</typeparam>
        /// <param name="builder">The prediction model builder.</param>
        /// <param name="stateSize">The size of the state space.</param>
        /// <param name="actionSize">The size of the action space.</param>
        /// <param name="autoTuneEntropyCoefficient">Whether to automatically tune the entropy coefficient.</param>
        /// <returns>The prediction model builder for method chaining.</returns>
        public static PredictionModelBuilder<T> UseSAC<T>(
            this PredictionModelBuilder<T> builder,
            int stateSize,
            int actionSize,
            bool autoTuneEntropyCoefficient = true)
        {
            // Create options with default values
            var options = new SACOptions
            {
                StateSize = stateSize,
                ActionSize = actionSize,
                AutoTuneEntropyCoefficient = autoTuneEntropyCoefficient,
                // SAC is for continuous action spaces only
                IsContinuous = true
            };

            return builder.UseSAC(options);
        }
        
        /// <summary>
        /// Creates a DQN model for discrete action spaces.
        /// </summary>
        /// <typeparam name="T">The numeric type used for calculations.</typeparam>
        /// <param name="builder">The prediction model builder.</param>
        /// <param name="options">The options for the DQN algorithm.</param>
        /// <returns>The prediction model builder for method chaining.</returns>
        public static PredictionModelBuilder<T> UseDQN<T>(
            this PredictionModelBuilder<T> builder,
            DQNOptions options)
        {
            // Create the model and set it in the builder
            var model = new DQNModel<T>(options);
            builder.SetModel(model);
            return builder;
        }
        
        /// <summary>
        /// Creates a DQN model for discrete action spaces using default options.
        /// </summary>
        /// <typeparam name="T">The numeric type used for calculations.</typeparam>
        /// <param name="builder">The prediction model builder.</param>
        /// <param name="stateSize">The size of the state space.</param>
        /// <param name="actionSize">The size of the action space.</param>
        /// <param name="useDoubleDQN">Whether to use Double DQN to reduce overestimation bias.</param>
        /// <param name="useDuelingDQN">Whether to use Dueling DQN architecture.</param>
        /// <returns>The prediction model builder for method chaining.</returns>
        public static PredictionModelBuilder<T> UseDQN<T>(
            this PredictionModelBuilder<T> builder,
            int stateSize,
            int actionSize,
            bool useDoubleDQN = true,
            bool useDuelingDQN = true)
        {
            // Create options with default values
            var options = new DQNOptions
            {
                StateSize = stateSize,
                ActionSize = actionSize,
                UseDoubleDQN = useDoubleDQN,
                UseDuelingDQN = useDuelingDQN,
                // DQN is for discrete action spaces only
                IsContinuous = false
            };

            return builder.UseDQN(options);
        }
        
        /// <summary>
        /// Creates a Rainbow DQN model for discrete action spaces.
        /// </summary>
        /// <typeparam name="T">The numeric type used for calculations.</typeparam>
        /// <param name="builder">The prediction model builder.</param>
        /// <param name="options">The options for the Rainbow DQN algorithm.</param>
        /// <returns>The prediction model builder for method chaining.</returns>
        /// <remarks>
        /// <para>
        /// Rainbow DQN combines multiple improvements to the DQN algorithm:
        /// - Double DQN: Reduces overestimation bias
        /// - Dueling networks: Separates state value and action advantage estimation
        /// - Prioritized experience replay: Focuses on important transitions
        /// - Noisy networks: Provides better exploration through parameter noise
        /// - Multi-step learning: Propagates rewards faster with n-step returns
        /// - Distributional RL (C51): Models full distribution of returns
        /// </para>
        /// </remarks>
        public static PredictionModelBuilder<T> UseRainbowDQN<T>(
            this PredictionModelBuilder<T> builder,
            RainbowDQNOptions options)
        {
            // Create the model and set it in the builder
            var model = new RainbowDQNModel<T>(options);
            builder.SetModel(model);
            return builder;
        }
        
        /// <summary>
        /// Creates a Rainbow DQN model for discrete action spaces using default options.
        /// </summary>
        /// <typeparam name="T">The numeric type used for calculations.</typeparam>
        /// <param name="builder">The prediction model builder.</param>
        /// <param name="stateSize">The size of the state space.</param>
        /// <param name="actionSize">The size of the action space.</param>
        /// <param name="useDistributionalRL">Whether to use distributional RL (C51).</param>
        /// <param name="useNoisyNetworks">Whether to use noisy networks for exploration.</param>
        /// <returns>The prediction model builder for method chaining.</returns>
        public static PredictionModelBuilder<T> UseRainbowDQN<T>(
            this PredictionModelBuilder<T> builder,
            int stateSize,
            int actionSize,
            bool useDistributionalRL = true,
            bool useNoisyNetworks = true)
        {
            // Create options with default values
            var options = new RainbowDQNOptions
            {
                StateSize = stateSize,
                ActionSize = actionSize,
                UseDistributionalRL = useDistributionalRL,
                UseNoisyNetworks = useNoisyNetworks,
                // Rainbow DQN is for discrete action spaces only
                IsContinuous = false
            };

            return builder.UseRainbowDQN(options);
        }

        /// <summary>
        /// Extends the existing prediction model builder with reinforcement learning capabilities.
        /// </summary>
        /// <typeparam name="T">The numeric type used for calculations.</typeparam>
        /// <param name="builder">The prediction model builder.</param>
        /// <returns>The prediction model builder for method chaining.</returns>
        public static PredictionModelBuilder<T> WithReinforcementLearning<T>(
            this PredictionModelBuilder<T> builder)
        {
            // This method is a placeholder to enable discovery of the extension methods
            // in IDEs through method chaining. It doesn't change the builder.
            return builder;
        }

        /// <summary>
        /// Trains a reinforcement learning model on an experience tuple.
        /// </summary>
        /// <typeparam name="T">The numeric type used for calculations.</typeparam>
        /// <param name="model">The reinforcement learning model.</param>
        /// <param name="state">The state before the action was taken.</param>
        /// <param name="action">The action that was taken.</param>
        /// <param name="reward">The reward received after taking the action.</param>
        /// <param name="nextState">The state after the action was taken.</param>
        /// <param name="done">A flag indicating whether the episode ended after this action.</param>
        public static void LearnFromExperience<T>(
            this IFullModel<T, Tensor<T>, Tensor<T>> model,
            Tensor<T> state,
            object action,
            T reward,
            Tensor<T> nextState,
            bool done)
        {
            // If the model is a ReinforcementLearningModelBase, use its Update method
            if (model is ReinforcementLearningModelBase<T> rlModel)
            {
                // Convert the action to a Vector<T> if it's not already one
                Vector<T> actionVector;
                if (action is Vector<T> vectorAction)
                {
                    actionVector = vectorAction;
                }
                else if (action is int discreteAction)
                {
                    // Create a single-element vector for discrete actions
                    actionVector = new Vector<T>(1);
                    actionVector[0] = MathHelper.GetNumericOperations<T>().FromDouble(discreteAction);
                }
                else if (action is T numericAction)
                {
                    // Create a single-element vector for numeric actions
                    actionVector = new Vector<T>(1);
                    actionVector[0] = numericAction;
                }
                else
                {
                    throw new ArgumentException("Action must be Vector<T>, int, or T");
                }
                
                // Call the common Update method
                rlModel.Update(state, actionVector, reward, nextState, done);
            }
            else
            {
                throw new InvalidOperationException("Model is not a reinforcement learning model");
            }
        }

        /// <summary>
        /// Selects an action for the given state using a reinforcement learning model.
        /// </summary>
        /// <typeparam name="T">The numeric type used for calculations.</typeparam>
        /// <param name="model">The reinforcement learning model.</param>
        /// <param name="state">The current state.</param>
        /// <param name="isTraining">Whether the agent should explore (true) or exploit (false).</param>
        /// <returns>The selected action.</returns>
        public static object SelectAction<T>(
            this IFullModel<T, Tensor<T>, Tensor<T>> model,
            Tensor<T> state,
            bool isTraining = true)
        {
            // If the model is a ReinforcementLearningModelBase, use its SelectAction method
            if (model is ReinforcementLearningModelBase<T> rlModel)
            {
                // Get the action as a Vector<T>
                Vector<T> actionVector = rlModel.SelectAction(state, isTraining);
                
                // If the vector has only one element and is a discrete action model, return as an integer
                if (actionVector.Length == 1 && !rlModel.IsContinuous)
                {
                    return Convert.ToInt32(actionVector[0]);
                }
                else
                {
                    // Otherwise return the vector itself for continuous actions
                    return actionVector;
                }
            }
            else
            {
                throw new InvalidOperationException("Model is not a reinforcement learning model");
            }
        }
    }
}