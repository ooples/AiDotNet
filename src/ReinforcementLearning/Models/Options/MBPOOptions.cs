namespace AiDotNet.ReinforcementLearning.Models.Options
{
    /// <summary>
    /// Options for configuring the Model-Based Policy Optimization (MBPO) reinforcement learning algorithm.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Model-Based Policy Optimization is a hybrid model-based/model-free reinforcement learning algorithm
    /// that learns a model of the environment dynamics and uses it to generate synthetic experiences
    /// to augment the real experiences collected from the environment. This helps improve sample efficiency 
    /// and accelerates learning compared to purely model-free approaches.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b>
    /// MBPO is a powerful approach for financial markets because it:
    /// - Learns faster by creating a "simulator" of market behavior
    /// - Needs less real market data to develop good strategies
    /// - Can test trading ideas in the simulation before risking real money
    /// - Adapts to changing market conditions by continuously updating its model
    /// 
    /// Think of it like having a virtual trading sandbox where the agent can practice
    /// thousands of trades for every real market interaction, dramatically speeding up learning.
    /// </para>
    /// </remarks>
    public class MBPOOptions : ReinforcementLearningOptions
    {
        /// <summary>
        /// Gets or sets the ensemble size for the dynamics model.
        /// </summary>
        /// <remarks>
        /// <para>
        /// The number of independent dynamics models to train in the ensemble. Using multiple models
        /// helps capture uncertainty in the environment dynamics predictions and improves robustness.
        /// Typical values range from 3 to 10.
        /// </para>
        /// <para>
        /// <b>For Beginners:</b>
        /// This is how many different "market simulators" the system trains. Having multiple models
        /// helps capture different possible market behaviors and reduces the risk of making mistakes
        /// based on any single model's prediction. It's like getting second and third opinions.
        /// </para>
        /// </remarks>
        public int EnsembleSize { get; set; } = 5;
        
        /// <summary>
        /// Gets or sets the hidden layer sizes for the dynamics model neural networks.
        /// </summary>
        /// <remarks>
        /// <para>
        /// Specifies the structure of the neural networks used in the dynamics model.
        /// Larger networks can capture more complex dynamics but require more data to train effectively.
        /// </para>
        /// <para>
        /// <b>For Beginners:</b>
        /// This defines how complex the "market simulator" is. Larger numbers mean a more sophisticated 
        /// model that can potentially capture more subtle market patterns, but also requires more data 
        /// to train properly.
        /// </para>
        /// </remarks>
        public int[] ModelHiddenSizes { get; set; } = new int[] { 200, 200, 200 };
        
        /// <summary>
        /// Gets or sets the hidden layer sizes for the policy network.
        /// </summary>
        /// <remarks>
        /// <para>
        /// Specifies the structure of the neural network used for the policy.
        /// This determines how complex the trading strategy can be.
        /// </para>
        /// <para>
        /// <b>For Beginners:</b>
        /// This defines how sophisticated the trading strategy can be. Larger networks
        /// can learn more complex decision rules but might be harder to train.
        /// </para>
        /// </remarks>
        public int[] PolicyHiddenSizes { get; set; } = new int[] { 256, 256 };
        
        /// <summary>
        /// Gets or sets the hidden layer sizes for the value/Q-function network.
        /// </summary>
        /// <remarks>
        /// <para>
        /// Specifies the structure of the neural network used for the value or Q-function.
        /// This determines how accurately the model can estimate future returns.
        /// </para>
        /// <para>
        /// <b>For Beginners:</b>
        /// This defines how sophisticated the model is at evaluating potential future returns
        /// from different trading strategies. Larger networks can capture more complex relationships
        /// between market states and expected returns.
        /// </para>
        /// </remarks>
        public int[] ValueHiddenSizes { get; set; } = new int[] { 256, 256 };
        
        /// <summary>
        /// Gets or sets the learning rate for the dynamics model.
        /// </summary>
        /// <remarks>
        /// <para>
        /// The learning rate determines how quickly the dynamics model adapts to new data.
        /// Higher values enable faster adaptation but may cause instability.
        /// </para>
        /// <para>
        /// <b>For Beginners:</b>
        /// This controls how quickly the market simulator learns from new data. 
        /// Higher values mean faster adaptation to changing market conditions, but
        /// may cause overreaction to random market noise.
        /// </para>
        /// </remarks>
        public double ModelLearningRate { get; set; } = 0.001;
        
        /// <summary>
        /// Gets or sets the learning rate for the policy network.
        /// </summary>
        /// <remarks>
        /// <para>
        /// The learning rate determines how quickly the policy adapts to new information.
        /// Higher values enable faster learning but may cause instability.
        /// </para>
        /// <para>
        /// <b>For Beginners:</b>
        /// This controls how quickly the trading strategy adapts to new information.
        /// Higher values mean faster learning but may lead to inconsistent trading behavior.
        /// </para>
        /// </remarks>
        public double PolicyLearningRate { get; set; } = 0.0003;
        
        /// <summary>
        /// Gets or sets the learning rate for the value/Q-function network.
        /// </summary>
        /// <remarks>
        /// <para>
        /// The learning rate determines how quickly the value function adapts to new information.
        /// Higher values enable faster learning but may cause instability.
        /// </para>
        /// <para>
        /// <b>For Beginners:</b>
        /// This controls how quickly the system updates its estimates of future returns.
        /// Higher values mean faster adaptation to new market information.
        /// </para>
        /// </remarks>
        public double ValueLearningRate { get; set; } = 0.0003;
        
        /// <summary>
        /// Gets or sets the rollout horizon for the model predictions.
        /// </summary>
        /// <remarks>
        /// <para>
        /// The number of timesteps to predict into the future when generating synthetic experiences.
        /// Shorter horizons are more accurate but provide less information.
        /// </para>
        /// <para>
        /// <b>For Beginners:</b>
        /// This is how far into the future the market simulator predicts when generating
        /// synthetic training data. Shorter horizons are more accurate but give less information
        /// about long-term consequences of trading decisions.
        /// </para>
        /// </remarks>
        public int RolloutHorizon { get; set; } = 1;
        
        /// <summary>
        /// Gets or sets the ratio of model-generated samples to real samples.
        /// </summary>
        /// <remarks>
        /// <para>
        /// The ratio of model-generated experiences to real experiences used in training.
        /// Higher values use more synthetic data, increasing sample efficiency but potentially
        /// introducing model bias.
        /// </para>
        /// <para>
        /// <b>For Beginners:</b>
        /// This determines how many simulated market interactions the agent learns from
        /// compared to real market data. Higher values mean more learning from simulation,
        /// which speeds up learning but might introduce errors if the simulator isn't perfect.
        /// </para>
        /// </remarks>
        public int ModelRatio { get; set; } = 20;
        
        /// <summary>
        /// Gets or sets the number of epochs to train the dynamics model per iteration.
        /// </summary>
        /// <remarks>
        /// <para>
        /// The number of training passes over the collected data when updating the dynamics model.
        /// More epochs provide better model fitting but may lead to overfitting.
        /// </para>
        /// <para>
        /// <b>For Beginners:</b>
        /// This is how thoroughly the market simulator is trained on each batch of data.
        /// More epochs mean more learning from each piece of market data, but might lead
        /// to overemphasizing patterns that won't repeat in the future.
        /// </para>
        /// </remarks>
        public int ModelEpochs { get; set; } = 50;
        
        /// <summary>
        /// Gets or sets the number of epochs to train the policy per iteration.
        /// </summary>
        /// <remarks>
        /// <para>
        /// The number of training passes over the collected data when updating the policy.
        /// More epochs provide better policy learning but may cause instability.
        /// </para>
        /// <para>
        /// <b>For Beginners:</b>
        /// This is how thoroughly the trading strategy is updated based on each batch of
        /// experiences. More epochs mean extracting more information from each trading experience.
        /// </para>
        /// </remarks>
        public int PolicyEpochs { get; set; } = 20;
        
        /// <summary>
        /// Gets or sets a value indicating whether to use a probabilistic dynamics model.
        /// </summary>
        /// <remarks>
        /// <para>
        /// When enabled, the dynamics model predicts not just the next state but also the uncertainty
        /// in that prediction, allowing for exploration through the model's uncertainty.
        /// </para>
        /// <para>
        /// <b>For Beginners:</b>
        /// When turned on, the market simulator doesn't just predict what will happen next,
        /// but also how confident it is in that prediction. This helps the system explore different
        /// trading strategies in areas where the market behavior is less certain, potentially
        /// discovering more robust approaches.
        /// </para>
        /// </remarks>
        public bool ProbabilisticModel { get; set; } = true;
        
        /// <summary>
        /// Gets or sets a value indicating whether to use an ensemble of policy networks.
        /// </summary>
        /// <remarks>
        /// <para>
        /// When enabled, trains multiple policy networks with different initializations,
        /// helping capture different trading strategies and improve robustness.
        /// </para>
        /// <para>
        /// <b>For Beginners:</b>
        /// When turned on, the system trains multiple trading strategies simultaneously
        /// and can either select the best one or combine them. This helps find diverse
        /// approaches to the market and reduces the risk of getting stuck with a suboptimal strategy.
        /// </para>
        /// </remarks>
        public bool EnsemblePolicy { get; set; } = false;
        
        /// <summary>
        /// Gets or sets the size of the policy ensemble if enabled.
        /// </summary>
        /// <remarks>
        /// <para>
        /// The number of independent policy networks to train if EnsemblePolicy is enabled.
        /// Only used when EnsemblePolicy is true.
        /// </para>
        /// <para>
        /// <b>For Beginners:</b>
        /// This is how many different trading strategies to train simultaneously when using
        /// the ensemble approach. Each strategy might specialize in different market conditions.
        /// </para>
        /// </remarks>
        public int PolicyEnsembleSize { get; set; } = 3;
        
        /// <summary>
        /// Gets or sets the initial temperature for Soft Actor-Critic updates.
        /// </summary>
        /// <remarks>
        /// <para>
        /// Initial entropy regularization coefficient for the policy. Higher values encourage
        /// more exploration. This is usually automatically adjusted during training.
        /// </para>
        /// <para>
        /// <b>For Beginners:</b>
        /// This controls how much the trading strategy balances exploration (trying new things)
        /// versus exploitation (using what already works well). Higher values encourage more
        /// diverse trading behaviors, which can be important in volatile markets.
        /// </para>
        /// </remarks>
        public double InitialTemperature { get; set; } = 0.1;
        
        /// <summary>
        /// Gets or sets a value indicating whether to use automatic entropy tuning.
        /// </summary>
        /// <remarks>
        /// <para>
        /// When enabled, the entropy regularization coefficient (temperature) is automatically
        /// adjusted during training to maintain a target level of policy entropy.
        /// </para>
        /// <para>
        /// <b>For Beginners:</b>
        /// When turned on, the system automatically adjusts how much it explores versus exploits
        /// based on how uncertain it is. This helps the system naturally become more focused
        /// as it learns more about the market, while still exploring when needed.
        /// </para>
        /// </remarks>
        public bool AutoTuneEntropy { get; set; } = true;
        
        /// <summary>
        /// Gets or sets the model training batch size.
        /// </summary>
        /// <remarks>
        /// <para>
        /// The number of samples to use in each training batch for the dynamics model.
        /// </para>
        /// <para>
        /// <b>For Beginners:</b>
        /// This is how many market examples the simulator learns from at once. Larger batches
        /// make training more stable but slower, while smaller batches allow quicker adaptation
        /// to new information.
        /// </para>
        /// </remarks>
        public int ModelBatchSize { get; set; } = 256;
        
        /// <summary>
        /// Gets or sets the minimum number of real experiences before starting model training.
        /// </summary>
        /// <remarks>
        /// <para>
        /// The algorithm will collect this many real experiences before starting to train
        /// the dynamics model and generate synthetic experiences.
        /// </para>
        /// <para>
        /// <b>For Beginners:</b>
        /// This is how much real market data needs to be collected before the system starts
        /// building and using the market simulator. Setting this too low might result in a
        /// poor simulator that gives misleading training signals.
        /// </para>
        /// </remarks>
        public int RealExpBeforeModel { get; set; } = 5000;
        
        /// <summary>
        /// Gets or sets a value indicating whether to use a reward predictor model.
        /// </summary>
        /// <remarks>
        /// <para>
        /// When enabled, the dynamics model also predicts rewards in addition to next states,
        /// allowing for more complete synthetic experiences.
        /// </para>
        /// <para>
        /// <b>For Beginners:</b>
        /// When turned on, the market simulator doesn't just predict future market states but
        /// also the rewards (profits/losses) that would result from different actions. This provides
        /// more complete information for training the trading strategy.
        /// </para>
        /// </remarks>
        public bool ModelPredictRewards { get; set; } = true;
        
        /// <summary>
        /// Gets or sets a value indicating whether to use model rollouts with branching.
        /// </summary>
        /// <remarks>
        /// <para>
        /// When enabled, model rollouts can branch into multiple possible futures,
        /// exploring different action sequences and providing more diverse synthetic experiences.
        /// </para>
        /// <para>
        /// <b>For Beginners:</b>
        /// When turned on, the system explores multiple possible future scenarios from each
        /// starting point, creating a tree of potential market outcomes rather than just a single
        /// path. This helps discover more robust trading strategies that work across different
        /// possible market developments.
        /// </para>
        /// </remarks>
        public bool BranchingRollouts { get; set; } = false;
        
        /// <summary>
        /// Gets or sets the number of branches in branching rollouts.
        /// </summary>
        /// <remarks>
        /// <para>
        /// The number of different future paths to explore from each state when using branching rollouts.
        /// Only used when BranchingRollouts is true.
        /// </para>
        /// <para>
        /// <b>For Beginners:</b>
        /// This is how many different possible future scenarios the system explores from each
        /// starting point when using branching rollouts. More branches provide a wider exploration
        /// of possible market developments.
        /// </para>
        /// </remarks>
        public int NumBranches { get; set; } = 4;

        /// <summary>
        /// Gets or sets the frequency of model training.
        /// </summary>
        /// <remarks>
        /// How many environment steps between each model training update.
        /// </remarks>
        public int ModelTrainingFrequency { get; set; } = 250;

        /// <summary>
        /// Gets or sets the frequency of model rollouts.
        /// </summary>
        /// <remarks>
        /// How many environment steps between each batch of model rollouts.
        /// </remarks>
        public int ModelRolloutFrequency { get; set; } = 250;

        /// <summary>
        /// Gets or sets the number of synthetic experiences to generate per rollout.
        /// </summary>
        /// <remarks>
        /// The number of synthetic experiences to generate from the dynamics model
        /// during each rollout phase.
        /// </remarks>
        public int NumSyntheticExperiences { get; set; } = 100;
    }
}