namespace AiDotNet.ReinforcementLearning.Models.Options
{
    /// <summary>
    /// Options for configuring the Quantile Regression Deep Q-Network (QR-DQN) reinforcement learning algorithm.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Quantile Regression DQN (QR-DQN) is a distributional reinforcement learning approach that
    /// models the full distribution of returns rather than just the expected value. It does this by
    /// estimating a set of quantiles of the return distribution, providing a much richer representation
    /// of uncertainty and risk.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b>
    /// QR-DQN is particularly powerful for financial applications because it:
    /// - Models the full range of possible returns, not just the average
    /// - Gives you information about risk (how volatile the returns might be)
    /// - Allows for risk-sensitive decision making
    /// - Can better handle market volatility and extreme events
    /// 
    /// Think of it like this: Instead of just predicting that a trade will make $10 on average,
    /// it might tell you there's a 10% chance of losing $5, a 60% chance of making $8-12, and
    /// a 30% chance of making $15+. This is crucial information for trading decisions.
    /// </para>
    /// </remarks>
    public class QRDQNOptions : ReinforcementLearningOptions
    {
        /// <summary>
        /// Gets or sets the number of quantiles to use for the return distribution.
        /// </summary>
        /// <remarks>
        /// <para>
        /// This parameter determines the granularity of the return distribution. More quantiles
        /// provide a finer-grained distribution but require more computational resources.
        /// Typical values range from 5 (coarse) to 200 (very fine-grained).
        /// </para>
        /// <para>
        /// <b>For Beginners:</b>
        /// This is like deciding how many "buckets" to split possible returns into.
        /// With 5 quantiles, you might have very rough categories like "big loss", 
        /// "small loss", "break even", "small gain", "big gain". With 50 quantiles,
        /// you get a much more detailed picture of the possible outcomes.
        /// </para>
        /// </remarks>
        public int NumQuantiles { get; set; } = 50;
        
        /// <summary>
        /// Gets or sets the quantile huber loss kappa parameter.
        /// </summary>
        /// <remarks>
        /// <para>
        /// The kappa parameter determines the threshold where the Huber loss switches from 
        /// quadratic to linear. It helps make the training more robust to outliers.
        /// A value of 1.0 is commonly used.
        /// </para>
        /// <para>
        /// <b>For Beginners:</b>
        /// This is a technical parameter that helps the model learn more reliably,
        /// especially when the market has "outlier" events like flash crashes or sudden spikes.
        /// The default value works well for most applications.
        /// </para>
        /// </remarks>
        public double HuberKappa { get; set; } = 1.0;
        
        /// <summary>
        /// Gets or sets a value indicating whether to use the CVaR risk measure for action selection.
        /// </summary>
        /// <remarks>
        /// <para>
        /// Conditional Value at Risk (CVaR) focuses on the worst possible outcomes rather than
        /// the average. When enabled, it makes the agent more conservative and risk-averse
        /// in its decision making.
        /// </para>
        /// <para>
        /// <b>For Beginners:</b>
        /// When this is on, the agent becomes more cautious, focusing on avoiding bad outcomes
        /// rather than just maximizing average returns. This can be critical for trading when 
        /// capital preservation is a priority. It's like telling the agent "be extra careful about
        /// the worst-case scenarios."
        /// </para>
        /// </remarks>
        public bool UseCVaR { get; set; } = false;
        
        /// <summary>
        /// Gets or sets the CVaR risk level (alpha value between 0 and 1).
        /// </summary>
        /// <remarks>
        /// <para>
        /// This parameter determines how conservative the CVaR measure should be.
        /// Lower values (e.g., 0.1) focus on more extreme worst-case scenarios,
        /// while higher values (e.g., 0.5) are less conservative.
        /// Only used when UseCVaR is true.
        /// </para>
        /// <para>
        /// <b>For Beginners:</b>
        /// This controls how cautious the risk-aware agent should be:
        /// - 0.05 means "focus on the worst 5% of possible outcomes"
        /// - 0.25 means "focus on the worst 25% of possible outcomes"
        /// 
        /// The lower the number, the more cautious and risk-averse the agent becomes.
        /// </para>
        /// </remarks>
        public double CVaRAlpha { get; set; } = 0.25;
        
        /// <summary>
        /// Gets or sets the risk distortion measure for risk-sensitive policy.
        /// </summary>
        /// <remarks>
        /// <para>
        /// A number between 0 and 1 that determines how much the agent distorts probabilities 
        /// when evaluating risks. A value of 0.0 means no distortion (risk-neutral), while
        /// values closer to 1.0 represent increasing risk-aversion.
        /// </para>
        /// <para>
        /// <b>For Beginners:</b>
        /// This is like adjusting how "pessimistic" the agent should be. At 0.0, the agent 
        /// evaluates risks accurately. As the value increases toward 1.0, the agent starts
        /// overweighting the probability of negative outcomes, becoming increasingly cautious.
        /// For volatile markets, a moderate setting of 0.2-0.5 can provide good balance.
        /// </para>
        /// </remarks>
        public double RiskDistortion { get; set; } = 0.0;
        
        /// <summary>
        /// Gets or sets a value indicating whether to use noisy networks for exploration.
        /// </summary>
        /// <remarks>
        /// <para>
        /// Noisy networks add parametric noise to the weights of the neural network,
        /// providing a more sophisticated exploration strategy than epsilon-greedy.
        /// It's particularly useful for complex environments like financial markets.
        /// </para>
        /// <para>
        /// <b>For Beginners:</b>
        /// This enables a smarter way for the agent to explore different trading strategies.
        /// Rather than just taking random actions occasionally (as in traditional approaches),
        /// noisy networks add "intelligent noise" to the decision process. This often leads to 
        /// more effective exploration of trading strategies, especially in complex markets.
        /// </para>
        /// </remarks>
        public bool UseNoisyNetworks { get; set; } = true;
        
        /// <summary>
        /// Gets or sets the initial value of sigma for noisy networks.
        /// </summary>
        /// <remarks>
        /// <para>
        /// This parameter controls the initial magnitude of the noise in noisy networks.
        /// Higher values encourage more exploration initially. Typical values are 0.1-0.5.
        /// Only used when UseNoisyNetworks is true.
        /// </para>
        /// <para>
        /// <b>For Beginners:</b>
        /// This determines how much "noise" or randomness is initially added to the agent's
        /// decision making process. Higher values mean more exploration of different strategies.
        /// The noise naturally reduces over time as the agent learns what works.
        /// </para>
        /// </remarks>
        public double InitialNoiseStd { get; set; } = 0.5;
        
        /// <summary>
        /// Gets or sets the hidden layer sizes for the neural network.
        /// </summary>
        /// <remarks>
        /// <para>
        /// This array specifies the number of neurons in each hidden layer of the neural network.
        /// More complex architectures can capture more complex patterns but may be harder to train.
        /// </para>
        /// <para>
        /// <b>For Beginners:</b>
        /// This defines the structure of the neural network's "brain". Larger numbers and more
        /// layers mean a more powerful model that can potentially learn more complex trading patterns,
        /// but might also be slower to train and could overfit on smaller datasets.
        /// </para>
        /// </remarks>
        public int[] HiddenLayerSizes { get; set; } = new int[] { 128, 128 };
        
        /// <summary>
        /// Gets or sets a value indicating whether to use double DQN.
        /// </summary>
        /// <remarks>
        /// <para>
        /// Double DQN helps reduce overestimation bias in Q-value estimates by using
        /// a separate target network for action selection and evaluation.
        /// </para>
        /// <para>
        /// <b>For Beginners:</b>
        /// This is a technique that helps the agent avoid being overly optimistic about
        /// potential trading returns. In financial markets, this can prevent the agent from
        /// taking excessive risks based on overestimated rewards. It's generally recommended
        /// to keep this enabled.
        /// </para>
        /// </remarks>
        public bool UseDoubleDQN { get; set; } = true;
        
        /// <summary>
        /// Gets or sets a value indicating whether to use prioritized experience replay.
        /// </summary>
        /// <remarks>
        /// <para>
        /// Prioritized experience replay gives more importance to experiences from which
        /// the agent can learn more (typically those with large errors), improving learning efficiency.
        /// </para>
        /// <para>
        /// <b>For Beginners:</b>
        /// This helps the agent learn more efficiently by focusing on the most informative
        /// market situations. For example, it might pay more attention to unusual market moves
        /// or situations where its predictions were very wrong. This is especially valuable in
        /// financial markets where certain rare events can be critically important.
        /// </para>
        /// </remarks>
        public new bool UsePrioritizedReplay { get; set; } = true;
        
        /// <summary>
        /// Gets or sets the prioritized replay alpha parameter for importance sampling.
        /// </summary>
        /// <remarks>
        /// <para>
        /// Controls how much prioritization is used. 0 means no prioritization,
        /// 1 means full prioritization. Only used when UsePrioritizedReplay is true.
        /// </para>
        /// <para>
        /// <b>For Beginners:</b>
        /// This controls how strongly the agent focuses on surprising or informative market events.
        /// Higher values mean the agent pays much more attention to unusual situations where
        /// it was very wrong, potentially learning faster but maybe becoming too fixated on rare events.
        /// </para>
        /// </remarks>
        public double PriorityAlpha { get; set; } = 0.6;
        
        /// <summary>
        /// Gets or sets the prioritized replay beta parameter for importance sampling bias correction.
        /// </summary>
        /// <remarks>
        /// <para>
        /// Controls how much to correct for the bias introduced by prioritized sampling.
        /// 0 means no correction, 1 means full correction. Usually annealed from a starting
        /// value to 1.0 over the course of training.
        /// </para>
        /// <para>
        /// <b>For Beginners:</b>
        /// This is a technical parameter that helps ensure the agent doesn't become too biased
        /// by focusing on certain market events more than others. It should generally start at
        /// a moderate value and gradually increase to 1.0 during training.
        /// </para>
        /// </remarks>
        public double PriorityBetaStart { get; set; } = 0.4;
        
        /// <summary>
        /// Gets or sets the number of atoms to use for categorical DQN.
        /// </summary>
        /// <remarks>
        /// <para>
        /// This is an alternative to quantile regression that uses a fixed grid of returns.
        /// This parameter is provided for compatibility with other distributional RL approaches.
        /// Not used in standard QR-DQN.
        /// </para>
        /// <para>
        /// <b>For Beginners:</b>
        /// This is an alternative technical approach for modeling return distributions that 
        /// isn't typically used in QR-DQN, but is provided for flexibility and experimentation.
        /// Most users can leave this at the default value.
        /// </para>
        /// </remarks>
        public int NumAtoms { get; set; } = 51;

        /// <summary>
        /// Gets or sets the network architecture for the Q-network.
        /// </summary>
        /// <remarks>
        /// Specifies the hidden layer sizes for the neural network.
        /// This is an alias for HiddenLayerSizes for compatibility.
        /// </remarks>
        public int[] NetworkArchitecture => HiddenLayerSizes;

        /// <summary>
        /// Gets or sets the risk metric to use for action selection.
        /// </summary>
        /// <remarks>
        /// Determines which risk measure is used when evaluating actions.
        /// Options might include "mean", "cvar", "var", etc.
        /// </remarks>
        public string RiskMetric { get; set; } = "mean";

        /// <summary>
        /// Gets or sets the risk level for risk-sensitive action selection.
        /// </summary>
        /// <remarks>
        /// A value between 0 and 1 that controls the risk preference.
        /// Lower values indicate more risk-averse behavior.
        /// </remarks>
        public double RiskLevel { get; set; } = 0.5;
    }
}