namespace AiDotNet.ReinforcementLearning.Models.Options
{
    /// <summary>
    /// Configuration options for the Hierarchical Risk-Aware Reinforcement Learning (HRARL) model.
    /// </summary>
    /// <remarks>
    /// The HRARL model combines hierarchical reinforcement learning with risk-aware decision making,
    /// making it particularly suitable for financial applications where managing risk is as important
    /// as maximizing returns. The model uses a multi-level decision hierarchy where high-level policies
    /// make strategic decisions (like asset allocation) while low-level policies make tactical decisions
    /// (like specific trade execution).
    /// 
    /// For beginners: This model works like having both a Chief Investment Officer (who makes big-picture
    /// decisions about risk levels and asset allocation) and a Trader (who executes specific trades) working
    /// together. The model learns at both levels while explicitly considering financial risk, making it
    /// ideal for real-world trading scenarios where managing downside is crucial.
    /// </remarks>
    public class HRARLOptions : ReinforcementLearningOptions
    {
        /// <summary>
        /// Gets or sets the number of hierarchical levels in the model.
        /// </summary>
        /// <remarks>
        /// For beginners: This controls how many layers of decision-making the model uses.
        /// Typically 2 (strategic and tactical) or 3 (strategic, tactical, and execution).
        /// More levels allow for more complex strategies but are harder to train.
        /// </remarks>
        public int NumHierarchicalLevels { get; set; } = 2;
        
        /// <summary>
        /// Gets or sets the hidden dimension size for the high-level policy networks.
        /// </summary>
        /// <remarks>
        /// For beginners: This controls how complex the strategic decision-making can be.
        /// Larger values allow for more sophisticated strategies but require more data to train.
        /// </remarks>
        public int HighLevelHiddenDimension { get; set; } = 256;
        
        /// <summary>
        /// Gets or sets the hidden dimension size for the low-level policy networks.
        /// </summary>
        /// <remarks>
        /// For beginners: This controls how complex the tactical decision-making can be.
        /// The low-level policy focuses on executing the strategy defined by the high-level policy.
        /// </remarks>
        public int LowLevelHiddenDimension { get; set; } = 128;
        
        /// <summary>
        /// Gets or sets the default risk aversion parameter (higher = more risk-averse).
        /// </summary>
        /// <remarks>
        /// For beginners: This determines how cautious the model will be. Higher values
        /// prioritize avoiding losses over making gains. Think of it as the model's
        /// risk tolerance setting.
        /// </remarks>
        public double RiskAversionParameter { get; set; } = 0.5;
        
        /// <summary>
        /// Gets or sets whether the risk aversion parameter is adaptive.
        /// </summary>
        /// <remarks>
        /// For beginners: When true, the model will automatically adjust its risk tolerance
        /// based on market conditions. For example, becoming more cautious during volatile
        /// periods and more aggressive during stable uptrends.
        /// </remarks>
        public bool UseAdaptiveRiskAversion { get; set; } = true;
        
        /// <summary>
        /// Gets or sets the time horizon for the high-level policy in environment steps.
        /// </summary>
        /// <remarks>
        /// For beginners: This is how far ahead the strategic level plans. Longer horizons
        /// enable more long-term strategic planning but make immediate decisions less precise.
        /// In trading terms, this might be quarterly or monthly planning.
        /// </remarks>
        public int HighLevelTimeHorizon { get; set; } = 50;
        
        /// <summary>
        /// Gets or sets the time horizon for the low-level policy in environment steps.
        /// </summary>
        /// <remarks>
        /// For beginners: This is how far ahead the tactical level plans. In trading terms,
        /// this might be daily or hourly decision-making to execute the strategic plan.
        /// </remarks>
        public int LowLevelTimeHorizon { get; set; } = 10;
        
        /// <summary>
        /// Gets or sets the discount factor for future rewards at the high level.
        /// </summary>
        /// <remarks>
        /// For beginners: This controls how much the strategic level cares about long-term
        /// versus short-term results. Values closer to 1 mean more emphasis on long-term performance.
        /// </remarks>
        public double HighLevelGamma { get; set; } = 0.99;
        
        /// <summary>
        /// Gets or sets the discount factor for future rewards at the low level.
        /// </summary>
        /// <remarks>
        /// For beginners: This controls how much the tactical level cares about immediate
        /// versus slightly delayed results. Usually slightly lower than the high-level gamma.
        /// </remarks>
        public double LowLevelGamma { get; set; } = 0.97;
        
        /// <summary>
        /// Gets or sets the learning rate for the high-level policy.
        /// </summary>
        /// <remarks>
        /// For beginners: Controls how quickly the strategic decision-making adapts to new information.
        /// Usually lower than the low-level learning rate since strategic decisions change more slowly.
        /// </remarks>
        public double HighLevelLearningRate { get; set; } = 0.0001;
        
        /// <summary>
        /// Gets or sets the learning rate for the low-level policy.
        /// </summary>
        /// <remarks>
        /// For beginners: Controls how quickly the tactical decision-making adapts to new information.
        /// Usually higher than the high-level rate since tactical decisions need to adapt more quickly.
        /// </remarks>
        public double LowLevelLearningRate { get; set; } = 0.0003;
        
        /// <summary>
        /// Gets or sets the entropy coefficient for the high-level policy.
        /// </summary>
        /// <remarks>
        /// For beginners: This encourages the strategic level to explore different approaches
        /// rather than sticking to a single strategy. Higher values lead to more diverse strategic planning.
        /// </remarks>
        public double HighLevelEntropyCoef { get; set; } = 0.01;
        
        /// <summary>
        /// Gets or sets the entropy coefficient for the low-level policy.
        /// </summary>
        /// <remarks>
        /// For beginners: This encourages the tactical level to try different ways to execute
        /// the strategic plan. Usually higher than the high-level coefficient to allow for
        /// tactical flexibility.
        /// </remarks>
        public double LowLevelEntropyCoef { get; set; } = 0.02;
        
        /// <summary>
        /// Gets or sets the type of risk metric to use.
        /// 0 = Variance, 1 = Value at Risk (VaR), 2 = Conditional Value at Risk (CVaR)
        /// </summary>
        /// <remarks>
        /// For beginners: This determines how the model measures risk.
        /// - Variance: Considers all deviations from expected return
        /// - VaR: Focuses on a specific worst-case percentile (e.g., worst 5% of outcomes)
        /// - CVaR: Considers the average of all outcomes worse than VaR (most conservative)
        /// </remarks>
        public int RiskMetricType { get; set; } = 2;
        
        /// <summary>
        /// Gets or sets the confidence level for VaR/CVaR (e.g., 0.05 for 95% VaR).
        /// </summary>
        /// <remarks>
        /// For beginners: If using VaR or CVaR, this sets how extreme the worst-case scenario is.
        /// 0.05 means focusing on the worst 5% of possible outcomes. Lower values (like 0.01)
        /// mean focusing on even more rare but severe downside scenarios.
        /// </remarks>
        public double ConfidenceLevel { get; set; } = 0.05;
        
        /// <summary>
        /// Gets or sets whether to use a recurrent neural network for the high-level policy.
        /// </summary>
        /// <remarks>
        /// For beginners: When true, the strategic level will consider patterns over time,
        /// not just the current market state. This helps it identify trends and market regimes,
        /// similar to how a human strategist considers how markets have been evolving.
        /// </remarks>
        public bool UseRecurrentHighLevelPolicy { get; set; } = true;
        
        /// <summary>
        /// Gets or sets whether to use intrinsic rewards to encourage exploration.
        /// </summary>
        /// <remarks>
        /// For beginners: When true, the model receives rewards for exploring new market states
        /// and strategies, which helps it discover approaches it might otherwise miss.
        /// This is especially useful in changing market conditions.
        /// </remarks>
        public bool UseIntrinsicRewards { get; set; } = true;
        
        /// <summary>
        /// Gets or sets the intrinsic reward scale factor.
        /// </summary>
        /// <remarks>
        /// For beginners: This controls how much the model is rewarded for exploration
        /// compared to actual financial returns. Higher values encourage more exploration.
        /// </remarks>
        public double IntrinsicRewardScale { get; set; } = 0.1;
        
        /// <summary>
        /// Gets or sets whether to use a target network for more stable learning.
        /// </summary>
        /// <remarks>
        /// For beginners: When true, the model uses a more stable version of itself for
        /// evaluating actions, which helps prevent erratic learning behavior and makes
        /// training more reliable.
        /// </remarks>
        public bool UseTargetNetwork { get; set; } = true;
        
        /// <summary>
        /// Gets or sets the update frequency for the target network.
        /// </summary>
        /// <remarks>
        /// For beginners: How often the stable version of the model is updated to reflect
        /// recent learning. Higher values mean more stability but slower adaptation.
        /// </remarks>
        public new int TargetUpdateFrequency { get; set; } = 1000;
        
        /// <summary>
        /// Gets or sets whether to pretrain the model on historical data.
        /// </summary>
        /// <remarks>
        /// For beginners: When true, the model first learns from historical market data
        /// before trading with real money. This gives it a baseline understanding of market
        /// patterns similar to how human traders study market history.
        /// </remarks>
        public bool UsePretraining { get; set; } = false;
        
        /// <summary>
        /// Gets or sets whether to use hindsight experience replay for more efficient learning.
        /// </summary>
        /// <remarks>
        /// For beginners: When true, the model learns not just from what happened, but also
        /// from what could have happened with different actions. This accelerates learning
        /// by extracting more knowledge from each experience.
        /// </remarks>
        public bool UseHindsightExperienceReplay { get; set; } = true;
        
        /// <summary>
        /// Gets or sets the batch size for training updates.
        /// </summary>
        /// <remarks>
        /// For beginners: This controls how many experiences the model learns from at once.
        /// Larger batch sizes provide more stable learning but require more memory.
        /// </remarks>
        public new int BatchSize { get; set; } = 32;
    }
}