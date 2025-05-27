using AiDotNet.Enums;
using AiDotNet.Models.Options;

namespace AiDotNet.ReinforcementLearning.Models.Options
{
    /// <summary>
    /// Configuration options for the Multi-Agent Transformer model for financial market simulation and prediction.
    /// </summary>
    /// <remarks>
    /// The Multi-Agent Transformer model combines transformer architecture with multi-agent reinforcement learning to 
    /// model complex market dynamics and interactions between different market participants. This approach is particularly 
    /// effective for financial markets where the actions of various traders, institutions, and regulators 
    /// collectively influence market behavior.
    /// 
    /// For beginners: This model treats the market as a system of interacting agents (like traders, market makers, etc.) 
    /// and uses powerful transformer neural networks (similar to those in large language models) to understand how these 
    /// agents interact and influence each other's behavior over time. This helps predict market movements by modeling 
    /// the "market conversation" rather than just looking at price patterns.
    /// </remarks>
    public class MultiAgentTransformerOptions : ReinforcementLearningOptions
    {
        /// <summary>
        /// Gets or sets the number of agents to model in the market environment.
        /// </summary>
        /// <remarks>
        /// For beginners: This represents different types of market participants such as retail traders, institutions,
        /// market makers, etc. Each "agent" represents a different trading strategy or market role.
        /// </remarks>
        public int NumAgents { get; set; } = 4;

        /// <summary>
        /// Gets or sets the hidden dimension size for the transformer model.
        /// </summary>
        /// <remarks>
        /// For beginners: This controls how much information the model can remember and process about market conditions.
        /// Larger values allow for more complex market patterns to be learned but require more data and computation.
        /// </remarks>
        public int HiddenDimension { get; set; } = 128;

        /// <summary>
        /// Gets or sets the number of attention heads in the transformer model.
        /// </summary>
        /// <remarks>
        /// For beginners: Attention heads allow the model to focus on different aspects of market data simultaneously.
        /// More heads can help the model capture different relationships in the data (like price-volume relationships,
        /// correlations between different assets, etc.).
        /// </remarks>
        public int NumHeads { get; set; } = 8;

        /// <summary>
        /// Gets or sets the number of transformer layers to use.
        /// </summary>
        /// <remarks>
        /// For beginners: More layers allow the model to learn more complex patterns, similar to how deeper neural 
        /// networks can recognize more complex features in images. For financial markets, deeper models can 
        /// potentially capture more complex market behaviors.
        /// </remarks>
        public int NumLayers { get; set; } = 4;

        /// <summary>
        /// Gets or sets the length of market history sequences to consider.
        /// </summary>
        /// <remarks>
        /// For beginners: This is how many past time steps (like days or hours) the model looks at to make predictions.
        /// Longer sequences allow the model to capture longer-term patterns but require more memory and computation.
        /// </remarks>
        public int SequenceLength { get; set; } = 50;

        /// <summary>
        /// Gets or sets whether to use an attention mask to prevent information leakage from future to past.
        /// </summary>
        /// <remarks>
        /// For beginners: Setting this to true ensures the model only looks at past data when making predictions,
        /// which is important for realistic market simulation. If false, the model could "cheat" by looking at
        /// future data that wouldn't be available in real trading.
        /// </remarks>
        public bool UseCausalMask { get; set; } = true;

        /// <summary>
        /// Gets or sets the type of positional encoding to use for the transformer.
        /// </summary>
        /// <remarks>
        /// For beginners: This helps the model understand the order of data points in a sequence. Different encoding
        /// types work better for different types of time series patterns in financial data.
        /// </remarks>
        public PositionalEncodingType PositionalEncodingType { get; set; } = PositionalEncodingType.Sinusoidal;

        /// <summary>
        /// Gets or sets the type of communication allowed between agents.
        /// 0 = No communication, 1 = Global communication, 2 = Local communication with neighbors
        /// </summary>
        /// <remarks>
        /// For beginners: This controls how the model simulates information sharing between market participants.
        /// Different settings model different market efficiency scenarios - from completely independent agents
        /// to fully informed market participants.
        /// </remarks>
        public int CommunicationMode { get; set; } = 1;

        /// <summary>
        /// Gets or sets whether to use centralized training with decentralized execution.
        /// </summary>
        /// <remarks>
        /// For beginners: When true, during training the model can access all market information, but during prediction
        /// each agent only sees its own information. This helps the model learn better while maintaining realistic
        /// market simulation where traders don't have perfect information.
        /// </remarks>
        public bool UseCentralizedTraining { get; set; } = true;

        /// <summary>
        /// Gets or sets the learning rate for the transformer networks.
        /// </summary>
        /// <remarks>
        /// For beginners: Controls how quickly the model learns from data. Too high can cause unstable learning,
        /// too low can make learning too slow. Financial markets often require careful tuning of this parameter.
        /// </remarks>
        public double TransformerLearningRate { get; set; } = 0.0003;

        /// <summary>
        /// Gets or sets the weight for the entropy bonus to encourage exploration.
        /// </summary>
        /// <remarks>
        /// For beginners: This encourages the model to try different trading strategies rather than
        /// sticking with one approach. Higher values lead to more diverse trading behavior.
        /// </remarks>
        public double EntropyCoefficient { get; set; } = 0.01;

        /// <summary>
        /// Gets or sets whether to use self-play for training agents.
        /// </summary>
        /// <remarks>
        /// For beginners: When enabled, the model learns by competing against copies of itself,
        /// helping it discover more advanced strategies through competitive co-evolution,
        /// similar to how chess AI improves by playing against itself.
        /// </remarks>
        public bool UseSelfPlay { get; set; } = true;

        /// <summary>
        /// Gets or sets the risk aversion parameter (higher values = more risk-averse behavior).
        /// </summary>
        /// <remarks>
        /// For beginners: Controls how cautious the trading strategy will be. Higher values
        /// prioritize consistent returns over potentially higher but riskier returns.
        /// </remarks>
        public double RiskAversionParameter { get; set; } = 0.5;

        /// <summary>
        /// Gets or sets whether to model market impact of agent actions.
        /// </summary>
        /// <remarks>
        /// For beginners: When true, the model accounts for how large trades can themselves
        /// move the market price, which is important for realistic simulation of large trading strategies.
        /// </remarks>
        public bool ModelMarketImpact { get; set; } = true;
        
        /// <summary>
        /// Gets or sets the state dimension for the agents.
        /// </summary>
        /// <remarks>
        /// The number of features that describe the market state at each time step.
        /// </remarks>
        public int StateDimension { get; set; } = 10;
        
        /// <summary>
        /// Gets or sets the action dimension for the agents.
        /// </summary>
        /// <remarks>
        /// The number of possible actions each agent can take (e.g., buy/sell/hold amounts).
        /// </remarks>
        public int ActionDimension { get; set; } = 3;
        
        /// <summary>
        /// Gets or sets the entropy coefficient.
        /// </summary>
        /// <remarks>
        /// Alias for EntropyCoefficient for consistency with other models.
        /// </remarks>
        public double EntropyCoef 
        { 
            get => EntropyCoefficient; 
            set => EntropyCoefficient = value; 
        }
        
        /// <summary>
        /// Gets or sets the risk aversion parameter.
        /// </summary>
        /// <remarks>
        /// Alias for RiskAversionParameter for consistency with other models.
        /// </remarks>
        public double RiskAversion 
        { 
            get => RiskAversionParameter; 
            set => RiskAversionParameter = value; 
        }
    }
}