using AiDotNet.Enums;
using AiDotNet.Interfaces;

namespace AiDotNet.ReinforcementLearning.Models.Options
{
    /// <summary>
    /// Options for configuring the Decision Transformer reinforcement learning algorithm.
    /// </summary>
    /// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
    /// <remarks>
    /// <para>
    /// Decision Transformer is a transformer-based architecture that reformulates reinforcement learning
    /// as a sequence modeling problem. It predicts actions based on the history of states, actions, 
    /// and target returns, making it particularly effective for financial market prediction.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b>
    /// The Decision Transformer is a modern approach to reinforcement learning that works by:
    /// - Treating the problem as "predicting the next word in a sentence" but for actions
    /// - Looking at a window of past market states to make decisions
    /// - Considering both what happened and what return you want to achieve
    /// - Using the same transformer architecture that powers large language models
    /// 
    /// This model is especially good for stock market prediction because it:
    /// - Can be trained on historical data without interacting with real markets
    /// - Handles long-term patterns in price data effectively
    /// - Adapts to different market regimes (bull, bear, sideways)
    /// - Makes predictions that account for desired risk/return profiles
    /// </para>
    /// </remarks>
    public class DecisionTransformerOptions<T> : ReinforcementLearningOptions
    {
        /// <summary>
        /// Gets or sets the length of the context window (number of previous timesteps to consider).
        /// </summary>
        /// <remarks>
        /// <para>
        /// The context length determines how many previous observations, actions, and returns
        /// the transformer considers when predicting the next action. A larger context length
        /// captures longer-term dependencies but requires more memory and computation.
        /// </para>
        /// <para>
        /// <b>For Beginners:</b>
        /// This is like how many days of market history the model looks at to make a decision.
        /// Longer history means more context but slower training.
        /// </para>
        /// </remarks>
        public int ContextLength { get; set; } = 20;

        /// <summary>
        /// Gets or sets the number of transformer layers in the model.
        /// </summary>
        /// <remarks>
        /// <para>
        /// The number of transformer layers controls the depth of the model. More layers can
        /// capture more complex patterns but increase the computational cost and the risk of overfitting.
        /// </para>
        /// <para>
        /// <b>For Beginners:</b>
        /// Think of this as the "depth" of the model's thinking. More layers can find more complex
        /// patterns but might also "overthink" on noisy market data.
        /// </para>
        /// </remarks>
        public int NumTransformerLayers { get; set; } = 4;

        /// <summary>
        /// Gets or sets the number of attention heads in each transformer layer.
        /// </summary>
        /// <remarks>
        /// <para>
        /// Attention heads allow the model to focus on different aspects of the input data
        /// simultaneously. More heads can capture more diverse patterns and relationships.
        /// </para>
        /// <para>
        /// <b>For Beginners:</b>
        /// This controls how many different patterns the model can look for at once.
        /// For example, one head might focus on volume trends while another watches price movements.
        /// </para>
        /// </remarks>
        public int NumHeads { get; set; } = 8;

        /// <summary>
        /// Gets or sets the dimension of the embedding vectors used in the transformer.
        /// </summary>
        /// <remarks>
        /// <para>
        /// The embedding dimension determines how much information can be encoded about each
        /// observation, action, or return. Larger dimensions can capture more nuanced representations
        /// but increase the model size.
        /// </para>
        /// <para>
        /// <b>For Beginners:</b>
        /// This is how "rich" the model's internal representation is. Higher values let it capture
        /// more subtle patterns in market data but make training slower.
        /// </para>
        /// </remarks>
        public int EmbeddingDim { get; set; } = 128;

        /// <summary>
        /// Gets or sets a value indicating whether the model is conditioned on target returns.
        /// </summary>
        /// <remarks>
        /// <para>
        /// When enabled, the model uses target returns as part of its input, allowing it to
        /// generate actions that aim to achieve specific return targets. When disabled, the
        /// model predicts actions based only on past states and actions.
        /// </para>
        /// <para>
        /// <b>For Beginners:</b>
        /// When turned on, you can tell the model "I want to make X% return" and it will try to
        /// suggest trades that achieve that goal. When off, it just tries to predict the best action
        /// without a specific target.
        /// </para>
        /// </remarks>
        public bool ReturnConditioned { get; set; } = true;

        /// <summary>
        /// Gets or sets a value indicating whether to train the model on historical data (offline learning).
        /// </summary>
        /// <remarks>
        /// <para>
        /// When enabled, the model learns from a dataset of historical trajectories without
        /// interacting with the environment. This is particularly useful for financial applications
        /// where live training can be costly or risky.
        /// </para>
        /// <para>
        /// <b>For Beginners:</b>
        /// When set to true, the model learns from historical market data rather than by
        /// trial and error in live markets. This is safer and more practical for financial applications.
        /// </para>
        /// </remarks>
        public bool OfflineTraining { get; set; } = true;

        /// <summary>
        /// Gets or sets the learning rate for the transformer model.
        /// </summary>
        /// <remarks>
        /// <para>
        /// The learning rate controls how quickly the model adapts to the training data.
        /// A higher rate can lead to faster convergence but may cause instability, while
        /// a lower rate provides more stability but slower learning.
        /// </para>
        /// <para>
        /// <b>For Beginners:</b>
        /// This controls how quickly the model updates its understanding of market patterns.
        /// Too high = erratic learning, too low = painfully slow progress.
        /// </para>
        /// </remarks>
        public double TransformerLearningRate { get; set; } = 1e-4;

        /// <summary>
        /// Gets or sets the maximum sequence length for trajectories in training data.
        /// </summary>
        /// <remarks>
        /// <para>
        /// This parameter limits the length of trajectories used during training. Longer
        /// trajectories provide more context but require more memory and computation.
        /// In financial applications, this might correspond to the maximum trading sequence
        /// length to consider.
        /// </para>
        /// <para>
        /// <b>For Beginners:</b>
        /// This is the maximum length of a trading sequence that the model will learn from.
        /// For stock trading, this might be how many days of continuous trading to consider as one lesson.
        /// </para>
        /// </remarks>
        public int MaxTrajectoryLength { get; set; } = 1000;

        /// <summary>
        /// Gets or sets the type of positional encoding to use in the transformer.
        /// </summary>
        /// <remarks>
        /// <para>
        /// Positional encoding helps the transformer understand the order of elements in the sequence.
        /// Different encoding types have different properties and may be more suitable for certain tasks.
        /// </para>
        /// <para>
        /// <b>For Beginners:</b>
        /// This determines how the model keeps track of "when" things happened in your market data sequence.
        /// The transformer needs this since it looks at all time steps at once, unlike humans who process sequentially.
        /// </para>
        /// </remarks>
        public PositionalEncodingType PositionalEncodingType { get; set; } = PositionalEncodingType.Sinusoidal;

        /// <summary>
        /// Gets or sets a value indicating whether to normalize returns during training.
        /// </summary>
        /// <remarks>
        /// <para>
        /// Normalizing returns can improve training stability by bringing all return values
        /// into a similar range. This is especially important for financial data where
        /// returns can vary significantly.
        /// </para>
        /// <para>
        /// <b>For Beginners:</b>
        /// When turned on, the model scales all returns to a similar range, which helps it
        /// learn more effectively without being thrown off by occasional huge gains or losses.
        /// </para>
        /// </remarks>
        public bool NormalizeReturns { get; set; } = true;

        /// <summary>
        /// Gets or sets the dropout rate for the transformer layers.
        /// </summary>
        /// <remarks>
        /// <para>
        /// Dropout is a regularization technique that helps prevent overfitting by randomly
        /// "dropping out" some neurons during training. The dropout rate specifies the
        /// probability of dropping a neuron.
        /// </para>
        /// <para>
        /// <b>For Beginners:</b>
        /// This helps prevent the model from memorizing the exact patterns in historical data
        /// too closely, making it more likely to generalize well to future market conditions.
        /// </para>
        /// </remarks>
        public double DropoutRate { get; set; } = 0.1;

        /// <summary>
        /// Gets or sets the maximum size of the replay buffer for storing experiences.
        /// </summary>
        /// <remarks>
        /// <para>
        /// The replay buffer stores historical experiences (states, actions, rewards) that
        /// the agent can learn from. A larger buffer allows the agent to learn from more
        /// diverse experiences but requires more memory.
        /// </para>
        /// <para>
        /// <b>For Beginners:</b>
        /// This is like the agent's memory capacity - how many past trading experiences
        /// it can remember and learn from. More memory means better learning but uses more RAM.
        /// </para>
        /// </remarks>
        public int MaxBufferSize { get; set; } = 100000;

        /// <summary>
        /// Gets or sets how often the model should be updated with new training data (in steps).
        /// </summary>
        /// <remarks>
        /// <para>
        /// The update frequency controls how often the agent trains on batches from its replay buffer.
        /// A lower value means more frequent updates (more learning but potentially unstable),
        /// while a higher value means less frequent updates (more stable but slower learning).
        /// </para>
        /// <para>
        /// <b>For Beginners:</b>
        /// This is like how often the model "studies" its past experiences. Lower numbers = more studying
        /// (faster learning but might get nervous/unstable), higher numbers = less studying (slower but steadier).
        /// </para>
        /// </remarks>
        public int UpdateFrequency { get; set; } = 4;

        /// <summary>
        /// Gets or sets the optimizer instance to use for training the transformer model.
        /// </summary>
        /// <remarks>
        /// <para>
        /// The optimizer determines how the model parameters are updated during training.
        /// If null, a default AdamOptimizer with the specified learning rate will be used,
        /// which is generally recommended for transformer-based models.
        /// </para>
        /// <para>
        /// <b>For Beginners:</b>
        /// The optimizer is like the "learning strategy" - how the model adjusts its understanding
        /// based on mistakes. Adam is generally a good choice for financial prediction models
        /// because it adapts well to different types of patterns in market data.
        /// </para>
        /// </remarks>
        public IGradientBasedOptimizer<T, Vector<T>, Vector<T>>? Optimizer { get; set; } = null;
    }
}