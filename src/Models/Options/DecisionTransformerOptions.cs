using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.ReinforcementLearning.Agents;

namespace AiDotNet.Models.Options;

/// <summary>
/// Configuration options for Decision Transformer agents.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Decision Transformer (Chen et al. 2021, "Decision Transformer: Reinforcement Learning via Sequence
/// Modeling") treats reinforcement learning as sequence modeling. A GPT-style causal transformer reads the
/// last <see cref="ContextLength"/> timesteps as interleaved (return-to-go, state, action) tokens and
/// predicts each action from the output at its state token. It is trained by supervised regression (or
/// cross-entropy for discrete actions) on trajectory segments sampled from an offline buffer, and at test
/// time it is conditioned on the return the caller wants to achieve.
/// </para>
/// <para><b>For Beginners:</b>
/// Decision Transformer is a radically different approach to RL. Instead of learning
/// "what action is best", it learns "what action was taken when the outcome was X".
/// Then at test time, you tell it "I want outcome X" and it generates actions.
///
/// Key innovation:
/// - **Sequence Modeling**: Uses transformers (like GPT) instead of RL algorithms
/// - **Return Conditioning**: Specify desired return, get action sequence
/// - **Offline-Friendly**: Works excellently with fixed datasets
/// - **No Value Functions**: No Q-networks or critics needed
///
/// Think of it like: "Show me examples of successful chess games, and I'll learn
/// to play moves that lead to success."
///
/// Famous for: Berkeley/Meta research showing transformers can replace RL algorithms
/// </para>
/// <para>
/// Defaults follow the paper's OpenAI Gym configuration (Chen et al. 2021, Appendix A, Gym hyperparameter table): 3 layers,
/// 1 head, embedding dimension 128, context length 20, dropout 0.1.
/// </para>
/// </remarks>
public class DecisionTransformerOptions<T> : ReinforcementLearningOptions<T>
{
    /// <summary>
    /// Gets or sets the length of the observation (state) vector.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Each state is embedded into a state token by its own linear map (Chen et al. 2021, Sec. 3), so this
    /// fixes the input width of that embedding.
    /// </para>
    /// <para><b>For Beginners:</b> How many numbers describe what the agent sees at one moment, for example
    /// 4 for a cart-pole (position, velocity, angle, angular velocity).</para>
    /// </remarks>
    /// <value>The state dimension; must be positive. Defaults to 16.</value>
    public int StateSize { get; init; } = 16;

    /// <summary>
    /// Gets or sets the length of the action vector (or the number of actions when discrete).
    /// </summary>
    /// <remarks>
    /// <para>
    /// For continuous control this is the action dimension the tanh head predicts; for discrete control
    /// (<see cref="IsContinuous"/> false) it is the number of actions, and actions are one-hot vectors.
    /// </para>
    /// <para><b>For Beginners:</b> How many numbers make up one action: the joint torques of a robot, or
    /// the number of buttons the agent can press.</para>
    /// </remarks>
    /// <value>The action dimension or action count; must be positive. Defaults to 4.</value>
    public int ActionSize { get; init; } = 4;

    /// <summary>
    /// Gets or sets whether actions are continuous vectors or discrete one-hot choices.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Continuous: the action head is linear followed by tanh and the objective is the mean-squared error
    /// to the dataset action (Chen et al. 2021, Sec. 3 and the Gym configuration). Discrete: the head emits
    /// logits and the objective is cross-entropy to the dataset action (the paper's Atari setting); a
    /// greedy query returns the arg-max one-hot action and an exploratory one samples the softmax.
    /// </para>
    /// <para><b>For Beginners:</b> Leave this true for robots and other "how much" actions; set it false
    /// for games where the agent picks one of a fixed set of moves.</para>
    /// </remarks>
    /// <value><c>true</c> (the default) for continuous actions; <c>false</c> for discrete actions.</value>
    public bool IsContinuous { get; init; } = true;

    /// <summary>
    /// Gets or sets the transformer's hidden (token embedding) width.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Every token -- return-to-go, state and action -- is projected to this width before the transformer
    /// (Chen et al. 2021, Sec. 3). The paper's Gym configuration uses 128 (Appendix A, Gym hyperparameter table).
    /// </para>
    /// <para><b>For Beginners:</b> How many numbers the model uses internally to describe each token.
    /// Bigger is more expressive but slower.</para>
    /// </remarks>
    /// <value>The embedding width; must be positive and divisible by <see cref="NumHeads"/>. Defaults to 128.</value>
    public int EmbeddingDim { get; init; } = 128;

    /// <summary>
    /// Gets or sets the number of transformer blocks.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Each block is a GPT-2 style pre-LayerNorm block: causal self-attention and a position-wise
    /// feed-forward network, each with a residual connection. The paper's Gym configuration uses 3
    /// (Appendix A, Gym hyperparameter table).
    /// </para>
    /// <para><b>For Beginners:</b> How many times the model re-reads the sequence to refine its
    /// understanding. More layers can learn more complex behaviour but train more slowly.</para>
    /// </remarks>
    /// <value>The number of blocks; must be positive. Defaults to 3.</value>
    public int NumLayers { get; init; } = 3;

    /// <summary>
    /// Gets or sets the number of attention heads in each block's causal self-attention.
    /// </summary>
    /// <remarks>
    /// <para>
    /// The embedding is split into this many heads that attend independently and are then recombined, as
    /// in GPT-2. The paper's Gym configuration uses 1 head (Appendix A, Gym hyperparameter table); its Atari configuration
    /// uses 8.
    /// </para>
    /// <para><b>For Beginners:</b> Each head can focus on a different relationship in the history (for
    /// example one on the last action, one on the target return). One head is the paper's choice for
    /// continuous control.</para>
    /// </remarks>
    /// <value>The head count; must be positive and divide <see cref="EmbeddingDim"/>. Defaults to 1.</value>
    public int NumHeads { get; init; } = 1;

    /// <summary>
    /// Gets or sets the context length K: how many most-recent timesteps the model conditions on.
    /// </summary>
    /// <remarks>
    /// <para>
    /// The transformer sees the last K timesteps, i.e. 3K tokens (Chen et al. 2021, Sec. 3). Training
    /// samples segments of up to K consecutive timesteps. The paper uses K = 20 for HalfCheetah, Hopper and
    /// Walker (Appendix A, Gym hyperparameter table).
    /// </para>
    /// <para><b>For Beginners:</b> How far back the agent remembers when choosing its next action.</para>
    /// </remarks>
    /// <value>The number of timesteps; must be positive. Defaults to 20.</value>
    public int ContextLength { get; init; } = 20;

    /// <summary>
    /// Gets or sets the dropout probability used throughout the transformer during training.
    /// </summary>
    /// <remarks>
    /// <para>
    /// As in GPT-2, dropout is applied to the summed token embeddings and to the output of every attention
    /// and feed-forward sublayer before its residual addition. It is active only while training; action
    /// selection is deterministic. The paper uses 0.1 (Appendix A, Gym hyperparameter table).
    /// </para>
    /// <para><b>For Beginners:</b> During training a random fraction of the model's internal signals is
    /// switched off each step, which stops it from memorising the dataset too closely.</para>
    /// </remarks>
    /// <value>A probability in [0, 1); 0 disables dropout. Defaults to 0.1.</value>
    public double DropoutRate { get; init; } = 0.1;

    /// <summary>
    /// Gets or sets the maximum episode length, which sizes the learned timestep embedding.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Decision Transformer adds a learned embedding of the absolute episode timestep to each of the three
    /// tokens of that timestep (Chen et al. 2021, Sec. 3: "an embedding for each timestep is learned and
    /// added to each token"). The embedding table has one row per timestep up to this bound; later
    /// timesteps share the last row. The reference Gym implementation uses 1000, the Gym episode limit.
    /// </para>
    /// <para><b>For Beginners:</b> The longest episode you expect. The model learns a separate "clock"
    /// signal for every step up to this length.</para>
    /// </remarks>
    /// <value>The number of distinct timestep embeddings; must be positive. Defaults to 1000.</value>
    public int MaxEpisodeLength { get; init; } = 1000;

    /// <summary>
    /// Gets or sets the capacity of the offline trajectory buffer, in transitions.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Decision Transformer trains on segments sampled from a fixed dataset of whole trajectories (Chen et
    /// al. 2021, Sec. 3, Algorithm 1). Trajectories arrive through <c>LoadOfflineData</c> or, episode by
    /// episode, through <c>StoreExperience</c>. When the stored transitions would exceed this capacity the
    /// oldest trajectories are evicted first; a single trajectory longer than the capacity keeps only its
    /// most recent transitions. The D4RL datasets the paper trains on hold up to about one million
    /// transitions.
    /// </para>
    /// <para><b>For Beginners:</b> How many recorded steps the agent keeps to learn from. When it is full,
    /// the oldest recorded games are forgotten first.</para>
    /// </remarks>
    /// <value>The maximum number of stored transitions; must be positive. Defaults to 1,000,000.</value>
    public int BufferSize { get; init; } = 1000000;

    /// <summary>
    /// Gets or sets the return the agent is asked to achieve at the start of each episode.
    /// </summary>
    /// <remarks>
    /// <para>
    /// At evaluation the first return-to-go token is this target and each later one is the previous one
    /// minus the reward just received (Chen et al. 2021, Sec. 3 and Algorithm 1). The paper picks a value
    /// per environment (the reference implementation evaluates Hopper at 3600, for example). When <c>null</c> the agent uses
    /// the highest episode return it has stored, the dataset maximum that the paper's Fig. 4 shows the
    /// model can reliably reach; with no stored data it uses zero.
    /// </para>
    /// <para><b>For Beginners:</b> The score you want the agent to aim for. Asking for a higher score makes
    /// it imitate the better trajectories it has seen.</para>
    /// </remarks>
    /// <value>The desired episode return, or <c>null</c> (the default) for the best stored episode return.</value>
    public double? TargetReturn { get; init; }

    /// <summary>
    /// Gets or sets the divisor applied to returns-to-go before they are embedded.
    /// </summary>
    /// <remarks>
    /// <para>
    /// The return token's input is the return-to-go divided by this scale, which keeps returns in the
    /// thousands from swamping the embedding. The reference Gym implementation divides by 1000; the Atari
    /// one does not scale. Rewards and target returns are always supplied unscaled.
    /// </para>
    /// <para><b>For Beginners:</b> If your episode scores are large (hundreds or thousands), set this to a
    /// similar magnitude so the model sees numbers near 1.</para>
    /// </remarks>
    /// <value>A positive, finite divisor. Defaults to 1.0 (no scaling).</value>
    public double ReturnScale { get; init; } = 1.0;

    /// <summary>
    /// Gets or sets a factory for the optimizer the transformer trains with.
    /// </summary>
    /// <remarks>
    /// <para>
    /// When this is <c>null</c>, the transformer trains with Adam (β1 0.9, β2 0.999, ε 1e-8) at the agent's
    /// <c>LearningRate</c>. The paper trains with AdamW (learning rate 1e-4, weight decay 1e-4, Appendix A);
    /// supply that here to reproduce it exactly.
    /// </para>
    /// <para>
    /// The agent invokes the factory once for every network it builds, so the factory must create a new
    /// optimizer instance on each call. A shared instance would share its step counter, and Adam's bias
    /// correction would then advance on every network's step instead of only on that network's own.
    /// </para>
    /// <para><b>For Beginners:</b> The optimizer is the rule that turns gradients into weight changes. Leave
    /// this unset to get Adam at the learning rate above; supply a factory such as
    /// <c>() =&gt; new AdamOptimizer&lt;T, Tensor&lt;T&gt;, Tensor&lt;T&gt;&gt;(null, myOptions)</c> to choose a
    /// different optimizer or its settings. Return a new optimizer from every call.
    /// </para>
    /// </remarks>
    /// <value>A delegate that returns a new optimizer on each call, or <c>null</c> (the default) for Adam at the
    /// agent's <c>LearningRate</c>.</value>
    public Func<IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>>? OptimizerFactory { get; init; }
}
