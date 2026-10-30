using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.ReinforcementLearning.Agents;

namespace AiDotNet.Models.Options;

/// <summary>
/// Configuration options for Dreamer agents (Hafner et al. 2020, "Dream to Control: Learning Behaviors by
/// Latent Imagination", ICLR 2020).
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Dreamer learns a recurrent state-space model (RSSM) of the environment from sequences of experience and
/// then learns an actor and a value model purely from trajectories imagined inside that model. The model
/// state is the pair of a deterministic recurrent state h_t and a stochastic Gaussian state z_t; the
/// representation model q(z_t | h_t, o_t) infers z_t from the observation, the transition model
/// p(z_t | h_t) predicts it without the observation, and the observation, reward and discount models read
/// the combined state (paper Section 4, Equations 9-10; Appendix A). The paper writes the representation model
/// as p and the transition model as q; these docs use the RSSM's posterior (q) and prior (p) names instead.
/// </para>
/// <para><b>For Beginners:</b>
/// Dreamer learns a "mental model" of how the environment works, then uses that
/// model to imagine future scenarios and learn from them - like practising a game in your head.
///
/// Key components:
/// - **World Model (RSSM)**: Learns environment dynamics in a compact latent space
/// - **Representation Model**: Combines the memory with the current observation
/// - **Transition Model**: Predicts the next latent state without seeing the observation
/// - **Reward / Discount Models**: Predict rewards and whether the episode continues
/// - **Actor and Value Models**: Learn the policy from imagined trajectories
///
/// Advantages: very sample efficient, because most learning happens in imagination.
/// </para>
/// </remarks>
public class DreamerOptions<T> : ReinforcementLearningOptions<T>
{
    private int _observationSize = 1;
    private int _actionSize = 1;

    /// <summary>
    /// Initializes the options with the hyperparameters of Hafner et al. (2020), Appendix A.
    /// </summary>
    /// <remarks>
    /// <para>
    /// The only inherited default changed here is <c>BatchSize</c>: the paper trains on batches of
    /// B = 50 sequences ("We draw batches of 50 sequences of length 50", Appendix A).
    /// </para>
    /// <para><b>For Beginners:</b> Creating the options with <c>new DreamerOptions&lt;T&gt;()</c> gives you the
    /// settings the paper used on the DeepMind Control Suite; you normally only need to set the observation
    /// and action sizes.</para>
    /// </remarks>
    public DreamerOptions()
    {
        BatchSize = 50;
    }

    /// <summary>
    /// Gets the width of each observation vector the agent receives.
    /// </summary>
    /// <remarks>
    /// <para>
    /// The observation is encoded by a dense encoder (the paper's convolutional encoder is for 64x64 images;
    /// for vector observations the encoder is three dense layers, as every non-convolutional function in
    /// Appendix A is) and reconstructed by the observation model q(o_t | s_t).
    /// </para>
    /// <para><b>For Beginners:</b> How many numbers describe what the agent sees at each step.</para>
    /// </remarks>
    /// <value>A positive integer; default 1.</value>
    public int ObservationSize
    {
        get => _observationSize;
        init
        {
            if (value <= 0)
            {
                throw new ArgumentException("ObservationSize must be positive", nameof(ObservationSize));
            }
            _observationSize = value;
        }
    }

    /// <summary>
    /// Gets the width of the continuous action vector the agent emits.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Actions are drawn from a tanh-transformed Gaussian (paper Equation 3), so every component lies in
    /// [-1, 1].
    /// </para>
    /// <para><b>For Beginners:</b> How many continuous controls the agent sets at each step, each between
    /// -1 and 1.</para>
    /// </remarks>
    /// <value>A positive integer; default 1.</value>
    public int ActionSize
    {
        get => _actionSize;
        init
        {
            if (value <= 0)
            {
                throw new ArgumentException("ActionSize must be positive", nameof(ActionSize));
            }
            _actionSize = value;
        }
    }

    /// <summary>
    /// Gets the size of the RSSM's deterministic recurrent state h_t.
    /// </summary>
    /// <remarks>
    /// <para>
    /// h_t is the hidden state of the GRU (Cho et al. 2014) that carries the model's memory:
    /// h_t = GRU(h_{t-1}, [z_{t-1}, a_{t-1}]). Dreamer uses the RSSM of PlaNet (Hafner et al. 2018), whose
    /// deterministic state has 200 units; the RSSM's own hidden layers (before the GRU and before the prior
    /// and posterior heads) use the same width, as in the authors' implementation.
    /// </para>
    /// <para><b>For Beginners:</b> The size of the agent's "memory" of what happened earlier in the episode.
    /// Bigger memories can track more, but train more slowly.</para>
    /// </remarks>
    /// <value>A positive integer; default 200.</value>
    public int DeterministicSize { get; init; } = 200;

    /// <summary>
    /// Gets the size of the RSSM's stochastic state z_t.
    /// </summary>
    /// <remarks>
    /// <para>
    /// z_t is a diagonal Gaussian. The representation model infers it as q(z_t | h_t, o_t) and the transition
    /// model predicts it as p(z_t | h_t). "Distributions in latent space are 30-dimensional diagonal
    /// Gaussians" (Appendix A).
    /// </para>
    /// <para><b>For Beginners:</b> The size of the uncertain part of the agent's belief about the current
    /// situation. It is what lets the model represent "I am not sure exactly what happens next".</para>
    /// </remarks>
    /// <value>A positive integer; default 30.</value>
    public int StochasticSize { get; init; } = 30;

    /// <summary>
    /// Gets the width of the dense layers of the encoder, observation, reward, discount, action and value
    /// models.
    /// </summary>
    /// <remarks>
    /// <para>
    /// "We ... implement all other functions as three dense layers of size 300 with ELU activations"
    /// (Appendix A). The encoder's last dense layer is the observation embedding the representation model
    /// reads.
    /// </para>
    /// <para><b>For Beginners:</b> How many neurons each layer of the agent's helper networks has.</para>
    /// </remarks>
    /// <value>A positive integer; default 300.</value>
    public int HiddenSize { get; init; } = 300;

    /// <summary>
    /// Gets the length L of each sequence sampled from the replay data to train the world model.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Every training step draws <c>BatchSize</c> sequences of <c>BatchLength</c> consecutive transitions
    /// from one episode and unrolls the RSSM along them (Algorithm 1, "Draw B data sequences"; Appendix A
    /// uses L = 50). While the stored episodes are still shorter than this, the agent trains on the longest
    /// sequences its data contains.
    /// </para>
    /// <para><b>For Beginners:</b> How many consecutive steps of real experience the world model studies at a
    /// time. Longer sequences teach it longer-term memory.</para>
    /// </remarks>
    /// <value>A positive integer; default 50.</value>
    public int BatchLength { get; init; } = 50;

    /// <summary>
    /// Gets the imagination horizon H: how many steps the actor and value models look ahead inside the
    /// learned model.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Imagined trajectories {s_tau, a_tau} for tau = t..t+H branch from every model state of the sampled
    /// sequences (Section 3, Algorithm 1). Appendix A uses H = 15.
    /// </para>
    /// <para><b>For Beginners:</b> How far into the imagined future the agent plans when it learns its
    /// policy.</para>
    /// </remarks>
    /// <value>A positive integer; default 15.</value>
    public int ImaginationHorizon { get; init; } = 15;

    /// <summary>
    /// Gets the lambda of the V-lambda value targets.
    /// </summary>
    /// <remarks>
    /// <para>
    /// V_lambda (Equation 6) is the exponentially weighted average of the k-step estimates V_N^k
    /// (Equation 5); lambda = 0 uses only the one-step estimate, lambda = 1 only the full-horizon one.
    /// Appendix A uses lambda = 0.95.
    /// </para>
    /// <para><b>For Beginners:</b> How much the agent trusts long imagined futures over its own value
    /// guesses. Values close to 1 lean on the imagined rewards; values close to 0 lean on the value model.</para>
    /// </remarks>
    /// <value>A value in [0, 1]; default 0.95.</value>
    public double Lambda { get; init; } = 0.95;

    /// <summary>
    /// Gets the scale beta of the KL regularizer between the representation (posterior) and transition
    /// (prior) models.
    /// </summary>
    /// <remarks>
    /// <para>
    /// The world-model objective contains J_D = -beta KL(representation || transition) (Equation 10; the paper
    /// writes the representation model as p(s_t | s_{t-1}, a_{t-1}, o_t) and the transition model as
    /// q(s_t | s_{t-1}, a_{t-1})). "We do not scale the KL regularizers (beta = 1)" for continuous control;
    /// the discrete-control experiments use beta = 0.1 (Appendix A).
    /// </para>
    /// <para><b>For Beginners:</b> How strongly the model is pushed to predict its next state without
    /// peeking at the next observation. Larger values give a model that imagines more accurately but
    /// remembers less detail of each observation.</para>
    /// </remarks>
    /// <value>A non-negative value; default 1.0.</value>
    public double KLScale { get; init; } = 1.0;

    /// <summary>
    /// Gets the free-nats threshold below which the KL regularizer is not optimized.
    /// </summary>
    /// <remarks>
    /// <para>
    /// The KL term, averaged over the batch and the sequence, is clipped from below at this many nats
    /// ("clip them below 3 free nats as in PlaNet", Appendix A), so the regularizer stops pulling once the
    /// posterior is within that budget of the prior.
    /// </para>
    /// <para><b>For Beginners:</b> A small allowance of information the model may take from each observation
    /// without being penalized for it.</para>
    /// </remarks>
    /// <value>A non-negative value; default 3.0.</value>
    public double FreeNats { get; init; } = 3.0;

    /// <summary>
    /// Gets the weight of the reward log-likelihood ln q(r_t | s_t) in the world-model objective.
    /// </summary>
    /// <remarks>
    /// <para>
    /// The paper's objective (Equation 10) sums the reward term J_R with weight 1. This option scales that
    /// term, as the "reward" loss scale of the Dreamer line (DreamerV2's <c>loss_scales.reward</c>) does;
    /// the default of 1 is the paper's objective.
    /// </para>
    /// <para><b>For Beginners:</b> How much the world model cares about predicting rewards correctly
    /// compared to reconstructing observations.</para>
    /// </remarks>
    /// <value>A non-negative value; default 1.0.</value>
    public double RewardScale { get; init; } = 1.0;

    /// <summary>
    /// Gets the weight of the discount (episode-continuation) predictor's log-likelihood in the world-model
    /// objective; zero disables the predictor.
    /// </summary>
    /// <remarks>
    /// <para>
    /// "In tasks with early termination, the world model also predicts the discount factor from each latent
    /// state to weigh the time steps in Equations 7 and 8 by the cumulative product of the predicted
    /// discount factors" (Section 3). It is "a binary classifier that is trained towards the soft labels of 0
    /// and gamma" (Appendix A). The paper does not state the weight of this term; the default of 10 is the
    /// authors' reference implementation (<c>pcont_scale = 10</c>, github.com/danijar/dreamer). With 0 the
    /// predictor is neither trained nor used, and imagination discounts every step by the constant
    /// <c>DiscountFactor</c>, which is the paper's setting for tasks without early termination.
    /// </para>
    /// <para><b>For Beginners:</b> How much the world model cares about predicting when an episode ends. Set
    /// it to 0 for tasks that never end early.</para>
    /// </remarks>
    /// <value>A non-negative value; default 10.0.</value>
    public double ContinueScale { get; init; } = 10.0;

    /// <summary>
    /// Gets or sets a factory for the optimizer each of this agent's networks trains with.
    /// </summary>
    /// <remarks>
    /// <para>
    /// When this is <c>null</c>, each network trains with Adam (β1 0.9, β2 0.999, ε 1e-8) at the agent's
    /// <c>LearningRate</c>, with each step's gradient rescaled to a global norm of at most 100 ("scale down
    /// gradient norms that exceed 100", Appendix A).
    /// </para>
    /// <para>
    /// The agent invokes the factory once for every network it builds, so the factory must create a new
    /// optimizer instance on each call. A shared instance would share its step counter, and Adam's bias
    /// correction would then advance on every network's step instead of only on that network's own.
    /// </para>
    /// <para><b>For Beginners:</b> The optimizer is the rule that turns each network's gradients into weight
    /// changes. Leave this unset to get Adam at the learning rate above; supply a factory such as
    /// <c>() =&gt; new AdamOptimizer&lt;T, Tensor&lt;T&gt;, Tensor&lt;T&gt;&gt;(null, myOptions)</c> to choose a
    /// different optimizer or its settings. Return a new optimizer from every call.
    /// </para>
    /// </remarks>
    /// <value>A delegate that returns a new optimizer on each call, or <c>null</c> (the default) for Adam at the
    /// agent's <c>LearningRate</c>.</value>
    public Func<IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>>? OptimizerFactory { get; init; }
}
