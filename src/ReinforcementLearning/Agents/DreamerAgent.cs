using AiDotNet.LearningRateSchedulers;
using AiDotNet.ActivationFunctions;
using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.LossFunctions;
using AiDotNet.Models;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Optimizers;
using AiDotNet.ReinforcementLearning.ReplayBuffers;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Validation;

namespace AiDotNet.ReinforcementLearning.Agents.Dreamer;

/// <summary>
/// Dreamer agent (Hafner et al. 2020, "Dream to Control: Learning Behaviors by Latent Imagination").
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// <b>World model (Section 4, Equations 9-10; Appendix A).</b> A recurrent state-space model (RSSM, Hafner et
/// al. 2018) whose model state s_t = (h_t, z_t) pairs a deterministic GRU state h_t
/// (<see cref="DreamerOptions{T}.DeterministicSize"/>) with a diagonal-Gaussian stochastic state z_t
/// (<see cref="DreamerOptions{T}.StochasticSize"/>):
/// h_t = GRU(h_{t-1}, [z_{t-1}, a_{t-1}]); transition (prior) p(z_t | h_t); representation (posterior)
/// q(z_t | h_t, e_t) with e_t = enc(o_t); observation model q(o_t | s_t); reward model q(r_t | s_t); and, for
/// tasks with early termination, a discount model trained toward the soft labels 0 and gamma (Section 3,
/// Appendix A). The model maximises Equation 10 on <see cref="DreamerOptions{T}.BatchLength"/>-step sequences
/// drawn from the replay data: reconstruction + <see cref="DreamerOptions{T}.RewardScale"/> * reward
/// + <see cref="DreamerOptions{T}.ContinueScale"/> * discount log-likelihoods, minus
/// <see cref="DreamerOptions{T}.KLScale"/> * KL(posterior || prior) clipped below
/// <see cref="DreamerOptions{T}.FreeNats"/>. (The paper writes the representation model as p and the transition
/// model as q; this class uses the RSSM's posterior / prior names.)
/// </para>
/// <para>
/// <b>Behaviour learning (Section 3, Equations 2-8).</b> From every posterior state of the sampled sequences
/// the agent imagines <see cref="DreamerOptions{T}.ImaginationHorizon"/> steps with the transition model and
/// a tanh-transformed Gaussian action model (Equation 3), computes the V-lambda targets of Equation 6, and
/// trains the action model by backpropagating V-lambda through the imagined dynamics (Equation 7) and the
/// value model by regressing V-lambda (Equation 8). The world model is fixed while behaviours are learned.
/// </para>
/// <para><b>For Beginners:</b>
/// Dreamer learns a "mental model" of how the environment works, then uses that
/// model to imagine future scenarios and plan actions - like chess players
/// thinking multiple moves ahead.
///
/// Key components:
/// - **Representation Model**: Combines the agent's memory with the current observation
/// - **Transition Model**: Predicts the next latent state without seeing the observation
/// - **Reward / Discount Models**: Predict rewards and whether the episode continues
/// - **Value Model**: Estimates how good an imagined state is
/// - **Action Model**: Learns the policy entirely in imagination
///
/// Advantages: Sample efficient, works with images, enables planning
/// </para>
/// </remarks>
/// <example>
/// <code>
/// // Create a Dreamer agent for an 8-dimensional observation and 2 continuous actions
/// var options = new DreamerOptions&lt;double&gt; { ObservationSize = 8, ActionSize = 2 };
/// var agent = new DreamerAgent&lt;double&gt;(options);
///
/// // Act: the agent filters its latent belief over the episode and returns an action in [-1, 1]
/// var observation = new Vector&lt;double&gt;(new double[8]);
/// var action = agent.SelectAction(observation);
/// </code>
/// </example>
[ModelDomain(ModelDomain.MachineLearning)]
[ModelCategory(ModelCategory.ReinforcementLearningAgent)]
[ModelCategory(ModelCategory.NeuralNetwork)]
[ModelTask(ModelTask.Classification)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("Dream to Control: Learning Behaviors by Latent Imagination",
    "https://arxiv.org/abs/1912.01603",
    Year = 2020,
    Authors = "Hafner, D., Lillicrap, T., Ba, J., & Norouzi, M.")]
[PaperOptimizer(OptimizerKind.Adam,
                Source = "Hafner et al. 2020, Learning updates: batches of 50 sequences of length 50 "
                        + "train the world, value and action models using Adam. No single learning rate "
                        + "is declared because the paper sets 6e-4, 8e-5 and 8e-5 for those three models "
                        + "respectively, and one optimizer over all of them could not honour any single "
                        + "value. No reference batch size is declared either, since the batch is given "
                        + "as sequences rather than examples.")]
public partial class DreamerAgent<T> : DeepReinforcementLearningAgentBase<T>
{

    /// <inheritdoc />
    /// <remarks>Every network this agent owns, in the order Networks yields them: encoder, the RSSM's
    /// transition input, GRU gate and GRU candidate layers, prior, posterior, then the observation, reward,
    /// discount, action and value models.</remarks>
    protected override void RegisterComponents()
    {
        foreach (var network in Networks) RegisterParameterComponent(network);
    }

    /// <summary>Lower bound added to the RSSM's softplus standard deviations (PlaNet / Dreamer RSSM, min_stddev 0.1).</summary>
    private const double MinimumStateStd = 0.1;

    /// <summary>"The action model outputs a tanh mean scaled by a factor of 5" (Appendix A).</summary>
    private const double ActionMeanScale = 5.0;

    /// <summary>Initial standard deviation of the action distribution before the tanh (authors' implementation, init_std 5).</summary>
    private const double ActionInitialStd = 5.0;

    /// <summary>Lower bound of the action model's standard deviation (authors' implementation, min_std 1e-4).</summary>
    private const double MinimumActionStd = 1e-4;

    /// <summary>"executing the predicted mode action with Normal(0, 0.3) exploration noise" (Appendix A).</summary>
    private const double ExplorationStd = 0.3;

    /// <summary>
    /// "scale down gradient norms that exceed 100" (Appendix A), per model group: the authors' implementation clips
    /// the global norm of each optimizer's variable set, and it has three optimizers, so the world model's networks
    /// are clipped together as one group, and the action and value models each on their own.
    /// </summary>
    private const double MaxGradientNorm = 100.0;

    // Deterministic base seed for each network's lazy weight initialisation when the options carry no Seed, so a
    // Clone()'d agent (rebuilt from configuration, then given the serialized parameters) resolves exactly the
    // same weights for any network that had not materialized yet (Clone_ShouldProduceSamePolicy).
    private const int UnseededInitializationBase = 9001;

    private DreamerOptions<T> _options;

    // ----- World model (RSSM + heads) -----
    private NeuralNetwork<T> _encoderNetwork;           // o_t -> e_t (three dense ELU layers)
    private NeuralNetwork<T> _transitionInputNetwork;   // [z_{t-1}, a_{t-1}] -> x_t (dense ELU)
    private NeuralNetwork<T> _gruGateNetwork;           // [x_t, h_{t-1}] -> reset and update gate pre-activations
    private NeuralNetwork<T> _gruCandidateNetwork;      // [x_t, r_t * h_{t-1}] -> candidate pre-activation
    private NeuralNetwork<T> _priorNetwork;             // h_t -> (mean, std) of p(z_t | h_t)
    private NeuralNetwork<T> _posteriorNetwork;         // [h_t, e_t] -> (mean, std) of q(z_t | h_t, e_t)
    private NeuralNetwork<T> _observationNetwork;       // s_t -> mean of q(o_t | s_t)
    private NeuralNetwork<T> _rewardNetwork;            // s_t -> mean of q(r_t | s_t)
    private NeuralNetwork<T> _continueNetwork;          // s_t -> logit of the discount predictor

    // ----- Behaviour -----
    private NeuralNetwork<T> _actorNetwork;             // s_t -> (mean, std) before the tanh of q(a_t | s_t)
    private NeuralNetwork<T> _valueNetwork;             // s_t -> v(s_t)

    private SequenceReplay _replay;
    private int _updateCount;

    // The agent's filtered belief for acting (Figure 3c: "encodes the history of the episode to compute the current
    // model state"). Episode-local and rebuilt from the observations, so it is not model state.
    [Scratch]
    private Vector<T> _beliefDeterministic;
    [Scratch]
    private Vector<T> _beliefStochastic;
    [Scratch]
    private Vector<T> _previousAction;
    private bool _beliefActive;

    /// <summary>The world-model terms of the most recent training step, for diagnostics and tests.</summary>
    [Scratch]
    private WorldModelTerms _lastWorldModelTerms;

    /// <summary>The sequence length the most recent training step unrolled the world model over.</summary>
    [Scratch]
    private int _lastSequenceLength;

    /// <summary>
    /// Initializes a new instance with default settings.
    /// </summary>
    /// <remarks>
    /// The zero-argument default builds a small, self-consistent continuous-control toy
    /// agent: a 4-dimensional observation and a 4-dimensional continuous action. Four is the
    /// canonical low-dimensional control state (e.g. CartPole's [cart position, cart velocity,
    /// pole angle, pole angular velocity]) and keeps <see cref="DreamerOptions{T}.ObservationSize"/> and
    /// <see cref="DreamerOptions{T}.ActionSize"/> aligned so that a plain
    /// <c>(state, target)</c> transition of that width is accepted by
    /// <see cref="StoreExperience"/> without any environment-specific configuration.
    /// Real tasks should pass a fully-specified <see cref="DreamerOptions{T}"/>.
    /// </remarks>
    public DreamerAgent()
        : this(new DreamerOptions<T> { ObservationSize = 4, ActionSize = 4 })
    {
    }

    /// <summary>
    /// Initializes a Dreamer agent from <paramref name="options"/>.
    /// </summary>
    /// <param name="options">The agent's configuration.</param>
    /// <exception cref="ArgumentOutOfRangeException">A size, scale or horizon option is outside its valid range.</exception>
    public DreamerAgent(DreamerOptions<T> options)
        : base(options)
    {
        Guard.NotNull(options);
        _options = options;
        Options = _options;
        ValidateOptions(options);

        _updateCount = 0;

        int observation = options.ObservationSize;
        int action = options.ActionSize;
        int deterministic = options.DeterministicSize;
        int stochastic = options.StochasticSize;
        int hidden = options.HiddenSize;
        int feature = deterministic + stochastic;

        // Appendix A: "implement all other functions as three dense layers of size 300 with ELU activations";
        // the RSSM's internal layers use the deterministic width, as PlaNet's RSSM (hidden = deter = 200) does.
        // Appendix A also gives "learning rates of 6 x 10^-4 for the world model, 8 x 10^-5 for the value model, and
        // 8 x 10^-5 for the action model": every world-model network at LearningRate, the action and value models at
        // their own.
        _encoderNetwork = BuildNetwork(0, LearningRate, observation, hidden, hiddenLayers: 2, hiddenWidth: hidden, outputIsElu: true);
        _transitionInputNetwork = BuildNetwork(1, LearningRate, stochastic + action, deterministic, hiddenLayers: 0, hiddenWidth: 0, outputIsElu: true);
        _gruGateNetwork = BuildNetwork(2, LearningRate, 2 * deterministic, 2 * deterministic, hiddenLayers: 0, hiddenWidth: 0, outputIsElu: false);
        _gruCandidateNetwork = BuildNetwork(3, LearningRate, 2 * deterministic, deterministic, hiddenLayers: 0, hiddenWidth: 0, outputIsElu: false);
        _priorNetwork = BuildNetwork(4, LearningRate, deterministic, 2 * stochastic, hiddenLayers: 1, hiddenWidth: deterministic, outputIsElu: false);
        _posteriorNetwork = BuildNetwork(5, LearningRate, deterministic + hidden, 2 * stochastic, hiddenLayers: 1, hiddenWidth: deterministic, outputIsElu: false);
        _observationNetwork = BuildNetwork(6, LearningRate, feature, observation, hiddenLayers: 3, hiddenWidth: hidden, outputIsElu: false);
        _rewardNetwork = BuildNetwork(7, LearningRate, feature, 1, hiddenLayers: 3, hiddenWidth: hidden, outputIsElu: false);
        _continueNetwork = BuildNetwork(8, LearningRate, feature, 1, hiddenLayers: 3, hiddenWidth: hidden, outputIsElu: false);
        _actorNetwork = BuildNetwork(9, _options.ActorLearningRate, feature, 2 * action, hiddenLayers: 3, hiddenWidth: hidden, outputIsElu: false);
        _valueNetwork = BuildNetwork(10, _options.ValueLearningRate, feature, 1, hiddenLayers: 3, hiddenWidth: hidden, outputIsElu: false);

        Networks.Add(_encoderNetwork);
        Networks.Add(_transitionInputNetwork);
        Networks.Add(_gruGateNetwork);
        Networks.Add(_gruCandidateNetwork);
        Networks.Add(_priorNetwork);
        Networks.Add(_posteriorNetwork);
        Networks.Add(_observationNetwork);
        Networks.Add(_rewardNetwork);
        Networks.Add(_continueNetwork);
        Networks.Add(_actorNetwork);
        Networks.Add(_valueNetwork);

        _replay = new SequenceReplay(options.ReplayBufferSize);

        _beliefDeterministic = new Vector<T>(deterministic);
        _beliefStochastic = new Vector<T>(stochastic);
        _previousAction = new Vector<T>(action);
        _beliefActive = false;
        _lastWorldModelTerms = new WorldModelTerms(NumOps.Zero, NumOps.Zero, NumOps.Zero, NumOps.Zero, NumOps.Zero);
        _lastSequenceLength = 0;
    }

    private static void ValidateOptions(DreamerOptions<T> options)
    {
        RequirePositive(options.DeterministicSize, nameof(options.DeterministicSize));
        RequirePositive(options.StochasticSize, nameof(options.StochasticSize));
        RequirePositive(options.HiddenSize, nameof(options.HiddenSize));
        RequirePositive(options.BatchLength, nameof(options.BatchLength));
        RequirePositive(options.ImaginationHorizon, nameof(options.ImaginationHorizon));
        RequirePositive(options.BatchSize, nameof(options.BatchSize));
        RequirePositive(options.ReplayBufferSize, nameof(options.ReplayBufferSize));
        RequireNonNegative(options.KLScale, nameof(options.KLScale));
        RequireNonNegative(options.FreeNats, nameof(options.FreeNats));
        RequireNonNegative(options.RewardScale, nameof(options.RewardScale));
        RequireNonNegative(options.ContinueScale, nameof(options.ContinueScale));
        RequirePositiveRate(options.ActorLearningRate, nameof(options.ActorLearningRate));
        RequirePositiveRate(options.ValueLearningRate, nameof(options.ValueLearningRate));
        if (double.IsNaN(options.Lambda) || options.Lambda < 0.0 || options.Lambda > 1.0)
        {
            throw new ArgumentOutOfRangeException(nameof(options.Lambda), options.Lambda,
                "Lambda must lie in [0, 1].");
        }
    }

    private static void RequirePositive(int value, string name)
    {
        if (value <= 0)
            throw new ArgumentOutOfRangeException(name, value, $"{name} must be positive.");
    }

    private static void RequirePositiveRate(T value, string name)
    {
        double rate = MathHelper.GetNumericOperations<T>().ToDouble(value);
        if (double.IsNaN(rate) || double.IsInfinity(rate) || rate <= 0.0)
            throw new ArgumentOutOfRangeException(name, rate, $"{name} must be positive and finite.");
    }

    private static void RequireNonNegative(double value, string name)
    {
        if (double.IsNaN(value) || double.IsInfinity(value) || value < 0.0)
            throw new ArgumentOutOfRangeException(name, value, $"{name} must be finite and non-negative.");
    }

    /// <summary>The index of the action model in <c>Networks</c>; every world-model network precedes it.</summary>
    private const int ActorNetworkIndex = 9;

    /// <summary>
    /// The clip a network's default optimizer applies to its own gradient. The action and value models are each a
    /// group of one network, so that clip is their group clip. The world model's networks get none: they are
    /// clipped jointly in <see cref="TrainWorldModel"/>, and clipping them again one by one would be a second,
    /// per-network bound the authors' implementation does not have.
    /// </summary>
    private static double? OwnGradientClip(int networkIndex) => networkIndex >= ActorNetworkIndex ? MaxGradientNorm : null;

    /// <summary>
    /// Builds one dense network: <paramref name="hiddenLayers"/> ELU layers of <paramref name="hiddenWidth"/> units, then
    /// an output layer of <paramref name="outputSize"/> units that is ELU (an embedding) or linear (a distribution head).
    /// </summary>
    private NeuralNetwork<T> BuildNetwork(int networkIndex, T learningRate, int inputSize, int outputSize, int hiddenLayers, int hiddenWidth, bool outputIsElu)
    {
        // Explicit layers handed to the architecture, so NeuralNetwork does not prepend its own default stack.
        var layers = new List<ILayer<T>>();
        for (int i = 0; i < hiddenLayers; i++)
        {
            layers.Add(new DenseLayer<T>(hiddenWidth, (IActivationFunction<T>)new ELUActivation<T>()));
        }
        layers.Add(outputIsElu
            ? new DenseLayer<T>(outputSize, (IActivationFunction<T>)new ELUActivation<T>())
            : new DenseLayer<T>(outputSize, (IActivationFunction<T>)new IdentityActivation<T>()));

        // Dense layers resolve their weights lazily on the first forward. Pin a per-layer seed (distinct per network)
        // so two agents built from the same options start from the same weights; explicitly supplied layers are not
        // wired by NeuralNetwork.InitializeLayers and would otherwise draw from the shared generator.
        int baseSeed = _options.Seed ?? UnseededInitializationBase;
        for (int i = 0; i < layers.Count; i++)
        {
            if (layers[i] is LayerBase<T> layer)
                layer.RandomSeed = unchecked(baseSeed * 31 + networkIndex * 1009 + i);
        }

        var architecture = new NeuralNetworkArchitecture<T>(
            inputType: InputType.OneDimensional,
            taskType: NeuralNetworkTaskType.Regression,
            complexity: NetworkComplexity.Medium,
            inputSize: inputSize,
            outputSize: outputSize,
            layers: layers);
        return new NeuralNetwork<T>(
            architecture,
            _options.OptimizerFactory?.Invoke() ?? CreateNetworkOptimizer(learningRate, OwnGradientClip(networkIndex)),
            new MeanSquaredErrorLoss<T>());
    }

    // =====================================================================================================
    // Acting
    // =====================================================================================================

    /// <summary>
    /// Advances the agent's latent belief with <paramref name="observation"/> and returns an action.
    /// </summary>
    /// <remarks>
    /// Algorithm 1, environment interaction: s_t ~ p(s_t | s_{t-1}, a_{t-1}, o_t) is computed from the history of the
    /// episode, then a_t from the action model. In training the posterior is sampled and the mode action receives
    /// Normal(0, 0.3) exploration noise (Appendix A); otherwise the posterior mean and the mode action are used. The
    /// belief is reset by <see cref="ResetEpisode"/> and when a terminal transition is stored.
    /// </remarks>
    public override Vector<T> SelectAction(Vector<T> observation, bool training = true)
    {
        Guard.NotNull(observation);
        if (observation.Length != _options.ObservationSize)
            throw new ArgumentException($"Observation length must be {_options.ObservationSize}, got {observation.Length}.", nameof(observation));

        if (!_beliefActive)
        {
            _beliefDeterministic = new Vector<T>(_options.DeterministicSize);
            _beliefStochastic = new Vector<T>(_options.StochasticSize);
            _previousAction = new Vector<T>(_options.ActionSize);
        }

        var (action, deterministic, stochastic) = Act(
            _beliefDeterministic, _beliefStochastic, _previousAction, observation, training);

        _beliefDeterministic = deterministic;
        _beliefStochastic = stochastic;
        _previousAction = action;
        _beliefActive = true;
        return action;
    }

    /// <summary>
    /// The deterministic action for <paramref name="input"/> taken as the first observation of an episode.
    /// </summary>
    /// <remarks>
    /// Stateless, so repeated calls agree: the belief starts from the zero state, the posterior mean is used and the
    /// mode action is returned. <see cref="SelectAction"/> is the stateful, history-filtering policy.
    /// </remarks>
    public override Vector<T> Predict(Vector<T> input)
    {
        Guard.NotNull(input);
        if (input.Length != _options.ObservationSize)
            throw new ArgumentException($"Observation length must be {_options.ObservationSize}, got {input.Length}.", nameof(input));

        var (action, _, _) = Act(
            new Vector<T>(_options.DeterministicSize),
            new Vector<T>(_options.StochasticSize),
            new Vector<T>(_options.ActionSize),
            input,
            explore: false);
        return action;
    }

    private (Vector<T> Action, Vector<T> Deterministic, Vector<T> Stochastic) Act(
        Vector<T> deterministic, Vector<T> stochastic, Vector<T> previousAction, Vector<T> observation, bool explore)
    {
        Func<NeuralNetwork<T>, Tensor<T>, Tensor<T>> infer = (network, x) => network.Predict(x);

        var (h, _, _) = ImagineStep(Row(stochastic), Row(previousAction), Row(deterministic), infer);
        var embedding = infer(_encoderNetwork, Row(observation));
        var (posteriorMean, posteriorStd) = Posterior(h, embedding, infer);
        var z = explore
            ? Engine.TensorAdd(posteriorMean, Engine.TensorMultiply(posteriorStd, StandardNormalTensor(1, _options.StochasticSize)))
            : posteriorMean;

        var actorRaw = infer(_actorNetwork, Engine.TensorConcatenate([h, z], axis: 1));
        var (actionMean, _) = ActionDistribution(actorRaw);
        // The mode action: tanh of the Gaussian's mean (Equation 3 with epsilon = 0).
        var mode = Engine.TensorTanh(actionMean).ToVector();

        var action = new Vector<T>(_options.ActionSize);
        T low = NumOps.FromDouble(-1.0);
        for (int i = 0; i < action.Length; i++)
        {
            T value = mode[i];
            if (explore)
            {
                value = NumOps.Add(value, NumOps.Multiply(NumOps.FromDouble(ExplorationStd), StandardNormal()));
                value = MathHelper.Clamp<T>(value, low, NumOps.One);
            }
            action[i] = value;
        }

        return (action, h.ToVector(), z.ToVector());
    }

    /// <inheritdoc />
    public override void StoreExperience(Vector<T> observation, Vector<T> action, T reward, Vector<T> nextObservation, bool done)
    {
        // Validate transition shapes at this public ingestion boundary so a malformed experience
        // can't enter the replay data and later cause shape failures deep in Train().
        Guard.NotNull(observation);
        Guard.NotNull(action);
        Guard.NotNull(nextObservation);
        if (observation.Length != _options.ObservationSize)
            throw new ArgumentException($"Observation length must be {_options.ObservationSize}, got {observation.Length}.", nameof(observation));
        if (nextObservation.Length != _options.ObservationSize)
            throw new ArgumentException($"Next observation length must be {_options.ObservationSize}, got {nextObservation.Length}.", nameof(nextObservation));
        if (action.Length != _options.ActionSize)
            throw new ArgumentException($"Action length must be {_options.ActionSize}, got {action.Length}.", nameof(action));

        _replay.Add(new Experience<T, Vector<T>, Vector<T>>(observation, action, reward, nextObservation, done));
        if (done) _beliefActive = false;
    }

    // =====================================================================================================
    // Training (Algorithm 1: dynamics learning, then behaviour learning)
    // =====================================================================================================

    /// <inheritdoc />
    public override T Train()
    {
        if (_replay.Count < _options.BatchSize)
        {
            return NumOps.Zero;
        }

        // Use the rate the BASE class already resolved, exactly as this type's constructor does for
        // LearningRate and for exactly the same reason: `DiscountFactor` is declared `T?`, but for a
        // value-type T -- float and double, i.e. every real use -- that is NOT Nullable<T>, so an
        // unconfigured option reads as default(T) == 0 while `is not null` is still TRUE. Re-deriving gamma
        // from the raw option therefore evaluated to gamma = 0 for every default-constructed DreamerAgent,
        // and the 0.99 fallback never ran.
        //
        // Zero gamma made the entire behaviour-learning objective identically zero: every imagined return
        // collapsed to its first reward and the action model received no gradient through the dynamics --
        // measured as 0 of 12 reachable tensors in each of the actor, dynamics and value heads, and 203408
        // published gradients with max |g| = 0.
        //
        // The base resolves this properly, treating default(T) as "not configured": see
        // ReinforcementLearningAgentBase, which checks `is null || == zero` before falling back.
        T gamma = DiscountFactor;

        int batch = _options.BatchSize;
        int length = _replay.LongestSequence(_options.BatchLength);
        var starts = _replay.SampleStarts(length, batch, Random);
        _lastSequenceLength = length;

        var data = BuildSequenceBatch(starts, length, gamma);
        var (worldModelLoss, start) = TrainWorldModel(data);
        var (actorLoss, valueLoss) = TrainBehaviour(start, gamma);

        _updateCount++;
        return NumOps.Add(worldModelLoss, NumOps.Add(actorLoss, valueLoss));
    }

    /// <summary>
    /// The B sequences of one step, time-major: observation row t*B + b is o_t of sequence b (t = 0..L), and the
    /// reward / discount rows (t-1)*B + b belong to o_t for t = 1..L.
    /// </summary>
    private SequenceBatch BuildSequenceBatch(int[] starts, int length, T gamma)
    {
        int batch = starts.Length;
        int obsSize = _options.ObservationSize;
        int actionSize = _options.ActionSize;

        var observations = new Tensor<T>([(length + 1) * batch, obsSize]);
        var actions = new Tensor<T>[length];
        var rewards = new Tensor<T>([length * batch, 1]);
        var discounts = new Tensor<T>([length * batch, 1]);
        for (int t = 0; t < length; t++) actions[t] = new Tensor<T>([batch, actionSize]);

        for (int b = 0; b < batch; b++)
        {
            var first = _replay.At(starts[b]);
            for (int j = 0; j < obsSize; j++) observations[b, j] = first.State[j];

            for (int t = 0; t < length; t++)
            {
                var transition = _replay.At(starts[b] + t);
                int nextRow = (t + 1) * batch + b;
                for (int j = 0; j < obsSize; j++) observations[nextRow, j] = transition.NextState[j];
                for (int j = 0; j < actionSize; j++) actions[t][b, j] = transition.Action[j];

                int row = t * batch + b;
                rewards[row, 0] = transition.Reward;
                // "a binary classifier that is trained towards the soft labels of 0 and gamma" (Appendix A).
                discounts[row, 0] = transition.Done ? NumOps.Zero : gamma;
            }
        }

        return new SequenceBatch(observations, actions, rewards, discounts, batch, length);
    }

    /// <summary>
    /// One world-model update on Equation 10, every RSSM and head network stepped from one tape.
    /// </summary>
    /// <returns>The world-model loss and the detached posterior states the behaviour step imagines from.</returns>
    private (T Loss, ImaginationStart Start) TrainWorldModel(SequenceBatch data)
    {
        int batch = data.Batch;
        int length = data.Length;
        int deterministicSize = _options.DeterministicSize;
        int stochasticSize = _options.StochasticSize;
        bool predictDiscount = _options.ContinueScale > 0.0;

        Func<NeuralNetwork<T>, Tensor<T>, Tensor<T>> train = (network, x) => network.ForwardForTraining(x);

        using var tape = new GradientTape<T>();

        var embeddings = train(_encoderNetwork, data.Observations);
        var h = new Tensor<T>([batch, deterministicSize]);
        var z = new Tensor<T>([batch, stochasticSize]);
        var features = new Tensor<T>[length + 1];
        var deterministicStates = new Tensor<T>[length + 1];
        var stochasticStates = new Tensor<T>[length + 1];
        var divergences = new Tensor<T>[length + 1];

        for (int t = 0; t <= length; t++)
        {
            // The first step of every sequence starts from the zero state and a zero previous action.
            var previousAction = t == 0 ? new Tensor<T>([batch, _options.ActionSize]) : data.Actions[t - 1];
            var (hNext, priorMean, priorStd) = ImagineStep(z, previousAction, h, train);
            var embedding = Engine.TensorNarrow(embeddings, 0, t * batch, batch);
            var (posteriorMean, posteriorStd) = Posterior(hNext, embedding, train);

            // Reparameterised posterior sample, so reconstruction gradients reach the representation model.
            z = Engine.TensorAdd(posteriorMean, Engine.TensorMultiply(posteriorStd, StandardNormalTensor(batch, stochasticSize)));
            h = hNext;

            features[t] = Engine.TensorConcatenate([h, z], axis: 1);
            deterministicStates[t] = h;
            stochasticStates[t] = z;
            divergences[t] = GaussianKl(posteriorMean, posteriorStd, priorMean, priorStd);
        }

        var allFeatures = Engine.TensorConcatenate(features, axis: 0);           // [(L+1)B, D+S]
        var nextFeatures = Engine.TensorNarrow(allFeatures, 0, batch, length * batch); // states of o_1..o_L

        T half = NumOps.FromDouble(0.5);

        // J_O: Gaussian observation log-likelihood with unit variance, summed over dimensions.
        var reconstructionError = Engine.TensorSubtract(train(_observationNetwork, allFeatures), data.Observations);
        var reconstruction = Engine.TensorMultiplyScalar(
            MeanAll(Engine.ReduceSum(Engine.TensorSquare(reconstructionError), [1], keepDims: false)), half);

        // J_R: Gaussian reward log-likelihood with unit variance.
        var rewardError = Engine.TensorSubtract(train(_rewardNetwork, nextFeatures), data.Rewards);
        var rewardLoss = Engine.TensorMultiplyScalar(MeanAll(Engine.TensorSquare(rewardError)), half);

        // J_D: KL(posterior || prior) averaged over the batch and the sequence, clipped below the free nats.
        var divergence = MeanAll(Engine.TensorConcatenate(divergences, axis: 0));
        var clippedDivergence = Engine.TensorMax(divergence, NumOps.FromDouble(_options.FreeNats));

        var total = Engine.TensorAdd(
            reconstruction,
            Engine.TensorAdd(
                Engine.TensorMultiplyScalar(rewardLoss, NumOps.FromDouble(_options.RewardScale)),
                Engine.TensorMultiplyScalar(clippedDivergence, NumOps.FromDouble(_options.KLScale))));

        T continueValue = NumOps.Zero;
        if (predictDiscount)
        {
            // Bernoulli log-likelihood of the soft label y in {0, gamma} under logit l:
            // -[y log sigmoid(l) + (1 - y) log(1 - sigmoid(l))] = softplus(l) - y * l.
            var logits = train(_continueNetwork, nextFeatures);
            var continueLoss = MeanAll(Engine.TensorSubtract(Engine.Softplus(logits), Engine.TensorMultiply(data.Discounts, logits)));
            total = Engine.TensorAdd(total, Engine.TensorMultiplyScalar(continueLoss, NumOps.FromDouble(_options.ContinueScale)));
            continueValue = continueLoss[0];
        }

        var members = new List<(NeuralNetworkBase<T> Network, IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? Optimizer)>
        {
            (_encoderNetwork, null),
            (_transitionInputNetwork, null),
            (_gruGateNetwork, null),
            (_gruCandidateNetwork, null),
            (_priorNetwork, null),
            (_posteriorNetwork, null),
            (_observationNetwork, null),
            (_rewardNetwork, null),
        };
        if (predictDiscount) members.Add((_continueNetwork, null));

        _lastWorldModelTerms = new WorldModelTerms(reconstruction[0], rewardLoss[0], divergence[0], continueValue, total[0]);
        // The world model is one group under one global-norm clip (Appendix A), whatever optimizer each network has.
        T loss = NeuralNetworkBase<T>.StepOnTape(tape, total, members, MaxGradientNorm);

        // Imagination starts from every posterior state, with gradients stopped (Section 3: "Imagined trajectories
        // start at the true model states s_t of observation sequences drawn from the agent's past experience").
        var start = new ImaginationStart(
            Detach(Engine.TensorConcatenate(deterministicStates, axis: 0)),
            Detach(Engine.TensorConcatenate(stochasticStates, axis: 0)));
        return (loss, start);
    }

    /// <summary>
    /// One action-model update (Equation 7) and one value-model update (Equation 8) on trajectories imagined
    /// from <paramref name="start"/>; the world model is not updated here.
    /// </summary>
    private (T ActorLoss, T ValueLoss) TrainBehaviour(ImaginationStart start, T gamma)
    {
        int rows = start.Deterministic.Shape[0];
        int horizon = _options.ImaginationHorizon;
        int stochasticSize = _options.StochasticSize;
        bool predictDiscount = _options.ContinueScale > 0.0;
        T lambda = NumOps.FromDouble(_options.Lambda);
        var startFeatures = Engine.TensorConcatenate([start.Deterministic, start.Stochastic], axis: 1);

        Tensor<T>? valueStates = null;
        Tensor<T>? valueTargets = null;
        Tensor<T>? valueWeights = null;

        Func<NeuralNetwork<T>, Tensor<T>, Tensor<T>> train = (network, x) => network.ForwardForTraining(x);

        // TrainWithCustomLoss differentiates only the action model's tensors, so the transition, reward, discount
        // and value models supply analytic gradients through the imagined trajectory without being updated by it.
        // They are forwarded with ForwardForTraining: Predict runs in a NoGradScope and would detach them.
        T actorLoss = _actorNetwork.TrainWithCustomLoss(startFeatures, startActorOutput =>
        {
            var h = start.Deterministic;
            var z = start.Stochastic;
            var feature = startFeatures;
            var trajectory = new Tensor<T>[horizon + 1];
            trajectory[0] = feature;

            for (int k = 0; k < horizon; k++)
            {
                // The action model reads the state with its gradient stopped, as the authors' implementation does;
                // the action's own gradient still flows into the transition model and on to every later state.
                var actorRaw = k == 0 ? startActorOutput : _actorNetwork.ForwardForTraining(Engine.StopGradient(feature));
                var (mean, std) = ActionDistribution(actorRaw);
                // Equation 3: a = tanh(mu + sigma * epsilon), epsilon ~ Normal(0, I).
                var action = Engine.TensorTanh(Engine.TensorAdd(mean, Engine.TensorMultiply(std, StandardNormalTensor(rows, _options.ActionSize))));

                var (hNext, priorMean, priorStd) = ImagineStep(z, action, h, train);
                z = Engine.TensorAdd(priorMean, Engine.TensorMultiply(priorStd, StandardNormalTensor(rows, stochasticSize)));
                h = hNext;
                feature = Engine.TensorConcatenate([h, z], axis: 1);
                trajectory[k + 1] = feature;
            }

            var states = Engine.TensorConcatenate(trajectory, axis: 0);   // [(H+1)N, D+S], time-major
            var rewards = SplitRows(train(_rewardNetwork, states), horizon + 1, rows);
            var values = SplitRows(train(_valueNetwork, states), horizon + 1, rows);
            Tensor<T>[] discounts;
            if (predictDiscount)
            {
                // The discount model predicts gamma * P(continue): its labels are 0 and gamma.
                discounts = SplitRows(Engine.TensorSigmoid(train(_continueNetwork, states)), horizon + 1, rows);
            }
            else
            {
                discounts = new Tensor<T>[horizon + 1];
                for (int k = 0; k <= horizon; k++) discounts[k] = Filled(rows, 1, gamma);
            }

            var returns = ComputeLambdaReturns(rewards, values, discounts, lambda);
            var weights = CumulativeDiscountWeights(discounts);

            // Equation 7 over tau = t..t+H, each step weighted by the cumulative predicted discount (Section 3).
            var weighted = new Tensor<T>[horizon + 1];
            for (int k = 0; k <= horizon; k++) weighted[k] = Engine.TensorMultiply(weights[k], returns[k]);
            var objective = MeanAll(Engine.TensorConcatenate(weighted, axis: 0));

            // Equation 8 regresses v(s_tau) onto V_lambda(s_tau) for tau = t..t+H-1; at tau = t+H the target is
            // v(s_{t+H}) itself, a term that is identically zero once the target's gradient is stopped.
            valueStates = Detach(Engine.TensorNarrow(states, 0, 0, horizon * rows));
            valueTargets = Detach(Engine.TensorConcatenate(returns.Take(horizon).ToArray(), axis: 0));
            valueWeights = Detach(Engine.TensorConcatenate(weights.Take(horizon).ToArray(), axis: 0));

            // Ascending the objective is minimising its negation.
            return Engine.TensorNegate(objective);
        });

        if (valueStates is null || valueTargets is null || valueWeights is null)
        {
            throw new InvalidOperationException("The action-model step did not produce value targets.");
        }

        var targets = valueTargets;
        var targetWeights = valueWeights;
        T half = NumOps.FromDouble(0.5);
        T valueLoss = _valueNetwork.TrainWithCustomLoss(valueStates, predicted =>
        {
            // Equation 8: 1/2 || v(s_tau) - V_lambda(s_tau) ||^2, weighted by the cumulative predicted discount.
            var error = Engine.TensorSubtract(predicted, targets);
            var weightedSquare = Engine.TensorMultiply(targetWeights, Engine.TensorSquare(error));
            return Engine.TensorMultiplyScalar(MeanAll(weightedSquare), half);
        });

        return (actorLoss, valueLoss);
    }

    // =====================================================================================================
    // RSSM and distribution helpers
    // =====================================================================================================

    /// <summary>
    /// The RSSM transition: h_t = GRU(h_{t-1}, x_t) with x_t = dense([z_{t-1}, a_{t-1}]), and the prior
    /// p(z_t | h_t) as a diagonal Gaussian.
    /// </summary>
    /// <remarks>
    /// The GRU is Cho et al. (2014): r = sigmoid(W_r [x, h]), u = sigmoid(W_u [x, h]),
    /// c = tanh(W_c [x, r * h]), h' = (1 - u) * h + u * c.
    /// </remarks>
    private (Tensor<T> Deterministic, Tensor<T> PriorMean, Tensor<T> PriorStd) ImagineStep(
        Tensor<T> stochastic, Tensor<T> action, Tensor<T> deterministic, Func<NeuralNetwork<T>, Tensor<T>, Tensor<T>> run)
    {
        int size = _options.DeterministicSize;
        var x = run(_transitionInputNetwork, Engine.TensorConcatenate([stochastic, action], axis: 1));
        var gates = run(_gruGateNetwork, Engine.TensorConcatenate([x, deterministic], axis: 1));
        var reset = Engine.TensorSigmoid(Engine.TensorNarrow(gates, 1, 0, size));
        var update = Engine.TensorSigmoid(Engine.TensorNarrow(gates, 1, size, size));
        var candidate = Engine.TensorTanh(run(_gruCandidateNetwork,
            Engine.TensorConcatenate([x, Engine.TensorMultiply(reset, deterministic)], axis: 1)));
        var next = Engine.TensorAdd(deterministic, Engine.TensorMultiply(update, Engine.TensorSubtract(candidate, deterministic)));

        var (mean, std) = StateDistribution(run(_priorNetwork, next));
        return (next, mean, std);
    }

    /// <summary>The representation model q(z_t | h_t, e_t) as a diagonal Gaussian.</summary>
    private (Tensor<T> Mean, Tensor<T> Std) Posterior(
        Tensor<T> deterministic, Tensor<T> embedding, Func<NeuralNetwork<T>, Tensor<T>, Tensor<T>> run)
        => StateDistribution(run(_posteriorNetwork, Engine.TensorConcatenate([deterministic, embedding], axis: 1)));

    /// <summary>Splits a [N, 2S] head into the mean and the standard deviation softplus(raw) + 0.1.</summary>
    private (Tensor<T> Mean, Tensor<T> Std) StateDistribution(Tensor<T> raw)
    {
        int size = _options.StochasticSize;
        var mean = Engine.TensorNarrow(raw, 1, 0, size);
        var std = Engine.TensorAddScalar(Engine.Softplus(Engine.TensorNarrow(raw, 1, size, size)), NumOps.FromDouble(MinimumStateStd));
        return (mean, std);
    }

    /// <summary>
    /// The action model's Gaussian before the tanh (Appendix A): mean 5 * tanh(mu / 5) and standard deviation
    /// softplus(raw + c) + 1e-4, with c chosen so the initial standard deviation is 5.
    /// </summary>
    private (Tensor<T> Mean, Tensor<T> Std) ActionDistribution(Tensor<T> raw)
    {
        int size = _options.ActionSize;
        T scale = NumOps.FromDouble(ActionMeanScale);
        var mean = Engine.TensorMultiplyScalar(
            Engine.TensorTanh(Engine.TensorMultiplyScalar(Engine.TensorNarrow(raw, 1, 0, size), NumOps.FromDouble(1.0 / ActionMeanScale))),
            scale);
        // softplus^{-1}(5) = ln(e^5 - 1).
        T stdOffset = NumOps.FromDouble(Math.Log(Math.Exp(ActionInitialStd) - 1.0));
        var std = Engine.TensorAddScalar(
            Engine.Softplus(Engine.TensorAddScalar(Engine.TensorNarrow(raw, 1, size, size), stdOffset)),
            NumOps.FromDouble(MinimumActionStd));
        return (mean, std);
    }

    /// <summary>
    /// KL(q || p) between diagonal Gaussians, summed over dimensions: one value per row.
    /// </summary>
    /// <remarks>
    /// sum_i [ ln(sigma_p / sigma_q) + (sigma_q^2 + (mu_q - mu_p)^2) / (2 sigma_p^2) - 1/2 ].
    /// </remarks>
    internal Tensor<T> GaussianKl(Tensor<T> posteriorMean, Tensor<T> posteriorStd, Tensor<T> priorMean, Tensor<T> priorStd)
    {
        var meanGap = Engine.TensorSubtract(posteriorMean, priorMean);
        var numerator = Engine.TensorAdd(Engine.TensorSquare(posteriorStd), Engine.TensorSquare(meanGap));
        var ratio = Engine.TensorDivide(numerator, Engine.TensorMultiplyScalar(Engine.TensorSquare(priorStd), NumOps.FromDouble(2.0)));
        var logRatio = Engine.TensorSubtract(Engine.TensorLog(priorStd), Engine.TensorLog(posteriorStd));
        var perDimension = Engine.TensorAddScalar(Engine.TensorAdd(logRatio, ratio), NumOps.FromDouble(-0.5));
        return Engine.ReduceSum(perDimension, [1], keepDims: false);
    }

    /// <summary>
    /// V-lambda along an imagined trajectory s_t..s_{t+H} (Equation 6), by its recursive form:
    /// V_lambda(s_{t+H}) = v(s_{t+H}) and
    /// V_lambda(s_tau) = r_tau + d_tau * ((1 - lambda) v(s_{tau+1}) + lambda V_lambda(s_{tau+1})),
    /// where d_tau is the (predicted) discount gamma.
    /// </summary>
    /// <param name="rewards">r_tau for tau = t..t+H, each [N, 1].</param>
    /// <param name="values">v(s_tau) for tau = t..t+H, each [N, 1].</param>
    /// <param name="discounts">d_tau for tau = t..t+H, each [N, 1].</param>
    /// <param name="lambda">The lambda of Equation 6.</param>
    /// <returns>V_lambda(s_tau) for tau = t..t+H.</returns>
    internal Tensor<T>[] ComputeLambdaReturns(Tensor<T>[] rewards, Tensor<T>[] values, Tensor<T>[] discounts, T lambda)
    {
        int last = values.Length - 1;
        var returns = new Tensor<T>[values.Length];
        returns[last] = values[last];
        T oneMinusLambda = NumOps.Subtract(NumOps.One, lambda);
        for (int k = last - 1; k >= 0; k--)
        {
            var blend = Engine.TensorAdd(
                Engine.TensorMultiplyScalar(values[k + 1], oneMinusLambda),
                Engine.TensorMultiplyScalar(returns[k + 1], lambda));
            returns[k] = Engine.TensorAdd(rewards[k], Engine.TensorMultiply(discounts[k], blend));
        }
        return returns;
    }

    /// <summary>
    /// The weight of each imagined step: the cumulative product of the predicted discounts before it, with the
    /// gradient stopped (Section 3; w_t = 1).
    /// </summary>
    private Tensor<T>[] CumulativeDiscountWeights(Tensor<T>[] discounts)
    {
        int rows = discounts[0].Shape[0];
        var weights = new Tensor<T>[discounts.Length];
        var running = new T[rows];
        for (int i = 0; i < rows; i++) running[i] = NumOps.One;
        for (int k = 0; k < discounts.Length; k++)
        {
            weights[k] = new Tensor<T>((T[])running.Clone(), [rows, 1]);
            var discount = discounts[k].ToVector();
            for (int i = 0; i < rows; i++) running[i] = NumOps.Multiply(running[i], discount[i]);
        }
        return weights;
    }

    private Tensor<T>[] SplitRows(Tensor<T> stacked, int parts, int rows)
    {
        var result = new Tensor<T>[parts];
        for (int k = 0; k < parts; k++) result[k] = Engine.TensorNarrow(stacked, 0, k * rows, rows);
        return result;
    }

    private Tensor<T> MeanAll(Tensor<T> tensor)
    {
        var axes = new int[tensor.Shape.Length];
        for (int i = 0; i < axes.Length; i++) axes[i] = i;
        return Engine.ReduceMean(tensor, axes, keepDims: false);
    }

    /// <summary>A value copy with no tape history, so later steps cannot reach back through it.</summary>
    private static Tensor<T> Detach(Tensor<T> tensor)
        => new Tensor<T>(tensor.ToVector().ToArray(), tensor.Shape.ToArray());

    private static Tensor<T> Row(Vector<T> vector)
        => new Tensor<T>(vector.ToArray(), [1, vector.Length]);

    private static Tensor<T> Filled(int rows, int columns, T value)
    {
        var data = new T[rows * columns];
        for (int i = 0; i < data.Length; i++) data[i] = value;
        return new Tensor<T>(data, [rows, columns]);
    }

    private Tensor<T> StandardNormalTensor(int rows, int columns)
    {
        var data = new T[rows * columns];
        for (int i = 0; i < data.Length; i++) data[i] = StandardNormal();
        return new Tensor<T>(data, [rows, columns]);
    }

    /// <summary>A standard normal draw from the agent's (seedable) generator, by the Box-Muller transform.</summary>
    private T StandardNormal()
    {
        double u1 = 1.0 - Random.NextDouble();
        double u2 = Random.NextDouble();
        return NumOps.FromDouble(Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2));
    }

    // =====================================================================================================
    // Diagnostics
    // =====================================================================================================

    /// <summary>The world-model terms of the most recent training step (Equation 10, before weighting).</summary>
    internal WorldModelTerms LastWorldModelTerms => _lastWorldModelTerms;

    /// <summary>The sequence length L the most recent training step unrolled the world model over.</summary>
    internal int LastSequenceLength => _lastSequenceLength;

    /// <summary>The replay data, in insertion order.</summary>
    internal SequenceReplay Replay => _replay;

    /// <inheritdoc />
    public override Dictionary<string, T> GetMetrics()
    {
        return new Dictionary<string, T>
        {
            ["updates"] = NumOps.FromDouble(_updateCount),
            ["buffer_size"] = NumOps.FromDouble(_replay.Count),
            ["reconstruction_loss"] = _lastWorldModelTerms.Reconstruction,
            ["reward_loss"] = _lastWorldModelTerms.Reward,
            ["kl_divergence"] = _lastWorldModelTerms.Divergence,
            ["continue_loss"] = _lastWorldModelTerms.Continue,
        };
    }

    /// <inheritdoc />
    public override void ResetEpisode()
    {
        // A new episode starts from the zero model state and a zero previous action.
        _beliefActive = false;
    }

    public Task<Vector<T>> PredictAsync(Vector<T> input)
    {
        return Task.FromResult(Predict(input));
    }

    public Task TrainAsync()
    {
        Train();
        return Task.CompletedTask;
    }

    /// <inheritdoc />
    public override ModelMetadata<T> GetModelMetadata()
    {
        var metadata = new ModelMetadata<T>
        {
            Name = "Dreamer",
            Description = "Dreamer model-based RL agent: learns a recurrent state-space world model (GRU " +
                "deterministic state, Gaussian stochastic state, observation, reward and discount models) and " +
                "an actor and value model trained on lambda-returns in latent imagination " +
                "(Hafner et al. 2020, 'Dream to Control').",
            FeatureCount = _options.ObservationSize,
            Complexity = ParameterCount,
        };

        metadata.SetProperty("ObservationSize", _options.ObservationSize);
        metadata.SetProperty("ActionSize", _options.ActionSize);
        metadata.SetProperty("DeterministicSize", _options.DeterministicSize);
        metadata.SetProperty("StochasticSize", _options.StochasticSize);
        metadata.SetProperty("HiddenSize", _options.HiddenSize);
        metadata.SetProperty("BatchLength", _options.BatchLength);
        metadata.SetProperty("ImaginationHorizon", _options.ImaginationHorizon);
        metadata.SetProperty("UpdateCount", _updateCount);

        return metadata;
    }

    /// <inheritdoc />
    public override int FeatureCount => _options.ObservationSize;

    /// <summary>The world-model terms of one training step.</summary>
    /// <param name="Reconstruction">-J_O: the observation negative log-likelihood (up to a constant).</param>
    /// <param name="Reward">-J_R: the reward negative log-likelihood (up to a constant), before RewardScale.</param>
    /// <param name="Divergence">The mean KL(posterior || prior) before free nats and KLScale.</param>
    /// <param name="Continue">The discount-predictor negative log-likelihood before ContinueScale (0 when disabled).</param>
    /// <param name="Total">The weighted objective the step differentiated.</param>
    internal readonly record struct WorldModelTerms(T Reconstruction, T Reward, T Divergence, T Continue, T Total);

    private readonly record struct ImaginationStart(Tensor<T> Deterministic, Tensor<T> Stochastic);

    private sealed record SequenceBatch(
        Tensor<T> Observations, Tensor<T>[] Actions, Tensor<T> Rewards, Tensor<T> Discounts, int Batch, int Length);

    /// <summary>
    /// The agent's replay data: transitions in insertion order, from which training draws runs of consecutive
    /// transitions that stay inside one episode (Algorithm 1: "Draw B data sequences {(a_t, o_t, r_t)}_{t=k}^{k+L}").
    /// </summary>
    internal sealed class SequenceReplay
    {
        private readonly Experience<T, Vector<T>, Vector<T>>[] _items;
        private int _first;
        private int _count;

        internal SequenceReplay(int capacity)
        {
            if (capacity <= 0) throw new ArgumentOutOfRangeException(nameof(capacity), capacity, "Capacity must be positive.");
            _items = new Experience<T, Vector<T>, Vector<T>>[capacity];
        }

        /// <summary>The number of stored transitions.</summary>
        internal int Count => _count;

        /// <summary>Appends a transition, overwriting the oldest once full.</summary>
        internal void Add(Experience<T, Vector<T>, Vector<T>> transition)
        {
            if (_count < _items.Length)
            {
                _items[(_first + _count) % _items.Length] = transition;
                _count++;
            }
            else
            {
                _items[_first] = transition;
                _first = (_first + 1) % _items.Length;
            }
        }

        /// <summary>The transition at <paramref name="index"/> in insertion order (0 is the oldest kept).</summary>
        internal Experience<T, Vector<T>, Vector<T>> At(int index) => _items[(_first + index) % _items.Length];

        /// <summary>
        /// For each index, how many transitions from it stay in one episode: up to and including the next terminal
        /// transition, or to the newest one.
        /// </summary>
        private int[] RunLengths()
        {
            var runs = new int[_count];
            for (int i = _count - 1; i >= 0; i--)
            {
                runs[i] = At(i).Done || i == _count - 1 ? 1 : 1 + runs[i + 1];
            }
            return runs;
        }

        /// <summary>min(<paramref name="maximum"/>, the longest within-episode run stored).</summary>
        internal int LongestSequence(int maximum)
        {
            int longest = 0;
            foreach (int run in RunLengths()) longest = Math.Max(longest, run);
            return Math.Min(maximum, longest);
        }

        /// <summary>
        /// Draws <paramref name="count"/> start indices, uniformly with replacement, of runs of
        /// <paramref name="length"/> consecutive transitions that do not cross an episode boundary (only the last
        /// transition of a run may be terminal).
        /// </summary>
        internal int[] SampleStarts(int length, int count, Random random)
        {
            var runs = RunLengths();
            var valid = new List<int>();
            for (int i = 0; i < runs.Length; i++)
            {
                if (runs[i] >= length) valid.Add(i);
            }

            if (valid.Count == 0)
            {
                throw new InvalidOperationException(
                    $"No stored run of {length} consecutive transitions stays inside one episode.");
            }

            var starts = new int[count];
            for (int i = 0; i < count; i++) starts[i] = valid[random.Next(valid.Count)];
            return starts;
        }
    }
}
