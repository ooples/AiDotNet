using AiDotNet.LearningRateSchedulers;
using AiDotNet.ActivationFunctions;
using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.Models;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Validation;

namespace AiDotNet.ReinforcementLearning.Agents.DecisionTransformer;

/// <summary>
/// Decision Transformer agent for offline reinforcement learning.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Decision Transformer treats RL as sequence modeling, using transformer architecture
/// to predict actions conditioned on desired returns-to-go.
/// </para>
/// <para>
/// The model follows Chen et al. 2021, Sec. 3. A trajectory is the token sequence
/// (R̂_1, s_1, a_1, R̂_2, s_2, a_2, ...), where R̂_t = Σ_{t'≥t} r_{t'} is the return-to-go. The last K timesteps
/// (3K tokens) are embedded by a separate linear map per modality, a learned embedding of the episode timestep is
/// added to each token, and the result is layer-normalised and passed through a GPT-2 style causal transformer
/// (pre-LayerNorm blocks of causal multi-head self-attention and a GELU feed-forward network, with dropout). The
/// action a_t is predicted by a linear head (tanh for continuous actions) from the output at the state token s_t,
/// which by causality has seen R̂_t, s_t and the earlier timesteps but not a_t itself. Training is supervised:
/// segments of up to K timesteps are sampled from an offline trajectory buffer and the predicted actions are
/// regressed onto the dataset actions (mean-squared error, or cross-entropy for discrete actions). At evaluation the
/// first return-to-go is the target return and each later one is the previous one minus the reward received
/// (Algorithm 1).
/// </para>
/// <para>
/// Implementation notes. The three modality embeddings and the timestep embedding are realised as one bias-free
/// linear layer over a block-structured token feature vector: a one-hot modality indicator (whose weights are the
/// per-modality biases), the return-to-go, the state, the action, and a one-hot timestep (whose weights are the
/// timestep embedding table). Each token fills only its own modality's block, so the layer computes exactly
/// W_m x + b_m + E[t] for the token's modality m. Segments shorter than K are padded on the RIGHT rather than on the
/// left as the reference implementation does; under the causal mask no real token can attend to a later padding
/// token, so this is equivalent to the reference's attention-masked left padding, and padded positions are excluded
/// from the loss. The library's transformer block applies dropout to sublayer outputs and the embedding; GPT-2's
/// additional dropout on the attention probabilities is not applied.
/// </para>
/// <para><b>For Beginners:</b>
/// Instead of learning "what's the best action", Decision Transformer learns
/// "what action was taken when the outcome was X". At test time, you specify
/// the desired outcome, and it generates the action sequence.
///
/// Key innovation:
/// - **Return Conditioning**: Specify target return, get actions that achieve it
/// - **Sequence Modeling**: Uses transformers like GPT for temporal dependencies
/// - **No RL Updates**: Just supervised learning on (return, state, action) sequences
/// - **Offline-First**: Designed for learning from fixed datasets
///
/// Think of it as: "Show me examples of successful games, and I'll learn to
/// generate moves that lead to that level of success."
///
/// Famous for: Berkeley/Meta research simplifying RL to sequence modeling
/// </para>
/// </remarks>
/// <example>
/// <code>
/// // Create a Decision Transformer for offline RL via sequence modeling
/// var options = new DecisionTransformerOptions&lt;double&gt; { StateSize = 4, ActionSize = 2, ContextLength = 20 };
/// var agent = new DecisionTransformerAgent&lt;double&gt;(options);
///
/// // Select an action conditioned on the desired return
/// var state = new Vector&lt;double&gt;(new double[] { 0.5, -0.3, 1.0, 0.2 });
/// var action = agent.SelectAction(state);
/// </code>
/// </example>
[ModelDomain(ModelDomain.MachineLearning)]
[ModelCategory(ModelCategory.ReinforcementLearningAgent)]
[ModelCategory(ModelCategory.NeuralNetwork)]
[ModelCategory(ModelCategory.Transformer)]
[ModelTask(ModelTask.Classification)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("Decision Transformer: Reinforcement Learning via Sequence Modeling",
    "https://arxiv.org/abs/2106.01345",
    Year = 2021,
    Authors = "Chen, L., Lu, K., Rajeswaran, A., Lee, K., Grover, A., Laskin, M., Abbeel, P., Srinivas, A., & Mordatch, I.")]
[PaperOptimizer(OptimizerKind.AdamW, LearningRate = 1e-4, WeightDecay = 1e-4,
                ReferenceBatchSize = 64, WarmupSteps = 100000,
                Schedule = LearningRateSchedulerType.LinearWarmup,
                Source = "Chen et al. 2021, Appendix A: AdamW following PyTorch defaults, trained for "
                        + "10^5 gradient steps with a learning rate of 1e-4, a weight decay of 1e-4, a "
                        + "batch size of 64 and linear warmup over the first 10^5 training steps. The "
                        + "exponents render as a bare 105 in extraction.")]
public partial class DecisionTransformerAgent<T> : DeepReinforcementLearningAgentBase<T>, IGradientComputable<T, Vector<T>, Vector<T>>
{

    /// <inheritdoc />
    /// <remarks>The same components, in the same order, that the hand-written
    /// GetParameters concatenated -- that order is the serialization order.</remarks>
    protected override void RegisterComponents()
    {
        RegisterParameterComponent(_transformerNetwork);
    }

    /// <summary>Tokens per timestep: return-to-go, state, action (Chen et al. 2021, Sec. 3).</summary>
    private const int TokensPerStep = 3;

    /// <summary>Position of the return-to-go token within a timestep.</summary>
    private const int ReturnToken = 0;

    /// <summary>Position of the state token within a timestep; the action is read from its output.</summary>
    private const int StateToken = 1;

    /// <summary>Position of the action token within a timestep.</summary>
    private const int ActionToken = 2;

    private readonly DecisionTransformerOptions<T> _options;

    private INeuralNetwork<T> _transformerNetwork;

    /// <summary>The offline dataset: whole trajectories, oldest first, holding at most BufferSize transitions.</summary>
    private readonly List<StoredTrajectory> _trajectories = new();

    /// <summary>Transitions held in <see cref="_trajectories"/>; recomputed from it, never persisted on its own.</summary>
    [Scratch]
    private int _storedTransitions;

    /// <summary>Transitions of the episode currently arriving through StoreExperience, committed when it ends.</summary>
    private readonly List<(Vector<T> State, Vector<T> Action, T Reward)> _pendingEpisode = new();

    /// <summary>The highest episode return committed to the buffer, the default conditioning target.</summary>
    private T _bestEpisodeReturn;

    /// <summary>Whether any episode has been committed, i.e. whether <see cref="_bestEpisodeReturn"/> is meaningful.</summary>
    private bool _hasEpisodeReturn;

    private int _updateCount;

    /// <summary>The evaluation context: the last K states, actions and returns-to-go of the current episode.</summary>
    [Scratch]
    private SequenceContext<T> _currentContext;

    /// <summary>The running return-to-go R̂ of the current episode (Algorithm 1: R̂ ← R̂ − r after each step).</summary>
    [Scratch]
    private T _currentReturnToGo;

    /// <summary>Whether <see cref="_currentReturnToGo"/> has been initialised for the current episode.</summary>
    [Scratch]
    private bool _episodeConditioned;

    /// <summary>The absolute timestep of the next step of the current episode.</summary>
    [Scratch]
    private int _episodeStep;

    public DecisionTransformerAgent() : this(new DecisionTransformerOptions<T>()) { }

    /// <summary>
    /// Initializes a Decision Transformer agent from <paramref name="options"/>.
    /// </summary>
    /// <param name="options">The agent's configuration.</param>
    /// <exception cref="ArgumentOutOfRangeException">An option is outside its valid range.</exception>
    public DecisionTransformerAgent(DecisionTransformerOptions<T> options)
        : base(options)
    {
        Guard.NotNull(options);
        _options = options;
        Options = _options;
        ValidateOptions(options);

        _updateCount = 0;
        _bestEpisodeReturn = NumOps.Zero;
        _hasEpisodeReturn = false;
        _currentContext = new SequenceContext<T>();
        _currentReturnToGo = NumOps.Zero;
        _episodeConditioned = false;
        _episodeStep = 0;

        _transformerNetwork = CreateTransformerNetwork();

        // Register network with base class
        Networks.Add(_transformerNetwork);
    }

    private static void ValidateOptions(DecisionTransformerOptions<T> options)
    {
        if (options.StateSize <= 0)
            throw new ArgumentOutOfRangeException(nameof(options), options.StateSize, "StateSize must be positive.");
        if (options.ActionSize <= 0)
            throw new ArgumentOutOfRangeException(nameof(options), options.ActionSize, "ActionSize must be positive.");
        if (options.EmbeddingDim <= 0)
            throw new ArgumentOutOfRangeException(nameof(options), options.EmbeddingDim, "EmbeddingDim must be positive.");
        if (options.NumLayers <= 0)
            throw new ArgumentOutOfRangeException(nameof(options), options.NumLayers, "NumLayers must be positive.");
        if (options.NumHeads <= 0)
            throw new ArgumentOutOfRangeException(nameof(options), options.NumHeads, "NumHeads must be positive.");
        if (options.EmbeddingDim % options.NumHeads != 0)
            throw new ArgumentOutOfRangeException(nameof(options), options.NumHeads,
                $"NumHeads ({options.NumHeads}) must divide EmbeddingDim ({options.EmbeddingDim}).");
        if (options.ContextLength <= 0)
            throw new ArgumentOutOfRangeException(nameof(options), options.ContextLength, "ContextLength must be positive.");
        if (double.IsNaN(options.DropoutRate) || options.DropoutRate < 0.0 || options.DropoutRate >= 1.0)
            throw new ArgumentOutOfRangeException(nameof(options), options.DropoutRate, "DropoutRate must be in [0, 1).");
        if (options.MaxEpisodeLength <= 0)
            throw new ArgumentOutOfRangeException(nameof(options), options.MaxEpisodeLength, "MaxEpisodeLength must be positive.");
        if (options.BufferSize <= 0)
            throw new ArgumentOutOfRangeException(nameof(options), options.BufferSize, "BufferSize must be positive.");
        if (options.BatchSize <= 0)
            throw new ArgumentOutOfRangeException(nameof(options), options.BatchSize, "BatchSize must be positive.");
        if (double.IsNaN(options.ReturnScale) || double.IsInfinity(options.ReturnScale) || options.ReturnScale <= 0.0)
            throw new ArgumentOutOfRangeException(nameof(options), options.ReturnScale, "ReturnScale must be positive and finite.");
        if (options.TargetReturn is double target && (double.IsNaN(target) || double.IsInfinity(target)))
            throw new ArgumentOutOfRangeException(nameof(options), target, "TargetReturn must be finite when set.");
    }

    // Token feature layout (Chen et al. 2021, Sec. 3: one linear embedding per modality plus a learned timestep
    // embedding). Every token is [modality one-hot | R̂ | s | a | timestep one-hot]; a token fills only its own block.
    private int ReturnFeature => TokensPerStep;
    private int StateFeatureOffset => ReturnFeature + 1;
    private int ActionFeatureOffset => StateFeatureOffset + _options.StateSize;
    private int TimestepFeatureOffset => ActionFeatureOffset + _options.ActionSize;
    private int TokenFeatureCount => TimestepFeatureOffset + _options.MaxEpisodeLength;
    private int SequenceTokenCount => TokensPerStep * _options.ContextLength;

    /// <summary>The transformer network, exposed to the test assembly for structural checks.</summary>
    internal INeuralNetwork<T> TransformerNetwork => _transformerNetwork;

    /// <summary>Number of trajectories currently held in the offline buffer.</summary>
    internal int StoredTrajectoryCount => _trajectories.Count;

    /// <summary>Number of transitions currently held in the offline buffer.</summary>
    internal int StoredTransitionCount => _storedTransitions;

    /// <summary>The returns-to-go of one stored trajectory, as they are embedded during training (unscaled).</summary>
    internal IReadOnlyList<T> GetStoredReturnsToGo(int trajectoryIndex) => _trajectories[trajectoryIndex].ReturnsToGo;

    /// <summary>The returns-to-go of the current evaluation context, oldest first (unscaled).</summary>
    internal IReadOnlyList<T> CurrentContextReturnsToGo => _currentContext.ReturnsToGo;

    private INeuralNetwork<T> CreateTransformerNetwork()
    {
        int featureCount = TokenFeatureCount;
        int hidden = _options.EmbeddingDim;
        var identity = (IActivationFunction<T>)new IdentityActivation<T>();

        var layers = new List<ILayer<T>>();

        // Token embedding: W_m x + b_m + E[t], bias-free because the modality indicator supplies b_m.
        var embedding = new DenseLayer<T>(hidden, identity, null, BiasMode.Never);
        embedding.ResolveShapesOnly(new[] { featureCount });
        layers.Add(embedding);

        // embed_ln, then GPT-2's embedding dropout.
        layers.Add(new LayerNormalizationLayer<T>(hidden));
        if (_options.DropoutRate > 0.0) layers.Add(new DropoutLayer<T>(_options.DropoutRate));

        // GPT-2 blocks: pre-LayerNorm, causal multi-head self-attention, 4x GELU MLP, residual dropout.
        for (int i = 0; i < _options.NumLayers; i++)
        {
            var block = new TransformerEncoderBlock<T>(
                hiddenSize: hidden,
                numHeads: _options.NumHeads,
                ffnDim: 4 * hidden,
                dropoutRate: _options.DropoutRate,
                ffnActivation: new GELUActivation<T>(),
                normPlacement: TransformerNormPlacement.PreNorm);
            if (block.AttentionLayer is not MultiHeadAttentionLayer<T> attention)
                throw new InvalidOperationException(
                    "TransformerEncoderBlock no longer hosts a MultiHeadAttentionLayer, so the causal mask cannot be set.");
            // A token may attend only to itself and earlier tokens, so the state token s_t never sees a_t.
            attention.UseCausalMask = true;
            layers.Add(block);
        }

        // GPT-2's final LayerNorm, then the action head: tanh for continuous actions, logits for discrete ones.
        layers.Add(new LayerNormalizationLayer<T>(hidden));
        var head = new DenseLayer<T>(
            _options.ActionSize,
            _options.IsContinuous ? new TanhActivation<T>() : identity);
        head.ResolveShapesOnly(new[] { hidden });
        layers.Add(head);

        var architecture = new NeuralNetworkArchitecture<T>(
            inputType: InputType.OneDimensional,
            taskType: NeuralNetworkTaskType.Regression,
            complexity: NetworkComplexity.Medium,
            inputSize: featureCount,
            outputSize: _options.ActionSize,
            layers: layers);

        return new NeuralNetwork<T>(architecture,
            optimizer: _options.OptimizerFactory?.Invoke() ?? CreateNetworkOptimizer(LearningRate),
            lossFunction: LossFunction);
    }

    /// <summary>
    /// Load offline dataset into the trajectory buffer.
    /// </summary>
    /// <param name="trajectories">Complete trajectories of (state, action, reward) steps; returns-to-go are computed here.</param>
    /// <remarks>
    /// Each trajectory's return-to-go is R̂_t = Σ_{t'=t}^{T} r_{t'} (Chen et al. 2021, Sec. 3) and its timesteps
    /// count from zero. Trajectories beyond <see cref="DecisionTransformerOptions{T}.BufferSize"/> evict the oldest.
    /// </remarks>
    public void LoadOfflineData(List<List<(Vector<T> state, Vector<T> action, T reward)>> trajectories)
    {
        if (trajectories is null) throw new ArgumentNullException(nameof(trajectories));

        foreach (var trajectory in trajectories)
        {
            if (trajectory is null) throw new ArgumentException("A trajectory is null.", nameof(trajectories));

            var steps = new List<(Vector<T> State, Vector<T> Action, T Reward)>(trajectory.Count);
            foreach (var (state, action, reward) in trajectory)
            {
                ValidateState(state);
                ValidateAction(action);
                steps.Add((CopyVector(state), CopyVector(action), reward));
            }

            CommitTrajectory(steps);
        }
    }

    public override Vector<T> SelectAction(Vector<T> state, bool training = true)
    {
        return SelectActionWithReturn(state, CurrentReturnToGo(), training);
    }

    /// <summary>
    /// Select action conditioned on desired return-to-go.
    /// </summary>
    /// <param name="state">The current observation s_t.</param>
    /// <param name="targetReturn">The return-to-go R̂_t to condition this step on (unscaled).</param>
    /// <param name="training">For discrete actions, true samples the softmax and false takes the arg-max; the
    /// continuous policy is deterministic either way.</param>
    /// <returns>The action a_t.</returns>
    /// <remarks>
    /// Appends (R̂_t, s_t) to the episode context, runs the transformer over the last K timesteps, and reads a_t from
    /// the output at s_t (Algorithm 1). The return-to-go given here becomes the episode's running return-to-go, from
    /// which <see cref="StoreExperience"/> subtracts each reward.
    /// </remarks>
    public Vector<T> SelectActionWithReturn(Vector<T> state, T targetReturn, bool training = true)
    {
        ValidateState(state);

        _currentReturnToGo = targetReturn;
        _episodeConditioned = true;
        int timestep = _episodeStep;
        _episodeStep++;

        _currentContext.States.Add(CopyVector(state));
        _currentContext.ReturnsToGo.Add(targetReturn);

        // Keep the last K timesteps; the action list trails the states by the current, still-unknown action.
        while (_currentContext.Length > _options.ContextLength)
        {
            _currentContext.States.RemoveAt(0);
            _currentContext.ReturnsToGo.RemoveAt(0);
            if (_currentContext.Actions.Count > 0) _currentContext.Actions.RemoveAt(0);
        }

        int length = _currentContext.Length;
        var tokens = new T[SequenceTokenCount * TokenFeatureCount];
        int firstTimestep = timestep - (length - 1);
        for (int i = 0; i < length; i++)
        {
            Vector<T>? action = i < _currentContext.Actions.Count ? _currentContext.Actions[i] : null;
            WriteTimestepTokens(tokens, i, _currentContext.ReturnsToGo[i], _currentContext.States[i], action,
                firstTimestep + i);
        }

        var output = PredictStateTokenOutput(tokens, length - 1);
        var chosen = ToAction(output, sample: training && !_options.IsContinuous);

        _currentContext.Actions.Add(CopyVector(chosen));
        return chosen;
    }

    public override void StoreExperience(Vector<T> state, Vector<T> action, T reward, Vector<T> nextState, bool done)
    {
        ValidateState(state);
        ValidateAction(action);

        _pendingEpisode.Add((CopyVector(state), CopyVector(action), reward));

        // If this transition is the step the context just acted on, record the action actually taken.
        int last = _currentContext.Length - 1;
        if (last >= 0 && _currentContext.Actions.Count == _currentContext.Length
            && VectorsEqual(_currentContext.States[last], state))
        {
            _currentContext.Actions[last] = CopyVector(action);
        }

        // Algorithm 1: the next return-to-go is the current one minus the reward just received.
        _currentReturnToGo = NumOps.Subtract(CurrentReturnToGo(), reward);
        _episodeConditioned = true;

        if (done)
        {
            CommitTrajectory(_pendingEpisode);
            _pendingEpisode.Clear();
            ResetEpisode();
        }
    }

    /// <summary>
    /// One supervised update on a batch of trajectory segments (Chen et al. 2021, Sec. 3 and Algorithm 1).
    /// </summary>
    /// <returns>The batch loss, or zero while the buffer holds fewer transitions than the batch size.</returns>
    public override T Train()
    {
        int batchSize = _options.BatchSize;
        if (_storedTransitions == 0 || _storedTransitions < batchSize)
        {
            return NumOps.Zero;
        }

        int contextLength = _options.ContextLength;
        int sequenceTokens = SequenceTokenCount;
        int featureCount = TokenFeatureCount;
        int actionSize = _options.ActionSize;

        var tokens = new T[batchSize * sequenceTokens * featureCount];
        var targets = new T[batchSize * sequenceTokens * actionSize];
        var labelledRows = new List<int>(batchSize * contextLength);

        for (int b = 0; b < batchSize; b++)
        {
            // A segment starts at a uniformly drawn stored transition, which picks a trajectory with probability
            // proportional to its length and a start uniformly within it -- the reference implementation's sampler.
            var (trajectory, start) = LocateTransition(Random.Next(_storedTransitions));
            int length = Math.Min(contextLength, trajectory.Length - start);
            int sampleTokenBase = b * sequenceTokens;

            for (int i = 0; i < length; i++)
            {
                int step = start + i;
                WriteTimestepTokens(tokens, sampleTokenBase / TokensPerStep + i,
                    trajectory.ReturnsToGo[step], trajectory.States[step], trajectory.Actions[step],
                    trajectory.Timesteps[step]);

                int row = sampleTokenBase + TokensPerStep * i + StateToken;
                labelledRows.Add(row);
                WriteActionLabel(targets, row, trajectory.Actions[step]);
            }
        }

        var tokenTensor = new Tensor<T>(tokens, new[] { batchSize, sequenceTokens, featureCount });
        var targetTensor = new Tensor<T>(targets, new[] { batchSize, sequenceTokens, actionSize });
        T loss = TrainOnSequence(tokenTensor, targetTensor, labelledRows.ToArray());

        _updateCount++;
        TrainingSteps++;
        LossHistory.Add(loss);
        return loss;
    }

    /// <summary>
    /// Supervised (online) training entry from <see cref="IFullModel{T,TInput,TOutput}"/>: trains
    /// the transformer to predict <paramref name="target"/> (the desired action) from
    /// <paramref name="state"/>. Decision Transformer is return-conditioned behavior cloning, so a
    /// single supervised step on (return-to-go, state) → action IS the training objective. The base
    /// <see cref="ReinforcementLearningAgentBase{T}.Train(Vector{T},Vector{T})"/> stores a Q-learning
    /// transition then calls the parameterless <see cref="Train()"/>, which for this agent would
    /// only buffer a one-step episode. Route the supervised pair straight into a transformer step
    /// instead, conditioned exactly as <see cref="Predict"/> conditions a query: a one-timestep
    /// context at timestep zero whose return-to-go is the episode's starting target.
    /// </summary>
    public override void Train(Vector<T> state, Vector<T> target)
    {
        if (state is null) throw new ArgumentNullException(nameof(state));
        if (target is null) throw new ArgumentNullException(nameof(target));
        ValidateState(state);
        if (target.Length == 0) throw new ArgumentException("target must contain at least one element.", nameof(target));

        var (tokens, targets, rows) = BuildSingleStepExample(state, target);
        TrainOnSequence(tokens, targets, rows);
        _updateCount++;
    }

    private T TrainOnSequence(Tensor<T> tokens, Tensor<T> targets, int[] labelledRows)
    {
        var objective = new DecisionTransformerActionLoss<T>(
            labelledRows, _options.ActionSize, _options.IsContinuous, LossFunction);
        var network = (NeuralNetworkBase<T>)_transformerNetwork;
        return network.TrainWithCustomLoss(tokens, output => objective.ComputeTapeLoss(output, targets));
    }

    /// <summary>The token sequence, label tensor and labelled row for one state at the start of an episode.</summary>
    private (Tensor<T> Tokens, Tensor<T> Targets, int[] Rows) BuildSingleStepExample(Vector<T> state, Vector<T> target)
    {
        int sequenceTokens = SequenceTokenCount;
        int actionSize = _options.ActionSize;

        var tokens = new T[sequenceTokens * TokenFeatureCount];
        WriteTimestepTokens(tokens, 0, InitialReturnToGo(), state, null, 0);

        var label = new Vector<T>(actionSize);
        if (_options.IsContinuous)
        {
            int copy = Math.Min(target.Length, actionSize);
            for (int i = 0; i < copy; i++) label[i] = target[i];
        }
        else
        {
            label[Math.Min(ArgMax(target), actionSize - 1)] = NumOps.One;
        }

        var targets = new T[sequenceTokens * actionSize];
        WriteActionLabel(targets, StateToken, label);

        return (
            new Tensor<T>(tokens, new[] { 1, sequenceTokens, TokenFeatureCount }),
            new Tensor<T>(targets, new[] { 1, sequenceTokens, actionSize }),
            new[] { StateToken });
    }

    /// <summary>
    /// Writes the three tokens of one timestep into a flattened <c>[tokens, features]</c> buffer.
    /// </summary>
    /// <param name="tokens">The token buffer; rows are tokens and must be zero where nothing is written.</param>
    /// <param name="stepSlot">The timestep slot in the flattened buffer (token row = 3 * slot).</param>
    /// <param name="returnToGo">R̂_t, unscaled.</param>
    /// <param name="state">s_t.</param>
    /// <param name="action">a_t, or null when it is not yet known (the step being predicted).</param>
    /// <param name="timestep">The absolute episode timestep, clipped to the embedding table.</param>
    private void WriteTimestepTokens(T[] tokens, int stepSlot, T returnToGo, Vector<T> state, Vector<T>? action, int timestep)
    {
        int featureCount = TokenFeatureCount;
        int clippedTimestep = Math.Min(Math.Max(timestep, 0), _options.MaxEpisodeLength - 1);
        int firstRow = stepSlot * TokensPerStep;

        for (int token = 0; token < TokensPerStep; token++)
        {
            int rowBase = (firstRow + token) * featureCount;
            tokens[rowBase + token] = NumOps.One;
            tokens[rowBase + TimestepFeatureOffset + clippedTimestep] = NumOps.One;
        }

        tokens[(firstRow + ReturnToken) * featureCount + ReturnFeature] =
            NumOps.Divide(returnToGo, NumOps.FromDouble(_options.ReturnScale));

        int stateBase = (firstRow + StateToken) * featureCount + StateFeatureOffset;
        for (int i = 0; i < _options.StateSize; i++) tokens[stateBase + i] = state[i];

        if (action is not null)
        {
            int actionBase = (firstRow + ActionToken) * featureCount + ActionFeatureOffset;
            for (int i = 0; i < _options.ActionSize; i++) tokens[actionBase + i] = action[i];
        }
    }

    /// <summary>Writes the action label for one state-token row: the action itself, or its one-hot when discrete.</summary>
    private void WriteActionLabel(T[] targets, int row, Vector<T> action)
    {
        int actionSize = _options.ActionSize;
        int rowBase = row * actionSize;
        if (_options.IsContinuous)
        {
            for (int i = 0; i < actionSize; i++) targets[rowBase + i] = action[i];
        }
        else
        {
            targets[rowBase + ArgMax(action)] = NumOps.One;
        }
    }

    /// <summary>Runs one token sequence and returns the head output at the state token of timestep slot <paramref name="stepSlot"/>.</summary>
    private Vector<T> PredictStateTokenOutput(T[] tokens, int stepSlot)
    {
        var input = new Tensor<T>(tokens, new[] { 1, SequenceTokenCount, TokenFeatureCount });
        var output = _transformerNetwork.Predict(input).ToVector();

        int actionSize = _options.ActionSize;
        int row = stepSlot * TokensPerStep + StateToken;
        var result = new Vector<T>(actionSize);
        for (int i = 0; i < actionSize; i++) result[i] = output[row * actionSize + i];
        return result;
    }

    /// <summary>
    /// Runs the transformer over an explicit context and returns the head output at every state token, in order.
    /// </summary>
    /// <remarks>
    /// Exposed to the test assembly so the causal structure can be checked directly: the output at s_t must not
    /// change when anything at a later position (a_t, or any later timestep) changes.
    /// </remarks>
    internal Vector<T>[] PredictContext(
        IReadOnlyList<T> returnsToGo,
        IReadOnlyList<Vector<T>> states,
        IReadOnlyList<Vector<T>?> actions,
        int firstTimestep)
    {
        int length = states.Count;
        if (length == 0 || length > _options.ContextLength || returnsToGo.Count != length || actions.Count != length)
            throw new ArgumentException("The context must hold 1..ContextLength timesteps with matching lists.");

        var tokens = new T[SequenceTokenCount * TokenFeatureCount];
        for (int i = 0; i < length; i++)
            WriteTimestepTokens(tokens, i, returnsToGo[i], states[i], actions[i], firstTimestep + i);

        var input = new Tensor<T>(tokens, new[] { 1, SequenceTokenCount, TokenFeatureCount });
        var output = _transformerNetwork.Predict(input).ToVector();

        int actionSize = _options.ActionSize;
        var results = new Vector<T>[length];
        for (int t = 0; t < length; t++)
        {
            int row = t * TokensPerStep + StateToken;
            results[t] = new Vector<T>(actionSize);
            for (int i = 0; i < actionSize; i++) results[t][i] = output[row * actionSize + i];
        }

        return results;
    }

    /// <summary>Turns the head output into an action: the tanh output itself, or a one-hot of the arg-max or a softmax draw.</summary>
    private Vector<T> ToAction(Vector<T> headOutput, bool sample)
    {
        if (_options.IsContinuous) return headOutput;

        int chosen = sample ? SampleSoftmax(headOutput) : ArgMax(headOutput);
        var oneHot = new Vector<T>(_options.ActionSize);
        oneHot[chosen] = NumOps.One;
        return oneHot;
    }

    private int SampleSoftmax(Vector<T> logits)
    {
        double max = double.NegativeInfinity;
        for (int i = 0; i < logits.Length; i++) max = Math.Max(max, NumOps.ToDouble(logits[i]));

        var weights = new double[logits.Length];
        double sum = 0.0;
        for (int i = 0; i < logits.Length; i++)
        {
            weights[i] = Math.Exp(NumOps.ToDouble(logits[i]) - max);
            sum += weights[i];
        }

        double draw = Random.NextDouble() * sum;
        for (int i = 0; i < weights.Length; i++)
        {
            draw -= weights[i];
            if (draw <= 0.0) return i;
        }

        return weights.Length - 1;
    }

    private int ArgMax(Vector<T> values)
    {
        int best = 0;
        for (int i = 1; i < values.Length; i++)
            if (NumOps.GreaterThan(values[i], values[best])) best = i;
        return best;
    }

    /// <summary>The episode's starting return-to-go: the configured target, else the best stored episode return.</summary>
    private T InitialReturnToGo()
    {
        if (_options.TargetReturn is double target) return NumOps.FromDouble(target);
        return _hasEpisodeReturn ? _bestEpisodeReturn : NumOps.Zero;
    }

    /// <summary>The running return-to-go, initialised to <see cref="InitialReturnToGo"/> at the start of an episode.</summary>
    private T CurrentReturnToGo() => _episodeConditioned ? _currentReturnToGo : InitialReturnToGo();

    /// <summary>
    /// Adds a finished trajectory to the buffer with its returns-to-go and timesteps, evicting the oldest
    /// trajectories beyond <see cref="DecisionTransformerOptions{T}.BufferSize"/> transitions.
    /// </summary>
    private void CommitTrajectory(List<(Vector<T> State, Vector<T> Action, T Reward)> steps)
    {
        int count = steps.Count;
        if (count == 0) return;

        // R̂_t = r_t + R̂_{t+1}, undiscounted (Sec. 3).
        var returnsToGo = new T[count];
        T running = NumOps.Zero;
        for (int t = count - 1; t >= 0; t--)
        {
            running = NumOps.Add(steps[t].Reward, running);
            returnsToGo[t] = running;
        }

        if (!_hasEpisodeReturn || NumOps.GreaterThan(returnsToGo[0], _bestEpisodeReturn))
        {
            _bestEpisodeReturn = returnsToGo[0];
            _hasEpisodeReturn = true;
        }

        // A trajectory longer than the whole buffer keeps its most recent transitions.
        int keep = Math.Min(count, _options.BufferSize);
        int first = count - keep;
        var states = new Vector<T>[keep];
        var actions = new Vector<T>[keep];
        var kept = new T[keep];
        var timesteps = new int[keep];
        for (int i = 0; i < keep; i++)
        {
            states[i] = steps[first + i].State;
            actions[i] = steps[first + i].Action;
            kept[i] = returnsToGo[first + i];
            timesteps[i] = first + i;
        }

        _trajectories.Add(new StoredTrajectory(states, actions, kept, timesteps));
        _storedTransitions += keep;

        while (_storedTransitions > _options.BufferSize && _trajectories.Count > 1)
        {
            _storedTransitions -= _trajectories[0].Length;
            _trajectories.RemoveAt(0);
        }
    }

    /// <summary>Maps a flat transition index over the buffer to its trajectory and step.</summary>
    private (StoredTrajectory Trajectory, int Step) LocateTransition(int index)
    {
        foreach (var trajectory in _trajectories)
        {
            if (index < trajectory.Length) return (trajectory, index);
            index -= trajectory.Length;
        }

        throw new InvalidOperationException("Transition index is outside the stored trajectories.");
    }

    private void ValidateState(Vector<T> state)
    {
        if (state is null) throw new ArgumentNullException(nameof(state));
        if (state.Length != _options.StateSize)
            throw new ArgumentException(
                $"State length must be {_options.StateSize}, got {state.Length}.", nameof(state));
    }

    private void ValidateAction(Vector<T> action)
    {
        if (action is null) throw new ArgumentNullException(nameof(action));
        if (action.Length != _options.ActionSize)
            throw new ArgumentException(
                $"Action length must be {_options.ActionSize}, got {action.Length}.", nameof(action));
    }

    private static Vector<T> CopyVector(Vector<T> source)
    {
        var copy = new Vector<T>(source.Length);
        for (int i = 0; i < source.Length; i++) copy[i] = source[i];
        return copy;
    }

    private bool VectorsEqual(Vector<T> left, Vector<T> right)
    {
        if (left.Length != right.Length) return false;
        for (int i = 0; i < left.Length; i++)
            if (!NumOps.Equals(left[i], right[i])) return false;
        return true;
    }

    public override Dictionary<string, T> GetMetrics()
    {
        return new Dictionary<string, T>
        {
            ["updates"] = NumOps.FromDouble(_updateCount),
            ["buffer_size"] = NumOps.FromDouble(_storedTransitions),
            ["trajectories"] = NumOps.FromDouble(_trajectories.Count)
        };
    }

    /// <inheritdoc/>
    /// <remarks>
    /// Ends the current episode: transitions stored since the last terminal step are committed to the buffer as a
    /// (truncated) trajectory, and the evaluation context and running return-to-go start afresh.
    /// </remarks>
    public override void ResetEpisode()
    {
        if (_pendingEpisode.Count > 0)
        {
            CommitTrajectory(_pendingEpisode);
            _pendingEpisode.Clear();
        }

        _currentContext = new SequenceContext<T>();
        _currentReturnToGo = NumOps.Zero;
        _episodeConditioned = false;
        _episodeStep = 0;
    }

    public override Vector<T> Predict(Vector<T> input)
    {
        // Pure, stateless single-step inference (the IFullModel.Predict contract): two Predict(state)
        // calls must return the same action. SelectAction is the EPISODIC entry — it appends to the
        // rolling _currentContext, so calling it from Predict would make repeated Predict(state) calls
        // diverge and break determinism and clone parity. Predict instead answers for a fresh episode:
        // a one-timestep context at timestep zero, conditioned on the episode's starting return-to-go,
        // run without touching _currentContext. Discrete actions return the arg-max one-hot.
        if (input is null) throw new ArgumentNullException(nameof(input));
        ValidateState(input);

        var tokens = new T[SequenceTokenCount * TokenFeatureCount];
        WriteTimestepTokens(tokens, 0, InitialReturnToGo(), input, null, 0);
        return ToAction(PredictStateTokenOutput(tokens, 0), sample: false);
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

    public override ModelMetadata<T> GetModelMetadata()
    {
        return new ModelMetadata<T>
        {
            FeatureCount = _options.StateSize,
            Complexity = ParameterCount,
        };
    }

    public override int FeatureCount => _options.StateSize;

    /// <summary>
    /// Computes gradients of the loss with respect to this agent's parameters, without updating them.
    /// </summary>
    /// <param name="input">The input state.</param>
    /// <param name="target">The target output.</param>
    /// <param name="lossFunction">The loss to differentiate, or null to use the agent's own.</param>
    /// <returns>A gradient vector the same length as <c>GetParameters()</c>.</returns>
    /// <remarks>
    /// Returns PARAMETER-space gradients, as <c>IGradientComputable</c> documents and as
    /// <c>ApplyGradients</c> requires. The objective is the one <see cref="Train(Vector{T}, Vector{T})"/> steps on:
    /// the action predicted at the state token of a one-timestep context, against <paramref name="target"/>
    /// (regressed with <paramref name="lossFunction"/> for continuous actions, cross-entropy for discrete ones).
    /// The gradient comes from the network's own tape pass, so no derivative is written by hand here.
    /// </remarks>
    public Vector<T> ComputeGradients(
        Vector<T> input,
        Vector<T> target,
        ILossFunction<T>? lossFunction = null)
    {
        if (input is null) throw new ArgumentNullException(nameof(input));
        if (target is null) throw new ArgumentNullException(nameof(target));
        ValidateState(input);

        if (_transformerNetwork is not IGradientComputable<T, Tensor<T>, Tensor<T>> computable)
        {
            throw new NotSupportedException(
                $"{GetType().Name} cannot compute parameter gradients because its network "
                + $"({_transformerNetwork.GetType().Name}) does not implement "
                + "IGradientComputable<T, Tensor<T>, Tensor<T>>.");
        }

        var (tokens, targets, rows) = BuildSingleStepExample(input, target);
        var objective = new DecisionTransformerActionLoss<T>(
            rows, _options.ActionSize, _options.IsContinuous, lossFunction ?? LossFunction);
        return computable.ComputeGradients(tokens, targets, objective);
    }

    public override void SaveModel(string filepath)
    {
        var data = Serialize();
        System.IO.File.WriteAllBytes(filepath, data);
    }

    public override void LoadModel(string filepath)
    {
        var data = System.IO.File.ReadAllBytes(filepath);
        Deserialize(data);
    }

    /// <summary>One stored trajectory: states, actions, returns-to-go and absolute timesteps, step-aligned.</summary>
    private sealed class StoredTrajectory
    {
        public StoredTrajectory(Vector<T>[] states, Vector<T>[] actions, T[] returnsToGo, int[] timesteps)
        {
            States = states;
            Actions = actions;
            ReturnsToGo = returnsToGo;
            Timesteps = timesteps;
        }

        public Vector<T>[] States { get; }

        public Vector<T>[] Actions { get; }

        public T[] ReturnsToGo { get; }

        public int[] Timesteps { get; }

        public int Length => States.Length;
    }
}
