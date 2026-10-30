using AiDotNet.ActivationFunctions;
using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.LinearAlgebra;
using AiDotNet.LossFunctions;
using AiDotNet.Models;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Optimizers;
using AiDotNet.ReinforcementLearning.ReplayBuffers;

namespace AiDotNet.ReinforcementLearning.Agents.WorldModels;

/// <summary>
/// The stage of the World Models training cycle that the next <see cref="WorldModelsAgent{T}.Train"/> call runs.
/// </summary>
public enum WorldModelsTrainingStage
{
    /// <summary>V: the variational autoencoder is trained on stored observations.</summary>
    Vision,

    /// <summary>M: the MDN-RNN is trained on V's encodings of stored sequences.</summary>
    Memory,

    /// <summary>C: the controller is evolved with CMA-ES inside rollouts dreamed by M.</summary>
    Controller
}

/// <summary>
/// Where a <see cref="WorldModelsAgent{T}"/> stands in its V -> M -> C training cycle.
/// </summary>
/// <remarks>
/// Plain data so it travels with the agent through Clone and serialization: a copy taken mid-stage resumes the same
/// epoch order and, in the Controller stage, the same CMA-ES search.
/// </remarks>
public sealed class WorldModelsTrainingCycle
{
    /// <summary>Gets or sets the stage the next Train() call runs.</summary>
    public WorldModelsTrainingStage Stage { get; set; } = WorldModelsTrainingStage.Vision;

    /// <summary>Gets or sets whether the current stage's epoch order has been drawn.</summary>
    public bool StagePrepared { get; set; }

    /// <summary>Gets or sets the shuffled replay slots (Vision) or window starts (Memory) of the current epoch.</summary>
    public List<int> EpochOrder { get; set; } = new List<int>();

    /// <summary>Gets or sets the position in <see cref="EpochOrder"/> of the next batch.</summary>
    public int EpochCursor { get; set; }

    /// <summary>Gets or sets the number of epochs the current stage has completed.</summary>
    public int EpochsCompleted { get; set; }

    /// <summary>Gets or sets the Controller stage's CMA-ES search, or null outside that stage.</summary>
    public CMAESSearchState? ControllerSearch { get; set; }
}

/// <summary>
/// World Models agent (Ha &amp; Schmidhuber 2018): a VAE (V), an MDN-RNN (M) and a CMA-ES-evolved controller (C)
/// trained inside M's dream.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// V encodes an observation x into q(z|x) = N(mu, diag(exp(logvar))) and decodes z back, trained on
/// ||x - D(z)||^2 + VAEBeta * KL(q(z|x) || N(0, I)) (appendix, "Variational Autoencoder"). M is an LSTM stack
/// whose output parameterises, for every latent dimension, a mixture of NumMixtures Gaussians over z_{t+1}, plus
/// the reward and the episode-end flag: P(z_{t+1}, r_{t+1}, d_{t+1} | a_t, z_t, h_t) (sections "MDN-RNN (M) Model"
/// and "Iterative Training Procedure"). C is a_t = tanh(W_c [z_t h_t] + b_c), optionally with hidden tanh layers
/// (section "Controller (C) Model"), and is trained by CMA-ES on the average cumulative reward of dream rollouts
/// (appendix, "Evolution Strategies"; section "Training Inside of the Dream").
/// </para>
/// <para>
/// <b>What one <see cref="Train"/> call does.</b> The paper trains V, then M on V's encodings, then C, and repeats
/// the cycle as new experience arrives ("Iterative Training Procedure"). Each call advances that cycle by one unit
/// of work, so a call's cost stays bounded:
/// <list type="bullet">
/// <item><description><b>Vision</b>: one optimizer step of V on the next BatchSize stored transitions of the current
/// epoch. After VAEEpochs full passes over the observations stored when the stage began, the stage moves to
/// Memory.</description></item>
/// <item><description><b>Memory</b>: one optimizer step of M (BPTT through the LSTM) on the next BatchSize sequence
/// windows of the current epoch, each at most RNNSequenceLength long and never crossing an episode end; inputs
/// are z sampled afresh from V's posterior on every step, as the paper does to avoid overfitting a single sample.
/// After RNNEpochs passes, the stage moves to Controller.</description></item>
/// <item><description><b>Controller</b>: one CMA-ES generation of ControllerPopulationSize candidates, each scored
/// by the mean cumulative predicted reward of ControllerEvaluationRollouts dream rollouts of up to RolloutLength
/// steps, started from V's encodings of stored observations; the controller becomes the search distribution's
/// mean. No external environment is needed. The search state persists across calls (and through Clone and
/// serialization); after ControllerGenerations generations the cycle restarts at Vision.</description></item>
/// </list>
/// The returned value is the loss of the unit just run: the VAE loss, the MDN-RNN loss, or the negated mean dream
/// return of the generation's candidates.
/// </para>
/// <para><b>For Beginners:</b>
/// World Models is inspired by how humans learn: we build mental models
/// of the world, then make decisions based on those models rather than
/// raw sensory input.
///
/// Three components (V-M-C):
/// - **V (VAE)**: Compresses observations into compact codes
/// - **M (MDN-RNN)**: Learns what code, reward and episode end come next
/// - **C (Controller)**: A tiny policy acting on the code and M's memory
/// - **Learning in Dreams**: C is improved entirely inside M's imagined rollouts
/// </para>
/// </remarks>
/// <example>
/// <code>
/// // Create a World Models agent that learns in its own dreams
/// var options = new WorldModelsOptions&lt;double&gt;
/// {
///     ObservationWidth = 8, ObservationHeight = 1, ObservationChannels = 1,
///     ActionSize = 2, LatentSize = 4, RNNHiddenSize = 16
/// };
/// var agent = new WorldModelsAgent&lt;double&gt;(options);
///
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
[ResearchPaper("World Models",
    "https://arxiv.org/abs/1803.10122",
    Year = 2018,
    Authors = "Ha, D. & Schmidhuber, J.")]
public partial class WorldModelsAgent<T> : DeepReinforcementLearningAgentBase<T>
{

    /// <inheritdoc />
    /// <remarks>Every network this agent owns, in the order Networks yields them: V's encoder, V's decoder,
    /// M.</remarks>
    protected override void RegisterComponents()
    {
        foreach (var network in Networks) RegisterParameterComponent(network);
    }
    private WorldModelsOptions<T> _options;

    // V: VAE for spatial compression
    private NeuralNetwork<T> _vaeEncoder;
    private NeuralNetwork<T> _vaeDecoder;

    // M: MDN-RNN — the LSTM stack and its mixture-density head, also kept individually so a dream can step them
    // one time step at a time from a carried (h, c) state.
    private NeuralNetwork<T> _mdnRnn;
    private readonly List<LSTMLayer<T>> _lstmLayers;
    private readonly DenseLayer<T> _mdnHead;

    // M's recurrent state while acting in the real environment, one (h, c) pair per LSTM layer, and the top
    // layer's h that C reads.
    [AiDotNet.Attributes.Scratch]
    private Tensor<T>?[] _hiddenStates;
    [AiDotNet.Attributes.Scratch]
    private Tensor<T>?[] _cellStates;
    [AiDotNet.Attributes.Scratch]
    private Tensor<T> _controllerMemory;

    // C: the controller's weights and biases, flattened layer by layer (W row-major [in, out], then b).
    [AiDotNet.Attributes.FittedParameter]
    private Vector<T> _controllerParameters;
    private readonly int[] _controllerLayerSizes;

    private UniformReplayBuffer<T, Vector<T>, Vector<T>> _replayBuffer;
    private int _updateCount;

    // Where the V -> M -> C cycle stands, including the Controller stage's CMA-ES search; carried through Clone and
    // serialization so a copy resumes the cycle (and the search) exactly where the original was.
    private WorldModelsTrainingCycle _cycle;
    private T _lastDreamReturn;

    // Deterministic base seeds for each sub-network's lazy weight init, so a
    // Clone()'d agent reproduces the original's policy (the networks resolve
    // their weights on first forward and would otherwise draw from the shared
    // non-deterministic RNG). Distinct bases keep the networks' inits
    // independent.
    private const int EncoderSeedBase = 7001;
    private const int DecoderSeedBase = 7101;
    private const int RNNSeedBase = 7201;
    private const int ControllerSeedBase = 7301;

    private int ObservationSize => _options.ObservationWidth * _options.ObservationHeight * _options.ObservationChannels;

    // M's head width: per latent dimension NumMixtures logits, means and log standard deviations, then reward and
    // the done logit.
    private int MixtureBlock => _options.LatentSize * _options.NumMixtures;
    private int MdnOutputSize => 3 * MixtureBlock + 2;

    /// <summary>
    /// Initializes a new instance with default settings.
    /// </summary>
    public WorldModelsAgent()
        : this(new WorldModelsOptions<T> { ActionSize = 2 })
    {
    }

    /// <summary>
    /// Initializes a new World Models agent.
    /// </summary>
    /// <param name="options">The agent's configuration.</param>
    /// <exception cref="ArgumentOutOfRangeException">An option is outside its valid range.</exception>
    public WorldModelsAgent(WorldModelsOptions<T> options) : base(options)
    {
        Guard.NotNull(options);
        ValidateOptions(options);
        _options = options;
        Options = _options;
        _updateCount = 0;

        // V: observation -> (mu, logvar); z -> reconstruction.
        _vaeEncoder = CreateEncoderNetwork(ObservationSize, _options.LatentSize * 2);
        _vaeDecoder = CreateDecoderNetwork(_options.LatentSize, ObservationSize);

        // M: [z_t, a_t] sequences -> mixture parameters over z_{t+1}, reward and done.
        _lstmLayers = new List<LSTMLayer<T>>();
        for (int i = 0; i < _options.RNNLayers; i++) _lstmLayers.Add(new LSTMLayer<T>(_options.RNNHiddenSize));
        _mdnHead = new DenseLayer<T>(MdnOutputSize, (IActivationFunction<T>)new IdentityActivation<T>());
        _mdnRnn = CreateMdnRnnNetwork();

        // Resolve every network's lazy weights now, from their fixed seeds, so the parameter layout is complete and
        // stable from construction: M is otherwise first forwarded only once training reaches the Memory stage, and
        // a parameter vector whose length changes mid-training cannot be compared or restored. M resolves its time
        // axis from this first forward, so it is resolved at the RNNSequenceLength every training batch is padded to;
        // the dream and SelectAction step the layers directly, one time step at a time.
        _vaeEncoder.Predict(new Tensor<T>(new[] { 1, ObservationSize }));
        _vaeDecoder.Predict(new Tensor<T>(new[] { 1, _options.LatentSize }));
        _mdnRnn.Predict(new Tensor<T>(new[] { 1, _options.RNNSequenceLength, _options.LatentSize + _options.ActionSize }));

        // C: [z, h] -> (hidden tanh layers) -> tanh action.
        var sizes = new List<int> { _options.LatentSize + _options.RNNHiddenSize };
        sizes.AddRange(_options.ControllerLayers);
        sizes.Add(_options.ActionSize);
        _controllerLayerSizes = sizes.ToArray();
        _controllerParameters = InitializeControllerParameters();

        _hiddenStates = new Tensor<T>?[_options.RNNLayers];
        _cellStates = new Tensor<T>?[_options.RNNLayers];
        _controllerMemory = new Tensor<T>(new[] { 1, _options.RNNHiddenSize });

        _replayBuffer = new UniformReplayBuffer<T, Vector<T>, Vector<T>>(_options.ReplayBufferSize);

        _cycle = new WorldModelsTrainingCycle();
        _lastDreamReturn = NumOps.Zero;

        Networks.Add(_vaeEncoder);
        Networks.Add(_vaeDecoder);
        Networks.Add(_mdnRnn);
    }

    private static void ValidateOptions(WorldModelsOptions<T> options)
    {
        static void Positive(int value, string name)
        {
            if (value <= 0) throw new ArgumentOutOfRangeException(name, value, $"{name} must be positive.");
        }

        Positive(options.ObservationWidth, nameof(options.ObservationWidth));
        Positive(options.ObservationHeight, nameof(options.ObservationHeight));
        Positive(options.ObservationChannels, nameof(options.ObservationChannels));
        Positive(options.ActionSize, nameof(options.ActionSize));
        Positive(options.LatentSize, nameof(options.LatentSize));
        Positive(options.RNNHiddenSize, nameof(options.RNNHiddenSize));
        Positive(options.RNNLayers, nameof(options.RNNLayers));
        Positive(options.NumMixtures, nameof(options.NumMixtures));
        Positive(options.VAEEpochs, nameof(options.VAEEpochs));
        Positive(options.RNNEpochs, nameof(options.RNNEpochs));
        Positive(options.RNNSequenceLength, nameof(options.RNNSequenceLength));
        Positive(options.ControllerGenerations, nameof(options.ControllerGenerations));
        Positive(options.ControllerEvaluationRollouts, nameof(options.ControllerEvaluationRollouts));
        Positive(options.RolloutLength, nameof(options.RolloutLength));
        Positive(options.BatchSize, nameof(options.BatchSize));
        Positive(options.ReplayBufferSize, nameof(options.ReplayBufferSize));

        if (options.ControllerPopulationSize < 4)
        {
            throw new ArgumentOutOfRangeException(nameof(options.ControllerPopulationSize),
                options.ControllerPopulationSize, "CMA-ES needs a population of at least 4.");
        }

        if (double.IsNaN(options.VAEBeta) || double.IsInfinity(options.VAEBeta) || options.VAEBeta < 0.0)
        {
            throw new ArgumentOutOfRangeException(nameof(options.VAEBeta), options.VAEBeta,
                "VAEBeta must be non-negative and finite.");
        }

        if (double.IsNaN(options.Temperature) || double.IsInfinity(options.Temperature) || options.Temperature <= 0.0)
        {
            throw new ArgumentOutOfRangeException(nameof(options.Temperature), options.Temperature,
                "Temperature must be positive and finite.");
        }

        if (double.IsNaN(options.ControllerInitialStepSize) || double.IsInfinity(options.ControllerInitialStepSize)
            || options.ControllerInitialStepSize <= 0.0)
        {
            throw new ArgumentOutOfRangeException(nameof(options.ControllerInitialStepSize),
                options.ControllerInitialStepSize, "ControllerInitialStepSize must be positive and finite.");
        }

        if (options.VAEEncoderChannels is null)
            throw new ArgumentNullException(nameof(options.VAEEncoderChannels));
        foreach (var width in options.VAEEncoderChannels) Positive(width, nameof(options.VAEEncoderChannels));

        if (options.ControllerLayers is null)
            throw new ArgumentNullException(nameof(options.ControllerLayers));
        foreach (var width in options.ControllerLayers) Positive(width, nameof(options.ControllerLayers));
    }

    private NeuralNetwork<T> CreateEncoderNetwork(int inputSize, int outputSize)
    {
        // Build the layer list explicitly and hand it to the architecture so
        // NeuralNetwork.InitializeLayers uses EXACTLY these layers. Passing an
        // architecture with no layers makes InitializeLayers fall back to
        // CreateDefaultNeuralNetworkLayers, which sizes a hidden layer to ~2x
        // the (image-flattened) input — for a 64x64x3 = 12,288-wide observation
        // that alone is a single 12,288x24,576 weight (~2.4 GB at FP64).
        var layers = new List<ILayer<T>>();
        foreach (var channels in _options.VAEEncoderChannels)
        {
            layers.Add(new DenseLayer<T>(channels, (IActivationFunction<T>)new ReLUActivation<T>()));
        }
        layers.Add(new DenseLayer<T>(outputSize, (IActivationFunction<T>)new IdentityActivation<T>()));

        // Pin a deterministic per-layer RandomSeed so the lazy weight
        // initialization (DenseLayer resolves its weights on first forward) is
        // reproducible. NeuralNetwork.InitializeLayers does NOT wire explicitly
        // supplied custom layers, so without this they would fall back to the
        // shared non-deterministic RNG and a Clone()'d agent would initialize a
        // DIFFERENT policy than the original (Clone_ShouldProduceSamePolicy).
        AssignDeterministicSeeds(layers, EncoderSeedBase);

        var architecture = new NeuralNetworkArchitecture<T>(
            inputType: InputType.OneDimensional,
            taskType: NeuralNetworkTaskType.Regression,
            complexity: NetworkComplexity.Medium,
            inputSize: inputSize,
            outputSize: outputSize,
            layers: layers);
        return new NeuralNetwork<T>(architecture, _options.OptimizerFactory?.Invoke() ?? CreateNetworkOptimizer(LearningRate), new MeanSquaredErrorLoss<T>());
    }

    /// <summary>
    /// Assigns a deterministic, reproducible <see cref="LayerBase{T}.RandomSeed"/>
    /// to each layer (base seed + index) so lazy weight initialization is
    /// identical across agent instances built from the same options — required
    /// for Clone() to reproduce the original's policy.
    /// </summary>
    private static void AssignDeterministicSeeds(List<ILayer<T>> layers, int baseSeed)
    {
        for (int i = 0; i < layers.Count; i++)
        {
            if (layers[i] is LayerBase<T> layer)
            {
                layer.RandomSeed = baseSeed + i;
            }
        }
    }

    private NeuralNetwork<T> CreateDecoderNetwork(int inputSize, int outputSize)
    {
        // Explicit layers (mirror of the encoder, reversed) — see
        // CreateEncoderNetwork for why we must not let the architecture
        // auto-generate default layers. The sigmoid output matches the paper's
        // pixel intensities in [0, 1].
        var reversedChannels = new List<int>(_options.VAEEncoderChannels);
        reversedChannels.Reverse();

        var layers = new List<ILayer<T>>();
        foreach (var channels in reversedChannels)
        {
            layers.Add(new DenseLayer<T>(channels, (IActivationFunction<T>)new ReLUActivation<T>()));
        }
        layers.Add(new DenseLayer<T>(outputSize, (IActivationFunction<T>)new SigmoidActivation<T>()));

        AssignDeterministicSeeds(layers, DecoderSeedBase);

        var architecture = new NeuralNetworkArchitecture<T>(
            inputType: InputType.OneDimensional,
            taskType: NeuralNetworkTaskType.Regression,
            complexity: NetworkComplexity.Medium,
            inputSize: inputSize,
            outputSize: outputSize,
            layers: layers);
        return new NeuralNetwork<T>(architecture, _options.OptimizerFactory?.Invoke() ?? CreateNetworkOptimizer(LearningRate), new MeanSquaredErrorLoss<T>());
    }

    private NeuralNetwork<T> CreateMdnRnnNetwork()
    {
        // [B, T, z + a] -> LSTM x RNNLayers -> [B, T, H] -> linear MDN head -> [B, T, 3 * L * K + 2].
        var layers = new List<ILayer<T>>();
        foreach (var lstm in _lstmLayers) layers.Add(lstm);
        layers.Add(_mdnHead);

        AssignDeterministicSeeds(layers, RNNSeedBase);

        // Declared per sample as [time, features] with the time axis at RNNSequenceLength, the length every training
        // batch is padded to, so the layer shapes the network infers match what training actually feeds it.
        var architecture = new NeuralNetworkArchitecture<T>(
            inputType: InputType.TwoDimensional,
            taskType: NeuralNetworkTaskType.Regression,
            complexity: NetworkComplexity.Medium,
            inputHeight: _options.RNNSequenceLength,
            inputWidth: _options.LatentSize + _options.ActionSize,
            outputSize: MdnOutputSize,
            layers: layers);
        return new NeuralNetwork<T>(architecture, _options.OptimizerFactory?.Invoke() ?? CreateNetworkOptimizer(LearningRate), new MeanSquaredErrorLoss<T>());
    }

    private Vector<T> InitializeControllerParameters()
    {
        // Small uniform weights from a fixed seed, so two agents built from the same options start from the same
        // controller (Clone reproduces the policy) while different inputs still give different actions.
        var random = RandomHelper.CreateSeededRandom(ControllerSeedBase);
        var parameters = new Vector<T>(ControllerParameterCount(_controllerLayerSizes));
        for (int i = 0; i < parameters.Length; i++)
        {
            parameters[i] = NumOps.FromDouble((random.NextDouble() - 0.5) * 0.1);
        }
        return parameters;
    }

    private static int ControllerParameterCount(int[] sizes)
    {
        int count = 0;
        for (int i = 0; i + 1 < sizes.Length; i++) count += sizes[i] * sizes[i + 1] + sizes[i + 1];
        return count;
    }

    /// <summary>The number of controller parameters (867 for the paper's Car Racing configuration).</summary>
    internal int ControllerParameterLength => _controllerParameters.Length;

    /// <summary>The stage the next <see cref="Train"/> call runs.</summary>
    internal WorldModelsTrainingStage CurrentStage => _cycle.Stage;

    /// <summary>A copy of the controller's flattened parameters.</summary>
    internal Vector<T> GetControllerParameters() => _controllerParameters.Clone();

    /// <summary>Replaces the controller's flattened parameters.</summary>
    internal void SetControllerParameters(Vector<T> parameters)
    {
        Guard.NotNull(parameters);
        if (parameters.Length != _controllerParameters.Length)
            throw new ArgumentException($"Expected {_controllerParameters.Length} controller parameters, got {parameters.Length}.", nameof(parameters));
        _controllerParameters = parameters.Clone();
    }

    // ===== Acting =====

    /// <inheritdoc />
    /// <remarks>
    /// Encodes the observation with V (a posterior sample when <paramref name="training"/>, the mean otherwise),
    /// computes a_t = C([z_t, h_t]), then advances M by one step on [z_t, a_t] so the next call sees h_{t+1}.
    /// </remarks>
    public override Vector<T> SelectAction(Vector<T> observation, bool training = true)
    {
        Guard.NotNull(observation);
        if (observation.Length != ObservationSize)
            throw new ArgumentException($"Observation length must be {ObservationSize}, got {observation.Length}.", nameof(observation));

        var batch = new Tensor<T>(new[] { 1, ObservationSize });
        for (int i = 0; i < ObservationSize; i++) batch[i] = observation[i];
        var (mean, logVar) = Encode(batch);
        var z = training ? SampleLatent(mean, logVar) : mean;

        var action = ControllerForward(BuildController(_controllerParameters), z, _controllerMemory);
        _controllerMemory = StepMemory(z, action, _hiddenStates, _cellStates);

        var result = new Vector<T>(_options.ActionSize);
        for (int i = 0; i < _options.ActionSize; i++) result[i] = action[i];
        return result;
    }

    /// <inheritdoc />
    public override void StoreExperience(Vector<T> observation, Vector<T> action, T reward, Vector<T> nextObservation, bool done)
    {
        // Validate transition shapes at this public input boundary so a malformed experience can't
        // enter the replay buffer and later overflow the batches built in Train().
        if (observation.Length != ObservationSize)
            throw new ArgumentException($"Observation length must be {ObservationSize}, got {observation.Length}.", nameof(observation));
        if (nextObservation.Length != ObservationSize)
            throw new ArgumentException($"Next observation length must be {ObservationSize}, got {nextObservation.Length}.", nameof(nextObservation));
        if (action.Length != _options.ActionSize)
            throw new ArgumentException($"Action length must be {_options.ActionSize}, got {action.Length}.", nameof(action));

        _replayBuffer.Add(new Experience<T, Vector<T>, Vector<T>>(observation, action, reward, nextObservation, done));

        if (done) ResetMemory();
    }

    private void ResetMemory()
    {
        _hiddenStates = new Tensor<T>?[_options.RNNLayers];
        _cellStates = new Tensor<T>?[_options.RNNLayers];
        _controllerMemory = new Tensor<T>(new[] { 1, _options.RNNHiddenSize });
    }

    // ===== Training =====

    /// <inheritdoc />
    /// <remarks>Advances the V → M → C cycle by one unit; see the class remarks for exactly what each stage does
    /// per call.</remarks>
    public override T Train()
    {
        int batchSize = _options.BatchSize;
        if (batchSize <= 0 || _replayBuffer.Count < batchSize)
        {
            return NumOps.Zero;
        }

        // A cycle restored into an agent whose replay buffer holds fewer transitions than it was drawn over (a clone
        // refilled from scratch) cannot index its epoch order, so that stage's epochs start again over what is stored.
        if (!_cycle.StagePrepared || !EpochOrderFitsBuffer()) PrepareStage();

        T loss;
        switch (_cycle.Stage)
        {
            case WorldModelsTrainingStage.Vision:
                loss = TrainVisionStep(NextBatch(batchSize));
                if (_cycle.EpochsCompleted >= _options.VAEEpochs) EnterStage(WorldModelsTrainingStage.Memory);
                break;
            case WorldModelsTrainingStage.Memory:
                loss = TrainMemoryStep(NextBatch(batchSize));
                if (_cycle.EpochsCompleted >= _options.RNNEpochs) EnterStage(WorldModelsTrainingStage.Controller);
                break;
            default:
                if (EvolveControllerGeneration(out loss)) EnterStage(WorldModelsTrainingStage.Vision);
                break;
        }

        _updateCount++;
        return loss;
    }

    private bool EpochOrderFitsBuffer()
    {
        foreach (int slot in _cycle.EpochOrder)
        {
            if (slot >= _replayBuffer.Count) return false;
        }
        return true;
    }

    private void EnterStage(WorldModelsTrainingStage stage)
    {
        _cycle.Stage = stage;
        _cycle.StagePrepared = false;
    }

    /// <summary>
    /// Fixes the data a stage's epochs run over: every stored transition for V, the episode-respecting sequence
    /// windows for M (identified by their first slot).
    /// </summary>
    private void PrepareStage()
    {
        _cycle.EpochOrder = _cycle.Stage switch
        {
            WorldModelsTrainingStage.Vision => Enumerable.Range(0, _replayBuffer.Count).ToList(),
            WorldModelsTrainingStage.Memory => SequenceWindowStarts(),
            _ => new List<int>(),
        };
        Shuffle(_cycle.EpochOrder);
        _cycle.EpochCursor = 0;
        _cycle.EpochsCompleted = 0;
        _cycle.StagePrepared = true;
    }

    /// <summary>Cuts the stored timeline, oldest first, into windows of at most RNNSequenceLength that end at an
    /// episode end, and returns the physical slot each window starts at.</summary>
    private List<int> SequenceWindowStarts()
    {
        var starts = new List<int>();
        int count = _replayBuffer.Count;
        bool wrapped = count >= _replayBuffer.Capacity;
        int oldest = wrapped ? _replayBuffer.NextWriteSlot : 0;
        int logical = 0;
        while (logical < count)
        {
            int physical = (oldest + logical) % _replayBuffer.Capacity;
            starts.Add(physical);
            int length = _replayBuffer.SequenceFromSlot(physical, _options.RNNSequenceLength).Count;
            logical += Math.Max(1, length);
        }
        return starts;
    }

    /// <summary>The next items of the current epoch; reaching the end of the epoch counts it and reshuffles.</summary>
    private List<int> NextBatch(int batchSize)
    {
        var batch = new List<int>(batchSize);
        int take = Math.Min(batchSize, _cycle.EpochOrder.Count - _cycle.EpochCursor);
        for (int i = 0; i < take; i++) batch.Add(_cycle.EpochOrder[_cycle.EpochCursor + i]);
        _cycle.EpochCursor += take;
        if (_cycle.EpochCursor >= _cycle.EpochOrder.Count)
        {
            _cycle.EpochsCompleted++;
            Shuffle(_cycle.EpochOrder);
            _cycle.EpochCursor = 0;
        }
        return batch;
    }

    private void Shuffle(List<int> items)
    {
        for (int i = items.Count - 1; i > 0; i--)
        {
            int j = Random.Next(i + 1);
            (items[i], items[j]) = (items[j], items[i]);
        }
    }

    /// <summary>
    /// One step of V on the ELBO: loss = mean over the batch of ||x - D(mu + sigma * eps)||^2
    /// + VAEBeta * KL(N(mu, sigma^2) || N(0, I)).
    /// </summary>
    /// <remarks>
    /// Encoder and decoder each step their own optimizer on the gradient of this one loss at the same noise eps
    /// and the same pre-step weights: the encoder's step differentiates through the decoder (reparameterisation),
    /// and the decoder's step sees the z that pre-step encoder produced, which is the decoder's exact gradient of the
    /// joint loss because the KL term does not depend on the decoder.
    /// </remarks>
    private T TrainVisionStep(List<int> slots)
    {
        int n = slots.Count;
        int obsSize = ObservationSize;
        int latent = _options.LatentSize;

        var observations = new Tensor<T>(new[] { n, obsSize });
        for (int b = 0; b < n; b++)
        {
            var state = _replayBuffer.SequenceFromSlot(slots[b], 1)[0].State;
            for (int j = 0; j < obsSize; j++) observations[b * obsSize + j] = state[j];
        }

        var noise = GaussianTensor(new[] { n, latent });
        var (mean, logVar) = Encode(observations);
        var z = Reparameterize(mean, logVar, noise);

        T beta = NumOps.FromDouble(_options.VAEBeta);
        T inverseBatch = NumOps.FromDouble(1.0 / n);
        var decoder = _vaeDecoder;

        T loss = _vaeEncoder.TrainWithCustomLoss(observations, encoded =>
        {
            var mu = Engine.TensorNarrow(encoded, 1, 0, latent);
            var lv = Engine.TensorNarrow(encoded, 1, latent, latent);
            var std = Engine.TensorExp(Engine.TensorMultiplyScalar(lv, NumOps.FromDouble(0.5)));
            var sample = Engine.TensorAdd(mu, Engine.TensorMultiply(std, noise));
            var reconstruction = decoder.ForwardForTraining(sample);
            return VaeObjective(reconstruction, observations, mu, lv, beta, inverseBatch);
        });

        _vaeDecoder.TrainWithCustomLoss(z, reconstruction =>
            ReconstructionTerm(reconstruction, observations, inverseBatch));

        return loss;
    }

    private Tensor<T> ReconstructionTerm(Tensor<T> reconstruction, Tensor<T> observations, T inverseBatch)
    {
        var error = Engine.TensorSquare(Engine.TensorSubtract(reconstruction, observations));
        return Engine.TensorMultiplyScalar(Engine.ReduceSum(error, null, keepDims: false), inverseBatch);
    }

    /// <summary>The VAE loss of the paper: L2 reconstruction plus beta times the KL to the unit Gaussian,
    /// averaged over the batch.</summary>
    internal Tensor<T> VaeObjective(Tensor<T> reconstruction, Tensor<T> observations, Tensor<T> mu, Tensor<T> logVar,
        T beta, T inverseBatch)
    {
        var reconstructionTerm = ReconstructionTerm(reconstruction, observations, inverseBatch);

        // KL(N(mu, sigma^2) || N(0, 1)) = -1/2 * sum(1 + logvar - mu^2 - exp(logvar)).
        var inner = Engine.TensorSubtract(
            Engine.TensorSubtract(Engine.TensorAddScalar(logVar, NumOps.One), Engine.TensorSquare(mu)),
            Engine.TensorExp(logVar));
        var kl = Engine.TensorMultiplyScalar(Engine.ReduceSum(inner, null, keepDims: false),
            NumOps.Multiply(NumOps.FromDouble(-0.5), inverseBatch));

        return Engine.TensorAdd(reconstructionTerm, Engine.TensorMultiplyScalar(kl, beta));
    }

    /// <summary>
    /// One BPTT step of M on the next batch of sequence windows: the mixture negative log-likelihood of z_{t+1}
    /// (averaged over latent dimensions), plus the squared reward error and the done cross-entropy, averaged over
    /// the valid steps.
    /// </summary>
    private T TrainMemoryStep(List<int> windowStarts)
    {
        var windows = new List<List<Experience<T, Vector<T>, Vector<T>>>>();
        foreach (var start in windowStarts)
        {
            var window = _replayBuffer.SequenceFromSlot(start, _options.RNNSequenceLength);
            if (window.Count > 0) windows.Add(window);
        }
        if (windows.Count == 0) return NumOps.Zero;

        int n = windows.Count;
        // Every batch is padded to RNNSequenceLength (the shape M was resolved at); the mask drops padded steps, and
        // because the LSTM is causal the padding after a window's end never reaches its valid outputs.
        int steps = _options.RNNSequenceLength;
        int latent = _options.LatentSize;
        int actionSize = _options.ActionSize;
        int obsSize = ObservationSize;
        int total = windows.Sum(w => w.Count);

        // Encode every state and next state once, then draw z from V's posterior (fresh noise on every step,
        // as the paper resamples z from the stored mu and sigma for each batch).
        var states = new Tensor<T>(new[] { total, obsSize });
        var nextStates = new Tensor<T>(new[] { total, obsSize });
        int row = 0;
        foreach (var window in windows)
        {
            foreach (var experience in window)
            {
                for (int j = 0; j < obsSize; j++)
                {
                    states[row * obsSize + j] = experience.State[j];
                    nextStates[row * obsSize + j] = experience.NextState[j];
                }
                row++;
            }
        }

        var (stateMean, stateLogVar) = Encode(states);
        var (nextMean, nextLogVar) = Encode(nextStates);
        var z = SampleLatent(stateMean, stateLogVar);
        var zNext = SampleLatent(nextMean, nextLogVar);

        var inputs = new Tensor<T>(new[] { n, steps, latent + actionSize });
        var zTarget = new Tensor<T>(new[] { n, steps, latent });
        var rewardTarget = new Tensor<T>(new[] { n, steps, 1 });
        var doneTarget = new Tensor<T>(new[] { n, steps, 1 });
        var mask = new Tensor<T>(new[] { n, steps, 1 });

        row = 0;
        for (int b = 0; b < n; b++)
        {
            var window = windows[b];
            for (int t = 0; t < window.Count; t++)
            {
                var experience = window[t];
                int cell = b * steps + t;
                for (int d = 0; d < latent; d++)
                {
                    inputs[cell * (latent + actionSize) + d] = z[row * latent + d];
                    zTarget[cell * latent + d] = zNext[row * latent + d];
                }
                for (int a = 0; a < actionSize; a++)
                {
                    inputs[cell * (latent + actionSize) + latent + a] = experience.Action[a];
                }
                rewardTarget[cell] = experience.Reward;
                doneTarget[cell] = experience.Done ? NumOps.One : NumOps.Zero;
                mask[cell] = NumOps.One;
                row++;
            }
        }

        T inverseValid = NumOps.FromDouble(1.0 / total);
        return _mdnRnn.TrainWithCustomLoss(inputs, output =>
            MdnObjective(output, zTarget, rewardTarget, doneTarget, mask, inverseValid));
    }

    /// <summary>
    /// M's loss on head output <c>[B, T, 3LK + 2]</c>: per step, the mean over latent dimensions of
    /// -log sum_k pi_k N(z_{t+1}; mu_k, sigma_k), plus (r_hat - r)^2, plus the logistic loss of the done flag; masked
    /// and averaged over the valid steps.
    /// </summary>
    internal Tensor<T> MdnObjective(Tensor<T> output, Tensor<T> zTarget, Tensor<T> rewardTarget, Tensor<T> doneTarget,
        Tensor<T> mask, T inverseValid)
    {
        int b = output.Shape[0];
        int steps = output.Shape[1];
        int latent = _options.LatentSize;
        int mixtures = _options.NumMixtures;
        int block = MixtureBlock;
        var mixtureShape = new[] { b, steps, latent, mixtures };

        var logits = Engine.Reshape(Engine.TensorNarrow(output, 2, 0, block), mixtureShape);
        var means = Engine.Reshape(Engine.TensorNarrow(output, 2, block, block), mixtureShape);
        var logSigma = Engine.Reshape(Engine.TensorNarrow(output, 2, 2 * block, block), mixtureShape);
        var rewardPrediction = Engine.TensorNarrow(output, 2, 3 * block, 1);
        var doneLogit = Engine.TensorNarrow(output, 2, 3 * block + 1, 1);

        // log N(z; mu, sigma) = -1/2 ((z - mu) / sigma)^2 - log sigma - 1/2 log(2 pi).
        var target = Engine.TensorBroadcastTo(Engine.Reshape(zTarget, new[] { b, steps, latent, 1 }), mixtureShape);
        var standardized = Engine.TensorMultiply(Engine.TensorSubtract(target, means),
            Engine.TensorExp(Engine.TensorNegate(logSigma)));
        var logDensity = Engine.TensorAddScalar(
            Engine.TensorSubtract(Engine.TensorMultiplyScalar(Engine.TensorSquare(standardized), NumOps.FromDouble(-0.5)), logSigma),
            NumOps.FromDouble(-0.5 * Math.Log(2.0 * Math.PI)));

        // -log sum_k softmax(logits)_k N_k = LSE(logits) - LSE(logits + log N).
        var negativeLogLikelihood = Engine.TensorSubtract(
            LogSumExpLastAxis(logits),
            LogSumExpLastAxis(Engine.TensorAdd(logits, logDensity)));
        var latentTerm = Engine.TensorMultiplyScalar(
            Engine.ReduceSum(negativeLogLikelihood, new[] { 2 }, keepDims: true),
            NumOps.FromDouble(1.0 / latent));

        var rewardTerm = Engine.TensorSquare(Engine.TensorSubtract(rewardPrediction, rewardTarget));

        // Logistic loss: log(1 + e^l) - y l.
        var doneTerm = Engine.TensorSubtract(
            Engine.TensorLog(Engine.TensorAddScalar(Engine.TensorExp(doneLogit), NumOps.One)),
            Engine.TensorMultiply(doneTarget, doneLogit));

        var perStep = Engine.TensorMultiply(Engine.TensorAdd(Engine.TensorAdd(latentTerm, rewardTerm), doneTerm), mask);
        return Engine.TensorMultiplyScalar(Engine.ReduceSum(perStep, null, keepDims: false), inverseValid);
    }

    /// <summary>Tape-tracked, numerically stable log-sum-exp over the last axis (removed from the shape).</summary>
    private Tensor<T> LogSumExpLastAxis(Tensor<T> x)
    {
        int axis = x.Shape.Length - 1;
        var max = Engine.ReduceMax(x, new[] { axis }, keepDims: true, out _);
        var shifted = Engine.TensorSubtract(x, Engine.TensorBroadcastTo(max, x.Shape.ToArray()));
        var sum = Engine.ReduceSum(Engine.TensorExp(shifted), new[] { axis }, keepDims: true);
        var result = Engine.TensorAdd(Engine.TensorLog(sum), max);
        var outShape = new int[axis];
        for (int i = 0; i < axis; i++) outShape[i] = x.Shape[i];
        return Engine.Reshape(result, outShape);
    }

    // ===== Controller evolution in the dream =====

    /// <summary>
    /// Runs one CMA-ES generation (the library's Hansen implementation, through its ask/tell interface) in the dream:
    /// samples ControllerPopulationSize controllers around the current search distribution, scores each by its mean
    /// dream return, updates the distribution, and installs the distribution's mean as the controller. The search
    /// state is kept between calls, so successive Controller-stage calls continue one search; it starts from the
    /// current controller at the stage's first generation.
    /// </summary>
    /// <returns><c>true</c> once ControllerGenerations generations have run (or the search's step size has
    /// collapsed), ending the stage.</returns>
    private bool EvolveControllerGeneration(out T loss)
    {
        var cma = CMAESOptimizer<T, Vector<T>, Vector<T>>.CreateForFunction(
            new CMAESOptimizerOptions<T, Vector<T>, Vector<T>>
            {
                PopulationSize = _options.ControllerPopulationSize,
                InitialStepSize = _options.ControllerInitialStepSize,
            });

        var search = _cycle.ControllerSearch ?? cma.CreateSearchState(_controllerParameters);
        _cycle.ControllerSearch = search;

        var generation = cma.Ask(search, Random);
        var fitness = new T[generation.Points.Count];
        T totalReturn = NumOps.Zero;
        for (int k = 0; k < fitness.Length; k++)
        {
            T dreamReturn = DreamReturn(generation.Points[k]);
            totalReturn = NumOps.Add(totalReturn, dreamReturn);
            // CMA-ES minimises; the paper's fitness is the cumulative reward, so its negation is minimised.
            fitness[k] = NumOps.Negate(dreamReturn);
        }
        cma.Tell(search, generation, fitness);

        // The controller is the distribution's mean, the estimate CMA-ES itself recommends under a noisy fitness.
        var mean = new Vector<T>(search.Dimension);
        for (int i = 0; i < mean.Length; i++) mean[i] = NumOps.FromDouble(search.Mean[i]);
        _controllerParameters = mean;

        _lastDreamReturn = NumOps.Divide(totalReturn, NumOps.FromDouble(fitness.Length));
        loss = NumOps.Negate(_lastDreamReturn);

        bool finished = search.Generation >= _options.ControllerGenerations || search.IsDegenerate;
        if (finished) _cycle.ControllerSearch = null;
        return finished;
    }

    /// <summary>
    /// The mean cumulative reward of ControllerEvaluationRollouts dream rollouts of the given controller. Each starts
    /// from z sampled from V's posterior of a random stored observation with M's state at zero, and runs for up to
    /// RolloutLength steps: a_t = C([z_t, h_t]); M steps on [z_t, a_t]; z_{t+1} is sampled from the mixture at
    /// temperature tau; the predicted reward is accumulated; the rollout ends once M predicts done with
    /// probability above 50% (appendix, "MDN-RNN").
    /// </summary>
    internal T DreamReturn(Vector<T> controllerParameters)
    {
        int rollouts = _options.ControllerEvaluationRollouts;
        int obsSize = ObservationSize;
        int latent = _options.LatentSize;
        int block = MixtureBlock;
        int outputSize = MdnOutputSize;

        var starts = new Tensor<T>(new[] { rollouts, obsSize });
        for (int b = 0; b < rollouts; b++)
        {
            var state = _replayBuffer.SequenceFromSlot(Random.Next(_replayBuffer.Count), 1)[0].State;
            for (int j = 0; j < obsSize; j++) starts[b * obsSize + j] = state[j];
        }
        var (mean, logVar) = Encode(starts);
        var z = SampleLatent(mean, logVar);

        var controller = BuildController(controllerParameters);
        var hidden = new Tensor<T>?[_options.RNNLayers];
        var cell = new Tensor<T>?[_options.RNNLayers];
        var memory = new Tensor<T>(new[] { rollouts, _options.RNNHiddenSize });
        var alive = new bool[rollouts];
        for (int b = 0; b < rollouts; b++) alive[b] = true;
        var returns = new T[rollouts];
        for (int b = 0; b < rollouts; b++) returns[b] = NumOps.Zero;
        int remaining = rollouts;

        var weights = new double[_options.NumMixtures];
        for (int step = 0; step < _options.RolloutLength && remaining > 0; step++)
        {
            var action = ControllerForward(controller, z, memory);
            var top = StepMemory(z, action, hidden, cell);
            memory = top;
            var output = _mdnHead.Forward(Engine.Reshape(top, new[] { rollouts, 1, _options.RNNHiddenSize }));

            var nextZ = new Tensor<T>(new[] { rollouts, latent });
            for (int b = 0; b < rollouts; b++)
            {
                int offset = b * outputSize;
                for (int d = 0; d < latent; d++)
                {
                    nextZ[b * latent + d] = SampleMixtureLatent(output, offset, d, weights);
                }

                if (!alive[b]) continue;
                returns[b] = NumOps.Add(returns[b], output[offset + 3 * block]);
                // sigmoid(l) > 0.5 exactly when l > 0.
                if (NumOps.GreaterThan(output[offset + 3 * block + 1], NumOps.Zero))
                {
                    alive[b] = false;
                    remaining--;
                }
            }
            z = nextZ;
            DreamStepsSimulated += rollouts;
        }

        DreamEvaluations++;

        T sum = NumOps.Zero;
        for (int b = 0; b < rollouts; b++) sum = NumOps.Add(sum, returns[b]);
        return NumOps.Divide(sum, NumOps.FromDouble(rollouts));
    }

    /// <summary>The number of controllers scored in the dream so far.</summary>
    internal int DreamEvaluations { get; private set; }

    /// <summary>The number of single-rollout dream steps simulated so far.</summary>
    internal long DreamStepsSimulated { get; private set; }

    /// <summary>
    /// Samples latent dimension <paramref name="dimension"/> of z_{t+1} from M's mixture output (the row starting at
    /// <paramref name="offset"/>) at temperature tau: a component k ~ softmax(logits / tau), then
    /// z = mu_k + sigma_k * sqrt(tau) * eps.
    /// </summary>
    /// <remarks>
    /// This is the authors' reference implementation (hardmaru/WorldModelsExperiments). carracing/rnn/rnn.py,
    /// <c>sample_sequence</c>: <c>logmix2 = np.copy(logmix)/temperature</c>, then
    /// <c>rand_gaussian = np.random.randn(OUTWIDTH)*np.sqrt(temperature)</c> and
    /// <c>next_x = chosen_mean+np.exp(chosen_logstd)*rand_gaussian</c>. doomrnn/doomrnn.py, <c>DoomCoverRNNEnv._step</c>,
    /// does the same (<c>logmix2 = np.copy(logmix)/temperature</c>;
    /// <c>rand_gaussian = self.np_random.randn(OUTWIDTH)*np.sqrt(temperature)</c>;
    /// <c>next_z = chosen_mean+np.exp(chosen_logstd)*rand_gaussian</c>) and ends the dream when
    /// <c>logrestart[0] &gt; 0</c>, i.e. P(done) &gt; 0.5, as <see cref="DreamReturn"/> does.
    /// </remarks>
    internal T SampleMixtureLatent(Tensor<T> output, int offset, int dimension, double[]? weights = null)
    {
        int mixtures = _options.NumMixtures;
        int block = MixtureBlock;
        double temperature = _options.Temperature;
        weights ??= new double[mixtures];
        int component = SampleComponent(output, offset + dimension * mixtures, mixtures, temperature, weights);
        T mu = output[offset + block + dimension * mixtures + component];
        T sigma = NumOps.Exp(output[offset + 2 * block + dimension * mixtures + component]);
        T scale = NumOps.Multiply(sigma, NumOps.FromDouble(Math.Sqrt(temperature)));
        return NumOps.Add(mu, NumOps.Multiply(scale, NumOps.FromDouble(NextGaussian())));
    }

    /// <summary>Samples a mixture component from softmax(logits / tau).</summary>
    private int SampleComponent(Tensor<T> output, int offset, int mixtures, double temperature, double[] weights)
    {
        double max = double.NegativeInfinity;
        for (int k = 0; k < mixtures; k++)
        {
            weights[k] = NumOps.ToDouble(output[offset + k]) / temperature;
            if (weights[k] > max) max = weights[k];
        }
        double total = 0.0;
        for (int k = 0; k < mixtures; k++)
        {
            weights[k] = Math.Exp(weights[k] - max);
            total += weights[k];
        }
        double u = Random.NextDouble() * total;
        for (int k = 0; k < mixtures - 1; k++)
        {
            u -= weights[k];
            if (u < 0.0) return k;
        }
        return mixtures - 1;
    }

    // ===== Shared pieces =====

    /// <summary>Encodes <c>[N, obs]</c> observations into V's posterior mean and log-variance, each <c>[N, L]</c>.</summary>
    internal (Tensor<T> Mean, Tensor<T> LogVar) Encode(Tensor<T> observations)
    {
        int n = observations.Shape[0];
        int latent = _options.LatentSize;
        var encoded = _vaeEncoder.Predict(observations);
        var mean = new Tensor<T>(new[] { n, latent });
        var logVar = new Tensor<T>(new[] { n, latent });
        for (int b = 0; b < n; b++)
        {
            for (int d = 0; d < latent; d++)
            {
                mean[b * latent + d] = encoded[b * 2 * latent + d];
                logVar[b * latent + d] = encoded[b * 2 * latent + latent + d];
            }
        }
        return (mean, logVar);
    }

    private Tensor<T> SampleLatent(Tensor<T> mean, Tensor<T> logVar)
        => Reparameterize(mean, logVar, GaussianTensor(mean.Shape.ToArray()));

    private Tensor<T> Reparameterize(Tensor<T> mean, Tensor<T> logVar, Tensor<T> noise)
    {
        var z = new Tensor<T>(mean.Shape.ToArray());
        T half = NumOps.FromDouble(0.5);
        for (int i = 0; i < z.Length; i++)
        {
            T std = NumOps.Exp(NumOps.Multiply(logVar[i], half));
            z[i] = NumOps.Add(mean[i], NumOps.Multiply(std, noise[i]));
        }
        return z;
    }

    private Tensor<T> GaussianTensor(int[] shape)
    {
        var tensor = new Tensor<T>(shape);
        for (int i = 0; i < tensor.Length; i++) tensor[i] = NumOps.FromDouble(NextGaussian());
        return tensor;
    }

    /// <summary>A standard normal draw from the agent's (seedable) generator, by Box-Muller.</summary>
    private double NextGaussian()
    {
        double u1 = 1.0 - Random.NextDouble();
        double u2 = Random.NextDouble();
        return Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2);
    }

    /// <summary>Unflattens controller parameters into per-layer weight <c>[in, out]</c> and bias <c>[1, out]</c>.</summary>
    private List<(Tensor<T> Weights, Tensor<T> Bias)> BuildController(Vector<T> parameters)
    {
        if (parameters.Length != ControllerParameterCount(_controllerLayerSizes))
            throw new ArgumentException("The controller parameter vector has the wrong length.", nameof(parameters));

        var layers = new List<(Tensor<T>, Tensor<T>)>();
        int offset = 0;
        for (int l = 0; l + 1 < _controllerLayerSizes.Length; l++)
        {
            int input = _controllerLayerSizes[l];
            int output = _controllerLayerSizes[l + 1];
            var weights = new Tensor<T>(new[] { input, output });
            for (int i = 0; i < input * output; i++) weights[i] = parameters[offset + i];
            offset += input * output;
            var bias = new Tensor<T>(new[] { 1, output });
            for (int i = 0; i < output; i++) bias[i] = parameters[offset + i];
            offset += output;
            layers.Add((weights, bias));
        }
        return layers;
    }

    /// <summary>C: a = tanh(... tanh([z, h] W_1 + b_1) ... W_n + b_n), on <c>[B, L]</c> and <c>[B, H]</c>.</summary>
    private Tensor<T> ControllerForward(List<(Tensor<T> Weights, Tensor<T> Bias)> controller, Tensor<T> z, Tensor<T> memory)
    {
        var x = Engine.TensorConcatenate(new[] { z, memory }, axis: 1);
        foreach (var (weights, bias) in controller)
        {
            var linear = Engine.TensorMatMul(x, weights);
            linear = Engine.TensorAdd(linear, Engine.TensorBroadcastTo(bias, linear.Shape.ToArray()));
            x = Engine.TensorTanh(linear);
        }
        return x;
    }

    /// <summary>
    /// Steps M's LSTM stack once on <c>[z, a]</c> from the carried per-layer (h, c), updating them in place, and
    /// returns the top layer's new h, <c>[B, H]</c>.
    /// </summary>
    private Tensor<T> StepMemory(Tensor<T> z, Tensor<T> action, Tensor<T>?[] hidden, Tensor<T>?[] cell)
    {
        int b = z.Shape[0];
        var x = Engine.Reshape(Engine.TensorConcatenate(new[] { z, action }, axis: 1),
            new[] { b, 1, _options.LatentSize + _options.ActionSize });
        for (int i = 0; i < _lstmLayers.Count; i++)
        {
            x = _lstmLayers[i].ForwardFromState(x, hidden[i], cell[i], out var finalHidden, out var finalCell);
            hidden[i] = finalHidden;
            cell[i] = finalCell;
        }
        return Engine.Reshape(x, new[] { b, _options.RNNHiddenSize });
    }

    /// <summary>M's mixture-density head.</summary>
    internal DenseLayer<T> MixtureHead => _mdnHead;

    /// <summary>M's LSTM layers, bottom first.</summary>
    internal IReadOnlyList<LSTMLayer<T>> MemoryLayers => _lstmLayers;

    /// <summary>The Controller stage's CMA-ES search, or null outside the stage.</summary>
    internal CMAESSearchState? ControllerSearch => _cycle.ControllerSearch;

    /// <summary>The number of sequence windows M's epochs currently run over.</summary>
    internal int SequenceWindowCount => SequenceWindowStarts().Count;

    /// <inheritdoc />
    public override Dictionary<string, T> GetMetrics()
    {
        return new Dictionary<string, T>
        {
            ["updates"] = NumOps.FromDouble(_updateCount),
            ["buffer_size"] = NumOps.FromDouble(_replayBuffer.Count),
            ["stage"] = NumOps.FromDouble((int)_cycle.Stage),
            ["dream_return"] = _lastDreamReturn
        };
    }

    /// <inheritdoc />
    public override void ResetEpisode()
    {
        ResetMemory();
    }

    /// <inheritdoc />
    public override Vector<T> Predict(Vector<T> input)
    {
        // Predict is the pure inference contract and MUST be side-effect free:
        // SelectAction advances M's recurrent state for sequential rollout, so
        // snapshot it before and restore it after, keeping repeated Predict
        // calls deterministic (Policy_ShouldBeDeterministic) and independent of
        // prior inference calls.
        var savedHidden = (Tensor<T>?[])_hiddenStates.Clone();
        var savedCell = (Tensor<T>?[])_cellStates.Clone();
        var savedMemory = _controllerMemory;

        var action = SelectAction(input, training: false);

        _hiddenStates = savedHidden;
        _cellStates = savedCell;
        _controllerMemory = savedMemory;
        return action;
    }

    /// <summary>Asynchronously predicts the deterministic action for an observation.</summary>
    public Task<Vector<T>> PredictAsync(Vector<T> input)
    {
        return Task.FromResult(Predict(input));
    }

    /// <summary>Asynchronously runs one <see cref="Train"/> call.</summary>
    public Task TrainAsync()
    {
        Train();
        return Task.CompletedTask;
    }

    /// <inheritdoc />
    public override ModelMetadata<T> GetModelMetadata()
    {
        return new ModelMetadata<T>
        {
        };
    }

    /// <inheritdoc />
    public override int FeatureCount => ObservationSize;

    /// <inheritdoc />
    public override void SaveModel(string filepath)
    {
        var data = Serialize();
        System.IO.File.WriteAllBytes(filepath, data);
    }

    /// <inheritdoc />
    public override void LoadModel(string filepath)
    {
        var data = System.IO.File.ReadAllBytes(filepath);
        Deserialize(data);
    }
}
