using AiDotNet.ActivationFunctions;
using AiDotNet.Attributes;
using AiDotNet.Autodiff;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.NeuralNetworks.SyntheticData;
using AiDotNet.Optimizers;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Training;

namespace AiDotNet.NeuralNetworks.SyntheticData;

/// <summary>
/// TimeGAN generator for synthesizing realistic time-series tabular data while preserving
/// temporal dynamics using an embedding-supervisor-adversarial training framework.
/// </summary>
/// <remarks>
/// <para>
/// TimeGAN uses five jointly trained components in a shared latent space:
///
/// <code>
///  Real Data ──► Embedder ──► H_real ──► Recovery ──► Reconstructed Data
///                                │
///                                ▼
///                          Supervisor ──► H_supervised
///                                │
///  Noise ──────► Generator ──► H_fake ──► Discriminator ──► Real/Fake?
///                                │
///                          Supervisor ──► H_fake_supervised
/// </code>
///
/// Training has three phases:
/// 1. <b>Embedding phase</b>: Train embedder + recovery to reconstruct data
/// 2. <b>Supervised phase</b>: Train supervisor to predict next-step embeddings
/// 3. <b>Joint phase</b>: Train all 5 components together with combined losses
///
/// This implementation follows the standard neural network architecture pattern with:
/// - Proper inheritance from NeuralNetworkBase
/// - Layers = generator network (user-overridable via Architecture)
/// - Auxiliary networks (embedder, recovery, supervisor, discriminator) are internal
/// </para>
/// <para>
/// <b>For Beginners:</b> TimeGAN works by:
///
/// 1. Learning to compress time-series into a simpler space (embedding)
/// 2. Learning how data moves through time in that space (supervisor)
/// 3. Learning to generate realistic fake data using both spatial and temporal info
///
/// Example usage:
/// <code>
/// var architecture = new NeuralNetworkArchitecture&lt;double&gt;(
///     inputFeatures: 64,
///     outputSize: 64
/// );
/// var options = new TimeGANOptions&lt;double&gt;
/// {
///     SequenceLength = 24,
///     HiddenDimension = 64,
///     Epochs = 2000
/// };
/// var timegan = new TimeGANGenerator&lt;double&gt;(architecture, options);
/// timegan.Fit(data, columns, epochs: 2000);
/// var synthetic = timegan.Generate(1000);
/// </code>
/// </para>
/// <para>
/// Reference: "Time-series Generative Adversarial Networks" (Yoon et al., NeurIPS 2019)
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
[ModelDomain(ModelDomain.MachineLearning)]
[ModelCategory(ModelCategory.NeuralNetwork)]
[ModelCategory(ModelCategory.SyntheticDataGenerator)]
[ModelTask(ModelTask.Generation)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("Time-series Generative Adversarial Networks",
    "https://papers.nips.cc/paper/2019/hash/c9efe5f26cd17ba6216bbe2a7d26d490-Abstract.html",
    Year = 2019,
    Authors = "Jinsung Yoon, Daniel Jarrett, Mihaela van der Schaar")]
public partial class TimeGANGenerator<T> : NeuralSyntheticTabularGeneratorBase<T>, ISyntheticTabularGenerator<T>
{
    private readonly TimeGANOptions<T> _options;
    // One dedicated optimizer per training phase (Yoon et al. 2019 uses separate
    // solvers: embedder, supervisor, generator, discriminator). See CTGANGenerator
    // for why a single shared AdamOptimizer corrupts its flat moment buffer across
    // networks of different parameter counts and diverges.
    private IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> _embedderOptimizer;
    private IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> _supervisorOptimizer;
    private IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> _generatorOptimizer;
    private IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> _discriminatorOptimizer;
    private ILossFunction<T> _lossFunction;

    // Synthetic tabular data infrastructure
    private List<ColumnMetadata> _columns = new();
    private int _dataWidth;
    private Random _random;

    // HiddenDimension and NumLayers determine every component's shape. Options remain mutable so a
    // caller can configure a later Fit, but changing them must not reinterpret materialized weights.
    private int _materializedHiddenDimension;
    private int _materializedNumLayers;

    // Embedder (auxiliary): data → latent
    private readonly List<GRULayer<T>> _embedderLayers = new();
    private FullyConnectedLayer<T>? _embedderOutput;
    [Scratch]
    private readonly List<Tensor<T>> _embedderPreActs = new();

    // Recovery (auxiliary): latent → data
    private readonly List<FullyConnectedLayer<T>> _recoveryLayers = new();
    private FullyConnectedLayer<T>? _recoveryOutput;
    [Scratch]
    private readonly List<Tensor<T>> _recoveryPreActs = new();

    // Generator output head and pre-activation cache (Layers = generator recurrent stack)
    private FullyConnectedLayer<T>? _generatorOutput;
    [Scratch]
    private readonly List<Tensor<T>> _generatorPreActs = new();

    // Supervisor (auxiliary): latent_t → latent_{t+1}
    private readonly List<GRULayer<T>> _supervisorLayers = new();
    private FullyConnectedLayer<T>? _supervisorOutput;
    [Scratch]
    private readonly List<Tensor<T>> _supervisorPreActs = new();

    // Discriminator (auxiliary): bidirectional latent sequence → per-step real/fake
    private readonly List<GRULayer<T>> _discriminatorForwardLayers = new();
    private readonly List<GRULayer<T>> _discriminatorBackwardLayers = new();
    private readonly List<DropoutLayer<T>> _discDropoutLayers = new();
    private FullyConnectedLayer<T>? _discriminatorOutput;
    [Scratch]
    private readonly List<Tensor<T>> _discPreActs = new();

    // Whether custom layers are being used
    private bool _usingCustomLayers;

    /// <summary>
    /// Gets the TimeGAN-specific options.
    /// </summary>
    /// <remarks>
    /// Changes to <see cref="TimeGANOptions{T}.HiddenDimension"/> or
    /// <see cref="TimeGANOptions{T}.NumLayers"/> take effect on the next <see cref="Fit"/> call.
    /// Existing trained layers, inference, cloning, serialization, and metadata retain the topology
    /// that was materialized by the most recent construction or fit.
    /// </remarks>
    public TimeGANOptions<T> TimeGanOptions => _options;

    /// <inheritdoc />
    public IReadOnlyList<ColumnMetadata> Columns => _columns.AsReadOnly();

    /// <inheritdoc />
    public bool IsFitted { get; private set; }

    /// <summary>
    /// Initializes a new TimeGAN generator with the specified architecture.
    /// </summary>
    /// <param name="architecture">The neural network architecture defining input/output dimensions and optional custom layers.</param>
    /// <param name="options">TimeGAN-specific options for temporal generation configuration.</param>
    /// <param name="optimizer">Gradient-based optimizer (defaults to Adam).</param>
    /// <param name="lossFunction">Loss function (defaults based on task type).</param>
    /// <param name="maxGradNorm">Maximum gradient norm for clipping (default 5.0).</param>
    /// <summary>
    /// Initializes a new instance with default architecture settings.
    /// </summary>
    public TimeGANGenerator()
        : this(new NeuralNetworkArchitecture<T>(
            inputType: Enums.InputType.OneDimensional,
            taskType: Enums.NeuralNetworkTaskType.Regression,
            inputSize: 10,
            outputSize: 10))
    {
    }

    public TimeGANGenerator(
        NeuralNetworkArchitecture<T> architecture,
        TimeGANOptions<T>? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
        ILossFunction<T>? lossFunction = null,
        double maxGradNorm = 5.0)
        : base(architecture, lossFunction ?? NeuralNetworkHelper<T>.GetDefaultLossFunction(architecture.TaskType), maxGradNorm)
    {
        _options = options ?? new TimeGANOptions<T>();
        ValidateOptions();
        _lossFunction = lossFunction ?? NeuralNetworkHelper<T>.GetDefaultLossFunction(architecture.TaskType);
        AdamOptimizer<T, Tensor<T>, Tensor<T>> MakeAdam() =>
            new(this, new Models.Options.AdamOptimizerOptions<T, Tensor<T>, Tensor<T>>
            {
                InitialLearningRate = _options.LearningRate,
                Beta1 = 0.5,
                Beta2 = 0.9,
                UseAdaptiveLearningRate = false,
                UseAMSGrad = false,
            });
        _generatorOptimizer = optimizer ?? MakeAdam();
        _discriminatorOptimizer = MakeAdam();
        _embedderOptimizer = MakeAdam();
        _supervisorOptimizer = MakeAdam();
        _random = _options.Seed.HasValue
            ? RandomHelper.CreateSeededRandom(_options.Seed.Value)
            : RandomHelper.CreateSecureRandom();

        InitializeLayers();
    }

    private void ValidateOptions()
    {
        if (_options.NumLayers < 1)
        {
            throw new ArgumentOutOfRangeException(
                nameof(TimeGANOptions<T>.NumLayers),
                _options.NumLayers,
                "TimeGAN requires at least one layer.");
        }
    }

    #region Layer Initialization (GANDALF Pattern)

    /// <summary>
    /// Initializes the generator layers (Layers = generator network, user-overridable).
    /// </summary>
    protected override void InitializeLayers()
    {
        int hiddenDim = _options.HiddenDimension;
        int numLayers = _options.NumLayers;

        if (Architecture.Layers is not null && Architecture.Layers.Count > 0)
        {
            Layers.AddRange(Architecture.Layers);
            _generatorOutput = null;
            _usingCustomLayers = true;
        }
        else
        {
            var identity = new IdentityActivation<T>() as IActivationFunction<T>;

            for (int i = 0; i < numLayers; i++)
            {
                Layers.Add(new GRULayer<T>(hiddenDim, returnSequences: true));
            }
            _generatorOutput = new FullyConnectedLayer<T>(hiddenDim, hiddenDim, identity);
            _usingCustomLayers = false;

            // Before fitting, public activation inspection accepts the architecture width. Materialize
            // the lazy recurrent stack with that contract now so ParameterLayout is stable and cannot
            // later bind the generator to whichever probe happens to reach it first.
            int initialNoiseWidth = Architecture.InputSize > 0 ? Architecture.InputSize : hiddenDim;
            _ = GeneratorForwardBatched(new Tensor<T>([1, 1, initialNoiseWidth]), isTraining: false);
        }

        _materializedHiddenDimension = hiddenDim;
        _materializedNumLayers = numLayers;
    }

    private void RebuildAllNetworks()
    {
        int hiddenDim = _options.HiddenDimension;
        int numLayers = _options.NumLayers;
        var identity = new IdentityActivation<T>() as IActivationFunction<T>;

        // Rebuild generator (Layers) if not using custom. TimeGAN's generator is recurrent:
        // every output at t depends on the current noise and the preceding generated state.
        if (!_usingCustomLayers)
        {
            Layers.Clear();
            for (int i = 0; i < numLayers; i++)
            {
                Layers.Add(new GRULayer<T>(hiddenDim, returnSequences: true));
            }
            _generatorOutput = new FullyConnectedLayer<T>(hiddenDim, hiddenDim, identity);
        }
        else
        {
            // A custom architecture remains a complete, caller-owned generator definition.
            _generatorOutput = null;
        }

        // Embedder: paper §4.1 recurrent temporal mapping, followed by a pointwise head.
        _embedderLayers.Clear();
        for (int i = 0; i < numLayers; i++)
        {
            _embedderLayers.Add(new GRULayer<T>(hiddenDim, returnSequences: true));
        }
        _embedderOutput = new FullyConnectedLayer<T>(hiddenDim, hiddenDim, identity);

        // Recovery: latent -> latent hidden stack, then a projection back out to the data width.
        _recoveryLayers.Clear();
        for (int i = 0; i < numLayers; i++)
        {
            _recoveryLayers.Add(new FullyConnectedLayer<T>(hiddenDim, hiddenDim, identity));
        }
        _recoveryOutput = new FullyConnectedLayer<T>(hiddenDim, _dataWidth, identity);

        // Supervisor: the authors use one fewer recurrent layer than the generator. At NumLayers == 1
        // the pointwise output head remains the complete supervisor stage.
        _supervisorLayers.Clear();
        for (int i = 0; i < numLayers - 1; i++)
        {
            _supervisorLayers.Add(new GRULayer<T>(hiddenDim, returnSequences: true));
        }
        _supervisorOutput = new FullyConnectedLayer<T>(hiddenDim, hiddenDim, identity);

        // Discriminator: paper §4.2 uses aligned forward/backward recurrent states and a
        // feedforward per-step classification head. Keep the two directions explicit here so the
        // backward result can be reversed back into the original time order before concatenation.
        _discriminatorForwardLayers.Clear();
        _discriminatorBackwardLayers.Clear();
        _discDropoutLayers.Clear();
        for (int i = 0; i < numLayers; i++)
        {
            _discriminatorForwardLayers.Add(new GRULayer<T>(hiddenDim, returnSequences: true));
            _discriminatorBackwardLayers.Add(new GRULayer<T>(hiddenDim, returnSequences: true));
            _discDropoutLayers.Add(new DropoutLayer<T>(_options.DiscriminatorDropout));
        }
        _discriminatorOutput = new FullyConnectedLayer<T>(2 * hiddenDim, 1, identity);

        // Materialize every lazy recurrent component before any phase collects its parameter list.
        // Otherwise the first tape step would see zero GRU parameters and silently train only heads.
        int probeLength = Math.Max(2, _options.SequenceLength);
        var hiddenProbe = new Tensor<T>([1, probeLength, hiddenDim]);
        _ = EmbedderForwardBatched(new Tensor<T>([1, probeLength, _dataWidth]), isTraining: false);
        if (!_usingCustomLayers)
        {
            _ = GeneratorForwardBatched(hiddenProbe, isTraining: false);
        }
        _ = SupervisorForwardBatched(hiddenProbe, isTraining: false);
        _ = RecoveryForwardBatched(hiddenProbe, isTraining: false);
        _ = DiscriminatorForwardBatched(hiddenProbe, isTraining: false);

        // Commit the snapshot only after every component was rebuilt successfully.
        _materializedHiddenDimension = hiddenDim;
        _materializedNumLayers = numLayers;
    }

    #endregion

    #region ISyntheticTabularGenerator Implementation

    /// <inheritdoc />
    public void Fit(Matrix<T> data, IReadOnlyList<ColumnMetadata> columns, int epochs)
    {
        // Options remain publicly accessible after construction, so validate again before using
        // them to rebuild the component networks.
        ValidateOptions();

        // Reject configurations that would silently no-op training. With
        // epochs <= 0 every phase loop runs zero iterations and IsFitted
        // would still flip true at the bottom (untrained model marked
        // ready for Generate). With SequenceLength < 2 there are no
        // (xt, xt+1) pairs so Phase 2 (supervisor) and the supervised
        // term of Phase 3 (joint) become no-ops — the TimeGAN variant
        // described in the class docs is no longer what's being trained.
        if (epochs <= 0)
            throw new ArgumentOutOfRangeException(nameof(epochs), epochs, "epochs must be greater than 0.");
        if (_options.SequenceLength < 2)
            throw new InvalidOperationException(
                "TimeGAN requires SequenceLength >= 2 for the supervisor + temporal-supervision objectives. "
                + $"Got SequenceLength = {_options.SequenceLength}.");

        _columns = new List<ColumnMetadata>(columns);
        _dataWidth = data.Columns;
        int seqLen = _options.SequenceLength;

        RebuildAllNetworks();

        var sequences = PrepareSequences(data, seqLen);
        if (sequences.Count == 0)
        {
            throw new InvalidOperationException(
                "TimeGAN.Fit: no usable sequences produced from the input data "
                + $"(data rows = {data.Rows}, SequenceLength = {seqLen}). "
                + "Need at least one window of length >= SequenceLength.");
        }
        if (!sequences.Exists(seq => seq.Rows >= 2))
        {
            throw new InvalidOperationException(
                "TimeGAN.Fit: every prepared sequence has fewer than two timesteps, "
                + "so the supervisor's next-step objective can never produce a pair. "
                + "Increase the input row count or lower SequenceLength.");
        }

        int batchSize = Math.Min(_options.BatchSize, sequences.Count);
        // Honor the caller's total epochs budget across the three phases —
        // the prior `phaseDuration = max(1, epochs/3)` formulation ran each
        // phase that many times, so the model actually trained for
        // 3·phaseDuration passes (over-training when epochs < 3 since
        // Math.Max(1, …) floors to 1 per phase, and dropping the remainder
        // for non-multiples of 3). Split as base + remainder distribution.
        int baseEpochs = epochs / 3;
        int remainder = epochs % 3;
        int phase1Epochs = baseEpochs + (remainder > 0 ? 1 : 0);
        int phase2Epochs = baseEpochs + (remainder > 1 ? 1 : 0);
        int phase3Epochs = baseEpochs;
        T lr = NumOps.FromDouble(_options.LearningRate / Math.Max(batchSize, 1));

        // Paper-faithful TimeGAN (Yoon et al. 2019) 3-phase training:
        // Phase 1: embedder + recovery learn the latent space via reconstruction.
        for (int epoch = 0; epoch < phase1Epochs; epoch++)
        {
            for (int b = 0; b < sequences.Count; b += batchSize)
            {
                int end = Math.Min(b + batchSize, sequences.Count);
                TrainEmbeddingStepBatched(sequences, b, end);
            }
        }

        // Phase 2: supervisor learns next-step prediction in latent space.
        for (int epoch = 0; epoch < phase2Epochs; epoch++)
        {
            for (int b = 0; b < sequences.Count; b += batchSize)
            {
                int end = Math.Min(b + batchSize, sequences.Count);
                TrainSupervisedStepBatched(sequences, b, end);
            }
        }

        // Phase 3: joint adversarial training. Per Yoon 2019 §3.3 schedule:
        // generator/supervisor step + critic step + embedder fine-tune per batch.
        for (int epoch = 0; epoch < phase3Epochs; epoch++)
        {
            for (int b = 0; b < sequences.Count; b += batchSize)
            {
                int end = Math.Min(b + batchSize, sequences.Count);
                TrainDiscriminatorStepBatched(sequences, b, end);
                TrainGeneratorStepBatched(sequences, b, end);
                TrainEmbeddingStepBatched(sequences, b, end);
            }
        }

        IsFitted = true;
    }

    /// <inheritdoc />
    public Task FitAsync(Matrix<T> data, IReadOnlyList<ColumnMetadata> columns, int epochs, CancellationToken ct = default)
    {
        return Task.Run(() =>
        {
            ct.ThrowIfCancellationRequested();
            Fit(data, columns, epochs);
        }, ct);
    }

    /// <inheritdoc />
    public Matrix<T> Generate(int numSamples, Vector<T>? conditionColumn = null, Vector<T>? conditionValue = null)
    {
        if (!IsFitted || _recoveryOutput is null)
        {
            throw new InvalidOperationException("Generator must be fitted before generating data.");
        }

        int seqLen = _options.SequenceLength;
        int hiddenDim = _materializedHiddenDimension;

        int numSequences = (int)Math.Ceiling((double)numSamples / seqLen);
        var result = new Matrix<T>(numSamples, _dataWidth);
        if (numSequences == 0)
        {
            return result;
        }

        // Generate complete sequences in one recurrent pass. Processing one vector at a time resets
        // a stateless GRU at every t and degenerates back into independent row generation.
        var noise = GenerateNoiseBatchTensor(numSequences, seqLen, hiddenDim);
        var generated = GeneratorForwardBatched(noise, isTraining: false);
        var supervised = SupervisorForwardBatched(generated, isTraining: false);
        var recovered = RecoveryForwardBatched(supervised, isTraining: false);

        for (int row = 0; row < numSamples; row++)
        {
            int sequence = row / seqLen;
            int timestep = row % seqLen;
            for (int j = 0; j < _dataWidth; j++)
            {
                result[row, j] = recovered[sequence, timestep, j];
            }
        }

        return result;
    }

    #endregion

    #region Forward Passes with Manual Activation and Pre-Activation Caching

    private Vector<T> EmbedderForward(Vector<T> x, bool isTraining)
    {
        _embedderPreActs.Clear();
        var current = EmbedderForwardBatched(VectorToTensor(x), isTraining);
        return TensorToVector(current, current.Length);
    }

    private Vector<T> RecoveryForward(Vector<T> h, bool isTraining)
    {
        _recoveryPreActs.Clear();
        var current = RecoveryForwardBatched(VectorToTensor(h), isTraining);
        return TensorToVector(current, current.Length);
    }

    private Vector<T> GeneratorForward(Vector<T> noise, bool isTraining)
    {
        _generatorPreActs.Clear();
        var current = GeneratorForwardBatched(VectorToTensor(noise), isTraining);
        return TensorToVector(current, current.Length);
    }

    private Vector<T> SupervisorForward(Vector<T> h, bool isTraining)
    {
        _supervisorPreActs.Clear();
        var current = SupervisorForwardBatched(VectorToTensor(h), isTraining);
        return TensorToVector(current, current.Length);
    }

    private Vector<T> DiscriminatorForward(Vector<T> h, bool isTraining)
    {
        _discPreActs.Clear();
        var current = DiscriminatorForwardBatched(VectorToTensor(h), isTraining);
        return TensorToVector(current, current.Length);
    }

    #endregion

    #region Backward Passes

    #endregion

    #region Parameter Updates

    private void UpdateEmbedder(T lr)
    {
        foreach (var layer in _embedderLayers) layer.UpdateParameters(lr);
        _embedderOutput?.UpdateParameters(lr);
    }

    private void UpdateRecovery(T lr)
    {
        foreach (var layer in _recoveryLayers) layer.UpdateParameters(lr);
        _recoveryOutput?.UpdateParameters(lr);
    }

    private void UpdateGenerator(T lr)
    {
        foreach (var layer in Layers) layer.UpdateParameters(lr);
        _generatorOutput?.UpdateParameters(lr);
    }

    private void UpdateSupervisor(T lr)
    {
        foreach (var layer in _supervisorLayers) layer.UpdateParameters(lr);
        _supervisorOutput?.UpdateParameters(lr);
    }

    private void UpdateDiscriminator(T lr)
    {
        foreach (var layer in _discriminatorForwardLayers) layer.UpdateParameters(lr);
        foreach (var layer in _discriminatorBackwardLayers) layer.UpdateParameters(lr);
        _discriminatorOutput?.UpdateParameters(lr);
    }

    #endregion

    #region Training Phases

    /// <summary>
    /// Paper-faithful TimeGAN Phase 1 (Yoon et al. 2019 §3.1):
    /// joint embedder + recovery training on the reconstruction objective
    /// <c>L_R = E[||x - r(e(x))||_2^2]</c>. Tape-tracked so backprop flows
    /// through both networks in a single optimizer step.
    /// </summary>
    private void TrainEmbeddingStepBatched(List<Matrix<T>> sequences, int startIdx, int endIdx)
    {
        var xBatch = BuildSequenceBatch(sequences, startIdx, endIdx);
        if (xBatch.Shape[0] == 0) return;

        var embedderRecoveryLayers = new List<ILayer<T>>();
        embedderRecoveryLayers.AddRange(_embedderLayers);
        if (_embedderOutput is not null) embedderRecoveryLayers.Add(_embedderOutput);
        embedderRecoveryLayers.AddRange(_recoveryLayers);
        if (_recoveryOutput is not null) embedderRecoveryLayers.Add(_recoveryOutput);
        var paramsList = TapeTrainingStep<T>.CollectParameters(embedderRecoveryLayers);

        // GPU-RESIDENT fast path — Phase 1 embedder + recovery reconstruction.
        var trainableEmbRec = embedderRecoveryLayers.OfType<ITrainableLayer<T>>().ToList();
        if (trainableEmbRec.Count > 0)
        {
            Tensor<T> Fwd(Tensor<T> x) => RecoveryForwardBatched(EmbedderForwardBatched(x, true), true);
            Tensor<T> Loss(Tensor<T> r, Tensor<T> x)
            {
                var d = Engine.TensorSubtract(r, x);
                var s = Engine.TensorMultiply(d, d);
                var axes = Enumerable.Range(0, s.Shape.Length).ToArray();
                var m = Engine.ReduceMean(s, axes, keepDims: false);
                return Engine.TensorMultiplyScalar(m, NumOps.FromDouble(_options.ReconstructionWeight));
            }
            if (AiDotNet.Training.GpuResidentFusedStep<T>.TryStep(
                    trainableEmbRec, xBatch, xBatch,
                    forward: Fwd, computeLoss: Loss,
                    optimizer: _embedderOptimizer,
                    out T _))
            {
                return;
            }
        }

        using var tape = new GradientTape<T>();

        var hBatch = EmbedderForwardBatched(xBatch, isTraining: true);
        var rBatch = RecoveryForwardBatched(hBatch, isTraining: true);

        // L_R = mean((x - r)^2) * reconstruction_weight
        var diff = Engine.TensorSubtract(rBatch, xBatch);
        var sq = Engine.TensorMultiply(diff, diff);
        var allAxes = Enumerable.Range(0, sq.Shape.Length).ToArray();
        var meanSq = Engine.ReduceMean(sq, allAxes, keepDims: false);
        var lossTensor = Engine.TensorMultiplyScalar(meanSq, NumOps.FromDouble(_options.ReconstructionWeight));

        var grads = tape.ComputeGradients(lossTensor, paramsList);
        T lossValue = lossTensor.Length > 0 ? lossTensor[0] : NumOps.Zero;

        Tensor<T> ComputeForward(Tensor<T> inp, Tensor<T> _) =>
            RecoveryForwardBatched(EmbedderForwardBatched(inp, true), true);
        Tensor<T> RecomputeLoss(Tensor<T> pred, Tensor<T> target) => Engine.TensorMultiplyScalar(
            Engine.ReduceMean(
                Engine.TensorMultiply(Engine.TensorSubtract(pred, target), Engine.TensorSubtract(pred, target)),
                allAxes, keepDims: false),
            NumOps.FromDouble(_options.ReconstructionWeight));

        var context = new TapeStepContext<T>(
            paramsList, grads, lossValue,
            xBatch, xBatch, ComputeForward, RecomputeLoss,
            parameterBuffer: null);
        _embedderOptimizer.Step(context);
    }

    /// <summary>
    /// Paper-faithful TimeGAN Phase 2 (Yoon et al. 2019 §3.2):
    /// supervisor next-step prediction in the embedded space.
    /// <c>L_S = E[||h_{t+1} - s(h_t)||_2^2]</c>. Embedder is frozen
    /// (Phase 1 produced it).
    /// </summary>
    private void TrainSupervisedStepBatched(List<Matrix<T>> sequences, int startIdx, int endIdx)
    {
        var xBatch = BuildSequenceBatch(sequences, startIdx, endIdx);
        if (xBatch.Shape[0] == 0 || xBatch.Shape[1] < 2) return;

        var supervisorLayers = new List<ILayer<T>>();
        supervisorLayers.AddRange(_supervisorLayers);
        if (_supervisorOutput is not null) supervisorLayers.Add(_supervisorOutput);
        var paramsList = TapeTrainingStep<T>.CollectParameters(supervisorLayers);

        // Embedder runs OUTSIDE the tape (frozen for this step).
        var embeddings = EmbedderForwardBatched(xBatch, isTraining: false);
        var (ht, htNext) = BuildAdjacentLatentBatch(embeddings);

        // GPU-RESIDENT fast path — supervisor's next-step prediction. Embedder
        // is frozen; supervisor's layers are the only trainable set here.
        var trainableSup = supervisorLayers.OfType<ITrainableLayer<T>>().ToList();
        if (trainableSup.Count > 0)
        {
            Tensor<T> Fwd(Tensor<T> h) => SupervisorForwardBatched(h, isTraining: true);
            Tensor<T> Loss(Tensor<T> pred, Tensor<T> tgt)
            {
                var d = Engine.TensorSubtract(pred, tgt);
                var s = Engine.TensorMultiply(d, d);
                var axes = Enumerable.Range(0, s.Shape.Length).ToArray();
                return Engine.ReduceMean(s, axes, keepDims: false);
            }
            if (AiDotNet.Training.GpuResidentFusedStep<T>.TryStep(
                    trainableSup, ht, htNext,
                    forward: Fwd, computeLoss: Loss,
                    optimizer: _supervisorOptimizer,
                    out T _))
            {
                return;
            }
        }

        using var tape = new GradientTape<T>();

        var htPred = SupervisorForwardBatched(ht, isTraining: true);
        var diff = Engine.TensorSubtract(htPred, htNext);
        var sq = Engine.TensorMultiply(diff, diff);
        var allAxes = Enumerable.Range(0, sq.Shape.Length).ToArray();
        var lossTensor = Engine.ReduceMean(sq, allAxes, keepDims: false);

        var grads = tape.ComputeGradients(lossTensor, paramsList);
        T lossValue = lossTensor.Length > 0 ? lossTensor[0] : NumOps.Zero;

        Tensor<T> ComputeForward(Tensor<T> inp, Tensor<T> _) => SupervisorForwardBatched(inp, true);
        Tensor<T> RecomputeLoss(Tensor<T> pred, Tensor<T> target) => Engine.ReduceMean(
            Engine.TensorMultiply(Engine.TensorSubtract(pred, target), Engine.TensorSubtract(pred, target)),
            allAxes, keepDims: false);

        var context = new TapeStepContext<T>(
            paramsList, grads, lossValue,
            ht, htNext, ComputeForward, RecomputeLoss,
            parameterBuffer: null);
        _supervisorOptimizer.Step(context);
    }

    /// <summary>
    /// Paper-faithful TimeGAN Phase 3 critic step (Yoon et al. 2019 §3.3):
    /// the discriminator learns to distinguish real embedded sequences from
    /// supervisor-rolled-out fake embedded sequences. BCE-with-logits via
    /// tape-tracked Engine.Sigmoid + Engine.TensorLog.
    /// </summary>
    private void TrainDiscriminatorStepBatched(List<Matrix<T>> sequences, int startIdx, int endIdx)
    {
        var xBatch = BuildSequenceBatch(sequences, startIdx, endIdx);
        if (xBatch.Shape[0] == 0) return;
        int batchSize = xBatch.Shape[0];
        int sequenceLength = xBatch.Shape[1];
        int hiddenDim = _materializedHiddenDimension;

        // Real embedded sequence: x -> embedder. Fake: noise -> generator -> supervisor.
        // Both produced OUTSIDE the critic's tape so the critic only updates its own params.
        var realEmb = EmbedderForwardBatched(xBatch, isTraining: false);
        var noise = GenerateNoiseBatchTensor(batchSize, sequenceLength, hiddenDim);
        var fakeEmb = GeneratorForwardBatched(noise, isTraining: false);
        var fakeSup = SupervisorForwardBatched(fakeEmb, isTraining: false);

        using var tape = new GradientTape<T>();

        var discLayers = new List<ILayer<T>>();
        discLayers.AddRange(_discriminatorForwardLayers);
        discLayers.AddRange(_discriminatorBackwardLayers);
        if (_discriminatorOutput is not null) discLayers.Add(_discriminatorOutput);
        var paramsList = TapeTrainingStep<T>.CollectParameters(discLayers);

        var realScores = DiscriminatorForwardBatched(realEmb, isTraining: true);
        var fakeScores = DiscriminatorForwardBatched(fakeSup, isTraining: true);

        var allAxes = Enumerable.Range(0, realScores.Shape.Length).ToArray();
        var lossReal = Engine.TensorNegate(Engine.ReduceMean(LogSigmoid(realScores), allAxes, keepDims: false));
        var lossFake = Engine.TensorNegate(Engine.ReduceMean(LogSigmoid(Engine.TensorNegate(fakeScores)), allAxes, keepDims: false));
        var lossTensor = Engine.TensorAdd(lossReal, lossFake);

        var grads = tape.ComputeGradients(lossTensor, paramsList);
        T lossValue = lossTensor.Length > 0 ? lossTensor[0] : NumOps.Zero;

        // Replay-correct closure for the critic: lossTensor was
        // (lossReal + lossFake), so RecomputeLoss must replay BOTH BCE
        // terms to stay tied to the objective that produced `grads`.
        // ComputeForward receives the real-embedded input and returns the
        // real discriminator scores; RecomputeLoss captures fakeSup so it
        // can re-run the discriminator on the fake side too, then build
        // the same -log σ(realScore) + -log σ(-fakeScore) sum.
        var capturedFakeSup = fakeSup;
        Tensor<T> ComputeForward(Tensor<T> inp, Tensor<T> _) => DiscriminatorForwardBatched(inp, true);
        Tensor<T> RecomputeLoss(Tensor<T> predReal, Tensor<T> _)
        {
            var predFake = DiscriminatorForwardBatched(capturedFakeSup, true);
            var lossR = Engine.TensorNegate(Engine.ReduceMean(LogSigmoid(predReal), allAxes, keepDims: false));
            var lossF = Engine.TensorNegate(Engine.ReduceMean(LogSigmoid(Engine.TensorNegate(predFake)), allAxes, keepDims: false));
            return Engine.TensorAdd(lossR, lossF);
        }

        var context = new TapeStepContext<T>(
            paramsList, grads, lossValue,
            realEmb, realEmb, ComputeForward, RecomputeLoss,
            parameterBuffer: null);
        _discriminatorOptimizer.Step(context);
    }

    /// <summary>
    /// Paper-faithful TimeGAN Phase 3 generator + supervisor joint step:
    /// non-saturating adversarial loss + supervised next-step loss in the
    /// embedded space (Yoon et al. 2019 §3.3 joint training).
    /// </summary>
    private void TrainGeneratorStepBatched(List<Matrix<T>> sequences, int startIdx, int endIdx)
    {
        int hiddenDim = _materializedHiddenDimension;

        // Phase 3 joint loss (Yoon et al. 2019 §3.3) =
        //   L_U (unsupervised adversarial, non-saturating)
        //   + γ · L_S (supervised next-step MSE on real sequence pairs).
        // Without the L_S term the supervisor is updated only through the
        // adversarial gradient — the next-step temporal structure that
        // L_S explicitly preserves is lost once joint training begins.
        var xBatch = BuildSequenceBatch(sequences, startIdx, endIdx);
        if (xBatch.Shape[0] == 0 || xBatch.Shape[1] < 2) return;
        int supervisedBatch = xBatch.Shape[0];
        int sequenceLength = xBatch.Shape[1];

        // The embedder is frozen in this step. Run the full real sequence first, then shift the
        // recurrent embeddings; embedding x_t and x_(t+1) as unrelated rows loses their histories.
        var realEmbeddings = EmbedderForwardBatched(xBatch, isTraining: false);
        var (ht, htNext) = BuildAdjacentLatentBatch(realEmbeddings);

        using var tape = new GradientTape<T>();

        var genSupLayers = new List<ILayer<T>>();
        genSupLayers.AddRange(Layers);
        if (_generatorOutput is not null) genSupLayers.Add(_generatorOutput);
        genSupLayers.AddRange(_supervisorLayers);
        if (_supervisorOutput is not null) genSupLayers.Add(_supervisorOutput);
        var paramsList = TapeTrainingStep<T>.CollectParameters(genSupLayers);

        // Adversarial term: minimize -log σ(D(s(g(z))))
        int advBatch = Math.Max(1, supervisedBatch);
        var noise = GenerateNoiseBatchTensor(advBatch, sequenceLength, hiddenDim);
        var fakeEmb = GeneratorForwardBatched(noise, isTraining: true);
        var fakeSup = SupervisorForwardBatched(fakeEmb, isTraining: true);
        var fakeScores = DiscriminatorForwardBatched(fakeSup, isTraining: false);
        var advAxes = Enumerable.Range(0, fakeScores.Shape.Length).ToArray();
        var advLoss = Engine.TensorNegate(Engine.ReduceMean(LogSigmoid(fakeScores), advAxes, keepDims: false));

        Tensor<T> lossTensor;
        if (supervisedBatch > 0)
        {
            // Supervised term: L_S = E[||h_{t+1} − s(h_t)||_2^2] on real
            // sequence pairs. Embedder is frozen by Phase 1 and runs
            // outside the tape; the supervisor remains tape-tracked so its
            // gradient flows.
            var htPred = SupervisorForwardBatched(ht, isTraining: true);
            var diff = Engine.TensorSubtract(htPred, htNext);
            var sq = Engine.TensorMultiply(diff, diff);
            var supAxes = Enumerable.Range(0, sq.Shape.Length).ToArray();
            var supLoss = Engine.ReduceMean(sq, supAxes, keepDims: false);

            // Yoon 2019 uses equal weighting between L_U and L_S in the
            // joint phase; expose γ via options later if needed.
            var weightedSup = Engine.TensorMultiplyScalar(supLoss, NumOps.FromDouble(_options.SupervisedWeight));
            lossTensor = Engine.TensorAdd(advLoss, weightedSup);
        }
        else
        {
            // No paired data available (sequence too short for next-step
            // pairing) — fall back to adversarial-only for this batch.
            lossTensor = advLoss;
        }

        var grads = tape.ComputeGradients(lossTensor, paramsList);
        T lossValue = lossTensor.Length > 0 ? lossTensor[0] : NumOps.Zero;

        // Replay-correct closure for the joint phase: lossTensor combined
        // adversarial (advLoss) AND, when supervisedBatch > 0, the weighted
        // supervised next-step term. RecomputeLoss must reproduce the full
        // sum so optimizer.Step replays stay tied to the objective that
        // produced `grads` — replaying only the adversarial part silently
        // drops the temporal-supervision phase-3 contribution.
        int capturedSupervisedBatch = supervisedBatch;
        var capturedHt = ht;
        var capturedHtNext = htNext;
        var capturedAdvAxes = advAxes;
        double capturedSupWeight = _options.SupervisedWeight;
        Tensor<T> ComputeForward(Tensor<T> inp, Tensor<T> _) => DiscriminatorForwardBatched(
            SupervisorForwardBatched(GeneratorForwardBatched(inp, true), true), false);
        Tensor<T> RecomputeLoss(Tensor<T> pred, Tensor<T> _)
        {
            var adv = Engine.TensorNegate(Engine.ReduceMean(LogSigmoid(pred), capturedAdvAxes, keepDims: false));
            if (capturedSupervisedBatch <= 0) return adv;
            var htPred = SupervisorForwardBatched(capturedHt, isTraining: true);
            var diff = Engine.TensorSubtract(htPred, capturedHtNext);
            var sq = Engine.TensorMultiply(diff, diff);
            var supAxes = Enumerable.Range(0, sq.Shape.Length).ToArray();
            var supLoss = Engine.ReduceMean(sq, supAxes, keepDims: false);
            var weightedSup = Engine.TensorMultiplyScalar(supLoss, NumOps.FromDouble(capturedSupWeight));
            return Engine.TensorAdd(adv, weightedSup);
        }

        var context = new TapeStepContext<T>(
            paramsList, grads, lossValue,
            noise, noise, ComputeForward, RecomputeLoss,
            parameterBuffer: null);
        _generatorOptimizer.Step(context);
    }

    /// <summary>
    /// Builds a batch of complete sequences as <c>[batch, time, features]</c>.
    /// Keeping the time axis is required for every recurrent TimeGAN component.
    /// </summary>
    private Tensor<T> BuildSequenceBatch(List<Matrix<T>> sequences, int startIdx, int endIdx)
    {
        int boundedEnd = Math.Min(endIdx, sequences.Count);
        int batchSize = Math.Max(0, boundedEnd - startIdx);
        if (batchSize == 0)
        {
            return new Tensor<T>([0, 0, _dataWidth]);
        }

        int sequenceLength = sequences[startIdx].Rows;
        var batch = new Tensor<T>([batchSize, sequenceLength, _dataWidth]);
        for (int s = startIdx; s < boundedEnd; s++)
        {
            var sequence = sequences[s];
            if (sequence.Rows != sequenceLength)
            {
                throw new InvalidOperationException(
                    "TimeGAN recurrent batches require equal sequence lengths within a batch.");
            }

            int batchIndex = s - startIdx;
            int columns = Math.Min(_dataWidth, sequence.Columns);
            for (int t = 0; t < sequenceLength; t++)
                for (int j = 0; j < columns; j++)
                    batch[batchIndex, t, j] = sequence[t, j];
        }

        return batch;
    }

    /// <summary>
    /// Splits a recurrent embedding sequence into aligned current/next latent sequences.
    /// Engine narrowing preserves the gradient connection to the supervisor input.
    /// </summary>
    private (Tensor<T> Current, Tensor<T> Next) BuildAdjacentLatentBatch(Tensor<T> embeddings)
    {
        if (embeddings.Shape.Length != 3)
        {
            throw new ArgumentException(
                $"Expected recurrent embeddings shaped [batch, time, hidden], got rank {embeddings.Shape.Length}.",
                nameof(embeddings));
        }

        int sequenceLength = embeddings.Shape[1];
        if (sequenceLength < 2)
        {
            int batchSize = embeddings.Shape[0];
            int hiddenDimension = embeddings.Shape[2];
            return (
                new Tensor<T>([batchSize, 0, hiddenDimension]),
                new Tensor<T>([batchSize, 0, hiddenDimension]));
        }

        return (
            Engine.TensorNarrow(embeddings, 1, 0, sequenceLength - 1),
            Engine.TensorNarrow(embeddings, 1, 1, sequenceLength - 1));
    }

    private Tensor<T> GenerateNoiseBatchTensor(int batchSize, int sequenceLength, int dim)
    {
        int totalElements = checked(batchSize * sequenceLength * dim);
        // Box–Muller via the seeded _random so TimeGANOptions.Seed makes
        // Fit reproducible — Engine.TensorRandomUniformRange bypasses _random
        // and breaks the seed contract that the rest of the sampler stack
        // (and Generate via its own seeded paths) honours.
        var noiseData = new T[totalElements];
        for (int i = 0; i < totalElements; i += 2)
        {
            double u1 = 1.0 - _random.NextDouble();   // ∈ (0, 1] keeps log finite
            double u2 = _random.NextDouble();
            double r = Math.Sqrt(-2.0 * Math.Log(u1));
            double theta = 2.0 * Math.PI * u2;
            noiseData[i] = NumOps.FromDouble(r * Math.Cos(theta));
            if (i + 1 < totalElements)
                noiseData[i + 1] = NumOps.FromDouble(r * Math.Sin(theta));
        }
        return new Tensor<T>(noiseData, [batchSize, sequenceLength, dim]);
    }

    // ----- Batched, tape-tracked recurrent forward methods -----

    private Tensor<T> EmbedderForwardBatched(Tensor<T> x, bool isTraining)
    {
        var current = x;
        foreach (var layer in _embedderLayers) current = layer.Forward(current);
        if (_embedderOutput is not null)
        {
            current = Engine.Sigmoid(ApplyPointwise(_embedderOutput, current));
        }
        return current;
    }

    private Tensor<T> RecoveryForwardBatched(Tensor<T> h, bool isTraining)
    {
        var current = h;
        foreach (var layer in _recoveryLayers)
        {
            current = ApplyPointwise(layer, current);
            current = Engine.Sigmoid(current);
        }
        if (_recoveryOutput is not null) current = ApplyPointwise(_recoveryOutput, current);
        return current;
    }

    private Tensor<T> GeneratorForwardBatched(Tensor<T> noise, bool isTraining)
    {
        var current = noise;
        foreach (var layer in Layers)
        {
            current = layer.Forward(current);
            if (_usingCustomLayers)
            {
                // Preserve the established custom-layer contract: Architecture.Layers describes the
                // complete generator stack and each layer receives the historical sigmoid transform.
                current = Engine.Sigmoid(current);
            }
        }
        if (_generatorOutput is not null)
        {
            current = Engine.Sigmoid(ApplyPointwise(_generatorOutput, current));
        }
        return current;
    }

    private Tensor<T> SupervisorForwardBatched(Tensor<T> h, bool isTraining)
    {
        var current = h;
        foreach (var layer in _supervisorLayers) current = layer.Forward(current);
        if (_supervisorOutput is not null)
        {
            current = Engine.Sigmoid(ApplyPointwise(_supervisorOutput, current));
        }
        return current;
    }

    private Tensor<T> DiscriminatorForwardBatched(Tensor<T> h, bool isTraining)
    {
        var current = h;
        for (int i = 0; i < _discriminatorForwardLayers.Count; i++)
        {
            var forward = _discriminatorForwardLayers[i].Forward(current);
            var reversedInput = ReverseTimeSequence(current);
            var backwardReversed = _discriminatorBackwardLayers[i].Forward(reversedInput);
            var backward = ReverseTimeSequence(backwardReversed);
            current = Engine.Concat([forward, backward], forward.Shape.Length - 1);

            if (i < _discDropoutLayers.Count)
            {
                _discDropoutLayers[i].SetTrainingMode(isTraining);
                current = _discDropoutLayers[i].Forward(current);
            }
        }
        if (_discriminatorOutput is not null) current = ApplyPointwise(_discriminatorOutput, current);
        return current;
    }

    private Tensor<T> ApplyPointwise(FullyConnectedLayer<T> layer, Tensor<T> input)
    {
        if (input.Shape.Length <= 2)
        {
            return layer.Forward(input);
        }

        // A dense head in a temporal model is applied independently at each [batch,time] position.
        // Flattening the leading axes makes that contract explicit and ensures weight gradients are
        // reduced into a matrix rather than retaining a spurious batch dimension.
        int inputWidth = input.Shape[^1];
        var flattened = Engine.Reshape(input, [input.Length / inputWidth, inputWidth]);
        var projected = layer.Forward(flattened);
        int[] outputShape = input.Shape.ToArray();
        outputShape[^1] = projected.Shape[^1];
        return Engine.Reshape(projected, outputShape);
    }

    private Tensor<T> ReverseTimeSequence(Tensor<T> sequence)
    {
        int timeAxis = sequence.Shape.Length switch
        {
            1 => -1,
            2 => 0,
            3 => 1,
            _ => throw new ArgumentException(
                $"TimeGAN recurrent components expect rank-1, rank-2, or rank-3 tensors; got rank {sequence.Shape.Length}.",
                nameof(sequence))
        };

        if (timeAxis < 0 || sequence.Shape[timeAxis] <= 1)
        {
            return sequence;
        }

        int sequenceLength = sequence.Shape[timeAxis];
        var timesteps = new Tensor<T>[sequenceLength];
        for (int t = 0; t < sequenceLength; t++)
        {
            timesteps[t] = Engine.TensorNarrow(sequence, timeAxis, sequenceLength - 1 - t, 1);
        }

        return Engine.Concat(timesteps, timeAxis);
    }

    // Numerically stable log σ(x) = -softplus(-x). The naive
    // log(σ(x)) = log(1 / (1 + exp(-x))) underflows to -∞ for confident
    // negative scores (sigmoid saturates at 0). The softplus form keeps
    // the dynamic range intact because softplus(z) = log(1+exp(z)) is
    // implemented via the stable max(z,0) + log(1+exp(-|z|)) identity
    // inside Engine.Softplus.
    private Tensor<T> LogSigmoid(Tensor<T> x) =>
        Engine.TensorNegate(Engine.Softplus(Engine.TensorNegate(x)));

    #endregion

    #region Discriminator Layer List

    /// <summary>
    /// Builds a combined list of discriminator layers (forward/backward GRUs + dropout + output)
    /// for gradient-penalty and related analyses.
    /// </summary>
    private IReadOnlyList<ILayer<T>> BuildDiscLayerList()
    {
        var allLayers = new List<ILayer<T>>();
        for (int i = 0; i < _discDropoutLayers.Count; i++)
        {
            allLayers.Add(_discriminatorForwardLayers[i]);
            allLayers.Add(_discriminatorBackwardLayers[i]);
            allLayers.Add(_discDropoutLayers[i]);
        }
        if (_discriminatorOutput is not null)
        {
            allLayers.Add(_discriminatorOutput);
        }
        return allLayers;
    }

    #endregion

    #region NeuralNetworkBase Overrides

    private Tensor<T> GetGeneratorNoise(Tensor<T> input)
    {
        if (input is null)
            throw new ArgumentNullException(nameof(input));

        int inputWidth = input.Shape.Length == 0 ? 0 : input.Shape[^1];
        if (IsFitted && inputWidth != _materializedHiddenDimension)
        {
            throw new ArgumentException(
                $"A fitted TimeGAN generator requires latent input with exactly "
                + $"{_materializedHiddenDimension} values (the HiddenDimension used by the fitted topology), "
                + $"but received a final dimension of {inputWidth}.",
                nameof(input));
        }

        return input;
    }

    private Tensor<T> RestoreInputRank(Tensor<T> output, Tensor<T> input)
    {
        return input.Shape.Length switch
        {
            1 => Engine.Reshape(output, [output.Shape[^1]]),
            2 when output.Shape.Length == 3 && output.Shape[0] == 1 =>
                Engine.Reshape(output, [output.Shape[1], output.Shape[2]]),
            _ => output
        };
    }

    /// <inheritdoc />
    protected override Tensor<T> PredictCore(Tensor<T> input)
    {
        if (!IsFitted) return input;

        var noise = GetGeneratorNoise(input);
        var genOut = GeneratorForwardBatched(noise, isTraining: false);
        var supOut = SupervisorForwardBatched(genOut, isTraining: false);
        var recOut = RecoveryForwardBatched(supOut, isTraining: false);
        return RestoreInputRank(recOut, input);
    }

    /// <summary>
    /// Reports the three stages of the synthesis path: generator, supervisor and recovery.
    /// </summary>
    /// <remarks>
    /// <para>
    /// THE BASE STRATEGIES BOTH COME BACK EMPTY HERE, and not because the stacks are unobservable.
    /// <see cref="PredictCore"/> opens with <c>if (!IsFitted) return input;</c>, so on a freshly
    /// constructed generator no layer is invoked at all: the observer fallback records nothing, and
    /// the sequential fold has nothing to fold because the embedder, recovery and supervisor stacks
    /// live in their own lists rather than in <see cref="Layers"/>. The base then returned an empty
    /// dictionary, which it documents as a failure to answer rather than an answer of "no
    /// activations".
    /// </para>
    /// <para>
    /// InitializeLayers constructs and initialises every stack, so the pipeline is well defined
    /// before fitting -- structural introspection should not depend on fit state. This runs the same
    /// generator -> supervisor -> recovery chain PredictCore runs after fitting, so the reported
    /// activations are the model's real computation rather than a reconstruction of it.
    /// </para>
    /// </remarks>
    public override Dictionary<string, Tensor<T>> GetNamedLayerActivations(Tensor<T> input)
    {
        var activations = new Dictionary<string, Tensor<T>>();

        var noise = GetGeneratorNoise(input);
        var current = GeneratorForwardBatched(noise, isTraining: false);
        activations["Generator"] = RestoreInputRank(current, input);

        // ONLY STAGES THAT ACTUALLY EXIST. RebuildAllNetworks -- which Fit calls -- is what creates
        // the supervisor and recovery stacks; InitializeLayers alone populates just the generator's
        // Layers. SupervisorForward and RecoveryForward iterate their stack, so on an unfitted model
        // those loops have nothing to run and return their input UNCHANGED. Reporting them anyway
        // published the generator's output three times under three names: an identity value dressed
        // as a distinct stage, which passes a non-empty check while describing nothing.
        //
        // Reporting only the initialised stacks keeps every entry a real computation. The generator
        // always exists after InitializeLayers, so the result is never empty.
        // GATE ON THE OUTPUT HEAD, NOT THE HIDDEN LAYER LIST. SupervisorForward and RecoveryForward
        // apply _supervisorOutput / _recoveryOutput independently of their loops, so a stack with an
        // empty layer list but a constructed head still performs a real projection -- which is
        // exactly the shape NumLayers == 1 produces for the supervisor. Gating on Count > 0 would
        // have dropped a stage that does genuine work; the head is what RebuildAllNetworks creates
        // last, so its presence is the honest signal that the stage exists.
        if (_supervisorOutput is not null)
        {
            current = SupervisorForwardBatched(current, isTraining: false);
            activations["Supervisor"] = RestoreInputRank(current, input);
        }

        if (_recoveryOutput is not null)
        {
            current = RecoveryForwardBatched(current, isTraining: false);
            activations["Recovery"] = RestoreInputRank(current, input);
        }

        return activations;
    }

    /// <inheritdoc />
    public override void Train(Tensor<T> input, Tensor<T> expectedOutput)
    {
        var predicted = Predict(input);
        var loss = _lossFunction.CalculateLoss(
            TensorToVector(predicted, predicted.Length),
            TensorToVector(expectedOutput, expectedOutput.Length));
    }

    /// <inheritdoc />
    public override Dictionary<string, T> GetFeatureImportance()
    {
        var importance = new Dictionary<string, T>();
        for (int i = 0; i < _columns.Count; i++)
        {
            importance[_columns[i].Name] = NumOps.FromDouble(1.0 / Math.Max(_columns.Count, 1));
        }
        return importance;
    }

    /// <inheritdoc />
    public override ModelMetadata<T> GetModelMetadata()
    {
        return new ModelMetadata<T>
        {
            AdditionalInfo = new Dictionary<string, object>
            {
                ["GeneratorType"] = "TimeGAN",
                ["SequenceLength"] = _options.SequenceLength,
                ["HiddenDimension"] = _materializedHiddenDimension,
                ["NumLayers"] = _materializedNumLayers,
                ["DataWidth"] = _dataWidth,
                ["IsFitted"] = IsFitted
            }
        };
    }

    #endregion

    #region Manual Activation Functions

    private static Tensor<T> ApplySigmoid(Tensor<T> input)
    {
        var numOps = MathHelper.GetNumericOperations<T>();
        var result = new Tensor<T>(input._shape);
        for (int i = 0; i < input.Length; i++)
        {
            double v = numOps.ToDouble(input[i]);
            double clamped = Math.Min(Math.Max(v, -20.0), 20.0);
            result[i] = numOps.FromDouble(1.0 / (1.0 + Math.Exp(-clamped)));
        }
        return result;
    }

    private static Tensor<T> ApplySigmoidDerivative(Tensor<T> grad, Tensor<T> preActivation)
    {
        var numOps = MathHelper.GetNumericOperations<T>();
        var result = new Tensor<T>(grad._shape);
        int len = Math.Min(grad.Length, preActivation.Length);
        for (int i = 0; i < len; i++)
        {
            double v = numOps.ToDouble(preActivation[i]);
            double clamped = Math.Min(Math.Max(v, -20.0), 20.0);
            double sig = 1.0 / (1.0 + Math.Exp(-clamped));
            double deriv = sig * (1.0 - sig);
            result[i] = numOps.FromDouble(numOps.ToDouble(grad[i]) * deriv);
        }
        return result;
    }

    private static Tensor<T> ApplyLeakyReLU(Tensor<T> input, double alpha)
    {
        var numOps = MathHelper.GetNumericOperations<T>();
        var result = new Tensor<T>(input._shape);
        for (int i = 0; i < input.Length; i++)
        {
            double v = numOps.ToDouble(input[i]);
            result[i] = numOps.FromDouble(v >= 0 ? v : alpha * v);
        }
        return result;
    }

    private static Tensor<T> ApplyLeakyReLUDerivative(Tensor<T> grad, Tensor<T> preActivation, double alpha)
    {
        var numOps = MathHelper.GetNumericOperations<T>();
        var result = new Tensor<T>(grad._shape);
        int len = Math.Min(grad.Length, preActivation.Length);
        for (int i = 0; i < len; i++)
        {
            double pre = numOps.ToDouble(preActivation[i]);
            double deriv = pre >= 0 ? 1.0 : alpha;
            result[i] = numOps.FromDouble(numOps.ToDouble(grad[i]) * deriv);
        }
        return result;
    }

    #endregion

    #region Data Preparation

    private List<Matrix<T>> PrepareSequences(Matrix<T> data, int seqLen)
    {
        var sequences = new List<Matrix<T>>();

        if (data.Rows < seqLen)
        {
            var seq = new Matrix<T>(data.Rows, data.Columns);
            for (int i = 0; i < data.Rows; i++)
                for (int j = 0; j < data.Columns; j++)
                    seq[i, j] = data[i, j];
            sequences.Add(seq);
            return sequences;
        }

        int numSequences = data.Rows - seqLen + 1;
        for (int start = 0; start < numSequences; start++)
        {
            var seq = new Matrix<T>(seqLen, data.Columns);
            for (int t = 0; t < seqLen; t++)
                for (int j = 0; j < data.Columns; j++)
                    seq[t, j] = data[start + t, j];
            sequences.Add(seq);
        }

        return sequences;
    }

    #endregion

    #region Helpers

    private static double SigmoidScalar(double x)
    {
        return 1.0 / (1.0 + Math.Exp(-Math.Min(Math.Max(x, -20.0), 20.0)));
    }

    private Vector<T> CreateStandardNormalVector(int length)
    {
        var v = new Vector<T>(length);
        for (int i = 0; i < length; i++)
        {
            double u1 = 1.0 - _random.NextDouble();
            double u2 = _random.NextDouble();
            double normal = Math.Sqrt(-2.0 * Math.Log(Math.Max(u1, 1e-10))) * Math.Cos(2.0 * Math.PI * u2);
            v[i] = NumOps.FromDouble(normal);
        }
        return v;
    }

    private static Tensor<T> SanitizeAndClipGradient(Tensor<T> grad, double maxNorm)
    {
        var numOps = MathHelper.GetNumericOperations<T>();
        double normSq = 0;
        for (int i = 0; i < grad.Length; i++)
        {
            double val = numOps.ToDouble(grad[i]);
            if (double.IsNaN(val) || double.IsInfinity(val))
            {
                grad[i] = numOps.Zero;
                continue;
            }
            normSq += val * val;
        }
        double norm = Math.Sqrt(normSq);
        if (norm > maxNorm)
        {
            double scale = maxNorm / norm;
            for (int i = 0; i < grad.Length; i++)
                grad[i] = numOps.FromDouble(numOps.ToDouble(grad[i]) * scale);
        }
        return grad;
    }

    private static Tensor<T> CloneTensor(Tensor<T> source)
    {
        var clone = new Tensor<T>(source._shape);
        for (int i = 0; i < source.Length; i++) clone[i] = source[i];
        return clone;
    }

    private static Vector<T> GetRow(Matrix<T> matrix, int row)
    {
        var v = new Vector<T>(matrix.Columns);
        for (int j = 0; j < matrix.Columns; j++) v[j] = matrix[row, j];
        return v;
    }

    private static Tensor<T> VectorToTensor(Vector<T> v)
    {
        var t = new Tensor<T>([v.Length]);
        for (int i = 0; i < v.Length; i++) t[i] = v[i];
        return t;
    }

    private static Vector<T> TensorToVector(Tensor<T> t, int length)
    {
        var v = new Vector<T>(length);
        int copyLen = Math.Min(length, t.Length);
        for (int i = 0; i < copyLen; i++) v[i] = t[i];
        return v;
    }

    #endregion

}
