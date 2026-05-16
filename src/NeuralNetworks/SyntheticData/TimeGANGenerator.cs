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
public class TimeGANGenerator<T> : NeuralNetworkBase<T>, ISyntheticTabularGenerator<T>
{
    private readonly TimeGANOptions<T> _options;
    private IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> _optimizer;
    private ILossFunction<T> _lossFunction;

    // Synthetic tabular data infrastructure
    private List<ColumnMetadata> _columns = new();
    private int _dataWidth;
    private Random _random;

    // Embedder (auxiliary): data → latent
    private readonly List<FullyConnectedLayer<T>> _embedderLayers = new();
    private FullyConnectedLayer<T>? _embedderOutput;
    private readonly List<Tensor<T>> _embedderPreActs = new();

    // Recovery (auxiliary): latent → data
    private readonly List<FullyConnectedLayer<T>> _recoveryLayers = new();
    private FullyConnectedLayer<T>? _recoveryOutput;
    private readonly List<Tensor<T>> _recoveryPreActs = new();

    // Generator pre-activation cache (Layers = generator)
    private readonly List<Tensor<T>> _generatorPreActs = new();

    // Supervisor (auxiliary): latent_t → latent_{t+1}
    private readonly List<FullyConnectedLayer<T>> _supervisorLayers = new();
    private FullyConnectedLayer<T>? _supervisorOutput;
    private readonly List<Tensor<T>> _supervisorPreActs = new();

    // Discriminator (auxiliary): latent → real/fake
    private readonly List<FullyConnectedLayer<T>> _discriminatorLayers = new();
    private readonly List<DropoutLayer<T>> _discDropoutLayers = new();
    private FullyConnectedLayer<T>? _discriminatorOutput;
    private readonly List<Tensor<T>> _discPreActs = new();

    // Whether custom layers are being used
    private bool _usingCustomLayers;

    /// <summary>
    /// Gets the TimeGAN-specific options.
    /// </summary>
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
        _lossFunction = lossFunction ?? NeuralNetworkHelper<T>.GetDefaultLossFunction(architecture.TaskType);
        _optimizer = optimizer ?? new AdamOptimizer<T, Tensor<T>, Tensor<T>>(this);
        _random = _options.Seed.HasValue
            ? RandomHelper.CreateSeededRandom(_options.Seed.Value)
            : RandomHelper.CreateSecureRandom();

        InitializeLayers();
    }

    #region Layer Initialization (GANDALF Pattern)

    /// <summary>
    /// Initializes the generator layers (Layers = generator network, user-overridable).
    /// </summary>
    protected override void InitializeLayers()
    {
        if (Architecture.Layers is not null && Architecture.Layers.Count > 0)
        {
            Layers.AddRange(Architecture.Layers);
            _usingCustomLayers = true;
        }
        else
        {
            int hiddenDim = _options.HiddenDimension;
            var identity = new IdentityActivation<T>() as IActivationFunction<T>;

            for (int i = 0; i < _options.NumLayers; i++)
            {
                Layers.Add(new FullyConnectedLayer<T>(hiddenDim, identity));
            }
            _usingCustomLayers = false;
        }
    }

    private void RebuildAllNetworks()
    {
        int hiddenDim = _options.HiddenDimension;
        var identity = new IdentityActivation<T>() as IActivationFunction<T>;

        // Rebuild generator (Layers) if not using custom
        if (!_usingCustomLayers)
        {
            Layers.Clear();
            for (int i = 0; i < _options.NumLayers; i++)
            {
                int layerInput = i == 0 ? hiddenDim : hiddenDim;
                Layers.Add(new FullyConnectedLayer<T>(hiddenDim, identity));
            }
        }

        // Generator output head (always auxiliary)
        // (Note: we store this separately as the last layer is a projection to hiddenDim)

        // Embedder
        _embedderLayers.Clear();
        for (int i = 0; i < _options.NumLayers; i++)
        {
            int layerInput = i == 0 ? _dataWidth : hiddenDim;
            _embedderLayers.Add(new FullyConnectedLayer<T>(hiddenDim, identity));
        }
        _embedderOutput = new FullyConnectedLayer<T>(hiddenDim, identity);

        // Recovery
        _recoveryLayers.Clear();
        for (int i = 0; i < _options.NumLayers; i++)
        {
            _recoveryLayers.Add(new FullyConnectedLayer<T>(hiddenDim, identity));
        }
        _recoveryOutput = new FullyConnectedLayer<T>(_dataWidth, identity);

        // Supervisor
        _supervisorLayers.Clear();
        for (int i = 0; i < _options.NumLayers - 1; i++)
        {
            _supervisorLayers.Add(new FullyConnectedLayer<T>(hiddenDim, identity));
        }
        _supervisorOutput = new FullyConnectedLayer<T>(hiddenDim, identity);

        // Discriminator
        _discriminatorLayers.Clear();
        _discDropoutLayers.Clear();
        for (int i = 0; i < _options.NumLayers; i++)
        {
            _discriminatorLayers.Add(new FullyConnectedLayer<T>(hiddenDim, identity));
            _discDropoutLayers.Add(new DropoutLayer<T>(_options.DiscriminatorDropout));
        }
        _discriminatorOutput = new FullyConnectedLayer<T>(1, identity);
    }

    #endregion

    #region ISyntheticTabularGenerator Implementation

    /// <inheritdoc />
    public void Fit(Matrix<T> data, IReadOnlyList<ColumnMetadata> columns, int epochs)
    {
        _columns = new List<ColumnMetadata>(columns);
        _dataWidth = data.Columns;
        int hiddenDim = _options.HiddenDimension;
        int seqLen = _options.SequenceLength;

        RebuildAllNetworks();

        var sequences = PrepareSequences(data, seqLen);
        if (sequences.Count == 0)
        {
            IsFitted = true;
            return;
        }

        int batchSize = Math.Min(_options.BatchSize, sequences.Count);
        int phaseDuration = Math.Max(1, epochs / 3);
        T lr = NumOps.FromDouble(_options.LearningRate / Math.Max(batchSize, 1));

        // Paper-faithful TimeGAN (Yoon et al. 2019) 3-phase training:
        // Phase 1: embedder + recovery learn the latent space via reconstruction.
        for (int epoch = 0; epoch < phaseDuration; epoch++)
        {
            for (int b = 0; b < sequences.Count; b += batchSize)
            {
                int end = Math.Min(b + batchSize, sequences.Count);
                TrainEmbeddingStepBatched(sequences, b, end);
            }
        }

        // Phase 2: supervisor learns next-step prediction in latent space.
        for (int epoch = 0; epoch < phaseDuration; epoch++)
        {
            for (int b = 0; b < sequences.Count; b += batchSize)
            {
                int end = Math.Min(b + batchSize, sequences.Count);
                TrainSupervisedStepBatched(sequences, b, end);
            }
        }

        // Phase 3: joint adversarial training. Per Yoon 2019 §3.3 schedule:
        // generator/supervisor step + critic step + embedder fine-tune per batch.
        for (int epoch = 0; epoch < phaseDuration; epoch++)
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
        int hiddenDim = _options.HiddenDimension;

        int numSequences = (int)Math.Ceiling((double)numSamples / seqLen);
        var allRows = new List<Vector<T>>();

        for (int s = 0; s < numSequences; s++)
        {
            var noiseSeq = new List<Vector<T>>();
            for (int t = 0; t < seqLen; t++)
            {
                noiseSeq.Add(CreateStandardNormalVector(hiddenDim));
            }

            var fakeEmbeddings = new List<Vector<T>>();
            for (int t = 0; t < seqLen; t++)
            {
                var genOut = GeneratorForward(noiseSeq[t], isTraining: false);
                fakeEmbeddings.Add(genOut);
            }

            var supervisedEmbeddings = new List<Vector<T>>();
            for (int t = 0; t < seqLen; t++)
            {
                var supOut = SupervisorForward(fakeEmbeddings[t], isTraining: false);
                supervisedEmbeddings.Add(supOut);
            }

            for (int t = 0; t < seqLen; t++)
            {
                var recOut = RecoveryForward(supervisedEmbeddings[t], isTraining: false);
                allRows.Add(recOut);
            }
        }

        var result = new Matrix<T>(numSamples, _dataWidth);
        for (int i = 0; i < numSamples && i < allRows.Count; i++)
        {
            for (int j = 0; j < _dataWidth && j < allRows[i].Length; j++)
            {
                result[i, j] = allRows[i][j];
            }
        }

        return result;
    }

    #endregion

    #region Forward Passes with Manual Activation and Pre-Activation Caching

    private Vector<T> EmbedderForward(Vector<T> x, bool isTraining)
    {
        _embedderPreActs.Clear();
        var current = VectorToTensor(x);

        for (int i = 0; i < _embedderLayers.Count; i++)
        {
            current = _embedderLayers[i].Forward(current);
            _embedderPreActs.Add(CloneTensor(current));
            current = ApplySigmoid(current);
        }

        if (_embedderOutput is not null)
        {
            current = _embedderOutput.Forward(current);
        }

        return TensorToVector(current, current.Length);
    }

    private Vector<T> RecoveryForward(Vector<T> h, bool isTraining)
    {
        _recoveryPreActs.Clear();
        var current = VectorToTensor(h);

        for (int i = 0; i < _recoveryLayers.Count; i++)
        {
            current = _recoveryLayers[i].Forward(current);
            _recoveryPreActs.Add(CloneTensor(current));
            current = ApplySigmoid(current);
        }

        if (_recoveryOutput is not null)
        {
            current = _recoveryOutput.Forward(current);
        }

        return TensorToVector(current, current.Length);
    }

    private Vector<T> GeneratorForward(Vector<T> noise, bool isTraining)
    {
        _generatorPreActs.Clear();
        var current = VectorToTensor(noise);

        for (int i = 0; i < Layers.Count; i++)
        {
            current = Layers[i].Forward(current);
            _generatorPreActs.Add(CloneTensor(current));
            current = ApplySigmoid(current);
        }

        return TensorToVector(current, current.Length);
    }

    private Vector<T> SupervisorForward(Vector<T> h, bool isTraining)
    {
        _supervisorPreActs.Clear();
        var current = VectorToTensor(h);

        for (int i = 0; i < _supervisorLayers.Count; i++)
        {
            current = _supervisorLayers[i].Forward(current);
            _supervisorPreActs.Add(CloneTensor(current));
            current = ApplySigmoid(current);
        }

        if (_supervisorOutput is not null)
        {
            current = _supervisorOutput.Forward(current);
        }

        return TensorToVector(current, current.Length);
    }

    private Vector<T> DiscriminatorForward(Vector<T> h, bool isTraining)
    {
        _discPreActs.Clear();
        var current = VectorToTensor(h);

        for (int i = 0; i < _discriminatorLayers.Count; i++)
        {
            current = _discriminatorLayers[i].Forward(current);
            _discPreActs.Add(CloneTensor(current));
            current = ApplyLeakyReLU(current, 0.2);

            if (i < _discDropoutLayers.Count)
            {
                _discDropoutLayers[i].SetTrainingMode(isTraining);
                current = _discDropoutLayers[i].Forward(current);
            }
        }

        if (_discriminatorOutput is not null)
        {
            current = _discriminatorOutput.Forward(current);
        }

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
    }

    private void UpdateSupervisor(T lr)
    {
        foreach (var layer in _supervisorLayers) layer.UpdateParameters(lr);
        _supervisorOutput?.UpdateParameters(lr);
    }

    private void UpdateDiscriminator(T lr)
    {
        foreach (var layer in _discriminatorLayers) layer.UpdateParameters(lr);
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
        var xBatch = BuildFlattenedSequenceBatch(sequences, startIdx, endIdx);
        if (xBatch.Shape[0] == 0) return;

        using var tape = new GradientTape<T>();

        var embedderRecoveryLayers = new List<ILayer<T>>();
        embedderRecoveryLayers.AddRange(_embedderLayers);
        if (_embedderOutput is not null) embedderRecoveryLayers.Add(_embedderOutput);
        embedderRecoveryLayers.AddRange(_recoveryLayers);
        if (_recoveryOutput is not null) embedderRecoveryLayers.Add(_recoveryOutput);
        var paramsList = TapeTrainingStep<T>.CollectParameters(embedderRecoveryLayers);

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
        _optimizer.Step(context);
    }

    /// <summary>
    /// Paper-faithful TimeGAN Phase 2 (Yoon et al. 2019 §3.2):
    /// supervisor next-step prediction in the embedded space.
    /// <c>L_S = E[||h_{t+1} - s(h_t)||_2^2]</c>. Embedder is frozen
    /// (Phase 1 produced it).
    /// </summary>
    private void TrainSupervisedStepBatched(List<Matrix<T>> sequences, int startIdx, int endIdx)
    {
        var (xt, xtNext) = BuildPairedSequenceBatch(sequences, startIdx, endIdx);
        if (xt.Shape[0] == 0) return;

        using var tape = new GradientTape<T>();

        var supervisorLayers = new List<ILayer<T>>();
        supervisorLayers.AddRange(_supervisorLayers);
        if (_supervisorOutput is not null) supervisorLayers.Add(_supervisorOutput);
        var paramsList = TapeTrainingStep<T>.CollectParameters(supervisorLayers);

        // Embedder runs OUTSIDE the tape (frozen for this step).
        var ht = EmbedderForwardBatched(xt, isTraining: false);
        var htNext = EmbedderForwardBatched(xtNext, isTraining: false);

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
        _optimizer.Step(context);
    }

    /// <summary>
    /// Paper-faithful TimeGAN Phase 3 critic step (Yoon et al. 2019 §3.3):
    /// the discriminator learns to distinguish real embedded sequences from
    /// supervisor-rolled-out fake embedded sequences. BCE-with-logits via
    /// tape-tracked Engine.Sigmoid + Engine.TensorLog.
    /// </summary>
    private void TrainDiscriminatorStepBatched(List<Matrix<T>> sequences, int startIdx, int endIdx)
    {
        var xBatch = BuildFlattenedSequenceBatch(sequences, startIdx, endIdx);
        if (xBatch.Shape[0] == 0) return;
        int batchSize = xBatch.Shape[0];
        int hiddenDim = _options.HiddenDimension;

        // Real embedded sequence: x -> embedder. Fake: noise -> generator -> supervisor.
        // Both produced OUTSIDE the critic's tape so the critic only updates its own params.
        var realEmb = EmbedderForwardBatched(xBatch, isTraining: false);
        var noise = GenerateNoiseBatchTensor(batchSize, hiddenDim);
        var fakeEmb = GeneratorForwardBatched(noise, isTraining: false);
        var fakeSup = SupervisorForwardBatched(fakeEmb, isTraining: false);

        using var tape = new GradientTape<T>();

        var discLayers = new List<ILayer<T>>();
        discLayers.AddRange(_discriminatorLayers);
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

        Tensor<T> ComputeForward(Tensor<T> inp, Tensor<T> _) => DiscriminatorForwardBatched(inp, true);
        Tensor<T> RecomputeLoss(Tensor<T> pred, Tensor<T> _) =>
            Engine.TensorNegate(Engine.ReduceMean(LogSigmoid(pred), allAxes, keepDims: false));

        var context = new TapeStepContext<T>(
            paramsList, grads, lossValue,
            realEmb, realEmb, ComputeForward, RecomputeLoss,
            parameterBuffer: null);
        _optimizer.Step(context);
    }

    /// <summary>
    /// Paper-faithful TimeGAN Phase 3 generator + supervisor joint step:
    /// non-saturating adversarial loss + supervised next-step loss in the
    /// embedded space (Yoon et al. 2019 §3.3 joint training).
    /// </summary>
    private void TrainGeneratorStepBatched(List<Matrix<T>> sequences, int startIdx, int endIdx)
    {
        int hiddenDim = _options.HiddenDimension;

        // Phase 3 joint loss (Yoon et al. 2019 §3.3) =
        //   L_U (unsupervised adversarial, non-saturating)
        //   + γ · L_S (supervised next-step MSE on real sequence pairs).
        // Without the L_S term the supervisor is updated only through the
        // adversarial gradient — the next-step temporal structure that
        // L_S explicitly preserves is lost once joint training begins.
        var (xt, xtNext) = BuildPairedSequenceBatch(sequences, startIdx, endIdx);
        int supervisedBatch = xt.Shape[0];

        using var tape = new GradientTape<T>();

        var genSupLayers = new List<ILayer<T>>();
        genSupLayers.AddRange(Layers);
        genSupLayers.AddRange(_supervisorLayers);
        if (_supervisorOutput is not null) genSupLayers.Add(_supervisorOutput);
        var paramsList = TapeTrainingStep<T>.CollectParameters(genSupLayers);

        // Adversarial term: minimize -log σ(D(s(g(z))))
        int advBatch = Math.Max(1, supervisedBatch);
        var noise = GenerateNoiseBatchTensor(advBatch, hiddenDim);
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
            var ht = EmbedderForwardBatched(xt, isTraining: false);
            var htNext = EmbedderForwardBatched(xtNext, isTraining: false);
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

        Tensor<T> ComputeForward(Tensor<T> inp, Tensor<T> _) => DiscriminatorForwardBatched(
            SupervisorForwardBatched(GeneratorForwardBatched(inp, true), true), false);
        Tensor<T> RecomputeLoss(Tensor<T> pred, Tensor<T> _) =>
            Engine.TensorNegate(Engine.ReduceMean(LogSigmoid(pred), advAxes, keepDims: false));

        var context = new TapeStepContext<T>(
            paramsList, grads, lossValue,
            noise, noise, ComputeForward, RecomputeLoss,
            parameterBuffer: null);
        _optimizer.Step(context);
    }

    /// <summary>
    /// Flattens timesteps across a slice of sequences into a single
    /// <c>[batchSize, dataWidth]</c> tensor for batched processing.
    /// Each row is one timestep observation.
    /// </summary>
    private Tensor<T> BuildFlattenedSequenceBatch(List<Matrix<T>> sequences, int startIdx, int endIdx)
    {
        int totalRows = 0;
        for (int s = startIdx; s < endIdx && s < sequences.Count; s++) totalRows += sequences[s].Rows;
        var batch = new Tensor<T>([Math.Max(1, totalRows), _dataWidth]);
        int idx = 0;
        for (int s = startIdx; s < endIdx && s < sequences.Count; s++)
        {
            var seq = sequences[s];
            int cols = Math.Min(_dataWidth, seq.Columns);
            for (int t = 0; t < seq.Rows; t++)
            {
                for (int j = 0; j < cols; j++) batch[idx, j] = seq[t, j];
                idx++;
            }
        }
        // If the slice yielded zero rows, return an empty-batch tensor with
        // first-dim zero so callers can early-exit on shape check.
        return totalRows == 0 ? new Tensor<T>([0, _dataWidth]) : batch;
    }

    /// <summary>
    /// Builds the paired (x_t, x_{t+1}) batches from sequence slices for the
    /// supervisor's next-step prediction objective.
    /// </summary>
    private (Tensor<T> xt, Tensor<T> xtNext) BuildPairedSequenceBatch(List<Matrix<T>> sequences, int startIdx, int endIdx)
    {
        int totalPairs = 0;
        for (int s = startIdx; s < endIdx && s < sequences.Count; s++)
            if (sequences[s].Rows >= 2) totalPairs += sequences[s].Rows - 1;

        if (totalPairs == 0)
            return (new Tensor<T>([0, _dataWidth]), new Tensor<T>([0, _dataWidth]));

        var xt = new Tensor<T>([totalPairs, _dataWidth]);
        var xtNext = new Tensor<T>([totalPairs, _dataWidth]);
        int idx = 0;
        for (int s = startIdx; s < endIdx && s < sequences.Count; s++)
        {
            var seq = sequences[s];
            if (seq.Rows < 2) continue;
            int cols = Math.Min(_dataWidth, seq.Columns);
            for (int t = 0; t < seq.Rows - 1; t++)
            {
                for (int j = 0; j < cols; j++)
                {
                    xt[idx, j] = seq[t, j];
                    xtNext[idx, j] = seq[t + 1, j];
                }
                idx++;
            }
        }
        return (xt, xtNext);
    }

    private Tensor<T> GenerateNoiseBatchTensor(int batchSize, int dim)
    {
        int totalElements = batchSize * dim;
        int halfElements = (totalElements + 1) / 2;
        var u2 = Engine.TensorRandomUniformRange<T>([halfElements], NumOps.Zero, NumOps.One);
        var u1Temp = Engine.TensorRandomUniformRange<T>([halfElements], NumOps.Zero, NumOps.One);
        var u1 = Engine.ScalarMinusTensor(NumOps.One, u1Temp);
        var radius = Engine.TensorSqrt(Engine.TensorMultiplyScalar(Engine.TensorLog(u1), NumOps.FromDouble(-2.0)));
        var theta = Engine.TensorMultiplyScalar(u2, NumOps.FromDouble(2.0 * Math.PI));
        var z1 = Engine.TensorMultiply(radius, Engine.TensorCos(theta));
        var z2 = Engine.TensorMultiply(radius, Engine.TensorSin(theta));
        var noiseData = new T[totalElements];
        var z1Arr = z1.ToArray();
        var z2Arr = z2.ToArray();
        for (int i = 0; i < halfElements; i++)
        {
            int idx = i * 2;
            if (idx < totalElements) noiseData[idx] = z1Arr[i];
            if (idx + 1 < totalElements) noiseData[idx + 1] = z2Arr[i];
        }
        return new Tensor<T>(noiseData, [batchSize, dim]);
    }

    // ----- Batched, tape-tracked forward methods (Engine.Sigmoid / LeakyReLU) -----

    private Tensor<T> EmbedderForwardBatched(Tensor<T> x, bool isTraining)
    {
        var current = x;
        foreach (var l in _embedderLayers) { current = l.Forward(current); current = Engine.Sigmoid(current); }
        if (_embedderOutput is not null) current = _embedderOutput.Forward(current);
        return current;
    }

    private Tensor<T> RecoveryForwardBatched(Tensor<T> h, bool isTraining)
    {
        var current = h;
        foreach (var l in _recoveryLayers) { current = l.Forward(current); current = Engine.Sigmoid(current); }
        if (_recoveryOutput is not null) current = _recoveryOutput.Forward(current);
        return current;
    }

    private Tensor<T> GeneratorForwardBatched(Tensor<T> noise, bool isTraining)
    {
        var current = noise;
        foreach (var l in Layers) { current = l.Forward(current); current = Engine.Sigmoid(current); }
        return current;
    }

    private Tensor<T> SupervisorForwardBatched(Tensor<T> h, bool isTraining)
    {
        var current = h;
        foreach (var l in _supervisorLayers) { current = l.Forward(current); current = Engine.Sigmoid(current); }
        if (_supervisorOutput is not null) current = _supervisorOutput.Forward(current);
        return current;
    }

    private Tensor<T> DiscriminatorForwardBatched(Tensor<T> h, bool isTraining)
    {
        var current = h;
        T leakySlope = NumOps.FromDouble(0.2);
        for (int i = 0; i < _discriminatorLayers.Count; i++)
        {
            current = _discriminatorLayers[i].Forward(current);
            current = Engine.LeakyReLU(current, leakySlope);
            if (i < _discDropoutLayers.Count)
            {
                _discDropoutLayers[i].SetTrainingMode(isTraining);
                current = _discDropoutLayers[i].Forward(current);
            }
        }
        if (_discriminatorOutput is not null) current = _discriminatorOutput.Forward(current);
        return current;
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
    /// Builds a combined list of discriminator layers (dense + dropout + output)
    /// for gradient-penalty and related analyses.
    /// </summary>
    private IReadOnlyList<ILayer<T>> BuildDiscLayerList()
    {
        var allLayers = new List<ILayer<T>>();
        for (int i = 0; i < _discDropoutLayers.Count; i++)
        {
            allLayers.Add(_discriminatorLayers[i]);
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

    /// <inheritdoc />
    public override Tensor<T> Predict(Tensor<T> input)
    {
        if (!IsFitted) return input;

        var noise = TensorToVector(input, input.Length);
        var genOut = GeneratorForward(noise, isTraining: false);
        var supOut = SupervisorForward(genOut, isTraining: false);
        var recOut = RecoveryForward(supOut, isTraining: false);
        return VectorToTensor(recOut);
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
    public override void UpdateParameters(Vector<T> parameters)
    {
        int startIndex = 0;
        foreach (var layer in Layers)
        {
            int layerParameterCount = checked((int)layer.ParameterCount);
            if (layerParameterCount > 0)
            {
                Vector<T> layerParameters = parameters.SubVector(startIndex, layerParameterCount);
                layer.UpdateParameters(layerParameters);
                startIndex += layerParameterCount;
            }
        }
    }

    /// <inheritdoc />
    protected override void SerializeNetworkSpecificData(BinaryWriter writer)
    {
        writer.Write(_options.SequenceLength);
        writer.Write(_options.HiddenDimension);
        writer.Write(_options.NumLayers);
        writer.Write(_dataWidth);
        writer.Write(IsFitted);
    }

    /// <inheritdoc />
    protected override void DeserializeNetworkSpecificData(BinaryReader reader)
    {
        _ = reader.ReadInt32(); // SequenceLength
        _ = reader.ReadInt32(); // HiddenDimension
        _ = reader.ReadInt32(); // NumLayers
        _dataWidth = reader.ReadInt32();
        IsFitted = reader.ReadBoolean();
    }

    /// <inheritdoc />
    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance()
    {
        return new TimeGANGenerator<T>(Architecture, _options);
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
                ["HiddenDimension"] = _options.HiddenDimension,
                ["NumLayers"] = _options.NumLayers,
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
