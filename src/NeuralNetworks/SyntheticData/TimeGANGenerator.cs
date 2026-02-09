using AiDotNet.ActivationFunctions;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.NeuralNetworks.SyntheticData;
using AiDotNet.Optimizers;
using AiDotNet.Tensors.Helpers;

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
                Layers.Add(new FullyConnectedLayer<T>(hiddenDim, hiddenDim, identity));
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
                Layers.Add(new FullyConnectedLayer<T>(layerInput, hiddenDim, identity));
            }
        }

        // Generator output head (always auxiliary)
        // (Note: we store this separately as the last layer is a projection to hiddenDim)

        // Embedder
        _embedderLayers.Clear();
        for (int i = 0; i < _options.NumLayers; i++)
        {
            int layerInput = i == 0 ? _dataWidth : hiddenDim;
            _embedderLayers.Add(new FullyConnectedLayer<T>(layerInput, hiddenDim, identity));
        }
        _embedderOutput = new FullyConnectedLayer<T>(hiddenDim, hiddenDim, identity);

        // Recovery
        _recoveryLayers.Clear();
        for (int i = 0; i < _options.NumLayers; i++)
        {
            _recoveryLayers.Add(new FullyConnectedLayer<T>(hiddenDim, hiddenDim, identity));
        }
        _recoveryOutput = new FullyConnectedLayer<T>(hiddenDim, _dataWidth, identity);

        // Supervisor
        _supervisorLayers.Clear();
        for (int i = 0; i < _options.NumLayers - 1; i++)
        {
            _supervisorLayers.Add(new FullyConnectedLayer<T>(hiddenDim, hiddenDim, identity));
        }
        _supervisorOutput = new FullyConnectedLayer<T>(hiddenDim, hiddenDim, identity);

        // Discriminator
        _discriminatorLayers.Clear();
        _discDropoutLayers.Clear();
        for (int i = 0; i < _options.NumLayers; i++)
        {
            _discriminatorLayers.Add(new FullyConnectedLayer<T>(hiddenDim, hiddenDim, identity));
            _discDropoutLayers.Add(new DropoutLayer<T>(_options.DiscriminatorDropout));
        }
        _discriminatorOutput = new FullyConnectedLayer<T>(hiddenDim, 1, identity);
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

        // Phase 1: Embedding training
        for (int epoch = 0; epoch < phaseDuration; epoch++)
        {
            for (int b = 0; b < sequences.Count; b += batchSize)
            {
                int end = Math.Min(b + batchSize, sequences.Count);
                TrainEmbeddingStep(sequences, b, end, lr);
            }
        }

        // Phase 2: Supervised training
        for (int epoch = 0; epoch < phaseDuration; epoch++)
        {
            for (int b = 0; b < sequences.Count; b += batchSize)
            {
                int end = Math.Min(b + batchSize, sequences.Count);
                TrainSupervisedStep(sequences, b, end, lr);
            }
        }

        // Phase 3: Joint training
        for (int epoch = 0; epoch < phaseDuration; epoch++)
        {
            for (int b = 0; b < sequences.Count; b += batchSize)
            {
                int end = Math.Min(b + batchSize, sequences.Count);
                TrainGeneratorJointStep(sequences, b, end, lr);
                TrainGeneratorJointStep(sequences, b, end, lr);
                TrainDiscriminatorStep(sequences, b, end, lr);
                TrainEmbeddingStep(sequences, b, end, lr);
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

    private Tensor<T> BackwardEmbedder(Tensor<T> grad)
    {
        var current = grad;
        if (_embedderOutput is not null) current = _embedderOutput.Backward(current);

        for (int i = _embedderLayers.Count - 1; i >= 0; i--)
        {
            if (i < _embedderPreActs.Count) current = ApplySigmoidDerivative(current, _embedderPreActs[i]);
            current = _embedderLayers[i].Backward(current);
        }
        return current;
    }

    private Tensor<T> BackwardRecovery(Tensor<T> grad)
    {
        var current = grad;
        if (_recoveryOutput is not null) current = _recoveryOutput.Backward(current);

        for (int i = _recoveryLayers.Count - 1; i >= 0; i--)
        {
            if (i < _recoveryPreActs.Count) current = ApplySigmoidDerivative(current, _recoveryPreActs[i]);
            current = _recoveryLayers[i].Backward(current);
        }
        return current;
    }

    private Tensor<T> BackwardGenerator(Tensor<T> grad)
    {
        var current = grad;
        for (int i = Layers.Count - 1; i >= 0; i--)
        {
            if (i < _generatorPreActs.Count) current = ApplySigmoidDerivative(current, _generatorPreActs[i]);
            current = Layers[i].Backward(current);
        }
        return current;
    }

    private Tensor<T> BackwardSupervisor(Tensor<T> grad)
    {
        var current = grad;
        if (_supervisorOutput is not null) current = _supervisorOutput.Backward(current);

        for (int i = _supervisorLayers.Count - 1; i >= 0; i--)
        {
            if (i < _supervisorPreActs.Count) current = ApplySigmoidDerivative(current, _supervisorPreActs[i]);
            current = _supervisorLayers[i].Backward(current);
        }
        return current;
    }

    private void BackwardDiscriminator(Tensor<T> grad)
    {
        var current = grad;
        if (_discriminatorOutput is not null) current = _discriminatorOutput.Backward(current);

        for (int i = _discriminatorLayers.Count - 1; i >= 0; i--)
        {
            if (i < _discDropoutLayers.Count) current = _discDropoutLayers[i].Backward(current);
            if (i < _discPreActs.Count) current = ApplyLeakyReLUDerivative(current, _discPreActs[i], 0.2);
            current = _discriminatorLayers[i].Backward(current);
        }
    }

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

    private void TrainEmbeddingStep(List<Matrix<T>> sequences, int startIdx, int endIdx, T lr)
    {
        for (int s = startIdx; s < endIdx; s++)
        {
            var seq = sequences[s];
            for (int t = 0; t < seq.Rows; t++)
            {
                var x = GetRow(seq, t);
                var embedding = EmbedderForward(x, isTraining: true);
                var recovered = RecoveryForward(embedding, isTraining: true);

                var grad = new Tensor<T>([recovered.Length]);
                int gradLen = Math.Min(grad.Length, x.Length);
                for (int j = 0; j < gradLen; j++)
                {
                    double diff = NumOps.ToDouble(recovered[j]) - NumOps.ToDouble(x[j]);
                    grad[j] = NumOps.FromDouble(2.0 * diff * _options.ReconstructionWeight);
                }
                grad = SanitizeAndClipGradient(grad, 5.0);

                var recoveryGrad = BackwardRecovery(grad);
                BackwardEmbedder(recoveryGrad);
                UpdateRecovery(lr);
                UpdateEmbedder(lr);
            }
        }
    }

    private void TrainSupervisedStep(List<Matrix<T>> sequences, int startIdx, int endIdx, T lr)
    {
        int hiddenDim = _options.HiddenDimension;
        for (int s = startIdx; s < endIdx; s++)
        {
            var seq = sequences[s];
            if (seq.Rows < 2) continue;

            for (int t = 0; t < seq.Rows - 1; t++)
            {
                var xt = GetRow(seq, t);
                var xtNext = GetRow(seq, t + 1);
                var ht = EmbedderForward(xt, isTraining: false);
                var htNext = EmbedderForward(xtNext, isTraining: false);
                var htPred = SupervisorForward(ht, isTraining: true);

                var grad = new Tensor<T>([htPred.Length]);
                for (int j = 0; j < grad.Length && j < htNext.Length; j++)
                {
                    double diff = NumOps.ToDouble(htPred[j]) - NumOps.ToDouble(htNext[j]);
                    grad[j] = NumOps.FromDouble(2.0 * diff);
                }

                BackwardSupervisor(grad);
                UpdateSupervisor(lr);
            }
        }
    }

    private void TrainGeneratorJointStep(List<Matrix<T>> sequences, int startIdx, int endIdx, T lr)
    {
        int hiddenDim = _options.HiddenDimension;
        for (int s = startIdx; s < endIdx; s++)
        {
            var seq = sequences[s];
            for (int t = 0; t < seq.Rows; t++)
            {
                var noise = CreateStandardNormalVector(hiddenDim);
                var fakeEmb = GeneratorForward(noise, isTraining: true);
                var fakeSup = SupervisorForward(fakeEmb, isTraining: true);
                var discFake = DiscriminatorForward(fakeSup, isTraining: false);

                double dScore = NumOps.ToDouble(discFake[0]);
                double sigD = SigmoidScalar(dScore);
                double advGrad = -(1.0 - sigD);

                // Compute discriminator input gradient using TapeLayerBridge autodiff
                var allDiscLayers = BuildDiscLayerList();
                var supGrad = TapeLayerBridge<T>.ComputeInputGradient(
                    VectorToTensor(fakeSup),
                    allDiscLayers,
                    TapeLayerBridge<T>.HiddenActivation.LeakyReLU,
                    applyActivationOnLast: false);

                // Scale by advGrad (chain rule for non-saturating loss)
                for (int g = 0; g < supGrad.Length; g++)
                {
                    supGrad[g] = NumOps.FromDouble(NumOps.ToDouble(supGrad[g]) * advGrad);
                }
                supGrad = SanitizeAndClipGradient(supGrad, 5.0);

                if (t < seq.Rows - 1)
                {
                    var xtNext = GetRow(seq, t + 1);
                    var htNext = EmbedderForward(xtNext, isTraining: false);
                    for (int j = 0; j < hiddenDim && j < supGrad.Length; j++)
                    {
                        double diff = NumOps.ToDouble(fakeSup[j]) - NumOps.ToDouble(htNext[j]);
                        supGrad[j] = NumOps.Add(supGrad[j],
                            NumOps.FromDouble(2.0 * diff * _options.SupervisedWeight));
                    }
                }

                _ = GeneratorForward(noise, isTraining: true);
                _ = SupervisorForward(fakeEmb, isTraining: true);

                var genGrad = BackwardSupervisor(supGrad);
                BackwardGenerator(genGrad);
                UpdateGenerator(lr);
            }
        }
    }

    private void TrainDiscriminatorStep(List<Matrix<T>> sequences, int startIdx, int endIdx, T lr)
    {
        int hiddenDim = _options.HiddenDimension;
        for (int s = startIdx; s < endIdx; s++)
        {
            var seq = sequences[s];
            for (int t = 0; t < seq.Rows; t++)
            {
                var x = GetRow(seq, t);
                var realEmb = EmbedderForward(x, isTraining: false);
                var discReal = DiscriminatorForward(realEmb, isTraining: true);

                double realScore = NumOps.ToDouble(discReal[0]);
                double sigReal = SigmoidScalar(realScore);
                var gradReal = new Tensor<T>([1]);
                gradReal[0] = NumOps.FromDouble(-(1.0 - sigReal));
                BackwardDiscriminator(gradReal);
                UpdateDiscriminator(lr);

                var noise = CreateStandardNormalVector(hiddenDim);
                var fakeEmb = GeneratorForward(noise, isTraining: false);
                var fakeSup = SupervisorForward(fakeEmb, isTraining: false);
                var discFake = DiscriminatorForward(fakeSup, isTraining: true);

                double fakeScore = NumOps.ToDouble(discFake[0]);
                double sigFake = SigmoidScalar(fakeScore);
                var gradFake = new Tensor<T>([1]);
                gradFake[0] = NumOps.FromDouble(sigFake);
                BackwardDiscriminator(gradFake);
                UpdateDiscriminator(lr);
            }
        }
    }

    #endregion

    #region Discriminator Layer List

    /// <summary>
    /// Builds a combined list of discriminator layers for TapeLayerBridge.
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
            int layerParameterCount = layer.ParameterCount;
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
            ModelType = ModelType.NeuralNetwork,
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
        var result = new Tensor<T>(input.Shape);
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
        var result = new Tensor<T>(grad.Shape);
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
        var result = new Tensor<T>(input.Shape);
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
        var result = new Tensor<T>(grad.Shape);
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
        var clone = new Tensor<T>(source.Shape);
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

    #region IJitCompilable Override

    /// <summary>
    /// TimeGAN uses 5 interacting networks (embedding, recovery, generator, discriminator, supervisor)
    /// which cannot be represented as a single computation graph.
    /// </summary>
    public override bool SupportsJitCompilation => false;

    #endregion
}
