using AiDotNet.ActivationFunctions;
using AiDotNet.Attributes;
using AiDotNet.Autodiff;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Optimizers;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Training;

namespace AiDotNet.NeuralNetworks.SyntheticData;

/// <summary>
/// MedSynth generator for privacy-preserving medical tabular data synthesis using a
/// VAE/GAN hybrid with clinical validity constraints and optional differential privacy.
/// </summary>
/// <remarks>
/// <para>
/// MedSynth combines VAE and GAN approaches with medical domain constraints:
///
/// <code>
///  Data --> Encoder --> (mean, logvar) --> z --> Decoder --> Reconstructed --> Constraint Layer
///                                                     |
///                                              Discriminator --> Real/Fake?
/// </code>
///
/// Training alternates between three objectives:
/// 1. <b>VAE loss</b>: Reconstruction + KL divergence + constraint violation penalty
/// 2. <b>Discriminator loss</b>: BCE on real vs fake samples
/// 3. <b>Adversarial loss</b>: Non-saturating generator loss through discriminator input gradients
///
/// This implementation follows the standard neural network architecture pattern with:
/// - Proper inheritance from NeuralNetworkBase
/// - Layer-based architecture using ILayer components
/// - Full autodiff and JIT compilation support
/// </para>
/// <para>
/// <b>For Beginners:</b> MedSynth ensures generated medical data is:
///
/// 1. Realistic (VAE reconstruction + GAN adversarial training)
/// 2. Valid (no impossible lab values or vital signs)
/// 3. Private (optional differential privacy protection)
///
/// If you provide custom layers in the architecture, those will be used directly
/// for the decoder network. Otherwise, standard layers are created.
///
/// Example usage:
/// <code>
/// var architecture = new NeuralNetworkArchitecture&lt;double&gt;(inputFeatures: 64, outputSize: 50);
/// var options = new MedSynthOptions&lt;double&gt; { LatentDimension = 64, EnablePrivacy = true };
/// var medsynth = new MedSynthGenerator&lt;double&gt;(architecture, options);
/// medsynth.Fit(data, columns, epochs: 500);
/// var synthetic = medsynth.Generate(1000);
/// </code>
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
[ModelDomain(ModelDomain.MachineLearning)]
[ModelCategory(ModelCategory.NeuralNetwork)]
[ModelCategory(ModelCategory.SyntheticDataGenerator)]
[ModelTask(ModelTask.Generation)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("Generation and Evaluation of Synthetic Patient Data",
    "https://arxiv.org/abs/1909.02662",
    Year = 2020,
    Authors = "Andrew Yale, Saloni Dash, Ritik Dutta, Isabelle Guyon, Adrien Pavao, Kristin P. Bennett")]
public class MedSynthGenerator<T> : NeuralNetworkBase<T>, ISyntheticTabularGenerator<T>
{
    private readonly MedSynthOptions<T> _options;
    // One dedicated optimizer per sub-network (discriminator / generator / VAE).
    // See CTGANGenerator: a single shared AdamOptimizer corrupts its flat moment
    // buffer across networks of different parameter counts. MedSynth trains three
    // distinct param sets, so it needs three independent optimizers.
    private IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> _generatorOptimizer;
    private IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> _discriminatorOptimizer;
    private IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> _vaeOptimizer;
    private ILossFunction<T> _lossFunction;
    private Random _random;

    // ISyntheticTabularGenerator state
    private List<ColumnMetadata> _columns = new();
    private TabularDataTransformer<T>? _transformer;

    // VAE encoder (auxiliary - not user-overridable)
    private readonly List<FullyConnectedLayer<T>> _encoderLayers = new();
    private FullyConnectedLayer<T>? _meanHead;
    private FullyConnectedLayer<T>? _logvarHead;
    private readonly List<Tensor<T>> _encoderPreActs = new();

    // Decoder hidden FCs (paired 1:1 with _decoderBN). NOT the full
    // Layers list — Layers also contains encoder/VAE-head/discriminator
    // sub-graphs, and walking it in the decoder forward path would feed
    // latent noise through the encoder and discriminator weights. The
    // decoder pass MUST use this slice only.
    private readonly List<FullyConnectedLayer<T>> _decoderLayers = new();
    private readonly List<BatchNormalizationLayer<T>> _decoderBN = new();
    private FullyConnectedLayer<T>? _decoderOutput;
    private readonly List<Tensor<T>> _decoderPreActs = new();

    // Discriminator (auxiliary - not user-overridable)
    private readonly List<FullyConnectedLayer<T>> _discLayers = new();
    private readonly List<DropoutLayer<T>> _discDropout = new();
    private FullyConnectedLayer<T>? _discOutput;
    private readonly List<Tensor<T>> _discPreActs = new();

    // Clinical constraints (learned from data)
    private double[]? _colMin;
    private double[]? _colMax;

    // Whether custom layers are being used
    private bool _usingCustomLayers;

    private int _dataWidth;

    /// <summary>
    /// Gets the MedSynth-specific options.
    /// </summary>
    public new MedSynthOptions<T> Options => _options;

    /// <inheritdoc />
    public IReadOnlyList<ColumnMetadata> Columns => _columns.AsReadOnly();

    /// <inheritdoc />
    public bool IsFitted { get; private set; }

    /// <summary>
    /// Initializes a new MedSynth generator with the specified architecture.
    /// </summary>
    /// <param name="architecture">The neural network architecture defining input/output dimensions and optional custom layers.</param>
    /// <param name="options">MedSynth-specific options for generation configuration.</param>
    /// <param name="optimizer">Gradient-based optimizer (defaults to Adam).</param>
    /// <param name="lossFunction">Loss function (defaults based on task type).</param>
    /// <param name="maxGradNorm">Maximum gradient norm for clipping (default 5.0).</param>
    /// <summary>
    /// Initializes a new instance with default architecture settings.
    /// </summary>
    public MedSynthGenerator()
        : this(new NeuralNetworkArchitecture<T>(
            inputType: Enums.InputType.OneDimensional,
            taskType: Enums.NeuralNetworkTaskType.Regression,
            inputSize: 10,
            outputSize: 10))
    {
    }

    public MedSynthGenerator(
        NeuralNetworkArchitecture<T> architecture,
        MedSynthOptions<T>? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
        ILossFunction<T>? lossFunction = null,
        double maxGradNorm = 5.0)
        : base(architecture, lossFunction ?? NeuralNetworkHelper<T>.GetDefaultLossFunction(architecture.TaskType), maxGradNorm)
    {
        _options = options ?? new MedSynthOptions<T>();
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
        _vaeOptimizer = MakeAdam();
        _random = _options.Seed.HasValue
            ? RandomHelper.CreateSeededRandom(_options.Seed.Value)
            : RandomHelper.CreateSecureRandom();

        InitializeLayers();
    }

    #region Layer Initialization (GANDALF Pattern)

    /// <summary>
    /// Initializes the layers of the MedSynth network based on the provided architecture.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method sets up the decoder network:
    /// - If you provided custom layers, those are used for the decoder
    /// - Otherwise, standard decoder layers are created
    ///
    /// The encoder, discriminator, mean/logvar heads, and decoder output are always
    /// created internally and are not user-overridable.
    /// </para>
    /// </remarks>
    protected override void InitializeLayers()
    {
        Layers.Clear();

        if (Architecture.Layers is not null && Architecture.Layers.Count > 0)
        {
            Layers.AddRange(Architecture.Layers);
            _usingCustomLayers = true;
        }
        else
        {
            _dataWidth = Math.Max(1, Architecture.OutputSize);
            var allLayers = LayerHelper<T>.CreateDefaultMedSynthLayers(
                _dataWidth, _options.LatentDimension,
                _options.EncoderDimensions, _options.DiscriminatorDimensions,
                _options.DiscriminatorDropout).ToList();
            Layers.AddRange(allLayers);
            _usingCustomLayers = false;
        }

        ExtractMedSynthLayerReferences();
    }

    /// <summary>
    /// Before Fit() supplies the real transformed width, adapt the encoder/decoder/
    /// discriminator layout to the actual input width so the model is a valid network
    /// for any 1-D input — the generated ModelFamily tests call Train()/Predict()
    /// directly without Fit(). Once fitted, the width is fixed by the transformer.
    /// </summary>
    private void EnsureSizedForInput(Tensor<T> input)
    {
        if (!IsFitted && !_usingCustomLayers && input.Length != _dataWidth && input.Length > 0)
        {
            _dataWidth = input.Length;
            RebuildLayersWithActualDimensions();
        }
    }

    /// <summary>
    /// Deterministic VAE reconstruction (encode → latent mean → decode), tape-connected
    /// and shared by <see cref="Predict"/> and <see cref="ForwardForTraining"/>. Uses the
    /// latent MEAN (no sampling) so inference is deterministic; the full sampled VAE+GAN
    /// objective runs in Fit.
    /// </summary>
    private Tensor<T> ReconstructForward(Tensor<T> input)
    {
        EnsureSizedForInput(input);
        var hidden = EncoderForwardBatched(input);
        var z = _meanHead is not null ? _meanHead.Forward(hidden) : hidden;
        return DecoderForwardBatched(z, isTraining: IsTrainingMode);
    }

    /// <summary>
    /// Extracts private layer references as aliases from the unified Layers list.
    /// </summary>
    private void ExtractMedSynthLayerReferences()
    {
        _encoderLayers.Clear();
        _decoderLayers.Clear();
        _decoderBN.Clear();
        _discLayers.Clear();
        _discDropout.Clear();

        int idx = 0;
        var dims = _options.EncoderDimensions;
        var discDims = _options.DiscriminatorDimensions;

        // Encoder layers
        for (int i = 0; i < dims.Length && idx < Layers.Count; i++)
        {
            if (Layers[idx] is FullyConnectedLayer<T> enc)
                _encoderLayers.Add(enc);
            idx++;
        }

        // VAE heads
        if (idx < Layers.Count && Layers[idx] is FullyConnectedLayer<T> mean)
        { _meanHead = mean; idx++; }
        if (idx < Layers.Count && Layers[idx] is FullyConnectedLayer<T> logvar)
        { _logvarHead = logvar; idx++; }

        // Decoder layers (FC + BN pairs). Capture the FC alias too so the
        // decoder forward path can walk decoder-only weights without
        // touching encoder/discriminator sub-graphs that share Layers.
        for (int i = 0; i < dims.Length && idx < Layers.Count; i++)
        {
            if (Layers[idx] is FullyConnectedLayer<T> decFc)
                _decoderLayers.Add(decFc);
            idx++; // advance past FC
            if (idx < Layers.Count && Layers[idx] is BatchNormalizationLayer<T> bn)
            { _decoderBN.Add(bn); idx++; }
        }

        // Decoder output
        if (idx < Layers.Count && Layers[idx] is FullyConnectedLayer<T> decOut)
        { _decoderOutput = decOut; idx++; }

        // Discriminator layers (FC + Dropout pairs)
        for (int i = 0; i < discDims.Length && idx < Layers.Count; i++)
        {
            if (Layers[idx] is FullyConnectedLayer<T> disc)
            { _discLayers.Add(disc); idx++; }
            if (idx < Layers.Count && Layers[idx] is DropoutLayer<T> drop)
            { _discDropout.Add(drop); idx++; }
        }

        // Discriminator output
        if (idx < Layers.Count && Layers[idx] is FullyConnectedLayer<T> discOut)
        { _discOutput = discOut; idx++; }
    }

    /// <summary>
    /// Rebuilds all layers with actual data dimensions discovered during Fit().
    /// </summary>
    private void RebuildLayersWithActualDimensions()
    {
        if (!_usingCustomLayers)
        {
            // Rebuild ALL layers via LayerHelper with actual data dimensions
            Layers.Clear();
            var allLayers = LayerHelper<T>.CreateDefaultMedSynthLayers(
                _dataWidth, _options.LatentDimension,
                _options.EncoderDimensions, _options.DiscriminatorDimensions,
                _options.DiscriminatorDropout).ToList();
            Layers.AddRange(allLayers);

            ExtractMedSynthLayerReferences();
        }
    }

    #endregion

    #region ISyntheticTabularGenerator Implementation

    /// <inheritdoc />
    public void Fit(Matrix<T> data, IReadOnlyList<ColumnMetadata> columns, int epochs)
    {
        _columns = new List<ColumnMetadata>(columns);

        _transformer = new TabularDataTransformer<T>(_options.VGMModes, _random);
        _transformer.Fit(data, columns);
        _dataWidth = _transformer.TransformedWidth;
        var transformedData = _transformer.Transform(data);

        // Learn clinical constraints from data
        LearnConstraints(transformedData);

        // Build all networks with actual dimensions
        RebuildLayersWithActualDimensions();

        // Compute noise multiplier for DP if enabled
        double noiseMultiplier = 0;
        if (_options.EnablePrivacy)
        {
            noiseMultiplier = ComputeNoiseMultiplier(data.Rows, epochs);
        }

        int batchSize = Math.Min(_options.BatchSize, data.Rows);
        T lr = NumOps.FromDouble(_options.LearningRate / Math.Max(batchSize, 1));

        for (int epoch = 0; epoch < epochs; epoch++)
        {
            for (int b = 0; b < data.Rows; b += batchSize)
            {
                int end = Math.Min(b + batchSize, data.Rows);

                // Paper-faithful MedSynth (VAE+GAN hybrid) training step:
                // 1) VAE half — encoder + (μ, log σ²) + decoder updated against
                //    reconstruction + KL + clinical-constraint losses. This
                //    is what makes MedSynth a VAE+GAN hybrid rather than a
                //    plain GAN; dropping it (the prior code path did) leaves
                //    _encoderLayers / _meanHead / _logvarHead untrained and
                //    relies on post-hoc clamping in Generate().
                // 2) GAN inner loop — DP-SGD discriminator updates (Abadi
                //    2016) interleaved with non-saturating generator updates
                //    (Goodfellow 2014 §3).
                TrainVaeStepBatched(transformedData, b, end);
                for (int d = 0; d < _options.DiscriminatorSteps; d++)
                {
                    TrainDiscriminatorStepBatched(transformedData, b, end, noiseMultiplier);
                }
                TrainGeneratorStepBatched(end - b);
            }
        }

        IsFitted = true;
    }

    /// <inheritdoc />
    public Task FitAsync(Matrix<T> data, IReadOnlyList<ColumnMetadata> columns, int epochs,
        CancellationToken cancellationToken = default)
    {
        return Task.Run(() => Fit(data, columns, epochs), cancellationToken);
    }

    /// <inheritdoc />
    public Matrix<T> Generate(int numSamples, Vector<T>? conditionColumn = null, Vector<T>? conditionValue = null)
    {
        if (_transformer is null || _decoderOutput is null)
        {
            throw new InvalidOperationException("Generator is not fitted. Call Fit() before Generate().");
        }

        int latentDim = _options.LatentDimension;
        var result = new Matrix<T>(numSamples, _dataWidth);

        for (int i = 0; i < numSamples; i++)
        {
            var z = CreateStandardNormalVector(latentDim);
            var decoded = DecoderForward(z, isTraining: false);
            var constrained = ApplyConstraints(decoded);
            var activated = ApplyOutputActivations(constrained);

            for (int j = 0; j < _dataWidth && j < activated.Length; j++)
            {
                result[i, j] = activated[j];
            }
        }

        return _transformer.InverseTransform(result);
    }

    #endregion

    #region NeuralNetworkBase Overrides

    /// <inheritdoc />
    protected override Tensor<T> PredictCore(Tensor<T> input)
    {
        // Deterministic VAE reconstruction of the input row (encode → latent mean →
        // decode). Adapts to the input width when unfitted so the model is a valid
        // network for the generated ModelFamily tests (which Predict without Fit).
        return ReconstructForward(input);
    }

    /// <summary>
    /// Training forward — the same deterministic VAE reconstruction as <see cref="Predict"/>,
    /// overridden so the tape-based <see cref="NeuralNetworkBase{T}.Train"/> path trains the
    /// encoder + latent-mean head + decoder rather than walking the full Layers list (which
    /// also holds the discriminator and would mis-chain encoder→heads→decoder→discriminator).
    /// </summary>
    public override Tensor<T> ForwardForTraining(Tensor<T> input) => ReconstructForward(input);

    /// <inheritdoc />
    public override void Train(Tensor<T> input, Tensor<T> expectedOutput)
    {
        SetTrainingMode(true);
        try
        {
            TrainWithTape(input, expectedOutput, _vaeOptimizer);
        }
        finally
        {
            SetTrainingMode(false);
        }
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
        writer.Write(_dataWidth);
        writer.Write(IsFitted);
    }

    /// <inheritdoc />
    protected override void DeserializeNetworkSpecificData(BinaryReader reader)
    {
        _dataWidth = reader.ReadInt32();
        IsFitted = reader.ReadBoolean();

        // The base deserializer rebuilt Layers with fresh instances; re-bind the
        // typed encoder/head/decoder/discriminator references from it so the
        // VAE reconstruction forward uses the deserialized (trained) weights
        // rather than the clone's discarded constructor init.
        if (!_usingCustomLayers) ExtractMedSynthLayerReferences();
    }

    /// <inheritdoc />
    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance()
    {
        return new MedSynthGenerator<T>(Architecture, _options);
    }

    /// <inheritdoc />
    public override Dictionary<string, T> GetFeatureImportance()
    {
        return new Dictionary<string, T>();
    }

    /// <inheritdoc />
    public override ModelMetadata<T> GetModelMetadata()
    {
        return new ModelMetadata<T>
        {
            AdditionalInfo = new Dictionary<string, object>
            {
                ["GeneratorType"] = "MedSynth",
                ["LatentDimension"] = _options.LatentDimension,
                ["EnablePrivacy"] = _options.EnablePrivacy,
                ["Epsilon"] = _options.Epsilon,
                ["IsFitted"] = IsFitted
            }
        };
    }

    #endregion

    #region Training

    /// <summary>
    /// Paper-faithful DP-SGD discriminator step (Abadi et al. 2016 §3 layered
    /// over MedSynth's VAE+GAN hybrid). Computes the BCE discriminator loss
    /// gradient via <see cref="GradientTape{T}"/>, applies the canonical
    /// DP-SGD primitive (per-parameter L2 clip + Gaussian noise on the
    /// gradient, not on the parameter), then hands the noised gradient to
    /// the optimizer through a <see cref="TapeStepContext{T}"/>.
    /// Replaces the prior manual <c>DiscriminatorForward → UpdateDiscriminator</c>
    /// path which threw "Backward pass must be called before updating
    /// parameters" on the tape-only autodiff stack.
    /// </summary>
    private void TrainDiscriminatorStepBatched(Matrix<T> data, int startRow, int endRow, double noiseMultiplier)
    {
        if (_discOutput is null) return;

        int batchSize = endRow - startRow;
        if (batchSize <= 0) return;

        if (noiseMultiplier > 0)
        {
            // Paper-faithful per-example DP-SGD (Abadi et al. 2016
            // Algorithm 1). The earlier implementation called
            // tape.ComputeGradients once on the FULL batch — that returns
            // already-aggregated gradients, after which a single
            // per-parameter L2 clip + noise no longer matches the
            // Abadi mechanism. The privacy proof's L2-sensitivity bound
            // requires clipping every per-example gradient INDIVIDUALLY
            // (against the GLOBAL norm across all parameters concatenated)
            // BEFORE aggregation. We therefore replay forward+backward L
            // times — one microbatch per example — clip each per-example
            // gradient by min(1, C / ||g_x||_2), accumulate the clipped
            // gradients, add a single Gaussian noise draw N(0, σ²C²I)
            // to the sum, and divide by L.
            TrainDiscriminatorStepPerExampleDPSGD(data, startRow, endRow, noiseMultiplier);
            return;
        }

        // Non-DP fast path: single batched forward+backward.
        var (realBatch, fakeBatch) = BuildRealAndFakeBatches(data, startRow, endRow);

        using var tape = new GradientTape<T>();
        var discParams = TapeTrainingStep<T>.CollectParameters(BuildDiscLayerList());

        var realScores = DiscriminatorForwardBatched(realBatch, isTraining: true);
        var fakeScores = DiscriminatorForwardBatched(fakeBatch, isTraining: true);

        // BCE-with-logits objective (Goodfellow 2014 GAN, also used by
        // MedSynth per paper §3.2): D maximizes log σ(D(real)) + log(1 - σ(D(fake))).
        // Tape-friendly form: -log σ(real) - log σ(-fake) where σ is sigmoid.
        var allAxes = Enumerable.Range(0, realScores.Shape.Length).ToArray();
        var negFakeScores = Engine.TensorNegate(fakeScores);
        var lossReal = Engine.TensorNegate(Engine.ReduceMean(LogSigmoid(realScores), allAxes, keepDims: false));
        var lossFake = Engine.TensorNegate(Engine.ReduceMean(LogSigmoid(negFakeScores), allAxes, keepDims: false));
        var lossTensor = Engine.TensorAdd(lossReal, lossFake);

        var grads = tape.ComputeGradients(lossTensor, discParams);

        T lossValue = lossTensor.Length > 0 ? lossTensor[0] : NumOps.Zero;
        // Replay-correct closure: lossTensor was (lossReal + lossFake), so
        // RecomputeLoss must replay BOTH BCE terms to stay tied to the
        // objective that produced `grads`. Capture fakeBatch in the closure
        // so the fake side can be re-scored on optimizer.Step replay.
        var capturedFakeBatch = fakeBatch;
        Tensor<T> ComputeForward(Tensor<T> inp, Tensor<T> _) => DiscriminatorForwardBatched(inp, true);
        Tensor<T> RecomputeLoss(Tensor<T> predReal, Tensor<T> _)
        {
            var predFake = DiscriminatorForwardBatched(capturedFakeBatch, true);
            var lossR = Engine.TensorNegate(Engine.ReduceMean(LogSigmoid(predReal), allAxes, keepDims: false));
            var lossF = Engine.TensorNegate(Engine.ReduceMean(LogSigmoid(Engine.TensorNegate(predFake)), allAxes, keepDims: false));
            return Engine.TensorAdd(lossR, lossF);
        }

        var context = new TapeStepContext<T>(
            discParams, grads, lossValue,
            realBatch, realBatch, ComputeForward, RecomputeLoss,
            parameterBuffer: null);
        _discriminatorOptimizer.Step(context);
    }

    /// <summary>
    /// Per-example DP-SGD discriminator step (Abadi et al. 2016 §3, Algorithm 1).
    /// Replays forward+backward once per example, clips each per-example
    /// gradient against the GLOBAL L2 norm across all parameters
    /// concatenated, sums, adds a single Gaussian noise draw, and averages.
    /// </summary>
    private void TrainDiscriminatorStepPerExampleDPSGD(Matrix<T> data, int startRow, int endRow, double noiseMultiplier)
    {
        int batchSize = endRow - startRow;
        var discLayerList = BuildDiscLayerList();
        var discParams = TapeTrainingStep<T>.CollectParameters(discLayerList);

        // Accumulator for sum of clipped per-example gradients
        var gradSum = new Dictionary<Tensor<T>, Tensor<T>>(TensorReferenceComparer<Tensor<T>>.Instance);
        foreach (var p in discParams)
        {
            var zero = new Tensor<T>(p._shape);
            zero.Fill(NumOps.Zero);
            gradSum[p] = zero;
        }

        double clipNorm = _options.ClipNorm;
        double noiseStd = clipNorm * noiseMultiplier;
        T lossSum = NumOps.Zero;

        // Capture the EXACT per-example (real, fake) tensors that produced
        // the per-example losses + clipped gradients, so the replay closure
        // can reconstruct the same objective. Replay built from
        // BuildRealAndFakeBatches(...startRow, endRow) draws fresh noise
        // (= different fake rows), which decouples the replayed scalar loss
        // from the noisedAvgGrads — the optimizer's replay would compute a
        // loss tied to a different objective than the gradients it applies.
        var perExampleReal = new List<Tensor<T>>(endRow - startRow);
        var perExampleFake = new List<Tensor<T>>(endRow - startRow);

        for (int row = startRow; row < endRow; row++)
        {
            var (realBatch, fakeBatch) = BuildRealAndFakeBatches(data, row, row + 1);
            perExampleReal.Add(realBatch);
            perExampleFake.Add(fakeBatch);

            using var tape = new GradientTape<T>();
            var realScores = DiscriminatorForwardBatched(realBatch, isTraining: true);
            var fakeScores = DiscriminatorForwardBatched(fakeBatch, isTraining: true);

            var perExampleAxes = Enumerable.Range(0, realScores.Shape.Length).ToArray();
            var negFakeScores = Engine.TensorNegate(fakeScores);
            var lossReal = Engine.TensorNegate(Engine.ReduceMean(LogSigmoid(realScores), perExampleAxes, keepDims: false));
            var lossFake = Engine.TensorNegate(Engine.ReduceMean(LogSigmoid(negFakeScores), perExampleAxes, keepDims: false));
            var lossTensor = Engine.TensorAdd(lossReal, lossFake);

            if (lossTensor.Length > 0)
                lossSum = NumOps.Add(lossSum, lossTensor[0]);

            var grads = tape.ComputeGradients(lossTensor, discParams);

            // GLOBAL L2 norm across all parameter gradients concatenated.
            // Required by Abadi's L2-sensitivity bound — per-tensor norms
            // do NOT provide the same privacy guarantee.
            double globalSqSum = 0.0;
            foreach (var g in grads.Values)
            {
                for (int i = 0; i < g.Length; i++)
                {
                    double v = NumOps.ToDouble(g[i]);
                    globalSqSum += v * v;
                }
            }
            double globalNorm = Math.Sqrt(globalSqSum + 1e-12);
            double clipFactor = Math.Min(1.0, clipNorm / globalNorm);
            T clipFactorT = NumOps.FromDouble(clipFactor);

            foreach (var kvp in grads)
            {
                var scaled = Engine.TensorMultiplyScalar(kvp.Value, clipFactorT);
                gradSum[kvp.Key] = Engine.TensorAdd(gradSum[kvp.Key], scaled);
            }
        }

        // Add Gaussian noise to the SUM, then average by batchSize.
        var noisedAvgGrads = new Dictionary<Tensor<T>, Tensor<T>>(TensorReferenceComparer<Tensor<T>>.Instance);
        double invBatch = 1.0 / batchSize;
        foreach (var kvp in gradSum)
        {
            var noisy = new Tensor<T>(kvp.Value._shape);
            for (int i = 0; i < kvp.Value.Length; i++)
            {
                double sumVal = NumOps.ToDouble(kvp.Value[i]);
                double u1 = Math.Max(1e-10, _random.NextDouble());
                double u2 = _random.NextDouble();
                double zn = Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2);
                noisy[i] = NumOps.FromDouble((sumVal + zn * noiseStd) * invBatch);
            }
            noisedAvgGrads[kvp.Key] = noisy;
        }

        T avgLoss = NumOps.Divide(lossSum, NumOps.FromDouble(batchSize));

        // Replay-correct closure: each per-example lossTensor was
        // (lossReal + lossFake); the noisedAvgGrads collapse that across
        // the batch but the corresponding scalar loss is still the mean of
        // (lossReal + lossFake). RecomputeLoss must replay BOTH BCE terms,
        // AND must use the EXACT same (real, fake) tensors that the per-
        // example loop used to compute grads — drawing fresh fake noise
        // here would tie the replayed loss to a different objective than
        // the noisedAvgGrads (silent training drift on any optimizer.Step
        // that exercises the replay path).
        var stackedReal = Engine.TensorConcatenate([.. perExampleReal], axis: 0);
        var stackedFake = Engine.TensorConcatenate([.. perExampleFake], axis: 0);
        var allAxesFull = Enumerable.Range(0, stackedReal.Shape.Length).ToArray();
        var capturedFake = stackedFake;
        Tensor<T> ComputeForward(Tensor<T> inp, Tensor<T> _) => DiscriminatorForwardBatched(inp, true);
        Tensor<T> RecomputeLoss(Tensor<T> predReal, Tensor<T> _)
        {
            var predFake = DiscriminatorForwardBatched(capturedFake, true);
            var lossR = Engine.TensorNegate(Engine.ReduceMean(LogSigmoid(predReal), allAxesFull, keepDims: false));
            var lossF = Engine.TensorNegate(Engine.ReduceMean(LogSigmoid(Engine.TensorNegate(predFake)), allAxesFull, keepDims: false));
            return Engine.TensorAdd(lossR, lossF);
        }

        var context = new TapeStepContext<T>(
            discParams, noisedAvgGrads, avgLoss,
            stackedReal, stackedReal, ComputeForward, RecomputeLoss,
            parameterBuffer: null);
        _discriminatorOptimizer.Step(context);
    }

    /// <summary>
    /// Paper-faithful generator/decoder step for MedSynth (no DP — adversarial
    /// gradients flow into the generator which never directly touches real data,
    /// covered by the data-processing inequality). Minimizes
    /// <c>-E[log σ(D(G(z)))]</c> (non-saturating Goodfellow 2014 generator loss)
    /// plus a constraint penalty when the clinical-constraint layer is wired up.
    /// </summary>
    private void TrainGeneratorStepBatched(int batchSize)
    {
        using var tape = new GradientTape<T>();

        // Generator's trainable surface = decoder FCs + per-layer BN +
        // output projection — NOT the full Layers list (Layers also holds
        // the encoder, VAE heads, and discriminator sub-graph; capturing
        // those in genParams would move the critic / encoder during the
        // generator step and corrupt the GAN training equilibrium).
        var generatorLayers = new List<ILayer<T>>(_decoderLayers.Count + _decoderBN.Count + 1);
        generatorLayers.AddRange(_decoderLayers);
        generatorLayers.AddRange(_decoderBN);
        if (_decoderOutput is not null) generatorLayers.Add(_decoderOutput);
        var genParams = TapeTrainingStep<T>.CollectParameters(generatorLayers);

        var noiseBatch = GenerateNoiseBatchTensor(batchSize);
        var fakeBatch = DecoderForwardBatched(noiseBatch, isTraining: true);
        var fakeScores = DiscriminatorForwardBatched(fakeBatch, isTraining: false);

        // Non-saturating generator loss (Goodfellow 2014 §3): maximize log σ(D(G(z)))
        // i.e. minimize -log σ(D(G(z))).
        var allAxes = Enumerable.Range(0, fakeScores.Shape.Length).ToArray();
        var lossTensor = Engine.TensorNegate(Engine.ReduceMean(LogSigmoid(fakeScores), allAxes, keepDims: false));

        var grads = tape.ComputeGradients(lossTensor, genParams);
        T lossValue = lossTensor.Length > 0 ? lossTensor[0] : NumOps.Zero;

        Tensor<T> ComputeForward(Tensor<T> inp, Tensor<T> _) =>
            DiscriminatorForwardBatched(DecoderForwardBatched(inp, true), false);
        Tensor<T> RecomputeLoss(Tensor<T> pred, Tensor<T> _) =>
            Engine.TensorNegate(Engine.ReduceMean(LogSigmoid(pred), allAxes, keepDims: false));

        var context = new TapeStepContext<T>(
            genParams, grads, lossValue,
            noiseBatch, noiseBatch, ComputeForward, RecomputeLoss,
            parameterBuffer: null);
        _generatorOptimizer.Step(context);
    }

    /// <summary>
    /// VAE-half of the MedSynth (VAE+GAN hybrid) training loop. Trains the
    /// encoder, the VAE heads (μ, log σ²), and the decoder against the
    /// composite VAE objective
    /// <code>
    ///   L_VAE = MSE(real, x_recon)                              // reconstruction
    ///         + KLWeight · KL( N(μ, σ²) ‖ N(0, I) )            // KL divergence
    ///         + ConstraintWeight · ConstraintPenalty(x_recon)   // clinical bounds
    /// </code>
    /// per Kingma 2013 §3 + the clinical-constraint extension. Without this
    /// step the documented VAE+GAN hybrid regresses to a plain GAN with
    /// post-hoc clamping in Generate(...) (CodeRabbit blocking concern).
    /// </summary>
    private void TrainVaeStepBatched(Matrix<T> data, int startRow, int endRow)
    {
        int batchSize = endRow - startRow;
        if (batchSize <= 0) return;
        if (_meanHead is null || _logvarHead is null || _decoderOutput is null) return;

        // Build the real batch tensor (rows of the transformed dataset).
        var realBatch = new Tensor<T>([batchSize, _dataWidth]);
        for (int b = 0; b < batchSize; b++)
        {
            int row = startRow + b;
            int cols = Math.Min(_dataWidth, data.Columns);
            for (int j = 0; j < cols; j++) realBatch[b, j] = data[row, j];
        }

        // Trainable surface = encoder + (μ, log σ²) heads + decoder FCs + BN +
        // output. Discriminator stays out of the VAE tape (its forward is not
        // run here at all).
        var vaeLayers = new List<ILayer<T>>(
            _encoderLayers.Count + 2 + _decoderLayers.Count + _decoderBN.Count + 1);
        vaeLayers.AddRange(_encoderLayers);
        vaeLayers.Add(_meanHead);
        vaeLayers.Add(_logvarHead);
        vaeLayers.AddRange(_decoderLayers);
        vaeLayers.AddRange(_decoderBN);
        vaeLayers.Add(_decoderOutput);
        var vaeParams = TapeTrainingStep<T>.CollectParameters(vaeLayers);

        using var tape = new GradientTape<T>();

        // Encoder + heads.
        var hidden = EncoderForwardBatched(realBatch);
        var mean = _meanHead.Forward(hidden);
        var logvar = _logvarHead.Forward(hidden);

        // Reparameterize: z = μ + ε ⊙ exp(½ · log σ²), ε ~ N(0, I) (seeded).
        int latentDim = _options.LatentDimension;
        var epsData = new T[batchSize * latentDim];
        for (int i = 0; i < epsData.Length; i += 2)
        {
            double u1 = 1.0 - _random.NextDouble();
            double u2 = _random.NextDouble();
            double r = Math.Sqrt(-2.0 * Math.Log(u1));
            double theta = 2.0 * Math.PI * u2;
            epsData[i] = NumOps.FromDouble(r * Math.Cos(theta));
            if (i + 1 < epsData.Length)
                epsData[i + 1] = NumOps.FromDouble(r * Math.Sin(theta));
        }
        var epsTensor = new Tensor<T>(epsData, [batchSize, latentDim]);
        var halfLogvar = Engine.TensorMultiplyScalar(logvar, NumOps.FromDouble(0.5));
        var stddev = Engine.TensorExp(halfLogvar);
        var z = Engine.TensorAdd(mean, Engine.TensorMultiply(stddev, epsTensor));

        // Decoder forward — tape-tracked. Matches DecoderForwardBatched.
        var recon = DecoderForwardBatched(z, isTraining: true);

        // Reconstruction loss: per-element MSE → scalar mean.
        var diff = Engine.TensorSubtract(recon, realBatch);
        var sq = Engine.TensorMultiply(diff, diff);
        var reconAxes = Enumerable.Range(0, sq.Shape.Length).ToArray();
        var reconLoss = Engine.ReduceMean(sq, reconAxes, keepDims: false);

        // KL( N(μ, σ²) ‖ N(0, I) ) = 0.5 · Σ(σ² + μ² − 1 − log σ²)  per element,
        // then mean over the batch. σ² = exp(log σ²).
        var variance = Engine.TensorExp(logvar);
        var meanSq = Engine.TensorMultiply(mean, mean);
        var ones = new Tensor<T>(logvar._shape);
        Engine.TensorFill(ones, NumOps.One);
        var klPerElement = Engine.TensorAdd(
            Engine.TensorSubtract(Engine.TensorAdd(variance, meanSq), ones),
            Engine.TensorNegate(logvar));
        var klAxes = Enumerable.Range(0, klPerElement.Shape.Length).ToArray();
        var klLoss = Engine.TensorMultiplyScalar(
            Engine.ReduceMean(klPerElement, klAxes, keepDims: false),
            NumOps.FromDouble(0.5));

        // Constraint penalty: per-column squared violation of the
        // [colMin, colMax] bounds learned in LearnConstraints, mean over
        // (batch, columns). Uses tape-tracked ReLU(x − colMax) for the
        // upper-bound violation and ReLU(colMin − x) for the lower-bound,
        // squared and summed. Skipped when the constraint table is null
        // (no Fit-time learning happened).
        Tensor<T> totalLoss = Engine.TensorAdd(
            reconLoss,
            Engine.TensorMultiplyScalar(klLoss, NumOps.FromDouble(_options.KLWeight)));
        if (_colMin is not null && _colMax is not null && _options.ConstraintWeight > 0.0)
        {
            int cols = Math.Min(_dataWidth, _colMin.Length);
            // Tile the per-column bounds across the batch to a full [batch, cols]
            // tensor — TensorSubtract requires matching shapes (it does not
            // broadcast a [1, cols] row against [batch, cols]).
            var lowerBoundArr = new T[batchSize * cols];
            var upperBoundArr = new T[batchSize * cols];
            for (int b = 0; b < batchSize; b++)
            {
                for (int j = 0; j < cols; j++)
                {
                    lowerBoundArr[b * cols + j] = NumOps.FromDouble(_colMin[j]);
                    upperBoundArr[b * cols + j] = NumOps.FromDouble(_colMax[j]);
                }
            }
            var lower = new Tensor<T>(lowerBoundArr, [batchSize, cols]);
            var upper = new Tensor<T>(upperBoundArr, [batchSize, cols]);
            var upperViol = Engine.ReLU(Engine.TensorSubtract(recon, upper));
            var lowerViol = Engine.ReLU(Engine.TensorSubtract(lower, recon));
            var viol = Engine.TensorAdd(
                Engine.TensorMultiply(upperViol, upperViol),
                Engine.TensorMultiply(lowerViol, lowerViol));
            var violAxes = Enumerable.Range(0, viol.Shape.Length).ToArray();
            var constraintLoss = Engine.ReduceMean(viol, violAxes, keepDims: false);
            totalLoss = Engine.TensorAdd(
                totalLoss,
                Engine.TensorMultiplyScalar(constraintLoss, NumOps.FromDouble(_options.ConstraintWeight)));
        }

        var grads = tape.ComputeGradients(totalLoss, vaeParams);
        T lossValue = totalLoss.Length > 0 ? totalLoss[0] : NumOps.Zero;

        // Replay closure recomputes the full VAE composite loss against the
        // exact (realBatch, eps, lower/upper bounds) captured during the
        // forward pass. Re-sampling ε here on replay would change the
        // reparameterized z and decouple the replayed loss from the grads;
        // re-using the captured ε keeps the optimizer.Step replay tied to
        // the same objective that produced grads.
        var capturedReal = realBatch;
        var capturedEps = epsTensor;
        var capturedReconAxes = reconAxes;
        var capturedKlAxes = klAxes;
        double capturedKLWeight = _options.KLWeight;
        double capturedConstraintWeight = _options.ConstraintWeight;
        T[]? capturedLowerArr = null, capturedUpperArr = null;
        int capturedCols = 0;
        if (_colMin is not null && _colMax is not null && capturedConstraintWeight > 0.0)
        {
            capturedCols = Math.Min(_dataWidth, _colMin.Length);
            capturedLowerArr = new T[capturedCols];
            capturedUpperArr = new T[capturedCols];
            for (int j = 0; j < capturedCols; j++)
            {
                capturedLowerArr[j] = NumOps.FromDouble(_colMin[j]);
                capturedUpperArr[j] = NumOps.FromDouble(_colMax[j]);
            }
        }

        Tensor<T> ComputeForward(Tensor<T> inp, Tensor<T> _)
        {
            var h = EncoderForwardBatched(inp);
            var m = _meanHead.Forward(h);
            var lv = _logvarHead.Forward(h);
            var hl = Engine.TensorMultiplyScalar(lv, NumOps.FromDouble(0.5));
            var sd = Engine.TensorExp(hl);
            var zz = Engine.TensorAdd(m, Engine.TensorMultiply(sd, capturedEps));
            return DecoderForwardBatched(zz, isTraining: true);
        }
        Tensor<T> RecomputeLoss(Tensor<T> reconReplay, Tensor<T> _)
        {
            var d = Engine.TensorSubtract(reconReplay, capturedReal);
            var s = Engine.TensorMultiply(d, d);
            var rL = Engine.ReduceMean(s, capturedReconAxes, keepDims: false);
            // KL term needs μ/log σ² which aren't in this closure's signature.
            // Recompute via a second encoder pass on capturedReal — identical
            // input → identical heads → identical KL contribution.
            var h2 = EncoderForwardBatched(capturedReal);
            var m2 = _meanHead.Forward(h2);
            var lv2 = _logvarHead.Forward(h2);
            var v2 = Engine.TensorExp(lv2);
            var ms2 = Engine.TensorMultiply(m2, m2);
            var o2 = new Tensor<T>(lv2._shape);
            Engine.TensorFill(o2, NumOps.One);
            var klE = Engine.TensorAdd(
                Engine.TensorSubtract(Engine.TensorAdd(v2, ms2), o2),
                Engine.TensorNegate(lv2));
            var klL = Engine.TensorMultiplyScalar(
                Engine.ReduceMean(klE, capturedKlAxes, keepDims: false),
                NumOps.FromDouble(0.5));
            var total = Engine.TensorAdd(rL,
                Engine.TensorMultiplyScalar(klL, NumOps.FromDouble(capturedKLWeight)));
            if (capturedLowerArr is not null && capturedUpperArr is not null)
            {
                var lwr = new Tensor<T>(capturedLowerArr, [1, capturedCols]);
                var upr = new Tensor<T>(capturedUpperArr, [1, capturedCols]);
                var uv = Engine.ReLU(Engine.TensorSubtract(reconReplay, upr));
                var lvw = Engine.ReLU(Engine.TensorSubtract(lwr, reconReplay));
                var vi = Engine.TensorAdd(
                    Engine.TensorMultiply(uv, uv),
                    Engine.TensorMultiply(lvw, lvw));
                var viAxes = Enumerable.Range(0, vi.Shape.Length).ToArray();
                var cL = Engine.ReduceMean(vi, viAxes, keepDims: false);
                total = Engine.TensorAdd(total,
                    Engine.TensorMultiplyScalar(cL, NumOps.FromDouble(capturedConstraintWeight)));
            }
            return total;
        }

        var context = new TapeStepContext<T>(
            vaeParams, grads, lossValue,
            realBatch, realBatch, ComputeForward, RecomputeLoss,
            parameterBuffer: null);
        _vaeOptimizer.Step(context);
    }

    /// <summary>
    /// Numerically-stable <c>log(sigmoid(x))</c> = <c>-softplus(-x)</c>, expressed
    /// via tape-tracked Engine ops so backprop flows correctly through the
    /// BCE-with-logits loss.
    /// </summary>
    private Tensor<T> LogSigmoid(Tensor<T> x)
    {
        // Numerically stable log σ(x) = -softplus(-x). The naive
        // log(σ(x)) underflows to -∞ for sufficiently negative scores
        // (sigmoid → 0). Engine.Softplus is internally computed with the
        // max(z,0) + log(1+exp(-|z|)) identity so the dynamic range
        // stays intact for confident discriminator outputs.
        return Engine.TensorNegate(Engine.Softplus(Engine.TensorNegate(x)));
    }

    /// <summary>
    /// Builds [batchSize, dataWidth] real + fake batches for one DP-SGD critic step.
    /// </summary>
    private (Tensor<T> realBatch, Tensor<T> fakeBatch) BuildRealAndFakeBatches(
        Matrix<T> data, int startRow, int endRow)
    {
        int batchSize = endRow - startRow;
        var realBatch = new Tensor<T>([batchSize, _dataWidth]);
        for (int b = 0; b < batchSize; b++)
        {
            int row = startRow + b;
            int cols = Math.Min(_dataWidth, data.Columns);
            for (int j = 0; j < cols; j++) realBatch[b, j] = data[row, j];
        }
        var noiseBatch = GenerateNoiseBatchTensor(batchSize);
        var fakeBatch = DecoderForwardBatched(noiseBatch, isTraining: false);
        return (realBatch, fakeBatch);
    }

    private Tensor<T> GenerateNoiseBatchTensor(int batchSize)
    {
        int latentDim = _options.LatentDimension;
        int totalElements = batchSize * latentDim;
        // Box–Muller via the seeded _random so MedSynthOptions.Seed makes
        // Fit reproducible — Engine.TensorRandomUniformRange bypasses _random
        // and breaks the seed contract that Generate(...) still honours.
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
        return new Tensor<T>(noiseData, [batchSize, latentDim]);
    }

    /// <summary>
    /// Batched, tape-tracked decoder forward (replaces per-row
    /// <see cref="DecoderForward"/>). Uses <see cref="Engine.ReLU"/> instead of
    /// the manual <c>ApplyReLU</c> indexed loop so the tape sees the activation.
    /// </summary>
    private Tensor<T> DecoderForwardBatched(Tensor<T> z, bool isTraining)
    {
        var current = z;
        for (int i = 0; i < _decoderLayers.Count; i++)
        {
            current = _decoderLayers[i].Forward(current);
            if (i < _decoderBN.Count)
            {
                _decoderBN[i].SetTrainingMode(isTraining);
                current = _decoderBN[i].Forward(current);
            }
            current = Engine.ReLU(current);
        }
        if (_decoderOutput is not null)
            current = _decoderOutput.Forward(current);
        return current;
    }

    /// <summary>
    /// Batched, tape-tracked encoder forward. Mirrors per-row
    /// <see cref="EncoderForward(Vector{T})"/> but on a <c>[batch, dataWidth]</c>
    /// tensor and using tape-tracked <see cref="Engine.ReLU"/> so the VAE
    /// step's backward flows through the encoder weights.
    /// </summary>
    private Tensor<T> EncoderForwardBatched(Tensor<T> input)
    {
        var current = input;
        for (int i = 0; i < _encoderLayers.Count; i++)
        {
            current = _encoderLayers[i].Forward(current);
            current = Engine.ReLU(current);
        }
        return current;
    }

    /// <summary>
    /// Batched, tape-tracked discriminator forward. Same paper-faithful
    /// LeakyReLU(0.2) + Dropout architecture as
    /// <see cref="DiscriminatorForward"/>, but using <see cref="Engine.LeakyReLU"/>.
    /// </summary>
    private Tensor<T> DiscriminatorForwardBatched(Tensor<T> input, bool isTraining)
    {
        var current = input;
        T leakySlope = NumOps.FromDouble(0.2);
        for (int i = 0; i < _discLayers.Count; i++)
        {
            current = _discLayers[i].Forward(current);
            current = Engine.LeakyReLU(current, leakySlope);
            if (i < _discDropout.Count)
            {
                _discDropout[i].SetTrainingMode(isTraining);
                current = _discDropout[i].Forward(current);
            }
        }
        if (_discOutput is not null)
            current = _discOutput.Forward(current);
        return current;
    }

    private double ComputeNoiseMultiplier(int dataSize, int epochs)
    {
        double delta = 1.0 / (dataSize * dataSize);
        int batchSize = Math.Min(_options.BatchSize, dataSize);
        double samplingRate = (double)batchSize / dataSize;
        int totalSteps = epochs * (dataSize / Math.Max(batchSize, 1));

        double noiseMultiplier = 1.0;
        for (int attempt = 0; attempt < 100; attempt++)
        {
            double eps = samplingRate * Math.Sqrt(totalSteps * 2.0 * Math.Log(1.0 / delta)) * noiseMultiplier;
            if (eps <= _options.Epsilon) break;
            noiseMultiplier *= 1.1;
        }

        return noiseMultiplier;
    }

    #endregion

    #region Forward Passes with Manual Activation

    private Vector<T> EncoderForward(Vector<T> x)
    {
        _encoderPreActs.Clear();
        var current = VectorToTensor(x);

        for (int i = 0; i < _encoderLayers.Count; i++)
        {
            current = _encoderLayers[i].Forward(current);
            _encoderPreActs.Add(CloneTensor(current));
            current = ApplyReLU(current);
        }

        int hiddenDim = _options.EncoderDimensions.Length > 0
            ? _options.EncoderDimensions[^1] : _dataWidth;
        return TensorToVector(current, hiddenDim);
    }

    private Vector<T> DecoderForward(Vector<T> z, bool isTraining)
    {
        _decoderPreActs.Clear();
        var current = VectorToTensor(z);

        for (int i = 0; i < _decoderLayers.Count; i++)
        {
            current = _decoderLayers[i].Forward(current);
            if (i < _decoderBN.Count)
            {
                _decoderBN[i].SetTrainingMode(isTraining);
                current = _decoderBN[i].Forward(current);
            }
            _decoderPreActs.Add(CloneTensor(current));
            current = ApplyReLU(current);
        }

        if (_decoderOutput is not null)
        {
            current = _decoderOutput.Forward(current);
        }

        return TensorToVector(current, _dataWidth);
    }

    private Vector<T> DiscriminatorForward(Vector<T> x, bool isTraining)
    {
        _discPreActs.Clear();
        var current = VectorToTensor(x);

        for (int i = 0; i < _discLayers.Count; i++)
        {
            current = _discLayers[i].Forward(current);
            _discPreActs.Add(CloneTensor(current));
            current = ApplyLeakyReLU(current, 0.2);

            if (i < _discDropout.Count)
            {
                _discDropout[i].SetTrainingMode(isTraining);
                current = _discDropout[i].Forward(current);
            }
        }

        if (_discOutput is not null)
        {
            current = _discOutput.Forward(current);
        }

        return TensorToVector(current, current.Length);
    }

    #endregion

    #region Backward Passes with Manual Activation Derivatives

    private void UpdateDiscriminator(T lr)
    {
        foreach (var layer in _discLayers) layer.UpdateParameters(lr);
        _discOutput?.UpdateParameters(lr);
    }

    #endregion

    #region Discriminator Layer List

    /// <summary>
    /// Builds a combined list of discriminator layers (dense + dropout + output)
    /// for gradient-penalty and related analyses.
    /// </summary>
    private IReadOnlyList<ILayer<T>> BuildDiscLayerList()
    {
        var allLayers = new List<ILayer<T>>();
        for (int i = 0; i < _discDropout.Count; i++)
        {
            allLayers.Add(_discLayers[i]);
            allLayers.Add(_discDropout[i]);
        }
        if (_discOutput is not null)
        {
            allLayers.Add(_discOutput);
        }
        return allLayers;
    }

    #endregion

    #region VAE Helpers

    private Vector<T> Reparameterize(Vector<T> mean, Vector<T> logvar)
    {
        int dim = mean.Length;
        var z = new Vector<T>(dim);
        for (int i = 0; i < dim; i++)
        {
            double u1 = Math.Max(1e-10, _random.NextDouble());
            double u2 = _random.NextDouble();
            double eps = Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2);
            double m = NumOps.ToDouble(mean[i]);
            double lv = NumOps.ToDouble(logvar[i]);
            z[i] = NumOps.FromDouble(m + eps * Math.Exp(0.5 * lv));
        }
        return z;
    }

    #endregion

    #region Manual Activation Functions

    private static Tensor<T> ApplyReLU(Tensor<T> input)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        var result = new Tensor<T>(input._shape);
        for (int i = 0; i < input.Length; i++)
        {
            double v = ops.ToDouble(input[i]);
            result[i] = ops.FromDouble(v > 0 ? v : 0);
        }
        return result;
    }

    private static Tensor<T> ApplyReLUDerivative(Tensor<T> grad, Tensor<T> preActivation)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        var result = new Tensor<T>(grad._shape);
        int len = Math.Min(grad.Length, preActivation.Length);
        for (int i = 0; i < len; i++)
        {
            double pre = ops.ToDouble(preActivation[i]);
            result[i] = pre > 0 ? grad[i] : ops.Zero;
        }
        return result;
    }

    private static Tensor<T> ApplyLeakyReLU(Tensor<T> input, double alpha)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        var result = new Tensor<T>(input._shape);
        for (int i = 0; i < input.Length; i++)
        {
            double v = ops.ToDouble(input[i]);
            result[i] = ops.FromDouble(v >= 0 ? v : alpha * v);
        }
        return result;
    }

    private static Tensor<T> ApplyLeakyReLUDerivative(Tensor<T> grad, Tensor<T> preActivation, double alpha)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        var result = new Tensor<T>(grad._shape);
        int len = Math.Min(grad.Length, preActivation.Length);
        for (int i = 0; i < len; i++)
        {
            double pre = ops.ToDouble(preActivation[i]);
            double deriv = pre >= 0 ? 1.0 : alpha;
            result[i] = ops.FromDouble(ops.ToDouble(grad[i]) * deriv);
        }
        return result;
    }

    #endregion

    #region Constraints & Activations

    private void LearnConstraints(Matrix<T> data)
    {
        _colMin = new double[data.Columns];
        _colMax = new double[data.Columns];

        for (int j = 0; j < data.Columns; j++)
        {
            double min = double.MaxValue;
            double max = double.MinValue;
            for (int i = 0; i < data.Rows; i++)
            {
                double v = NumOps.ToDouble(data[i, j]);
                if (v < min) min = v;
                if (v > max) max = v;
            }
            double range = max - min;
            _colMin[j] = min - 0.1 * range;
            _colMax[j] = max + 0.1 * range;
        }
    }

    private Vector<T> ApplyConstraints(Vector<T> decoded)
    {
        if (_colMin is null || _colMax is null) return decoded;

        var constrained = new Vector<T>(decoded.Length);
        for (int j = 0; j < decoded.Length; j++)
        {
            double val = NumOps.ToDouble(decoded[j]);
            constrained[j] = NumOps.FromDouble(Math.Min(Math.Max(val, _colMin[j]), _colMax[j]));
        }
        return constrained;
    }

    private Vector<T> ApplyOutputActivations(Vector<T> constrained)
    {
        if (_transformer is null) return constrained;

        var output = VectorToTensor(constrained);
        var result = new Tensor<T>(output._shape);
        int idx = 0;

        for (int col = 0; col < _columns.Count && idx < output.Length; col++)
        {
            var transform = _transformer.GetTransformInfo(col);
            if (transform.IsContinuous)
            {
                if (idx < output.Length)
                {
                    result[idx] = NumOps.FromDouble(Math.Tanh(NumOps.ToDouble(output[idx])));
                    idx++;
                }
                int numModes = transform.Width - 1;
                if (numModes > 0) ApplySoftmax(output, result, ref idx, numModes);
            }
            else
            {
                ApplySoftmax(output, result, ref idx, transform.Width);
            }
        }

        return TensorToVector(result, _dataWidth);
    }

    private static void ApplySoftmax(Tensor<T> input, Tensor<T> output, ref int idx, int count)
    {
        if (count <= 0) return;
        int actualCount = Math.Min(count, input.Length - idx);
        if (actualCount <= 0) return;
        var slice = new Tensor<T>([actualCount]);
        input.Data.Span.Slice(idx, actualCount).CopyTo(slice.Data.Span);
        var result = AiDotNetEngine.Current.Softmax(slice, -1);
        result.Data.Span.CopyTo(output.Data.Span.Slice(idx, actualCount));
        idx += actualCount;
    }

    #endregion

    #region Helpers

    private static double Sigmoid(double x)
    {
        return 1.0 / (1.0 + Math.Exp(-Math.Min(Math.Max(x, -20.0), 20.0)));
    }

    private static Tensor<T> CloneTensor(Tensor<T> source)
    {
        var clone = new Tensor<T>(source._shape);
        for (int i = 0; i < source.Length; i++)
        {
            clone[i] = source[i];
        }
        return clone;
    }

    private static Vector<T> GetRow(Matrix<T> matrix, int row)
    {
        var v = new Vector<T>(matrix.Columns);
        for (int j = 0; j < matrix.Columns; j++) v[j] = matrix[row, j];
        return v;
    }

    /// <summary>
    /// Creates a vector of standard normal random values using Box-Muller transform.
    /// </summary>
    private Vector<T> CreateStandardNormalVector(int size)
    {
        var v = new Vector<T>(size);
        for (int i = 0; i < size; i++)
        {
            double u1 = Math.Max(1e-10, _random.NextDouble());
            double u2 = _random.NextDouble();
            double normal = Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2);
            v[i] = NumOps.FromDouble(normal);
        }
        return v;
    }

    /// <summary>
    /// Sanitizes a gradient tensor by replacing NaN/Inf values with zero and applying gradient clipping.
    /// </summary>
    private static Tensor<T> SanitizeAndClipGradient(Tensor<T> grad, double maxNorm)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        double normSq = 0;
        for (int i = 0; i < grad.Length; i++)
        {
            double v = ops.ToDouble(grad[i]);
            if (double.IsNaN(v) || double.IsInfinity(v))
            {
                grad[i] = ops.Zero;
            }
            else
            {
                normSq += v * v;
            }
        }

        double norm = Math.Sqrt(normSq);
        if (norm > maxNorm)
        {
            double scale = maxNorm / norm;
            for (int i = 0; i < grad.Length; i++)
            {
                grad[i] = ops.FromDouble(ops.ToDouble(grad[i]) * scale);
            }
        }

        return grad;
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
