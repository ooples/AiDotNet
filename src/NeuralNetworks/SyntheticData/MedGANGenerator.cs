using AiDotNet.ActivationFunctions;
using AiDotNet.Attributes;
using AiDotNet.Autodiff;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Optimizers;
using AiDotNet.SelfSupervisedLearning.Losses;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Training;

namespace AiDotNet.NeuralNetworks.SyntheticData;

/// <summary>
/// medGAN — generates synthetic patient records by pairing a pre-trained autoencoder with a
/// generative adversarial network whose generator produces points in the autoencoder's latent space.
/// </summary>
/// <remarks>
/// <para>
/// Choi et al., "Generating Multi-label Discrete Patient Records using Generative Adversarial
/// Networks" (arXiv:1703.06490). GANs are built for continuous data, but patient records are
/// high-dimensional and discrete — binary diagnosis codes or integer counts. medGAN's answer is to
/// never make the generator emit a record at all. Instead an autoencoder is pre-trained on the real
/// records, and the generator learns only to produce a plausible point in that autoencoder's
/// continuous latent space; the pre-trained decoder turns that point into a record:
/// </para>
/// <code>
///   z ~ N(0, 1) --> G --> Dec --> x~            (synthesis)
///                                  \
///                                   D --> real / fake
///                                  /
///   x (real record) --------------
/// </code>
/// <para>
/// The composition is <c>x~ = Dec(G(z))</c>: "the pre-trained decoder Dec can pick up the right
/// signals from G(z) to convert it to the patient record Dec(G(z))". The decoder is not frozen after
/// pre-training — it is updated jointly with the generator by the adversarial objective, so the
/// parameter set of the generator step is <c>theta_(g,dec)</c>.
/// </para>
/// <para>
/// <b>The three efficiency devices.</b> The paper names three, and all three are implemented here
/// rather than described:
/// </para>
/// <list type="number">
/// <item><description><b>Minibatch averaging</b> — the discriminator sees each sample concatenated
/// with the average of its minibatch, so it can judge a sample against the batch's overall
/// composition. This is medGAN's remedy for mode collapse, and it works because for binary
/// variables the average IS the maximum-likelihood estimate of each code's Bernoulli success
/// probability, and for counts it estimates the binomial mean. A generator that collapses onto one
/// record produces a batch average that gives it away immediately.</description></item>
/// <item><description><b>Batch normalization</b> in the generator, with moving-average decay
/// 0.99.</description></item>
/// <item><description><b>Shortcut connections</b> in the generator:
/// <c>x_k = ReLU(BN_k(W_k x_(k-1))) + x_(k-1)</c>. The addition is what forces every generator width
/// to equal the embedding dimension.</description></item>
/// </list>
/// <para>
/// The discriminator deliberately has NEITHER batch normalization NOR shortcut connections.
/// </para>
/// <para>
/// <b>The two-stage objective.</b> Pre-training minimizes reconstruction error over the real records
/// — cross entropy for binary variables, squared error for counts. The GAN then optimizes
/// </para>
/// <code>
///   theta_d       &lt;- ascend  1/m sum_i [ log D(x_i, xbar) + log(1 - D(x_z_i, xbar_z)) ]
///   theta_(g,dec) &lt;- ascend  1/m sum_i   log D(x_z_i, xbar_z)
/// </code>
/// <para>
/// with k = 2 discriminator updates per generator update. The second line is the non-saturating
/// generator objective, and the presence of <c>theta_dec</c> in it is the detail that makes the
/// decoder adapt to the generator rather than the other way round.
/// </para>
/// <para>
/// <b>For Beginners:</b> Real patient records cannot be shared, but fake ones with the same
/// statistical shape can. Learning to invent them directly is hard because a record is a list of
/// yes/no answers, and the usual machinery needs smooth numbers. So this does it in two stages.
/// First it learns to squash a real record down to a short list of numbers and rebuild it — that is
/// the autoencoder. Then it learns to invent new short lists that rebuild into records a critic
/// cannot distinguish from real ones. The critic gets one extra hint: alongside each record it sees
/// the average of the whole batch, which makes it obvious if the inventor keeps producing the same
/// record over and over.
/// </para>
/// <para>
/// <b>Beyond the paper</b>, and off by default so that the default configuration is exactly
/// medGAN's: <see cref="MedGANOptions{T}.EnablePrivacy"/> trains the discriminator under DP-SGD
/// (Abadi et al. 2016) for a formal privacy bound where the paper offers only an empirical
/// observation, and <see cref="MedGANOptions{T}.ConstraintWeight"/> penalizes generated values
/// outside the range observed in training.
/// </para>
/// <example>
/// <code>
/// var data = new Matrix&lt;double&gt;(new double[,] { { 1.0, 2.0 }, { 3.0, 4.0 }, { 5.0, 6.0 }, { 7.0, 8.0 } });
/// var architecture = new NeuralNetworkArchitecture&lt;double&gt;(
///     inputType: InputType.OneDimensional, taskType: NeuralNetworkTaskType.Regression,
///     inputSize: 64, outputSize: 64);
/// var medgan = new MedGANGenerator&lt;double&gt;(architecture);
/// var columns = new[]
/// {
///     new ColumnMetadata("age", ColumnDataType.Continuous, columnIndex: 0),
///     new ColumnMetadata("sex", ColumnDataType.Discrete, new[] { "f", "m" }, columnIndex: 1)
/// };
/// medgan.Fit(data, columns, epochs: 1000);
/// var synthetic = medgan.Generate(1000);
/// </code>
/// </example>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
[ModelDomain(ModelDomain.MachineLearning)]
[ModelCategory(ModelCategory.NeuralNetwork)]
[ModelCategory(ModelCategory.SyntheticDataGenerator)]
[ModelTask(ModelTask.Generation)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("Generating Multi-label Discrete Patient Records using Generative Adversarial Networks",
    "https://arxiv.org/abs/1703.06490",
    Year = 2017,
    Authors = "Edward Choi, Siddharth Biswal, Bradley Malin, Jon Duke, Walter F. Stewart, Jimeng Sun")]
public partial class MedGANGenerator<T> : NeuralSyntheticTabularGeneratorBase<T>, ISyntheticTabularGenerator<T>
{
    private readonly MedGANOptions<T> _options;

    // One dedicated optimizer per parameter set. See CTGANGenerator: a single shared AdamOptimizer
    // corrupts its flat moment buffer across networks of different parameter counts. medGAN trains
    // three distinct sets — the autoencoder during pre-training, then theta_d and theta_(g,dec).
    private IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> _generatorOptimizer;
    private IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> _discriminatorOptimizer;
    private IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> _autoencoderOptimizer;
    private ILossFunction<T> _lossFunction;
    private Random _random;

    // ISyntheticTabularGenerator state
    private List<ColumnMetadata> _columns = new();
    private TabularDataTransformer<T>? _transformer;

    // --- Autoencoder ---
    // Encoder: hidden widths then the embedding projection. Decoder: the mirror, ending at the
    // record width. Pre-trained together; afterwards only the decoder keeps training, as part of
    // theta_(g,dec).
    private readonly List<FullyConnectedLayer<T>> _encoderLayers = new();
    private readonly List<FullyConnectedLayer<T>> _decoderLayers = new();

    // --- Generator ---
    // Paired 1:1 — every generator FC is followed by a BatchNorm and carries a shortcut. These are
    // views into Layers, NOT the full list: walking Layers in the generator forward path would run
    // latent noise through the autoencoder and discriminator weights.
    private readonly List<FullyConnectedLayer<T>> _generatorLayers = new();
    private readonly List<BatchNormalizationLayer<T>> _generatorBN = new();

    // --- Discriminator ---
    // No BatchNorm and no shortcuts, per the paper.
    private readonly List<FullyConnectedLayer<T>> _discLayers = new();
    private FullyConnectedLayer<T>? _discOutput;

    // Per-column output group layout of the transformed representation, used by the MixedTabular
    // decoder activation and reconstruction loss. Null until Fit supplies a transformer.
    private List<(int Start, int Width, bool Softmax)>? _outputGroups;

    // Observed per-column range, for the optional out-of-range penalty.
    private double[]? _colMin;
    private double[]? _colMax;

    private bool _usingCustomLayers;
    private int _dataWidth;

    /// <summary>
    /// Gets the medGAN-specific options.
    /// </summary>
    public new MedGANOptions<T> Options => _options;

    /// <inheritdoc />
    public IReadOnlyList<ColumnMetadata> Columns => _columns.AsReadOnly();

    /// <inheritdoc />
    public bool IsFitted { get; private set; }

    /// <summary>
    /// Gets the dimension r of the random prior z, which equals the embedding dimension unless
    /// <see cref="MedGANOptions{T}.NoiseDimension"/> overrides it.
    /// </summary>
    public int NoiseDimension => _options.NoiseDimension ?? _options.EmbeddingDimension;

    /// <summary>
    /// Gets the width the discriminator actually consumes: twice the record width when minibatch
    /// averaging is on, because each sample arrives concatenated with the batch average.
    /// </summary>
    public int DiscriminatorInputWidth =>
        _options.UseMinibatchAveraging ? _dataWidth * 2 : _dataWidth;

    /// <summary>
    /// Initializes a new instance with default architecture settings.
    /// </summary>
    public MedGANGenerator()
        : this(new NeuralNetworkArchitecture<T>(
            inputType: Enums.InputType.OneDimensional,
            taskType: Enums.NeuralNetworkTaskType.Regression,
            inputSize: 10,
            outputSize: 10))
    {
    }

    /// <summary>
    /// Initializes a new medGAN generator with the specified architecture.
    /// </summary>
    /// <param name="architecture">The architecture defining input/output dimensions and optional custom layers.</param>
    /// <param name="options">medGAN options; defaults reproduce the paper's hyperparameters.</param>
    /// <param name="optimizer">Gradient-based optimizer for the generator (defaults to Adam at the paper's 1e-3).</param>
    /// <param name="lossFunction">Loss function used by the base class's generic training path.</param>
    /// <param name="maxGradNorm">Maximum gradient norm for clipping (default 5.0).</param>
    /// <exception cref="ArgumentException">
    /// Thrown when a generator width differs from the embedding dimension. The generator's shortcut
    /// connection is an addition, so mismatched widths cannot be added; rejecting this at
    /// construction is deliberate, since silently dropping the shortcut would remove one of the
    /// paper's three named contributions without saying so.
    /// </exception>
    public MedGANGenerator(
        NeuralNetworkArchitecture<T> architecture,
        MedGANOptions<T>? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
        ILossFunction<T>? lossFunction = null,
        double maxGradNorm = 5.0)
        : base(architecture, lossFunction ?? NeuralNetworkHelper<T>.GetDefaultLossFunction(architecture.TaskType), maxGradNorm)
    {
        _options = options ?? new MedGANOptions<T>();

        Guard.Positive(_options.EmbeddingDimension, nameof(_options.EmbeddingDimension));
        foreach (int width in _options.GeneratorDimensions)
        {
            if (width != _options.EmbeddingDimension)
            {
                throw new ArgumentException(
                    $"Every generator width must equal EmbeddingDimension ({_options.EmbeddingDimension}) " +
                    $"because medGAN's shortcut connection adds a layer's output to its input; got {width}. " +
                    "Change EmbeddingDimension, or set GeneratorDimensions to widths that match it.",
                    nameof(options));
            }
        }
        if (NoiseDimension != _options.EmbeddingDimension)
        {
            throw new ArgumentException(
                $"NoiseDimension ({NoiseDimension}) must equal EmbeddingDimension " +
                $"({_options.EmbeddingDimension}): the generator's first layer also carries a shortcut, " +
                "so the prior and the embedding space must have the same width.",
                nameof(options));
        }

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
        _autoencoderOptimizer = MakeAdam();
        _random = _options.Seed.HasValue
            ? RandomHelper.CreateSeededRandom(_options.Seed.Value)
            : RandomHelper.CreateSecureRandom();

        InitializeLayers();
    }

    #region Layer Initialization

    /// <inheritdoc />
    /// <remarks>
    /// Builds encoder, decoder, generator and discriminator into the single <c>Layers</c> list in
    /// the fixed order <see cref="ExtractMedGANLayerReferences"/> walks. Custom architecture layers,
    /// when supplied, replace the whole stack.
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
            SeedLayerInitialization();
            Layers.AddRange(LayerHelper<T>.CreateDefaultMedGANLayers(
                _dataWidth, _options.EmbeddingDimension,
                _options.AutoencoderDimensions, _options.GeneratorDimensions,
                _options.DiscriminatorDimensions, _options.UseMinibatchAveraging));
            _usingCustomLayers = false;
        }

        ExtractMedGANLayerReferences();
    }

    /// <summary>
    /// Before Fit() supplies the real transformed width, adapt the layout to the actual input width
    /// so the model is a valid network for any 1-D input — the generated ModelFamily tests call
    /// Train()/Predict() directly without Fit(). Once fitted, the width is fixed by the transformer.
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
    /// Binds the typed encoder/decoder/generator/discriminator views into the unified Layers list.
    /// </summary>
    /// <remarks>
    /// Must be re-run after deserialization: the base deserializer replaces every entry in
    /// <c>Layers</c> with a fresh instance, which would leave these views pointing at the discarded
    /// constructor-initialized weights.
    /// </remarks>
    private void ExtractMedGANLayerReferences()
    {
        _encoderLayers.Clear();
        _decoderLayers.Clear();
        _generatorLayers.Clear();
        _generatorBN.Clear();
        _discLayers.Clear();
        _discOutput = null;

        int idx = 0;
        int aeHidden = _options.AutoencoderDimensions.Length;

        // Encoder: hidden widths, then the embedding projection.
        for (int i = 0; i <= aeHidden && idx < Layers.Count; i++)
        {
            if (Layers[idx] is FullyConnectedLayer<T> enc) _encoderLayers.Add(enc);
            idx++;
        }

        // Decoder: the mirror, ending at the record width.
        for (int i = 0; i <= aeHidden && idx < Layers.Count; i++)
        {
            if (Layers[idx] is FullyConnectedLayer<T> dec) _decoderLayers.Add(dec);
            idx++;
        }

        // Generator: hidden shortcut layers plus the final embedding projection, each FC paired
        // with a BatchNorm.
        int generatorFcCount = _options.GeneratorDimensions.Length + 1;
        for (int i = 0; i < generatorFcCount && idx < Layers.Count; i++)
        {
            if (Layers[idx] is FullyConnectedLayer<T> gen) _generatorLayers.Add(gen);
            idx++;
            if (idx < Layers.Count && Layers[idx] is BatchNormalizationLayer<T> bn)
            {
                _generatorBN.Add(bn);
                idx++;
            }
        }

        // Discriminator hidden layers, then its scalar output.
        for (int i = 0; i < _options.DiscriminatorDimensions.Length && idx < Layers.Count; i++)
        {
            if (Layers[idx] is FullyConnectedLayer<T> disc) _discLayers.Add(disc);
            idx++;
        }
        if (idx < Layers.Count && Layers[idx] is FullyConnectedLayer<T> discOut)
        {
            _discOutput = discOut;
            idx++;
        }
    }

    /// <summary>
    /// Makes weight initialization obey <see cref="MedGANOptions{T}.Seed"/>.
    /// </summary>
    /// <remarks>
    /// Without this, each layer's lazy weight init falls back to the process-shared, order-dependent
    /// <c>RandomHelper.ThreadSafeRandom</c>, so two runs with the same seed start from different
    /// weights and Fit is not reproducible — the seed would only be governing the noise draws, which
    /// is a seed contract that silently covers half the model. With <c>Seed</c> null the scope is
    /// inert and the non-reproducible production default is preserved.
    /// </remarks>
    private void SeedLayerInitialization() =>
        LayerInitializationSeedScope.ResetForModelConstruction(_options.Seed);

    /// <summary>
    /// Rebuilds all layers with the actual data dimensions discovered during Fit().
    /// </summary>
    private void RebuildLayersWithActualDimensions()
    {
        if (_usingCustomLayers) return;

        Layers.Clear();
        SeedLayerInitialization();
        Layers.AddRange(LayerHelper<T>.CreateDefaultMedGANLayers(
            _dataWidth, _options.EmbeddingDimension,
            _options.AutoencoderDimensions, _options.GeneratorDimensions,
            _options.DiscriminatorDimensions, _options.UseMinibatchAveraging));

        ExtractMedGANLayerReferences();
    }

    #endregion

    #region ISyntheticTabularGenerator Implementation

    /// <inheritdoc />
    /// <remarks>
    /// Runs medGAN's two stages in order: the autoencoder is pre-trained on the real records first,
    /// and only then does the GAN begin, with k = 2 discriminator updates per generator update. The
    /// ordering is load-bearing — the generator's whole job is to hit a latent space that already
    /// means something, so starting the GAN against an untrained decoder is training against noise.
    /// </remarks>
    public void Fit(Matrix<T> data, IReadOnlyList<ColumnMetadata> columns, int epochs)
    {
        _columns = new List<ColumnMetadata>(columns);

        _transformer = new TabularDataTransformer<T>(_options.VGMModes, _random);
        _transformer.Fit(data, columns);
        _dataWidth = _transformer.TransformedWidth;
        var transformedData = _transformer.Transform(data);

        BuildOutputGroups();
        LearnObservedRanges(transformedData);
        RebuildLayersWithActualDimensions();

        double noiseMultiplier = _options.EnablePrivacy
            ? ComputeNoiseMultiplier(data.Rows, epochs)
            : 0.0;

        int batchSize = Math.Min(_options.BatchSize, data.Rows);
        if (batchSize <= 0) { IsFitted = true; return; }

        // --- Stage 1: pre-train the autoencoder on the real records ---
        int pretrainEpochs = _options.AutoencoderPretrainEpochs
            ?? Math.Max(1, (int)Math.Round(epochs * _options.AutoencoderPretrainFraction));
        pretrainEpochs = Math.Min(pretrainEpochs, epochs);

        for (int epoch = 0; epoch < pretrainEpochs; epoch++)
        {
            for (int b = 0; b < data.Rows; b += batchSize)
            {
                PretrainAutoencoderStep(transformedData, b, Math.Min(b + batchSize, data.Rows));
            }
        }

        // --- Stage 2: the GAN, with the decoder still training as part of theta_(g,dec) ---
        for (int epoch = pretrainEpochs; epoch < epochs; epoch++)
        {
            for (int b = 0; b < data.Rows; b += batchSize)
            {
                int end = Math.Min(b + batchSize, data.Rows);
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
    /// <remarks>
    /// Synthesis is exactly <c>Dec(G(z))</c> — the generator never emits a record, only a latent
    /// point for the pre-trained decoder to expand.
    /// </remarks>
    public Matrix<T> Generate(int numSamples, Vector<T>? conditionColumn = null, Vector<T>? conditionValue = null)
    {
        if (_transformer is null || _decoderLayers.Count == 0)
        {
            throw new InvalidOperationException("Generator is not fitted. Call Fit() before Generate().");
        }
        if (numSamples <= 0) return new Matrix<T>(0, _dataWidth);

        // Batched so the whole draw is one pass of engine ops. BatchNorm runs in inference mode
        // here, so a sample's output does not depend on which other samples were drawn with it.
        var noise = GenerateNoiseBatchTensor(numSamples);
        var latent = GeneratorForwardBatched(noise, isTraining: false);
        var records = DecoderForwardBatched(latent, applyOutputActivation: true);
        records = ClampToObservedRange(records);

        var result = new Matrix<T>(numSamples, _dataWidth);
        for (int i = 0; i < numSamples; i++)
        {
            for (int j = 0; j < _dataWidth; j++) result[i, j] = records[i, j];
        }

        return _transformer.InverseTransform(result);
    }

    #endregion

    #region NeuralNetworkBase Overrides

    /// <inheritdoc />
    /// <remarks>
    /// Autoencoder reconstruction <c>Dec(Enc(x))</c>. This is medGAN's own compression path, and the
    /// only part of the model that maps a record to a record — the GAN half maps noise to a record
    /// and so has no meaning for <c>Predict</c>.
    /// </remarks>
    protected override Tensor<T> PredictCore(Tensor<T> input)
    {
        EnsureSizedForInput(input);
        var embedding = EncoderForwardBatched(input);
        return DecoderForwardBatched(embedding, applyOutputActivation: true);
    }

    /// <summary>
    /// Training forward — the same autoencoder reconstruction as <see cref="PredictCore"/>,
    /// overridden so the tape-based <see cref="NeuralNetworkBase{T}.Train"/> path trains encoder +
    /// decoder rather than walking the full Layers list (which also holds the generator and
    /// discriminator, and would mis-chain encoder to generator to discriminator).
    /// </summary>
    public override Tensor<T> ForwardForTraining(Tensor<T> input)
    {
        EnsureSizedForInput(input);
        var embedding = EncoderForwardBatched(input);
        return DecoderForwardBatched(embedding, applyOutputActivation: true);
    }

    /// <inheritdoc />
    /// <remarks>
    /// Overridden because the base implementation walks <c>Layers</c> as one straight-through stack,
    /// feeding each layer's output to the next. medGAN's <c>Layers</c> holds FOUR disjoint
    /// sub-networks — encoder, decoder, generator, discriminator — so that walk would push a record
    /// into the generator's prior slot and an embedding into the discriminator's record slot. The
    /// activations reported here follow the model's real graphs instead: the autoencoder path a
    /// record actually takes, then the synthesis path <c>Dec(G(z))</c>, then the critic's verdict.
    /// </remarks>
    public override Dictionary<string, Tensor<T>> GetNamedLayerActivations(Tensor<T> input)
    {
        if (_usingCustomLayers) return base.GetNamedLayerActivations(input);

        EnsureSizedForInput(input);
        var activations = new Dictionary<string, Tensor<T>>();

        var record = input.Rank == 2 ? input : Engine.Reshape(input, [1, input.Length]);

        // Autoencoder path: what a real record does.
        var embedding = EncoderForwardBatched(record);
        activations["Encoder_Embedding"] = embedding.Clone();
        activations["Decoder_Reconstruction"] =
            DecoderForwardBatched(embedding, applyOutputActivation: true).Clone();

        // Synthesis path: Dec(G(z)), the composition that defines medGAN.
        var noise = GenerateNoiseBatchTensor(record.Shape[0]);
        var latent = GeneratorForwardBatched(noise, isTraining: false);
        activations["Generator_Latent"] = latent.Clone();
        var synthetic = DecoderForwardBatched(latent, applyOutputActivation: true);
        activations["Generator_Record"] = synthetic.Clone();

        // Critic: the logit it assigns the real record, with minibatch averaging applied.
        activations["Discriminator_Logit"] = DiscriminatorForwardBatched(record).Clone();

        return activations;
    }

    /// <inheritdoc />
    public override void Train(Tensor<T> input, Tensor<T> expectedOutput)
    {
        SetTrainingMode(true);
        try
        {
            TrainWithTape(input, expectedOutput, _autoencoderOptimizer);
        }
        finally
        {
            SetTrainingMode(false);
        }
    }

    // UpdateParameters re-sliced the flat vector across Layers by hand -- the base walks
    // exactly the same enumeration, so this said nothing the base does not already say.
    /// <inheritdoc />


    /// <inheritdoc />


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
                ["GeneratorType"] = "medGAN",
                ["EmbeddingDimension"] = _options.EmbeddingDimension,
                ["NoiseDimension"] = NoiseDimension,
                ["DataType"] = _options.DataType.ToString(),
                ["UseMinibatchAveraging"] = _options.UseMinibatchAveraging,
                ["DiscriminatorSteps"] = _options.DiscriminatorSteps,
                ["EnablePrivacy"] = _options.EnablePrivacy,
                ["Epsilon"] = _options.Epsilon,
                ["IsFitted"] = IsFitted
            }
        };
    }

    #endregion

    #region Training

    /// <summary>
    /// One autoencoder pre-training step: minimize the reconstruction error of the real records.
    /// </summary>
    /// <remarks>
    /// The loss is the paper's Eq. 2 (mean squared, for counts) or Eq. 3 (cross entropy, for binary
    /// variables), selected by <see cref="MedGANOptions{T}.DataType"/> and generalized per-group for
    /// mixed tabular data. Only the encoder and decoder are on this tape; the GAN has not started.
    /// </remarks>
    private void PretrainAutoencoderStep(Matrix<T> data, int startRow, int endRow)
    {
        int batchSize = endRow - startRow;
        if (batchSize <= 0 || _encoderLayers.Count == 0 || _decoderLayers.Count == 0) return;

        var realBatch = BuildRealBatch(data, startRow, endRow);

        var aeLayers = new List<ILayer<T>>(_encoderLayers.Count + _decoderLayers.Count);
        aeLayers.AddRange(_encoderLayers);
        aeLayers.AddRange(_decoderLayers);
        var aeParams = TapeTrainingStep<T>.CollectParameters(aeLayers);
        if (aeParams.Count == 0) return;

        using var tape = new GradientTape<T>();
        var embedding = EncoderForwardBatched(realBatch);
        var logits = DecoderForwardBatched(embedding, applyOutputActivation: false);
        var lossTensor = ReconstructionLoss(logits, realBatch);

        var grads = tape.ComputeGradients(lossTensor, aeParams);
        T lossValue = lossTensor.Length > 0 ? lossTensor[0] : NumOps.Zero;

        var capturedReal = realBatch;
        Tensor<T> ComputeForward(Tensor<T> inp, Tensor<T> _) =>
            DecoderForwardBatched(EncoderForwardBatched(inp), applyOutputActivation: false);
        Tensor<T> RecomputeLoss(Tensor<T> replayLogits, Tensor<T> _) =>
            ReconstructionLoss(replayLogits, capturedReal);

        var context = new TapeStepContext<T>(
            aeParams, grads, lossValue,
            realBatch, realBatch, ComputeForward, RecomputeLoss,
            parameterBuffer: null);
        _autoencoderOptimizer.Step(context);
    }

    /// <summary>
    /// One discriminator update: ascend <c>log D(x, xbar) + log(1 - D(x_z, xbar_z))</c>.
    /// </summary>
    /// <remarks>
    /// The reference implementation applies a sigmoid and then takes <c>log(y + eps)</c>. This uses
    /// the log-sigmoid of the raw logit instead, which is the same objective without the epsilon
    /// fudge and without the underflow that makes a confident discriminator's loss saturate to
    /// negative infinity.
    /// </remarks>
    private void TrainDiscriminatorStepBatched(Matrix<T> data, int startRow, int endRow, double noiseMultiplier)
    {
        if (_discOutput is null) return;

        int batchSize = endRow - startRow;
        if (batchSize <= 0) return;

        if (noiseMultiplier > 0)
        {
            // Abadi et al. 2016 Algorithm 1 requires clipping every PER-EXAMPLE gradient before
            // aggregation; a single clip of an already-aggregated batch gradient does not satisfy
            // the L2-sensitivity bound the privacy proof rests on.
            TrainDiscriminatorStepPerExampleDPSGD(data, startRow, endRow, noiseMultiplier);
            return;
        }

        var (realBatch, fakeBatch) = BuildRealAndFakeBatches(data, startRow, endRow);

        // GPU-RESIDENT fast path — pack (real, fake) into one persistent input along axis 0 so both
        // score sets come from a single forward, then split inside the loss.
        var trainableDisc = BuildDiscLayerList().OfType<ITrainableLayer<T>>().ToList();
        if (trainableDisc.Count > 0)
        {
            int realN = realBatch.Shape[0];
            int fakeN = fakeBatch.Shape[0];
            var stacked = Engine.TensorConcatenate([realBatch, fakeBatch], axis: 0);
            var target = new Tensor<T>(new[] { 1 });
            // Minibatch averaging must be computed WITHIN each half. Averaging across the stacked
            // real+fake tensor would leak the real batch's composition into the fake samples'
            // companion vector and hand the discriminator a constant, defeating the mechanism.
            Tensor<T> Fwd(Tensor<T> both)
            {
                var (r, f) = SplitStacked(both, realN, fakeN);
                return Engine.TensorConcatenate(
                    [DiscriminatorForwardBatched(r), DiscriminatorForwardBatched(f)], axis: 0);
            }
            Tensor<T> Loss(Tensor<T> allScores, Tensor<T> _)
            {
                var (rScores, fScores) = SplitStacked(allScores, realN, fakeN);
                return DiscriminatorLoss(rScores, fScores);
            }
            if (GpuResidentFusedStep<T>.TryStep(
                    trainableDisc, stacked, target,
                    forward: Fwd, computeLoss: Loss,
                    optimizer: _discriminatorOptimizer,
                    out T _))
            {
                return;
            }
        }

        using var tape = new GradientTape<T>();
        var discParams = TapeTrainingStep<T>.CollectParameters(BuildDiscLayerList());

        var realScores = DiscriminatorForwardBatched(realBatch);
        var fakeScores = DiscriminatorForwardBatched(fakeBatch);
        var lossTensor = DiscriminatorLoss(realScores, fakeScores);

        var grads = tape.ComputeGradients(lossTensor, discParams);
        T lossValue = lossTensor.Length > 0 ? lossTensor[0] : NumOps.Zero;

        // Replay closure must reproduce BOTH terms against the SAME fake batch — resampling here
        // would tie the replayed loss to a different objective than the gradients.
        var capturedFakeBatch = fakeBatch;
        Tensor<T> ComputeForward(Tensor<T> inp, Tensor<T> _) => DiscriminatorForwardBatched(inp);
        Tensor<T> RecomputeLoss(Tensor<T> predReal, Tensor<T> _) =>
            DiscriminatorLoss(predReal, DiscriminatorForwardBatched(capturedFakeBatch));

        var context = new TapeStepContext<T>(
            discParams, grads, lossValue,
            realBatch, realBatch, ComputeForward, RecomputeLoss,
            parameterBuffer: null);
        _discriminatorOptimizer.Step(context);
    }

    /// <summary>
    /// Per-example DP-SGD discriminator step (Abadi et al. 2016 Algorithm 1). Replays
    /// forward+backward once per example, clips each per-example gradient against the GLOBAL L2 norm
    /// across all parameters concatenated, sums, adds a single Gaussian draw, and averages.
    /// </summary>
    /// <remarks>
    /// Beyond the paper — see <see cref="MedGANOptions{T}.EnablePrivacy"/>.
    /// </remarks>
    private void TrainDiscriminatorStepPerExampleDPSGD(Matrix<T> data, int startRow, int endRow, double noiseMultiplier)
    {
        int batchSize = endRow - startRow;
        var discParams = TapeTrainingStep<T>.CollectParameters(BuildDiscLayerList());

        // Pre-materialize per-example (real, fake) batches so both the fused path and the replay
        // closure see the SAME sampled fake data.
        var perExampleReal = new List<Tensor<T>>(batchSize);
        var perExampleFake = new List<Tensor<T>>(batchSize);
        for (int row = startRow; row < endRow; row++)
        {
            var (rb, fb) = BuildRealAndFakeBatches(data, row, row + 1);
            perExampleReal.Add(rb);
            perExampleFake.Add(fb);
        }

        T lossSum = NumOps.Zero;
        using var dpSgdStep = new DpSgdFusedStep<T>();
        bool dpFusedRan = dpSgdStep.TryStep(
            parameters: discParams,
            perExampleSlotData: exIdx => new[] { perExampleReal[exIdx], perExampleFake[exIdx] },
            forward: slots => DiscriminatorForwardBatched(slots[0]),
            computeLoss: (realScores, slots) =>
            {
                var fakeScores = DiscriminatorForwardBatched(slots[1]);
                var lossTensor = DiscriminatorLoss(realScores, fakeScores);
                if (lossTensor.Length > 0) lossSum = NumOps.Add(lossSum, lossTensor[0]);
                return lossTensor;
            },
            batchSize: batchSize,
            clipNorm: _options.ClipNorm,
            noiseMultiplier: noiseMultiplier,
            rng: _random,
            out var noisedAvgGrads);

        // Eager fallback: same clip-BEFORE-aggregate contract, by manual accumulation.
        if (!dpFusedRan)
        {
            noisedAvgGrads = new Dictionary<Tensor<T>, Tensor<T>>(TensorReferenceComparer<Tensor<T>>.Instance);
            var gradSum = new Dictionary<Tensor<T>, Tensor<T>>(TensorReferenceComparer<Tensor<T>>.Instance);
            foreach (var p in discParams)
            {
                var zero = new Tensor<T>(p._shape);
                zero.Fill(NumOps.Zero);
                gradSum[p] = zero;
            }
            for (int row = startRow; row < endRow; row++)
            {
                var realBatch = perExampleReal[row - startRow];
                var fakeBatch = perExampleFake[row - startRow];
                using var tape = new GradientTape<T>();
                var lossTensor = DiscriminatorLoss(
                    DiscriminatorForwardBatched(realBatch), DiscriminatorForwardBatched(fakeBatch));
                if (lossTensor.Length > 0) lossSum = NumOps.Add(lossSum, lossTensor[0]);
                var grads = tape.ComputeGradients(lossTensor, discParams);

                // GLOBAL L2 norm across ALL parameter gradients (the sensitivity contract).
                T normSquared = NumOps.Zero;
                foreach (var g in grads.Values)
                {
                    var perParamSum = Engine.ReduceSum(Engine.TensorMultiply(g, g), axes: null, keepDims: false);
                    normSquared = NumOps.Add(normSquared, perParamSum.Length > 0 ? perParamSum[0] : NumOps.Zero);
                }
                double clipFactor = Math.Min(1.0, _options.ClipNorm / Math.Sqrt(NumOps.ToDouble(normSquared) + 1e-12));
                var clipFactorT = NumOps.FromDouble(clipFactor);
                foreach (var kvp in grads)
                {
                    gradSum[kvp.Key] = Engine.TensorAdd(
                        gradSum[kvp.Key], Engine.TensorMultiplyScalar(kvp.Value, clipFactorT));
                }
            }

            double invBatch = 1.0 / batchSize;
            double noiseStdD = _options.ClipNorm * noiseMultiplier * invBatch;
            var invBatchT = NumOps.FromDouble(invBatch);
            var noiseStdT = NumOps.FromDouble(noiseStdD);
            foreach (var kvp in gradSum)
            {
                var scaledSum = Engine.TensorMultiplyScalar(kvp.Value, invBatchT);
                if (noiseStdD > 0)
                {
                    var noise = new Tensor<T>(kvp.Value._shape);
                    Engine.TensorRandomNormalInto(noise, NumOps.Zero, noiseStdT);
                    noisedAvgGrads[kvp.Key] = Engine.TensorAdd(scaledSum, noise);
                }
                else
                {
                    noisedAvgGrads[kvp.Key] = scaledSum;
                }
            }
        }

        var stackedReal = Engine.TensorConcatenate([.. perExampleReal], axis: 0);
        var stackedFake = Engine.TensorConcatenate([.. perExampleFake], axis: 0);
        T avgLoss = NumOps.Divide(lossSum, NumOps.FromDouble(batchSize));

        var capturedFake = stackedFake;
        Tensor<T> ComputeForward(Tensor<T> inp, Tensor<T> _) => DiscriminatorForwardBatched(inp);
        Tensor<T> RecomputeLoss(Tensor<T> predReal, Tensor<T> _) =>
            DiscriminatorLoss(predReal, DiscriminatorForwardBatched(capturedFake));

        var context = new TapeStepContext<T>(
            discParams, noisedAvgGrads, avgLoss,
            stackedReal, stackedReal, ComputeForward, RecomputeLoss,
            parameterBuffer: null);
        _discriminatorOptimizer.Step(context);
    }

    /// <summary>
    /// One generator update: ascend <c>log D(x_z, xbar_z)</c> over <c>theta_(g,dec)</c>.
    /// </summary>
    /// <remarks>
    /// <para>
    /// The trainable surface is the generator AND the decoder — that joint set is what the paper
    /// writes as <c>theta_(g,dec)</c>, and it is why the decoder is described as pre-trained rather
    /// than frozen. The encoder is deliberately absent: after pre-training it has no further role,
    /// and including it here would let adversarial gradients reshape the latent space the generator
    /// is trying to hit.
    /// </para>
    /// <para>
    /// No differential privacy is applied on this step even when privacy is enabled: the generator
    /// never touches real data, so its guarantee follows from the discriminator's by
    /// post-processing.
    /// </para>
    /// </remarks>
    private void TrainGeneratorStepBatched(int batchSize)
    {
        if (batchSize <= 0 || _generatorLayers.Count == 0 || _decoderLayers.Count == 0) return;

        var genLayers = new List<ILayer<T>>(
            _generatorLayers.Count + _generatorBN.Count + _decoderLayers.Count);
        genLayers.AddRange(_generatorLayers);
        genLayers.AddRange(_generatorBN);
        genLayers.AddRange(_decoderLayers);
        var genParams = TapeTrainingStep<T>.CollectParameters(genLayers);
        if (genParams.Count == 0) return;

        var noiseBatch = GenerateNoiseBatchTensor(batchSize);

        // GPU-RESIDENT fast path — the fused plan captures the whole
        // (z -> G -> Dec -> D-frozen -> non-saturating loss) chain. The discriminator's layers are
        // not in genParams, so no gradients accumulate to them here.
        var trainableGen = genLayers.OfType<ITrainableLayer<T>>().ToList();
        if (trainableGen.Count > 0)
        {
            var target = new Tensor<T>(new[] { 1 });
            Tensor<T> Fwd(Tensor<T> nb) => SynthesizeForDiscriminator(nb, isTraining: true);
            Tensor<T> Loss(Tensor<T> fake, Tensor<T> _) => GeneratorLoss(fake);
            if (GpuResidentFusedStep<T>.TryStep(
                    trainableGen, noiseBatch, target,
                    forward: Fwd, computeLoss: Loss,
                    optimizer: _generatorOptimizer,
                    out T _))
            {
                return;
            }
        }

        using var tape = new GradientTape<T>();
        var fakeBatch = SynthesizeForDiscriminator(noiseBatch, isTraining: true);
        var lossTensor = GeneratorLoss(fakeBatch);

        var grads = tape.ComputeGradients(lossTensor, genParams);
        T lossValue = lossTensor.Length > 0 ? lossTensor[0] : NumOps.Zero;

        Tensor<T> ComputeForward(Tensor<T> inp, Tensor<T> _) => SynthesizeForDiscriminator(inp, isTraining: true);
        Tensor<T> RecomputeLoss(Tensor<T> fake, Tensor<T> _) => GeneratorLoss(fake);

        var context = new TapeStepContext<T>(
            genParams, grads, lossValue,
            noiseBatch, noiseBatch, ComputeForward, RecomputeLoss,
            parameterBuffer: null);
        _generatorOptimizer.Step(context);
    }

    /// <summary>
    /// <c>Dec(G(z))</c> with the decoder's output activation applied, i.e. exactly what the
    /// discriminator is shown as a fake sample.
    /// </summary>
    internal Tensor<T> SynthesizeForDiscriminator(Tensor<T> noise, bool isTraining)
    {
        var latent = GeneratorForwardBatched(noise, isTraining);
        return DecoderForwardBatched(latent, applyOutputActivation: true);
    }

    /// <summary>
    /// <c>-mean log sigmoid(D(real)) - mean log(1 - sigmoid(D(fake)))</c>, the descent form of the
    /// discriminator's ascent objective, written through log-sigmoid so it stays finite.
    /// </summary>
    private Tensor<T> DiscriminatorLoss(Tensor<T> realScores, Tensor<T> fakeScores)
    {
        var axes = Enumerable.Range(0, realScores.Shape.Length).ToArray();
        var lossReal = Engine.TensorNegate(Engine.ReduceMean(LogSigmoid(realScores), axes, keepDims: false));
        var lossFake = Engine.TensorNegate(
            Engine.ReduceMean(LogSigmoid(Engine.TensorNegate(fakeScores)), axes, keepDims: false));
        return Engine.TensorAdd(lossReal, lossFake);
    }

    /// <summary>
    /// <c>-mean log sigmoid(D(Dec(G(z))))</c>, plus the optional out-of-range penalty.
    /// </summary>
    private Tensor<T> GeneratorLoss(Tensor<T> fakeRecords)
    {
        var fakeScores = DiscriminatorForwardBatched(fakeRecords);
        var axes = Enumerable.Range(0, fakeScores.Shape.Length).ToArray();
        Tensor<T> loss = Engine.TensorNegate(
            Engine.ReduceMean(LogSigmoid(fakeScores), axes, keepDims: false));

        // Beyond the paper, disabled by default (ConstraintWeight = 0).
        if (_options.ConstraintWeight > 0.0 && _colMin is not null && _colMax is not null)
        {
            loss = Engine.TensorAdd(loss, Engine.TensorMultiplyScalar(
                OutOfRangePenalty(fakeRecords), NumOps.FromDouble(_options.ConstraintWeight)));
        }
        return loss;
    }

    /// <summary>
    /// Numerically-stable <c>log(sigmoid(x))</c> = <c>-softplus(-x)</c>, through tape-tracked engine
    /// ops so backprop flows correctly.
    /// </summary>
    private Tensor<T> LogSigmoid(Tensor<T> x)
    {
        // Engine.Softplus uses the max(z,0) + log(1+exp(-|z|)) identity internally, so the dynamic
        // range survives for confident discriminator outputs where naive log(sigmoid(x)) underflows.
        return Engine.TensorNegate(Engine.Softplus(Engine.TensorNegate(x)));
    }

    private double ComputeNoiseMultiplier(int dataSize, int epochs)
    {
        double delta = 1.0 / ((double)dataSize * dataSize);
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

    // The forward passes below are `internal` rather than `private` so the mechanism tests can
    // assert medGAN's actual mechanisms — minibatch averaging, the generator shortcuts, the
    // Dec(G(z)) composition — instead of proxies for them. AiDotNetTests has InternalsVisibleTo.
    #region Forward Passes

    /// <summary>
    /// Encoder: feedforward down to the embedding, with the autoencoder activation on every layer
    /// including the embedding projection.
    /// </summary>
    internal Tensor<T> EncoderForwardBatched(Tensor<T> input)
    {
        var current = input;
        for (int i = 0; i < _encoderLayers.Count; i++)
        {
            current = AutoencoderActivation(_encoderLayers[i].Forward(current));
        }
        return current;
    }

    /// <summary>
    /// Decoder: the encoder mirrored. Hidden layers use the autoencoder activation; the final
    /// projection uses the output activation selected by <see cref="MedGANOptions{T}.DataType"/>,
    /// unless <paramref name="applyOutputActivation"/> is false so a loss can consume raw logits.
    /// </summary>
    internal Tensor<T> DecoderForwardBatched(Tensor<T> embedding, bool applyOutputActivation)
    {
        var current = embedding;
        for (int i = 0; i < _decoderLayers.Count - 1; i++)
        {
            current = AutoencoderActivation(_decoderLayers[i].Forward(current));
        }
        if (_decoderLayers.Count > 0)
        {
            current = _decoderLayers[^1].Forward(current);
            if (applyOutputActivation) current = DecoderOutputActivation(current);
        }
        return current;
    }

    /// <summary>
    /// Generator: <c>x_k = ReLU(BN_k(W_k x_(k-1))) + x_(k-1)</c> for every layer, with the final
    /// projection into the embedding space using tanh (binary) or ReLU (counts) before its own
    /// shortcut.
    /// </summary>
    /// <remarks>
    /// The shortcut is applied to the LAST layer too, not only the hidden ones. That is what the
    /// reference implementation does, and it is why the prior dimension has to equal the embedding
    /// dimension rather than merely the hidden widths matching each other.
    /// </remarks>
    internal Tensor<T> GeneratorForwardBatched(Tensor<T> z, bool isTraining)
    {
        var current = z;
        int last = _generatorLayers.Count - 1;
        for (int i = 0; i < _generatorLayers.Count; i++)
        {
            var h = _generatorLayers[i].Forward(current);
            if (i < _generatorBN.Count)
            {
                _generatorBN[i].SetTrainingMode(isTraining);
                h = _generatorBN[i].Forward(h);
            }
            h = i == last
                ? (_options.DataType == MedGANDataType.Count ? Engine.ReLU(h) : Engine.Tanh(h))
                : Engine.ReLU(h);
            current = Engine.TensorAdd(h, current);
        }
        return current;
    }

    /// <summary>
    /// Discriminator: plain feedforward with ReLU hidden activations and a scalar logit output,
    /// preceded by minibatch averaging.
    /// </summary>
    /// <remarks>
    /// The returned value is a LOGIT, not a probability. The paper's discriminator ends in a
    /// sigmoid; that sigmoid lives in <see cref="LogSigmoid"/> inside the losses instead, which is
    /// the same function composed differently and does not underflow.
    /// </remarks>
    internal Tensor<T> DiscriminatorForwardBatched(Tensor<T> records)
    {
        var current = ApplyMinibatchAveraging(records);
        for (int i = 0; i < _discLayers.Count; i++)
        {
            current = Engine.ReLU(_discLayers[i].Forward(current));
        }
        if (_discOutput is not null) current = _discOutput.Forward(current);
        return current;
    }

    /// <summary>
    /// Concatenates the minibatch average onto every sample: <c>[x_i ; xbar]</c>.
    /// </summary>
    /// <remarks>
    /// <para>
    /// medGAN's remedy for mode collapse. The average is computed over the samples PRESENT in this
    /// call, which is what makes it informative — for binary variables it is the maximum-likelihood
    /// estimate of each code's Bernoulli success probability, so a generator that emits the same
    /// record repeatedly produces an average that betrays it.
    /// </para>
    /// <para>
    /// Real and fake batches are therefore averaged separately, never jointly.
    /// </para>
    /// </remarks>
    internal Tensor<T> ApplyMinibatchAveraging(Tensor<T> records)
    {
        if (!_options.UseMinibatchAveraging || records.Rank != 2) return records;

        int batch = records.Shape[0];
        int width = records.Shape[1];
        var average = Engine.ReduceMean(records, [0], keepDims: true);          // [1, W]
        var tiled = Engine.TensorBroadcastTo(average, [batch, width]);          // [B, W]
        return Engine.TensorConcatenate([records, tiled], axis: 1);             // [B, 2W]
    }

    /// <summary>
    /// The autoencoder's hidden activation: tanh for binary and mixed data, ReLU for counts.
    /// </summary>
    private Tensor<T> AutoencoderActivation(Tensor<T> x) =>
        _options.DataType == MedGANDataType.Count ? Engine.ReLU(x) : Engine.Tanh(x);

    /// <summary>
    /// The decoder's output activation: sigmoid for binary, ReLU for counts, and for mixed tabular
    /// data tanh on each mode-normalized scalar with a softmax over each one-hot group.
    /// </summary>
    private Tensor<T> DecoderOutputActivation(Tensor<T> logits)
    {
        switch (_options.DataType)
        {
            case MedGANDataType.Binary:
                return Engine.Sigmoid(logits);

            case MedGANDataType.Count:
                return Engine.ReLU(logits);

            default:
                // Unfitted (no transformer yet): tanh over the whole vector. It is the right shape
                // for the mode-normalized scalars that dominate the representation, and this path
                // only runs before Fit has established the group layout.
                if (_outputGroups is null || logits.Rank != 2) return Engine.Tanh(logits);

                var parts = new List<Tensor<T>>(_outputGroups.Count);
                foreach (var (start, width, softmax) in _outputGroups)
                {
                    var slice = SliceColumns(logits, start, width);
                    parts.Add(softmax ? Engine.Softmax(slice, -1) : Engine.Tanh(slice));
                }
                return Engine.TensorConcatenate([.. parts], axis: 1);
        }
    }

    #endregion

    #region Reconstruction Loss

    /// <summary>
    /// The autoencoder's pre-training loss over raw decoder logits.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Binary — Eq. 3, the cross entropy, computed from logits as
    /// <c>softplus(z) - x*z</c>, which is algebraically identical to
    /// <c>-(x log sigmoid(z) + (1-x) log(1 - sigmoid(z)))</c> and does not need the reference
    /// implementation's epsilon inside the log.
    /// </para>
    /// <para>
    /// Count — Eq. 2, the squared error against <c>ReLU(z)</c>.
    /// </para>
    /// <para>
    /// Mixed tabular — the generalization of both: squared error on the mode-normalized scalars and
    /// softmax cross entropy on each one-hot group, which is the multi-class form of Eq. 3.
    /// </para>
    /// <para>
    /// All three sum over features and average over the batch, matching the paper's
    /// <c>sum</c>-inside, <c>mean</c>-outside form.
    /// </para>
    /// </remarks>
    internal Tensor<T> ReconstructionLoss(Tensor<T> logits, Tensor<T> target)
    {
        switch (_options.DataType)
        {
            case MedGANDataType.Binary:
            {
                // softplus(z) - x*z, summed over features, averaged over the batch.
                var perElement = Engine.TensorSubtract(
                    Engine.Softplus(logits), Engine.TensorMultiply(target, logits));
                return MeanOverBatchOfFeatureSum(perElement);
            }

            case MedGANDataType.Count:
            {
                var diff = Engine.TensorSubtract(Engine.ReLU(logits), target);
                return MeanOverBatchOfFeatureSum(Engine.TensorMultiply(diff, diff));
            }

            default:
            {
                if (_outputGroups is null || logits.Rank != 2)
                {
                    var d = Engine.TensorSubtract(Engine.Tanh(logits), target);
                    return MeanOverBatchOfFeatureSum(Engine.TensorMultiply(d, d));
                }

                Tensor<T>? total = null;
                foreach (var (start, width, softmax) in _outputGroups)
                {
                    var zGroup = SliceColumns(logits, start, width);
                    var xGroup = SliceColumns(target, start, width);
                    Tensor<T> term;
                    if (softmax)
                    {
                        // -sum_k x_k log softmax(z)_k over the group, per sample.
                        var logProbs = ObjectiveOps.LogSoftmax(zGroup, axis: 1);
                        term = Engine.TensorNegate(
                            Engine.ReduceSum(Engine.TensorMultiply(xGroup, logProbs), [1], keepDims: false));
                    }
                    else
                    {
                        var d = Engine.TensorSubtract(Engine.Tanh(zGroup), xGroup);
                        term = Engine.ReduceSum(Engine.TensorMultiply(d, d), [1], keepDims: false);
                    }
                    total = total is null ? term : Engine.TensorAdd(total, term);
                }

                if (total is null) return new Tensor<T>([1]);
                return Engine.ReduceMean(total, [0], keepDims: false);
            }
        }
    }

    /// <summary>
    /// Sums over every non-batch axis, then averages over the batch — the paper's
    /// <c>mean_i sum_j</c> form rather than a flat mean over all elements.
    /// </summary>
    private Tensor<T> MeanOverBatchOfFeatureSum(Tensor<T> perElement)
    {
        if (perElement.Rank < 2)
        {
            return Engine.ReduceSum(perElement, axes: null, keepDims: false);
        }
        var featureAxes = Enumerable.Range(1, perElement.Rank - 1).ToArray();
        var perSample = Engine.ReduceSum(perElement, featureAxes, keepDims: false);
        return Engine.ReduceMean(perSample, [0], keepDims: false);
    }

    /// <summary>
    /// Squared violation of the per-column range observed during Fit, averaged over the batch.
    /// Beyond the paper; see <see cref="MedGANOptions{T}.ConstraintWeight"/>.
    /// </summary>
    private Tensor<T> OutOfRangePenalty(Tensor<T> records)
    {
        if (_colMin is null || _colMax is null || records.Rank != 2) return new Tensor<T>([1]);

        int batch = records.Shape[0];
        int cols = Math.Min(records.Shape[1], _colMin.Length);
        if (cols <= 0) return new Tensor<T>([1]);

        var lowerArr = new T[batch * cols];
        var upperArr = new T[batch * cols];
        for (int b = 0; b < batch; b++)
        {
            for (int j = 0; j < cols; j++)
            {
                lowerArr[b * cols + j] = NumOps.FromDouble(_colMin[j]);
                upperArr[b * cols + j] = NumOps.FromDouble(_colMax[j]);
            }
        }
        var view = cols == records.Shape[1] ? records : SliceColumns(records, 0, cols);
        var lower = new Tensor<T>(lowerArr, [batch, cols]);
        var upper = new Tensor<T>(upperArr, [batch, cols]);
        var over = Engine.ReLU(Engine.TensorSubtract(view, upper));
        var under = Engine.ReLU(Engine.TensorSubtract(lower, view));
        var violation = Engine.TensorAdd(
            Engine.TensorMultiply(over, over), Engine.TensorMultiply(under, under));
        return MeanOverBatchOfFeatureSum(violation);
    }

    #endregion

    #region Batch Construction and Layout

    private Tensor<T> BuildRealBatch(Matrix<T> data, int startRow, int endRow)
    {
        int batchSize = endRow - startRow;
        var realBatch = new Tensor<T>([batchSize, _dataWidth]);
        int cols = Math.Min(_dataWidth, data.Columns);
        for (int b = 0; b < batchSize; b++)
        {
            int row = startRow + b;
            for (int j = 0; j < cols; j++) realBatch[b, j] = data[row, j];
        }
        return realBatch;
    }

    /// <summary>
    /// Builds a real batch and the matching synthetic batch <c>Dec(G(z))</c> for a critic step.
    /// </summary>
    private (Tensor<T> realBatch, Tensor<T> fakeBatch) BuildRealAndFakeBatches(
        Matrix<T> data, int startRow, int endRow)
    {
        var realBatch = BuildRealBatch(data, startRow, endRow);
        var noiseBatch = GenerateNoiseBatchTensor(endRow - startRow);
        var fakeBatch = SynthesizeForDiscriminator(noiseBatch, isTraining: false);
        return (realBatch, fakeBatch);
    }

    internal Tensor<T> GenerateNoiseBatchTensor(int batchSize)
    {
        int noiseDim = NoiseDimension;
        int totalElements = batchSize * noiseDim;
        // Box-Muller via the seeded _random so MedGANOptions.Seed makes Fit reproducible —
        // Engine.TensorRandomUniformRange bypasses _random and breaks the seed contract.
        var noiseData = new T[totalElements];
        for (int i = 0; i < totalElements; i += 2)
        {
            double u1 = 1.0 - _random.NextDouble();   // in (0, 1] keeps log finite
            double u2 = _random.NextDouble();
            double r = Math.Sqrt(-2.0 * Math.Log(u1));
            double theta = 2.0 * Math.PI * u2;
            noiseData[i] = NumOps.FromDouble(r * Math.Cos(theta));
            if (i + 1 < totalElements)
                noiseData[i + 1] = NumOps.FromDouble(r * Math.Sin(theta));
        }
        return new Tensor<T>(noiseData, [batchSize, noiseDim]);
    }

    /// <summary>
    /// Builds the discriminator's layer list (hidden layers plus output) for parameter collection.
    /// </summary>
    private IReadOnlyList<ILayer<T>> BuildDiscLayerList()
    {
        var all = new List<ILayer<T>>(_discLayers.Count + 1);
        all.AddRange(_discLayers);
        if (_discOutput is not null) all.Add(_discOutput);
        return all;
    }

    /// <summary>Takes <paramref name="width"/> columns starting at <paramref name="start"/>.</summary>
    private Tensor<T> SliceColumns(Tensor<T> t, int start, int width)
    {
        var sliceStart = new int[t.Rank];
        sliceStart[1] = start;
        var sliceShape = t._shape.ToArray();
        sliceShape[1] = width;
        return Engine.TensorSlice(t, sliceStart, sliceShape);
    }

    /// <summary>Splits a tensor stacked along axis 0 back into its two halves.</summary>
    private (Tensor<T> First, Tensor<T> Second) SplitStacked(Tensor<T> stacked, int firstCount, int secondCount)
    {
        var firstShape = stacked._shape.ToArray();
        firstShape[0] = firstCount;
        var secondShape = stacked._shape.ToArray();
        secondShape[0] = secondCount;
        var firstStart = new int[stacked.Rank];
        var secondStart = new int[stacked.Rank];
        secondStart[0] = firstCount;
        return (Engine.TensorSlice(stacked, firstStart, firstShape),
                Engine.TensorSlice(stacked, secondStart, secondShape));
    }

    /// <summary>
    /// Records the column layout of the transformed representation: for each original column, a
    /// mode-normalized scalar plus a one-hot mode indicator (continuous) or a single one-hot group
    /// (categorical). Drives the MixedTabular activation and reconstruction loss.
    /// </summary>
    private void BuildOutputGroups()
    {
        if (_transformer is null) { _outputGroups = null; return; }

        var groups = new List<(int, int, bool)>();
        int offset = 0;
        for (int col = 0; col < _columns.Count; col++)
        {
            var info = _transformer.GetTransformInfo(col);
            if (info.IsContinuous)
            {
                groups.Add((offset, 1, false));                 // mode-normalized scalar
                offset += 1;
                int modes = info.Width - 1;
                if (modes > 0)
                {
                    groups.Add((offset, modes, true));          // one-hot mode indicator
                    offset += modes;
                }
            }
            else if (info.Width > 0)
            {
                groups.Add((offset, info.Width, true));         // one-hot category
                offset += info.Width;
            }
        }

        // Any tail the transformer reports but the loop did not cover is treated as scalars rather
        // than silently dropped, which would exclude those columns from the loss entirely.
        if (offset < _dataWidth) groups.Add((offset, _dataWidth - offset, false));

        _outputGroups = groups.Count > 0 ? groups : null;
    }

    private void LearnObservedRanges(Matrix<T> data)
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

    /// <summary>
    /// Clamps generated records into the observed range. Applied only when the out-of-range penalty
    /// is enabled, so that the default path returns the decoder's output untouched as the paper's
    /// does.
    /// </summary>
    private Tensor<T> ClampToObservedRange(Tensor<T> records)
    {
        if (_options.ConstraintWeight <= 0.0 || _colMin is null || _colMax is null) return records;

        var clamped = new Tensor<T>(records._shape);
        int width = records.Rank == 2 ? records.Shape[1] : records.Length;
        for (int i = 0; i < records.Length; i++)
        {
            int col = width > 0 ? i % width : 0;
            double v = NumOps.ToDouble(records[i]);
            if (col < _colMin.Length)
            {
                v = Math.Min(Math.Max(v, _colMin[col]), _colMax[col]);
            }
            clamped[i] = NumOps.FromDouble(v);
        }
        return clamped;
    }

    #endregion
}
