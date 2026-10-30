using AiDotNet.Attributes;
using AiDotNet.Autodiff;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.NeuralNetworks.SyntheticData;
using AiDotNet.Optimizers;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Training;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.NeuralNetworks.SyntheticData;

/// <summary>
/// CTAB-GAN+ generator for high-quality synthetic tabular data with auxiliary classifier
/// discriminator, mixed-type encoder, and information loss.
/// </summary>
/// <remarks>
/// <para>
/// This implementation follows the standard neural network architecture pattern with:
/// - Proper inheritance from NeuralNetworkBase
/// - Layer-based architecture using ILayer components
/// - Engine-based tensor operations for CPU/GPU acceleration
/// - Full autodiff and JIT compilation support
/// </para>
/// <para>
/// CTAB-GAN+ extends the CTGAN architecture with:
/// - <b>ACGAN-style discriminator</b>: Outputs both real/fake score and class label predictions
/// - <b>Information loss</b>: Penalizes statistical divergence between real and generated data
/// - <b>Log-frequency encoding</b>: Better handling of long-tail categorical distributions
/// - <b>Conditional training</b>: Same training-by-sampling as CTGAN
/// </para>
/// <para>
/// <b>For Beginners:</b> CTAB-GAN+ works like CTGAN but with smarter training:
///
/// <b>Architecture:</b>
/// <code>
/// Generator:      noise + condition -> [FC+BN+ReLU] x L (residual) -> fake row
/// Discriminator:  row + condition -> [FC+LeakyReLU+Dropout] x L -> {real/fake score, class logits}
/// </code>
///
/// <b>Three-part loss for discriminator:</b>
/// 1. Wasserstein loss (real vs fake)
/// 2. Auxiliary classifier loss (correct class prediction)
/// 3. Gradient penalty (training stability)
///
/// <b>Three-part loss for generator:</b>
/// 1. Fool discriminator (maximize fake score)
/// 2. Correct class via auxiliary classifier
/// 3. Information loss (match data statistics)
///
/// If you provide custom layers in the architecture, those will be used directly
/// for the generator network. If not, the network creates industry-standard
/// CTAB-GAN+ layers based on the original research paper specifications.
///
/// Example usage:
/// <code>
/// var architecture = new NeuralNetworkArchitecture&lt;double&gt;(
///     inputFeatures: 10,
///     outputSize: 10
/// );
/// var options = new CTABGANPlusOptions&lt;double&gt;
/// {
///     GeneratorDimensions = new[] { 256, 256 },
///     DiscriminatorDimensions = new[] { 256, 256 },
///     ClassifierWeight = 0.5,
///     Epochs = 300
/// };
/// var generator = new CTABGANPlusGenerator&lt;double&gt;(architecture, options);
/// generator.Fit(data, columns, epochs: 300);
/// var synthetic = generator.Generate(1000);
/// </code>
/// </para>
/// <para>
/// Reference: "CTAB-GAN: Effective Table Data Synthesizing" (Zhao et al., 2021)
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
[ModelDomain(ModelDomain.MachineLearning)]
[ModelCategory(ModelCategory.NeuralNetwork)]
[ModelCategory(ModelCategory.SyntheticDataGenerator)]
[ModelTask(ModelTask.Generation)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("CTAB-GAN+: Enhancing Tabular Data Synthesis",
    "https://arxiv.org/abs/2204.00401",
    Year = 2023,
    Authors = "Zilong Zhao, Aditya Kunar, Robert Birke, Lydia Y. Chen")]
public partial class CTABGANPlusGenerator<T> : NeuralSyntheticTabularGeneratorBase<T>, ISyntheticTabularGenerator<T>
{
    private readonly CTABGANPlusOptions<T> _options;
    private IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> _optimizer;
    private ILossFunction<T> _lossFunction;

    // Synthetic tabular data infrastructure
    private TabularDataTransformer<T>? _transformer;
    private CTGANDataSampler<T>? _sampler;
    private List<ColumnMetadata> _columns = new();
    private int _dataWidth;
    private int _condWidth;
    private int _packedInputDim;
    private Random _random;

    // WGAN-GP Adam instances for the critic and the auxiliary classifier, separate from the generator's so their
    // moment estimates never cross-contaminate.
    private IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> _discriminatorOptimizer;
    private IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> _classifierOptimizer;

    // Held across Train calls. The fused plan owns the critic optimizer's state, so a step object built per call
    // re-traced and recompiled every call and restarted the optimizer at t = 1.
    private AiDotNet.Training.WganGpFusedStep<T>? _wganFusedStep;

    // Each categorical column's (offset, width) in the transformed row, in the conditional vector's column order.
    private readonly List<(int Offset, int Width)> _catOutputBlocks = new();

    // Generator batch normalization layers (auxiliary, always created to match Layers)
    private readonly List<BatchNormalizationLayer<T>> _genBNLayers = new();

    // Discriminator layers (auxiliary, not user-overridable)
    private readonly List<ILayer<T>> _discLayers = new();
    private readonly List<(int InputSize, int OutputSize)> _discLayerDims = new();

    // Auxiliary classifier (Zhao et al. 2021 §3.1): an MLP of the critic's hidden widths and a class head, predicting
    // the target column from the rest of a row. It is a separate network from the critic.
    private readonly List<ILayer<T>> _classifierLayers = new();
    private FullyConnectedLayer<T>? _classifierHead;
    private int _numClasses;
    private int _targetColTransformOffset;
    // False when the data has no categorical target column, which leaves the classifier and its loss out.
    private bool _hasClassifierTarget;

    // Cached pre-activations for proper backward passes
    [Scratch]
    private readonly List<Tensor<T>> _genPreActivations = new();

    // Whether custom layers are being used (disables residual connection logic)
    private bool _usingCustomLayers;

    /// <summary>
    /// Gets the CTAB-GAN+-specific options.
    /// </summary>
    public new CTABGANPlusOptions<T> Options => _options;

    /// <inheritdoc />
    public IReadOnlyList<ColumnMetadata> Columns => _columns.AsReadOnly();

    /// <inheritdoc />
    public bool IsFitted { get; private set; }

    /// <summary>
    /// Initializes a new CTAB-GAN+ generator with the specified architecture.
    /// </summary>
    /// <param name="architecture">The neural network architecture defining input/output dimensions and optional custom layers.</param>
    /// <param name="options">CTAB-GAN+-specific options for generator and discriminator configuration.</param>
    /// <param name="optimizer">Gradient-based optimizer (defaults to Adam).</param>
    /// <param name="lossFunction">Loss function (defaults based on task type).</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This constructor creates a CTAB-GAN+ network based on the architecture you provide.
    ///
    /// If you provide custom layers in the architecture, those will be used directly
    /// for the generator network. If not, the network will create industry-standard
    /// CTAB-GAN+ generator layers based on the original research paper specifications.
    ///
    /// The discriminator (with auxiliary classifier head) is always created internally.
    /// </para>
    /// </remarks>
    public CTABGANPlusGenerator()
        : this(new NeuralNetworkArchitecture<T>(InputType.OneDimensional, NeuralNetworkTaskType.Generative, inputSize: 128, outputSize: 128))
    {
    }

    public CTABGANPlusGenerator(NeuralNetworkArchitecture<T> architecture,
        CTABGANPlusOptions<T>? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
        ILossFunction<T>? lossFunction = null)
        : base(architecture, lossFunction ?? NeuralNetworkHelper<T>.GetDefaultLossFunction(architecture.TaskType), (options ??= new CTABGANPlusOptions<T>()).MaxGradNorm)
    {
        _options = options;
        _lossFunction = lossFunction ?? NeuralNetworkHelper<T>.GetDefaultLossFunction(architecture.TaskType);
        // WGAN-GP Adam configuration (Gulrajani et al. 2017): beta1 = 0.5, beta2 = 0.9.
        AdamOptimizer<T, Tensor<T>, Tensor<T>> MakeAdam() =>
            new(this, new AdamOptimizerOptions<T, Tensor<T>, Tensor<T>>
            {
                InitialLearningRate = _options.LearningRate,
                Beta1 = 0.5,
                Beta2 = 0.9,
                UseAdaptiveLearningRate = false,
                UseAMSGrad = false,
            });
        _optimizer = optimizer ?? MakeAdam();
        _discriminatorOptimizer = MakeAdam();
        _classifierOptimizer = MakeAdam();
        _random = _options.Seed.HasValue
            ? RandomHelper.CreateSeededRandom(_options.Seed.Value)
            : RandomHelper.CreateSecureRandom();

        InitializeLayers();
    }

    /// <inheritdoc/>
    protected override void Dispose(bool disposing)
    {
        if (disposing)
        {
            _wganFusedStep?.Dispose();
            _wganFusedStep = null;
        }
        base.Dispose(disposing);
    }

    #region Layer Initialization (GANDALF Pattern)

    /// <summary>
    /// Initializes the layers of the CTAB-GAN+ network based on the provided architecture.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This method either uses custom layers provided in the architecture or creates
    /// default CTAB-GAN+ generator layers following the original paper specifications.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> This method sets up the generator network structure:
    /// - If you provided custom layers, those are used for the generator
    /// - Otherwise, it creates the standard CTAB-GAN+ generator architecture:
    ///   Dense layers with batch normalization and residual connections
    ///
    /// The discriminator and auxiliary classifier are always created internally.
    /// </para>
    /// </remarks>
    protected override void InitializeLayers()
    {
        if (Architecture.Layers != null && Architecture.Layers.Count > 0)
        {
            Layers.AddRange(Architecture.Layers);
            ValidateCustomLayers(Layers);
            _usingCustomLayers = true;
        }
        else
        {
            int inputDim = _options.EmbeddingDimension + Architecture.CalculatedInputSize;
            int outputDim = Architecture.OutputSize;
            Layers.AddRange(LayerHelper<T>.CreateDefaultCTABGANPlusGeneratorLayers(
                inputDim, outputDim, _options.GeneratorDimensions));

            _genBNLayers.Clear();
            foreach (int dim in _options.GeneratorDimensions)
            {
                _genBNLayers.Add(new BatchNormalizationLayer<T>());
            }
            _usingCustomLayers = false;
        }
    }

    /// <summary>
    /// Rebuilds the default generator and discriminator layers with actual data dimensions
    /// discovered during Fit(). Only rebuilds generator layers when not using custom layers.
    /// </summary>
    /// <param name="genInputDim">Actual generator input dimension (embedding + conditional width).</param>
    /// <param name="genOutputDim">Actual generator output dimension (transformed data width).</param>
    /// <param name="discInputDim">Actual discriminator input dimension (packed data).</param>
    private void RebuildLayersWithActualDimensions(int genInputDim, int genOutputDim, int discInputDim)
    {
        if (!_usingCustomLayers)
        {
            Layers.Clear();
            Layers.AddRange(LayerHelper<T>.CreateDefaultCTABGANPlusGeneratorLayers(
                genInputDim, genOutputDim, _options.GeneratorDimensions));

            _genBNLayers.Clear();
            foreach (int dim in _options.GeneratorDimensions)
            {
                _genBNLayers.Add(new BatchNormalizationLayer<T>());
            }
        }

        // Discriminator is always rebuilt with actual dimensions
        _discLayers.Clear();
        _discLayerDims.Clear();
        _discLayers.AddRange(LayerHelper<T>.CreateDefaultCTABGANPlusDiscriminatorLayers(
            discInputDim, _options.DiscriminatorDimensions, _options.DiscriminatorDropout));

        BuildDiscriminatorDimensionMap(discInputDim);

        // Auxiliary classifier: hidden dense layers of the critic's widths (LeakyReLU applied in the forward), then
        // the class head.
        var identity = new IdentityActivation<T>() as IActivationFunction<T>;
        _classifierLayers.Clear();
        foreach (int dim in _options.DiscriminatorDimensions)
        {
            _classifierLayers.Add(new DenseLayer<T>(dim, identity));
        }
        _classifierHead = new FullyConnectedLayer<T>(_numClasses, identity);
        _wganFusedStep?.Dispose();
        _wganFusedStep = null;
    }

    /// <summary>
    /// Records each critic dense layer's (input, output) widths; the last entry marks the score head.
    /// </summary>
    private void BuildDiscriminatorDimensionMap(int inputDim)
    {
        _discLayerDims.Clear();

        int prevDim = inputDim;
        var discDims = _options.DiscriminatorDimensions;

        for (int i = 0; i < discDims.Length; i++)
        {
            _discLayerDims.Add((prevDim, discDims[i]));
            prevDim = discDims[i];
        }
        _discLayerDims.Add((prevDim, 1));
    }

    #endregion

    #region Neural Network Methods (GANDALF Pattern)

    /// <summary>
    /// Runs the generator forward pass to create synthetic data from a noise input.
    /// </summary>
    /// <param name="input">The input tensor (noise vector + conditional vector).</param>
    /// <returns>The generated synthetic data tensor.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This takes random noise and transforms it into a fake data row.
    /// The generator uses residual connections where each hidden layer receives both the
    /// previous output and the original input concatenated together.
    /// </para>
    /// </remarks>
    protected override Tensor<T> PredictCore(Tensor<T> input)
    {

        if (_usingCustomLayers)
        {
            Tensor<T> current = input;
            foreach (var layer in Layers)
            {
                current = layer.Forward(current);
            }
            return current;
        }

        return GeneratorForwardWithResidual(input);
    }

    // UpdateParameters re-sliced the flat vector across Layers by hand -- the base walks
    // exactly the same enumeration, so this said nothing the base does not already say.
    #endregion

    #region ISyntheticTabularGenerator<T> Implementation

    /// <summary>
    /// Fits the CTAB-GAN+ generator to the provided real tabular data.
    /// </summary>
    /// <param name="data">The real data matrix where each row is a sample and each column is a feature.</param>
    /// <param name="columns">Metadata describing each column (type, categories, etc.).</param>
    /// <param name="epochs">Number of training epochs. When null, the model's published Epochs from its options is used.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This is the "learning" step. The generator studies your real data:
    /// 1. Fits VGM transformer for mode-specific normalization
    /// 2. Identifies the target column for the auxiliary classifier
    /// 3. Trains generator, discriminator, and classifier in alternating WGAN-GP steps
    /// After fitting, call Generate() to create new synthetic rows.
    /// </para>
    /// </remarks>
    public void Fit(Matrix<T> data, IReadOnlyList<ColumnMetadata> columns, int? epochs = null)
    {
        int epochCount = epochs ?? _options.Epochs;
        ValidateFitInputs(data, columns, epochCount);

        var transformedData = PrepareForData(data, columns);
        int numPacks = PacksPerBatch(_options.BatchSize);
        int numBatches = Math.Max(1, data.Rows / (numPacks * _options.PacSize));

        for (int epoch = 0; epoch < epochCount; epoch++)
        {
            for (int batch = 0; batch < numBatches; batch++)
            {
                TrainBatch(transformedData, numPacks);
            }
        }

        IsFitted = true;
    }

    /// <inheritdoc />
    public async Task FitAsync(Matrix<T> data, IReadOnlyList<ColumnMetadata> columns, int? epochs = null, CancellationToken ct = default)
    {
        int epochCount = epochs ?? _options.Epochs;
        ValidateFitInputs(data, columns, epochCount);

        await Task.Run(() =>
        {
            ct.ThrowIfCancellationRequested();
            var transformedData = PrepareForData(data, columns);
            int numPacks = PacksPerBatch(_options.BatchSize);
            int numBatches = Math.Max(1, data.Rows / (numPacks * _options.PacSize));

            for (int epoch = 0; epoch < epochCount; epoch++)
            {
                ct.ThrowIfCancellationRequested();
                for (int batch = 0; batch < numBatches; batch++)
                {
                    TrainBatch(transformedData, numPacks);
                }
            }
        }, ct).ConfigureAwait(false);

        IsFitted = true;
    }

    /// <inheritdoc />
    /// <remarks>
    /// One Fit batch on <paramref name="realData"/>: the critic steps, the auxiliary classifier step, then the generator
    /// step, over as many packs as the rows fill. The first call (or one with a new column count) prepares the
    /// transformer, sampler and networks from these rows, every column continuous.
    /// </remarks>
    protected override void TrainOnBatch(Tensor<T> realData)
    {
        var rows = RowsOf(realData);
        if (_transformer is null || _columns.Count != rows.Columns)
            PrepareForData(rows, ContinuousColumns(rows.Columns));
        var transformer = _transformer ?? throw new InvalidOperationException("The data transformer was not prepared.");
        TrainBatch(transformer.Transform(rows), PacksPerBatch(rows.Rows));
        IsFitted = true;
    }

    private int PacksPerBatch(int rows) => Math.Max(1, rows / _options.PacSize);

    /// <summary>
    /// Fits the VGM transformer and the conditional sampler to <paramref name="data"/>, finds the classifier's target
    /// column and builds the networks for the transformed width; returns the transformed data.
    /// </summary>
    private Matrix<T> PrepareForData(Matrix<T> data, IReadOnlyList<ColumnMetadata> columns)
    {
        _columns = PrepareColumns(data, columns);

        // Step 1: Fit transformer
        _transformer = new TabularDataTransformer<T>(_options.VGMModes, _random);
        _transformer.Fit(data, _columns);
        _dataWidth = _transformer.TransformedWidth;
        BuildCategoricalOutputBlocks();

        // Step 2: Fit sampler
        _sampler = new CTGANDataSampler<T>(_random);
        _sampler.Fit(data, _columns);
        _condWidth = _sampler.ConditionalVectorWidth;

        // Step 3: Identify target column for auxiliary classifier
        IdentifyTargetColumn(_columns);

        // Step 4: Compute packed input dimension
        _packedInputDim = (_dataWidth + _condWidth) * _options.PacSize;

        // Step 5: Rebuild layers with actual data dimensions
        int genInputDim = _options.EmbeddingDimension + _condWidth;
        RebuildLayersWithActualDimensions(genInputDim, _dataWidth, _packedInputDim);

        // Step 6: Transform data
        return _transformer.Transform(data);
    }

    /// <summary>
    /// One batch of the paper's schedule (Zhao et al. 2023 §3): DiscriminatorSteps WGAN-GP critic steps, one step of the
    /// auxiliary classifier on real rows, then one generator step, all through GradientTape.
    /// </summary>
    private void TrainBatch(Matrix<T> transformedData, int numPacks)
    {
        for (int dStep = 0; dStep < _options.DiscriminatorSteps; dStep++)
        {
            TrainDiscriminatorStepBatched(transformedData, numPacks);
        }
        if (_hasClassifierTarget)
        {
            TrainClassifierStepBatched(transformedData, numPacks * _options.PacSize);
        }
        TrainGeneratorStepBatched(transformedData, numPacks);
    }

    /// <summary>
    /// Generates new synthetic tabular data rows.
    /// </summary>
    /// <param name="numSamples">The number of synthetic rows to generate.</param>
    /// <param name="conditionColumn">Optional conditioning column indices.</param>
    /// <param name="conditionValue">Optional conditioning values.</param>
    /// <returns>A matrix of synthetic data with the same column structure as the training data.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> After fitting, this creates new fake-but-realistic rows by:
    /// 1. Generating random noise
    /// 2. Running through the generator to create transformed data
    /// 3. Applying inverse VGM transform to restore original column distributions
    /// </para>
    /// </remarks>
    public Matrix<T> Generate(int numSamples, Vector<T>? conditionColumn = null, Vector<T>? conditionValue = null)
    {
        if (!IsFitted || _transformer is null || _sampler is null)
        {
            throw new InvalidOperationException(
                "The generator must be fitted before generating data. Call Fit() first.");
        }

        if (numSamples <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(numSamples), "Number of samples must be positive.");
        }

        var transformedRows = new Matrix<T>(numSamples, _dataWidth);

        for (int i = 0; i < numSamples; i++)
        {
            var noise = CreateStandardNormalVector(_options.EmbeddingDimension);

            Vector<T> condVector;
            if (conditionColumn is not null && conditionValue is not null && i < conditionColumn.Length)
            {
                int colIdx = (int)NumOps.ToDouble(conditionColumn[i]);
                int catVal = (int)NumOps.ToDouble(conditionValue[i]);
                condVector = _sampler.CreateConditionVector(colIdx, catVal);
            }
            else
            {
                condVector = _sampler.SampleRandomConditionVector();
            }

            var genInput = ConcatVectors(noise, condVector);
            var fakeRow = Predict(VectorToTensor(genInput));

            for (int j = 0; j < _dataWidth && j < fakeRow.Length; j++)
            {
                transformedRows[i, j] = fakeRow[j];
            }
        }

        return _transformer.InverseTransform(transformedRows);
    }

    #endregion

    #region Target Column Identification

    /// <summary>
    /// Identifies the target column for the auxiliary classifier.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The auxiliary classifier needs a "target" column to predict.
    /// This method finds it (using the user-specified index or auto-detecting the first
    /// categorical column) and computes its offset in the transformed data representation.
    /// </para>
    /// </remarks>
    private void IdentifyTargetColumn(IReadOnlyList<ColumnMetadata> columns)
    {
        int targetIdx = _options.TargetColumnIndex;

        // Auto-detect first categorical column if not specified
        if (targetIdx < 0)
        {
            for (int i = 0; i < columns.Count; i++)
            {
                if (columns[i].IsCategorical)
                {
                    targetIdx = i;
                    break;
                }
            }
        }

        // The target's one-hot block in the transformed row. Its width is the transformer's (the categories actually
        // seen), which is what the classifier predicts and the slices below read.
        if (targetIdx >= 0 && targetIdx < columns.Count && columns[targetIdx].IsCategorical && _transformer is not null
            && _transformer.GetTransformInfo(targetIdx) is { IsContinuous: false, Width: >= 2 } target)
        {
            _hasClassifierTarget = true;
            _numClasses = target.Width;
            _targetColTransformOffset = target.StartOffset;
        }
        else
        {
            _hasClassifierTarget = false;
            _numClasses = 2;
            _targetColTransformOffset = 0;
        }
    }

    #endregion

    #region GAN Training Steps

    /// <summary>
    /// WGAN-GP critic step (Gulrajani et al. 2017) over PacSize-packed samples (PacGAN, Lin et al. 2018), through the
    /// shared critic step and its single replay-safe gradient penalty.
    /// </summary>
    private void TrainDiscriminatorStepBatched(Matrix<T> transformedData, int numPacks)
    {
        if (_sampler is null) return;

        var (realPacked, fakePacked) = BuildPackedRealAndFakeBatches(transformedData, numPacks);
        AiDotNet.Training.WassersteinCriticStep<T>.Step(
            this,
            _discLayers,
            realPacked,
            fakePacked,
            input => DiscriminatorForwardBatched(input, isTraining: true).Score,
            _options.GradientPenaltyWeight,
            _discriminatorOptimizer,
            _random,
            ref _wganFusedStep);
    }

    /// <summary>
    /// Auxiliary classifier step (Zhao et al. 2021 §3.1): cross-entropy of the classifier's prediction of the target
    /// column from the rest of each real row.
    /// </summary>
    private void TrainClassifierStepBatched(Matrix<T> transformedData, int batchSize)
    {
        var head = _classifierHead;
        if (head is null) return;

        var real = new Tensor<T>([batchSize, _dataWidth]);
        for (int b = 0; b < batchSize; b++)
        {
            int row = _random.Next(transformedData.Rows);
            for (int j = 0; j < _dataWidth; j++) real[b, j] = transformedData[row, j];
        }
        var labels = Engine.TensorSlice(real, [0, _targetColTransformOffset], [batchSize, _numClasses]);

        var classifierLayers = new List<ILayer<T>>(_classifierLayers) { head };
        var trainable = classifierLayers.OfType<ITrainableLayer<T>>().ToList();
        AiDotNet.Training.FusedTrainingStep<T>.Step(
            head, trainable, real, labels,
            forward: ClassifierForwardBatched,
            computeLoss: ClassifierCrossEntropy,
            optimizer: _classifierOptimizer,
            extraTensors: TapeTrainingStep<T>.CollectParameters(classifierLayers));
    }

    /// <summary>
    /// Generator step (Zhao et al. 2023 §3): the WGAN loss -E[D(G(z, c))], plus the conditional loss that makes the
    /// generator honour its condition (Xu et al. 2019 §4.3), ClassifierWeight times the classifier's cross-entropy
    /// against each generated row's own target, and InformationWeight times the information loss. The critic and
    /// classifier score the samples in evaluation mode and are not in the parameter group, so only the generator moves.
    /// </summary>
    private void TrainGeneratorStepBatched(Matrix<T> transformedData, int numPacks)
    {
        if (_sampler is null) return;

        var generatorLayers = new List<ILayer<T>>();
        generatorLayers.AddRange(Layers);
        foreach (var bn in _genBNLayers) generatorLayers.Add(bn);
        var trainableGenLayers = generatorLayers.OfType<ITrainableLayer<T>>().ToList();
        if (trainableGenLayers.Count == 0) return;

        int totalSamples = numPacks * _options.PacSize;
        int embedDim = _options.EmbeddingDimension;
        int condDim = _condWidth;
        var noiseBatch = GenerateNoiseBatchTensor(totalSamples);
        var (condBatch, maskBatch) = SampleCondMaskBatch(totalSamples);
        // The step's input carries [noise | cond | mask] side by side so a replayed plan sees fresh samples.
        var fusedInput = Engine.TensorConcatenate([noiseBatch, condBatch, maskBatch], axis: 1);

        // The real batch the information loss compares against: packed rows with their own conditions.
        var (realPacked, _) = BuildPackedRealAndFakeBatches(transformedData, numPacks, includeFake: false);

        // Fwd's intermediates, reused by Loss (the fused-step contract runs Fwd before Loss).
        Tensor<T>? capturedAct = null;
        Tensor<T>? capturedCond = null;
        Tensor<T>? capturedMask = null;
        Tensor<T>? capturedFakeFeatures = null;
        Tensor<T> Fwd(Tensor<T> packed)
        {
            var gi = Engine.TensorSlice(packed, [0, 0], [totalSamples, embedDim + condDim]);
            var act = ApplyOutputActivationsBatched(GeneratorForwardWithResidualBatched(gi));
            var cond = Engine.TensorSlice(packed, [0, embedDim], [totalSamples, condDim]);
            capturedAct = act;
            capturedCond = cond;
            capturedMask = Engine.TensorSlice(packed, [0, embedDim + condDim], [totalSamples, condDim]);
            var withCond = condDim > 0 ? Engine.TensorConcatenate([act, cond], axis: 1) : act;
            var (score, features) = DiscriminatorForwardBatched(withCond.Reshape([numPacks, _packedInputDim]), false);
            capturedFakeFeatures = features;
            return score;
        }
        Tensor<T> Loss(Tensor<T> scores, Tensor<T> _)
        {
            var act = capturedAct;
            var cond = capturedCond;
            var mask = capturedMask;
            var fakeFeatures = capturedFakeFeatures;
            if (act is null || cond is null || mask is null || fakeFeatures is null)
                throw new InvalidOperationException(
                    "CTAB-GAN+ generator step: the forward's intermediates were not captured before the loss ran.");

            var axes = Enumerable.Range(0, scores.Shape.Length).ToArray();
            var loss = Engine.TensorNegate(Engine.ReduceMean(scores, axes, keepDims: false));
            if (condDim > 0 && _catOutputBlocks.Count > 0)
            {
                loss = Engine.TensorAdd(loss, ConditionalCrossEntropy(act, cond, mask));
            }
            if (_hasClassifierTarget && _options.ClassifierWeight > 0)
            {
                // Each generated row's own target block is its label: a constant, so the gradient reaches the
                // generator through the classifier's view of the rest of the row.
                var labels = Engine.TensorSlice(act, [0, _targetColTransformOffset], [totalSamples, _numClasses]);
                var ce = ClassifierCrossEntropy(ClassifierForwardBatched(act), new Tensor<T>(labels.ToArray(), labels._shape));
                loss = Engine.TensorAdd(loss, Engine.TensorMultiplyScalar(ce, NumOps.FromDouble(_options.ClassifierWeight)));
            }
            if (_options.InformationWeight > 0)
            {
                var realFeatures = DiscriminatorForwardBatched(realPacked, false).Features;
                var info = InformationLoss(new Tensor<T>(realFeatures.ToArray(), realFeatures._shape), fakeFeatures);
                loss = Engine.TensorAdd(loss, Engine.TensorMultiplyScalar(info, NumOps.FromDouble(_options.InformationWeight)));
            }
            return loss;
        }

        AiDotNet.Training.FusedTrainingStep<T>.Step(
            this, trainableGenLayers, fusedInput, new Tensor<T>(new[] { 1 }),
            forward: Fwd,
            computeLoss: Loss,
            optimizer: _optimizer,
            extraTensors: TapeTrainingStep<T>.CollectParameters(generatorLayers));
    }

    /// <summary>
    /// Information loss (Zhao et al. 2021 §3.1): the L2 distances between the batch mean and between the batch standard
    /// deviation of the critic's last hidden features on real and on generated packs.
    /// </summary>
    private Tensor<T> InformationLoss(Tensor<T> realFeatures, Tensor<T> fakeFeatures)
    {
        var eps = NumOps.FromDouble(1e-8);
        Tensor<T> L2(Tensor<T> a, Tensor<T> b)
            => Engine.TensorSqrt(Engine.TensorAddScalar(
                Engine.ReduceSum(Engine.TensorSquare(Engine.TensorSubtract(a, b)), [0], keepDims: false), eps));
        Tensor<T> Std(Tensor<T> f)
            => Engine.TensorSqrt(Engine.TensorAddScalar(Engine.ReduceVariance(f, [0], keepDims: false), eps));

        var meanTerm = L2(Engine.ReduceMean(realFeatures, [0], keepDims: false), Engine.ReduceMean(fakeFeatures, [0], keepDims: false));
        var stdTerm = L2(Std(realFeatures), Std(fakeFeatures));
        return Engine.TensorAdd(meanTerm, stdTerm);
    }

    /// <summary>
    /// The mean softmax cross-entropy of <paramref name="logits"/> [N, classes] against one-hot (or soft)
    /// <paramref name="labels"/> [N, classes].
    /// </summary>
    private Tensor<T> ClassifierCrossEntropy(Tensor<T> logits, Tensor<T> labels)
    {
        var logp = Engine.TensorLog(Engine.TensorAddScalar(Engine.Softmax(logits, axis: 1), NumOps.FromDouble(1e-8)));
        var perRow = Engine.ReduceSum(Engine.TensorMultiply(labels, logp), [1], keepDims: false);
        return Engine.TensorNegate(Engine.ReduceMean(perRow, [0], keepDims: false));
    }

    /// <summary>
    /// The auxiliary classifier's logits [N, classes] for transformed rows [N, dataWidth], from every block but the
    /// target column's: the hidden layers with LeakyReLU (slope 0.2), then the class head.
    /// </summary>
    private Tensor<T> ClassifierForwardBatched(Tensor<T> rows)
    {
        int n = rows.Shape[0];
        int width = rows.Shape[1];
        int end = _targetColTransformOffset + _numClasses;
        var parts = new List<Tensor<T>>(2);
        if (_targetColTransformOffset > 0)
            parts.Add(Engine.TensorSlice(rows, [0, 0], [n, _targetColTransformOffset]));
        if (end < width)
            parts.Add(Engine.TensorSlice(rows, [0, end], [n, width - end]));
        var current = parts.Count switch
        {
            0 => new Tensor<T>([n, 1]),
            1 => parts[0],
            _ => Engine.TensorConcatenate(parts.ToArray(), axis: 1),
        };

        T leakySlope = NumOps.FromDouble(0.2);
        foreach (var layer in _classifierLayers)
        {
            current = Engine.LeakyReLU(layer.Forward(current), leakySlope);
        }
        var head = _classifierHead ?? throw new InvalidOperationException("The auxiliary classifier was not built.");
        return head.Forward(current);
    }

    /// <summary>
    /// Records each categorical column's (offset, width) in the transformed row, in the conditional vector's order.
    /// </summary>
    private void BuildCategoricalOutputBlocks()
    {
        _catOutputBlocks.Clear();
        if (_transformer is null) return;
        for (int col = 0; col < _columns.Count; col++)
        {
            var info = _transformer.GetTransformInfo(col);
            if (!info.IsContinuous)
                _catOutputBlocks.Add((info.StartOffset, info.Width));
        }
    }

    /// <summary>
    /// Samples conditional vectors with their masks (Xu et al. 2019 §4.3): [N, condWidth] one-hot conditions and
    /// [N, condWidth] masks that are 1 over the selected column's block.
    /// </summary>
    private (Tensor<T> Cond, Tensor<T> Mask) SampleCondMaskBatch(int batchSize)
    {
        var cond = new Tensor<T>([batchSize, _condWidth]);
        var mask = new Tensor<T>([batchSize, _condWidth]);
        if (_sampler is null || _condWidth == 0) return (cond, mask);
        for (int b = 0; b < batchSize; b++)
        {
            var (cv, mv, _, _, _) = _sampler.SampleCondVecWithMask();
            for (int j = 0; j < _condWidth; j++) { cond[b, j] = cv[j]; mask[b, j] = mv[j]; }
        }
        return (cond, mask);
    }

    /// <summary>
    /// Conditional loss (Xu et al. 2019 §4.3): the cross-entropy between each condition and the generator's softmax
    /// over the conditioned column, masked to that column.
    /// </summary>
    private Tensor<T> ConditionalCrossEntropy(Tensor<T> fakeActivated, Tensor<T> condBatch, Tensor<T> maskBatch)
    {
        int batch = fakeActivated.Shape[0];
        var blocks = new List<Tensor<T>>(_catOutputBlocks.Count);
        foreach (var (offset, width) in _catOutputBlocks)
            blocks.Add(Engine.TensorSlice(fakeActivated, [0, offset], [batch, width]));
        var alignedGen = blocks.Count == 1 ? blocks[0] : Engine.TensorConcatenate(blocks.ToArray(), axis: 1);

        var logp = Engine.TensorLog(Engine.TensorAddScalar(alignedGen, NumOps.FromDouble(1e-8)));
        var masked = Engine.TensorMultiply(Engine.TensorMultiply(maskBatch, condBatch), logp);
        var perRow = Engine.ReduceSum(masked, [1], keepDims: false);
        return Engine.TensorNegate(Engine.ReduceMean(perRow, [0], keepDims: false));
    }

    /// <summary>
    /// The packed real batch [numPacks, packedInputDim] (sampled rows with the conditions that drew them) and, unless
    /// <paramref name="includeFake"/> is false, the packed generated batch for the same conditions.
    /// </summary>
    private (Tensor<T> RealPacked, Tensor<T> FakePacked) BuildPackedRealAndFakeBatches(
        Matrix<T> transformedData, int numPacks, bool includeFake = true)
    {
        int totalSamples = numPacks * _options.PacSize;
        var (condBatch, rowIndices) = SampleConditionalBatchWithRows(transformedData.Rows, totalSamples);

        var realFlat = new Tensor<T>([totalSamples, _dataWidth]);
        int cols = Math.Min(_dataWidth, transformedData.Columns);
        for (int s = 0; s < totalSamples; s++)
        {
            for (int j = 0; j < cols; j++) realFlat[s, j] = transformedData[rowIndices[s], j];
        }
        var realSingles = _condWidth > 0 ? Engine.TensorConcatenate([realFlat, condBatch], axis: 1) : realFlat;
        var realPacked = realSingles.Reshape([numPacks, _packedInputDim]);
        if (!includeFake) return (realPacked, realPacked);

        var genInput = Engine.TensorConcatenate([GenerateNoiseBatchTensor(totalSamples), condBatch], axis: 1);
        var fakeActivated = ApplyOutputActivationsBatched(GeneratorForwardWithResidualBatched(genInput));
        var fakeSingles = _condWidth > 0 ? Engine.TensorConcatenate([fakeActivated, condBatch], axis: 1) : fakeActivated;
        return (realPacked, fakeSingles.Reshape([numPacks, _packedInputDim]));
    }

    /// <summary>
    /// Samples a condition per sample by training-by-sampling (Xu et al. 2019 §4.3) with the row it drew.
    /// </summary>
    private (Tensor<T> CondBatch, int[] RowIndices) SampleConditionalBatchWithRows(int dataRows, int batchSize)
    {
        var batch = new Tensor<T>([batchSize, _condWidth]);
        var rowIndices = new int[batchSize];
        if (_sampler is null) return (batch, rowIndices);

        for (int b = 0; b < batchSize; b++)
        {
            var (cv, rowIndex) = _sampler.SampleConditionAndRow();
            int cols = Math.Min(_condWidth, cv.Length);
            for (int j = 0; j < cols; j++) batch[b, j] = cv[j];
            rowIndices[b] = rowIndex >= 0 && rowIndex < dataRows ? rowIndex : _random.Next(dataRows);
        }

        return (batch, rowIndices);
    }

    /// <summary>
    /// [batchSize, EmbeddingDimension] standard-normal noise.
    /// </summary>
    private Tensor<T> GenerateNoiseBatchTensor(int batchSize)
    {
        var noise = new Tensor<T>([batchSize, _options.EmbeddingDimension]);
        for (int i = 0; i < noise.Length; i++) noise[i] = SampleStandardNormal();
        return noise;
    }

    /// <summary>
    /// Batched, tape-tracked <see cref="GeneratorForwardWithResidual"/>: each hidden layer sees the previous output
    /// concatenated with the input, then batch normalisation and ReLU.
    /// </summary>
    private Tensor<T> GeneratorForwardWithResidualBatched(Tensor<T> input)
    {
        if (_usingCustomLayers)
        {
            var c = input;
            foreach (var l in Layers) c = l.Forward(c);
            return c;
        }

        var h = input;
        for (int i = 0; i < Layers.Count - 1; i++)
        {
            if (i > 0) h = Engine.TensorConcatenate([h, input], axis: 1);
            h = Layers[i].Forward(h);
            if (i < _genBNLayers.Count) h = _genBNLayers[i].Forward(h);
            h = Engine.ReLU(h);
        }
        h = Engine.TensorConcatenate([h, input], axis: 1);
        return Layers[^1].Forward(h);
    }

    /// <summary>
    /// Batched, tape-tracked <see cref="ApplyOutputActivations"/>: tanh on each continuous value, softmax over each mode
    /// and each categorical block.
    /// </summary>
    private Tensor<T> ApplyOutputActivationsBatched(Tensor<T> output)
    {
        if (_transformer is null) return Engine.TensorTanh(output);

        int batch = output.Shape[0];
        int totalWidth = output.Shape[1];
        var blocks = new List<Tensor<T>>(_columns.Count * 2);
        int idx = 0;
        for (int col = 0; col < _columns.Count && idx < totalWidth; col++)
        {
            var transform = _transformer.GetTransformInfo(col);
            if (transform.IsContinuous)
            {
                blocks.Add(Engine.TensorTanh(Engine.TensorSlice(output, [0, idx], [batch, 1])));
                idx++;
                int modeLength = Math.Min(transform.Width - 1, totalWidth - idx);
                if (modeLength > 0)
                {
                    blocks.Add(Engine.Softmax(Engine.TensorSlice(output, [0, idx], [batch, modeLength]), axis: 1));
                    idx += modeLength;
                }
            }
            else
            {
                int blockLength = Math.Min(transform.Width, totalWidth - idx);
                blocks.Add(Engine.Softmax(Engine.TensorSlice(output, [0, idx], [batch, blockLength]), axis: 1));
                idx += blockLength;
            }
        }
        if (idx < totalWidth)
        {
            blocks.Add(Engine.TensorTanh(Engine.TensorSlice(output, [0, idx], [batch, totalWidth - idx])));
        }
        return blocks.Count == 1 ? blocks[0] : Engine.TensorConcatenate(blocks.ToArray(), axis: 1);
    }

    #endregion

    #region Discriminator Forward/Backward

    /// <summary>
    /// Batched, tape-tracked critic forward over packed rows: dense layers with LeakyReLU (slope 0.2), dropout while
    /// training. Returns the score [numPacks, 1] and the last hidden activation the information loss compares.
    /// </summary>
    private (Tensor<T> Score, Tensor<T> Features) DiscriminatorForwardBatched(Tensor<T> input, bool isTraining)
    {
        var current = input;
        var features = input;
        T leakySlope = NumOps.FromDouble(0.2);
        int layerIdx = 0;

        for (int i = 0; i < _discLayers.Count; i++)
        {
            if (_discLayers[i] is DropoutLayer<T> dropout)
            {
                if (isTraining) current = dropout.Forward(current);
                continue;
            }

            bool isLastDense = layerIdx == _discLayerDims.Count - 1;
            if (isLastDense) features = current;
            current = _discLayers[i].Forward(current);
            if (!isLastDense) current = Engine.LeakyReLU(current, leakySlope);
            layerIdx++;
        }

        return (current, features);
    }

    #endregion

    #region Generator Forward/Backward with Residual Connections

    /// <summary>
    /// Generator forward pass with CTGAN-style residual connections:
    /// Each hidden layer receives [previous_output, original_input].
    /// </summary>
    private Tensor<T> GeneratorForwardWithResidual(Tensor<T> input)
    {
        _genPreActivations.Clear();
        var inputTensor = input;
        var current = input;

        for (int i = 0; i < Layers.Count - 1; i++)
        {
            if (i > 0)
            {
                current = ConcatTensors(current, inputTensor);
            }

            current = Layers[i].Forward(current);

            if (i < _genBNLayers.Count)
            {
                current = _genBNLayers[i].Forward(current);
            }

            _genPreActivations.Add(CloneTensor(current));
            current = ApplyReLU(current);
        }

        current = ConcatTensors(current, inputTensor);
        current = Layers[^1].Forward(current);

        return ApplyOutputActivations(current);
    }

    #endregion

    #region Activation Functions

    private Tensor<T> ApplyReLU(Tensor<T> input)
    {
        var result = new Tensor<T>(input._shape);
        for (int i = 0; i < input.Length; i++)
        {
            result[i] = NumOps.GreaterThan(input[i], NumOps.Zero) ? input[i] : NumOps.Zero;
        }
        return result;
    }

    /// <summary>
    /// Applies tanh to continuous columns and softmax to categorical/mode columns.
    /// </summary>
    private Tensor<T> ApplyOutputActivations(Tensor<T> output)
    {
        if (_transformer is null) return output;

        var result = new Tensor<T>(output._shape);
        int idx = 0;

        for (int col = 0; col < _columns.Count && idx < output.Length; col++)
        {
            var transform = _transformer.GetTransformInfo(col);

            if (transform.IsContinuous)
            {
                if (idx < output.Length)
                {
                    double val = NumOps.ToDouble(output[idx]);
                    result[idx] = NumOps.FromDouble(Math.Tanh(val));
                    idx++;
                }

                int numModes = transform.Width - 1;
                if (numModes > 0)
                {
                    ApplySoftmaxBlock(output, result, ref idx, numModes);
                }
            }
            else
            {
                ApplySoftmaxBlock(output, result, ref idx, transform.Width);
            }
        }

        return result;
    }

    private void ApplySoftmaxBlock(Tensor<T> input, Tensor<T> output, ref int idx, int count)
    {
        if (count <= 0) return;
        int actualCount = Math.Min(count, input.Length - idx);
        if (actualCount <= 0) return;
        var slice = new Tensor<T>([actualCount]);
        input.Data.Span.Slice(idx, actualCount).CopyTo(slice.Data.Span);
        var result = Engine.Softmax(slice, -1);
        result.Data.Span.CopyTo(output.Data.Span.Slice(idx, actualCount));
        idx += actualCount;
    }

    #endregion

    #region Serialization and Model Metadata (GANDALF Pattern)

    /// <inheritdoc/>
    public override ModelMetadata<T> GetModelMetadata()
    {
        return new ModelMetadata<T>
        {
            AdditionalInfo = new Dictionary<string, object>
            {
                { "EmbeddingDimension", _options.EmbeddingDimension },
                { "GeneratorDimensions", _options.GeneratorDimensions },
                { "DiscriminatorDimensions", _options.DiscriminatorDimensions },
                { "BatchSize", _options.BatchSize },
                { "PacSize", _options.PacSize },
                { "GradientPenaltyWeight", _options.GradientPenaltyWeight },
                { "ClassifierWeight", _options.ClassifierWeight },
                { "InformationWeight", _options.InformationWeight },
                { "NumClasses", _numClasses },
                { "GeneratorLayerCount", Layers.Count },
                { "GeneratorLayerTypes", Layers.Select(l => l.GetType().Name).ToArray() }
            },
            ModelDataProvider = () => SerializeForMetadata()
        };
    }

    /// <inheritdoc/>


    /// <inheritdoc/>


    /// <inheritdoc/>
    public override Dictionary<string, T> GetFeatureImportance()
    {
        var importance = new Dictionary<string, T>();
        int numFeatures = Architecture.CalculatedInputSize;
        var uniformValue = NumOps.FromDouble(1.0 / Math.Max(numFeatures, 1));
        for (int f = 0; f < numFeatures; f++)
        {
            importance[$"feature_{f}"] = uniformValue;
        }
        return importance;
    }

    #endregion

    #region Input Validation and Column Management

    private static void ValidateFitInputs(Matrix<T> data, IReadOnlyList<ColumnMetadata> columns, int epochs)
    {
        if (data.Rows == 0 || data.Columns == 0)
        {
            throw new ArgumentException("Data matrix must not be empty.", nameof(data));
        }

        if (columns.Count == 0)
        {
            throw new ArgumentException("Column metadata list must not be empty.", nameof(columns));
        }

        if (columns.Count != data.Columns)
        {
            throw new ArgumentException(
                $"Column metadata count ({columns.Count}) must match data column count ({data.Columns}).",
                nameof(columns));
        }

        if (epochs <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(epochs), "Epochs must be positive.");
        }
    }

    private List<ColumnMetadata> PrepareColumns(Matrix<T> data, IReadOnlyList<ColumnMetadata> columns)
    {
        var prepared = new List<ColumnMetadata>(columns.Count);

        for (int col = 0; col < columns.Count; col++)
        {
            var meta = columns[col].Clone();
            meta.ColumnIndex = col;

            if (meta.IsNumerical)
            {
                ComputeColumnStatistics(data, col, meta);
            }
            else if (meta.IsCategorical && meta.NumCategories == 0)
            {
                var categories = new HashSet<string>();
                for (int row = 0; row < data.Rows; row++)
                {
                    var val = NumOps.ToDouble(data[row, col]);
                    categories.Add(val.ToString(System.Globalization.CultureInfo.InvariantCulture));
                }
                meta.Categories = categories.OrderBy(c => c, StringComparer.Ordinal).ToList().AsReadOnly();
            }

            prepared.Add(meta);
        }

        return prepared;
    }

    private void ComputeColumnStatistics(Matrix<T> data, int colIndex, ColumnMetadata meta)
    {
        int n = data.Rows;
        double sum = 0;
        double min = double.MaxValue;
        double max = double.MinValue;

        for (int row = 0; row < n; row++)
        {
            double val = NumOps.ToDouble(data[row, colIndex]);
            sum += val;
            if (val < min) min = val;
            if (val > max) max = val;
        }

        double mean = sum / n;
        double sumSqDiff = 0;
        for (int row = 0; row < n; row++)
        {
            double val = NumOps.ToDouble(data[row, colIndex]);
            double diff = val - mean;
            sumSqDiff += diff * diff;
        }

        double std = n > 1 ? Math.Sqrt(sumSqDiff / (n - 1)) : 1.0;
        if (std < 1e-10) std = 1e-10;

        meta.Min = min;
        meta.Max = max;
        meta.Mean = mean;
        meta.Std = std;
    }

    #endregion

    #region Random Sampling Utilities

    private T SampleStandardNormal()
    {
        double u1 = 1.0 - _random.NextDouble();
        double u2 = _random.NextDouble();
        double z = Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2);
        return NumOps.FromDouble(z);
    }

    private Vector<T> CreateStandardNormalVector(int length)
    {
        var v = new Vector<T>(length);
        for (int i = 0; i < length; i++)
        {
            v[i] = SampleStandardNormal();
        }
        return v;
    }

    #endregion

    #region Helpers

    private static Vector<T> GetRow(Matrix<T> matrix, int row)
    {
        var v = new Vector<T>(matrix.Columns);
        for (int j = 0; j < matrix.Columns; j++) v[j] = matrix[row, j];
        return v;
    }

    private static Vector<T> ConcatVectors(Vector<T> a, Vector<T> b)
    {
        var result = new Vector<T>(a.Length + b.Length);
        for (int i = 0; i < a.Length; i++) result[i] = a[i];
        for (int i = 0; i < b.Length; i++) result[a.Length + i] = b[i];
        return result;
    }

    private static Tensor<T> VectorToTensor(Vector<T> v)
    {
        var t = new Tensor<T>([v.Length]);
        for (int i = 0; i < v.Length; i++) t[i] = v[i];
        return t;
    }

    private static Tensor<T> ConcatTensors(Tensor<T> a, Tensor<T> b)
    {
        var result = new Tensor<T>([a.Length + b.Length]);
        for (int i = 0; i < a.Length; i++) result[i] = a[i];
        for (int i = 0; i < b.Length; i++) result[a.Length + i] = b[i];
        return result;
    }

    private static Tensor<T> CloneTensor(Tensor<T> source)
    {
        var clone = new Tensor<T>(source._shape);
        for (int i = 0; i < source.Length; i++) clone[i] = source[i];
        return clone;
    }

    #endregion

}
