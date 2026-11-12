using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks.SyntheticData;
using AiDotNet.Optimizers;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Training;

namespace AiDotNet.NeuralNetworks.SyntheticData;

/// <summary>
/// CopulaGAN generator for synthetic tabular data, combining Gaussian copula transformations
/// with the CTGAN training pipeline for improved continuous column modeling.
/// </summary>
/// <remarks>
/// <para>
/// CopulaGAN applies a two-stage preprocessing to continuous columns:
/// 1. <b>Gaussian copula transform</b>: CDF then quantile then standard normal
/// 2. <b>VGM normalization</b>: Standard CTGAN mode-specific normalization
///
/// This implementation follows the standard neural network architecture pattern with:
/// - Proper inheritance from NeuralNetworkBase
/// - Layer-based architecture using ILayer components
/// - Engine-based tensor operations for CPU/GPU acceleration
/// - Full autodiff and JIT compilation support
/// </para>
/// <para>
/// <b>For Beginners:</b> CopulaGAN is CTGAN with an extra "normalizing" step:
///
/// <b>Training:</b>
/// <code>
/// Raw data -> [Copula: reshape to bell curve] -> [VGM: CTGAN preprocessing] -> [CTGAN training]
/// </code>
///
/// <b>Generation:</b>
/// <code>
/// Random noise -> [Generator] -> [Inverse VGM] -> [Inverse Copula] -> Synthetic data
/// </code>
///
/// If you provide custom layers in the architecture, those will be used directly
/// for the generator network. If not, the network creates industry-standard
/// CopulaGAN layers based on the original research paper specifications.
///
/// Example usage:
/// <code>
/// var architecture = new NeuralNetworkArchitecture&lt;double&gt;(
///     inputFeatures: 10,
///     outputSize: 10
/// );
/// var options = new CopulaGANOptions&lt;double&gt;
/// {
///     EmbeddingDimension = 128,
///     GeneratorDimensions = new[] { 256, 256 },
///     BatchSize = 500,
///     Epochs = 300
/// };
/// var generator = new CopulaGANGenerator&lt;double&gt;(architecture, options);
/// generator.Fit(data, columns, epochs: 300);
/// var synthetic = generator.Generate(1000);
/// </code>
/// </para>
/// <para>
/// Reference: "Synthesizing Tabular Data using Copulas" (2020)
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
[ModelDomain(ModelDomain.MachineLearning)]
[ModelCategory(ModelCategory.NeuralNetwork)]
[ModelCategory(ModelCategory.SyntheticDataGenerator)]
[ModelTask(ModelTask.Generation)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("Modeling Tabular Data using Conditional GAN",
    "https://arxiv.org/abs/1907.00503",
    Year = 2019,
    Authors = "Lei Xu, Maria Skoularidou, Alfredo Cuesta-Infante, Kalyan Veeramachaneni")]
public partial class CopulaGANGenerator<T> : NeuralSyntheticTabularGeneratorBase<T>, ISyntheticTabularGenerator<T>
{
    // Held across Train calls. The fused plan owns the optimizer state, so a step object built per call
    // re-traced and recompiled every call and restarted the optimizer at t = 1 (Adam ~ lr*sign(g)).
    private AiDotNet.Training.WganGpFusedStep<T>? _wganFusedStep;

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

    private readonly CopulaGANOptions<T> _options;
    // Separate generator/discriminator optimizers — see CTGANGenerator for the
    // full rationale: AdamOptimizer keeps one flat (_m,_v) moment buffer sized to
    // the parameter count and resets the timestep when that count changes, so a
    // single optimizer shared across G and D (different param counts) wiped the
    // moments every alternating Step and diverged worse with more epochs.
    private IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> _generatorOptimizer;
    private IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> _discriminatorOptimizer;
    private ILossFunction<T> _lossFunction;

    // Synthetic tabular data infrastructure
    private TabularDataTransformer<T>? _transformer;
    private CTGANDataSampler<T>? _sampler;
    private List<ColumnMetadata> _columns = new();
    private int _dataWidth;
    private int _condWidth;
    private int _packedInputDim;
    private Random _random;

    // Generator batch normalization layers (auxiliary, always created to match Layers)
    private readonly List<BatchNormalizationLayer<T>> _genBNLayers = new();

    // Discriminator layers (auxiliary, not user-overridable)
    private readonly List<ILayer<T>> _discLayers = new();
    private readonly List<DropoutLayer<T>> _discDropoutLayers = new();
    private readonly List<(int InputSize, int OutputSize)> _discLayerDims = new();

    // Copula transform state: sorted values and empirical CDF per continuous column
    private readonly List<T[]> _sortedValues = new();
    private readonly List<int> _continuousColumnIndices = new();

    // Cached pre-activations for proper backward passes
    [Scratch]
    private Matrix<T>? _realDataSource;
    [Scratch]
    private Tensor<T>? _realDataTensor;

    // Whether custom layers are being used (disables residual connection logic)
    private bool _usingCustomLayers;

    // Pre-allocated training buffers to eliminate per-row GC pressure
    [Scratch]
    private Tensor<T>? _oneGrad;
    [Scratch]
    private Tensor<T>? _negOneGrad;
    [Scratch]
    private Vector<T>? _packedRealBuf;
    [Scratch]
    private Vector<T>? _packedFakeBuf;
    [Scratch]
    private Vector<T>? _noiseBuf;
    [Scratch]
    private Vector<T>? _genInputBuf;
    [Scratch]
    private Vector<T>? _realSingleBuf;
    [Scratch]
    private Vector<T>? _fakeSingleBuf;
    [Scratch]
    private Vector<T>? _realRowBuf;
    [Scratch]
    private Vector<T>? _fakeRowBuf;
    [Scratch]
    private Vector<T>? _interpolatedBuf;
    [Scratch]
    private Tensor<T>? _sampleGradBuf;

    /// <summary>
    /// Gets the CopulaGAN-specific options.
    /// </summary>
    public new CopulaGANOptions<T> Options => _options;

    /// <inheritdoc />
    public IReadOnlyList<ColumnMetadata> Columns => _columns.AsReadOnly();

    /// <inheritdoc />
    public bool IsFitted { get; private set; }

    /// <summary>
    /// Initializes a new CopulaGAN generator with the specified architecture.
    /// </summary>
    /// <param name="architecture">The neural network architecture defining input/output dimensions and optional custom layers.</param>
    /// <param name="options">CopulaGAN-specific options for generator and discriminator configuration.</param>
    /// <param name="optimizer">Gradient-based optimizer (defaults to Adam).</param>
    /// <param name="lossFunction">Loss function (defaults based on task type).</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This constructor creates a CopulaGAN network based on the architecture you provide.
    ///
    /// If you provide custom layers in the architecture, those will be used directly
    /// for the generator network. If not, the network will create industry-standard
    /// CopulaGAN generator layers based on the original research paper specifications.
    ///
    /// Example usage:
    /// <code>
    /// var architecture = new NeuralNetworkArchitecture&lt;double&gt;(
    ///     inputFeatures: 10,
    ///     outputSize: 10
    /// );
    /// var options = new CopulaGANOptions&lt;double&gt; { EmbeddingDimension = 128 };
    /// var generator = new CopulaGANGenerator&lt;double&gt;(architecture, options);
    /// </code>
    /// </para>
    /// </remarks>
    /// <summary>
    /// Initializes a new instance with default architecture settings.
    /// </summary>
    public CopulaGANGenerator()
        : this(new NeuralNetworkArchitecture<T>(
            inputType: Enums.InputType.OneDimensional,
            taskType: Enums.NeuralNetworkTaskType.Regression,
            inputSize: 10,
            outputSize: 10))
    {
    }

    public CopulaGANGenerator(NeuralNetworkArchitecture<T> architecture,
        CopulaGANOptions<T>? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
        ILossFunction<T>? lossFunction = null)
        : base(architecture, lossFunction ?? NeuralNetworkHelper<T>.GetDefaultLossFunction(architecture.TaskType), (options ??= new CopulaGANOptions<T>()).MaxGradNorm)
    {
        _options = options;
        base.Options = _options;
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
        _random = _options.Seed.HasValue
            ? RandomHelper.CreateSeededRandom(_options.Seed.Value)
            : RandomHelper.CreateSecureRandom();

        InitializeLayers();
    }

    #region Layer Initialization (GANDALF Pattern)

    /// <summary>
    /// Initializes the layers of the CopulaGAN network based on the provided architecture.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This method either uses custom layers provided in the architecture or creates
    /// default CopulaGAN generator layers following the original paper specifications.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> This method sets up the generator network structure:
    /// - If you provided custom layers, those are used for the generator
    /// - Otherwise, it creates the standard CopulaGAN generator architecture:
    ///   Dense layers with batch normalization and residual connections
    ///
    /// The discriminator is always created internally and is not user-overridable.
    /// </para>
    /// </remarks>
    protected override void InitializeLayers()
    {
        if (Architecture.Layers != null && Architecture.Layers.Count > 0)
        {
            // Use the layers provided by the user for the generator
            Layers.AddRange(Architecture.Layers);
            ValidateCustomLayers(Layers);
            _usingCustomLayers = true;
        }
        else
        {
            // Create default generator layers
            // Input = EmbeddingDimension (noise) + estimated conditional width
            int inputDim = _options.EmbeddingDimension + Architecture.CalculatedInputSize;
            int outputDim = Architecture.OutputSize;
            Layers.AddRange(LayerHelper<T>.CreateDefaultCopulaGANGeneratorLayers(
                inputDim, outputDim, _options.GeneratorDimensions));

            // Create matching batch normalization layers (one per hidden layer)
            _genBNLayers.Clear();
            foreach (int dim in _options.GeneratorDimensions)
            {
                _genBNLayers.Add(new BatchNormalizationLayer<T>(dim));
            }
            _usingCustomLayers = false;
        }
    }

    /// <summary>
    /// Rebuilds the default generator and discriminator layers with actual data dimensions
    /// discovered during Fit(). Only called when not using custom layers.
    /// </summary>
    /// <param name="genInputDim">Actual generator input dimension (embedding + conditional width).</param>
    /// <param name="genOutputDim">Actual generator output dimension (transformed data width).</param>
    /// <param name="discInputDim">Actual discriminator input dimension (packed data).</param>
    private void RebuildLayersWithActualDimensions(int genInputDim, int genOutputDim, int discInputDim)
    {
        if (!_usingCustomLayers)
        {
            Layers.Clear();
            Layers.AddRange(LayerHelper<T>.CreateDefaultCopulaGANGeneratorLayers(
                genInputDim, genOutputDim, _options.GeneratorDimensions));

            _genBNLayers.Clear();
            foreach (int dim in _options.GeneratorDimensions)
            {
                _genBNLayers.Add(new BatchNormalizationLayer<T>(dim));
            }
        }

        // Discriminator is always rebuilt with actual dimensions
        _discLayers.Clear();
        _discDropoutLayers.Clear();
        _discLayerDims.Clear();
        _discLayers.AddRange(LayerHelper<T>.CreateDefaultCopulaGANDiscriminatorLayers(
            discInputDim, _options.DiscriminatorDimensions, _options.DiscriminatorDropout));

        // Build dimension map for manual gradient computation
        BuildDiscriminatorDimensionMap(discInputDim);
    }

    /// <summary>
    /// Builds a dimension map for the discriminator layers to support manual backward pass.
    /// </summary>
    private void BuildDiscriminatorDimensionMap(int inputDim)
    {
        _discLayerDims.Clear();
        _discDropoutLayers.Clear();

        int prevDim = inputDim;
        var discDims = _options.DiscriminatorDimensions;

        for (int i = 0; i < discDims.Length; i++)
        {
            _discLayerDims.Add((prevDim, discDims[i]));
            _discDropoutLayers.Add(new DropoutLayer<T>(_options.DiscriminatorDropout));
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

        // If using custom layers, do simple sequential forward
        if (_usingCustomLayers)
        {
            Tensor<T> current = input;
            foreach (var layer in Layers)
            {
                current = layer.Forward(current);
            }
            return current;
        }

        // Default generator with residual connections + BN + ReLU
        return GeneratorForwardWithResidual(input);
    }

    /// <summary>
    /// Trains the CopulaGAN network using the provided input and expected output.
    /// </summary>
    /// <param name="input">The input noise tensor.</param>
    /// <param name="expectedOutput">The expected real data tensor.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This runs a single training step for the generator.
    /// The full GAN training loop (alternating discriminator/generator) happens in Fit().
    /// This method is provided for compatibility with the NeuralNetworkBase pattern.
    /// </para>
    /// </remarks>
    protected override bool TryApplyLearningRule(Tensor<T> input, Tensor<T> expectedOutput)
    {
        // CopulaGAN is trained adversarially through Fit() / FitAsync (the copula
        // transform is fit statistically and the GAN is trained with the
        // alternating critic/generator loop). The NeuralNetworkBase supervised
        // Train(input, expected) contract does not map onto a GAN's minimax
        // objective — silently running a forward + loss but skipping every
        // parameter update would report a misleading "training" loss while
        // making zero learning progress, which is worse than failing fast.
        throw new NotSupportedException(
            "CopulaGAN is trained adversarially; the supervised Train(input, expected) " +
            "contract does not map onto its minimax objective. Use Fit(IEnumerable<Tensor<T>>) " +
            "or FitAsync to drive the alternating discriminator/generator loop, which fits " +
            "the copula transform and runs the WGAN critic + generator updates.");
    }

    // UpdateParameters re-sliced the flat vector across Layers by hand -- the base walks
    // exactly the same enumeration, so this said nothing the base does not already say.
    #endregion

    #region ISyntheticTabularGenerator<T> Implementation

    /// <summary>
    /// Fits the CopulaGAN generator to the provided real tabular data.
    /// </summary>
    /// <param name="data">The real data matrix where each row is a sample and each column is a feature.</param>
    /// <param name="columns">Metadata describing each column (type, categories, etc.).</param>
    /// <param name="epochs">Number of training epochs. When null, the model's published Epochs from its options is used.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This is the "learning" step. The generator studies your real data:
    /// 1. Applies Gaussian copula transform to normalize continuous columns
    /// 2. Fits VGM transformer for mode-specific normalization
    /// 3. Trains generator and discriminator in alternating WGAN-GP steps
    /// After fitting, call Generate() to create new synthetic rows.
    /// </para>
    /// </remarks>
    public void Fit(Matrix<T> data, IReadOnlyList<ColumnMetadata> columns, int? epochs = null)
    {
        try
        {
            FitCore(data, columns, epochs);
        }
        finally
        {
            // The cached training rows only serve the training loop: a fitted generator, or one whose Fit failed,
            // does not keep a copy of its training data.
            _realDataSource = null;
            _realDataTensor = null;
        }
    }

    private void FitCore(Matrix<T> data, IReadOnlyList<ColumnMetadata> columns, int? epochs)
    {
        int epochCount = epochs ?? _options.Epochs;
        ValidateFitInputs(data, columns, epochCount);

        _columns = PrepareColumns(data, columns);

        // Step 1: Identify continuous columns and apply Gaussian copula transform
        _continuousColumnIndices.Clear();
        _sortedValues.Clear();
        for (int col = 0; col < columns.Count; col++)
        {
            if (columns[col].IsNumerical)
            {
                _continuousColumnIndices.Add(col);
            }
        }
        var copulaData = ApplyCopulaTransform(data);

        // Step 2: Fit the VGM transformer on copula-transformed data
        _transformer = new TabularDataTransformer<T>(_options.VGMModes, _random);
        _transformer.Fit(copulaData, _columns);
        _dataWidth = _transformer.TransformedWidth;

        // Step 3: Fit the data sampler
        _sampler = new CTGANDataSampler<T>(_random);
        _sampler.Fit(copulaData, _columns);
        _condWidth = _sampler.ConditionalVectorWidth;

        // Step 4: Compute packed input dimension
        _packedInputDim = (_dataWidth + _condWidth) * _options.PacSize;

        // Step 5: Rebuild layers with actual data dimensions
        int genInputDim = _options.EmbeddingDimension + _condWidth;
        BuildFromConstructionSequence(() => RebuildLayersWithActualDimensions(genInputDim, _dataWidth, _packedInputDim));

        // Step 6: Transform data
        var transformedData = _transformer.Transform(copulaData);

        // Step 7: Training loop
        T lr = NumOps.FromDouble(_options.LearningRate);
        int pacSize = _options.PacSize;
        int batchSize = _options.BatchSize;
        int numPacks = Math.Max(1, batchSize / pacSize);
        int numBatches = Math.Max(1, data.Rows / (numPacks * pacSize));

        for (int epoch = 0; epoch < epochCount; epoch++)
        {
            for (int batch = 0; batch < numBatches; batch++)
            {
                // Paper-faithful WGAN-GP schedule: DiscriminatorSteps critic
                // updates per generator update. Both via GradientTape to
                // satisfy the autodiff migration (LayerBase.cs:1593).
                for (int dStep = 0; dStep < _options.DiscriminatorSteps; dStep++)
                {
                    TrainDiscriminatorStepBatched(transformedData, numPacks);
                }
                TrainGeneratorStepBatched(transformedData, numPacks);
            }
        }

        IsFitted = true;
    }

    /// <inheritdoc />
    public async Task FitAsync(Matrix<T> data, IReadOnlyList<ColumnMetadata> columns, int? epochs = null, CancellationToken ct = default)
    {
        int epochCount = epochs ?? _options.Epochs;
        ValidateFitInputs(data, columns, epochCount);

        _columns = PrepareColumns(data, columns);

        await Task.Run(() =>
        {
            ct.ThrowIfCancellationRequested();

            _continuousColumnIndices.Clear();
            _sortedValues.Clear();
            for (int col = 0; col < columns.Count; col++)
            {
                if (columns[col].IsNumerical)
                {
                    _continuousColumnIndices.Add(col);
                }
            }
            var copulaData = ApplyCopulaTransform(data);

            _transformer = new TabularDataTransformer<T>(_options.VGMModes, _random);
            _transformer.Fit(copulaData, _columns);
            _dataWidth = _transformer.TransformedWidth;

            _sampler = new CTGANDataSampler<T>(_random);
            _sampler.Fit(copulaData, _columns);
            _condWidth = _sampler.ConditionalVectorWidth;

            _packedInputDim = (_dataWidth + _condWidth) * _options.PacSize;

            int genInputDim = _options.EmbeddingDimension + _condWidth;
            BuildFromConstructionSequence(() => RebuildLayersWithActualDimensions(genInputDim, _dataWidth, _packedInputDim));

            // Pre-allocate training buffers
            InitializeTrainingBuffers(genInputDim);

            var transformedData = _transformer.Transform(copulaData);

            T lr = NumOps.FromDouble(_options.LearningRate);
            int pacSize = _options.PacSize;
            int batchSize = _options.BatchSize;
            int numPacks = Math.Max(1, batchSize / pacSize);
            int numBatches = Math.Max(1, data.Rows / (numPacks * pacSize));

            for (int epoch = 0; epoch < epochCount; epoch++)
            {
                ct.ThrowIfCancellationRequested();
                for (int batch = 0; batch < numBatches; batch++)
                {
                    for (int dStep = 0; dStep < _options.DiscriminatorSteps; dStep++)
                    {
                        TrainDiscriminatorStepBatched(transformedData, numPacks);
                    }
                    TrainGeneratorStepBatched(transformedData, numPacks);
                }
            }
        }, ct).ConfigureAwait(false);

        IsFitted = true;
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
    /// 3. Applying inverse VGM transform
    /// 4. Applying inverse copula transform to restore original distributions
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

        int[] conditionColumns = conditionColumn is not null ? SyntheticTensorOps<T>.TruncateToInt(conditionColumn) : Array.Empty<int>();
        int[] conditionValues = conditionValue is not null ? SyntheticTensorOps<T>.TruncateToInt(conditionValue) : Array.Empty<int>();
        var generated = new Tensor<T>(new[] { numSamples, _dataWidth });

        // Rows are generated one at a time, in order, so each row's noise and condition draws keep their RNG order.
        for (int i = 0; i < numSamples; i++)
        {
            var noise = CreateStandardNormalVector(_options.EmbeddingDimension);

            Vector<T> condVector;
            if (conditionColumn is not null && conditionValue is not null && i < conditionColumn.Length)
            {
                condVector = _sampler.CreateConditionVector(conditionColumns[i], conditionValues[i]);
            }
            else
            {
                condVector = _sampler.SampleRandomConditionVector();
            }

            var genInput = ConcatVectors(noise, condVector);
            SyntheticTensorOps<T>.WriteRow(generated, i, Predict(VectorToTensor(genInput)), _dataWidth);
        }

        // CopulaGAN samples like sdv-dev/CTGAN: each mode and category block is drawn from its distribution
        // (gumbel_softmax then argmax), not set to its most likely entry.
        // Inverse VGM transform, then inverse copula transform, in bounded blocks so only one block's
        // intermediates are alive at a time.
        TabularDataTransformer<T> transformer = _transformer;
        int columnCount = _columns.Count;
        return SyntheticTensorOps<T>.DecodeInBlocks(generated, block => ApplyInverseCopulaTransform(transformer.InverseTransform(
            SyntheticTensorOps<T>.SampleCategoricalBlocks(block, transformer, columnCount, _random).ToMatrix())));
    }

    #endregion

    #region GAN Training Steps

    #region Training Buffer Management

    private void InitializeTrainingBuffers(int genInputDim)
    {
        int singleDim = _dataWidth + _condWidth;
        _oneGrad = new Tensor<T>([1]);
        _oneGrad[0] = NumOps.One;
        _negOneGrad = new Tensor<T>([1]);
        _negOneGrad[0] = NumOps.Negate(NumOps.One);
        _packedRealBuf = new Vector<T>(_packedInputDim);
        _packedFakeBuf = new Vector<T>(_packedInputDim);
        _noiseBuf = new Vector<T>(_options.EmbeddingDimension);
        _genInputBuf = new Vector<T>(genInputDim);
        _realSingleBuf = new Vector<T>(singleDim);
        _fakeSingleBuf = new Vector<T>(singleDim);
        _realRowBuf = new Vector<T>(_dataWidth);
        _fakeRowBuf = new Vector<T>(_dataWidth);
        _interpolatedBuf = new Vector<T>(_packedInputDim);
        _sampleGradBuf = new Tensor<T>([_dataWidth]);
    }

    #endregion

    /// <summary>
    /// Paper-faithful WGAN-GP critic step (Xu et al. 2019 CTGAN +
    /// Patki et al. 2016 Gaussian copula) over PacGAN-packed batches.
    /// Tape-tracked through <see cref="DiscriminatorForwardBatched"/>.
    /// </summary>
    private void TrainDiscriminatorStepBatched(Matrix<T> transformedData, int numPacks)
    {
        if (_sampler is null) return;

        var (realPacked, fakePacked) = BuildPackedRealAndFakeBatches(transformedData, numPacks);

        // The shared WGAN-GP critic step: the GPU-resident primitive, the fused compiled plan or the shared eager
        // tape step, with one replay-safe gradient penalty (Gulrajani et al. 2017; #1844 / #1845).
        AiDotNet.Training.WassersteinCriticStep<T>.Step(
            this,
            _discLayers,
            realPacked,
            fakePacked,
            input => DiscriminatorForwardBatched(input, isTraining: true),
            _options.GradientPenaltyWeight,
            _discriminatorOptimizer,
            _random,
            ref _wganFusedStep, maxGradNorm: _options.MaxGradNorm);
    }

    /// <summary>
    /// Paper-faithful WGAN-GP generator step. Minimizes -E[D(G(z,c),c)].
    /// </summary>
    private void TrainGeneratorStepBatched(Matrix<T> transformedData, int numPacks)
    {
        if (_sampler is null) return;

        var generatorLayers = new List<ILayer<T>>();
        generatorLayers.AddRange(Layers);
        foreach (var bn in _genBNLayers) generatorLayers.Add(bn);
        var genParams = TapeTrainingStep<T>.CollectParameters(generatorLayers);

        int pacSize = _options.PacSize;
        int total = numPacks * pacSize;

        var noiseBatch = GenerateNoiseBatchTensor(total);
        var (condBatch, maskBatch) = SyntheticTensorOps<T>.SampleCondMaskBatch(_sampler, total, _condWidth);
        var catBlocks = SyntheticTensorOps<T>.CategoricalOutputBlocks(_transformer, _columns.Count);

        // GPU-RESIDENT fast path: the persistent input packs [noise | cond | mask | Gumbel noise] so a replayed plan
        // uses a fresh condition, mask and Gumbel draw each step.
        var trainableGenLayers = generatorLayers.OfType<ITrainableLayer<T>>().ToList();
        if (trainableGenLayers.Count > 0)
        {
            int embedDim = _options.EmbeddingDimension;
            int condDim = _condWidth;
            var gumbel = SyntheticTensorOps<T>.GumbelNoise([total, _dataWidth], _random);
            var fusedInput = Engine.TensorConcatenate([noiseBatch, condBatch, maskBatch, gumbel], axis: 1);
            var target = new Tensor<T>(new[] { 1 });
            Tensor<T>? capturedHead = null;
            Tensor<T>? capturedCond = null;
            Tensor<T>? capturedMask = null;
            Tensor<T> Fwd(Tensor<T> packedInput)
            {
                var ginp = Engine.TensorSlice(packedInput, [0, 0], [total, embedDim + condDim]);
                var faked = GeneratorForwardWithResidualBatched(ginp);
                var gumbelFromInput = Engine.TensorSlice(packedInput, [0, embedDim + 2 * condDim], [total, _dataWidth]);
                var act = ApplyOutputActivationsBatched(faked, gumbelFromInput);
                var condFromInput = Engine.TensorSlice(packedInput, [0, embedDim], [total, condDim]);
                capturedHead = faked;
                capturedCond = condFromInput;
                capturedMask = Engine.TensorSlice(packedInput, [0, embedDim + condDim], [total, condDim]);
                var withCond = Engine.TensorConcatenate([act, condFromInput], axis: 1);
                var packed = withCond.Reshape([numPacks, _packedInputDim]);
                return DiscriminatorForwardBatched(packed, false);
            }
            // sdv-dev/CTGAN's generator objective, which CopulaGAN inherits: -E[D(G(z, c))] + the conditional
            // cross-entropy that makes the generator honour c.
            Tensor<T> Loss(Tensor<T> scores, Tensor<T> _)
            {
                var axes = Enumerable.Range(0, scores.Shape.Length).ToArray();
                var lossT = Engine.TensorNegate(Engine.ReduceMean(scores, axes, keepDims: false));
                if (condDim > 0 && catBlocks.Count > 0)
                {
                    if (capturedHead is null || capturedCond is null || capturedMask is null)
                        throw new InvalidOperationException(
                            "CopulaGAN fused step: generator intermediates were not captured by the forward closure.");
                    lossT = Engine.TensorAdd(lossT,
                        SyntheticTensorOps<T>.ConditionalCrossEntropy(capturedHead, capturedCond, capturedMask, catBlocks));
                }
                return lossT;
            }
            // One step through the shared training step: the fused plan when it applies, otherwise the shared
            // eager tape step on the same objective. The critic scores the samples in evaluation mode and is not
            // in the generator's parameter group, so only the generator moves.
            AiDotNet.Training.FusedTrainingStep<T>.Step(
                this, trainableGenLayers, fusedInput, target,
                forward: Fwd,
                computeLoss: Loss,
                optimizer: _generatorOptimizer,
                maxGradNorm: _options.MaxGradNorm,
                extraTensors: genParams);
        }
    }

    private (Tensor<T> realPacked, Tensor<T> fakePacked) BuildPackedRealAndFakeBatches(
        Matrix<T> transformedData,
        int numPacks)
    {
        int pacSize = _options.PacSize;
        int total = numPacks * pacSize;

        var noiseBatch = GenerateNoiseBatchTensor(total);
        var (condBatch, rowIndices) = SampleConditionalBatchWithRows(total);
        var genInput = Engine.TensorConcatenate([noiseBatch, condBatch], axis: 1);

        var fakeFlat = GeneratorForwardWithResidualBatched(genInput);
        var fakeActivated = ApplyOutputActivationsBatched(fakeFlat, SyntheticTensorOps<T>.GumbelNoise(fakeFlat._shape, _random));
        var fakeSingles = Engine.TensorConcatenate([fakeActivated, condBatch], axis: 1);

        if (_realDataTensor is null || !ReferenceEquals(_realDataSource, transformedData))
        {
            _realDataTensor = Tensor<T>.FromMatrix(transformedData);
            _realDataSource = transformedData;
        }
        var realFlat = SyntheticTensorOps<T>.GatherRows(_realDataTensor, rowIndices, _dataWidth);
        var realSingles = Engine.TensorConcatenate([realFlat, condBatch], axis: 1);

        return (realSingles.Reshape([numPacks, _packedInputDim]),
                fakeSingles.Reshape([numPacks, _packedInputDim]));
    }

    private Tensor<T> GenerateNoiseBatchTensor(int batchSize)
    {
        int embedDim = _options.EmbeddingDimension;
        int totalElements = batchSize * embedDim;
        int halfElements = (totalElements + 1) / 2;
        var u2 = Engine.TensorRandomUniformRange<T>([halfElements], NumOps.Zero, NumOps.One, _random.Next());
        var u1Temp = Engine.TensorRandomUniformRange<T>([halfElements], NumOps.Zero, NumOps.One, _random.Next());
        var u1 = Engine.ScalarMinusTensor(NumOps.One, u1Temp);
        var radius = Engine.TensorSqrt(Engine.TensorMultiplyScalar(Engine.TensorLog(u1), NumOps.FromDouble(-2.0)));
        var theta = Engine.TensorMultiplyScalar(u2, NumOps.FromDouble(2.0 * Math.PI));
        var z1 = Engine.TensorMultiply(radius, Engine.TensorCos(theta));
        var z2 = Engine.TensorMultiply(radius, Engine.TensorSin(theta));
        return SyntheticTensorOps<T>.InterleaveNormalPairs(z1, z2, batchSize, embedDim);
    }

    private (Tensor<T> CondBatch, int[] RowIndices) SampleConditionalBatchWithRows(int batchSize)
        => SyntheticTensorOps<T>.SampleConditionsWithRows(_sampler, batchSize, _condWidth);

    /// <summary>
    /// Batched, tape-tracked generator forward with residual skip connections.
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
        int numHidden = Layers.Count - 1;
        for (int i = 0; i < numHidden; i++)
        {
            if (i > 0) h = Engine.TensorConcatenate([h, input], axis: 1);
            h = Layers[i].Forward(h);
            if (i < _genBNLayers.Count) h = _genBNLayers[i].Forward(h);
            h = Engine.ReLU(h);
        }
        h = Engine.TensorConcatenate([h, input], axis: 1);
        h = Layers[^1].Forward(h);
        return h;
    }

    /// <summary>
    /// Batched, tape-tracked discriminator forward (LeakyReLU 0.2 + Dropout).
    /// </summary>
    private Tensor<T> DiscriminatorForwardBatched(Tensor<T> input, bool isTraining)
    {
        var current = input;
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
            current = _discLayers[i].Forward(current);
            if (!isLastDense) current = Engine.LeakyReLU(current, leakySlope);
            layerIdx++;
        }
        return current;
    }

    /// <summary>
    /// Batched per-column VGM output activation: Tanh on continuous mode
    /// value, Softmax on mode probabilities, Softmax on categorical blocks. With Gumbel noise the softmaxes
    /// are Gumbel-softmax samples at <see cref="CopulaGANOptions{T}.GumbelTemperature"/>, as in sdv-dev/CTGAN.
    /// </summary>
    private Tensor<T> ApplyOutputActivationsBatched(Tensor<T> output, Tensor<T>? gumbelNoise = null)
        => SyntheticTensorOps<T>.ApplyOutputActivationsBatched(
            output, _transformer, _columns.Count, gumbelNoise, _options.GumbelTemperature);

    #endregion

    #region Discriminator Forward/Backward

    #endregion

    #region Gradient Penalty

    #endregion

    #region Generator Forward/Backward with Residual Connections

    /// <summary>
    /// Generator forward pass with CTGAN-style residual connections:
    /// Each hidden layer receives [previous_output, original_input].
    /// </summary>
    private Tensor<T> GeneratorForwardWithResidual(Tensor<T> input)
    {
        var inputTensor = input;
        var current = input;

        // Hidden layers with residual connections + BN + ReLU
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

            current = ApplyReLU(current);
        }

        // Output layer with residual connection
        current = ConcatTensors(current, inputTensor);
        current = Layers[^1].Forward(current);

        return ApplyOutputActivations(current);
    }

    #endregion

    #region Copula Transform

    /// <summary>
    /// Applies Gaussian copula transform to continuous columns:
    /// value -> empirical CDF -> inverse normal CDF (quantile function).
    /// </summary>
    private Matrix<T> ApplyCopulaTransform(Matrix<T> data)
    {
        var result = data.Clone();
        foreach (int col in _continuousColumnIndices)
        {
            var (z, sorted) = SyntheticTensorOps<T>.GaussianCopulaForward(Tensor<T>.FromVector(data.GetColumn(col)));
            _sortedValues.Add(sorted.ToArray());
            result.SetColumn(col, z.ToVector());
        }

        return result;
    }

    /// <summary>
    /// Applies inverse Gaussian copula transform to continuous columns.
    /// </summary>
    private Matrix<T> ApplyInverseCopulaTransform(Matrix<T> data)
    {
        var result = data.Clone();
        for (int idx = 0; idx < _continuousColumnIndices.Count && idx < _sortedValues.Count; idx++)
        {
            int col = _continuousColumnIndices[idx];
            var original = SyntheticTensorOps<T>.GaussianCopulaInverse(
                Tensor<T>.FromVector(data.GetColumn(col)), Tensor<T>.FromVector(new Vector<T>(_sortedValues[idx])));
            result.SetColumn(col, original.ToVector());
        }

        return result;
    }

    #endregion

    #region Activation Functions

    private Tensor<T> ApplyReLU(Tensor<T> input) => SyntheticTensorOps<T>.ReLU(input);

    /// <summary>
    /// Applies tanh to continuous columns and softmax to categorical/mode columns.
    /// </summary>
    private Tensor<T> ApplyOutputActivations(Tensor<T> output)
    {
        if (_transformer is null) return output;
        return SyntheticTensorOps<T>.ApplyOutputActivations(output, _transformer, _columns.Count);
    }

    #endregion

    #region Gradient Safety Utilities

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
        => SyntheticTabularGeneratorBase<T>.PrepareColumns(data, columns);

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

    private static Vector<T> ConcatVectors(Vector<T> a, Vector<T> b) => SyntheticTensorOps<T>.ConcatVectors(a, b);

    private static Tensor<T> VectorToTensor(Vector<T> v) => Tensor<T>.FromVector(v);

    private static Tensor<T> ConcatTensors(Tensor<T> a, Tensor<T> b) => SyntheticTensorOps<T>.Concat1D(a, b);

    #endregion

}
