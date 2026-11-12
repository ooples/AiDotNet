using AiDotNet.Attributes;
using AiDotNet.Autodiff;
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
/// Conditional Tabular GAN (CTGAN) for generating realistic synthetic tabular data.
/// </summary>
/// <remarks>
/// <para>
/// CTGAN combines several innovations for effective tabular data generation:
/// - <b>VGM normalization</b>: Handles multi-modal continuous distributions
/// - <b>Conditional generation</b>: Training-by-sampling ensures all categories are represented
/// - <b>WGAN-GP loss</b>: Wasserstein distance with gradient penalty for stable training
/// - <b>Residual generator</b>: Skip connections in the generator for better gradient flow
/// - <b>PacGAN</b>: Packing multiple samples to prevent mode collapse
///
/// This implementation follows the standard neural network architecture pattern with:
/// - Proper inheritance from NeuralNetworkBase
/// - Layer-based architecture using ILayer components
/// - Engine-based tensor operations for CPU/GPU acceleration
/// - Full autodiff and JIT compilation support
/// </para>
/// <para>
/// <b>For Beginners:</b> CTGAN works like a forgery competition:
///
/// 1. The <b>Generator</b> starts with random noise and tries to create realistic table rows
/// 2. The <b>Discriminator</b> sees both real and generated rows and tries to tell them apart
/// 3. They train together: the generator gets better at fooling the discriminator,
///    and the discriminator gets better at spotting fakes
/// 4. Eventually, the generator produces rows that are indistinguishable from real data
///
/// If you provide custom layers in the architecture, those will be used directly
/// for the generator network. If not, the network creates industry-standard
/// CTGAN layers based on the original research paper specifications.
///
/// Example usage:
/// <code>
/// var architecture = new NeuralNetworkArchitecture&lt;double&gt;(
///     inputFeatures: 10,
///     outputSize: 10
/// );
/// var options = new CTGANOptions&lt;double&gt;
/// {
///     EmbeddingDimension = 128,
///     GeneratorDimensions = new[] { 256, 256 },
///     BatchSize = 500,
///     Epochs = 300
/// };
/// var generator = new CTGANGenerator&lt;double&gt;(architecture, options);
/// generator.Fit(data, columns, epochs: 300);
/// var synthetic = generator.Generate(1000);
/// </code>
/// </para>
/// <para>
/// Reference: "Modeling Tabular Data using Conditional GAN" (Xu et al., NeurIPS 2019)
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
public partial class CTGANGenerator<T> : NeuralSyntheticTabularGeneratorBase<T>, ISyntheticTabularGenerator<T>
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

    private readonly CTGANOptions<T> _options;
    // SEPARATE optimizers for generator and discriminator. They MUST be distinct
    // instances: AdamOptimizer keeps a single flat (_m,_v) moment buffer sized to
    // the parameter count and resets the timestep whenever that count changes, so
    // sharing one optimizer across the generator and discriminator (different
    // param counts) wiped the moments every alternating Step and pinned bias
    // correction at t=1 — the root cause of CTGAN diverging worse with more
    // epochs. The golden GAN base (GenerativeAdversarialNetwork<T>) likewise keeps
    // separate optimizers.
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

    // Generated-output offsets of each categorical column's softmax block, in
    // categorical-column order — the SAME order the conditional vector lays out its
    // category blocks. Used to align the generator's output categorical
    // distributions with the conditional one-hot so the §4.3 conditional
    // cross-entropy loss can be computed. (offset, width) per categorical column.
    private readonly List<(int Offset, int Width)> _catOutputBlocks = new();

    // Generator batch normalization layers (auxiliary, always created to match Layers)
    private readonly List<BatchNormalizationLayer<T>> _genBNLayers = new();

    // Discriminator layers (auxiliary, not user-overridable)
    private readonly List<ILayer<T>> _discLayers = new();
    private readonly List<DropoutLayer<T>> _discDropoutLayers = new();
    private readonly List<(int InputSize, int OutputSize)> _discLayerDims = new();

    // Tensor view of the transformed training matrix, rebuilt when a different matrix is passed in.
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
    /// Gets the CTGAN-specific options.
    /// </summary>
    public new CTGANOptions<T> Options => _options;

    /// <inheritdoc />
    public IReadOnlyList<ColumnMetadata> Columns => _columns.AsReadOnly();

    /// <inheritdoc />
    public bool IsFitted { get; private set; }

    /// <summary>
    /// Initializes a new CTGAN generator with the specified architecture.
    /// </summary>
    /// <param name="architecture">The neural network architecture defining input/output dimensions and optional custom layers.</param>
    /// <param name="options">CTGAN-specific options for generator and discriminator configuration.</param>
    /// <param name="optimizer">Gradient-based optimizer (defaults to Adam).</param>
    /// <param name="lossFunction">Loss function (defaults based on task type).</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This constructor creates a CTGAN network based on the architecture you provide.
    ///
    /// If you provide custom layers in the architecture, those will be used directly
    /// for the generator network. If not, the network will create industry-standard
    /// CTGAN generator layers based on the original research paper specifications.
    ///
    /// Example usage:
    /// <code>
    /// var architecture = new NeuralNetworkArchitecture&lt;double&gt;(
    ///     inputFeatures: 10,
    ///     outputSize: 10
    /// );
    /// var options = new CTGANOptions&lt;double&gt; { EmbeddingDimension = 128 };
    /// var generator = new CTGANGenerator&lt;double&gt;(architecture, options);
    /// </code>
    /// </para>
    /// </remarks>
    /// <summary>
    /// Initializes a new instance with default architecture settings.
    /// </summary>
    public CTGANGenerator()
        : this(new NeuralNetworkArchitecture<T>(
            inputType: Enums.InputType.OneDimensional,
            taskType: Enums.NeuralNetworkTaskType.Generative,
            inputSize: 10,
            outputSize: 10))
    {
    }

    public CTGANGenerator(NeuralNetworkArchitecture<T> architecture,
        CTGANOptions<T>? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
        ILossFunction<T>? lossFunction = null)
        : base(architecture, lossFunction ?? NeuralNetworkHelper<T>.GetDefaultLossFunction(architecture.TaskType), (options ??= new CTGANOptions<T>()).MaxGradNorm)
    {
        _options = options;
        _lossFunction = lossFunction ?? NeuralNetworkHelper<T>.GetDefaultLossFunction(architecture.TaskType);

        // WGAN-GP Adam configuration (Gulrajani et al. 2017 / Xu et al. 2019):
        // β1=0.5, β2=0.9, no adaptive-LR mutation (the GP already regularizes the
        // critic). Two independent instances so generator and discriminator
        // moment estimates never cross-contaminate.
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

        int? seed = _options.Seed;
        _random = seed.HasValue
            ? RandomHelper.CreateSeededRandom(seed.Value)
            : RandomHelper.CreateSecureRandom();

        InitializeLayers();
    }

    #region Layer Initialization (GANDALF Pattern)

    /// <inheritdoc />
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
            Layers.AddRange(LayerHelper<T>.CreateDefaultCTGANGeneratorLayers(
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
    /// Rebuilds generator and discriminator layers with actual data dimensions discovered during Fit().
    /// </summary>
    private void RebuildLayersWithActualDimensions(int genInputDim, int genOutputDim, int discInputDim)
    {
        if (!_usingCustomLayers)
        {
            Layers.Clear();
            Layers.AddRange(LayerHelper<T>.CreateDefaultCTGANGeneratorLayers(
                genInputDim, genOutputDim, _options.GeneratorDimensions));

            _genBNLayers.Clear();
            foreach (int dim in _options.GeneratorDimensions)
            {
                _genBNLayers.Add(new BatchNormalizationLayer<T>());
            }
        }

        // Discriminator is always rebuilt with actual dimensions
        _discLayers.Clear();
        _discDropoutLayers.Clear();
        _discLayerDims.Clear();
        _discLayers.AddRange(LayerHelper<T>.CreateDefaultCTGANDiscriminatorLayers(
            discInputDim, _options.DiscriminatorDimensions, _options.DiscriminatorDropout));

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

    /// <inheritdoc />
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
    /// Fits the CTGAN generator to the provided real tabular data.
    /// </summary>
    /// <param name="data">The real data matrix where each row is a sample and each column is a feature.</param>
    /// <param name="columns">Metadata describing each column (type, categories, etc.).</param>
    /// <param name="epochs">Number of training epochs. When null, the model's published Epochs from its options is used.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This is the "learning" step. The generator studies your real data:
    /// 1. Fits VGM transformer for data normalization
    /// 2. Builds conditional vectors for training-by-sampling
    /// 3. Trains generator and discriminator in alternating WGAN-GP steps
    /// After fitting, call Generate() to create new synthetic rows.
    /// </para>
    /// </remarks>
    public void Fit(Matrix<T> data, IReadOnlyList<ColumnMetadata> columns, int? epochs = null)
    {
        int epochCount = epochs ?? _options.Epochs;
        ValidateFitInputs(data, columns, epochCount);

        _columns = PrepareColumns(data, columns);

        // Step 1: Fit the VGM transformer
        _transformer = new TabularDataTransformer<T>(_options.VGMModes, _random);
        _transformer.Fit(data, _columns);
        _dataWidth = _transformer.TransformedWidth;
        BuildCategoricalOutputBlocks();

        // Step 2: Fit the data sampler
        _sampler = new CTGANDataSampler<T>(_random);
        _sampler.Fit(data, _columns);
        _condWidth = _sampler.ConditionalVectorWidth;

        // Step 3: Compute packed input dimension
        _packedInputDim = (_dataWidth + _condWidth) * _options.PacSize;

        // Step 4: Rebuild layers with actual data dimensions
        int genInputDim = _options.EmbeddingDimension + _condWidth;
        BuildFromConstructionSequence(() => RebuildLayersWithActualDimensions(genInputDim, _dataWidth, _packedInputDim));

        // Step 5: Transform data
        var transformedData = _transformer.Transform(data);

        // Step 6: Training loop
        T lr = NumOps.FromDouble(_options.LearningRate);
        int pacSize = _options.PacSize;
        int batchSize = _options.BatchSize;
        int numPacks = Math.Max(1, batchSize / pacSize);
        int numBatches = Math.Max(1, data.Rows / (numPacks * pacSize));

        for (int epoch = 0; epoch < epochCount; epoch++)
        {
            for (int batch = 0; batch < numBatches; batch++)
            {
                // Paper's training schedule: DiscriminatorSteps critic steps
                // for each generator step. Both run through GradientTape so
                // gradients propagate through the tape-tracked forwards.
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

            _transformer = new TabularDataTransformer<T>(_options.VGMModes, _random);
            _transformer.Fit(data, _columns);
            _dataWidth = _transformer.TransformedWidth;
            BuildCategoricalOutputBlocks();

            _sampler = new CTGANDataSampler<T>(_random);
            _sampler.Fit(data, _columns);
            _condWidth = _sampler.ConditionalVectorWidth;

            _packedInputDim = (_dataWidth + _condWidth) * _options.PacSize;

            int genInputDim = _options.EmbeddingDimension + _condWidth;
            BuildFromConstructionSequence(() => RebuildLayersWithActualDimensions(genInputDim, _dataWidth, _packedInputDim));

            // Pre-allocate training buffers to avoid per-row GC pressure
            InitializeTrainingBuffers(genInputDim);

            var transformedData = _transformer.Transform(data);

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
        var fakeRows = new List<Tensor<T>>(numSamples);

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
            fakeRows.Add(Predict(VectorToTensor(genInput)));
        }

        // sdv-dev/CTGAN samples each mode and category with gumbel_softmax and then takes the argmax: a draw from
        // the block's distribution, not its most likely entry.
        var sampled = SyntheticTensorOps<T>.SampleCategoricalBlocks(
            SyntheticTensorOps<T>.StackRows(fakeRows, _dataWidth), _transformer, _columns.Count, _random);
        return _transformer.InverseTransform(sampled.ToMatrix());
    }

    #endregion

    #region Training Buffer Management

    /// <summary>
    /// Pre-allocates reusable buffers for the training loop to eliminate per-row GC pressure.
    /// Called once before training begins when all dimensions are known.
    /// </summary>
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

    #region GAN Training Steps

    /// <summary>
    /// Paper-faithful WGAN-GP critic step (Xu et al. 2019 §3.4, after Gulrajani
    /// et al. 2017) over <see cref="_options.PacSize"/>-packed samples (PacGAN
    /// to prevent mode collapse). Uses <see cref="GradientTape{T}"/> +
    /// <see cref="TapeStepContext{T}"/> so backprop flows through every
    /// tape-tracked op in <see cref="DiscriminatorForwardBatched"/>. Replaces
    /// the prior per-row <c>DiscriminatorForward → UpdateParameters</c>
    /// pattern, which the codebase's autodiff migration (LayerBase.cs:1593)
    /// made invalid — that path now throws
    /// "Backward pass must be called before updating parameters."
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
            ref _wganFusedStep);
    }

    /// <summary>
    /// Paper-faithful WGAN-GP generator step (Xu et al. 2019). Minimizes
    /// <c>-E[D(G(z, c))]</c> so the generator pushes its conditional samples
    /// into regions the critic scores high. Tape-tracked through
    /// <see cref="GeneratorForwardWithResidualBatched"/> + the critic's
    /// frozen forward, so generator parameters get gradients via the critic
    /// but the critic itself does not update on this step.
    /// </summary>
    /// <summary>
    /// Records the generated-output offset + width of each categorical column's
    /// softmax block, in categorical-column order (the order the conditional vector
    /// uses), so <see cref="ConditionalCrossEntropy"/> can align them.
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
    /// Samples a batch of conditional vectors together with their masks (Xu 2019
    /// §4.3). Returns ([N, condWidth] one-hot conditions, [N, condWidth] masks that
    /// are 1 over the selected column's block). Both are tape constants.
    /// </summary>
    private (Tensor<T> Cond, Tensor<T> Mask) SampleCondMaskBatch(int batchSize)
    {
        if (_sampler is null || _condWidth == 0)
            return (new Tensor<T>([batchSize, _condWidth]), new Tensor<T>([batchSize, _condWidth]));

        var conds = new List<Vector<T>>(batchSize);
        var masks = new List<Vector<T>>(batchSize);
        for (int b = 0; b < batchSize; b++)
        {
            var (cv, mv, _, _, _) = _sampler.SampleCondVecWithMask();
            conds.Add(cv);
            masks.Add(mv);
        }
        return (SyntheticTensorOps<T>.StackRows(conds, _condWidth), SyntheticTensorOps<T>.StackRows(masks, _condWidth));
    }

    /// <summary>
    /// Conditional generator loss term (Xu 2019 §4.3): cross-entropy between the
    /// conditional one-hot and the generator's output distribution for the
    /// conditioned discrete column, masked to that column. With a one-hot condition
    /// inside the masked block this reduces to -log p(selected category), averaged
    /// over the batch — the term that forces the generator to actually honour the
    /// conditional vector instead of ignoring it.
    /// </summary>
    private Tensor<T> ConditionalCrossEntropy(Tensor<T> generatorLogits, Tensor<T> condBatch, Tensor<T> maskBatch)
    {
        int batch = generatorLogits.Shape[0];
        // sdv-dev/CTGAN _cond_loss: cross-entropy of each conditioned column's category LOGITS against the requested
        // one-hot, not the log of the (Gumbel-noised, temperature 0.2) sample. Blocks are gathered in
        // categorical-column order -> [batch, condWidth], aligning with the conditional-vector layout.
        var blocks = new List<Tensor<T>>(_catOutputBlocks.Count);
        foreach (var (offset, width) in _catOutputBlocks)
            blocks.Add(Engine.TensorLogSoftmax(Engine.TensorSlice(generatorLogits, [0, offset], [batch, width]), 1));
        var logp = blocks.Count == 1 ? blocks[0] : Engine.TensorConcatenate(blocks.ToArray(), axis: 1);

        // CE = -mean_b sum_j mask*cond*log_softmax(logits).
        var masked = Engine.TensorMultiply(Engine.TensorMultiply(maskBatch, condBatch), logp);
        var perRow = Engine.ReduceSum(masked, [1], keepDims: false);     // [batch]
        var meanCe = Engine.ReduceMean(perRow, [0], keepDims: false);     // scalar
        return Engine.TensorNegate(meanCe);
    }

    private void TrainGeneratorStepBatched(Matrix<T> transformedData, int numPacks)
    {
        if (_sampler is null) return;

        // Generator's trainable surface = Layers + per-layer BN.
        var generatorLayers = new List<ILayer<T>>();
        generatorLayers.AddRange(Layers);
        foreach (var bn in _genBNLayers) generatorLayers.Add(bn);
        var genParams = TapeTrainingStep<T>.CollectParameters(generatorLayers);

        int pacSize = _options.PacSize;
        int singleDim = _dataWidth + _condWidth;
        int genInputDim = _options.EmbeddingDimension + _condWidth;

        // Build the conditional generator input batch [numPacks * pacSize, genInputDim],
        // capturing the mask so the conditional cross-entropy term can be computed.
        int totalSamples = numPacks * pacSize;
        var noiseBatch = GenerateNoiseBatchTensor(totalSamples);
        var (condBatch, maskBatch) = SampleCondMaskBatch(totalSamples);
        var genInput = Engine.TensorConcatenate([noiseBatch, condBatch], axis: 1);

        // GPU-RESIDENT fast path — pack (genInput, maskBatch) into one persistent
        // input tensor so replay uses fresh noise + cond + mask each step. The
        // closure slices them back out. Conditional CE and -avgFake both captured
        // on the fused plan.
        var trainableGenLayers = generatorLayers.OfType<ITrainableLayer<T>>().ToList();
        if (trainableGenLayers.Count > 0)
        {
            int embedDim = _options.EmbeddingDimension;
            int condDim = _condWidth;
            // Fused-step input encodes [noise | cond | mask | Gumbel noise] side-by-side (columns), so a replayed
            // plan draws fresh Gumbel noise each step too.
            var gumbel = SyntheticTensorOps<T>.GumbelNoise([totalSamples, _dataWidth], _random);
            var fusedInput = Engine.TensorConcatenate([genInput, maskBatch, gumbel], axis: 1);
            var target = new Tensor<T>(new[] { 1 });
            // Closure-captured intermediates from Fwd's single generator pass —
            // reused by Loss so we don't re-run GeneratorForwardWithResidualBatched
            // and ApplyOutputActivationsBatched for the conditional-CE term.
            // Fwd/Loss ordering is guaranteed by the fused-step contract.
            Tensor<T>? capturedHead = null;
            Tensor<T>? capturedCond = null;
            Tensor<T>? capturedMask = null;
            Tensor<T> Fwd(Tensor<T> packed)
            {
                var gi = Engine.TensorSlice(packed, [0, 0], [totalSamples, embedDim + condDim]);
                var faked = GeneratorForwardWithResidualBatched(gi);
                var gumbelFromInput = Engine.TensorSlice(packed, [0, embedDim + 2 * condDim], [totalSamples, _dataWidth]);
                var act = ApplyOutputActivationsBatched(faked, gumbelFromInput);
                var condFromInput = Engine.TensorSlice(packed, [0, embedDim], [totalSamples, condDim]);
                capturedHead = faked;
                capturedCond = condFromInput;
                capturedMask = Engine.TensorSlice(packed, [0, embedDim + condDim], [totalSamples, condDim]);
                var withCond = Engine.TensorConcatenate([act, condFromInput], axis: 1);
                var pk = withCond.Reshape([numPacks, _packedInputDim]);
                return DiscriminatorForwardBatched(pk, false);
            }
            Tensor<T> Loss(Tensor<T> scores, Tensor<T> _)
            {
                var axes = Enumerable.Range(0, scores.Shape.Length).ToArray();
                var lossT = Engine.TensorNegate(Engine.ReduceMean(scores, axes, keepDims: false));
                if (_condWidth > 0 && _catOutputBlocks.Count > 0)
                {
                    var head = capturedHead;
                    var condFromInput = capturedCond;
                    var maskFromInput = capturedMask;
                    if (head is null || condFromInput is null || maskFromInput is null)
                        throw new InvalidOperationException(
                            "CTGAN fused step: generator intermediates were not captured by Fwd. " +
                            "This indicates the fused-step framework called the loss closure before " +
                            "the forward closure, violating its documented Fwd-then-Loss ordering.");
                    var ce = ConditionalCrossEntropy(head, condFromInput, maskFromInput);
                    lossT = Engine.TensorAdd(lossT, ce);
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
                extraTensors: genParams);
        }
    }

    /// <summary>
    /// Builds the packed real + fake batches for one critic step.
    /// Each pack groups <see cref="_options.PacSize"/> independent samples
    /// (PacGAN, Lin et al. 2017) so the critic sees the joint distribution
    /// of pacSize samples instead of one sample at a time — this is what
    /// prevents mode collapse on tabular distributions with heavily
    /// imbalanced categorical columns.
    /// </summary>
    private (Tensor<T> realPacked, Tensor<T> fakePacked) BuildPackedRealAndFakeBatches(
        Matrix<T> transformedData,
        int numPacks)
    {
        int pacSize = _options.PacSize;
        int singleDim = _dataWidth + _condWidth;
        int totalSamples = numPacks * pacSize;

        // Per-sample noise + conditional vectors that drive the generator.
        var noiseBatch = GenerateNoiseBatchTensor(totalSamples);
        var (condBatch, rowIndices) = SampleConditionalBatchWithRows(totalSamples);
        var genInput = Engine.TensorConcatenate([noiseBatch, condBatch], axis: 1);

        var fakeFlat = GeneratorForwardWithResidualBatched(genInput);
        var fakeActivated = ApplyOutputActivationsBatched(fakeFlat, SyntheticTensorOps<T>.GumbelNoise(fakeFlat._shape, _random));
        // [totalSamples, dataWidth] concat [totalSamples, condWidth] -> [totalSamples, singleDim]
        var fakeSingles = Engine.TensorConcatenate([fakeActivated, condBatch], axis: 1);

        // Real-sample matrix: random rows of transformedData paired with each sample's condVector.
        if (_realDataTensor is null || !ReferenceEquals(_realDataSource, transformedData))
        {
            _realDataTensor = Tensor<T>.FromMatrix(transformedData);
            _realDataSource = transformedData;
        }
        var realFlat = SyntheticTensorOps<T>.GatherRows(_realDataTensor, rowIndices, _dataWidth);
        var realSingles = Engine.TensorConcatenate([realFlat, condBatch], axis: 1);

        // Pack into [numPacks, pacSize * singleDim].
        var realPacked = realSingles.Reshape([numPacks, _packedInputDim]);
        var fakePacked = fakeSingles.Reshape([numPacks, _packedInputDim]);
        return (realPacked, fakePacked);
    }

    /// <summary>
    /// Batched Box-Muller standard-normal noise generator
    /// (matches <see cref="GenerativeAdversarialNetwork{T}.GenerateRandomNoiseTensor"/>).
    /// </summary>
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

        // Interleave z1[i], z2[i] and drop the spare value when the element count is odd.
        var interleaved = Engine.TensorStack([z1, z2], 1).Reshape([2 * halfElements]);
        var noise = 2 * halfElements == totalElements
            ? interleaved
            : Engine.TensorSlice(interleaved, [0], [totalElements]);
        return noise.Reshape([batchSize, embedDim]);
    }

    private (Tensor<T> CondBatch, int[] RowIndices) SampleConditionalBatchWithRows(int batchSize)
    {
        var rowIndices = new int[batchSize];
        if (_sampler is null) return (new Tensor<T>([batchSize, _condWidth]), rowIndices);

        var conds = new List<Vector<T>>(batchSize);
        for (int b = 0; b < batchSize; b++)
        {
            var (cv, rowIndex) = _sampler.SampleConditionAndRow();
            conds.Add(cv);
            rowIndices[b] = rowIndex;
        }

        return (SyntheticTensorOps<T>.StackRows(conds, _condWidth), rowIndices);
    }

    /// <summary>
    /// Batched, tape-tracked version of <see cref="GeneratorForwardWithResidual"/>.
    /// Uses <see cref="Engine.TensorConcatenate"/> for skip connections and
    /// <see cref="Engine.ReLU"/> for hidden activation. Same paper-faithful
    /// residual + BN architecture as the per-sample variant.
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
        int numHiddenLayers = Layers.Count - 1;
        for (int i = 0; i < numHiddenLayers; i++)
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
    /// Batched, tape-tracked critic forward pass.
    /// Uses <see cref="Engine.LeakyReLU"/> (alpha=0.2 per paper) and
    /// <see cref="DropoutLayer{T}"/>'s tape-tracked Forward when training.
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
    /// Batched per-column output activations (Xu et al. 2019 §3.1): per the
    /// VGM column encoding, continuous columns emit a Tanh-bounded mode
    /// value followed by softmax over mode probabilities; categorical
    /// columns emit a softmax over the one-hot block. With Gumbel noise those
    /// softmaxes are Gumbel-softmax samples at <see cref="CTGANOptions{T}.GumbelTemperature"/>,
    /// as sdv-dev/CTGAN's training does. Runs through
    /// tape-tracked Engine ops so backprop flows from critic to generator.
    /// </summary>
    private Tensor<T> ApplyOutputActivationsBatched(Tensor<T> output, Tensor<T>? gumbelNoise = null)
        => SyntheticTensorOps<T>.ApplyOutputActivationsBatched(
            output, _transformer, _columns.Count, gumbelNoise, _options.GumbelTemperature);

    #endregion

    #region Gradient Penalty

    #endregion

    #region Forward Passes

    /// <summary>
    /// Generator forward pass with residual connections: each hidden layer receives
    /// both the previous output and the original input concatenated.
    /// </summary>
    private Tensor<T> GeneratorForwardWithResidual(Tensor<T> input)
    {
        var current = input;

        // Find dense layers (skip batch norm which we handle separately)
        var denseLayers = new List<ILayer<T>>();
        foreach (var layer in Layers)
        {
            denseLayers.Add(layer);
        }

        int numHiddenLayers = denseLayers.Count - 1;

        for (int i = 0; i < numHiddenLayers; i++)
        {
            if (i > 0)
            {
                current = ConcatTensors(current, input);
            }

            current = denseLayers[i].Forward(current);

            if (i < _genBNLayers.Count)
            {
                current = _genBNLayers[i].Forward(current);
            }

            current = ApplyReLU(current);
        }

        // Final layer with residual connection
        current = ConcatTensors(current, input);
        current = denseLayers[^1].Forward(current);

        return ApplyOutputActivations(current);
    }

    #endregion

    #region Backward Passes

    #endregion

    #region Activation Functions

    private Tensor<T> ApplyReLU(Tensor<T> input) => SyntheticTensorOps<T>.ReLU(input);

    private Tensor<T> ApplyOutputActivations(Tensor<T> output)
    {
        if (_transformer is null) return output;
        return SyntheticTensorOps<T>.ApplyOutputActivations(output, _transformer, Columns.Count);
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
            throw new ArgumentException("Data matrix must not be empty.", nameof(data));
        if (columns.Count == 0)
            throw new ArgumentException("Column metadata list must not be empty.", nameof(columns));
        if (columns.Count != data.Columns)
            throw new ArgumentException(
                $"Column metadata count ({columns.Count}) must match data column count ({data.Columns}).", nameof(columns));
        if (epochs <= 0)
            throw new ArgumentOutOfRangeException(nameof(epochs), "Epochs must be positive.");
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
        for (int i = 0; i < length; i++) v[i] = SampleStandardNormal();
        return v;
    }

    #endregion

    #region Helpers

    private static Vector<T> ConcatVectors(Vector<T> a, Vector<T> b) => SyntheticTensorOps<T>.ConcatVectors(a, b);

    private static Tensor<T> VectorToTensor(Vector<T> v) => Tensor<T>.FromVector(v);

    private static Tensor<T> ConcatTensors(Tensor<T> a, Tensor<T> b) => SyntheticTensorOps<T>.Concat1D(a, b);

    #endregion

}
