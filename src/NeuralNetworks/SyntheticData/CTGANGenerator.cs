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
public class CTGANGenerator<T> : NeuralNetworkBase<T>, ISyntheticTabularGenerator<T>
{
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

    // Cached pre-activations for proper backward passes
    private readonly List<Tensor<T>> _genPreActivations = new();
    private readonly List<Tensor<T>> _discPreActivations = new();

    // Whether custom layers are being used (disables residual connection logic)
    private bool _usingCustomLayers;

    // Pre-allocated training buffers to eliminate per-row GC pressure
    private Tensor<T>? _oneGrad;
    private Tensor<T>? _negOneGrad;
    private Vector<T>? _packedRealBuf;
    private Vector<T>? _packedFakeBuf;
    private Vector<T>? _noiseBuf;
    private Vector<T>? _genInputBuf;
    private Vector<T>? _realSingleBuf;
    private Vector<T>? _fakeSingleBuf;
    private Vector<T>? _realRowBuf;
    private Vector<T>? _fakeRowBuf;
    private Vector<T>? _interpolatedBuf;
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
    /// <param name="maxGradNorm">Maximum gradient norm for clipping (default 5.0).</param>
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

    public CTGANGenerator(
        NeuralNetworkArchitecture<T> architecture,
        CTGANOptions<T>? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
        ILossFunction<T>? lossFunction = null,
        double maxGradNorm = 5.0)
        : base(architecture, lossFunction ?? NeuralNetworkHelper<T>.GetDefaultLossFunction(architecture.TaskType), maxGradNorm)
    {
        _options = options ?? new CTGANOptions<T>();
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
        if (TryForwardGpuOptimized(input, out var gpuResult))
            return gpuResult;

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

    /// <inheritdoc />
    public override void Train(Tensor<T> input, Tensor<T> expectedOutput)
    {
        Tensor<T> prediction = Predict(input);
        LastLoss = _lossFunction.CalculateLoss(prediction.ToVector(), expectedOutput.ToVector());
        Tensor<T> error = prediction.Subtract(expectedOutput);
        UpdateNetworkParameters();
    }

    private void UpdateNetworkParameters()
    {
        _generatorOptimizer.UpdateParameters(Layers);
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

    #endregion

    #region ISyntheticTabularGenerator<T> Implementation

    /// <summary>
    /// Fits the CTGAN generator to the provided real tabular data.
    /// </summary>
    /// <param name="data">The real data matrix where each row is a sample and each column is a feature.</param>
    /// <param name="columns">Metadata describing each column (type, categories, etc.).</param>
    /// <param name="epochs">Number of training epochs.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This is the "learning" step. The generator studies your real data:
    /// 1. Fits VGM transformer for data normalization
    /// 2. Builds conditional vectors for training-by-sampling
    /// 3. Trains generator and discriminator in alternating WGAN-GP steps
    /// After fitting, call Generate() to create new synthetic rows.
    /// </para>
    /// </remarks>
    public void Fit(Matrix<T> data, IReadOnlyList<ColumnMetadata> columns, int epochs)
    {
        ValidateFitInputs(data, columns, epochs);

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
        RebuildLayersWithActualDimensions(genInputDim, _dataWidth, _packedInputDim);

        // Step 5: Transform data
        var transformedData = _transformer.Transform(data);

        // Step 6: Training loop
        T lr = NumOps.FromDouble(_options.LearningRate);
        int pacSize = _options.PacSize;
        int batchSize = _options.BatchSize;
        int numPacks = Math.Max(1, batchSize / pacSize);
        int numBatches = Math.Max(1, data.Rows / (numPacks * pacSize));

        for (int epoch = 0; epoch < epochs; epoch++)
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
    public async Task FitAsync(Matrix<T> data, IReadOnlyList<ColumnMetadata> columns, int epochs, CancellationToken ct = default)
    {
        ValidateFitInputs(data, columns, epochs);

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
            RebuildLayersWithActualDimensions(genInputDim, _dataWidth, _packedInputDim);

            // Pre-allocate training buffers to avoid per-row GC pressure
            InitializeTrainingBuffers(genInputDim);

            var transformedData = _transformer.Transform(data);

            T lr = NumOps.FromDouble(_options.LearningRate);
            int pacSize = _options.PacSize;
            int batchSize = _options.BatchSize;
            int numPacks = Math.Max(1, batchSize / pacSize);
            int numBatches = Math.Max(1, data.Rows / (numPacks * pacSize));

            for (int epoch = 0; epoch < epochs; epoch++)
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

    /// <summary>
    /// Fills a pre-allocated vector with standard normal random values (Box-Muller).
    /// </summary>
    private void FillStandardNormal(Vector<T> buffer)
    {
        for (int i = 0; i < buffer.Length; i++)
        {
            double u1 = 1.0 - _random.NextDouble();
            double u2 = _random.NextDouble();
            double z = Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2);
            buffer[i] = NumOps.FromDouble(z);
        }
    }

    /// <summary>
    /// Fills a pre-allocated vector with a row from the data matrix.
    /// </summary>
    private static void FillRow(Matrix<T> data, int row, Vector<T> buffer)
    {
        int cols = Math.Min(data.Columns, buffer.Length);
        for (int j = 0; j < cols; j++)
            buffer[j] = data[row, j];
    }

    /// <summary>
    /// Concatenates two vectors into a pre-allocated destination buffer.
    /// </summary>
    private static void ConcatInto(Vector<T> a, Vector<T> b, Vector<T> dest)
    {
        int aLen = a.Length;
        for (int i = 0; i < aLen; i++) dest[i] = a[i];
        int bLen = Math.Min(b.Length, dest.Length - aLen);
        for (int i = 0; i < bLen; i++) dest[aLen + i] = b[i];
    }

    /// <summary>
    /// Fills a pre-allocated vector from a tensor.
    /// </summary>
    private static void FillFromTensor(Tensor<T> src, Vector<T> dest)
    {
        int len = Math.Min(src.Length, dest.Length);
        for (int i = 0; i < len; i++) dest[i] = src[i];
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

        using var tape = new GradientTape<T>();
        var discParams = TapeTrainingStep<T>.CollectParameters(_discLayers);

        var realScores = DiscriminatorForwardBatched(realPacked, isTraining: true);
        var fakeScores = DiscriminatorForwardBatched(fakePacked, isTraining: true);

        var allAxes = Enumerable.Range(0, realScores.Shape.Length).ToArray();
        var avgReal = Engine.ReduceMean(realScores, allAxes, keepDims: false);
        var avgFake = Engine.ReduceMean(fakeScores, allAxes, keepDims: false);
        // WGAN critic minimizes E[D(fake)] - E[D(real)].
        var wassersteinLoss = Engine.TensorSubtract(avgFake, avgReal);
        var gradientPenalty = ComputeGradientPenalty(realPacked, fakePacked);
        var weightedGradientPenalty = Engine.TensorMultiplyScalar(
            gradientPenalty,
            NumOps.FromDouble(_options.GradientPenaltyWeight));
        var lossTensor = Engine.TensorAdd(wassersteinLoss, weightedGradientPenalty);

        var grads = tape.ComputeGradients(lossTensor, discParams);
        T lossValue = lossTensor.Length > 0 ? lossTensor[0] : NumOps.Zero;

        Tensor<T> ComputeForward(Tensor<T> inp, Tensor<T> _) => DiscriminatorForwardBatched(inp, true);
        Tensor<T> RecomputeLoss(Tensor<T> pred, Tensor<T> _)
        {
            var recomputedAvgReal = Engine.ReduceMean(pred, allAxes, keepDims: false);
            var recomputedFakeScores = DiscriminatorForwardBatched(fakePacked, true);
            var recomputedAvgFake = Engine.ReduceMean(recomputedFakeScores, allAxes, keepDims: false);
            var recomputedWasserstein = Engine.TensorSubtract(recomputedAvgFake, recomputedAvgReal);
            var recomputedGradientPenalty = ComputeGradientPenalty(realPacked, fakePacked);
            return Engine.TensorAdd(
                recomputedWasserstein,
                Engine.TensorMultiplyScalar(recomputedGradientPenalty, NumOps.FromDouble(_options.GradientPenaltyWeight)));
        }

        var context = new TapeStepContext<T>(
            discParams, grads, lossValue,
            realPacked, realPacked, ComputeForward, RecomputeLoss,
            parameterBuffer: null);
        _discriminatorOptimizer.Step(context);
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
    /// Conditional generator loss term (Xu 2019 §4.3): cross-entropy between the
    /// conditional one-hot and the generator's output distribution for the
    /// conditioned discrete column, masked to that column. With a one-hot condition
    /// inside the masked block this reduces to -log p(selected category), averaged
    /// over the batch — the term that forces the generator to actually honour the
    /// conditional vector instead of ignoring it.
    /// </summary>
    private Tensor<T> ConditionalCrossEntropy(Tensor<T> fakeActivated, Tensor<T> condBatch, Tensor<T> maskBatch)
    {
        int batch = fakeActivated.Shape[0];
        // Gather the generator's categorical softmax blocks in categorical-column
        // order → [batch, condWidth], aligning with the conditional-vector layout.
        var blocks = new List<Tensor<T>>(_catOutputBlocks.Count);
        foreach (var (offset, width) in _catOutputBlocks)
            blocks.Add(Engine.TensorSlice(fakeActivated, [0, offset], [batch, width]));
        var alignedGen = blocks.Count == 1 ? blocks[0] : Engine.TensorConcatenate(blocks.ToArray(), axis: 1);

        // CE = -mean_b sum_j mask*cond*log(alignedGen + eps).
        var logp = Engine.TensorLog(Engine.TensorAddScalar(alignedGen, NumOps.FromDouble(1e-8)));
        var masked = Engine.TensorMultiply(Engine.TensorMultiply(maskBatch, condBatch), logp);
        var perRow = Engine.ReduceSum(masked, [1], keepDims: false);     // [batch]
        var meanCe = Engine.ReduceMean(perRow, [0], keepDims: false);     // scalar
        return Engine.TensorNegate(meanCe);
    }

    private void TrainGeneratorStepBatched(Matrix<T> transformedData, int numPacks)
    {
        if (_sampler is null) return;

        using var tape = new GradientTape<T>();

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
        var noiseBatch = GenerateNoiseBatchTensor(numPacks * pacSize);
        var (condBatch, maskBatch) = SampleCondMaskBatch(numPacks * pacSize);
        var genInput = Engine.TensorConcatenate([noiseBatch, condBatch], axis: 1);

        // Forward through generator → produces [numPacks * pacSize, dataWidth].
        var fakeFlat = GeneratorForwardWithResidualBatched(genInput);
        var fakeActivated = ApplyOutputActivationsBatched(fakeFlat);

        // PacGAN packing: reshape [numPacks * pacSize, singleDim] -> [numPacks, packedInputDim].
        // Re-attach the conditional vector to each sample, then pack.
        var fakeWithCond = Engine.TensorConcatenate([fakeActivated, condBatch], axis: 1);
        var fakePacked = fakeWithCond.Reshape([numPacks, _packedInputDim]);

        var fakeScores = DiscriminatorForwardBatched(fakePacked, isTraining: false);
        var allAxes = Enumerable.Range(0, fakeScores.Shape.Length).ToArray();
        var avgFake = Engine.ReduceMean(fakeScores, allAxes, keepDims: false);
        // Generator loss = -E[D(G(z, c))] + conditional cross-entropy (Xu 2019 §4.3).
        var lossTensor = Engine.TensorNegate(avgFake);
        if (_condWidth > 0 && _catOutputBlocks.Count > 0)
        {
            var ce = ConditionalCrossEntropy(fakeActivated, condBatch, maskBatch);
            lossTensor = Engine.TensorAdd(lossTensor, ce);
        }

        var grads = tape.ComputeGradients(lossTensor, genParams);
        T lossValue = lossTensor.Length > 0 ? lossTensor[0] : NumOps.Zero;

        Tensor<T> ComputeForward(Tensor<T> inp, Tensor<T> _)
        {
            var faked = GeneratorForwardWithResidualBatched(inp);
            var act = ApplyOutputActivationsBatched(faked);
            var withCond = Engine.TensorConcatenate([act, condBatch], axis: 1);
            var packed = withCond.Reshape([numPacks, _packedInputDim]);
            return DiscriminatorForwardBatched(packed, false);
        }
        Tensor<T> RecomputeLoss(Tensor<T> pred, Tensor<T> _) => Engine.TensorNegate(Engine.ReduceMean(pred, allAxes, keepDims: false));

        var context = new TapeStepContext<T>(
            genParams, grads, lossValue,
            genInput, genInput, ComputeForward, RecomputeLoss,
            parameterBuffer: null);
        _generatorOptimizer.Step(context);
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
        var fakeActivated = ApplyOutputActivationsBatched(fakeFlat);
        // [totalSamples, dataWidth] concat [totalSamples, condWidth] -> [totalSamples, singleDim]
        var fakeSingles = Engine.TensorConcatenate([fakeActivated, condBatch], axis: 1);

        // Real-sample matrix: random rows of transformedData paired with each sample's condVector.
        var realFlat = new Tensor<T>([totalSamples, _dataWidth]);
        for (int s = 0; s < totalSamples; s++)
        {
            int rowIdx = rowIndices[s];
            int cols = Math.Min(_dataWidth, transformedData.Columns);
            for (int j = 0; j < cols; j++) realFlat[s, j] = transformedData[rowIdx, j];
        }
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
        return new Tensor<T>(noiseData, [batchSize, embedDim]);
    }

    /// <summary>
    /// Samples a batch of conditional vectors from the CTGAN sampler (one
    /// per sample). The conditional vector encodes a column-condition for
    /// training-by-sampling (Xu et al. 2019 §3.2) so the generator learns
    /// to honour rare categories.
    /// </summary>
    private Tensor<T> SampleConditionalBatchTensor(int batchSize)
    {
        if (_sampler is null) return new Tensor<T>([batchSize, _condWidth]);

        var batch = new Tensor<T>([batchSize, _condWidth]);
        for (int b = 0; b < batchSize; b++)
        {
            var (cv, _) = _sampler.SampleConditionAndRow();
            int cols = Math.Min(_condWidth, cv.Length);
            for (int j = 0; j < cols; j++) batch[b, j] = cv[j];
        }
        return batch;
    }

    private (Tensor<T> CondBatch, int[] RowIndices) SampleConditionalBatchWithRows(int batchSize)
    {
        var batch = new Tensor<T>([batchSize, _condWidth]);
        var rowIndices = new int[batchSize];
        if (_sampler is null) return (batch, rowIndices);

        for (int b = 0; b < batchSize; b++)
        {
            var (cv, rowIndex) = _sampler.SampleConditionAndRow();
            int cols = Math.Min(_condWidth, cv.Length);
            for (int j = 0; j < cols; j++) batch[b, j] = cv[j];
            rowIndices[b] = rowIndex;
        }

        return (batch, rowIndices);
    }

    private Tensor<T> ComputeGradientPenalty(Tensor<T> realPacked, Tensor<T> fakePacked)
    {
        int batchSize = Math.Max(1, realPacked.Shape[0]);
        int elementsPerSample = Math.Max(1, realPacked.Length / batchSize);

        var epsilon = Engine.TensorRandomUniformRange<T>([batchSize, 1], NumOps.Zero, NumOps.One);
        var epsilonBroadcast = Engine.TensorTile(epsilon, [1, elementsPerSample]).Reshape([realPacked.Length]);
        var ones = new Tensor<T>([realPacked.Length]);
        Engine.TensorFill(ones, NumOps.One);
        var oneMinusEpsilon = Engine.TensorSubtract(ones, epsilonBroadcast);

        var realFlat = realPacked.Reshape([realPacked.Length]);
        var fakeFlat = fakePacked.Reshape([fakePacked.Length]);
        var interpolatedFlat = Engine.TensorAdd(
            Engine.TensorMultiply(epsilonBroadcast, realFlat),
            Engine.TensorMultiply(oneMinusEpsilon, fakeFlat));
        var interpolated = interpolatedFlat.Reshape(realPacked._shape);

        Tensor<T> inputGradients;
        using (var gradientTape = new GradientTape<T>())
        {
            var scores = DiscriminatorForwardBatched(interpolated, true);
            var scoreAxes = Enumerable.Range(0, scores.Shape.Length).ToArray();
            var summedScores = Engine.ReduceSum(scores, scoreAxes, keepDims: false);
            var gradients = gradientTape.ComputeGradients(summedScores, [interpolated]);
            inputGradients = gradients.TryGetValue(interpolated, out var gradient)
                ? gradient
                : new Tensor<T>(interpolated._shape);
        }

        var gradientsReshaped = inputGradients.Reshape([batchSize, elementsPerSample]);
        var gradientSquared = Engine.TensorMultiply(gradientsReshaped, gradientsReshaped);
        var gradientNormSquared = Engine.ReduceSum(gradientSquared, [1], keepDims: false);
        var gradientNorm = Engine.TensorSqrt(Engine.TensorAddScalar(gradientNormSquared, NumOps.FromDouble(1e-12)));
        var targetNorm = new Tensor<T>(gradientNorm._shape);
        Engine.TensorFill(targetNorm, NumOps.One);
        var deviation = Engine.TensorSubtract(gradientNorm, targetNorm);
        var penalty = Engine.TensorMultiply(deviation, deviation);
        var penaltyAxes = Enumerable.Range(0, penalty.Shape.Length).ToArray();
        return Engine.ReduceMean(penalty, penaltyAxes, keepDims: false);
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
    /// Batched, tape-tracked version of <see cref="DiscriminatorForward"/>.
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
    /// columns emit a softmax over the one-hot block. Runs through
    /// tape-tracked Engine ops so backprop flows from critic to generator.
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
                var valueSlice = Engine.TensorSlice(output, [0, idx], [batch, 1]);
                blocks.Add(Engine.TensorTanh(valueSlice));
                idx++;

                int numModes = transform.Width - 1;
                int modeLength = Math.Min(numModes, totalWidth - idx);
                if (modeLength > 0)
                {
                    var modeSlice = Engine.TensorSlice(output, [0, idx], [batch, modeLength]);
                    blocks.Add(Engine.Softmax(modeSlice, axis: 1));
                    idx += modeLength;
                }
            }
            else
            {
                int blockLength = Math.Min(transform.Width, totalWidth - idx);
                var catSlice = Engine.TensorSlice(output, [0, idx], [batch, blockLength]);
                blocks.Add(Engine.Softmax(catSlice, axis: 1));
                idx += blockLength;
            }
        }

        if (idx < totalWidth)
        {
            var tail = Engine.TensorSlice(output, [0, idx], [batch, totalWidth - idx]);
            blocks.Add(Engine.TensorTanh(tail));
        }

        if (blocks.Count == 1) return blocks[0];
        return Engine.TensorConcatenate(blocks.ToArray(), axis: 1);
    }

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
        _genPreActivations.Clear();
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

            _genPreActivations.Add(CloneTensor(current));
            current = ApplyReLU(current);
        }

        // Final layer with residual connection
        current = ConcatTensors(current, input);
        current = denseLayers[^1].Forward(current);

        return ApplyOutputActivations(current);
    }

    /// <summary>
    /// Discriminator forward pass with LeakyReLU and dropout.
    /// </summary>
    private Tensor<T> DiscriminatorForward(Tensor<T> input, bool isTraining)
    {
        _discPreActivations.Clear();
        var current = input;
        int layerIdx = 0;

        for (int i = 0; i < _discLayers.Count; i++)
        {
            if (_discLayers[i] is DropoutLayer<T> dropout)
            {
                if (isTraining)
                {
                    current = dropout.Forward(current);
                }
                continue;
            }

            bool isLastDense = (layerIdx == _discLayerDims.Count - 1);

            current = _discLayers[i].Forward(current);

            if (!isLastDense)
            {
                _discPreActivations.Add(CloneTensor(current));
                current = ApplyLeakyReLU(current);
            }

            layerIdx++;
        }

        return current;
    }

    #endregion

    #region Backward Passes

    private void UpdateGeneratorParameters(T learningRate)
    {
        foreach (var layer in Layers)
        {
            layer.UpdateParameters(learningRate);
        }
        foreach (var bn in _genBNLayers)
        {
            bn.UpdateParameters(learningRate);
        }
    }

    private void UpdateDiscriminatorParameters(T learningRate)
    {
        foreach (var layer in _discLayers)
        {
            layer.UpdateParameters(learningRate);
        }
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

    private Tensor<T> ApplyReLUDerivative(Tensor<T> gradOutput, Tensor<T> preActivation)
    {
        int len = Math.Min(gradOutput.Length, preActivation.Length);
        var result = new Tensor<T>(gradOutput._shape);
        for (int i = 0; i < len; i++)
        {
            result[i] = NumOps.GreaterThan(preActivation[i], NumOps.Zero) ? gradOutput[i] : NumOps.Zero;
        }
        return result;
    }

    private Tensor<T> ApplyLeakyReLU(Tensor<T> input)
    {
        var result = new Tensor<T>(input._shape);
        T slope = NumOps.FromDouble(0.2);
        for (int i = 0; i < input.Length; i++)
        {
            double val = NumOps.ToDouble(input[i]);
            result[i] = val > 0 ? input[i] : NumOps.Multiply(slope, input[i]);
        }
        return result;
    }

    private Tensor<T> ApplyLeakyReLUDerivative(Tensor<T> gradOutput, Tensor<T> preActivation)
    {
        int len = Math.Min(gradOutput.Length, preActivation.Length);
        var result = new Tensor<T>(gradOutput._shape);
        T slope = NumOps.FromDouble(0.2);
        for (int i = 0; i < len; i++)
        {
            if (NumOps.GreaterThan(preActivation[i], NumOps.Zero))
            {
                result[i] = gradOutput[i];
            }
            else
            {
                result[i] = NumOps.Multiply(slope, gradOutput[i]);
            }
        }
        return result;
    }

    private Tensor<T> ApplyOutputActivations(Tensor<T> output)
    {
        if (_transformer is null) return output;

        var result = new Tensor<T>(output._shape);
        int idx = 0;

        for (int col = 0; col < Columns.Count && idx < output.Length; col++)
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

        return result;
    }

    private void ApplySoftmax(Tensor<T> input, Tensor<T> output, ref int idx, int count)
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

    #region Gradient Safety Utilities

    private void SanitizeTensor(Tensor<T> tensor)
    {
        for (int i = 0; i < tensor.Length; i++)
        {
            double v = NumOps.ToDouble(tensor[i]);
            if (double.IsNaN(v) || double.IsInfinity(v))
            {
                tensor[i] = NumOps.Zero;
            }
        }
    }

    private Tensor<T> ClipGradientNorm(Tensor<T> grad, double maxNorm)
    {
        if (maxNorm <= 0) return grad;

        double normSq = 0;
        for (int i = 0; i < grad.Length; i++)
        {
            double v = NumOps.ToDouble(grad[i]);
            normSq += v * v;
        }

        double norm = Math.Sqrt(normSq);
        if (norm <= maxNorm) return grad;

        double scale = maxNorm / norm;
        var clipped = new Tensor<T>(grad._shape);
        for (int i = 0; i < grad.Length; i++)
        {
            clipped[i] = NumOps.FromDouble(NumOps.ToDouble(grad[i]) * scale);
        }
        return clipped;
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
                { "GeneratorLayerCount", Layers.Count },
                { "GeneratorLayerTypes", Layers.Select(l => l.GetType().Name).ToArray() }
            },
            ModelData = SerializeForMetadata()
        };
    }

    /// <inheritdoc/>
    protected override void SerializeNetworkSpecificData(BinaryWriter writer)
    {
        writer.Write(_options.EmbeddingDimension);
        writer.Write(_options.GeneratorDimensions.Length);
        foreach (var dim in _options.GeneratorDimensions) writer.Write(dim);
        writer.Write(_options.DiscriminatorDimensions.Length);
        foreach (var dim in _options.DiscriminatorDimensions) writer.Write(dim);
        writer.Write(_options.BatchSize);
        writer.Write(_options.LearningRate);
        writer.Write(_options.GradientPenaltyWeight);
        writer.Write(_options.PacSize);
        writer.Write(_options.VGMModes);
        writer.Write(_options.DiscriminatorDropout);
    }

    /// <inheritdoc/>
    protected override void DeserializeNetworkSpecificData(BinaryReader reader)
    {
        // Options are reconstructed from serialized data
    }

    /// <inheritdoc/>
    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance()
    {
        return new CTGANGenerator<T>(
            Architecture,
            _options,
            _generatorOptimizer,
            _lossFunction);
    }

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
        double sum = 0, min = double.MaxValue, max = double.MinValue;
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
            double diff = NumOps.ToDouble(data[row, colIndex]) - mean;
            sumSqDiff += diff * diff;
        }
        double std = n > 1 ? Math.Sqrt(sumSqDiff / (n - 1)) : 1.0;
        if (std < 1e-10) std = 1e-10;
        meta.Min = min; meta.Max = max; meta.Mean = mean; meta.Std = std;
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
        for (int i = 0; i < length; i++) v[i] = SampleStandardNormal();
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

    private static Vector<T> TensorToVector(Tensor<T> t, int length)
    {
        var v = new Vector<T>(length);
        int copyLen = Math.Min(length, t.Length);
        for (int i = 0; i < copyLen; i++) v[i] = t[i];
        return v;
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
