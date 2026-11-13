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

    // Generator batch normalization layers (auxiliary, always created to match Layers)
    private readonly List<BatchNormalizationLayer<T>> _genBNLayers = new();

    // Discriminator layers (auxiliary, not user-overridable)
    private readonly List<ILayer<T>> _discLayers = new();
    private readonly List<DropoutLayer<T>> _discDropoutLayers = new();
    private readonly List<(int InputSize, int OutputSize)> _discLayerDims = new();
    private int _discLastHiddenDim;

    private bool _hasClassificationTarget;
    // Auxiliary classifier head (separate from discriminator's critic head)
    private FullyConnectedLayer<T>? _classifierHead;
    private int _numClasses;
    private int _targetColTransformOffset;

    // Whether custom layers are being used (disables residual connection logic)
    private bool _usingCustomLayers;

    // Pre-allocated training buffers to avoid per-row GC pressure
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
        base.Options = _options;
        _lossFunction = lossFunction ?? NeuralNetworkHelper<T>.GetDefaultLossFunction(architecture.TaskType);
        _optimizer = optimizer ?? new AdamOptimizer<T, Tensor<T>, Tensor<T>>(this,
            new AdamOptimizerOptions<T, Tensor<T>, Tensor<T>>
            {
                InitialLearningRate = _options.LearningRate
            });
        _random = _options.Seed.HasValue
            ? RandomHelper.CreateSeededRandom(_options.Seed.Value)
            : RandomHelper.CreateSecureRandom();

        InitializeLayers();
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
                _genBNLayers.Add(new BatchNormalizationLayer<T>(dim));
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
                _genBNLayers.Add(new BatchNormalizationLayer<T>(dim));
            }
        }

        // Discriminator is always rebuilt with actual dimensions
        _discLayers.Clear();
        _discDropoutLayers.Clear();
        _discLayerDims.Clear();
        _discLayers.AddRange(LayerHelper<T>.CreateDefaultCTABGANPlusDiscriminatorLayers(
            discInputDim, _options.DiscriminatorDimensions, _options.DiscriminatorDropout));

        BuildDiscriminatorDimensionMap(discInputDim);

        // Build auxiliary classifier head from last hidden dim
        var identity = new IdentityActivation<T>() as IActivationFunction<T>;
        _classifierHead = new FullyConnectedLayer<T>(_discLastHiddenDim, _numClasses, identity);
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
        _discLastHiddenDim = prevDim;
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

    /// <summary>
    /// Trains the CTAB-GAN+ network using the provided input and expected output.
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
        // CTAB-GAN+ is trained adversarially through Fit() / FitAsync (alternating
        // critic and generator updates with the auxiliary classifier and
        // information loss). The NeuralNetworkBase supervised Train(input,
        // expected) contract does not map onto a GAN's minimax objective —
        // silently running a forward + loss but skipping every parameter
        // update would report a misleading "training" loss while making zero
        // learning progress, which is worse than failing fast.
        throw new NotSupportedException(
            "CTAB-GAN+ is trained adversarially; the supervised Train(input, expected) " +
            "contract does not map onto its minimax objective. Use Fit(IEnumerable<Tensor<T>>) " +
            "or FitAsync to drive the alternating discriminator/generator loop with the " +
            "auxiliary classifier and information losses.");
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

        // Step 1: Fit transformer
        _transformer = new TabularDataTransformer<T>(_options.VGMModes, _random);
        _transformer.Fit(data, _columns);
        _dataWidth = _transformer.TransformedWidth;

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
        BuildFromConstructionSequence(() => RebuildLayersWithActualDimensions(genInputDim, _dataWidth, _packedInputDim));
        InitializeTrainingBuffers(genInputDim);

        // Step 6: Transform data
        var transformedData = _transformer.Transform(data);

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
                for (int dStep = 0; dStep < _options.DiscriminatorSteps; dStep++)
                {
                    TrainNativeBatch(transformedData, numPacks, critic: true);
                }

                TrainNativeBatch(transformedData, numPacks, critic: false);
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

            _sampler = new CTGANDataSampler<T>(_random);
            _sampler.Fit(data, _columns);
            _condWidth = _sampler.ConditionalVectorWidth;

            IdentifyTargetColumn(_columns);

            _packedInputDim = (_dataWidth + _condWidth) * _options.PacSize;

            int genInputDim = _options.EmbeddingDimension + _condWidth;
            BuildFromConstructionSequence(() => RebuildLayersWithActualDimensions(genInputDim, _dataWidth, _packedInputDim));
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
                        TrainNativeBatch(transformedData, numPacks, critic: true);
                    TrainNativeBatch(transformedData, numPacks, critic: false);
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

        // CTAB-GAN+ samples like sdv-dev/CTGAN: each mode and category block is drawn from its distribution
        // (gumbel_softmax then argmax), not set to its most likely entry.
        // Decoded in bounded blocks, so only one block's intermediates are alive at a time.
        TabularDataTransformer<T> transformer = _transformer;
        int columnCount = _columns.Count;
        return SyntheticTensorOps<T>.DecodeInBlocks(generated, block => transformer.InverseTransform(
            SyntheticTensorOps<T>.SampleCategoricalBlocks(block, transformer, columnCount, _random).ToMatrix()));
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
        _hasClassificationTarget = targetIdx >= 0 && targetIdx < columns.Count && columns[targetIdx].IsCategorical;

        if (targetIdx >= 0 && targetIdx < columns.Count && columns[targetIdx].IsCategorical)
        {
            _numClasses = Math.Max(columns[targetIdx].NumCategories, 2);

            // Find the transform offset for the target column
            if (_transformer is not null)
            {
                _targetColTransformOffset = 0;
                for (int c = 0; c < targetIdx && c < columns.Count; c++)
                {
                    var info = _transformer.GetTransformInfo(c);
                    _targetColTransformOffset += info.Width;
                }
            }
        }
        else
        {
            _numClasses = 2; // Binary fallback
            _targetColTransformOffset = 0;
        }
    }

    #endregion

    #region GAN Training Steps

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

        current = ConcatTensors(current, inputTensor);
        current = Layers[^1].Forward(current);

        return ApplyOutputActivations(current);
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
        => SyntheticTabularGeneratorBase<T>.PrepareColumns(data, columns);

    #endregion

    #region Training Buffer Management

    /// <summary>
    /// Pre-allocates all training buffers once to eliminate per-row GC pressure.
    /// Must be called after RebuildLayersWithActualDimensions when all dimensions are known.
    /// </summary>
    private void InitializeTrainingBuffers(int genInputDim)
    {
        _oneGrad = new Tensor<T>([1]);
        _oneGrad[0] = NumOps.One;
        _negOneGrad = new Tensor<T>([1]);
        _negOneGrad[0] = NumOps.Negate(NumOps.One);

        _packedRealBuf = new Vector<T>(_packedInputDim);
        _packedFakeBuf = new Vector<T>(_packedInputDim);
        _noiseBuf = new Vector<T>(_options.EmbeddingDimension);
        _genInputBuf = new Vector<T>(genInputDim);

        int singleDim = _dataWidth + _condWidth;
        _realSingleBuf = new Vector<T>(singleDim);
        _fakeSingleBuf = new Vector<T>(singleDim);
        _realRowBuf = new Vector<T>(_dataWidth);
        _fakeRowBuf = new Vector<T>(_dataWidth);
        _interpolatedBuf = new Vector<T>(_packedInputDim);
        _sampleGradBuf = new Tensor<T>([_dataWidth]);
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

    private static Vector<T> ConcatVectors(Vector<T> a, Vector<T> b) => SyntheticTensorOps<T>.ConcatVectors(a, b);

    private static Tensor<T> VectorToTensor(Vector<T> v) => Tensor<T>.FromVector(v);

    private static Tensor<T> ConcatTensors(Tensor<T> a, Tensor<T> b) => SyntheticTensorOps<T>.Concat1D(a, b);

    #endregion

}
