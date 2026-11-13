using AiDotNet.LearningRateSchedulers;
using AiDotNet.ActivationFunctions;
using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Optimizers;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.NeuralNetworks.SyntheticData;

/// <summary>
/// TabDDPM (Tabular Denoising Diffusion Probabilistic Model) for generating synthetic tabular data.
/// </summary>
/// <remarks>
/// <para>
/// TabDDPM applies diffusion models to tabular data with separate processes for different feature types:
/// - <b>Gaussian diffusion</b> for continuous/numerical features (noise prediction)
/// - <b>Multinomial diffusion</b> for categorical features (category probability prediction)
/// - <b>Shared MLP denoiser</b> with sinusoidal timestep embedding processes both types jointly
/// - <b>Simple preprocessing</b>: Quantile normalization for continuous, integer encoding for categorical
/// </para>
/// <para>
/// This implementation follows the standard neural network architecture pattern with:
/// - Proper inheritance from NeuralNetworkBase
/// - Layer-based architecture using ILayer components
/// - Engine-based tensor operations for CPU/GPU acceleration
/// - Full autodiff and JIT compilation support
/// </para>
/// <para>
/// <b>For Beginners:</b> TabDDPM generates data by learning to "undo noise."
///
/// <b>Training:</b>
/// <code>
/// 1. Take a real data row
/// 2. Pick a random noise level (timestep t)
/// 3. Add noise: numbers get Gaussian noise, categories get randomly flipped
/// 4. Feed noisy data + timestep to the MLP
/// 5. MLP predicts: what noise was added (for numbers) / what original category (for categories)
/// 6. Compare predictions to truth and update the MLP
/// </code>
///
/// <b>Generation:</b>
/// <code>
/// 1. Start with pure random noise (numbers = random, categories = random)
/// 2. For t = T, T-1, T-2, ..., 1, 0:
///    a. Feed noisy data + timestep to MLP
///    b. MLP predicts the noise/categories
///    c. Remove a small amount of noise
/// 3. Final result is a clean, realistic data row
/// </code>
///
/// If you provide custom layers in the architecture, those will be used directly
/// for the denoiser MLP. If not, the network creates industry-standard
/// TabDDPM layers based on the original research paper specifications.
///
/// Example usage:
/// <code>
/// var architecture = new NeuralNetworkArchitecture&lt;double&gt;(
///     inputFeatures: 10,
///     outputSize: 10
/// );
/// var options = new TabDDPMOptions&lt;double&gt;
/// {
///     NumTimesteps = 1000,
///     MLPDimensions = new[] { 256, 256, 256 },
///     BatchSize = 4096,
///     Epochs = 1000
/// };
/// var tabddpm = new TabDDPMGenerator&lt;double&gt;(architecture, options);
/// tabddpm.Fit(data, columns, epochs: 1000);
/// var synthetic = tabddpm.Generate(1000);
/// </code>
///
/// TabDDPM often produces higher-quality synthetic data than CTGAN/TVAE, especially
/// for complex distributions, at the cost of slower generation (many denoising steps).
/// </para>
/// <para>
/// Reference: "TabDDPM: Modelling Tabular Data with Diffusion Models" (Kotelnikov et al., ICML 2023)
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
[ModelDomain(ModelDomain.MachineLearning)]
[ModelCategory(ModelCategory.NeuralNetwork)]
[ModelCategory(ModelCategory.SyntheticDataGenerator)]
[ModelTask(ModelTask.Generation)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("TabDDPM: Modelling Tabular Data with Diffusion Models",
    "https://arxiv.org/abs/2209.15421",
    Year = 2023,
    Authors = "Akim Kotelnikov, Dmitry Baranchuk, Ivan Rubachev, Artem Babenko")]
[PaperOptimizer(OptimizerKind.AdamW, Provenance = RecipeProvenance.Searched,
                Source = "Kotelnikov et al. 2023: the learning rate is drawn from LogUniform[1e-5, 1e-2] and the weight decay from {0, LogUniform[1e-6, 1e-3]} by hyperparameter search, so the paper states a space rather than a value and none is declared. Built by the model rather than by the factory because it constructs explicit options; the declaration verifies those values instead of replacing them.")]
public partial class TabDDPMGenerator<T> : NeuralSyntheticTabularGeneratorBase<T>, ISyntheticTabularGenerator<T>
{
    // Held across Train calls. The fused plan owns the optimizer state, so a step object built per call
    // re-traced and recompiled every call and restarted the optimizer at t = 1 (Adam ~ lr*sign(g)).
    private AiDotNet.Training.MultiSlotFusedStep<T>? _fusedMultiSlotStep;

    /// <inheritdoc/>
    protected override void Dispose(bool disposing)
    {
        if (disposing)
        {
            _fusedMultiSlotStep?.Dispose();
            _fusedMultiSlotStep = null;
        }
        base.Dispose(disposing);
    }

    private readonly TabDDPMOptions<T> _options;
    private IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> _optimizer;
    private ILossFunction<T> _lossFunction;

    // Synthetic tabular data infrastructure
    private List<ColumnMetadata> _columns = new();
    private Random _random;

    // Diffusion processes
    private GaussianDiffusion<T>? _gaussianDiffusion;
    private MultinomialDiffusion<T>? _multinomialDiffusion;

    // Output heads (auxiliary, not user-overridable)
    private FullyConnectedLayer<T>? _numericalOutputHead;
    private FullyConnectedLayer<T>? _categoricalOutputHead;

    // Timestep embedding layer (auxiliary)
    private FullyConnectedLayer<T>? _timestepProjection;

    // Feature dimensions
    private int _numNumericalFeatures;
    private int _numCategoricalFeatures;
    private int _totalCategoricalWidth;
    private int _inputWidth;
    private int _lastHiddenDim;
    [Scratch]
    private Tensor<T>? _lastMLPOutput;

    // Column layout tracking
    private readonly List<int> _numericalColumnIndices = new();
    private readonly List<int> _categoricalColumnIndices = new();
    private readonly List<int> _categoricalColumnWidths = new();

    // Preprocessing stats
    [AiDotNet.Attributes.FittedParameter(InputSized = true)]
    private Tensor<T> _quantileMeans = new Tensor<T>(new[] { 1, 0 });
    [AiDotNet.Attributes.FittedParameter(InputSized = true)]
    private Tensor<T> _quantileStds = new Tensor<T>(new[] { 1, 0 });
    // Per numerical column [1, J]: the fitted data's minimum and maximum (the clamp range of the reconstruction).
    [AiDotNet.Attributes.FittedParameter(InputSized = true)]
    private Tensor<T> _numericalMins = new Tensor<T>(new[] { 1, 0 });
    [AiDotNet.Attributes.FittedParameter(InputSized = true)]
    private Tensor<T> _numericalMaxs = new Tensor<T>(new[] { 1, 0 });

    // Whether custom layers are being used
    private bool _usingCustomLayers;

    /// <summary>
    /// Gets the TabDDPM-specific options.
    /// </summary>
    public new TabDDPMOptions<T> Options => _options;

    /// <inheritdoc />
    public IReadOnlyList<ColumnMetadata> Columns => _columns.AsReadOnly();

    /// <inheritdoc />
    public bool IsFitted { get; private set; }

    /// <summary>
    /// Initializes a new TabDDPM generator with the specified architecture.
    /// </summary>
    /// <param name="architecture">The neural network architecture defining input/output dimensions and optional custom layers.</param>
    /// <param name="options">TabDDPM-specific options for diffusion and MLP configuration.</param>
    /// <param name="optimizer">Gradient-based optimizer (defaults to Adam).</param>
    /// <param name="lossFunction">Loss function (defaults based on task type).</param>
    /// <summary>
    /// Initializes a new instance with default architecture settings.
    /// </summary>
    public TabDDPMGenerator()
        : this(new NeuralNetworkArchitecture<T>(
            inputType: Enums.InputType.OneDimensional,
            taskType: Enums.NeuralNetworkTaskType.Regression,
            inputSize: 10,
            outputSize: 10))
    {
    }

    public TabDDPMGenerator(NeuralNetworkArchitecture<T> architecture,
        TabDDPMOptions<T>? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
        ILossFunction<T>? lossFunction = null)
        : base(architecture, lossFunction ?? NeuralNetworkHelper<T>.GetDefaultLossFunction(architecture.TaskType), (options ??= new TabDDPMOptions<T>()).MaxGradNorm)
    {
        _options = options;
        _lossFunction = lossFunction ?? NeuralNetworkHelper<T>.GetDefaultLossFunction(architecture.TaskType);
        _optimizer = PaperOptimizerFactory.VerifyHandBuilt(this,
            optimizer ?? new AdamOptimizer<T, Tensor<T>, Tensor<T>>(this,
                new AdamOptimizerOptions<T, Tensor<T>, Tensor<T>>
                {
                    InitialLearningRate = _options.LearningRate
                }));

        int? seed = _options.Seed;
        _random = seed.HasValue
            ? RandomHelper.CreateSeededRandom(seed.Value)
            : RandomHelper.CreateSecureRandom();

        InitializeLayers();
    }

    #region Layer Initialization (GANDALF Pattern)

    /// <inheritdoc/>
    protected override void InitializeLayers()
    {
        if (Architecture.Layers != null && Architecture.Layers.Count > 0)
        {
            Layers.AddRange(Architecture.Layers);
            _usingCustomLayers = true;
        }
        else
        {
            // Create default denoiser MLP layers
            int inputDim = Architecture.CalculatedInputSize + _options.TimestepEmbeddingDimension;
            Layers.AddRange(LayerHelper<T>.CreateDefaultTabDDPMDenoiserLayers(
                inputDim, _options.MLPDimensions, _options.DropoutRate));
            _usingCustomLayers = false;
        }

        // Timestep projection is always internal, and maps TimestepEmbeddingDimension onto itself.
        // Both widths are declared for the same reason as TabSynGenerator: the (outputSize,
        // activation) overload leaves the input at -1, and a lazily-bound layer permanently takes
        // the width of the first tensor that reaches it -- which is not necessarily the sinusoidal
        // embedding it is meant to project.
        int timestepEmbeddingDimension = _options.TimestepEmbeddingDimension;
        _timestepProjection = new FullyConnectedLayer<T>(
            timestepEmbeddingDimension,
            timestepEmbeddingDimension,
            new SiLUActivation<T>() as IActivationFunction<T>);
    }

    /// <summary>
    /// Rebuilds denoiser layers with actual data dimensions discovered during Fit().
    /// </summary>
    private void RebuildLayersWithActualDimensions()
    {
        _inputWidth = _numNumericalFeatures + _totalCategoricalWidth + _options.TimestepEmbeddingDimension;
        _lastHiddenDim = _options.MLPDimensions.Length > 0
            ? _options.MLPDimensions[^1]
            : _inputWidth;

        if (!_usingCustomLayers)
        {
            Layers.Clear();
            Layers.AddRange(LayerHelper<T>.CreateDefaultTabDDPMDenoiserLayers(
                _inputWidth, _options.MLPDimensions, _options.DropoutRate));
        }

        // Always rebuild output heads with actual dimensions
        var identity = new IdentityActivation<T>() as IActivationFunction<T>;
        int lastHidden = _usingCustomLayers
            ? GetLastLayerOutputSize()
            : _lastHiddenDim;

        // lastHidden is the INPUT width and was already being computed above -- the comment says
        // "with actual dimensions" -- but both heads were constructed through the
        // (outputSize, activation) overload, discarding it and leaving the input at -1 for lazy
        // inference. A lazily-bound layer keeps the width of the first tensor that reaches it, so a
        // head bound to a narrow probe and then rejected the real MLP hidden state:
        // "Matrix dimensions incompatible: [1,64] x [3,5]".
        if (_numNumericalFeatures > 0)
        {
            _numericalOutputHead = new FullyConnectedLayer<T>(lastHidden, _numNumericalFeatures, identity);
        }

        if (_totalCategoricalWidth > 0)
        {
            _categoricalOutputHead = new FullyConnectedLayer<T>(lastHidden, _totalCategoricalWidth, identity);
        }
    }

    private int GetLastLayerOutputSize()
    {
        if (Layers.Count == 0) return _inputWidth;
        var shape = Layers[^1].GetOutputShape();
        if (shape is not null && shape.Length > 0) return shape[0];
        return _options.MLPDimensions.Length > 0 ? _options.MLPDimensions[^1] : _inputWidth;
    }

    #endregion

    #region Neural Network Methods (GANDALF Pattern)

    /// <inheritdoc/>
    protected override Tensor<T> PredictCore(Tensor<T> input)
    {

        Tensor<T> current = input;
        foreach (var layer in Layers)
        {
            current = layer.Forward(current);
        }
        return current;
    }

    // UpdateParameters re-sliced the flat vector across Layers by hand -- the base walks
    // exactly the same enumeration, so this said nothing the base does not already say.
    #endregion

    #region ISyntheticTabularGenerator<T> Implementation

    /// <inheritdoc/>
    public void Fit(Matrix<T> data, IReadOnlyList<ColumnMetadata> columns, int? epochs = null)
    {
        int epochCount = epochs ?? _options.Epochs;
        ValidateFitInputs(data, columns, epochCount);

        _columns = PrepareColumns(data, columns);

        // Step 1: Analyze column layout
        AnalyzeColumns(_columns);

        // Step 2: Preprocess data
        var (numericalData, categoricalData) = PreprocessData(data, _columns);

        // Step 3: Initialize diffusion processes
        _gaussianDiffusion = new GaussianDiffusion<T>(
            _options.NumTimesteps, _options.BetaStart, _options.BetaEnd,
            _options.BetaSchedule, _random);

        _multinomialDiffusion = new MultinomialDiffusion<T>(
            _options.NumCategoricalDiffusionSteps, _options.BetaStart, _options.BetaEnd, _random);

        // Step 4: Build layers with actual dimensions
        BuildFromConstructionSequence(() => RebuildLayersWithActualDimensions());

        // Step 5: Training loop
        int batchSize = Math.Min(_options.BatchSize, data.Rows);
        int numBatches = Math.Max(1, data.Rows / batchSize);
        T scaledLr = NumOps.FromDouble(_options.LearningRate / batchSize);

        SetTrainingMode(true);
        try
        {
            for (int epoch = 0; epoch < epochCount; epoch++)
            {
                for (int batch = 0; batch < numBatches; batch++)
                {
                    int startRow = batch * batchSize;
                    int endRow = Math.Min(startRow + batchSize, data.Rows);
                    TrainBatch(numericalData, categoricalData, startRow, endRow, scaledLr);
                }
            }
        }
        finally
        {
            SetTrainingMode(false);
        }

        IsFitted = true;
    }

    /// <inheritdoc/>
    public async Task FitAsync(Matrix<T> data, IReadOnlyList<ColumnMetadata> columns, int? epochs = null, CancellationToken ct = default)
    {
        int epochCount = epochs ?? _options.Epochs;
        ValidateFitInputs(data, columns, epochCount);

        _columns = PrepareColumns(data, columns);

        await Task.Run(() =>
        {
            ct.ThrowIfCancellationRequested();

            AnalyzeColumns(_columns);
            var (numericalData, categoricalData) = PreprocessData(data, _columns);

            _gaussianDiffusion = new GaussianDiffusion<T>(
                _options.NumTimesteps, _options.BetaStart, _options.BetaEnd,
                _options.BetaSchedule, _random);

            _multinomialDiffusion = new MultinomialDiffusion<T>(
                _options.NumCategoricalDiffusionSteps, _options.BetaStart, _options.BetaEnd, _random);

            BuildFromConstructionSequence(() => RebuildLayersWithActualDimensions());

            int batchSize = Math.Min(_options.BatchSize, data.Rows);
            int numBatches = Math.Max(1, data.Rows / batchSize);
            T scaledLr = NumOps.FromDouble(_options.LearningRate / batchSize);

            SetTrainingMode(true);
            try
            {
                for (int epoch = 0; epoch < epochCount; epoch++)
                {
                    ct.ThrowIfCancellationRequested();
                    for (int batch = 0; batch < numBatches; batch++)
                    {
                        int startRow = batch * batchSize;
                        int endRow = Math.Min(startRow + batchSize, data.Rows);
                        TrainBatch(numericalData, categoricalData, startRow, endRow, scaledLr);
                    }
                }
            }
            finally
            {
                SetTrainingMode(false);
            }

            IsFitted = true;
        }, ct).ConfigureAwait(false);
    }

    /// <inheritdoc/>
    public Matrix<T> Generate(int numSamples, Vector<T>? conditionColumn = null, Vector<T>? conditionValue = null)
    {
        if (_gaussianDiffusion is null || _multinomialDiffusion is null || !IsFitted)
        {
            throw new InvalidOperationException("Generator must be fitted before generating data. Call Fit() first.");
        }

        // Start from pure noise
        var numericalSamples = new Matrix<T>(numSamples, _numNumericalFeatures);
        var categoricalSamples = InitializeRandomCategories(numSamples);

        // Initialize numerical features with standard normal noise
        for (int i = 0; i < numSamples; i++)
        {
            for (int j = 0; j < _numNumericalFeatures; j++)
            {
                numericalSamples[i, j] = SampleStandardNormal();
            }
        }

        // Reverse diffusion: denoise step by step
        int numSteps = _options.NumTimesteps;
        int catSteps = _options.NumCategoricalDiffusionSteps;

        for (int t = numSteps - 1; t >= 0; t--)
        {
            for (int i = 0; i < numSamples; i++)
            {
                var numFeats = GetMatrixRow(numericalSamples, i);
                var catFeats = GetMatrixRow(categoricalSamples, i);
                var timeEmbed = CreateTimestepEmbedding(t);

                var (predictedNoise, predictedLogits) = DenoiserForward(numFeats, catFeats, timeEmbed);

                numericalSamples = SetMatrixRow(numericalSamples, i,
                    _gaussianDiffusion.DenoisingStep(numFeats, predictedNoise, t));

                int catT = (int)((long)t * catSteps / numSteps);
                if (catT < catSteps && _totalCategoricalWidth > 0)
                {
                    var denoisedCat = DenoiseCategorical(catFeats, predictedLogits, catT);
                    categoricalSamples = SetMatrixRow(categoricalSamples, i, denoisedCat);
                }
            }
        }

        return ReconstructData(numericalSamples, categoricalSamples, numSamples);
    }

    #endregion

    #region Column Analysis and Preprocessing

    private void AnalyzeColumns(IReadOnlyList<ColumnMetadata> columns)
    {
        _numericalColumnIndices.Clear();
        _categoricalColumnIndices.Clear();
        _categoricalColumnWidths.Clear();

        _numNumericalFeatures = 0;
        _totalCategoricalWidth = 0;

        for (int col = 0; col < columns.Count; col++)
        {
            if (columns[col].IsNumerical)
            {
                _numericalColumnIndices.Add(col);
                _numNumericalFeatures++;
            }
            else if (columns[col].IsCategorical)
            {
                _categoricalColumnIndices.Add(col);
                int numCats = Math.Max(columns[col].NumCategories, 2);
                _categoricalColumnWidths.Add(numCats);
                _totalCategoricalWidth += numCats;
            }
        }

        _numCategoricalFeatures = _categoricalColumnIndices.Count;
    }

    private (Matrix<T> Numerical, Matrix<T> Categorical) PreprocessData(
        Matrix<T> data, IReadOnlyList<ColumnMetadata> columns)
    {
        int n = data.Rows;

        var dataTensor = Tensor<T>.FromMatrix(data);

        // Quantile normalization for numerical columns: (x - mean) / std per column, with the sample (n - 1)
        // std (1 for a single row or a constant column).
        Matrix<T> numerical;
        if (_numNumericalFeatures == 0)
        {
            numerical = new Matrix<T>(n, 0);
            _quantileMeans = new Tensor<T>(new[] { 1, 0 });
            _quantileStds = new Tensor<T>(new[] { 1, 0 });
            _numericalMins = new Tensor<T>(new[] { 1, 0 });
            _numericalMaxs = new Tensor<T>(new[] { 1, 0 });
        }
        else
        {
            var columnIds = new Tensor<int>(_numericalColumnIndices.ToArray(), new[] { _numNumericalFeatures });
            var values = Engine.TensorIndexSelect(dataTensor, columnIds, 1);                      // [n, J]

            // A narrow T (Half, BFloat16) is normalized in float: its sum of squares overflows past 65504.
            Tensor<T> mean, std, normalized;
            if (typeof(T) == typeof(Half) || typeof(T) == typeof(AiDotNet.Tensors.NumericOperations.BFloat16))
            {
                var wide = NormalizeColumns(values.Cast<float>(), n);
                (mean, std, normalized) = (wide.Mean.Cast<T>(), wide.Std.Cast<T>(), wide.Normalized.Cast<T>());
            }
            else
            {
                (mean, std, normalized) = NormalizeColumns(values, n);
            }

            _quantileMeans = mean;
            _quantileStds = std;
            _numericalMins = Engine.TensorNegate(Engine.ReduceMax(Engine.TensorNegate(values), new[] { 0 }, true));
            _numericalMaxs = Engine.ReduceMax(values, new[] { 0 }, true);
            numerical = normalized.ToMatrix();
        }

        // One-hot encoding for categorical columns: the rounded (half to even) value selects the category;
        // values outside [0, numCats) leave the block empty.
        Matrix<T> categorical;
        if (_numCategoricalFeatures == 0 || n == 0)
        {
            categorical = new Matrix<T>(n, _totalCategoricalWidth);
        }
        else
        {
            var categoryIds = new Tensor<int>(_categoricalColumnIndices.ToArray(), new[] { _numCategoricalFeatures });
            var rounded = Engine.TensorRound(Engine.TensorIndexSelect(dataTensor, categoryIds, 1)).Cast<int>();   // [n, Jc]
            var blocks = new Tensor<T>[_numCategoricalFeatures];
            for (int j = 0; j < _numCategoricalFeatures; j++)
            {
                int numCats = _categoricalColumnWidths[j];
                var ids = Engine.Reshape(Engine.TensorSlice(rounded, new[] { 0, j }, new[] { n, 1 }), new[] { n });
                var inRange = Engine.TensorMultiply(
                    Engine.TensorGreaterThan(ids, -1), Engine.TensorLessThan(ids, numCats));
                var oneHot = Engine.TensorOneHot<T>(Engine.TensorMultiply(ids, inRange), numCats);
                blocks[j] = Engine.TensorMultiply(
                    oneHot, Engine.TensorBroadcastTo(Engine.Reshape(inRange.Cast<T>(), new[] { n, 1 }), new[] { n, numCats }));
            }
            categorical = Engine.TensorConcatenate(blocks, 1).ToMatrix();
        }

        return (numerical, categorical);
    }

    /// <summary>
    /// Per column of <c>[n, J]</c> values: the mean, the sample (n - 1) std (1 for a single row or a constant
    /// column) and the values standardized with them.
    /// </summary>
    private (Tensor<TValue> Mean, Tensor<TValue> Std, Tensor<TValue> Normalized) NormalizeColumns<TValue>(
        Tensor<TValue> values, int n)
    {
        var ops = MathHelper.GetNumericOperations<TValue>();
        var shape = values._shape;
        var mean = Engine.ReduceMean(values, new[] { 0 }, true);                              // [1, J]
        var centered = Engine.TensorSubtract(values, Engine.TensorBroadcastTo(mean, shape));
        var ones = Engine.TensorOnesLike(mean);
        Tensor<TValue> std;
        if (n > 1)
        {
            var variance = Engine.TensorDivideScalar(
                Engine.ReduceSum(Engine.TensorMultiply(centered, centered), new[] { 0 }, true), ops.FromDouble(n - 1));
            std = Engine.TensorSqrt(variance);
            var nearZero = Engine.TensorAdd(Engine.TensorLessThan(std, ops.FromDouble(1e-10)),
                Engine.TensorEquals(std, ops.Zero));
            std = Engine.TensorWhere(nearZero, ones, std);
        }
        else
        {
            std = ones;
        }

        return (mean, std, Engine.TensorDivide(centered, Engine.TensorBroadcastTo(std, shape)));
    }

    #endregion

    #region Timestep Embedding

    private Vector<T> CreateTimestepEmbedding(int timestep)
    {
        int dim = _options.TimestepEmbeddingDimension;
        var embedTensor = BuildRawTimestepEmbedding(timestep);

        if (_timestepProjection is not null)
        {
            var projected = _timestepProjection.Forward(embedTensor);
            return TensorToVector(projected, dim);
        }

        return embedTensor.ToVector();
    }

    /// <summary>
    /// Tape-connected timestep embedding: the sinusoidal features projected
    /// through the learnable <see cref="_timestepProjection"/>, returned as a
    /// <see cref="Tensor{T}"/> so its gradient flows back to the projection
    /// during training. The Vector-returning <see cref="CreateTimestepEmbedding"/>
    /// detaches (via TensorToVector) and is kept for sampling/inference.
    /// </summary>
    private Tensor<T> CreateTimestepEmbeddingTensor(int timestep)
    {
        var embedTensor = BuildRawTimestepEmbedding(timestep);
        return _timestepProjection is not null
            ? _timestepProjection.Forward(embedTensor)
            : embedTensor;
    }

    #endregion

    #region Denoiser Forward

    private (Vector<T> NoisePred, Vector<T> CatLogits) DenoiserForward(
        Vector<T> numericalFeatures, Vector<T> categoricalFeatures, Vector<T> timestepEmbed)
    {
        // [numerical | categorical | timestep] as one input row.
        var current = Engine.TensorConcatenate(
            new[] { VectorToTensor(numericalFeatures), VectorToTensor(categoricalFeatures), VectorToTensor(timestepEmbed) }, 0);

        // Forward through MLP layers
        foreach (var layer in Layers)
        {
            current = layer.Forward(current);
        }
        _lastMLPOutput = current;

        // Split output via separate heads (each keeps its leading values, zero-padded to the head width)
        var noisePred = _numericalOutputHead is not null
            ? TensorToVector(_numericalOutputHead.Forward(current), _numNumericalFeatures)
            : new Vector<T>(_numNumericalFeatures);
        var catLogits = _categoricalOutputHead is not null
            ? TensorToVector(_categoricalOutputHead.Forward(current), _totalCategoricalWidth)
            : new Vector<T>(_totalCategoricalWidth);

        return (noisePred, catLogits);
    }

    #endregion

    #region Training

    private void TrainBatch(Matrix<T> numericalData, Matrix<T> categoricalData,
        int startRow, int endRow, T scaledLearningRate)
    {
        if (_gaussianDiffusion is null || _multinomialDiffusion is null) return;

        // MultiSlotFusedStep cache: reused across rows so the compiled plan
        // is compiled once (on the first row) and replayed per subsequent row
        // by refreshing slot data. See ooples/AiDotNet#1846.
        AiDotNet.Training.MultiSlotFusedStep<T>? multiSlotStep = null;
        bool fusedMapped = NeuralNetworks.NeuralNetworkBase<T>.TryMapToFusedOptimizerConfig(_optimizer, out _);
        for (int row = startRow; row < endRow; row++)
        {
            int t = _gaussianDiffusion.SampleTimestep();
            int catT = (int)((long)t * _options.NumCategoricalDiffusionSteps / _options.NumTimesteps);

            var numClean = GetMatrixRow(numericalData, row);
            var catClean = GetMatrixRow(categoricalData, row);

            Vector<T> numNoisy;
            Vector<T> actualNoise;
            if (_numNumericalFeatures > 0)
            {
                (numNoisy, actualNoise) = _gaussianDiffusion.AddNoise(numClean, t);
            }
            else
            {
                numNoisy = numClean;
                actualNoise = new Vector<T>(0);
            }

            Vector<T> catNoisy;
            if (_totalCategoricalWidth > 0 && catT < _multinomialDiffusion.NumTimesteps)
            {
                catNoisy = _multinomialDiffusion.AddNoise(catClean, catT);
            }
            else
            {
                catNoisy = catClean;
            }

            // Preferred fused path: MultiSlotFusedStep with the raw sinusoidal
            // timestep encoding as a persistent slot. The learnable
            // _timestepProjection stays INSIDE the compiled forward closure
            // (which _plan.Step() replays per row) so its weights participate
            // in the backward pass. See ooples/AiDotNet#1846.
            // The fused step resolves lazily created weights itself, so it is tried from the first row: an eager first
            // row would leave its optimizer moments behind when the plan takes over.
            if (fusedMapped)
            {
                var slots = BuildTabDDPMSlots(numNoisy, actualNoise, catNoisy, catClean, t);
                if (slots is not null)
                {
                    multiSlotStep = _fusedMultiSlotStep ??= new AiDotNet.Training.MultiSlotFusedStep<T>(this);
                    Tensor<T> ForwardFromSlots(IReadOnlyList<Tensor<T>> s)
                    {
                        // s[0] numNoisy, s[1] actualNoise, s[2] catNoisy,
                        // s[3] catClean, s[4] rawSinusoidalTimeEmbed.
                        var timeEmbed = _timestepProjection is not null
                            ? _timestepProjection.Forward(s[4])
                            : s[4];
                        var (noisePred, catLogits) = DenoiserForwardFromTensors(s[0], s[2], timeEmbed);
                        // Concat both heads into a single output tensor so the
                        // fused-step signature (Tensor<T> forward output) is
                        // satisfied. Loss closure splits it back.
                        return Engine.TensorConcatenate(new[] { noisePred, catLogits }, axis: 0);
                    }
                    Tensor<T> ComputeLossFromSlots(Tensor<T> pred, IReadOnlyList<Tensor<T>> s)
                    {
                        // Split forward output back into (noisePred, catLogits).
                        int noisePredLen = _numNumericalFeatures > 0 && _numericalOutputHead is not null
                            ? s[1].Length : 0;
                        int catLogitsLen = _totalCategoricalWidth > 0 && _categoricalOutputHead is not null
                            ? s[3].Length : 0;
                        Tensor<T> noisePredT, catLogitsT;
                        if (noisePredLen > 0 && catLogitsLen > 0)
                        {
                            noisePredT = Engine.TensorSlice(pred, new[] { 0 }, new[] { noisePredLen });
                            catLogitsT = Engine.TensorSlice(pred, new[] { noisePredLen }, new[] { catLogitsLen });
                        }
                        else if (noisePredLen > 0)
                        {
                            noisePredT = pred;
                            catLogitsT = new Tensor<T>(new[] { 0 });
                        }
                        else
                        {
                            noisePredT = new Tensor<T>(new[] { 0 });
                            catLogitsT = pred;
                        }
                        return ComputeDiffusionLossTapeFromTensors(noisePredT, s[1], catLogitsT, s[3]);
                    }
                    if (multiSlotStep.TryStep(
                            parameterProvider: CollectModelTrainableTensors,
                            zeroGradAction: null,
                            freshSlotData: slots,
                            forward: ForwardFromSlots,
                            computeLoss: ComputeLossFromSlots,
                            optimizer: _optimizer,
                            out T _))
                    {
                        continue;
                    }
                }
            }

            // Eager fallback: tape-connected diffusion training step: run the
            // denoiser forward on the tape, build the TabDDPM hybrid loss
            // (ε-prediction MSE for the Gaussian-diffused numerical features +
            // softmax cross-entropy for the multinomial-diffused categorical
            // features, Kotelnikov et al. 2023), and backpropagate through the
            // MLP + output heads + timestep projection in one optimizer step.
            using var tape = new GradientTape<T>();
            var timeEmbed2 = CreateTimestepEmbeddingTensor(t);
            var (predictedNoise, predictedLogits) = DenoiserForwardTensors(numNoisy, catNoisy, timeEmbed2);
            var loss = ComputeDiffusionLossTape(predictedNoise, actualNoise, predictedLogits, catClean);
            BackwardAndStepOnPrecomputedLoss(tape, loss, _optimizer);
        }
    }

    /// <summary>Builds the raw sinusoidal timestep encoding (no learnable projection
    /// applied — that runs INSIDE the compiled forward). Matches the first-half
    /// of <see cref="CreateTimestepEmbeddingTensor"/>.</summary>
    private Tensor<T> BuildRawTimestepEmbedding(int timestep)
    {
        // [sin(angle_i), cos(angle_i)] with angle_i = timestep * exp(-ln(10000) i / halfDim); an odd dim
        // leaves the last entry 0.
        int dim = _options.TimestepEmbeddingDimension;
        int halfDim = dim / 2;
        if (halfDim == 0)
            return new Tensor<T>(new[] { dim });
        // The angles are formed in double: in a narrow T such as Half, timestep * frequency is off by up to
        // half a radian.
        var frequencies = Engine.TensorExp(Engine.TensorMultiplyScalar(
            Engine.TensorArange<double>(0, halfDim, 1), -Math.Log(10000.0) / halfDim));
        var angles = Engine.TensorMultiplyScalar(frequencies, (double)timestep);
        var parts = dim % 2 == 0
            ? new[] { Engine.TensorSin(angles).Cast<T>(), Engine.TensorCos(angles).Cast<T>() }
            : new[] { Engine.TensorSin(angles).Cast<T>(), Engine.TensorCos(angles).Cast<T>(), new Tensor<T>(new[] { 1 }) };
        return Engine.TensorConcatenate(parts, 0);
    }

    /// <summary>Assembles the persistent-slot data list for the MultiSlotFusedStep
    /// wire-up. Returns null when any required feature vector's length is zero
    /// (indicates a degenerate row) so the caller falls back to eager training.</summary>
    private IReadOnlyList<Tensor<T>>? BuildTabDDPMSlots(
        Vector<T> numNoisy, Vector<T> actualNoise,
        Vector<T> catNoisy, Vector<T> catClean, int timestep)
    {
        // Degenerate: no features at all — can't run the denoiser.
        if (numNoisy.Length == 0 && catNoisy.Length == 0) return null;

        // Use zero-length tensors as placeholders for absent modality (matches
        // the DenoiserForwardTensors branching pattern). MultiSlotFusedStep's
        // shape-key includes zero-length slots so a shape change in either
        // modality triggers a plan recompile.
        return new[]
        {
            VectorToTensor(numNoisy),
            VectorToTensor(actualNoise),
            VectorToTensor(catNoisy),
            VectorToTensor(catClean),
            BuildRawTimestepEmbedding(timestep),
        };
    }

    /// <summary>Tensor-input variant of <see cref="DenoiserForwardTensors"/>. Used by the
    /// MultiSlotFusedStep forward closure so the numerical/categorical/timestep inputs
    /// are all persistent slots (their references are captured once at trace, their
    /// data is refreshed per replay).</summary>
    private (Tensor<T> NoisePred, Tensor<T> CatLogits) DenoiserForwardFromTensors(
        Tensor<T> numericalNoisy, Tensor<T> categoricalNoisy, Tensor<T> projectedTimeEmbed)
    {
        var timeT = projectedTimeEmbed.Rank == 1
            ? projectedTimeEmbed
            : Engine.Reshape(projectedTimeEmbed, new[] { projectedTimeEmbed.Length });

        Tensor<T> current;
        if (numericalNoisy.Length > 0 && categoricalNoisy.Length > 0)
            current = Engine.TensorConcatenate(new[] { numericalNoisy, categoricalNoisy, timeT }, 0);
        else if (numericalNoisy.Length > 0)
            current = Engine.TensorConcatenate(new[] { numericalNoisy, timeT }, 0);
        else if (categoricalNoisy.Length > 0)
            current = Engine.TensorConcatenate(new[] { categoricalNoisy, timeT }, 0);
        else
            current = timeT;

        foreach (var layer in Layers)
            current = layer.Forward(current);

        var noisePred = _numericalOutputHead is not null && _numNumericalFeatures > 0
            ? _numericalOutputHead.Forward(current)
            : new Tensor<T>(new[] { 0 });
        var catLogits = _categoricalOutputHead is not null && _totalCategoricalWidth > 0
            ? _categoricalOutputHead.Forward(current)
            : new Tensor<T>(new[] { 0 });
        return (noisePred, catLogits);
    }

    /// <summary>Tensor-input variant of <see cref="ComputeDiffusionLossTape"/> — target
    /// noise and target categorical values come in as slot tensors instead of Vector&lt;T&gt;.</summary>
    private Tensor<T> ComputeDiffusionLossTapeFromTensors(
        Tensor<T> noisePred, Tensor<T> actualNoise, Tensor<T> catLogits, Tensor<T> catClean)
    {
        Tensor<T>? loss = null;

        if (_numNumericalFeatures > 0 && actualNoise.Length > 0)
        {
            var pred = noisePred.Rank == 1 ? noisePred : Engine.Reshape(noisePred, new[] { noisePred.Length });
            var diff = Engine.TensorSubtract(pred, actualNoise);
            loss = ReduceToScalar(Engine.TensorSquare(diff));
        }

        if (_totalCategoricalWidth > 0 && catClean.Length > 0)
        {
            var logits = catLogits.Rank == 1 ? catLogits : Engine.Reshape(catLogits, new[] { catLogits.Length });
            int offset = 0;
            for (int j = 0; j < _numCategoricalFeatures; j++)
            {
                int numCats = _categoricalColumnWidths[j];
                if (numCats > 0)
                {
                    var ce = SoftmaxCrossEntropy(logits, catClean, offset, numCats);
                    loss = loss is null ? ce : Engine.TensorAdd(loss, ce);
                    offset += numCats;
                }
            }
        }

        return loss ?? ReduceToScalar(Engine.TensorSquare(VectorToTensor(new Vector<T>(1))));
    }

    /// <summary>
    /// Tape-connected denoiser forward returning the numerical ε-prediction and
    /// the categorical logits as <see cref="Tensor{T}"/> outputs (the vector-
    /// returning <see cref="DenoiserForward"/> is kept for sampling/inference).
    /// </summary>
    private (Tensor<T> NoisePred, Tensor<T> CatLogits) DenoiserForwardTensors(
        Vector<T> numericalFeatures, Vector<T> categoricalFeatures, Tensor<T> timestepEmbed)
    {
        // Build the denoiser input tape-connected. The numerical/categorical
        // features are leaf data, but the timestep embedding flows through the
        // learnable _timestepProjection, so CONCATENATE the projected embedding
        // tensor (rather than flattening everything through a detached Vector) to
        // keep its gradient path intact — otherwise _timestepProjection never
        // trains (its forward would happen off-tape).
        var timeT = timestepEmbed.Rank == 1
            ? timestepEmbed
            : Engine.Reshape(timestepEmbed, new[] { timestepEmbed.Length });

        Tensor<T> current;
        if (numericalFeatures.Length > 0 && categoricalFeatures.Length > 0)
            current = Engine.TensorConcatenate(new[] { VectorToTensor(numericalFeatures), VectorToTensor(categoricalFeatures), timeT }, 0);
        else if (numericalFeatures.Length > 0)
            current = Engine.TensorConcatenate(new[] { VectorToTensor(numericalFeatures), timeT }, 0);
        else if (categoricalFeatures.Length > 0)
            current = Engine.TensorConcatenate(new[] { VectorToTensor(categoricalFeatures), timeT }, 0);
        else
            current = timeT;

        foreach (var layer in Layers)
        {
            current = layer.Forward(current);
        }

        var noisePred = _numericalOutputHead is not null && _numNumericalFeatures > 0
            ? _numericalOutputHead.Forward(current)
            : new Tensor<T>([0]);
        var catLogits = _categoricalOutputHead is not null && _totalCategoricalWidth > 0
            ? _categoricalOutputHead.Forward(current)
            : new Tensor<T>([0]);
        return (noisePred, catLogits);
    }

    /// <summary>Tape-connected TabDDPM loss: numerical ε-MSE + per-group categorical softmax cross-entropy.</summary>
    private Tensor<T> ComputeDiffusionLossTape(Tensor<T> noisePred, Vector<T> actualNoise, Tensor<T> catLogits, Vector<T> catClean)
    {
        Tensor<T>? loss = null;

        if (_numNumericalFeatures > 0)
        {
            var pred = noisePred.Rank == 1 ? noisePred : Engine.Reshape(noisePred, new[] { noisePred.Length });
            var target = VectorToTensor(actualNoise);
            var diff = Engine.TensorSubtract(pred, target);
            loss = ReduceToScalar(Engine.TensorSquare(diff));
        }

        if (_totalCategoricalWidth > 0)
        {
            var logits = catLogits.Rank == 1 ? catLogits : Engine.Reshape(catLogits, new[] { catLogits.Length });
            var tgt = VectorToTensor(catClean);
            int offset = 0;
            for (int j = 0; j < _numCategoricalFeatures; j++)
            {
                int numCats = _categoricalColumnWidths[j];
                if (numCats > 0)
                {
                    var ce = SoftmaxCrossEntropy(logits, tgt, offset, numCats);
                    loss = loss is null ? ce : Engine.TensorAdd(loss, ce);
                    offset += numCats;
                }
            }
        }

        return loss ?? ReduceToScalar(Engine.TensorSquare(VectorToTensor(new Vector<T>(1))));
    }

    /// <summary>Tape-connected softmax cross-entropy −Σ target·log(softmax(logit_slice)) over a categorical group.</summary>
    private Tensor<T> SoftmaxCrossEntropy(Tensor<T> logits, Tensor<T> tgt, int start, int count)
    {
        var logitSlice = Engine.TensorSlice(logits, new[] { start }, new[] { count });
        var tgtSlice = Engine.TensorSlice(tgt, new[] { start }, new[] { count });
        var logProbs = Engine.TensorLogSoftmax(logitSlice, axis: 0);
        return Engine.TensorNegate(ReduceToScalar(Engine.TensorMultiply(tgtSlice, logProbs)));
    }

    /// <summary>Reduces a tensor to a scalar [1] by summing all elements (tape-connected).</summary>
    private Tensor<T> ReduceToScalar(Tensor<T> t)
        => Engine.ReduceSum(t, Enumerable.Range(0, t.Shape.Length).ToArray(), keepDims: false);

    #endregion

    #region Backward Pass

    #endregion

    #region Sampling Helpers

    private Matrix<T> InitializeRandomCategories(int numSamples)
    {
        // Every category of a column starts at 1 / numCats.
        if (_totalCategoricalWidth == 0 || numSamples == 0)
            return new Matrix<T>(numSamples, _totalCategoricalWidth);

        var blockWidth = new int[_totalCategoricalWidth];
        int offset = 0;
        foreach (int numCats in _categoricalColumnWidths)
        {
            blockWidth.AsSpan(offset, numCats).Fill(numCats);
            offset += numCats;
        }

        var uniformRow = Engine.Reshape(
            Engine.TensorReciprocal(new Tensor<int>(blockWidth, new[] { _totalCategoricalWidth }).Cast<T>()),
            new[] { 1, _totalCategoricalWidth });
        return Engine.TensorTile(uniformRow, new[] { numSamples, 1 }).ToMatrix();
    }

    private Vector<T> DenoiseCategorical(Vector<T> noisyFeats, Vector<T> predictedLogits, int catT)
    {
        if (_multinomialDiffusion is null) return noisyFeats;

        // Inputs past the end of either vector read as zero; each block's result is zero-padded to its width.
        var noisy = PadTo(noisyFeats, _totalCategoricalWidth);
        var logits = PadTo(predictedLogits, _totalCategoricalWidth);
        var blocks = new Tensor<T>[_numCategoricalFeatures];
        int offset = 0;

        for (int j = 0; j < _numCategoricalFeatures; j++)
        {
            int numCats = _categoricalColumnWidths[j];
            var denoised = _multinomialDiffusion.DenoisingStep(
                noisy.GetSubVector(offset, numCats), logits.GetSubVector(offset, numCats), catT);
            blocks[j] = Tensor<T>.FromVector(PadTo(denoised, numCats));
            offset += numCats;
        }

        return _numCategoricalFeatures == 0
            ? new Vector<T>(_totalCategoricalWidth)
            : Engine.TensorConcatenate(blocks, 0).ToVector();
    }

    #endregion

    #region Data Reconstruction

    private Matrix<T> ReconstructData(Matrix<T> numericalSamples, Matrix<T> categoricalSamples, int numSamples)
    {
        if (numSamples == 0 || _columns.Count == 0)
            return new Matrix<T>(numSamples, _columns.Count);

        // Numerical columns: undo the normalization (x * std + mean), clamp to the fitted range and round
        // the discrete ones (half to even, as Math.Round).
        Tensor<T>? numerical = null;
        if (_numNumericalFeatures > 0)
        {
            var shape = new[] { numSamples, _numNumericalFeatures };
            var restored = Engine.TensorAdd(
                Engine.TensorMultiply(Tensor<T>.FromMatrix(numericalSamples), Engine.TensorBroadcastTo(_quantileStds, shape)),
                Engine.TensorBroadcastTo(_quantileMeans, shape));
            restored = Engine.TensorMax(
                Engine.TensorBroadcastTo(_numericalMins, shape),
                Engine.TensorMin(Engine.TensorBroadcastTo(_numericalMaxs, shape), restored));
            var discrete = _columns.Where(c => c.IsNumerical).Select(c => c.DataType == ColumnDataType.Discrete ? 1 : 0).ToArray();
            var discreteMask = Engine.TensorBroadcastTo(
                new Tensor<int>(discrete, new[] { 1, _numNumericalFeatures }).Cast<T>(), shape);
            numerical = Engine.TensorWhere(discreteMask, Engine.TensorRound(restored), restored);
        }

        // Categorical columns: the first highest category of each block (NaN never wins).
        var categoricalTensor = _totalCategoricalWidth > 0 ? Tensor<T>.FromMatrix(categoricalSamples) : null;
        if (categoricalTensor is not null)
        {
            categoricalTensor = Engine.TensorWhere(
                Engine.TensorIsNan(categoricalTensor),
                Engine.TensorFullLike(categoricalTensor, NumOps.FromDouble(double.NegativeInfinity)),
                categoricalTensor);
        }

        var outputColumns = new Tensor<T>[_columns.Count];
        int numIdx = 0;
        int catColIdx = 0;
        int catOffset = 0;
        for (int col = 0; col < _columns.Count; col++)
        {
            if (_columns[col].IsNumerical && numerical is not null)
            {
                outputColumns[col] = Engine.TensorSlice(numerical, new[] { 0, numIdx }, new[] { numSamples, 1 });
                numIdx++;
            }
            else if (_columns[col].IsCategorical && categoricalTensor is not null)
            {
                int numCats = _categoricalColumnWidths[catColIdx];
                var block = Engine.TensorSlice(categoricalTensor, new[] { 0, catOffset }, new[] { numSamples, numCats });
                outputColumns[col] = Engine.Reshape(Engine.TensorArgMax(block, 1).Cast<T>(), new[] { numSamples, 1 });
                catOffset += numCats;
                catColIdx++;
            }
            else
            {
                outputColumns[col] = new Tensor<T>(new[] { numSamples, 1 });
            }
        }

        return Engine.TensorConcatenate(outputColumns, 1).ToMatrix();
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
                { "NumTimesteps", _options.NumTimesteps },
                { "MLPDimensions", _options.MLPDimensions },
                { "TimestepEmbeddingDimension", _options.TimestepEmbeddingDimension },
                { "BatchSize", _options.BatchSize },
                { "NumNumericalFeatures", _numNumericalFeatures },
                { "TotalCategoricalWidth", _totalCategoricalWidth },
                { "DenoiserLayerCount", Layers.Count },
                { "DenoiserLayerTypes", Layers.Select(l => l.GetType().Name).ToArray() }
            },
            ModelDataProvider = () => SerializeForMetadata()
        };
    }

    /// <inheritdoc/>


    private static void WriteIntList(BinaryWriter writer, List<int> values)
    {
        writer.Write(values.Count);
        foreach (int value in values) writer.Write(value);
    }

    private static void ReadIntListInto(BinaryReader reader, List<int> target)
    {
        target.Clear();
        int count = reader.ReadInt32();
        for (int i = 0; i < count; i++) target.Add(reader.ReadInt32());
    }

    private static void WriteDoubleArray(BinaryWriter writer, double[] values)
    {
        writer.Write(values.Length);
        foreach (double value in values) writer.Write(value);
    }

    private static double[] ReadDoubleArray(BinaryReader reader)
    {
        int count = reader.ReadInt32();
        var values = new double[count];
        for (int i = 0; i < count; i++) values[i] = reader.ReadDouble();
        return values;
    }

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
        if (n == 0)
        {
            // An empty column keeps the empty-scan values: no min/max, mean 0/0, unit std.
            meta.Min = double.MaxValue;
            meta.Max = double.MinValue;
            meta.Mean = 0.0 / n;
            meta.Std = 1.0;
            return;
        }

        var column = Tensor<T>.FromVector(data.GetColumn(colIndex));

        // A T other than double is reduced in float with the column first divided by its largest magnitude:
        // the squared deviations of a finite float column (|x| near 1e20) overflow float, and a Half sum
        // overflows past 65504. Double columns keep the plain double reduction.
        var (mean, sumSqDiff) = typeof(T) == typeof(double)
            ? ColumnMoments(column, prescale: false)
            : ColumnMoments(column.Cast<float>(), prescale: true);

        // Sample standard deviation (n - 1), floored at 1e-10.
        double std = n > 1 ? Math.Sqrt(sumSqDiff / (n - 1)) : 1.0;
        if (std < 1e-10) std = 1e-10;

        meta.Min = NumOps.ToDouble(Engine.TensorMinValue(column));
        meta.Max = NumOps.ToDouble(Engine.TensorMaxValue(column));
        meta.Mean = mean;
        meta.Std = std;
    }

    /// <summary>
    /// The mean of a column and the sum of its squared deviations from that mean. With
    /// <paramref name="prescale"/>, a column whose n * max|x|^2 could overflow float is divided by its largest
    /// finite magnitude first and the moments are scaled back in double, so neither sum can overflow
    /// <typeparamref name="TValue"/>.
    /// </summary>
    private (double Mean, double SumOfSquaredDeviations) ColumnMoments<TValue>(Tensor<TValue> column, bool prescale)
    {
        var ops = MathHelper.GetNumericOperations<TValue>();
        double scale = 1.0;
        if (prescale)
        {
            double largest = ops.ToDouble(Engine.TensorMaxValue(Engine.TensorAbs(column)));
            // Only when n * largest^2 could overflow float, so ordinary columns keep their unscaled values.
            if (largest > 0 && !double.IsInfinity(largest) && !double.IsNaN(largest)
                && largest * largest * column.Length >= 1e36)
            {
                scale = largest;
                column = Engine.TensorDivideScalar(column, ops.FromDouble(largest));
            }
        }

        TValue mean = Engine.TensorMean(column);
        double sumOfSquares = ops.ToDouble(Engine.TensorSumOfSquares(Engine.TensorSubtractScalar(column, mean)));
        return (ops.ToDouble(mean) * scale, sumOfSquares * scale * scale);
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

    #endregion

    #region Helpers

    private static Vector<T> GetMatrixRow(Matrix<T> data, int row)
        => data.Columns == 0 ? new Vector<T>(0) : data.GetRow(row);

    /// <summary>Overwrites the leading min(columns, values.Length) entries of row <paramref name="row"/>.</summary>
    private static Matrix<T> SetMatrixRow(Matrix<T> data, int row, Vector<T> values)
    {
        int copyLen = Math.Min(data.Columns, values.Length);
        if (copyLen == 0)
            return data;
        var current = data.GetRow(row);
        var head = values.GetSubVector(0, copyLen);
        data.SetRow(row, copyLen == data.Columns
            ? head
            : AiDotNetEngine.Current.TensorConcatenate(
                new[] { Tensor<T>.FromVector(head), Tensor<T>.FromVector(current.GetSubVector(copyLen, data.Columns - copyLen)) }, 0)
                .ToVector());
        return data;
    }

    private static Tensor<T> VectorToTensor(Vector<T> v) => Tensor<T>.FromVector(v);

    /// <summary>The first min(length, t.Length) values of <paramref name="t"/>, zero-padded to <paramref name="length"/>.</summary>
    private static Vector<T> TensorToVector(Tensor<T> t, int length)
    {
        int copyLen = Math.Min(length, t.Length);
        if (copyLen == 0)
            return new Vector<T>(length);
        var engine = AiDotNetEngine.Current;
        var flat = engine.Reshape(t, new[] { t.Length });
        var head = copyLen == t.Length ? flat : engine.TensorSlice(flat, new[] { 0 }, new[] { copyLen });
        if (copyLen < length)
            head = engine.TensorConcatenate(new[] { head, new Tensor<T>(new[] { length - copyLen }) }, 0);
        return head.ToVector();
    }

    /// <summary>The first min(length, v.Length) values of <paramref name="v"/>, zero-padded to <paramref name="length"/>.</summary>
    private static Vector<T> PadTo(Vector<T> v, int length)
        => v.Length == length ? v : TensorToVector(Tensor<T>.FromVector(v), length);

    #endregion

}
