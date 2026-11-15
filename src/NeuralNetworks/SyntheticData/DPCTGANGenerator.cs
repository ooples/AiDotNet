using AiDotNet.Attributes;
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
/// Differentially Private CTGAN (DP-CTGAN) for generating synthetic tabular data
/// with formal (epsilon, delta)-differential privacy guarantees.
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
/// DP-CTGAN modifies the CTGAN training loop to provide differential privacy:
/// - <b>Per-sample gradient clipping</b>: Each sample's gradient is clipped to a fixed L2 norm
/// - <b>Gaussian noise</b>: Calibrated noise added to aggregated clipped gradients
/// - <b>Privacy accounting</b>: Tracks cumulative privacy cost using moments accountant
/// - <b>Early stopping</b>: Training halts when privacy budget is exhausted
/// </para>
/// <para>
/// <b>For Beginners:</b> DP-CTGAN works exactly like CTGAN but with a "privacy shield":
///
/// <b>Standard CTGAN training step:</b>
/// <code>
/// 1. Compute gradient for each sample
/// 2. Average gradients
/// 3. Update model
/// </code>
///
/// <b>DP-CTGAN training step:</b>
/// <code>
/// 1. Compute gradient for each sample
/// 2. CLIP each gradient to max norm C     -- limits individual influence
/// 3. Average clipped gradients
/// 4. ADD Gaussian noise (scaled by C)     -- obscures individual contributions
/// 5. Update model with noisy gradient
/// 6. TRACK privacy budget spent           -- ensures total privacy bound
/// </code>
///
/// If you provide custom layers in the architecture, those will be used directly
/// for the generator network. If not, the network creates industry-standard
/// CTGAN layers.
///
/// Example usage:
/// <code>
/// var architecture = new NeuralNetworkArchitecture&lt;double&gt;(
///     inputFeatures: 10,
///     outputSize: 10
/// );
/// var options = new DPCTGANOptions&lt;double&gt;
/// {
///     Epsilon = 3.0,
///     Delta = 1e-5,
///     ClipNorm = 1.0,
///     Epochs = 300
/// };
/// var generator = new DPCTGANGenerator&lt;double&gt;(architecture, options);
/// generator.Fit(data, columns, epochs: 300);
/// var synthetic = generator.Generate(1000);
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
[ResearchPaper("Deep Learning with Differential Privacy",
    "https://arxiv.org/abs/1607.00133",
    Year = 2016,
    Authors = "Martin Abadi, Andy Chu, Ian Goodfellow, H. Brendan McMahan, Ilya Mironov, Kunal Talwar, Li Zhang")]
public partial class DPCTGANGenerator<T> : NeuralSyntheticTabularGeneratorBase<T>, ISyntheticTabularGenerator<T>
{
    private readonly DPCTGANOptions<T> _options;
    // Separate G/D optimizers (see CTGANGenerator for the divergence rationale).
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

    // Cached pre-activations for proper backward passes
    [Scratch]
    private Matrix<T>? _realDataSource;
    [Scratch]
    private Tensor<T>? _realDataTensor;

    // Whether custom layers are being used (disables residual connection logic)
    private bool _usingCustomLayers;

    // Privacy accounting
    private double _computedNoiseMultiplier;
    private double _cumulativeEpsilon;

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
    private Tensor<T>? _sampleGradBuf;

    /// <summary>
    /// Gets the cumulative privacy cost (epsilon) spent so far during training.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This tells you how much of your privacy budget has been used.
    /// Once this reaches the target epsilon, training stops to preserve privacy.
    /// </para>
    /// </remarks>
    public double CumulativeEpsilon => _cumulativeEpsilon;

    /// <summary>
    /// Gets the DP-CTGAN-specific options.
    /// </summary>
    public new DPCTGANOptions<T> Options => _options;

    /// <inheritdoc />
    public IReadOnlyList<ColumnMetadata> Columns => _columns.AsReadOnly();

    /// <inheritdoc />
    public bool IsFitted { get; private set; }

    /// <summary>
    /// Initializes a new DP-CTGAN generator with the specified architecture.
    /// </summary>
    /// <param name="architecture">The neural network architecture defining input/output dimensions and optional custom layers.</param>
    /// <param name="options">DP-CTGAN-specific options for privacy, generator, and discriminator configuration.</param>
    /// <param name="optimizer">Gradient-based optimizer (defaults to Adam).</param>
    /// <param name="lossFunction">Loss function (defaults based on task type).</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This constructor creates a DP-CTGAN network. The key privacy parameters
    /// (epsilon, delta, clipNorm) are in the options object. Lower epsilon = more privacy.
    /// </para>
    /// </remarks>
    /// <summary>
    /// Initializes a new instance with default architecture settings.
    /// </summary>
    public DPCTGANGenerator()
        : this(new NeuralNetworkArchitecture<T>(
            inputType: Enums.InputType.OneDimensional,
            taskType: Enums.NeuralNetworkTaskType.Regression,
            inputSize: 10,
            outputSize: 10))
    {
    }

    public DPCTGANGenerator(NeuralNetworkArchitecture<T> architecture,
        DPCTGANOptions<T>? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
        ILossFunction<T>? lossFunction = null)
        : base(architecture, lossFunction ?? NeuralNetworkHelper<T>.GetDefaultLossFunction(architecture.TaskType), (options ??= new DPCTGANOptions<T>()).MaxGradNorm)
    {
        _options = options;
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
            Layers.AddRange(LayerHelper<T>.CreateDefaultDPCTGANGeneratorLayers(
                inputDim, outputDim, _options.GeneratorDimensions));

            _genBNLayers.Clear();
            foreach (int dim in _options.GeneratorDimensions)
            {
                // The residual generator knows each hidden width at construction time. Leaving
                // BatchNorm deferred lets the model-wide shape walk resolve this auxiliary list
                // against the raw architecture input instead (10 for the default fixture), after
                // which the first 256-wide hidden activation cannot be normalized. Bind the
                // explicit per-feature contract just as the dense layer contract is bound above.
                _genBNLayers.Add(new BatchNormalizationLayer<T>(dim));
            }
            _usingCustomLayers = false;
        }
    }

    private void RebuildLayersWithActualDimensions(int genInputDim, int genOutputDim, int discInputDim)
    {
        if (!_usingCustomLayers)
        {
            Layers.Clear();
            Layers.AddRange(LayerHelper<T>.CreateDefaultDPCTGANGeneratorLayers(
                genInputDim, genOutputDim, _options.GeneratorDimensions));

            _genBNLayers.Clear();
            foreach (int dim in _options.GeneratorDimensions)
            {
                _genBNLayers.Add(new BatchNormalizationLayer<T>(dim));
            }
        }

        _discLayers.Clear();
        _discDropoutLayers.Clear();
        _discLayerDims.Clear();
        _discLayers.AddRange(LayerHelper<T>.CreateDefaultDPCTGANDiscriminatorLayers(
            discInputDim, _options.DiscriminatorDimensions, _options.DiscriminatorDropout));

        BuildDiscriminatorDimensionMap(discInputDim);
    }

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
    /// Fits the DP-CTGAN generator to the provided real tabular data with differential privacy.
    /// </summary>
    /// <param name="data">The real data matrix.</param>
    /// <param name="columns">Column metadata.</param>
    /// <param name="epochs">Number of training epochs (may stop early if privacy budget exhausted). When null, the model's published Epochs from its options is used.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Training proceeds like CTGAN but with privacy protections.
    /// Training will stop early if the privacy budget (epsilon) is exhausted before
    /// all epochs complete. Check CumulativeEpsilon after training to see how much
    /// privacy was actually consumed.
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

        // Step 3: Setup dimensions
        _packedInputDim = (_dataWidth + _condWidth) * _options.PacSize;

        // Step 4: Rebuild layers with actual dimensions
        int genInputDim = _options.EmbeddingDimension + _condWidth;
        BuildFromConstructionSequence(() => RebuildLayersWithActualDimensions(genInputDim, _dataWidth, _packedInputDim));
        InitializeTrainingBuffers(genInputDim);

        // Step 5: Transform data
        var transformedData = _transformer.Transform(data);

        // Step 6: Compute noise multiplier from privacy budget
        ComputeNoiseMultiplier(data.Rows, epochCount);
        _cumulativeEpsilon = 0;

        // Step 7: Training loop with privacy budget check
        T lr = NumOps.FromDouble(_options.LearningRate);
        int pacSize = _options.PacSize;
        int batchSize = _options.BatchSize;
        int numPacks = Math.Max(1, batchSize / pacSize);
        int numBatches = Math.Max(1, data.Rows / (numPacks * pacSize));

        bool privacyBudgetExhausted = false;
        for (int epoch = 0; epoch < epochCount && !privacyBudgetExhausted; epoch++)
        {
            if (_cumulativeEpsilon >= _options.Epsilon)
            {
                break; // Privacy budget exhausted
            }

            for (int batch = 0; batch < numBatches && !privacyBudgetExhausted; batch++)
            {
                // DP-SGD-faithful schedule (Abadi et al. 2016 §3 over CTGAN's
                // PacGAN-packed WGAN-GP). Critic step applies the per-step
                // (clip, noise) primitive on its tape-computed gradients;
                // generator step does the standard untrained-critic max.
                //
                // CRITICAL: charge privacy budget for EVERY critic step,
                // not just once per outer batch. Each TrainDiscriminatorStepBatchedDP
                // touches real data and consumes one DP-SGD privacy slot;
                // accumulating only the outer-loop's stepEpsilon under-counts
                // by a factor of DiscriminatorSteps and lets the early-stop
                // guard (_cumulativeEpsilon >= _options.Epsilon) leak budget.
                // Check budget INSIDE the inner loop too — without that
                // mid-epoch exit, a critic step can push the run past the
                // (ε, δ) target and the remaining steps in that epoch still
                // execute, consuming extra privacy.
                for (int dStep = 0; dStep < _options.DiscriminatorSteps; dStep++)
                {
                    TrainDiscriminatorStepBatchedDP(transformedData, numPacks);
                    _cumulativeEpsilon += ComputeStepPrivacyCost(data.Rows, numPacks * pacSize);
                    if (_cumulativeEpsilon >= _options.Epsilon)
                    {
                        privacyBudgetExhausted = true;
                        break;
                    }
                }
                if (!privacyBudgetExhausted) TrainGeneratorStepBatched(transformedData, numPacks);
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

            _packedInputDim = (_dataWidth + _condWidth) * _options.PacSize;

            int genInputDim = _options.EmbeddingDimension + _condWidth;
            BuildFromConstructionSequence(() => RebuildLayersWithActualDimensions(genInputDim, _dataWidth, _packedInputDim));
            InitializeTrainingBuffers(genInputDim);

            var transformedData = _transformer.Transform(data);

            ComputeNoiseMultiplier(data.Rows, epochCount);
            _cumulativeEpsilon = 0;

            T lr = NumOps.FromDouble(_options.LearningRate);
            int pacSize = _options.PacSize;
            int batchSize = _options.BatchSize;
            int numPacks = Math.Max(1, batchSize / pacSize);
            int numBatches = Math.Max(1, data.Rows / (numPacks * pacSize));

            bool privacyBudgetExhausted = false;
            for (int epoch = 0; epoch < epochCount && !privacyBudgetExhausted; epoch++)
            {
                ct.ThrowIfCancellationRequested();
                if (_cumulativeEpsilon >= _options.Epsilon) break;

                for (int batch = 0; batch < numBatches && !privacyBudgetExhausted; batch++)
                {
                    // Charge privacy budget per critic step — each one touches
                    // real data and is its own DP-SGD privacy event. Charging
                    // only the outer-loop step under-counts by DiscriminatorSteps
                    // and leaks budget past the early-stop guard. The mid-batch
                    // privacy-exhaustion break stops further critic + generator
                    // updates so a single critic step can't push the run past
                    // the (ε, δ) target and still execute the remaining steps.
                    for (int dStep = 0; dStep < _options.DiscriminatorSteps; dStep++)
                    {
                        TrainDiscriminatorStepBatchedDP(transformedData, numPacks);
                        _cumulativeEpsilon += ComputeStepPrivacyCost(data.Rows, numPacks * pacSize);
                        if (_cumulativeEpsilon >= _options.Epsilon)
                        {
                            privacyBudgetExhausted = true;
                            break;
                        }
                    }
                    if (privacyBudgetExhausted) break;
                    TrainGeneratorStepBatched(transformedData, numPacks);
                }
            }
        }, ct).ConfigureAwait(false);

        IsFitted = true;
    }

    /// <inheritdoc />
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

        // DP-CTGAN samples like sdv-dev/CTGAN: each mode and category block is drawn from its distribution
        // (gumbel_softmax then argmax), not set to its most likely entry.
        // Decoded in bounded blocks, so only one block's intermediates are alive at a time.
        TabularDataTransformer<T> transformer = _transformer;
        int columnCount = _columns.Count;
        return SyntheticTensorOps<T>.DecodeInBlocks(generated, block => transformer.InverseTransform(
            SyntheticTensorOps<T>.SampleCategoricalBlocks(block, transformer, columnCount, _random).ToMatrix()));
    }

    #endregion

    #region Privacy Mechanisms

    /// <summary>
    /// Computes the noise multiplier from the privacy budget if not manually specified.
    /// </summary>
    private void ComputeNoiseMultiplier(int datasetSize, int epochs)
    {
        if (_options.NoiseMultiplier > 0)
        {
            _computedNoiseMultiplier = _options.NoiseMultiplier;
            return;
        }

        // Approximate noise multiplier from privacy budget
        // Using simplified Gaussian mechanism: sigma >= sqrt(2 * ln(1.25/delta)) * sensitivity / epsilon
        //
        // Per-step privacy events: each TrainDiscriminatorStepBatchedDP call
        // is a separate privacy event under Abadi 2016 §3 (the privacy budget
        // is charged per critic step in the training loop). DiscriminatorSteps
        // > 1 multiplies the number of events per outer batch, so the
        // perStepEpsilon must be divided across DiscriminatorSteps × batches ×
        // epochs — otherwise the auto-derived sigma is under-noised and the
        // reported (ε, δ) bound is overstated.
        int totalSteps = epochs
            * Math.Max(1, datasetSize / _options.BatchSize)
            * Math.Max(1, _options.DiscriminatorSteps);
        double perStepEpsilon = _options.Epsilon / Math.Max(totalSteps, 1);
        double sensitivity = _options.ClipNorm;
        double minNoise = Math.Sqrt(2.0 * Math.Log(1.25 / _options.Delta)) * sensitivity / Math.Max(perStepEpsilon, 1e-10);
        _computedNoiseMultiplier = Math.Max(minNoise, 0.1);
    }

    /// <summary>
    /// Computes the privacy cost (epsilon) for a single training step.
    /// </summary>
    private double ComputeStepPrivacyCost(int datasetSize, int batchSize)
    {
        // Simplified moments accountant approximation
        double q = Math.Min(1.0, (double)batchSize / datasetSize); // sampling rate
        double sigma = _computedNoiseMultiplier;
        return q * Math.Sqrt(2.0) / Math.Max(sigma, 1e-10);
    }

    #endregion

    #region GAN Training Steps

    /// <summary>
    /// Paper-faithful DP-SGD critic step (Abadi et al. 2016
    /// "Deep Learning with Differential Privacy" §3, layered over Xu et al.
    /// 2019 CTGAN's PacGAN-packed WGAN). Computes per-packed-example WGAN
    /// critic gradients, then applies the canonical DP-SGD primitive:
    ///   1. Clip each example's full gradient vector to <see cref="_options.ClipNorm"/>.
    ///   2. Average the clipped gradients and add Gaussian noise.
    ///   3. Hand the noised averaged gradient to the optimizer via TapeStepContext.
    /// Replaces the prior <c>UpdateDiscriminatorParametersDP</c> path that
    /// clipped/noised the parameters directly (not paper-faithful — DP-SGD
    /// operates on gradients) and threw "Backward pass must be called…" on
    /// the tape-only autodiff stack.
    /// </summary>
    private void TrainDiscriminatorStepBatchedDP(Matrix<T> transformedData, int numPacks)
    {
        if (_sampler is null) return;

        var (realPacked, fakePacked) = BuildPackedRealAndFakeBatches(transformedData, numPacks);

        var discParams = TapeTrainingStep<T>.CollectParameters(_discLayers);

        // The shared DP-SGD step (Abadi et al. 2016): each packed example's critic objective, E[D(fake)] - E[D(real)]
        // + lambda * GP (Gulrajani et al. 2017, the shared replay-safe penalty), is clipped to the global L2 norm BEFORE
        // aggregation, then averaged, noised and applied (the GPU-resident primitive when available).
        int exampleCount = Math.Max(1, realPacked.Shape[0]);
        double gpWeight = _options.GradientPenaltyWeight;
        Tensor<T> CriticForward(Tensor<T> x) => DiscriminatorForwardBatched(x, isTraining: true);
        AiDotNet.Training.DpSgdTrainingStep<T>.Step(
            discParams,
            exampleCount,
            exIdx => new[] { ExtractPackedExample(realPacked, exIdx), ExtractPackedExample(fakePacked, exIdx) },
            slots => CriticForward(slots[0]),
            (realScore, slots) =>
            {
                var fakeScore = CriticForward(slots[1]);
                var axes = Enumerable.Range(0, realScore.Shape.Length).ToArray();
                var wasserstein = Engine.TensorSubtract(
                    Engine.ReduceMean(fakeScore, axes, keepDims: false),
                    Engine.ReduceMean(realScore, axes, keepDims: false));
                var penalty = AiDotNet.Training.WassersteinCriticStep<T>.GradientPenalty(
                    Engine, slots[0], slots[1], CriticForward, _random);
                return Engine.TensorAdd(wasserstein, Engine.TensorMultiplyScalar(penalty, NumOps.FromDouble(gpWeight)));
            },
            _options.ClipNorm,
            _computedNoiseMultiplier,
            _random,
            _discriminatorOptimizer);
    }

    /// <summary>
    /// Non-DP generator step (Abadi et al. 2016 only requires privacy on
    /// the discriminator-side gradients since the generator never sees real
    /// data directly — privacy comes via the data-processing inequality).
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
        // uses a fresh condition, mask and Gumbel draw each step. Disc layers are not in the trainable set.
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
            // The CTGAN generator objective DP-CTGAN keeps: -E[D(G(z, c))] + the conditional cross-entropy.
            Tensor<T> Loss(Tensor<T> scores, Tensor<T> _)
            {
                var axes = Enumerable.Range(0, scores.Shape.Length).ToArray();
                var lossT = Engine.TensorNegate(Engine.ReduceMean(scores, axes, keepDims: false));
                if (condDim > 0 && catBlocks.Count > 0)
                {
                    if (capturedHead is null || capturedCond is null || capturedMask is null)
                        throw new InvalidOperationException(
                            "DP-CTGAN fused step: generator intermediates were not captured by the forward closure.");
                    lossT = Engine.TensorAdd(lossT,
                        SyntheticTensorOps<T>.ConditionalCrossEntropy(capturedHead, capturedCond, capturedMask, catBlocks));
                }
                return lossT;
            }
            // One step through the shared training step: the fused plan when it applies, otherwise the shared eager
            // tape step on the same objective.
            FusedTrainingStep<T>.Step(
                this, trainableGenLayers, fusedInput, target,
                forward: Fwd,
                computeLoss: Loss,
                optimizer: _generatorOptimizer,
                extraTensors: genParams);
        }
    }

    private Tensor<T> ExtractPackedExample(Tensor<T> packed, int rowIndex)
        => Engine.TensorSlice(packed, [rowIndex, 0], [1, _packedInputDim]);

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
        // Box–Muller via the seeded _random so DPCTGANOptions.Seed makes the
        // packed training path reproducible — Engine.TensorRandomUniformRange
        // bypasses _random and breaks the seed contract that the rest of the
        // sampler stack (CTGANDataSampler, _random.NextDouble paths) honours.
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
        return new Tensor<T>(noiseData, [batchSize, embedDim]);
    }

    private (Tensor<T> CondBatch, int[] RowIndices) SampleConditionalBatchWithRows(int batchSize)
        => SyntheticTensorOps<T>.SampleConditionsWithRows(_sampler, batchSize, _condWidth);

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

    private Tensor<T> ApplyOutputActivationsBatched(Tensor<T> output, Tensor<T>? gumbelNoise = null)
        => SyntheticTensorOps<T>.ApplyOutputActivationsBatched(
            output, _transformer, _columns.Count, gumbelNoise, _options.GumbelTemperature);

    #endregion

    #region Discriminator Forward/Backward

    #endregion

    #region Generator Forward/Backward with Residual Connections

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
                { "Epsilon", _options.Epsilon },
                { "Delta", _options.Delta },
                { "ClipNorm", _options.ClipNorm },
                { "CumulativeEpsilon", _cumulativeEpsilon },
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
