using AiDotNet.ActivationFunctions;
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
/// Causal-GAN generator that learns causal graph structure (directed acyclic graph)
/// and generates synthetic data respecting causal relationships between features.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Causal-GAN discovers causal structure using a NOTEARS-style continuous relaxation
/// for learning a directed acyclic graph (DAG), then generates each feature as a
/// function of its causal parents via structural equation models.
/// </para>
/// <para>
/// Architecture:
/// <code>
/// Generator:   noise --> [FC+BN+ReLU] x L (with residual) --> raw features
///              raw features --> causal mixing (I + W^T) * raw --> structured features
/// Discriminator: features --> [FC+LeakyReLU+Dropout] x L --> real/fake score (WGAN)
/// </code>
/// </para>
/// <para>
/// <b>For Beginners:</b> Causal-GAN learns which features cause other features:
///
/// Instead of just learning that "Age and Income are related" (correlation),
/// it learns "Education causes higher Income" (causation).
///
/// This has two benefits:
/// 1. Generated data respects cause-effect chains, producing more realistic samples
/// 2. You can simulate "what-if" scenarios (interventions) on specific features
///
/// The model learns a weighted adjacency matrix W where W[i,j] means feature i
/// influences feature j. A DAG penalty (NOTEARS) ensures no circular dependencies.
///
/// The training uses:
/// - WGAN-GP loss for stable adversarial training
/// - NOTEARS penalty via augmented Lagrangian for DAG acyclicity constraint
/// - L1 sparsity on the adjacency matrix for interpretable causal structure
///
/// The NOTEARS constraint is: h(W) = tr(e^(W * W)) - d = 0
/// where d is the number of features. This is zero if and only if W encodes a DAG.
/// The matrix exponential is computed by scaling and squaring, as scipy.linalg.expm in the NOTEARS reference code.
///
/// This implementation follows the standard neural network architecture pattern with:
/// - Proper inheritance from NeuralNetworkBase
/// - Layer-based architecture using ILayer components
/// - Engine-based tensor operations for CPU/GPU acceleration
/// - Full autodiff and JIT compilation support
///
/// If you provide custom layers in the architecture, those will be used directly
/// for the generator network. Otherwise, the network creates the standard architecture.
///
/// Example usage:
/// <code>
/// var architecture = new NeuralNetworkArchitecture&lt;double&gt;(
///     inputFeatures: 10,
///     outputSize: 10
/// );
/// var options = new CausalGANOptions&lt;double&gt;
/// {
///     DAGPenaltyWeight = 0.5,
///     Epochs = 300
/// };
/// var generator = new CausalGANGenerator&lt;double&gt;(architecture, options);
/// generator.Fit(data, columns, epochs: 300);
/// var synthetic = generator.Generate(1000);
/// </code>
/// </para>
/// <para>
/// Reference: Zheng et al., "DAGs with NO TEARS: Continuous Optimization for Structure Learning" (NeurIPS 2018)
/// </para>
/// </remarks>
[ModelDomain(ModelDomain.MachineLearning)]
[ModelCategory(ModelCategory.NeuralNetwork)]
[ModelCategory(ModelCategory.SyntheticDataGenerator)]
[ModelTask(ModelTask.Generation)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("CausalGAN: Learning Causal Implicit Generative Models with Adversarial Training",
    "https://arxiv.org/abs/1709.02023",
    Year = 2018,
    Authors = "Murat Kocaoglu, Christopher Snyder, Alexandros G. Dimakis, Sriram Vishwanath")]
public partial class CausalGANGenerator<T> : NeuralSyntheticTabularGeneratorBase<T>, ISyntheticTabularGenerator<T>
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

    private readonly CausalGANOptions<T> _options;
    // Separate G/D optimizers (see CTGANGenerator for the divergence rationale).
    private IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> _generatorOptimizer;
    private IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> _discriminatorOptimizer;
    private ILossFunction<T> _lossFunction;

    // Synthetic tabular data infrastructure
    private TabularDataTransformer<T>? _transformer;
    private List<ColumnMetadata> _columns = new();
    private int _dataWidth;
    private Random _random;

    // Causal structure: adjacency matrix W (numFeatures x numFeatures)
    // W[i,j] > 0 means feature i causally influences feature j
    [Buffer(Name = "causal-adjacency", Availability = AiDotNet.Models.Parameters.ParameterAvailability.Conditional)]
    private Matrix<T>? _adjacency;

    // X^T X / n of the transformed training rows: the NOTEARS least-squares loss (1/2n)||X - XW||^2 has gradient
    // (1/n) X^T (XW - X) = Gram (W - I), so the Gram is all the adjacency update needs from the data. Fit rebuilds it
    // from the rows it is given, so it is training scratch, not state.
    [Scratch]
    private Matrix<T>? _dataGram;

    // Augmented Lagrangian parameters for NOTEARS DAG constraint
    private double _lagrangeAlpha;
    private double _lagrangeRho;
    private double _prevDagConstraint;

    // Generator batch normalization layers (auxiliary, always created to match Layers)
    private readonly List<BatchNormalizationLayer<T>> _genBNLayers = new();

    // Discriminator layers (auxiliary, not user-overridable)
    private readonly List<FullyConnectedLayer<T>> _discLayers = new();
    private readonly List<DropoutLayer<T>> _discDropoutLayers = new();
    private readonly List<(int InputSize, int OutputSize)> _discLayerDims = new();

    // Tensor view of the transformed training matrix, rebuilt when a different matrix is passed in.
    [Scratch]
    private Matrix<T>? _realDataSource;
    [Scratch]
    private Tensor<T>? _realDataTensor;

    // Whether custom layers are being used
    private bool _usingCustomLayers;

    /// <summary>
    /// Gets the CausalGAN-specific options.
    /// </summary>
    public new CausalGANOptions<T> Options => _options;

    /// <inheritdoc />
    public IReadOnlyList<ColumnMetadata> Columns => _columns.AsReadOnly();

    /// <inheritdoc />
    public bool IsFitted { get; private set; }

    /// <summary>
    /// Initializes a new instance of the <see cref="CausalGANGenerator{T}"/> class.
    /// </summary>
    /// <param name="architecture">The neural network architecture defining input/output dimensions and optional custom layers.</param>
    /// <param name="options">CausalGAN-specific configuration options.</param>
    /// <param name="optimizer">Gradient-based optimizer (defaults to Adam).</param>
    /// <param name="lossFunction">Loss function (defaults based on task type).</param>
    /// <summary>
    /// Initializes a new instance with default architecture settings.
    /// </summary>
    public CausalGANGenerator()
        : this(new NeuralNetworkArchitecture<T>(
            inputType: Enums.InputType.OneDimensional,
            taskType: Enums.NeuralNetworkTaskType.Regression,
            inputSize: 10,
            outputSize: 10))
    {
    }

    public CausalGANGenerator(NeuralNetworkArchitecture<T> architecture,
        CausalGANOptions<T>? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
        ILossFunction<T>? lossFunction = null)
        : base(architecture, lossFunction ?? NeuralNetworkHelper<T>.GetDefaultLossFunction(architecture.TaskType), (options ??= new CausalGANOptions<T>()).MaxGradNorm)
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

        _lagrangeAlpha = 0.0;
        _lagrangeRho = 1.0;
        _prevDagConstraint = double.MaxValue;

        InitializeLayers();
    }

    #region Layer Initialization (GANDALF Pattern)

    /// <summary>
    /// Initializes the layers of the CausalGAN generator based on the provided architecture.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Layers = generator hidden layers (user-overridable).
    /// The discriminator is always auxiliary and not user-overridable.
    /// Generator uses residual connections with BatchNorm and manual ReLU.
    /// </para>
    /// </remarks>
    protected override void InitializeLayers()
    {
        if (Architecture.Layers != null && Architecture.Layers.Count > 0)
        {
            Layers.AddRange(Architecture.Layers);
            _usingCustomLayers = true;
        }
        else
        {
            // Create default generator layers (placeholder dims until Fit())
            BuildDefaultGeneratorLayers(_options.EmbeddingDimension, Architecture.OutputSize);
            _usingCustomLayers = false;
        }
    }

    /// <summary>
    /// Builds default generator layers with residual connections, BatchNorm, and manual ReLU.
    /// All layers use IdentityActivation so we control activation ordering.
    /// </summary>
    private void BuildDefaultGeneratorLayers(int inputDim, int outputDim)
    {
        Layers.Clear();
        _genBNLayers.Clear();

        var identity = new IdentityActivation<T>() as IActivationFunction<T>;
        var dims = _options.HiddenDimensions;

        for (int i = 0; i < dims.Length; i++)
        {
            // Residual: input dimension includes original input concatenated. This width MUST be
            // passed to the layer explicitly: the output-size-only FullyConnectedLayer constructor
            // infers its input from the preceding layer's output (dims[i-1]), which ignores the
            // concatenated original input, so the second block was built [256,256] while the forward
            // pass fed it [1, 256+inputDim]. That threw "Matrix dimensions incompatible:
            // [1,266] x [256,256]" on every Forward-only invariant (Predict_ShouldBeDeterministic,
            // Clone_ShouldProduceIdenticalOutput, BatchConsistency_SingleMatchesBatch); training-path
            // tests masked it because Fit() rebuilds the stack via RebuildLayersWithActualDimensions.
            int layerInput = i == 0 ? inputDim : dims[i - 1] + inputDim;
            Layers.Add(new FullyConnectedLayer<T>(layerInput, dims[i], identity));
            _genBNLayers.Add(new BatchNormalizationLayer<T>(dims[i]));
        }

        // Output layer: produces raw features (no activation). Its input is likewise the last hidden
        // width plus the concatenated original input.
        int lastHidden = dims.Length > 0 ? dims[^1] + inputDim : inputDim;
        Layers.Add(new FullyConnectedLayer<T>(lastHidden, outputDim, identity));
    }

    /// <summary>
    /// Rebuilds generator and discriminator layers with actual data dimensions discovered during Fit().
    /// </summary>
    private void RebuildLayersWithActualDimensions(int genInputDim, int genOutputDim, int discInputDim)
    {
        if (!_usingCustomLayers)
        {
            BuildDefaultGeneratorLayers(genInputDim, genOutputDim);
        }

        // Discriminator is always rebuilt
        BuildDiscriminator(discInputDim);
    }

    /// <summary>
    /// Builds the discriminator network with Dropout and manual LeakyReLU.
    /// Tracks layer dimensions for ComputeDiscriminatorInputGradient.
    /// </summary>
    private void BuildDiscriminator(int inputDim)
    {
        _discLayers.Clear();
        _discDropoutLayers.Clear();
        _discLayerDims.Clear();

        var identity = new IdentityActivation<T>() as IActivationFunction<T>;
        var dims = _options.HiddenDimensions;

        for (int i = 0; i < dims.Length; i++)
        {
            int layerInput = i == 0 ? inputDim : dims[i - 1];
            _discLayers.Add(new FullyConnectedLayer<T>(dims[i], identity));
            _discLayerDims.Add((layerInput, dims[i]));
            _discDropoutLayers.Add(new DropoutLayer<T>(_options.DiscriminatorDropout));
        }

        // Output layer: single scalar (raw Wasserstein score, no activation)
        int lastHidden = dims.Length > 0 ? dims[^1] : inputDim;
        _discLayers.Add(new FullyConnectedLayer<T>(1, identity));
        _discLayerDims.Add((lastHidden, 1));
    }

    #endregion

    #region Neural Network Methods (GANDALF Pattern)

    /// <summary>
    /// Runs the generator forward pass with residual connections and pre-activation caching.
    /// </summary>
    protected override Tensor<T> PredictCore(Tensor<T> input)
    {

        if (_usingCustomLayers)
        {
            var current = input;
            foreach (var layer in Layers)
            {
                current = layer.Forward(current);
            }
            return current;
        }

        return GeneratorForward(input);
    }

    // UpdateParameters re-sliced the flat vector across Layers by hand -- the base walks
    // exactly the same enumeration, so this said nothing the base does not already say.
    #endregion

    #region ISyntheticTabularGenerator<T> Implementation

    /// <summary>
    /// Fits the CausalGAN generator to the provided real tabular data.
    /// </summary>
    /// <param name="data">The real data matrix.</param>
    /// <param name="columns">Metadata describing each column.</param>
    /// <param name="epochs">Number of training epochs. When null, the model's published Epochs from its options is used.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method:
    /// 1. Transforms data using VGM normalization
    /// 2. Initializes the causal adjacency matrix
    /// 3. Trains generator and discriminator using WGAN-GP
    /// 4. Updates the causal structure using augmented Lagrangian (NOTEARS)
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

        // Step 1: Fit the data transformer
        _transformer = new TabularDataTransformer<T>(_options.VGMModes, _random);
        _transformer.Fit(data, _columns);
        _dataWidth = _transformer.TransformedWidth;
        var transformedData = _transformer.Transform(data);
        _dataGram = ComputeDataGram(transformedData);

        // Step 2: Initialize adjacency matrix with small random values (zero diagonal)
        _adjacency = InitialAdjacency(_dataWidth);

        // Step 3: Reset augmented Lagrangian parameters
        _lagrangeAlpha = 0.0;
        _lagrangeRho = 1.0;
        _prevDagConstraint = double.MaxValue;

        // Step 4: Build networks with actual dimensions
        BuildFromConstructionSequence(() => RebuildLayersWithActualDimensions(_options.EmbeddingDimension, _dataWidth, _dataWidth));

        // Step 5: Training loop
        T lr = NumOps.FromDouble(_options.LearningRate);
        int batchSize = Math.Min(_options.BatchSize, data.Rows);
        int numBatches = Math.Max(1, data.Rows / batchSize);

        for (int epoch = 0; epoch < epochCount; epoch++)
        {
            for (int batch = 0; batch < numBatches; batch++)
            {
                // Paper-faithful CausalGAN training schedule (Kocaoglu et al.
                // 2018 + NOTEARS / Zheng et al. 2018 §3.2). Critic + generator
                // both via GradientTape; the causal adjacency matrix is
                // updated separately via augmented Lagrangian after each
                // epoch.
                for (int dStep = 0; dStep < _options.DiscriminatorSteps; dStep++)
                {
                    TrainDiscriminatorStepBatched(transformedData, batchSize);
                }
                TrainGeneratorStepBatched(batchSize);
            }

            UpdateAdjacencyAugmentedLagrangian(lr);
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
            var transformedData = _transformer.Transform(data);
            _dataGram = ComputeDataGram(transformedData);

            _adjacency = InitialAdjacency(_dataWidth);

            _lagrangeAlpha = 0.0;
            _lagrangeRho = 1.0;
            _prevDagConstraint = double.MaxValue;

            BuildFromConstructionSequence(() => RebuildLayersWithActualDimensions(_options.EmbeddingDimension, _dataWidth, _dataWidth));

            T lr = NumOps.FromDouble(_options.LearningRate);
            int batchSize = Math.Min(_options.BatchSize, data.Rows);
            int numBatches = Math.Max(1, data.Rows / batchSize);

            for (int epoch = 0; epoch < epochCount; epoch++)
            {
                ct.ThrowIfCancellationRequested();
                for (int batch = 0; batch < numBatches; batch++)
                {
                    for (int dStep = 0; dStep < _options.DiscriminatorSteps; dStep++)
                    {
                        TrainDiscriminatorStepBatched(transformedData, batchSize);
                    }
                    TrainGeneratorStepBatched(batchSize);
                }
                UpdateAdjacencyAugmentedLagrangian(lr);
            }
        }, ct).ConfigureAwait(false);

        IsFitted = true;
    }

    /// <summary>
    /// Generates new synthetic tabular data rows respecting the learned causal structure.
    /// </summary>
    public Matrix<T> Generate(int numSamples, Vector<T>? conditionColumn = null, Vector<T>? conditionValue = null)
    {
        if (!IsFitted || _transformer is null || Layers.Count == 0)
        {
            throw new InvalidOperationException(
                "The generator must be fitted before generating data. Call Fit() first.");
        }

        if (numSamples <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(numSamples), "Number of samples must be positive.");
        }

        var generated = new Tensor<T>(new[] { numSamples, _dataWidth });

        // Rows are generated one at a time, in order, so each row's noise draws keep their RNG order.
        for (int i = 0; i < numSamples; i++)
        {
            var noise = CreateStandardNormalVector(_options.EmbeddingDimension);
            var genOutput = GeneratorForward(VectorToTensor(noise));

            // Apply causal structure: y = (I + W^T) * x
            if (_adjacency is not null)
            {
                genOutput = ApplyCausalStructure(genOutput);
            }

            // Apply output activations
            SyntheticTensorOps<T>.WriteRow(generated, i, ApplyOutputActivations(genOutput), _dataWidth);
        }

        // The VGM head is decoded as in CTGAN: each mode and category block is drawn from its distribution
        // (gumbel_softmax then argmax), not set to its most likely entry.
        // Decoded in bounded blocks, so only one block's intermediates are alive at a time.
        TabularDataTransformer<T> transformer = _transformer;
        int columnCount = _columns.Count;
        return SyntheticTensorOps<T>.DecodeInBlocks(generated, block => transformer.InverseTransform(
            SyntheticTensorOps<T>.SampleCategoricalBlocks(block, transformer, columnCount, _random).ToMatrix()));
    }

    #endregion

    #region Forward Passes

    /// <summary>
    /// Generator forward pass with residual connections and pre-activation caching.
    /// </summary>
    private Tensor<T> GeneratorForward(Tensor<T> input)
    {
        var inputTensor = input;
        var current = input;

        for (int i = 0; i < Layers.Count - 1; i++)
        {
            // Residual: concatenate original noise input
            if (i > 0)
            {
                current = ConcatTensors(current, inputTensor);
            }

            // FC(identity) --> BN --> cache --> ReLU
            current = Layers[i].Forward(current);
            if (i < _genBNLayers.Count)
            {
                current = _genBNLayers[i].Forward(current);
            }

            current = ApplyReLU(current);
        }

        // Final output layer with residual connection (no activation)
        current = ConcatTensors(current, inputTensor);
        current = Layers[^1].Forward(current);

        return current;
    }

    /// <summary>
    /// Reports the per-layer activations of the GENERATOR, following the same residual path
    /// <see cref="GeneratorForward"/> uses.
    /// </summary>
    /// <param name="input">The latent/noise input, sized <c>CausalGANOptions.EmbeddingDimension</c>.</param>
    /// <returns>Named activations for each generator block, in forward order.</returns>
    /// <remarks>
    /// <para>
    /// The base implementation walks <c>Layers</c> as a plain sequential chain, which is NOT this
    /// model's topology: every block after the first (and the output layer) receives its predecessor's
    /// activation CONCATENATED with the original latent input, so its weights are
    /// <c>[hidden + latent, hidden]</c>. Feeding a bare <c>[1, hidden]</c> activation into the next
    /// block therefore threw "Matrix dimensions incompatible: [1,256] x [384,256]". Mirroring
    /// <see cref="GeneratorForward"/> here keeps the reported activations identical to the ones the
    /// forward pass actually computes (BatchNorm and ReLU included), which is what makes this probe
    /// meaningful rather than merely non-throwing.
    /// </para>
    /// </remarks>
    public override Dictionary<string, Tensor<T>> GetNamedLayerActivations(Tensor<T> input)
    {
        var activations = new Dictionary<string, Tensor<T>>();
        if (_usingCustomLayers)
            return base.GetNamedLayerActivations(input);
        if (Layers.Count == 0)
            return activations;

        var inputTensor = input;
        var current = input;

        for (int i = 0; i < Layers.Count - 1; i++)
        {
            if (i > 0)
                current = ConcatTensors(current, inputTensor);

            current = Layers[i].Forward(current);
            if (i < _genBNLayers.Count)
                current = _genBNLayers[i].Forward(current);

            current = ApplyReLU(current);
            activations[$"Layer_{i}_{Layers[i].GetType().Name}"] = current.Clone();
        }

        current = ConcatTensors(current, inputTensor);
        current = Layers[^1].Forward(current);
        activations[$"Layer_{Layers.Count - 1}_{Layers[^1].GetType().Name}"] = current.Clone();

        return activations;
    }

    #endregion

    #region Training Steps

    /// <summary>
    /// Paper-faithful WGAN-GP critic step over the causal-structured fake
    /// distribution (Kocaoglu et al. 2018 §4 / Gulrajani et al. 2017). Fake
    /// samples are generated through the residual generator, then have the
    /// causal adjacency matrix applied so the critic learns the joint
    /// causal distribution. Tape-based gradients via
    /// <see cref="GradientTape{T}"/> + <see cref="TapeStepContext{T}"/>.
    /// </summary>
    private void TrainDiscriminatorStepBatched(Matrix<T> transformedData, int batchSize)
    {
        var (realBatch, fakeBatch) = BuildRealAndFakeBatches(transformedData, batchSize);
        var discLayerList = _discLayers.Cast<ILayer<T>>().ToList();

        // The shared WGAN-GP critic step: the GPU-resident primitive, the fused compiled plan or the shared eager
        // tape step, with one replay-safe gradient penalty (Gulrajani et al. 2017; #1844 / #1845).
        AiDotNet.Training.WassersteinCriticStep<T>.Step(
            this,
            discLayerList,
            realBatch,
            fakeBatch,
            input => DiscriminatorForwardBatched(input, isTraining: true),
            _options.GradientPenaltyWeight,
            _discriminatorOptimizer,
            _random,
            ref _wganFusedStep, maxGradNorm: _options.MaxGradNorm);
    }

    /// <summary>
    /// Paper-faithful generator step: minimize -E[D(G(z) ⊗ A)] where the
    /// fake samples flow through the residual generator then the causal
    /// adjacency mask (Kocaoglu et al. 2018). Tape-tracked end-to-end.
    /// </summary>
    private void TrainGeneratorStepBatched(int batchSize)
    {
        var generatorLayers = new List<ILayer<T>>();
        generatorLayers.AddRange(Layers);
        foreach (var bn in _genBNLayers) generatorLayers.Add(bn);
        var genParams = TapeTrainingStep<T>.CollectParameters(generatorLayers);

        var noiseBatch = GenerateNoiseBatchTensor(batchSize);

        // GPU-RESIDENT fast path — noise → gen → (optional causal) → activation → disc-frozen → -avgFake.
        // Disc layers not in the trainable set, so no gradients accumulate to them.
        var trainableGenLayers = generatorLayers.OfType<ITrainableLayer<T>>().ToList();
        if (trainableGenLayers.Count > 0)
        {
            var target = new Tensor<T>(new[] { 1 });
            Tensor<T> Fwd(Tensor<T> nb)
            {
                var f = GeneratorForwardBatched(nb);
                var c = _adjacency is not null ? ApplyCausalStructureBatched(f) : f;
                var a = ApplyOutputActivationsBatched(c);
                return DiscriminatorForwardBatched(a, false);
            }
            Tensor<T> Loss(Tensor<T> scores, Tensor<T> _)
            {
                var axes = Enumerable.Range(0, scores.Shape.Length).ToArray();
                return Engine.TensorNegate(Engine.ReduceMean(scores, axes, keepDims: false));
            }
            // One step through the shared training step: the fused plan when it applies, otherwise the shared
            // eager tape step on the same objective. The critic scores the samples in evaluation mode and is not
            // in the generator's parameter group, so only the generator moves.
            AiDotNet.Training.FusedTrainingStep<T>.Step(
                this, trainableGenLayers, noiseBatch, target,
                forward: Fwd,
                computeLoss: Loss,
                optimizer: _generatorOptimizer,
                maxGradNorm: _options.MaxGradNorm,
                extraTensors: genParams);
        }
    }

    private (Tensor<T> realBatch, Tensor<T> fakeBatch) BuildRealAndFakeBatches(
        Matrix<T> transformedData, int batchSize)
    {
        var noiseBatch = GenerateNoiseBatchTensor(batchSize);
        var fakeRaw = GeneratorForwardBatched(noiseBatch);
        var fakeCausal = _adjacency is not null ? ApplyCausalStructureBatched(fakeRaw) : fakeRaw;
        var fakeBatch = ApplyOutputActivationsBatched(fakeCausal);

        var rowIndices = new int[batchSize];
        for (int b = 0; b < batchSize; b++)
        {
            rowIndices[b] = _random.Next(transformedData.Rows);
        }

        if (_realDataTensor is null || !ReferenceEquals(_realDataSource, transformedData))
        {
            _realDataTensor = Tensor<T>.FromMatrix(transformedData);
            _realDataSource = transformedData;
        }
        var realBatch = SyntheticTensorOps<T>.GatherRows(_realDataTensor, rowIndices, _dataWidth);
        return (realBatch, fakeBatch);
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

    /// <summary>
    /// Batched generator forward, tape-tracked. Same residual + BN + ReLU
    /// architecture as <see cref="GeneratorForward"/>, but using
    /// <see cref="Engine.TensorConcatenate"/> + <see cref="Engine.ReLU"/>.
    /// </summary>
    private Tensor<T> GeneratorForwardBatched(Tensor<T> noise)
    {
        if (_usingCustomLayers)
        {
            var c = noise;
            foreach (var l in Layers) c = l.Forward(c);
            return c;
        }
        var h = noise;
        int numHidden = Layers.Count - 1;
        for (int i = 0; i < numHidden; i++)
        {
            if (i > 0) h = Engine.TensorConcatenate([h, noise], axis: 1);
            h = Layers[i].Forward(h);
            if (i < _genBNLayers.Count) h = _genBNLayers[i].Forward(h);
            h = Engine.ReLU(h);
        }
        h = Engine.TensorConcatenate([h, noise], axis: 1);
        h = Layers[^1].Forward(h);
        return h;
    }

    /// <summary>
    /// Batched discriminator forward (LeakyReLU 0.2 + dropout), tape-tracked.
    /// </summary>
    private Tensor<T> DiscriminatorForwardBatched(Tensor<T> input, bool isTraining)
    {
        var current = input;
        T leakySlope = NumOps.FromDouble(0.2);
        for (int i = 0; i < _discLayers.Count - 1; i++)
        {
            current = _discLayers[i].Forward(current);
            current = Engine.LeakyReLU(current, leakySlope);
            if (isTraining && i < _discDropoutLayers.Count)
                current = _discDropoutLayers[i].Forward(current);
        }
        current = _discLayers[^1].Forward(current);
        return current;
    }

    /// <summary>
    /// Applies the learned causal adjacency matrix to a batch of fake samples
    /// (Kocaoglu et al. 2018). Each sample row x is multiplied by
    /// (I + A), where A is the dataWidth x dataWidth causal adjacency matrix.
    /// Implemented as a tape-tracked TensorMatMul.
    /// </summary>
    private Tensor<T> ApplyCausalStructureBatched(Tensor<T> fakeBatch)
    {
        if (_adjacency is null) return fakeBatch;

        // Build (I + A) as a Tensor<T> of shape [dataWidth, dataWidth].
        var iPlusA = Engine.TensorAdd(Engine.TensorEye<T>(_dataWidth), Tensor<T>.FromMatrix(_adjacency));

        // [B, d] × [d, d] -> [B, d]
        return Engine.TensorMatMul(fakeBatch, iPlusA);
    }

    /// <summary>
    /// Batched per-column VGM output activation (matches CTGAN /
    /// TableGAN's tape-tracked Tanh + per-block Softmax dispatch).
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

    #region Causal Structure Learning

    /// <summary>
    /// Applies the learned causal structure: y = (I + W^T) * x.
    /// </summary>
    private Tensor<T> ApplyCausalStructure(Tensor<T> features)
    {
        if (_adjacency is null) return features;

        // y_j = x_j + sum_{i != j} W_ij x_i over the first min(dataWidth, length) entries; later entries are 0.
        var engine = Engine;
        int n = Math.Min(_dataWidth, features.Length);
        var flat = features.Reshape([features.Length]);
        var x = engine.TensorSlice(flat, [0], [n]).Reshape([1, n]);
        var weights = engine.TensorSlice(Tensor<T>.FromMatrix(_adjacency), [0, 0], [n, n]);
        var identity = engine.TensorEye<T>(n);
        var offDiagonal = engine.TensorMultiply(weights, engine.ScalarMinusTensor(NumOps.One, identity));
        var y = engine.TensorMatMul(x, engine.TensorAdd(identity, offDiagonal)).Reshape([n]);
        if (n < features.Length)
        {
            y = engine.TensorConcatenate([y, new Tensor<T>([features.Length - n])], 0);
        }

        return y.Reshape(features._shape);
    }

    /// <summary>
    /// Computes the NOTEARS acyclicity constraint and its gradient.
    /// h(W) = tr(e^(W hadamard W)) - d; gradient = 2 * e^(W hadamard W) hadamard W.
    /// Uses a scaling-and-squaring matrix exponential (SyntheticTensorOps.MatrixExponential).
    /// </summary>
    private (T ConstraintValue, Tensor<T> Gradient) ComputeNOTEARSConstraintAndGradient()
    {
        if (_adjacency is null)
        {
            return (NumOps.Zero, new Tensor<T>([0, 0]));
        }

        // Zheng et al. (2018): h(W) = tr(e^(W o W)) - d, grad h = (e^(W o W))^T o 2W, with the exact matrix
        // exponential the reference code takes from scipy.linalg.expm (not a truncated Taylor series).
        var engine = Engine;
        var w = Tensor<T>.FromMatrix(_adjacency);
        var expMatrix = SyntheticTensorOps<T>.MatrixExponential(engine.TensorMultiply(w, w));
        T constraintValue = NumOps.Subtract(engine.TensorTrace(expMatrix), NumOps.FromDouble(_dataWidth));
        var gradient = engine.TensorMultiplyScalar(
            engine.TensorMultiply(engine.TensorTranspose(expMatrix), w), NumOps.FromDouble(2.0));
        return (constraintValue, gradient);
    }

    /// <summary>
    /// Small uniform weights in [-0.005, 0.005) off the diagonal and zero on it: no variable starts as its own
    /// parent, as DAG-structure learning requires (Zheng et al. 2018's acyclicity term is zero on a DAG).
    /// </summary>
    private Matrix<T> InitialAdjacency(int width)
    {
        var draws = Engine.TensorRandomUniformRange<T>([width, width], NumOps.FromDouble(-0.005), NumOps.FromDouble(0.005), _random.Next());
        var offDiagonal = Engine.ScalarMinusTensor(NumOps.One, Engine.TensorEye<T>(width));
        return Engine.TensorMultiply(draws, offDiagonal).ToMatrix();
    }

    /// <summary>
    /// X^T X / n of the transformed rows, the data statistic of the NOTEARS least-squares score.
    /// </summary>
    private Matrix<T> ComputeDataGram(Matrix<T> transformedData)
    {
        if (transformedData.Columns != _dataWidth)
            throw new InvalidOperationException(
                $"The transformed data has {transformedData.Columns} columns but the adjacency spans {_dataWidth}.");
        if (transformedData.Rows == 0)
            throw new InvalidOperationException("CausalGAN needs at least one training row to score the causal structure.");
        return transformedData.Transpose().Multiply(transformedData).Multiply(NumOps.FromDouble(1.0 / transformedData.Rows));
    }

    /// <summary>
    /// Updates the adjacency matrix using the augmented Lagrangian method for the NOTEARS constraint.
    /// </summary>
    /// <remarks>
    /// NOTEARS (Zheng et al., 2018) minimises the least-squares score (1/2n)||X - XW||^2 + lambda ||W||_1 subject to
    /// h(W) = 0, so each step follows the score's gradient (1/n) X^T (XW - X) alongside the acyclicity and sparsity terms.
    /// Without the score, W never leaves its near-zero start and the acyclicity penalty has nothing to push against.
    /// </remarks>
    private void UpdateAdjacencyAugmentedLagrangian(T lr)
    {
        if (_adjacency is null) return;

        var engine = Engine;
        double adjLr = NumOps.ToDouble(lr) * 0.01;

        var (hT, notearGrad) = ComputeNOTEARSConstraintAndGradient();
        double h = NumOps.ToDouble(hT);
        var w = Tensor<T>.FromMatrix(_adjacency);
        // Gram (W - I) = Gram W - Gram.
        var scoreGrad = _dataGram is null
            ? new Tensor<T>([_dataWidth, _dataWidth])
            : Tensor<T>.FromMatrix(_dataGram.Multiply(_adjacency).Subtract(_dataGram));
        // The configured linear acyclicity penalty accompanies the augmented constraint.
        double lagrangianCoeff = _options.DAGPenaltyWeight + _lagrangeAlpha + _lagrangeRho * h;

        // W <- W - lr (data + DAG + sparsity gradients), with the diagonal held at zero.
        var step = engine.TensorAdd(engine.TensorAdd(scoreGrad,
                engine.TensorMultiplyScalar(notearGrad, NumOps.FromDouble(lagrangianCoeff))),
            engine.TensorMultiplyScalar(engine.TensorSign(w), NumOps.FromDouble(_options.SparsityWeight)));
        var updated = engine.TensorSubtract(w, engine.TensorMultiplyScalar(step, NumOps.FromDouble(adjLr)));
        var offDiagonal = engine.ScalarMinusTensor(NumOps.One, engine.TensorEye<T>(_dataWidth));
        _adjacency = engine.TensorMultiply(updated, offDiagonal).ToMatrix();

        // Update augmented Lagrangian parameters
        _lagrangeAlpha += _lagrangeRho * h;

        double gamma = 0.25;
        if (Math.Abs(h) > gamma * Math.Abs(_prevDagConstraint) && _prevDagConstraint < double.MaxValue)
        {
            _lagrangeRho = Math.Min(_lagrangeRho * 10.0, 1e16);
        }

        _prevDagConstraint = h;
    }

    #endregion

    #region Gradient Penalty

    #endregion

    #region Backward Passes

    #endregion

    #region Activation Functions

    private Tensor<T> ApplyReLU(Tensor<T> input) => SyntheticTensorOps<T>.ReLU(input);

    /// <summary>
    /// Applies output activations per column type: tanh for continuous, softmax for categorical.
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
                { "HiddenDimensions", _options.HiddenDimensions },
                { "DAGPenaltyWeight", _options.DAGPenaltyWeight },
                { "SparsityWeight", _options.SparsityWeight },
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

    private static Tensor<T> VectorToTensor(Vector<T> v) => Tensor<T>.FromVector(v);

    private static Tensor<T> ConcatTensors(Tensor<T> a, Tensor<T> b) => SyntheticTensorOps<T>.Concat1D(a, b);

    #endregion

}
