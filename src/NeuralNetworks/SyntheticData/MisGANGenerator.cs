using AiDotNet.ActivationFunctions;
using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Models;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Optimizers;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.NeuralNetworks.SyntheticData;

/// <summary>
/// MisGAN generator for learning from incomplete data using dual generator/discriminator
/// pairs for both data values and missingness patterns.
/// </summary>
/// <remarks>
/// <para>
/// MisGAN uses a dual-GAN architecture with four networks:
/// - <b>Data generator (G_x)</b>: Generates complete data rows from noise
/// - <b>Data discriminator (D_x)</b>: Judges masked data (only observed values) as real or fake
/// - <b>Mask generator (G_m)</b>: Generates realistic missingness patterns from noise
/// - <b>Mask discriminator (D_m)</b>: Judges missingness patterns as real or fake
///
/// Key innovations:
/// - <b>Masked discrimination</b>: D_x only sees data * mask (observed values), not complete rows
/// - <b>Missingness modeling</b>: G_m learns the missing data mechanism (MCAR/MAR/MNAR)
/// - <b>WGAN-GP training</b>: Both discriminators use Wasserstein distance with gradient penalty
/// - <b>Residual generators</b>: Skip connections for better gradient flow
///
/// This implementation follows the standard neural network architecture pattern with:
/// - Proper inheritance from NeuralNetworkBase
/// - Layer-based architecture using ILayer components
/// - Full autodiff and JIT compilation support
/// </para>
/// <para>
/// <b>For Beginners:</b> MisGAN handles datasets with missing values:
///
/// Real datasets often have missing values (e.g., patients who skip blood tests,
/// customers who don't fill in all survey questions). MisGAN learns two things:
///
/// 1. <b>What the complete data looks like</b> (the data generator)
/// 2. <b>Which values tend to be missing</b> (the mask generator)
///
/// The data discriminator only sees observed values (masking out missing ones), just like
/// in real life. This forces the data generator to produce realistic complete rows, even
/// though it only gets feedback on partial observations.
///
/// If you provide custom layers in the architecture, those will be used directly
/// for the data generator network. If not, the network creates industry-standard
/// MisGAN layers based on the original research paper specifications.
///
/// Architecture:
/// <code>
/// Data Generator:   z --> [FC+BN+ReLU] x L (with residual) --> complete row
/// Data Discriminator: (row * mask) --> [FC+LeakyReLU+Dropout] x L --> real/fake score
/// Mask Generator:   z --> [FC+BN+ReLU] x L (with residual) --> sigmoid --> mask
/// Mask Discriminator: mask --> [FC+LeakyReLU+Dropout] x L --> real/fake score
/// </code>
///
/// Training loop (per batch):
/// <code>
/// 1. Generate real masks from data, fake masks from G_m
/// 2. Train D_x: score real masked data high, fake masked data low (+ GP)
/// 3. Train D_m: score real masks high, fake masks low (+ GP)
/// 4. Train G_x: make D_x score generated masked data high
/// 5. Train G_m: make D_m score generated masks high
/// </code>
///
/// Example usage:
/// <code>
/// var architecture = new NeuralNetworkArchitecture&lt;double&gt;(
///     inputFeatures: 10,
///     outputSize: 10
/// );
/// var options = new MisGANOptions&lt;double&gt;
/// {
///     MissingRate = 0.2,
///     Epochs = 300
/// };
/// var generator = new MisGANGenerator&lt;double&gt;(architecture, options);
/// generator.Fit(data, columns, epochs: 300);
/// var synthetic = generator.Generate(1000);
/// </code>
/// </para>
/// <para>
/// Reference: "MisGAN: Learning from Incomplete Data with GANs" (Li et al., ICLR 2019)
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
[ModelDomain(ModelDomain.MachineLearning)]
[ModelCategory(ModelCategory.NeuralNetwork)]
[ModelCategory(ModelCategory.SyntheticDataGenerator)]
[ModelTask(ModelTask.Generation)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("MisGAN: Learning from Incomplete Data with Generative Adversarial Networks",
    "https://arxiv.org/abs/1902.09599",
    Year = 2019,
    Authors = "Steven Cheng-Xian Li, Bo Jiang, Benjamin Marlin")]
public class MisGANGenerator<T> : NeuralNetworkBase<T>, ISyntheticTabularGenerator<T>
{
    private readonly MisGANOptions<T> _options;
    private IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> _optimizer;
    private ILossFunction<T> _lossFunction;

    // Synthetic tabular data infrastructure
    private TabularDataTransformer<T>? _transformer;
    private List<ColumnMetadata> _columns = new();
    private int _dataWidth;
    private Random _random;

    // Data generator batch normalization layers (auxiliary, always paired with Layers)
    private readonly List<BatchNormalizationLayer<T>> _dataGenBNLayers = new();

    // Data discriminator layers (auxiliary, not user-overridable)
    private readonly List<FullyConnectedLayer<T>> _dataDiscLayers = new();
    private readonly List<DropoutLayer<T>> _dataDiscDropoutLayers = new();

    // Mask generator: FC(identity) + BN pairs with residual + manual ReLU
    private readonly List<FullyConnectedLayer<T>> _maskGenLayers = new();
    private readonly List<BatchNormalizationLayer<T>> _maskGenBNLayers = new();

    // Mask discriminator: FC(identity) + Dropout with manual LeakyReLU
    private readonly List<FullyConnectedLayer<T>> _maskDiscLayers = new();
    private readonly List<DropoutLayer<T>> _maskDiscDropoutLayers = new();

    // Pre-activation caches for each network

    // Whether custom layers are being used for the data generator
    private bool _usingCustomLayers;

    /// <summary>
    /// Gets the MisGAN-specific options.
    /// </summary>
    public new MisGANOptions<T> Options => _options;

    /// <inheritdoc />
    public IReadOnlyList<ColumnMetadata> Columns => _columns.AsReadOnly();

    /// <inheritdoc />
    public bool IsFitted { get; private set; }

    /// <summary>
    /// Initializes a new MisGAN generator with the specified architecture.
    /// </summary>
    /// <param name="architecture">The neural network architecture defining input/output dimensions and optional custom layers.</param>
    /// <param name="options">MisGAN-specific options for generator and discriminator configuration.</param>
    /// <param name="optimizer">Gradient-based optimizer (defaults to Adam).</param>
    /// <param name="lossFunction">Loss function (defaults based on task type).</param>
    /// <param name="maxGradNorm">Maximum gradient norm for clipping (default 5.0).</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This constructor creates a MisGAN network.
    ///
    /// If you provide custom layers in the architecture, those will be used directly
    /// for the data generator network. If not, the network will create industry-standard
    /// MisGAN data generator layers based on the original research paper specifications.
    ///
    /// The data discriminator, mask generator, and mask discriminator are always created
    /// internally and are not user-overridable.
    /// </para>
    /// </remarks>
    /// <summary>
    /// Initializes a new instance with default architecture settings.
    /// </summary>
    public MisGANGenerator()
        : this(new NeuralNetworkArchitecture<T>(
            inputType: Enums.InputType.OneDimensional,
            taskType: Enums.NeuralNetworkTaskType.Regression,
            inputSize: 10,
            outputSize: 10))
    {
    }

    public MisGANGenerator(
        NeuralNetworkArchitecture<T> architecture,
        MisGANOptions<T>? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
        ILossFunction<T>? lossFunction = null,
        double maxGradNorm = 5.0)
        : base(architecture, lossFunction ?? NeuralNetworkHelper<T>.GetDefaultLossFunction(architecture.TaskType), maxGradNorm)
    {
        _options = options ?? new MisGANOptions<T>();
        _lossFunction = lossFunction ?? NeuralNetworkHelper<T>.GetDefaultLossFunction(architecture.TaskType);
        _optimizer = optimizer ?? new AdamOptimizer<T, Tensor<T>, Tensor<T>>(this);
        _random = _options.Seed.HasValue
            ? RandomHelper.CreateSeededRandom(_options.Seed.Value)
            : RandomHelper.CreateSecureRandom();

        InitializeLayers();
    }

    #region Layer Initialization (GANDALF Pattern)

    /// <summary>
    /// Initializes the layers of the MisGAN network based on the provided architecture.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method sets up the data generator network structure:
    /// - If you provided custom layers in the architecture, those are used for the data generator
    /// - Otherwise, it creates the standard MisGAN data generator with residual connections
    ///
    /// The data discriminator, mask generator, and mask discriminator are always created
    /// internally and are not user-overridable.
    /// </para>
    /// </remarks>
    protected override void InitializeLayers()
    {
        if (Architecture.Layers != null && Architecture.Layers.Count > 0)
        {
            // Use the layers provided by the user for the data generator
            Layers.AddRange(Architecture.Layers);
            _usingCustomLayers = true;
        }
        else
        {
            // Create default data generator layers (residual FC + BN)
            var identity = new IdentityActivation<T>() as IActivationFunction<T>;
            int inputDim = _options.EmbeddingDimension;
            var dims = _options.HiddenDimensions;

            for (int i = 0; i < dims.Length; i++)
            {
                int layerInput = i == 0 ? inputDim : dims[i - 1] + inputDim;
                var fcLayer = new FullyConnectedLayer<T>(dims[i], identity);
                Layers.Add(fcLayer);
                _dataGenBNLayers.Add(new BatchNormalizationLayer<T>());
            }

            int lastHidden = dims.Length > 0 ? dims[^1] + inputDim : inputDim;
            Layers.Add(new FullyConnectedLayer<T>(Architecture.OutputSize, identity));
            _usingCustomLayers = false;
        }
    }

    /// <summary>
    /// Builds the data discriminator network (auxiliary, not user-overridable).
    /// </summary>
    private void BuildDataDiscriminator()
    {
        _dataDiscLayers.Clear();
        _dataDiscDropoutLayers.Clear();

        var identity = new IdentityActivation<T>() as IActivationFunction<T>;
        var dims = _options.HiddenDimensions;

        for (int i = 0; i < dims.Length; i++)
        {
            int layerInput = i == 0 ? _dataWidth : dims[i - 1];
            _dataDiscLayers.Add(new FullyConnectedLayer<T>(dims[i], identity));
            _dataDiscDropoutLayers.Add(new DropoutLayer<T>(_options.DiscriminatorDropout));
        }

        int lastHidden = dims.Length > 0 ? dims[^1] : _dataWidth;
        _dataDiscLayers.Add(new FullyConnectedLayer<T>(1, identity));
    }

    /// <summary>
    /// Builds the mask generator network (auxiliary, not user-overridable).
    /// </summary>
    private void BuildMaskGenerator()
    {
        _maskGenLayers.Clear();
        _maskGenBNLayers.Clear();

        var identity = new IdentityActivation<T>() as IActivationFunction<T>;
        int inputDim = _options.EmbeddingDimension;
        var dims = _options.HiddenDimensions;

        for (int i = 0; i < dims.Length; i++)
        {
            int layerInput = i == 0 ? inputDim : dims[i - 1] + inputDim;
            _maskGenLayers.Add(new FullyConnectedLayer<T>(dims[i], identity));
            _maskGenBNLayers.Add(new BatchNormalizationLayer<T>());
        }

        int lastHidden = dims.Length > 0 ? dims[^1] + inputDim : inputDim;
        _maskGenLayers.Add(new FullyConnectedLayer<T>(_dataWidth, identity));
    }

    /// <summary>
    /// Builds the mask discriminator network (auxiliary, not user-overridable).
    /// </summary>
    private void BuildMaskDiscriminator()
    {
        _maskDiscLayers.Clear();
        _maskDiscDropoutLayers.Clear();

        var identity = new IdentityActivation<T>() as IActivationFunction<T>;
        var dims = _options.HiddenDimensions;

        for (int i = 0; i < dims.Length; i++)
        {
            int layerInput = i == 0 ? _dataWidth : dims[i - 1];
            _maskDiscLayers.Add(new FullyConnectedLayer<T>(dims[i], identity));
            _maskDiscDropoutLayers.Add(new DropoutLayer<T>(_options.DiscriminatorDropout));
        }

        int lastHidden = dims.Length > 0 ? dims[^1] : _dataWidth;
        _maskDiscLayers.Add(new FullyConnectedLayer<T>(1, identity));
    }

    #endregion

    #region ISyntheticTabularGenerator Implementation

    /// <inheritdoc />
    public void Fit(Matrix<T> data, IReadOnlyList<ColumnMetadata> columns, int epochs)
    {
        _columns = columns.ToList();
        _transformer = new TabularDataTransformer<T>(_options.VGMModes, _random);
        _transformer.Fit(data, columns);
        _dataWidth = _transformer.TransformedWidth;
        var transformedData = _transformer.Transform(data);

        // Re-initialize data generator layers with actual data width if not using custom layers
        if (!_usingCustomLayers)
        {
            Layers.Clear();
            _dataGenBNLayers.Clear();

            var identity = new IdentityActivation<T>() as IActivationFunction<T>;
            int inputDim = _options.EmbeddingDimension;
            var dims = _options.HiddenDimensions;

            for (int i = 0; i < dims.Length; i++)
            {
                int layerInput = i == 0 ? inputDim : dims[i - 1] + inputDim;
                Layers.Add(new FullyConnectedLayer<T>(dims[i], identity));
                _dataGenBNLayers.Add(new BatchNormalizationLayer<T>());
            }

            int lastHidden = dims.Length > 0 ? dims[^1] + inputDim : inputDim;
            Layers.Add(new FullyConnectedLayer<T>(_dataWidth, identity));
        }

        // Build auxiliary networks
        BuildDataDiscriminator();
        BuildMaskGenerator();
        BuildMaskDiscriminator();

        // Training loop
        int batchSize = Math.Min(_options.BatchSize, data.Rows);
        int numBatches = Math.Max(1, data.Rows / batchSize);

        for (int epoch = 0; epoch < epochs; epoch++)
        {
            for (int batch = 0; batch < numBatches; batch++)
            {
                int batchStart = batch * batchSize;
                int batchEnd = Math.Min(batchStart + batchSize, data.Rows);
                int actualBatchSize = batchEnd - batchStart;

                T scaledLr = NumOps.FromDouble(_options.LearningRate / actualBatchSize);

                SetTrainingMode(true);
                try
                {
                    // WGAN: several critic (discriminator) updates per generator update.
                    for (int dStep = 0; dStep < Math.Max(1, _options.DiscriminatorSteps); dStep++)
                    {
                        TrainDataDiscriminatorStep(transformedData, batchStart, batchEnd);
                        TrainMaskDiscriminatorStep(transformedData, batchStart, batchEnd);
                    }

                    // Generator updates (data + mask).
                    TrainDataGeneratorStep(batchStart, batchEnd);
                    TrainMaskGeneratorStep(batchStart, batchEnd);
                }
                finally
                {
                    SetTrainingMode(false);
                }
            }
        }

        IsFitted = true;
    }

    /// <inheritdoc />
    public async Task FitAsync(Matrix<T> data, IReadOnlyList<ColumnMetadata> columns, int epochs,
        CancellationToken cancellationToken = default)
    {
        await Task.Run(() =>
        {
            cancellationToken.ThrowIfCancellationRequested();
            Fit(data, columns, epochs);
        }, cancellationToken);
    }

    /// <inheritdoc />
    public Matrix<T> Generate(int numSamples, Vector<T>? conditionColumn = null, Vector<T>? conditionValue = null)
    {
        if (_transformer is null)
        {
            throw new InvalidOperationException("Generator is not fitted. Call Fit() first.");
        }

        var transformedRows = new Matrix<T>(numSamples, _dataWidth);

        for (int i = 0; i < numSamples; i++)
        {
            var noise = CreateStandardNormalVector(_options.EmbeddingDimension);
            var fakeRow = DataGeneratorForward(VectorToTensor(noise));
            for (int j = 0; j < _dataWidth; j++)
            {
                transformedRows[i, j] = j < fakeRow.Length ? fakeRow[j] : NumOps.Zero;
            }
        }

        return _transformer.InverseTransform(transformedRows);
    }

    #endregion

    #region Forward Passes

    /// <summary>
    /// Forward pass through the data generator with residual connections.
    /// Uses Layers (the user-overridable generator layers from NeuralNetworkBase).
    /// </summary>
    private Tensor<T> DataGeneratorForward(Tensor<T> input)
    {

        if (_usingCustomLayers)
        {
            // Custom layers: simple sequential forward
            var current = input;
            for (int i = 0; i < Layers.Count; i++)
            {
                current = Layers[i].Forward(current);
            }
            return current;
        }

        // Default layers: residual connections with BN + ReLU
        var h = input;
        var originalInput = CloneTensor(input);

        for (int i = 0; i < Layers.Count - 1; i++)
        {
            if (i > 0) h = ConcatTensors(h, originalInput);
            h = Layers[i].Forward(h);
            h = _dataGenBNLayers[i].Forward(h);
            h = ApplyReLU(h);
        }

        h = ConcatTensors(h, originalInput);
        h = Layers[^1].Forward(h);
        return ApplyOutputActivations(h);
    }

    /// <summary>
    /// Forward pass through the mask generator with residual connections.
    /// Output is passed through sigmoid to produce mask probabilities.
    /// </summary>
    private Tensor<T> MaskGeneratorForward(Tensor<T> input)
    {
        var current = input;
        var originalInput = CloneTensor(input);

        for (int i = 0; i < _maskGenLayers.Count - 1; i++)
        {
            if (i > 0) current = ConcatTensors(current, originalInput);
            current = _maskGenLayers[i].Forward(current);
            current = _maskGenBNLayers[i].Forward(current);
            current = ApplyReLU(current);
        }

        current = ConcatTensors(current, originalInput);
        current = _maskGenLayers[^1].Forward(current);

        // Apply sigmoid for mask probabilities [0, 1]
        return ApplySigmoid(current);
    }

    /// <summary>
    /// Forward pass through the data discriminator. Input is masked data.
    /// </summary>
    private Tensor<T> DataDiscriminatorForward(Tensor<T> input, bool isTraining)
    {
        var current = input;

        for (int i = 0; i < _dataDiscLayers.Count - 1; i++)
        {
            current = _dataDiscLayers[i].Forward(current);
            current = ApplyLeakyReLU(current);
            if (isTraining) current = _dataDiscDropoutLayers[i].Forward(current);
        }

        current = _dataDiscLayers[^1].Forward(current);
        return current;
    }

    /// <summary>
    /// Forward pass through the mask discriminator.
    /// </summary>
    private Tensor<T> MaskDiscriminatorForward(Tensor<T> input, bool isTraining)
    {
        var current = input;

        for (int i = 0; i < _maskDiscLayers.Count - 1; i++)
        {
            current = _maskDiscLayers[i].Forward(current);
            current = ApplyLeakyReLU(current);
            if (isTraining) current = _maskDiscDropoutLayers[i].Forward(current);
        }

        current = _maskDiscLayers[^1].Forward(current);
        return current;
    }

    #endregion

    #region Training Steps (WGAN)

    // WGAN critic loss = E[D(fake)] - E[D(real)] (minimized w.r.t. the critic);
    // generator loss = -E[D(fake)]. Lipschitz continuity of each critic is enforced
    // by weight clipping (Arjovsky et al. 2017). All forwards are tape-connected so
    // autodiff backpropagates through the masked-data / mask GANs (Li et al. 2019).
    private const double WGanClip = 0.01;

    private void TrainDataDiscriminatorStep(Matrix<T> transformedData, int batchStart, int batchEnd)
    {
        // D_x discriminates real data under an OBSERVED mask versus generated data
        // under a mask sampled from the mask generator G_m (Li et al. 2019) — this
        // couples G_m to the data GAN instead of using a fixed Bernoulli mask for
        // both sides. Weight clipping runs after EVERY critic step (WGAN).
        var criticLayers = BuildDataDiscLayerList();
        for (int i = batchStart; i < batchEnd; i++)
        {
            var realRow = GetRow(transformedData, i);
            var realMask = VectorToTensor(CreateRandomMask(_dataWidth, _options.MissingRate));
            var dataNoise = VectorToTensor(CreateStandardNormalVector(_options.EmbeddingDimension));
            var maskNoise = VectorToTensor(CreateStandardNormalVector(_options.EmbeddingDimension));

            using var tape = new GradientTape<T>();
            var realScore = DataDiscriminatorForward(ElementwiseMultiplyTensor(VectorToTensor(realRow), realMask), isTraining: true);
            var fakeRow = DataGeneratorForward(dataNoise);
            var fakeMask = MaskGeneratorForward(maskNoise);
            var fakeScore = DataDiscriminatorForward(ElementwiseMultiplyTensor(fakeRow, fakeMask), isTraining: true);
            var loss = Engine.TensorSubtract(ReduceToScalar(fakeScore), ReduceToScalar(realScore));
            TapeStepOver(tape, loss, criticLayers);
            ClipWeights(criticLayers);
        }
    }

    private void TrainMaskDiscriminatorStep(Matrix<T> transformedData, int batchStart, int batchEnd)
    {
        var criticLayers = BuildMaskDiscLayerList();
        for (int i = batchStart; i < batchEnd; i++)
        {
            var realMask = VectorToTensor(CreateRandomMask(_dataWidth, _options.MissingRate));
            var noise = VectorToTensor(CreateStandardNormalVector(_options.EmbeddingDimension));

            using var tape = new GradientTape<T>();
            var realScore = MaskDiscriminatorForward(realMask, isTraining: true);
            var fakeMask = MaskGeneratorForward(noise);
            var fakeScore = MaskDiscriminatorForward(fakeMask, isTraining: true);
            var loss = Engine.TensorSubtract(ReduceToScalar(fakeScore), ReduceToScalar(realScore));
            TapeStepOver(tape, loss, criticLayers);
            ClipWeights(criticLayers);
        }
    }

    private void TrainDataGeneratorStep(int batchStart, int batchEnd)
    {
        // G_x is optimized against D_x using a mask drawn from G_m, so the data and
        // mask generators are trained jointly against the masked-data critic.
        for (int i = batchStart; i < batchEnd; i++)
        {
            var dataNoise = VectorToTensor(CreateStandardNormalVector(_options.EmbeddingDimension));
            var maskNoise = VectorToTensor(CreateStandardNormalVector(_options.EmbeddingDimension));

            using var tape = new GradientTape<T>();
            var fakeRow = DataGeneratorForward(dataNoise);
            var fakeMask = MaskGeneratorForward(maskNoise);
            var fakeScore = DataDiscriminatorForward(ElementwiseMultiplyTensor(fakeRow, fakeMask), isTraining: true);
            var loss = Engine.TensorNegate(ReduceToScalar(fakeScore));
            TapeStepOver(tape, loss, BuildDataGenLayerList());
        }
    }

    private void TrainMaskGeneratorStep(int batchStart, int batchEnd)
    {
        for (int i = batchStart; i < batchEnd; i++)
        {
            var noise = VectorToTensor(CreateStandardNormalVector(_options.EmbeddingDimension));

            using var tape = new GradientTape<T>();
            var fakeMask = MaskGeneratorForward(noise);
            var fakeScore = MaskDiscriminatorForward(fakeMask, isTraining: true);
            var loss = Engine.TensorNegate(ReduceToScalar(fakeScore));
            TapeStepOver(tape, loss, BuildMaskGenLayerList());
        }
    }

    #endregion

    #region Tape Step Helpers

    /// <summary>Runs one optimizer step over the given sub-network's parameters from a tape-tracked loss.</summary>
    private void TapeStepOver(GradientTape<T> tape, Tensor<T> loss, IReadOnlyList<ILayer<T>> layers)
    {
        var trainable = Training.TapeTrainingStep<T>.CollectParameters(layers);
        if (trainable.Count == 0) return;
        var grads = tape.ComputeGradients(loss, trainable);
        T lossValue = loss.Length > 0 ? loss[0] : NumOps.Zero;
        LastLoss = lossValue;
        var ctx = new AiDotNet.Tensors.Engines.Autodiff.TapeStepContext<T>(trainable, grads, lossValue);
        _optimizer.Step(ctx);
    }

    private Tensor<T> ReduceToScalar(Tensor<T> t)
        => Engine.ReduceSum(t, Enumerable.Range(0, t.Shape.Length).ToArray(), keepDims: false);

    /// <summary>WGAN weight clipping: clamp each critic parameter to [-c, c] for Lipschitz continuity.</summary>
    private void ClipWeights(IReadOnlyList<ILayer<T>> layers)
    {
        T lo = NumOps.FromDouble(-WGanClip);
        T hi = NumOps.FromDouble(WGanClip);
        foreach (var layer in layers)
        {
            var ps = layer.GetParameters();
            if (ps.Length == 0) continue;
            bool changed = false;
            for (int i = 0; i < ps.Length; i++)
            {
                if (NumOps.GreaterThan(ps[i], hi)) { ps[i] = hi; changed = true; }
                else if (NumOps.LessThan(ps[i], lo)) { ps[i] = lo; changed = true; }
            }
            if (changed) layer.UpdateParameters(ps);
        }
    }

    private IReadOnlyList<ILayer<T>> BuildDataGenLayerList()
    {
        var all = new List<ILayer<T>>(Layers);
        all.AddRange(_dataGenBNLayers);
        return all;
    }

    private IReadOnlyList<ILayer<T>> BuildMaskGenLayerList()
    {
        var all = new List<ILayer<T>>(_maskGenLayers);
        all.AddRange(_maskGenBNLayers);
        return all;
    }

    private IReadOnlyList<ILayer<T>> BuildDataDiscLayerList()
    {
        var all = new List<ILayer<T>>(_dataDiscLayers);
        all.AddRange(_dataDiscDropoutLayers);
        return all;
    }

    private IReadOnlyList<ILayer<T>> BuildMaskDiscLayerList()
    {
        var all = new List<ILayer<T>>(_maskDiscLayers);
        all.AddRange(_maskDiscDropoutLayers);
        return all;
    }

    #endregion

    #region NeuralNetworkBase Required Overrides

    /// <inheritdoc />
    public override Tensor<T> Predict(Tensor<T> input)
    {
        return DataGeneratorForward(input);
    }

    /// <inheritdoc />
    public override void Train(Tensor<T> input, Tensor<T> expectedOutput)
    {
        // The full adversarial training (data + mask GANs) runs in Fit. This
        // single-step entry point trains the data generator (the Layers chain,
        // i.e. what Predict runs) to reconstruct the target via a tape-connected
        // step, satisfying the NeuralNetworkBase training contract. The previous
        // body was a no-op, so the generated training invariants saw no change.
        SetTrainingMode(true);
        try
        {
            using var tape = new GradientTape<T>();
            var output = DataGeneratorForward(input);
            var flatOut = output.Rank == 1 ? output : Engine.Reshape(output, new[] { output.Length });
            var target = expectedOutput.Rank == 1 ? expectedOutput : Engine.Reshape(expectedOutput, new[] { expectedOutput.Length });
            // Honor the ctor-supplied / task-default loss function (tape-connected)
            // rather than hardcoding squared error; fall back to MSE only if the
            // configured loss does not expose a tape implementation.
            Tensor<T> loss = LossFunction is LossFunctions.LossFunctionBase<T> tapeLoss
                ? tapeLoss.ComputeTapeLoss(flatOut, target)
                : ReduceToScalar(Engine.TensorSquare(Engine.TensorSubtract(flatOut, target)));
            TapeStepOver(tape, loss, BuildDataGenLayerList());
        }
        finally
        {
            SetTrainingMode(false);
        }
    }

    /// <inheritdoc />
    public override void SetTrainingMode(bool isTraining)
    {
        base.SetTrainingMode(isTraining);
        // The auxiliary BatchNorm/Dropout sub-networks live outside Layers, so the
        // base walk does not reach them; propagate the mode explicitly so BatchNorm
        // uses running stats and Dropout is disabled at inference (Predict/Generate).
        foreach (var bn in _dataGenBNLayers) bn.SetTrainingMode(isTraining);
        foreach (var bn in _maskGenBNLayers) bn.SetTrainingMode(isTraining);
        foreach (var drop in _dataDiscDropoutLayers) drop.SetTrainingMode(isTraining);
        foreach (var drop in _maskDiscDropoutLayers) drop.SetTrainingMode(isTraining);
    }

    /// <inheritdoc />
    public override void UpdateParameters(Vector<T> parameters)
    {
        int offset = 0;
        foreach (var layer in Layers)
        {
            var layerParams = layer.GetParameters();
            int count = layerParams.Length;
            if (offset + count <= parameters.Length)
            {
                var slice = new Vector<T>(count);
                for (int i = 0; i < count; i++)
                {
                    slice[i] = parameters[offset + i];
                }
                layer.UpdateParameters(slice);
                offset += count;
            }
        }
    }

    /// <inheritdoc />
    protected override void SerializeNetworkSpecificData(BinaryWriter writer)
    {
        writer.Write(_options.EmbeddingDimension);
        writer.Write(_options.MissingRate);
        writer.Write(_dataWidth);
        writer.Write(IsFitted);
    }

    /// <inheritdoc />
    protected override void DeserializeNetworkSpecificData(BinaryReader reader)
    {
        _ = reader.ReadInt32(); // EmbeddingDimension
        _ = reader.ReadDouble(); // MissingRate
        _dataWidth = reader.ReadInt32();
        IsFitted = reader.ReadBoolean();
    }

    /// <inheritdoc />
    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance()
    {
        return new MisGANGenerator<T>(Architecture, _options);
    }

    /// <inheritdoc />
    public override Dictionary<string, T> GetFeatureImportance()
    {
        var importance = new Dictionary<string, T>();
        for (int i = 0; i < _columns.Count; i++)
        {
            importance[$"feature_{i}"] = NumOps.One;
        }
        return importance;
    }

    /// <inheritdoc />
    public override ModelMetadata<T> GetModelMetadata()
    {
        return new ModelMetadata<T>
        {
            AdditionalInfo = new Dictionary<string, object>
            {
                ["generator_type"] = "MisGAN",
                ["embedding_dimension"] = _options.EmbeddingDimension,
                ["hidden_dimensions"] = _options.HiddenDimensions,
                ["missing_rate"] = _options.MissingRate,
                ["gradient_penalty_weight"] = _options.GradientPenaltyWeight,
                ["is_fitted"] = IsFitted,
                ["data_width"] = _dataWidth,
                ["using_custom_layers"] = _usingCustomLayers
            }
        };
    }

    #endregion

    #region Activation Functions

    private Tensor<T> ApplyReLU(Tensor<T> input) => Engine.TensorReLU(input);


    private Tensor<T> ApplyLeakyReLU(Tensor<T> input) => Engine.TensorLeakyReLU(input, NumOps.FromDouble(0.2));


    private Tensor<T> ApplySigmoid(Tensor<T> input) => Engine.TensorSigmoid(input);

    private Tensor<T> ApplyOutputActivations(Tensor<T> output)
    {
        if (_transformer is null) return output;

        var flat = output.Rank == 1 ? output : Engine.Reshape(output, new[] { output.Length });
        var blocks = new List<Tensor<T>>();
        int idx = 0;
        for (int col = 0; col < _columns.Count && idx < flat.Length; col++)
        {
            var transform = _transformer.GetTransformInfo(col);
            if (transform.IsContinuous)
            {
                // tanh on the normalized value scalar.
                blocks.Add(Engine.TensorTanh(Engine.TensorSlice(flat, new[] { idx }, new[] { 1 })));
                idx++;
                int numModes = transform.Width - 1;
                if (numModes > 0)
                {
                    blocks.Add(Engine.TensorSoftmax(Engine.TensorSlice(flat, new[] { idx }, new[] { numModes }), axis: 0));
                    idx += numModes;
                }
            }
            else
            {
                blocks.Add(Engine.TensorSoftmax(Engine.TensorSlice(flat, new[] { idx }, new[] { transform.Width }), axis: 0));
                idx += transform.Width;
            }
        }

        if (blocks.Count == 0) return flat;
        if (idx < flat.Length) blocks.Add(Engine.TensorSlice(flat, new[] { idx }, new[] { flat.Length - idx }));
        return Engine.TensorConcatenate(blocks.ToArray(), axis: 0);
    }


    #endregion

    #region Helpers

    /// <summary>
    /// Sanitizes a gradient tensor by clamping NaN/Inf and clipping to max norm.
    /// </summary>
    private Tensor<T> SafeGradient(Tensor<T> grad, double maxNorm)
    {
        var result = new Tensor<T>(grad._shape);
        double normSq = 0;

        for (int i = 0; i < grad.Length; i++)
        {
            double val = NumOps.ToDouble(grad[i]);
            if (double.IsNaN(val) || double.IsInfinity(val)) val = 0;
            result[i] = NumOps.FromDouble(val);
            normSq += val * val;
        }

        double norm = Math.Sqrt(normSq + 1e-12);
        if (norm > maxNorm)
        {
            double scale = maxNorm / norm;
            for (int i = 0; i < result.Length; i++)
            {
                result[i] = NumOps.FromDouble(NumOps.ToDouble(result[i]) * scale);
            }
        }

        return result;
    }

    /// <summary>
    /// Creates a standard normal random vector of the given size.
    /// </summary>
    private Vector<T> CreateStandardNormalVector(int size)
    {
        var v = new Vector<T>(size);
        for (int i = 0; i < size; i++)
        {
            // Box-Muller transform
            double u1 = Math.Max(1e-10, _random.NextDouble());
            double u2 = _random.NextDouble();
            double z = Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2);
            v[i] = NumOps.FromDouble(z);
        }
        return v;
    }

    /// <summary>
    /// Creates a random binary mask vector where each element is 1 (observed) with probability (1 - missingRate).
    /// </summary>
    private Vector<T> CreateRandomMask(int width, double missingRate)
    {
        var mask = new Vector<T>(width);
        for (int j = 0; j < width; j++)
        {
            mask[j] = NumOps.FromDouble(_random.NextDouble() > missingRate ? 1.0 : 0.0);
        }
        return mask;
    }

    private Vector<T> ElementwiseMultiply(Vector<T> a, Vector<T> b)
    {
        int len = Math.Min(a.Length, b.Length);
        var result = new Vector<T>(len);
        for (int i = 0; i < len; i++)
            result[i] = NumOps.Multiply(a[i], b[i]);
        return result;
    }

    private Tensor<T> ElementwiseMultiplyTensor(Tensor<T> a, Tensor<T> b)
    {
        var a1 = a.Rank == 1 ? a : Engine.Reshape(a, new[] { a.Length });
        var b1 = b.Rank == 1 ? b : Engine.Reshape(b, new[] { b.Length });
        return Engine.TensorMultiply(a1, b1);
    }

    private static Vector<T> GetRow(Matrix<T> matrix, int row)
    {
        var numOps = MathHelper.GetNumericOperations<T>();
        var v = new Vector<T>(matrix.Columns);
        for (int j = 0; j < matrix.Columns; j++)
            v[j] = matrix[row, j];
        return v;
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

    private Tensor<T> ConcatTensors(Tensor<T> a, Tensor<T> b)
    {
        var a1 = a.Rank == 1 ? a : Engine.Reshape(a, new[] { a.Length });
        var b1 = b.Rank == 1 ? b : Engine.Reshape(b, new[] { b.Length });
        return Engine.TensorConcatenate(new[] { a1, b1 }, axis: 0);
    }

    private static Tensor<T> CloneTensor(Tensor<T> source)
    {
        var clone = new Tensor<T>(source._shape);
        for (int i = 0; i < source.Length; i++)
            clone[i] = source[i];
        return clone;
    }

    #endregion

}
