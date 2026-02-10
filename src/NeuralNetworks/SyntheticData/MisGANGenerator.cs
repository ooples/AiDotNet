using AiDotNet.ActivationFunctions;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Models;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Optimizers;
using AiDotNet.Tensors.Helpers;

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
    private readonly List<Tensor<T>> _dataGenPreActs = new();
    private readonly List<Tensor<T>> _dataDiscPreActs = new();
    private readonly List<Tensor<T>> _maskGenPreActs = new();
    private readonly List<Tensor<T>> _maskDiscPreActs = new();

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
                var fcLayer = new FullyConnectedLayer<T>(layerInput, dims[i], identity);
                Layers.Add(fcLayer);
                _dataGenBNLayers.Add(new BatchNormalizationLayer<T>(dims[i]));
            }

            int lastHidden = dims.Length > 0 ? dims[^1] + inputDim : inputDim;
            Layers.Add(new FullyConnectedLayer<T>(lastHidden, Architecture.OutputSize, identity));
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
            _dataDiscLayers.Add(new FullyConnectedLayer<T>(layerInput, dims[i], identity));
            _dataDiscDropoutLayers.Add(new DropoutLayer<T>(_options.DiscriminatorDropout));
        }

        int lastHidden = dims.Length > 0 ? dims[^1] : _dataWidth;
        _dataDiscLayers.Add(new FullyConnectedLayer<T>(lastHidden, 1, identity));
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
            _maskGenLayers.Add(new FullyConnectedLayer<T>(layerInput, dims[i], identity));
            _maskGenBNLayers.Add(new BatchNormalizationLayer<T>(dims[i]));
        }

        int lastHidden = dims.Length > 0 ? dims[^1] + inputDim : inputDim;
        _maskGenLayers.Add(new FullyConnectedLayer<T>(lastHidden, _dataWidth, identity));
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
            _maskDiscLayers.Add(new FullyConnectedLayer<T>(layerInput, dims[i], identity));
            _maskDiscDropoutLayers.Add(new DropoutLayer<T>(_options.DiscriminatorDropout));
        }

        int lastHidden = dims.Length > 0 ? dims[^1] : _dataWidth;
        _maskDiscLayers.Add(new FullyConnectedLayer<T>(lastHidden, 1, identity));
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
                Layers.Add(new FullyConnectedLayer<T>(layerInput, dims[i], identity));
                _dataGenBNLayers.Add(new BatchNormalizationLayer<T>(dims[i]));
            }

            int lastHidden = dims.Length > 0 ? dims[^1] + inputDim : inputDim;
            Layers.Add(new FullyConnectedLayer<T>(lastHidden, _dataWidth, identity));
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

                // Train both discriminators
                for (int dStep = 0; dStep < _options.DiscriminatorSteps; dStep++)
                {
                    TrainDataDiscriminatorStep(transformedData, batchStart, batchEnd, scaledLr);
                    TrainMaskDiscriminatorStep(transformedData, batchStart, batchEnd, scaledLr);
                }

                // Train both generators
                TrainDataGeneratorStep(actualBatchSize, scaledLr);
                TrainMaskGeneratorStep(actualBatchSize, scaledLr);
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
        _dataGenPreActs.Clear();

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
            _dataGenPreActs.Add(CloneTensor(h));
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
        _maskGenPreActs.Clear();
        var current = input;
        var originalInput = CloneTensor(input);

        for (int i = 0; i < _maskGenLayers.Count - 1; i++)
        {
            if (i > 0) current = ConcatTensors(current, originalInput);
            current = _maskGenLayers[i].Forward(current);
            current = _maskGenBNLayers[i].Forward(current);
            _maskGenPreActs.Add(CloneTensor(current));
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
        _dataDiscPreActs.Clear();
        var current = input;

        for (int i = 0; i < _dataDiscLayers.Count - 1; i++)
        {
            current = _dataDiscLayers[i].Forward(current);
            _dataDiscPreActs.Add(CloneTensor(current));
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
        _maskDiscPreActs.Clear();
        var current = input;

        for (int i = 0; i < _maskDiscLayers.Count - 1; i++)
        {
            current = _maskDiscLayers[i].Forward(current);
            _maskDiscPreActs.Add(CloneTensor(current));
            current = ApplyLeakyReLU(current);
            if (isTraining) current = _maskDiscDropoutLayers[i].Forward(current);
        }

        current = _maskDiscLayers[^1].Forward(current);
        return current;
    }

    #endregion

    #region Training Steps

    /// <summary>
    /// Trains data discriminator with WGAN loss + gradient penalty.
    /// D_x sees real data * mask vs generated data * mask.
    /// </summary>
    private void TrainDataDiscriminatorStep(Matrix<T> transformedData, int batchStart, int batchEnd, T scaledLr)
    {
        for (int i = batchStart; i < batchEnd; i++)
        {
            var realRow = GetRow(transformedData, i);

            // Create real mask (simulate missingness)
            var mask = CreateRandomMask(_dataWidth, _options.MissingRate);
            var maskedReal = ElementwiseMultiply(realRow, mask);

            // WGAN: minimize E[D(fake)] - E[D(real)]
            // Real: backward with gradient = -1 (maximize D(real))
            _ = DataDiscriminatorForward(VectorToTensor(maskedReal), isTraining: true);
            var realGrad = new Tensor<T>([1]);
            realGrad[0] = NumOps.Negate(NumOps.One);
            BackwardDataDiscriminator(realGrad);
            UpdateDataDiscriminatorParameters(scaledLr);

            // Fake: generate data, apply generated mask
            var dataNoise = CreateStandardNormalVector(_options.EmbeddingDimension);
            var fakeRow = DataGeneratorForward(VectorToTensor(dataNoise));
            var maskedFake = ElementwiseMultiplyTensor(fakeRow, VectorToTensor(mask));

            // Fake: backward with gradient = +1 (minimize D(fake))
            _ = DataDiscriminatorForward(maskedFake, isTraining: true);
            var fakeGrad = new Tensor<T>([1]);
            fakeGrad[0] = NumOps.One;
            BackwardDataDiscriminator(fakeGrad);
            UpdateDataDiscriminatorParameters(scaledLr);

            // Gradient penalty
            ApplyDataGradientPenalty(maskedReal, TensorToVector(maskedFake, _dataWidth), scaledLr);
        }
    }

    /// <summary>
    /// Trains mask discriminator with WGAN loss + gradient penalty.
    /// D_m sees real masks vs generated masks.
    /// </summary>
    private void TrainMaskDiscriminatorStep(Matrix<T> transformedData, int batchStart, int batchEnd, T scaledLr)
    {
        for (int i = batchStart; i < batchEnd; i++)
        {
            // Create real mask
            var realMask = CreateRandomMask(_dataWidth, _options.MissingRate);

            // WGAN real: gradient = -1
            _ = MaskDiscriminatorForward(VectorToTensor(realMask), isTraining: true);
            var realGrad = new Tensor<T>([1]);
            realGrad[0] = NumOps.Negate(NumOps.One);
            BackwardMaskDiscriminator(realGrad);
            UpdateMaskDiscriminatorParameters(scaledLr);

            // Generate fake mask
            var maskNoise = CreateStandardNormalVector(_options.EmbeddingDimension);
            var fakeMask = MaskGeneratorForward(VectorToTensor(maskNoise));

            // WGAN fake: gradient = +1
            _ = MaskDiscriminatorForward(fakeMask, isTraining: true);
            var fakeGrad = new Tensor<T>([1]);
            fakeGrad[0] = NumOps.One;
            BackwardMaskDiscriminator(fakeGrad);
            UpdateMaskDiscriminatorParameters(scaledLr);

            // Gradient penalty
            ApplyMaskGradientPenalty(realMask, TensorToVector(fakeMask, _dataWidth), scaledLr);
        }
    }

    /// <summary>
    /// Trains data generator: minimize -E[D_x(G_x(z) * mask)].
    /// </summary>
    private void TrainDataGeneratorStep(int batchSize, T scaledLr)
    {
        for (int i = 0; i < batchSize; i++)
        {
            var noise = CreateStandardNormalVector(_options.EmbeddingDimension);
            var mask = CreateRandomMask(_dataWidth, _options.MissingRate);

            // Forward through data generator
            var fakeRow = DataGeneratorForward(VectorToTensor(noise));
            var maskedFake = ElementwiseMultiplyTensor(fakeRow, VectorToTensor(mask));

            // Forward through data discriminator (eval)
            _ = DataDiscriminatorForward(maskedFake, isTraining: false);

            // Backprop through discriminator to get gradient w.r.t. input using autodiff
            var allDataDiscLayers = BuildDataDiscLayerList();
            var discInputGrad = TapeLayerBridge<T>.ComputeInputGradient(
                maskedFake,
                allDataDiscLayers,
                TapeLayerBridge<T>.HiddenActivation.LeakyReLU,
                applyActivationOnLast: false);

            // Negate for generator loss: minimize -E[D(fake)]
            for (int g = 0; g < discInputGrad.Length; g++)
            {
                discInputGrad[g] = NumOps.Negate(discInputGrad[g]);
            }

            // Sanitize and clip gradient to prevent NaN propagation and exploding gradients
            discInputGrad = SafeGradient(discInputGrad, 5.0);

            // Apply mask to gradient (only observe gradients for non-masked features)
            for (int j = 0; j < _dataWidth && j < discInputGrad.Length; j++)
            {
                discInputGrad[j] = NumOps.Multiply(discInputGrad[j], mask[j]);
            }

            // Re-forward through data generator for caches
            _ = DataGeneratorForward(VectorToTensor(noise));

            // Backward through data generator
            BackwardDataGenerator(discInputGrad);
            UpdateDataGeneratorParameters(scaledLr);
        }
    }

    /// <summary>
    /// Trains mask generator: minimize -E[D_m(G_m(z))].
    /// </summary>
    private void TrainMaskGeneratorStep(int batchSize, T scaledLr)
    {
        for (int i = 0; i < batchSize; i++)
        {
            var noise = CreateStandardNormalVector(_options.EmbeddingDimension);

            // Forward through mask generator
            var fakeMask = MaskGeneratorForward(VectorToTensor(noise));

            // Forward through mask discriminator (eval)
            _ = MaskDiscriminatorForward(fakeMask, isTraining: false);

            // Backprop through mask discriminator to get gradient w.r.t. input using autodiff
            var allMaskDiscLayers = BuildMaskDiscLayerList();
            var discInputGrad = TapeLayerBridge<T>.ComputeInputGradient(
                fakeMask,
                allMaskDiscLayers,
                TapeLayerBridge<T>.HiddenActivation.LeakyReLU,
                applyActivationOnLast: false);

            // Negate for generator loss: minimize -E[D(fake)]
            for (int g = 0; g < discInputGrad.Length; g++)
            {
                discInputGrad[g] = NumOps.Negate(discInputGrad[g]);
            }

            // Sanitize and clip gradient to prevent NaN propagation and exploding gradients
            discInputGrad = SafeGradient(discInputGrad, 5.0);

            // Re-forward through mask generator for caches
            _ = MaskGeneratorForward(VectorToTensor(noise));

            // Backward through mask generator
            BackwardMaskGenerator(discInputGrad);
            UpdateMaskGeneratorParameters(scaledLr);
        }
    }

    #endregion

    #region Gradient Penalty

    private void ApplyDataGradientPenalty(Vector<T> real, Vector<T> fake, T scaledLr)
    {
        var allLayers = BuildDataDiscLayerList();
        ApplyGradientPenaltyGeneric(real, fake, scaledLr, allLayers,
            input => DataDiscriminatorForward(input, isTraining: false),
            BackwardDataDiscriminator, UpdateDataDiscriminatorParameters);
    }

    private void ApplyMaskGradientPenalty(Vector<T> real, Vector<T> fake, T scaledLr)
    {
        var allLayers = BuildMaskDiscLayerList();
        ApplyGradientPenaltyGeneric(real, fake, scaledLr, allLayers,
            input => MaskDiscriminatorForward(input, isTraining: false),
            BackwardMaskDiscriminator, UpdateMaskDiscriminatorParameters);
    }

    /// <summary>
    /// Generic WGAN-GP gradient penalty for any discriminator using TapeLayerBridge autodiff.
    /// </summary>
    private void ApplyGradientPenaltyGeneric(
        Vector<T> real, Vector<T> fake, T scaledLr,
        IReadOnlyList<ILayer<T>> allDiscLayers,
        Func<Tensor<T>, Tensor<T>> forwardFn,
        Action<Tensor<T>> backwardFn,
        Action<T> updateFn)
    {
        double alpha = _random.NextDouble();
        int len = Math.Min(real.Length, fake.Length);
        var interpolated = new Vector<T>(len);

        for (int i = 0; i < len; i++)
        {
            interpolated[i] = NumOps.Add(
                NumOps.Multiply(NumOps.FromDouble(alpha), real[i]),
                NumOps.Multiply(NumOps.FromDouble(1.0 - alpha), fake[i]));
        }

        // Compute input gradient using TapeLayerBridge autodiff
        var interpolatedTensor = VectorToTensor(interpolated);
        var inputGrad = TapeLayerBridge<T>.ComputeInputGradient(
            interpolatedTensor,
            allDiscLayers,
            TapeLayerBridge<T>.HiddenActivation.LeakyReLU,
            applyActivationOnLast: false);

        // Compute L2 norm
        double gradNormSq = 0;
        for (int i = 0; i < inputGrad.Length; i++)
        {
            double g = NumOps.ToDouble(inputGrad[i]);
            gradNormSq += g * g;
        }
        double gradNorm = Math.Sqrt(gradNormSq + 1e-12);

        double penaltyGradScale = 2.0 * _options.GradientPenaltyWeight * (gradNorm - 1.0) / gradNorm;

        if (Math.Abs(penaltyGradScale) > 1e-10)
        {
            _ = forwardFn(VectorToTensor(interpolated));
            var penaltyGrad = new Tensor<T>([1]);
            penaltyGrad[0] = NumOps.FromDouble(penaltyGradScale);
            backwardFn(penaltyGrad);
            updateFn(scaledLr);
        }
    }

    /// <summary>
    /// Builds a combined list of data discriminator layers for TapeLayerBridge.
    /// </summary>
    private IReadOnlyList<ILayer<T>> BuildDataDiscLayerList()
    {
        var allLayers = new List<ILayer<T>>();
        for (int i = 0; i < _dataDiscDropoutLayers.Count; i++)
        {
            allLayers.Add(_dataDiscLayers[i]);
            allLayers.Add(_dataDiscDropoutLayers[i]);
        }
        allLayers.Add(_dataDiscLayers[^1]); // output layer
        return allLayers;
    }

    /// <summary>
    /// Builds a combined list of mask discriminator layers for TapeLayerBridge.
    /// </summary>
    private IReadOnlyList<ILayer<T>> BuildMaskDiscLayerList()
    {
        var allLayers = new List<ILayer<T>>();
        for (int i = 0; i < _maskDiscDropoutLayers.Count; i++)
        {
            allLayers.Add(_maskDiscLayers[i]);
            allLayers.Add(_maskDiscDropoutLayers[i]);
        }
        allLayers.Add(_maskDiscLayers[^1]); // output layer
        return allLayers;
    }


    #endregion

    #region Backward Passes

    private void BackwardDataDiscriminator(Tensor<T> gradOutput)
    {
        var current = gradOutput;
        current = _dataDiscLayers[^1].Backward(current);
        for (int i = _dataDiscLayers.Count - 2; i >= 0; i--)
        {
            if (i < _dataDiscPreActs.Count)
                current = ApplyLeakyReLUDerivative(current, _dataDiscPreActs[i]);
            current = _dataDiscLayers[i].Backward(current);
        }
    }

    private void BackwardMaskDiscriminator(Tensor<T> gradOutput)
    {
        var current = gradOutput;
        current = _maskDiscLayers[^1].Backward(current);
        for (int i = _maskDiscLayers.Count - 2; i >= 0; i--)
        {
            if (i < _maskDiscPreActs.Count)
                current = ApplyLeakyReLUDerivative(current, _maskDiscPreActs[i]);
            current = _maskDiscLayers[i].Backward(current);
        }
    }

    private void BackwardDataGenerator(Tensor<T> gradOutput)
    {
        if (_usingCustomLayers)
        {
            var current = gradOutput;
            for (int i = Layers.Count - 1; i >= 0; i--)
            {
                current = Layers[i].Backward(current);
            }
            return;
        }

        BackwardResidualGenerator(gradOutput, Layers, _dataGenBNLayers, _dataGenPreActs,
            _options.EmbeddingDimension);
    }

    private void BackwardMaskGenerator(Tensor<T> gradOutput)
    {
        BackwardResidualGenerator(gradOutput, _maskGenLayers.Cast<ILayer<T>>().ToList(), _maskGenBNLayers, _maskGenPreActs,
            _options.EmbeddingDimension);
    }

    /// <summary>
    /// Generic backward pass through a residual generator (shared by data and mask generators).
    /// </summary>
    private static void BackwardResidualGenerator(
        Tensor<T> gradOutput,
        List<ILayer<T>> layers,
        List<BatchNormalizationLayer<T>> bnLayers,
        List<Tensor<T>> preActivations,
        int inputDim)
    {
        var numOps = MathHelper.GetNumericOperations<T>();
        var current = gradOutput;

        // Backward through output layer
        current = layers[^1].Backward(current);

        // Split off residual gradient
        int lastHiddenDim = current.Length - inputDim;
        if (lastHiddenDim > 0)
        {
            var hiddenGrad = new Tensor<T>([lastHiddenDim]);
            for (int j = 0; j < lastHiddenDim && j < current.Length; j++)
                hiddenGrad[j] = current[j];
            current = hiddenGrad;
        }

        // Backward through hidden layers
        for (int i = layers.Count - 2; i >= 0; i--)
        {
            if (i < preActivations.Count)
                current = ApplyReLUDerivativeStatic(current, preActivations[i], numOps);

            current = bnLayers[i].Backward(current);
            current = layers[i].Backward(current);

            if (i > 0)
            {
                int prevDim = current.Length - inputDim;
                if (prevDim > 0)
                {
                    var hiddenGrad = new Tensor<T>([prevDim]);
                    for (int j = 0; j < prevDim && j < current.Length; j++)
                        hiddenGrad[j] = current[j];
                    current = hiddenGrad;
                }
            }
        }
    }

    private void UpdateDataGeneratorParameters(T learningRate)
    {
        foreach (var l in Layers) l.UpdateParameters(learningRate);
        foreach (var bn in _dataGenBNLayers) bn.UpdateParameters(learningRate);
    }

    private void UpdateDataDiscriminatorParameters(T learningRate)
    {
        foreach (var l in _dataDiscLayers) l.UpdateParameters(learningRate);
    }

    private void UpdateMaskGeneratorParameters(T learningRate)
    {
        foreach (var l in _maskGenLayers) l.UpdateParameters(learningRate);
        foreach (var bn in _maskGenBNLayers) bn.UpdateParameters(learningRate);
    }

    private void UpdateMaskDiscriminatorParameters(T learningRate)
    {
        foreach (var l in _maskDiscLayers) l.UpdateParameters(learningRate);
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
        // MisGAN uses its own specialized training via Fit/FitAsync.
        // Standard Train is not applicable for GAN-style training.
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
            ModelType = ModelType.NeuralNetwork,
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

    private Tensor<T> ApplyReLU(Tensor<T> input)
    {
        var result = new Tensor<T>(input.Shape);
        for (int i = 0; i < input.Length; i++)
            result[i] = NumOps.ToDouble(input[i]) > 0 ? input[i] : NumOps.Zero;
        return result;
    }

    private static Tensor<T> ApplyReLUDerivativeStatic(Tensor<T> gradOutput, Tensor<T> preActivation,
        INumericOperations<T> numOps)
    {
        int len = Math.Min(gradOutput.Length, preActivation.Length);
        var result = new Tensor<T>(gradOutput.Shape);
        for (int i = 0; i < len; i++)
            result[i] = numOps.ToDouble(preActivation[i]) > 0 ? gradOutput[i] : numOps.Zero;
        return result;
    }

    private Tensor<T> ApplyLeakyReLU(Tensor<T> input)
    {
        var result = new Tensor<T>(input.Shape);
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
        var result = new Tensor<T>(gradOutput.Shape);
        T slope = NumOps.FromDouble(0.2);
        for (int i = 0; i < len; i++)
        {
            result[i] = NumOps.ToDouble(preActivation[i]) > 0
                ? gradOutput[i]
                : NumOps.Multiply(slope, gradOutput[i]);
        }
        return result;
    }

    private Tensor<T> ApplySigmoid(Tensor<T> input)
    {
        var result = new Tensor<T>(input.Shape);
        for (int i = 0; i < input.Length; i++)
        {
            double val = NumOps.ToDouble(input[i]);
            double clampedVal = Math.Min(Math.Max(val, -20.0), 20.0);
            result[i] = NumOps.FromDouble(1.0 / (1.0 + Math.Exp(-clampedVal)));
        }
        return result;
    }

    private Tensor<T> ApplyOutputActivations(Tensor<T> output)
    {
        if (_transformer is null) return output;

        var result = new Tensor<T>(output.Shape);
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

        double maxVal = double.MinValue;
        for (int m = 0; m < count && (idx + m) < input.Length; m++)
        {
            double v = NumOps.ToDouble(input[idx + m]);
            if (v > maxVal) maxVal = v;
        }

        double sumExp = 0;
        for (int m = 0; m < count && (idx + m) < input.Length; m++)
            sumExp += Math.Exp(NumOps.ToDouble(input[idx + m]) - maxVal);

        for (int m = 0; m < count && idx < input.Length; m++)
        {
            double expVal = Math.Exp(NumOps.ToDouble(input[idx]) - maxVal);
            output[idx] = NumOps.FromDouble(expVal / Math.Max(sumExp, 1e-10));
            idx++;
        }
    }

    #endregion

    #region Helpers

    /// <summary>
    /// Sanitizes a gradient tensor by clamping NaN/Inf and clipping to max norm.
    /// </summary>
    private Tensor<T> SafeGradient(Tensor<T> grad, double maxNorm)
    {
        var result = new Tensor<T>(grad.Shape);
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
        int len = Math.Min(a.Length, b.Length);
        var result = new Tensor<T>(a.Shape);
        for (int i = 0; i < len; i++)
            result[i] = NumOps.Multiply(a[i], b[i]);
        return result;
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

    private static Tensor<T> ConcatTensors(Tensor<T> a, Tensor<T> b)
    {
        var result = new Tensor<T>([a.Length + b.Length]);
        for (int i = 0; i < a.Length; i++) result[i] = a[i];
        for (int i = 0; i < b.Length; i++) result[a.Length + i] = b[i];
        return result;
    }

    private static Tensor<T> CloneTensor(Tensor<T> source)
    {
        var clone = new Tensor<T>(source.Shape);
        for (int i = 0; i < source.Length; i++)
            clone[i] = source[i];
        return clone;
    }

    #endregion

    #region IJitCompilable Override

    /// <summary>
    /// MisGAN uses dual generator/discriminator pairs (data + mask) which cannot be represented as a single computation graph.
    /// </summary>
    public override bool SupportsJitCompilation => false;

    #endregion
}
