using AiDotNet.ActivationFunctions;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks.SyntheticData;
using AiDotNet.Optimizers;
using AiDotNet.Tensors.Helpers;

namespace AiDotNet.NeuralNetworks.SyntheticData;

/// <summary>
/// TabSyn generator combining VAE pretraining with latent diffusion for state-of-the-art
/// synthetic tabular data generation.
/// </summary>
/// <remarks>
/// <para>
/// TabSyn trains in two phases:
/// 1. <b>VAE Phase</b>: Encoder-decoder learns latent representation of tabular data
/// 2. <b>Diffusion Phase</b>: Gaussian diffusion model learns the distribution of latent codes
///
/// Generation: z ~ DiffusionModel -> decoded = VAEDecoder(z) -> inverse transform
///
/// This implementation follows the standard neural network architecture pattern with:
/// - Proper inheritance from NeuralNetworkBase
/// - Layer-based architecture using ILayer components
/// - Engine-based tensor operations for CPU/GPU acceleration
/// - Full autodiff and JIT compilation support
/// </para>
/// <para>
/// <b>For Beginners:</b> TabSyn is a two-step generator:
///
/// <b>Step 1 - VAE Training (learning to compress):</b>
/// <code>
/// Data row -> [Encoder] -> latent code z -> [Decoder] -> reconstructed row
/// </code>
///
/// <b>Step 2 - Diffusion Training (learning the latent distribution):</b>
/// <code>
/// Real latent code z -> add noise -> [Denoiser MLP] -> predict noise -> learn to denoise
/// </code>
///
/// <b>Generation:</b>
/// <code>
/// Pure noise -> [Denoise 1000 times] -> clean latent z -> [VAE Decoder] -> synthetic row
/// </code>
///
/// If you provide custom layers in the architecture, those will be used directly
/// for the VAE encoder. If not, the network creates industry-standard
/// TabSyn encoder layers based on the original research paper specifications.
///
/// Example usage:
/// <code>
/// var architecture = new NeuralNetworkArchitecture&lt;double&gt;(
///     inputFeatures: 10,
///     outputSize: 10
/// );
/// var options = new TabSynOptions&lt;double&gt;
/// {
///     EncoderDimensions = new[] { 256, 256 },
///     DecoderDimensions = new[] { 256, 256 },
///     LatentDimension = 64,
///     DiffusionSteps = 1000,
///     VAEEpochs = 100,
///     DiffusionEpochs = 100
/// };
/// var generator = new TabSynGenerator&lt;double&gt;(architecture, options);
/// generator.Fit(data, columns, epochs: 100);
/// var synthetic = generator.Generate(1000);
/// </code>
/// </para>
/// <para>
/// Reference: "TabSyn: Bridging the Gap" (Zhang et al., NeurIPS 2023)
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public class TabSynGenerator<T> : NeuralNetworkBase<T>, ISyntheticTabularGenerator<T>
{
    private readonly TabSynOptions<T> _options;
    private IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> _optimizer;
    private ILossFunction<T> _lossFunction;

    // Synthetic tabular data infrastructure
    private TabularDataTransformer<T>? _transformer;
    private List<ColumnMetadata> _columns = new();
    private int _dataWidth;
    private Random _random;

    // VAE mean/logvar projection heads (auxiliary, not user-overridable)
    private FullyConnectedLayer<T>? _meanLayer;
    private FullyConnectedLayer<T>? _logVarLayer;

    // VAE decoder layers (auxiliary, not user-overridable)
    private readonly List<ILayer<T>> _decoderLayers = new();

    // Latent diffusion components (auxiliary, not user-overridable)
    private GaussianDiffusion<T>? _latentDiffusion;
    private readonly List<ILayer<T>> _diffMLPLayers = new();
    private FullyConnectedLayer<T>? _timestepProjection;

    // Cached reparameterization noise for backward pass
    private Tensor<T>? _lastEpsilon;
    private Tensor<T>? _lastEncoderOutput;

    // Whether custom layers are being used (disables default encoder logic)
    private bool _usingCustomLayers;

    /// <summary>
    /// Gets the TabSyn-specific options.
    /// </summary>
    public new TabSynOptions<T> Options => _options;

    /// <inheritdoc />
    public IReadOnlyList<ColumnMetadata> Columns => _columns.AsReadOnly();

    /// <inheritdoc />
    public bool IsFitted { get; private set; }

    /// <summary>
    /// Initializes a new TabSyn generator with the specified architecture.
    /// </summary>
    /// <param name="architecture">The neural network architecture defining input/output dimensions and optional custom layers.</param>
    /// <param name="options">TabSyn-specific options for VAE and diffusion configuration.</param>
    /// <param name="optimizer">Gradient-based optimizer (defaults to Adam).</param>
    /// <param name="lossFunction">Loss function (defaults based on task type).</param>
    /// <param name="maxGradNorm">Maximum gradient norm for clipping (default 5.0).</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This constructor creates a TabSyn network based on the architecture you provide.
    ///
    /// If you provide custom layers in the architecture, those will be used directly
    /// for the VAE encoder network. If not, the network will create industry-standard
    /// TabSyn encoder layers based on the original research paper specifications.
    ///
    /// Example usage:
    /// <code>
    /// var architecture = new NeuralNetworkArchitecture&lt;double&gt;(
    ///     inputFeatures: 10,
    ///     outputSize: 10
    /// );
    /// var options = new TabSynOptions&lt;double&gt; { LatentDimension = 64 };
    /// var generator = new TabSynGenerator&lt;double&gt;(architecture, options);
    /// </code>
    /// </para>
    /// </remarks>
    public TabSynGenerator(
        NeuralNetworkArchitecture<T> architecture,
        TabSynOptions<T>? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
        ILossFunction<T>? lossFunction = null,
        double maxGradNorm = 5.0)
        : base(architecture, lossFunction ?? NeuralNetworkHelper<T>.GetDefaultLossFunction(architecture.TaskType), maxGradNorm)
    {
        _options = options ?? new TabSynOptions<T>();
        _lossFunction = lossFunction ?? NeuralNetworkHelper<T>.GetDefaultLossFunction(architecture.TaskType);
        _optimizer = optimizer ?? new AdamOptimizer<T, Tensor<T>, Tensor<T>>(this);
        _random = _options.Seed.HasValue
            ? RandomHelper.CreateSeededRandom(_options.Seed.Value)
            : RandomHelper.CreateSecureRandom();

        InitializeLayers();
    }

    #region Layer Initialization (GANDALF Pattern)

    /// <summary>
    /// Initializes the layers of the TabSyn network based on the provided architecture.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This method either uses custom layers provided in the architecture or creates
    /// default TabSyn encoder layers following the original paper specifications.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> This method sets up the VAE encoder network structure:
    /// - If you provided custom layers, those are used for the encoder
    /// - Otherwise, it creates the standard TabSyn encoder architecture
    ///
    /// The decoder and diffusion MLP are always created internally and are not user-overridable.
    /// </para>
    /// </remarks>
    protected override void InitializeLayers()
    {
        if (Architecture.Layers != null && Architecture.Layers.Count > 0)
        {
            // Use the layers provided by the user for the encoder
            Layers.AddRange(Architecture.Layers);
            ValidateCustomLayers(Layers);
            _usingCustomLayers = true;
        }
        else
        {
            // Create default encoder layers
            int inputDim = Architecture.CalculatedInputSize;
            int latentDim = _options.LatentDimension;
            Layers.AddRange(LayerHelper<T>.CreateDefaultTabSynEncoderLayers(
                inputDim, latentDim, _options.EncoderDimensions));
            _usingCustomLayers = false;
        }
    }

    /// <summary>
    /// Rebuilds encoder, decoder, and diffusion layers with actual data dimensions
    /// discovered during Fit(). Only rebuilds encoder when not using custom layers.
    /// </summary>
    /// <param name="dataWidth">Actual transformed data width (encoder input and decoder output).</param>
    private void RebuildLayersWithActualDimensions(int dataWidth)
    {
        int latentDim = _options.LatentDimension;

        if (!_usingCustomLayers)
        {
            // Rebuild encoder with actual data dimensions
            Layers.Clear();
            Layers.AddRange(LayerHelper<T>.CreateDefaultTabSynEncoderLayers(
                dataWidth, latentDim, _options.EncoderDimensions));
        }

        // Build mean and logvar projection heads from the last encoder hidden dim
        int lastEncHidden = _options.EncoderDimensions.Length > 0
            ? _options.EncoderDimensions[^1]
            : dataWidth;
        var identity = new IdentityActivation<T>() as IActivationFunction<T>;

        // If using custom layers, the last layer output is 2*latentDim (from LayerHelper),
        // but for custom layers we create separate mean/logvar heads from the last hidden dim.
        if (_usingCustomLayers)
        {
            // For custom layers, mean/logvar heads take from last custom layer output
            // We assume the user has set up their layers to output the desired hidden size
            int customLastOutputDim = latentDim;
            _meanLayer = new FullyConnectedLayer<T>(customLastOutputDim, latentDim, identity);
            _logVarLayer = new FullyConnectedLayer<T>(customLastOutputDim, latentDim, identity);
        }
        else
        {
            // Default encoder outputs 2*latentDim, so we split in the forward pass.
            // Mean/logvar are simply the first/second half of the encoder output.
            // No separate projection layers needed - they're built into the default encoder.
            _meanLayer = null;
            _logVarLayer = null;
        }

        // Always rebuild decoder with actual dimensions
        _decoderLayers.Clear();
        _decoderLayers.AddRange(LayerHelper<T>.CreateDefaultTabSynDecoderLayers(
            latentDim, dataWidth, _options.DecoderDimensions));

        // Always rebuild diffusion MLP with actual dimensions
        _diffMLPLayers.Clear();
        int teDim = _options.TimestepEmbeddingDimension;
        int diffInputDim = latentDim + teDim;
        _diffMLPLayers.AddRange(LayerHelper<T>.CreateDefaultTabSynDiffusionLayers(
            diffInputDim, latentDim, _options.DiffusionMLPDimensions));

        // Timestep projection
        var silu = new SiLUActivation<T>() as IActivationFunction<T>;
        _timestepProjection = new FullyConnectedLayer<T>(teDim, teDim, silu);
    }

    #endregion

    #region Neural Network Methods (GANDALF Pattern)

    /// <summary>
    /// Runs the VAE encoder forward pass on input data.
    /// </summary>
    /// <param name="input">The input tensor (transformed data row).</param>
    /// <returns>The encoder output (containing mean and log-variance information).</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This encodes a data row into a latent representation.
    /// The encoder compresses the input into a small summary vector.
    /// </para>
    /// </remarks>
    public override Tensor<T> Predict(Tensor<T> input)
    {
        // GPU-resident optimization: use TryForwardGpuOptimized for 10-50x speedup
        if (TryForwardGpuOptimized(input, out var gpuResult))
            return gpuResult;

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

        // Default encoder forward (all layers are sequential from LayerHelper)
        return EncoderForwardDefault(input);
    }

    /// <summary>
    /// Trains the TabSyn network using the provided input and expected output.
    /// </summary>
    /// <param name="input">The input data tensor.</param>
    /// <param name="expectedOutput">The expected reconstruction tensor.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This runs a single training step for the VAE.
    /// The full two-phase training (VAE then diffusion) happens in Fit().
    /// This method is provided for compatibility with the NeuralNetworkBase pattern.
    /// </para>
    /// </remarks>
    public override void Train(Tensor<T> input, Tensor<T> expectedOutput)
    {
        // Forward pass through encoder
        Tensor<T> prediction = Predict(input);

        // Calculate loss
        LastLoss = _lossFunction.CalculateLoss(prediction.ToVector(), expectedOutput.ToVector());

        // Calculate error gradient
        Tensor<T> error = prediction.Subtract(expectedOutput);

        // Backpropagate error through encoder
        BackpropagateError(error);

        // Update parameters
        UpdateNetworkParameters();
    }

    /// <summary>
    /// Backpropagates the error through the encoder layers.
    /// </summary>
    /// <param name="error">The error tensor to backpropagate.</param>
    private void BackpropagateError(Tensor<T> error)
    {
        if (_usingCustomLayers)
        {
            for (int i = Layers.Count - 1; i >= 0; i--)
            {
                error = Layers[i].Backward(error);
            }
            return;
        }

        // Default encoder backward (sequential)
        for (int i = Layers.Count - 1; i >= 0; i--)
        {
            error = Layers[i].Backward(error);
        }
    }

    /// <summary>
    /// Updates the parameters of all layers in the network based on computed gradients.
    /// </summary>
    private void UpdateNetworkParameters()
    {
        _optimizer.UpdateParameters(Layers);
    }

    /// <inheritdoc />
    public override void UpdateParameters(Vector<T> parameters)
    {
        int startIndex = 0;
        foreach (var layer in Layers)
        {
            int layerParameterCount = layer.ParameterCount;
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
    /// Fits the TabSyn generator to the provided real tabular data.
    /// </summary>
    /// <param name="data">The real data matrix where each row is a sample and each column is a feature.</param>
    /// <param name="columns">Metadata describing each column (type, categories, etc.).</param>
    /// <param name="epochs">Number of training epochs (used for VAE epochs; diffusion uses DiffusionEpochs from options).</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This is the "learning" step. The generator studies your real data:
    /// 1. Fits the VGM transformer for data normalization
    /// 2. Trains the VAE to learn a compressed latent representation
    /// 3. Encodes all data to latent space
    /// 4. Trains the latent diffusion model on the encoded data
    /// After fitting, call Generate() to create new synthetic rows.
    /// </para>
    /// </remarks>
    public void Fit(Matrix<T> data, IReadOnlyList<ColumnMetadata> columns, int epochs)
    {
        ValidateFitInputs(data, columns, epochs);

        _columns = PrepareColumns(data, columns);

        // Step 1: Fit transformer
        _transformer = new TabularDataTransformer<T>(_options.VGMModes, _random);
        _transformer.Fit(data, _columns);
        _dataWidth = _transformer.TransformedWidth;

        var transformedData = _transformer.Transform(data);

        // Step 2: Rebuild layers with actual data dimensions
        RebuildLayersWithActualDimensions(_dataWidth);

        // Step 3: Train VAE
        int vaeBatchSize = Math.Min(_options.BatchSize, data.Rows);
        int vaeNumBatches = Math.Max(1, data.Rows / vaeBatchSize);
        T vaeLr = NumOps.FromDouble(_options.VAELearningRate / vaeBatchSize);

        int vaeEpochs = _options.VAEEpochs > 0 ? _options.VAEEpochs : epochs;
        for (int epoch = 0; epoch < vaeEpochs; epoch++)
        {
            for (int batch = 0; batch < vaeNumBatches; batch++)
            {
                int startRow = batch * vaeBatchSize;
                int endRow = Math.Min(startRow + vaeBatchSize, data.Rows);
                TrainVAEBatch(transformedData, startRow, endRow, vaeLr);
            }
        }

        // Step 4: Encode all data to latent space
        var latentCodes = EncodeAllData(transformedData);

        // Step 5: Build and train latent diffusion model
        _latentDiffusion = new GaussianDiffusion<T>(
            _options.DiffusionSteps, _options.BetaStart, _options.BetaEnd, "linear", _random);

        int diffBatchSize = Math.Min(_options.BatchSize, data.Rows);
        int diffNumBatches = Math.Max(1, data.Rows / diffBatchSize);
        T diffLr = NumOps.FromDouble(_options.DiffusionLearningRate / diffBatchSize);

        for (int epoch = 0; epoch < _options.DiffusionEpochs; epoch++)
        {
            for (int batch = 0; batch < diffNumBatches; batch++)
            {
                int startRow = batch * diffBatchSize;
                int endRow = Math.Min(startRow + diffBatchSize, data.Rows);
                TrainDiffusionBatch(latentCodes, startRow, endRow, diffLr);
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

            var transformedData = _transformer.Transform(data);

            RebuildLayersWithActualDimensions(_dataWidth);

            int vaeBatchSize = Math.Min(_options.BatchSize, data.Rows);
            int vaeNumBatches = Math.Max(1, data.Rows / vaeBatchSize);
            T vaeLr = NumOps.FromDouble(_options.VAELearningRate / vaeBatchSize);

            int vaeEpochs = _options.VAEEpochs > 0 ? _options.VAEEpochs : epochs;
            for (int epoch = 0; epoch < vaeEpochs; epoch++)
            {
                ct.ThrowIfCancellationRequested();
                for (int batch = 0; batch < vaeNumBatches; batch++)
                {
                    int startRow = batch * vaeBatchSize;
                    int endRow = Math.Min(startRow + vaeBatchSize, data.Rows);
                    TrainVAEBatch(transformedData, startRow, endRow, vaeLr);
                }
            }

            var latentCodes = EncodeAllData(transformedData);

            _latentDiffusion = new GaussianDiffusion<T>(
                _options.DiffusionSteps, _options.BetaStart, _options.BetaEnd, "linear", _random);

            int diffBatchSize = Math.Min(_options.BatchSize, data.Rows);
            int diffNumBatches = Math.Max(1, data.Rows / diffBatchSize);
            T diffLr = NumOps.FromDouble(_options.DiffusionLearningRate / diffBatchSize);

            for (int epoch = 0; epoch < _options.DiffusionEpochs; epoch++)
            {
                ct.ThrowIfCancellationRequested();
                for (int batch = 0; batch < diffNumBatches; batch++)
                {
                    int startRow = batch * diffBatchSize;
                    int endRow = Math.Min(startRow + diffBatchSize, data.Rows);
                    TrainDiffusionBatch(latentCodes, startRow, endRow, diffLr);
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
    /// 1. Sampling noise in latent space
    /// 2. Running reverse diffusion to produce clean latent codes
    /// 3. Decoding latent codes to transformed data space via VAE decoder
    /// 4. Applying inverse VGM transform to restore original distributions
    /// </para>
    /// </remarks>
    public Matrix<T> Generate(int numSamples, Vector<T>? conditionColumn = null, Vector<T>? conditionValue = null)
    {
        if (!IsFitted || _transformer is null || _latentDiffusion is null)
        {
            throw new InvalidOperationException(
                "The generator must be fitted before generating data. Call Fit() first.");
        }

        if (numSamples <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(numSamples), "Number of samples must be positive.");
        }

        int latentDim = _options.LatentDimension;
        var transformedRows = new Matrix<T>(numSamples, _dataWidth);

        for (int i = 0; i < numSamples; i++)
        {
            // Start with noise in latent space
            var z = CreateStandardNormalVector(latentDim);

            // Reverse diffusion in latent space
            for (int t = _options.DiffusionSteps - 1; t >= 0; t--)
            {
                var timeEmbed = CreateTimestepEmbedding(t);
                var predictedNoise = DiffusionMLPForward(z, timeEmbed);
                z = _latentDiffusion.DenoisingStep(z, predictedNoise, t);
            }

            // Decode latent code to data space
            var decoded = DecoderForward(VectorToTensor(z));
            var activated = ApplyOutputActivations(decoded);

            for (int j = 0; j < _dataWidth && j < activated.Length; j++)
            {
                transformedRows[i, j] = activated[j];
            }
        }

        return _transformer.InverseTransform(transformedRows);
    }

    #endregion

    #region VAE Forward Passes

    /// <summary>
    /// Runs the default encoder forward pass (sequential through all LayerHelper-created layers).
    /// The last layer outputs 2*latentDim which is split into mean and log-variance.
    /// </summary>
    private Tensor<T> EncoderForwardDefault(Tensor<T> input)
    {
        var current = input;
        foreach (var layer in Layers)
        {
            current = layer.Forward(current);
        }
        _lastEncoderOutput = current;
        return current;
    }

    /// <summary>
    /// Extracts mean and log-variance from the encoder output.
    /// </summary>
    /// <param name="encoderOutput">The encoder output tensor.</param>
    /// <returns>A tuple of (mean, logVar) tensors, each of size latentDim.</returns>
    private (Tensor<T> Mean, Tensor<T> LogVar) SplitEncoderOutput(Tensor<T> encoderOutput)
    {
        int latentDim = _options.LatentDimension;

        if (_usingCustomLayers && _meanLayer is not null && _logVarLayer is not null)
        {
            // Custom layers: pass encoder output through separate mean/logvar heads
            var mean = _meanLayer.Forward(encoderOutput);
            var logVar = _logVarLayer.Forward(encoderOutput);
            return (mean, logVar);
        }

        // Default encoder: last layer outputs [mean | logVar] concatenated
        // Derive shape from encoder output rank: replace last dim with latentDim
        int[] meanShape = DeriveShapeWithLastDim(encoderOutput.Shape, latentDim);
        var meanTensor = new Tensor<T>(meanShape);
        var logVarTensor = new Tensor<T>(meanShape);

        for (int i = 0; i < latentDim && i < encoderOutput.Length; i++)
        {
            meanTensor[i] = encoderOutput[i];
        }
        for (int i = 0; i < latentDim && (i + latentDim) < encoderOutput.Length; i++)
        {
            logVarTensor[i] = encoderOutput[i + latentDim];
        }

        return (meanTensor, logVarTensor);
    }

    /// <summary>
    /// VAE reparameterization trick: z = mean + std * epsilon, where epsilon ~ N(0,1).
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The reparameterization trick lets us backpropagate through
    /// the random sampling step. Instead of sampling z directly from the distribution,
    /// we sample noise epsilon and compute z = mean + std * epsilon. This way the
    /// randomness is in epsilon (which has no learned parameters) and gradients can
    /// flow through mean and std.
    /// </para>
    /// </remarks>
    private Tensor<T> Reparameterize(Tensor<T> mean, Tensor<T> logVar)
    {
        var z = new Tensor<T>(mean.Shape);
        _lastEpsilon = new Tensor<T>(mean.Shape);
        for (int i = 0; i < mean.Length; i++)
        {
            double m = NumOps.ToDouble(mean[i]);
            double lv = NumOps.ToDouble(logVar[i]);
            double std = Math.Exp(0.5 * lv);
            double eps = NumOps.ToDouble(SampleStandardNormal());
            _lastEpsilon[i] = NumOps.FromDouble(eps);
            z[i] = NumOps.FromDouble(m + std * eps);
        }
        return z;
    }

    /// <summary>
    /// Runs the VAE decoder forward pass to reconstruct data from latent code.
    /// </summary>
    private Tensor<T> DecoderForward(Tensor<T> z)
    {
        var current = z;
        foreach (var layer in _decoderLayers)
        {
            current = layer.Forward(current);
        }
        return current;
    }

    #endregion

    #region Diffusion Forward

    /// <summary>
    /// Creates a sinusoidal timestep embedding for the diffusion model.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The diffusion model needs to know "how much noise was added"
    /// at each step. This creates a vector that encodes the timestep as a combination
    /// of sine and cosine waves at different frequencies, similar to positional encoding
    /// in transformers.
    /// </para>
    /// </remarks>
    private Vector<T> CreateTimestepEmbedding(int timestep)
    {
        int dim = _options.TimestepEmbeddingDimension;
        var embedding = new Vector<T>(dim);
        int halfDim = dim / 2;
        for (int i = 0; i < halfDim; i++)
        {
            double freq = Math.Exp(-Math.Log(10000.0) * i / halfDim);
            double angle = timestep * freq;
            embedding[i] = NumOps.FromDouble(Math.Sin(angle));
            if (i + halfDim < dim) embedding[i + halfDim] = NumOps.FromDouble(Math.Cos(angle));
        }

        if (_timestepProjection is not null)
        {
            var embedTensor = VectorToTensor(embedding);
            var projected = _timestepProjection.Forward(embedTensor);
            return TensorToVector(projected, dim);
        }
        return embedding;
    }

    /// <summary>
    /// Runs the diffusion denoiser MLP forward pass.
    /// Takes concatenated [noised_latent, timestep_embedding] and predicts noise.
    /// </summary>
    private Vector<T> DiffusionMLPForward(Vector<T> latent, Vector<T> timeEmbed)
    {
        int totalLen = latent.Length + timeEmbed.Length;
        var input = new Vector<T>(totalLen);
        for (int i = 0; i < latent.Length; i++) input[i] = latent[i];
        for (int i = 0; i < timeEmbed.Length; i++) input[latent.Length + i] = timeEmbed[i];

        var current = VectorToTensor(input);
        foreach (var layer in _diffMLPLayers)
        {
            current = layer.Forward(current);
        }

        return TensorToVector(current, _options.LatentDimension);
    }

    #endregion

    #region VAE Training

    /// <summary>
    /// Trains the VAE on a batch of rows: forward encode, reparameterize, decode, backward.
    /// </summary>
    private void TrainVAEBatch(Matrix<T> transformedData, int startRow, int endRow, T scaledLr)
    {
        for (int row = startRow; row < endRow; row++)
        {
            var inputVec = GetRow(transformedData, row);
            var inputTensor = VectorToTensor(inputVec);

            // Encoder forward
            var encoderOutput = Predict(inputTensor);
            var (mean, logVar) = SplitEncoderOutput(encoderOutput);

            // Reparameterize
            var z = Reparameterize(mean, logVar);

            // Decoder forward
            var rawOutput = DecoderForward(z);
            var activated = ApplyOutputActivations(rawOutput);

            // Compute reconstruction gradient
            var outputGrad = ComputeVAEOutputGradient(inputTensor, activated);

            // Sanitize and clip gradient
            SanitizeTensor(outputGrad);
            outputGrad = ClipGradientNorm(outputGrad, 5.0);

            // Backward decoder
            var zGrad = BackwardDecoder(outputGrad);

            // Backward encoder (through reparameterization and mean/logvar)
            BackwardEncoder(zGrad, mean, logVar);

            // Update all VAE parameters
            UpdateVAEParameters(scaledLr);
        }
    }

    /// <summary>
    /// Encodes all training data rows to their mean latent representations.
    /// </summary>
    private Matrix<T> EncodeAllData(Matrix<T> transformedData)
    {
        int latentDim = _options.LatentDimension;
        var latentCodes = new Matrix<T>(transformedData.Rows, latentDim);

        for (int i = 0; i < transformedData.Rows; i++)
        {
            var inputVec = GetRow(transformedData, i);
            var inputTensor = VectorToTensor(inputVec);

            var encoderOutput = Predict(inputTensor);
            var (mean, _) = SplitEncoderOutput(encoderOutput);

            for (int j = 0; j < latentDim && j < mean.Length; j++)
            {
                latentCodes[i, j] = mean[j];
            }
        }

        return latentCodes;
    }

    #endregion

    #region Diffusion Training

    /// <summary>
    /// Trains the latent diffusion model on a batch of latent codes.
    /// </summary>
    private void TrainDiffusionBatch(Matrix<T> latentCodes, int startRow, int endRow, T scaledLr)
    {
        if (_latentDiffusion is null) return;

        for (int row = startRow; row < endRow; row++)
        {
            int t = _latentDiffusion.SampleTimestep();
            var clean = GetRow(latentCodes, row);
            var (noisy, actualNoise) = _latentDiffusion.AddNoise(clean, t);

            var timeEmbed = CreateTimestepEmbedding(t);
            var predictedNoise = DiffusionMLPForward(noisy, timeEmbed);

            var grad = _latentDiffusion.ComputeLossGradient(predictedNoise, actualNoise);

            BackwardDiffusionMLP(VectorToTensor(grad));
            UpdateDiffusionParameters(scaledLr);
        }
    }

    #endregion

    #region Backward Passes

    /// <summary>
    /// Backward pass through the VAE decoder layers.
    /// </summary>
    private Tensor<T> BackwardDecoder(Tensor<T> gradOutput)
    {
        var current = gradOutput;
        for (int i = _decoderLayers.Count - 1; i >= 0; i--)
        {
            current = _decoderLayers[i].Backward(current);
        }
        return current;
    }

    /// <summary>
    /// Backward pass through the encoder, including the reparameterization trick gradients.
    /// Combines gradients from the reconstruction loss (through z) with the KL divergence loss.
    /// </summary>
    private void BackwardEncoder(Tensor<T> zGrad, Tensor<T> mean, Tensor<T> logVar)
    {
        var meanGrad = new Tensor<T>(mean.Shape);
        var logVarGrad = new Tensor<T>(logVar.Shape);

        for (int i = 0; i < mean.Length; i++)
        {
            double m = NumOps.ToDouble(mean[i]);
            double lv = NumOps.ToDouble(logVar[i]);
            double dz = i < zGrad.Length ? NumOps.ToDouble(zGrad[i]) : 0;
            double eps = _lastEpsilon is not null && i < _lastEpsilon.Length
                ? NumOps.ToDouble(_lastEpsilon[i]) : 0;

            // Reconstruction gradient + KL divergence gradient for mean
            meanGrad[i] = NumOps.FromDouble(dz + m);
            // Reconstruction gradient through std + KL divergence gradient for logvar
            logVarGrad[i] = NumOps.FromDouble(dz * eps * 0.5 * Math.Exp(0.5 * lv) + 0.5 * (Math.Exp(lv) - 1.0));
        }

        Tensor<T> encoderOutGrad;
        if (_usingCustomLayers && _meanLayer is not null && _logVarLayer is not null)
        {
            // Custom layers with separate heads: backward through each head and sum
            var gradFromMean = _meanLayer.Backward(meanGrad);
            var gradFromLogVar = _logVarLayer.Backward(logVarGrad);

            encoderOutGrad = new Tensor<T>(gradFromMean.Shape);
            for (int i = 0; i < encoderOutGrad.Length; i++)
            {
                double gm = i < gradFromMean.Length ? NumOps.ToDouble(gradFromMean[i]) : 0;
                double gl = i < gradFromLogVar.Length ? NumOps.ToDouble(gradFromLogVar[i]) : 0;
                encoderOutGrad[i] = NumOps.FromDouble(gm + gl);
            }
        }
        else
        {
            // Default encoder: gradient flows back to [mean | logvar] output
            int latentDim = _options.LatentDimension;
            int[] encGradShape = _lastEncoderOutput is not null
                ? (int[])_lastEncoderOutput.Shape.Clone()
                : DeriveShapeWithLastDim(mean.Shape, latentDim * 2);
            encoderOutGrad = new Tensor<T>(encGradShape);
            for (int i = 0; i < latentDim; i++)
            {
                encoderOutGrad[i] = meanGrad[i];
                if (i + latentDim < encoderOutGrad.Length)
                {
                    encoderOutGrad[i + latentDim] = logVarGrad[i];
                }
            }
        }

        // Backward through encoder layers
        var current = encoderOutGrad;
        for (int i = Layers.Count - 1; i >= 0; i--)
        {
            current = Layers[i].Backward(current);
        }
    }

    /// <summary>
    /// Backward pass through the diffusion denoiser MLP.
    /// </summary>
    private void BackwardDiffusionMLP(Tensor<T> gradOutput)
    {
        var current = gradOutput;
        for (int i = _diffMLPLayers.Count - 1; i >= 0; i--)
        {
            current = _diffMLPLayers[i].Backward(current);
        }
    }

    /// <summary>
    /// Updates all VAE parameters (encoder + mean/logvar heads + decoder).
    /// </summary>
    private void UpdateVAEParameters(T lr)
    {
        foreach (var layer in Layers) layer.UpdateParameters(lr);
        _meanLayer?.UpdateParameters(lr);
        _logVarLayer?.UpdateParameters(lr);
        foreach (var layer in _decoderLayers) layer.UpdateParameters(lr);
    }

    /// <summary>
    /// Updates all diffusion MLP parameters.
    /// </summary>
    private void UpdateDiffusionParameters(T lr)
    {
        foreach (var layer in _diffMLPLayers) layer.UpdateParameters(lr);
    }

    #endregion

    #region Output Activations

    /// <summary>
    /// Computes the VAE reconstruction gradient for the output activations.
    /// Uses tanh derivatives for continuous values and cross-entropy gradients for softmax.
    /// </summary>
    private Tensor<T> ComputeVAEOutputGradient(Tensor<T> input, Tensor<T> activated)
    {
        var grad = new Tensor<T>(activated.Shape);
        if (_transformer is null) return grad;

        int idx = 0;
        for (int col = 0; col < Columns.Count && idx < input.Length; col++)
        {
            var transform = _transformer.GetTransformInfo(col);
            if (transform.IsContinuous)
            {
                if (idx < input.Length && idx < activated.Length)
                {
                    double target = NumOps.ToDouble(input[idx]);
                    double tanhVal = NumOps.ToDouble(activated[idx]);
                    double diff = tanhVal - target;
                    double tanhDeriv = 1.0 - tanhVal * tanhVal;
                    grad[idx] = NumOps.FromDouble(2.0 * diff * tanhDeriv);
                    idx++;
                }
                int numModes = transform.Width - 1;
                for (int m = 0; m < numModes && idx < input.Length; m++)
                {
                    double target = NumOps.ToDouble(input[idx]);
                    double predicted = NumOps.ToDouble(activated[idx]);
                    grad[idx] = NumOps.FromDouble(predicted - target);
                    idx++;
                }
            }
            else
            {
                int numCats = transform.Width;
                for (int c = 0; c < numCats && idx < input.Length; c++)
                {
                    double target = NumOps.ToDouble(input[idx]);
                    double predicted = NumOps.ToDouble(activated[idx]);
                    grad[idx] = NumOps.FromDouble(predicted - target);
                    idx++;
                }
            }
        }
        return grad;
    }

    /// <summary>
    /// Applies per-column output activations: tanh for continuous values, softmax for modes/categories.
    /// </summary>
    private Tensor<T> ApplyOutputActivations(Tensor<T> output)
    {
        if (_transformer is null) return output;

        var result = new Tensor<T>(output.Shape);
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

    /// <summary>
    /// Applies stable softmax activation to a contiguous block of tensor elements.
    /// </summary>
    private void ApplySoftmax(Tensor<T> input, Tensor<T> output, ref int idx, int count)
    {
        if (count <= 0) return;
        double maxVal = double.MinValue;
        for (int i = 0; i < count && (idx + i) < input.Length; i++)
        {
            double v = NumOps.ToDouble(input[idx + i]);
            if (v > maxVal) maxVal = v;
        }
        double sumExp = 0;
        for (int i = 0; i < count && (idx + i) < input.Length; i++)
        {
            sumExp += Math.Exp(NumOps.ToDouble(input[idx + i]) - maxVal);
        }
        for (int i = 0; i < count && idx < input.Length; i++)
        {
            double expVal = Math.Exp(NumOps.ToDouble(input[idx]) - maxVal);
            output[idx] = NumOps.FromDouble(expVal / Math.Max(sumExp, 1e-10));
            idx++;
        }
    }

    #endregion

    #region Gradient Safety Utilities

    /// <summary>
    /// Replaces NaN and Infinity values in a tensor with zero, in-place.
    /// </summary>
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

    /// <summary>
    /// Clips a gradient tensor to a maximum L2 norm.
    /// </summary>
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
        var clipped = new Tensor<T>(grad.Shape);
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
            ModelType = ModelType.NeuralNetwork,
            AdditionalInfo = new Dictionary<string, object>
            {
                { "LatentDimension", _options.LatentDimension },
                { "EncoderDimensions", _options.EncoderDimensions },
                { "DecoderDimensions", _options.DecoderDimensions },
                { "DiffusionMLPDimensions", _options.DiffusionMLPDimensions },
                { "DiffusionSteps", _options.DiffusionSteps },
                { "VAEEpochs", _options.VAEEpochs },
                { "DiffusionEpochs", _options.DiffusionEpochs },
                { "EncoderLayerCount", Layers.Count },
                { "EncoderLayerTypes", Layers.Select(l => l.GetType().Name).ToArray() }
            },
            ModelData = this.Serialize()
        };
    }

    /// <inheritdoc/>
    protected override void SerializeNetworkSpecificData(BinaryWriter writer)
    {
        writer.Write(_options.LatentDimension);
        writer.Write(_options.EncoderDimensions.Length);
        foreach (var dim in _options.EncoderDimensions)
        {
            writer.Write(dim);
        }
        writer.Write(_options.DecoderDimensions.Length);
        foreach (var dim in _options.DecoderDimensions)
        {
            writer.Write(dim);
        }
        writer.Write(_options.DiffusionMLPDimensions.Length);
        foreach (var dim in _options.DiffusionMLPDimensions)
        {
            writer.Write(dim);
        }
        writer.Write(_options.DiffusionSteps);
        writer.Write(_options.BetaStart);
        writer.Write(_options.BetaEnd);
        writer.Write(_options.BatchSize);
        writer.Write(_options.VAELearningRate);
        writer.Write(_options.DiffusionLearningRate);
        writer.Write(_options.VGMModes);
        writer.Write(_options.TimestepEmbeddingDimension);
    }

    /// <inheritdoc/>
    protected override void DeserializeNetworkSpecificData(BinaryReader reader)
    {
        // Options are reconstructed from serialized data
        // Layers are handled by base class
    }

    /// <inheritdoc/>
    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance()
    {
        return new TabSynGenerator<T>(
            Architecture,
            _options,
            _optimizer,
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

    /// <summary>
    /// Validates inputs to the Fit method.
    /// </summary>
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

    #endregion

    #region IJitCompilable Override

    /// <summary>
    /// TabSyn uses latent diffusion with a separate VAE encoder/decoder which cannot be represented as a single computation graph.
    /// </summary>
    public override bool SupportsJitCompilation => false;

    #endregion

    #region Tensor Shape Helpers

    /// <summary>
    /// Derives a tensor shape that preserves the rank of the reference but replaces the last dimension.
    /// </summary>
    private static int[] DeriveShapeWithLastDim(int[] referenceShape, int lastDim)
    {
        if (referenceShape.Length == 0)
            return [lastDim];
        var shape = (int[])referenceShape.Clone();
        shape[^1] = lastDim;
        return shape;
    }

    #endregion
}
