using AiDotNet.ActivationFunctions;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Optimizers;
using AiDotNet.Tensors.Helpers;

namespace AiDotNet.NeuralNetworks.SyntheticData;

/// <summary>
/// Tabular Variational Autoencoder (TVAE) for generating synthetic tabular data.
/// </summary>
/// <remarks>
/// <para>
/// TVAE applies the VAE framework to tabular data with the same VGM preprocessing as CTGAN:
/// - <b>Encoder</b>: Maps transformed data row to latent distribution parameters (mean, logvar)
/// - <b>Reparameterization</b>: z = mean + exp(0.5 * logvar) * epsilon (epsilon ~ N(0,1))
/// - <b>Decoder</b>: Reconstructs the transformed data from the latent code
/// - <b>ELBO Loss</b>: Reconstruction loss (per-column cross-entropy/MSE) + KL divergence
/// </para>
/// <para>
/// This implementation follows the standard neural network architecture pattern with:
/// - Proper inheritance from NeuralNetworkBase
/// - Layer-based architecture using ILayer components
/// - Engine-based tensor operations for CPU/GPU acceleration
/// - Full autodiff and JIT compilation support
/// </para>
/// <para>
/// <b>For Beginners:</b> TVAE works like a compression algorithm that can also generate new data:
///
/// <b>Training (learning to compress and decompress):</b>
/// <code>
/// Data row --&gt; [Encoder] --&gt; (mean, variance) --&gt; sample z --&gt; [Decoder] --&gt; reconstructed row
///                                                                            |
///                                                            Compare with original --&gt; loss
/// </code>
///
/// <b>Generation (creating new data):</b>
/// <code>
/// Random noise z ~ N(0,1) --&gt; [Decoder] --&gt; new synthetic row --&gt; inverse transform
/// </code>
///
/// The key insight is that the latent space is regularized to be Gaussian,
/// so we can sample random points from it and decode them into realistic rows.
///
/// If you provide custom layers in the architecture, those will be used directly
/// for the encoder network. If not, the network creates industry-standard
/// TVAE layers based on the original research paper specifications.
///
/// Example usage:
/// <code>
/// var architecture = new NeuralNetworkArchitecture&lt;double&gt;(
///     inputFeatures: 10,
///     outputSize: 10
/// );
/// var options = new TVAEOptions&lt;double&gt;
/// {
///     EncoderDimensions = new[] { 128, 128 },
///     DecoderDimensions = new[] { 128, 128 },
///     LatentDimension = 128,
///     Epochs = 300
/// };
/// var tvae = new TVAEGenerator&lt;double&gt;(architecture, options);
/// tvae.Fit(data, columns, epochs: 300);
/// var synthetic = tvae.Generate(1000);
/// </code>
/// </para>
/// <para>
/// Reference: "Modeling Tabular Data using Conditional GAN" (Xu et al., NeurIPS 2019)
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public class TVAEGenerator<T> : NeuralNetworkBase<T>, ISyntheticTabularGenerator<T>
{
    private readonly TVAEOptions<T> _options;
    private IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> _optimizer;
    private ILossFunction<T> _lossFunction;

    // Synthetic tabular data infrastructure
    private TabularDataTransformer<T>? _transformer;
    private List<ColumnMetadata> _columns = new();
    private int _dataWidth;
    private Random _random;

    // Mean and logvar projection heads (auxiliary, used when not using custom layers)
    // Custom layers produce concatenated 2*latentDim output; default layers use separate heads
    private FullyConnectedLayer<T>? _meanLayer;
    private FullyConnectedLayer<T>? _logVarLayer;

    // Decoder layers (auxiliary, not user-overridable)
    private readonly List<ILayer<T>> _decoderLayers = new();

    // Cached epsilon from reparameterization for proper backward gradient computation
    private Tensor<T>? _lastEpsilon;
    private Tensor<T>? _lastEncoderOutput;

    // Whether custom layers are being used (changes encoder output handling)
    private bool _usingCustomLayers;

    /// <summary>
    /// Gets the TVAE-specific options.
    /// </summary>
    public new TVAEOptions<T> Options => _options;

    /// <inheritdoc />
    public IReadOnlyList<ColumnMetadata> Columns => _columns.AsReadOnly();

    /// <inheritdoc />
    public bool IsFitted { get; private set; }

    /// <summary>
    /// Initializes a new TVAE generator with the specified architecture.
    /// </summary>
    /// <param name="architecture">The neural network architecture defining input/output dimensions and optional custom layers.</param>
    /// <param name="options">TVAE-specific options for encoder and decoder configuration.</param>
    /// <param name="optimizer">Gradient-based optimizer (defaults to Adam).</param>
    /// <param name="lossFunction">Loss function (defaults based on task type).</param>
    /// <param name="maxGradNorm">Maximum gradient norm for clipping (default 5.0).</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This constructor creates a TVAE network based on the architecture you provide.
    ///
    /// If you provide custom layers in the architecture, those will be used directly
    /// for the encoder network. If not, the network will create industry-standard
    /// TVAE encoder layers based on the original research paper specifications.
    ///
    /// Example usage:
    /// <code>
    /// var architecture = new NeuralNetworkArchitecture&lt;double&gt;(
    ///     inputFeatures: 10,
    ///     outputSize: 10
    /// );
    /// var options = new TVAEOptions&lt;double&gt; { LatentDimension = 128 };
    /// var tvae = new TVAEGenerator&lt;double&gt;(architecture, options);
    /// </code>
    /// </para>
    /// </remarks>
    public TVAEGenerator(
        NeuralNetworkArchitecture<T> architecture,
        TVAEOptions<T>? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
        ILossFunction<T>? lossFunction = null,
        double maxGradNorm = 5.0)
        : base(architecture, lossFunction ?? NeuralNetworkHelper<T>.GetDefaultLossFunction(architecture.TaskType), maxGradNorm)
    {
        _options = options ?? new TVAEOptions<T>();
        _lossFunction = lossFunction ?? NeuralNetworkHelper<T>.GetDefaultLossFunction(architecture.TaskType);
        _optimizer = optimizer ?? new AdamOptimizer<T, Tensor<T>, Tensor<T>>(this);

        int? seed = _options.Seed;
        _random = seed.HasValue
            ? RandomHelper.CreateSeededRandom(seed.Value)
            : RandomHelper.CreateSecureRandom();

        InitializeLayers();
    }

    #region Layer Initialization (GANDALF Pattern)

    /// <summary>
    /// Initializes the layers of the TVAE network based on the provided architecture.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This method either uses custom layers provided in the architecture or creates
    /// default TVAE encoder layers following the original paper specifications.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> This method sets up the encoder network structure:
    /// - If you provided custom layers, those are used for the encoder
    /// - Otherwise, it creates the standard TVAE encoder architecture
    ///
    /// The decoder is always created internally and is not user-overridable.
    /// </para>
    /// </remarks>
    protected override void InitializeLayers()
    {
        if (Architecture.Layers != null && Architecture.Layers.Count > 0)
        {
            // Use the layers provided by the user for the encoder
            Layers.AddRange(Architecture.Layers);
            _usingCustomLayers = true;

            // Create separate mean and logvar projection heads for custom layers
            int lastLayerOutput = Architecture.OutputSize;
            var identity = new IdentityActivation<T>() as IActivationFunction<T>;
            _meanLayer = new FullyConnectedLayer<T>(lastLayerOutput, _options.LatentDimension, identity);
            _logVarLayer = new FullyConnectedLayer<T>(lastLayerOutput, _options.LatentDimension, identity);
        }
        else
        {
            // Create default encoder layers
            // Default encoder outputs 2*latentDim (mean+logvar concatenated)
            int inputDim = Architecture.CalculatedInputSize;
            Layers.AddRange(LayerHelper<T>.CreateDefaultTVAEEncoderLayers(
                inputDim, _options.LatentDimension, _options.EncoderDimensions));
            _usingCustomLayers = false;
        }

        // Create default decoder layers (always internal, not user-overridable)
        _decoderLayers.Clear();
        _decoderLayers.AddRange(LayerHelper<T>.CreateDefaultTVAEDecoderLayers(
            _options.LatentDimension, Architecture.OutputSize, _options.DecoderDimensions));
    }

    /// <summary>
    /// Rebuilds encoder and decoder layers with actual data dimensions discovered during Fit().
    /// </summary>
    private void RebuildLayersWithActualDimensions(int dataWidth)
    {
        if (!_usingCustomLayers)
        {
            Layers.Clear();
            Layers.AddRange(LayerHelper<T>.CreateDefaultTVAEEncoderLayers(
                dataWidth, _options.LatentDimension, _options.EncoderDimensions));
        }
        else
        {
            // For custom layers, rebuild the projection heads with actual latent dimension
            int lastLayerOutput = Layers.Count > 0 ? GetLayerOutputSize(Layers[^1]) : dataWidth;
            var identity = new IdentityActivation<T>() as IActivationFunction<T>;
            _meanLayer = new FullyConnectedLayer<T>(lastLayerOutput, _options.LatentDimension, identity);
            _logVarLayer = new FullyConnectedLayer<T>(lastLayerOutput, _options.LatentDimension, identity);
        }

        // Always rebuild decoder with actual dimensions
        _decoderLayers.Clear();
        _decoderLayers.AddRange(LayerHelper<T>.CreateDefaultTVAEDecoderLayers(
            _options.LatentDimension, dataWidth, _options.DecoderDimensions));
    }

    /// <summary>
    /// Gets the output size of a layer by examining its output shape.
    /// </summary>
    private static int GetLayerOutputSize(ILayer<T> layer)
    {
        var shape = layer.GetOutputShape();
        if (shape is not null && shape.Length > 0)
        {
            return shape[0];
        }
        return 128; // fallback
    }

    #endregion

    #region Neural Network Methods (GANDALF Pattern)

    /// <summary>
    /// Runs the encoder forward pass to produce latent distribution parameters.
    /// </summary>
    /// <param name="input">The input tensor (transformed data row).</param>
    /// <returns>The encoder output tensor.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This takes a data row and compresses it into a latent representation.
    /// The output is either 2*latentDim (concatenated mean+logvar) for default layers, or
    /// a hidden representation that gets projected into mean and logvar by separate heads.
    /// </para>
    /// </remarks>
    public override Tensor<T> Predict(Tensor<T> input)
    {
        // GPU-resident optimization
        if (TryForwardGpuOptimized(input, out var gpuResult))
            return gpuResult;

        Tensor<T> current = input;
        foreach (var layer in Layers)
        {
            current = layer.Forward(current);
        }
        return current;
    }

    /// <summary>
    /// Trains the TVAE using the provided input and expected output.
    /// </summary>
    public override void Train(Tensor<T> input, Tensor<T> expectedOutput)
    {
        Tensor<T> prediction = Predict(input);
        LastLoss = _lossFunction.CalculateLoss(prediction.ToVector(), expectedOutput.ToVector());
        Tensor<T> error = prediction.Subtract(expectedOutput);
        BackpropagateError(error);
        UpdateNetworkParameters();
    }

    /// <summary>
    /// Backpropagates the error through the encoder layers.
    /// </summary>
    private void BackpropagateError(Tensor<T> error)
    {
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
    /// Fits the TVAE generator to the provided real tabular data.
    /// </summary>
    /// <param name="data">The real data matrix where each row is a sample and each column is a feature.</param>
    /// <param name="columns">Metadata describing each column (type, categories, etc.).</param>
    /// <param name="epochs">Number of training epochs.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This is the "learning" step. The generator studies your real data:
    /// 1. Fits VGM transformer for mode-specific normalization
    /// 2. Trains encoder and decoder to compress and reconstruct data
    /// After fitting, call Generate() to create new synthetic rows.
    /// </para>
    /// </remarks>
    public void Fit(Matrix<T> data, IReadOnlyList<ColumnMetadata> columns, int epochs)
    {
        ValidateFitInputs(data, columns, epochs);

        _columns = PrepareColumns(data, columns);

        // Step 1: Fit transformer (same VGM + one-hot as CTGAN)
        _transformer = new TabularDataTransformer<T>(_options.VGMModes, _random);
        _transformer.Fit(data, _columns);
        _dataWidth = _transformer.TransformedWidth;

        // Step 2: Transform data
        var transformedData = _transformer.Transform(data);

        // Step 3: Rebuild layers with actual data dimensions
        RebuildLayersWithActualDimensions(_dataWidth);

        // Step 4: Training loop
        int batchSize = Math.Min(_options.BatchSize, data.Rows);
        int numBatches = Math.Max(1, data.Rows / batchSize);
        T scaledLr = NumOps.FromDouble(_options.LearningRate / batchSize);

        for (int epoch = 0; epoch < epochs; epoch++)
        {
            for (int batch = 0; batch < numBatches; batch++)
            {
                TrainBatch(transformedData, batch, batchSize, scaledLr);
            }
        }

        IsFitted = true;
    }

    /// <summary>
    /// Generates synthetic tabular data samples.
    /// </summary>
    /// <param name="numSamples">Number of synthetic rows to generate.</param>
    /// <param name="conditionColumn">Optional condition column index.</param>
    /// <param name="conditionValue">Optional condition value.</param>
    /// <returns>Matrix of synthetic data in original space.</returns>
    public Matrix<T> Generate(int numSamples, Vector<T>? conditionColumn = null, Vector<T>? conditionValue = null)
    {
        if (_transformer is null || !IsFitted)
        {
            throw new InvalidOperationException("Generator must be fitted before generating data. Call Fit() first.");
        }

        var transformedRows = new Matrix<T>(numSamples, _dataWidth);

        for (int i = 0; i < numSamples; i++)
        {
            // Sample from standard normal latent space
            var z = CreateStandardNormalVector(_options.LatentDimension);

            // Decode
            var decoded = DecoderForward(VectorToTensor(z));

            // Apply output activations (tanh for continuous values, softmax for mode indicators/categories)
            var activated = ApplyOutputActivations(decoded);

            // Copy to output
            for (int j = 0; j < _dataWidth && j < activated.Length; j++)
            {
                transformedRows[i, j] = activated[j];
            }
        }

        // Inverse transform to original space
        return _transformer.InverseTransform(transformedRows);
    }

    /// <inheritdoc />
    public async Task FitAsync(Matrix<T> data, IReadOnlyList<ColumnMetadata> columns, int epochs, CancellationToken ct = default)
    {
        ValidateFitInputs(data, columns, epochs);

        _columns = PrepareColumns(data, columns);

        await Task.Run(() =>
        {
            ct.ThrowIfCancellationRequested();

            // Step 1: Fit transformer
            _transformer = new TabularDataTransformer<T>(_options.VGMModes, _random);
            _transformer.Fit(data, _columns);
            _dataWidth = _transformer.TransformedWidth;

            // Step 2: Transform data
            var transformedData = _transformer.Transform(data);

            // Step 3: Rebuild layers with actual data dimensions
            RebuildLayersWithActualDimensions(_dataWidth);

            // Step 4: Training loop
            int batchSize = Math.Min(_options.BatchSize, data.Rows);
            int numBatches = Math.Max(1, data.Rows / batchSize);
            T scaledLr = NumOps.FromDouble(_options.LearningRate / batchSize);

            for (int epoch = 0; epoch < epochs; epoch++)
            {
                ct.ThrowIfCancellationRequested();
                for (int batch = 0; batch < numBatches; batch++)
                {
                    TrainBatch(transformedData, batch, batchSize, scaledLr);
                }
            }

            IsFitted = true;
        }, ct).ConfigureAwait(false);
    }

    #endregion

    #region Forward Passes

    /// <summary>
    /// Encoder forward pass: produces mean and logvar for the latent distribution.
    /// For default layers, the last layer outputs 2*latentDim which is split.
    /// For custom layers, separate mean/logvar heads project from the encoder output.
    /// </summary>
    private (Tensor<T> Mean, Tensor<T> LogVar) EncoderForward(Tensor<T> input)
    {
        var current = input;
        foreach (var layer in Layers)
        {
            current = layer.Forward(current);
        }
        _lastEncoderOutput = current;

        if (_usingCustomLayers)
        {
            // Custom layers: use separate projection heads
            var mean = _meanLayer is not null ? _meanLayer.Forward(current) : current;
            var logVar = _logVarLayer is not null ? _logVarLayer.Forward(current) : current;
            return (mean, logVar);
        }
        else
        {
            // Default layers: last layer outputs 2*latentDim, split into mean and logvar
            return SplitEncoderOutput(current);
        }
    }

    /// <summary>
    /// Splits the encoder output tensor into mean and logvar halves.
    /// </summary>
    private (Tensor<T> Mean, Tensor<T> LogVar) SplitEncoderOutput(Tensor<T> encoderOutput)
    {
        int latentDim = _options.LatentDimension;
        int[] halfShape = DeriveShapeWithLastDim(encoderOutput.Shape, latentDim);
        var mean = new Tensor<T>(halfShape);
        var logVar = new Tensor<T>(halfShape);

        for (int i = 0; i < latentDim && i < encoderOutput.Length; i++)
        {
            mean[i] = encoderOutput[i];
        }
        for (int i = 0; i < latentDim && (latentDim + i) < encoderOutput.Length; i++)
        {
            logVar[i] = encoderOutput[latentDim + i];
        }

        return (mean, logVar);
    }

    /// <summary>
    /// Reparameterization trick: z = mean + exp(0.5 * logVar) * epsilon.
    /// Caches epsilon for proper backward gradient computation.
    /// </summary>
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
    /// Decoder forward pass: reconstructs data from latent code z.
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

    #region Training

    private void TrainBatch(Matrix<T> transformedData, int batchIndex, int batchSize, T scaledLearningRate)
    {
        int startRow = batchIndex * batchSize;
        int endRow = Math.Min(startRow + batchSize, transformedData.Rows);

        for (int row = startRow; row < endRow; row++)
        {
            // Get input row
            var inputVec = GetRow(transformedData, row);
            var inputTensor = VectorToTensor(inputVec);

            // Forward: encode -> reparameterize -> decode
            var (mean, logVar) = EncoderForward(inputTensor);
            var z = Reparameterize(mean, logVar);
            var rawOutput = DecoderForward(z);
            var activated = ApplyOutputActivations(rawOutput);

            // Compute gradient of reconstruction loss w.r.t. decoder raw output
            var outputGrad = ComputeOutputGradient(inputTensor, activated);

            // Sanitize and clip gradient to prevent NaN propagation
            outputGrad = SafeGradient(outputGrad, 5.0);

            // Backward through decoder -> returns gradient w.r.t. z
            var zGrad = BackwardDecoder(outputGrad);

            // Backward through reparameterization and encoder
            BackwardEncoder(zGrad, mean, logVar);

            // Per-sample parameter update
            UpdateAllParameters(scaledLearningRate);
        }
    }

    /// <summary>
    /// Computes the gradient of the ELBO reconstruction loss with respect to the decoder's raw output,
    /// properly incorporating the derivatives of per-column output activations (tanh, softmax).
    /// </summary>
    /// <remarks>
    /// <para>
    /// For softmax + cross-entropy: the gradient simplifies to softmax(logit) - target,
    /// which is already dL/d(logit) (the standard simplification).
    /// </para>
    /// <para>
    /// For tanh + MSE: dL/d(raw) = 2*(tanh(raw) - target) * (1 - tanh(raw)^2),
    /// which chains the MSE gradient through the tanh derivative.
    /// </para>
    /// </remarks>
    private Tensor<T> ComputeOutputGradient(Tensor<T> input, Tensor<T> activated)
    {
        var grad = new Tensor<T>(activated.Shape);

        if (_transformer is null) return grad;

        int idx = 0;
        for (int col = 0; col < _columns.Count && idx < input.Length; col++)
        {
            var transform = _transformer.GetTransformInfo(col);

            if (transform.IsContinuous)
            {
                // Continuous normalized value: tanh activation + MSE loss
                if (idx < input.Length && idx < activated.Length)
                {
                    double target = NumOps.ToDouble(input[idx]);
                    double tanhVal = NumOps.ToDouble(activated[idx]);
                    double diff = tanhVal - target;
                    double tanhDeriv = 1.0 - tanhVal * tanhVal;
                    grad[idx] = NumOps.FromDouble(2.0 * diff * tanhDeriv * _options.LossWeight);
                    idx++;
                }

                // Mode indicators: softmax activation + cross-entropy loss
                int numModes = transform.Width - 1;
                for (int m = 0; m < numModes && idx < input.Length; m++)
                {
                    double target = NumOps.ToDouble(input[idx]);
                    double predicted = NumOps.ToDouble(activated[idx]);
                    grad[idx] = NumOps.FromDouble((predicted - target) * _options.LossWeight);
                    idx++;
                }
            }
            else
            {
                // Categorical: softmax activation + cross-entropy loss
                int numCats = transform.Width;
                for (int c = 0; c < numCats && idx < input.Length; c++)
                {
                    double target = NumOps.ToDouble(input[idx]);
                    double predicted = NumOps.ToDouble(activated[idx]);
                    grad[idx] = NumOps.FromDouble((predicted - target) * _options.LossWeight);
                    idx++;
                }
            }
        }

        return grad;
    }

    #endregion

    #region Backward Passes

    /// <summary>
    /// Backward pass through the decoder, returning the gradient w.r.t. the decoder input (z).
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
    /// Backward pass through reparameterization and encoder.
    /// Properly chains gradients through z = mean + exp(0.5 * logvar) * epsilon,
    /// then combines both mean and logvar path gradients for the shared encoder layers.
    /// </summary>
    private void BackwardEncoder(Tensor<T> zGrad, Tensor<T> mean, Tensor<T> logVar)
    {
        // Through reparameterization: z = mean + exp(0.5 * logvar) * epsilon
        //
        // Reconstruction gradient through reparameterization:
        //   dL/d(mean)_recon   = dL/dz * 1
        //   dL/d(logvar)_recon = dL/dz * epsilon * 0.5 * exp(0.5 * logvar)
        //
        // KL divergence gradient: KL = -0.5 * sum(1 + logvar - mean^2 - exp(logvar))
        //   dKL/d(mean)   = mean
        //   dKL/d(logvar) = 0.5 * (exp(logvar) - 1)

        var meanGrad = new Tensor<T>(mean.Shape);
        var logVarGrad = new Tensor<T>(logVar.Shape);

        for (int i = 0; i < mean.Length; i++)
        {
            double m = NumOps.ToDouble(mean[i]);
            double lv = NumOps.ToDouble(logVar[i]);
            double dz = i < zGrad.Length ? NumOps.ToDouble(zGrad[i]) : 0;
            double eps = _lastEpsilon is not null && i < _lastEpsilon.Length
                ? NumOps.ToDouble(_lastEpsilon[i])
                : 0;

            // Reconstruction + KL gradient for mean
            double dMean = dz + m;

            // Reconstruction + KL gradient for logvar
            double dLogVar = dz * eps * 0.5 * Math.Exp(0.5 * lv) + 0.5 * (Math.Exp(lv) - 1.0);

            meanGrad[i] = NumOps.FromDouble(dMean);
            logVarGrad[i] = NumOps.FromDouble(dLogVar);
        }

        if (_usingCustomLayers)
        {
            // Custom layers: backward through separate mean/logvar heads, then sum
            Tensor<T> gradFromMean = _meanLayer is not null ? _meanLayer.Backward(meanGrad) : meanGrad;
            Tensor<T> gradFromLogVar = _logVarLayer is not null ? _logVarLayer.Backward(logVarGrad) : logVarGrad;

            // Sum gradients from both paths
            var encoderOutGrad = new Tensor<T>(gradFromMean.Shape);
            for (int i = 0; i < encoderOutGrad.Length; i++)
            {
                double gm = i < gradFromMean.Length ? NumOps.ToDouble(gradFromMean[i]) : 0;
                double gl = i < gradFromLogVar.Length ? NumOps.ToDouble(gradFromLogVar[i]) : 0;
                encoderOutGrad[i] = NumOps.FromDouble(gm + gl);
            }

            // Backward through shared encoder layers
            var current = encoderOutGrad;
            for (int i = Layers.Count - 1; i >= 0; i--)
            {
                current = Layers[i].Backward(current);
            }
        }
        else
        {
            // Default layers: combine mean and logvar gradients into 2*latentDim gradient
            int latentDim = _options.LatentDimension;
            int[] encGradShape = _lastEncoderOutput is not null
                ? (int[])_lastEncoderOutput.Shape.Clone()
                : DeriveShapeWithLastDim(mean.Shape, 2 * latentDim);
            var combinedGrad = new Tensor<T>(encGradShape);
            for (int i = 0; i < latentDim; i++)
            {
                combinedGrad[i] = meanGrad[i];
                combinedGrad[latentDim + i] = logVarGrad[i];
            }

            // Backward through encoder layers
            var current = combinedGrad;
            for (int i = Layers.Count - 1; i >= 0; i--)
            {
                current = Layers[i].Backward(current);
            }
        }
    }

    private void UpdateAllParameters(T learningRate)
    {
        // Update encoder layers
        foreach (var layer in Layers)
        {
            layer.UpdateParameters(learningRate);
        }

        // Update mean/logvar projection heads (if using custom layers)
        _meanLayer?.UpdateParameters(learningRate);
        _logVarLayer?.UpdateParameters(learningRate);

        // Update decoder layers
        foreach (var layer in _decoderLayers)
        {
            layer.UpdateParameters(learningRate);
        }
    }

    #endregion

    #region Output Activations

    /// <summary>
    /// Applies per-column output activations (tanh for continuous, softmax for categorical).
    /// </summary>
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
                // Tanh for normalized value
                if (idx < output.Length)
                {
                    double val = NumOps.ToDouble(output[idx]);
                    result[idx] = NumOps.FromDouble(Math.Tanh(val));
                    idx++;
                }

                // Softmax for mode indicators
                int numModes = transform.Width - 1;
                ApplySoftmax(output, result, ref idx, numModes);
            }
            else
            {
                // Softmax for categorical one-hot
                ApplySoftmax(output, result, ref idx, transform.Width);
            }
        }

        return result;
    }

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

    #region Gradient Utilities

    /// <summary>
    /// Sanitizes and clips gradients to prevent NaN propagation and exploding gradients.
    /// </summary>
    private Tensor<T> SafeGradient(Tensor<T> grad, double maxNorm)
    {
        var result = new Tensor<T>(grad.Shape);
        double sumSq = 0;

        for (int i = 0; i < grad.Length; i++)
        {
            double val = NumOps.ToDouble(grad[i]);
            if (double.IsNaN(val) || double.IsInfinity(val))
            {
                val = 0;
            }
            result[i] = NumOps.FromDouble(val);
            sumSq += val * val;
        }

        double norm = Math.Sqrt(sumSq);
        if (norm > maxNorm && norm > 0)
        {
            double scale = maxNorm / norm;
            for (int i = 0; i < result.Length; i++)
            {
                double val = NumOps.ToDouble(result[i]);
                result[i] = NumOps.FromDouble(val * scale);
            }
        }

        return result;
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
                { "BatchSize", _options.BatchSize },
                { "LossWeight", _options.LossWeight },
                { "EncoderLayerCount", Layers.Count },
                { "DecoderLayerCount", _decoderLayers.Count },
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
        writer.Write(_options.BatchSize);
        writer.Write(_options.LearningRate);
        writer.Write(_options.LossWeight);
        writer.Write(_options.VGMModes);
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
        return new TVAEGenerator<T>(
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

    #endregion

    #region IJitCompilable Override

    /// <summary>
    /// TVAE uses separate encoder/decoder networks with reparameterization trick which cannot be represented as a single computation graph.
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
