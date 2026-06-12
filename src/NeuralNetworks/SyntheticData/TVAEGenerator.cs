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

    // Encoder layers (the forward iterates this typed list; all sub-networks are
    // ALSO registered in the shared Layers collection so the tape-training
    // parameter collection / optimizer sees encoder + heads + decoder together).
    private readonly List<ILayer<T>> _encoderLayers = new();

    // Mean and logvar projection heads (auxiliary, used when not using custom layers)
    // Custom layers produce concatenated 2*latentDim output; default layers use separate heads
    private FullyConnectedLayer<T>? _meanLayer;
    private FullyConnectedLayer<T>? _logVarLayer;

    // Decoder layers (auxiliary, not user-overridable)
    private readonly List<ILayer<T>> _decoderLayers = new();

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
    /// <summary>
    /// Initializes a new instance with default architecture settings.
    /// </summary>
    public TVAEGenerator()
        : this(new NeuralNetworkArchitecture<T>(
            inputType: Enums.InputType.OneDimensional,
            taskType: Enums.NeuralNetworkTaskType.Regression,
            inputSize: 10,
            outputSize: 10))
    {
    }

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
        _encoderLayers.Clear();
        if (Architecture.Layers != null && Architecture.Layers.Count > 0)
        {
            // Use the layers provided by the user for the encoder
            _encoderLayers.AddRange(Architecture.Layers);
            _usingCustomLayers = true;

            // Create separate mean and logvar projection heads for custom layers
            var identity = new IdentityActivation<T>() as IActivationFunction<T>;
            _meanLayer = new FullyConnectedLayer<T>(_options.LatentDimension, identity);
            _logVarLayer = new FullyConnectedLayer<T>(_options.LatentDimension, identity);
        }
        else
        {
            // Create default encoder layers
            // Default encoder outputs 2*latentDim (mean+logvar concatenated)
            int inputDim = Architecture.CalculatedInputSize;
            _encoderLayers.AddRange(LayerHelper<T>.CreateDefaultTVAEEncoderLayers(
                inputDim, _options.LatentDimension, _options.EncoderDimensions));
            _usingCustomLayers = false;
        }

        // Create default decoder layers (always internal, not user-overridable)
        _decoderLayers.Clear();
        _decoderLayers.AddRange(LayerHelper<T>.CreateDefaultTVAEDecoderLayers(
            _options.LatentDimension, Architecture.OutputSize, _options.DecoderDimensions));

        RegisterAllLayers();
    }

    /// <summary>
    /// Rebuilds encoder and decoder layers with actual data dimensions discovered during Fit().
    /// </summary>
    private void RebuildLayersWithActualDimensions(int dataWidth)
    {
        if (!_usingCustomLayers)
        {
            _encoderLayers.Clear();
            _encoderLayers.AddRange(LayerHelper<T>.CreateDefaultTVAEEncoderLayers(
                dataWidth, _options.LatentDimension, _options.EncoderDimensions));
        }
        else
        {
            // For custom layers, rebuild the projection heads with actual latent dimension
            var identity = new IdentityActivation<T>() as IActivationFunction<T>;
            _meanLayer = new FullyConnectedLayer<T>(_options.LatentDimension, identity);
            _logVarLayer = new FullyConnectedLayer<T>(_options.LatentDimension, identity);
        }

        // Always rebuild decoder with actual dimensions
        _decoderLayers.Clear();
        _decoderLayers.AddRange(LayerHelper<T>.CreateDefaultTVAEDecoderLayers(
            _options.LatentDimension, dataWidth, _options.DecoderDimensions));

        RegisterAllLayers();
    }

    /// <summary>
    /// Registers every trainable sub-network (encoder, mean/logvar heads, decoder)
    /// in the shared <see cref="NeuralNetworkBase{T}.Layers"/> collection in a
    /// stable order so the tape-based training parameter collection and
    /// GetParameters/GetParameterGradients/UpdateParameters see the full set.
    /// The forward pass uses the typed lists (<see cref="_encoderLayers"/> /
    /// <see cref="_decoderLayers"/>), not Layers, so encoder and decoder stay
    /// distinct stages.
    /// </summary>
    private void RegisterAllLayers()
    {
        Layers.Clear();
        Layers.AddRange(_encoderLayers);
        if (_meanLayer is not null) Layers.Add(_meanLayer);
        if (_logVarLayer is not null) Layers.Add(_logVarLayer);
        Layers.AddRange(_decoderLayers);
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
        EnsureSizedForInput(input);

        // Deterministic VAE reconstruction: encode, take the latent MEAN (no
        // sampling, so Predict is deterministic), decode, apply per-column output
        // activations. Sampling-based generation lives in Generate().
        var (mean, _) = EncoderForward(input);
        var raw = DecoderForward(mean);
        return ApplyOutputActivations(raw);
    }

    /// <summary>
    /// When the generator has not yet been fitted (e.g. the generated ModelFamily
    /// tests call Train()/Predict() directly), adapt the encoder/decoder layout to
    /// the actual input width so the model is a valid network for any 1-D input.
    /// Once fitted, the width is fixed by the TabularDataTransformer.
    /// </summary>
    private void EnsureSizedForInput(Tensor<T> input)
    {
        if (!IsFitted && input.Length != _dataWidth)
        {
            _dataWidth = input.Length;
            RebuildLayersWithActualDimensions(_dataWidth);
        }
    }

    /// <summary>
    /// Trains the TVAE on a single sample by running one tape-connected ELBO
    /// optimization step (encode → reparameterize → decode → reconstruction + KL).
    /// </summary>
    public override void Train(Tensor<T> input, Tensor<T> expectedOutput)
    {
        SetTrainingMode(true);
        try
        {
            ElboStep(input);
        }
        finally
        {
            SetTrainingMode(false);
        }
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

        SetTrainingMode(true);
        try
        {
            for (int epoch = 0; epoch < epochs; epoch++)
            {
                for (int batch = 0; batch < numBatches; batch++)
                {
                    TrainBatch(transformedData, batch, batchSize, scaledLr);
                }
            }
        }
        finally
        {
            SetTrainingMode(false);
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

            SetTrainingMode(true);
            try
            {
                for (int epoch = 0; epoch < epochs; epoch++)
                {
                    ct.ThrowIfCancellationRequested();
                    for (int batch = 0; batch < numBatches; batch++)
                    {
                        TrainBatch(transformedData, batch, batchSize, scaledLr);
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
        foreach (var layer in _encoderLayers)
        {
            current = layer.Forward(current);
        }

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
    /// Splits the encoder output tensor into mean and logvar halves, using
    /// tape-connected <see cref="NeuralNetworkBase{T}.Engine"/> slices so the
    /// reparameterization gradient flows back into the encoder.
    /// </summary>
    private (Tensor<T> Mean, Tensor<T> LogVar) SplitEncoderOutput(Tensor<T> encoderOutput)
    {
        int latentDim = _options.LatentDimension;
        var flat = encoderOutput.Rank == 1 ? encoderOutput : Engine.Reshape(encoderOutput, new[] { encoderOutput.Length });
        var mean = Engine.TensorSlice(flat, new[] { 0 }, new[] { latentDim });
        var logVar = Engine.TensorSlice(flat, new[] { latentDim }, new[] { latentDim });
        return (mean, logVar);
    }

    /// <summary>
    /// Reparameterization trick: z = mean + exp(0.5 * logVar) ⊙ epsilon, computed
    /// with tape-connected <see cref="NeuralNetworkBase{T}.Engine"/> ops (epsilon is
    /// a sampled constant leaf) so gradients flow into mean and logVar. This is the
    /// standard low-variance VAE gradient estimator (Kingma &amp; Welling, 2014).
    /// </summary>
    private Tensor<T> Reparameterize(Tensor<T> mean, Tensor<T> logVar)
    {
        var eps = new Tensor<T>(mean._shape);
        for (int i = 0; i < eps.Length; i++) eps[i] = SampleStandardNormal();

        // std = exp(0.5 * logVar); z = mean + std ⊙ eps
        var std = Engine.TensorExp(Engine.TensorMultiplyScalar(logVar, NumOps.FromDouble(0.5)));
        return Engine.TensorAdd(mean, Engine.TensorMultiply(std, eps));
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
            ElboStep(VectorToTensor(GetRow(transformedData, row)));
        }
    }

    /// <summary>
    /// Runs one tape-connected ELBO optimization step on a single transformed
    /// input row: encode → reparameterize → decode, then compute the evidence
    /// lower bound (per-column reconstruction loss + KL divergence) as a
    /// tape-tracked scalar and let autodiff backpropagate through the whole VAE
    /// (encoder, reparameterization, decoder) before the optimizer step. Replaces
    /// the previous hand-rolled gradient path whose decoder/encoder backward was
    /// never actually wired, so no parameter received a gradient.
    /// </summary>
    private void ElboStep(Tensor<T> input)
    {
        EnsureSizedForInput(input);
        using var tape = new GradientTape<T>();
        var (mean, logVar) = EncoderForward(input);
        var z = Reparameterize(mean, logVar);
        var rawOutput = DecoderForward(z);
        var loss = ComputeElboLossTape(rawOutput, input, mean, logVar);
        BackwardAndStepOnPrecomputedLoss(tape, loss, _optimizer);
    }

    /// <summary>
    /// Computes the tape-connected negative ELBO: per-column reconstruction loss
    /// (tanh + MSE for the continuous normalized value, softmax + cross-entropy
    /// for mode indicators and categorical one-hots — the CTGAN/TVAE loss of Xu
    /// et al. 2019) scaled by <see cref="TVAEOptions{T}.LossWeight"/>, plus the
    /// standard Gaussian KL divergence to the N(0,I) prior.
    /// </summary>
    private Tensor<T> ComputeElboLossTape(Tensor<T> rawOutput, Tensor<T> target, Tensor<T> mean, Tensor<T> logVar)
    {
        var raw = rawOutput.Rank == 1 ? rawOutput : Engine.Reshape(rawOutput, new[] { rawOutput.Length });
        var tgt = target.Rank == 1 ? target : Engine.Reshape(target, new[] { target.Length });

        Tensor<T>? recon = null;
        int idx = 0;
        if (_transformer is not null)
        {
            for (int col = 0; col < _columns.Count && idx < raw.Length; col++)
            {
                var transform = _transformer.GetTransformInfo(col);
                if (transform.IsContinuous)
                {
                    // Continuous normalized value: tanh + MSE.
                    var rawScalar = Engine.TensorSlice(raw, new[] { idx }, new[] { 1 });
                    var tgtScalar = Engine.TensorSlice(tgt, new[] { idx }, new[] { 1 });
                    var diff = Engine.TensorSubtract(Engine.TensorTanh(rawScalar), tgtScalar);
                    recon = AddScalarLoss(recon, ReduceToScalar(Engine.TensorSquare(diff)));
                    idx++;

                    // Mode indicators: softmax + cross-entropy.
                    int numModes = transform.Width - 1;
                    if (numModes > 0)
                    {
                        recon = AddScalarLoss(recon, SoftmaxCrossEntropy(raw, tgt, idx, numModes));
                        idx += numModes;
                    }
                }
                else
                {
                    // Categorical one-hot: softmax + cross-entropy.
                    recon = AddScalarLoss(recon, SoftmaxCrossEntropy(raw, tgt, idx, transform.Width));
                    idx += transform.Width;
                }
            }
        }

        recon ??= ReduceToScalar(Engine.TensorSquare(Engine.TensorSubtract(raw, tgt)));
        recon = Engine.TensorMultiplyScalar(recon, NumOps.FromDouble(_options.LossWeight));

        // KL(N(mean, exp(logVar)) || N(0, I)) = -0.5 * Σ(1 + logVar - mean² - exp(logVar)).
        var meanSq = Engine.TensorSquare(mean);
        var expLogVar = Engine.TensorExp(logVar);
        var klTerm = Engine.TensorSubtract(
            Engine.TensorSubtract(Engine.TensorAddScalar(logVar, NumOps.One), meanSq),
            expLogVar);
        var kl = Engine.TensorMultiplyScalar(ReduceToScalar(klTerm), NumOps.FromDouble(-0.5));

        return Engine.TensorAdd(recon, kl);
    }

    /// <summary>Tape-connected softmax cross-entropy −Σ target·log(softmax(raw_slice)) over a column slice.</summary>
    private Tensor<T> SoftmaxCrossEntropy(Tensor<T> raw, Tensor<T> tgt, int start, int count)
    {
        if (count <= 0) return ReduceToScalar(Engine.TensorSlice(raw, new[] { start }, new[] { 0 }));
        var rawSlice = Engine.TensorSlice(raw, new[] { start }, new[] { count });
        var tgtSlice = Engine.TensorSlice(tgt, new[] { start }, new[] { count });
        var logProbs = Engine.TensorLogSoftmax(rawSlice, axis: 0);
        var ce = Engine.TensorNegate(ReduceToScalar(Engine.TensorMultiply(tgtSlice, logProbs)));
        return ce;
    }

    /// <summary>Reduces a tensor to a scalar [1] by summing all elements (tape-connected).</summary>
    private Tensor<T> ReduceToScalar(Tensor<T> t)
    {
        var axes = Enumerable.Range(0, t.Shape.Length).ToArray();
        return Engine.ReduceSum(t, axes, keepDims: false);
    }

    private Tensor<T> AddScalarLoss(Tensor<T>? acc, Tensor<T> term)
        => acc is null ? term : Engine.TensorAdd(acc, term);

    #endregion

    #region Output Activations

    /// <summary>
    /// Applies per-column output activations (tanh for continuous, softmax for categorical).
    /// </summary>
    private Tensor<T> ApplyOutputActivations(Tensor<T> output)
    {
        if (_transformer is null) return output;

        var result = new Tensor<T>(output._shape);
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
        int actualCount = Math.Min(count, input.Length - idx);
        if (actualCount <= 0) return;
        var slice = new Tensor<T>([actualCount]);
        input.Data.Span.Slice(idx, actualCount).CopyTo(slice.Data.Span);
        var result = Engine.Softmax(slice, -1);
        result.Data.Span.CopyTo(output.Data.Span.Slice(idx, actualCount));
        idx += actualCount;
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
                { "LatentDimension", _options.LatentDimension },
                { "EncoderDimensions", _options.EncoderDimensions },
                { "DecoderDimensions", _options.DecoderDimensions },
                { "BatchSize", _options.BatchSize },
                { "LossWeight", _options.LossWeight },
                { "EncoderLayerCount", Layers.Count },
                { "DecoderLayerCount", _decoderLayers.Count },
                { "EncoderLayerTypes", Layers.Select(l => l.GetType().Name).ToArray() }
            },
            ModelData = SerializeForMetadata()
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

        // Structural layout so a deserialized clone can re-bind its typed layer
        // references (encoder / mean+logvar heads / decoder) out of the shared
        // Layers collection and reproduce the identical VAE forward.
        writer.Write(_dataWidth);
        writer.Write(IsFitted);
        writer.Write(_usingCustomLayers);
        writer.Write(_encoderLayers.Count);
        writer.Write(_meanLayer is not null);
        writer.Write(_decoderLayers.Count);
    }

    /// <inheritdoc/>
    protected override void DeserializeNetworkSpecificData(BinaryReader reader)
    {
        _ = reader.ReadInt32();    // LatentDimension
        int encDims = reader.ReadInt32();
        for (int i = 0; i < encDims; i++) _ = reader.ReadInt32();
        int decDims = reader.ReadInt32();
        for (int i = 0; i < decDims; i++) _ = reader.ReadInt32();
        _ = reader.ReadInt32();    // BatchSize
        _ = reader.ReadDouble();   // LearningRate
        _ = reader.ReadDouble();   // LossWeight
        _ = reader.ReadInt32();    // VGMModes

        _dataWidth = reader.ReadInt32();
        IsFitted = reader.ReadBoolean();
        _usingCustomLayers = reader.ReadBoolean();
        int encoderCount = reader.ReadInt32();
        bool hasHeads = reader.ReadBoolean();
        int decoderCount = reader.ReadInt32();

        // The base deserializer rebuilt Layers; re-bind the typed references the
        // VAE forward uses (encoder, mean/logvar heads, decoder) from it.
        ExtractLayerReferences(encoderCount, hasHeads, decoderCount);
    }

    /// <summary>
    /// Re-binds <see cref="_encoderLayers"/>, <see cref="_meanLayer"/>,
    /// <see cref="_logVarLayer"/>, and <see cref="_decoderLayers"/> from the shared
    /// <see cref="NeuralNetworkBase{T}.Layers"/> collection using the serialized
    /// split, after deserialization replaced Layers with fresh instances.
    /// </summary>
    private void ExtractLayerReferences(int encoderCount, bool hasHeads, int decoderCount)
    {
        int expected = encoderCount + (hasHeads ? 2 : 0) + decoderCount;
        if (Layers.Count != expected) return;

        _encoderLayers.Clear();
        _decoderLayers.Clear();
        _meanLayer = null;
        _logVarLayer = null;

        int idx = 0;
        for (int i = 0; i < encoderCount; i++) _encoderLayers.Add(Layers[idx++]);
        if (hasHeads)
        {
            _meanLayer = Layers[idx++] as FullyConnectedLayer<T>;
            _logVarLayer = Layers[idx++] as FullyConnectedLayer<T>;
        }
        for (int i = 0; i < decoderCount; i++) _decoderLayers.Add(Layers[idx++]);
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
