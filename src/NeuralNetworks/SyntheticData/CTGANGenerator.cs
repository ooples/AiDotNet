using AiDotNet.Autodiff;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks.SyntheticData;
using AiDotNet.Optimizers;
using AiDotNet.Tensors.Helpers;

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
public class CTGANGenerator<T> : NeuralNetworkBase<T>, ISyntheticTabularGenerator<T>
{
    private readonly CTGANOptions<T> _options;
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
        _optimizer = optimizer ?? new AdamOptimizer<T, Tensor<T>, Tensor<T>>(this);

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
                _genBNLayers.Add(new BatchNormalizationLayer<T>(dim));
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
                _genBNLayers.Add(new BatchNormalizationLayer<T>(dim));
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
    public override Tensor<T> Predict(Tensor<T> input)
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
        BackpropagateError(error);
        UpdateNetworkParameters();
    }

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

        BackwardGeneratorWithResidual(error);
    }

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
                for (int dStep = 0; dStep < _options.DiscriminatorSteps; dStep++)
                {
                    TrainDiscriminatorStep(transformedData, numPacks, lr);
                }

                TrainGeneratorStep(numPacks, lr);
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
                        TrainDiscriminatorStep(transformedData, numPacks, lr);
                    }
                    TrainGeneratorStep(numPacks, lr);
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
    /// Trains the discriminator for one step using WGAN-GP objective.
    /// </summary>
    private void TrainDiscriminatorStep(Matrix<T> transformedData, int numPacks, T learningRate)
    {
        if (_sampler is null || _packedRealBuf is null || _packedFakeBuf is null ||
            _noiseBuf is null || _genInputBuf is null || _realSingleBuf is null ||
            _fakeSingleBuf is null || _realRowBuf is null || _fakeRowBuf is null ||
            _oneGrad is null || _negOneGrad is null) return;

        int pacSize = _options.PacSize;
        int singleDim = _dataWidth + _condWidth;
        T scaledLr = NumOps.FromDouble(NumOps.ToDouble(learningRate) / numPacks);

        for (int p = 0; p < numPacks; p++)
        {
            // Zero the packed buffers
            for (int i = 0; i < _packedInputDim; i++)
            {
                _packedRealBuf[i] = NumOps.Zero;
                _packedFakeBuf[i] = NumOps.Zero;
            }

            for (int s = 0; s < pacSize; s++)
            {
                var (condVector, rowIdx) = _sampler.SampleConditionAndRow();
                FillRow(transformedData, rowIdx, _realRowBuf);
                ConcatInto(_realRowBuf, condVector, _realSingleBuf);

                FillStandardNormal(_noiseBuf);
                ConcatInto(_noiseBuf, condVector, _genInputBuf);
                var fakeTransformed = Predict(VectorToTensor(_genInputBuf));
                FillFromTensor(fakeTransformed, _fakeRowBuf);
                ConcatInto(_fakeRowBuf, condVector, _fakeSingleBuf);

                for (int d = 0; d < singleDim; d++)
                {
                    _packedRealBuf[s * singleDim + d] = d < _realSingleBuf.Length ? _realSingleBuf[d] : NumOps.Zero;
                    _packedFakeBuf[s * singleDim + d] = d < _fakeSingleBuf.Length ? _fakeSingleBuf[d] : NumOps.Zero;
                }
            }

            // Train on fake: D(fake) should be low
            _ = DiscriminatorForward(VectorToTensor(_packedFakeBuf), isTraining: true);
            BackwardDiscriminator(_oneGrad);
            UpdateDiscriminatorParameters(scaledLr);

            // Train on real: D(real) should be high
            _ = DiscriminatorForward(VectorToTensor(_packedRealBuf), isTraining: true);
            BackwardDiscriminator(_negOneGrad);
            UpdateDiscriminatorParameters(scaledLr);

            // Gradient penalty
            ApplyGradientPenalty(_packedRealBuf, _packedFakeBuf, scaledLr);
        }
    }

    /// <summary>
    /// Trains the generator for one step: minimize -E[D(G(z))].
    /// </summary>
    private void TrainGeneratorStep(int numPacks, T learningRate)
    {
        if (_sampler is null || _packedFakeBuf is null || _noiseBuf is null ||
            _genInputBuf is null || _fakeRowBuf is null || _fakeSingleBuf is null ||
            _sampleGradBuf is null) return;

        int pacSize = _options.PacSize;
        int singleDim = _dataWidth + _condWidth;
        T scaledLr = NumOps.FromDouble(NumOps.ToDouble(learningRate) / numPacks);

        for (int p = 0; p < numPacks; p++)
        {
            // Must keep per-sample noise/cond for backprop second pass
            var noises = new List<Vector<T>>(pacSize);
            var condVectors = new List<Vector<T>>(pacSize);

            // Zero packed fake buffer
            for (int i = 0; i < _packedInputDim; i++)
                _packedFakeBuf[i] = NumOps.Zero;

            for (int s = 0; s < pacSize; s++)
            {
                var condVector = _sampler.SampleRandomConditionVector();
                condVectors.Add(condVector);

                var noise = CreateStandardNormalVector(_options.EmbeddingDimension);
                noises.Add(noise);

                ConcatInto(noise, condVector, _genInputBuf);
                var fakeTransformed = Predict(VectorToTensor(_genInputBuf));
                FillFromTensor(fakeTransformed, _fakeRowBuf);
                ConcatInto(_fakeRowBuf, condVector, _fakeSingleBuf);

                for (int d = 0; d < singleDim; d++)
                {
                    _packedFakeBuf[s * singleDim + d] = d < _fakeSingleBuf.Length ? _fakeSingleBuf[d] : NumOps.Zero;
                }
            }

            // Discriminator score on fake pack (eval mode - no dropout)
            _ = DiscriminatorForward(VectorToTensor(_packedFakeBuf), isTraining: false);

            // Generator loss: minimize -E[D(fake)]
            // Compute dD/dInput using GradientTape autodiff, then negate for generator gradient
            var discInputGrad = TapeLayerBridge<T>.ComputeInputGradient(
                VectorToTensor(_packedFakeBuf),
                _discLayers,
                TapeLayerBridge<T>.HiddenActivation.LeakyReLU,
                applyActivationOnLast: false);
            // Negate: generator wants to maximize D(fake), so gradient is -dD/dx
            for (int g = 0; g < discInputGrad.Length; g++)
            {
                discInputGrad[g] = NumOps.Negate(discInputGrad[g]);
            }

            // Sanitize and clip
            SanitizeTensor(discInputGrad);
            discInputGrad = ClipGradientNorm(discInputGrad, 5.0);

            T perSampleLr = NumOps.FromDouble(NumOps.ToDouble(scaledLr) / pacSize);
            for (int s = 0; s < pacSize; s++)
            {
                // Reuse sample gradient buffer
                for (int d = 0; d < _dataWidth; d++)
                    _sampleGradBuf[d] = NumOps.Zero;
                for (int d = 0; d < _dataWidth && (s * singleDim + d) < discInputGrad.Length; d++)
                {
                    _sampleGradBuf[d] = discInputGrad[s * singleDim + d];
                }

                ConcatInto(noises[s], condVectors[s], _genInputBuf);
                _ = Predict(VectorToTensor(_genInputBuf));

                BackpropagateError(_sampleGradBuf);
                UpdateGeneratorParameters(perSampleLr);
            }
        }
    }

    #endregion

    #region Gradient Penalty

    /// <summary>
    /// Applies WGAN-GP gradient penalty using GradientTape automatic differentiation.
    /// </summary>
    /// <remarks>
    /// Uses TapeLayerBridge to compute ∇_x D(x) automatically instead of manual
    /// ManualLinearBackward chains. The GradientTape records the discriminator forward
    /// pass through TensorOperations and computes exact gradients via reverse-mode autodiff.
    /// </remarks>
    private void ApplyGradientPenalty(Vector<T> packedReal, Vector<T> packedFake, T scaledLr)
    {
        if (_interpolatedBuf is null) return;

        double alpha = _random.NextDouble();
        int len = Math.Min(packedReal.Length, Math.Min(packedFake.Length, _interpolatedBuf.Length));

        for (int i = 0; i < len; i++)
        {
            _interpolatedBuf[i] = NumOps.Add(
                NumOps.Multiply(NumOps.FromDouble(alpha), packedReal[i]),
                NumOps.Multiply(NumOps.FromDouble(1.0 - alpha), packedFake[i]));
        }

        // Compute gradient penalty using GradientTape autodiff
        var interpolatedTensor = VectorToTensor(_interpolatedBuf);
        var inputGrad = TapeLayerBridge<T>.ComputeInputGradient(
            interpolatedTensor,
            _discLayers,
            TapeLayerBridge<T>.HiddenActivation.LeakyReLU,
            applyActivationOnLast: false);

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
            _ = DiscriminatorForward(VectorToTensor(_interpolatedBuf), isTraining: false);

            var penaltyGrad = new Tensor<T>([1]);
            penaltyGrad[0] = NumOps.FromDouble(penaltyGradScale);
            BackwardDiscriminator(penaltyGrad);
            UpdateDiscriminatorParameters(scaledLr);
        }
    }

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

    private void BackwardGeneratorWithResidual(Tensor<T> gradOutput)
    {
        int inputDim = _options.EmbeddingDimension + _condWidth;
        var current = gradOutput;
        var denseLayers = new List<ILayer<T>>(Layers);

        // Backward through output layer
        current = denseLayers[^1].Backward(current);

        // Split off residual gradient
        int lastHiddenDim = current.Length - inputDim;
        if (lastHiddenDim > 0)
        {
            var hiddenGrad = new Tensor<T>([lastHiddenDim]);
            for (int j = 0; j < lastHiddenDim && j < current.Length; j++)
            {
                hiddenGrad[j] = current[j];
            }
            current = hiddenGrad;
        }

        // Backward through hidden layers
        int numHiddenLayers = denseLayers.Count - 1;
        for (int i = numHiddenLayers - 1; i >= 0; i--)
        {
            if (i < _genPreActivations.Count)
            {
                current = ApplyReLUDerivative(current, _genPreActivations[i]);
            }

            if (i < _genBNLayers.Count)
            {
                current = _genBNLayers[i].Backward(current);
            }

            current = denseLayers[i].Backward(current);

            if (i > 0)
            {
                int prevDim = current.Length - inputDim;
                if (prevDim > 0)
                {
                    var hiddenGrad = new Tensor<T>([prevDim]);
                    for (int j = 0; j < prevDim && j < current.Length; j++)
                    {
                        hiddenGrad[j] = current[j];
                    }
                    current = hiddenGrad;
                }
            }
        }
    }

    private void BackwardDiscriminator(Tensor<T> gradOutput)
    {
        var current = gradOutput;
        int layerIdx = _discLayerDims.Count - 1;

        for (int i = _discLayers.Count - 1; i >= 0; i--)
        {
            if (_discLayers[i] is DropoutLayer<T>)
            {
                continue;
            }

            current = _discLayers[i].Backward(current);
            layerIdx--;

            if (layerIdx >= 0 && layerIdx < _discPreActivations.Count)
            {
                current = ApplyLeakyReLUDerivative(current, _discPreActivations[layerIdx]);
            }
        }
    }

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
        var result = new Tensor<T>(input.Shape);
        for (int i = 0; i < input.Length; i++)
        {
            result[i] = NumOps.ToDouble(input[i]) > 0 ? input[i] : NumOps.Zero;
        }
        return result;
    }

    private Tensor<T> ApplyReLUDerivative(Tensor<T> gradOutput, Tensor<T> preActivation)
    {
        int len = Math.Min(gradOutput.Length, preActivation.Length);
        var result = new Tensor<T>(gradOutput.Shape);
        for (int i = 0; i < len; i++)
        {
            result[i] = NumOps.ToDouble(preActivation[i]) > 0 ? gradOutput[i] : NumOps.Zero;
        }
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
            if (NumOps.ToDouble(preActivation[i]) > 0)
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
                { "EmbeddingDimension", _options.EmbeddingDimension },
                { "GeneratorDimensions", _options.GeneratorDimensions },
                { "DiscriminatorDimensions", _options.DiscriminatorDimensions },
                { "BatchSize", _options.BatchSize },
                { "PacSize", _options.PacSize },
                { "GradientPenaltyWeight", _options.GradientPenaltyWeight },
                { "GeneratorLayerCount", Layers.Count },
                { "GeneratorLayerTypes", Layers.Select(l => l.GetType().Name).ToArray() }
            },
            ModelData = this.Serialize()
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
        var clone = new Tensor<T>(source.Shape);
        for (int i = 0; i < source.Length; i++) clone[i] = source[i];
        return clone;
    }

    #endregion

    #region IJitCompilable Override

    /// <summary>
    /// Gets whether this CTGAN generator supports JIT compilation for accelerated generation.
    /// </summary>
    /// <remarks>
    /// <para>
    /// CTGAN supports JIT compilation when the model is fitted and using the default
    /// (non-custom) layer configuration. The computation graph exports the generator
    /// MLP forward pass with batch normalization and residual connections.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> After training, you can JIT compile the generator for faster
    /// synthetic data creation. This compiles the neural network forward pass (the
    /// computationally expensive part) into optimized native code.
    /// </para>
    /// </remarks>
    public override bool SupportsJitCompilation =>
        IsFitted && Layers.Count > 1 && !_usingCustomLayers &&
        _genBNLayers.Count > 0 &&
        Layers.All(l => l.SupportsJitCompilation) &&
        _genBNLayers.All(l => l.SupportsJitCompilation);

    /// <summary>
    /// Exports the CTGAN generator network as a computation graph for JIT compilation.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Exports the generator MLP with residual connections and batch normalization.
    /// The graph covers: noise input → FC + BN + ReLU (with residual concat) → output FC.
    /// Column-specific output activations (Tanh/Softmax) are applied separately after
    /// the JIT-compiled forward pass.
    /// </para>
    /// </remarks>
    public override ComputationNode<T> ExportComputationGraph(List<ComputationNode<T>> inputNodes)
    {
        if (!SupportsJitCompilation)
        {
            throw new NotSupportedException(
                $"{GetType().Name} does not support JIT compilation in its current configuration.");
        }

        int genInputDim = _options.EmbeddingDimension + _condWidth;
        var hiddenLayers = Layers.Take(Layers.Count - 1).ToList();

        return TapeLayerBridge<T>.ExportMLPGeneratorGraph(
            inputNodes,
            genInputDim,
            hiddenLayers,
            _genBNLayers.Cast<ILayer<T>>().ToList(),
            Layers[^1],
            TapeLayerBridge<T>.HiddenActivation.ReLU,
            TapeLayerBridge<T>.HiddenActivation.None,
            useResidualConcat: true);
    }

    #endregion
}
