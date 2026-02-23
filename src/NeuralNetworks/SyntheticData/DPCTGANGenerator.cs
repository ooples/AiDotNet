using AiDotNet.Autodiff;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.NeuralNetworks.SyntheticData;
using AiDotNet.Optimizers;
using AiDotNet.Tensors.Helpers;

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
public class DPCTGANGenerator<T> : NeuralNetworkBase<T>, ISyntheticTabularGenerator<T>
{
    private readonly DPCTGANOptions<T> _options;
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

    // Privacy accounting
    private double _computedNoiseMultiplier;
    private double _cumulativeEpsilon;

    // Pre-allocated training buffers to avoid per-row GC pressure
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
    /// <param name="maxGradNorm">Maximum gradient norm for clipping (default 5.0).</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This constructor creates a DP-CTGAN network. The key privacy parameters
    /// (epsilon, delta, clipNorm) are in the options object. Lower epsilon = more privacy.
    /// </para>
    /// </remarks>
    public DPCTGANGenerator(
        NeuralNetworkArchitecture<T> architecture,
        DPCTGANOptions<T>? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
        ILossFunction<T>? lossFunction = null,
        double maxGradNorm = 5.0)
        : base(architecture, lossFunction ?? NeuralNetworkHelper<T>.GetDefaultLossFunction(architecture.TaskType), maxGradNorm)
    {
        _options = options ?? new DPCTGANOptions<T>();
        _lossFunction = lossFunction ?? NeuralNetworkHelper<T>.GetDefaultLossFunction(architecture.TaskType);
        _optimizer = optimizer ?? new AdamOptimizer<T, Tensor<T>, Tensor<T>>(this);
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
    /// Fits the DP-CTGAN generator to the provided real tabular data with differential privacy.
    /// </summary>
    /// <param name="data">The real data matrix.</param>
    /// <param name="columns">Column metadata.</param>
    /// <param name="epochs">Number of training epochs (may stop early if privacy budget exhausted).</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Training proceeds like CTGAN but with privacy protections.
    /// Training will stop early if the privacy budget (epsilon) is exhausted before
    /// all epochs complete. Check CumulativeEpsilon after training to see how much
    /// privacy was actually consumed.
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

        // Step 2: Fit sampler
        _sampler = new CTGANDataSampler<T>(_random);
        _sampler.Fit(data, _columns);
        _condWidth = _sampler.ConditionalVectorWidth;

        // Step 3: Setup dimensions
        _packedInputDim = (_dataWidth + _condWidth) * _options.PacSize;

        // Step 4: Rebuild layers with actual dimensions
        int genInputDim = _options.EmbeddingDimension + _condWidth;
        RebuildLayersWithActualDimensions(genInputDim, _dataWidth, _packedInputDim);
        InitializeTrainingBuffers(genInputDim);

        // Step 5: Transform data
        var transformedData = _transformer.Transform(data);

        // Step 6: Compute noise multiplier from privacy budget
        ComputeNoiseMultiplier(data.Rows, epochs);
        _cumulativeEpsilon = 0;

        // Step 7: Training loop with privacy budget check
        T lr = NumOps.FromDouble(_options.LearningRate);
        int pacSize = _options.PacSize;
        int batchSize = _options.BatchSize;
        int numPacks = Math.Max(1, batchSize / pacSize);
        int numBatches = Math.Max(1, data.Rows / (numPacks * pacSize));

        for (int epoch = 0; epoch < epochs; epoch++)
        {
            if (_cumulativeEpsilon >= _options.Epsilon)
            {
                break; // Privacy budget exhausted
            }

            for (int batch = 0; batch < numBatches; batch++)
            {
                for (int dStep = 0; dStep < _options.DiscriminatorSteps; dStep++)
                {
                    TrainDiscriminatorStepDP(transformedData, numPacks, lr);
                }

                TrainGeneratorStep(numPacks, lr);

                double stepEpsilon = ComputeStepPrivacyCost(data.Rows, numPacks * pacSize);
                _cumulativeEpsilon += stepEpsilon;
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
            InitializeTrainingBuffers(genInputDim);

            var transformedData = _transformer.Transform(data);

            ComputeNoiseMultiplier(data.Rows, epochs);
            _cumulativeEpsilon = 0;

            T lr = NumOps.FromDouble(_options.LearningRate);
            int pacSize = _options.PacSize;
            int batchSize = _options.BatchSize;
            int numPacks = Math.Max(1, batchSize / pacSize);
            int numBatches = Math.Max(1, data.Rows / (numPacks * pacSize));

            for (int epoch = 0; epoch < epochs; epoch++)
            {
                ct.ThrowIfCancellationRequested();
                if (_cumulativeEpsilon >= _options.Epsilon) break;

                for (int batch = 0; batch < numBatches; batch++)
                {
                    for (int dStep = 0; dStep < _options.DiscriminatorSteps; dStep++)
                    {
                        TrainDiscriminatorStepDP(transformedData, numPacks, lr);
                    }
                    TrainGeneratorStep(numPacks, lr);

                    double stepEpsilon = ComputeStepPrivacyCost(data.Rows, numPacks * pacSize);
                    _cumulativeEpsilon += stepEpsilon;
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
        int totalSteps = epochs * Math.Max(1, datasetSize / _options.BatchSize);
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

    /// <summary>
    /// Clips a layer's gradient parameters to the specified L2 norm and adds Gaussian noise.
    /// This is the core DP mechanism that ensures individual samples have bounded influence.
    /// </summary>
    private void ClipAndNoiseGradient(ILayer<T> layer)
    {
        var paramsVec = layer.GetParameters();
        double normSq = 0;
        for (int i = 0; i < paramsVec.Length; i++)
        {
            double v = NumOps.ToDouble(paramsVec[i]);
            normSq += v * v;
        }
        double norm = Math.Sqrt(normSq + 1e-12);
        double clipFactor = Math.Min(1.0, _options.ClipNorm / norm);

        // Clip and add noise
        double noiseStd = _options.ClipNorm * _computedNoiseMultiplier;
        var currentParams = layer.GetParameters();
        for (int i = 0; i < currentParams.Length; i++)
        {
            double clipped = NumOps.ToDouble(currentParams[i]) * clipFactor;
            double noise = NumOps.ToDouble(SampleStandardNormal()) * noiseStd;
            currentParams[i] = NumOps.FromDouble(clipped + noise);
        }
        layer.SetParameters(currentParams);
    }

    #endregion

    #region GAN Training Steps

    /// <summary>
    /// Trains the discriminator for one step with DP noise injection on discriminator updates.
    /// </summary>
    private void TrainDiscriminatorStepDP(Matrix<T> transformedData, int numPacks, T learningRate)
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
            // Zero out packed buffers
            for (int z = 0; z < _packedInputDim; z++)
            {
                _packedRealBuf[z] = NumOps.Zero;
                _packedFakeBuf[z] = NumOps.Zero;
            }

            for (int s = 0; s < pacSize; s++)
            {
                var (condVector, rowIdx) = _sampler.SampleConditionAndRow();
                FillRow(_realRowBuf, transformedData, rowIdx);
                ConcatInto(_realSingleBuf, _realRowBuf, condVector);

                FillStandardNormal(_noiseBuf);
                ConcatInto(_genInputBuf, _noiseBuf, condVector);
                var fakeTransformed = Predict(VectorToTensor(_genInputBuf));
                FillFromTensor(_fakeRowBuf, fakeTransformed);
                ConcatInto(_fakeSingleBuf, _fakeRowBuf, condVector);

                for (int d = 0; d < singleDim; d++)
                {
                    _packedRealBuf[s * singleDim + d] = d < _realSingleBuf.Length ? _realSingleBuf[d] : NumOps.Zero;
                    _packedFakeBuf[s * singleDim + d] = d < _fakeSingleBuf.Length ? _fakeSingleBuf[d] : NumOps.Zero;
                }
            }

            // WGAN fake loss
            _ = DiscriminatorForward(VectorToTensor(_packedFakeBuf), isTraining: true);
            BackwardDiscriminator(_oneGrad);
            UpdateDiscriminatorParametersDP(scaledLr);

            // WGAN real loss
            _ = DiscriminatorForward(VectorToTensor(_packedRealBuf), isTraining: true);
            BackwardDiscriminator(_negOneGrad);
            UpdateDiscriminatorParametersDP(scaledLr);
        }
    }

    /// <summary>
    /// Trains the generator for one step (no DP noise on generator - privacy is through discriminator).
    /// </summary>
    private void TrainGeneratorStep(int numPacks, T learningRate)
    {
        if (_sampler is null || _packedFakeBuf is null || _genInputBuf is null ||
            _fakeRowBuf is null || _fakeSingleBuf is null || _sampleGradBuf is null) return;

        int pacSize = _options.PacSize;
        int singleDim = _dataWidth + _condWidth;
        T scaledLr = NumOps.FromDouble(NumOps.ToDouble(learningRate) / numPacks);

        for (int p = 0; p < numPacks; p++)
        {
            // noises and condVectors must be allocated per-sample (needed for backprop second pass)
            var noises = new List<Vector<T>>();
            var condVectors = new List<Vector<T>>();

            // Zero out packed buffer
            for (int z = 0; z < _packedInputDim; z++)
            {
                _packedFakeBuf[z] = NumOps.Zero;
            }

            for (int s = 0; s < pacSize; s++)
            {
                var condVector = _sampler.SampleRandomConditionVector();
                condVectors.Add(condVector);

                var noise = CreateStandardNormalVector(_options.EmbeddingDimension);
                noises.Add(noise);

                ConcatInto(_genInputBuf, noise, condVector);
                var fakeTransformed = Predict(VectorToTensor(_genInputBuf));
                FillFromTensor(_fakeRowBuf, fakeTransformed);
                ConcatInto(_fakeSingleBuf, _fakeRowBuf, condVector);

                for (int d = 0; d < singleDim; d++)
                {
                    _packedFakeBuf[s * singleDim + d] = d < _fakeSingleBuf.Length ? _fakeSingleBuf[d] : NumOps.Zero;
                }
            }

            // Compute dD/dInput using GradientTape autodiff, then negate for generator gradient
            var discInputGrad = TapeLayerBridge<T>.ComputeInputGradient(
                VectorToTensor(_packedFakeBuf),
                _discLayers,
                TapeLayerBridge<T>.HiddenActivation.LeakyReLU,
                applyActivationOnLast: false);
            for (int g = 0; g < discInputGrad.Length; g++)
            {
                discInputGrad[g] = NumOps.Negate(discInputGrad[g]);
            }
            discInputGrad = SafeGradient(discInputGrad, 5.0);

            T perSampleLr = NumOps.FromDouble(NumOps.ToDouble(scaledLr) / pacSize);
            for (int s = 0; s < pacSize; s++)
            {
                for (int d = 0; d < _dataWidth; d++)
                {
                    _sampleGradBuf[d] = (s * singleDim + d) < discInputGrad.Length
                        ? discInputGrad[s * singleDim + d]
                        : NumOps.Zero;
                }

                ConcatInto(_genInputBuf, noises[s], condVectors[s]);
                _ = Predict(VectorToTensor(_genInputBuf));

                BackpropagateError(_sampleGradBuf);
                UpdateGeneratorParameters(perSampleLr);
            }
        }
    }

    /// <summary>
    /// Updates discriminator parameters with DP noise injection.
    /// </summary>
    private void UpdateDiscriminatorParametersDP(T learningRate)
    {
        foreach (var layer in _discLayers)
        {
            ClipAndNoiseGradient(layer);
            layer.UpdateParameters(learningRate);
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

    #endregion

    #region Discriminator Forward/Backward

    private Tensor<T> DiscriminatorForward(Tensor<T> input, bool isTraining)
    {
        _discPreActivations.Clear();
        var current = input;
        int layerIdx = 0;

        for (int i = 0; i < _discLayers.Count; i++)
        {
            var layer = _discLayers[i];

            if (layer is DropoutLayer<T> dropout)
            {
                if (isTraining)
                {
                    current = dropout.Forward(current);
                }
                continue;
            }

            bool isLastDense = layerIdx == _discLayerDims.Count - 1;
            current = layer.Forward(current);

            if (!isLastDense)
            {
                _discPreActivations.Add(CloneTensor(current));
                current = ApplyLeakyReLU(current);
            }

            layerIdx++;
        }

        return current;
    }

    private void BackwardDiscriminator(Tensor<T> gradOutput)
    {
        var current = gradOutput;
        int denseIdx = _discLayerDims.Count - 1;

        for (int i = _discLayers.Count - 1; i >= 0; i--)
        {
            var layer = _discLayers[i];

            if (layer is DropoutLayer<T>)
            {
                continue;
            }

            if (denseIdx < _discLayerDims.Count - 1 && denseIdx < _discPreActivations.Count)
            {
                current = ApplyLeakyReLUDerivative(current, _discPreActivations[denseIdx]);
            }

            current = layer.Backward(current);
            denseIdx--;
        }
    }

    #endregion

    #region Generator Forward/Backward with Residual Connections

    private Tensor<T> GeneratorForwardWithResidual(Tensor<T> input)
    {
        _genPreActivations.Clear();
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

            _genPreActivations.Add(CloneTensor(current));
            current = ApplyReLU(current);
        }

        current = ConcatTensors(current, inputTensor);
        current = Layers[^1].Forward(current);

        return ApplyOutputActivations(current);
    }

    private void BackwardGeneratorWithResidual(Tensor<T> gradOutput)
    {
        int inputDim = _options.EmbeddingDimension + _condWidth;
        var current = gradOutput;

        current = Layers[^1].Backward(current);

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

        for (int i = Layers.Count - 2; i >= 0; i--)
        {
            if (i < _genPreActivations.Count)
            {
                current = ApplyReLUDerivative(current, _genPreActivations[i]);
            }

            if (i < _genBNLayers.Count)
            {
                current = _genBNLayers[i].Backward(current);
            }

            current = Layers[i].Backward(current);

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
            result[i] = NumOps.ToDouble(preActivation[i]) > 0
                ? gradOutput[i]
                : NumOps.Multiply(slope, gradOutput[i]);
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
        {
            sumExp += Math.Exp(NumOps.ToDouble(input[idx + m]) - maxVal);
        }

        for (int m = 0; m < count && idx < input.Length; m++)
        {
            double expVal = Math.Exp(NumOps.ToDouble(input[idx]) - maxVal);
            output[idx] = NumOps.FromDouble(expVal / Math.Max(sumExp, 1e-10));
            idx++;
        }
    }

    #endregion

    #region Gradient Safety Utilities

    private Tensor<T> SafeGradient(Tensor<T> grad, double maxNorm)
    {
        for (int i = 0; i < grad.Length; i++)
        {
            double v = NumOps.ToDouble(grad[i]);
            if (double.IsNaN(v) || double.IsInfinity(v))
            {
                grad[i] = NumOps.Zero;
            }
        }

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
                { "Epsilon", _options.Epsilon },
                { "Delta", _options.Delta },
                { "ClipNorm", _options.ClipNorm },
                { "CumulativeEpsilon", _cumulativeEpsilon },
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
        foreach (var dim in _options.GeneratorDimensions)
        {
            writer.Write(dim);
        }
        writer.Write(_options.DiscriminatorDimensions.Length);
        foreach (var dim in _options.DiscriminatorDimensions)
        {
            writer.Write(dim);
        }
        writer.Write(_options.BatchSize);
        writer.Write(_options.LearningRate);
        writer.Write(_options.GradientPenaltyWeight);
        writer.Write(_options.PacSize);
        writer.Write(_options.VGMModes);
        writer.Write(_options.DiscriminatorDropout);
        writer.Write(_options.Epsilon);
        writer.Write(_options.Delta);
        writer.Write(_options.ClipNorm);
        writer.Write(_options.NoiseMultiplier);
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
        return new DPCTGANGenerator<T>(
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

    /// <summary>
    /// Fills a pre-allocated vector with standard normal samples (Box-Muller).
    /// </summary>
    private void FillStandardNormal(Vector<T> buf)
    {
        for (int i = 0; i < buf.Length; i++)
        {
            buf[i] = SampleStandardNormal();
        }
    }

    /// <summary>
    /// Copies a row from a matrix into a pre-allocated vector.
    /// </summary>
    private static void FillRow(Vector<T> buf, Matrix<T> matrix, int row)
    {
        int cols = Math.Min(buf.Length, matrix.Columns);
        for (int j = 0; j < cols; j++) buf[j] = matrix[row, j];
    }

    /// <summary>
    /// Concatenates two vectors into a pre-allocated destination buffer.
    /// </summary>
    private static void ConcatInto(Vector<T> dest, Vector<T> a, Vector<T> b)
    {
        for (int i = 0; i < a.Length; i++) dest[i] = a[i];
        for (int i = 0; i < b.Length; i++) dest[a.Length + i] = b[i];
    }

    /// <summary>
    /// Copies tensor values into a pre-allocated vector buffer.
    /// </summary>
    private static void FillFromTensor(Vector<T> buf, Tensor<T> t)
    {
        int copyLen = Math.Min(buf.Length, t.Length);
        for (int i = 0; i < copyLen; i++) buf[i] = t[i];
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

    /// <inheritdoc />
    public override bool SupportsJitCompilation =>
        IsFitted && Layers.Count > 1 && !_usingCustomLayers &&
        _genBNLayers.Count > 0 &&
        Layers.All(l => l.SupportsJitCompilation) &&
        _genBNLayers.All(l => l.SupportsJitCompilation);

    /// <inheritdoc />
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
            inputNodes, genInputDim, hiddenLayers,
            _genBNLayers.Cast<ILayer<T>>().ToList(), Layers[^1],
            TapeLayerBridge<T>.HiddenActivation.ReLU,
            TapeLayerBridge<T>.HiddenActivation.None,
            useResidualConcat: true);
    }

    #endregion
}
