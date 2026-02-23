using AiDotNet.Autodiff;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks.SyntheticData;
using AiDotNet.Optimizers;
using AiDotNet.Tensors.Helpers;

namespace AiDotNet.NeuralNetworks.SyntheticData;

/// <summary>
/// CopulaGAN generator for synthetic tabular data, combining Gaussian copula transformations
/// with the CTGAN training pipeline for improved continuous column modeling.
/// </summary>
/// <remarks>
/// <para>
/// CopulaGAN applies a two-stage preprocessing to continuous columns:
/// 1. <b>Gaussian copula transform</b>: CDF then quantile then standard normal
/// 2. <b>VGM normalization</b>: Standard CTGAN mode-specific normalization
///
/// This implementation follows the standard neural network architecture pattern with:
/// - Proper inheritance from NeuralNetworkBase
/// - Layer-based architecture using ILayer components
/// - Engine-based tensor operations for CPU/GPU acceleration
/// - Full autodiff and JIT compilation support
/// </para>
/// <para>
/// <b>For Beginners:</b> CopulaGAN is CTGAN with an extra "normalizing" step:
///
/// <b>Training:</b>
/// <code>
/// Raw data -> [Copula: reshape to bell curve] -> [VGM: CTGAN preprocessing] -> [CTGAN training]
/// </code>
///
/// <b>Generation:</b>
/// <code>
/// Random noise -> [Generator] -> [Inverse VGM] -> [Inverse Copula] -> Synthetic data
/// </code>
///
/// If you provide custom layers in the architecture, those will be used directly
/// for the generator network. If not, the network creates industry-standard
/// CopulaGAN layers based on the original research paper specifications.
///
/// Example usage:
/// <code>
/// var architecture = new NeuralNetworkArchitecture&lt;double&gt;(
///     inputFeatures: 10,
///     outputSize: 10
/// );
/// var options = new CopulaGANOptions&lt;double&gt;
/// {
///     EmbeddingDimension = 128,
///     GeneratorDimensions = new[] { 256, 256 },
///     BatchSize = 500,
///     Epochs = 300
/// };
/// var generator = new CopulaGANGenerator&lt;double&gt;(architecture, options);
/// generator.Fit(data, columns, epochs: 300);
/// var synthetic = generator.Generate(1000);
/// </code>
/// </para>
/// <para>
/// Reference: "Synthesizing Tabular Data using Copulas" (2020)
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public class CopulaGANGenerator<T> : NeuralNetworkBase<T>, ISyntheticTabularGenerator<T>
{
    private readonly CopulaGANOptions<T> _options;
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

    // Copula transform state: sorted values and empirical CDF per continuous column
    private readonly List<double[]> _sortedValues = new();
    private readonly List<int> _continuousColumnIndices = new();

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
    /// Gets the CopulaGAN-specific options.
    /// </summary>
    public new CopulaGANOptions<T> Options => _options;

    /// <inheritdoc />
    public IReadOnlyList<ColumnMetadata> Columns => _columns.AsReadOnly();

    /// <inheritdoc />
    public bool IsFitted { get; private set; }

    /// <summary>
    /// Initializes a new CopulaGAN generator with the specified architecture.
    /// </summary>
    /// <param name="architecture">The neural network architecture defining input/output dimensions and optional custom layers.</param>
    /// <param name="options">CopulaGAN-specific options for generator and discriminator configuration.</param>
    /// <param name="optimizer">Gradient-based optimizer (defaults to Adam).</param>
    /// <param name="lossFunction">Loss function (defaults based on task type).</param>
    /// <param name="maxGradNorm">Maximum gradient norm for clipping (default 5.0).</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This constructor creates a CopulaGAN network based on the architecture you provide.
    ///
    /// If you provide custom layers in the architecture, those will be used directly
    /// for the generator network. If not, the network will create industry-standard
    /// CopulaGAN generator layers based on the original research paper specifications.
    ///
    /// Example usage:
    /// <code>
    /// var architecture = new NeuralNetworkArchitecture&lt;double&gt;(
    ///     inputFeatures: 10,
    ///     outputSize: 10
    /// );
    /// var options = new CopulaGANOptions&lt;double&gt; { EmbeddingDimension = 128 };
    /// var generator = new CopulaGANGenerator&lt;double&gt;(architecture, options);
    /// </code>
    /// </para>
    /// </remarks>
    public CopulaGANGenerator(
        NeuralNetworkArchitecture<T> architecture,
        CopulaGANOptions<T>? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
        ILossFunction<T>? lossFunction = null,
        double maxGradNorm = 5.0)
        : base(architecture, lossFunction ?? NeuralNetworkHelper<T>.GetDefaultLossFunction(architecture.TaskType), maxGradNorm)
    {
        _options = options ?? new CopulaGANOptions<T>();
        _lossFunction = lossFunction ?? NeuralNetworkHelper<T>.GetDefaultLossFunction(architecture.TaskType);
        _optimizer = optimizer ?? new AdamOptimizer<T, Tensor<T>, Tensor<T>>(this);
        _random = _options.Seed.HasValue
            ? RandomHelper.CreateSeededRandom(_options.Seed.Value)
            : RandomHelper.CreateSecureRandom();

        InitializeLayers();
    }

    #region Layer Initialization (GANDALF Pattern)

    /// <summary>
    /// Initializes the layers of the CopulaGAN network based on the provided architecture.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This method either uses custom layers provided in the architecture or creates
    /// default CopulaGAN generator layers following the original paper specifications.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> This method sets up the generator network structure:
    /// - If you provided custom layers, those are used for the generator
    /// - Otherwise, it creates the standard CopulaGAN generator architecture:
    ///   Dense layers with batch normalization and residual connections
    ///
    /// The discriminator is always created internally and is not user-overridable.
    /// </para>
    /// </remarks>
    protected override void InitializeLayers()
    {
        if (Architecture.Layers != null && Architecture.Layers.Count > 0)
        {
            // Use the layers provided by the user for the generator
            Layers.AddRange(Architecture.Layers);
            ValidateCustomLayers(Layers);
            _usingCustomLayers = true;
        }
        else
        {
            // Create default generator layers
            // Input = EmbeddingDimension (noise) + estimated conditional width
            int inputDim = _options.EmbeddingDimension + Architecture.CalculatedInputSize;
            int outputDim = Architecture.OutputSize;
            Layers.AddRange(LayerHelper<T>.CreateDefaultCopulaGANGeneratorLayers(
                inputDim, outputDim, _options.GeneratorDimensions));

            // Create matching batch normalization layers (one per hidden layer)
            _genBNLayers.Clear();
            foreach (int dim in _options.GeneratorDimensions)
            {
                _genBNLayers.Add(new BatchNormalizationLayer<T>(dim));
            }
            _usingCustomLayers = false;
        }
    }

    /// <summary>
    /// Rebuilds the default generator and discriminator layers with actual data dimensions
    /// discovered during Fit(). Only called when not using custom layers.
    /// </summary>
    /// <param name="genInputDim">Actual generator input dimension (embedding + conditional width).</param>
    /// <param name="genOutputDim">Actual generator output dimension (transformed data width).</param>
    /// <param name="discInputDim">Actual discriminator input dimension (packed data).</param>
    private void RebuildLayersWithActualDimensions(int genInputDim, int genOutputDim, int discInputDim)
    {
        if (!_usingCustomLayers)
        {
            Layers.Clear();
            Layers.AddRange(LayerHelper<T>.CreateDefaultCopulaGANGeneratorLayers(
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
        _discLayers.AddRange(LayerHelper<T>.CreateDefaultCopulaGANDiscriminatorLayers(
            discInputDim, _options.DiscriminatorDimensions, _options.DiscriminatorDropout));

        // Build dimension map for manual gradient computation
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

    /// <summary>
    /// Runs the generator forward pass to create synthetic data from a noise input.
    /// </summary>
    /// <param name="input">The input tensor (noise vector + conditional vector).</param>
    /// <returns>The generated synthetic data tensor.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This takes random noise and transforms it into a fake data row.
    /// The generator uses residual connections where each hidden layer receives both the
    /// previous output and the original input concatenated together.
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

        // Default generator with residual connections + BN + ReLU
        return GeneratorForwardWithResidual(input);
    }

    /// <summary>
    /// Trains the CopulaGAN network using the provided input and expected output.
    /// </summary>
    /// <param name="input">The input noise tensor.</param>
    /// <param name="expectedOutput">The expected real data tensor.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This runs a single training step for the generator.
    /// The full GAN training loop (alternating discriminator/generator) happens in Fit().
    /// This method is provided for compatibility with the NeuralNetworkBase pattern.
    /// </para>
    /// </remarks>
    public override void Train(Tensor<T> input, Tensor<T> expectedOutput)
    {
        // Forward pass through generator
        Tensor<T> prediction = Predict(input);

        // Calculate loss
        LastLoss = _lossFunction.CalculateLoss(prediction.ToVector(), expectedOutput.ToVector());

        // Calculate error gradient
        Tensor<T> error = prediction.Subtract(expectedOutput);

        // Backpropagate error through generator
        BackpropagateError(error);

        // Update generator parameters
        UpdateNetworkParameters();
    }

    /// <summary>
    /// Backpropagates the error through the generator layers.
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

        // Default generator backward with residual connections
        BackwardGeneratorWithResidual(error);
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
    /// Fits the CopulaGAN generator to the provided real tabular data.
    /// </summary>
    /// <param name="data">The real data matrix where each row is a sample and each column is a feature.</param>
    /// <param name="columns">Metadata describing each column (type, categories, etc.).</param>
    /// <param name="epochs">Number of training epochs.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This is the "learning" step. The generator studies your real data:
    /// 1. Applies Gaussian copula transform to normalize continuous columns
    /// 2. Fits VGM transformer for mode-specific normalization
    /// 3. Trains generator and discriminator in alternating WGAN-GP steps
    /// After fitting, call Generate() to create new synthetic rows.
    /// </para>
    /// </remarks>
    public void Fit(Matrix<T> data, IReadOnlyList<ColumnMetadata> columns, int epochs)
    {
        ValidateFitInputs(data, columns, epochs);

        _columns = PrepareColumns(data, columns);

        // Step 1: Identify continuous columns and apply Gaussian copula transform
        _continuousColumnIndices.Clear();
        _sortedValues.Clear();
        for (int col = 0; col < columns.Count; col++)
        {
            if (columns[col].IsNumerical)
            {
                _continuousColumnIndices.Add(col);
            }
        }
        var copulaData = ApplyCopulaTransform(data);

        // Step 2: Fit the VGM transformer on copula-transformed data
        _transformer = new TabularDataTransformer<T>(_options.VGMModes, _random);
        _transformer.Fit(copulaData, _columns);
        _dataWidth = _transformer.TransformedWidth;

        // Step 3: Fit the data sampler
        _sampler = new CTGANDataSampler<T>(_random);
        _sampler.Fit(copulaData, _columns);
        _condWidth = _sampler.ConditionalVectorWidth;

        // Step 4: Compute packed input dimension
        _packedInputDim = (_dataWidth + _condWidth) * _options.PacSize;

        // Step 5: Rebuild layers with actual data dimensions
        int genInputDim = _options.EmbeddingDimension + _condWidth;
        RebuildLayersWithActualDimensions(genInputDim, _dataWidth, _packedInputDim);

        // Step 6: Transform data
        var transformedData = _transformer.Transform(copulaData);

        // Step 7: Training loop
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

            _continuousColumnIndices.Clear();
            _sortedValues.Clear();
            for (int col = 0; col < columns.Count; col++)
            {
                if (columns[col].IsNumerical)
                {
                    _continuousColumnIndices.Add(col);
                }
            }
            var copulaData = ApplyCopulaTransform(data);

            _transformer = new TabularDataTransformer<T>(_options.VGMModes, _random);
            _transformer.Fit(copulaData, _columns);
            _dataWidth = _transformer.TransformedWidth;

            _sampler = new CTGANDataSampler<T>(_random);
            _sampler.Fit(copulaData, _columns);
            _condWidth = _sampler.ConditionalVectorWidth;

            _packedInputDim = (_dataWidth + _condWidth) * _options.PacSize;

            int genInputDim = _options.EmbeddingDimension + _condWidth;
            RebuildLayersWithActualDimensions(genInputDim, _dataWidth, _packedInputDim);

            // Pre-allocate training buffers
            InitializeTrainingBuffers(genInputDim);

            var transformedData = _transformer.Transform(copulaData);

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
    /// <param name="numSamples">The number of synthetic rows to generate.</param>
    /// <param name="conditionColumn">Optional conditioning column indices.</param>
    /// <param name="conditionValue">Optional conditioning values.</param>
    /// <returns>A matrix of synthetic data with the same column structure as the training data.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> After fitting, this creates new fake-but-realistic rows by:
    /// 1. Generating random noise
    /// 2. Running through the generator to create transformed data
    /// 3. Applying inverse VGM transform
    /// 4. Applying inverse copula transform to restore original distributions
    /// </para>
    /// </remarks>
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

        // Inverse VGM transform
        var copulaSpace = _transformer.InverseTransform(transformedRows);

        // Inverse copula transform
        return ApplyInverseCopulaTransform(copulaSpace);
    }

    #endregion

    #region GAN Training Steps

    #region Training Buffer Management

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

    private void FillStandardNormal(Vector<T> buffer)
    {
        for (int i = 0; i < buffer.Length; i++)
        {
            double u1 = 1.0 - _random.NextDouble();
            double u2 = _random.NextDouble();
            buffer[i] = NumOps.FromDouble(Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2));
        }
    }

    private static void FillRow(Matrix<T> data, int row, Vector<T> buffer)
    {
        int cols = Math.Min(data.Columns, buffer.Length);
        for (int j = 0; j < cols; j++) buffer[j] = data[row, j];
    }

    private static void ConcatInto(Vector<T> a, Vector<T> b, Vector<T> dest)
    {
        int aLen = a.Length;
        for (int i = 0; i < aLen; i++) dest[i] = a[i];
        int bLen = Math.Min(b.Length, dest.Length - aLen);
        for (int i = 0; i < bLen; i++) dest[aLen + i] = b[i];
    }

    private static void FillFromTensor(Tensor<T> src, Vector<T> dest)
    {
        int len = Math.Min(src.Length, dest.Length);
        for (int i = 0; i < len; i++) dest[i] = src[i];
    }

    #endregion

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

            _ = DiscriminatorForward(VectorToTensor(_packedFakeBuf), isTraining: true);
            BackwardDiscriminator(_oneGrad);
            UpdateDiscriminatorParameters(scaledLr);

            _ = DiscriminatorForward(VectorToTensor(_packedRealBuf), isTraining: true);
            BackwardDiscriminator(_negOneGrad);
            UpdateDiscriminatorParameters(scaledLr);

            ApplyGradientPenalty(_packedRealBuf, _packedFakeBuf, scaledLr);
        }
    }

    /// <summary>
    /// Trains the generator for one step using WGAN-GP objective.
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
            var noises = new List<Vector<T>>(pacSize);
            var condVectors = new List<Vector<T>>(pacSize);
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

            var discInputGrad = TapeLayerBridge<T>.ComputeInputGradient(
                VectorToTensor(_packedFakeBuf),
                _discLayers,
                TapeLayerBridge<T>.HiddenActivation.LeakyReLU,
                applyActivationOnLast: false);
            for (int g = 0; g < discInputGrad.Length; g++)
                discInputGrad[g] = NumOps.Negate(discInputGrad[g]);
            discInputGrad = SafeGradient(discInputGrad, 5.0);

            T perSampleLr = NumOps.FromDouble(NumOps.ToDouble(scaledLr) / pacSize);
            for (int s = 0; s < pacSize; s++)
            {
                for (int d = 0; d < _dataWidth; d++)
                    _sampleGradBuf[d] = NumOps.Zero;
                for (int d = 0; d < _dataWidth && (s * singleDim + d) < discInputGrad.Length; d++)
                    _sampleGradBuf[d] = discInputGrad[s * singleDim + d];

                ConcatInto(noises[s], condVectors[s], _genInputBuf);
                _ = Predict(VectorToTensor(_genInputBuf));

                BackpropagateError(_sampleGradBuf);
                UpdateGeneratorParameters(perSampleLr);
            }
        }
    }

    #endregion

    #region Discriminator Forward/Backward

    /// <summary>
    /// Runs the discriminator forward pass.
    /// </summary>
    private Tensor<T> DiscriminatorForward(Tensor<T> input, bool isTraining)
    {
        _discPreActivations.Clear();
        var current = input;
        int layerIdx = 0;

        for (int i = 0; i < _discLayers.Count; i++)
        {
            var layer = _discLayers[i];

            // Skip dropout layers in this loop - they're interleaved
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

    /// <summary>
    /// Runs the discriminator backward pass.
    /// </summary>
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

    /// <summary>
    /// Updates discriminator parameters with a given learning rate.
    /// </summary>
    private void UpdateDiscriminatorParameters(T learningRate)
    {
        foreach (var layer in _discLayers)
        {
            layer.UpdateParameters(learningRate);
        }
    }

    /// <summary>
    /// Updates generator parameters with a given learning rate.
    /// </summary>
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

    #region Gradient Penalty

    /// <summary>
    /// Applies WGAN-GP gradient penalty to the discriminator.
    /// </summary>
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

    #region Generator Forward/Backward with Residual Connections

    /// <summary>
    /// Generator forward pass with CTGAN-style residual connections:
    /// Each hidden layer receives [previous_output, original_input].
    /// </summary>
    private Tensor<T> GeneratorForwardWithResidual(Tensor<T> input)
    {
        _genPreActivations.Clear();
        var inputTensor = input;
        var current = input;

        // Hidden layers with residual connections + BN + ReLU
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

        // Output layer with residual connection
        current = ConcatTensors(current, inputTensor);
        current = Layers[^1].Forward(current);

        return ApplyOutputActivations(current);
    }

    /// <summary>
    /// Generator backward pass with residual connection handling.
    /// </summary>
    private void BackwardGeneratorWithResidual(Tensor<T> gradOutput)
    {
        int inputDim = _options.EmbeddingDimension + _condWidth;
        var current = gradOutput;

        // Backward through output layer
        current = Layers[^1].Backward(current);

        // Extract hidden part of gradient (skip the input residual part)
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

            // Extract hidden part for next layer
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

    #region Copula Transform

    /// <summary>
    /// Applies Gaussian copula transform to continuous columns:
    /// value -> empirical CDF -> inverse normal CDF (quantile function).
    /// </summary>
    private Matrix<T> ApplyCopulaTransform(Matrix<T> data)
    {
        var result = new Matrix<T>(data.Rows, data.Columns);

        for (int i = 0; i < data.Rows; i++)
        {
            for (int j = 0; j < data.Columns; j++)
            {
                result[i, j] = data[i, j];
            }
        }

        foreach (int col in _continuousColumnIndices)
        {
            var values = new double[data.Rows];
            for (int i = 0; i < data.Rows; i++)
            {
                values[i] = NumOps.ToDouble(data[i, col]);
            }

            var sorted = new double[data.Rows];
            Array.Copy(values, sorted, data.Rows);
            Array.Sort(sorted);
            _sortedValues.Add(sorted);

            for (int i = 0; i < data.Rows; i++)
            {
                int rank = Array.BinarySearch(sorted, values[i]);
                if (rank < 0) rank = ~rank;
                int lo = rank;
                while (lo > 0 && sorted[lo - 1] == values[i]) lo--;
                int hi = rank;
                while (hi < sorted.Length - 1 && sorted[hi + 1] == values[i]) hi++;
                double midRank = (lo + hi) / 2.0;

                double u = (midRank + 1.0) / (data.Rows + 1.0);
                u = Math.Max(1e-6, Math.Min(1.0 - 1e-6, u));
                double z = NormalQuantile(u);
                result[i, col] = NumOps.FromDouble(z);
            }
        }

        return result;
    }

    /// <summary>
    /// Applies inverse Gaussian copula transform to continuous columns.
    /// </summary>
    private Matrix<T> ApplyInverseCopulaTransform(Matrix<T> data)
    {
        var result = new Matrix<T>(data.Rows, data.Columns);

        for (int i = 0; i < data.Rows; i++)
        {
            for (int j = 0; j < data.Columns; j++)
            {
                result[i, j] = data[i, j];
            }
        }

        for (int idx = 0; idx < _continuousColumnIndices.Count && idx < _sortedValues.Count; idx++)
        {
            int col = _continuousColumnIndices[idx];
            var sorted = _sortedValues[idx];
            int n = sorted.Length;

            for (int i = 0; i < data.Rows; i++)
            {
                double z = NumOps.ToDouble(data[i, col]);
                double u = NormalCDF(z);
                u = Math.Max(0, Math.Min(1, u));

                double rank = u * (n + 1.0) - 1.0;
                rank = Math.Max(0, Math.Min(n - 1, rank));
                int lo = (int)Math.Floor(rank);
                int hi = Math.Min(lo + 1, n - 1);
                double frac = rank - lo;

                double original = sorted[lo] * (1.0 - frac) + sorted[hi] * frac;
                result[i, col] = NumOps.FromDouble(original);
            }
        }

        return result;
    }

    /// <summary>
    /// Approximation of the inverse normal CDF using Beasley-Springer-Moro algorithm.
    /// </summary>
    private static double NormalQuantile(double p)
    {
        if (p <= 0) return -8.0;
        if (p >= 1) return 8.0;

        double t;
        if (p < 0.5)
        {
            t = Math.Sqrt(-2.0 * Math.Log(p));
            return -(2.515517 + t * (0.802853 + t * 0.010328))
                   / (1.0 + t * (1.432788 + t * (0.189269 + t * 0.001308)));
        }
        else
        {
            t = Math.Sqrt(-2.0 * Math.Log(1.0 - p));
            return (2.515517 + t * (0.802853 + t * 0.010328))
                   / (1.0 + t * (1.432788 + t * (0.189269 + t * 0.001308)));
        }
    }

    /// <summary>
    /// Approximation of the normal CDF using the error function.
    /// </summary>
    private static double NormalCDF(double x)
    {
        return 0.5 * (1.0 + Erf(x / Math.Sqrt(2.0)));
    }

    /// <summary>
    /// Approximation of the error function using Abramowitz and Stegun formula 7.1.26.
    /// </summary>
    private static double Erf(double x)
    {
        double sign = x >= 0 ? 1.0 : -1.0;
        x = Math.Abs(x);

        const double a1 = 0.254829592;
        const double a2 = -0.284496736;
        const double a3 = 1.421413741;
        const double a4 = -1.453152027;
        const double a5 = 1.061405429;
        const double p = 0.3275911;

        double t = 1.0 / (1.0 + p * x);
        double y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * Math.Exp(-x * x);

        return sign * y;
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

    /// <summary>
    /// Applies tanh to continuous columns and softmax to categorical/mode columns.
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

    /// <summary>
    /// Applies NaN sanitization and gradient norm clipping in a single operation.
    /// </summary>
    private Tensor<T> SafeGradient(Tensor<T> grad, double maxNorm)
    {
        // Sanitize NaN/Infinity
        for (int i = 0; i < grad.Length; i++)
        {
            double v = NumOps.ToDouble(grad[i]);
            if (double.IsNaN(v) || double.IsInfinity(v))
            {
                grad[i] = NumOps.Zero;
            }
        }

        // Clip gradient norm
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
        return new CopulaGANGenerator<T>(
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
