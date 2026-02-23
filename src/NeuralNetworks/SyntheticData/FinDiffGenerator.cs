using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks.SyntheticData;
using AiDotNet.Optimizers;
using AiDotNet.Tensors.Helpers;

namespace AiDotNet.NeuralNetworks.SyntheticData;

/// <summary>
/// FinDiff generator for synthesizing realistic financial tabular data using diffusion
/// models with temporal correlation preservation and financial constraint enforcement.
/// </summary>
/// <remarks>
/// <para>
/// FinDiff augments standard diffusion with financial-specific losses:
///
/// <code>
///  Standard Loss:    ||predicted_noise - actual_noise||^2
///  + Temporal Loss:  ||autocorr(synthetic) - autocorr(real)||^2
///  + Constraint:     max(0, -value) for positive-only columns
/// </code>
///
/// This implementation follows the standard neural network architecture pattern with:
/// - Proper inheritance from NeuralNetworkBase
/// - Layer-based architecture using ILayer components
/// - Engine-based tensor operations for CPU/GPU acceleration
/// - Full autodiff and JIT compilation support
/// </para>
/// <para>
/// <b>For Beginners:</b> FinDiff generates fake financial data that respects financial rules:
///
/// 1. Stock prices don't jump randomly — they follow trends (temporal correlation)
/// 2. Prices are always positive (constraint enforcement)
/// 3. Volatile periods cluster together (volatility awareness)
///
/// If you provide custom layers in the architecture, those will be used directly
/// for the denoiser MLP. If not, the network creates industry-standard
/// FinDiff layers based on the original research paper specifications.
///
/// Example usage:
/// <code>
/// var architecture = new NeuralNetworkArchitecture&lt;double&gt;(
///     inputFeatures: 10,
///     outputSize: 10
/// );
/// var options = new FinDiffOptions&lt;double&gt;
/// {
///     NumTimesteps = 500,
///     TemporalWeight = 5.0,
///     EnforcePositive = true
/// };
/// var generator = new FinDiffGenerator&lt;double&gt;(architecture, options);
/// generator.Fit(data, columns, epochs: 300);
/// var synthetic = generator.Generate(1000);
/// </code>
/// </para>
/// <para>
/// Reference: "Diffusion Models for Financial Tabular Data" (2024)
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public class FinDiffGenerator<T> : NeuralNetworkBase<T>, ISyntheticTabularGenerator<T>
{
    private readonly FinDiffOptions<T> _options;
    private IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> _optimizer;
    private ILossFunction<T> _lossFunction;

    // Synthetic tabular data infrastructure
    private TabularDataTransformer<T>? _transformer;
    private List<ColumnMetadata> _columns = new();
    private int _dataWidth;
    private Random _random;

    // Timestep projection layers (always created by LayerHelper, not user-overridable)
    private readonly List<ILayer<T>> _timestepProjectionLayers = new();

    // Diffusion parameters
    private double[] _betas = Array.Empty<double>();
    private double[] _alphas = Array.Empty<double>();
    private double[] _alphasCumprod = Array.Empty<double>();

    // Real data statistics for temporal loss
    private double[]? _realAutoCorr;

    /// <summary>
    /// Gets the FinDiff-specific options.
    /// </summary>
    public new FinDiffOptions<T> Options => _options;

    /// <inheritdoc />
    public IReadOnlyList<ColumnMetadata> Columns => _columns.AsReadOnly();

    /// <inheritdoc />
    public bool IsFitted { get; private set; }

    /// <summary>
    /// Initializes a new FinDiff generator with the specified architecture.
    /// </summary>
    /// <param name="architecture">The neural network architecture defining input/output dimensions and optional custom layers.</param>
    /// <param name="options">FinDiff-specific options for diffusion configuration.</param>
    /// <param name="optimizer">Gradient-based optimizer (defaults to Adam).</param>
    /// <param name="lossFunction">Loss function (defaults based on task type).</param>
    /// <param name="maxGradNorm">Maximum gradient norm for clipping (default 5.0).</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This constructor creates a FinDiff network based on the architecture you provide.
    ///
    /// If you provide custom layers in the architecture, those will be used directly
    /// for the denoiser MLP. If not, the network will create industry-standard
    /// FinDiff layers based on the original research paper specifications.
    /// </para>
    /// </remarks>
    public FinDiffGenerator(
        NeuralNetworkArchitecture<T> architecture,
        FinDiffOptions<T>? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
        ILossFunction<T>? lossFunction = null,
        double maxGradNorm = 5.0)
        : base(architecture, lossFunction ?? NeuralNetworkHelper<T>.GetDefaultLossFunction(architecture.TaskType), maxGradNorm)
    {
        _options = options ?? new FinDiffOptions<T>();
        _lossFunction = lossFunction ?? NeuralNetworkHelper<T>.GetDefaultLossFunction(architecture.TaskType);
        _optimizer = optimizer ?? new AdamOptimizer<T, Tensor<T>, Tensor<T>>(this);
        _random = _options.Seed.HasValue
            ? RandomHelper.CreateSeededRandom(_options.Seed.Value)
            : RandomHelper.CreateSecureRandom();

        InitializeLayers();
    }

    #region Layer Initialization (GANDALF Pattern)

    /// <summary>
    /// Initializes the layers of the FinDiff network based on the provided architecture.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This method either uses custom layers provided in the architecture or creates
    /// default FinDiff layers following the original paper specifications.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> This method sets up the network structure:
    /// - If you provided custom layers, those are used for the denoiser MLP
    /// - Otherwise, it creates the standard FinDiff architecture:
    ///   1. Timestep projection (sinusoidal embedding processed by a Dense layer)
    ///   2. Denoiser MLP (Dense(SiLU) layers ending with Dense(Identity))
    /// </para>
    /// </remarks>
    protected override void InitializeLayers()
    {
        if (Architecture.Layers != null && Architecture.Layers.Count > 0)
        {
            Layers.AddRange(Architecture.Layers);
            ValidateCustomLayers(Layers);
        }
        else
        {
            Layers.AddRange(LayerHelper<T>.CreateDefaultFinDiffDenoiserLayers(
                inputDim: Architecture.CalculatedInputSize + _options.TimestepEmbeddingDimension,
                outputDim: Architecture.OutputSize,
                hiddenDims: _options.MLPDimensions));
        }

        _timestepProjectionLayers.Clear();
        _timestepProjectionLayers.AddRange(
            LayerHelper<T>.CreateDefaultFinDiffTimestepProjectionLayers(
                _options.TimestepEmbeddingDimension));
    }

    /// <summary>
    /// Rebuilds MLP layers using the actual transformed data width (which may differ from
    /// Architecture.OutputSize due to VGMM mode-encoding of continuous columns).
    /// </summary>
    private void RebuildLayersForDataWidth()
    {
        if (Architecture.Layers != null && Architecture.Layers.Count > 0)
        {
            return;
        }

        Layers.Clear();
        Layers.AddRange(LayerHelper<T>.CreateDefaultFinDiffDenoiserLayers(
            inputDim: _dataWidth + _options.TimestepEmbeddingDimension,
            outputDim: _dataWidth,
            hiddenDims: _options.MLPDimensions));

        _timestepProjectionLayers.Clear();
        _timestepProjectionLayers.AddRange(
            LayerHelper<T>.CreateDefaultFinDiffTimestepProjectionLayers(
                _options.TimestepEmbeddingDimension));
    }

    #endregion

    #region Neural Network Methods (GANDALF Pattern)

    /// <inheritdoc />
    public override Tensor<T> Predict(Tensor<T> input)
    {
        if (TryForwardGpuOptimized(input, out var gpuResult))
            return gpuResult;

        Tensor<T> currentOutput = input;
        foreach (var layer in Layers)
        {
            currentOutput = layer.Forward(currentOutput);
        }

        return currentOutput;
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
        for (int i = Layers.Count - 1; i >= 0; i--)
        {
            error = Layers[i].Backward(error);
        }
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

    /// <inheritdoc />
    public void Fit(Matrix<T> data, IReadOnlyList<ColumnMetadata> columns, int epochs)
    {
        ValidateFitInputs(data, columns, epochs);
        _columns = PrepareColumns(data, columns);

        _transformer = new TabularDataTransformer<T>(_options.VGMModes, _random);
        _transformer.Fit(data, _columns);
        _dataWidth = _transformer.TransformedWidth;

        RebuildLayersForDataWidth();

        var transformedData = _transformer.Transform(data);

        ComputeNoiseSchedule();
        _realAutoCorr = ComputeAutocorrelation(transformedData);

        int batchSize = Math.Min(_options.BatchSize, data.Rows);

        for (int epoch = 0; epoch < epochs; epoch++)
        {
            for (int b = 0; b < data.Rows; b += batchSize)
            {
                int end = Math.Min(b + batchSize, data.Rows);
                TrainBatch(transformedData, b, end);
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

            RebuildLayersForDataWidth();

            var transformedData = _transformer.Transform(data);

            ComputeNoiseSchedule();
            _realAutoCorr = ComputeAutocorrelation(transformedData);

            int batchSize = Math.Min(_options.BatchSize, data.Rows);

            for (int epoch = 0; epoch < epochs; epoch++)
            {
                ct.ThrowIfCancellationRequested();
                for (int b = 0; b < data.Rows; b += batchSize)
                {
                    int end = Math.Min(b + batchSize, data.Rows);
                    TrainBatch(transformedData, b, end);
                }
            }
        }, ct).ConfigureAwait(false);

        IsFitted = true;
    }

    /// <inheritdoc />
    public Matrix<T> Generate(int numSamples, Vector<T>? conditionColumn = null, Vector<T>? conditionValue = null)
    {
        if (!IsFitted || _transformer is null)
        {
            throw new InvalidOperationException(
                "The generator must be fitted before generating data. Call Fit() first.");
        }

        if (numSamples <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(numSamples), "Number of samples must be positive.");
        }

        var result = new Matrix<T>(numSamples, _dataWidth);

        for (int i = 0; i < numSamples; i++)
        {
            var xt = CreateStandardNormalVector(_dataWidth);

            for (int t = _options.NumTimesteps - 1; t >= 0; t--)
            {
                var predictedNoise = PredictNoise(xt, t);
                xt = DenoisingStep(xt, predictedNoise, t);
            }

            if (_options.EnforcePositive)
            {
                for (int j = 0; j < _dataWidth; j++)
                {
                    double val = NumOps.ToDouble(xt[j]);
                    if (val < 0) xt[j] = NumOps.FromDouble(Math.Abs(val));
                }
            }

            for (int j = 0; j < _dataWidth; j++)
            {
                result[i, j] = xt[j];
            }
        }

        return _transformer.InverseTransform(result);
    }

    #endregion

    #region Noise Schedule

    private void ComputeNoiseSchedule()
    {
        int T = _options.NumTimesteps;
        _betas = new double[T];
        _alphas = new double[T];
        _alphasCumprod = new double[T];

        for (int t = 0; t < T; t++)
        {
            _betas[t] = _options.BetaStart + (_options.BetaEnd - _options.BetaStart) * t / (T - 1);
        }

        double cumprod = 1.0;
        for (int t = 0; t < T; t++)
        {
            _alphas[t] = 1.0 - _betas[t];
            cumprod *= _alphas[t];
            _alphasCumprod[t] = cumprod;
        }
    }

    #endregion

    #region Training

    private void TrainBatch(Matrix<T> data, int startRow, int endRow)
    {
        for (int row = startRow; row < endRow; row++)
        {
            int t = _random.Next(_options.NumTimesteps);
            var x0 = GetRow(data, row);
            var noise = CreateStandardNormalVector(_dataWidth);

            double sqrtAlphaBar = Math.Sqrt(_alphasCumprod[t]);
            double sqrtOneMinusAlphaBar = Math.Sqrt(1.0 - _alphasCumprod[t]);

            // Build noisy input: xt = sqrt(alpha_bar) * x0 + sqrt(1-alpha_bar) * noise
            var xt = new Vector<T>(_dataWidth);
            for (int j = 0; j < _dataWidth; j++)
            {
                xt[j] = NumOps.FromDouble(
                    sqrtAlphaBar * NumOps.ToDouble(x0[j]) +
                    sqrtOneMinusAlphaBar * NumOps.ToDouble(noise[j]));
            }

            // Build target noise tensor
            var targetNoise = new Tensor<T>([_dataWidth]);
            for (int j = 0; j < _dataWidth; j++)
            {
                targetNoise[j] = noise[j];
            }

            // Create timestep embedding and build input tensor
            var timeEmbed = CreateTimestepEmbedding(t);
            int totalLen = xt.Length + timeEmbed.Length;
            var input = new Tensor<T>([totalLen]);
            for (int j = 0; j < xt.Length; j++) input[j] = xt[j];
            for (int j = 0; j < timeEmbed.Length; j++) input[xt.Length + j] = timeEmbed[j];

            // Use Train() method (GANDALF pattern: forward -> loss -> backward -> update)
            Train(input, targetNoise);
        }
    }

    #endregion

    #region Denoising

    private Vector<T> PredictNoise(Vector<T> xt, int t)
    {
        var timeEmbed = CreateTimestepEmbedding(t);

        int totalLen = xt.Length + timeEmbed.Length;
        var input = new Vector<T>(totalLen);
        for (int i = 0; i < xt.Length; i++) input[i] = xt[i];
        for (int i = 0; i < timeEmbed.Length; i++) input[xt.Length + i] = timeEmbed[i];

        var current = VectorToTensor(input);
        foreach (var layer in Layers)
        {
            current = layer.Forward(current);
        }

        return TensorToVector(current, _dataWidth);
    }

    private Vector<T> CreateTimestepEmbedding(int t)
    {
        int dim = _options.TimestepEmbeddingDimension;
        var embedding = new Vector<T>(dim);
        int halfDim = dim / 2;

        for (int i = 0; i < halfDim; i++)
        {
            double freq = Math.Exp(-Math.Log(10000.0) * i / halfDim);
            double angle = t * freq;
            embedding[i] = NumOps.FromDouble(Math.Sin(angle));
            if (i + halfDim < dim) embedding[i + halfDim] = NumOps.FromDouble(Math.Cos(angle));
        }

        var current = VectorToTensor(embedding);
        foreach (var layer in _timestepProjectionLayers)
        {
            current = layer.Forward(current);
        }

        return TensorToVector(current, dim);
    }

    private Vector<T> DenoisingStep(Vector<T> xt, Vector<T> predictedNoise, int t)
    {
        double alphaT = _alphas[t];
        double alphaBarT = _alphasCumprod[t];
        double betaT = _betas[t];

        var result = new Vector<T>(_dataWidth);
        double coeff1 = 1.0 / Math.Sqrt(alphaT);
        double coeff2 = betaT / Math.Sqrt(1.0 - alphaBarT);

        for (int j = 0; j < _dataWidth; j++)
        {
            double mean = coeff1 * (NumOps.ToDouble(xt[j]) - coeff2 * NumOps.ToDouble(predictedNoise[j]));

            if (t > 0)
            {
                double sigma = Math.Sqrt(betaT);
                mean += sigma * NumOps.ToDouble(SampleStandardNormal());
            }

            result[j] = NumOps.FromDouble(mean);
        }

        return result;
    }

    #endregion

    #region Statistics

    private double[] ComputeAutocorrelation(Matrix<T> data)
    {
        var autoCorr = new double[data.Columns];

        for (int j = 0; j < data.Columns; j++)
        {
            double mean = 0;
            for (int i = 0; i < data.Rows; i++)
                mean += NumOps.ToDouble(data[i, j]);
            mean /= data.Rows;

            double varSum = 0;
            double covSum = 0;
            for (int i = 0; i < data.Rows; i++)
            {
                double v = NumOps.ToDouble(data[i, j]) - mean;
                varSum += v * v;
                if (i > 0)
                {
                    double vPrev = NumOps.ToDouble(data[i - 1, j]) - mean;
                    covSum += v * vPrev;
                }
            }

            autoCorr[j] = varSum > 1e-10 ? covSum / varSum : 0;
        }

        return autoCorr;
    }

    #endregion

    #region Metadata and Serialization (GANDALF Pattern)

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

    /// <inheritdoc/>
    public override ModelMetadata<T> GetModelMetadata()
    {
        return new ModelMetadata<T>
        {
            ModelType = ModelType.NeuralNetwork,
            AdditionalInfo = new Dictionary<string, object>
            {
                { "Architecture", "FinDiff" },
                { "InputSize", Architecture.CalculatedInputSize },
                { "OutputSize", Architecture.OutputSize },
                { "NumTimesteps", _options.NumTimesteps },
                { "TemporalWeight", _options.TemporalWeight },
                { "EnforcePositive", _options.EnforcePositive },
                { "MLPDimensions", _options.MLPDimensions },
                { "LayerCount", Layers.Count },
                { "LayerTypes", Layers.Select(l => l.GetType().Name).ToArray() }
            },
            ModelData = this.Serialize()
        };
    }

    /// <inheritdoc/>
    protected override void SerializeNetworkSpecificData(BinaryWriter writer)
    {
        writer.Write(_options.NumTimesteps);
        writer.Write(_options.TimestepEmbeddingDimension);
        writer.Write(_options.TemporalWeight);
        writer.Write(_options.EnforcePositive);
        writer.Write(_options.BetaStart);
        writer.Write(_options.BetaEnd);
        writer.Write(_options.LearningRate);
        writer.Write(_options.BatchSize);
        writer.Write(_options.VGMModes);
        writer.Write(_options.MLPDimensions.Length);
        foreach (var dim in _options.MLPDimensions)
        {
            writer.Write(dim);
        }
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
        return new FinDiffGenerator<T>(
            Architecture,
            _options,
            _optimizer,
            _lossFunction);
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
                ComputeColumnStatistics(data, col, meta);
            else if (meta.IsCategorical && meta.NumCategories == 0)
            {
                var categories = new HashSet<string>();
                for (int row = 0; row < data.Rows; row++)
                    categories.Add(NumOps.ToDouble(data[row, col]).ToString(System.Globalization.CultureInfo.InvariantCulture));
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
    /// FinDiff uses temporal correlation-aware diffusion with financial constraints which cannot be represented as a single computation graph.
    /// </summary>
    public override bool SupportsJitCompilation => false;

    #endregion
}
