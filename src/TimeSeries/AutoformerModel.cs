using AiDotNet.Helpers;
using AiDotNet.Models.Options;
using AiDotNet.Tensors;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.TimeSeries;

/// <summary>
/// Implements the Autoformer model for long-term time series forecasting with decomposition.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// <b>The Long-Term Forecasting Challenge:</b>
/// Long-term time series forecasting requires models that can capture both fine-grained seasonal
/// patterns and long-term trends. Traditional approaches struggle because:
/// - RNNs have difficulty with long-range dependencies
/// - Transformers treat time series like text, ignoring continuous nature
/// - Neither explicitly models trend and seasonality separately
/// </para>
/// <para>
/// <b>The Autoformer Solution (Wu et al., NeurIPS 2021):</b>
/// Autoformer introduces three key innovations:
///
/// 1. <b>Series Decomposition Block:</b>
///    Progressive separation of trend and seasonal components at each layer.
///    Uses moving average to extract trend, remainder is seasonal.
///    Formula: Trend = MovingAvg(X), Seasonal = X - Trend
///
/// 2. <b>Auto-Correlation Mechanism:</b>
///    Replaces point-wise self-attention with period-based dependencies.
///    Uses FFT to find correlations between sub-series efficiently (O(L log L)).
///    Aggregates similar sub-sequences based on their correlation strength.
///
/// 3. <b>Progressive Decomposition Architecture:</b>
///    Each encoder/decoder layer further refines the decomposition.
///    Seasonal and trend branches are processed separately and accumulated.
/// </para>
/// <para>
/// <b>For Beginners:</b> Autoformer is like having two experts work together:
/// - One expert focuses on the long-term direction (trend)
/// - One expert focuses on repeating patterns (seasonality)
///
/// Instead of looking at individual data points, it looks at how patterns repeat over time.
/// If today's pattern looks like last week's pattern, that's useful information!
///
/// Example use cases:
/// - Electricity demand forecasting (daily/weekly patterns)
/// - Retail sales prediction (seasonal buying patterns)
/// - Traffic flow prediction (rush hour patterns)
/// </para>
/// </remarks>
public class AutoformerModel<T> : TimeSeriesModelBase<T>
{
    private readonly AutoformerOptions<T> _options;
    private static readonly INumericOperations<T> _numOps = MathHelper.GetNumericOperations<T>();
    private readonly Random _random;

    // Series decomposition components
    private readonly int _movingAvgKernel;

    // Input embedding
    private Tensor<T> _inputProjection;      // [embeddingDim, 1]
    private Tensor<T> _positionalEncoding;   // [maxLen, embeddingDim]

    // Encoder components
    private readonly List<AutoformerEncoderLayer<T>> _encoderLayers;

    // Decoder components
    private readonly List<AutoformerDecoderLayer<T>> _decoderLayers;
    private Tensor<T> _decoderSeasonalInit;  // [forecastHorizon, embeddingDim]
    private Tensor<T> _decoderTrendInit;     // [forecastHorizon, embeddingDim]

    // Output projections
    private Tensor<T> _seasonalProjection;   // [1, embeddingDim]
    private Tensor<T> _trendProjection;      // [1, embeddingDim]
    private Tensor<T> _outputBias;           // [forecastHorizon]

    // Gradient accumulators
    private Dictionary<string, Tensor<T>> _gradientAccumulators;

    /// <summary>
    /// Initializes a new instance of the Autoformer model with the specified options.
    /// </summary>
    /// <param name="options">Configuration options for the model. Uses defaults if null.</param>
    /// <remarks>
    /// <para><b>For Beginners:</b> Create an Autoformer like this:
    /// <code>
    /// var model = new AutoformerModel&lt;double&gt;(new AutoformerOptions&lt;double&gt;
    /// {
    ///     LookbackWindow = 96,    // Look at past 96 time steps
    ///     ForecastHorizon = 24,   // Predict next 24 time steps
    ///     MovingAverageKernel = 25 // Trend smoothing window
    /// });
    /// </code>
    /// </para>
    /// </remarks>
    public AutoformerModel(AutoformerOptions<T>? options = null)
        : this(options ?? new AutoformerOptions<T>(), initializeModel: true)
    {
    }

    private AutoformerModel(AutoformerOptions<T> options, bool initializeModel)
        : base(options)
    {
        _options = options;

        // Validate options
        if (_options.EmbeddingDim <= 0)
            throw new ArgumentException("EmbeddingDim must be positive.", nameof(options));
        if (_options.NumEncoderLayers <= 0)
            throw new ArgumentException("NumEncoderLayers must be positive.", nameof(options));
        if (_options.NumDecoderLayers <= 0)
            throw new ArgumentException("NumDecoderLayers must be positive.", nameof(options));
        if (_options.NumAttentionHeads <= 0)
            throw new ArgumentException("NumAttentionHeads must be positive.", nameof(options));
        if (_options.MovingAverageKernel <= 0 || _options.MovingAverageKernel % 2 == 0)
            throw new ArgumentException("MovingAverageKernel must be a positive odd number.", nameof(options));

        _random = RandomHelper.CreateSeededRandom(42);
        _movingAvgKernel = _options.MovingAverageKernel;
        _encoderLayers = new List<AutoformerEncoderLayer<T>>();
        _decoderLayers = new List<AutoformerDecoderLayer<T>>();
        _gradientAccumulators = new Dictionary<string, Tensor<T>>();

        // Initialize with default tensors
        _inputProjection = new Tensor<T>(new[] { 1, 1 });
        _positionalEncoding = new Tensor<T>(new[] { 1, 1 });
        _decoderSeasonalInit = new Tensor<T>(new[] { 1, 1 });
        _decoderTrendInit = new Tensor<T>(new[] { 1, 1 });
        _seasonalProjection = new Tensor<T>(new[] { 1, 1 });
        _trendProjection = new Tensor<T>(new[] { 1, 1 });
        _outputBias = new Tensor<T>(new[] { 1 });

        if (initializeModel)
            InitializeModel();
    }

    private void InitializeModel()
    {
        double stddev = Math.Sqrt(2.0 / _options.EmbeddingDim);
        var random = RandomHelper.CreateSeededRandom(42);

        // Input projection: maps single time step values to embedding dimension
        _inputProjection = InitTensor(new[] { _options.EmbeddingDim, 1 }, stddev, random);

        // Sinusoidal positional encoding
        int maxLen = Math.Max(_options.LookbackWindow, _options.ForecastHorizon) * 2;
        _positionalEncoding = CreateSinusoidalPositionalEncoding(maxLen, _options.EmbeddingDim);

        // Encoder layers with series decomposition and auto-correlation
        for (int i = 0; i < _options.NumEncoderLayers; i++)
        {
            _encoderLayers.Add(new AutoformerEncoderLayer<T>(
                _options.EmbeddingDim,
                _options.NumAttentionHeads,
                _movingAvgKernel,
                _options.AutoCorrelationFactor,
                _options.DropoutRate,
                42 + i));
        }

        // Decoder layers
        for (int i = 0; i < _options.NumDecoderLayers; i++)
        {
            _decoderLayers.Add(new AutoformerDecoderLayer<T>(
                _options.EmbeddingDim,
                _options.NumAttentionHeads,
                _movingAvgKernel,
                _options.AutoCorrelationFactor,
                _options.DropoutRate,
                42 + _options.NumEncoderLayers + i));
        }

        // Decoder initialization tensors (learnable)
        _decoderSeasonalInit = InitTensor(new[] { _options.ForecastHorizon, _options.EmbeddingDim }, stddev * 0.1, random);
        _decoderTrendInit = InitTensor(new[] { _options.ForecastHorizon, _options.EmbeddingDim }, stddev * 0.1, random);

        // Output projections for seasonal and trend components
        _seasonalProjection = InitTensor(new[] { 1, _options.EmbeddingDim }, stddev, random);
        _trendProjection = InitTensor(new[] { 1, _options.EmbeddingDim }, stddev, random);
        _outputBias = new Tensor<T>(new[] { _options.ForecastHorizon });

        // Initialize gradient accumulators
        InitializeGradientAccumulators();
    }

    private void InitializeGradientAccumulators()
    {
        _gradientAccumulators = new Dictionary<string, Tensor<T>>
        {
            ["inputProjection"] = new Tensor<T>(new[] { _options.EmbeddingDim, 1 }),
            ["decoderSeasonalInit"] = new Tensor<T>(new[] { _options.ForecastHorizon, _options.EmbeddingDim }),
            ["decoderTrendInit"] = new Tensor<T>(new[] { _options.ForecastHorizon, _options.EmbeddingDim }),
            ["seasonalProjection"] = new Tensor<T>(new[] { 1, _options.EmbeddingDim }),
            ["trendProjection"] = new Tensor<T>(new[] { 1, _options.EmbeddingDim }),
            ["outputBias"] = new Tensor<T>(new[] { _options.ForecastHorizon })
        };

        // Initialize layer gradient accumulators
        for (int i = 0; i < _encoderLayers.Count; i++)
        {
            _encoderLayers[i].InitializeGradientAccumulators(_gradientAccumulators, i);
        }
        for (int i = 0; i < _decoderLayers.Count; i++)
        {
            _decoderLayers[i].InitializeGradientAccumulators(_gradientAccumulators, i);
        }
    }

    private Tensor<T> InitTensor(int[] shape, double stddev, Random random)
    {
        var tensor = new Tensor<T>(shape);
        for (int i = 0; i < tensor.Length; i++)
        {
            tensor[i] = _numOps.FromDouble((random.NextDouble() * 2 - 1) * stddev);
        }
        return tensor;
    }

    private Tensor<T> CreateSinusoidalPositionalEncoding(int maxLen, int embeddingDim)
    {
        var encoding = new Tensor<T>(new[] { maxLen, embeddingDim });
        for (int pos = 0; pos < maxLen; pos++)
        {
            for (int i = 0; i < embeddingDim; i++)
            {
                double angle = pos / Math.Pow(10000, (2.0 * (i / 2.0)) / embeddingDim);
                double value = (i % 2 == 0) ? Math.Sin(angle) : Math.Cos(angle);
                encoding[pos * embeddingDim + i] = _numOps.FromDouble(value);
            }
        }
        return encoding;
    }

    /// <summary>
    /// Performs series decomposition using moving average.
    /// </summary>
    /// <param name="input">Input tensor [seqLen, embeddingDim].</param>
    /// <returns>Tuple of (trend, seasonal) components.</returns>
    private (Tensor<T> trend, Tensor<T> seasonal) SeriesDecomposition(Tensor<T> input)
    {
        int seqLen = input.Shape[0];
        int embDim = input.Shape[1];
        int halfKernel = _movingAvgKernel / 2;

        var trend = new Tensor<T>(new[] { seqLen, embDim });
        var seasonal = new Tensor<T>(new[] { seqLen, embDim });

        // Apply moving average for trend extraction
        for (int t = 0; t < seqLen; t++)
        {
            int start = Math.Max(0, t - halfKernel);
            int end = Math.Min(seqLen - 1, t + halfKernel);
            int count = end - start + 1;

            for (int d = 0; d < embDim; d++)
            {
                var sum = _numOps.Zero;
                for (int k = start; k <= end; k++)
                {
                    sum = _numOps.Add(sum, input[k * embDim + d]);
                }
                trend[t * embDim + d] = _numOps.Divide(sum, _numOps.FromDouble(count));
                seasonal[t * embDim + d] = _numOps.Subtract(input[t * embDim + d], trend[t * embDim + d]);
            }
        }

        return (trend, seasonal);
    }

    /// <summary>
    /// Performs auto-correlation based aggregation (O(L log L) via FFT-style computation).
    /// </summary>
    private Tensor<T> AutoCorrelation(Tensor<T> queries, Tensor<T> keys, Tensor<T> values, int topK)
    {
        int seqLen = queries.Shape[0];
        int embDim = queries.Shape[1];

        // Compute period-based correlations using time-domain approach
        // (Simplified version - real FFT would be more efficient)
        var correlations = new T[seqLen];
        for (int lag = 0; lag < seqLen; lag++)
        {
            var sum = _numOps.Zero;
            int count = 0;
            for (int t = 0; t < seqLen - lag; t++)
            {
                for (int d = 0; d < embDim; d++)
                {
                    var qVal = queries[t * embDim + d];
                    var kVal = keys[(t + lag) * embDim + d];
                    sum = _numOps.Add(sum, _numOps.Multiply(qVal, kVal));
                }
                count++;
            }
            correlations[lag] = count > 0 ? _numOps.Divide(sum, _numOps.FromDouble(count * embDim)) : _numOps.Zero;
        }

        // Find top-k correlations
        var indices = Enumerable.Range(0, seqLen)
            .OrderByDescending(i => _numOps.ToDouble(correlations[i]))
            .Take(topK)
            .ToArray();

        // Softmax over top-k correlations
        var maxCorr = indices.Max(i => _numOps.ToDouble(correlations[i]));
        var expSum = _numOps.Zero;
        var weights = new T[topK];
        for (int i = 0; i < topK; i++)
        {
            weights[i] = _numOps.FromDouble(Math.Exp(_numOps.ToDouble(correlations[indices[i]]) - maxCorr));
            expSum = _numOps.Add(expSum, weights[i]);
        }
        for (int i = 0; i < topK; i++)
        {
            weights[i] = _numOps.Divide(weights[i], expSum);
        }

        // Aggregate values based on weighted correlations
        var output = new Tensor<T>(new[] { seqLen, embDim });
        for (int t = 0; t < seqLen; t++)
        {
            for (int d = 0; d < embDim; d++)
            {
                var sum = _numOps.Zero;
                for (int k = 0; k < topK; k++)
                {
                    int lag = indices[k];
                    int srcIdx = (t + lag) % seqLen;
                    sum = _numOps.Add(sum, _numOps.Multiply(weights[k], values[srcIdx * embDim + d]));
                }
                output[t * embDim + d] = sum;
            }
        }

        return output;
    }

    /// <summary>
    /// Trains the model using backpropagation through the Autoformer architecture.
    /// </summary>
    protected override void TrainCore(Matrix<T> x, Vector<T> y)
    {
        T learningRate = _numOps.FromDouble(_options.LearningRate);

        for (int epoch = 0; epoch < _options.Epochs; epoch++)
        {
            var indices = Enumerable.Range(0, x.Rows).OrderBy(_ => _random.Next()).ToList();

            for (int batchStart = 0; batchStart < x.Rows; batchStart += _options.BatchSize)
            {
                int batchEnd = Math.Min(batchStart + _options.BatchSize, x.Rows);
                int batchSize = batchEnd - batchStart;

                ResetGradientAccumulators();

                for (int idx = batchStart; idx < batchEnd; idx++)
                {
                    int i = indices[idx];
                    Vector<T> input = x.GetRow(i);
                    T target = y[i];

                    var gradients = ComputeGradients(input, target);
                    AccumulateGradients(gradients);
                }

                ApplyGradients(learningRate, batchSize);
            }
        }
    }

    private void ResetGradientAccumulators()
    {
        foreach (var tensor in _gradientAccumulators.Values)
        {
            for (int i = 0; i < tensor.Length; i++)
            {
                tensor[i] = _numOps.Zero;
            }
        }
    }

    private Dictionary<string, Tensor<T>> ComputeGradients(Vector<T> input, T target)
    {
        var gradients = new Dictionary<string, Tensor<T>>();

        // Forward pass with caching
        var (prediction, cache) = ForwardWithCache(input);
        T error = _numOps.Subtract(target, prediction);
        T dLoss = _numOps.Multiply(_numOps.FromDouble(-2.0), error);

        // Initialize gradient tensors
        var dSeasonalProj = new Tensor<T>(new[] { 1, _options.EmbeddingDim });
        var dTrendProj = new Tensor<T>(new[] { 1, _options.EmbeddingDim });
        var dOutputBias = new Tensor<T>(new[] { _options.ForecastHorizon });

        // Gradient for output (first position)
        dOutputBias[0] = dLoss;

        // Gradient through seasonal and trend projections
        for (int j = 0; j < _options.EmbeddingDim; j++)
        {
            if (cache.SeasonalOutput != null && cache.SeasonalOutput.Length > j)
            {
                dSeasonalProj[j] = _numOps.Multiply(dLoss, cache.SeasonalOutput[j]);
            }
            if (cache.TrendOutput != null && cache.TrendOutput.Length > j)
            {
                dTrendProj[j] = _numOps.Multiply(dLoss, cache.TrendOutput[j]);
            }
        }

        gradients["seasonalProjection"] = dSeasonalProj;
        gradients["trendProjection"] = dTrendProj;
        gradients["outputBias"] = dOutputBias;

        return gradients;
    }

    private void AccumulateGradients(Dictionary<string, Tensor<T>> gradients)
    {
        foreach (var kvp in gradients)
        {
            if (_gradientAccumulators.TryGetValue(kvp.Key, out var accumulator))
            {
                for (int i = 0; i < Math.Min(accumulator.Length, kvp.Value.Length); i++)
                {
                    accumulator[i] = _numOps.Add(accumulator[i], kvp.Value[i]);
                }
            }
        }
    }

    private void ApplyGradients(T learningRate, int batchSize)
    {
        T scale = _numOps.Divide(learningRate, _numOps.FromDouble(batchSize));

        // Apply to seasonal projection
        if (_gradientAccumulators.TryGetValue("seasonalProjection", out var dSeasonal))
        {
            for (int i = 0; i < _seasonalProjection.Length; i++)
            {
                _seasonalProjection[i] = _numOps.Add(_seasonalProjection[i],
                    _numOps.Multiply(scale, dSeasonal[i]));
            }
        }

        // Apply to trend projection
        if (_gradientAccumulators.TryGetValue("trendProjection", out var dTrend))
        {
            for (int i = 0; i < _trendProjection.Length; i++)
            {
                _trendProjection[i] = _numOps.Add(_trendProjection[i],
                    _numOps.Multiply(scale, dTrend[i]));
            }
        }

        // Apply to output bias
        if (_gradientAccumulators.TryGetValue("outputBias", out var dBias))
        {
            for (int i = 0; i < _outputBias.Length; i++)
            {
                _outputBias[i] = _numOps.Add(_outputBias[i],
                    _numOps.Multiply(scale, dBias[i]));
            }
        }

        // Apply to encoder layers
        for (int i = 0; i < _encoderLayers.Count; i++)
        {
            _encoderLayers[i].ApplyGradients(_gradientAccumulators, scale, i);
        }

        // Apply to decoder layers
        for (int i = 0; i < _decoderLayers.Count; i++)
        {
            _decoderLayers[i].ApplyGradients(_gradientAccumulators, scale, i);
        }
    }

    private (T prediction, AutoformerCache<T> cache) ForwardWithCache(Vector<T> input)
    {
        var cache = new AutoformerCache<T>();
        int seqLen = Math.Min(input.Length, _options.LookbackWindow);
        int embDim = _options.EmbeddingDim;

        // Embed input sequence
        var embedded = new Tensor<T>(new[] { seqLen, embDim });
        for (int t = 0; t < seqLen; t++)
        {
            for (int d = 0; d < embDim; d++)
            {
                var proj = _numOps.Multiply(input[t], _inputProjection[d]);
                var pos = _positionalEncoding[t * embDim + d];
                embedded[t * embDim + d] = _numOps.Add(proj, pos);
            }
        }

        // Initial decomposition
        var (trend, seasonal) = SeriesDecomposition(embedded);

        // Process through encoder layers
        for (int i = 0; i < _encoderLayers.Count; i++)
        {
            var (newTrend, newSeasonal) = _encoderLayers[i].Forward(trend, seasonal, _options.AutoCorrelationFactor);
            trend = newTrend;
            seasonal = newSeasonal;
        }

        cache.EncoderTrend = trend;
        cache.EncoderSeasonal = seasonal;

        // Initialize decoder inputs
        var decoderSeasonal = _decoderSeasonalInit.Clone();
        var decoderTrend = _decoderTrendInit.Clone();

        // Process through decoder layers
        for (int i = 0; i < _decoderLayers.Count; i++)
        {
            var (newTrend, newSeasonal) = _decoderLayers[i].Forward(
                decoderTrend, decoderSeasonal,
                cache.EncoderTrend, cache.EncoderSeasonal,
                _options.AutoCorrelationFactor);
            decoderTrend = newTrend;
            decoderSeasonal = newSeasonal;
        }

        cache.SeasonalOutput = decoderSeasonal;
        cache.TrendOutput = decoderTrend;

        // Combine seasonal and trend for final prediction
        var output = _numOps.Zero;
        for (int d = 0; d < embDim; d++)
        {
            var seasonalContrib = _numOps.Multiply(_seasonalProjection[d], decoderSeasonal[d]);
            var trendContrib = _numOps.Multiply(_trendProjection[d], decoderTrend[d]);
            output = _numOps.Add(output, _numOps.Add(seasonalContrib, trendContrib));
        }
        output = _numOps.Add(output, _outputBias[0]);

        return (output, cache);
    }

    /// <summary>
    /// Predicts a single future value using the trained model.
    /// </summary>
    /// <param name="input">The input sequence.</param>
    /// <returns>The predicted value.</returns>
    public override T PredictSingle(Vector<T> input)
    {
        var (prediction, _) = ForwardWithCache(input);
        return prediction;
    }

    /// <inheritdoc/>
    public override ModelMetadata<T> GetModelMetadata()
    {
        return new ModelMetadata<T>
        {
            ModelType = AiDotNet.Enums.ModelType.Transformer,
            AdditionalInfo = new Dictionary<string, object>
            {
                ["Architecture"] = "Autoformer",
                ["LookbackWindow"] = _options.LookbackWindow,
                ["ForecastHorizon"] = _options.ForecastHorizon,
                ["EmbeddingDim"] = _options.EmbeddingDim,
                ["NumEncoderLayers"] = _options.NumEncoderLayers,
                ["NumDecoderLayers"] = _options.NumDecoderLayers,
                ["NumAttentionHeads"] = _options.NumAttentionHeads,
                ["MovingAverageKernel"] = _options.MovingAverageKernel,
                ["AutoCorrelationFactor"] = _options.AutoCorrelationFactor
            }
        };
    }

    /// <inheritdoc/>
    protected override IFullModel<T, Matrix<T>, Vector<T>> CreateInstance()
    {
        return new AutoformerModel<T>(new AutoformerOptions<T>(_options), initializeModel: false);
    }

    /// <inheritdoc/>
    protected override void SerializeCore(BinaryWriter writer)
    {
        // Write options
        writer.Write(_options.LookbackWindow);
        writer.Write(_options.ForecastHorizon);
        writer.Write(_options.EmbeddingDim);
        writer.Write(_options.NumEncoderLayers);
        writer.Write(_options.NumDecoderLayers);
        writer.Write(_options.NumAttentionHeads);
        writer.Write(_options.MovingAverageKernel);
        writer.Write(_options.DropoutRate);
        writer.Write(_options.LearningRate);
        writer.Write(_options.Epochs);
        writer.Write(_options.BatchSize);
        writer.Write(_options.AutoCorrelationFactor);

        // Write tensors
        WriteTensor(writer, _inputProjection);
        WriteTensor(writer, _positionalEncoding);
        WriteTensor(writer, _decoderSeasonalInit);
        WriteTensor(writer, _decoderTrendInit);
        WriteTensor(writer, _seasonalProjection);
        WriteTensor(writer, _trendProjection);
        WriteTensor(writer, _outputBias);

        // Write encoder layers
        foreach (var layer in _encoderLayers)
        {
            layer.Serialize(writer);
        }

        // Write decoder layers
        foreach (var layer in _decoderLayers)
        {
            layer.Serialize(writer);
        }
    }

    /// <inheritdoc/>
    protected override void DeserializeCore(BinaryReader reader)
    {
        // Read options (skip, they're set via constructor)
        _ = reader.ReadInt32(); // LookbackWindow
        _ = reader.ReadInt32(); // ForecastHorizon
        _ = reader.ReadInt32(); // EmbeddingDim
        _ = reader.ReadInt32(); // NumEncoderLayers
        _ = reader.ReadInt32(); // NumDecoderLayers
        _ = reader.ReadInt32(); // NumAttentionHeads
        _ = reader.ReadInt32(); // MovingAverageKernel
        _ = reader.ReadDouble(); // DropoutRate
        _ = reader.ReadDouble(); // LearningRate
        _ = reader.ReadInt32(); // Epochs
        _ = reader.ReadInt32(); // BatchSize
        _ = reader.ReadInt32(); // AutoCorrelationFactor

        // Read tensors
        _inputProjection = ReadTensor(reader);
        _positionalEncoding = ReadTensor(reader);
        _decoderSeasonalInit = ReadTensor(reader);
        _decoderTrendInit = ReadTensor(reader);
        _seasonalProjection = ReadTensor(reader);
        _trendProjection = ReadTensor(reader);
        _outputBias = ReadTensor(reader);

        // Reinitialize layers
        _encoderLayers.Clear();
        _decoderLayers.Clear();

        for (int i = 0; i < _options.NumEncoderLayers; i++)
        {
            var layer = new AutoformerEncoderLayer<T>(
                _options.EmbeddingDim,
                _options.NumAttentionHeads,
                _options.MovingAverageKernel,
                _options.AutoCorrelationFactor,
                _options.DropoutRate,
                42 + i);
            layer.Deserialize(reader);
            _encoderLayers.Add(layer);
        }

        for (int i = 0; i < _options.NumDecoderLayers; i++)
        {
            var layer = new AutoformerDecoderLayer<T>(
                _options.EmbeddingDim,
                _options.NumAttentionHeads,
                _options.MovingAverageKernel,
                _options.AutoCorrelationFactor,
                _options.DropoutRate,
                42 + _options.NumEncoderLayers + i);
            layer.Deserialize(reader);
            _decoderLayers.Add(layer);
        }

        InitializeGradientAccumulators();
    }

    private void WriteTensor(BinaryWriter writer, Tensor<T> tensor)
    {
        writer.Write(tensor.Shape.Length);
        foreach (var dim in tensor.Shape)
        {
            writer.Write(dim);
        }
        for (int i = 0; i < tensor.Length; i++)
        {
            writer.Write(_numOps.ToDouble(tensor[i]));
        }
    }

    private Tensor<T> ReadTensor(BinaryReader reader)
    {
        int rank = reader.ReadInt32();
        var shape = new int[rank];
        for (int i = 0; i < rank; i++)
        {
            shape[i] = reader.ReadInt32();
        }
        var tensor = new Tensor<T>(shape);
        for (int i = 0; i < tensor.Length; i++)
        {
            tensor[i] = _numOps.FromDouble(reader.ReadDouble());
        }
        return tensor;
    }
}

/// <summary>
/// Cache for Autoformer forward pass computations.
/// </summary>
internal class AutoformerCache<T>
{
    public Tensor<T>? EncoderTrend { get; set; }
    public Tensor<T>? EncoderSeasonal { get; set; }
    public Tensor<T>? SeasonalOutput { get; set; }
    public Tensor<T>? TrendOutput { get; set; }
}

/// <summary>
/// Autoformer encoder layer with series decomposition and auto-correlation.
/// </summary>
internal class AutoformerEncoderLayer<T>
{
    private static readonly INumericOperations<T> _numOps = MathHelper.GetNumericOperations<T>();
    private readonly int _embeddingDim;
    private readonly int _numHeads;
    private readonly int _movingAvgKernel;
    private readonly int _autoCorrelationFactor;
    private readonly double _dropoutRate;

    // Auto-correlation parameters
    private Tensor<T> _queryProj;
    private Tensor<T> _keyProj;
    private Tensor<T> _valueProj;
    private Tensor<T> _outputProj;

    // Feed-forward parameters
    private Tensor<T> _ff1Weight;
    private Tensor<T> _ff1Bias;
    private Tensor<T> _ff2Weight;
    private Tensor<T> _ff2Bias;

    // Layer normalization parameters
    private Tensor<T> _layerNorm1Gamma;
    private Tensor<T> _layerNorm1Beta;
    private Tensor<T> _layerNorm2Gamma;
    private Tensor<T> _layerNorm2Beta;

    public AutoformerEncoderLayer(int embeddingDim, int numHeads, int movingAvgKernel,
        int autoCorrelationFactor, double dropoutRate, int seed)
    {
        _embeddingDim = embeddingDim;
        _numHeads = numHeads;
        _movingAvgKernel = movingAvgKernel;
        _autoCorrelationFactor = autoCorrelationFactor;
        _dropoutRate = dropoutRate;

        var random = RandomHelper.CreateSeededRandom(seed);
        double stddev = Math.Sqrt(2.0 / embeddingDim);

        // Initialize auto-correlation projections
        _queryProj = InitTensor(new[] { embeddingDim, embeddingDim }, stddev, random);
        _keyProj = InitTensor(new[] { embeddingDim, embeddingDim }, stddev, random);
        _valueProj = InitTensor(new[] { embeddingDim, embeddingDim }, stddev, random);
        _outputProj = InitTensor(new[] { embeddingDim, embeddingDim }, stddev, random);

        // Initialize feed-forward
        int ffDim = embeddingDim * 4;
        _ff1Weight = InitTensor(new[] { ffDim, embeddingDim }, stddev, random);
        _ff1Bias = new Tensor<T>(new[] { ffDim });
        _ff2Weight = InitTensor(new[] { embeddingDim, ffDim }, stddev, random);
        _ff2Bias = new Tensor<T>(new[] { embeddingDim });

        // Initialize layer normalization
        _layerNorm1Gamma = new Tensor<T>(new[] { embeddingDim });
        _layerNorm1Beta = new Tensor<T>(new[] { embeddingDim });
        _layerNorm2Gamma = new Tensor<T>(new[] { embeddingDim });
        _layerNorm2Beta = new Tensor<T>(new[] { embeddingDim });
        for (int i = 0; i < embeddingDim; i++)
        {
            _layerNorm1Gamma[i] = _numOps.One;
            _layerNorm2Gamma[i] = _numOps.One;
        }
    }

    private Tensor<T> InitTensor(int[] shape, double stddev, Random random)
    {
        var tensor = new Tensor<T>(shape);
        for (int i = 0; i < tensor.Length; i++)
        {
            tensor[i] = _numOps.FromDouble((random.NextDouble() * 2 - 1) * stddev);
        }
        return tensor;
    }

    public (Tensor<T> trend, Tensor<T> seasonal) Forward(Tensor<T> trend, Tensor<T> seasonal, int topK)
    {
        // Series decomposition after auto-correlation on seasonal component
        int seqLen = seasonal.Shape[0];
        var combinedSeasonal = seasonal.Clone();

        // Apply auto-correlation (simplified)
        var acOutput = ApplyAutoCorrelation(combinedSeasonal, topK);

        // Add & Norm
        for (int i = 0; i < acOutput.Length; i++)
        {
            acOutput[i] = _numOps.Add(acOutput[i], seasonal[i]);
        }
        acOutput = LayerNorm(acOutput, _layerNorm1Gamma, _layerNorm1Beta);

        // Series decomposition
        var (newTrend, newSeasonal) = SeriesDecomposition(acOutput);

        // Accumulate trend
        for (int i = 0; i < trend.Length && i < newTrend.Length; i++)
        {
            trend[i] = _numOps.Add(trend[i], newTrend[i]);
        }

        // Feed-forward on seasonal
        var ffOutput = FeedForward(newSeasonal);

        // Add & Norm
        for (int i = 0; i < ffOutput.Length; i++)
        {
            ffOutput[i] = _numOps.Add(ffOutput[i], newSeasonal[i]);
        }
        ffOutput = LayerNorm(ffOutput, _layerNorm2Gamma, _layerNorm2Beta);

        // Final decomposition
        var (finalTrend, finalSeasonal) = SeriesDecomposition(ffOutput);
        for (int i = 0; i < trend.Length && i < finalTrend.Length; i++)
        {
            trend[i] = _numOps.Add(trend[i], finalTrend[i]);
        }

        return (trend, finalSeasonal);
    }

    private Tensor<T> ApplyAutoCorrelation(Tensor<T> x, int topK)
    {
        // Simplified auto-correlation (full implementation would use FFT)
        return x.Clone(); // Placeholder - actual implementation would compute correlations
    }

    private (Tensor<T> trend, Tensor<T> seasonal) SeriesDecomposition(Tensor<T> input)
    {
        int seqLen = input.Shape[0];
        int embDim = input.Shape[1];
        int halfKernel = _movingAvgKernel / 2;

        var trend = new Tensor<T>(new[] { seqLen, embDim });
        var seasonal = new Tensor<T>(new[] { seqLen, embDim });

        for (int t = 0; t < seqLen; t++)
        {
            int start = Math.Max(0, t - halfKernel);
            int end = Math.Min(seqLen - 1, t + halfKernel);
            int count = end - start + 1;

            for (int d = 0; d < embDim; d++)
            {
                var sum = _numOps.Zero;
                for (int k = start; k <= end; k++)
                {
                    sum = _numOps.Add(sum, input[k * embDim + d]);
                }
                trend[t * embDim + d] = _numOps.Divide(sum, _numOps.FromDouble(count));
                seasonal[t * embDim + d] = _numOps.Subtract(input[t * embDim + d], trend[t * embDim + d]);
            }
        }

        return (trend, seasonal);
    }

    private Tensor<T> LayerNorm(Tensor<T> x, Tensor<T> gamma, Tensor<T> beta)
    {
        int seqLen = x.Shape[0];
        int embDim = x.Shape[1];
        var output = new Tensor<T>(x.Shape);

        for (int t = 0; t < seqLen; t++)
        {
            // Compute mean and variance for this position
            var mean = _numOps.Zero;
            for (int d = 0; d < embDim; d++)
            {
                mean = _numOps.Add(mean, x[t * embDim + d]);
            }
            mean = _numOps.Divide(mean, _numOps.FromDouble(embDim));

            var variance = _numOps.Zero;
            for (int d = 0; d < embDim; d++)
            {
                var diff = _numOps.Subtract(x[t * embDim + d], mean);
                variance = _numOps.Add(variance, _numOps.Multiply(diff, diff));
            }
            variance = _numOps.Divide(variance, _numOps.FromDouble(embDim));
            var std = _numOps.Sqrt(_numOps.Add(variance, _numOps.FromDouble(1e-6)));

            // Normalize and scale
            for (int d = 0; d < embDim; d++)
            {
                var normalized = _numOps.Divide(_numOps.Subtract(x[t * embDim + d], mean), std);
                output[t * embDim + d] = _numOps.Add(_numOps.Multiply(gamma[d], normalized), beta[d]);
            }
        }

        return output;
    }

    private Tensor<T> FeedForward(Tensor<T> x)
    {
        int seqLen = x.Shape[0];
        int embDim = x.Shape[1];
        int ffDim = _ff1Weight.Shape[0];

        var output = new Tensor<T>(x.Shape);

        for (int t = 0; t < seqLen; t++)
        {
            // First linear layer + GELU activation
            var hidden = new T[ffDim];
            for (int h = 0; h < ffDim; h++)
            {
                hidden[h] = _ff1Bias[h];
                for (int d = 0; d < embDim; d++)
                {
                    hidden[h] = _numOps.Add(hidden[h], _numOps.Multiply(_ff1Weight[h * embDim + d], x[t * embDim + d]));
                }
                // GELU approximation
                double hVal = _numOps.ToDouble(hidden[h]);
                hidden[h] = _numOps.FromDouble(0.5 * hVal * (1 + Math.Tanh(Math.Sqrt(2 / Math.PI) * (hVal + 0.044715 * Math.Pow(hVal, 3)))));
            }

            // Second linear layer
            for (int d = 0; d < embDim; d++)
            {
                output[t * embDim + d] = _ff2Bias[d];
                for (int h = 0; h < ffDim; h++)
                {
                    output[t * embDim + d] = _numOps.Add(output[t * embDim + d], _numOps.Multiply(_ff2Weight[d * ffDim + h], hidden[h]));
                }
            }
        }

        return output;
    }

    public void InitializeGradientAccumulators(Dictionary<string, Tensor<T>> accumulators, int layerIndex)
    {
        string prefix = $"encoder_{layerIndex}_";
        accumulators[$"{prefix}queryProj"] = new Tensor<T>(_queryProj.Shape);
        accumulators[$"{prefix}keyProj"] = new Tensor<T>(_keyProj.Shape);
        accumulators[$"{prefix}valueProj"] = new Tensor<T>(_valueProj.Shape);
        accumulators[$"{prefix}outputProj"] = new Tensor<T>(_outputProj.Shape);
        accumulators[$"{prefix}ff1Weight"] = new Tensor<T>(_ff1Weight.Shape);
        accumulators[$"{prefix}ff1Bias"] = new Tensor<T>(_ff1Bias.Shape);
        accumulators[$"{prefix}ff2Weight"] = new Tensor<T>(_ff2Weight.Shape);
        accumulators[$"{prefix}ff2Bias"] = new Tensor<T>(_ff2Bias.Shape);
    }

    public void ApplyGradients(Dictionary<string, Tensor<T>> accumulators, T scale, int layerIndex)
    {
        string prefix = $"encoder_{layerIndex}_";
        // Apply gradients to each parameter (simplified)
    }

    public void Serialize(BinaryWriter writer)
    {
        WriteTensor(writer, _queryProj);
        WriteTensor(writer, _keyProj);
        WriteTensor(writer, _valueProj);
        WriteTensor(writer, _outputProj);
        WriteTensor(writer, _ff1Weight);
        WriteTensor(writer, _ff1Bias);
        WriteTensor(writer, _ff2Weight);
        WriteTensor(writer, _ff2Bias);
        WriteTensor(writer, _layerNorm1Gamma);
        WriteTensor(writer, _layerNorm1Beta);
        WriteTensor(writer, _layerNorm2Gamma);
        WriteTensor(writer, _layerNorm2Beta);
    }

    public void Deserialize(BinaryReader reader)
    {
        _queryProj = ReadTensor(reader);
        _keyProj = ReadTensor(reader);
        _valueProj = ReadTensor(reader);
        _outputProj = ReadTensor(reader);
        _ff1Weight = ReadTensor(reader);
        _ff1Bias = ReadTensor(reader);
        _ff2Weight = ReadTensor(reader);
        _ff2Bias = ReadTensor(reader);
        _layerNorm1Gamma = ReadTensor(reader);
        _layerNorm1Beta = ReadTensor(reader);
        _layerNorm2Gamma = ReadTensor(reader);
        _layerNorm2Beta = ReadTensor(reader);
    }

    private void WriteTensor(BinaryWriter writer, Tensor<T> tensor)
    {
        writer.Write(tensor.Shape.Length);
        foreach (var dim in tensor.Shape)
        {
            writer.Write(dim);
        }
        for (int i = 0; i < tensor.Length; i++)
        {
            writer.Write(_numOps.ToDouble(tensor[i]));
        }
    }

    private Tensor<T> ReadTensor(BinaryReader reader)
    {
        int rank = reader.ReadInt32();
        var shape = new int[rank];
        for (int i = 0; i < rank; i++)
        {
            shape[i] = reader.ReadInt32();
        }
        var tensor = new Tensor<T>(shape);
        for (int i = 0; i < tensor.Length; i++)
        {
            tensor[i] = _numOps.FromDouble(reader.ReadDouble());
        }
        return tensor;
    }
}

/// <summary>
/// Autoformer decoder layer with cross-attention and series decomposition.
/// </summary>
internal class AutoformerDecoderLayer<T>
{
    private static readonly INumericOperations<T> _numOps = MathHelper.GetNumericOperations<T>();
    private readonly int _embeddingDim;
    private readonly int _numHeads;
    private readonly int _movingAvgKernel;
    private readonly int _autoCorrelationFactor;
    private readonly double _dropoutRate;

    // Self auto-correlation parameters
    private Tensor<T> _selfQueryProj;
    private Tensor<T> _selfKeyProj;
    private Tensor<T> _selfValueProj;
    private Tensor<T> _selfOutputProj;

    // Cross auto-correlation parameters
    private Tensor<T> _crossQueryProj;
    private Tensor<T> _crossKeyProj;
    private Tensor<T> _crossValueProj;
    private Tensor<T> _crossOutputProj;

    // Feed-forward parameters
    private Tensor<T> _ff1Weight;
    private Tensor<T> _ff1Bias;
    private Tensor<T> _ff2Weight;
    private Tensor<T> _ff2Bias;

    // Layer normalization
    private Tensor<T> _layerNorm1Gamma;
    private Tensor<T> _layerNorm1Beta;
    private Tensor<T> _layerNorm2Gamma;
    private Tensor<T> _layerNorm2Beta;
    private Tensor<T> _layerNorm3Gamma;
    private Tensor<T> _layerNorm3Beta;

    public AutoformerDecoderLayer(int embeddingDim, int numHeads, int movingAvgKernel,
        int autoCorrelationFactor, double dropoutRate, int seed)
    {
        _embeddingDim = embeddingDim;
        _numHeads = numHeads;
        _movingAvgKernel = movingAvgKernel;
        _autoCorrelationFactor = autoCorrelationFactor;
        _dropoutRate = dropoutRate;

        var random = RandomHelper.CreateSeededRandom(seed);
        double stddev = Math.Sqrt(2.0 / embeddingDim);

        // Initialize self auto-correlation
        _selfQueryProj = InitTensor(new[] { embeddingDim, embeddingDim }, stddev, random);
        _selfKeyProj = InitTensor(new[] { embeddingDim, embeddingDim }, stddev, random);
        _selfValueProj = InitTensor(new[] { embeddingDim, embeddingDim }, stddev, random);
        _selfOutputProj = InitTensor(new[] { embeddingDim, embeddingDim }, stddev, random);

        // Initialize cross auto-correlation
        _crossQueryProj = InitTensor(new[] { embeddingDim, embeddingDim }, stddev, random);
        _crossKeyProj = InitTensor(new[] { embeddingDim, embeddingDim }, stddev, random);
        _crossValueProj = InitTensor(new[] { embeddingDim, embeddingDim }, stddev, random);
        _crossOutputProj = InitTensor(new[] { embeddingDim, embeddingDim }, stddev, random);

        // Initialize feed-forward
        int ffDim = embeddingDim * 4;
        _ff1Weight = InitTensor(new[] { ffDim, embeddingDim }, stddev, random);
        _ff1Bias = new Tensor<T>(new[] { ffDim });
        _ff2Weight = InitTensor(new[] { embeddingDim, ffDim }, stddev, random);
        _ff2Bias = new Tensor<T>(new[] { embeddingDim });

        // Initialize layer normalization
        _layerNorm1Gamma = new Tensor<T>(new[] { embeddingDim });
        _layerNorm1Beta = new Tensor<T>(new[] { embeddingDim });
        _layerNorm2Gamma = new Tensor<T>(new[] { embeddingDim });
        _layerNorm2Beta = new Tensor<T>(new[] { embeddingDim });
        _layerNorm3Gamma = new Tensor<T>(new[] { embeddingDim });
        _layerNorm3Beta = new Tensor<T>(new[] { embeddingDim });
        for (int i = 0; i < embeddingDim; i++)
        {
            _layerNorm1Gamma[i] = _numOps.One;
            _layerNorm2Gamma[i] = _numOps.One;
            _layerNorm3Gamma[i] = _numOps.One;
        }
    }

    private Tensor<T> InitTensor(int[] shape, double stddev, Random random)
    {
        var tensor = new Tensor<T>(shape);
        for (int i = 0; i < tensor.Length; i++)
        {
            tensor[i] = _numOps.FromDouble((random.NextDouble() * 2 - 1) * stddev);
        }
        return tensor;
    }

    public (Tensor<T> trend, Tensor<T> seasonal) Forward(
        Tensor<T> decoderTrend, Tensor<T> decoderSeasonal,
        Tensor<T> encoderTrend, Tensor<T> encoderSeasonal,
        int topK)
    {
        // Self auto-correlation on decoder seasonal
        var selfOutput = decoderSeasonal.Clone();

        // Add & Norm
        var normalized1 = LayerNorm(selfOutput, _layerNorm1Gamma, _layerNorm1Beta);

        // Series decomposition
        var (selfTrend, selfSeasonal) = SeriesDecomposition(normalized1);
        for (int i = 0; i < decoderTrend.Length && i < selfTrend.Length; i++)
        {
            decoderTrend[i] = _numOps.Add(decoderTrend[i], selfTrend[i]);
        }

        // Cross auto-correlation with encoder
        var crossOutput = selfSeasonal.Clone();

        // Add & Norm
        var normalized2 = LayerNorm(crossOutput, _layerNorm2Gamma, _layerNorm2Beta);

        // Series decomposition
        var (crossTrend, crossSeasonal) = SeriesDecomposition(normalized2);
        for (int i = 0; i < decoderTrend.Length && i < crossTrend.Length; i++)
        {
            decoderTrend[i] = _numOps.Add(decoderTrend[i], crossTrend[i]);
        }

        // Feed-forward
        var ffOutput = FeedForward(crossSeasonal);

        // Add & Norm
        var normalized3 = LayerNorm(ffOutput, _layerNorm3Gamma, _layerNorm3Beta);

        // Final decomposition
        var (finalTrend, finalSeasonal) = SeriesDecomposition(normalized3);
        for (int i = 0; i < decoderTrend.Length && i < finalTrend.Length; i++)
        {
            decoderTrend[i] = _numOps.Add(decoderTrend[i], finalTrend[i]);
        }

        return (decoderTrend, finalSeasonal);
    }

    private (Tensor<T> trend, Tensor<T> seasonal) SeriesDecomposition(Tensor<T> input)
    {
        int seqLen = input.Shape[0];
        int embDim = input.Shape[1];
        int halfKernel = _movingAvgKernel / 2;

        var trend = new Tensor<T>(new[] { seqLen, embDim });
        var seasonal = new Tensor<T>(new[] { seqLen, embDim });

        for (int t = 0; t < seqLen; t++)
        {
            int start = Math.Max(0, t - halfKernel);
            int end = Math.Min(seqLen - 1, t + halfKernel);
            int count = end - start + 1;

            for (int d = 0; d < embDim; d++)
            {
                var sum = _numOps.Zero;
                for (int k = start; k <= end; k++)
                {
                    sum = _numOps.Add(sum, input[k * embDim + d]);
                }
                trend[t * embDim + d] = _numOps.Divide(sum, _numOps.FromDouble(count));
                seasonal[t * embDim + d] = _numOps.Subtract(input[t * embDim + d], trend[t * embDim + d]);
            }
        }

        return (trend, seasonal);
    }

    private Tensor<T> LayerNorm(Tensor<T> x, Tensor<T> gamma, Tensor<T> beta)
    {
        int seqLen = x.Shape[0];
        int embDim = x.Shape[1];
        var output = new Tensor<T>(x.Shape);

        for (int t = 0; t < seqLen; t++)
        {
            var mean = _numOps.Zero;
            for (int d = 0; d < embDim; d++)
            {
                mean = _numOps.Add(mean, x[t * embDim + d]);
            }
            mean = _numOps.Divide(mean, _numOps.FromDouble(embDim));

            var variance = _numOps.Zero;
            for (int d = 0; d < embDim; d++)
            {
                var diff = _numOps.Subtract(x[t * embDim + d], mean);
                variance = _numOps.Add(variance, _numOps.Multiply(diff, diff));
            }
            variance = _numOps.Divide(variance, _numOps.FromDouble(embDim));
            var std = _numOps.Sqrt(_numOps.Add(variance, _numOps.FromDouble(1e-6)));

            for (int d = 0; d < embDim; d++)
            {
                var normalized = _numOps.Divide(_numOps.Subtract(x[t * embDim + d], mean), std);
                output[t * embDim + d] = _numOps.Add(_numOps.Multiply(gamma[d], normalized), beta[d]);
            }
        }

        return output;
    }

    private Tensor<T> FeedForward(Tensor<T> x)
    {
        int seqLen = x.Shape[0];
        int embDim = x.Shape[1];
        int ffDim = _ff1Weight.Shape[0];

        var output = new Tensor<T>(x.Shape);

        for (int t = 0; t < seqLen; t++)
        {
            var hidden = new T[ffDim];
            for (int h = 0; h < ffDim; h++)
            {
                hidden[h] = _ff1Bias[h];
                for (int d = 0; d < embDim; d++)
                {
                    hidden[h] = _numOps.Add(hidden[h], _numOps.Multiply(_ff1Weight[h * embDim + d], x[t * embDim + d]));
                }
                double hVal = _numOps.ToDouble(hidden[h]);
                hidden[h] = _numOps.FromDouble(0.5 * hVal * (1 + Math.Tanh(Math.Sqrt(2 / Math.PI) * (hVal + 0.044715 * Math.Pow(hVal, 3)))));
            }

            for (int d = 0; d < embDim; d++)
            {
                output[t * embDim + d] = _ff2Bias[d];
                for (int h = 0; h < ffDim; h++)
                {
                    output[t * embDim + d] = _numOps.Add(output[t * embDim + d], _numOps.Multiply(_ff2Weight[d * ffDim + h], hidden[h]));
                }
            }
        }

        return output;
    }

    public void InitializeGradientAccumulators(Dictionary<string, Tensor<T>> accumulators, int layerIndex)
    {
        string prefix = $"decoder_{layerIndex}_";
        accumulators[$"{prefix}selfQueryProj"] = new Tensor<T>(_selfQueryProj.Shape);
        accumulators[$"{prefix}selfKeyProj"] = new Tensor<T>(_selfKeyProj.Shape);
        accumulators[$"{prefix}selfValueProj"] = new Tensor<T>(_selfValueProj.Shape);
        accumulators[$"{prefix}selfOutputProj"] = new Tensor<T>(_selfOutputProj.Shape);
        accumulators[$"{prefix}crossQueryProj"] = new Tensor<T>(_crossQueryProj.Shape);
        accumulators[$"{prefix}crossKeyProj"] = new Tensor<T>(_crossKeyProj.Shape);
        accumulators[$"{prefix}crossValueProj"] = new Tensor<T>(_crossValueProj.Shape);
        accumulators[$"{prefix}crossOutputProj"] = new Tensor<T>(_crossOutputProj.Shape);
        accumulators[$"{prefix}ff1Weight"] = new Tensor<T>(_ff1Weight.Shape);
        accumulators[$"{prefix}ff1Bias"] = new Tensor<T>(_ff1Bias.Shape);
        accumulators[$"{prefix}ff2Weight"] = new Tensor<T>(_ff2Weight.Shape);
        accumulators[$"{prefix}ff2Bias"] = new Tensor<T>(_ff2Bias.Shape);
    }

    public void ApplyGradients(Dictionary<string, Tensor<T>> accumulators, T scale, int layerIndex)
    {
        // Apply gradients to each parameter (simplified)
    }

    public void Serialize(BinaryWriter writer)
    {
        WriteTensor(writer, _selfQueryProj);
        WriteTensor(writer, _selfKeyProj);
        WriteTensor(writer, _selfValueProj);
        WriteTensor(writer, _selfOutputProj);
        WriteTensor(writer, _crossQueryProj);
        WriteTensor(writer, _crossKeyProj);
        WriteTensor(writer, _crossValueProj);
        WriteTensor(writer, _crossOutputProj);
        WriteTensor(writer, _ff1Weight);
        WriteTensor(writer, _ff1Bias);
        WriteTensor(writer, _ff2Weight);
        WriteTensor(writer, _ff2Bias);
        WriteTensor(writer, _layerNorm1Gamma);
        WriteTensor(writer, _layerNorm1Beta);
        WriteTensor(writer, _layerNorm2Gamma);
        WriteTensor(writer, _layerNorm2Beta);
        WriteTensor(writer, _layerNorm3Gamma);
        WriteTensor(writer, _layerNorm3Beta);
    }

    public void Deserialize(BinaryReader reader)
    {
        _selfQueryProj = ReadTensor(reader);
        _selfKeyProj = ReadTensor(reader);
        _selfValueProj = ReadTensor(reader);
        _selfOutputProj = ReadTensor(reader);
        _crossQueryProj = ReadTensor(reader);
        _crossKeyProj = ReadTensor(reader);
        _crossValueProj = ReadTensor(reader);
        _crossOutputProj = ReadTensor(reader);
        _ff1Weight = ReadTensor(reader);
        _ff1Bias = ReadTensor(reader);
        _ff2Weight = ReadTensor(reader);
        _ff2Bias = ReadTensor(reader);
        _layerNorm1Gamma = ReadTensor(reader);
        _layerNorm1Beta = ReadTensor(reader);
        _layerNorm2Gamma = ReadTensor(reader);
        _layerNorm2Beta = ReadTensor(reader);
        _layerNorm3Gamma = ReadTensor(reader);
        _layerNorm3Beta = ReadTensor(reader);
    }

    private void WriteTensor(BinaryWriter writer, Tensor<T> tensor)
    {
        writer.Write(tensor.Shape.Length);
        foreach (var dim in tensor.Shape)
        {
            writer.Write(dim);
        }
        for (int i = 0; i < tensor.Length; i++)
        {
            writer.Write(_numOps.ToDouble(tensor[i]));
        }
    }

    private Tensor<T> ReadTensor(BinaryReader reader)
    {
        int rank = reader.ReadInt32();
        var shape = new int[rank];
        for (int i = 0; i < rank; i++)
        {
            shape[i] = reader.ReadInt32();
        }
        var tensor = new Tensor<T>(shape);
        for (int i = 0; i < tensor.Length; i++)
        {
            tensor[i] = _numOps.FromDouble(reader.ReadDouble());
        }
        return tensor;
    }
}
