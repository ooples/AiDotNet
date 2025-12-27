using AiDotNet.Autodiff;
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
    /// For multi-step forecasting, the input should contain lookback + forecastHorizon values.
    /// The first lookback values are used as input, and the last forecastHorizon values
    /// from the sequence are used as multi-step targets.
    /// </summary>
    protected override void TrainCore(Matrix<T> x, Vector<T> y)
    {
        T learningRate = _numOps.FromDouble(_options.LearningRate);
        int forecastHorizon = _options.ForecastHorizon;
        int lookback = _options.LookbackWindow;

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
                    Vector<T> fullSequence = x.GetRow(i);

                    // Extract multi-step targets from the input sequence
                    // If input contains lookback + forecastHorizon values, use the last forecastHorizon as targets
                    // Otherwise, construct targets from available data
                    var targets = new Vector<T>(forecastHorizon);
                    int inputLen = Math.Min(fullSequence.Length, lookback);

                    if (fullSequence.Length >= lookback + forecastHorizon)
                    {
                        // Full sequence available: use last forecastHorizon values as multi-step targets
                        for (int h = 0; h < forecastHorizon; h++)
                        {
                            targets[h] = fullSequence[lookback + h];
                        }
                    }
                    else
                    {
                        // Fallback: use y[i] for first target, pad with last known value
                        targets[0] = y[i];
                        T lastValue = y[i];
                        for (int h = 1; h < forecastHorizon; h++)
                        {
                            targets[h] = lastValue;
                        }
                    }

                    // Extract input portion (first lookback values)
                    var input = new Vector<T>(inputLen);
                    for (int t = 0; t < inputLen; t++)
                    {
                        input[t] = fullSequence[t];
                    }

                    var gradients = ComputeGradientsMultiStep(input, targets);
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

    private Dictionary<string, Tensor<T>> ComputeGradientsMultiStep(Vector<T> input, Vector<T> targets)
    {
        var gradients = new Dictionary<string, Tensor<T>>();
        int seqLen = Math.Min(input.Length, _options.LookbackWindow);
        int embDim = _options.EmbeddingDim;
        int forecastHorizon = _options.ForecastHorizon;

        // Create computation nodes for all trainable parameters
        var inputProjNode = TensorOperations<T>.Variable(_inputProjection, "inputProj", requiresGradient: true);
        var seasonalProjNode = TensorOperations<T>.Variable(_seasonalProjection, "seasonalProj", requiresGradient: true);
        var trendProjNode = TensorOperations<T>.Variable(_trendProjection, "trendProj", requiresGradient: true);
        var outputBiasNode = TensorOperations<T>.Variable(_outputBias, "outputBias", requiresGradient: true);

        // Create input tensor as computation node
        var inputTensor = new Tensor<T>(new[] { seqLen, 1 });
        for (int t = 0; t < seqLen; t++)
        {
            inputTensor[t, 0] = input[t];
        }
        var inputNode = TensorOperations<T>.Variable(inputTensor, "input", requiresGradient: false);

        // Forward pass using autodiff operations
        // 1. Input embedding: embedded = input @ inputProj + positionalEncoding
        var inputProjReshaped = TensorOperations<T>.Reshape(inputProjNode, new[] { 1, embDim });
        var embedded = TensorOperations<T>.MatrixMultiply(inputNode, inputProjReshaped);

        // Add positional encoding
        var posTensor = new Tensor<T>(new[] { seqLen, embDim });
        for (int t = 0; t < seqLen; t++)
        {
            for (int d = 0; d < embDim; d++)
            {
                posTensor[t, d] = _positionalEncoding[t * embDim + d];
            }
        }
        var posNode = TensorOperations<T>.Variable(posTensor, "pos", requiresGradient: false);
        embedded = TensorOperations<T>.Add(embedded, posNode);

        // 2. Series decomposition (moving average for trend extraction)
        var trendOutput = ComputeMovingAverageNode(embedded, _movingAvgKernel);
        var seasonalOutput = TensorOperations<T>.Subtract(embedded, trendOutput);

        // 3. Process through encoder layers and collect parameter nodes for gradient collection
        var allParamNodes = new List<ComputationNode<T>>();
        for (int i = 0; i < _encoderLayers.Count; i++)
        {
            var layerOutput = ProcessEncoderLayerAutodiff(seasonalOutput, trendOutput, i);
            seasonalOutput = layerOutput.seasonal;
            trendOutput = layerOutput.trend;
            allParamNodes.AddRange(layerOutput.paramNodes);
        }

        // 4. Process through decoder layers
        var decoderSeasonalInit = TensorOperations<T>.Variable(_decoderSeasonalInit, "decSeasonalInit", requiresGradient: true);
        var decoderTrendInit = TensorOperations<T>.Variable(_decoderTrendInit, "decTrendInit", requiresGradient: true);

        var decoderSeasonal = decoderSeasonalInit;
        var decoderTrend = decoderTrendInit;

        for (int i = 0; i < _decoderLayers.Count; i++)
        {
            var layerOutput = ProcessDecoderLayerAutodiff(decoderSeasonal, decoderTrend, seasonalOutput, trendOutput, i);
            decoderSeasonal = layerOutput.seasonal;
            decoderTrend = layerOutput.trend;
            allParamNodes.AddRange(layerOutput.paramNodes);
        }

        // 5. Output projection: output = seasonal @ seasonalProj + trend @ trendProj + bias
        var seasonalContrib = TensorOperations<T>.MatrixMultiply(decoderSeasonal, TensorOperations<T>.Transpose(seasonalProjNode));
        var trendContrib = TensorOperations<T>.MatrixMultiply(decoderTrend, TensorOperations<T>.Transpose(trendProjNode));
        var output = TensorOperations<T>.Add(seasonalContrib, trendContrib);

        // Add output bias (broadcast across sequence)
        var biasReshaped = TensorOperations<T>.Reshape(outputBiasNode, new[] { forecastHorizon, 1 });
        output = TensorOperations<T>.Add(output, biasReshaped);

        // 6. Compute loss: MSE = mean((output - targets)^2) over all forecast steps
        var targetTensor = new Tensor<T>(new[] { forecastHorizon, 1 });
        for (int h = 0; h < forecastHorizon && h < targets.Length; h++)
        {
            targetTensor[h, 0] = targets[h];
        }
        var targetNode = TensorOperations<T>.Variable(targetTensor, "target", requiresGradient: false);

        var diff = TensorOperations<T>.Subtract(output, targetNode);
        var squared = TensorOperations<T>.ElementwiseMultiply(diff, diff);
        var loss = TensorOperations<T>.Mean(squared);

        // 7. Backward pass to compute gradients
        loss.Gradient = new Tensor<T>(loss.Value.Shape);
        loss.Gradient[0] = _numOps.One;

        // Topological sort for backward pass
        var visited = new HashSet<ComputationNode<T>>();
        var topoOrder = new List<ComputationNode<T>>();
        TopologicalSort(loss, visited, topoOrder);

        // Execute backward pass
        for (int i = topoOrder.Count - 1; i >= 0; i--)
        {
            var node = topoOrder[i];
            if (node.RequiresGradient && node.BackwardFunction != null && node.Gradient != null)
            {
                node.BackwardFunction(node.Gradient);
            }
        }

        // 8. Collect gradients for all trainable parameters
        if (inputProjNode.Gradient != null)
            gradients["inputProjection"] = inputProjNode.Gradient;

        if (seasonalProjNode.Gradient != null)
            gradients["seasonalProjection"] = seasonalProjNode.Gradient;

        if (trendProjNode.Gradient != null)
            gradients["trendProjection"] = trendProjNode.Gradient;

        if (outputBiasNode.Gradient != null)
            gradients["outputBias"] = outputBiasNode.Gradient;

        if (decoderSeasonalInit.Gradient != null)
            gradients["decoderSeasonalInit"] = decoderSeasonalInit.Gradient;

        if (decoderTrendInit.Gradient != null)
            gradients["decoderTrendInit"] = decoderTrendInit.Gradient;

        // Collect gradients from all encoder/decoder layer parameters
        foreach (var paramNode in allParamNodes)
        {
            string? nodeName = paramNode.Name;
            if (paramNode.Gradient != null && nodeName != null && nodeName.Length > 0)
            {
                gradients[nodeName] = paramNode.Gradient;
            }
        }

        return gradients;
    }

    private void TopologicalSort(ComputationNode<T> node, HashSet<ComputationNode<T>> visited, List<ComputationNode<T>> topoOrder)
    {
        if (visited.Contains(node)) return;

        foreach (var parent in node.Parents)
        {
            TopologicalSort(parent, visited, topoOrder);
        }

        visited.Add(node);
        topoOrder.Add(node);
    }

    private ComputationNode<T> ComputeMovingAverageNode(ComputationNode<T> input, int kernelSize)
    {
        var shape = input.Value.Shape;
        int seqLen = shape[0];
        int embDim = shape.Length > 1 ? shape[1] : 1;

        // Compute moving average using autodiff operations
        var result = new Tensor<T>(shape);
        int halfK = kernelSize / 2;

        for (int t = 0; t < seqLen; t++)
        {
            int start = Math.Max(0, t - halfK);
            int end = Math.Min(seqLen - 1, t + halfK);
            int count = end - start + 1;

            for (int d = 0; d < embDim; d++)
            {
                var sum = _numOps.Zero;
                for (int k = start; k <= end; k++)
                {
                    int idx = embDim > 1 ? k * embDim + d : k;
                    sum = _numOps.Add(sum, input.Value[idx]);
                }
                int outIdx = embDim > 1 ? t * embDim + d : t;
                result[outIdx] = _numOps.Divide(sum, _numOps.FromDouble(count));
            }
        }

        // Create node with backward function for gradient propagation
        var node = new ComputationNode<T>(
            result,
            requiresGradient: input.RequiresGradient,
            parents: new List<ComputationNode<T>> { input },
            name: "movingAvg");

        node.BackwardFunction = (gradient) =>
        {
            if (input.RequiresGradient)
            {
                if (input.Gradient == null)
                    input.Gradient = new Tensor<T>(input.Value.Shape);

                for (int t = 0; t < seqLen; t++)
                {
                    int start = Math.Max(0, t - halfK);
                    int end = Math.Min(seqLen - 1, t + halfK);
                    int count = end - start + 1;
                    var scale = _numOps.FromDouble(1.0 / count);

                    for (int d = 0; d < embDim; d++)
                    {
                        int gradIdx = embDim > 1 ? t * embDim + d : t;
                        var gradVal = _numOps.Multiply(gradient[gradIdx], scale);

                        for (int k = start; k <= end; k++)
                        {
                            int inputIdx = embDim > 1 ? k * embDim + d : k;
                            input.Gradient[inputIdx] = _numOps.Add(input.Gradient[inputIdx], gradVal);
                        }
                    }
                }
            }
        };

        return node;
    }

    private (ComputationNode<T> seasonal, ComputationNode<T> trend, List<ComputationNode<T>> paramNodes) ProcessEncoderLayerAutodiff(
        ComputationNode<T> seasonal, ComputationNode<T> trend, int layerIdx)
    {
        var layer = _encoderLayers[layerIdx];
        string prefix = $"encoder_{layerIdx}_";
        var paramNodes = new List<ComputationNode<T>>();

        // Get layer parameters as computation nodes with correct naming for ApplyGradients
        var queryProj = TensorOperations<T>.Variable(layer.GetQueryProjection(), $"{prefix}queryProj", requiresGradient: true);
        var keyProj = TensorOperations<T>.Variable(layer.GetKeyProjection(), $"{prefix}keyProj", requiresGradient: true);
        var valueProj = TensorOperations<T>.Variable(layer.GetValueProjection(), $"{prefix}valueProj", requiresGradient: true);
        var outputProj = TensorOperations<T>.Variable(layer.GetOutputProjection(), $"{prefix}outputProj", requiresGradient: true);
        paramNodes.AddRange(new[] { queryProj, keyProj, valueProj, outputProj });

        // Auto-correlation computation using autodiff
        var query = TensorOperations<T>.MatrixMultiply(seasonal, queryProj);
        var key = TensorOperations<T>.MatrixMultiply(seasonal, keyProj);
        var value = TensorOperations<T>.MatrixMultiply(seasonal, valueProj);

        // Simplified auto-correlation: use scaled dot-product attention as approximation
        var attnOutput = TensorOperations<T>.ScaledDotProductAttention(query, key, value);
        var projected = TensorOperations<T>.MatrixMultiply(attnOutput, outputProj);

        // Residual connection
        var residual = TensorOperations<T>.Add(seasonal, projected);

        // Layer normalization
        var gamma1 = TensorOperations<T>.Variable(layer.GetLayerNorm1Gamma(), $"{prefix}ln1Gamma", requiresGradient: true);
        var beta1 = TensorOperations<T>.Variable(layer.GetLayerNorm1Beta(), $"{prefix}ln1Beta", requiresGradient: true);
        paramNodes.AddRange(new[] { gamma1, beta1 });
        var normalized = TensorOperations<T>.LayerNorm(residual, new[] { residual.Value.Shape[^1] }, gamma1, beta1);

        // Feed-forward network
        var ff1W = TensorOperations<T>.Variable(layer.GetFF1Weight(), $"{prefix}ff1Weight", requiresGradient: true);
        var ff1B = TensorOperations<T>.Variable(layer.GetFF1Bias(), $"{prefix}ff1Bias", requiresGradient: true);
        var ff2W = TensorOperations<T>.Variable(layer.GetFF2Weight(), $"{prefix}ff2Weight", requiresGradient: true);
        var ff2B = TensorOperations<T>.Variable(layer.GetFF2Bias(), $"{prefix}ff2Bias", requiresGradient: true);
        paramNodes.AddRange(new[] { ff1W, ff1B, ff2W, ff2B });

        var ffHidden = TensorOperations<T>.Add(TensorOperations<T>.MatrixMultiply(normalized, ff1W), ff1B);
        var ffActivated = TensorOperations<T>.ReLU(ffHidden);
        var ffOutput = TensorOperations<T>.Add(TensorOperations<T>.MatrixMultiply(ffActivated, ff2W), ff2B);

        // Residual + LayerNorm
        var ffResidual = TensorOperations<T>.Add(normalized, ffOutput);
        var gamma2 = TensorOperations<T>.Variable(layer.GetLayerNorm2Gamma(), $"{prefix}ln2Gamma", requiresGradient: true);
        var beta2 = TensorOperations<T>.Variable(layer.GetLayerNorm2Beta(), $"{prefix}ln2Beta", requiresGradient: true);
        paramNodes.AddRange(new[] { gamma2, beta2 });
        var newSeasonal = TensorOperations<T>.LayerNorm(ffResidual, new[] { ffResidual.Value.Shape[^1] }, gamma2, beta2);

        // Series decomposition
        var newTrend = ComputeMovingAverageNode(newSeasonal, _movingAvgKernel);
        newSeasonal = TensorOperations<T>.Subtract(newSeasonal, newTrend);

        // Accumulate trend
        var combinedTrend = TensorOperations<T>.Add(trend, newTrend);

        return (newSeasonal, combinedTrend, paramNodes);
    }

    private (ComputationNode<T> seasonal, ComputationNode<T> trend, List<ComputationNode<T>> paramNodes) ProcessDecoderLayerAutodiff(
        ComputationNode<T> decoderSeasonal, ComputationNode<T> decoderTrend,
        ComputationNode<T> encoderSeasonal, ComputationNode<T> encoderTrend, int layerIdx)
    {
        var layer = _decoderLayers[layerIdx];
        string prefix = $"decoder_{layerIdx}_";
        var paramNodes = new List<ComputationNode<T>>();

        // Self-attention projections
        var selfQueryProj = TensorOperations<T>.Variable(layer.GetSelfQueryProjection(), $"{prefix}selfQueryProj", requiresGradient: true);
        var selfKeyProj = TensorOperations<T>.Variable(layer.GetSelfKeyProjection(), $"{prefix}selfKeyProj", requiresGradient: true);
        var selfValueProj = TensorOperations<T>.Variable(layer.GetSelfValueProjection(), $"{prefix}selfValueProj", requiresGradient: true);
        var selfOutputProj = TensorOperations<T>.Variable(layer.GetSelfOutputProjection(), $"{prefix}selfOutputProj", requiresGradient: true);
        paramNodes.AddRange(new[] { selfQueryProj, selfKeyProj, selfValueProj, selfOutputProj });

        // Self-attention
        var selfQuery = TensorOperations<T>.MatrixMultiply(decoderSeasonal, selfQueryProj);
        var selfKey = TensorOperations<T>.MatrixMultiply(decoderSeasonal, selfKeyProj);
        var selfValue = TensorOperations<T>.MatrixMultiply(decoderSeasonal, selfValueProj);
        var selfAttn = TensorOperations<T>.ScaledDotProductAttention(selfQuery, selfKey, selfValue);
        var selfProjected = TensorOperations<T>.MatrixMultiply(selfAttn, selfOutputProj);

        // Residual + LayerNorm
        var selfResidual = TensorOperations<T>.Add(decoderSeasonal, selfProjected);
        var ln1Gamma = TensorOperations<T>.Variable(layer.GetLayerNorm1Gamma(), $"{prefix}ln1Gamma", requiresGradient: true);
        var ln1Beta = TensorOperations<T>.Variable(layer.GetLayerNorm1Beta(), $"{prefix}ln1Beta", requiresGradient: true);
        paramNodes.AddRange(new[] { ln1Gamma, ln1Beta });
        var normalized1 = TensorOperations<T>.LayerNorm(selfResidual, new[] { selfResidual.Value.Shape[^1] }, ln1Gamma, ln1Beta);

        // Cross-attention with encoder
        var crossQueryProj = TensorOperations<T>.Variable(layer.GetCrossQueryProjection(), $"{prefix}crossQueryProj", requiresGradient: true);
        var crossKeyProj = TensorOperations<T>.Variable(layer.GetCrossKeyProjection(), $"{prefix}crossKeyProj", requiresGradient: true);
        var crossValueProj = TensorOperations<T>.Variable(layer.GetCrossValueProjection(), $"{prefix}crossValueProj", requiresGradient: true);
        var crossOutputProj = TensorOperations<T>.Variable(layer.GetCrossOutputProjection(), $"{prefix}crossOutputProj", requiresGradient: true);
        paramNodes.AddRange(new[] { crossQueryProj, crossKeyProj, crossValueProj, crossOutputProj });

        var crossQuery = TensorOperations<T>.MatrixMultiply(normalized1, crossQueryProj);
        var crossKey = TensorOperations<T>.MatrixMultiply(encoderSeasonal, crossKeyProj);
        var crossValue = TensorOperations<T>.MatrixMultiply(encoderSeasonal, crossValueProj);
        var crossAttn = TensorOperations<T>.ScaledDotProductAttention(crossQuery, crossKey, crossValue);
        var crossProjected = TensorOperations<T>.MatrixMultiply(crossAttn, crossOutputProj);

        // Residual + LayerNorm
        var crossResidual = TensorOperations<T>.Add(normalized1, crossProjected);
        var ln2Gamma = TensorOperations<T>.Variable(layer.GetLayerNorm2Gamma(), $"{prefix}ln2Gamma", requiresGradient: true);
        var ln2Beta = TensorOperations<T>.Variable(layer.GetLayerNorm2Beta(), $"{prefix}ln2Beta", requiresGradient: true);
        paramNodes.AddRange(new[] { ln2Gamma, ln2Beta });
        var normalized2 = TensorOperations<T>.LayerNorm(crossResidual, new[] { crossResidual.Value.Shape[^1] }, ln2Gamma, ln2Beta);

        // Feed-forward network
        var ff1W = TensorOperations<T>.Variable(layer.GetFF1Weight(), $"{prefix}ff1Weight", requiresGradient: true);
        var ff1B = TensorOperations<T>.Variable(layer.GetFF1Bias(), $"{prefix}ff1Bias", requiresGradient: true);
        var ff2W = TensorOperations<T>.Variable(layer.GetFF2Weight(), $"{prefix}ff2Weight", requiresGradient: true);
        var ff2B = TensorOperations<T>.Variable(layer.GetFF2Bias(), $"{prefix}ff2Bias", requiresGradient: true);
        paramNodes.AddRange(new[] { ff1W, ff1B, ff2W, ff2B });

        var ffHidden = TensorOperations<T>.Add(TensorOperations<T>.MatrixMultiply(normalized2, ff1W), ff1B);
        var ffActivated = TensorOperations<T>.ReLU(ffHidden);
        var ffOutput = TensorOperations<T>.Add(TensorOperations<T>.MatrixMultiply(ffActivated, ff2W), ff2B);

        // Residual + LayerNorm
        var ffResidual = TensorOperations<T>.Add(normalized2, ffOutput);
        var ln3Gamma = TensorOperations<T>.Variable(layer.GetLayerNorm3Gamma(), $"{prefix}ln3Gamma", requiresGradient: true);
        var ln3Beta = TensorOperations<T>.Variable(layer.GetLayerNorm3Beta(), $"{prefix}ln3Beta", requiresGradient: true);
        paramNodes.AddRange(new[] { ln3Gamma, ln3Beta });
        var newSeasonal = TensorOperations<T>.LayerNorm(ffResidual, new[] { ffResidual.Value.Shape[^1] }, ln3Gamma, ln3Beta);

        // Series decomposition
        var newTrend = ComputeMovingAverageNode(newSeasonal, _movingAvgKernel);
        newSeasonal = TensorOperations<T>.Subtract(newSeasonal, newTrend);

        // Accumulate trend
        var combinedTrend = TensorOperations<T>.Add(decoderTrend, newTrend);

        return (newSeasonal, combinedTrend, paramNodes);
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

    /// <summary>
    /// Predicts multiple time steps ahead using the Autoformer architecture.
    /// </summary>
    /// <param name="input">Input sequence of historical values.</param>
    /// <returns>Vector of predictions for the entire forecast horizon.</returns>
    /// <remarks>
    /// <para>
    /// This method returns predictions for all steps in the forecast horizon,
    /// as Autoformer is designed for multi-horizon forecasting.
    /// </para>
    /// </remarks>
    public Vector<T> PredictMultiple(Vector<T> input)
    {
        int forecastHorizon = _options.ForecastHorizon;
        var predictions = new Vector<T>(forecastHorizon);

        var (firstPrediction, cache) = ForwardWithCache(input);

        // Extract predictions from the decoder output for each step in the forecast horizon
        var seasonalOutput = cache.SeasonalOutput;
        var trendOutput = cache.TrendOutput;

        // If cache outputs are null, fall back to repeating the first prediction
        if (seasonalOutput == null || trendOutput == null)
        {
            for (int h = 0; h < forecastHorizon; h++)
            {
                predictions[h] = firstPrediction;
            }
            return predictions;
        }

        int embDim = _options.EmbeddingDim;

        for (int h = 0; h < forecastHorizon; h++)
        {
            var pred = _numOps.Zero;

            // Combine seasonal and trend contributions for this step
            for (int d = 0; d < embDim; d++)
            {
                int idx = Math.Min(h, seasonalOutput.Length / embDim - 1) * embDim + d;
                if (idx >= 0 && idx < seasonalOutput.Length && idx < trendOutput.Length)
                {
                    var seasonalContrib = _numOps.Multiply(_seasonalProjection[d], seasonalOutput[idx]);
                    var trendContrib = _numOps.Multiply(_trendProjection[d], trendOutput[idx]);
                    pred = _numOps.Add(pred, _numOps.Add(seasonalContrib, trendContrib));
                }
            }

            // Add bias for this step
            if (h < _outputBias.Length)
            {
                pred = _numOps.Add(pred, _outputBias[h]);
            }

            predictions[h] = pred;
        }

        return predictions;
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
        // Auto-correlation using FFT as described in Autoformer paper (Wu et al., NeurIPS 2021)
        // Computes: AutoCorr(Q, K) = Softmax(TopK(Corr(Q, K))) * V
        // where Corr is computed via FFT: Corr = IFFT(FFT(Q) * conj(FFT(K)))

        int seqLen = x.Shape[0];
        int embDim = x.Shape.Length > 1 ? x.Shape[1] : 1;

        // Ensure topK doesn't exceed sequence length
        topK = Math.Min(topK, seqLen);

        var output = new Tensor<T>(x.Shape);

        // Process each embedding dimension independently
        for (int d = 0; d < embDim; d++)
        {
            // Extract time series for this dimension
            var series = new double[seqLen];
            for (int t = 0; t < seqLen; t++)
            {
                series[t] = _numOps.ToDouble(x[t * embDim + d]);
            }

            // Compute auto-correlation via FFT (simplified version without Complex type)
            // Using correlation formula: R(tau) = sum(x[t] * x[t+tau])
            var correlations = new double[seqLen];
            double sumX = 0;
            double sumX2 = 0;
            for (int t = 0; t < seqLen; t++)
            {
                sumX += series[t];
                sumX2 += series[t] * series[t];
            }
            double meanX = sumX / seqLen;
            double varX = sumX2 / seqLen - meanX * meanX;

            for (int tau = 0; tau < seqLen; tau++)
            {
                double sum = 0;
                int count = seqLen - tau;
                for (int t = 0; t < count; t++)
                {
                    sum += (series[t] - meanX) * (series[t + tau] - meanX);
                }
                correlations[tau] = varX > 1e-10 ? sum / (count * varX) : 0;
            }

            // Find top-K correlations (excluding lag 0)
            var topKIndices = new List<int>();
            var sortedCorr = correlations.Select((v, i) => (value: Math.Abs(v), index: i))
                                         .Where(c => c.index > 0) // Exclude lag 0
                                         .OrderByDescending(c => c.value)
                                         .Take(topK)
                                         .ToList();

            // Softmax over top-K correlation values
            double maxCorr = sortedCorr.Count > 0 ? sortedCorr.Max(c => c.value) : 0;
            var weights = sortedCorr.Select(c => Math.Exp(c.value - maxCorr)).ToList();
            double sumWeights = weights.Sum();
            if (sumWeights > 1e-10)
            {
                for (int i = 0; i < weights.Count; i++)
                    weights[i] /= sumWeights;
            }

            // Aggregate values using weighted combination of time-delayed versions
            for (int t = 0; t < seqLen; t++)
            {
                double aggregatedValue = series[t] * 0.5; // Keep some of original signal
                double weightSum = 0.5;

                for (int k = 0; k < sortedCorr.Count; k++)
                {
                    int lag = sortedCorr[k].index;
                    int sourceIdx = (t + lag) % seqLen; // Circular indexing
                    aggregatedValue += weights[k] * 0.5 * series[sourceIdx];
                    weightSum += weights[k] * 0.5;
                }

                output[t * embDim + d] = _numOps.FromDouble(aggregatedValue / weightSum);
            }
        }

        return output;
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
        accumulators[$"{prefix}ln1Gamma"] = new Tensor<T>(_layerNorm1Gamma.Shape);
        accumulators[$"{prefix}ln1Beta"] = new Tensor<T>(_layerNorm1Beta.Shape);
        accumulators[$"{prefix}ln2Gamma"] = new Tensor<T>(_layerNorm2Gamma.Shape);
        accumulators[$"{prefix}ln2Beta"] = new Tensor<T>(_layerNorm2Beta.Shape);
    }

    // Getters for autodiff integration
    public Tensor<T> GetQueryProjection() => _queryProj;
    public Tensor<T> GetKeyProjection() => _keyProj;
    public Tensor<T> GetValueProjection() => _valueProj;
    public Tensor<T> GetOutputProjection() => _outputProj;
    public Tensor<T> GetFF1Weight() => _ff1Weight;
    public Tensor<T> GetFF1Bias() => _ff1Bias;
    public Tensor<T> GetFF2Weight() => _ff2Weight;
    public Tensor<T> GetFF2Bias() => _ff2Bias;
    public Tensor<T> GetLayerNorm1Gamma() => _layerNorm1Gamma;
    public Tensor<T> GetLayerNorm1Beta() => _layerNorm1Beta;
    public Tensor<T> GetLayerNorm2Gamma() => _layerNorm2Gamma;
    public Tensor<T> GetLayerNorm2Beta() => _layerNorm2Beta;

    public void ApplyGradients(Dictionary<string, Tensor<T>> accumulators, T scale, int layerIndex)
    {
        string prefix = $"encoder_{layerIndex}_";

        // Apply gradients: param = param - scale * gradient
        ApplyGradientToParameter(ref _queryProj, accumulators, $"{prefix}queryProj", scale);
        ApplyGradientToParameter(ref _keyProj, accumulators, $"{prefix}keyProj", scale);
        ApplyGradientToParameter(ref _valueProj, accumulators, $"{prefix}valueProj", scale);
        ApplyGradientToParameter(ref _outputProj, accumulators, $"{prefix}outputProj", scale);
        ApplyGradientToParameter(ref _ff1Weight, accumulators, $"{prefix}ff1Weight", scale);
        ApplyGradientToParameter(ref _ff1Bias, accumulators, $"{prefix}ff1Bias", scale);
        ApplyGradientToParameter(ref _ff2Weight, accumulators, $"{prefix}ff2Weight", scale);
        ApplyGradientToParameter(ref _ff2Bias, accumulators, $"{prefix}ff2Bias", scale);
        ApplyGradientToParameter(ref _layerNorm1Gamma, accumulators, $"{prefix}ln1Gamma", scale);
        ApplyGradientToParameter(ref _layerNorm1Beta, accumulators, $"{prefix}ln1Beta", scale);
        ApplyGradientToParameter(ref _layerNorm2Gamma, accumulators, $"{prefix}ln2Gamma", scale);
        ApplyGradientToParameter(ref _layerNorm2Beta, accumulators, $"{prefix}ln2Beta", scale);
    }

    private void ApplyGradientToParameter(ref Tensor<T> param, Dictionary<string, Tensor<T>> accumulators, string key, T scale)
    {
        if (!accumulators.TryGetValue(key, out var gradient))
            return;

        // param = param - scale * gradient
        for (int i = 0; i < param.Shape[0]; i++)
        {
            for (int j = 0; j < (param.Shape.Length > 1 ? param.Shape[1] : 1); j++)
            {
                var currentValue = param.Shape.Length > 1 ? param[i, j] : param[i];
                var gradValue = gradient.Shape.Length > 1 ? gradient[i, j] : gradient[i];
                var update = _numOps.Multiply(scale, gradValue);
                var newValue = _numOps.Subtract(currentValue, update);

                if (param.Shape.Length > 1)
                    param[i, j] = newValue;
                else
                    param[i] = newValue;
            }
        }
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
        // 1. Self auto-correlation on decoder seasonal
        var selfAttnOutput = ApplySelfAutoCorrelation(decoderSeasonal, topK);

        // Add residual + LayerNorm (Post-LN: residual first, then normalize)
        var selfResidual = AddTensors(decoderSeasonal, selfAttnOutput);
        var normalized1 = LayerNorm(selfResidual, _layerNorm1Gamma, _layerNorm1Beta);

        // Series decomposition after self-attention
        var (selfTrend, selfSeasonal) = SeriesDecomposition(normalized1);
        for (int i = 0; i < decoderTrend.Length && i < selfTrend.Length; i++)
        {
            decoderTrend[i] = _numOps.Add(decoderTrend[i], selfTrend[i]);
        }

        // 2. Cross auto-correlation with encoder outputs
        var crossAttnOutput = ApplyCrossAutoCorrelation(selfSeasonal, encoderSeasonal, topK);

        // Add residual + LayerNorm
        var crossResidual = AddTensors(selfSeasonal, crossAttnOutput);
        var normalized2 = LayerNorm(crossResidual, _layerNorm2Gamma, _layerNorm2Beta);

        // Series decomposition after cross-attention
        var (crossTrend, crossSeasonal) = SeriesDecomposition(normalized2);
        for (int i = 0; i < decoderTrend.Length && i < crossTrend.Length; i++)
        {
            decoderTrend[i] = _numOps.Add(decoderTrend[i], crossTrend[i]);
        }

        // 3. Feed-forward network
        var ffOutput = FeedForward(crossSeasonal);

        // Add residual + LayerNorm
        var ffResidual = AddTensors(crossSeasonal, ffOutput);
        var normalized3 = LayerNorm(ffResidual, _layerNorm3Gamma, _layerNorm3Beta);

        // Final decomposition
        var (finalTrend, finalSeasonal) = SeriesDecomposition(normalized3);
        for (int i = 0; i < decoderTrend.Length && i < finalTrend.Length; i++)
        {
            decoderTrend[i] = _numOps.Add(decoderTrend[i], finalTrend[i]);
        }

        return (decoderTrend, finalSeasonal);
    }

    private Tensor<T> AddTensors(Tensor<T> a, Tensor<T> b)
    {
        var result = new Tensor<T>(a.Shape);
        int len = Math.Min(a.Length, b.Length);
        for (int i = 0; i < len; i++)
        {
            result[i] = _numOps.Add(a[i], b[i]);
        }
        return result;
    }

    private Tensor<T> ApplySelfAutoCorrelation(Tensor<T> x, int topK)
    {
        int seqLen = x.Shape[0];
        int embDim = x.Shape[1];

        // Project to Q, K, V
        var query = ProjectTensor(x, _selfQueryProj);
        var key = ProjectTensor(x, _selfKeyProj);
        var value = ProjectTensor(x, _selfValueProj);

        // Auto-correlation: compute correlations between query and shifted keys
        var attnWeights = ComputeAutoCorrelationWeights(query, key, topK);
        var attnOutput = AggregateWithCorrelation(value, attnWeights, topK);

        // Project output
        return ProjectTensor(attnOutput, _selfOutputProj);
    }

    private Tensor<T> ApplyCrossAutoCorrelation(Tensor<T> query, Tensor<T> encoderOutput, int topK)
    {
        int seqLen = query.Shape[0];
        int embDim = query.Shape[1];

        // Project decoder query and encoder key/value
        var q = ProjectTensor(query, _crossQueryProj);
        var k = ProjectTensor(encoderOutput, _crossKeyProj);
        var v = ProjectTensor(encoderOutput, _crossValueProj);

        // Cross-correlation attention
        var attnWeights = ComputeCrossCorrelationWeights(q, k, topK);
        var attnOutput = AggregateWithCrossCorrelation(v, attnWeights, topK);

        // Project output
        return ProjectTensor(attnOutput, _crossOutputProj);
    }

    private Tensor<T> ProjectTensor(Tensor<T> x, Tensor<T> weight)
    {
        int seqLen = x.Shape[0];
        int inDim = x.Shape[1];
        int outDim = weight.Shape[0];
        var result = new Tensor<T>(new[] { seqLen, outDim });

        for (int t = 0; t < seqLen; t++)
        {
            for (int o = 0; o < outDim; o++)
            {
                var sum = _numOps.Zero;
                for (int i = 0; i < inDim; i++)
                {
                    sum = _numOps.Add(sum, _numOps.Multiply(x[t * inDim + i], weight[o * inDim + i]));
                }
                result[t * outDim + o] = sum;
            }
        }
        return result;
    }

    private T[] ComputeAutoCorrelationWeights(Tensor<T> query, Tensor<T> key, int topK)
    {
        int seqLen = query.Shape[0];
        int embDim = query.Shape[1];

        // Compute correlation for each lag
        var correlations = new T[seqLen];
        for (int lag = 0; lag < seqLen; lag++)
        {
            var corr = _numOps.Zero;
            int count = 0;
            for (int t = 0; t < seqLen - lag; t++)
            {
                for (int d = 0; d < embDim; d++)
                {
                    var q = query[t * embDim + d];
                    var k = key[(t + lag) * embDim + d];
                    corr = _numOps.Add(corr, _numOps.Multiply(q, k));
                    count++;
                }
            }
            correlations[lag] = count > 0 ? _numOps.Divide(corr, _numOps.FromDouble(count)) : _numOps.Zero;
        }

        // Find top-k correlations and apply softmax
        var indices = Enumerable.Range(0, seqLen)
            .OrderByDescending(i => _numOps.ToDouble(correlations[i]))
            .Take(topK)
            .ToArray();

        var weights = new T[topK];
        var maxCorr = indices.Length > 0 ? _numOps.ToDouble(correlations[indices[0]]) : 0.0;
        var expSum = _numOps.Zero;

        for (int i = 0; i < topK && i < indices.Length; i++)
        {
            weights[i] = _numOps.FromDouble(Math.Exp(_numOps.ToDouble(correlations[indices[i]]) - maxCorr));
            expSum = _numOps.Add(expSum, weights[i]);
        }

        if (_numOps.ToDouble(expSum) > 1e-10)
        {
            for (int i = 0; i < weights.Length; i++)
            {
                weights[i] = _numOps.Divide(weights[i], expSum);
            }
        }

        return weights;
    }

    private T[] ComputeCrossCorrelationWeights(Tensor<T> query, Tensor<T> key, int topK)
    {
        int queryLen = query.Shape[0];
        int keyLen = key.Shape[0];
        int embDim = query.Shape[1];

        // Compute attention scores between query and key positions
        var scores = new T[keyLen];
        for (int k = 0; k < keyLen; k++)
        {
            var score = _numOps.Zero;
            for (int q = 0; q < queryLen; q++)
            {
                for (int d = 0; d < embDim; d++)
                {
                    score = _numOps.Add(score, _numOps.Multiply(query[q * embDim + d], key[k * embDim + d]));
                }
            }
            scores[k] = _numOps.Divide(score, _numOps.FromDouble(Math.Sqrt(embDim)));
        }

        // Top-k softmax
        var indices = Enumerable.Range(0, keyLen)
            .OrderByDescending(i => _numOps.ToDouble(scores[i]))
            .Take(topK)
            .ToArray();

        var weights = new T[topK];
        var maxScore = indices.Length > 0 ? _numOps.ToDouble(scores[indices[0]]) : 0.0;
        var expSum = _numOps.Zero;

        for (int i = 0; i < topK && i < indices.Length; i++)
        {
            weights[i] = _numOps.FromDouble(Math.Exp(_numOps.ToDouble(scores[indices[i]]) - maxScore));
            expSum = _numOps.Add(expSum, weights[i]);
        }

        if (_numOps.ToDouble(expSum) > 1e-10)
        {
            for (int i = 0; i < weights.Length; i++)
            {
                weights[i] = _numOps.Divide(weights[i], expSum);
            }
        }

        return weights;
    }

    private Tensor<T> AggregateWithCorrelation(Tensor<T> value, T[] weights, int topK)
    {
        int seqLen = value.Shape[0];
        int embDim = value.Shape[1];
        var output = new Tensor<T>(new[] { seqLen, embDim });

        for (int t = 0; t < seqLen; t++)
        {
            for (int d = 0; d < embDim; d++)
            {
                var sum = _numOps.Zero;
                for (int k = 0; k < topK && k < weights.Length; k++)
                {
                    int srcIdx = (t + k) % seqLen;
                    sum = _numOps.Add(sum, _numOps.Multiply(weights[k], value[srcIdx * embDim + d]));
                }
                output[t * embDim + d] = sum;
            }
        }
        return output;
    }

    private Tensor<T> AggregateWithCrossCorrelation(Tensor<T> value, T[] weights, int topK)
    {
        int seqLen = value.Shape[0];
        int embDim = value.Shape[1];
        var output = new Tensor<T>(new[] { seqLen, embDim });

        for (int t = 0; t < seqLen; t++)
        {
            for (int d = 0; d < embDim; d++)
            {
                var sum = _numOps.Zero;
                for (int k = 0; k < topK && k < weights.Length; k++)
                {
                    int srcIdx = Math.Min(k, seqLen - 1);
                    sum = _numOps.Add(sum, _numOps.Multiply(weights[k], value[srcIdx * embDim + d]));
                }
                output[t * embDim + d] = sum;
            }
        }
        return output;
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
        accumulators[$"{prefix}ln1Gamma"] = new Tensor<T>(_layerNorm1Gamma.Shape);
        accumulators[$"{prefix}ln1Beta"] = new Tensor<T>(_layerNorm1Beta.Shape);
        accumulators[$"{prefix}ln2Gamma"] = new Tensor<T>(_layerNorm2Gamma.Shape);
        accumulators[$"{prefix}ln2Beta"] = new Tensor<T>(_layerNorm2Beta.Shape);
        accumulators[$"{prefix}ln3Gamma"] = new Tensor<T>(_layerNorm3Gamma.Shape);
        accumulators[$"{prefix}ln3Beta"] = new Tensor<T>(_layerNorm3Beta.Shape);
    }

    // Getters for autodiff integration
    public Tensor<T> GetSelfQueryProjection() => _selfQueryProj;
    public Tensor<T> GetSelfKeyProjection() => _selfKeyProj;
    public Tensor<T> GetSelfValueProjection() => _selfValueProj;
    public Tensor<T> GetSelfOutputProjection() => _selfOutputProj;
    public Tensor<T> GetCrossQueryProjection() => _crossQueryProj;
    public Tensor<T> GetCrossKeyProjection() => _crossKeyProj;
    public Tensor<T> GetCrossValueProjection() => _crossValueProj;
    public Tensor<T> GetCrossOutputProjection() => _crossOutputProj;
    public Tensor<T> GetFF1Weight() => _ff1Weight;
    public Tensor<T> GetFF1Bias() => _ff1Bias;
    public Tensor<T> GetFF2Weight() => _ff2Weight;
    public Tensor<T> GetFF2Bias() => _ff2Bias;
    public Tensor<T> GetLayerNorm1Gamma() => _layerNorm1Gamma;
    public Tensor<T> GetLayerNorm1Beta() => _layerNorm1Beta;
    public Tensor<T> GetLayerNorm2Gamma() => _layerNorm2Gamma;
    public Tensor<T> GetLayerNorm2Beta() => _layerNorm2Beta;
    public Tensor<T> GetLayerNorm3Gamma() => _layerNorm3Gamma;
    public Tensor<T> GetLayerNorm3Beta() => _layerNorm3Beta;

    public void ApplyGradients(Dictionary<string, Tensor<T>> accumulators, T scale, int layerIndex)
    {
        string prefix = $"decoder_{layerIndex}_";

        // Apply gradients: param = param - scale * gradient
        ApplyGradientToParameter(ref _selfQueryProj, accumulators, $"{prefix}selfQueryProj", scale);
        ApplyGradientToParameter(ref _selfKeyProj, accumulators, $"{prefix}selfKeyProj", scale);
        ApplyGradientToParameter(ref _selfValueProj, accumulators, $"{prefix}selfValueProj", scale);
        ApplyGradientToParameter(ref _selfOutputProj, accumulators, $"{prefix}selfOutputProj", scale);
        ApplyGradientToParameter(ref _crossQueryProj, accumulators, $"{prefix}crossQueryProj", scale);
        ApplyGradientToParameter(ref _crossKeyProj, accumulators, $"{prefix}crossKeyProj", scale);
        ApplyGradientToParameter(ref _crossValueProj, accumulators, $"{prefix}crossValueProj", scale);
        ApplyGradientToParameter(ref _crossOutputProj, accumulators, $"{prefix}crossOutputProj", scale);
        ApplyGradientToParameter(ref _ff1Weight, accumulators, $"{prefix}ff1Weight", scale);
        ApplyGradientToParameter(ref _ff1Bias, accumulators, $"{prefix}ff1Bias", scale);
        ApplyGradientToParameter(ref _ff2Weight, accumulators, $"{prefix}ff2Weight", scale);
        ApplyGradientToParameter(ref _ff2Bias, accumulators, $"{prefix}ff2Bias", scale);
        ApplyGradientToParameter(ref _layerNorm1Gamma, accumulators, $"{prefix}ln1Gamma", scale);
        ApplyGradientToParameter(ref _layerNorm1Beta, accumulators, $"{prefix}ln1Beta", scale);
        ApplyGradientToParameter(ref _layerNorm2Gamma, accumulators, $"{prefix}ln2Gamma", scale);
        ApplyGradientToParameter(ref _layerNorm2Beta, accumulators, $"{prefix}ln2Beta", scale);
        ApplyGradientToParameter(ref _layerNorm3Gamma, accumulators, $"{prefix}ln3Gamma", scale);
        ApplyGradientToParameter(ref _layerNorm3Beta, accumulators, $"{prefix}ln3Beta", scale);
    }

    private void ApplyGradientToParameter(ref Tensor<T> param, Dictionary<string, Tensor<T>> accumulators, string key, T scale)
    {
        if (!accumulators.TryGetValue(key, out var gradient))
            return;

        // param = param - scale * gradient
        for (int i = 0; i < param.Shape[0]; i++)
        {
            for (int j = 0; j < (param.Shape.Length > 1 ? param.Shape[1] : 1); j++)
            {
                var currentValue = param.Shape.Length > 1 ? param[i, j] : param[i];
                var gradValue = gradient.Shape.Length > 1 ? gradient[i, j] : gradient[i];
                var update = _numOps.Multiply(scale, gradValue);
                var newValue = _numOps.Subtract(currentValue, update);

                if (param.Shape.Length > 1)
                    param[i, j] = newValue;
                else
                    param[i] = newValue;
            }
        }
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
