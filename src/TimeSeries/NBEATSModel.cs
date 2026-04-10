using AiDotNet.Attributes;
using AiDotNet.Autodiff;
using AiDotNet.Enums;
using AiDotNet.LossFunctions;
using AiDotNet.Optimizers;
using AiDotNet.Models.Options;
using AiDotNet.Tensors.Engines.Autodiff;

namespace AiDotNet.TimeSeries;

/// <summary>
/// Implements the N-BEATS (Neural Basis Expansion Analysis for Time Series) model for forecasting.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// N-BEATS is a deep neural architecture based on backward and forward residual links and
/// a very deep stack of fully-connected layers. The architecture has the following key features:
/// </para>
/// <list type="bullet">
/// <item>Doubly residual stacking: Each block produces a backcast (reconstruction) and forecast</item>
/// <item>Hierarchical decomposition: Multiple stacks focus on different aspects (trend, seasonality)</item>
/// <item>Interpretability: Can use polynomial and Fourier basis for explainable forecasts</item>
/// <item>No manual feature engineering: Learns directly from raw time series data</item>
/// </list>
/// <para>
/// The original paper: Oreshkin et al., "N-BEATS: Neural basis expansion analysis for
/// interpretable time series forecasting" (ICLR 2020).
/// </para>
/// <para><b>For Beginners:</b> N-BEATS is a state-of-the-art neural network for time series
/// forecasting that automatically learns patterns from your data. Unlike traditional methods
/// that require you to manually specify trends and seasonality, N-BEATS figures these out
/// on its own.
///
/// Key advantages:
/// - No need for manual feature engineering (the model learns what's important)
/// - Can capture complex, non-linear patterns
/// - Provides interpretable components (trend, seasonality) when configured to do so
/// - Works well for both short-term and long-term forecasting
///
/// The model works by stacking many "blocks" together, where each block tries to:
/// 1. Understand what patterns are in the input (backcast)
/// 2. Predict the future based on those patterns (forecast)
/// 3. Pass the unexplained patterns to the next block
///
/// This allows the model to decompose complex time series into simpler components.
/// </para>
/// </remarks>
/// <example>
/// <code>
/// // Create an N-BEATS model with interpretable trend and seasonality stacks
/// var options = new NBEATSModelOptions&lt;double&gt;();
/// var nbeats = new NBEATSModel&lt;double&gt;(options);
/// nbeats.Train(trainingMatrix, trainingLabels);
/// Vector&lt;double&gt; forecast = nbeats.Predict(inputMatrix);
/// </code>
/// </example>
[ModelDomain(ModelDomain.TimeSeries)]
[ModelCategory(ModelCategory.NeuralNetwork)]
[ModelCategory(ModelCategory.TimeSeriesModel)]
[ModelTask(ModelTask.Forecasting)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Matrix<>), typeof(Vector<>))]
[ResearchPaper("N-BEATS: Neural basis expansion analysis for interpretable time series forecasting", "https://arxiv.org/abs/1905.10437", Year = 2020, Authors = "Boris N. Oreshkin, Dmitri Carpov, Nicolas Chapados, Yoshua Bengio")]
public class NBEATSModel<T> : TimeSeriesModelBase<T>
{
    private readonly NBEATSModelOptions<T> _options;
    private readonly List<NBEATSBlock<T>> _blocks;
    private Vector<T> _trainingSeries = Vector<T>.Empty();

    // Normalization statistics computed during training
    private T _normMean = MathHelper.GetNumericOperations<T>().Zero;
    private T _normStd = MathHelper.GetNumericOperations<T>().One;

    /// <summary>
    /// Initializes a new instance of the NBEATSModel class.
    /// </summary>
    /// <param name="options">Configuration options for the N-BEATS model. If null, default options are used.</param>
    /// <remarks>
    /// <para><b>For Beginners:</b> This creates a new N-BEATS model with the specified configuration.
    /// The options control things like:
    /// - How far back to look (lookback window)
    /// - How far forward to predict (forecast horizon)
    /// - How complex the model should be (number of stacks, blocks, layer sizes)
    /// - Whether to use interpretable components
    ///
    /// If you don't provide options, sensible defaults will be used.
    /// </para>
    /// </remarks>
    public NBEATSModel(NBEATSModelOptions<T>? options = null) : base(options ?? new NBEATSModelOptions<T>())
    {
        _options = options ?? new NBEATSModelOptions<T>();
        Options = _options;
        _blocks = new List<NBEATSBlock<T>>();

        // Validate options
        ValidateNBEATSOptions();

        // Initialize blocks
        InitializeBlocks();
    }

    /// <summary>
    /// Validates the N-BEATS specific options.
    /// </summary>
    private void ValidateNBEATSOptions()
    {
        if (_options.LookbackWindow <= 0)
        {
            throw new ArgumentException("Lookback window must be positive.", nameof(_options.LookbackWindow));
        }

        if (_options.ForecastHorizon <= 0)
        {
            throw new ArgumentException("Forecast horizon must be positive.", nameof(_options.ForecastHorizon));
        }

        if (_options.NumStacks <= 0)
        {
            throw new ArgumentException("Number of stacks must be positive.", nameof(_options.NumStacks));
        }

        if (_options.NumBlocksPerStack <= 0)
        {
            throw new ArgumentException("Number of blocks per stack must be positive.", nameof(_options.NumBlocksPerStack));
        }

        if (_options.HiddenLayerSize <= 0)
        {
            throw new ArgumentException("Hidden layer size must be positive.", nameof(_options.HiddenLayerSize));
        }

        if (_options.NumHiddenLayers <= 0)
        {
            throw new ArgumentException("Number of hidden layers must be positive.", nameof(_options.NumHiddenLayers));
        }

        if (_options.PolynomialDegree < 1)
        {
            throw new ArgumentException("Polynomial degree must be at least 1.", nameof(_options.PolynomialDegree));
        }

        if (_options.Epochs <= 0)
        {
            throw new ArgumentException("Number of epochs must be positive.", nameof(_options.Epochs));
        }

        if (_options.BatchSize <= 0)
        {
            throw new ArgumentException("Batch size must be positive.", nameof(_options.BatchSize));
        }

        if (_options.LearningRate <= 0)
        {
            throw new ArgumentException("Learning rate must be positive.", nameof(_options.LearningRate));
        }
    }

    /// <summary>
    /// Initializes all blocks in the N-BEATS architecture.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> This creates all the individual blocks that make up
    /// the N-BEATS model. The number of blocks is determined by NumStacks * NumBlocksPerStack.
    ///
    /// Each block is initialized with the same architecture but different random weights,
    /// allowing them to learn different aspects of the time series.
    /// </para>
    /// </remarks>
    private void InitializeBlocks()
    {
        _blocks.Clear();

        // Calculate theta sizes for basis expansion
        int thetaSizeBackcast;
        int thetaSizeForecast;

        if (_options.UseInterpretableBasis)
        {
            // For polynomial basis, theta size is polynomial degree + 1
            thetaSizeBackcast = _options.PolynomialDegree + 1;
            thetaSizeForecast = _options.PolynomialDegree + 1;
        }
        else
        {
            // For generic basis, theta size matches the output length
            thetaSizeBackcast = _options.LookbackWindow;
            thetaSizeForecast = _options.ForecastHorizon;
        }

        // Create all blocks
        int totalBlocks = _options.NumStacks * _options.NumBlocksPerStack;
        for (int i = 0; i < totalBlocks; i++)
        {
            var block = new NBEATSBlock<T>(
                _options.LookbackWindow,
                _options.ForecastHorizon,
                _options.HiddenLayerSize,
                _options.NumHiddenLayers,
                thetaSizeBackcast,
                thetaSizeForecast,
                _options.UseInterpretableBasis,
                _options.PolynomialDegree
            );
            _blocks.Add(block);
        }
    }

    /// <summary>
    /// Trains the N-BEATS model using tape-based automatic differentiation with Adam optimizer.
    /// Per Oreshkin et al. (2020), NBEATS uses Adam for optimization.
    /// </summary>
    protected override void TrainCore(Matrix<T> x, Vector<T> y)
    {
        // Store training series BEFORE training loop for cancellation safety
        _trainingSeries = new Vector<T>(y.Length);
        for (int i = 0; i < y.Length; i++)
            _trainingSeries[i] = y[i];
        ModelParameters = new Vector<T>(1);
        ModelParameters[0] = NumOps.FromDouble(y.Length);

        // Normalize the input series to zero mean / unit variance for stable gradient flow.
        T yMean = NumOps.Zero;
        for (int i = 0; i < y.Length; i++)
            yMean = NumOps.Add(yMean, y[i]);
        yMean = NumOps.Divide(yMean, NumOps.FromDouble(y.Length));

        T yVar = NumOps.Zero;
        for (int i = 0; i < y.Length; i++)
        {
            T diff = NumOps.Subtract(y[i], yMean);
            yVar = NumOps.Add(yVar, NumOps.Multiply(diff, diff));
        }
        yVar = NumOps.Divide(yVar, NumOps.FromDouble(y.Length));
        T yStd = NumOps.Sqrt(yVar);
        if (NumOps.LessThanOrEquals(yStd, NumOps.FromDouble(1e-10)))
            yStd = NumOps.One;

        _normMean = yMean;
        _normStd = yStd;

        // Create normalized copies
        Vector<T> yNorm = new Vector<T>(y.Length);
        for (int i = 0; i < y.Length; i++)
            yNorm[i] = NumOps.Divide(NumOps.Subtract(y[i], yMean), yStd);

        // Create Adam optimizer (per Oreshkin et al. 2020)
        var adamOptions = new AdamOptimizerOptions<T, Matrix<T>, Vector<T>>
        {
            InitialLearningRate = _options.LearningRate
        };
        var optimizer = new AdamOptimizer<T, Matrix<T>, Vector<T>>(null, adamOptions);

        // Collect all trainable parameters from all blocks
        var allBlocks = _blocks.Cast<Interfaces.ILayer<T>>().ToList();
        var trainableParams = Training.TapeTrainingStep<T>.CollectParameters(allBlocks, -1);

        // MSE loss function for tape-tracked computation
        // Per Oreshkin et al. 2020, N-BEATS uses MAE loss (not MSE)
        var maeLoss = new MeanAbsoluteErrorLoss<T>();

        int numSamples = x.Rows;
        var random = new Random(42);

        for (int epoch = 0; epoch < _options.Epochs; epoch++)
        {
            TrainingCancellationToken.ThrowIfCancellationRequested();

            var indices = Enumerable.Range(0, numSamples).OrderBy(_ => random.Next()).ToList();

            for (int batchStart = 0; batchStart < numSamples; batchStart += _options.BatchSize)
            {
                int batchEnd = Math.Min(batchStart + _options.BatchSize, numSamples);
                int batchSize = batchEnd - batchStart;

                for (int bi = 0; bi < batchSize; bi++)
                {
                    if (bi % 8 == 0) TrainingCancellationToken.ThrowIfCancellationRequested();
                    int idx = indices[batchStart + bi];
                    var lookback = ExtractNormalizedLookbackWindow(x, yNorm, idx);
                    T target = yNorm[idx];

                    // Create input tensor from lookback vector
                    var inputTensor = new Tensor<T>(new[] { _options.LookbackWindow }, lookback);
                    // Create target tensor (scalar)
                    var targetTensor = new Tensor<T>(new[] { 1 }, new Vector<T>(new[] { target }));

                    // Re-collect params — use version -1 to avoid ThreadStatic cache
                    // returning stale parameters from a different model on the same thread
                    trainableParams = Training.TapeTrainingStep<T>.CollectParameters(allBlocks, -1);

                    // Forward + backward under gradient tape
                    using var tape = new GradientTape<T>();

                    // Tape-tracked forward pass through all blocks with residual connections
                    var residual = inputTensor;
                    Tensor<T>? aggregatedForecast = null;

                    for (int blockIdx = 0; blockIdx < _blocks.Count; blockIdx++)
                    {
                        var (backcast, forecast) = _blocks[blockIdx].ForwardTape(residual);
                        // Residual: subtract backcast from input to this block
                        residual = Engine.TensorSubtract(residual, backcast);
                        // Accumulate forecast
                        if (aggregatedForecast is null)
                            aggregatedForecast = forecast;
                        else
                            aggregatedForecast = Engine.TensorAdd(aggregatedForecast, forecast);
                    }

                    // Extract the first forecast element as prediction
                    // Create slice weights: one-hot vector selecting index 0
                    var sliceWeights = new T[_options.ForecastHorizon];
                    sliceWeights[0] = NumOps.One;
                    var sliceTensor = new Tensor<T>(
                        new[] { 1, _options.ForecastHorizon },
                        new Vector<T>(sliceWeights));
                    var forecastCol = Engine.Reshape(aggregatedForecast!, [_options.ForecastHorizon, 1]);
                    var prediction = Engine.TensorMatMul(sliceTensor, forecastCol); // [1, 1]
                    prediction = Engine.Reshape(prediction, [1]); // [1]

                    // Compute MSE loss via tape-tracked operations
                    var lossTensor = maeLoss.ComputeTapeLoss(prediction, targetTensor);

                    // Compute gradients for all trainable parameters
                    var allGrads = tape.ComputeGradients(lossTensor, sources: null);
                    var grads = new Dictionary<Tensor<T>, Tensor<T>>(
                        Helpers.TensorReferenceComparer<Tensor<T>>.Instance);
                    foreach (var param in trainableParams)
                    {
                        if (allGrads.TryGetValue(param, out var grad))
                            grads[param] = grad;
                    }


                    // Create optimizer context and step
                    Tensor<T> ComputeForward(Tensor<T> inp, Tensor<T> tgt) => prediction; // not used by Adam
                    Tensor<T> ComputeLoss(Tensor<T> pred, Tensor<T> tgt) =>
                        maeLoss.ComputeTapeLoss(pred, tgt);

                    var context = new TapeStepContext<T>(
                        trainableParams, grads, lossTensor.Length > 0 ? lossTensor[0] : NumOps.Zero,
                        inputTensor, targetTensor, ComputeForward, ComputeLoss,
                        null);

                    optimizer.Step(context);

                }
            }
        }
    }

    /// <summary>
    /// Extracts a normalized lookback window for training.
    /// </summary>
    private Vector<T> ExtractNormalizedLookbackWindow(Matrix<T> x, Vector<T> yNorm, int sampleIdx)
    {
        var input = new Vector<T>(_options.LookbackWindow);
        if (x.Columns >= _options.LookbackWindow)
        {
            // Multi-variate: normalize each element
            for (int j = 0; j < _options.LookbackWindow; j++)
                input[j] = NumOps.Divide(NumOps.Subtract(x[sampleIdx, j], _normMean), _normStd);
        }
        else
        {
            // Univariate: use preceding normalized y values
            for (int j = 0; j < _options.LookbackWindow; j++)
            {
                int yIdx = sampleIdx - _options.LookbackWindow + j;
                input[j] = yIdx >= 0 ? yNorm[yIdx] : NumOps.Zero;
            }
        }
        return input;
    }

    public override Vector<T> Predict(Matrix<T> input)
    {
        int n = input.Rows;
        int trainN = _trainingSeries.Length;
        var predictions = new Vector<T>(n);

        // If the input has enough columns to serve as a lookback window, use rows directly
        if (input.Columns >= _options.LookbackWindow)
        {
            for (int i = 0; i < n; i++)
            {
                predictions[i] = PredictSingle(input.GetRow(i));
            }
            return predictions;
        }

        // Univariate case: use overlapping lookback windows from training series
        // for in-sample positions, autoregressive for out-of-sample.
        // Per Oreshkin et al. 2020: each prediction uses the preceding lookback
        // window from observed data when available, avoiding error compounding.
        var series = new List<T>(trainN);
        for (int i = 0; i < trainN; i++)
            series.Add(_trainingSeries[i]);

        for (int i = 0; i < n; i++)
        {
            // Use training series lookback for in-sample, autoregressive for out-of-sample
            int seriesIdx = i; // position in the original timeline
            var lookback = new Vector<T>(_options.LookbackWindow);

            for (int j = 0; j < _options.LookbackWindow; j++)
            {
                int idx = seriesIdx - _options.LookbackWindow + 1 + j;
                if (idx >= 0 && idx < series.Count)
                    lookback[j] = series[idx];
                else
                    lookback[j] = NumOps.Zero;
            }

            predictions[i] = PredictSingle(lookback);
        }

        return predictions;
    }

    /// <summary>
    /// Extracts a lookback window vector for a given sample index.
    /// </summary>
    private Vector<T> ExtractLookbackWindow(Matrix<T> x, Vector<T> y, int sampleIdx)
    {
        var input = new Vector<T>(_options.LookbackWindow);
        if (x.Columns >= _options.LookbackWindow)
        {
            for (int j = 0; j < _options.LookbackWindow; j++)
                input[j] = x[sampleIdx, j];
        }
        else
        {
            // Univariate: construct lookback window from preceding y values
            for (int j = 0; j < _options.LookbackWindow; j++)
            {
                int yIdx = sampleIdx - _options.LookbackWindow + j;
                input[j] = yIdx >= 0 ? y[yIdx] : NumOps.Zero;
            }
        }

        return input;
    }

    /// <summary>
    /// Predicts a single value based on the provided input vector.
    /// </summary>
    /// <param name="input">The input vector containing the lookback window of historical values.</param>
    /// <returns>The predicted value for the next time step.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method takes a window of historical values and
    /// predicts the next value. It runs the input through all the blocks in the model,
    /// each block contributing to the final prediction.
    /// </para>
    /// </remarks>
    public override T PredictSingle(Vector<T> input)
    {
        // If input is shorter than lookback window, construct from training series tail
        if (input.Length < _options.LookbackWindow && _trainingSeries.Length >= _options.LookbackWindow)
        {
            var lookback = new Vector<T>(_options.LookbackWindow);
            int start = _trainingSeries.Length - _options.LookbackWindow;
            for (int j = 0; j < _options.LookbackWindow; j++)
                lookback[j] = _trainingSeries[start + j];
            input = lookback;
        }

        if (input.Length != _options.LookbackWindow)
        {
            throw new ArgumentException(
                $"Input length ({input.Length}) must match lookback window ({_options.LookbackWindow}).",
                nameof(input));
        }

        // Normalize input using training statistics
        Vector<T> normalizedInput = new Vector<T>(input.Length);
        for (int i = 0; i < input.Length; i++)
            normalizedInput[i] = NumOps.Divide(NumOps.Subtract(input[i], _normMean), _normStd);

        Vector<T> residual = normalizedInput;
        Vector<T> aggregatedForecast = new Vector<T>(_options.ForecastHorizon);

        // Forward pass through all blocks (inference mode, no tape)
        for (int blockIdx = 0; blockIdx < _blocks.Count; blockIdx++)
        {
            var (backcast, forecast) = _blocks[blockIdx].ForwardInternal(residual);

            // Update residual for next block
            residual = (Vector<T>)Engine.Subtract(residual, backcast);

            // Accumulate forecast
            aggregatedForecast = (Vector<T>)Engine.Add(aggregatedForecast, forecast);
        }

        // Denormalize the forecast and return the first step
        return NumOps.Add(NumOps.Multiply(aggregatedForecast[0], _normStd), _normMean);
    }

    /// <summary>
    /// Generates forecasts for multiple future time steps.
    /// </summary>
    /// <param name="input">The input vector containing the lookback window of historical values.</param>
    /// <returns>A vector of forecasted values for all forecast horizon steps.</returns>
    public Vector<T> ForecastHorizon(Vector<T> input)
    {
        if (input.Length != _options.LookbackWindow)
        {
            throw new ArgumentException(
                $"Input length ({input.Length}) must match lookback window ({_options.LookbackWindow}).",
                nameof(input));
        }

        // Normalize input
        Vector<T> normalizedInput = new Vector<T>(input.Length);
        for (int i = 0; i < input.Length; i++)
            normalizedInput[i] = NumOps.Divide(NumOps.Subtract(input[i], _normMean), _normStd);

        Vector<T> residual = normalizedInput;
        Vector<T> aggregatedForecast = new Vector<T>(_options.ForecastHorizon);

        // Forward pass through all blocks
        for (int blockIdx = 0; blockIdx < _blocks.Count; blockIdx++)
        {
            var (backcast, forecast) = _blocks[blockIdx].ForwardInternal(residual);

            residual = (Vector<T>)Engine.Subtract(residual, backcast);
            aggregatedForecast = (Vector<T>)Engine.Add(aggregatedForecast, forecast);
        }

        // Denormalize forecast
        for (int i = 0; i < aggregatedForecast.Length; i++)
            aggregatedForecast[i] = NumOps.Add(NumOps.Multiply(aggregatedForecast[i], _normStd), _normMean);

        return aggregatedForecast;
    }

    /// <summary>
    /// Serializes model-specific data to the binary writer.
    /// </summary>
    protected override void SerializeCore(BinaryWriter writer)
    {
        // Write N-BEATS specific options
        writer.Write(_options.NumStacks);
        writer.Write(_options.NumBlocksPerStack);
        writer.Write(_options.PolynomialDegree);
        writer.Write(_options.LookbackWindow);
        writer.Write(_options.ForecastHorizon);
        writer.Write(_options.HiddenLayerSize);
        writer.Write(_options.NumHiddenLayers);
        writer.Write(_options.LearningRate);
        writer.Write(_options.Epochs);
        writer.Write(_options.BatchSize);
        writer.Write(_options.ShareWeightsInStack);
        writer.Write(_options.UseInterpretableBasis);

        // Write all block parameters
        writer.Write(_blocks.Count);
        foreach (var block in _blocks)
        {
            Vector<T> blockParams = block.GetParameters();
            writer.Write(blockParams.Length);
            for (int i = 0; i < blockParams.Length; i++)
            {
                writer.Write(Convert.ToDouble(blockParams[i]));
            }
        }
    }

    /// <summary>
    /// Deserializes model-specific data from the binary reader.
    /// </summary>
    protected override void DeserializeCore(BinaryReader reader)
    {
        // Read N-BEATS specific options
        _options.NumStacks = reader.ReadInt32();
        _options.NumBlocksPerStack = reader.ReadInt32();
        _options.PolynomialDegree = reader.ReadInt32();
        _options.LookbackWindow = reader.ReadInt32();
        _options.ForecastHorizon = reader.ReadInt32();
        _options.HiddenLayerSize = reader.ReadInt32();
        _options.NumHiddenLayers = reader.ReadInt32();
        _options.LearningRate = reader.ReadDouble();
        _options.Epochs = reader.ReadInt32();
        _options.BatchSize = reader.ReadInt32();
        _options.ShareWeightsInStack = reader.ReadBoolean();
        _options.UseInterpretableBasis = reader.ReadBoolean();

        // Reinitialize blocks with loaded options
        InitializeBlocks();

        // Read all block parameters
        int blockCount = reader.ReadInt32();
        if (blockCount != _blocks.Count)
        {
            throw new InvalidOperationException(
                $"Block count mismatch. Expected {_blocks.Count}, but serialized data contains {blockCount}.");
        }

        for (int i = 0; i < blockCount; i++)
        {
            int paramCount = reader.ReadInt32();
            Vector<T> blockParams = new Vector<T>(paramCount);
            for (int j = 0; j < paramCount; j++)
            {
                blockParams[j] = NumOps.FromDouble(reader.ReadDouble());
            }
            _blocks[i].SetParameters(blockParams);
        }
    }

    /// <summary>
    /// Gets metadata about the N-BEATS model.
    /// </summary>
    public override ModelMetadata<T> GetModelMetadata()
    {
        var metadata = new ModelMetadata<T>
        {
            Name = "N-BEATS",
            Description = "Neural Basis Expansion Analysis for Interpretable Time Series Forecasting",
            Complexity = ParameterCount,
            FeatureCount = _options.LookbackWindow,
            AdditionalInfo = new Dictionary<string, object>
            {
                { "InputDimension", _options.LookbackWindow },
                { "OutputDimension", _options.ForecastHorizon },
                { "TrainingMetrics", LastEvaluationMetrics ?? new Dictionary<string, T>() },
                { "Hyperparameters", new Dictionary<string, object>
                    {
                        { "NumStacks", _options.NumStacks },
                        { "NumBlocksPerStack", _options.NumBlocksPerStack },
                        { "PolynomialDegree", _options.PolynomialDegree },
                        { "LookbackWindow", _options.LookbackWindow },
                        { "ForecastHorizon", _options.ForecastHorizon },
                        { "HiddenLayerSize", _options.HiddenLayerSize },
                        { "NumHiddenLayers", _options.NumHiddenLayers },
                        { "UseInterpretableBasis", _options.UseInterpretableBasis }
                    }
                }
            }
        };
        return metadata;
    }

    /// <summary>
    /// Creates a new instance of the N-BEATS model.
    /// </summary>
    protected override IFullModel<T, Matrix<T>, Vector<T>> CreateInstance()
    {
        return new NBEATSModel<T>(new NBEATSModelOptions<T>(_options));
    }

    /// <summary>
    /// Gets the total number of trainable parameters in the model.
    /// </summary>
    public override int ParameterCount
    {
        get
        {
            int totalParams = 0;
            foreach (var block in _blocks)
            {
                totalParams += block.ParameterCount;
            }
            return totalParams;
        }
    }

    /// <summary>
    /// Gets all model parameters as a single vector.
    /// </summary>
    public override Vector<T> GetParameters()
    {
        var allParams = new List<T>();

        foreach (var block in _blocks)
        {
            Vector<T> blockParams = block.GetParameters();
            for (int i = 0; i < blockParams.Length; i++)
            {
                allParams.Add(blockParams[i]);
            }
        }

        return new Vector<T>(allParams.ToArray());
    }

    /// <summary>
    /// Sets all model parameters from a single vector.
    /// </summary>
    public override void SetParameters(Vector<T> parameters)
    {
        int expectedCount = ParameterCount;
        if (parameters.Length != expectedCount)
        {
            throw new ArgumentException(
                $"Expected {expectedCount} parameters, but got {parameters.Length}.",
                nameof(parameters));
        }

        int idx = 0;
        foreach (var block in _blocks)
        {
            int blockParamCount = block.ParameterCount;
            Vector<T> blockParams = new Vector<T>(blockParamCount);

            for (int i = 0; i < blockParamCount; i++)
            {
                blockParams[i] = parameters[idx++];
            }

            block.SetParameters(blockParams);
        }
    }

    /// <summary>
    /// Creates slice weights for extracting a single element from a vector.
    /// </summary>
    private T[] CreateSliceWeights(int index, int length, INumericOperations<T> numOps)
    {
        var weights = new T[length];
        for (int i = 0; i < length; i++)
        {
            weights[i] = i == index ? numOps.One : numOps.Zero;
        }
        return weights;
    }

    public override IFullModel<T, Matrix<T>, Vector<T>> Clone()
    {
        var clone = new NBEATSModel<T>(_options);
        // Copy trained blocks (read-only after training -- safe to share by reference)
        clone._blocks.Clear();
        clone._blocks.AddRange(_blocks);
        // Copy training series
        if (_trainingSeries.Length > 0)
            clone._trainingSeries = new Vector<T>(_trainingSeries);
        // Copy model parameters
        if (ModelParameters is not null && ModelParameters.Length > 0)
            clone.ModelParameters = new Vector<T>(ModelParameters);
        // Copy normalization parameters
        clone._normMean = _normMean;
        clone._normStd = _normStd;
        return clone;


}

    public override IFullModel<T, Matrix<T>, Vector<T>> DeepCopy() => Clone();

}
