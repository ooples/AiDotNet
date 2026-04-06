using AiDotNet.Attributes;
using AiDotNet.Autodiff;
using AiDotNet.Enums;

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
[ModelPaper("N-BEATS: Neural basis expansion analysis for interpretable time series forecasting", "https://arxiv.org/abs/1905.10437", Year = 2020, Authors = "Boris N. Oreshkin, Dmitri Carpov, Nicolas Chapados, Yoshua Bengio")]
public class NBEATSModel<T> : TimeSeriesModelBase<T>
{
    private readonly NBEATSModelOptions<T> _options;
    private readonly List<NBEATSBlock<T>> _blocks;
    private Vector<T> _trainingSeries = Vector<T>.Empty();

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
    /// Trains the N-BEATS model using mini-batch gradient descent.
    /// </summary>
    protected override void TrainCore(Matrix<T> x, Vector<T> y)
    {
        // Store training series BEFORE training loop for cancellation safety
        _trainingSeries = new Vector<T>(y.Length);
        for (int i = 0; i < y.Length; i++)
            _trainingSeries[i] = y[i];
        ModelParameters = new Vector<T>(1);
        ModelParameters[0] = NumOps.FromDouble(y.Length);

        T learningRate = NumOps.FromDouble(_options.LearningRate);
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
                    var lookback = ExtractLookbackWindow(x, y, idx);
                    T target = y[idx];

                    // Forward pass through all blocks
                    Vector<T> residual = lookback.Clone();
                    Vector<T> aggregatedForecast = new Vector<T>(_options.ForecastHorizon);

                    for (int blockIdx = 0; blockIdx < _blocks.Count; blockIdx++)
                    {
                        var (backcast, forecast) = _blocks[blockIdx].ForwardInternal(residual);
                        residual = (Vector<T>)Engine.Subtract(residual, backcast);
                        aggregatedForecast = (Vector<T>)Engine.Add(aggregatedForecast, forecast);
                    }

                    // Compute gradient of MSE loss on first forecast step
                    T prediction = aggregatedForecast[0];
                    T error = NumOps.Subtract(prediction, target);
                    T gradScale = NumOps.Multiply(NumOps.FromDouble(2.0 / batchSize), error);

                    // Update each block with scaled learning rate
                    T scaledLr = NumOps.Multiply(learningRate, gradScale);
                    for (int blockIdx = 0; blockIdx < _blocks.Count; blockIdx++)
                    {
                        _blocks[blockIdx].UpdateParameters(scaledLr);
                    }
                }
            }
        }
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

        // Univariate case: build lookback windows from training series + autoregressive predictions.
        // Construct a running series combining training data and new predictions.
        var series = new List<T>(trainN + n);
        for (int i = 0; i < trainN; i++)
            series.Add(_trainingSeries[i]);

        for (int i = 0; i < n; i++)
        {
            // Always produce a model prediction — never return stored training data
            int seriesLen = series.Count;
            if (seriesLen >= _options.LookbackWindow)
            {
                var lookback = new Vector<T>(_options.LookbackWindow);
                int start = seriesLen - _options.LookbackWindow;
                for (int j = 0; j < _options.LookbackWindow; j++)
                    lookback[j] = series[start + j];
                predictions[i] = PredictSingle(lookback);
            }
            else
            {
                // Not enough history: pad with zeros
                var lookback = new Vector<T>(_options.LookbackWindow);
                int offset = _options.LookbackWindow - seriesLen;
                for (int j = 0; j < seriesLen; j++)
                    lookback[offset + j] = series[j];
                predictions[i] = PredictSingle(lookback);
            }
            series.Add(predictions[i]);
        }

        return predictions;
    }

    /// <summary>
    /// Extracts a lookback window vector for a given sample index.
    /// </summary>
    /// <param name="x">Input features matrix.</param>
    /// <param name="y">Target values vector.</param>
    /// <param name="sampleIdx">The index of the current sample.</param>
    /// <returns>A vector of length <see cref="NBEATSModelOptions{T}.LookbackWindow"/>.</returns>
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

        Vector<T> residual = input.Clone();
        Vector<T> aggregatedForecast = new Vector<T>(_options.ForecastHorizon);

        // Forward pass through all blocks
        for (int blockIdx = 0; blockIdx < _blocks.Count; blockIdx++)
        {
            var (backcast, forecast) = _blocks[blockIdx].ForwardInternal(residual);

            // Update residual for next block - vectorized with Engine.Subtract
            residual = (Vector<T>)Engine.Subtract(residual, backcast);

            // Accumulate forecast - vectorized with Engine.Add
            aggregatedForecast = (Vector<T>)Engine.Add(aggregatedForecast, forecast);
        }

        // Return the first forecast step
        return aggregatedForecast[0];
    }

    /// <summary>
    /// Generates forecasts for multiple future time steps.
    /// </summary>
    /// <param name="input">The input vector containing the lookback window of historical values.</param>
    /// <returns>A vector of forecasted values for all forecast horizon steps.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method predicts multiple future time steps at once.
    /// Unlike PredictSingle which only returns the next value, this returns all values
    /// up to the forecast horizon.
    ///
    /// For example, if your forecast horizon is 7, this will predict the next 7 time steps.
    /// </para>
    /// </remarks>
    public Vector<T> ForecastHorizon(Vector<T> input)
    {
        if (input.Length != _options.LookbackWindow)
        {
            throw new ArgumentException(
                $"Input length ({input.Length}) must match lookback window ({_options.LookbackWindow}).",
                nameof(input));
        }

        Vector<T> residual = input.Clone();
        Vector<T> aggregatedForecast = new Vector<T>(_options.ForecastHorizon);

        // Forward pass through all blocks
        for (int blockIdx = 0; blockIdx < _blocks.Count; blockIdx++)
        {
            var (backcast, forecast) = _blocks[blockIdx].ForwardInternal(residual);

            // Update residual for next block - vectorized with Engine.Subtract
            residual = (Vector<T>)Engine.Subtract(residual, backcast);

            // Accumulate forecast - vectorized with Engine.Add
            aggregatedForecast = (Vector<T>)Engine.Add(aggregatedForecast, forecast);
        }

        return aggregatedForecast;
    }

    /// <summary>
    /// Serializes model-specific data to the binary writer.
    /// </summary>
    /// <param name="writer">The binary writer to write to.</param>
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
    /// <param name="reader">The binary reader to read from.</param>
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
    /// <returns>A ModelMetadata object containing information about the model.</returns>
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
    /// <returns>A new N-BEATS model instance with the same configuration.</returns>
    /// <remarks>
    /// Creates a deep copy of the model options to ensure the cloned model has an independent options instance.
    /// </remarks>
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
    /// <returns>A vector containing all trainable parameters from all blocks.</returns>
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
    /// <param name="parameters">A vector containing all trainable parameters.</param>
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
        // Copy trained blocks (read-only after training — safe to share by reference)
        clone._blocks.Clear();
        clone._blocks.AddRange(_blocks);
        // Copy training series
        if (_trainingSeries.Length > 0)
            clone._trainingSeries = new Vector<T>(_trainingSeries);
        // Copy model parameters
        if (ModelParameters is not null && ModelParameters.Length > 0)
            clone.ModelParameters = new Vector<T>(ModelParameters);
        return clone;
    

}

    public override IFullModel<T, Matrix<T>, Vector<T>> DeepCopy() => Clone();
}
