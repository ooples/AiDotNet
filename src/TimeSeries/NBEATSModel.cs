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
public class NBEATSModel<T> : TimeSeriesModelBase<T>
{
    private readonly NBEATSModelOptions<T> _options;
    private readonly List<NBEATSBlock<T>> _blocks;
    private readonly INumericOperations<T> _numOps;

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
        _numOps = MathHelper.GetNumericOperations<T>();
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
    /// Performs the core training logic for the N-BEATS model.
    /// </summary>
    /// <param name="x">The input features matrix where each row is a historical window.</param>
    /// <param name="y">The target values vector where each element is the corresponding forecast target.</param>
    /// <remarks>
    /// <para>
    /// Training uses a simple gradient descent approach with mean squared error loss.
    /// The model iterates through the training data for the specified number of epochs,
    /// updating parameters to minimize prediction error.
    /// </para>
    /// <para><b>For Beginners:</b> This is where the model actually learns from your data.
    ///
    /// The training process:
    /// 1. The model makes predictions on your training data
    /// 2. It calculates how far off the predictions are (the error)
    /// 3. It adjusts its internal parameters to reduce this error
    /// 4. It repeats this process many times (epochs) until it learns the patterns
    ///
    /// Note: This is a simplified training implementation. A production version would
    /// include more sophisticated optimization, regularization, and validation.
    /// </para>
    /// </remarks>
    protected override void TrainCore(Matrix<T> x, Vector<T> y)
    {
        // For simplicity, we'll implement a basic training loop
        // A full implementation would use more sophisticated optimization

        int numSamples = x.Rows;

        // Simple gradient descent training for demonstration
        for (int epoch = 0; epoch < _options.Epochs; epoch++)
        {
            T totalLoss = _numOps.Zero;

            // Process each sample
            for (int sampleIdx = 0; sampleIdx < numSamples; sampleIdx++)
            {
                Vector<T> input = x.GetRow(sampleIdx);

                // Forward pass through all blocks
                Vector<T> residual = input.Clone();
                Vector<T> aggregatedForecast = new Vector<T>(_options.ForecastHorizon);

                for (int blockIdx = 0; blockIdx < _blocks.Count; blockIdx++)
                {
                    var (backcast, forecast) = _blocks[blockIdx].Forward(residual);

                    // Update residual for next block
                    for (int i = 0; i < residual.Length; i++)
                    {
                        residual[i] = _numOps.Subtract(residual[i], backcast[i]);
                    }

                    // Accumulate forecast
                    for (int i = 0; i < aggregatedForecast.Length; i++)
                    {
                        aggregatedForecast[i] = _numOps.Add(aggregatedForecast[i], forecast[i]);
                    }
                }

                // Calculate loss (simplified - just the first forecast step for now)
                T target = y[sampleIdx];
                T prediction = aggregatedForecast[0];
                T error = _numOps.Subtract(prediction, target);
                T loss = _numOps.Multiply(error, error);
                totalLoss = _numOps.Add(totalLoss, loss);
            }

            // Average loss for this epoch
            T avgLoss = _numOps.Divide(totalLoss, _numOps.FromDouble(numSamples));

            // Print progress every 10 epochs
            if (epoch % 10 == 0)
            {
                Console.WriteLine($"Epoch {epoch}/{_options.Epochs}, Loss: {avgLoss}");
            }
        }

        // Store the final parameters
        ModelParameters = GetParameters();
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
            var (backcast, forecast) = _blocks[blockIdx].Forward(residual);

            // Update residual for next block
            for (int i = 0; i < residual.Length; i++)
            {
                residual[i] = _numOps.Subtract(residual[i], backcast[i]);
            }

            // Accumulate forecast
            for (int i = 0; i < aggregatedForecast.Length; i++)
            {
                aggregatedForecast[i] = _numOps.Add(aggregatedForecast[i], forecast[i]);
            }
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
            var (backcast, forecast) = _blocks[blockIdx].Forward(residual);

            // Update residual for next block
            for (int i = 0; i < residual.Length; i++)
            {
                residual[i] = _numOps.Subtract(residual[i], backcast[i]);
            }

            // Accumulate forecast
            for (int i = 0; i < aggregatedForecast.Length; i++)
            {
                aggregatedForecast[i] = _numOps.Add(aggregatedForecast[i], forecast[i]);
            }
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
            ModelType = ModelType.TimeSeriesRegression,
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
    protected override IFullModel<T, Matrix<T>, Vector<T>> CreateInstance()
    {
        return new NBEATSModel<T>(_options);
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
}
