using AiDotNet.Autodiff;

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
        int numSamples = x.Rows;
        T learningRate = NumOps.FromDouble(_options.LearningRate);

        // Training loop with epochs
        for (int epoch = 0; epoch < _options.Epochs; epoch++)
        {
            T epochLoss = NumOps.Zero;

            // Process in mini-batches
            for (int batchStart = 0; batchStart < numSamples; batchStart += _options.BatchSize)
            {
                int batchEnd = Math.Min(batchStart + _options.BatchSize, numSamples);
                int batchSize = batchEnd - batchStart;

                // Compute gradients for each block using numerical differentiation
                List<Vector<T>> blockGradients = new List<Vector<T>>();

                for (int blockIdx = 0; blockIdx < _blocks.Count; blockIdx++)
                {
                    Vector<T> currentParams = _blocks[blockIdx].GetParameters();
                    Vector<T> gradient = new Vector<T>(currentParams.Length);

                    // Numerical gradient computation (finite differences)
                    T epsilon = NumOps.FromDouble(1e-7);

                    for (int paramIdx = 0; paramIdx < currentParams.Length; paramIdx++)
                    {
                        // Save original value
                        T originalValue = currentParams[paramIdx];

                        // Compute loss with parameter + epsilon
                        currentParams[paramIdx] = NumOps.Add(originalValue, epsilon);
                        _blocks[blockIdx].SetParameters(currentParams);
                        T lossPlus = ComputeBatchLoss(x, y, batchStart, batchEnd);

                        // Compute loss with parameter - epsilon
                        currentParams[paramIdx] = NumOps.Subtract(originalValue, epsilon);
                        _blocks[blockIdx].SetParameters(currentParams);
                        T lossMinus = ComputeBatchLoss(x, y, batchStart, batchEnd);

                        // Restore original value
                        currentParams[paramIdx] = originalValue;
                        _blocks[blockIdx].SetParameters(currentParams);

                        // Compute gradient: (f(x+h) - f(x-h)) / (2h)
                        T gradValue = NumOps.Divide(
                            NumOps.Subtract(lossPlus, lossMinus),
                            NumOps.Multiply(NumOps.FromDouble(2.0), epsilon)
                        );
                        gradient[paramIdx] = gradValue;
                    }

                    blockGradients.Add(gradient);
                }

                // Update parameters using computed gradients
                for (int blockIdx = 0; blockIdx < _blocks.Count; blockIdx++)
                {
                    Vector<T> currentParams = _blocks[blockIdx].GetParameters();
                    Vector<T> gradient = blockGradients[blockIdx];

                    for (int paramIdx = 0; paramIdx < currentParams.Length; paramIdx++)
                    {
                        // Gradient descent: param = param - learningRate * gradient
                        T update = NumOps.Multiply(learningRate, gradient[paramIdx]);
                        currentParams[paramIdx] = NumOps.Subtract(currentParams[paramIdx], update);
                    }

                    _blocks[blockIdx].SetParameters(currentParams);
                }

                // Accumulate batch loss for monitoring
                epochLoss = NumOps.Add(epochLoss, ComputeBatchLoss(x, y, batchStart, batchEnd));
            }

            // Optional: Could log epoch loss here for debugging
            // epochLoss now contains the total loss for this epoch
        }
    }

    /// <summary>
    /// Computes the mean squared error loss for a batch of samples.
    /// </summary>
    /// <param name="x">Input features matrix.</param>
    /// <param name="y">Target values vector.</param>
    /// <param name="batchStart">Starting index of the batch (inclusive).</param>
    /// <param name="batchEnd">Ending index of the batch (exclusive).</param>
    /// <returns>The mean squared error for the batch.</returns>
    private T ComputeBatchLoss(Matrix<T> x, Vector<T> y, int batchStart, int batchEnd)
    {
        T totalLoss = NumOps.Zero;
        int batchSize = batchEnd - batchStart;

        for (int sampleIdx = batchStart; sampleIdx < batchEnd; sampleIdx++)
        {
            // Extract input vector for this sample
            Vector<T> input = new Vector<T>(_options.LookbackWindow);
            for (int j = 0; j < _options.LookbackWindow; j++)
            {
                input[j] = x[sampleIdx, j];
            }

            // Get prediction
            T prediction = PredictSingle(input);

            // Compute squared error
            T error = NumOps.Subtract(prediction, y[sampleIdx]);
            T squaredError = NumOps.Multiply(error, error);
            totalLoss = NumOps.Add(totalLoss, squaredError);
        }

        // Return mean squared error
        return NumOps.Divide(totalLoss, NumOps.FromDouble(batchSize));
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
            var (backcast, forecast) = _blocks[blockIdx].Forward(residual);

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
    /// Gets whether this model supports JIT compilation.
    /// </summary>
    /// <value>
    /// Returns <c>true</c> when the model has been trained and has initialized blocks.
    /// N-BEATS architecture can be represented as a computation graph with the doubly-residual
    /// stacking pattern, enabling JIT compilation for optimized inference.
    /// </value>
    /// <remarks>
    /// <para><b>For Beginners:</b> JIT (Just-In-Time) compilation converts the model's calculations
    /// into optimized native code that runs much faster. N-BEATS can be JIT compiled because
    /// its forward pass can be expressed as a series of matrix operations with residual connections.
    /// </para>
    /// </remarks>
    public override bool SupportsJitCompilation => _blocks.Count > 0;

    /// <summary>
    /// Exports the N-BEATS model as a computation graph for JIT compilation.
    /// </summary>
    /// <param name="inputNodes">A list to which input nodes will be added.</param>
    /// <returns>The output computation node representing the forecast.</returns>
    /// <remarks>
    /// <para>
    /// The computation graph represents the N-BEATS forward pass:
    /// 1. For each block, compute backcast and forecast from the current residual
    /// 2. Update residual: residual = residual - backcast
    /// 3. Accumulate forecast: total_forecast = total_forecast + block_forecast
    /// 4. Return the first element of the aggregated forecast
    /// </para>
    /// <para><b>For Beginners:</b> This converts the entire N-BEATS model into a computation graph
    /// that can be optimized by the JIT compiler. The graph chains all blocks together with
    /// their residual connections, allowing the JIT compiler to:
    /// - Fuse operations across blocks
    /// - Optimize memory usage
    /// - Generate fast native code
    ///
    /// Expected speedup: 3-5x for inference after JIT compilation.
    /// </para>
    /// </remarks>
    public override ComputationNode<T> ExportComputationGraph(List<ComputationNode<T>> inputNodes)
    {
        if (inputNodes == null)
        {
            throw new ArgumentNullException(nameof(inputNodes), "Input nodes list cannot be null.");
        }

        if (_blocks.Count == 0)
        {
            throw new InvalidOperationException("Cannot export computation graph: Model blocks are not initialized.");
        }

        // Create input node (lookback window)
        var inputShape = new int[] { _options.LookbackWindow };
        var inputTensor = new Tensor<T>(inputShape);
        var inputNode = TensorOperations<T>.Variable(inputTensor, "nbeats_input", requiresGradient: false);
        inputNodes.Add(inputNode);

        // Initialize residual as input
        var residual = inputNode;

        // Initialize aggregated forecast with zeros
        var zeroData = new T[_options.ForecastHorizon];
        var numOps = MathHelper.GetNumericOperations<T>();
        for (int i = 0; i < _options.ForecastHorizon; i++)
        {
            zeroData[i] = numOps.Zero;
        }
        var zeroTensor = new Tensor<T>(new[] { _options.ForecastHorizon }, new Vector<T>(zeroData));
        var aggregatedForecast = TensorOperations<T>.Constant(zeroTensor, "initial_forecast");

        // Process each block
        for (int blockIdx = 0; blockIdx < _blocks.Count; blockIdx++)
        {
            // Export block computation graph
            var (backcast, forecast) = _blocks[blockIdx].ExportComputationGraph(residual);

            // Update residual: residual = residual - backcast
            residual = TensorOperations<T>.Subtract(residual, backcast);

            // Accumulate forecast: aggregatedForecast = aggregatedForecast + forecast
            aggregatedForecast = TensorOperations<T>.Add(aggregatedForecast, forecast);
        }

        // Extract first element of forecast (for single-step prediction)
        // Create a slice tensor to extract the first element
        var sliceData = new T[1];
        sliceData[0] = numOps.One;
        var sliceTensor = new Tensor<T>(new[] { 1, _options.ForecastHorizon }, new Vector<T>(CreateSliceWeights(0, _options.ForecastHorizon, numOps)));
        var sliceNode = TensorOperations<T>.Constant(sliceTensor, "forecast_slice");

        // output[0] = slice @ aggregatedForecast
        var outputNode = TensorOperations<T>.MatrixVectorMultiply(sliceNode, aggregatedForecast);

        return outputNode;
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
}
