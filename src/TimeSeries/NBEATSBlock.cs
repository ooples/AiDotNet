namespace AiDotNet.TimeSeries;

/// <summary>
/// Represents a single block in the N-BEATS architecture.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// Each N-BEATS block consists of:
/// 1. A stack of fully connected layers (the "theta" network)
/// 2. A basis expansion layer for generating backcast (reconstruction of input)
/// 3. A basis expansion layer for generating forecast (prediction of future)
/// </para>
/// <para>
/// The block architecture implements a doubly residual stacking principle:
/// - Backcast residual: Input minus backcast is passed to the next block
/// - Forecast addition: Forecasts from all blocks are summed for the final prediction
/// </para>
/// <para><b>For Beginners:</b> A block is the basic building unit of N-BEATS. Think of it like
/// a specialized predictor that:
/// 1. Looks at the input time series
/// 2. Tries to reconstruct what it saw (backcast)
/// 3. Predicts the future (forecast)
/// 4. Passes the "leftover" patterns it couldn't explain to the next block
///
/// Multiple blocks work together, with each one focusing on different aspects of the data.
/// </para>
/// </remarks>
public class NBEATSBlock<T>
{
    private readonly INumericOperations<T> _numOps;
    private readonly int _lookbackWindow;
    private readonly int _forecastHorizon;
    private readonly int _hiddenLayerSize;
    private readonly int _numHiddenLayers;
    private readonly int _thetaSizeBackcast;
    private readonly int _thetaSizeForecast;
    private readonly bool _useInterpretableBasis;
    private readonly int _polynomialDegree;

    /// <summary>
    /// Weights for the fully connected layers (theta network).
    /// </summary>
    private List<Matrix<T>> _fcWeights;

    /// <summary>
    /// Biases for the fully connected layers (theta network).
    /// </summary>
    private List<Vector<T>> _fcBiases;

    /// <summary>
    /// Gets the total number of trainable parameters in the block.
    /// </summary>
    public int ParameterCount
    {
        get
        {
            int count = 0;
            foreach (var weight in _fcWeights)
            {
                count += weight.Rows * weight.Columns;
            }
            foreach (var bias in _fcBiases)
            {
                count += bias.Length;
            }
            return count;
        }
    }

    /// <summary>
    /// Initializes a new instance of the NBEATSBlock class.
    /// </summary>
    /// <param name="lookbackWindow">The number of historical time steps used as input.</param>
    /// <param name="forecastHorizon">The number of future time steps to predict.</param>
    /// <param name="hiddenLayerSize">The size of hidden layers in the fully connected network.</param>
    /// <param name="numHiddenLayers">The number of hidden layers.</param>
    /// <param name="thetaSizeBackcast">The size of the theta vector for backcast basis expansion.</param>
    /// <param name="thetaSizeForecast">The size of the theta vector for forecast basis expansion.</param>
    /// <param name="useInterpretableBasis">Whether to use interpretable basis functions.</param>
    /// <param name="polynomialDegree">The polynomial degree for trend basis (if interpretable).</param>
    /// <remarks>
    /// <para><b>For Beginners:</b> This creates a new block with specific parameters:
    /// - lookbackWindow: How far back in time the block looks
    /// - forecastHorizon: How far forward in time the block predicts
    /// - hiddenLayerSize: How many neurons in each hidden layer (bigger = more capacity)
    /// - numHiddenLayers: How many hidden layers (deeper = more complex patterns)
    /// - useInterpretableBasis: Whether to use human-understandable basis functions
    /// </para>
    /// </remarks>
    public NBEATSBlock(
        int lookbackWindow,
        int forecastHorizon,
        int hiddenLayerSize,
        int numHiddenLayers,
        int thetaSizeBackcast,
        int thetaSizeForecast,
        bool useInterpretableBasis,
        int polynomialDegree = 3)
    {
        if (lookbackWindow <= 0)
        {
            throw new ArgumentException("Lookback window must be positive.", nameof(lookbackWindow));
        }
        if (forecastHorizon <= 0)
        {
            throw new ArgumentException("Forecast horizon must be positive.", nameof(forecastHorizon));
        }
        if (hiddenLayerSize <= 0)
        {
            throw new ArgumentException("Hidden layer size must be positive.", nameof(hiddenLayerSize));
        }
        if (numHiddenLayers <= 0)
        {
            throw new ArgumentException("Number of hidden layers must be positive.", nameof(numHiddenLayers));
        }

        _numOps = MathHelper.GetNumericOperations<T>();
        _lookbackWindow = lookbackWindow;
        _forecastHorizon = forecastHorizon;
        _hiddenLayerSize = hiddenLayerSize;
        _numHiddenLayers = numHiddenLayers;
        _thetaSizeBackcast = thetaSizeBackcast;
        _thetaSizeForecast = thetaSizeForecast;
        _useInterpretableBasis = useInterpretableBasis;
        _polynomialDegree = polynomialDegree;

        _fcWeights = new List<Matrix<T>>();
        _fcBiases = new List<Vector<T>>();

        InitializeWeights();
    }

    /// <summary>
    /// Initializes the weights and biases for the fully connected layers.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Uses Xavier/Glorot initialization to set initial weights, which helps with
    /// training stability by keeping the scale of gradients roughly the same across layers.
    /// </para>
    /// <para><b>For Beginners:</b> This sets up the initial random values for all the
    /// weights and biases in the block. Good initialization is important for the model
    /// to learn effectively. We use a special technique (Xavier initialization) that
    /// has been proven to work well for neural networks.
    /// </para>
    /// </remarks>
    private void InitializeWeights()
    {
        var random = new Random(42);

        // First layer: lookbackWindow -> hiddenLayerSize
        int inputSize = _lookbackWindow;
        double stddev = Math.Sqrt(2.0 / (inputSize + _hiddenLayerSize));
        var weight = new Matrix<T>(_hiddenLayerSize, inputSize);
        for (int i = 0; i < weight.Rows; i++)
        {
            for (int j = 0; j < weight.Columns; j++)
            {
                weight[i, j] = _numOps.FromDouble(random.NextDouble() * stddev * 2 - stddev);
            }
        }
        _fcWeights.Add(weight);
        _fcBiases.Add(new Vector<T>(_hiddenLayerSize));

        // Hidden layers: hiddenLayerSize -> hiddenLayerSize
        for (int layer = 1; layer < _numHiddenLayers; layer++)
        {
            stddev = Math.Sqrt(2.0 / (_hiddenLayerSize + _hiddenLayerSize));
            weight = new Matrix<T>(_hiddenLayerSize, _hiddenLayerSize);
            for (int i = 0; i < weight.Rows; i++)
            {
                for (int j = 0; j < weight.Columns; j++)
                {
                    weight[i, j] = _numOps.FromDouble(random.NextDouble() * stddev * 2 - stddev);
                }
            }
            _fcWeights.Add(weight);
            _fcBiases.Add(new Vector<T>(_hiddenLayerSize));
        }

        // Output layer for backcast theta: hiddenLayerSize -> thetaSizeBackcast
        stddev = Math.Sqrt(2.0 / (_hiddenLayerSize + _thetaSizeBackcast));
        weight = new Matrix<T>(_thetaSizeBackcast, _hiddenLayerSize);
        for (int i = 0; i < weight.Rows; i++)
        {
            for (int j = 0; j < weight.Columns; j++)
            {
                weight[i, j] = _numOps.FromDouble(random.NextDouble() * stddev * 2 - stddev);
            }
        }
        _fcWeights.Add(weight);
        _fcBiases.Add(new Vector<T>(_thetaSizeBackcast));

        // Output layer for forecast theta: hiddenLayerSize -> thetaSizeForecast
        stddev = Math.Sqrt(2.0 / (_hiddenLayerSize + _thetaSizeForecast));
        weight = new Matrix<T>(_thetaSizeForecast, _hiddenLayerSize);
        for (int i = 0; i < weight.Rows; i++)
        {
            for (int j = 0; j < weight.Columns; j++)
            {
                weight[i, j] = _numOps.FromDouble(random.NextDouble() * stddev * 2 - stddev);
            }
        }
        _fcWeights.Add(weight);
        _fcBiases.Add(new Vector<T>(_thetaSizeForecast));
    }

    /// <summary>
    /// Performs the forward pass through the block.
    /// </summary>
    /// <param name="input">The input time series vector of length lookbackWindow.</param>
    /// <returns>A tuple containing (backcast, forecast) vectors.</returns>
    /// <remarks>
    /// <para>
    /// The forward pass:
    /// 1. Passes input through fully connected layers with ReLU activation
    /// 2. Computes theta parameters for backcast and forecast
    /// 3. Applies basis expansion to generate backcast and forecast
    /// </para>
    /// <para><b>For Beginners:</b> This is where the block actually processes the input data.
    /// It takes the historical time series, runs it through the neural network layers,
    /// and produces two outputs:
    /// - Backcast: The block's attempt to reconstruct the input (what it understood)
    /// - Forecast: The block's prediction of future values
    /// </para>
    /// </remarks>
    public (Vector<T> backcast, Vector<T> forecast) Forward(Vector<T> input)
    {
        if (input.Length != _lookbackWindow)
        {
            throw new ArgumentException(
                $"Input length ({input.Length}) must match lookback window ({_lookbackWindow}).",
                nameof(input));
        }

        // Pass through fully connected layers with ReLU activation
        Vector<T> x = input.Clone();

        for (int layer = 0; layer < _numHiddenLayers; layer++)
        {
            // Linear transformation: y = Wx + b
            Vector<T> linear = new Vector<T>(_fcWeights[layer].Rows);
            for (int i = 0; i < _fcWeights[layer].Rows; i++)
            {
                T sum = _fcBiases[layer][i];
                for (int j = 0; j < _fcWeights[layer].Columns; j++)
                {
                    sum = _numOps.Add(sum, _numOps.Multiply(_fcWeights[layer][i, j], x[j]));
                }
                linear[i] = sum;
            }

            // ReLU activation
            x = new Vector<T>(linear.Length);
            for (int i = 0; i < linear.Length; i++)
            {
                x[i] = _numOps.GreaterThan(linear[i], _numOps.Zero) ? linear[i] : _numOps.Zero;
            }
        }

        // Compute theta for backcast
        Vector<T> thetaBackcast = new Vector<T>(_thetaSizeBackcast);
        int backcastLayerIdx = _numHiddenLayers;
        for (int i = 0; i < _fcWeights[backcastLayerIdx].Rows; i++)
        {
            T sum = _fcBiases[backcastLayerIdx][i];
            for (int j = 0; j < _fcWeights[backcastLayerIdx].Columns; j++)
            {
                sum = _numOps.Add(sum, _numOps.Multiply(_fcWeights[backcastLayerIdx][i, j], x[j]));
            }
            thetaBackcast[i] = sum;
        }

        // Compute theta for forecast
        Vector<T> thetaForecast = new Vector<T>(_thetaSizeForecast);
        int forecastLayerIdx = _numHiddenLayers + 1;
        for (int i = 0; i < _fcWeights[forecastLayerIdx].Rows; i++)
        {
            T sum = _fcBiases[forecastLayerIdx][i];
            for (int j = 0; j < _fcWeights[forecastLayerIdx].Columns; j++)
            {
                sum = _numOps.Add(sum, _numOps.Multiply(_fcWeights[forecastLayerIdx][i, j], x[j]));
            }
            thetaForecast[i] = sum;
        }

        // Apply basis expansion
        Vector<T> backcast = ApplyBasisExpansion(thetaBackcast, _lookbackWindow);
        Vector<T> forecast = ApplyBasisExpansion(thetaForecast, _forecastHorizon);

        return (backcast, forecast);
    }

    /// <summary>
    /// Applies basis expansion to theta parameters to generate time series outputs.
    /// </summary>
    /// <param name="theta">The theta parameter vector.</param>
    /// <param name="outputLength">The desired output length (lookbackWindow or forecastHorizon).</param>
    /// <returns>The expanded time series vector.</returns>
    /// <remarks>
    /// <para>
    /// For interpretable basis, uses polynomial basis for trend.
    /// For generic basis, uses a learned linear transformation.
    /// </para>
    /// <para><b>For Beginners:</b> Basis expansion is how we convert the neural network's
    /// output (theta) into actual time series values. Think of theta as a compact representation
    /// and this function as expanding it into a full time series.
    ///
    /// - Interpretable basis: Uses mathematical functions like polynomials that we can understand
    /// - Generic basis: Uses learned transformations that may be more flexible
    /// </para>
    /// </remarks>
    private Vector<T> ApplyBasisExpansion(Vector<T> theta, int outputLength)
    {
        Vector<T> output = new Vector<T>(outputLength);

        if (_useInterpretableBasis)
        {
            // Polynomial basis for trend (interpretable)
            // Each time step t gets a polynomial: theta_0 + theta_1*t + theta_2*t^2 + ...
            for (int t = 0; t < outputLength; t++)
            {
                T value = _numOps.Zero;
                T tNormalized = _numOps.FromDouble((double)t / outputLength);

                for (int p = 0; p < Math.Min(theta.Length, _polynomialDegree + 1); p++)
                {
                    T power = _numOps.One;
                    for (int k = 0; k < p; k++)
                    {
                        power = _numOps.Multiply(power, tNormalized);
                    }
                    value = _numOps.Add(value, _numOps.Multiply(theta[p], power));
                }

                output[t] = value;
            }
        }
        else
        {
            // Generic basis: simple linear projection
            // In a full implementation, this would use a learned basis matrix
            // For now, we use a simple approach where theta directly maps to output
            for (int t = 0; t < outputLength; t++)
            {
                T value = _numOps.Zero;
                for (int k = 0; k < theta.Length; k++)
                {
                    // Use a simple projection where each theta contributes to each time step
                    T contribution = _numOps.Multiply(
                        theta[k],
                        _numOps.FromDouble(Math.Cos(2.0 * Math.PI * k * t / outputLength))
                    );
                    value = _numOps.Add(value, contribution);
                }
                output[t] = value;
            }
        }

        return output;
    }

    /// <summary>
    /// Gets all parameters (weights and biases) as a single vector.
    /// </summary>
    /// <returns>A vector containing all trainable parameters.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method collects all the weights and biases from
    /// all layers into one long vector. This is useful for saving the model or for
    /// certain optimization algorithms.
    /// </para>
    /// </remarks>
    public Vector<T> GetParameters()
    {
        var parameters = new List<T>();

        foreach (var weight in _fcWeights)
        {
            for (int i = 0; i < weight.Rows; i++)
            {
                for (int j = 0; j < weight.Columns; j++)
                {
                    parameters.Add(weight[i, j]);
                }
            }
        }

        foreach (var bias in _fcBiases)
        {
            for (int i = 0; i < bias.Length; i++)
            {
                parameters.Add(bias[i]);
            }
        }

        return new Vector<T>(parameters.ToArray());
    }

    /// <summary>
    /// Sets all parameters (weights and biases) from a single vector.
    /// </summary>
    /// <param name="parameters">A vector containing all trainable parameters.</param>
    /// <remarks>
    /// <para><b>For Beginners:</b> This is the opposite of GetParameters - it takes
    /// a long vector of numbers and distributes them back to all the weights and biases
    /// in the block. This is useful for loading a saved model.
    /// </para>
    /// </remarks>
    public void SetParameters(Vector<T> parameters)
    {
        if (parameters.Length != ParameterCount)
        {
            throw new ArgumentException(
                $"Expected {ParameterCount} parameters, but got {parameters.Length}.",
                nameof(parameters));
        }

        int idx = 0;

        foreach (var weight in _fcWeights)
        {
            for (int i = 0; i < weight.Rows; i++)
            {
                for (int j = 0; j < weight.Columns; j++)
                {
                    weight[i, j] = parameters[idx++];
                }
            }
        }

        foreach (var bias in _fcBiases)
        {
            for (int i = 0; i < bias.Length; i++)
            {
                bias[i] = parameters[idx++];
            }
        }
    }
}
