using AiDotNet.Attributes;
using AiDotNet.Autodiff;
using AiDotNet.Enums;

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
internal class NBEATSBlock<T> : NeuralNetworks.Layers.LayerBase<T>
{
    private readonly int _lookbackWindow;
    private readonly int _forecastHorizon;
    private readonly int _hiddenLayerSize;
    private readonly int _numHiddenLayers;
    private readonly int _thetaSizeBackcast;
    private readonly int _thetaSizeForecast;
    private readonly bool _useInterpretableBasis;
    private readonly int _polynomialDegree;

    /// <summary>
    /// Initializes a new instance with default settings.
    /// </summary>
    public NBEATSBlock()
        : this(64, 16, 128, 4, 64, 16, false)
    {
    }

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
    public override int ParameterCount
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
        : base(new[] { lookbackWindow }, new[] { lookbackWindow + forecastHorizon })
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
        var random = RandomHelper.CreateSeededRandom(42);

        // First layer: lookbackWindow -> hiddenLayerSize
        int inputSize = _lookbackWindow;
        double stddev = Math.Sqrt(2.0 / (inputSize + _hiddenLayerSize));
        var weight = new Matrix<T>(_hiddenLayerSize, inputSize);
        for (int i = 0; i < weight.Rows; i++)
        {
            for (int j = 0; j < weight.Columns; j++)
            {
                weight[i, j] = NumOps.FromDouble(random.NextDouble() * stddev * 2 - stddev);
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
                    weight[i, j] = NumOps.FromDouble(random.NextDouble() * stddev * 2 - stddev);
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
                weight[i, j] = NumOps.FromDouble(random.NextDouble() * stddev * 2 - stddev);
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
                weight[i, j] = NumOps.FromDouble(random.NextDouble() * stddev * 2 - stddev);
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
    /// <summary>
    /// LayerBase Forward — converts Tensor to Vector, runs internal forward, returns concatenated result.
    /// Output tensor layout: [backcast(lookbackWindow) | forecast(forecastHorizon)].
    /// </summary>
    public override Tensor<T> Forward(Tensor<T> input)
    {
        var inputVec = input.ToVector();
        var (backcast, forecast) = ForwardInternal(inputVec);
        var output = new Tensor<T>(new[] { _lookbackWindow + _forecastHorizon });
        for (int i = 0; i < backcast.Length; i++) output[i] = backcast[i];
        for (int i = 0; i < forecast.Length; i++) output[_lookbackWindow + i] = forecast[i];
        return output;
    }

    // Cached activations from forward pass for backward
    private Vector<T>? _lastInput;
    private List<Vector<T>> _preActivations = new();  // before ReLU
    private List<Vector<T>> _postActivations = new(); // after ReLU (layer outputs)
    private Vector<T>? _lastHiddenOutput; // output of last hidden layer (input to theta layers)

    // Stored gradients for UpdateParameters
    private List<Matrix<T>>? _weightGradients;
    private List<Vector<T>>? _biasGradients;

    /// <summary>
    /// Full analytical backward pass through the FC layers using chain rule.
    /// Computes dL/dInput and stores dL/dW, dL/db for UpdateParameters.
    /// </summary>
    public override Tensor<T> Backward(Tensor<T> outputGradient)
    {
        if (_lastInput is null || _postActivations.Count == 0)
            return new Tensor<T>(new[] { _lookbackWindow });

        _weightGradients = new List<Matrix<T>>();
        _biasGradients = new List<Vector<T>>();

        // Output gradient layout: [dL/d_backcast(lookbackWindow) | dL/d_forecast(forecastHorizon)]
        // Extract forecast output gradient
        var dForecast = new Vector<T>(_forecastHorizon);
        for (int i = 0; i < _forecastHorizon && i + _lookbackWindow < outputGradient.Length; i++)
            dForecast[i] = outputGradient[_lookbackWindow + i];

        // Extract backcast output gradient
        var dBackcast = new Vector<T>(_lookbackWindow);
        for (int i = 0; i < _lookbackWindow && i < outputGradient.Length; i++)
            dBackcast[i] = outputGradient[i];

        // Chain rule through basis expansion: output = BasisMatrix @ theta
        // => dL/d_theta = BasisMatrix^T @ dL/d_output
        var fcBasis = ComputeBasisMatrix(_thetaSizeForecast, _forecastHorizon);
        var fcThetaGrad = new Vector<T>(_thetaSizeForecast);
        for (int k = 0; k < _thetaSizeForecast; k++)
        {
            T sum = NumOps.Zero;
            for (int t = 0; t < _forecastHorizon; t++)
                sum = NumOps.Add(sum, NumOps.Multiply(fcBasis[t, k], dForecast[t]));
            fcThetaGrad[k] = sum;
        }

        var bcBasis = ComputeBasisMatrix(_thetaSizeBackcast, _lookbackWindow);
        var bcThetaGrad = new Vector<T>(_thetaSizeBackcast);
        for (int k = 0; k < _thetaSizeBackcast; k++)
        {
            T sum = NumOps.Zero;
            for (int t = 0; t < _lookbackWindow; t++)
                sum = NumOps.Add(sum, NumOps.Multiply(bcBasis[t, k], dBackcast[t]));
            bcThetaGrad[k] = sum;
        }

        // Backward through forecast theta layer using proper theta gradient
        int fcLayerIdx = _numHiddenLayers + 1;
        var fcW = _fcWeights[fcLayerIdx];
        var hiddenOut = _lastHiddenOutput ?? new Vector<T>(fcW.Columns);

        // dL/dW_forecast = dL/d_theta_forecast * hidden^T
        var wGrad = new Matrix<T>(fcW.Rows, fcW.Columns);
        for (int i = 0; i < fcW.Rows; i++)
            for (int j = 0; j < fcW.Columns; j++)
                wGrad[i, j] = NumOps.Multiply(fcThetaGrad[i], hiddenOut[j]);
        _weightGradients.Insert(0, wGrad);
        _biasGradients.Insert(0, fcThetaGrad.Clone());

        // dL/d_hidden from forecast layer: W_forecast^T * dL/d_theta_forecast
        var dHidden = new Vector<T>(fcW.Columns);
        for (int j = 0; j < fcW.Columns; j++)
        {
            T sum = NumOps.Zero;
            for (int i = 0; i < fcW.Rows; i++)
                sum = NumOps.Add(sum, NumOps.Multiply(fcW[i, j], fcThetaGrad[i]));
            dHidden[j] = sum;
        }

        // Backward through backcast theta layer using proper theta gradient
        int bcLayerIdx = _numHiddenLayers;
        var bcW = _fcWeights[bcLayerIdx];

        var bcWGrad = new Matrix<T>(bcW.Rows, bcW.Columns);
        for (int i = 0; i < bcW.Rows; i++)
            for (int j = 0; j < bcW.Columns; j++)
                bcWGrad[i, j] = NumOps.Multiply(bcThetaGrad[i], hiddenOut[j]);
        _weightGradients.Insert(0, bcWGrad);
        _biasGradients.Insert(0, bcThetaGrad.Clone());

        // Add backcast contribution to dHidden: W_backcast^T * dL/d_theta_backcast
        for (int j = 0; j < bcW.Columns; j++)
        {
            T sum = NumOps.Zero;
            for (int i = 0; i < bcW.Rows; i++)
                sum = NumOps.Add(sum, NumOps.Multiply(bcW[i, j], bcThetaGrad[i]));
            dHidden[j] = NumOps.Add(dHidden[j], sum);
        }

        // Backward through hidden layers (reverse order)
        var currentGrad = dHidden;
        for (int layer = _numHiddenLayers - 1; layer >= 0; layer--)
        {
            var preAct = _preActivations[layer];
            var w = _fcWeights[layer];

            // ReLU derivative: gradient passes through where preActivation > 0
            var reluGrad = new Vector<T>(currentGrad.Length);
            for (int i = 0; i < reluGrad.Length; i++)
                reluGrad[i] = NumOps.GreaterThan(preAct[i], NumOps.Zero) ? currentGrad[i] : NumOps.Zero;

            // dL/dW = reluGrad * input^T
            var layerInput = layer > 0 ? _postActivations[layer - 1] : _lastInput;
            var layerWGrad = new Matrix<T>(w.Rows, w.Columns);
            for (int i = 0; i < w.Rows; i++)
                for (int j = 0; j < w.Columns && j < layerInput!.Length; j++)
                    layerWGrad[i, j] = NumOps.Multiply(reluGrad[i], layerInput![j]);
            _weightGradients.Insert(0, layerWGrad);
            _biasGradients.Insert(0, reluGrad.Clone());

            // dL/d_input = W^T * reluGrad (always compute, including layer 0)
            currentGrad = new Vector<T>(w.Columns);
            for (int j = 0; j < w.Columns; j++)
            {
                T sum = NumOps.Zero;
                for (int i = 0; i < w.Rows; i++)
                    sum = NumOps.Add(sum, NumOps.Multiply(w[i, j], reluGrad[i]));
                currentGrad[j] = sum;
            }
        }

        // Convert input gradient to Tensor
        var inputGrad = new Tensor<T>(new[] { _lookbackWindow });
        for (int i = 0; i < _lookbackWindow && i < currentGrad.Length; i++)
            inputGrad[i] = currentGrad[i];
        return inputGrad;
    }

    public override bool SupportsTraining => true;
    public override bool SupportsJitCompilation => true;

    public override void ResetState() { /* stateless layer — no recurrent state to reset */ }

    private static Matrix<T> VectorToColumnMatrix(Vector<T> v)
    {
        var m = new Matrix<T>(v.Length, 1);
        for (int i = 0; i < v.Length; i++) m[i, 0] = v[i];
        return m;
    }

    public override void UpdateParameters(T learningRate)
    {
        if (_weightGradients is null || _biasGradients is null) return;

        for (int l = 0; l < _fcWeights.Count && l < _weightGradients.Count; l++)
        {
            var w = _fcWeights[l];
            var wg = _weightGradients[l];
            for (int i = 0; i < w.Rows; i++)
                for (int j = 0; j < w.Columns; j++)
                    w[i, j] = NumOps.Subtract(w[i, j], NumOps.Multiply(learningRate, wg[i, j]));
        }
        for (int l = 0; l < _fcBiases.Count && l < _biasGradients.Count; l++)
        {
            var b = _fcBiases[l];
            var bg = _biasGradients[l];
            for (int i = 0; i < b.Length && i < bg.Length; i++)
                b[i] = NumOps.Subtract(b[i], NumOps.Multiply(learningRate, bg[i]));
        }

        _weightGradients = null;
        _biasGradients = null;
    }

    public override ComputationNode<T> ExportComputationGraph(List<ComputationNode<T>> nodes)
    {
        if (nodes.Count > 0)
        {
            var (_, forecast) = ExportComputationGraph(nodes[0]);
            return forecast;
        }
        return TensorOperations<T>.Variable(new Tensor<T>(new[] { _forecastHorizon }), "nbeats_output");
    }

    public (Vector<T> backcast, Vector<T> forecast) ForwardInternal(Vector<T> input)
    {
        if (input.Length != _lookbackWindow)
        {
            throw new ArgumentException(
                $"Input length ({input.Length}) must match lookback window ({_lookbackWindow}).",
                nameof(input));
        }

        // Pass through fully connected layers with ReLU activation
        // Uses Engine from LayerBase for hardware-accelerated matrix-vector operations
        // Caches pre/post activations for Backward pass
        _lastInput = input.Clone();
        _preActivations.Clear();
        _postActivations.Clear();
        Vector<T> x = input.Clone();

        for (int layer = 0; layer < _numHiddenLayers; layer++)
        {
            // Accelerated linear transformation: y = Wx + b
            var wxMatrix = (Matrix<T>)Engine.MatrixMultiply(
                _fcWeights[layer],
                VectorToColumnMatrix(x));
            Vector<T> linear = new Vector<T>(_fcWeights[layer].Rows);
            for (int i = 0; i < linear.Length; i++)
            {
                linear[i] = NumOps.Add(_fcBiases[layer][i], wxMatrix[i, 0]);
            }
            _preActivations.Add(linear.Clone());

            // ReLU activation
            x = new Vector<T>(linear.Length);
            for (int i = 0; i < linear.Length; i++)
            {
                x[i] = NumOps.GreaterThan(linear[i], NumOps.Zero) ? linear[i] : NumOps.Zero;
            }
            _postActivations.Add(x.Clone());
        }
        _lastHiddenOutput = x.Clone();

        // Accelerated theta for backcast: theta = W_b * x + b_b
        int backcastLayerIdx = _numHiddenLayers;
        var bcWx = (Matrix<T>)Engine.MatrixMultiply(
            _fcWeights[backcastLayerIdx],
            VectorToColumnMatrix(x));
        Vector<T> thetaBackcast = new Vector<T>(_thetaSizeBackcast);
        for (int i = 0; i < _thetaSizeBackcast; i++)
        {
            thetaBackcast[i] = NumOps.Add(_fcBiases[backcastLayerIdx][i], bcWx[i, 0]);
        }

        // Accelerated theta for forecast: theta = W_f * x + b_f
        int forecastLayerIdx = _numHiddenLayers + 1;
        var fcWx = (Matrix<T>)Engine.MatrixMultiply(
            _fcWeights[forecastLayerIdx],
            VectorToColumnMatrix(x));
        Vector<T> thetaForecast = new Vector<T>(_thetaSizeForecast);
        for (int i = 0; i < _thetaSizeForecast; i++)
        {
            thetaForecast[i] = NumOps.Add(_fcBiases[forecastLayerIdx][i], fcWx[i, 0]);
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
    /// <summary>
    /// Computes the basis matrix B such that output = B @ theta.
    /// Shape: [outputLength, thetaSize].
    /// </summary>
    private Matrix<T> ComputeBasisMatrix(int thetaSize, int outputLength)
    {
        var basis = new Matrix<T>(outputLength, thetaSize);

        if (_useInterpretableBasis)
        {
            // Polynomial basis: B[t, p] = (t/outputLength)^p
            for (int t = 0; t < outputLength; t++)
            {
                double tNormalized = (double)t / outputLength;
                for (int p = 0; p < Math.Min(thetaSize, _polynomialDegree + 1); p++)
                {
                    basis[t, p] = NumOps.FromDouble(Math.Pow(tNormalized, p));
                }
            }
        }
        else
        {
            // Generic basis: B[t, k] = cos(2π * k * t / outputLength)
            for (int t = 0; t < outputLength; t++)
            {
                for (int k = 0; k < thetaSize; k++)
                {
                    basis[t, k] = NumOps.FromDouble(
                        Math.Cos(2.0 * Math.PI * k * t / outputLength));
                }
            }
        }

        return basis;
    }

    private Vector<T> ApplyBasisExpansion(Vector<T> theta, int outputLength)
    {
        Vector<T> output = new Vector<T>(outputLength);

        if (_useInterpretableBasis)
        {
            // Polynomial basis for trend (interpretable)
            // Each time step t gets a polynomial: theta_0 + theta_1*t + theta_2*t^2 + ...
            for (int t = 0; t < outputLength; t++)
            {
                T value = NumOps.Zero;
                T tNormalized = NumOps.FromDouble((double)t / outputLength);

                for (int p = 0; p < Math.Min(theta.Length, _polynomialDegree + 1); p++)
                {
                    T power = NumOps.One;
                    for (int k = 0; k < p; k++)
                    {
                        power = NumOps.Multiply(power, tNormalized);
                    }
                    value = NumOps.Add(value, NumOps.Multiply(theta[p], power));
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
                T value = NumOps.Zero;
                for (int k = 0; k < theta.Length; k++)
                {
                    // Use a simple projection where each theta contributes to each time step
                    T contribution = NumOps.Multiply(
                        theta[k],
                        NumOps.FromDouble(Math.Cos(2.0 * Math.PI * k * t / outputLength))
                    );
                    value = NumOps.Add(value, contribution);
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
    public override Vector<T> GetParameters()
    {
        var parameters = new List<T>();

        // VECTORIZED: Use row operations to collect weight parameters
        foreach (var weight in _fcWeights)
        {
            for (int i = 0; i < weight.Rows; i++)
            {
                Vector<T> row = weight.GetRow(i);
                parameters.AddRange(row);
            }
        }

        // VECTORIZED: Use AddRange to collect bias parameters
        foreach (var bias in _fcBiases)
        {
            parameters.AddRange(bias);
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
    public override void SetParameters(Vector<T> parameters)
    {
        if (parameters.Length != ParameterCount)
        {
            throw new ArgumentException(
                $"Expected {ParameterCount} parameters, but got {parameters.Length}.",
                nameof(parameters));
        }

        int idx = 0;

        // VECTORIZED: Use SetRow to assign weight parameters
        foreach (var weight in _fcWeights)
        {
            for (int i = 0; i < weight.Rows; i++)
            {
                T[] rowData = new T[weight.Columns];
                for (int j = 0; j < weight.Columns; j++)
                {
                    rowData[j] = parameters[idx++];
                }
                weight.SetRow(i, new Vector<T>(rowData));
            }
        }

        // VECTORIZED: Use Slice to assign bias parameters
        foreach (var bias in _fcBiases)
        {
            Vector<T> biasSlice = parameters.Slice(idx, bias.Length);
            for (int i = 0; i < bias.Length; i++)
            {
                bias[i] = biasSlice[i];
            }
            idx += bias.Length;
        }
    }

    /// <summary>
    /// Exports the block as computation graph nodes for JIT compilation.
    /// </summary>
    /// <param name="inputNode">The input computation node (residual from previous block).</param>
    /// <returns>A tuple containing (backcast, forecast) computation nodes.</returns>
    /// <remarks>
    /// <para>
    /// This method creates a computation graph that represents the forward pass through
    /// the N-BEATS block, enabling JIT compilation for optimized inference.
    /// </para>
    /// <para><b>For Beginners:</b> This converts the block's calculations into a format
    /// that can be optimized by the JIT compiler. The resulting computation graph
    /// represents:
    /// 1. Passing input through fully connected layers with ReLU activation
    /// 2. Computing theta parameters for backcast and forecast
    /// 3. Applying basis expansion to generate backcast and forecast
    /// </para>
    /// </remarks>
    public (ComputationNode<T> backcast, ComputationNode<T> forecast) ExportComputationGraph(ComputationNode<T> inputNode)
    {
        var numOps = MathHelper.GetNumericOperations<T>();

        // Start with the input
        var x = inputNode;

        // Pass through fully connected layers with ReLU activation
        for (int layer = 0; layer < _numHiddenLayers; layer++)
        {
            // Convert weight matrix to tensor [hidden_size, input_size]
            var weightTensor = MatrixToTensor(_fcWeights[layer]);
            var weightNode = TensorOperations<T>.Constant(weightTensor, $"block_fc{layer}_weight");

            // Convert bias to tensor [hidden_size]
            var biasTensor = VectorToTensor(_fcBiases[layer]);
            var biasNode = TensorOperations<T>.Constant(biasTensor, $"block_fc{layer}_bias");

            // Linear transformation: y = W @ x + b
            var linear = TensorOperations<T>.MatrixVectorMultiply(weightNode, x);
            linear = TensorOperations<T>.Add(linear, biasNode);

            // ReLU activation
            x = TensorOperations<T>.ReLU(linear);
        }

        // Compute theta for backcast
        var backcastWeightTensor = MatrixToTensor(_fcWeights[_numHiddenLayers]);
        var backcastWeightNode = TensorOperations<T>.Constant(backcastWeightTensor, "block_backcast_weight");
        var backcastBiasTensor = VectorToTensor(_fcBiases[_numHiddenLayers]);
        var backcastBiasNode = TensorOperations<T>.Constant(backcastBiasTensor, "block_backcast_bias");

        var thetaBackcast = TensorOperations<T>.MatrixVectorMultiply(backcastWeightNode, x);
        thetaBackcast = TensorOperations<T>.Add(thetaBackcast, backcastBiasNode);

        // Compute theta for forecast
        var forecastWeightTensor = MatrixToTensor(_fcWeights[_numHiddenLayers + 1]);
        var forecastWeightNode = TensorOperations<T>.Constant(forecastWeightTensor, "block_forecast_weight");
        var forecastBiasTensor = VectorToTensor(_fcBiases[_numHiddenLayers + 1]);
        var forecastBiasNode = TensorOperations<T>.Constant(forecastBiasTensor, "block_forecast_bias");

        var thetaForecast = TensorOperations<T>.MatrixVectorMultiply(forecastWeightNode, x);
        thetaForecast = TensorOperations<T>.Add(thetaForecast, forecastBiasNode);

        // Apply basis expansion
        var backcastNode = ApplyBasisExpansionGraph(thetaBackcast, _lookbackWindow, isBackcast: true);
        var forecastNode = ApplyBasisExpansionGraph(thetaForecast, _forecastHorizon, isBackcast: false);

        return (backcastNode, forecastNode);
    }

    /// <summary>
    /// Applies basis expansion in the computation graph.
    /// </summary>
    private ComputationNode<T> ApplyBasisExpansionGraph(ComputationNode<T> theta, int outputLength, bool isBackcast)
    {
        var numOps = MathHelper.GetNumericOperations<T>();

        if (_useInterpretableBasis)
        {
            // Polynomial basis expansion: output[t] = sum(theta[p] * t^p)
            // Create the basis matrix [output_length, theta_size] where basis[t, p] = (t/outputLength)^p
            var basisData = new T[outputLength * theta.Value.Shape[0]];
            int thetaSize = theta.Value.Shape[0];

            for (int t = 0; t < outputLength; t++)
            {
                double tNormalized = (double)t / outputLength;
                for (int p = 0; p < Math.Min(thetaSize, _polynomialDegree + 1); p++)
                {
                    double power = Math.Pow(tNormalized, p);
                    basisData[t * thetaSize + p] = numOps.FromDouble(power);
                }
            }

            var basisTensor = new Tensor<T>(new[] { outputLength, thetaSize }, new Vector<T>(basisData));
            var basisNode = TensorOperations<T>.Constant(basisTensor, isBackcast ? "backcast_basis" : "forecast_basis");

            // output = basis @ theta
            return TensorOperations<T>.MatrixVectorMultiply(basisNode, theta);
        }
        else
        {
            // Generic basis: Fourier-like projection
            // Create the basis matrix where basis[t, k] = cos(2π * k * t / outputLength)
            var basisData = new T[outputLength * theta.Value.Shape[0]];
            int thetaSize = theta.Value.Shape[0];

            for (int t = 0; t < outputLength; t++)
            {
                for (int k = 0; k < thetaSize; k++)
                {
                    double cosValue = Math.Cos(2.0 * Math.PI * k * t / outputLength);
                    basisData[t * thetaSize + k] = numOps.FromDouble(cosValue);
                }
            }

            var basisTensor = new Tensor<T>(new[] { outputLength, thetaSize }, new Vector<T>(basisData));
            var basisNode = TensorOperations<T>.Constant(basisTensor, isBackcast ? "backcast_basis" : "forecast_basis");

            // output = basis @ theta
            return TensorOperations<T>.MatrixVectorMultiply(basisNode, theta);
        }
    }

    /// <summary>
    /// Converts a Matrix to a Tensor for use in computation graphs.
    /// </summary>
    private Tensor<T> MatrixToTensor(Matrix<T> matrix)
    {
        var data = new T[matrix.Rows * matrix.Columns];
        for (int i = 0; i < matrix.Rows; i++)
        {
            for (int j = 0; j < matrix.Columns; j++)
            {
                data[i * matrix.Columns + j] = matrix[i, j];
            }
        }
        return new Tensor<T>(new[] { matrix.Rows, matrix.Columns }, new Vector<T>(data));
    }

    /// <summary>
    /// Converts a Vector to a Tensor for use in computation graphs.
    /// </summary>
    private Tensor<T> VectorToTensor(Vector<T> vector)
    {
        var data = new T[vector.Length];
        for (int i = 0; i < vector.Length; i++)
        {
            data[i] = vector[i];
        }
        return new Tensor<T>(new[] { vector.Length }, new Vector<T>(data));
    }
}
