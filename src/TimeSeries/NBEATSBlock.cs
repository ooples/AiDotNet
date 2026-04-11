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
    /// Weights for the fully connected layers (theta network), stored as Tensor&lt;T&gt;
    /// for tape-based automatic differentiation.
    /// </summary>
    private List<Tensor<T>> _fcWeights;

    /// <summary>
    /// Biases for the fully connected layers (theta network), stored as Tensor&lt;T&gt;
    /// for tape-based automatic differentiation.
    /// </summary>
    private List<Tensor<T>> _fcBiases;

    /// <summary>
    /// Precomputed basis matrix for backcast expansion: [lookbackWindow, thetaSizeBackcast].
    /// </summary>
    private Tensor<T> _basisBackcast;

    /// <summary>
    /// Precomputed basis matrix for forecast expansion: [forecastHorizon, thetaSizeForecast].
    /// </summary>
    private Tensor<T> _basisForecast;

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
                count += weight.Length;
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

        _fcWeights = new List<Tensor<T>>();
        _fcBiases = new List<Tensor<T>>();

        if (_useInterpretableBasis)
        {
            // Interpretable blocks: fixed polynomial basis (not trainable)
            // Per Oreshkin et al. 2020 Section 3.3
            _basisBackcast = ComputeBasisTensor(_thetaSizeBackcast, _lookbackWindow);
            _basisForecast = ComputeBasisTensor(_thetaSizeForecast, _forecastHorizon);
        }
        else
        {
            // Generic blocks: V_b and V_f are fully learnable linear functions.
            // Per Oreshkin et al. 2020 Section 3.2:
            // "In the generic architecture, we do not restrict g^b and g^f to a
            //  particular functional form, and instead make them fully learnable"
            // Initialize near identity for stable initial behavior.
            var data_b = new T[_lookbackWindow * _thetaSizeBackcast];
            var data_f = new T[_forecastHorizon * _thetaSizeForecast];
            for (int i = 0; i < _lookbackWindow; i++)
                for (int j = 0; j < _thetaSizeBackcast; j++)
                    data_b[i * _thetaSizeBackcast + j] = (i == j) ? NumOps.One : NumOps.Zero;
            for (int i = 0; i < _forecastHorizon; i++)
                for (int j = 0; j < _thetaSizeForecast; j++)
                    data_f[i * _thetaSizeForecast + j] = (i == j) ? NumOps.One : NumOps.Zero;
            _basisBackcast = new Tensor<T>(data_b, new[] { _lookbackWindow, _thetaSizeBackcast });
            _basisForecast = new Tensor<T>(data_f, new[] { _forecastHorizon, _thetaSizeForecast });
        }

        InitializeWeights();
    }

    /// <summary>
    /// Initializes the weights and biases for the fully connected layers.
    /// Uses He initialization for ReLU networks and registers all parameters as trainable
    /// for tape-based autodiff.
    /// </summary>
    private void InitializeWeights()
    {
        var random = RandomHelper.CreateSeededRandom(42);

        // First layer: lookbackWindow -> hiddenLayerSize
        int inputSize = _lookbackWindow;
        double stddev = Math.Sqrt(2.0 / inputSize);
        var weight = CreateWeightTensor(_hiddenLayerSize, inputSize, stddev, random);
        _fcWeights.Add(weight);
        RegisterTrainableParameter(weight, PersistentTensorRole.Weights);

        var bias = CreateBiasTensor(_hiddenLayerSize, 0.01);
        _fcBiases.Add(bias);
        RegisterTrainableParameter(bias, PersistentTensorRole.Biases);

        // Hidden layers: hiddenLayerSize -> hiddenLayerSize
        for (int layer = 1; layer < _numHiddenLayers; layer++)
        {
            stddev = Math.Sqrt(2.0 / _hiddenLayerSize);
            weight = CreateWeightTensor(_hiddenLayerSize, _hiddenLayerSize, stddev, random);
            _fcWeights.Add(weight);
            RegisterTrainableParameter(weight, PersistentTensorRole.Weights);

            bias = CreateBiasTensor(_hiddenLayerSize, 0.01);
            _fcBiases.Add(bias);
            RegisterTrainableParameter(bias, PersistentTensorRole.Biases);
        }

        // Output layer for backcast theta: hiddenLayerSize -> thetaSizeBackcast
        stddev = Math.Sqrt(2.0 / (_hiddenLayerSize + _thetaSizeBackcast));
        weight = CreateWeightTensor(_thetaSizeBackcast, _hiddenLayerSize, stddev, random);
        _fcWeights.Add(weight);
        RegisterTrainableParameter(weight, PersistentTensorRole.Weights);

        bias = CreateBiasTensor(_thetaSizeBackcast, 0.0);
        _fcBiases.Add(bias);
        RegisterTrainableParameter(bias, PersistentTensorRole.Biases);

        // Output layer for forecast theta: hiddenLayerSize -> thetaSizeForecast
        stddev = Math.Sqrt(2.0 / (_hiddenLayerSize + _thetaSizeForecast));
        weight = CreateWeightTensor(_thetaSizeForecast, _hiddenLayerSize, stddev, random);
        _fcWeights.Add(weight);
        RegisterTrainableParameter(weight, PersistentTensorRole.Weights);

        bias = CreateBiasTensor(_thetaSizeForecast, 0.0);
        _fcBiases.Add(bias);
        RegisterTrainableParameter(bias, PersistentTensorRole.Biases);

        // For generic blocks: register V_b and V_f as trainable
        // Per Oreshkin et al. 2020 Section 3.2
        if (!_useInterpretableBasis)
        {
            RegisterTrainableParameter(_basisBackcast, PersistentTensorRole.Weights);
            RegisterTrainableParameter(_basisForecast, PersistentTensorRole.Weights);
        }
    }

    /// <summary>
    /// Creates a weight tensor with He initialization.
    /// </summary>
    private Tensor<T> CreateWeightTensor(int rows, int cols, double stddev, Random random)
    {
        var data = new T[rows * cols];
        for (int i = 0; i < data.Length; i++)
        {
            data[i] = NumOps.FromDouble(random.NextDouble() * stddev * 2 - stddev);
        }
        return new Tensor<T>(new[] { rows, cols }, new Vector<T>(data));
    }

    /// <summary>
    /// Creates a bias tensor initialized to a constant value.
    /// </summary>
    private Tensor<T> CreateBiasTensor(int size, double initValue)
    {
        var data = new T[size];
        for (int i = 0; i < size; i++)
        {
            data[i] = NumOps.FromDouble(initValue);
        }
        return new Tensor<T>(new[] { size }, new Vector<T>(data));
    }

    /// <summary>
    /// LayerBase Forward -- uses tape-tracked Engine operations for automatic differentiation.
    /// Output tensor layout: [backcast(lookbackWindow) | forecast(forecastHorizon)].
    /// </summary>
    public override Tensor<T> Forward(Tensor<T> input)
    {
        // Use Engine.Reshape for tape-tracked reshaping
        var x = Engine.Reshape(input, [_lookbackWindow, 1]);

        // Pass through hidden layers with ReLU
        for (int layer = 0; layer < _numHiddenLayers; layer++)
        {
            // Linear: y = W @ x + b
            var linear = Engine.TensorMatMul(_fcWeights[layer], x);
            // Add bias: reshape bias to column [hidden, 1]
            var biasCol = Engine.Reshape(_fcBiases[layer], [_hiddenLayerSize, 1]);
            linear = Engine.TensorAdd(linear, biasCol);
            // ReLU activation
            x = Engine.ReLU(linear);
        }

        // Compute theta for backcast: [thetaSizeBackcast, 1]
        int backcastLayerIdx = _numHiddenLayers;
        var thetaBackcast = Engine.TensorMatMul(_fcWeights[backcastLayerIdx], x);
        var bcBiasCol = Engine.Reshape(_fcBiases[backcastLayerIdx], [_thetaSizeBackcast, 1]);
        thetaBackcast = Engine.TensorAdd(thetaBackcast, bcBiasCol);

        // Compute theta for forecast: [thetaSizeForecast, 1]
        int forecastLayerIdx = _numHiddenLayers + 1;
        var thetaForecast = Engine.TensorMatMul(_fcWeights[forecastLayerIdx], x);
        var fcBiasCol = Engine.Reshape(_fcBiases[forecastLayerIdx], [_thetaSizeForecast, 1]);
        thetaForecast = Engine.TensorAdd(thetaForecast, fcBiasCol);

        // Basis expansion: backcast = B_backcast @ theta_backcast
        var backcast = Engine.TensorMatMul(_basisBackcast, thetaBackcast); // [lookbackWindow, 1]
        // Basis expansion: forecast = B_forecast @ theta_forecast
        var forecast = Engine.TensorMatMul(_basisForecast, thetaForecast); // [forecastHorizon, 1]

        // Concatenate backcast and forecast into output: flatten to 1D
        var backcastFlat = Engine.Reshape(backcast, [_lookbackWindow]);
        var forecastFlat = Engine.Reshape(forecast, [_forecastHorizon]);

        // Engine.TensorConcatenate along axis 0 is a 1:1 replacement for the scalar
        // copy loop: it produces the same [lookbackWindow + forecastHorizon] 1D tensor
        // by copying backcastFlat elements followed by forecastFlat elements.
        var output = Engine.TensorConcatenate([backcastFlat, forecastFlat], axis: 0);

        return output;
    }

    /// <summary>
    /// Tape-tracked forward pass that returns separate backcast and forecast tensors.
    /// Used by the NBEATSModel during training for residual block-by-block processing.
    /// </summary>
    public (Tensor<T> backcast, Tensor<T> forecast) ForwardTape(Tensor<T> input)
    {
        // Use Engine.Reshape (tape-tracked) instead of tensor.Reshape (not tracked)
        var x = Engine.Reshape(input, [_lookbackWindow, 1]);

        // Pass through hidden layers with ReLU
        for (int layer = 0; layer < _numHiddenLayers; layer++)
        {
            var linear = Engine.TensorMatMul(_fcWeights[layer], x);
            var biasCol = Engine.Reshape(_fcBiases[layer], [_hiddenLayerSize, 1]);
            linear = Engine.TensorAdd(linear, biasCol);
            x = Engine.ReLU(linear);
        }

        // Compute theta for backcast
        int backcastLayerIdx = _numHiddenLayers;
        var thetaBackcast = Engine.TensorMatMul(_fcWeights[backcastLayerIdx], x);
        var bcBiasCol = Engine.Reshape(_fcBiases[backcastLayerIdx], [_thetaSizeBackcast, 1]);
        thetaBackcast = Engine.TensorAdd(thetaBackcast, bcBiasCol);

        // Compute theta for forecast
        int forecastLayerIdx = _numHiddenLayers + 1;
        var thetaForecast = Engine.TensorMatMul(_fcWeights[forecastLayerIdx], x);
        var fcBiasCol = Engine.Reshape(_fcBiases[forecastLayerIdx], [_thetaSizeForecast, 1]);
        thetaForecast = Engine.TensorAdd(thetaForecast, fcBiasCol);

        // Basis expansion — use Engine.Reshape for tape-tracked reshape
        var backcastRaw = Engine.TensorMatMul(_basisBackcast, thetaBackcast);
        var backcast = Engine.Reshape(backcastRaw, [_lookbackWindow]);
        var forecastRaw = Engine.TensorMatMul(_basisForecast, thetaForecast);
        var forecast = Engine.Reshape(forecastRaw, [_forecastHorizon]);

        return (backcast, forecast);
    }

    public override bool SupportsTraining => true;

    public override void ResetState() { /* stateless layer -- no recurrent state to reset */ }

    /// <summary>
    /// No-op: tape-based training handles parameter updates through the optimizer.
    /// </summary>
    public override void UpdateParameters(T learningRate) { }

    /// <summary>
    /// Non-tape forward pass for inference (used by PredictSingle).
    /// Uses plain matrix/vector operations without tape overhead.
    /// </summary>
    public (Vector<T> backcast, Vector<T> forecast) ForwardInternal(Vector<T> input)
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
            // Linear transformation: y = Wx + b using tensor operations
            var xCol = new Tensor<T>(new[] { x.Length, 1 }, x);
            var wxResult = Engine.TensorMatMul(_fcWeights[layer], xCol);
            Vector<T> linear = new Vector<T>(_hiddenLayerSize);
            var biasVec = _fcBiases[layer].ToVector();
            for (int i = 0; i < _hiddenLayerSize; i++)
            {
                linear[i] = NumOps.Add(biasVec[i], wxResult[i, 0]);
            }

            // ReLU activation
            x = new Vector<T>(linear.Length);
            for (int i = 0; i < linear.Length; i++)
            {
                x[i] = NumOps.GreaterThan(linear[i], NumOps.Zero) ? linear[i] : NumOps.Zero;
            }
        }

        // Compute theta for backcast
        int backcastLayerIdx = _numHiddenLayers;
        var xColTheta = new Tensor<T>(new[] { x.Length, 1 }, x);
        var bcWx = Engine.TensorMatMul(_fcWeights[backcastLayerIdx], xColTheta);
        var bcBiasVec = _fcBiases[backcastLayerIdx].ToVector();
        Vector<T> thetaBackcast = new Vector<T>(_thetaSizeBackcast);
        for (int i = 0; i < _thetaSizeBackcast; i++)
        {
            thetaBackcast[i] = NumOps.Add(bcBiasVec[i], bcWx[i, 0]);
        }

        // Compute theta for forecast
        int forecastLayerIdx = _numHiddenLayers + 1;
        var fcWx = Engine.TensorMatMul(_fcWeights[forecastLayerIdx], xColTheta);
        var fcBiasVec = _fcBiases[forecastLayerIdx].ToVector();
        Vector<T> thetaForecast = new Vector<T>(_thetaSizeForecast);
        for (int i = 0; i < _thetaSizeForecast; i++)
        {
            thetaForecast[i] = NumOps.Add(fcBiasVec[i], fcWx[i, 0]);
        }

        // Apply basis expansion
        Vector<T> backcast = ApplyBasisExpansion(thetaBackcast, _lookbackWindow);
        Vector<T> forecast = ApplyBasisExpansion(thetaForecast, _forecastHorizon);

        return (backcast, forecast);
    }

    /// <summary>
    /// Computes the basis matrix as a Tensor for tape-tracked operations.
    /// Shape: [outputLength, thetaSize].
    /// </summary>
    private Tensor<T> ComputeBasisTensor(int thetaSize, int outputLength)
    {
        var data = new T[outputLength * thetaSize];

        if (_useInterpretableBasis)
        {
            for (int t = 0; t < outputLength; t++)
            {
                double tNormalized = (double)t / outputLength;
                for (int p = 0; p < Math.Min(thetaSize, _polynomialDegree + 1); p++)
                {
                    data[t * thetaSize + p] = NumOps.FromDouble(Math.Pow(tNormalized, p));
                }
            }
        }
        else
        {
            // Generic basis per Oreshkin et al. (2020): when thetaSize == outputLength,
            // theta IS the output directly (identity basis). When they differ, use a
            // simple identity-like mapping (1 on the diagonal, 0 elsewhere).
            for (int t = 0; t < outputLength; t++)
            {
                for (int k = 0; k < thetaSize; k++)
                {
                    data[t * thetaSize + k] = (t == k)
                        ? NumOps.One
                        : NumOps.Zero;
                }
            }
        }

        return new Tensor<T>(new[] { outputLength, thetaSize }, new Vector<T>(data));
    }

    /// <summary>
    /// Computes the basis matrix as a Matrix (for legacy operations).
    /// Shape: [outputLength, thetaSize].
    /// </summary>
    private Matrix<T> ComputeBasisMatrix(int thetaSize, int outputLength)
    {
        var basis = new Matrix<T>(outputLength, thetaSize);

        if (_useInterpretableBasis)
        {
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
            // Generic basis: identity matrix (theta IS the output)
            for (int t = 0; t < outputLength; t++)
            {
                for (int k = 0; k < thetaSize; k++)
                {
                    basis[t, k] = (t == k) ? NumOps.One : NumOps.Zero;
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
            // Generic basis: identity — theta[t] maps directly to output[t]
            for (int t = 0; t < outputLength; t++)
            {
                output[t] = (t < theta.Length) ? theta[t] : NumOps.Zero;
            }
        }

        return output;
    }

    /// <summary>
    /// Gets all parameters (weights and biases) as a single vector.
    /// </summary>
    public override Vector<T> GetParameters()
    {
        var parameters = new List<T>();

        foreach (var weight in _fcWeights)
        {
            var vec = weight.ToVector();
            parameters.AddRange(vec);
        }

        foreach (var bias in _fcBiases)
        {
            var vec = bias.ToVector();
            parameters.AddRange(vec);
        }

        return new Vector<T>(parameters.ToArray());
    }

    /// <summary>
    /// Sets all parameters (weights and biases) from a single vector.
    /// </summary>
    public override void SetParameters(Vector<T> parameters)
    {
        if (parameters.Length != ParameterCount)
        {
            throw new ArgumentException(
                $"Expected {ParameterCount} parameters, but got {parameters.Length}.",
                nameof(parameters));
        }

        int idx = 0;

        for (int w = 0; w < _fcWeights.Count; w++)
        {
            var weight = _fcWeights[w];
            int len = weight.Length;
            var data = new T[len];
            for (int i = 0; i < len; i++)
            {
                data[i] = parameters[idx++];
            }
            int rows = weight.Shape[0];
            int cols = weight.Shape[1];
            _fcWeights[w] = new Tensor<T>(new[] { rows, cols }, new Vector<T>(data));
        }

        for (int b = 0; b < _fcBiases.Count; b++)
        {
            var bias = _fcBiases[b];
            int len = bias.Length;
            var data = new T[len];
            for (int i = 0; i < len; i++)
            {
                data[i] = parameters[idx++];
            }
            _fcBiases[b] = new Tensor<T>(new[] { len }, new Vector<T>(data));
        }

        // Re-register trainable parameters after replacing tensors
        ReRegisterParameters();
    }

    /// <summary>
    /// Re-registers all weight and bias tensors as trainable parameters.
    /// Called after SetParameters replaces tensor instances.
    /// </summary>
    private void ReRegisterParameters()
    {
        // Clear and re-register (RegisterTrainableParameter handles dedup)
        foreach (var w in _fcWeights)
            RegisterTrainableParameter(w, PersistentTensorRole.Weights);
        foreach (var b in _fcBiases)
            RegisterTrainableParameter(b, PersistentTensorRole.Biases);
    }

    /// <summary>
    /// Exports the block as computation graph nodes for JIT compilation.
    /// </summary>
    public (ComputationNode<T> backcast, ComputationNode<T> forecast) ExportComputationGraph(ComputationNode<T> inputNode)
    {
        var numOps = MathHelper.GetNumericOperations<T>();
        var x = inputNode;

        for (int layer = 0; layer < _numHiddenLayers; layer++)
        {
            var weightTensor = _fcWeights[layer];
            var weightNode = TensorOperations<T>.Constant(weightTensor, $"block_fc{layer}_weight");

            var biasTensor = _fcBiases[layer];
            var biasNode = TensorOperations<T>.Constant(biasTensor, $"block_fc{layer}_bias");

            var linear = TensorOperations<T>.MatrixVectorMultiply(weightNode, x);
            linear = TensorOperations<T>.Add(linear, biasNode);

            x = TensorOperations<T>.ReLU(linear);
        }

        // Compute theta for backcast
        var backcastWeightNode = TensorOperations<T>.Constant(_fcWeights[_numHiddenLayers], "block_backcast_weight");
        var backcastBiasNode = TensorOperations<T>.Constant(_fcBiases[_numHiddenLayers], "block_backcast_bias");

        var thetaBackcast = TensorOperations<T>.MatrixVectorMultiply(backcastWeightNode, x);
        thetaBackcast = TensorOperations<T>.Add(thetaBackcast, backcastBiasNode);

        // Compute theta for forecast
        var forecastWeightNode = TensorOperations<T>.Constant(_fcWeights[_numHiddenLayers + 1], "block_forecast_weight");
        var forecastBiasNode = TensorOperations<T>.Constant(_fcBiases[_numHiddenLayers + 1], "block_forecast_bias");

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

            return TensorOperations<T>.MatrixVectorMultiply(basisNode, theta);
        }
        else
        {
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

            return TensorOperations<T>.MatrixVectorMultiply(basisNode, theta);
        }
    }
}
