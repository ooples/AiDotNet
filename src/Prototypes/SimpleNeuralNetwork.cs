using AiDotNet.Extensions;

namespace AiDotNet.Prototypes;

/// <summary>
/// Simple 2-layer neural network for prototype validation.
/// Demonstrates GPU acceleration through vectorized operations.
/// </summary>
/// <typeparam name="T">The numeric type for weights and calculations.</typeparam>
/// <remarks>
/// This is a minimal neural network for Phase A prototype validation.
/// It includes:
/// - Input layer → Hidden layer (with ReLU activation)
/// - Hidden layer → Output layer (linear)
/// - Backpropagation
/// - Adam optimizer integration
/// </remarks>
public class SimpleNeuralNetwork<T>
{
    private readonly int _inputSize;
    private readonly int _hiddenSize;
    private readonly int _outputSize;

    private PrototypeVector<T> _weightsInputHidden = null!;
    private PrototypeVector<T> _biasHidden = null!;
    private PrototypeVector<T> _weightsHiddenOutput = null!;
    private PrototypeVector<T> _biasOutput = null!;

    private PrototypeVector<T>? _lastInput;
    private PrototypeVector<T>? _lastHidden;
    private PrototypeVector<T>? _lastOutput;

    private readonly INumericOperations<T> _numOps;
    private readonly Random _random;

    /// <summary>
    /// Gets the total number of parameters in the network.
    /// </summary>
    public int ParameterCount => _inputSize * _hiddenSize + _hiddenSize + _hiddenSize * _outputSize + _outputSize;

    /// <summary>
    /// Initializes a new instance of the SimpleNeuralNetwork.
    /// </summary>
    /// <param name="inputSize">Number of input features.</param>
    /// <param name="hiddenSize">Number of hidden units.</param>
    /// <param name="outputSize">Number of output units.</param>
    /// <param name="seed">Random seed for weight initialization.</param>
    public SimpleNeuralNetwork(int inputSize, int hiddenSize, int outputSize, int? seed = null)
    {
        if (inputSize <= 0) throw new ArgumentException("Input size must be positive", nameof(inputSize));
        if (hiddenSize <= 0) throw new ArgumentException("Hidden size must be positive", nameof(hiddenSize));
        if (outputSize <= 0) throw new ArgumentException("Output size must be positive", nameof(outputSize));

        _inputSize = inputSize;
        _hiddenSize = hiddenSize;
        _outputSize = outputSize;
        _numOps = MathHelper.GetNumericOperations<T>();
        _random = seed.HasValue ? RandomHelper.CreateSeededRandom(seed.Value) : RandomHelper.CreateSecureRandom();

        InitializeWeights();
    }

    /// <summary>
    /// Initializes weights using Xavier/Glorot initialization.
    /// </summary>
    private void InitializeWeights()
    {
        // Xavier initialization: scale = sqrt(2 / (fan_in + fan_out))
        var scaleInputHidden = Math.Sqrt(2.0 / (_inputSize + _hiddenSize));
        var scaleHiddenOutput = Math.Sqrt(2.0 / (_hiddenSize + _outputSize));

        _weightsInputHidden = CreateRandomVector(_inputSize * _hiddenSize, scaleInputHidden);
        _biasHidden = PrototypeVector<T>.Zeros(_hiddenSize);

        _weightsHiddenOutput = CreateRandomVector(_hiddenSize * _outputSize, scaleHiddenOutput);
        _biasOutput = PrototypeVector<T>.Zeros(_outputSize);
    }

    /// <summary>
    /// Creates a random vector with values from normal distribution.
    /// </summary>
    private PrototypeVector<T> CreateRandomVector(int length, double scale)
    {
        var data = new T[length];
        for (int i = 0; i < length; i++)
        {
            data[i] = _numOps.FromDouble(_random.NextGaussian() * scale);
        }
        return new PrototypeVector<T>(data);
    }

    /// <summary>
    /// Forward pass through the network.
    /// </summary>
    /// <param name="input">Input vector.</param>
    /// <returns>Output vector.</returns>
    public PrototypeVector<T> Forward(PrototypeVector<T> input)
    {
        if (input.Length != _inputSize)
        {
            throw new ArgumentException($"Expected input size {_inputSize}, got {input.Length}");
        }

        _lastInput = input;

        // Input → Hidden: z_h = W_ih @ input + b_h
        var hiddenPreActivation = MatrixVectorMultiply(_weightsInputHidden, input, _hiddenSize, _inputSize);
        hiddenPreActivation = hiddenPreActivation.Add(_biasHidden);

        // ReLU activation: hidden = max(0, z_h)
        _lastHidden = ApplyReLU(hiddenPreActivation);

        // Hidden → Output: output = W_ho @ hidden + b_o
        var output = MatrixVectorMultiply(_weightsHiddenOutput, _lastHidden, _outputSize, _hiddenSize);
        output = output.Add(_biasOutput);

        _lastOutput = output;
        return output;
    }

    /// <summary>
    /// Backward pass (computes gradients).
    /// </summary>
    /// <param name="outputGradient">Gradient from loss function.</param>
    /// <returns>Tuple of (weightsInputHidden gradient, biasHidden gradient, weightsHiddenOutput gradient, biasOutput gradient).</returns>
    public (PrototypeVector<T>, PrototypeVector<T>, PrototypeVector<T>, PrototypeVector<T>) Backward(PrototypeVector<T> outputGradient)
    {
        if (_lastInput == null || _lastHidden == null || _lastOutput == null)
        {
            throw new InvalidOperationException("Must call Forward before Backward");
        }

        // Output layer gradients
        var biasOutputGrad = outputGradient;

        // Gradient w.r.t. weights hidden→output: dW_ho = gradient @ hidden^T
        var weightsHiddenOutputGrad = OuterProduct(outputGradient, _lastHidden);

        // Gradient w.r.t. hidden layer: dHidden = W_ho^T @ gradient
        var hiddenGrad = MatrixVectorMultiplyTranspose(_weightsHiddenOutput, outputGradient, _outputSize, _hiddenSize);

        // ReLU derivative: gradient = gradient * (hidden > 0)
        hiddenGrad = ApplyReLUDerivative(hiddenGrad, _lastHidden);

        // Hidden layer gradients
        var biasHiddenGrad = hiddenGrad;

        // Gradient w.r.t. weights input→hidden: dW_ih = gradient @ input^T
        var weightsInputHiddenGrad = OuterProduct(hiddenGrad, _lastInput);

        return (weightsInputHiddenGrad, biasHiddenGrad, weightsHiddenOutputGrad, biasOutputGrad);
    }

    /// <summary>
    /// Gets all parameters as a single flattened vector.
    /// </summary>
    public PrototypeVector<T> GetParameters()
    {
        var parameters = new T[ParameterCount];
        int idx = 0;

        // Flatten all weights and biases
        for (int i = 0; i < _weightsInputHidden.Length; i++) parameters[idx++] = _weightsInputHidden[i];
        for (int i = 0; i < _biasHidden.Length; i++) parameters[idx++] = _biasHidden[i];
        for (int i = 0; i < _weightsHiddenOutput.Length; i++) parameters[idx++] = _weightsHiddenOutput[i];
        for (int i = 0; i < _biasOutput.Length; i++) parameters[idx++] = _biasOutput[i];

        return new PrototypeVector<T>(parameters);
    }

    /// <summary>
    /// Sets all parameters from a single flattened vector.
    /// </summary>
    public void SetParameters(PrototypeVector<T> parameters)
    {
        if (parameters.Length != ParameterCount)
        {
            throw new ArgumentException($"Expected {ParameterCount} parameters, got {parameters.Length}");
        }

        int idx = 0;

        for (int i = 0; i < _weightsInputHidden.Length; i++) _weightsInputHidden[i] = parameters[idx++];
        for (int i = 0; i < _biasHidden.Length; i++) _biasHidden[i] = parameters[idx++];
        for (int i = 0; i < _weightsHiddenOutput.Length; i++) _weightsHiddenOutput[i] = parameters[idx++];
        for (int i = 0; i < _biasOutput.Length; i++) _biasOutput[i] = parameters[idx++];
    }

    #region Helper Methods

    /// <summary>
    /// Matrix-vector multiplication: result = matrix @ vector
    /// </summary>
    private PrototypeVector<T> MatrixVectorMultiply(PrototypeVector<T> matrix, PrototypeVector<T> vector, int rows, int cols)
    {
        var result = new T[rows];
        for (int i = 0; i < rows; i++)
        {
            var sum = _numOps.Zero;
            for (int j = 0; j < cols; j++)
            {
                sum = _numOps.Add(sum, _numOps.Multiply(matrix[i * cols + j], vector[j]));
            }
            result[i] = sum;
        }
        return new PrototypeVector<T>(result);
    }

    /// <summary>
    /// Matrix-vector multiplication with transposed matrix: result = matrix^T @ vector
    /// Matrix is (rows, cols), transpose is (cols, rows), result is (cols)
    /// </summary>
    private PrototypeVector<T> MatrixVectorMultiplyTranspose(PrototypeVector<T> matrix, PrototypeVector<T> vector, int rows, int cols)
    {
        var result = new T[cols];
        for (int i = 0; i < cols; i++)
        {
            var sum = _numOps.Zero;
            for (int j = 0; j < rows; j++)
            {
                sum = _numOps.Add(sum, _numOps.Multiply(matrix[j * cols + i], vector[j]));
            }
            result[i] = sum;
        }
        return new PrototypeVector<T>(result);
    }

    /// <summary>
    /// Outer product: result = a @ b^T (flattened)
    /// </summary>
    private PrototypeVector<T> OuterProduct(PrototypeVector<T> a, PrototypeVector<T> b)
    {
        var result = new T[a.Length * b.Length];
        int idx = 0;
        for (int i = 0; i < a.Length; i++)
        {
            for (int j = 0; j < b.Length; j++)
            {
                result[idx++] = _numOps.Multiply(a[i], b[j]);
            }
        }
        return new PrototypeVector<T>(result);
    }

    /// <summary>
    /// Applies ReLU activation: max(0, x)
    /// </summary>
    private PrototypeVector<T> ApplyReLU(PrototypeVector<T> input)
    {
        var result = new T[input.Length];
        for (int i = 0; i < input.Length; i++)
        {
            result[i] = _numOps.ToDouble(input[i]) > 0 ? input[i] : _numOps.Zero;
        }
        return new PrototypeVector<T>(result);
    }

    /// <summary>
    /// Applies ReLU derivative: gradient * (x > 0)
    /// </summary>
    private PrototypeVector<T> ApplyReLUDerivative(PrototypeVector<T> gradient, PrototypeVector<T> input)
    {
        var result = new T[gradient.Length];
        for (int i = 0; i < gradient.Length; i++)
        {
            result[i] = _numOps.ToDouble(input[i]) > 0 ? gradient[i] : _numOps.Zero;
        }
        return new PrototypeVector<T>(result);
    }

    #endregion

    /// <summary>
    /// Computes mean squared error loss.
    /// </summary>
    public T ComputeLoss(PrototypeVector<T> predicted, PrototypeVector<T> target)
    {
        var diff = predicted.Subtract(target);
        var squared = diff.Multiply(diff);

        var sum = _numOps.Zero;
        for (int i = 0; i < squared.Length; i++)
        {
            sum = _numOps.Add(sum, squared[i]);
        }

        return _numOps.Divide(sum, _numOps.FromDouble(squared.Length));
    }

    /// <summary>
    /// Computes gradient of MSE loss.
    /// </summary>
    public PrototypeVector<T> ComputeLossGradient(PrototypeVector<T> predicted, PrototypeVector<T> target)
    {
        var diff = predicted.Subtract(target);
        var scale = _numOps.FromDouble(2.0 / predicted.Length);
        return diff.Multiply(scale);
    }
}
