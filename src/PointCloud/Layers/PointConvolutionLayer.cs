using AiDotNet.ActivationFunctions;
using AiDotNet.Autodiff;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks.Layers;

namespace AiDotNet.PointCloud.Layers;

/// <summary>
/// Implements a convolution layer specifically designed for point cloud data.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <b>For Beginners:</b> Unlike regular convolutions for images, point cloud convolutions work on unordered 3D points.
///
/// Key differences from image convolutions:
/// - Images have regular grid structure (pixels in rows/columns)
/// - Point clouds are unordered sets of 3D coordinates
/// - Must be invariant to point order (permutation invariant)
/// - Must handle varying number of points
///
/// This layer learns features from point neighborhoods by:
/// - Finding nearby points (local neighborhood)
/// - Aggregating features from these neighbors
/// - Learning weights that work regardless of point order
///
/// Applications:
/// - Feature extraction from local 3D geometry
/// - Learning shape patterns in point clouds
/// - Building blocks for PointNet-style architectures
/// </remarks>
public class PointConvolutionLayer<T> : LayerBase<T>
{
    private readonly int _inputChannels;
    private readonly int _outputChannels;
    private readonly Matrix<T> _weights;
    private readonly Vector<T> _biases;
    private readonly Matrix<T> _weightGradients;
    private readonly Vector<T> _biasGradients;
    private Tensor<T>? _lastInput;
    private Tensor<T>? _lastPreActivation;

    /// <summary>
    /// Initializes a new instance of the PointConvolutionLayer class.
    /// </summary>
    /// <param name="inputChannels">Number of input feature channels.</param>
    /// <param name="outputChannels">Number of output feature channels.</param>
    /// <param name="activation">Optional activation function to apply.</param>
    /// <remarks>
    /// <b>For Beginners:</b> Creates a layer that transforms point features from one dimension to another.
    ///
    /// Parameters:
    /// - inputChannels: How many features each point currently has (e.g., 3 for XYZ, 6 for XYZ+RGB)
    /// - outputChannels: How many features each point should have after this layer
    ///
    /// Example:
    /// - Input: 1024 points with 3 features each (XYZ coordinates)
    /// - Layer: PointConvolutionLayer(3, 64)
    /// - Output: 1024 points with 64 learned features each
    ///
    /// The layer learns which combinations of input features are important.
    /// </remarks>
    public PointConvolutionLayer(int inputChannels, int outputChannels, IActivationFunction<T>? activation = null)
        : base([0, inputChannels], [0, outputChannels], activation ?? new IdentityActivation<T>())
    {
        _inputChannels = inputChannels;
        _outputChannels = outputChannels;

        // Initialize weights using He initialization
        _weights = InitializeWeights(inputChannels, outputChannels);
        _biases = new Vector<T>(outputChannels);
        _weightGradients = new Matrix<T>(inputChannels, outputChannels);
        _biasGradients = new Vector<T>(outputChannels);

        // Flatten parameters for base class
        int totalParams = inputChannels * outputChannels + outputChannels;
        Parameters = new Vector<T>(totalParams);
        UpdateParametersFromWeightsAndBiases();
    }

    /// <summary>
    /// Initializes weights using He initialization for better convergence.
    /// </summary>
    private Matrix<T> InitializeWeights(int inputDim, int outputDim)
    {
        var weights = new Matrix<T>(inputDim, outputDim);
        var numOps = NumOps;
        var random = Random;

        // He initialization: weights ~ N(0, sqrt(2/inputDim))
        var stddev = Math.Sqrt(2.0 / inputDim);

        for (int i = 0; i < inputDim; i++)
        {
            for (int j = 0; j < outputDim; j++)
            {
                var value = random.NextGaussian(0, stddev);
                weights[i, j] = numOps.FromDouble(value);
            }
        }

        return weights;
    }

    private void UpdateParametersFromWeightsAndBiases()
    {
        int idx = 0;
        for (int i = 0; i < _inputChannels; i++)
        {
            for (int j = 0; j < _outputChannels; j++)
            {
                Parameters[idx++] = _weights[i, j];
            }
        }
        for (int i = 0; i < _outputChannels; i++)
        {
            Parameters[idx++] = _biases[i];
        }
    }

    private void UpdateWeightsAndBiasesFromParameters()
    {
        int idx = 0;
        for (int i = 0; i < _inputChannels; i++)
        {
            for (int j = 0; j < _outputChannels; j++)
            {
                _weights[i, j] = Parameters[idx++];
            }
        }
        for (int i = 0; i < _outputChannels; i++)
        {
            _biases[i] = Parameters[idx++];
        }
    }

    public override Tensor<T> Forward(Tensor<T> input)
    {
        _lastInput = input;
        int numPoints = input.Shape[0];
        var output = new T[numPoints * _outputChannels];
        var numOps = NumOps;

        // Matrix multiplication: [N, inputChannels] x [inputChannels, outputChannels] = [N, outputChannels]
        for (int n = 0; n < numPoints; n++)
        {
            for (int outC = 0; outC < _outputChannels; outC++)
            {
                T sum = _biases[outC];
                for (int inC = 0; inC < _inputChannels; inC++)
                {
                    var inputVal = input.Data[n * _inputChannels + inC];
                    var weightVal = _weights[inC, outC];
                    sum = numOps.Add(sum, numOps.Multiply(inputVal, weightVal));
                }
                output[n * _outputChannels + outC] = sum;
            }
        }

        var preActivation = new Tensor<T>(output, [numPoints, _outputChannels]);
        _lastPreActivation = preActivation;

        // Apply activation if specified
        if (ScalarActivation != null)
        {
            var activated = new T[output.Length];
            for (int i = 0; i < output.Length; i++)
            {
                activated[i] = ScalarActivation.Activate(output[i]);
            }

            return new Tensor<T>(activated, [numPoints, _outputChannels]);
        }

        return preActivation;
    }

    public override Tensor<T> Backward(Tensor<T> outputGradient)
    {
        if (_lastInput == null)
        {
            throw new InvalidOperationException("Forward pass must be called before backward pass.");
        }

        int numPoints = _lastInput.Shape[0];
        var inputGradient = new T[numPoints * _inputChannels];
        var numOps = NumOps;

        // Apply activation derivative if needed
        var gradient = outputGradient.Data;
        if (ScalarActivation != null)
        {
            if (_lastPreActivation == null)
            {
                throw new InvalidOperationException("Forward pass must be called before backward pass.");
            }

            var activatedGradient = new Vector<T>(gradient.Length);
            for (int i = 0; i < gradient.Length; i++)
            {
                var preActivation = _lastPreActivation.Data[i];
                activatedGradient[i] = numOps.Multiply(outputGradient.Data[i], ScalarActivation.Derivative(preActivation));
            }

            gradient = activatedGradient;
        }

        // Compute weight gradients: dL/dW = X^T * dL/dY
        for (int inC = 0; inC < _inputChannels; inC++)
        {
            for (int outC = 0; outC < _outputChannels; outC++)
            {
                T gradSum = numOps.Zero;
                for (int n = 0; n < numPoints; n++)
                {
                    var inputVal = _lastInput.Data[n * _inputChannels + inC];
                    var outGrad = gradient[n * _outputChannels + outC];
                    gradSum = numOps.Add(gradSum, numOps.Multiply(inputVal, outGrad));
                }
                _weightGradients[inC, outC] = numOps.Add(_weightGradients[inC, outC], gradSum);
            }
        }

        // Compute bias gradients
        for (int outC = 0; outC < _outputChannels; outC++)
        {
            T gradSum = numOps.Zero;
            for (int n = 0; n < numPoints; n++)
            {
                gradSum = numOps.Add(gradSum, gradient[n * _outputChannels + outC]);
            }
            _biasGradients[outC] = numOps.Add(_biasGradients[outC], gradSum);
        }

        // Compute input gradient: dL/dX = dL/dY * W^T
        for (int n = 0; n < numPoints; n++)
        {
            for (int inC = 0; inC < _inputChannels; inC++)
            {
                T sum = numOps.Zero;
                for (int outC = 0; outC < _outputChannels; outC++)
                {
                    var outGrad = gradient[n * _outputChannels + outC];
                    var weightVal = _weights[inC, outC];
                    sum = numOps.Add(sum, numOps.Multiply(outGrad, weightVal));
                }
                inputGradient[n * _inputChannels + inC] = sum;
            }
        }

        return new Tensor<T>(inputGradient, [numPoints, _inputChannels]);
    }

    public override void UpdateParameters(T learningRate)
    {
        var numOps = NumOps;

        // Update weights
        for (int i = 0; i < _inputChannels; i++)
        {
            for (int j = 0; j < _outputChannels; j++)
            {
                var update = numOps.Multiply(learningRate, _weightGradients[i, j]);
                _weights[i, j] = numOps.Subtract(_weights[i, j], update);
            }
        }

        // Update biases
        for (int i = 0; i < _outputChannels; i++)
        {
            var update = numOps.Multiply(learningRate, _biasGradients[i]);
            _biases[i] = numOps.Subtract(_biases[i], update);
        }

        UpdateParametersFromWeightsAndBiases();
    }

    public override void ClearGradients()
    {
        var numOps = NumOps;
        for (int i = 0; i < _inputChannels; i++)
        {
            for (int j = 0; j < _outputChannels; j++)
            {
                _weightGradients[i, j] = numOps.Zero;
            }
        }
        for (int i = 0; i < _outputChannels; i++)
        {
            _biasGradients[i] = numOps.Zero;
        }
    }

    public override Vector<T> GetParameters()
    {
        UpdateParametersFromWeightsAndBiases();
        return Parameters.Clone();
    }

    public override void UpdateParameters(Vector<T> parameters)
    {
        if (parameters.Length != ParameterCount)
        {
            throw new ArgumentException("Parameter vector length does not match layer parameter count.", nameof(parameters));
        }

        Parameters = parameters;
        UpdateWeightsAndBiasesFromParameters();
    }

    public override void ResetState()
    {
        _lastInput = null;
        _lastPreActivation = null;
        ClearGradients();
    }

    public override bool SupportsJitCompilation => false;

    public override ComputationNode<T> ExportComputationGraph(List<ComputationNode<T>> inputNodes)
    {
        throw new NotSupportedException(
            "PointConvolutionLayer does not support computation graph export due to point cloud-specific operations.");
    }

    public override int ParameterCount => _inputChannels * _outputChannels + _outputChannels;

    public override bool SupportsTraining => true;
}

/// <summary>
/// Extension methods for Random class to support Gaussian distribution.
/// </summary>
internal static class RandomExtensions
{
    /// <summary>
    /// Generates a random number from a Gaussian (normal) distribution.
    /// </summary>
    public static double NextGaussian(this Random random, double mean = 0, double stddev = 1)
    {
        // Box-Muller transform
        double u1 = 1.0 - random.NextDouble();
        double u2 = 1.0 - random.NextDouble();
        double randStdNormal = Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Sin(2.0 * Math.PI * u2);
        return mean + stddev * randStdNormal;
    }
}
