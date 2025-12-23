using AiDotNet.Autodiff;
using AiDotNet.ActivationFunctions;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks.Layers;

namespace AiDotNet.PointCloud.Layers;

/// <summary>
/// Implements a Transformation Network (T-Net) for learning spatial transformations of point clouds.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <b>For Beginners:</b> T-Net learns to align and normalize point clouds before processing.
///
/// What T-Net does:
/// - Learns a transformation matrix to apply to input points
/// - Aligns point clouds to a canonical orientation
/// - Makes the network more robust to rotations and translations
/// - Helps the network focus on shape rather than orientation
///
/// How it works:
/// 1. Takes point cloud as input
/// 2. Processes it through small neural network
/// 3. Outputs a transformation matrix (e.g., 3x3 for spatial, KxK for feature)
/// 4. Applies this matrix to transform the input
///
/// Two types of T-Net in PointNet:
/// - Input T-Net: 3x3 matrix to align XYZ coordinates
/// - Feature T-Net: KxK matrix to align high-dimensional features
///
/// Benefits:
/// - Achieves invariance to rigid transformations
/// - Normalizes point cloud orientation
/// - Improves classification and segmentation accuracy
///
/// Example:
/// - Input: Point cloud that might be rotated randomly
/// - T-Net learns: "Rotate this cloud 45 degrees to align it"
/// - Output: Aligned point cloud in standard orientation
/// </remarks>
public class TNetLayer<T> : LayerBase<T>
{
    private readonly int _transformDim; // Dimension of transformation (e.g., 3 for XYZ, 64 for features)
    private readonly int _numFeatures;
    private readonly List<ILayer<T>> _mlpLayers;
    private readonly List<ILayer<T>> _fcLayers;
    private readonly MaxPoolingLayer<T> _maxPooling;
    private Tensor<T>? _lastInput;
    private Matrix<T>? _transformMatrix;
    private Tensor<T>? _lastTransformVector;

    /// <summary>
    /// Initializes a new instance of the TNetLayer class.
    /// </summary>
    /// <param name="transformDim">Dimension of the transformation matrix (3 for spatial, higher for features).</param>
    /// <param name="numFeatures">Number of feature channels in the input.</param>
    /// <param name="mlpChannels">Per-point MLP channels used before global pooling.</param>
    /// <param name="fcChannels">Fully connected channels used to predict the transformation.</param>
    /// <remarks>
    /// <b>For Beginners:</b> Creates a T-Net that learns transformations for point clouds.
    ///
    /// Parameters:
    /// - transformDim: Size of transformation matrix
    ///   - Use 3 for spatial transformation of XYZ coordinates (learns 3x3 matrix)
    ///   - Use 64, 128, etc. for feature transformation (learns KxK matrix)
    /// - numFeatures: How many features each point has
    ///
    /// The T-Net internally uses:
    /// - Convolution layers to process point features
    /// - Max pooling to get global information
    /// - Fully connected layers to predict transformation matrix
    ///
    /// Example:
    /// - TNetLayer(3, 3): Spatial transformer for XYZ coordinates
    /// - TNetLayer(64, 64): Feature transformer for 64-dimensional features
    /// </remarks>
    public TNetLayer(int transformDim, int numFeatures, int[]? mlpChannels = null, int[]? fcChannels = null)
        : base([0, numFeatures], [0, numFeatures])
    {
        if (transformDim <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(transformDim), "Transform dimension must be positive.");
        }
        if (numFeatures <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(numFeatures), "Number of features must be positive.");
        }
        if (transformDim > numFeatures)
        {
            throw new ArgumentOutOfRangeException(nameof(transformDim), "Transform dimension must be <= number of features.");
        }

        _transformDim = transformDim;
        _numFeatures = numFeatures;
        _mlpLayers = [];
        _fcLayers = [];

        var mlp = ValidateChannelArray(mlpChannels ?? new[] { 64, 128, 1024 }, nameof(mlpChannels));
        var fc = ValidateChannelArray(fcChannels ?? new[] { 512, 256 }, nameof(fcChannels));

        int inputChannels = numFeatures;
        foreach (var outChannels in mlp)
        {
            _mlpLayers.Add(new PointConvolutionLayer<T>(inputChannels, outChannels, new ReLUActivation<T>()));
            inputChannels = outChannels;
        }

        _maxPooling = new MaxPoolingLayer<T>(inputChannels);

        int fcInput = inputChannels;
        foreach (var hidden in fc)
        {
            _fcLayers.Add(new DenseLayer<T>(fcInput, hidden, activationFunction: new ReLUActivation<T>()));
            fcInput = hidden;
        }

        int outputDim = _transformDim * _transformDim;
        _fcLayers.Add(new DenseLayer<T>(fcInput, outputDim, activationFunction: new IdentityActivation<T>()));

        Parameters = GetParameters();
    }

    public override Tensor<T> Forward(Tensor<T> input)
    {
        _lastInput = input;

        Tensor<T> features = input;
        foreach (var layer in _mlpLayers)
        {
            features = layer.Forward(features);
        }

        features = _maxPooling.Forward(features);

        Tensor<T> transformVector = features;
        foreach (var layer in _fcLayers)
        {
            transformVector = layer.Forward(transformVector);
        }

        _lastTransformVector = transformVector;
        _transformMatrix = BuildTransformMatrix(transformVector);

        // Apply transformation to input
        return ApplyTransformation(input, _transformMatrix);
    }

    private Matrix<T> BuildTransformMatrix(Tensor<T> transformVector)
    {
        int expected = _transformDim * _transformDim;
        if (transformVector.Length != expected)
        {
            throw new InvalidOperationException("Transform vector has unexpected length.");
        }

        var matrix = new Matrix<T>(_transformDim, _transformDim);
        var numOps = NumOps;
        int index = 0;
        for (int r = 0; r < _transformDim; r++)
        {
            for (int c = 0; c < _transformDim; c++)
            {
                T value = transformVector.Data[index++];
                if (r == c)
                {
                    value = numOps.Add(value, numOps.One);
                }
                matrix[r, c] = value;
            }
        }

        return matrix;
    }

    private Tensor<T> ApplyTransformation(Tensor<T> input, Matrix<T> transform)
    {
        int numPoints = input.Shape[0];
        int numFeatures = input.Shape[1];
        var output = new T[numPoints * numFeatures];
        var numOps = NumOps;

        // Apply transformation: output = input * transform^T
        for (int n = 0; n < numPoints; n++)
        {
            for (int outF = 0; outF < _transformDim; outF++)
            {
                T sum = numOps.Zero;
                for (int inF = 0; inF < _transformDim; inF++)
                {
                    var inputVal = input.Data[n * numFeatures + inF];
                    var transformVal = transform[inF, outF];
                    sum = numOps.Add(sum, numOps.Multiply(inputVal, transformVal));
                }
                output[n * numFeatures + outF] = sum;
            }

            // Copy remaining features if numFeatures > transformDim
            for (int f = _transformDim; f < numFeatures; f++)
            {
                output[n * numFeatures + f] = input.Data[n * numFeatures + f];
            }
        }

        return new Tensor<T>(output, [numPoints, numFeatures]);
    }

    public override Tensor<T> Backward(Tensor<T> outputGradient)
    {
        if (_lastInput == null || _transformMatrix == null || _lastTransformVector == null)
        {
            throw new InvalidOperationException("Forward pass must be called before backward pass.");
        }

        int numPoints = outputGradient.Shape[0];
        int numFeatures = outputGradient.Shape[1];
        var inputGradient = new T[numPoints * numFeatures];
        var transformGrad = new T[_transformDim * _transformDim];
        var numOps = NumOps;

        for (int n = 0; n < numPoints; n++)
        {
            for (int outF = 0; outF < _transformDim; outF++)
            {
                var outGrad = outputGradient.Data[n * numFeatures + outF];
                for (int inF = 0; inF < _transformDim; inF++)
                {
                    var transformVal = _transformMatrix[inF, outF];
                    var inputVal = _lastInput.Data[n * numFeatures + inF];
                    inputGradient[n * numFeatures + inF] = numOps.Add(
                        inputGradient[n * numFeatures + inF],
                        numOps.Multiply(outGrad, transformVal));
                    int tIndex = inF * _transformDim + outF;
                    transformGrad[tIndex] = numOps.Add(
                        transformGrad[tIndex],
                        numOps.Multiply(inputVal, outGrad));
                }
            }

            for (int f = _transformDim; f < numFeatures; f++)
            {
                inputGradient[n * numFeatures + f] = outputGradient.Data[n * numFeatures + f];
            }
        }

        var transformGradTensor = new Tensor<T>(transformGrad, _lastTransformVector.Shape);
        Tensor<T> layerGradient = transformGradTensor;
        for (int i = _fcLayers.Count - 1; i >= 0; i--)
        {
            layerGradient = _fcLayers[i].Backward(layerGradient);
        }

        layerGradient = _maxPooling.Backward(layerGradient);
        for (int i = _mlpLayers.Count - 1; i >= 0; i--)
        {
            layerGradient = _mlpLayers[i].Backward(layerGradient);
        }

        var layerGradData = layerGradient.Data;
        for (int i = 0; i < inputGradient.Length; i++)
        {
            inputGradient[i] = numOps.Add(inputGradient[i], layerGradData[i]);
        }

        return new Tensor<T>(inputGradient, [numPoints, numFeatures]);
    }

    public override void UpdateParameters(T learningRate)
    {
        foreach (var layer in _mlpLayers)
        {
            layer.UpdateParameters(learningRate);
        }
        foreach (var layer in _fcLayers)
        {
            layer.UpdateParameters(learningRate);
        }
    }

    public override void ClearGradients()
    {
        foreach (var layer in _mlpLayers)
        {
            layer.ClearGradients();
        }
        foreach (var layer in _fcLayers)
        {
            layer.ClearGradients();
        }
    }

    public override Vector<T> GetParameters()
    {
        int totalParams = ParameterCount;
        var parameters = new Vector<T>(totalParams);
        int offset = 0;

        foreach (var layer in _mlpLayers)
        {
            var layerParameters = layer.GetParameters();
            for (int i = 0; i < layerParameters.Length; i++)
            {
                parameters[offset + i] = layerParameters[i];
            }

            offset += layerParameters.Length;
        }

        foreach (var layer in _fcLayers)
        {
            var layerParameters = layer.GetParameters();
            for (int i = 0; i < layerParameters.Length; i++)
            {
                parameters[offset + i] = layerParameters[i];
            }

            offset += layerParameters.Length;
        }

        Parameters = parameters;
        return parameters;
    }

    public override void UpdateParameters(Vector<T> parameters)
    {
        int offset = 0;
        foreach (var layer in _mlpLayers)
        {
            int layerParameterCount = layer.ParameterCount;
            if (layerParameterCount > 0)
            {
                var layerParameters = parameters.SubVector(offset, layerParameterCount);
                layer.UpdateParameters(layerParameters);
                offset += layerParameterCount;
            }
        }

        foreach (var layer in _fcLayers)
        {
            int layerParameterCount = layer.ParameterCount;
            if (layerParameterCount > 0)
            {
                var layerParameters = parameters.SubVector(offset, layerParameterCount);
                layer.UpdateParameters(layerParameters);
                offset += layerParameterCount;
            }
        }

        Parameters = parameters;
    }

    public override void ResetState()
    {
        _lastInput = null;
        _transformMatrix = null;
        _lastTransformVector = null;

        foreach (var layer in _mlpLayers)
        {
            layer.ResetState();
        }
        foreach (var layer in _fcLayers)
        {
            layer.ResetState();
        }

        _maxPooling.ResetState();
    }

    public override bool SupportsJitCompilation => false;

    public override ComputationNode<T> ExportComputationGraph(List<ComputationNode<T>> inputNodes)
    {
        throw new NotSupportedException(
            "TNetLayer does not support computation graph export due to point cloud-specific operations.");
    }

    public override int ParameterCount
    {
        get
        {
            int total = 0;
            foreach (var layer in _mlpLayers)
            {
                total += layer.ParameterCount;
            }
            foreach (var layer in _fcLayers)
            {
                total += layer.ParameterCount;
            }
            return total;
        }
    }

    public override bool SupportsTraining => true;

    private static int[] ValidateChannelArray(int[] values, string paramName)
    {
        if (values.Length == 0)
        {
            throw new ArgumentException("Channel array must not be empty.", paramName);
        }
        for (int i = 0; i < values.Length; i++)
        {
            if (values[i] <= 0)
            {
                throw new ArgumentOutOfRangeException(paramName, "Channel sizes must be positive.");
            }
        }

        return values;
    }
}
