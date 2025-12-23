using AiDotNet.Autodiff;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks.Layers;

namespace AiDotNet.PointCloud.Layers;

/// <summary>
/// Implements global max pooling for point clouds to extract global features.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <b>For Beginners:</b> Max pooling takes the maximum value across all points for each feature channel.
///
/// How it works:
/// - Input: N points, each with C features [N, C]
/// - Operation: For each feature channel, find the maximum value across all N points
/// - Output: A single vector of C features [1, C]
///
/// Why it's useful:
/// - Creates a global representation of the entire point cloud
/// - Achieves permutation invariance (order of points doesn't matter)
/// - Reduces dimensionality from many points to one feature vector
///
/// Example:
/// - Input: 1024 points with 64 features each = [1024, 64]
/// - Max pooling across points
/// - Output: 1 global feature vector with 64 features = [1, 64]
///
/// This is a key component in PointNet for making the network invariant to point order.
/// </remarks>
public class MaxPoolingLayer<T> : LayerBase<T>
{
    private readonly int _numFeatures;
    private int[]? _maxIndices; // Store indices of max values for backward pass
    private int _numPoints;

    /// <summary>
    /// Initializes a new instance of the MaxPoolingLayer class.
    /// </summary>
    /// <param name="numFeatures">Number of feature channels to pool.</param>
    /// <remarks>
    /// <b>For Beginners:</b> Creates a max pooling layer for point cloud global feature extraction.
    ///
    /// The number of features determines the output size:
    /// - If input is [N, 64], output will be [1, 64]
    /// - If input is [N, 128], output will be [1, 128]
    ///
    /// This layer has no trainable parameters - it's a fixed operation that
    /// selects the maximum value for each feature across all points.
    /// </remarks>
    public MaxPoolingLayer(int numFeatures)
        : base([0, numFeatures], [1, numFeatures])
    {
        _numFeatures = numFeatures;
        Parameters = Vector<T>.Empty(); // No trainable parameters
    }

    public override Tensor<T> Forward(Tensor<T> input)
    {
        _numPoints = input.Shape[0];
        _maxIndices = new int[_numFeatures];
        var output = new T[_numFeatures];
        var numOps = NumOps;

        // For each feature channel, find the maximum across all points
        for (int c = 0; c < _numFeatures; c++)
        {
            T maxVal = input.Data[c]; // Start with first point's value
            int maxIdx = 0;

            for (int n = 1; n < _numPoints; n++)
            {
                T currentVal = input.Data[n * _numFeatures + c];
                if (numOps.GreaterThan(currentVal, maxVal))
                {
                    maxVal = currentVal;
                    maxIdx = n;
                }
            }

            output[c] = maxVal;
            _maxIndices[c] = maxIdx;
        }

        return new Tensor<T>(output, [1, _numFeatures]);
    }

    public override Tensor<T> Backward(Tensor<T> outputGradient)
    {
        if (_maxIndices == null)
        {
            throw new InvalidOperationException("Forward pass must be called before backward pass.");
        }

        // Gradient flows only to the points that had maximum values
        var inputGradient = new T[_numPoints * _numFeatures];
        var numOps = NumOps;

        // Initialize all gradients to zero
        for (int i = 0; i < inputGradient.Length; i++)
        {
            inputGradient[i] = numOps.Zero;
        }

        // Assign gradient only to the max points
        for (int c = 0; c < _numFeatures; c++)
        {
            int maxIdx = _maxIndices[c];
            inputGradient[maxIdx * _numFeatures + c] = outputGradient.Data[c];
        }

        return new Tensor<T>(inputGradient, [_numPoints, _numFeatures]);
    }

    public override void UpdateParameters(T learningRate)
    {
        // No parameters to update
    }

    public override void ClearGradients()
    {
        // No gradients to clear
    }

    public override Vector<T> GetParameters()
    {
        return Vector<T>.Empty();
    }

    public override void ResetState()
    {
        _maxIndices = null;
        _numPoints = 0;
    }

    public override bool SupportsJitCompilation => false;

    public override ComputationNode<T> ExportComputationGraph(List<ComputationNode<T>> inputNodes)
    {
        throw new NotSupportedException(
            "MaxPoolingLayer does not support computation graph export due to point cloud-specific pooling.");
    }

    public override int ParameterCount => 0;

    public override bool SupportsTraining => false; // No parameters to update; still participates in backprop
}
