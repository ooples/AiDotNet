using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.PointCloud.Interfaces;
using AiDotNet.PointCloud.Layers;

namespace AiDotNet.PointCloud.Models;

/// <summary>
/// Implements the PointNet architecture for processing point cloud data.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> PointNet is a pioneering deep learning architecture designed to directly process point clouds.
/// </para>
/// <para>
/// Key innovations of PointNet:
/// - Directly processes unordered point sets (no need to convert to voxels or images)
/// - Permutation invariant: output doesn't change if you shuffle the input points
/// - Learns both local and global features
/// - Uses spatial transformer networks (T-Net) for alignment
/// </para>
/// <para>
/// Architecture overview:
/// 1. Input transformation: T-Net learns to align input points
/// 2. Multi-layer perceptron (MLP): Processes each point independently
/// 3. Feature transformation: Another T-Net aligns learned features
/// 4. More MLPs: Further feature extraction
/// 5. Max pooling: Aggregates information from all points
/// 6. Global feature vector: Represents the entire point cloud
/// 7. Classification/Segmentation: Task-specific layers
/// </para>
/// <para>
/// Why it's important:
/// - First successful deep learning approach for raw point clouds
/// - Achieves state-of-the-art results on ModelNet40 classification
/// - Foundation for many subsequent point cloud methods
/// - Widely used in robotics, autonomous driving, and 3D vision
/// </para>
/// <para>
/// Reference: "PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation"
/// by Qi et al., CVPR 2017
/// </para>
/// </remarks>
public class PointNet<T> : NeuralNetworkBase<T>, IPointCloudModel<T>, IPointCloudClassification<T>
{
    private readonly int _numClasses;
    private readonly bool _useInputTransform;
    private readonly bool _useFeatureTransform;
    private Vector<T>? _globalFeatures;

    /// <summary>
    /// Initializes a new instance of the PointNet class.
    /// </summary>
    /// <param name="numClasses">Number of output classes for classification.</param>
    /// <param name="useInputTransform">Whether to use input transformation network (T-Net).</param>
    /// <param name="useFeatureTransform">Whether to use feature transformation network.</param>
    /// <param name="lossFunction">Optional loss function for training.</param>
    /// <remarks>
    /// <b>For Beginners:</b> Creates a PointNet model for point cloud classification.
    ///
    /// Parameters:
    /// - numClasses: How many categories to classify (e.g., 40 for ModelNet40)
    /// - useInputTransform: Enable T-Net for input alignment (recommended: true)
    /// - useFeatureTransform: Enable T-Net for feature alignment (recommended: true)
    ///
    /// Example:
    /// - For ModelNet40 classification: PointNet(40, true, true)
    /// - For simple 10-class problem: PointNet(10, true, false)
    ///
    /// The transformations help make the network robust to rotations and deformations.
    /// </remarks>
    public PointNet(
        int numClasses,
        bool useInputTransform = true,
        bool useFeatureTransform = true,
        ILossFunction<T>? lossFunction = null)
        : base(CreateArchitecture(numClasses), lossFunction)
    {
        _numClasses = numClasses;
        _useInputTransform = useInputTransform;
        _useFeatureTransform = useFeatureTransform;

        InitializeLayers();
    }

    private static NeuralNetworkArchitecture<T> CreateArchitecture(int numClasses)
    {
        return new NeuralNetworkArchitecture<T>
        {
            InputType = InputType.ThreeDimensional,
            LayerSize = 1024, // Global feature dimension
            TaskType = TaskType.Classification,
            Layers = null // Will be initialized manually
        };
    }

    protected override void InitializeLayers()
    {
        // PointNet architecture based on the original paper

        // Input transform (3x3 transformation)
        if (_useInputTransform)
        {
            AddLayerToCollection(new TNetLayer<T>(3, 3));
        }

        // MLP (64, 64)
        AddLayerToCollection(new PointConvolutionLayer<T>(3, 64));
        AddLayerToCollection(new PointConvolutionLayer<T>(64, 64));

        // Feature transform (64x64 transformation)
        if (_useFeatureTransform)
        {
            AddLayerToCollection(new TNetLayer<T>(64, 64));
        }

        // MLP (64, 128, 1024)
        AddLayerToCollection(new PointConvolutionLayer<T>(64, 64));
        AddLayerToCollection(new PointConvolutionLayer<T>(64, 128));
        AddLayerToCollection(new PointConvolutionLayer<T>(128, 1024));

        // Max pooling to get global features
        AddLayerToCollection(new MaxPoolingLayer<T>(1024));

        // Classification head - MLP (512, 256, numClasses)
        // Note: Would need DenseLayer for fully connected layers
        // For now, using point convolution which acts similarly on [1, features]
        AddLayerToCollection(new PointConvolutionLayer<T>(1024, 512));
        AddLayerToCollection(new PointConvolutionLayer<T>(512, 256));
        AddLayerToCollection(new PointConvolutionLayer<T>(256, _numClasses));
    }

    public override Tensor<T> Forward(Tensor<T> input)
    {
        Tensor<T> x = input;

        // Process through all layers
        for (int i = 0; i < Layers.Count; i++)
        {
            _layerInputs[i] = x;
            x = Layers[i].Forward(x);
            _layerOutputs[i] = x;

            // Capture global features after max pooling
            if (Layers[i] is MaxPoolingLayer<T> && _globalFeatures == null)
            {
                // Convert tensor to vector for global features
                _globalFeatures = new Vector<T>(x.Data);
            }
        }

        return x;
    }

    public override Tensor<T> Backward(Tensor<T> outputGradient)
    {
        Tensor<T> gradient = outputGradient;

        // Backpropagate through layers in reverse order
        for (int i = Layers.Count - 1; i >= 0; i--)
        {
            gradient = Layers[i].Backward(gradient);
        }

        return gradient;
    }

    public Vector<T> ExtractGlobalFeatures(Tensor<T> pointCloud)
    {
        // Forward pass to populate global features
        _ = Forward(pointCloud);

        if (_globalFeatures == null)
        {
            throw new InvalidOperationException("Global features not extracted. Ensure forward pass completed.");
        }

        return _globalFeatures;
    }

    public Tensor<T> ExtractPointFeatures(Tensor<T> pointCloud)
    {
        Tensor<T> x = pointCloud;

        // Process through layers up to (but not including) max pooling
        for (int i = 0; i < Layers.Count; i++)
        {
            if (Layers[i] is MaxPoolingLayer<T>)
            {
                // Return features before global pooling
                return x;
            }

            x = Layers[i].Forward(x);
        }

        return x;
    }

    public Vector<T> ClassifyPointCloud(Tensor<T> pointCloud)
    {
        var output = Forward(pointCloud);

        // Output should be [1, numClasses], convert to vector
        return new Vector<T>(output.Data);
    }

    public override void Train(Tensor<T> input, Tensor<T> expectedOutput)
    {
        // Reset global features
        _globalFeatures = null;

        // Forward pass
        var prediction = Forward(input);

        // Compute loss
        if (LossFunction == null)
        {
            throw new InvalidOperationException("Loss function not set for training.");
        }

        var loss = LossFunction.ComputeLoss(prediction, expectedOutput);

        // Backward pass
        var lossGradient = LossFunction.ComputeGradient(prediction, expectedOutput);
        Backward(lossGradient);

        // Note: Parameter updates would typically be handled by an optimizer
        // For basic SGD, could call UpdateParameters on each layer
    }

    public override Tensor<T> Predict(Tensor<T> input)
    {
        SetTrainingMode(false);
        var output = Forward(input);
        return output;
    }
}
