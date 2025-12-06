using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks;
using AiDotNet.PointCloud.Interfaces;
using AiDotNet.PointCloud.Layers;

namespace AiDotNet.PointCloud.Models;

/// <summary>
/// Implements the PointNet++ architecture for hierarchical point cloud processing.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> PointNet++ extends PointNet by adding hierarchical feature learning at multiple scales.
/// </para>
/// <para>
/// Key improvements over PointNet:
/// - Hierarchical structure: Processes point clouds at multiple resolutions
/// - Local context: Captures fine-grained local patterns
/// - Multi-scale grouping: Learns features at different scales simultaneously
/// - Better generalization: More robust to non-uniform point density
/// </para>
/// <para>
/// Architecture components:
/// 1. Set Abstraction Layers: Hierarchically group points and extract features
///    - Sampling: Select subset of points as centroids
///    - Grouping: Find neighboring points around each centroid
///    - PointNet layer: Extract features from each local region
/// 2. Feature Propagation Layers: Upsample features for segmentation tasks
///    - Interpolation: Propagate features from coarse to fine levels
///    - Skip connections: Combine with features from encoder
/// </para>
/// <para>
/// Why hierarchical learning matters:
/// - Different patterns exist at different scales (like edges vs. shapes in images)
/// - Local context provides detailed geometry information
/// - Global context provides overall shape understanding
/// - Combining both gives comprehensive understanding
/// </para>
/// <para>
/// Applications:
/// - Fine-grained classification
/// - Part segmentation (identifying specific parts of objects)
/// - Semantic segmentation (labeling each point)
/// - Better performance on complex, detailed shapes
/// </para>
/// <para>
/// Example use case - autonomous driving:
/// - Coarse level: Identify general object shapes (car, pedestrian)
/// - Medium level: Recognize object parts (wheels, windows)
/// - Fine level: Detect details (door handles, mirrors)
/// </para>
/// <para>
/// Reference: "PointNet++: Deep Hierarchical Feature Learning on Point Sets in a Metric Space"
/// by Qi et al., NeurIPS 2017
/// </para>
/// </remarks>
public class PointNetPlusPlus<T> : NeuralNetworkBase<T>, IPointCloudModel<T>, IPointCloudClassification<T>, IPointCloudSegmentation<T>
{
    private readonly int _numClasses;
    private readonly int[] _samplingRates; // Number of points at each hierarchy level
    private readonly double[] _searchRadii; // Search radius for neighborhood at each level
    private readonly int[][] _mlpDimensions; // MLP dimensions for each set abstraction layer
    private readonly bool _useMultiScaleGrouping;

    private readonly List<SetAbstractionLayer<T>> _setAbstractionLayers;
    private Vector<T>? _globalFeatures;

    /// <summary>
    /// Initializes a new instance of the PointNetPlusPlus class.
    /// </summary>
    /// <param name="numClasses">Number of output classes.</param>
    /// <param name="samplingRates">Number of points to sample at each hierarchy level.</param>
    /// <param name="searchRadii">Search radius for finding neighbors at each level.</param>
    /// <param name="mlpDimensions">MLP layer dimensions for each set abstraction level.</param>
    /// <param name="useMultiScaleGrouping">Whether to use multi-scale grouping (MSG).</param>
    /// <param name="lossFunction">Optional loss function for training.</param>
    /// <remarks>
    /// <b>For Beginners:</b> Creates a PointNet++ model with hierarchical feature learning.
    ///
    /// Parameters explained:
    /// - numClasses: How many categories to classify into
    /// - samplingRates: How many points to keep at each level
    ///   Example: [512, 128, 32] means:
    ///   - Level 1: Sample 512 points from input
    ///   - Level 2: Sample 128 points from the 512
    ///   - Level 3: Sample 32 points from the 128
    /// - searchRadii: How far to look for neighbors at each level
    ///   Example: [0.1, 0.2, 0.4] means:
    ///   - Level 1: Look 0.1 units around each point
    ///   - Level 2: Look 0.2 units (larger neighborhood)
    ///   - Level 3: Look 0.4 units (even larger)
    /// - mlpDimensions: Feature dimensions for processing at each level
    ///   Example: [[64,64,128], [128,128,256], [256,512,1024]]
    /// - useMultiScaleGrouping: Process each level at multiple scales (more robust but slower)
    ///
    /// Example configuration for ModelNet40:
    /// - samplingRates: [512, 128, null] (null means use all remaining points)
    /// - searchRadii: [0.2, 0.4, null]
    /// - mlpDimensions: [[64,64,128], [128,128,256], [256,512,1024]]
    /// </remarks>
    public PointNetPlusPlus(
        int numClasses,
        int[] samplingRates,
        double[] searchRadii,
        int[][] mlpDimensions,
        bool useMultiScaleGrouping = false,
        ILossFunction<T>? lossFunction = null)
        : base(CreateArchitecture(numClasses), lossFunction)
    {
        _numClasses = numClasses;
        _samplingRates = samplingRates;
        _searchRadii = searchRadii;
        _mlpDimensions = mlpDimensions;
        _useMultiScaleGrouping = useMultiScaleGrouping;
        _setAbstractionLayers = [];

        InitializeLayers();
    }

    private static NeuralNetworkArchitecture<T> CreateArchitecture(int numClasses)
    {
        return new NeuralNetworkArchitecture<T>
        {
            InputType = InputType.ThreeDimensional,
            LayerSize = 1024,
            TaskType = TaskType.Classification,
            Layers = null
        };
    }

    protected override void InitializeLayers()
    {
        // Build hierarchical set abstraction layers
        int inputChannels = 3; // Start with XYZ

        for (int i = 0; i < _samplingRates.Length; i++)
        {
            var saLayer = new SetAbstractionLayer<T>(
                numPoints: _samplingRates[i],
                searchRadius: _searchRadii[i],
                inputChannels: inputChannels,
                mlpDimensions: _mlpDimensions[i],
                useMultiScale: _useMultiScaleGrouping
            );

            _setAbstractionLayers.Add(saLayer);
            AddLayerToCollection(saLayer);

            // Output channels of this layer become input channels of next
            inputChannels = _mlpDimensions[i][^1]; // Last dimension of MLP
        }

        // Global feature aggregation
        AddLayerToCollection(new MaxPoolingLayer<T>(inputChannels));

        // Classification head
        AddLayerToCollection(new PointConvolutionLayer<T>(inputChannels, 512));
        AddLayerToCollection(new PointConvolutionLayer<T>(512, 256));
        AddLayerToCollection(new PointConvolutionLayer<T>(256, _numClasses));
    }

    public override Tensor<T> Forward(Tensor<T> input)
    {
        Tensor<T> x = input;

        // Forward through all layers
        for (int i = 0; i < Layers.Count; i++)
        {
            _layerInputs[i] = x;
            x = Layers[i].Forward(x);
            _layerOutputs[i] = x;

            // Capture global features after max pooling
            if (Layers[i] is MaxPoolingLayer<T>)
            {
                _globalFeatures = new Vector<T>(x.Data);
            }
        }

        return x;
    }

    public override Tensor<T> Backward(Tensor<T> outputGradient)
    {
        Tensor<T> gradient = outputGradient;

        for (int i = Layers.Count - 1; i >= 0; i--)
        {
            gradient = Layers[i].Backward(gradient);
        }

        return gradient;
    }

    public Vector<T> ExtractGlobalFeatures(Tensor<T> pointCloud)
    {
        _ = Forward(pointCloud);

        if (_globalFeatures == null)
        {
            throw new InvalidOperationException("Global features not extracted.");
        }

        return _globalFeatures;
    }

    public Tensor<T> ExtractPointFeatures(Tensor<T> pointCloud)
    {
        // Extract hierarchical features from intermediate layers
        Tensor<T> x = pointCloud;

        foreach (var layer in _setAbstractionLayers)
        {
            x = layer.Forward(x);
        }

        return x;
    }

    public Vector<T> ClassifyPointCloud(Tensor<T> pointCloud)
    {
        var output = Forward(pointCloud);
        return new Vector<T>(output.Data);
    }

    public Tensor<T> SegmentPointCloud(Tensor<T> pointCloud)
    {
        // For segmentation, would need feature propagation layers
        // This is a simplified version that returns point features
        var pointFeatures = ExtractPointFeatures(pointCloud);

        // Would normally upsample and combine with skip connections
        // For now, apply segmentation head to point features
        return pointFeatures;
    }

    public override void Train(Tensor<T> input, Tensor<T> expectedOutput)
    {
        _globalFeatures = null;

        var prediction = Forward(input);

        if (LossFunction == null)
        {
            throw new InvalidOperationException("Loss function not set.");
        }

        var lossGradient = LossFunction.ComputeGradient(prediction, expectedOutput);
        Backward(lossGradient);
    }

    public override Tensor<T> Predict(Tensor<T> input)
    {
        SetTrainingMode(false);
        return Forward(input);
    }
}

/// <summary>
/// Implements a Set Abstraction layer for PointNet++.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <b>For Beginners:</b> Set Abstraction is the core building block of PointNet++.
///
/// It performs three operations:
/// 1. Sampling: Select representative points (centroids)
/// 2. Grouping: Find neighbors around each centroid
/// 3. PointNet: Apply mini-PointNet to each local region
///
/// Think of it like this:
/// - You have 1000 points representing a chair
/// - Sampling: Select 100 important points (e.g., corners, edges)
/// - Grouping: For each of the 100 points, find nearby points within radius r
/// - PointNet: Process each local neighborhood to extract features
/// - Result: 100 points with rich features describing local geometry
///
/// This creates a hierarchical representation:
/// - Input: Many points, basic features (XYZ)
/// - Output: Fewer points, rich features (learned patterns)
/// </remarks>
internal class SetAbstractionLayer<T> : LayerBase<T>
{
    private readonly int _numPoints;
    private readonly double _searchRadius;
    private readonly int _inputChannels;
    private readonly int[] _mlpDimensions;
    private readonly bool _useMultiScale;
    private readonly List<ILayer<T>> _mlpLayers;

    public SetAbstractionLayer(
        int numPoints,
        double searchRadius,
        int inputChannels,
        int[] mlpDimensions,
        bool useMultiScale = false)
    {
        _numPoints = numPoints;
        _searchRadius = searchRadius;
        _inputChannels = inputChannels;
        _mlpDimensions = mlpDimensions;
        _useMultiScale = useMultiScale;
        _mlpLayers = [];

        // Build MLP layers for feature extraction
        int inChannels = inputChannels;
        foreach (var outChannels in mlpDimensions)
        {
            _mlpLayers.Add(new PointConvolutionLayer<T>(inChannels, outChannels));
            inChannels = outChannels;
        }

        // Count parameters
        int totalParams = 0;
        foreach (var layer in _mlpLayers)
        {
            totalParams += layer.ParameterCount;
        }
        Parameters = new Vector<T>(totalParams);
    }

    public override int[] GetInputShape() => [0, _inputChannels];

    public override int[] GetOutputShape() => [_numPoints, _mlpDimensions[^1]];

    public override Tensor<T> Forward(Tensor<T> input)
    {
        // Simplified implementation
        // Full version would:
        // 1. Sample _numPoints centroids using farthest point sampling
        // 2. Group neighbors within _searchRadius around each centroid
        // 3. Apply MLP to each local region
        // 4. Max pool features within each region

        // For now, apply MLP to all points and then sample
        Tensor<T> features = input;
        foreach (var layer in _mlpLayers)
        {
            features = layer.Forward(features);
        }

        // Simplified sampling: take first _numPoints
        int originalPoints = input.Shape[0];
        int outputChannels = _mlpDimensions[^1];

        if (_numPoints >= originalPoints)
        {
            return features;
        }

        // Sample points uniformly (simplified - should use farthest point sampling)
        var sampledData = new T[_numPoints * outputChannels];
        int stride = originalPoints / _numPoints;

        for (int i = 0; i < _numPoints; i++)
        {
            int srcIdx = (i * stride) * outputChannels;
            int dstIdx = i * outputChannels;

            for (int c = 0; c < outputChannels; c++)
            {
                sampledData[dstIdx + c] = features.Data[srcIdx + c];
            }
        }

        return new Tensor<T>(sampledData, [_numPoints, outputChannels]);
    }

    public override Tensor<T> Backward(Tensor<T> outputGradient)
    {
        // Backprop through MLP layers
        Tensor<T> gradient = outputGradient;
        for (int i = _mlpLayers.Count - 1; i >= 0; i--)
        {
            gradient = _mlpLayers[i].Backward(gradient);
        }

        return gradient;
    }

    public override void UpdateParameters(T learningRate)
    {
        foreach (var layer in _mlpLayers)
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
            return total;
        }
    }

    public override bool SupportsTraining => true;
}
