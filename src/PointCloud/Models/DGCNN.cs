using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.PointCloud.Interfaces;
using AiDotNet.PointCloud.Layers;

namespace AiDotNet.PointCloud.Models;

/// <summary>
/// Implements Dynamic Graph CNN (DGCNN) for point cloud processing.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> DGCNN treats point clouds as graphs and uses edge convolutions to learn features.
/// </para>
/// <para>
/// Key innovations of DGCNN:
/// - Dynamic graph construction: Rebuilds neighborhood graph at each layer based on learned features
/// - Edge convolution: Learns features from edges connecting nearby points
/// - Captures local geometric structure more effectively than PointNet
/// - Adapts to the feature space, not just spatial coordinates
/// </para>
/// <para>
/// How DGCNN differs from PointNet:
/// - PointNet: Processes each point independently, then aggregates
/// - DGCNN: Explicitly models relationships between neighboring points
/// - Dynamic graphs: Neighborhoods change as features evolve through layers
/// </para>
/// <para>
/// Edge Convolution explained:
/// 1. For each point, find K nearest neighbors (in feature space or spatial)
/// 2. Compute edge features: combine point feature with neighbor features
/// 3. Apply MLP to edge features
/// 4. Aggregate (max pool) edge features for each point
/// 5. Result: New features that incorporate local structure
/// </para>
/// <para>
/// Why dynamic graphs are powerful:
/// - Early layers: Use spatial proximity (XYZ coordinates)
/// - Later layers: Use semantic similarity (learned features)
/// - Example: Points on same chair leg become neighbors even if spatially distant
/// </para>
/// <para>
/// Architecture:
/// 1. Multiple EdgeConv layers with increasing feature dimensions
/// 2. Each layer rebuilds k-NN graph based on current features
/// 3. Concatenate features from all EdgeConv layers
/// 4. Max pooling for global features
/// 5. Fully connected layers for classification/segmentation
/// </para>
/// <para>
/// Applications:
/// - Classification: Achieves state-of-the-art on ModelNet40
/// - Part segmentation: Excellent for identifying object parts
/// - Semantic segmentation: Captures fine-grained geometric details
/// - Better than PointNet at capturing local structure
/// </para>
/// <para>
/// Example - chair classification:
/// - Layer 1: Find spatial neighbors (nearby points)
/// - Layer 2: Find points with similar low-level features (edges, corners)
/// - Layer 3: Find points with similar mid-level features (vertical bars, flat surfaces)
/// - Layer 4: Find points with similar high-level features (legs, back, seat)
/// - Final: Combine all levels to recognize "chair"
/// </para>
/// <para>
/// Reference: "Dynamic Graph CNN for Learning on Point Clouds"
/// by Wang et al., ACM Transactions on Graphics 2019
/// </para>
/// </remarks>
public class DGCNN<T> : NeuralNetworkBase<T>, IPointCloudModel<T>, IPointCloudClassification<T>, IPointCloudSegmentation<T>
{
    private readonly int _numClasses;
    private readonly int _knnK; // Number of nearest neighbors
    private readonly int[] _edgeConvChannels; // Output channels for each EdgeConv layer
    private readonly bool _useDropout;
    private readonly double _dropoutRate;

    private readonly List<EdgeConvLayer<T>> _edgeConvLayers;
    private Vector<T>? _globalFeatures;

    /// <summary>
    /// Initializes a new instance of the DGCNN class.
    /// </summary>
    /// <param name="numClasses">Number of output classes for classification.</param>
    /// <param name="knnK">Number of nearest neighbors for graph construction.</param>
    /// <param name="edgeConvChannels">Output channel dimensions for each EdgeConv layer.</param>
    /// <param name="useDropout">Whether to use dropout regularization.</param>
    /// <param name="dropoutRate">Dropout rate (if dropout is enabled).</param>
    /// <param name="lossFunction">Optional loss function for training.</param>
    /// <remarks>
    /// <b>For Beginners:</b> Creates a DGCNN model for point cloud processing with dynamic graphs.
    ///
    /// Parameters explained:
    /// - numClasses: How many categories to classify (e.g., 40 for ModelNet40)
    /// - knnK: How many neighbors to consider for each point
    ///   - Typical values: 20-40
    ///   - Larger K: More context, but more computation
    ///   - Smaller K: Faster, but might miss important relationships
    /// - edgeConvChannels: Feature dimensions at each EdgeConv layer
    ///   Example: [64, 64, 128, 256]
    ///   - Layer 1: 64-dimensional features
    ///   - Layer 2: 64-dimensional features
    ///   - Layer 3: 128-dimensional features
    ///   - Layer 4: 256-dimensional features
    /// - useDropout: Prevents overfitting by randomly dropping neurons during training
    /// - dropoutRate: Fraction of neurons to drop (e.g., 0.5 means drop 50%)
    ///
    /// Example configuration for ModelNet40:
    /// - knnK: 20
    /// - edgeConvChannels: [64, 64, 128, 256]
    /// - useDropout: true
    /// - dropoutRate: 0.5
    ///
    /// The network will:
    /// 1. Build k-NN graph (find 20 nearest neighbors for each point)
    /// 2. Apply EdgeConv to learn from local neighborhoods
    /// 3. Rebuild graph based on new features
    /// 4. Repeat for all EdgeConv layers
    /// 5. Aggregate global features and classify
    /// </remarks>
    public DGCNN(
        int numClasses,
        int knnK = 20,
        int[]? edgeConvChannels = null,
        bool useDropout = true,
        double dropoutRate = 0.5,
        ILossFunction<T>? lossFunction = null)
        : base(CreateArchitecture(numClasses), lossFunction)
    {
        _numClasses = numClasses;
        _knnK = knnK;
        _edgeConvChannels = edgeConvChannels ?? [64, 64, 128, 256];
        _useDropout = useDropout;
        _dropoutRate = dropoutRate;
        _edgeConvLayers = [];

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
        // Build EdgeConv layers
        int inputChannels = 3; // Start with XYZ coordinates

        foreach (var outChannels in _edgeConvChannels)
        {
            var edgeConvLayer = new EdgeConvLayer<T>(inputChannels, outChannels, _knnK);
            _edgeConvLayers.Add(edgeConvLayer);
            AddLayerToCollection(edgeConvLayer);

            inputChannels = outChannels;
        }

        // Concatenate features from all EdgeConv layers
        int totalFeatures = _edgeConvChannels.Sum();

        // Global feature aggregation
        AddLayerToCollection(new MaxPoolingLayer<T>(totalFeatures));

        // Classification head with dropout
        AddLayerToCollection(new PointConvolutionLayer<T>(totalFeatures, 512));
        // Would add dropout layer here if implemented
        AddLayerToCollection(new PointConvolutionLayer<T>(512, 256));
        // Would add dropout layer here if implemented
        AddLayerToCollection(new PointConvolutionLayer<T>(256, _numClasses));
    }

    public override Tensor<T> Forward(Tensor<T> input)
    {
        var edgeFeatures = new List<Tensor<T>>();
        Tensor<T> x = input;

        // Process through EdgeConv layers and collect features
        foreach (var edgeConvLayer in _edgeConvLayers)
        {
            x = edgeConvLayer.Forward(x);
            edgeFeatures.Add(x);
        }

        // Concatenate all edge features
        x = ConcatenateFeatures(edgeFeatures);

        // Continue through remaining layers (max pooling and classification head)
        int startIdx = _edgeConvLayers.Count;
        for (int i = startIdx; i < Layers.Count; i++)
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

    private Tensor<T> ConcatenateFeatures(List<Tensor<T>> features)
    {
        if (features.Count == 0)
        {
            throw new ArgumentException("No features to concatenate.");
        }

        int numPoints = features[0].Shape[0];
        int totalChannels = features.Sum(f => f.Shape[1]);
        var concatenated = new T[numPoints * totalChannels];

        for (int n = 0; n < numPoints; n++)
        {
            int outIdx = 0;
            foreach (var feature in features)
            {
                int featureChannels = feature.Shape[1];
                for (int c = 0; c < featureChannels; c++)
                {
                    concatenated[n * totalChannels + outIdx++] = feature.Data[n * featureChannels + c];
                }
            }
        }

        return new Tensor<T>(concatenated, [numPoints, totalChannels]);
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
        var edgeFeatures = new List<Tensor<T>>();
        Tensor<T> x = pointCloud;

        foreach (var edgeConvLayer in _edgeConvLayers)
        {
            x = edgeConvLayer.Forward(x);
            edgeFeatures.Add(x);
        }

        return ConcatenateFeatures(edgeFeatures);
    }

    public Vector<T> ClassifyPointCloud(Tensor<T> pointCloud)
    {
        var output = Forward(pointCloud);
        return new Vector<T>(output.Data);
    }

    public Tensor<T> SegmentPointCloud(Tensor<T> pointCloud)
    {
        // For segmentation, concatenate global features with point features
        var pointFeatures = ExtractPointFeatures(pointCloud);
        var globalFeatures = ExtractGlobalFeatures(pointCloud);

        // Would normally repeat global features and concatenate with point features
        // Then apply segmentation head
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
/// Implements an Edge Convolution layer for DGCNN.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <b>For Beginners:</b> EdgeConv is the key building block of DGCNN.
///
/// What EdgeConv does:
/// 1. Build a graph: Connect each point to its K nearest neighbors
/// 2. Compute edge features: For each edge, combine features from both endpoints
/// 3. Transform: Apply MLP to edge features
/// 4. Aggregate: Max pool edge features for each point
///
/// Edge feature computation:
/// - For point i with neighbor j: edge_ij = MLP(concat(feature_i, feature_j - feature_i))
/// - This captures both the point's feature and its relationship to neighbors
///
/// Example:
/// - Point at chair leg: High features, neighbors also high (smooth region)
/// - Point at edge: High features, neighbors vary (captures edge information)
/// - The difference (feature_j - feature_i) captures local geometry
///
/// Dynamic aspect:
/// - Early layers: Neighbors are spatially close points
/// - Later layers: Neighbors are semantically similar points
/// - Graph structure adapts as features evolve
/// </remarks>
internal class EdgeConvLayer<T> : LayerBase<T>
{
    private readonly int _inputChannels;
    private readonly int _outputChannels;
    private readonly int _k; // Number of nearest neighbors
    private readonly PointConvolutionLayer<T> _mlp;
    private Tensor<T>? _lastInput;
    private int[,]? _knnIndices; // Store k-NN indices for backward pass

    public EdgeConvLayer(int inputChannels, int outputChannels, int k)
    {
        _inputChannels = inputChannels;
        _outputChannels = outputChannels;
        _k = k;

        // MLP processes edge features (point + neighbor difference)
        // Input: 2 * inputChannels (point feature + difference feature)
        _mlp = new PointConvolutionLayer<T>(2 * inputChannels, outputChannels);

        Parameters = _mlp.GetParameters();
    }

    public override int[] GetInputShape() => [0, _inputChannels];

    public override int[] GetOutputShape() => [0, _outputChannels];

    public override Tensor<T> Forward(Tensor<T> input)
    {
        _lastInput = input;
        int numPoints = input.Shape[0];

        // Build k-NN graph based on current features
        _knnIndices = BuildKNNGraph(input, _k);

        // Compute edge features
        var edgeFeatures = ComputeEdgeFeatures(input, _knnIndices);

        // Apply MLP to edge features
        var transformedEdges = _mlp.Forward(edgeFeatures);

        // Aggregate: max pool over neighbors for each point
        var output = AggregateEdgeFeatures(transformedEdges, numPoints);

        return output;
    }

    private int[,] BuildKNNGraph(Tensor<T> features, int k)
    {
        int numPoints = features.Shape[0];
        var knnIndices = new int[numPoints, k];
        var numOps = NumOps;

        // For each point, find k nearest neighbors
        for (int i = 0; i < numPoints; i++)
        {
            var distances = new List<(double dist, int idx)>();

            for (int j = 0; j < numPoints; j++)
            {
                if (i == j) continue;

                // Compute Euclidean distance in feature space
                double distSq = 0;
                for (int c = 0; c < _inputChannels; c++)
                {
                    var diff = numOps.Subtract(
                        features.Data[i * _inputChannels + c],
                        features.Data[j * _inputChannels + c]
                    );
                    var diffDouble = numOps.ToDouble(diff);
                    distSq += diffDouble * diffDouble;
                }

                distances.Add((Math.Sqrt(distSq), j));
            }

            // Sort by distance and take top k
            var topK = distances.OrderBy(d => d.dist).Take(k).ToArray();
            for (int kIdx = 0; kIdx < k && kIdx < topK.Length; kIdx++)
            {
                knnIndices[i, kIdx] = topK[kIdx].idx;
            }
        }

        return knnIndices;
    }

    private Tensor<T> ComputeEdgeFeatures(Tensor<T> input, int[,] knnIndices)
    {
        int numPoints = input.Shape[0];
        int k = knnIndices.GetLength(1);
        var numOps = NumOps;

        // Edge features: [numPoints * k, 2 * inputChannels]
        var edgeFeatures = new T[numPoints * k * 2 * _inputChannels];

        for (int i = 0; i < numPoints; i++)
        {
            for (int kIdx = 0; kIdx < k; kIdx++)
            {
                int neighborIdx = knnIndices[i, kIdx];
                int outIdx = (i * k + kIdx) * 2 * _inputChannels;

                // First half: point feature
                for (int c = 0; c < _inputChannels; c++)
                {
                    edgeFeatures[outIdx + c] = input.Data[i * _inputChannels + c];
                }

                // Second half: difference feature (neighbor - point)
                for (int c = 0; c < _inputChannels; c++)
                {
                    var neighborFeature = input.Data[neighborIdx * _inputChannels + c];
                    var pointFeature = input.Data[i * _inputChannels + c];
                    edgeFeatures[outIdx + _inputChannels + c] = numOps.Subtract(neighborFeature, pointFeature);
                }
            }
        }

        return new Tensor<T>(edgeFeatures, [numPoints * k, 2 * _inputChannels]);
    }

    private Tensor<T> AggregateEdgeFeatures(Tensor<T> edgeFeatures, int numPoints)
    {
        int k = edgeFeatures.Shape[0] / numPoints;
        var output = new T[numPoints * _outputChannels];
        var numOps = NumOps;

        // Max pool over k neighbors for each point
        for (int i = 0; i < numPoints; i++)
        {
            for (int c = 0; c < _outputChannels; c++)
            {
                T maxVal = edgeFeatures.Data[(i * k) * _outputChannels + c];

                for (int kIdx = 1; kIdx < k; kIdx++)
                {
                    T val = edgeFeatures.Data[(i * k + kIdx) * _outputChannels + c];
                    if (numOps.Compare(val, maxVal) > 0)
                    {
                        maxVal = val;
                    }
                }

                output[i * _outputChannels + c] = maxVal;
            }
        }

        return new Tensor<T>(output, [numPoints, _outputChannels]);
    }

    public override Tensor<T> Backward(Tensor<T> outputGradient)
    {
        // Simplified backward pass
        // Full implementation would backprop through aggregation and k-NN graph
        return _mlp.Backward(outputGradient);
    }

    public override void UpdateParameters(T learningRate)
    {
        _mlp.UpdateParameters(learningRate);
    }

    public override void ClearGradients()
    {
        _mlp.ClearGradients();
    }

    public override int ParameterCount => _mlp.ParameterCount;

    public override bool SupportsTraining => true;
}
