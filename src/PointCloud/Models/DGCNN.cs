using System.Collections.Generic;
using System.Linq;
using AiDotNet.Autodiff;
using AiDotNet.ActivationFunctions;
using AiDotNet.Interfaces;
using AiDotNet.Tensors;
using AiDotNet.Models.Options;
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
    private int _numClasses;
    private int _inputFeatureDim;
    private int _knnK; // Number of nearest neighbors
    private int[] _edgeConvChannels; // Output channels for each EdgeConv layer
    private int[] _classifierChannels;
    private bool _useDropout;
    private double _dropoutRate;
    private T _learningRate;

    private readonly List<EdgeConvLayer<T>> _edgeConvLayers;
    private readonly List<ILayer<T>> _classificationHeadLayers;
    private Vector<T>? _globalFeatures;

    /// <summary>
    /// Initializes a new instance of the DGCNN class with default options.
    /// </summary>
    public DGCNN()
        : this(new DGCNNOptions(), null)
    {
    }

    /// <summary>
    /// Initializes a new instance of the DGCNN class with configurable options.
    /// </summary>
    /// <param name="options">Configuration options for the DGCNN model.</param>
    /// <param name="lossFunction">Optional loss function for training.</param>
    public DGCNN(DGCNNOptions options, ILossFunction<T>? lossFunction = null)
        : base(CreateArchitecture(options.NumClasses, options.InputFeatureDim), lossFunction ?? NeuralNetworkHelper<T>.GetDefaultLossFunction(NeuralNetworkTaskType.MultiClassClassification))
    {
        if (options == null)
        {
            throw new ArgumentNullException(nameof(options));
        }
        if (options.NumClasses <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(options.NumClasses), "Number of classes must be positive.");
        }
        if (options.InputFeatureDim <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(options.InputFeatureDim), "InputFeatureDim must be positive.");
        }
        if (options.KnnK <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(options.KnnK), "KnnK must be positive.");
        }
        if (options.DropoutRate < 0.0 || options.DropoutRate >= 1.0)
        {
            throw new ArgumentOutOfRangeException(nameof(options.DropoutRate), "DropoutRate must be in [0, 1).");
        }

        _numClasses = options.NumClasses;
        _inputFeatureDim = options.InputFeatureDim;
        _knnK = options.KnnK;
        _edgeConvChannels = ValidatePositiveArray(options.EdgeConvChannels, nameof(options.EdgeConvChannels));
        _classifierChannels = options.ClassifierChannels == null || options.ClassifierChannels.Length == 0
            ? []
            : ValidatePositiveArray(options.ClassifierChannels, nameof(options.ClassifierChannels));
        _useDropout = options.UseDropout;
        _dropoutRate = options.DropoutRate;
        _learningRate = NumOps.FromDouble(options.LearningRate);

        _edgeConvLayers = [];
        _classificationHeadLayers = [];
        InitializeLayers();
    }

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
        : this(new DGCNNOptions
            {
                NumClasses = numClasses,
                KnnK = knnK,
                EdgeConvChannels = edgeConvChannels ?? [64, 64, 128, 256],
                UseDropout = useDropout,
                DropoutRate = dropoutRate
            },
            lossFunction)
    {
    }

    private static NeuralNetworkArchitecture<T> CreateArchitecture(int numClasses, int inputFeatureDim = 3)
    {
        return new NeuralNetworkArchitecture<T>(
            inputType: InputType.ThreeDimensional,
            taskType: NeuralNetworkTaskType.MultiClassClassification,
            complexity: NetworkComplexity.Medium,
            inputHeight: 1,
            inputWidth: 1,
            inputDepth: inputFeatureDim,
            outputSize: numClasses);
    }

    protected override void InitializeLayers()
    {
        ClearLayers();
        _edgeConvLayers.Clear();
        _classificationHeadLayers.Clear();

        // Build EdgeConv layers
        int inputChannels = _inputFeatureDim; // Start with XYZ coordinates

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
        AddLayerToCollection(new AiDotNet.PointCloud.Layers.MaxPoolingLayer<T>(totalFeatures));

        int classifierInput = totalFeatures;
        if (_classifierChannels.Length == 0)
        {
            var outputLayer = new DenseLayer<T>(
                classifierInput,
                _numClasses,
                activationFunction: new IdentityActivation<T>());
            AddLayerToCollection(outputLayer);
            _classificationHeadLayers.Add(outputLayer);
            return;
        }

        foreach (var hidden in _classifierChannels)
        {
            var dense = new DenseLayer<T>(
                classifierInput,
                hidden,
                activationFunction: new ReLUActivation<T>());
            AddLayerToCollection(dense);
            _classificationHeadLayers.Add(dense);
            classifierInput = hidden;

            if (_useDropout && _dropoutRate > 0.0)
            {
                var dropout = new DropoutLayer<T>(_dropoutRate);
                AddLayerToCollection(dropout);
                _classificationHeadLayers.Add(dropout);
            }
        }

        var output = new DenseLayer<T>(
            classifierInput,
            _numClasses,
            activationFunction: new IdentityActivation<T>());
        AddLayerToCollection(output);
        _classificationHeadLayers.Add(output);
    }

    public override Tensor<T> ForwardWithMemory(Tensor<T> input)
    {
        if (input.Shape.Length != 2 || input.Shape[1] != _inputFeatureDim)
        {
            throw new ArgumentException($"Input must have shape [N, {_inputFeatureDim}].", nameof(input));
        }

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
            if (Layers[i] is AiDotNet.PointCloud.Layers.MaxPoolingLayer<T>)
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

    public override Tensor<T> Backpropagate(Tensor<T> outputGradient)
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
        bool originalMode = IsTrainingMode;
        SetTrainingMode(false);

        try
        {
            _ = ForwardWithMemory(pointCloud);

            if (_globalFeatures == null)
            {
                throw new InvalidOperationException("Global features not extracted.");
            }

            return _globalFeatures;
        }
        finally
        {
            SetTrainingMode(originalMode);
        }
    }

    public Tensor<T> ExtractPointFeatures(Tensor<T> pointCloud)
    {
        if (pointCloud.Shape.Length != 2 || pointCloud.Shape[1] != _inputFeatureDim)
        {
            throw new ArgumentException($"Input must have shape [N, {_inputFeatureDim}].", nameof(pointCloud));
        }

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
        var output = Predict(pointCloud);
        return new Vector<T>(output.Data);
    }

    public Tensor<T> SegmentPointCloud(Tensor<T> pointCloud)
    {
        bool originalMode = IsTrainingMode;
        SetTrainingMode(false);

        try
        {
            var pointFeatures = ExtractPointFeatures(pointCloud);
            Tensor<T> x = pointFeatures;
            foreach (var layer in _classificationHeadLayers)
            {
                x = layer.Forward(x);
            }

            return x;
        }
        finally
        {
            SetTrainingMode(originalMode);
        }
    }

    public override void Train(Tensor<T> input, Tensor<T> expectedOutput)       
    {
        SetTrainingMode(true);

        _globalFeatures = null;

        var prediction = ForwardWithMemory(input);

        if (LossFunction == null)
        {
            throw new InvalidOperationException("Loss function not set.");
        }

        var loss = LossFunction.ComputeLoss(prediction, expectedOutput);
        LastLoss = loss;

        var lossGradient = LossFunction.ComputeGradient(prediction, expectedOutput);
        Backpropagate(lossGradient);

        // Basic SGD parameter update
        foreach (var layer in Layers)
        {
            if (layer.SupportsTraining && layer.ParameterCount > 0)
            {
                layer.UpdateParameters(_learningRate);
            }
        }
    }

    public override Tensor<T> Predict(Tensor<T> input)
    {
        SetTrainingMode(false);
        return ForwardWithMemory(input);
    }

    public override bool SupportsTraining => true;

    public override void UpdateParameters(Vector<T> parameters)
    {
        int startIndex = 0;
        foreach (var layer in Layers)
        {
            int layerParameterCount = layer.ParameterCount;
            if (layerParameterCount > 0)
            {
                Vector<T> layerParameters = parameters.SubVector(startIndex, layerParameterCount);
                layer.UpdateParameters(layerParameters);
                startIndex += layerParameterCount;
            }
        }
    }

    public override ModelMetadata<T> GetModelMetadata()
    {
        return new ModelMetadata<T>
        {
            ModelType = ModelType.NeuralNetwork,
            AdditionalInfo = new Dictionary<string, object>
            {
                { "ModelName", "DGCNN" },
                { "NumClasses", _numClasses },
                { "InputFeatureDim", _inputFeatureDim },
                { "KnnK", _knnK },
                { "EdgeConvChannels", _edgeConvChannels },
                { "ClassifierChannels", _classifierChannels },
                { "UseDropout", _useDropout },
                { "DropoutRate", _dropoutRate },
                { "LearningRate", NumOps.ToDouble(_learningRate) },
                { "TotalParameters", ParameterCount },
                { "TotalLayers", Layers.Count },
                { "TaskType", Architecture.TaskType.ToString() }
            },
            ModelData = this.Serialize()
        };
    }

    protected override void SerializeNetworkSpecificData(BinaryWriter writer)
    {
        writer.Write(_numClasses);
        writer.Write(_inputFeatureDim);
        writer.Write(_knnK);
        writer.Write(_useDropout);
        writer.Write(_dropoutRate);
        writer.Write(NumOps.ToDouble(_learningRate));
        WriteIntArray(writer, _edgeConvChannels);
        WriteIntArray(writer, _classifierChannels);
    }

    protected override void DeserializeNetworkSpecificData(BinaryReader reader)
    {
        _numClasses = reader.ReadInt32();
        _inputFeatureDim = reader.ReadInt32();
        _knnK = reader.ReadInt32();
        _useDropout = reader.ReadBoolean();
        _dropoutRate = reader.ReadDouble();
        _learningRate = NumOps.FromDouble(reader.ReadDouble());
        _edgeConvChannels = ReadIntArray(reader, nameof(_edgeConvChannels), allowEmpty: false);
        _classifierChannels = ReadIntArray(reader, nameof(_classifierChannels), allowEmpty: true);

        _edgeConvLayers.Clear();
        _classificationHeadLayers.Clear();
        bool afterPooling = false;
        foreach (var layer in Layers)
        {
            if (layer is EdgeConvLayer<T> edgeLayer)
            {
                _edgeConvLayers.Add(edgeLayer);
            }
            if (layer is AiDotNet.PointCloud.Layers.MaxPoolingLayer<T>)
            {
                afterPooling = true;
                continue;
            }
            if (afterPooling && (layer is DenseLayer<T> || layer is DropoutLayer<T>))
            {
                _classificationHeadLayers.Add(layer);
            }
        }
    }

    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance()
    {
        return new DGCNN<T>(
            new DGCNNOptions
            {
                NumClasses = _numClasses,
                InputFeatureDim = _inputFeatureDim,
                KnnK = _knnK,
                EdgeConvChannels = _edgeConvChannels,
                ClassifierChannels = _classifierChannels,
                UseDropout = _useDropout,
                DropoutRate = _dropoutRate,
                LearningRate = NumOps.ToDouble(_learningRate)
            },
            LossFunction);
    }

    private static int[] ValidatePositiveArray(int[]? values, string paramName)
    {
        if (values == null)
        {
            throw new ArgumentNullException(paramName);
        }
        if (values.Length == 0)
        {
            throw new ArgumentException("Array must not be empty.", paramName);
        }
        for (int i = 0; i < values.Length; i++)
        {
            if (values[i] <= 0)
            {
                throw new ArgumentOutOfRangeException(paramName, "Values must be positive.");
            }
        }

        return values;
    }

    private static void WriteIntArray(BinaryWriter writer, int[] values)
    {
        writer.Write(values.Length);
        for (int i = 0; i < values.Length; i++)
        {
            writer.Write(values[i]);
        }
    }

    private static int[] ReadIntArray(BinaryReader reader, string paramName, bool allowEmpty)
    {
        int length = reader.ReadInt32();
        if (length < 0)
        {
            throw new InvalidOperationException("Serialized array length must be non-negative.");
        }
        if (length == 0 && !allowEmpty)
        {
            throw new InvalidOperationException($"Serialized array '{paramName}' must not be empty.");
        }
        var values = new int[length];
        for (int i = 0; i < length; i++)
        {
            values[i] = reader.ReadInt32();
        }

        if (length == 0)
        {
            return values;
        }

        return ValidatePositiveArray(values, paramName);
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
    private int[,]? _maxIndices; // Store max neighbor indices for backward pass

    public EdgeConvLayer(int inputChannels, int outputChannels, int k)
        : base([0, inputChannels], [0, outputChannels])
    {
        _inputChannels = inputChannels;
        _outputChannels = outputChannels;
        _k = k;

        // MLP processes edge features (point + neighbor difference)
        // Input: 2 * inputChannels (point feature + difference feature)
        _mlp = new PointConvolutionLayer<T>(2 * inputChannels, outputChannels, new ReLUActivation<T>());

        Parameters = _mlp.GetParameters();
    }

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

        // Vectorized k-NN using IEngine operations
        // Compute pairwise squared distances: [numPoints, numPoints]
        var distancesSq = Engine.PairwiseDistanceSquared(features, features);

        // Set diagonal to large value to exclude self-connections
        T largeVal = NumOps.FromDouble(1e30);
        for (int i = 0; i < numPoints; i++)
        {
            distancesSq[i, i] = largeVal;
        }

        // Get k smallest distances per row (largest=false)
        var (_, indices) = Engine.TopK(distancesSq, k, axis: 1, largest: false);

        // Copy to int[,] result array
        for (int i = 0; i < numPoints; i++)
        {
            for (int j = 0; j < k; j++)
            {
                knnIndices[i, j] = indices[i, j];
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
        _maxIndices = new int[numPoints, _outputChannels];

        // Max pool over k neighbors for each point
        for (int i = 0; i < numPoints; i++)
        {
            for (int c = 0; c < _outputChannels; c++)
            {
                T maxVal = edgeFeatures.Data[(i * k) * _outputChannels + c];    
                int maxIdx = 0;

                for (int kIdx = 1; kIdx < k; kIdx++)
                {
                    T val = edgeFeatures.Data[(i * k + kIdx) * _outputChannels + c];
                    if (numOps.GreaterThan(val, maxVal))
                    {
                        maxVal = val;
                        maxIdx = kIdx;
                    }
                }

                output[i * _outputChannels + c] = maxVal;
                _maxIndices[i, c] = maxIdx;
            }
        }

        return new Tensor<T>(output, [numPoints, _outputChannels]);
    }

    public override Tensor<T> Backward(Tensor<T> outputGradient)
    {
        if (_lastInput == null || _knnIndices == null || _maxIndices == null)
        {
            throw new InvalidOperationException("Forward pass must be called before backward pass.");
        }

        int numPoints = _lastInput.Shape[0];
        int k = _knnIndices.GetLength(1);

        // Backprop through max pooling over neighbors
        var edgeGradients = new T[numPoints * k * _outputChannels];
        for (int i = 0; i < numPoints; i++)
        {
            for (int c = 0; c < _outputChannels; c++)
            {
                int maxIdx = _maxIndices[i, c];
                int edgeIdx = (i * k + maxIdx) * _outputChannels + c;
                edgeGradients[edgeIdx] = outputGradient.Data[i * _outputChannels + c];
            }
        }

        var edgeGradientTensor = new Tensor<T>(edgeGradients, [numPoints * k, _outputChannels]);
        var edgeFeatureGradient = _mlp.Backward(edgeGradientTensor);

        // Map edge feature gradients back to input features
        var inputGradient = new T[numPoints * _inputChannels];
        var numOps = NumOps;

        for (int i = 0; i < numPoints; i++)
        {
            for (int kIdx = 0; kIdx < k; kIdx++)
            {
                int neighborIdx = _knnIndices[i, kIdx];
                int baseIdx = (i * k + kIdx) * 2 * _inputChannels;

                for (int c = 0; c < _inputChannels; c++)
                {
                    var gradSelf = edgeFeatureGradient.Data[baseIdx + c];
                    var gradDiff = edgeFeatureGradient.Data[baseIdx + _inputChannels + c];

                    inputGradient[i * _inputChannels + c] = numOps.Add(
                        inputGradient[i * _inputChannels + c],
                        numOps.Subtract(gradSelf, gradDiff));

                    inputGradient[neighborIdx * _inputChannels + c] = numOps.Add(
                        inputGradient[neighborIdx * _inputChannels + c],
                        gradDiff);
                }
            }
        }

        return new Tensor<T>(inputGradient, [numPoints, _inputChannels]);
    }

    public override void UpdateParameters(T learningRate)
    {
        _mlp.UpdateParameters(learningRate);
    }

    public override void ClearGradients()
    {
        _mlp.ClearGradients();
    }

    public override Vector<T> GetParameters()
    {
        return _mlp.GetParameters();
    }

    public override void UpdateParameters(Vector<T> parameters)
    {
        if (parameters.Length != ParameterCount)
        {
            throw new ArgumentException("Parameter vector length does not match layer parameter count.", nameof(parameters));
        }

        _mlp.UpdateParameters(parameters);
        Parameters = _mlp.GetParameters();
    }

    public override void ResetState()
    {
        _lastInput = null;
        _knnIndices = null;
        _maxIndices = null;
        _mlp.ResetState();
    }

    public override bool SupportsJitCompilation => false;

    public override ComputationNode<T> ExportComputationGraph(List<ComputationNode<T>> inputNodes)
    {
        throw new NotSupportedException(
            "EdgeConvLayer does not support computation graph export due to dynamic k-NN construction.");
    }

    public override int ParameterCount => _mlp.ParameterCount;

    public override bool SupportsTraining => true;
}
