using System.Collections.Generic;
using System.Linq;
using AiDotNet.ActivationFunctions;
using AiDotNet.Autodiff;
using AiDotNet.Interfaces;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.PointCloud.Interfaces;
using AiDotNet.PointCloud.Layers;
using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Tensors;

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
/// <example>
/// <code>
/// // Create a DGCNN model for point cloud classification with dynamic edge convolution
/// var dgcnn = new DGCNN&lt;float&gt;(
///     numClasses: 40,
///     knnK: 20,
///     edgeConvChannels: new[] { 64, 64, 128, 256 },
///     useDropout: true,
///     dropoutRate: 0.5);
/// </code>
/// </example>
[ModelDomain(ModelDomain.Vision)]
[ModelDomain(ModelDomain.MachineLearning)]
[ModelCategory(ModelCategory.NeuralNetwork)]
[ModelCategory(ModelCategory.GraphNetwork)]
[ModelTask(ModelTask.Classification)]
[ModelTask(ModelTask.Segmentation)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Vector<>))]
[ResearchPaper("Dynamic Graph CNN for Learning on Point Clouds", "https://doi.org/10.1145/3326362", Year = 2019, Authors = "Yue Wang, Yongbin Sun, Ziwei Liu, Sanjay E. Sarma, Michael M. Bronstein, Justin M. Solomon")]
public class DGCNN<T> : NeuralNetworkBase<T>, IPointCloudModel<T>, IPointCloudClassification<T>, IPointCloudSegmentation<T>
{
    private readonly DGCNNOptions _options;

    /// <inheritdoc/>
    public override ModelOptions GetOptions() => _options;

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
    public DGCNN(DGCNNOptions options, ILossFunction<T>? lossFunction = null, DGCNNOptions? modelOptions = null)
        : base(CreateArchitecture(options.NumClasses, options.InputFeatureDim), lossFunction ?? NeuralNetworkHelper<T>.GetDefaultLossFunction(NeuralNetworkTaskType.MultiClassClassification))
    {
        _options = modelOptions ?? new DGCNNOptions();
        Options = _options;

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

        if (Architecture?.Layers != null && Architecture.Layers.Count > 0)
        {
            foreach (var layer in Architecture.Layers)
            {
                AddLayerToCollection(layer);
            }

            return;
        }

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
                _numClasses,
                activationFunction: new IdentityActivation<T>());
            AddLayerToCollection(outputLayer);
            _classificationHeadLayers.Add(outputLayer);
            return;
        }

        foreach (var hidden in _classifierChannels)
        {
            var dense = new DenseLayer<T>(
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
            _numClasses,
            activationFunction: new IdentityActivation<T>());
        AddLayerToCollection(output);
        _classificationHeadLayers.Add(output);
    }

    /// <summary>
    /// Routes the tape-based training forward through <see cref="ForwardWithMemory"/> — the
    /// DGCNN architecture (multi-scale EdgeConv skip-concatenation → global max-pool → head)
    /// is NOT a plain sequential layer stack, so the base's default sequential
    /// <c>ForwardForTraining</c> ran the wrong graph (and the wrong global-pool input width).
    /// With the EdgeConv gather/aggregate and the skip-concatenation now built from
    /// tape-tracked Engine ops, this makes training use the same differentiable graph as
    /// inference, so the gradient reaches every EdgeConv/head parameter.
    /// </summary>
    public override Tensor<T> ForwardForTraining(Tensor<T> input) => ForwardWithMemory(input);

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
                _globalFeatures = new Vector<T>(x.Data.ToArray());
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

        // Concatenate the per-EdgeConv features along the channel axis with a tape-tracked
        // Engine.Concat. The prior scalar-loop copy into a fresh new Tensor severed the
        // autodiff tape, blocking the gradient from flowing back into the EdgeConv layers
        // (DGCNN's multi-scale skip aggregation, Wang et al. 2019).
        return features.Count == 1 ? features[0] : Engine.Concat(features, 1);
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
        return new Vector<T>(output.Data.ToArray());
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
        try
        {
            TrainWithTape(input, expectedOutput);
        }
        finally
        {
            SetTrainingMode(false);
        }
    }

    protected override Tensor<T> PredictCore(Tensor<T> input)
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
            int layerParameterCount = checked((int)layer.ParameterCount);
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
            ModelData = SerializeForMetadata()
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
        int total = numPoints * k;

        // Flattened gather indices [numPoints*k]: for each point, its own index repeated
        // k times (self), and its k neighbor indices. The kNN indices themselves are
        // NON-differentiable (an arg-sort in feature space — exactly as DGCNN / PyTorch
        // treat them), but GATHERING the point features by them IS differentiable, so the
        // gradient flows back to the input point features through the edge features.
        var selfIndices = new int[total];
        var neighborIndices = new int[total];
        for (int i = 0; i < numPoints; i++)
        {
            for (int kk = 0; kk < k; kk++)
            {
                selfIndices[i * k + kk] = i;
                neighborIndices[i * k + kk] = knnIndices[i, kk];
            }
        }

        var xi = Engine.TensorGather(input, new Tensor<int>(selfIndices, [total]), axis: 0);      // [P*k, C]
        var xj = Engine.TensorGather(input, new Tensor<int>(neighborIndices, [total]), axis: 0);  // [P*k, C]
        var diff = Engine.TensorSubtract(xj, xi);                                                 // [P*k, C]

        // DGCNN edge feature (Wang et al. 2019): concat([x_i, x_j - x_i]) -> [P*k, 2C].
        // Tape-tracked Engine ops throughout; the prior scalar-loop build detached the
        // tape (a fresh new Tensor from Data.Span reads), so no gradient reached the
        // input or, downstream, the MLP weights — training diverged.
        return Engine.Concat(new[] { xi, diff }, 1);                                              // [P*k, 2C]
    }

    private Tensor<T> AggregateEdgeFeatures(Tensor<T> edgeFeatures, int numPoints)
    {
        int k = edgeFeatures.Shape[0] / numPoints;

        // Symmetric max-pool over the k neighbors per point (DGCNN's permutation-
        // invariant aggregation): reshape [P*k, outC] -> [P, k, outC], max over the k
        // axis -> [P, outC]. Engine.ReduceMax is tape-tracked (its backward routes the
        // gradient to the arg-max neighbor), so the whole EdgeConv is differentiable
        // end-to-end. The prior scalar-loop max wrote a fresh new Tensor, severing the
        // tape so the MLP weights received zero gradient and the loss diverged.
        var reshaped = Engine.Reshape(edgeFeatures, [numPoints, k, _outputChannels]);
        var pooled = Engine.ReduceMax(reshaped, new[] { 1 }, keepDims: false, out var argMax); // [P, outC]

        // Preserve the per-(point,channel) arg-max neighbor index for the legacy manual
        // ComputeGradients path (unused by the tape training path, kept for compatibility).
        _maxIndices = new int[numPoints, _outputChannels];
        if (argMax is not null && argMax.Length >= numPoints * _outputChannels)
        {
            for (int i = 0; i < numPoints; i++)
                for (int c = 0; c < _outputChannels; c++)
                    _maxIndices[i, c] = argMax[i * _outputChannels + c];
        }

        return pooled;
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

    public override long ParameterCount => _mlp.ParameterCount;

    public override bool SupportsTraining => true;
}
