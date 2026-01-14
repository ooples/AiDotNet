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
    private int _numClasses;
    private int _inputFeatureDim;
    private int[] _samplingRates; // Number of points at each hierarchy level
    private double[] _searchRadii; // Search radius for neighborhood at each level
    private int[] _neighborSamples;
    private int[][] _mlpDimensions; // MLP dimensions for each set abstraction layer
    private bool _useMultiScaleGrouping;
    private double[][]? _multiScaleRadii;
    private int[][][]? _multiScaleMlpDimensions;
    private int[][]? _multiScaleNeighborSamples;
    private int[] _classifierChannels;
    private bool _useDropout;
    private double _dropoutRate;
    private T _learningRate;

    private readonly List<SetAbstractionLayer<T>> _setAbstractionLayers;
    private readonly List<ILayer<T>> _classificationHeadLayers;
    private Vector<T>? _globalFeatures;

    /// <summary>
    /// Initializes a new instance of the PointNetPlusPlus class with default options.
    /// </summary>
    public PointNetPlusPlus()
        : this(new PointNetPlusPlusOptions(), null)
    {
    }

    /// <summary>
    /// Initializes a new instance of the PointNetPlusPlus class with configurable options.
    /// </summary>
    /// <param name="options">Configuration options for the PointNet++ model.</param>
    /// <param name="lossFunction">Optional loss function for training.</param>
    public PointNetPlusPlus(PointNetPlusPlusOptions options, ILossFunction<T>? lossFunction = null)
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
        if (options.DropoutRate < 0.0 || options.DropoutRate >= 1.0)
        {
            throw new ArgumentOutOfRangeException(nameof(options.DropoutRate), "DropoutRate must be in [0, 1).");
        }

        _numClasses = options.NumClasses;
        _inputFeatureDim = options.InputFeatureDim;
        _samplingRates = ValidatePositiveArray(options.SamplingRates, nameof(options.SamplingRates));
        _searchRadii = ValidatePositiveArray(options.SearchRadii, nameof(options.SearchRadii));
        _neighborSamples = ValidatePositiveArray(options.NeighborSamples, nameof(options.NeighborSamples));
        _mlpDimensions = ValidateMlpDimensions(options.MlpDimensions, nameof(options.MlpDimensions));
        _useMultiScaleGrouping = options.UseMultiScaleGrouping;
        _classifierChannels = options.ClassifierChannels == null || options.ClassifierChannels.Length == 0
            ? []
            : ValidatePositiveArray(options.ClassifierChannels, nameof(options.ClassifierChannels));
        _useDropout = options.UseDropout;
        _dropoutRate = options.DropoutRate;
        _learningRate = NumOps.FromDouble(options.LearningRate);

        _multiScaleRadii = options.MultiScaleRadii;
        _multiScaleMlpDimensions = options.MultiScaleMlpDimensions;
        _multiScaleNeighborSamples = options.MultiScaleNeighborSamples;

        if (_samplingRates.Length == 0)
        {
            throw new ArgumentException("At least one sampling rate is required.", nameof(options.SamplingRates));
        }
        if (_samplingRates.Length != _searchRadii.Length || _samplingRates.Length != _mlpDimensions.Length)
        {
            throw new ArgumentException("SamplingRates, SearchRadii, and MlpDimensions must have the same length.");
        }
        if (_neighborSamples.Length != _samplingRates.Length)
        {
            throw new ArgumentException("NeighborSamples must have the same length as SamplingRates.", nameof(options.NeighborSamples));
        }

        if (_useMultiScaleGrouping)
        {
            _multiScaleRadii = BuildMultiScaleRadii(_searchRadii, _multiScaleRadii);
            _multiScaleMlpDimensions = BuildMultiScaleMlpDimensions(_mlpDimensions, _multiScaleRadii, _multiScaleMlpDimensions);
            _multiScaleNeighborSamples = BuildMultiScaleNeighborSamples(_neighborSamples, _multiScaleRadii, _multiScaleNeighborSamples);
        }
        else
        {
            _multiScaleRadii = null;
            _multiScaleMlpDimensions = null;
            _multiScaleNeighborSamples = null;
        }

        _setAbstractionLayers = [];
        _classificationHeadLayers = [];
        InitializeLayers();
    }

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
        : this(new PointNetPlusPlusOptions
        {
            NumClasses = numClasses,
            SamplingRates = samplingRates,
            SearchRadii = searchRadii,
            MlpDimensions = mlpDimensions,
            UseMultiScaleGrouping = useMultiScaleGrouping
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
        _setAbstractionLayers.Clear();
        _classificationHeadLayers.Clear();

        int inputChannels = _inputFeatureDim;

        for (int i = 0; i < _samplingRates.Length; i++)
        {
            SetAbstractionLayer<T> saLayer;
            if (_useMultiScaleGrouping)
            {
                if (_multiScaleRadii == null || _multiScaleMlpDimensions == null || _multiScaleNeighborSamples == null)
                {
                    throw new InvalidOperationException("Multi-scale grouping configuration is missing.");
                }

                saLayer = new SetAbstractionLayer<T>(
                    numPoints: _samplingRates[i],
                    radii: _multiScaleRadii[i],
                    inputChannels: inputChannels,
                    mlpDimensions: _multiScaleMlpDimensions[i],
                    neighborSamples: _multiScaleNeighborSamples[i]);
            }
            else
            {
                saLayer = new SetAbstractionLayer<T>(
                    numPoints: _samplingRates[i],
                    searchRadius: _searchRadii[i],
                    inputChannels: inputChannels,
                    mlpDimensions: _mlpDimensions[i],
                    neighborSamples: _neighborSamples[i]);
            }

            _setAbstractionLayers.Add(saLayer);
            AddLayerToCollection(saLayer);
            inputChannels = saLayer.OutputChannels;
        }

        AddLayerToCollection(new AiDotNet.PointCloud.Layers.MaxPoolingLayer<T>(inputChannels));

        int classifierInput = inputChannels;
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

        Tensor<T> x = input;

        // Forward through all layers
        for (int i = 0; i < Layers.Count; i++)
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
        var output = Predict(pointCloud);
        return new Vector<T>(output.Data.ToArray());
    }

    public Tensor<T> SegmentPointCloud(Tensor<T> pointCloud)
    {
        if (pointCloud.Shape.Length != 2 || pointCloud.Shape[1] != _inputFeatureDim)
        {
            throw new ArgumentException($"Input must have shape [N, {_inputFeatureDim}].", nameof(pointCloud));
        }

        bool originalMode = IsTrainingMode;
        SetTrainingMode(false);

        try
        {
            Tensor<T> features = pointCloud;
            foreach (var layer in _setAbstractionLayers)
            {
                features = layer.Forward(features);
            }

            var lastLayer = _setAbstractionLayers.Count > 0 ? _setAbstractionLayers[^1] : null;
            if (lastLayer == null || lastLayer.LastCentroidPositions == null)
            {
                throw new InvalidOperationException("Centroid positions are not available for segmentation.");
            }

            var upsampled = UpsampleFeatures(pointCloud, lastLayer.LastCentroidPositions, features);
            Tensor<T> x = upsampled;
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

    private Tensor<T> UpsampleFeatures(
        Tensor<T> pointCloud,
        Tensor<T> centroidPositions,
        Tensor<T> centroidFeatures)
    {
        int numPoints = pointCloud.Shape[0];
        int numCentroids = centroidPositions.Shape[0];
        int featureDim = centroidFeatures.Shape[1];

        if (centroidPositions.Shape.Length != 2 || centroidPositions.Shape[1] != 3)
        {
            throw new ArgumentException("Centroid positions must have shape [M, 3].", nameof(centroidPositions));
        }
        if (centroidFeatures.Shape.Length != 2 || centroidFeatures.Shape[0] != numCentroids)
        {
            throw new ArgumentException("Centroid features must have shape [M, F].", nameof(centroidFeatures));
        }

        var pointData = pointCloud.Data.Span;
        var centroidPosData = centroidPositions.Data.Span;
        var centroidFeatureData = centroidFeatures.Data.Span;
        var output = new T[numPoints * featureDim];

        for (int i = 0; i < numPoints; i++)
        {
            double px = NumOps.ToDouble(pointData[i * _inputFeatureDim]);
            double py = NumOps.ToDouble(pointData[i * _inputFeatureDim + 1]);
            double pz = NumOps.ToDouble(pointData[i * _inputFeatureDim + 2]);

            double bestDist = double.PositiveInfinity;
            int bestIdx = 0;
            for (int c = 0; c < numCentroids; c++)
            {
                double cx = NumOps.ToDouble(centroidPosData[c * 3]);
                double cy = NumOps.ToDouble(centroidPosData[c * 3 + 1]);
                double cz = NumOps.ToDouble(centroidPosData[c * 3 + 2]);
                double dx = px - cx;
                double dy = py - cy;
                double dz = pz - cz;
                double dist = dx * dx + dy * dy + dz * dz;
                if (dist < bestDist)
                {
                    bestDist = dist;
                    bestIdx = c;
                }
            }

            int dstOffset = i * featureDim;
            int srcOffset = bestIdx * featureDim;
            for (int f = 0; f < featureDim; f++)
            {
                output[dstOffset + f] = centroidFeatureData[srcOffset + f];
            }
        }

        return new Tensor<T>(output, [numPoints, featureDim]);
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
                { "ModelName", "PointNetPlusPlus" },
                { "NumClasses", _numClasses },
                { "InputFeatureDim", _inputFeatureDim },
                { "SamplingRates", _samplingRates },
                { "SearchRadii", _searchRadii },
                { "NeighborSamples", _neighborSamples },
                { "UseMultiScaleGrouping", _useMultiScaleGrouping },
                { "MultiScaleRadii", _multiScaleRadii ?? Array.Empty<double[]>() },
                { "MultiScaleMlpDimensions", _multiScaleMlpDimensions ?? Array.Empty<int[][]>() },
                { "MultiScaleNeighborSamples", _multiScaleNeighborSamples ?? Array.Empty<int[]>() },
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
        writer.Write(_useMultiScaleGrouping);
        writer.Write(_useDropout);
        writer.Write(_dropoutRate);
        writer.Write(NumOps.ToDouble(_learningRate));

        WriteIntArray(writer, _samplingRates);
        WriteDoubleArray(writer, _searchRadii);
        WriteIntArray(writer, _neighborSamples);
        WriteIntJagged(writer, _mlpDimensions);
        WriteIntArray(writer, _classifierChannels);

        bool hasMultiScale = _multiScaleRadii != null && _multiScaleMlpDimensions != null && _multiScaleNeighborSamples != null;
        writer.Write(hasMultiScale);
        if (hasMultiScale)
        {
            var multiScaleRadii = _multiScaleRadii;
            var multiScaleMlpDimensions = _multiScaleMlpDimensions;
            var multiScaleNeighborSamples = _multiScaleNeighborSamples;
            if (multiScaleRadii == null || multiScaleMlpDimensions == null || multiScaleNeighborSamples == null)
            {
                throw new InvalidOperationException("Multi-scale configuration is missing.");
            }

            WriteDoubleJagged(writer, multiScaleRadii);
            WriteIntJagged3(writer, multiScaleMlpDimensions);
            WriteIntJagged(writer, multiScaleNeighborSamples);
        }
    }

    protected override void DeserializeNetworkSpecificData(BinaryReader reader)
    {
        _numClasses = reader.ReadInt32();
        _inputFeatureDim = reader.ReadInt32();
        _useMultiScaleGrouping = reader.ReadBoolean();
        _useDropout = reader.ReadBoolean();
        _dropoutRate = reader.ReadDouble();
        _learningRate = NumOps.FromDouble(reader.ReadDouble());

        _samplingRates = ReadIntArray(reader, nameof(_samplingRates), allowEmpty: false);
        _searchRadii = ReadDoubleArray(reader, nameof(_searchRadii), allowEmpty: false);
        _neighborSamples = ReadIntArray(reader, nameof(_neighborSamples), allowEmpty: false);
        _mlpDimensions = ReadIntJagged(reader, nameof(_mlpDimensions), allowEmpty: false);
        _classifierChannels = ReadIntArray(reader, nameof(_classifierChannels), allowEmpty: true);

        bool hasMultiScale = reader.ReadBoolean();
        if (hasMultiScale)
        {
            _multiScaleRadii = ReadDoubleJagged(reader, nameof(_multiScaleRadii));
            _multiScaleMlpDimensions = ReadIntJagged3(reader, nameof(_multiScaleMlpDimensions));
            _multiScaleNeighborSamples = ReadIntJagged(reader, nameof(_multiScaleNeighborSamples), allowEmpty: false);
            if (_multiScaleRadii == null || _multiScaleMlpDimensions == null || _multiScaleNeighborSamples == null)
            {
                throw new InvalidOperationException("Serialized multi-scale configuration is incomplete.");
            }
        }
        else
        {
            _multiScaleRadii = null;
            _multiScaleMlpDimensions = null;
            _multiScaleNeighborSamples = null;
        }

        _setAbstractionLayers.Clear();
        _classificationHeadLayers.Clear();
        bool afterPooling = false;
        foreach (var layer in Layers)
        {
            if (layer is SetAbstractionLayer<T> saLayer)
            {
                _setAbstractionLayers.Add(saLayer);
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
        return new PointNetPlusPlus<T>(
            new PointNetPlusPlusOptions
            {
                NumClasses = _numClasses,
                InputFeatureDim = _inputFeatureDim,
                SamplingRates = _samplingRates,
                SearchRadii = _searchRadii,
                NeighborSamples = _neighborSamples,
                MlpDimensions = _mlpDimensions,
                UseMultiScaleGrouping = _useMultiScaleGrouping,
                MultiScaleRadii = _multiScaleRadii,
                MultiScaleMlpDimensions = _multiScaleMlpDimensions,
                MultiScaleNeighborSamples = _multiScaleNeighborSamples,
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

    private static double[] ValidatePositiveArray(double[]? values, string paramName)
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
            if (values[i] <= 0.0)
            {
                throw new ArgumentOutOfRangeException(paramName, "Values must be positive.");
            }
        }

        return values;
    }

    private static int[][] ValidateMlpDimensions(int[][]? values, string paramName)
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
            if (values[i] == null || values[i].Length == 0)
            {
                throw new ArgumentException("Each MLP dimension array must be non-empty.", paramName);
            }
            ValidatePositiveArray(values[i], paramName);
        }

        return values;
    }

    private static double[][] BuildMultiScaleRadii(double[] baseRadii, double[][]? radii)
    {
        if (radii == null || radii.Length == 0)
        {
            var generated = new double[baseRadii.Length][];
            for (int i = 0; i < baseRadii.Length; i++)
            {
                generated[i] = new[] { baseRadii[i], baseRadii[i] * 2.0 };
            }
            return generated;
        }
        if (radii.Length != baseRadii.Length)
        {
            throw new ArgumentException("MultiScaleRadii must match the number of abstraction layers.");
        }
        for (int i = 0; i < radii.Length; i++)
        {
            if (radii[i] == null || radii[i].Length == 0)
            {
                throw new ArgumentException("Each MultiScaleRadii entry must be non-empty.");
            }
            for (int j = 0; j < radii[i].Length; j++)
            {
                if (radii[i][j] <= 0.0)
                {
                    throw new ArgumentOutOfRangeException(nameof(radii), "Multi-scale radii must be positive.");
                }
            }
        }

        return radii;
    }

    private static int[][][] BuildMultiScaleMlpDimensions(
        int[][] baseDims,
        double[][] radii,
        int[][][]? multiScaleMlp)
    {
        if (multiScaleMlp == null || multiScaleMlp.Length == 0)
        {
            var generated = new int[baseDims.Length][][];
            for (int i = 0; i < baseDims.Length; i++)
            {
                generated[i] = new int[radii[i].Length][];
                for (int s = 0; s < radii[i].Length; s++)
                {
                    generated[i][s] = baseDims[i];
                }
            }
            return generated;
        }

        if (multiScaleMlp.Length != baseDims.Length)
        {
            throw new ArgumentException("MultiScaleMlpDimensions must match the number of abstraction layers.");
        }
        for (int i = 0; i < multiScaleMlp.Length; i++)
        {
            if (multiScaleMlp[i] == null || multiScaleMlp[i].Length != radii[i].Length)
            {
                throw new ArgumentException("MultiScaleMlpDimensions must match the number of scales per layer.");
            }
            for (int s = 0; s < multiScaleMlp[i].Length; s++)
            {
                if (multiScaleMlp[i][s] == null || multiScaleMlp[i][s].Length == 0)
                {
                    throw new ArgumentException("Each multi-scale MLP dimension array must be non-empty.");
                }
                ValidatePositiveArray(multiScaleMlp[i][s], nameof(multiScaleMlp));
            }
        }

        return multiScaleMlp;
    }

    private static int[][] BuildMultiScaleNeighborSamples(
        int[] baseSamples,
        double[][] radii,
        int[][]? multiScaleSamples)
    {
        if (multiScaleSamples == null || multiScaleSamples.Length == 0)
        {
            var generated = new int[baseSamples.Length][];
            for (int i = 0; i < baseSamples.Length; i++)
            {
                generated[i] = new int[radii[i].Length];
                for (int s = 0; s < radii[i].Length; s++)
                {
                    generated[i][s] = baseSamples[i];
                }
            }
            return generated;
        }

        if (multiScaleSamples.Length != baseSamples.Length)
        {
            throw new ArgumentException("MultiScaleNeighborSamples must match the number of abstraction layers.");
        }
        for (int i = 0; i < multiScaleSamples.Length; i++)
        {
            if (multiScaleSamples[i] == null || multiScaleSamples[i].Length != radii[i].Length)
            {
                throw new ArgumentException("MultiScaleNeighborSamples must match the number of scales per layer.");
            }
            ValidatePositiveArray(multiScaleSamples[i], nameof(multiScaleSamples));
        }

        return multiScaleSamples;
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

    private static void WriteDoubleArray(BinaryWriter writer, double[] values)
    {
        writer.Write(values.Length);
        for (int i = 0; i < values.Length; i++)
        {
            writer.Write(values[i]);
        }
    }

    private static double[] ReadDoubleArray(BinaryReader reader, string paramName, bool allowEmpty)
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
        var values = new double[length];
        for (int i = 0; i < length; i++)
        {
            values[i] = reader.ReadDouble();
        }

        if (length == 0)
        {
            return values;
        }

        return ValidatePositiveArray(values, paramName);
    }

    private static void WriteIntJagged(BinaryWriter writer, int[][] values)
    {
        writer.Write(values.Length);
        for (int i = 0; i < values.Length; i++)
        {
            WriteIntArray(writer, values[i]);
        }
    }

    private static int[][] ReadIntJagged(BinaryReader reader, string paramName, bool allowEmpty)
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
        var values = new int[length][];
        for (int i = 0; i < length; i++)
        {
            values[i] = ReadIntArray(reader, paramName, allowEmpty: false);
        }

        return values;
    }

    private static void WriteDoubleJagged(BinaryWriter writer, double[][] values)
    {
        writer.Write(values.Length);
        for (int i = 0; i < values.Length; i++)
        {
            WriteDoubleArray(writer, values[i]);
        }
    }

    private static double[][]? ReadDoubleJagged(BinaryReader reader, string paramName)
    {
        int length = reader.ReadInt32();
        if (length <= 0)
        {
            return null;
        }
        var values = new double[length][];
        for (int i = 0; i < length; i++)
        {
            values[i] = ReadDoubleArray(reader, paramName, allowEmpty: false);
        }

        return values;
    }

    private static void WriteIntJagged3(BinaryWriter writer, int[][][] values)
    {
        writer.Write(values.Length);
        for (int i = 0; i < values.Length; i++)
        {
            WriteIntJagged(writer, values[i]);
        }
    }

    private static int[][][]? ReadIntJagged3(BinaryReader reader, string paramName)
    {
        int length = reader.ReadInt32();
        if (length <= 0)
        {
            return null;
        }
        var values = new int[length][][];
        for (int i = 0; i < length; i++)
        {
            values[i] = ReadIntJagged(reader, paramName, allowEmpty: false);
        }

        return values;
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
    private sealed class ScaleBranch
    {
        private readonly int[] _mlpDimensions;

        public ScaleBranch(double radius, int neighborSamples, int inputChannels, int[] mlpDimensions)
        {
            if (radius <= 0.0)
            {
                throw new ArgumentOutOfRangeException(nameof(radius), "Search radius must be positive.");
            }
            if (neighborSamples <= 0)
            {
                throw new ArgumentOutOfRangeException(nameof(neighborSamples), "NeighborSamples must be positive.");
            }

            _mlpDimensions = ValidateChannelArray(mlpDimensions, nameof(mlpDimensions));
            Radius = radius;
            NeighborSamples = neighborSamples;
            MlpLayers = [];

            int inChannels = inputChannels;
            foreach (var outChannels in _mlpDimensions)
            {
                MlpLayers.Add(new PointConvolutionLayer<T>(inChannels, outChannels, new ReLUActivation<T>()));
                inChannels = outChannels;
            }
        }

        public double Radius { get; }
        public int NeighborSamples { get; }
        public List<ILayer<T>> MlpLayers { get; }
        public int OutputChannels => _mlpDimensions[^1];
        public int[]? NeighborCounts { get; set; }
        public int[,]? NeighborIndices { get; set; }
        public int[]? MaxIndices { get; set; }
    }

    private readonly int _numPoints;
    private readonly int _inputChannels;
    private readonly List<ScaleBranch> _branches;
    private Tensor<T>? _lastInput;
    private int[]? _centroidIndices;
    private Tensor<T>? _lastCentroidPositions;
    private readonly int _outputChannels;

    public Tensor<T>? LastCentroidPositions => _lastCentroidPositions;
    public int OutputChannels => _outputChannels;

    public SetAbstractionLayer(
        int numPoints,
        double searchRadius,
        int inputChannels,
        int[] mlpDimensions,
        int neighborSamples)
        : base([0, inputChannels], [0, mlpDimensions[^1]])
    {
        if (numPoints <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(numPoints), "Number of points must be positive.");
        }
        if (inputChannels < 3)
        {
            throw new ArgumentOutOfRangeException(nameof(inputChannels), "InputChannels must be at least 3.");
        }

        _numPoints = numPoints;
        _inputChannels = inputChannels;
        _branches = [new ScaleBranch(searchRadius, neighborSamples, inputChannels, mlpDimensions)];
        _outputChannels = _branches[0].OutputChannels;

        Parameters = GetParameters();
    }

    public SetAbstractionLayer(
        int numPoints,
        double[] radii,
        int inputChannels,
        int[][] mlpDimensions,
        int[] neighborSamples)
        : base([0, inputChannels], [0, CalculateOutputChannels(mlpDimensions)])
    {
        if (numPoints <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(numPoints), "Number of points must be positive.");
        }
        if (inputChannels < 3)
        {
            throw new ArgumentOutOfRangeException(nameof(inputChannels), "InputChannels must be at least 3.");
        }
        if (radii == null || radii.Length == 0)
        {
            throw new ArgumentException("Radii must not be empty.", nameof(radii));
        }
        if (mlpDimensions == null || mlpDimensions.Length != radii.Length)
        {
            throw new ArgumentException("MlpDimensions must match the number of radii.", nameof(mlpDimensions));
        }
        if (neighborSamples == null || neighborSamples.Length != radii.Length)
        {
            throw new ArgumentException("NeighborSamples must match the number of radii.", nameof(neighborSamples));
        }

        _numPoints = numPoints;
        _inputChannels = inputChannels;
        _branches = [];

        for (int i = 0; i < radii.Length; i++)
        {
            _branches.Add(new ScaleBranch(radii[i], neighborSamples[i], inputChannels, mlpDimensions[i]));
        }

        _outputChannels = _branches.Sum(branch => branch.OutputChannels);
        Parameters = GetParameters();
    }

    public override Tensor<T> Forward(Tensor<T> input)
    {
        if (input.Shape.Length != 2 || input.Shape[1] != _inputChannels)
        {
            throw new ArgumentException($"Input must have shape [N, {_inputChannels}].", nameof(input));
        }

        _lastInput = input;
        int numPoints = input.Shape[0];
        if (numPoints == 0)
        {
            return new Tensor<T>([], [0, _outputChannels]);
        }

        int numCentroids = Math.Min(_numPoints, numPoints);
        var positions = ExtractPositions(input);
        _centroidIndices = FarthestPointSampling(positions, numCentroids);
        _lastCentroidPositions = BuildCentroidPositions(input, _centroidIndices);

        var outputs = new List<Tensor<T>>();
        foreach (var branch in _branches)
        {
            outputs.Add(ProcessBranch(branch, input, positions, _centroidIndices));
        }

        if (outputs.Count == 1)
        {
            return outputs[0];
        }

        var combined = new T[numCentroids * _outputChannels];
        int channelOffset = 0;
        foreach (var branchOutput in outputs)
        {
            int branchChannels = branchOutput.Shape[1];
            for (int c = 0; c < numCentroids; c++)
            {
                int dstBase = c * _outputChannels + channelOffset;
                int srcBase = c * branchChannels;
                for (int ch = 0; ch < branchChannels; ch++)
                {
                    combined[dstBase + ch] = branchOutput.Data.Span[srcBase + ch];
                }
            }

            channelOffset += branchChannels;
        }

        return new Tensor<T>(combined, [numCentroids, _outputChannels]);
    }

    public override Tensor<T> Backward(Tensor<T> outputGradient)
    {
        if (_lastInput == null || _centroidIndices == null)
        {
            throw new InvalidOperationException("Forward pass must be called before backward pass.");
        }
        if (outputGradient.Shape.Length != 2 || outputGradient.Shape[1] != _outputChannels)
        {
            throw new ArgumentException($"Output gradient must have shape [M, {_outputChannels}].", nameof(outputGradient));
        }

        int numPoints = _lastInput.Shape[0];
        int numCentroids = _centroidIndices.Length;
        var numOps = NumOps;
        var inputGradient = new T[numPoints * _inputChannels];
        for (int i = 0; i < inputGradient.Length; i++)
        {
            inputGradient[i] = numOps.Zero;
        }

        int channelOffset = 0;
        foreach (var branch in _branches)
        {
            if (branch.NeighborCounts == null || branch.NeighborIndices == null || branch.MaxIndices == null)
            {
                throw new InvalidOperationException("Branch metadata missing for backward pass.");
            }

            int branchChannels = branch.OutputChannels;
            int maxNeighbors = branch.NeighborIndices.GetLength(1);
            var branchGradient = new T[numCentroids * maxNeighbors * branchChannels];

            for (int c = 0; c < numCentroids; c++)
            {
                for (int ch = 0; ch < branchChannels; ch++)
                {
                    int maxIdx = branch.MaxIndices[c * branchChannels + ch];
                    int gradIdx = (c * maxNeighbors + maxIdx) * branchChannels + ch;
                    int srcIdx = c * _outputChannels + channelOffset + ch;
                    branchGradient[gradIdx] = outputGradient.Data.Span[srcIdx];
                }
            }

            Tensor<T> gradTensor = new Tensor<T>(branchGradient, [numCentroids * maxNeighbors, branchChannels]);
            for (int i = branch.MlpLayers.Count - 1; i >= 0; i--)
            {
                gradTensor = branch.MlpLayers[i].Backward(gradTensor);
            }

            var gradData = gradTensor.Data.Span;
            for (int c = 0; c < numCentroids; c++)
            {
                int centroidIdx = _centroidIndices[c];
                int count = branch.NeighborCounts[c];
                for (int k = 0; k < count; k++)
                {
                    int neighborIdx = branch.NeighborIndices[c, k];
                    int baseIdx = (c * maxNeighbors + k) * _inputChannels;
                    for (int d = 0; d < 3; d++)
                    {
                        var gradVal = gradData[baseIdx + d];
                        int neighborOffset = neighborIdx * _inputChannels + d;
                        int centroidOffset = centroidIdx * _inputChannels + d;
                        inputGradient[neighborOffset] = numOps.Add(inputGradient[neighborOffset], gradVal);
                        inputGradient[centroidOffset] = numOps.Subtract(inputGradient[centroidOffset], gradVal);
                    }
                    for (int d = 3; d < _inputChannels; d++)
                    {
                        int neighborOffset = neighborIdx * _inputChannels + d;
                        inputGradient[neighborOffset] = numOps.Add(inputGradient[neighborOffset], gradData[baseIdx + d]);
                    }
                }
            }

            channelOffset += branchChannels;
        }

        return new Tensor<T>(inputGradient, [numPoints, _inputChannels]);
    }

    public override void UpdateParameters(T learningRate)
    {
        foreach (var branch in _branches)
        {
            foreach (var layer in branch.MlpLayers)
            {
                layer.UpdateParameters(learningRate);
            }
        }
    }

    public override void ClearGradients()
    {
        foreach (var branch in _branches)
        {
            foreach (var layer in branch.MlpLayers)
            {
                layer.ClearGradients();
            }
        }
    }

    public override Vector<T> GetParameters()
    {
        int totalParams = ParameterCount;
        var parameters = new Vector<T>(totalParams);
        int offset = 0;

        foreach (var branch in _branches)
        {
            foreach (var layer in branch.MlpLayers)
            {
                var layerParameters = layer.GetParameters();
                for (int i = 0; i < layerParameters.Length; i++)
                {
                    parameters[offset + i] = layerParameters[i];
                }

                offset += layerParameters.Length;
            }
        }

        Parameters = parameters;
        return parameters;
    }

    public override void UpdateParameters(Vector<T> parameters)
    {
        int offset = 0;
        foreach (var branch in _branches)
        {
            foreach (var layer in branch.MlpLayers)
            {
                int layerParameterCount = layer.ParameterCount;
                if (layerParameterCount > 0)
                {
                    var layerParameters = parameters.SubVector(offset, layerParameterCount);
                    layer.UpdateParameters(layerParameters);
                    offset += layerParameterCount;
                }
            }
        }

        Parameters = parameters;
    }

    public override void ResetState()
    {
        _lastInput = null;
        _centroidIndices = null;
        _lastCentroidPositions = null;

        foreach (var branch in _branches)
        {
            foreach (var layer in branch.MlpLayers)
            {
                layer.ResetState();
            }
            branch.NeighborCounts = null;
            branch.NeighborIndices = null;
            branch.MaxIndices = null;
        }
    }

    public override bool SupportsJitCompilation => false;

    public override ComputationNode<T> ExportComputationGraph(List<ComputationNode<T>> inputNodes)
    {
        throw new NotSupportedException(
            "SetAbstractionLayer does not support computation graph export due to point cloud-specific operations.");
    }

    public override int ParameterCount
    {
        get
        {
            int total = 0;
            foreach (var branch in _branches)
            {
                foreach (var layer in branch.MlpLayers)
                {
                    total += layer.ParameterCount;
                }
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

    private static int CalculateOutputChannels(int[][] mlpDimensions)
    {
        int total = 0;
        for (int i = 0; i < mlpDimensions.Length; i++)
        {
            total += mlpDimensions[i][^1];
        }
        return total;
    }

    private double[] ExtractPositions(Tensor<T> input)
    {
        int numPoints = input.Shape[0];
        var positions = new double[numPoints * 3];
        var data = input.Data.Span;

        for (int i = 0; i < numPoints; i++)
        {
            int baseIdx = i * _inputChannels;
            positions[i * 3] = NumOps.ToDouble(data[baseIdx]);
            positions[i * 3 + 1] = NumOps.ToDouble(data[baseIdx + 1]);
            positions[i * 3 + 2] = NumOps.ToDouble(data[baseIdx + 2]);
        }

        return positions;
    }

    private Tensor<T> BuildCentroidPositions(Tensor<T> input, int[] centroidIndices)
    {
        var data = input.Data.Span;
        var centroidData = new T[centroidIndices.Length * 3];
        for (int i = 0; i < centroidIndices.Length; i++)
        {
            int pointIdx = centroidIndices[i];
            int srcBase = pointIdx * _inputChannels;
            int dstBase = i * 3;
            centroidData[dstBase] = data[srcBase];
            centroidData[dstBase + 1] = data[srcBase + 1];
            centroidData[dstBase + 2] = data[srcBase + 2];
        }

        return new Tensor<T>(centroidData, [centroidIndices.Length, 3]);
    }

    private Tensor<T> ProcessBranch(
        ScaleBranch branch,
        Tensor<T> input,
        double[] positions,
        int[] centroidIndices)
    {
        int numCentroids = centroidIndices.Length;
        int maxNeighbors = Math.Min(branch.NeighborSamples, input.Shape[0]);
        if (maxNeighbors <= 0)
        {
            maxNeighbors = 1;
        }

        BuildNeighborIndices(positions, centroidIndices, branch.Radius, maxNeighbors, out int[,] neighborIndices, out int[] neighborCounts);
        branch.NeighborIndices = neighborIndices;
        branch.NeighborCounts = neighborCounts;

        var grouped = BuildGroupedFeatures(input, positions, centroidIndices, neighborIndices, neighborCounts, maxNeighbors);
        Tensor<T> features = grouped;
        foreach (var layer in branch.MlpLayers)
        {
            features = layer.Forward(features);
        }

        int outChannels = branch.OutputChannels;
        var output = new T[numCentroids * outChannels];
        var maxIndices = new int[numCentroids * outChannels];
        var data = features.Data.Span;
        var numOps = NumOps;

        for (int c = 0; c < numCentroids; c++)
        {
            int count = neighborCounts[c];
            for (int ch = 0; ch < outChannels; ch++)
            {
                int baseIdx = (c * maxNeighbors) * outChannels + ch;
                T maxVal = data[baseIdx];
                int maxIdx = 0;
                for (int k = 1; k < count; k++)
                {
                    int idx = (c * maxNeighbors + k) * outChannels + ch;
                    T val = data[idx];
                    if (numOps.GreaterThan(val, maxVal))
                    {
                        maxVal = val;
                        maxIdx = k;
                    }
                }
                output[c * outChannels + ch] = maxVal;
                maxIndices[c * outChannels + ch] = maxIdx;
            }
        }

        branch.MaxIndices = maxIndices;
        return new Tensor<T>(output, [numCentroids, outChannels]);
    }

    private Tensor<T> BuildGroupedFeatures(
        Tensor<T> input,
        double[] positions,
        int[] centroidIndices,
        int[,] neighborIndices,
        int[] neighborCounts,
        int maxNeighbors)
    {
        int numCentroids = centroidIndices.Length;
        var grouped = new T[numCentroids * maxNeighbors * _inputChannels];
        var data = input.Data.Span;
        var numOps = NumOps;

        for (int c = 0; c < numCentroids; c++)
        {
            int centroidIdx = centroidIndices[c];
            double cx = positions[centroidIdx * 3];
            double cy = positions[centroidIdx * 3 + 1];
            double cz = positions[centroidIdx * 3 + 2];
            int count = neighborCounts[c];

            for (int k = 0; k < maxNeighbors; k++)
            {
                int neighborIdx = neighborIndices[c, k];
                int baseIdx = (c * maxNeighbors + k) * _inputChannels;

                if (k < count)
                {
                    double nx = positions[neighborIdx * 3];
                    double ny = positions[neighborIdx * 3 + 1];
                    double nz = positions[neighborIdx * 3 + 2];
                    grouped[baseIdx] = numOps.FromDouble(nx - cx);
                    grouped[baseIdx + 1] = numOps.FromDouble(ny - cy);
                    grouped[baseIdx + 2] = numOps.FromDouble(nz - cz);

                    int featureBase = neighborIdx * _inputChannels;
                    for (int d = 3; d < _inputChannels; d++)
                    {
                        grouped[baseIdx + d] = data[featureBase + d];
                    }
                }
                else
                {
                    for (int d = 0; d < _inputChannels; d++)
                    {
                        grouped[baseIdx + d] = numOps.Zero;
                    }
                }
            }
        }

        return new Tensor<T>(grouped, [numCentroids * maxNeighbors, _inputChannels]);
    }

    private static void BuildNeighborIndices(
        double[] positions,
        int[] centroidIndices,
        double radius,
        int maxNeighbors,
        out int[,] neighborIndices,
        out int[] neighborCounts)
    {
        int numCentroids = centroidIndices.Length;
        int numPoints = positions.Length / 3;
        neighborIndices = new int[numCentroids, maxNeighbors];
        neighborCounts = new int[numCentroids];
        double radiusSq = radius * radius;

        for (int c = 0; c < numCentroids; c++)
        {
            int centroidIdx = centroidIndices[c];
            double cx = positions[centroidIdx * 3];
            double cy = positions[centroidIdx * 3 + 1];
            double cz = positions[centroidIdx * 3 + 2];
            var neighbors = new List<(double dist, int idx)>();

            for (int p = 0; p < numPoints; p++)
            {
                double dx = positions[p * 3] - cx;
                double dy = positions[p * 3 + 1] - cy;
                double dz = positions[p * 3 + 2] - cz;
                double dist = dx * dx + dy * dy + dz * dz;
                if (dist <= radiusSq)
                {
                    neighbors.Add((dist, p));
                }
            }

            if (neighbors.Count == 0)
            {
                neighbors.Add((0.0, centroidIdx));
            }

            if (neighbors.Count > maxNeighbors)
            {
                neighbors = neighbors.OrderBy(n => n.dist).Take(maxNeighbors).ToList();
            }

            neighborCounts[c] = neighbors.Count;
            for (int k = 0; k < maxNeighbors; k++)
            {
                neighborIndices[c, k] = k < neighbors.Count ? neighbors[k].idx : centroidIdx;
            }
        }
    }

    private static int[] FarthestPointSampling(double[] positions, int numCentroids)
    {
        int numPoints = positions.Length / 3;
        numCentroids = Math.Min(numCentroids, numPoints);
        var centroids = new int[numCentroids];
        var distances = new double[numPoints];
        for (int i = 0; i < distances.Length; i++)
        {
            distances[i] = double.PositiveInfinity;
        }

        int farthest = 0;
        for (int i = 0; i < numCentroids; i++)
        {
            centroids[i] = farthest;
            double cx = positions[farthest * 3];
            double cy = positions[farthest * 3 + 1];
            double cz = positions[farthest * 3 + 2];

            for (int p = 0; p < numPoints; p++)
            {
                double dx = positions[p * 3] - cx;
                double dy = positions[p * 3 + 1] - cy;
                double dz = positions[p * 3 + 2] - cz;
                double dist = dx * dx + dy * dy + dz * dz;
                if (dist < distances[p])
                {
                    distances[p] = dist;
                }
            }

            farthest = 0;
            double maxDist = distances[0];
            for (int p = 1; p < numPoints; p++)
            {
                if (distances[p] > maxDist)
                {
                    maxDist = distances[p];
                    farthest = p;
                }
            }
        }

        return centroids;
    }
}
