using System.Collections.Generic;
using AiDotNet.ActivationFunctions;
using AiDotNet.Interfaces;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.PointCloud.Interfaces;
using AiDotNet.PointCloud.Layers;
using PointNetModelOptions = AiDotNet.PointCloud.Options.PointNetOptions;

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
    private readonly PointNetModelOptions _options;

    /// <inheritdoc/>
    public override ModelOptions GetOptions() => _options;

    private int _numClasses;
    private int _inputFeatureDim;
    private int _inputTransformDim;
    private bool _useInputTransform;
    private bool _useFeatureTransform;
    private bool _useDropout;
    private double _dropoutRate;
    private T _learningRate;
    private int[] _inputMlpChannels;
    private int[] _featureMlpChannels;
    private int[] _classifierChannels;
    private int[] _inputTransformMlpChannels;
    private int[] _inputTransformFcChannels;
    private int[] _featureTransformMlpChannels;
    private int[] _featureTransformFcChannels;
    private Vector<T>? _globalFeatures;
    private readonly List<ILayer<T>> _classificationHeadLayers;

    /// <summary>
    /// Initializes a new instance of the PointNet class with default options.
    /// </summary>
    public PointNet()
        : this(new PointNetOptions(), null)
    {
    }

    /// <summary>
    /// Initializes a new instance of the PointNet class with configurable options.
    /// </summary>
    /// <param name="options">Configuration options for the PointNet model.</param>
    /// <param name="lossFunction">Optional loss function for training.</param>
    public PointNet(PointNetOptions options, ILossFunction<T>? lossFunction = null, PointNetModelOptions? modelOptions = null)
        : base(CreateArchitecture(options.NumClasses, options.InputFeatureDim), lossFunction ?? NeuralNetworkHelper<T>.GetDefaultLossFunction(NeuralNetworkTaskType.MultiClassClassification))
    {
        _options = modelOptions ?? new PointNetModelOptions();
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
        if (options.InputTransformDim <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(options.InputTransformDim), "InputTransformDim must be positive.");
        }
        if (options.InputTransformDim > options.InputFeatureDim)
        {
            throw new ArgumentOutOfRangeException(nameof(options.InputTransformDim), "InputTransformDim must be <= InputFeatureDim.");
        }
        if (options.DropoutRate < 0.0 || options.DropoutRate >= 1.0)
        {
            throw new ArgumentOutOfRangeException(nameof(options.DropoutRate), "DropoutRate must be in [0, 1).");
        }

        _numClasses = options.NumClasses;
        _inputFeatureDim = options.InputFeatureDim;
        _inputTransformDim = options.InputTransformDim;
        _useInputTransform = options.UseInputTransform;
        _useFeatureTransform = options.UseFeatureTransform;
        _useDropout = options.UseDropout;
        _dropoutRate = options.DropoutRate;
        _learningRate = NumOps.FromDouble(options.LearningRate);

        _inputMlpChannels = ValidateChannelArray(options.InputMlpChannels, nameof(options.InputMlpChannels));
        _featureMlpChannels = ValidateChannelArray(options.FeatureMlpChannels, nameof(options.FeatureMlpChannels));
        _classifierChannels = options.ClassifierChannels == null ? [] : ValidateNonNegativeArray(options.ClassifierChannels, nameof(options.ClassifierChannels));
        _inputTransformMlpChannels = ValidateChannelArray(options.InputTransformMlpChannels, nameof(options.InputTransformMlpChannels));
        _inputTransformFcChannels = ValidateChannelArray(options.InputTransformFcChannels, nameof(options.InputTransformFcChannels));
        _featureTransformMlpChannels = ValidateChannelArray(options.FeatureTransformMlpChannels, nameof(options.FeatureTransformMlpChannels));
        _featureTransformFcChannels = ValidateChannelArray(options.FeatureTransformFcChannels, nameof(options.FeatureTransformFcChannels));

        _classificationHeadLayers = [];
        InitializeLayers();
    }

    /// <summary>
    /// Initializes a new instance of the PointNet class.
    /// </summary>
    /// <param name="numClasses">Number of output classes for classification.</param>
    /// <param name="useInputTransform">Whether to use input transformation network (T-Net).</param>
    /// <param name="useFeatureTransform">Whether to use feature transformation network.</param>
    /// <param name="lossFunction">Optional loss function for training.</param>
    /// <remarks>
    /// <b>For Beginners:</b> Creates a PointNet model for point cloud classification.
    /// </remarks>
    public PointNet(
        int numClasses,
        bool useInputTransform = true,
        bool useFeatureTransform = true,
        ILossFunction<T>? lossFunction = null)
        : this(new PointNetOptions
        {
            NumClasses = numClasses,
            UseInputTransform = useInputTransform,
            UseFeatureTransform = useFeatureTransform
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
        _classificationHeadLayers.Clear();

        int inputChannels = _inputFeatureDim;

        if (_useInputTransform)
        {
            AddLayerToCollection(new TNetLayer<T>(
                _inputTransformDim,
                inputChannels,
                _inputTransformMlpChannels,
                _inputTransformFcChannels));
        }

        foreach (var outChannels in _inputMlpChannels)
        {
            AddLayerToCollection(new PointConvolutionLayer<T>(
                inputChannels,
                outChannels,
                new ReLUActivation<T>()));
            inputChannels = outChannels;
        }

        if (_useFeatureTransform)
        {
            AddLayerToCollection(new TNetLayer<T>(
                inputChannels,
                inputChannels,
                _featureTransformMlpChannels,
                _featureTransformFcChannels));
        }

        foreach (var outChannels in _featureMlpChannels)
        {
            AddLayerToCollection(new PointConvolutionLayer<T>(
                inputChannels,
                outChannels,
                new ReLUActivation<T>()));
            inputChannels = outChannels;
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

        // Process through all layers
        for (int i = 0; i < Layers.Count; i++)
        {
            _layerInputs[i] = x;
            x = Layers[i].Forward(x);
            _layerOutputs[i] = x;

            // Capture global features after max pooling
            if (Layers[i] is AiDotNet.PointCloud.Layers.MaxPoolingLayer<T>)
            {
                // Convert tensor to vector for global features
                _globalFeatures = new Vector<T>(x.Data.ToArray());
            }
        }

        return x;
    }

    public override Tensor<T> Backpropagate(Tensor<T> outputGradient)
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
        bool originalMode = IsTrainingMode;
        SetTrainingMode(false);

        try
        {
            _ = ForwardWithMemory(pointCloud);

            if (_globalFeatures == null)
            {
                throw new InvalidOperationException("Global features not extracted. Ensure forward pass completed.");
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

        Tensor<T> x = pointCloud;

        // Process through layers up to (but not including) max pooling
        for (int i = 0; i < Layers.Count; i++)
        {
            if (Layers[i] is AiDotNet.PointCloud.Layers.MaxPoolingLayer<T>)
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
        var output = Predict(pointCloud);

        // Output should be [1, numClasses], convert to vector
        return new Vector<T>(output.Data.ToArray());
    }

    public override void Train(Tensor<T> input, Tensor<T> expectedOutput)
    {
        SetTrainingMode(true);

        // Reset global features
        _globalFeatures = null;

        // Forward pass
        var prediction = ForwardWithMemory(input);

        // Compute loss
        if (LossFunction == null)
        {
            throw new InvalidOperationException("Loss function not set for training.");
        }

        var loss = LossFunction.ComputeLoss(prediction, expectedOutput);
        LastLoss = loss;

        // Backward pass
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
        var output = ForwardWithMemory(input);
        return output;
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
                { "ModelName", "PointNet" },
                { "NumClasses", _numClasses },
                { "InputFeatureDim", _inputFeatureDim },
                { "InputTransformDim", _inputTransformDim },
                { "UseInputTransform", _useInputTransform },
                { "UseFeatureTransform", _useFeatureTransform },
                { "InputMlpChannels", _inputMlpChannels },
                { "FeatureMlpChannels", _featureMlpChannels },
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
        writer.Write(_inputTransformDim);
        writer.Write(_useInputTransform);
        writer.Write(_useFeatureTransform);
        writer.Write(_useDropout);
        writer.Write(_dropoutRate);
        writer.Write(NumOps.ToDouble(_learningRate));
        WriteIntArray(writer, _inputMlpChannels);
        WriteIntArray(writer, _featureMlpChannels);
        WriteIntArray(writer, _classifierChannels);
        WriteIntArray(writer, _inputTransformMlpChannels);
        WriteIntArray(writer, _inputTransformFcChannels);
        WriteIntArray(writer, _featureTransformMlpChannels);
        WriteIntArray(writer, _featureTransformFcChannels);
    }

    protected override void DeserializeNetworkSpecificData(BinaryReader reader)
    {
        _numClasses = reader.ReadInt32();
        _inputFeatureDim = reader.ReadInt32();
        _inputTransformDim = reader.ReadInt32();
        _useInputTransform = reader.ReadBoolean();
        _useFeatureTransform = reader.ReadBoolean();
        _useDropout = reader.ReadBoolean();
        _dropoutRate = reader.ReadDouble();
        _learningRate = NumOps.FromDouble(reader.ReadDouble());
        _inputMlpChannels = ReadIntArray(reader, nameof(_inputMlpChannels), allowEmpty: false);
        _featureMlpChannels = ReadIntArray(reader, nameof(_featureMlpChannels), allowEmpty: false);
        _classifierChannels = ReadIntArray(reader, nameof(_classifierChannels), allowEmpty: true);
        _inputTransformMlpChannels = ReadIntArray(reader, nameof(_inputTransformMlpChannels), allowEmpty: false);
        _inputTransformFcChannels = ReadIntArray(reader, nameof(_inputTransformFcChannels), allowEmpty: false);
        _featureTransformMlpChannels = ReadIntArray(reader, nameof(_featureTransformMlpChannels), allowEmpty: false);
        _featureTransformFcChannels = ReadIntArray(reader, nameof(_featureTransformFcChannels), allowEmpty: false);

        _classificationHeadLayers.Clear();
        bool afterPooling = false;
        foreach (var layer in Layers)
        {
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
        return new PointNet<T>(
            new PointNetOptions
            {
                NumClasses = _numClasses,
                InputFeatureDim = _inputFeatureDim,
                InputTransformDim = _inputTransformDim,
                UseInputTransform = _useInputTransform,
                UseFeatureTransform = _useFeatureTransform,
                InputMlpChannels = _inputMlpChannels,
                FeatureMlpChannels = _featureMlpChannels,
                ClassifierChannels = _classifierChannels,
                InputTransformMlpChannels = _inputTransformMlpChannels,
                InputTransformFcChannels = _inputTransformFcChannels,
                FeatureTransformMlpChannels = _featureTransformMlpChannels,
                FeatureTransformFcChannels = _featureTransformFcChannels,
                UseDropout = _useDropout,
                DropoutRate = _dropoutRate,
                LearningRate = NumOps.ToDouble(_learningRate)
            },
            LossFunction);
    }

    private static int[] ValidateChannelArray(int[]? values, string paramName)
    {
        if (values == null)
        {
            throw new ArgumentNullException(paramName);
        }
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

    private static int[] ValidateNonNegativeArray(int[] values, string paramName)
    {
        if (values.Length == 0)
        {
            return values;
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

        return ValidateNonNegativeArray(values, paramName);
    }
}
