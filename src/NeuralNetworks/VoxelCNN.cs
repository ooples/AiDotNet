namespace AiDotNet.NeuralNetworks;

/// <summary>
/// Represents a Voxel-based 3D Convolutional Neural Network for processing volumetric data.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (typically float or double).</typeparam>
/// <remarks>
/// <para>
/// A Voxel CNN processes 3D volumetric data using 3D convolutions. This is useful for:
/// - 3D shape recognition from voxelized point clouds (e.g., ModelNet40)
/// - Medical image analysis (CT, MRI scans)
/// - Spatial occupancy prediction
/// </para>
/// <para>
/// <b>For Beginners:</b> Think of a VoxelCNN as a 3D version of a regular image classifier.
/// Instead of looking at a 2D image, it examines a 3D grid of "blocks" (voxels) to understand
/// 3D shapes. This is like how Minecraft represents the world - each block is either filled
/// or empty, and the pattern of blocks creates recognizable objects.
/// 
/// Applications include:
/// - Recognizing 3D objects from point cloud scans
/// - Detecting tumors in 3D medical scans
/// - Understanding room layouts from depth sensors
/// </para>
/// </remarks>
public class VoxelCNN<T> : NeuralNetworkBase<T>
{
    private ILossFunction<T> _lossFunction;
    private IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> _optimizer;

    /// <summary>
    /// Gets the voxel grid resolution used by this network.
    /// </summary>
    public int VoxelResolution { get; private set; }

    /// <summary>
    /// Initializes a new instance of the VoxelCNN class.
    /// </summary>
    /// <param name="architecture">The architecture defining the structure of the neural network.</param>
    /// <param name="voxelResolution">The resolution of the voxel grid (e.g., 32 for 32x32x32).</param>
    /// <param name="optimizer">The optimizer for training. Defaults to Adam if not specified.</param>
    /// <param name="lossFunction">The loss function. Defaults based on task type if not specified.</param>
    /// <param name="maxGradNorm">Maximum gradient norm for clipping. Defaults to 1.0.</param>
    public VoxelCNN(
        NeuralNetworkArchitecture<T> architecture,
        int voxelResolution = 32,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
        ILossFunction<T>? lossFunction = null,
        double maxGradNorm = 1.0)
        : base(architecture, lossFunction ?? NeuralNetworkHelper<T>.GetDefaultLossFunction(architecture.TaskType), maxGradNorm)
    {
        VoxelResolution = voxelResolution;
        _lossFunction = lossFunction ?? NeuralNetworkHelper<T>.GetDefaultLossFunction(architecture.TaskType);
        _optimizer = optimizer ?? new AdamOptimizer<T, Tensor<T>, Tensor<T>>(this);

        InitializeLayers();
    }

    /// <inheritdoc />
    protected override void InitializeLayers()
    {
        if (Architecture.Layers != null && Architecture.Layers.Count > 0)
        {
            Layers.AddRange(Architecture.Layers);
            ValidateCustomLayers(Layers);
        }
        else
        {
            // Create default VoxelCNN architecture
            Layers.AddRange(CreateDefaultLayers());
        }
    }

    private List<ILayer<T>> CreateDefaultLayers()
    {
        var layers = new List<ILayer<T>>();
        int numClasses = Architecture.OutputSize;

        // Layer 1: Conv3D 1->32, kernel 3, stride 1, padding 1
        layers.Add(new Layers.Conv3DLayer<T>(
            inputChannels: 1,
            outputChannels: 32,
            kernelSize: 3,
            inputDepth: VoxelResolution,
            inputHeight: VoxelResolution,
            inputWidth: VoxelResolution,
            stride: 1,
            padding: 1,
            activation: new ReLUActivation<T>()));

        // Layer 2: MaxPool3D kernel 2, stride 2 -> resolution / 2
        int res1 = VoxelResolution / 2;
        layers.Add(new Layers.MaxPool3DLayer<T>(
            inputShape: [32, VoxelResolution, VoxelResolution, VoxelResolution],
            poolSize: 2,
            stride: 2));

        // Layer 3: Conv3D 32->64, kernel 3, stride 1, padding 1
        layers.Add(new Layers.Conv3DLayer<T>(
            inputChannels: 32,
            outputChannels: 64,
            kernelSize: 3,
            inputDepth: res1,
            inputHeight: res1,
            inputWidth: res1,
            stride: 1,
            padding: 1,
            activation: new ReLUActivation<T>()));

        // Layer 4: MaxPool3D kernel 2, stride 2 -> resolution / 4
        int res2 = res1 / 2;
        layers.Add(new Layers.MaxPool3DLayer<T>(
            inputShape: [64, res1, res1, res1],
            poolSize: 2,
            stride: 2));

        // Layer 5: Conv3D 64->128, kernel 3, stride 1, padding 1
        layers.Add(new Layers.Conv3DLayer<T>(
            inputChannels: 64,
            outputChannels: 128,
            kernelSize: 3,
            inputDepth: res2,
            inputHeight: res2,
            inputWidth: res2,
            stride: 1,
            padding: 1,
            activation: new ReLUActivation<T>()));

        // Layer 6: Global Average Pooling (flatten to [128])
        layers.Add(new Layers.GlobalPoolingLayer<T>(
            inputShape: [128, res2, res2, res2],
            poolingType: PoolingType.Average,
            activationFunction: (IActivationFunction<T>?)null));

        // Layer 7: Dense 128 -> numClasses
        layers.Add(new Layers.DenseLayer<T>(
            inputSize: 128,
            outputSize: numClasses,
            activationFunction: Architecture.TaskType == NeuralNetworkTaskType.MultiClassClassification
                ? new SoftmaxActivation<T>()
                : new SigmoidActivation<T>()));

        return layers;
    }

    /// <summary>
    /// Performs a forward pass through the network.
    /// </summary>
    /// <param name="input">The input voxel grid tensor [batch, channels, depth, height, width] or [channels, depth, height, width].</param>
    /// <returns>The output predictions.</returns>
    public Tensor<T> Forward(Tensor<T> input)
    {
        Tensor<T> output = input;
        foreach (var layer in Layers)
        {
            output = layer.Forward(output);
        }
        return output;
    }

    /// <summary>
    /// Performs a backward pass through the network.
    /// </summary>
    /// <param name="outputGradient">The gradient of the loss with respect to the output.</param>
    /// <returns>The gradient of the loss with respect to the input.</returns>
    public Tensor<T> Backward(Tensor<T> outputGradient)
    {
        for (int i = Layers.Count - 1; i >= 0; i--)
        {
            outputGradient = Layers[i].Backward(outputGradient);
        }
        return outputGradient;
    }

    /// <inheritdoc />
    public override Tensor<T> Predict(Tensor<T> input)
    {
        return Forward(input);
    }

    /// <inheritdoc />
    public override void Train(Tensor<T> input, Tensor<T> expectedOutput)
    {
        // Forward pass
        var prediction = Forward(input);

        // Calculate loss
        var loss = _lossFunction.CalculateLoss(prediction.ToVector(), expectedOutput.ToVector());
        LastLoss = loss;

        // Calculate output gradient
        var outputGradient = _lossFunction.CalculateDerivative(prediction.ToVector(), expectedOutput.ToVector());
        var outputGradientTensor = new Tensor<T>(prediction.Shape, outputGradient);

        // Backward pass
        var gradients = new List<Tensor<T>>();
        var currentGradient = outputGradientTensor;
        for (int i = Layers.Count - 1; i >= 0; i--)
        {
            currentGradient = Layers[i].Backward(currentGradient);
            gradients.Insert(0, currentGradient);
        }

        // Apply gradient clipping
        ClipGradients(gradients);

        // Update parameters
        _optimizer.UpdateParameters(Layers);
    }

    /// <inheritdoc />
    public override void UpdateParameters(Vector<T> parameters)
    {
        int index = 0;
        foreach (var layer in Layers)
        {
            int layerParameterCount = layer.ParameterCount;
            var layerParameters = parameters.Slice(index, layerParameterCount);
            layer.UpdateParameters(layerParameters);
            index += layerParameterCount;
        }
    }

    /// <inheritdoc />
    public override ModelMetadata<T> GetModelMetadata()
    {
        return new ModelMetadata<T>
        {
            ModelType = ModelType.VoxelCNN,
            AdditionalInfo = new Dictionary<string, object>
            {
                { "VoxelResolution", VoxelResolution },
                { "InputShape", Architecture.GetInputShape() },
                { "OutputShape", Layers[Layers.Count - 1].GetOutputShape() },
                { "LayerCount", Layers.Count },
                { "LayerTypes", Layers.Select(l => l.GetType().Name).ToArray() }
            },
            ModelData = this.Serialize()
        };
    }

    /// <inheritdoc />
    protected override void SerializeNetworkSpecificData(BinaryWriter writer)
    {
        writer.Write(VoxelResolution);
    }

    /// <inheritdoc />
    protected override void DeserializeNetworkSpecificData(BinaryReader reader)
    {
        VoxelResolution = reader.ReadInt32();
    }

    /// <inheritdoc />
    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance()
    {
        return new VoxelCNN<T>(
            Architecture,
            VoxelResolution,
            _optimizer,
            _lossFunction,
            NumOps.ToDouble(MaxGradNorm));
    }
}
