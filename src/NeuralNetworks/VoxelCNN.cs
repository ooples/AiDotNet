using AiDotNet.Helpers;

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
    /// <summary>
    /// The loss function used to compute the error between predictions and targets.
    /// </summary>
    private readonly ILossFunction<T> _lossFunction;

    /// <summary>
    /// The optimizer used to update network parameters during training.
    /// </summary>
    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> _optimizer;

    /// <summary>
    /// Gets the voxel grid resolution used by this network.
    /// </summary>
    /// <remarks>
    /// <para>
    /// The voxel resolution determines the spatial dimensions of the input 3D grid.
    /// A resolution of 32 means the network expects 32x32x32 voxel grids.
    /// Higher resolutions capture more detail but require more computation.
    /// </para>
    /// </remarks>
    public int VoxelResolution { get; private set; }

    /// <summary>
    /// Gets the number of convolutional blocks in the network.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Each convolutional block consists of a Conv3D layer followed by a MaxPool3D layer.
    /// More blocks allow the network to learn more hierarchical features but increase
    /// computational cost and risk of overfitting.
    /// </para>
    /// </remarks>
    public int NumConvBlocks { get; private set; }

    /// <summary>
    /// Gets the base number of filters in the first convolutional layer.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This value doubles with each convolutional block. For example, with baseFilters=32
    /// and 3 blocks, the filter counts will be 32, 64, 128.
    /// </para>
    /// </remarks>
    public int BaseFilters { get; private set; }

    /// <summary>
    /// Initializes a new instance of the <see cref="VoxelCNN{T}"/> class.
    /// </summary>
    /// <param name="architecture">The architecture defining the structure of the neural network.</param>
    /// <param name="voxelResolution">The resolution of the voxel grid (e.g., 32 for 32x32x32). Default is 32.</param>
    /// <param name="numConvBlocks">Number of convolutional blocks. Default is 3.</param>
    /// <param name="baseFilters">Base number of filters in first conv layer. Default is 32.</param>
    /// <param name="optimizer">The optimizer for training. Defaults to Adam if not specified.</param>
    /// <param name="lossFunction">The loss function. Defaults based on task type if not specified.</param>
    /// <param name="maxGradNorm">Maximum gradient norm for clipping. Defaults to 1.0.</param>
    /// <exception cref="ArgumentNullException">Thrown when architecture is null.</exception>
    /// <exception cref="ArgumentException">Thrown when voxelResolution or numConvBlocks is not positive.</exception>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This constructor creates a VoxelCNN with the specified configuration.
    /// 
    /// Key parameters explained:
    /// - voxelResolution: The size of the 3D input grid (32 = 32x32x32 voxels)
    /// - numConvBlocks: How many conv+pool layers (more = deeper network)
    /// - baseFilters: Starting number of feature detectors (32 is a good default)
    /// </para>
    /// </remarks>
    public VoxelCNN(
        NeuralNetworkArchitecture<T> architecture,
        int voxelResolution = 32,
        int numConvBlocks = 3,
        int baseFilters = 32,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
        ILossFunction<T>? lossFunction = null,
        double maxGradNorm = 1.0)
        : base(architecture, lossFunction ?? NeuralNetworkHelper<T>.GetDefaultLossFunction(architecture.TaskType), maxGradNorm)
    {
        if (architecture == null)
            throw new ArgumentNullException(nameof(architecture));
        if (voxelResolution <= 0)
            throw new ArgumentException("Voxel resolution must be positive.", nameof(voxelResolution));
        if (numConvBlocks <= 0)
            throw new ArgumentException("Number of convolutional blocks must be positive.", nameof(numConvBlocks));
        if (baseFilters <= 0)
            throw new ArgumentException("Base filters must be positive.", nameof(baseFilters));

        // Minimum resolution depends on numConvBlocks (each block halves resolution)
        int minResolution = 1 << numConvBlocks; // 2^numConvBlocks
        if (voxelResolution < minResolution)
            throw new ArgumentOutOfRangeException(nameof(voxelResolution),
                $"VoxelResolution must be at least {minResolution} for {numConvBlocks} convolutional blocks.");

        VoxelResolution = voxelResolution;
        NumConvBlocks = numConvBlocks;
        BaseFilters = baseFilters;
        _lossFunction = lossFunction ?? NeuralNetworkHelper<T>.GetDefaultLossFunction(architecture.TaskType);
        _optimizer = optimizer ?? new AdamOptimizer<T, Tensor<T>, Tensor<T>>(this);

        InitializeLayers();
    }

    /// <summary>
    /// Initializes the layers of the VoxelCNN.
    /// </summary>
    /// <remarks>
    /// <para>
    /// If the architecture provides custom layers, those are used. Otherwise,
    /// default layers are created using <see cref="LayerHelper{T}.CreateDefaultVoxelCNNLayers"/>.
    /// </para>
    /// </remarks>
    protected override void InitializeLayers()
    {
        if (Architecture.Layers != null && Architecture.Layers.Count > 0)
        {
            Layers.AddRange(Architecture.Layers);
            ValidateCustomLayers(Layers);
        }
        else
        {
            Layers.AddRange(LayerHelper<T>.CreateDefaultVoxelCNNLayers(
                Architecture,
                VoxelResolution,
                NumConvBlocks,
                BaseFilters));
        }
    }

    /// <summary>
    /// Performs a forward pass through the network.
    /// </summary>
    /// <param name="input">
    /// The input voxel grid tensor with shape [batch, channels, depth, height, width] 
    /// or [channels, depth, height, width] for single samples.
    /// </param>
    /// <returns>The output predictions with shape [batch, numClasses] or [numClasses].</returns>
    /// <remarks>
    /// <para>
    /// The forward pass sequentially applies each layer's transformation to the input,
    /// producing class probabilities or scores for 3D shape classification.
    /// </para>
    /// </remarks>
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
    /// Performs a backward pass through the network to compute gradients.
    /// </summary>
    /// <param name="outputGradient">The gradient of the loss with respect to the output.</param>
    /// <returns>The gradient of the loss with respect to the input.</returns>
    /// <remarks>
    /// <para>
    /// The backward pass propagates gradients from the output back through each layer,
    /// computing gradients for all trainable parameters.
    /// </para>
    /// </remarks>
    public Tensor<T> Backward(Tensor<T> outputGradient)
    {
        for (int i = Layers.Count - 1; i >= 0; i--)
        {
            outputGradient = Layers[i].Backward(outputGradient);
        }
        return outputGradient;
    }

    /// <summary>
    /// Generates predictions for the given input.
    /// </summary>
    /// <param name="input">The input voxel grid tensor.</param>
    /// <returns>The predicted class probabilities or scores.</returns>
    /// <inheritdoc />
    public override Tensor<T> Predict(Tensor<T> input)
    {
        return Forward(input);
    }

    /// <summary>
    /// Trains the network on a single batch of input-output pairs.
    /// </summary>
    /// <param name="input">The input voxel grid tensor.</param>
    /// <param name="expectedOutput">The expected output (ground truth labels).</param>
    /// <remarks>
    /// <para>
    /// Training involves:
    /// 1. Forward pass to compute predictions
    /// 2. Loss calculation between predictions and expected output
    /// 3. Backward pass to compute gradients
    /// 4. Gradient clipping to prevent exploding gradients
    /// 5. Parameter update using the optimizer
    /// </para>
    /// </remarks>
    /// <inheritdoc />
    public override void Train(Tensor<T> input, Tensor<T> expectedOutput)
    {
        var prediction = Forward(input);

        var loss = _lossFunction.CalculateLoss(prediction.ToVector(), expectedOutput.ToVector());
        LastLoss = loss;

        var outputGradient = _lossFunction.CalculateDerivative(prediction.ToVector(), expectedOutput.ToVector());
        var outputGradientTensor = new Tensor<T>(prediction.Shape, outputGradient);

        var gradients = new List<Tensor<T>>();
        var currentGradient = outputGradientTensor;
        for (int i = Layers.Count - 1; i >= 0; i--)
        {
            currentGradient = Layers[i].Backward(currentGradient);
            gradients.Insert(0, currentGradient);
        }

        ClipGradients(gradients);
        _optimizer.UpdateParameters(Layers);
    }

    /// <summary>
    /// Updates the network parameters using a flat parameter vector.
    /// </summary>
    /// <param name="parameters">Vector containing all parameters to set.</param>
    /// <remarks>
    /// <para>
    /// This method distributes parameters from a flat vector to each layer
    /// based on their parameter counts.
    /// </para>
    /// </remarks>
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

    /// <summary>
    /// Gets metadata about this model for serialization and inspection.
    /// </summary>
    /// <returns>A <see cref="ModelMetadata{T}"/> object containing model information.</returns>
    /// <inheritdoc />
    public override ModelMetadata<T> GetModelMetadata()
    {
        return new ModelMetadata<T>
        {
            ModelType = ModelType.VoxelCNN,
            AdditionalInfo = new Dictionary<string, object>
            {
                { "VoxelResolution", VoxelResolution },
                { "NumConvBlocks", NumConvBlocks },
                { "BaseFilters", BaseFilters },
                { "InputShape", Architecture.GetInputShape() },
                { "OutputShape", Layers.Count > 0 ? Layers[Layers.Count - 1].GetOutputShape() : Array.Empty<int>() },
                { "LayerCount", Layers.Count },
                { "LayerTypes", Layers.Select(l => l.GetType().Name).ToArray() }
            },
            ModelData = this.Serialize()
        };
    }

    /// <summary>
    /// Serializes network-specific data to a binary stream.
    /// </summary>
    /// <param name="writer">The binary writer to serialize to.</param>
    /// <inheritdoc />
    protected override void SerializeNetworkSpecificData(BinaryWriter writer)
    {
        writer.Write(VoxelResolution);
        writer.Write(NumConvBlocks);
        writer.Write(BaseFilters);
    }

    /// <summary>
    /// Deserializes network-specific data from a binary stream.
    /// </summary>
    /// <param name="reader">The binary reader to deserialize from.</param>
    /// <inheritdoc />
    protected override void DeserializeNetworkSpecificData(BinaryReader reader)
    {
        VoxelResolution = reader.ReadInt32();
        NumConvBlocks = reader.ReadInt32();
        BaseFilters = reader.ReadInt32();
    }

    /// <summary>
    /// Creates a new instance of this model type for cloning purposes.
    /// </summary>
    /// <returns>A new <see cref="VoxelCNN{T}"/> instance with the same configuration.</returns>
    /// <inheritdoc />
    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance()
    {
        return new VoxelCNN<T>(
            Architecture,
            VoxelResolution,
            NumConvBlocks,
            BaseFilters,
            _optimizer,
            _lossFunction,
            NumOps.ToDouble(MaxGradNorm));
    }
}
