using AiDotNet.Helpers;

namespace AiDotNet.NeuralNetworks;

/// <summary>
/// Represents a 3D U-Net neural network for volumetric semantic segmentation.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (typically float or double).</typeparam>
/// <remarks>
/// <para>
/// A 3D U-Net extends the classic U-Net architecture to three dimensions for processing volumetric data.
/// It uses an encoder-decoder structure with skip connections to produce dense, per-voxel predictions
/// while preserving both local details and global context.
/// </para>
/// <para>
/// <b>For Beginners:</b> A 3D U-Net is like an intelligent 3D scanner that can identify and label
/// every single voxel (3D pixel) in a 3D volume.
///
/// Think of it like this:
/// - The encoder (left side of "U") looks at the big picture by progressively zooming out
/// - The decoder (right side of "U") zooms back in to produce detailed predictions
/// - Skip connections (horizontal lines in "U") preserve fine details from encoder to decoder
///
/// This is useful for:
/// - Medical imaging: Finding organs or tumors in CT/MRI scans
/// - 3D scene understanding: Segmenting objects in point clouds
/// - Part segmentation: Identifying different parts of 3D shapes
///
/// The "U" shape comes from the symmetric encoder-decoder design with skip connections.
/// </para>
/// </remarks>
public class UNet3D<T> : NeuralNetworkBase<T>
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
    /// The voxel resolution determines the spatial dimensions of the input and output 3D grids.
    /// A resolution of 32 means the network processes 32×32×32 voxel grids.
    /// Input and output have the same spatial resolution (dense prediction).
    /// </para>
    /// </remarks>
    public int VoxelResolution { get; private set; }

    /// <summary>
    /// Gets the number of encoder blocks in the network.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Each encoder block consists of two Conv3D layers followed by a MaxPool3D layer
    /// (except the last encoder block). More blocks allow deeper feature extraction
    /// but require higher input resolution and more computation.
    /// </para>
    /// </remarks>
    public int NumEncoderBlocks { get; private set; }

    /// <summary>
    /// Gets the base number of filters in the first encoder block.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This value doubles with each encoder block. For example, with baseFilters=32
    /// and 4 blocks, the filter counts will be 32, 64, 128, 256.
    /// </para>
    /// </remarks>
    public int BaseFilters { get; private set; }

    /// <summary>
    /// Gets the number of output classes (segmentation categories).
    /// </summary>
    /// <remarks>
    /// <para>
    /// For binary segmentation (foreground/background), this is 1.
    /// For multi-class segmentation, this equals the number of categories.
    /// </para>
    /// </remarks>
    public int NumClasses { get; private set; }

    /// <summary>
    /// Initializes a new instance of the <see cref="UNet3D{T}"/> class.
    /// </summary>
    /// <param name="architecture">The architecture defining the structure of the neural network.</param>
    /// <param name="voxelResolution">The resolution of the voxel grid (e.g., 32 for 32x32x32). Default is 32.</param>
    /// <param name="numEncoderBlocks">Number of encoder blocks. Default is 4.</param>
    /// <param name="baseFilters">Base number of filters in first encoder block. Default is 32.</param>
    /// <param name="optimizer">The optimizer for training. Defaults to Adam if not specified.</param>
    /// <param name="lossFunction">The loss function. Defaults based on task type if not specified.</param>
    /// <param name="maxGradNorm">Maximum gradient norm for clipping. Defaults to 1.0.</param>
    /// <exception cref="ArgumentNullException">Thrown when architecture is null.</exception>
    /// <exception cref="ArgumentException">Thrown when voxelResolution or numEncoderBlocks is not positive.</exception>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This constructor creates a 3D U-Net with the specified configuration.
    ///
    /// Key parameters explained:
    /// - voxelResolution: The size of the 3D input grid (32 = 32×32×32 voxels)
    /// - numEncoderBlocks: How many downsampling stages (more = deeper network)
    /// - baseFilters: Starting number of feature detectors (32 is a good default)
    /// </para>
    /// </remarks>
    public UNet3D(
        NeuralNetworkArchitecture<T> architecture,
        int voxelResolution = 32,
        int numEncoderBlocks = 4,
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
        if (numEncoderBlocks <= 0)
            throw new ArgumentException("Number of encoder blocks must be positive.", nameof(numEncoderBlocks));
        if (baseFilters <= 0)
            throw new ArgumentException("Base filters must be positive.", nameof(baseFilters));

        // Minimum resolution depends on numEncoderBlocks
        int minResolution = 1 << numEncoderBlocks; // 2^numEncoderBlocks
        if (voxelResolution < minResolution)
            throw new ArgumentOutOfRangeException(nameof(voxelResolution),
                $"VoxelResolution must be at least {minResolution} for {numEncoderBlocks} encoder blocks.");

        VoxelResolution = voxelResolution;
        NumEncoderBlocks = numEncoderBlocks;
        BaseFilters = baseFilters;
        NumClasses = architecture.OutputSize;
        _lossFunction = lossFunction ?? NeuralNetworkHelper<T>.GetDefaultLossFunction(architecture.TaskType);
        _optimizer = optimizer ?? new AdamOptimizer<T, Tensor<T>, Tensor<T>>(this);

        InitializeLayers();
    }

    /// <summary>
    /// Initializes the layers of the 3D U-Net.
    /// </summary>
    /// <remarks>
    /// <para>
    /// If the architecture provides custom layers, those are used. Otherwise,
    /// default layers are created using <see cref="LayerHelper{T}.CreateDefaultUNet3DLayers"/>.
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
            Layers.AddRange(LayerHelper<T>.CreateDefaultUNet3DLayers(
                Architecture,
                VoxelResolution,
                NumEncoderBlocks,
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
    /// <returns>
    /// The output segmentation map with shape [batch, numClasses, depth, height, width]
    /// or [numClasses, depth, height, width] for single samples.
    /// </returns>
    /// <remarks>
    /// <para>
    /// The forward pass sequentially applies each layer's transformation to the input,
    /// producing per-voxel class predictions for 3D semantic segmentation.
    /// </para>
    /// </remarks>
    public Tensor<T> Forward(Tensor<T> input)
    {
        // GPU-resident optimization: use TryForwardGpuOptimized for 10-50x speedup
        if (TryForwardGpuOptimized(input, out var gpuResult))
            return gpuResult;


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
    /// <returns>The predicted segmentation map.</returns>
    /// <inheritdoc />
    public override Tensor<T> Predict(Tensor<T> input)
    {
        return Forward(input);
    }

    /// <summary>
    /// Trains the network on a single batch of input-output pairs.
    /// </summary>
    /// <param name="input">The input voxel grid tensor.</param>
    /// <param name="expectedOutput">The expected segmentation map (ground truth labels).</param>
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
            ModelType = ModelType.UNet3D,
            AdditionalInfo = new Dictionary<string, object>
            {
                { "VoxelResolution", VoxelResolution },
                { "NumEncoderBlocks", NumEncoderBlocks },
                { "BaseFilters", BaseFilters },
                { "NumClasses", NumClasses },
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
        writer.Write(NumEncoderBlocks);
        writer.Write(BaseFilters);
        writer.Write(NumClasses);
    }

    /// <summary>
    /// Deserializes network-specific data from a binary stream.
    /// </summary>
    /// <param name="reader">The binary reader to deserialize from.</param>
    /// <inheritdoc />
    protected override void DeserializeNetworkSpecificData(BinaryReader reader)
    {
        VoxelResolution = reader.ReadInt32();
        NumEncoderBlocks = reader.ReadInt32();
        BaseFilters = reader.ReadInt32();
        NumClasses = reader.ReadInt32();
    }

    /// <summary>
    /// Creates a new instance of this model type for cloning purposes.
    /// </summary>
    /// <returns>A new <see cref="UNet3D{T}"/> instance with the same configuration.</returns>
    /// <inheritdoc />
    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance()
    {
        return new UNet3D<T>(
            Architecture,
            VoxelResolution,
            NumEncoderBlocks,
            BaseFilters,
            _optimizer,
            _lossFunction,
            NumOps.ToDouble(MaxGradNorm));
    }
}
