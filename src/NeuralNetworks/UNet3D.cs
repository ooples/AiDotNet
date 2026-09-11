using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks.Options;

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
/// <example>
/// <code>
/// var options = new UNet3DOptions { InputChannels = 1, OutputChannels = 4, BaseChannels = 32 };
/// var model = new UNet3D&lt;float&gt;(options);
/// var volume = Tensor&lt;float&gt;.Random(new[] { 1, 1, 64, 64, 64 });
/// var segmented = model.Predict(volume);
/// </code>
/// </example>
[ModelDomain(ModelDomain.Vision)]
[ModelDomain(ModelDomain.ThreeD)]
[ModelCategory(ModelCategory.NeuralNetwork)]
[ModelCategory(ModelCategory.ConvolutionalNetwork)]
[ModelTask(ModelTask.Segmentation)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("3D U-Net: Learning Dense Volumetric Segmentation from Sparse Annotation", "https://arxiv.org/abs/1606.06650", Year = 2016, Authors = "Ozgun Cicek, Ahmed Abdulkadir, Soeren S. Lienkamp, Thomas Brox, Olaf Ronneberger")]
public partial class UNet3D<T> : VolumetricModelLayoutBase<T>
{
    private readonly UNet3DOptions _options;

    /// <inheritdoc/>
    public override ModelOptions GetOptions() => _options;

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
    /// Initializes a new instance with default architecture settings.
    /// </summary>
    public UNet3D()
        : this(new NeuralNetworkArchitecture<T>(
            inputType: Enums.InputType.ThreeDimensional,
            taskType: Enums.NeuralNetworkTaskType.Regression,
            inputHeight: 32, inputWidth: 32, inputDepth: 32,
            outputSize: 1))
    {
    }

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
    public UNet3D(
        NeuralNetworkArchitecture<T> architecture,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
        ILossFunction<T>? lossFunction = null,
        UNet3DOptions? options = null)
        : this(options ?? new UNet3DOptions(), architecture, optimizer, lossFunction)
    {
    }

    /// <summary>
    /// Initializes the model from an already-resolved options instance.
    /// </summary>
    /// <remarks>
    /// <para>
    /// The base initializer needs MaxGradNorm and runs before the body, so the options must be
    /// resolved first. Options come first in the parameter list because a nullable and a
    /// non-nullable reference type are the same type to the compiler.
    /// </para>
    /// </remarks>
    private UNet3D(
        UNet3DOptions options,
        NeuralNetworkArchitecture<T> architecture,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer,
        ILossFunction<T>? lossFunction)
        : base(architecture, lossFunction ?? NeuralNetworkHelper<T>.GetDefaultLossFunction(architecture.TaskType), options.MaxGradNorm)
    {
        // Every dimension check the constructor used to make now lives on the options, including
        // the cross-field resolution-vs-blocks rule.
        options.Validate();
        _options = options;
        Options = _options;

        if (architecture == null)
            throw new ArgumentNullException(nameof(architecture));

        VoxelResolution = options.VoxelResolution;
        NumEncoderBlocks = options.NumEncoderBlocks;
        BaseFilters = options.BaseFilters;
        NumClasses = architecture.OutputSize;
        _lossFunction = lossFunction ?? NeuralNetworkHelper<T>.GetDefaultLossFunction(architecture.TaskType);
        // Built from the options rather than bare: a bare AdamOptimizer trains at its own
        // default and silently ignores anything the caller configured.
        _optimizer = optimizer ?? new AdamOptimizer<T, Tensor<T>, Tensor<T>>(
            this,
            new AdamOptimizerOptions<T, Tensor<T>, Tensor<T>>
            {
                InitialLearningRate = options.LearningRate
            });

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
        // A caller-supplied layer list has no encoder/decoder split to tap, so run it as given.
        if (Layers.Count != ExpectedGeneratedLayerCount(NumEncoderBlocks))
        {
            Tensor<T> sequential = input;
            foreach (var layer in Layers)
            {
                sequential = layer.Forward(sequential);
            }

            return sequential;
        }

        // Channels sit on axis 1 when the volume carries a batch dimension and axis 0 otherwise.
        int channelAxis = input.Rank >= 5 ? 1 : 0;

        var features = input;
        int li = 0;
        var taps = new Tensor<T>[NumEncoderBlocks];

        // Encoder: two convolutions per block, tapped before the pool that follows it.
        for (int block = 0; block < NumEncoderBlocks; block++)
        {
            features = Layers[li++].Forward(features);
            features = Layers[li++].Forward(features);
            taps[block] = features;

            if (block < NumEncoderBlocks - 1)
            {
                features = Layers[li++].Forward(features);
            }
        }

        // Bottleneck.
        features = Layers[li++].Forward(features);

        // Decoder: up-convolve (2x resolution, half the channels), concatenate the encoder tap
        // now at the matching resolution, then two convolutions.
        for (int block = NumEncoderBlocks - 2; block >= 0; block--)
        {
            features = Layers[li++].Forward(features);
            features = Engine.TensorConcatenate(new[] { features, taps[block] }, axis: channelAxis);
            features = Layers[li++].Forward(features);
            features = Layers[li++].Forward(features);
        }

        // 1x1x1 output projection.
        features = Layers[li++].Forward(features);

        return features;
    }

    /// <summary>
    /// Training must take the same skip path as inference.
    /// </summary>
    /// <param name="input">The input voxel grid tensor.</param>
    /// <returns>The network output.</returns>
    /// <remarks>
    /// <para>
    /// The decoder convolutions resolve their input channel count lazily, from whatever they are
    /// first handed. A plain sequential training pass would resolve them against NON-concatenated
    /// features and then mismatch on the first real forward, so the two paths cannot diverge.
    /// </para>
    /// </remarks>
    public override Tensor<T> ForwardForTraining(Tensor<T> input) => Forward(input);

    /// <summary>
    /// The number of layers <see cref="LayerHelper{T}.CreateDefaultUNet3DLayers"/> emits.
    /// </summary>
    /// <param name="numEncoderBlocks">The configured encoder block count.</param>
    /// <returns>The expected layer count.</returns>
    /// <remarks>
    /// <para>
    /// Two convolutions per encoder block, a pool after all but the last, one bottleneck
    /// convolution, an up-convolution plus two convolutions per decoder block, and the output
    /// projection. Forward compares against this so a caller-supplied architecture falls back to a
    /// sequential pass instead of indexing into a layout it does not have.
    /// </para>
    /// </remarks>
    private static int ExpectedGeneratedLayerCount(int numEncoderBlocks)
        => (2 * numEncoderBlocks) + (numEncoderBlocks - 1) + 1 + (3 * (numEncoderBlocks - 1)) + 1;

    /// <summary>Latches the one-time lazy shape resolution.</summary>
    private bool _lazyShapesResolved;

    /// <summary>
    /// Resolves the lazy convolutions through the real skip topology.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Conv3DLayer is built with placeholder shapes and pins its kernel's input channel count to
    /// whatever first reaches it. The base class resolves that by walking <see cref="Layers"/>
    /// sequentially, which cannot model the concatenations: it would size the first decoder
    /// convolution against its non-concatenated predecessor (256) rather than the concatenated
    /// 256 + 128 = 384, and then fail the first real forward.
    /// </para>
    /// <para>
    /// One inference-mode forward through the actual topology materialises every layer with its
    /// true channel count, and leaves ParameterCount non-zero before the first user forward, which
    /// is the contract the base method upholds.
    /// </para>
    /// </remarks>
    protected override void ResolveLazyLayerShapes()
    {
        if (_lazyShapesResolved) return;
        if (Layers is null || Layers.Count == 0) return;

        using (InferenceMode.Enter())
        {
            Forward(new Tensor<T>([1, VoxelResolution, VoxelResolution, VoxelResolution]));
        }

        _lazyShapesResolved = true;
    }

    /// <summary>
    /// Generates predictions for the given input.
    /// </summary>
    /// <param name="input">The input voxel grid tensor.</param>
    /// <returns>The predicted segmentation map.</returns>
    /// <inheritdoc />
    /// <summary>
    /// Routes inference through <see cref="NeuralNetworkBase{T}.PredictCompiled"/> for
    /// compiled-plan replay; <see cref="Forward"/> remains the eager fallback.
    /// </summary>
    protected override Tensor<T> PredictEager(Tensor<T> input) => Forward(input);

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
        SetTrainingMode(true);
        try
        {
            TrainWithTape(input, expectedOutput, _optimizer);
        }
        finally
        {
            SetTrainingMode(false);
        }
    }

    // UpdateParameters re-sliced the flat vector across Layers by hand -- the base walks
    // exactly the same enumeration, so this said nothing the base does not already say.
    /// <summary>
    /// Gets metadata about this model for serialization and inspection.
    /// </summary>
    /// <returns>A <see cref="ModelMetadata{T}"/> object containing model information.</returns>
    /// <inheritdoc />
    public override ModelMetadata<T> GetModelMetadata()
    {
        return new ModelMetadata<T>
        {
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
            ModelData = SerializeForMetadata()
        };
    }

    /// <summary>
    /// Serializes network-specific data to a binary stream.
    /// </summary>
    /// <param name="writer">The binary writer to serialize to.</param>
    /// <inheritdoc />


    /// <summary>
    /// Deserializes network-specific data from a binary stream.
    /// </summary>
    /// <param name="reader">The binary reader to deserialize from.</param>
    /// <inheritdoc />

}
