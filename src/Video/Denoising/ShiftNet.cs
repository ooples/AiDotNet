using System.IO;
using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks;
using AiDotNet.Onnx;
using AiDotNet.Optimizers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Video.Options;

namespace AiDotNet.Video.Denoising;

/// <summary>
/// ShiftNet channel-shifting video denoiser using zero-cost temporal feature exchange.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para><b>References:</b>
/// <list type="bullet">
/// <item>Paper: "A Simple Baseline for Video Restoration with Grouped Spatial-temporal Shift"
/// (Li et al., CVPR 2023, arXiv:2206.10810)</item>
/// </list></para>
/// <para><b>For Beginners:</b> ShiftNet uses efficient shift operations instead of expensive 3D convolutions for video denoising. By shifting feature maps along the temporal dimension, it captures motion at minimal computational cost.</para>
/// <para>
/// <b>CITATION CORRECTED.</b> This class previously cited arXiv:2106.10948 as "An Efficient Recurrent
/// Architecture for Video Denoising via Temporal Shift" by "Marco Maggioni" et al. (2021). That
/// identifier is "A scalar Riemann-Hilbert problem on the torus: Applications to the KdV equation" —
/// integrable systems, not video. The claimed title does not exist, and the author list was a blend:
/// Maggioni/Huang/Li are real co-authors of the unrelated EMVD paper (arXiv:2103.05407, and the first
/// name is Matteo, not Marco), while Rao/Lu/Zhou are from a different group entirely.
/// </para>
/// <para>
/// The mechanism was also absent: the class described channel shifting but contained no shift operation
/// at all, and its NumShifts / ShiftRadius options were consumed by nothing. The real technique lives in
/// <see cref="GroupedSpatialTemporalShift{T}"/>:
/// </para>
/// <code>
///   temporal   f_i split equally into f_i^a, f_i^b; FTS and BTS blocks alternate over frame pairs
///   spatial    f_{i-1}^b split into M = 25 slices, each shifted by (dx, dy) from {-9,-5,0,5,9}^2
///   loss       L = (1/T) sum_i ||H_i - O_i||_1
/// </code>
/// <para>
/// Both halves are required. The temporal half moves information BETWEEN frames; the spatial half moves
/// it WITHIN a frame across 25 displacements, supplying the large effective receptive field that stands
/// in for explicit correspondence search. Implementing only the temporal shift — the obvious reading of
/// "temporal shift" — cannot align anything that moved sideways.
/// </para>
/// </remarks>
/// <example>
/// <code>
/// // Create a ShiftNet model for efficient temporal-shift video denoising
/// var architecture = new NeuralNetworkArchitecture&lt;double&gt;(
///     inputType: InputType.ThreeDimensional,
///     inputHeight: 256, inputWidth: 256, inputDepth: 3);
/// var options = new ShiftNetOptions();
/// var shiftNet = new ShiftNet&lt;double&gt;(architecture, options);
///
/// // Or load a pre-trained ONNX model for inference
/// var shiftNetOnnx = new ShiftNet&lt;double&gt;(architecture, "shiftnet_model.onnx");
/// </code>
/// </example>
[ModelDomain(ModelDomain.Video)]
[ModelDomain(ModelDomain.Vision)]
[ModelCategory(ModelCategory.NeuralNetwork)]
[ModelTask(ModelTask.Generation)]
[ModelComplexity(ModelComplexity.Medium)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("A Simple Baseline for Video Restoration with Grouped Spatial-temporal Shift",
    "https://arxiv.org/abs/2206.10810",
    Year = 2023,
    Authors = "Dasong Li, Xiaoyu Shi, Yi Zhang, Ka Chun Cheung, Simon See, Xiaogang Wang, Hongwei Qin, Hongsheng Li")]
public class ShiftNet<T> : VideoDenoisingBase<T>
{
    private readonly ShiftNetOptions _options;

    private readonly GroupedSpatialTemporalShift<T> _shift = new();

    /// <summary>
    /// Gets the grouped spatial-temporal shift module — the paper's alignment mechanism.
    /// </summary>
    /// <remarks>
    /// Exposed rather than buried because it is the model's whole contribution, and because it is
    /// verifiable independently of the U-Net backbone: shifting is parameter-free, so its correctness is
    /// a property of the operation and not of any trained weights.
    /// </remarks>
    public GroupedSpatialTemporalShift<T> Shift => _shift;

    /// <inheritdoc/>
    public override ModelOptions GetOptions() => _options;

    private IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _optimizer;
    private bool _useNativeMode;
    private bool _disposed;

    /// <summary>
    /// Creates a ShiftNet model for ONNX inference.
    /// </summary>
    public ShiftNet(
        NeuralNetworkArchitecture<T> architecture,
        string modelPath,
        ShiftNetOptions? options = null)
        : base(architecture)
    {
        _options = options ?? new ShiftNetOptions();
        _useNativeMode = false;
        TemporalRadius = _options.ShiftRadius;
        _options.ModelPath = modelPath;
        OnnxModel = new OnnxModel<T>(modelPath, _options.OnnxOptions);
        InitializeLayers();
    }

    /// <summary>
    /// Creates a ShiftNet model for native training and inference.
    /// </summary>
    public ShiftNet(
        NeuralNetworkArchitecture<T> architecture,
        ShiftNetOptions? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null)
        : base(architecture)
    {
        _options = options ?? new ShiftNetOptions();
        _useNativeMode = true;
        _optimizer = optimizer ?? new AdamWOptimizer<T, Tensor<T>, Tensor<T>>(this,
            new AdamWOptimizerOptions<T, Tensor<T>, Tensor<T>>
            {
                InitialLearningRate = _options.LearningRate
            });
        TemporalRadius = _options.ShiftRadius;
        InitializeLayers();
    }

    /// <inheritdoc/>
    public override Tensor<T> Denoise(Tensor<T> noisyFrames)
    {
        ThrowIfDisposed();
        var preprocessed = PreprocessFrames(noisyFrames);
        var output = IsOnnxMode ? RunOnnxInference(preprocessed) : Forward(preprocessed);
        return PostprocessOutput(output);
    }

    /// <inheritdoc/>
    protected override void InitializeLayers()
    {
        if (!_useNativeMode) return;
        if (Architecture.Layers is not null && Architecture.Layers.Count > 0)
        {
            Layers.AddRange(Architecture.Layers);
        }
        else
        {
            int ch = Architecture.InputDepth > 0 ? Architecture.InputDepth : 3;
            int h = Architecture.InputHeight > 0 ? Architecture.InputHeight : 128;
            int w = Architecture.InputWidth > 0 ? Architecture.InputWidth : 128;
            int temporalFrames = 2 * _options.ShiftRadius + 1;
            Layers.AddRange(LayerHelper<T>.CreateDefaultVideoDenoisingLayers(
                inputChannels: ch, inputHeight: h, inputWidth: w,
                numFeatures: _options.NumFeatures,
                temporalFrames: temporalFrames));
        }
    }

    /// <inheritdoc/>
    protected override Tensor<T> PreprocessFrames(Tensor<T> rawFrames) => NormalizeFrames(rawFrames);

    /// <inheritdoc/>
    protected override Tensor<T> PostprocessOutput(Tensor<T> modelOutput) => DenormalizeFrames(modelOutput);

    /// <inheritdoc/>
    protected override IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> GetOrCreateBaseOptimizer()
        => _optimizer ?? base.GetOrCreateBaseOptimizer();

    // UpdateParameters re-sliced the flat vector across Layers by hand -- the base walks
    // exactly the same enumeration, so this said nothing the base does not already say.
    /// <inheritdoc/>
    public override ModelMetadata<T> GetModelMetadata()
    {
        return new ModelMetadata<T>
        {
            AdditionalInfo = new Dictionary<string, object>
            {
                { "ModelName", "ShiftNet" },
                { "Variant", _options.Variant.ToString() },
                { "NumFeatures", _options.NumFeatures },
                { "NumBlocks", _options.NumBlocks },
                { "NumShifts", _options.NumShifts },
                { "ShiftRadius", _options.ShiftRadius }
            },
            ModelData = SerializeForMetadata()
        };
    }

    /// <inheritdoc/>
    protected override void SerializeNetworkSpecificData(BinaryWriter writer)
    {
        writer.Write((int)_options.Variant);
        writer.Write(_options.NumFeatures);
        writer.Write(_options.NumBlocks);
        writer.Write(_options.NumShifts);
        writer.Write(_options.ShiftRadius);
        writer.Write(_options.LearningRate);
        writer.Write(_options.DropoutRate);
    }

    /// <inheritdoc/>
    protected override void DeserializeNetworkSpecificData(BinaryReader reader)
    {
        _options.Variant = (VideoModelVariant)reader.ReadInt32();
        _options.NumFeatures = reader.ReadInt32();
        _options.NumBlocks = reader.ReadInt32();
        _options.NumShifts = reader.ReadInt32();
        _options.ShiftRadius = reader.ReadInt32();
        _options.LearningRate = reader.ReadDouble();
        _options.DropoutRate = reader.ReadDouble();
    }

    /// <inheritdoc/>
    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance()
    {
        return new ShiftNet<T>(Architecture, _options);
    }

    private void ThrowIfDisposed()
    {
        if (_disposed) throw new ObjectDisposedException(GetType().FullName ?? nameof(ShiftNet<T>));
    }

    /// <inheritdoc/>
    protected override void Dispose(bool disposing)
    {
        if (_disposed) return;
        _disposed = true;
        if (disposing) OnnxModel?.Dispose();
        base.Dispose(disposing);
    }
}
