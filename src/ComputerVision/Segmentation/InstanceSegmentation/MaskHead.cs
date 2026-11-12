using System.IO;
using AiDotNet.ComputerVision.Detection.Backbones;
using AiDotNet.Interfaces;
using AiDotNet.Models.Parameters;
using AiDotNet.Tensors;

namespace AiDotNet.ComputerVision.Segmentation.InstanceSegmentation;

/// <summary>
/// Mask prediction head for instance segmentation.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> The mask head takes RoI-pooled features and predicts
/// a binary segmentation mask for each class. It typically uses a series of
/// convolutional layers followed by a transposed convolution for upsampling.</para>
///
/// <para>Key features:
/// - Multiple convolutional layers for feature processing
/// - Upsampling via transposed convolution
/// - Per-class mask prediction
/// - Configurable mask resolution
/// </para>
/// </remarks>
public class MaskHead<T> : IParameterSource<T>, IParameterChunkSource<T>, IParameterLayoutSource
{
    private readonly Conv2D<T> _conv1;
    private readonly Conv2D<T> _conv2;
    private readonly Conv2D<T> _conv3;
    private readonly Conv2D<T> _conv4;
    private readonly ConvTranspose2D<T> _deconv;
    private readonly Conv2D<T> _predictor;
    private readonly int _numClasses;
    private readonly int _maskResolution;

    /// <summary>
    /// Creates the Mask R-CNN mask head (He et al. 2017, FPN variant): four 3x3 convolutions of width
    /// 256, a 2x2 stride-2 transposed convolution that doubles the RoI resolution (14 to 28), then a
    /// 1x1 convolution giving one mask logit map per class. ReLU follows every layer but the last.
    /// </summary>
    /// <param name="inChannels">Channels of the pooled RoI features (256 for FPN).</param>
    /// <param name="numClasses">Foreground classes; one mask is predicted per class.</param>
    /// <param name="maskResolution">Output mask side, twice the pooled RoI side (28 in the paper).</param>
    public MaskHead(int inChannels, int numClasses, int maskResolution = 28)
    {
        if (inChannels <= 0) throw new ArgumentOutOfRangeException(nameof(inChannels));
        if (numClasses <= 0) throw new ArgumentOutOfRangeException(nameof(numClasses));
        if (maskResolution <= 0 || maskResolution % 2 != 0)
            throw new ArgumentOutOfRangeException(nameof(maskResolution),
                "The mask side must be a positive even number: the head doubles the pooled RoI side.");
        _numClasses = numClasses;
        _maskResolution = maskResolution;

        _conv1 = new Conv2D<T>(inChannels, 256, kernelSize: 3, padding: 1);
        _conv2 = new Conv2D<T>(256, 256, kernelSize: 3, padding: 1);
        _conv3 = new Conv2D<T>(256, 256, kernelSize: 3, padding: 1);
        _conv4 = new Conv2D<T>(256, 256, kernelSize: 3, padding: 1);
        _deconv = new ConvTranspose2D<T>(256, 256, kernelSize: 2, stride: 2);
        _predictor = new Conv2D<T>(256, numClasses, kernelSize: 1);
    }

    /// <summary>The side of the pooled RoI features this head reads: half the mask side (14).</summary>
    public int RoiSize => _maskResolution / 2;

    /// <summary>The side of the predicted masks (28).</summary>
    public int MaskResolution => _maskResolution;

    /// <summary>
    /// Predicts per-class mask logits on the tape.
    /// </summary>
    /// <param name="roiFeatures">Pooled RoI features [rois, channels, RoiSize, RoiSize].</param>
    /// <returns>Mask logits [rois, numClasses, MaskResolution, MaskResolution].</returns>
    public Tensor<T> Forward(Tensor<T> roiFeatures)
    {
        if (roiFeatures is null) throw new ArgumentNullException(nameof(roiFeatures));
        var engine = AiDotNetEngine.Current;
        var x = engine.ReLU(_conv1.Forward(roiFeatures));
        x = engine.ReLU(_conv2.Forward(x));
        x = engine.ReLU(_conv3.Forward(x));
        x = engine.ReLU(_conv4.Forward(x));
        x = engine.ReLU(_deconv.Forward(x));
        return _predictor.Forward(x);
    }

    /// <summary>
    /// Predicts the mask of one class for one RoI, as probabilities.
    /// </summary>
    /// <param name="roiFeatures">Features for a single RoI [1, channels, RoiSize, RoiSize].</param>
    /// <param name="classId">Foreground class whose mask to return.</param>
    /// <returns>Mask probabilities [MaskResolution, MaskResolution].</returns>
    public Tensor<T> PredictMask(Tensor<T> roiFeatures, int classId)
    {
        if (classId < 0 || classId >= _numClasses) throw new ArgumentOutOfRangeException(nameof(classId));
        var logits = Forward(roiFeatures);
        int height = logits.Shape[2];
        int width = logits.Shape[3];
        var engine = AiDotNetEngine.Current;
        var classMap = CvTensorOps<T>.Select(logits, new[] { classId }, 1);
        return engine.Reshape(engine.Sigmoid(classMap), new[] { height, width });
    }

    /// <summary>Gets the total parameter count.</summary>
    public long GetParameterCount() => Parameters.ParameterCount;

    /// <summary>Writes the parameters to a binary stream.</summary>
    public void WriteParameters(BinaryWriter writer)
    {
        writer.Write(_numClasses);
        writer.Write(_maskResolution);
        _conv1.WriteParameters(writer);
        _conv2.WriteParameters(writer);
        _conv3.WriteParameters(writer);
        _conv4.WriteParameters(writer);
        _deconv.WriteParameters(writer);
        _predictor.WriteParameters(writer);
    }

    /// <summary>Reads the parameters from a binary stream.</summary>
    public void ReadParameters(BinaryReader reader)
    {
        int numClasses = reader.ReadInt32();
        int maskRes = reader.ReadInt32();
        if (numClasses != _numClasses || maskRes != _maskResolution)
        {
            throw new InvalidOperationException(
                $"MaskHead configuration mismatch. Expected numClasses={_numClasses}, maskRes={_maskResolution}, " +
                $"got numClasses={numClasses}, maskRes={maskRes}");
        }
        _conv1.ReadParameters(reader);
        _conv2.ReadParameters(reader);
        _conv3.ReadParameters(reader);
        _conv4.ReadParameters(reader);
        _deconv.ReadParameters(reader);
        _predictor.ReadParameters(reader);
    }

    // MaskHead is public, so it forwards the parameter interfaces to an internal module, as RPN does.
    // Before this the model registry could not see inside the mask head: it was never trained,
    // counted only through a hand-written sum, and dropped by SetParameters and cloning.
    private DelegatingCvParameterModule<T>? _parameters;

    private DelegatingCvParameterModule<T> Parameters
        => _parameters ??= new DelegatingCvParameterModule<T>(
            () => new IParameterSource<T>?[] { _conv1, _conv2, _conv3, _conv4, _deconv, _predictor });

    /// <inheritdoc />
    long IParameterSource<T>.ParameterCount => Parameters.ParameterCount;

    /// <inheritdoc />
    IReadOnlyList<ParameterSlotDescriptor> IParameterLayoutSource.GetParameterLayout() => Parameters.GetParameterLayout();

    /// <inheritdoc />
    Vector<T> IParameterSource<T>.GetParameters() => Parameters.GetParameters();

    /// <inheritdoc />
    void IParameterSource<T>.SetParameters(Vector<T> parameters) => Parameters.SetParameters(parameters);

    /// <inheritdoc />
    IEnumerable<ParameterChunk<T>> IParameterChunkSource<T>.GetParameterStateChunks() => Parameters.GetParameterStateChunks();
}

/// <summary>
/// Prototype-based mask head for YOLO and SOLOv2.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> Instead of predicting masks directly for each instance,
/// prototype-based methods predict a set of prototype masks and per-instance coefficients.
/// The final mask is a linear combination of prototypes weighted by coefficients.</para>
/// </remarks>
public class PrototypeMaskHead<T> : IParameterSource<T>, IParameterChunkSource<T>, IParameterLayoutSource
{
    private readonly INumericOperations<T> _numOps;
    private readonly Conv2D<T> _protoConv1;
    private readonly Conv2D<T> _protoConv2;
    private readonly Conv2D<T> _protoConv3;
    private readonly Conv2D<T> _protoOut;
    private readonly int _numPrototypes;

    /// <summary>
    /// Number of mask prototypes.
    /// </summary>
    public int NumPrototypes => _numPrototypes;

    /// <summary>
    /// Creates a new prototype mask head.
    /// </summary>
    /// <param name="inChannels">Number of input feature channels.</param>
    /// <param name="numPrototypes">Number of prototype masks to generate.</param>
    public PrototypeMaskHead(int inChannels, int numPrototypes = 32)
    {
        _numOps = Tensors.Helpers.MathHelper.GetNumericOperations<T>();
        _numPrototypes = numPrototypes;

        _protoConv1 = new Conv2D<T>(inChannels, 256, kernelSize: 3, padding: 1);
        _protoConv2 = new Conv2D<T>(256, 256, kernelSize: 3, padding: 1);
        _protoConv3 = new Conv2D<T>(256, 256, kernelSize: 3, padding: 1);
        _protoOut = new Conv2D<T>(256, numPrototypes, kernelSize: 1);
    }

    /// <summary>
    /// Generates prototype masks from feature map.
    /// </summary>
    /// <param name="features">Feature map [batch, channels, height, width].</param>
    /// <returns>Prototype masks [batch, num_prototypes, height, width].</returns>
    public Tensor<T> GeneratePrototypes(Tensor<T> features)
    {
        var x = ApplyConvReLU(_protoConv1, features);
        x = Upsample2x(x);
        x = ApplyConvReLU(_protoConv2, x);
        x = Upsample2x(x);
        x = ApplyConvReLU(_protoConv3, x);
        x = _protoOut.Forward(x);

        return x;
    }

    /// <summary>
    /// Assembles instance mask from prototypes and coefficients.
    /// </summary>
    /// <param name="prototypes">Prototype masks [1, num_prototypes, h, w].</param>
    /// <param name="coefficients">Mask coefficients [num_prototypes].</param>
    /// <returns>Instance mask [h, w].</returns>
    public Tensor<T> AssembleMask(Tensor<T> prototypes, Tensor<T> coefficients)
    {
        int height = prototypes.Shape[2];
        int width = prototypes.Shape[3];

        var mask = new Tensor<T>(new[] { height, width });

        for (int h = 0; h < height; h++)
        {
            for (int w = 0; w < width; w++)
            {
                double val = 0;
                for (int p = 0; p < _numPrototypes; p++)
                {
                    double proto = _numOps.ToDouble(prototypes[0, p, h, w]);
                    double coef = _numOps.ToDouble(coefficients[p]);
                    val += proto * coef;
                }
                // Apply sigmoid
                mask[h, w] = _numOps.FromDouble(1.0 / (1.0 + Math.Exp(-val)));
            }
        }

        return mask;
    }

    private static Tensor<T> ApplyConvReLU(Conv2D<T> conv, Tensor<T> input)
        => AiDotNetEngine.Current.ReLU(conv.Forward(input));

    // Nearest-neighbour 2x upsampling, through the engine so the step stays on the gradient tape.
    private static Tensor<T> Upsample2x(Tensor<T> input)
        => AiDotNetEngine.Current.Interpolate(
            input, new[] { input.Shape[2] * 2, input.Shape[3] * 2 }, InterpolateMode.Nearest, alignCorners: false);

    public long GetParameterCount() => Parameters.ParameterCount;

    /// <summary>
    /// Writes parameters to binary writer.
    /// </summary>
    public void WriteParameters(BinaryWriter writer)
    {
        writer.Write(_numPrototypes);
        _protoConv1.WriteParameters(writer);
        _protoConv2.WriteParameters(writer);
        _protoConv3.WriteParameters(writer);
        _protoOut.WriteParameters(writer);
    }

    /// <summary>
    /// Reads parameters from binary reader.
    /// </summary>
    public void ReadParameters(BinaryReader reader)
    {
        int numProtos = reader.ReadInt32();
        if (numProtos != _numPrototypes)
        {
            throw new InvalidOperationException(
                $"PrototypeMaskHead configuration mismatch. Expected numPrototypes={_numPrototypes}, got {numProtos}");
        }
        _protoConv1.ReadParameters(reader);
        _protoConv2.ReadParameters(reader);
        _protoConv3.ReadParameters(reader);
        _protoOut.ReadParameters(reader);
    }

    private DelegatingCvParameterModule<T>? _parameters;

    private DelegatingCvParameterModule<T> Parameters
        => _parameters ??= new DelegatingCvParameterModule<T>(
            () => new IParameterSource<T>?[] { _protoConv1, _protoConv2, _protoConv3, _protoOut });

    /// <inheritdoc />
    long IParameterSource<T>.ParameterCount => Parameters.ParameterCount;

    /// <inheritdoc />
    IReadOnlyList<ParameterSlotDescriptor> IParameterLayoutSource.GetParameterLayout() => Parameters.GetParameterLayout();

    /// <inheritdoc />
    Vector<T> IParameterSource<T>.GetParameters() => Parameters.GetParameters();

    /// <inheritdoc />
    void IParameterSource<T>.SetParameters(Vector<T> parameters) => Parameters.SetParameters(parameters);

    /// <inheritdoc />
    IEnumerable<ParameterChunk<T>> IParameterChunkSource<T>.GetParameterStateChunks() => Parameters.GetParameterStateChunks();
}
