using AiDotNet.ComputerVision.Detection.Backbones;
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
public class MaskHead<T>
{
    private readonly INumericOperations<T> _numOps;
    private readonly Conv2D<T> _conv1;
    private readonly Conv2D<T> _conv2;
    private readonly Conv2D<T> _conv3;
    private readonly Conv2D<T> _conv4;
    private readonly Conv2D<T> _deconv;
    private readonly Conv2D<T> _predictor;
    private readonly int _numClasses;
    private readonly int _maskResolution;

    /// <summary>
    /// Creates a new mask head.
    /// </summary>
    /// <param name="inChannels">Number of input channels from RoI features.</param>
    /// <param name="numClasses">Number of classes to predict.</param>
    /// <param name="maskResolution">Output mask resolution.</param>
    public MaskHead(int inChannels, int numClasses, int maskResolution = 28)
    {
        _numOps = Tensors.Helpers.MathHelper.GetNumericOperations<T>();
        _numClasses = numClasses;
        _maskResolution = maskResolution;

        // Feature processing layers
        _conv1 = new Conv2D<T>(inChannels, 256, kernelSize: 3, padding: 1);
        _conv2 = new Conv2D<T>(256, 256, kernelSize: 3, padding: 1);
        _conv3 = new Conv2D<T>(256, 256, kernelSize: 3, padding: 1);
        _conv4 = new Conv2D<T>(256, 256, kernelSize: 3, padding: 1);

        // Upsampling layer (2x)
        _deconv = new Conv2D<T>(256, 256, kernelSize: 2, stride: 1);

        // Mask predictor
        _predictor = new Conv2D<T>(256, numClasses, kernelSize: 1);
    }

    /// <summary>
    /// Forward pass to predict masks from RoI features.
    /// </summary>
    /// <param name="roiFeatures">RoI-pooled features [num_rois, channels, height, width].</param>
    /// <returns>Mask predictions [num_rois, num_classes, mask_h, mask_w].</returns>
    public Tensor<T> Forward(Tensor<T> roiFeatures)
    {
        // Apply conv layers with ReLU
        var x = ApplyConvReLU(_conv1, roiFeatures);
        x = ApplyConvReLU(_conv2, x);
        x = ApplyConvReLU(_conv3, x);
        x = ApplyConvReLU(_conv4, x);

        // Upsample
        x = Upsample2x(x);
        x = ApplyConvReLU(_deconv, x);

        // Predict masks
        var masks = _predictor.Forward(x);

        return masks;
    }

    /// <summary>
    /// Predicts mask for a single RoI and class.
    /// </summary>
    /// <param name="roiFeatures">Features for single RoI [1, channels, h, w].</param>
    /// <param name="classId">Class ID to predict mask for.</param>
    /// <returns>Binary mask [mask_h, mask_w].</returns>
    public Tensor<T> PredictMask(Tensor<T> roiFeatures, int classId)
    {
        var allMasks = Forward(roiFeatures);

        // Extract mask for specific class
        int height = allMasks.Shape[2];
        int width = allMasks.Shape[3];

        var mask = new Tensor<T>(new[] { height, width });

        for (int h = 0; h < height; h++)
        {
            for (int w = 0; w < width; w++)
            {
                double val = _numOps.ToDouble(allMasks[0, classId, h, w]);
                // Apply sigmoid
                mask[h, w] = _numOps.FromDouble(1.0 / (1.0 + Math.Exp(-val)));
            }
        }

        return mask;
    }

    private Tensor<T> ApplyConvReLU(Conv2D<T> conv, Tensor<T> input)
    {
        var output = conv.Forward(input);

        for (int i = 0; i < output.Length; i++)
        {
            double val = _numOps.ToDouble(output[i]);
            output[i] = _numOps.FromDouble(Math.Max(0, val));
        }

        return output;
    }

    private Tensor<T> Upsample2x(Tensor<T> input)
    {
        int batch = input.Shape[0];
        int channels = input.Shape[1];
        int height = input.Shape[2];
        int width = input.Shape[3];

        int newH = height * 2;
        int newW = width * 2;

        var output = new Tensor<T>(new[] { batch, channels, newH, newW });

        for (int b = 0; b < batch; b++)
        {
            for (int c = 0; c < channels; c++)
            {
                for (int h = 0; h < newH; h++)
                {
                    for (int w = 0; w < newW; w++)
                    {
                        // Bilinear interpolation
                        double srcY = (double)h / newH * height;
                        double srcX = (double)w / newW * width;

                        int y0 = (int)Math.Floor(srcY);
                        int x0 = (int)Math.Floor(srcX);
                        int y1 = Math.Min(y0 + 1, height - 1);
                        int x1 = Math.Min(x0 + 1, width - 1);

                        double wy1 = srcY - y0;
                        double wy0 = 1.0 - wy1;
                        double wx1 = srcX - x0;
                        double wx0 = 1.0 - wx1;

                        double v00 = _numOps.ToDouble(input[b, c, y0, x0]);
                        double v01 = _numOps.ToDouble(input[b, c, y0, x1]);
                        double v10 = _numOps.ToDouble(input[b, c, y1, x0]);
                        double v11 = _numOps.ToDouble(input[b, c, y1, x1]);

                        double val = wy0 * (wx0 * v00 + wx1 * v01) + wy1 * (wx0 * v10 + wx1 * v11);
                        output[b, c, h, w] = _numOps.FromDouble(val);
                    }
                }
            }
        }

        return output;
    }

    /// <summary>
    /// Gets the total parameter count.
    /// </summary>
    public long GetParameterCount()
    {
        return _conv1.GetParameterCount() +
               _conv2.GetParameterCount() +
               _conv3.GetParameterCount() +
               _conv4.GetParameterCount() +
               _deconv.GetParameterCount() +
               _predictor.GetParameterCount();
    }
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
public class PrototypeMaskHead<T>
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

    private Tensor<T> ApplyConvReLU(Conv2D<T> conv, Tensor<T> input)
    {
        var output = conv.Forward(input);

        for (int i = 0; i < output.Length; i++)
        {
            double val = _numOps.ToDouble(output[i]);
            output[i] = _numOps.FromDouble(Math.Max(0, val));
        }

        return output;
    }

    private Tensor<T> Upsample2x(Tensor<T> input)
    {
        int batch = input.Shape[0];
        int channels = input.Shape[1];
        int height = input.Shape[2];
        int width = input.Shape[3];

        int newH = height * 2;
        int newW = width * 2;

        var output = new Tensor<T>(new[] { batch, channels, newH, newW });

        for (int b = 0; b < batch; b++)
        {
            for (int c = 0; c < channels; c++)
            {
                for (int h = 0; h < newH; h++)
                {
                    for (int w = 0; w < newW; w++)
                    {
                        // Nearest neighbor upsampling for speed
                        int srcH = h / 2;
                        int srcW = w / 2;
                        output[b, c, h, w] = input[b, c, srcH, srcW];
                    }
                }
            }
        }

        return output;
    }

    /// <summary>
    /// Gets the total parameter count.
    /// </summary>
    public long GetParameterCount()
    {
        return _protoConv1.GetParameterCount() +
               _protoConv2.GetParameterCount() +
               _protoConv3.GetParameterCount() +
               _protoOut.GetParameterCount();
    }
}
