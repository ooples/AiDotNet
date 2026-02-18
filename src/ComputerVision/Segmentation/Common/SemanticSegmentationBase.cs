using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks;
using AiDotNet.Optimizers;

namespace AiDotNet.ComputerVision.Segmentation.Common;

/// <summary>
/// Abstract base class for semantic segmentation models that assign a class label to every pixel.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> Semantic segmentation answers "what is this pixel?" for every pixel in an
/// image. This base class provides the shared infrastructure for models like SegFormer, SegNeXt,
/// InternImage, and DiffSeg — all of which produce a per-pixel class label map.
///
/// Extending this class gives you:
/// - Dual-mode support (native training + ONNX inference)
/// - Automatic class map extraction (argmax of logits)
/// - Probability map generation (softmax of logits)
/// - All common serialization and batch handling
/// </para>
/// </remarks>
public abstract class SemanticSegmentationBase<T> : SegmentationModelBase<T>, ISemanticSegmentation<T>
{
    /// <summary>
    /// Initializes the base in native (trainable) mode.
    /// </summary>
    protected SemanticSegmentationBase(
        NeuralNetworkArchitecture<T> architecture,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer,
        ILossFunction<T>? lossFunction,
        int numClasses)
        : base(architecture, optimizer, lossFunction, numClasses)
    {
    }

    /// <summary>
    /// Initializes the base in ONNX (inference-only) mode.
    /// </summary>
    protected SemanticSegmentationBase(
        NeuralNetworkArchitecture<T> architecture,
        string onnxModelPath,
        int numClasses)
        : base(architecture, onnxModelPath, numClasses)
    {
    }

    /// <inheritdoc/>
    public virtual Tensor<T> GetClassMap(Tensor<T> image)
    {
        var logits = Segment(image);
        return ArgmaxAlongClassDim(logits);
    }

    /// <inheritdoc/>
    public virtual Tensor<T> GetProbabilityMap(Tensor<T> image)
    {
        var logits = Segment(image);
        return SoftmaxAlongClassDim(logits);
    }

    /// <summary>
    /// Computes argmax along the class dimension (dim 0 for [C, H, W] or dim 1 for [B, C, H, W]).
    /// </summary>
    protected Tensor<T> ArgmaxAlongClassDim(Tensor<T> logits)
    {
        if (logits.Rank == 3)
        {
            // [C, H, W] → [H, W]
            int c = logits.Shape[0], h = logits.Shape[1], w = logits.Shape[2];
            var result = new Tensor<T>([h, w]);
            for (int row = 0; row < h; row++)
            {
                for (int col = 0; col < w; col++)
                {
                    int bestClass = 0;
                    T bestVal = logits[0, row, col];
                    for (int cls = 1; cls < c; cls++)
                    {
                        T val = logits[cls, row, col];
                        if (NumOps.GreaterThan(val, bestVal))
                        {
                            bestVal = val;
                            bestClass = cls;
                        }
                    }
                    result[row, col] = NumOps.FromDouble(bestClass);
                }
            }
            return result;
        }
        else
        {
            // [B, C, H, W] → [B, H, W]
            int b = logits.Shape[0], c = logits.Shape[1], h = logits.Shape[2], w = logits.Shape[3];
            var result = new Tensor<T>([b, h, w]);
            for (int batch = 0; batch < b; batch++)
            {
                for (int row = 0; row < h; row++)
                {
                    for (int col = 0; col < w; col++)
                    {
                        int bestClass = 0;
                        T bestVal = logits[batch, 0, row, col];
                        for (int cls = 1; cls < c; cls++)
                        {
                            T val = logits[batch, cls, row, col];
                            if (NumOps.GreaterThan(val, bestVal))
                            {
                                bestVal = val;
                                bestClass = cls;
                            }
                        }
                        result[batch, row, col] = NumOps.FromDouble(bestClass);
                    }
                }
            }
            return result;
        }
    }

    /// <summary>
    /// Computes softmax along the class dimension.
    /// </summary>
    protected Tensor<T> SoftmaxAlongClassDim(Tensor<T> logits)
    {
        var result = new Tensor<T>(logits.Shape);

        if (logits.Rank == 3)
        {
            int c = logits.Shape[0], h = logits.Shape[1], w = logits.Shape[2];
            for (int row = 0; row < h; row++)
            {
                for (int col = 0; col < w; col++)
                {
                    // Find max for numerical stability
                    T maxVal = logits[0, row, col];
                    for (int cls = 1; cls < c; cls++)
                    {
                        T val = logits[cls, row, col];
                        if (NumOps.GreaterThan(val, maxVal)) maxVal = val;
                    }

                    // Compute exp(x - max) and sum
                    T sumExp = NumOps.Zero;
                    for (int cls = 0; cls < c; cls++)
                    {
                        T expVal = NumOps.Exp(NumOps.Subtract(logits[cls, row, col], maxVal));
                        result[cls, row, col] = expVal;
                        sumExp = NumOps.Add(sumExp, expVal);
                    }

                    // Normalize
                    for (int cls = 0; cls < c; cls++)
                    {
                        result[cls, row, col] = NumOps.Divide(result[cls, row, col], sumExp);
                    }
                }
            }
        }
        else
        {
            int b = logits.Shape[0], c = logits.Shape[1], h = logits.Shape[2], w = logits.Shape[3];
            for (int batch = 0; batch < b; batch++)
            {
                for (int row = 0; row < h; row++)
                {
                    for (int col = 0; col < w; col++)
                    {
                        T maxVal = logits[batch, 0, row, col];
                        for (int cls = 1; cls < c; cls++)
                        {
                            T val = logits[batch, cls, row, col];
                            if (NumOps.GreaterThan(val, maxVal)) maxVal = val;
                        }

                        T sumExp = NumOps.Zero;
                        for (int cls = 0; cls < c; cls++)
                        {
                            T expVal = NumOps.Exp(NumOps.Subtract(logits[batch, cls, row, col], maxVal));
                            result[batch, cls, row, col] = expVal;
                            sumExp = NumOps.Add(sumExp, expVal);
                        }

                        for (int cls = 0; cls < c; cls++)
                        {
                            result[batch, cls, row, col] = NumOps.Divide(result[batch, cls, row, col], sumExp);
                        }
                    }
                }
            }
        }

        return result;
    }
}
