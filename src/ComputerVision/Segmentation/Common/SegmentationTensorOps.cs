using AiDotNet.Helpers;

namespace AiDotNet.ComputerVision.Segmentation.Common;

/// <summary>
/// Static helper methods for common segmentation tensor operations (argmax, softmax, etc.)
/// used by models implementing segmentation interfaces.
/// </summary>
public static class SegmentationTensorOps
{
    /// <summary>
    /// Computes argmax along the class dimension, producing a per-pixel class index map.
    /// </summary>
    /// <typeparam name="T">The numeric type.</typeparam>
    /// <param name="logits">Logits tensor [C, H, W] or [B, C, H, W].</param>
    /// <returns>Class index map [H, W] or [B, H, W].</returns>
    public static Tensor<T> ArgmaxAlongClassDim<T>(Tensor<T> logits)
    {
        var numOps = MathHelper.GetNumericOperations<T>();

        if (logits.Rank == 3)
        {
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
                        if (numOps.GreaterThan(val, bestVal))
                        {
                            bestVal = val;
                            bestClass = cls;
                        }
                    }
                    result[row, col] = numOps.FromDouble(bestClass);
                }
            }
            return result;
        }
        else
        {
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
                            if (numOps.GreaterThan(val, bestVal))
                            {
                                bestVal = val;
                                bestClass = cls;
                            }
                        }
                        result[batch, row, col] = numOps.FromDouble(bestClass);
                    }
                }
            }
            return result;
        }
    }

    /// <summary>
    /// Computes softmax along the class dimension, producing per-pixel probabilities.
    /// </summary>
    /// <typeparam name="T">The numeric type.</typeparam>
    /// <param name="logits">Logits tensor [C, H, W] or [B, C, H, W].</param>
    /// <returns>Probability map [C, H, W] or [B, C, H, W] with values in [0, 1].</returns>
    public static Tensor<T> SoftmaxAlongClassDim<T>(Tensor<T> logits)
    {
        var numOps = MathHelper.GetNumericOperations<T>();
        var result = new Tensor<T>(logits.Shape);

        if (logits.Rank == 3)
        {
            int c = logits.Shape[0], h = logits.Shape[1], w = logits.Shape[2];
            for (int row = 0; row < h; row++)
            {
                for (int col = 0; col < w; col++)
                {
                    T maxVal = logits[0, row, col];
                    for (int cls = 1; cls < c; cls++)
                    {
                        T val = logits[cls, row, col];
                        if (numOps.GreaterThan(val, maxVal)) maxVal = val;
                    }

                    T sumExp = numOps.Zero;
                    for (int cls = 0; cls < c; cls++)
                    {
                        T expVal = numOps.Exp(numOps.Subtract(logits[cls, row, col], maxVal));
                        result[cls, row, col] = expVal;
                        sumExp = numOps.Add(sumExp, expVal);
                    }

                    for (int cls = 0; cls < c; cls++)
                    {
                        result[cls, row, col] = numOps.Divide(result[cls, row, col], sumExp);
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
                            if (numOps.GreaterThan(val, maxVal)) maxVal = val;
                        }

                        T sumExp = numOps.Zero;
                        for (int cls = 0; cls < c; cls++)
                        {
                            T expVal = numOps.Exp(numOps.Subtract(logits[batch, cls, row, col], maxVal));
                            result[batch, cls, row, col] = expVal;
                            sumExp = numOps.Add(sumExp, expVal);
                        }

                        for (int cls = 0; cls < c; cls++)
                        {
                            result[batch, cls, row, col] = numOps.Divide(result[batch, cls, row, col], sumExp);
                        }
                    }
                }
            }
        }

        return result;
    }
}
