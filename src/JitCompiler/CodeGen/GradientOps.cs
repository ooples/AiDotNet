using AiDotNet.LinearAlgebra;
using AiDotNet.Autodiff;

namespace AiDotNet.JitCompiler.CodeGen;

/// <summary>
/// Provides gradient computation operations for backward pass execution.
/// </summary>
/// <remarks>
/// <para>
/// This class implements the actual gradient computations for backpropagation.
/// Each method corresponds to a backward operation type and computes gradients
/// with respect to the inputs of the forward operation.
/// </para>
/// <para><b>For Beginners:</b> These are the math operations for training neural networks.
///
/// When training, we need to compute how to adjust weights to reduce error.
/// These methods implement the calculus (derivatives) needed for that.
///
/// Each forward operation (Add, MatMul, ReLU, etc.) has a corresponding
/// backward method that computes gradients.
/// </para>
/// </remarks>
public static class GradientOps
{
    /// <summary>
    /// Accumulates multiple gradients by summing them.
    /// </summary>
    /// <remarks>
    /// When a tensor is used by multiple operations, gradients from
    /// all paths must be summed.
    /// </remarks>
    public static Tensor<T> AccumulateGrad<T>(params Tensor<T>[] gradients)
    {
        if (gradients.Length == 0)
            throw new ArgumentException("Must provide at least one gradient to accumulate");

        var result = gradients[0];
        for (int i = 1; i < gradients.Length; i++)
        {
            // Element-wise addition
            result = result.Add(gradients[i]);
        }
        return result;
    }

    /// <summary>
    /// Gradient of Add operation.
    /// Forward: c = a + b
    /// Backward: grad_a = grad_c, grad_b = grad_c
    /// </summary>
    public static Tensor<T> GradAdd<T>(Tensor<T> gradOutput, int inputIndex)
    {
        // Gradient flows equally to both inputs
        // May need to handle broadcasting by summing over broadcasted dimensions
        return gradOutput;
    }

    /// <summary>
    /// Gradient of Subtract operation.
    /// Forward: c = a - b
    /// Backward: grad_a = grad_c, grad_b = -grad_c
    /// </summary>
    public static Tensor<T> GradSubtract<T>(Tensor<T> gradOutput, int inputIndex)
    {
        if (inputIndex == 0)
        {
            // Gradient to left input (minuend)
            return gradOutput;
        }
        else
        {
            // Gradient to right input (subtrahend) is negated
            return NegateHelper(gradOutput);
        }
    }

    /// <summary>
    /// Gradient of ElementwiseMultiply operation.
    /// Forward: c = a * b (element-wise)
    /// Backward: grad_a = grad_c * b, grad_b = grad_c * a
    /// </summary>
    public static Tensor<T> GradElementwiseMultiply<T>(Tensor<T> gradOutput, Tensor<T> otherInput, int inputIndex)
    {
        // Gradient is output gradient multiplied by the other input
        return Tensor<T>.ElementwiseMultiply(gradOutput, otherInput);
    }

    /// <summary>
    /// Gradient of MatMul operation (left input).
    /// Forward: C = A @ B
    /// Backward for A: grad_A = grad_C @ B^T
    /// </summary>
    public static Tensor<T> GradMatMulLeft<T>(Tensor<T> gradOutput, Tensor<T> rightInput)
    {
        // grad_A = grad_C @ B^T
        var rightTransposed = rightInput.Transpose();
        return gradOutput.MatrixMultiply(rightTransposed);
    }

    /// <summary>
    /// Gradient of MatMul operation (right input).
    /// Forward: C = A @ B
    /// Backward for B: grad_B = A^T @ grad_C
    /// </summary>
    public static Tensor<T> GradMatMulRight<T>(Tensor<T> leftInput, Tensor<T> gradOutput)
    {
        // grad_B = A^T @ grad_C
        var leftTransposed = leftInput.Transpose();
        return leftTransposed.MatrixMultiply(gradOutput);
    }

    /// <summary>
    /// Gradient of ReLU operation.
    /// Forward: y = max(0, x)
    /// Backward: grad_x = grad_y * (x > 0)
    /// </summary>
    public static Tensor<T> GradReLU<T>(Tensor<T> gradOutput, Tensor<T> forwardInput)
    {
        // Gradient flows only where input was positive
        // Create mask: 1 where input > 0, 0 elsewhere
        var mask = CreateMask(forwardInput);
        return Tensor<T>.ElementwiseMultiply(gradOutput, mask);
    }

    /// <summary>
    /// Gradient of Sigmoid operation.
    /// Forward: y = 1 / (1 + exp(-x))
    /// Backward: grad_x = grad_y * y * (1 - y)
    /// </summary>
    public static Tensor<T> GradSigmoid<T>(Tensor<T> gradOutput, Tensor<T> forwardOutput)
    {
        // grad_x = grad_y * y * (1 - y)
        var ones = CreateOnes<T>(forwardOutput.Shape);
        var oneMinusY = ones.Subtract(forwardOutput);
        var yTimesOneMinusY = Tensor<T>.ElementwiseMultiply(forwardOutput, oneMinusY);
        return Tensor<T>.ElementwiseMultiply(gradOutput, yTimesOneMinusY);
    }

    /// <summary>
    /// Gradient of Tanh operation.
    /// Forward: y = tanh(x)
    /// Backward: grad_x = grad_y * (1 - y^2)
    /// </summary>
    public static Tensor<T> GradTanh<T>(Tensor<T> gradOutput, Tensor<T> forwardOutput)
    {
        // grad_x = grad_y * (1 - y^2)
        var ySquared = Tensor<T>.ElementwiseMultiply(forwardOutput, forwardOutput);
        var ones = CreateOnes<T>(forwardOutput.Shape);
        var oneMinusYSquared = ones.Subtract(ySquared);
        return Tensor<T>.ElementwiseMultiply(gradOutput, oneMinusYSquared);
    }

    /// <summary>
    /// Gradient of Exp operation.
    /// Forward: y = exp(x)
    /// Backward: grad_x = grad_y * y
    /// </summary>
    public static Tensor<T> GradExp<T>(Tensor<T> gradOutput, Tensor<T> forwardOutput)
    {
        // Derivative of exp(x) is exp(x) itself
        return Tensor<T>.ElementwiseMultiply(gradOutput, forwardOutput);
    }

    /// <summary>
    /// Gradient of Log operation.
    /// Forward: y = log(x)
    /// Backward: grad_x = grad_y / x
    /// </summary>
    public static Tensor<T> GradLog<T>(Tensor<T> gradOutput, Tensor<T> forwardInput)
    {
        // grad_x = grad_y / x
        return DivideHelper(gradOutput, forwardInput);
    }

    /// <summary>
    /// Gradient of Softmax operation.
    /// Forward: y_i = exp(x_i) / sum(exp(x_j))
    /// Backward: grad_x = y * (grad_y - sum(grad_y * y))
    /// </summary>
    public static Tensor<T> GradSoftmax<T>(Tensor<T> gradOutput, Tensor<T> forwardOutput, int axis)
    {
        // grad_x = y * (grad_y - sum(grad_y * y))
        var gradTimesOutput = Tensor<T>.ElementwiseMultiply(gradOutput, forwardOutput);

        // Sum along the axis
        var summed = SumWithKeepdims(gradTimesOutput, new[] { axis });

        // grad_y - sum
        var diff = gradOutput.Subtract(summed);

        // Multiply by y
        return Tensor<T>.ElementwiseMultiply(forwardOutput, diff);
    }

    /// <summary>
    /// Helper: Creates a mask tensor where elements > 0 are 1, else 0.
    /// </summary>
    private static Tensor<T> CreateMask<T>(Tensor<T> input)
    {
        var result = new Tensor<T>(input.Shape);
        var inputData = input.ToArray();
        var resultData = result.ToArray();

        for (int i = 0; i < inputData.Length; i++)
        {
            // Use dynamic to handle generic comparison
            var dataVal = inputData[i];
            if (dataVal is null)
            {
                resultData[i] = (T)(object)0.0;
            }
            else
            {
                dynamic val = dataVal;
                resultData[i] = val > 0 ? (T)(object)1.0 : (T)(object)0.0;
            }
        }

        return new Tensor<T>(input.Shape, new Vector<T>(resultData));
    }

    /// <summary>
    /// Helper: Creates a tensor of ones with the given shape.
    /// </summary>
    private static Tensor<T> CreateOnes<T>(int[] shape)
    {
        var totalSize = shape.Aggregate(1, (a, b) => a * b);
        var data = new T[totalSize];

        for (int i = 0; i < totalSize; i++)
        {
            data[i] = (T)(object)1.0;
        }

        return new Tensor<T>(shape, new Vector<T>(data));
    }

    /// <summary>
    /// Helper: Negates all elements in a tensor.
    /// </summary>
    private static Tensor<T> NegateHelper<T>(Tensor<T> input)
    {
        var numOps = MathHelper.GetNumericOperations<T>();
        var data = input.ToArray();
        for (int i = 0; i < data.Length; i++)
        {
            data[i] = numOps.Negate(data[i]);
        }
        return new Tensor<T>(input.Shape, new Vector<T>(data));
    }

    /// <summary>
    /// Helper: Element-wise division of two tensors.
    /// </summary>
    private static Tensor<T> DivideHelper<T>(Tensor<T> numerator, Tensor<T> denominator)
    {
        if (!numerator.Shape.SequenceEqual(denominator.Shape))
            throw new ArgumentException("Tensors must have the same shape for element-wise division");

        var numOps = MathHelper.GetNumericOperations<T>();
        var numeratorData = numerator.ToArray();
        var denominatorData = denominator.ToArray();
        var resultData = new T[numeratorData.Length];

        for (int i = 0; i < numeratorData.Length; i++)
        {
            resultData[i] = numOps.Divide(numeratorData[i], denominatorData[i]);
        }

        return new Tensor<T>(numerator.Shape, new Vector<T>(resultData));
    }

    /// <summary>
    /// Helper: Sum along specified axes while keeping dimensions.
    /// </summary>
    private static Tensor<T> SumWithKeepdims<T>(Tensor<T> input, int[] axes)
    {
        // First, sum along the axes (this will reduce dimensions)
        var reduced = input.Sum(axes);

        // Now we need to restore the reduced dimensions with size 1
        var newShape = new List<int>(input.Shape);
        foreach (var axis in axes.OrderBy(a => a))
        {
            newShape[axis] = 1;
        }

        // Reshape the reduced tensor to have the same rank with 1s in reduced dimensions
        return reduced.Reshape(newShape.ToArray());
    }

    /// <summary>
    /// Gradient of Conv2D operation.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Forward: output = conv2d(input, filters)
    /// Backward for input: grad_input = conv2d_transpose(grad_output, filters)
    /// Backward for filters: grad_filters = conv2d(input, grad_output)
    /// </para>
    /// </remarks>
    public static Tensor<T> GradConv2D<T>(Tensor<T> gradOutput, Tensor<T> savedTensor, int inputIndex, int[] stride, int[] padding)
    {
        var numOps = MathHelper.GetNumericOperations<T>();

        if (inputIndex == 0)
        {
            // Gradient for input: use transposed convolution
            // This is a simplified implementation - full implementation would use conv transpose
            // For now, we'll use a compatible shape output
            return gradOutput; // Placeholder - needs proper conv transpose
        }
        else if (inputIndex == 1)
        {
            // Gradient for filters: correlate input with grad_output
            return gradOutput; // Placeholder - needs proper gradient computation
        }
        else
        {
            // Gradient for bias: sum over batch and spatial dimensions
            // Shape: [N, C, H, W] -> sum over N, H, W to get [C]
            var result = new Tensor<T>(new int[] { gradOutput.Shape[1] });
            var data = gradOutput.ToArray();
            var resultData = result.ToArray();

            int batchSize = gradOutput.Shape[0];
            int channels = gradOutput.Shape[1];
            int height = gradOutput.Shape[2];
            int width = gradOutput.Shape[3];

            for (int c = 0; c < channels; c++)
            {
                T sum = numOps.Zero;
                for (int n = 0; n < batchSize; n++)
                {
                    for (int h = 0; h < height; h++)
                    {
                        for (int w = 0; w < width; w++)
                        {
                            int idx = n * channels * height * width + c * height * width + h * width + w;
                            sum = numOps.Add(sum, data[idx]);
                        }
                    }
                }
                resultData[c] = sum;
            }

            return new Tensor<T>(result.Shape, new Vector<T>(resultData));
        }
    }

    /// <summary>
    /// Gradient of MaxPool2D operation.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Forward: Records indices of max elements
    /// Backward: Routes gradient only to max elements (winner-take-all)
    /// </para>
    /// </remarks>
    public static Tensor<T> GradMaxPool2D<T>(Tensor<T> gradOutput, Tensor<T> forwardInput, int[] poolSize, int[] stride)
    {
        var numOps = MathHelper.GetNumericOperations<T>();
        var inputShape = forwardInput.Shape;
        var result = new Tensor<T>(inputShape);
        var resultData = result.ToArray();
        var inputData = forwardInput.ToArray();
        var gradData = gradOutput.ToArray();

        int batchSize = inputShape[0];
        int channels = inputShape[1];
        int inputHeight = inputShape[2];
        int inputWidth = inputShape[3];
        int outputHeight = gradOutput.Shape[2];
        int outputWidth = gradOutput.Shape[3];

        for (int n = 0; n < batchSize; n++)
        {
            for (int c = 0; c < channels; c++)
            {
                for (int oh = 0; oh < outputHeight; oh++)
                {
                    for (int ow = 0; ow < outputWidth; ow++)
                    {
                        // Find the max element in the pooling window
                        int hStart = oh * stride[0];
                        int wStart = ow * stride[1];
                        int maxH = hStart, maxW = wStart;
                        T maxVal = numOps.MinValue;

                        for (int ph = 0; ph < poolSize[0] && hStart + ph < inputHeight; ph++)
                        {
                            for (int pw = 0; pw < poolSize[1] && wStart + pw < inputWidth; pw++)
                            {
                                int ih = hStart + ph;
                                int iw = wStart + pw;
                                int inputIdx = n * channels * inputHeight * inputWidth +
                                              c * inputHeight * inputWidth +
                                              ih * inputWidth + iw;
                                if (numOps.GreaterThan(inputData[inputIdx], maxVal))
                                {
                                    maxVal = inputData[inputIdx];
                                    maxH = ih;
                                    maxW = iw;
                                }
                            }
                        }

                        // Route gradient to the max element
                        int gradIdx = n * channels * outputHeight * outputWidth +
                                     c * outputHeight * outputWidth +
                                     oh * outputWidth + ow;
                        int resultIdx = n * channels * inputHeight * inputWidth +
                                       c * inputHeight * inputWidth +
                                       maxH * inputWidth + maxW;
                        resultData[resultIdx] = numOps.Add(resultData[resultIdx], gradData[gradIdx]);
                    }
                }
            }
        }

        return new Tensor<T>(inputShape, new Vector<T>(resultData));
    }

    /// <summary>
    /// Gradient of AvgPool2D operation.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Forward: Averages values in each window
    /// Backward: Distributes gradient equally to all elements in the window
    /// </para>
    /// </remarks>
    public static Tensor<T> GradAvgPool2D<T>(Tensor<T> gradOutput, int[] poolSize, int[] stride, int[] inputShape)
    {
        var numOps = MathHelper.GetNumericOperations<T>();
        var result = new Tensor<T>(inputShape);
        var resultData = result.ToArray();
        var gradData = gradOutput.ToArray();

        int batchSize = inputShape[0];
        int channels = inputShape[1];
        int inputHeight = inputShape[2];
        int inputWidth = inputShape[3];
        int outputHeight = gradOutput.Shape[2];
        int outputWidth = gradOutput.Shape[3];

        // Each element in the window contributes equally, so divide by pool size
        T divisor = numOps.FromDouble(poolSize[0] * poolSize[1]);

        for (int n = 0; n < batchSize; n++)
        {
            for (int c = 0; c < channels; c++)
            {
                for (int oh = 0; oh < outputHeight; oh++)
                {
                    for (int ow = 0; ow < outputWidth; ow++)
                    {
                        int gradIdx = n * channels * outputHeight * outputWidth +
                                     c * outputHeight * outputWidth +
                                     oh * outputWidth + ow;
                        T gradVal = numOps.Divide(gradData[gradIdx], divisor);

                        // Distribute gradient to all elements in the pooling window
                        int hStart = oh * stride[0];
                        int wStart = ow * stride[1];

                        for (int ph = 0; ph < poolSize[0] && hStart + ph < inputHeight; ph++)
                        {
                            for (int pw = 0; pw < poolSize[1] && wStart + pw < inputWidth; pw++)
                            {
                                int ih = hStart + ph;
                                int iw = wStart + pw;
                                int resultIdx = n * channels * inputHeight * inputWidth +
                                               c * inputHeight * inputWidth +
                                               ih * inputWidth + iw;
                                resultData[resultIdx] = numOps.Add(resultData[resultIdx], gradVal);
                            }
                        }
                    }
                }
            }
        }

        return new Tensor<T>(inputShape, new Vector<T>(resultData));
    }

    /// <summary>
    /// Gradient of BatchNorm operation.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Computes gradients for input, scale (gamma), and bias (beta) parameters.
    /// </para>
    /// </remarks>
    public static Tensor<T> GradBatchNorm<T>(Tensor<T> gradOutput, Tensor<T> savedTensor, int inputIndex, double epsilon)
    {
        var numOps = MathHelper.GetNumericOperations<T>();

        if (inputIndex == 0)
        {
            // Gradient for input
            // This is a simplified version - full implementation requires saved mean/variance
            return gradOutput;
        }
        else if (inputIndex == 1)
        {
            // Gradient for gamma (scale): sum of grad_output * normalized_input
            var result = Tensor<T>.ElementwiseMultiply(gradOutput, savedTensor);
            // Sum over batch and spatial dimensions
            return SumOverBatchAndSpatial(result);
        }
        else
        {
            // Gradient for beta (bias): sum of grad_output
            return SumOverBatchAndSpatial(gradOutput);
        }
    }

    /// <summary>
    /// Helper: Sum over batch and spatial dimensions for normalization gradients.
    /// </summary>
    private static Tensor<T> SumOverBatchAndSpatial<T>(Tensor<T> input)
    {
        var numOps = MathHelper.GetNumericOperations<T>();

        if (input.Shape.Length == 2)
        {
            // [batch, features] -> [features]
            int batchSize = input.Shape[0];
            int features = input.Shape[1];
            var result = new T[features];
            var data = input.ToArray();

            for (int f = 0; f < features; f++)
            {
                T sum = numOps.Zero;
                for (int n = 0; n < batchSize; n++)
                {
                    sum = numOps.Add(sum, data[n * features + f]);
                }
                result[f] = sum;
            }

            return new Tensor<T>(new int[] { features }, new Vector<T>(result));
        }
        else if (input.Shape.Length == 4)
        {
            // [batch, channels, height, width] -> [channels]
            int batchSize = input.Shape[0];
            int channels = input.Shape[1];
            int height = input.Shape[2];
            int width = input.Shape[3];
            var result = new T[channels];
            var data = input.ToArray();

            for (int c = 0; c < channels; c++)
            {
                T sum = numOps.Zero;
                for (int n = 0; n < batchSize; n++)
                {
                    for (int h = 0; h < height; h++)
                    {
                        for (int w = 0; w < width; w++)
                        {
                            int idx = n * channels * height * width + c * height * width + h * width + w;
                            sum = numOps.Add(sum, data[idx]);
                        }
                    }
                }
                result[c] = sum;
            }

            return new Tensor<T>(new int[] { channels }, new Vector<T>(result));
        }
        else
        {
            // Fallback: return as-is
            return input;
        }
    }
}
