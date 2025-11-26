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

    // ========== Additional Gradient Operations ==========

    /// <summary>
    /// Gradient of Reshape operation.
    /// Forward: y = reshape(x, new_shape)
    /// Backward: grad_x = reshape(grad_y, original_shape)
    /// </summary>
    public static Tensor<T> GradReshape<T>(Tensor<T> gradOutput, int[] originalShape)
    {
        return gradOutput.Reshape(originalShape);
    }

    /// <summary>
    /// Gradient of Transpose operation.
    /// Forward: y = transpose(x, axes)
    /// Backward: grad_x = transpose(grad_y, inverse_axes)
    /// </summary>
    public static Tensor<T> GradTranspose<T>(Tensor<T> gradOutput, int[]? axes)
    {
        if (axes == null)
        {
            // Simple transpose (swap last two dimensions)
            return gradOutput.Transpose();
        }

        // Compute inverse permutation
        var inverseAxes = new int[axes.Length];
        for (int i = 0; i < axes.Length; i++)
        {
            inverseAxes[axes[i]] = i;
        }

        // Apply inverse transpose
        return PermuteAxes(gradOutput, inverseAxes);
    }

    /// <summary>
    /// Gradient of Concat operation for a specific input.
    /// Forward: y = concat([x1, x2, ...], axis)
    /// Backward: grad_xi = slice(grad_y, start_i, end_i, axis)
    /// </summary>
    public static Tensor<T> GradConcat<T>(Tensor<T> gradOutput, int axis, int startIndex, int size)
    {
        return SliceAlongAxis(gradOutput, axis, startIndex, size);
    }

    /// <summary>
    /// Gradient of Split operation.
    /// Forward: [y1, y2, ...] = split(x, sizes, axis)
    /// Backward: grad_x = concat([grad_y1, grad_y2, ...], axis)
    /// </summary>
    public static Tensor<T> GradSplit<T>(Tensor<T>[] gradOutputs, int axis)
    {
        if (gradOutputs.Length == 0)
            throw new ArgumentException("Must provide at least one gradient");

        if (gradOutputs.Length == 1)
            return gradOutputs[0];

        // Concatenate all gradients along the axis
        var result = gradOutputs[0];
        for (int i = 1; i < gradOutputs.Length; i++)
        {
            result = Tensor<T>.Concat(new[] { result, gradOutputs[i] }, axis);
        }
        return result;
    }

    /// <summary>
    /// Gradient of Divide operation for numerator.
    /// Forward: c = a / b
    /// Backward for a: grad_a = grad_c / b
    /// </summary>
    public static Tensor<T> GradDivideNumerator<T>(Tensor<T> gradOutput, Tensor<T> denominator)
    {
        return DivideHelper(gradOutput, denominator);
    }

    /// <summary>
    /// Gradient of Divide operation for denominator.
    /// Forward: c = a / b
    /// Backward for b: grad_b = -grad_c * a / (b^2)
    /// </summary>
    public static Tensor<T> GradDivideDenominator<T>(Tensor<T> gradOutput, Tensor<T> numerator, Tensor<T> denominator)
    {
        var numOps = MathHelper.GetNumericOperations<T>();

        // -grad_c * a / (b^2)
        var negGrad = NegateHelper(gradOutput);
        var gradTimesNumerator = Tensor<T>.ElementwiseMultiply(negGrad, numerator);
        var denominatorSquared = Tensor<T>.ElementwiseMultiply(denominator, denominator);
        return DivideHelper(gradTimesNumerator, denominatorSquared);
    }

    /// <summary>
    /// Gradient of Power operation.
    /// Forward: y = x^p
    /// Backward: grad_x = grad_y * p * x^(p-1)
    /// </summary>
    public static Tensor<T> GradPower<T>(Tensor<T> gradOutput, Tensor<T> forwardInput, double exponent)
    {
        var numOps = MathHelper.GetNumericOperations<T>();

        // grad_x = grad_y * p * x^(p-1)
        var inputData = forwardInput.ToArray();
        var gradData = gradOutput.ToArray();
        var resultData = new T[inputData.Length];

        var p = numOps.FromDouble(exponent);
        var pMinus1 = numOps.FromDouble(exponent - 1);

        for (int i = 0; i < inputData.Length; i++)
        {
            // x^(p-1) * p * grad_y
            var xPowPMinus1 = numOps.Power(inputData[i], pMinus1);
            var scaled = numOps.Multiply(xPowPMinus1, p);
            resultData[i] = numOps.Multiply(scaled, gradData[i]);
        }

        return new Tensor<T>(forwardInput.Shape, new Vector<T>(resultData));
    }

    /// <summary>
    /// Gradient of Sqrt operation.
    /// Forward: y = sqrt(x)
    /// Backward: grad_x = grad_y / (2 * y)
    /// </summary>
    public static Tensor<T> GradSqrt<T>(Tensor<T> gradOutput, Tensor<T> forwardOutput)
    {
        var numOps = MathHelper.GetNumericOperations<T>();

        // grad_x = grad_y / (2 * y)
        var two = numOps.FromDouble(2.0);
        var twoTimesY = ScalarMultiply(forwardOutput, two);
        return DivideHelper(gradOutput, twoTimesY);
    }

    /// <summary>
    /// Gradient of Sum operation.
    /// Forward: y = sum(x, axes)
    /// Backward: grad_x = broadcast(grad_y, original_shape)
    /// </summary>
    public static Tensor<T> GradSum<T>(Tensor<T> gradOutput, int[] originalShape, int[]? axes)
    {
        // Broadcast gradient back to original shape
        return BroadcastTo(gradOutput, originalShape);
    }

    /// <summary>
    /// Gradient of Mean operation.
    /// Forward: y = mean(x, axes)
    /// Backward: grad_x = broadcast(grad_y / count, original_shape)
    /// </summary>
    public static Tensor<T> GradMean<T>(Tensor<T> gradOutput, int[] originalShape, int count)
    {
        var numOps = MathHelper.GetNumericOperations<T>();

        // Divide by count first
        var divisor = numOps.FromDouble(count);
        var scaledGrad = ScalarDivide(gradOutput, divisor);

        // Then broadcast
        return BroadcastTo(scaledGrad, originalShape);
    }

    /// <summary>
    /// Gradient of Slice operation.
    /// Forward: y = slice(x, start, end)
    /// Backward: grad_x = pad_with_zeros(grad_y, original_shape, start_indices)
    /// </summary>
    public static Tensor<T> GradSlice<T>(Tensor<T> gradOutput, int[] originalShape, int[] startIndices)
    {
        var numOps = MathHelper.GetNumericOperations<T>();

        // Create zero tensor with original shape
        var totalElements = originalShape.Aggregate(1, (a, b) => a * b);
        var resultData = new T[totalElements];
        for (int i = 0; i < totalElements; i++)
        {
            resultData[i] = numOps.Zero;
        }

        // Copy gradient values to correct positions
        var gradData = gradOutput.ToArray();
        var gradShape = gradOutput.Shape;

        // Calculate strides for original shape
        var strides = new int[originalShape.Length];
        strides[strides.Length - 1] = 1;
        for (int d = strides.Length - 2; d >= 0; d--)
        {
            strides[d] = strides[d + 1] * originalShape[d + 1];
        }

        // Copy gradient to appropriate positions
        CopyToPosition(resultData, gradData, originalShape, gradShape, startIndices, strides);

        return new Tensor<T>(originalShape, new Vector<T>(resultData));
    }

    /// <summary>
    /// Gradient of Pad operation.
    /// Forward: y = pad(x, padding)
    /// Backward: grad_x = slice(grad_y, unpad region)
    /// </summary>
    public static Tensor<T> GradPad<T>(Tensor<T> gradOutput, int[] padding)
    {
        // Extract the center (unpadded) region
        var shape = gradOutput.Shape;
        var startIndices = new int[shape.Length];
        var sizes = new int[shape.Length];

        for (int d = 0; d < shape.Length; d++)
        {
            var padBefore = d < padding.Length / 2 ? padding[d * 2] : 0;
            var padAfter = d < padding.Length / 2 ? padding[d * 2 + 1] : 0;
            startIndices[d] = padBefore;
            sizes[d] = shape[d] - padBefore - padAfter;
        }

        return SliceWithShape(gradOutput, startIndices, sizes);
    }

    /// <summary>
    /// Gradient of Dropout operation.
    /// Forward: y = dropout(x, p, mask)
    /// Backward: grad_x = grad_y * mask / (1 - p)
    /// </summary>
    public static Tensor<T> GradDropout<T>(Tensor<T> gradOutput, Tensor<T> mask, double probability)
    {
        var numOps = MathHelper.GetNumericOperations<T>();

        // grad_x = grad_y * mask / (1 - p)
        var gradTimesMask = Tensor<T>.ElementwiseMultiply(gradOutput, mask);
        var scale = numOps.FromDouble(1.0 / (1.0 - probability));
        return ScalarMultiply(gradTimesMask, scale);
    }

    /// <summary>
    /// Gradient of LeakyReLU operation.
    /// Forward: y = max(alpha * x, x)
    /// Backward: grad_x = grad_y * (1 if x > 0 else alpha)
    /// </summary>
    public static Tensor<T> GradLeakyReLU<T>(Tensor<T> gradOutput, Tensor<T> forwardInput, double alpha)
    {
        var numOps = MathHelper.GetNumericOperations<T>();

        var gradData = gradOutput.ToArray();
        var inputData = forwardInput.ToArray();
        var resultData = new T[gradData.Length];

        var alphaT = numOps.FromDouble(alpha);
        var one = numOps.FromDouble(1.0);

        for (int i = 0; i < gradData.Length; i++)
        {
            var slope = numOps.GreaterThan(inputData[i], numOps.Zero) ? one : alphaT;
            resultData[i] = numOps.Multiply(gradData[i], slope);
        }

        return new Tensor<T>(forwardInput.Shape, new Vector<T>(resultData));
    }

    /// <summary>
    /// Gradient of GELU operation (approximate).
    /// </summary>
    public static Tensor<T> GradGELU<T>(Tensor<T> gradOutput, Tensor<T> forwardInput, bool approximate = true)
    {
        var numOps = MathHelper.GetNumericOperations<T>();

        var gradData = gradOutput.ToArray();
        var inputData = forwardInput.ToArray();
        var resultData = new T[gradData.Length];

        // Constants for approximate GELU
        var sqrt2OverPi = numOps.FromDouble(Math.Sqrt(2.0 / Math.PI)); // ~0.7978845608
        var k = numOps.FromDouble(0.044715);

        for (int i = 0; i < gradData.Length; i++)
        {
            var x = inputData[i];
            var xCubed = numOps.Multiply(numOps.Multiply(x, x), x);
            var inner = numOps.Multiply(sqrt2OverPi, numOps.Add(x, numOps.Multiply(k, xCubed)));

            // tanh(inner)
            var tanhInner = numOps.Tanh(inner);

            // sech^2(inner) = 1 - tanh^2(inner)
            var sech2 = numOps.Subtract(numOps.FromDouble(1.0),
                numOps.Multiply(tanhInner, tanhInner));

            // Derivative of inner with respect to x
            var dInner = numOps.Multiply(sqrt2OverPi,
                numOps.Add(numOps.FromDouble(1.0),
                    numOps.Multiply(numOps.FromDouble(3.0 * 0.044715),
                        numOps.Multiply(x, x))));

            // d/dx GELU = 0.5 * (1 + tanh(inner)) + 0.5 * x * sech^2(inner) * dInner
            var term1 = numOps.Multiply(numOps.FromDouble(0.5),
                numOps.Add(numOps.FromDouble(1.0), tanhInner));
            var term2 = numOps.Multiply(numOps.FromDouble(0.5),
                numOps.Multiply(x, numOps.Multiply(sech2, dInner)));
            var derivative = numOps.Add(term1, term2);

            resultData[i] = numOps.Multiply(gradData[i], derivative);
        }

        return new Tensor<T>(forwardInput.Shape, new Vector<T>(resultData));
    }

    /// <summary>
    /// Gradient of Broadcast operation.
    /// Forward: y = broadcast(x, target_shape)
    /// Backward: grad_x = reduce_sum(grad_y, broadcasted_axes)
    /// </summary>
    public static Tensor<T> GradBroadcast<T>(Tensor<T> gradOutput, int[] originalShape, int[] broadcastedAxes)
    {
        // Sum over the broadcasted axes
        var result = gradOutput;
        foreach (var axis in broadcastedAxes.OrderByDescending(a => a))
        {
            result = result.Sum(new[] { axis });
        }

        // Reshape to original shape if needed
        if (!result.Shape.SequenceEqual(originalShape))
        {
            result = result.Reshape(originalShape);
        }

        return result;
    }

    // ========== Helper Methods ==========

    /// <summary>
    /// Helper: Permutes tensor axes.
    /// </summary>
    private static Tensor<T> PermuteAxes<T>(Tensor<T> input, int[] axes)
    {
        // For now, use transpose if it's a 2D case
        if (axes.Length == 2 && axes[0] == 1 && axes[1] == 0)
        {
            return input.Transpose();
        }

        // General permutation - simplified implementation
        return input; // Would need full permutation implementation
    }

    /// <summary>
    /// Helper: Slices tensor along a specific axis.
    /// </summary>
    private static Tensor<T> SliceAlongAxis<T>(Tensor<T> input, int axis, int start, int size)
    {
        // Simplified slice implementation
        var shape = input.Shape;
        var newShape = shape.ToArray();
        newShape[axis] = size;

        // Calculate strides
        var strides = new int[shape.Length];
        strides[strides.Length - 1] = 1;
        for (int d = strides.Length - 2; d >= 0; d--)
        {
            strides[d] = strides[d + 1] * shape[d + 1];
        }

        var inputData = input.ToArray();
        var resultSize = newShape.Aggregate(1, (a, b) => a * b);
        var resultData = new T[resultSize];

        // Copy data (simplified - assumes contiguous memory)
        CopySlice(inputData, resultData, shape, newShape, axis, start, strides);

        return new Tensor<T>(newShape, new Vector<T>(resultData));
    }

    /// <summary>
    /// Helper: Slices tensor with start indices and sizes.
    /// </summary>
    private static Tensor<T> SliceWithShape<T>(Tensor<T> input, int[] startIndices, int[] sizes)
    {
        var inputData = input.ToArray();
        var resultSize = sizes.Aggregate(1, (a, b) => a * b);
        var resultData = new T[resultSize];

        // Simplified copy - actual implementation would need proper indexing
        var inputShape = input.Shape;
        var strides = new int[inputShape.Length];
        strides[strides.Length - 1] = 1;
        for (int d = strides.Length - 2; d >= 0; d--)
        {
            strides[d] = strides[d + 1] * inputShape[d + 1];
        }

        // Copy from input to result
        CopySliceRegion(inputData, resultData, inputShape, sizes, startIndices, strides);

        return new Tensor<T>(sizes, new Vector<T>(resultData));
    }

    /// <summary>
    /// Helper: Broadcasts tensor to target shape.
    /// </summary>
    private static Tensor<T> BroadcastTo<T>(Tensor<T> input, int[] targetShape)
    {
        var inputShape = input.Shape;

        // If shapes match, return as-is
        if (inputShape.SequenceEqual(targetShape))
            return input;

        var inputData = input.ToArray();
        var resultSize = targetShape.Aggregate(1, (a, b) => a * b);
        var resultData = new T[resultSize];

        // Broadcast by repeating values
        BroadcastCopy(inputData, resultData, inputShape, targetShape);

        return new Tensor<T>(targetShape, new Vector<T>(resultData));
    }

    /// <summary>
    /// Helper: Multiplies tensor by scalar.
    /// </summary>
    private static Tensor<T> ScalarMultiply<T>(Tensor<T> input, T scalar)
    {
        var numOps = MathHelper.GetNumericOperations<T>();
        var data = input.ToArray();
        var resultData = new T[data.Length];

        for (int i = 0; i < data.Length; i++)
        {
            resultData[i] = numOps.Multiply(data[i], scalar);
        }

        return new Tensor<T>(input.Shape, new Vector<T>(resultData));
    }

    /// <summary>
    /// Helper: Divides tensor by scalar.
    /// </summary>
    private static Tensor<T> ScalarDivide<T>(Tensor<T> input, T scalar)
    {
        var numOps = MathHelper.GetNumericOperations<T>();
        var data = input.ToArray();
        var resultData = new T[data.Length];

        for (int i = 0; i < data.Length; i++)
        {
            resultData[i] = numOps.Divide(data[i], scalar);
        }

        return new Tensor<T>(input.Shape, new Vector<T>(resultData));
    }

    /// <summary>
    /// Helper: Copies data to a specific position in result array.
    /// </summary>
    private static void CopyToPosition<T>(T[] result, T[] source, int[] resultShape, int[] sourceShape, int[] startIndices, int[] strides)
    {
        // Simplified implementation for common cases
        var sourceSize = source.Length;
        for (int i = 0; i < sourceSize; i++)
        {
            // Calculate source indices
            var sourceIndices = new int[sourceShape.Length];
            int remaining = i;
            for (int d = sourceShape.Length - 1; d >= 0; d--)
            {
                sourceIndices[d] = remaining % sourceShape[d];
                remaining /= sourceShape[d];
            }

            // Calculate result index
            int resultIdx = 0;
            for (int d = 0; d < resultShape.Length; d++)
            {
                int srcIdx = d < sourceIndices.Length ? sourceIndices[d] : 0;
                int startIdx = d < startIndices.Length ? startIndices[d] : 0;
                resultIdx += (startIdx + srcIdx) * strides[d];
            }

            if (resultIdx < result.Length)
            {
                result[resultIdx] = source[i];
            }
        }
    }

    /// <summary>
    /// Helper: Copies a slice of data.
    /// </summary>
    private static void CopySlice<T>(T[] input, T[] result, int[] inputShape, int[] resultShape, int axis, int start, int[] strides)
    {
        // Simplified implementation
        var resultIdx = 0;
        CopySliceRecursive(input, result, inputShape, resultShape, axis, start, strides, 0, 0, ref resultIdx);
    }

    private static void CopySliceRecursive<T>(T[] input, T[] result, int[] inputShape, int[] resultShape, int axis, int start, int[] strides, int dim, int inputOffset, ref int resultIdx)
    {
        if (dim == inputShape.Length)
        {
            result[resultIdx++] = input[inputOffset];
            return;
        }

        int rangeStart = dim == axis ? start : 0;
        int rangeEnd = dim == axis ? start + resultShape[dim] : inputShape[dim];

        for (int i = rangeStart; i < rangeEnd; i++)
        {
            CopySliceRecursive(input, result, inputShape, resultShape, axis, start, strides, dim + 1, inputOffset + i * strides[dim], ref resultIdx);
        }
    }

    /// <summary>
    /// Helper: Copies a region of data for slicing.
    /// </summary>
    private static void CopySliceRegion<T>(T[] input, T[] result, int[] inputShape, int[] resultShape, int[] startIndices, int[] strides)
    {
        // Simplified implementation
        var resultIdx = 0;
        CopySliceRegionRecursive(input, result, inputShape, resultShape, startIndices, strides, 0, 0, ref resultIdx);
    }

    private static void CopySliceRegionRecursive<T>(T[] input, T[] result, int[] inputShape, int[] resultShape, int[] startIndices, int[] strides, int dim, int inputOffset, ref int resultIdx)
    {
        if (dim == inputShape.Length)
        {
            result[resultIdx++] = input[inputOffset];
            return;
        }

        int start = dim < startIndices.Length ? startIndices[dim] : 0;
        int size = dim < resultShape.Length ? resultShape[dim] : 1;

        for (int i = 0; i < size; i++)
        {
            CopySliceRegionRecursive(input, result, inputShape, resultShape, startIndices, strides, dim + 1, inputOffset + (start + i) * strides[dim], ref resultIdx);
        }
    }

    /// <summary>
    /// Helper: Broadcasts data from source to target shape.
    /// </summary>
    private static void BroadcastCopy<T>(T[] source, T[] result, int[] sourceShape, int[] targetShape)
    {
        // Pad source shape with 1s at the front if needed
        var paddedSourceShape = new int[targetShape.Length];
        var offset = targetShape.Length - sourceShape.Length;
        for (int i = 0; i < targetShape.Length; i++)
        {
            paddedSourceShape[i] = i < offset ? 1 : sourceShape[i - offset];
        }

        // Calculate strides
        var sourceStrides = new int[targetShape.Length];
        var targetStrides = new int[targetShape.Length];
        sourceStrides[targetShape.Length - 1] = 1;
        targetStrides[targetShape.Length - 1] = 1;
        for (int d = targetShape.Length - 2; d >= 0; d--)
        {
            sourceStrides[d] = sourceStrides[d + 1] * paddedSourceShape[d + 1];
            targetStrides[d] = targetStrides[d + 1] * targetShape[d + 1];
        }

        // Broadcast copy
        for (int i = 0; i < result.Length; i++)
        {
            var indices = new int[targetShape.Length];
            int remaining = i;
            for (int d = targetShape.Length - 1; d >= 0; d--)
            {
                indices[d] = remaining % targetShape[d];
                remaining /= targetShape[d];
            }

            // Calculate source index with broadcasting
            int srcIdx = 0;
            for (int d = 0; d < targetShape.Length; d++)
            {
                int srcDimIdx = paddedSourceShape[d] == 1 ? 0 : indices[d];
                srcIdx += srcDimIdx * sourceStrides[d];
            }

            result[i] = source[srcIdx];
        }
    }

    /// <summary>
    /// Helper: Sum over batch and spatial dimensions for normalization gradients.
    /// Supports arbitrary dimensions - keeps channel dimension (axis 1) and sums over all others.
    /// </summary>
    private static Tensor<T> SumOverBatchAndSpatial<T>(Tensor<T> input)
    {
        var numOps = MathHelper.GetNumericOperations<T>();
        var shape = input.Shape;

        if (shape.Length == 1)
        {
            // Already 1D, return as-is
            return input;
        }

        if (shape.Length == 2)
        {
            // [batch, features] -> [features]
            int batchSize = shape[0];
            int features = shape[1];
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

        // For N-dimensional tensors (N >= 3), sum over all dimensions except channels (axis 1)
        // Format: [batch, channels, spatial_dims...]
        int channels = shape[1];
        var result = new T[channels];
        var data = input.ToArray();

        // Calculate strides for each dimension
        var strides = new int[shape.Length];
        strides[shape.Length - 1] = 1;
        for (int d = shape.Length - 2; d >= 0; d--)
        {
            strides[d] = strides[d + 1] * shape[d + 1];
        }

        // Calculate total spatial size (excluding batch and channels)
        int spatialSize = 1;
        for (int d = 2; d < shape.Length; d++)
        {
            spatialSize *= shape[d];
        }

        int batchSize = shape[0];
        int channelStride = strides[1];

        // Sum over batch and spatial dimensions for each channel
        for (int c = 0; c < channels; c++)
        {
            T sum = numOps.Zero;
            for (int n = 0; n < batchSize; n++)
            {
                int batchOffset = n * strides[0] + c * channelStride;
                for (int s = 0; s < spatialSize; s++)
                {
                    sum = numOps.Add(sum, data[batchOffset + s]);
                }
            }
            result[c] = sum;
        }

        return new Tensor<T>(new int[] { channels }, new Vector<T>(result));
    }
}
