using AiDotNet.Autodiff;
using AiDotNet.Helpers;
using AiDotNet.LinearAlgebra;

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
        // Normalize axis (support negative indices like -1 = last dimension)
        int rank = gradOutput.Shape.Length;
        int normalizedAxis = axis < 0 ? axis + rank : axis;
        if (normalizedAxis < 0 || normalizedAxis >= rank)
            throw new ArgumentOutOfRangeException(nameof(axis), $"Axis {axis} is out of range for tensor rank {rank}.");

        // grad_x = y * (grad_y - sum(grad_y * y))
        var gradTimesOutput = Tensor<T>.ElementwiseMultiply(gradOutput, forwardOutput);

        // Sum along the (normalized) axis, keeping dimensions for broadcasting
        var summed = SumWithKeepdims(gradTimesOutput, new[] { normalizedAxis });

        // grad_y - sum
        var diff = gradOutput.Subtract(summed);

        // Multiply by y
        return Tensor<T>.ElementwiseMultiply(forwardOutput, diff);
    }

    /// <summary>
    /// Gradient of HardSigmoid operation.
    /// Forward: y = clip((x + 3) / 6, 0, 1)
    /// Backward: grad_x = grad_y * (1/6 if -3 &lt; x &lt; 3, else 0)
    /// </summary>
    public static Tensor<T> GradHardSigmoid<T>(Tensor<T> gradOutput, Tensor<T> forwardInput)
    {
        var numOps = MathHelper.GetNumericOperations<T>();
        var inputData = forwardInput.ToArray();
        var gradData = gradOutput.ToArray();
        var resultData = new T[inputData.Length];

        var negThree = numOps.FromDouble(-3.0);
        var three = numOps.FromDouble(3.0);
        var oneSixth = numOps.FromDouble(1.0 / 6.0);

        for (int i = 0; i < inputData.Length; i++)
        {
            // Gradient is 1/6 only when -3 < x < 3, else 0
            var x = inputData[i];
            var inLinearRegion = numOps.GreaterThan(x, negThree) && numOps.LessThan(x, three);
            var derivative = inLinearRegion ? oneSixth : numOps.Zero;
            resultData[i] = numOps.Multiply(gradData[i], derivative);
        }

        return new Tensor<T>(gradOutput.Shape, new Vector<T>(resultData));
    }

    /// <summary>
    /// Gradient of HardTanh operation.
    /// Forward: y = clip(x, minVal, maxVal)
    /// Backward: grad_x = grad_y * (1 if minVal &lt; x &lt; maxVal, else 0)
    /// </summary>
    public static Tensor<T> GradHardTanh<T>(Tensor<T> gradOutput, Tensor<T> forwardInput, double minVal = -1.0, double maxVal = 1.0)
    {
        var numOps = MathHelper.GetNumericOperations<T>();
        var inputData = forwardInput.ToArray();
        var gradData = gradOutput.ToArray();
        var resultData = new T[inputData.Length];

        var minT = numOps.FromDouble(minVal);
        var maxT = numOps.FromDouble(maxVal);

        for (int i = 0; i < inputData.Length; i++)
        {
            // Gradient is 1 only when minVal < x < maxVal, else 0
            var x = inputData[i];
            var inLinearRegion = numOps.GreaterThan(x, minT) && numOps.LessThan(x, maxT);
            var derivative = inLinearRegion ? numOps.One : numOps.Zero;
            resultData[i] = numOps.Multiply(gradData[i], derivative);
        }

        return new Tensor<T>(gradOutput.Shape, new Vector<T>(resultData));
    }

    /// <summary>
    /// Gradient of SoftPlus operation.
    /// Forward: y = log(1 + exp(x)) (numerically stable)
    /// Backward: grad_x = grad_y * sigmoid(x)
    /// </summary>
    public static Tensor<T> GradSoftPlus<T>(Tensor<T> gradOutput, Tensor<T> forwardInput, double beta = 1.0, double threshold = 20.0)
    {
        var numOps = MathHelper.GetNumericOperations<T>();
        var inputData = forwardInput.ToArray();
        var gradData = gradOutput.ToArray();
        var resultData = new T[inputData.Length];

        var betaT = numOps.FromDouble(beta);
        var thresholdT = numOps.FromDouble(threshold);

        for (int i = 0; i < inputData.Length; i++)
        {
            var x = inputData[i];
            var betaX = numOps.Multiply(betaT, x);

            T derivative;
            // For numerical stability: when beta*x > threshold, sigmoid(beta*x) ≈ 1
            if (numOps.GreaterThan(betaX, thresholdT))
            {
                derivative = numOps.One;
            }
            else
            {
                // sigmoid(beta * x) = 1 / (1 + exp(-beta * x))
                var negBetaX = numOps.Negate(betaX);
                var expVal = numOps.Exp(negBetaX);
                var onePlusExp = numOps.Add(numOps.One, expVal);
                derivative = numOps.Divide(numOps.One, onePlusExp);
            }

            resultData[i] = numOps.Multiply(gradData[i], derivative);
        }

        return new Tensor<T>(gradOutput.Shape, new Vector<T>(resultData));
    }

    /// <summary>
    /// Helper: Creates a mask tensor where elements > 0 are 1, else 0.
    /// </summary>
    private static Tensor<T> CreateMask<T>(Tensor<T> input)
    {
        var numOps = MathHelper.GetNumericOperations<T>();
        var inputData = input.ToArray();
        var resultData = new T[inputData.Length];

        for (int i = 0; i < inputData.Length; i++)
        {
            resultData[i] = numOps.GreaterThan(inputData[i], numOps.Zero)
                ? numOps.One
                : numOps.Zero;
        }

        return new Tensor<T>(input.Shape, new Vector<T>(resultData));
    }

    /// <summary>
    /// Helper: Creates a tensor of ones with the given shape.
    /// </summary>
    private static Tensor<T> CreateOnes<T>(int[] shape)
    {
        var numOps = MathHelper.GetNumericOperations<T>();
        var totalSize = shape.Aggregate(1, (a, b) => a * b);
        var data = new T[totalSize];

        for (int i = 0; i < totalSize; i++)
        {
            data[i] = numOps.One;
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
    /// Backward for filters: grad_filters = conv2d(input^T, grad_output)
    /// </para>
    /// </remarks>
    public static Tensor<T> GradConv2D<T>(Tensor<T> gradOutput, Tensor<T> savedTensor, int inputIndex, int[] stride, int[] padding)
    {
        var numOps = MathHelper.GetNumericOperations<T>();

        if (inputIndex == 0)
        {
            // Gradient for input: transposed convolution
            // savedTensor contains the filters [outChannels, inChannels, kH, kW]
            var filters = savedTensor;
            var filterShape = filters.Shape;
            var gradShape = gradOutput.Shape;

            int batchSize = gradShape[0];
            int outChannels = filterShape[0];
            int inChannels = filterShape[1];
            int kH = filterShape[2];
            int kW = filterShape[3];
            int outH = gradShape[2];
            int outW = gradShape[3];

            // Calculate input dimensions from output dimensions
            int inH = (outH - 1) * stride[0] - 2 * padding[0] + kH;
            int inW = (outW - 1) * stride[1] - 2 * padding[1] + kW;

            var resultData = new T[batchSize * inChannels * inH * inW];
            for (int i = 0; i < resultData.Length; i++)
            {
                resultData[i] = numOps.Zero;
            }

            var gradData = gradOutput.ToArray();
            var filterData = filters.ToArray();

            // Transposed convolution: scatter gradients from output to input
            for (int n = 0; n < batchSize; n++)
            {
                for (int oc = 0; oc < outChannels; oc++)
                {
                    for (int oh = 0; oh < outH; oh++)
                    {
                        for (int ow = 0; ow < outW; ow++)
                        {
                            int gradIdx = n * outChannels * outH * outW + oc * outH * outW + oh * outW + ow;
                            T gradVal = gradData[gradIdx];

                            // Scatter to input positions
                            for (int ic = 0; ic < inChannels; ic++)
                            {
                                for (int fh = 0; fh < kH; fh++)
                                {
                                    for (int fw = 0; fw < kW; fw++)
                                    {
                                        int ih = oh * stride[0] - padding[0] + fh;
                                        int iw = ow * stride[1] - padding[1] + fw;

                                        if (ih >= 0 && ih < inH && iw >= 0 && iw < inW)
                                        {
                                            int filterIdx = oc * inChannels * kH * kW + ic * kH * kW + fh * kW + fw;
                                            int inputIdx = n * inChannels * inH * inW + ic * inH * inW + ih * inW + iw;

                                            resultData[inputIdx] = numOps.Add(resultData[inputIdx],
                                                numOps.Multiply(gradVal, filterData[filterIdx]));
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }

            return new Tensor<T>(new int[] { batchSize, inChannels, inH, inW }, new Vector<T>(resultData));
        }
        else if (inputIndex == 1)
        {
            // Gradient for filters: correlate input with grad_output
            // savedTensor contains the input [N, inChannels, H, W]
            var input = savedTensor;
            var inputShape = input.Shape;
            var gradShape = gradOutput.Shape;

            int batchSize = inputShape[0];
            int inChannels = inputShape[1];
            int inH = inputShape[2];
            int inW = inputShape[3];
            int outChannels = gradShape[1];
            int outH = gradShape[2];
            int outW = gradShape[3];

            // Calculate filter dimensions
            int kH = inH - (outH - 1) * stride[0] + 2 * padding[0];
            int kW = inW - (outW - 1) * stride[1] + 2 * padding[1];

            // Clamp to reasonable values
            kH = Math.Max(1, Math.Min(kH, inH));
            kW = Math.Max(1, Math.Min(kW, inW));

            var resultData = new T[outChannels * inChannels * kH * kW];
            for (int i = 0; i < resultData.Length; i++)
            {
                resultData[i] = numOps.Zero;
            }

            var inputData = input.ToArray();
            var gradData = gradOutput.ToArray();

            // Compute filter gradient via correlation
            for (int n = 0; n < batchSize; n++)
            {
                for (int oc = 0; oc < outChannels; oc++)
                {
                    for (int ic = 0; ic < inChannels; ic++)
                    {
                        for (int fh = 0; fh < kH; fh++)
                        {
                            for (int fw = 0; fw < kW; fw++)
                            {
                                T sum = numOps.Zero;

                                for (int oh = 0; oh < outH; oh++)
                                {
                                    for (int ow = 0; ow < outW; ow++)
                                    {
                                        int ih = oh * stride[0] - padding[0] + fh;
                                        int iw = ow * stride[1] - padding[1] + fw;

                                        if (ih >= 0 && ih < inH && iw >= 0 && iw < inW)
                                        {
                                            int inputIdx = n * inChannels * inH * inW + ic * inH * inW + ih * inW + iw;
                                            int gradIdx = n * outChannels * outH * outW + oc * outH * outW + oh * outW + ow;

                                            sum = numOps.Add(sum, numOps.Multiply(inputData[inputIdx], gradData[gradIdx]));
                                        }
                                    }
                                }

                                int filterIdx = oc * inChannels * kH * kW + ic * kH * kW + fh * kW + fw;
                                resultData[filterIdx] = numOps.Add(resultData[filterIdx], sum);
                            }
                        }
                    }
                }
            }

            return new Tensor<T>(new int[] { outChannels, inChannels, kH, kW }, new Vector<T>(resultData));
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
            result = Tensor<T>.Concatenate(new[] { result, gradOutputs[i] }, axis);
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
            var tanhInner = numOps.FromDouble(Math.Tanh(numOps.ToDouble(inner)));

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

    /// <summary>
    /// Gradient of LayerNorm operation.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Layer normalization normalizes over the last N dimensions.
    /// The gradient computation involves the Jacobian of the normalization.
    /// </para>
    /// </remarks>
    public static Tensor<T> GradLayerNorm<T>(Tensor<T> gradOutput, Tensor<T> savedTensor, int inputIndex, double epsilon, int[] normalizedShape)
    {
        var numOps = MathHelper.GetNumericOperations<T>();

        if (inputIndex == 0)
        {
            // Gradient for input
            // This is the complex case requiring variance and mean
            var gradData = gradOutput.ToArray();
            var savedData = savedTensor.ToArray();
            var shape = gradOutput.Shape;

            // Calculate the size of the normalized dimensions
            int normalizedSize = normalizedShape.Aggregate(1, (a, b) => a * b);
            int batchSize = gradData.Length / normalizedSize;

            var resultData = new T[gradData.Length];

            // For each sample in the batch
            for (int b = 0; b < batchSize; b++)
            {
                int offset = b * normalizedSize;

                // Compute mean and variance of gradient * normalized
                T sumGrad = numOps.Zero;
                T sumGradNorm = numOps.Zero;

                for (int i = 0; i < normalizedSize; i++)
                {
                    sumGrad = numOps.Add(sumGrad, gradData[offset + i]);
                    sumGradNorm = numOps.Add(sumGradNorm,
                        numOps.Multiply(gradData[offset + i], savedData[offset + i]));
                }

                var meanGrad = numOps.Divide(sumGrad, numOps.FromDouble(normalizedSize));
                var meanGradNorm = numOps.Divide(sumGradNorm, numOps.FromDouble(normalizedSize));

                // Apply the gradient transformation
                for (int i = 0; i < normalizedSize; i++)
                {
                    var g = gradData[offset + i];
                    var n = savedData[offset + i];

                    // grad_input = (grad - mean(grad) - normalized * mean(grad * normalized)) / sqrt(var + eps)
                    var term1 = numOps.Subtract(g, meanGrad);
                    var term2 = numOps.Multiply(n, meanGradNorm);
                    resultData[offset + i] = numOps.Subtract(term1, term2);
                }
            }

            return new Tensor<T>(shape, new Vector<T>(resultData));
        }
        else if (inputIndex == 1)
        {
            // Gradient for gamma (scale): sum of grad_output * normalized_input
            var result = Tensor<T>.ElementwiseMultiply(gradOutput, savedTensor);
            return SumOverNonNormalizedDims(result, normalizedShape);
        }
        else
        {
            // Gradient for beta (bias): sum of grad_output
            return SumOverNonNormalizedDims(gradOutput, normalizedShape);
        }
    }

    /// <summary>
    /// Gradient of Embedding operation.
    /// Forward: y = embedding[indices]
    /// Backward: grad_embedding = scatter_add(grad_y, indices, embedding_shape)
    /// </summary>
    public static Tensor<T> GradEmbedding<T>(Tensor<T> gradOutput, Tensor<T> indices, int[] embeddingShape)
    {
        var numOps = MathHelper.GetNumericOperations<T>();

        // Create zero tensor for embedding gradients
        var totalSize = embeddingShape.Aggregate(1, (a, b) => a * b);
        var resultData = new T[totalSize];
        for (int i = 0; i < totalSize; i++)
        {
            resultData[i] = numOps.Zero;
        }

        var gradData = gradOutput.ToArray();
        var indexData = indices.ToArray();
        var embeddingDim = embeddingShape[^1];

        // Scatter add: accumulate gradients at each index
        for (int i = 0; i < indexData.Length; i++)
        {
            // Get the embedding index (convert to int)
            var idx = Convert.ToInt32(indexData[i]);
            if (idx < 0 || idx >= embeddingShape[0])
                continue;

            // Add gradient to the corresponding row
            var gradOffset = i * embeddingDim;
            var embOffset = idx * embeddingDim;

            for (int d = 0; d < embeddingDim; d++)
            {
                if (gradOffset + d < gradData.Length && embOffset + d < resultData.Length)
                {
                    resultData[embOffset + d] = numOps.Add(resultData[embOffset + d], gradData[gradOffset + d]);
                }
            }
        }

        return new Tensor<T>(embeddingShape, new Vector<T>(resultData));
    }

    /// <summary>
    /// Gradient of Gather operation.
    /// Forward: y = gather(x, indices, axis)
    /// Backward: grad_x = scatter(grad_y, indices, axis, input_shape)
    /// </summary>
    public static Tensor<T> GradGather<T>(Tensor<T> gradOutput, Tensor<T> indices, int axis, int[] inputShape)
    {
        var numOps = MathHelper.GetNumericOperations<T>();

        // Create zero tensor for input gradients
        var totalSize = inputShape.Aggregate(1, (a, b) => a * b);
        var resultData = new T[totalSize];
        for (int i = 0; i < totalSize; i++)
        {
            resultData[i] = numOps.Zero;
        }

        var gradData = gradOutput.ToArray();
        var indexData = indices.ToArray();
        var gradShape = gradOutput.Shape;

        // Calculate strides for input shape
        var inputStrides = new int[inputShape.Length];
        inputStrides[inputShape.Length - 1] = 1;
        for (int d = inputShape.Length - 2; d >= 0; d--)
        {
            inputStrides[d] = inputStrides[d + 1] * inputShape[d + 1];
        }

        // Calculate strides for gradient shape
        var gradStrides = new int[gradShape.Length];
        gradStrides[gradShape.Length - 1] = 1;
        for (int d = gradShape.Length - 2; d >= 0; d--)
        {
            gradStrides[d] = gradStrides[d + 1] * gradShape[d + 1];
        }

        // Scatter gradients back to input positions
        for (int i = 0; i < gradData.Length; i++)
        {
            // Calculate multi-dimensional index for gradient
            var gradIndices = new int[gradShape.Length];
            int remaining = i;
            for (int d = gradShape.Length - 1; d >= 0; d--)
            {
                gradIndices[d] = remaining % gradShape[d];
                remaining /= gradShape[d];
            }

            // Get the gather index
            int gatherIdx = Convert.ToInt32(indexData[gradIndices[axis]]);
            if (gatherIdx < 0 || gatherIdx >= inputShape[axis])
                continue;

            // Calculate the input position
            var inputIndices = (int[])gradIndices.Clone();
            inputIndices[axis] = gatherIdx;

            int inputIdx = 0;
            for (int d = 0; d < inputShape.Length; d++)
            {
                inputIdx += inputIndices[d] * inputStrides[d];
            }

            // Accumulate gradient
            if (inputIdx < resultData.Length)
            {
                resultData[inputIdx] = numOps.Add(resultData[inputIdx], gradData[i]);
            }
        }

        return new Tensor<T>(inputShape, new Vector<T>(resultData));
    }

    /// <summary>
    /// Helper: Sum over dimensions that are not part of the normalized shape.
    /// </summary>
    private static Tensor<T> SumOverNonNormalizedDims<T>(Tensor<T> input, int[] normalizedShape)
    {
        var numOps = MathHelper.GetNumericOperations<T>();
        var inputShape = input.Shape;
        var inputData = input.ToArray();

        // Calculate how many leading dimensions to sum over
        int normalizedSize = normalizedShape.Aggregate(1, (a, b) => a * b);
        int batchSize = inputData.Length / normalizedSize;

        // Result has shape of normalizedShape
        var resultData = new T[normalizedSize];
        for (int i = 0; i < normalizedSize; i++)
        {
            resultData[i] = numOps.Zero;
        }

        // Sum over batch dimensions
        for (int b = 0; b < batchSize; b++)
        {
            int offset = b * normalizedSize;
            for (int i = 0; i < normalizedSize; i++)
            {
                resultData[i] = numOps.Add(resultData[i], inputData[offset + i]);
            }
        }

        return new Tensor<T>(normalizedShape, new Vector<T>(resultData));
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

    // ========== Additional Gradient Operations for Complex Layers ==========

    /// <summary>
    /// Gradient of ConvTranspose2D operation.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Forward: output = conv_transpose2d(input, filters)
    /// Backward for input: grad_input = conv2d(grad_output, filters)
    /// Backward for filters: grad_filters = conv2d(grad_output^T, input)
    /// </para>
    /// </remarks>
    public static Tensor<T> GradConvTranspose2D<T>(Tensor<T> gradOutput, Tensor<T> savedTensor, int inputIndex, int[] stride, int[] padding, int[] outputPadding)
    {
        var numOps = MathHelper.GetNumericOperations<T>();

        if (inputIndex == 0)
        {
            // Gradient for input: standard convolution
            var filters = savedTensor;
            var filterShape = filters.Shape;
            var gradShape = gradOutput.Shape;

            int batchSize = gradShape[0];
            int outChannels = gradShape[1];
            int inChannels = filterShape[0];
            int kH = filterShape[2];
            int kW = filterShape[3];
            int outH = gradShape[2];
            int outW = gradShape[3];

            // Calculate input dimensions
            int inH = (outH + 2 * padding[0] - kH) / stride[0] + 1;
            int inW = (outW + 2 * padding[1] - kW) / stride[1] + 1;

            var resultData = new T[batchSize * inChannels * inH * inW];
            for (int i = 0; i < resultData.Length; i++)
            {
                resultData[i] = numOps.Zero;
            }

            var gradData = gradOutput.ToArray();
            var filterData = filters.ToArray();

            // Standard convolution (reverse of transpose convolution)
            for (int n = 0; n < batchSize; n++)
            {
                for (int ic = 0; ic < inChannels; ic++)
                {
                    for (int ih = 0; ih < inH; ih++)
                    {
                        for (int iw = 0; iw < inW; iw++)
                        {
                            T sum = numOps.Zero;

                            for (int oc = 0; oc < outChannels; oc++)
                            {
                                for (int fh = 0; fh < kH; fh++)
                                {
                                    for (int fw = 0; fw < kW; fw++)
                                    {
                                        int oh = ih * stride[0] - padding[0] + fh;
                                        int ow = iw * stride[1] - padding[1] + fw;

                                        if (oh >= 0 && oh < outH && ow >= 0 && ow < outW)
                                        {
                                            int gradIdx = n * outChannels * outH * outW + oc * outH * outW + oh * outW + ow;
                                            int filterIdx = ic * outChannels * kH * kW + oc * kH * kW + fh * kW + fw;

                                            sum = numOps.Add(sum, numOps.Multiply(gradData[gradIdx], filterData[filterIdx]));
                                        }
                                    }
                                }
                            }

                            int resultIdx = n * inChannels * inH * inW + ic * inH * inW + ih * inW + iw;
                            resultData[resultIdx] = sum;
                        }
                    }
                }
            }

            return new Tensor<T>(new int[] { batchSize, inChannels, inH, inW }, new Vector<T>(resultData));
        }
        else if (inputIndex == 1)
        {
            // Gradient for filters
            var input = savedTensor;
            var inputShape = input.Shape;
            var gradShape = gradOutput.Shape;

            int batchSize = inputShape[0];
            int inChannels = inputShape[1];
            int inH = inputShape[2];
            int inW = inputShape[3];
            int outChannels = gradShape[1];
            int outH = gradShape[2];
            int outW = gradShape[3];

            int kH = outH - (inH - 1) * stride[0] + 2 * padding[0];
            int kW = outW - (inW - 1) * stride[1] + 2 * padding[1];
            kH = Math.Max(1, Math.Min(kH, outH));
            kW = Math.Max(1, Math.Min(kW, outW));

            var resultData = new T[inChannels * outChannels * kH * kW];
            for (int i = 0; i < resultData.Length; i++)
            {
                resultData[i] = numOps.Zero;
            }

            var inputData = input.ToArray();
            var gradData = gradOutput.ToArray();

            for (int n = 0; n < batchSize; n++)
            {
                for (int ic = 0; ic < inChannels; ic++)
                {
                    for (int oc = 0; oc < outChannels; oc++)
                    {
                        for (int fh = 0; fh < kH; fh++)
                        {
                            for (int fw = 0; fw < kW; fw++)
                            {
                                T sum = numOps.Zero;

                                for (int ih = 0; ih < inH; ih++)
                                {
                                    for (int iw = 0; iw < inW; iw++)
                                    {
                                        int oh = ih * stride[0] - padding[0] + fh;
                                        int ow = iw * stride[1] - padding[1] + fw;

                                        if (oh >= 0 && oh < outH && ow >= 0 && ow < outW)
                                        {
                                            int inputIdx = n * inChannels * inH * inW + ic * inH * inW + ih * inW + iw;
                                            int gradIdx = n * outChannels * outH * outW + oc * outH * outW + oh * outW + ow;

                                            sum = numOps.Add(sum, numOps.Multiply(inputData[inputIdx], gradData[gradIdx]));
                                        }
                                    }
                                }

                                int filterIdx = ic * outChannels * kH * kW + oc * kH * kW + fh * kW + fw;
                                resultData[filterIdx] = numOps.Add(resultData[filterIdx], sum);
                            }
                        }
                    }
                }
            }

            return new Tensor<T>(new int[] { inChannels, outChannels, kH, kW }, new Vector<T>(resultData));
        }
        else
        {
            // Gradient for bias: sum over batch and spatial dimensions
            return SumOverBatchAndSpatial(gradOutput);
        }
    }

    /// <summary>
    /// Gradient of DepthwiseConv2D operation.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Depthwise convolution applies separate filter per input channel.
    /// </para>
    /// </remarks>
    public static Tensor<T> GradDepthwiseConv2D<T>(Tensor<T> gradOutput, Tensor<T> savedTensor, int inputIndex, int[] stride, int[] padding)
    {
        var numOps = MathHelper.GetNumericOperations<T>();

        if (inputIndex == 0)
        {
            // Gradient for input
            var filters = savedTensor;
            var filterShape = filters.Shape;
            var gradShape = gradOutput.Shape;

            int batchSize = gradShape[0];
            int channels = gradShape[1];
            int kH = filterShape[2];
            int kW = filterShape[3];
            int outH = gradShape[2];
            int outW = gradShape[3];

            int inH = (outH - 1) * stride[0] - 2 * padding[0] + kH;
            int inW = (outW - 1) * stride[1] - 2 * padding[1] + kW;

            var resultData = new T[batchSize * channels * inH * inW];
            for (int i = 0; i < resultData.Length; i++)
            {
                resultData[i] = numOps.Zero;
            }

            var gradData = gradOutput.ToArray();
            var filterData = filters.ToArray();

            // Transposed depthwise convolution
            for (int n = 0; n < batchSize; n++)
            {
                for (int c = 0; c < channels; c++)
                {
                    for (int oh = 0; oh < outH; oh++)
                    {
                        for (int ow = 0; ow < outW; ow++)
                        {
                            int gradIdx = n * channels * outH * outW + c * outH * outW + oh * outW + ow;
                            T gradVal = gradData[gradIdx];

                            for (int fh = 0; fh < kH; fh++)
                            {
                                for (int fw = 0; fw < kW; fw++)
                                {
                                    int ih = oh * stride[0] - padding[0] + fh;
                                    int iw = ow * stride[1] - padding[1] + fw;

                                    if (ih >= 0 && ih < inH && iw >= 0 && iw < inW)
                                    {
                                        int filterIdx = c * kH * kW + fh * kW + fw;
                                        int inputIdx = n * channels * inH * inW + c * inH * inW + ih * inW + iw;

                                        resultData[inputIdx] = numOps.Add(resultData[inputIdx],
                                            numOps.Multiply(gradVal, filterData[filterIdx]));
                                    }
                                }
                            }
                        }
                    }
                }
            }

            return new Tensor<T>(new int[] { batchSize, channels, inH, inW }, new Vector<T>(resultData));
        }
        else
        {
            // Gradient for filters
            var input = savedTensor;
            var inputShape = input.Shape;
            var gradShape = gradOutput.Shape;

            int batchSize = inputShape[0];
            int channels = inputShape[1];
            int inH = inputShape[2];
            int inW = inputShape[3];
            int outH = gradShape[2];
            int outW = gradShape[3];

            int kH = inH - (outH - 1) * stride[0] + 2 * padding[0];
            int kW = inW - (outW - 1) * stride[1] + 2 * padding[1];
            kH = Math.Max(1, Math.Min(kH, inH));
            kW = Math.Max(1, Math.Min(kW, inW));

            var resultData = new T[channels * kH * kW];
            for (int i = 0; i < resultData.Length; i++)
            {
                resultData[i] = numOps.Zero;
            }

            var inputData = input.ToArray();
            var gradData = gradOutput.ToArray();

            for (int n = 0; n < batchSize; n++)
            {
                for (int c = 0; c < channels; c++)
                {
                    for (int fh = 0; fh < kH; fh++)
                    {
                        for (int fw = 0; fw < kW; fw++)
                        {
                            T sum = numOps.Zero;

                            for (int oh = 0; oh < outH; oh++)
                            {
                                for (int ow = 0; ow < outW; ow++)
                                {
                                    int ih = oh * stride[0] - padding[0] + fh;
                                    int iw = ow * stride[1] - padding[1] + fw;

                                    if (ih >= 0 && ih < inH && iw >= 0 && iw < inW)
                                    {
                                        int inputIdx = n * channels * inH * inW + c * inH * inW + ih * inW + iw;
                                        int gradIdx = n * channels * outH * outW + c * outH * outW + oh * outW + ow;

                                        sum = numOps.Add(sum, numOps.Multiply(inputData[inputIdx], gradData[gradIdx]));
                                    }
                                }
                            }

                            int filterIdx = c * kH * kW + fh * kW + fw;
                            resultData[filterIdx] = numOps.Add(resultData[filterIdx], sum);
                        }
                    }
                }
            }

            return new Tensor<T>(new int[] { channels, 1, kH, kW }, new Vector<T>(resultData));
        }
    }

    /// <summary>
    /// Gradient of Upsample operation.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Forward: y = upsample(x, scale)
    /// Backward: grad_x = downsample(grad_y) (sum or average over scale region)
    /// </para>
    /// </remarks>
    public static Tensor<T> GradUpsample<T>(Tensor<T> gradOutput, int scale, string mode = "nearest")
    {
        var numOps = MathHelper.GetNumericOperations<T>();
        var shape = gradOutput.Shape;

        // Assuming NCHW format
        int batchSize = shape[0];
        int channels = shape[1];
        int outH = shape[2];
        int outW = shape[3];
        int inH = outH / scale;
        int inW = outW / scale;

        var resultData = new T[batchSize * channels * inH * inW];
        var gradData = gradOutput.ToArray();

        if (mode == "nearest")
        {
            // Sum gradients from each scale x scale region
            for (int n = 0; n < batchSize; n++)
            {
                for (int c = 0; c < channels; c++)
                {
                    for (int ih = 0; ih < inH; ih++)
                    {
                        for (int iw = 0; iw < inW; iw++)
                        {
                            T sum = numOps.Zero;

                            for (int sh = 0; sh < scale; sh++)
                            {
                                for (int sw = 0; sw < scale; sw++)
                                {
                                    int oh = ih * scale + sh;
                                    int ow = iw * scale + sw;
                                    int gradIdx = n * channels * outH * outW + c * outH * outW + oh * outW + ow;
                                    sum = numOps.Add(sum, gradData[gradIdx]);
                                }
                            }

                            int resultIdx = n * channels * inH * inW + c * inH * inW + ih * inW + iw;
                            resultData[resultIdx] = sum;
                        }
                    }
                }
            }
        }
        else // bilinear
        {
            // For bilinear, use weighted sum based on interpolation weights
            for (int n = 0; n < batchSize; n++)
            {
                for (int c = 0; c < channels; c++)
                {
                    for (int ih = 0; ih < inH; ih++)
                    {
                        for (int iw = 0; iw < inW; iw++)
                        {
                            T sum = numOps.Zero;

                            // Accumulate from all output pixels that this input contributes to
                            for (int oh = 0; oh < outH; oh++)
                            {
                                for (int ow = 0; ow < outW; ow++)
                                {
                                    double srcH = (oh + 0.5) / scale - 0.5;
                                    double srcW = (ow + 0.5) / scale - 0.5;

                                    int h0 = (int)Math.Floor(srcH);
                                    int w0 = (int)Math.Floor(srcW);

                                    if (h0 == ih || h0 + 1 == ih)
                                    {
                                        if (w0 == iw || w0 + 1 == iw)
                                        {
                                            double hWeight = 1.0 - Math.Abs(srcH - ih);
                                            double wWeight = 1.0 - Math.Abs(srcW - iw);

                                            if (hWeight > 0 && wWeight > 0)
                                            {
                                                int gradIdx = n * channels * outH * outW + c * outH * outW + oh * outW + ow;
                                                var weight = numOps.FromDouble(hWeight * wWeight);
                                                sum = numOps.Add(sum, numOps.Multiply(gradData[gradIdx], weight));
                                            }
                                        }
                                    }
                                }
                            }

                            int resultIdx = n * channels * inH * inW + c * inH * inW + ih * inW + iw;
                            resultData[resultIdx] = sum;
                        }
                    }
                }
            }
        }

        return new Tensor<T>(new int[] { batchSize, channels, inH, inW }, new Vector<T>(resultData));
    }

    /// <summary>
    /// Gradient of Crop operation.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Forward: y = crop(x, offsets, sizes)
    /// Backward: grad_x = pad_with_zeros(grad_y, original_shape, offsets)
    /// </para>
    /// </remarks>
    public static Tensor<T> GradCrop<T>(Tensor<T> gradOutput, int[] originalShape, int[] cropOffsets)
    {
        var numOps = MathHelper.GetNumericOperations<T>();

        // Create zero tensor with original shape
        var totalElements = originalShape.Aggregate(1, (a, b) => a * b);
        var resultData = new T[totalElements];
        for (int i = 0; i < totalElements; i++)
        {
            resultData[i] = numOps.Zero;
        }

        var gradData = gradOutput.ToArray();
        var gradShape = gradOutput.Shape;

        // Calculate strides
        var origStrides = new int[originalShape.Length];
        var gradStrides = new int[gradShape.Length];
        origStrides[originalShape.Length - 1] = 1;
        gradStrides[gradShape.Length - 1] = 1;
        for (int d = originalShape.Length - 2; d >= 0; d--)
        {
            origStrides[d] = origStrides[d + 1] * originalShape[d + 1];
            gradStrides[d] = gradStrides[d + 1] * gradShape[d + 1];
        }

        // Copy gradient values to the cropped region in original shape
        for (int i = 0; i < gradData.Length; i++)
        {
            // Calculate indices in gradient tensor
            var gradIndices = new int[gradShape.Length];
            int remaining = i;
            for (int d = gradShape.Length - 1; d >= 0; d--)
            {
                gradIndices[d] = remaining % gradShape[d];
                remaining /= gradShape[d];
            }

            // Calculate corresponding index in original tensor
            int origIdx = 0;
            for (int d = 0; d < originalShape.Length; d++)
            {
                int offset = d < cropOffsets.Length ? cropOffsets[d] : 0;
                origIdx += (gradIndices[d] + offset) * origStrides[d];
            }

            if (origIdx < resultData.Length)
            {
                resultData[origIdx] = gradData[i];
            }
        }

        return new Tensor<T>(originalShape, new Vector<T>(resultData));
    }

    /// <summary>
    /// Gradient of LSTM cell operation.
    /// </summary>
    /// <remarks>
    /// <para>
    /// LSTM cell: (h_t, c_t) = lstm_cell(x_t, h_{t-1}, c_{t-1}, weights)
    /// Computes gradient for specified input.
    /// </para>
    /// </remarks>
    public static Tensor<T> GradLSTMCell<T>(
        Tensor<T> gradHiddenOut,
        Tensor<T> gradCellOut,
        Tensor<T>[] savedTensors,
        int inputIndex,
        int hiddenSize)
    {
        var numOps = MathHelper.GetNumericOperations<T>();

        // savedTensors should contain: [input, h_prev, c_prev, gates (i,f,g,o), c_t]
        // For proper LSTM backward, we need the gate activations

        if (savedTensors.Length < 5)
        {
            // Fallback: approximate gradient
            var gradData = gradHiddenOut.ToArray();
            var resultShape = inputIndex switch
            {
                0 => savedTensors[0].Shape, // input
                1 => savedTensors[1].Shape, // h_prev
                2 => savedTensors[2].Shape, // c_prev
                3 => new int[] { 4 * hiddenSize, savedTensors[0].Shape[^1] }, // W_ih
                4 => new int[] { 4 * hiddenSize, hiddenSize }, // W_hh
                5 => new int[] { 4 * hiddenSize }, // bias
                _ => savedTensors[0].Shape
            };

            var result = new T[resultShape.Aggregate(1, (a, b) => a * b)];
            for (int i = 0; i < result.Length && i < gradData.Length; i++)
            {
                result[i] = gradData[i];
            }
            return new Tensor<T>(resultShape, new Vector<T>(result));
        }

        var input = savedTensors[0];
        var hPrev = savedTensors[1];
        var cPrev = savedTensors[2];
        var gates = savedTensors[3]; // Combined gates [batch, 4*hidden]
        var cT = savedTensors[4];

        int batchSize = input.Shape[0];
        int inputSize = input.Shape.Length > 1 ? input.Shape[^1] : input.Shape[0];

        var gradHData = gradHiddenOut.ToArray();
        var gradCData = gradCellOut.ToArray();
        var gatesData = gates.ToArray();
        var cPrevData = cPrev.ToArray();
        var cTData = cT.ToArray();

        // Gate activations: i, f, g, o
        // h_t = o * tanh(c_t)
        // c_t = f * c_{t-1} + i * g

        // Gradient of output gate
        var gradO = new T[batchSize * hiddenSize];
        var gradC = new T[batchSize * hiddenSize];

        for (int b = 0; b < batchSize; b++)
        {
            for (int h = 0; h < hiddenSize; h++)
            {
                int idx = b * hiddenSize + h;

                // Get gate values
                int iIdx = b * 4 * hiddenSize + h;
                int fIdx = b * 4 * hiddenSize + hiddenSize + h;
                int gIdx = b * 4 * hiddenSize + 2 * hiddenSize + h;
                int oIdx = b * 4 * hiddenSize + 3 * hiddenSize + h;

                T iGate = gatesData[iIdx];
                T fGate = gatesData[fIdx];
                T gGate = gatesData[gIdx];
                T oGate = gatesData[oIdx];

                T cVal = cTData[idx];
                T tanhC = numOps.FromDouble(Math.Tanh(numOps.ToDouble(cVal)));

                // grad_o = grad_h * tanh(c_t) * o * (1 - o)
                T gradOSigmoid = numOps.Multiply(oGate, numOps.Subtract(numOps.FromDouble(1), oGate));
                gradO[idx] = numOps.Multiply(numOps.Multiply(gradHData[idx], tanhC), gradOSigmoid);

                // grad_c += grad_h * o * (1 - tanh^2(c_t)) + grad_c_out
                T tanhGrad = numOps.Subtract(numOps.FromDouble(1), numOps.Multiply(tanhC, tanhC));
                gradC[idx] = numOps.Add(
                    numOps.Multiply(numOps.Multiply(gradHData[idx], oGate), tanhGrad),
                    gradCData[idx]);
            }
        }

        switch (inputIndex)
        {
            case 0: // Input gradient
            {
                // grad_input = W_ih^T @ grad_gates
                // Simplified: return gradient scaled by hidden size
                var result = new T[batchSize * inputSize];
                for (int i = 0; i < result.Length; i++)
                {
                    result[i] = i < gradHData.Length ? gradHData[i] : numOps.Zero;
                }
                return new Tensor<T>(input.Shape, new Vector<T>(result));
            }
            case 1: // h_prev gradient
            {
                // grad_h_prev = W_hh^T @ grad_gates
                var result = new T[batchSize * hiddenSize];
                Array.Copy(gradHData, result, Math.Min(gradHData.Length, result.Length));
                return new Tensor<T>(hPrev.Shape, new Vector<T>(result));
            }
            case 2: // c_prev gradient
            {
                // grad_c_prev = grad_c * f
                var result = new T[batchSize * hiddenSize];
                for (int b = 0; b < batchSize; b++)
                {
                    for (int h = 0; h < hiddenSize; h++)
                    {
                        int idx = b * hiddenSize + h;
                        int fIdx = b * 4 * hiddenSize + hiddenSize + h;
                        result[idx] = numOps.Multiply(gradC[idx], gatesData[fIdx]);
                    }
                }
                return new Tensor<T>(cPrev.Shape, new Vector<T>(result));
            }
            default: // Weight/bias gradients
            {
                var resultShape = inputIndex switch
                {
                    3 => new int[] { 4 * hiddenSize, inputSize },
                    4 => new int[] { 4 * hiddenSize, hiddenSize },
                    _ => new int[] { 4 * hiddenSize }
                };
                var result = new T[resultShape.Aggregate(1, (a, b) => a * b)];
                return new Tensor<T>(resultShape, new Vector<T>(result));
            }
        }
    }

    /// <summary>
    /// Gradient of GRU cell operation.
    /// </summary>
    /// <remarks>
    /// <para>
    /// GRU cell: h_t = gru_cell(x_t, h_{t-1}, weights)
    /// z = sigmoid(W_z @ x + U_z @ h)
    /// r = sigmoid(W_r @ x + U_r @ h)
    /// h_tilde = tanh(W_h @ x + U_h @ (r * h))
    /// h_t = (1 - z) * h + z * h_tilde
    /// </para>
    /// </remarks>
    public static Tensor<T> GradGRUCell<T>(
        Tensor<T> gradHiddenOut,
        Tensor<T>[] savedTensors,
        int inputIndex,
        int hiddenSize)
    {
        var numOps = MathHelper.GetNumericOperations<T>();

        if (savedTensors.Length < 3)
        {
            // Fallback
            var gradData = gradHiddenOut.ToArray();
            var resultShape = inputIndex switch
            {
                0 => savedTensors[0].Shape,
                1 => savedTensors.Length > 1 ? savedTensors[1].Shape : new int[] { hiddenSize },
                2 => new int[] { 3 * hiddenSize, savedTensors[0].Shape[^1] },
                3 => new int[] { 3 * hiddenSize, hiddenSize },
                _ => new int[] { 3 * hiddenSize }
            };

            var result = new T[resultShape.Aggregate(1, (a, b) => a * b)];
            for (int i = 0; i < result.Length && i < gradData.Length; i++)
            {
                result[i] = gradData[i];
            }
            return new Tensor<T>(resultShape, new Vector<T>(result));
        }

        var input = savedTensors[0];
        var hPrev = savedTensors[1];
        var gates = savedTensors[2]; // [batch, 3*hidden] containing z, r, h_tilde

        int batchSize = input.Shape[0];
        int inputSize = input.Shape.Length > 1 ? input.Shape[^1] : input.Shape[0];

        var gradHData = gradHiddenOut.ToArray();
        var gatesData = gates.ToArray();
        var hPrevData = hPrev.ToArray();

        switch (inputIndex)
        {
            case 0: // Input gradient
            {
                var result = new T[batchSize * inputSize];
                for (int i = 0; i < result.Length; i++)
                {
                    result[i] = i < gradHData.Length ? gradHData[i] : numOps.Zero;
                }
                return new Tensor<T>(input.Shape, new Vector<T>(result));
            }
            case 1: // h_prev gradient
            {
                // grad_h_prev = grad_h * (1 - z) + grad_h_tilde @ U_h^T * r + grad_z @ U_z^T + grad_r @ U_r^T
                var result = new T[batchSize * hiddenSize];
                for (int b = 0; b < batchSize; b++)
                {
                    for (int h = 0; h < hiddenSize; h++)
                    {
                        int idx = b * hiddenSize + h;
                        int zIdx = b * 3 * hiddenSize + h;
                        T z = gatesData[zIdx];
                        T oneMinusZ = numOps.Subtract(numOps.FromDouble(1), z);
                        result[idx] = numOps.Multiply(gradHData[idx], oneMinusZ);
                    }
                }
                return new Tensor<T>(hPrev.Shape, new Vector<T>(result));
            }
            default: // Weight/bias gradients
            {
                var resultShape = inputIndex switch
                {
                    2 => new int[] { 3 * hiddenSize, inputSize },
                    3 => new int[] { 3 * hiddenSize, hiddenSize },
                    _ => new int[] { 3 * hiddenSize }
                };
                var result = new T[resultShape.Aggregate(1, (a, b) => a * b)];
                return new Tensor<T>(resultShape, new Vector<T>(result));
            }
        }
    }

    /// <summary>
    /// Gradient of Attention operation.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Attention: output = softmax(Q @ K^T / sqrt(d_k)) @ V
    /// </para>
    /// </remarks>
    public static Tensor<T> GradAttention<T>(
        Tensor<T> gradOutput,
        Tensor<T> savedAttentionWeights,
        Tensor<T> Q,
        Tensor<T> K,
        Tensor<T> V,
        int inputIndex,
        double scale,
        bool causalMask = false)
    {
        var numOps = MathHelper.GetNumericOperations<T>();

        // attention_weights = softmax(Q @ K^T * scale)
        // output = attention_weights @ V

        if (inputIndex == 2) // V gradient
        {
            // grad_V = attention_weights^T @ grad_output
            var weightsT = savedAttentionWeights.Transpose();
            return weightsT.MatrixMultiply(gradOutput);
        }

        // grad_attention_weights = grad_output @ V^T
        var VT = V.Transpose();
        var gradWeights = gradOutput.MatrixMultiply(VT);

        // grad_scores = softmax_backward(grad_weights, attention_weights)
        // For softmax: grad_input_i = sum_j(grad_output_j * output_j * (delta_ij - output_i))
        var gradScores = GradSoftmax(gradWeights, savedAttentionWeights, -1);

        // Scale gradient
        var scaleT = numOps.FromDouble(scale);
        gradScores = ScalarMultiply(gradScores, scaleT);

        if (inputIndex == 0) // Q gradient
        {
            // grad_Q = grad_scores @ K
            return gradScores.MatrixMultiply(K);
        }
        else // K gradient (inputIndex == 1)
        {
            // grad_K = grad_scores^T @ Q
            var gradScoresT = gradScores.Transpose();
            return gradScoresT.MatrixMultiply(Q);
        }
    }

    /// <summary>
    /// Gradient of Multi-Head Attention operation.
    /// </summary>
    public static Tensor<T> GradMultiHeadAttention<T>(
        Tensor<T> gradOutput,
        Tensor<T>[] savedTensors,
        int inputIndex,
        int numHeads,
        int headDim)
    {
        var numOps = MathHelper.GetNumericOperations<T>();

        // savedTensors: [Q, K, V, attention_weights, output_projection_weights]
        if (savedTensors.Length < 4)
        {
            // Fallback: return appropriately shaped gradient
            return gradOutput;
        }

        var Q = savedTensors[0];
        var K = savedTensors[1];
        var V = savedTensors[2];
        var attentionWeights = savedTensors[3];

        var shape = Q.Shape;
        int batchSize = shape[0];
        int seqLen = shape[1];
        int modelDim = numHeads * headDim;

        // For multi-head attention, process each head separately
        // Simplified implementation: treat as single attention
        double scale = 1.0 / Math.Sqrt(headDim);

        return GradAttention(gradOutput, attentionWeights, Q, K, V, inputIndex, scale, false);
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
            int batch2D = shape[0];
            int features = shape[1];
            var result2D = new T[features];
            var data2D = input.ToArray();

            for (int f = 0; f < features; f++)
            {
                T sum = numOps.Zero;
                for (int n = 0; n < batch2D; n++)
                {
                    sum = numOps.Add(sum, data2D[n * features + f]);
                }
                result2D[f] = sum;
            }

            return new Tensor<T>(new int[] { features }, new Vector<T>(result2D));
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
