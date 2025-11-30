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
        // Gradient to left input (minuend) is unchanged, gradient to right input (subtrahend) is negated
        return inputIndex == 0 ? gradOutput : NegateHelper(gradOutput);
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
}
