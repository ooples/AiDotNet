global using AiDotNet.Exceptions;

namespace AiDotNet.Validation;

/// <summary>
/// Provides validation methods for tensors and neural network operations.
/// </summary>
public static class TensorValidator
{
    /// <summary>
    /// Validates that a tensor's shape matches the expected shape.
    /// </summary>
    /// <typeparam name="T">The numeric type of the tensor.</typeparam>
    /// <param name="tensor">The tensor to validate.</param>
    /// <param name="expectedShape">The expected shape of the tensor.</param>
    /// <param name="component">The component performing the validation.</param>
    /// <param name="operation">The operation being performed.</param>
    /// <exception cref="TensorShapeMismatchException">Thrown when the tensor shape doesn't match the expected shape.</exception>
    public static void ValidateShape<T>(Tensor<T> tensor, int[] expectedShape, string component, string operation)
    {
        if (!tensor.Shape.SequenceEqual(expectedShape))
        {
            throw new TensorShapeMismatchException(expectedShape, tensor.Shape, component, operation);
        }
    }

    /// <summary>
    /// Validates that a forward pass has been performed before a dependent operation.
    /// </summary>
    /// <typeparam name="T">The numeric type of the tensor.</typeparam>
    /// <param name="lastInput">The input from the last forward pass.</param>
    /// <param name="componentName">The name of the component performing the validation.</param>
    /// <param name="componentType">The type of the component performing the validation.</param>
    /// <param name="operation">The operation being performed.</param>
    /// <exception cref="ForwardPassRequiredException">Thrown when an operation is attempted before a required forward pass.</exception>
    public static void ValidateForwardPassPerformed<T>(Tensor<T>? lastInput, string componentName, string componentType, string operation)
    {
        if (lastInput == null)
        {
            throw new ForwardPassRequiredException(componentName, componentType, operation);
        }
    }

    /// <summary>
    /// Validates that a forward pass has been performed before a backward pass in a layer.
    /// </summary>
    /// <typeparam name="T">The numeric type of the tensor.</typeparam>
    /// <param name="lastInput">The input from the last forward pass.</param>
    /// <param name="layerName">The name of the layer.</param>
    /// <param name="layerType">The type of the layer.</param>
    /// <exception cref="ForwardPassRequiredException">Thrown when a backward pass is attempted before a forward pass.</exception>
    public static void ValidateForwardPassPerformedForLayer<T>(Tensor<T>? lastInput, string layerName, string layerType)
    {
        if (lastInput == null)
        {
            throw new ForwardPassRequiredException(layerName, layerType);
        }
    }

    /// <summary>
    /// Validates that two tensors have the same shape.
    /// </summary>
    /// <typeparam name="T">The numeric type of the tensors.</typeparam>
    /// <param name="tensor1">The first tensor to compare.</param>
    /// <param name="tensor2">The second tensor to compare.</param>
    /// <param name="component">The component performing the validation.</param>
    /// <param name="operation">The operation being performed.</param>
    /// <exception cref="TensorShapeMismatchException">Thrown when the tensors have different shapes.</exception>
    public static void ValidateShapesMatch<T>(Tensor<T> tensor1, Tensor<T> tensor2, string component, string operation)
    {
        if (!tensor1.Shape.SequenceEqual(tensor2.Shape))
        {
            throw new TensorShapeMismatchException(tensor1.Shape, tensor2.Shape, component, operation);
        }
    }

    /// <summary>
    /// Validates that a tensor has the expected rank (number of dimensions).
    /// </summary>
    /// <typeparam name="T">The numeric type of the tensor.</typeparam>
    /// <param name="tensor">The tensor to validate.</param>
    /// <param name="expectedRank">The expected rank of the tensor.</param>
    /// <param name="component">The component performing the validation.</param>
    /// <param name="operation">The operation being performed.</param>
    /// <exception cref="TensorRankException">Thrown when the tensor has an incorrect rank.</exception>
    public static void ValidateRank<T>(Tensor<T> tensor, int expectedRank, string component, string operation)
    {
        if (tensor.Shape.Length != expectedRank)
        {
            throw new TensorRankException(expectedRank, tensor.Shape.Length, component, operation);
        }
    }

    /// <summary>
    /// Validates that a tensor is not null.
    /// </summary>
    /// <typeparam name="T">The numeric type of the tensor.</typeparam>
    /// <param name="tensor">The tensor to validate.</param>
    /// <param name="paramName">The name of the parameter being validated.</param>
    /// <param name="component">The component performing the validation.</param>
    /// <param name="operation">The operation being performed.</param>
    /// <exception cref="ArgumentNullException">Thrown when the tensor is null.</exception>
    public static void ValidateNotNull<T>(Tensor<T>? tensor, string paramName, string component, string operation)
    {
        if (tensor == null)
        {
            throw new ArgumentNullException(
                paramName, 
                $"Tensor cannot be null in {component} during {operation}.");
        }
    }

    /// <summary>
    /// Validates that a dimension in a tensor shape matches an expected value.
    /// </summary>
    /// <typeparam name="T">The numeric type of the tensor.</typeparam>
    /// <param name="tensor">The tensor to validate.</param>
    /// <param name="dimensionIndex">The index of the dimension to check.</param>
    /// <param name="expectedValue">The expected value of the dimension.</param>
    /// <param name="component">The component performing the validation.</param>
    /// <param name="operation">The operation being performed.</param>
    /// <exception cref="TensorDimensionException">Thrown when the dimension doesn't match the expected value.</exception>
    /// <exception cref="ArgumentOutOfRangeException">Thrown when the dimension index is out of range.</exception>
    public static void ValidateDimension<T>(Tensor<T> tensor, int dimensionIndex, int expectedValue, string component, string operation)
    {
        if (dimensionIndex < 0 || dimensionIndex >= tensor.Shape.Length)
        {
            throw new ArgumentOutOfRangeException(
                nameof(dimensionIndex), 
                $"Dimension index {dimensionIndex} is out of range for tensor with rank {tensor.Shape.Length} in {component} during {operation}.");
        }
    
        if (tensor.Shape[dimensionIndex] != expectedValue)
        {
            throw new TensorDimensionException(dimensionIndex, expectedValue, tensor.Shape[dimensionIndex], component, operation);
        }
    }

    /// <summary>
    /// Validates that two tensors have shapes that are compatible for broadcasting operations.
    /// </summary>
    /// <typeparam name="T">The numeric type of the tensors.</typeparam>
    /// <param name="tensor1">The first tensor to compare.</param>
    /// <param name="tensor2">The second tensor to compare.</param>
    /// <param name="component">The component performing the validation.</param>
    /// <param name="operation">The operation being performed.</param>
    /// <exception cref="TensorShapeMismatchException">Thrown when the tensors have incompatible shapes for broadcasting.</exception>
    public static void ValidateBroadcastCompatibility<T>(Tensor<T> tensor1, Tensor<T> tensor2, string component, string operation)
    {
        if (!AreBroadcastCompatible(tensor1.Shape, tensor2.Shape))
        {
            throw new TensorShapeMismatchException(
                tensor1.Shape, 
                tensor2.Shape, 
                component, 
                $"{operation} (broadcasting incompatible)");
        }
    }

    /// <summary>
    /// Determines if two shapes are compatible for broadcasting operations.
    /// </summary>
    /// <param name="shape1">The first shape.</param>
    /// <param name="shape2">The second shape.</param>
    /// <returns>True if the shapes are broadcast compatible, false otherwise.</returns>
    private static bool AreBroadcastCompatible(int[] shape1, int[] shape2)
    {
        int rank1 = shape1.Length;
        int rank2 = shape2.Length;
        int maxRank = Math.Max(rank1, rank2);

        for (int i = 0; i < maxRank; i++)
        {
            int dim1 = i < rank1 ? shape1[rank1 - 1 - i] : 1;
            int dim2 = i < rank2 ? shape2[rank2 - 1 - i] : 1;

            // For broadcasting compatibility, dimensions must either be equal,
            // or one of them must be 1 (which can be broadcast to match the other)
            if (dim1 != dim2 && dim1 != 1 && dim2 != 1)
                return false;
        }

        return true;
    }
}