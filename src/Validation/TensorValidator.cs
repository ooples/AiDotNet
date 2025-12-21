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
    /// <param name="component">Optional. The component performing the validation.</param>
    /// <param name="operation">Optional. The operation being performed.</param>
    /// <exception cref="TensorShapeMismatchException">Thrown when the tensor shape doesn't match the expected shape.</exception>
    public static void ValidateShape<T>(Tensor<T> tensor, int[] expectedShape, string component = "", string operation = "")
    {
        var (resolvedComponent, resolvedOperation) = ValidationHelper<T>.ResolveCallerInfo(component, operation);

        if (!tensor.Shape.SequenceEqual(expectedShape))
        {
            throw new TensorShapeMismatchException(expectedShape, tensor.Shape, resolvedComponent, resolvedOperation);
        }
    }

    /// <summary>
    /// Validates that a forward pass has been performed before a dependent operation.
    /// </summary>
    /// <typeparam name="T">The numeric type of the tensor.</typeparam>
    /// <param name="lastInput">The input from the last forward pass.</param>
    /// <param name="componentName">Optional. The name of the component performing the validation.</param>
    /// <param name="componentType">Optional. The type of the component performing the validation.</param>
    /// <param name="operation">Optional. The operation being performed.</param>
    /// <exception cref="ForwardPassRequiredException">Thrown when an operation is attempted before a required forward pass.</exception>
    public static void ValidateForwardPassPerformed<T>(Tensor<T>? lastInput, string componentName = "", string componentType = "", string operation = "")
    {
        var (resolvedComponent, resolvedOperation) = ValidationHelper<T>.ResolveCallerInfo(componentName, operation);

        if (lastInput == null)
        {
            throw new ForwardPassRequiredException(resolvedComponent, componentType.Length > 0 ? componentType : resolvedComponent, resolvedOperation);
        }
    }

    /// <summary>
    /// Validates that a forward pass has been performed before a backward pass in a layer.
    /// </summary>
    /// <typeparam name="T">The numeric type of the tensor.</typeparam>
    /// <param name="lastInput">The input from the last forward pass.</param>
    /// <param name="layerName">Optional. The name of the layer.</param>
    /// <param name="layerType">Optional. The type of the layer.</param>
    /// <exception cref="ForwardPassRequiredException">Thrown when a backward pass is attempted before a forward pass.</exception>
    public static void ValidateForwardPassPerformedForLayer<T>(Tensor<T>? lastInput, string layerName = "", string layerType = "")
    {
        var (resolvedComponent, _) = ValidationHelper<T>.ResolveCallerInfo(layerName);

        if (lastInput == null)
        {
            throw new ForwardPassRequiredException(resolvedComponent, layerType.Length > 0 ? layerType : resolvedComponent);
        }
    }

    /// <summary>
    /// Validates that two tensors have the same shape.
    /// </summary>
    /// <typeparam name="T">The numeric type of the tensors.</typeparam>
    /// <param name="tensor1">The first tensor to compare.</param>
    /// <param name="tensor2">The second tensor to compare.</param>
    /// <param name="component">Optional. The component performing the validation.</param>
    /// <param name="operation">Optional. The operation being performed.</param>
    /// <exception cref="TensorShapeMismatchException">Thrown when the tensors have different shapes.</exception>
    public static void ValidateShapesMatch<T>(Tensor<T> tensor1, Tensor<T> tensor2, string component = "", string operation = "")
    {
        var (resolvedComponent, resolvedOperation) = ValidationHelper<T>.ResolveCallerInfo(component, operation);

        if (!tensor1.Shape.SequenceEqual(tensor2.Shape))
        {
            throw new TensorShapeMismatchException(tensor1.Shape, tensor2.Shape, resolvedComponent, resolvedOperation);
        }
    }

    /// <summary>
    /// Validates that a tensor has the expected rank (number of dimensions).
    /// </summary>
    /// <typeparam name="T">The numeric type of the tensor.</typeparam>
    /// <param name="tensor">The tensor to validate.</param>
    /// <param name="expectedRank">The expected rank of the tensor.</param>
    /// <param name="component">Optional. The component performing the validation.</param>
    /// <param name="operation">Optional. The operation being performed.</param>
    /// <exception cref="InvalidTensorRankException">Thrown when the tensor rank doesn't match the expected rank.</exception>
    public static void ValidateRank<T>(Tensor<T> tensor, int expectedRank, string component = "", string operation = "")
    {
        if (tensor.Rank != expectedRank)
        {
            var (resolvedComponent, resolvedOperation) = ValidationHelper<T>.ResolveCallerInfo(component, operation);
            throw new TensorRankException(expectedRank, tensor.Rank, resolvedComponent, resolvedOperation);
        }
    }
}
