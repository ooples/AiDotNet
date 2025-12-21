namespace AiDotNet.Exceptions;

/// <summary>
/// Exception thrown when an operation is attempted before a required forward pass has been completed.
/// </summary>
public class ForwardPassRequiredException : AiDotNetException
{
    /// <summary>
    /// The name of the component where the exception occurred.
    /// </summary>
    public string ComponentName { get; }

    /// <summary>
    /// The type of the component where the exception occurred.
    /// </summary>
    public string ComponentType { get; }

    /// <summary>
    /// The operation that was attempted.
    /// </summary>
    public string Operation { get; }

    /// <summary>
    /// Initializes a new instance of the <see cref="ForwardPassRequiredException"/> class for a layer.
    /// </summary>
    /// <param name="layerName">The name of the layer where the exception occurred.</param>
    /// <param name="layerType">The type of the layer where the exception occurred.</param>
    public ForwardPassRequiredException(string layerName, string layerType)
        : base($"Forward pass must be called before backward pass in layer '{layerName}' of type {layerType}.")
    {
        ComponentName = layerName;
        ComponentType = layerType;
        Operation = "backward pass";
    }

    /// <summary>
    /// Initializes a new instance of the <see cref="ForwardPassRequiredException"/> class.
    /// </summary>
    /// <param name="componentName">The name of the component where the exception occurred.</param>
    /// <param name="componentType">The type of the component where the exception occurred.</param>
    /// <param name="operation">The operation that was attempted.</param>
    public ForwardPassRequiredException(string componentName, string componentType, string operation)
        : base($"Forward pass must be called before {operation} in {componentType} '{componentName}'.")
    {
        ComponentName = componentName;
        ComponentType = componentType;
        Operation = operation;
    }
}
