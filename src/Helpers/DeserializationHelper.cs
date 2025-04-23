global using System.Reflection;

namespace AiDotNet.Helpers;

public static class DeserializationHelper
{
    private static readonly Dictionary<string, Type> LayerTypes = [];

    static DeserializationHelper()
    {
        // Automatically discover and register all ILayer<T> implementations
        var layerTypes = Assembly.GetExecutingAssembly()
            .GetTypes()
            .Where(t => !t.IsAbstract && t.GetInterfaces()
                .Any(i => i.IsGenericType && i.GetGenericTypeDefinition() == typeof(ILayer<>)));

        foreach (var type in layerTypes)
        {
            LayerTypes[type.Name] = type;
        }
    }

    /// <summary>
    /// Creates a layer of the specified type during deserialization.
    /// </summary>
    /// <typeparam name="T">The numeric type used in the neural network.</typeparam>
    /// <param name="layerType">The type name of the layer to create.</param>
    /// <param name="inputShape">The input shape of the layer.</param>
    /// <param name="outputShape">The output shape of the layer.</param>
    /// <param name="additionalParams">Additional parameters needed for layer creation.</param>
    /// <returns>A new layer instance of the specified type.</returns>
    /// <remarks>
    /// <para>
    /// This method creates a new layer instance based on its type name and the specified input and output shapes.
    /// It uses reflection to dynamically create instances of layer types, allowing for easy addition of new layer types.
    /// Additional parameters can be provided for layers that require more information during instantiation.
    /// </para>
    /// <para><b>For Beginners:</b> This method is like a factory that can create any type of layer in our network.
    /// 
    /// When loading a saved network:
    /// - We need to recreate each layer with the correct settings
    /// - This method looks up how to create each type of layer
    /// - It sets up the layer with the right input and output sizes
    /// - For some layers, it uses extra information to set them up correctly
    /// 
    /// This design makes it easy to add new types of layers in the future without changing this method.
    /// </para>
    /// </remarks>
    public static ILayer<T> CreateLayerFromType<T>(string layerType, int[] inputShape, int[] outputShape, Dictionary<string, object>? additionalParams = null)
    {
        if (!LayerTypes.TryGetValue(layerType, out Type? type))
        {
            throw new NotSupportedException($"Layer type {layerType} is not supported for deserialization.");
        }

        if (type == null)
        {
            throw new InvalidOperationException($"Type for layer {layerType} was registered as null.");
        }

        // Prepare constructor parameters
        var parameters = new List<object> { inputShape };
        
        if (type == typeof(DenseLayer<T>))
        {
            parameters.Add(outputShape[0]);
        }
        else if (type == typeof(ConvolutionalLayer<T>))
        {
            int filterSize = additionalParams?.TryGetValue("FilterSize", out var fs) == true ? (int)fs : 3;
            int filterCount = outputShape[0];
            parameters.Add(filterSize);
            parameters.Add(filterCount);
        }
        else if (type == typeof(PoolingLayer<T>))
        {
            int poolSize = additionalParams?.TryGetValue("PoolSize", out var ps) == true ? (int)ps : 2;
            int stride = additionalParams?.TryGetValue("Stride", out var s) == true ? (int)s : 2;
            PoolingType poolingType = additionalParams?.TryGetValue("PoolingType", out var pt) == true 
                ? (PoolingType)pt : PoolingType.Max;
            parameters.Add(poolSize);
            parameters.Add(stride);
            parameters.Add(poolingType);
        }
        else if (type == typeof(ActivationLayer<T>))
        {
            ActivationFunction activationFunction = additionalParams?.TryGetValue("ActivationFunction", out var af) == true 
                ? (ActivationFunction)af : ActivationFunction.ReLU;
            parameters.Add(activationFunction);
        }
        
        // Use reflection to create the instance
        object? instance = Activator.CreateInstance(type, [.. parameters]);
        if (instance == null)
        {
            throw new InvalidOperationException($"Failed to create instance of layer type {layerType}.");
        }
        
        return (ILayer<T>)instance;
    }

    /// <summary>
    /// Deserializes and creates an instance of an interface based on the type name read from a BinaryReader.
    /// </summary>
    /// <typeparam name="TInterface">The interface type to deserialize.</typeparam>
    /// <param name="reader">The BinaryReader to read the type name from.</param>
    /// <returns>An instance of the deserialized interface, or null if no type name was provided.</returns>
    /// <exception cref="InvalidOperationException">Thrown when the type cannot be found or instantiated.</exception>
    public static TInterface? DeserializeInterface<TInterface>(BinaryReader reader) where TInterface : class
    {
        string typeName = reader.ReadString();
        if (string.IsNullOrEmpty(typeName))
        {
            return null;
        }

        Type? type = Type.GetType(typeName);
        if (type == null)
        {
            throw new InvalidOperationException($"Cannot find type {typeName}");
        }

        if (!typeof(TInterface).IsAssignableFrom(type))
        {
            throw new InvalidOperationException($"Type {typeName} does not implement interface {typeof(TInterface).Name}");
        }

        return (TInterface?)Activator.CreateInstance(type) 
            ?? throw new InvalidOperationException($"Failed to create instance of type {typeName}");
    }
}