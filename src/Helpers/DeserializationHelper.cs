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
        if (!LayerTypes.TryGetValue(layerType, out Type? openGenericType))
        {
            throw new NotSupportedException($"Layer type {layerType} is not supported for deserialization.");
        }

        if (openGenericType == null)
        {
            throw new InvalidOperationException($"Type for layer {layerType} was registered as null.");
        }

        // Validate input/output shapes
        if (inputShape is null || inputShape.Length == 0)
        {
            throw new ArgumentException("Input shape must have at least one dimension.", nameof(inputShape));
        }
        if (outputShape is null || outputShape.Length == 0)
        {
            throw new ArgumentException("Output shape must have at least one dimension.", nameof(outputShape));
        }

        // Close the generic type with the actual type parameter T
        Type type = openGenericType.IsGenericTypeDefinition
            ? openGenericType.MakeGenericType(typeof(T))
            : openGenericType;

        // Get the generic type definition for comparison (handles both open and closed types)
        // All layer types should be generic; if not, throw a descriptive error
        if (!openGenericType.IsGenericType)
        {
            throw new InvalidOperationException($"Layer type {layerType} is not a generic type. All ILayer<T> implementations must be generic.");
        }
        Type genericDef = openGenericType.IsGenericTypeDefinition
            ? openGenericType
            : openGenericType.GetGenericTypeDefinition();

        // Prepare constructor and parameters based on layer type
        object? instance;

        if (genericDef == typeof(DenseLayer<>))
        {
            // DenseLayer(int inputSize, int outputSize, IActivationFunction<T>? activationFunction = null)
            // Use specific constructor to avoid ambiguity with vector activation constructor
            var activationFuncType = typeof(IActivationFunction<>).MakeGenericType(typeof(T));
            var ctor = type.GetConstructor([typeof(int), typeof(int), activationFuncType]);
            if (ctor is null)
            {
                throw new InvalidOperationException($"Cannot find DenseLayer constructor with (int, int, IActivationFunction<T>).");
            }
            instance = ctor.Invoke([inputShape[0], outputShape[0], null]);
        }
        else if (genericDef == typeof(ConvolutionalLayer<>))
        {
            // ConvolutionalLayer(int inputDepth, int outputDepth, int kernelSize, int inputHeight, int inputWidth, int stride, int padding, IActivationFunction<T>?)
            int kernelSize = additionalParams?.TryGetValue("FilterSize", out var fs) == true ? (int)fs : 3;
            int stride = additionalParams?.TryGetValue("Stride", out var s) == true ? (int)s : 1;
            int padding = additionalParams?.TryGetValue("Padding", out var p) == true ? (int)p : 0;
            // inputShape format: [height, width, depth]
            int inputDepth = inputShape.Length > 2 ? inputShape[2] : inputShape[0];
            int inputHeight = inputShape.Length > 0 ? inputShape[0] : 1;
            int inputWidth = inputShape.Length > 1 ? inputShape[1] : 1;
            int outputDepth = outputShape[0];

            var activationFuncType = typeof(IActivationFunction<>).MakeGenericType(typeof(T));
            var ctor = type.GetConstructor([typeof(int), typeof(int), typeof(int), typeof(int), typeof(int), typeof(int), typeof(int), activationFuncType]);
            if (ctor is null)
            {
                throw new InvalidOperationException($"Cannot find ConvolutionalLayer constructor.");
            }
            instance = ctor.Invoke([inputDepth, outputDepth, kernelSize, inputHeight, inputWidth, stride, padding, null]);
        }
        else if (genericDef == typeof(PoolingLayer<>))
        {
            // PoolingLayer(int inputDepth, int inputHeight, int inputWidth, int poolSize, int stride, PoolingType type)
            int poolSize = additionalParams?.TryGetValue("PoolSize", out var ps) == true ? (int)ps : 2;
            int stride = additionalParams?.TryGetValue("Stride", out var s) == true ? (int)s : 2;
            PoolingType poolingType = additionalParams?.TryGetValue("PoolingType", out var pt) == true
                ? (PoolingType)pt : PoolingType.Max;
            // inputShape format: [height, width, depth]
            int inputDepth = inputShape.Length > 2 ? inputShape[2] : inputShape[0];
            int inputHeight = inputShape.Length > 0 ? inputShape[0] : 1;
            int inputWidth = inputShape.Length > 1 ? inputShape[1] : 1;

            var ctor = type.GetConstructor([typeof(int), typeof(int), typeof(int), typeof(int), typeof(int), typeof(PoolingType)]);
            if (ctor is null)
            {
                throw new InvalidOperationException($"Cannot find PoolingLayer constructor.");
            }
            instance = ctor.Invoke([inputDepth, inputHeight, inputWidth, poolSize, stride, poolingType]);
        }
        else if (genericDef == typeof(ActivationLayer<>))
        {
            // ActivationLayer(int[] inputShape, IActivationFunction<T> activationFunction)
            ActivationFunction activationFunctionEnum = additionalParams?.TryGetValue("ActivationFunction", out var af) == true
                ? (ActivationFunction)af : ActivationFunction.ReLU;
            // Use ActivationFunctionFactory to create the IActivationFunction from enum
            var factoryType = typeof(ActivationFunctionFactory<>).MakeGenericType(typeof(T));
            var createMethod = factoryType.GetMethod("CreateActivationFunction", BindingFlags.Public | BindingFlags.Static);
            if (createMethod is null)
            {
                throw new InvalidOperationException("Cannot find ActivationFunctionFactory.CreateActivationFunction method.");
            }
            object? activationFunction = createMethod.Invoke(null, [activationFunctionEnum]);
            if (activationFunction is null)
            {
                throw new InvalidOperationException($"Failed to create activation function for {activationFunctionEnum}.");
            }

            // Use specific constructor to avoid ambiguity with vector activation constructor
            var activationFuncType = typeof(IActivationFunction<>).MakeGenericType(typeof(T));
            var ctor = type.GetConstructor([typeof(int[]), activationFuncType]);
            if (ctor is null)
            {
                throw new InvalidOperationException($"Cannot find ActivationLayer constructor with (int[], IActivationFunction<T>).");
            }
            instance = ctor.Invoke([inputShape, activationFunction]);
        }
        else
        {
            // Default: pass inputShape as first parameter
            instance = Activator.CreateInstance(type, [inputShape]);
        }
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
