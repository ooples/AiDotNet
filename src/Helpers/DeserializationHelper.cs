global using System.Reflection;

namespace AiDotNet.Helpers;

public static class DeserializationHelper
{
    private static readonly Dictionary<string, Type> LayerTypes = new Dictionary<string, Type>();

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
        // Allow layerType to contain serialized constructor metadata, e.g. "MultiHeadAttentionLayer;HeadCount=8".
        if (TryParseLayerTypeIdentifier(layerType, out var parsedTypeName, out var parsedParams))
        {
            layerType = parsedTypeName;
            additionalParams = MergeParams(additionalParams, parsedParams);
        }

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
            instance = CreateDenseLayer<T>(type, inputShape, outputShape, additionalParams);
        }
        else if (genericDef == typeof(InputLayer<>))
        {
            // InputLayer(int inputSize)
            var ctor = type.GetConstructor(new Type[] { typeof(int) });
            if (ctor is null)
            {
                throw new InvalidOperationException("Cannot find InputLayer constructor with (int).");
            }

            instance = ctor.Invoke(new object[] { inputShape[0] });
        }
        else if (genericDef == typeof(ReshapeLayer<>))
        {
            // ReshapeLayer(int[] inputShape, int[] outputShape)
            var ctor = type.GetConstructor(new Type[] { typeof(int[]), typeof(int[]) });
            if (ctor is null)
            {
                throw new InvalidOperationException("Cannot find ReshapeLayer constructor with (int[], int[]).");
            }

            instance = ctor.Invoke(new object[] { inputShape, outputShape });
        }
        else if (genericDef == typeof(EmbeddingLayer<>))
        {
            // EmbeddingLayer(int vocabularySize, int embeddingDimension)
            int embeddingDim = outputShape[0];
            int vocabSize = TryGetInt(additionalParams, "VocabularySize")
                ?? TryGetInt(additionalParams, "VocabSize")
                ?? throw new InvalidOperationException("EmbeddingLayer requires VocabularySize metadata for deserialization.");

            var ctor = type.GetConstructor(new Type[] { typeof(int), typeof(int) });
            if (ctor is null)
            {
                throw new InvalidOperationException("Cannot find EmbeddingLayer constructor with (int, int).");
            }
            instance = ctor.Invoke(new object[] { vocabSize, embeddingDim });
        }
        else if (genericDef == typeof(PositionalEncodingLayer<>))
        {
            // PositionalEncodingLayer(int maxSequenceLength, int embeddingSize)
            if (inputShape.Length < 2)
            {
                throw new InvalidOperationException("PositionalEncodingLayer requires input shape [maxSequenceLength, embeddingSize].");
            }

            int maxSeqLen = inputShape[0];
            int embDim = inputShape[1];

            var ctor = type.GetConstructor(new Type[] { typeof(int), typeof(int) });
            if (ctor is null)
            {
                throw new InvalidOperationException("Cannot find PositionalEncodingLayer constructor with (int, int).");
            }
            instance = ctor.Invoke(new object[] { maxSeqLen, embDim });
        }
        else if (genericDef == typeof(DropoutLayer<>))
        {
            // DropoutLayer(double dropoutRate = 0.5)
            double rate = TryGetDouble(additionalParams, "DropoutRate") ?? 0.5;
            var ctor = type.GetConstructor(new Type[] { typeof(double) });
            if (ctor is null)
            {
                throw new InvalidOperationException("Cannot find DropoutLayer constructor with (double).");
            }
            instance = ctor.Invoke(new object[] { rate });
        }
        else if (genericDef == typeof(LayerNormalizationLayer<>))
        {
            // LayerNormalizationLayer(int featureSize, double epsilon = ...)
            int featureSize = inputShape[0];
            double epsilon = TryGetDouble(additionalParams, "Epsilon") ?? NumericalStabilityHelper.LargeEpsilon;
            var ctor = type.GetConstructor(new Type[] { typeof(int), typeof(double) });
            if (ctor is null)
            {
                throw new InvalidOperationException("Cannot find LayerNormalizationLayer constructor with (int, double).");
            }
            instance = ctor.Invoke(new object[] { featureSize, epsilon });
        }
        else if (genericDef == typeof(BatchNormalizationLayer<>))
        {
            // BatchNormalizationLayer(int featureSize, double epsilon = ..., double momentum = ...)
            int featureSize = inputShape[0];
            double epsilon = TryGetDouble(additionalParams, "Epsilon") ?? NumericalStabilityHelper.LargeEpsilon;
            double momentum = TryGetDouble(additionalParams, "Momentum") ?? 0.9;
            var ctor = type.GetConstructor([typeof(int), typeof(double), typeof(double)]);
            if (ctor is null)
            {
                throw new InvalidOperationException("Cannot find BatchNormalizationLayer constructor with (int, double, double).");
            }
            instance = ctor.Invoke([featureSize, epsilon, momentum]);
        }
        else if (genericDef == typeof(MultiHeadAttentionLayer<>))
        {
            instance = CreateMultiHeadAttentionLayer<T>(type, inputShape, additionalParams);
        }
        else if (genericDef == typeof(SelfAttentionLayer<>))
        {
            // SelfAttentionLayer(int sequenceLength, int embeddingDimension, int headCount = 8, IActivationFunction<T>? = null)
            if (inputShape.Length < 2)
            {
                throw new InvalidOperationException("SelfAttentionLayer requires input shape [sequenceLength, embeddingDimension].");
            }

            int seqLen = inputShape[0];
            int embDim = inputShape[1];
            int headCount = TryGetInt(additionalParams, "HeadCount") ?? ResolveDefaultHeadCount(embDim);

            var activationFuncType = typeof(IActivationFunction<>).MakeGenericType(typeof(T));
            var ctor = type.GetConstructor(new Type[] { typeof(int), typeof(int), typeof(int), activationFuncType });
            if (ctor is null)
            {
                throw new InvalidOperationException("Cannot find SelfAttentionLayer constructor with (int, int, int, IActivationFunction<T>).");
            }
            object? activation = TryCreateActivationInstance(additionalParams, "ScalarActivationType", activationFuncType);
            instance = ctor.Invoke(new object?[] { seqLen, embDim, headCount, activation });
        }
        else if (genericDef == typeof(AttentionLayer<>))
        {
            // AttentionLayer(int inputSize, int attentionSize, IActivationFunction<T>? = null)
            int inputSize = inputShape[0];
            int attentionSize = outputShape[0];

            var activationFuncType = typeof(IActivationFunction<>).MakeGenericType(typeof(T));
            var ctor = type.GetConstructor(new Type[] { typeof(int), typeof(int), activationFuncType });
            if (ctor is null)
            {
                throw new InvalidOperationException("Cannot find AttentionLayer constructor with (int, int, IActivationFunction<T>).");
            }
            object? activation = TryCreateActivationInstance(additionalParams, "ScalarActivationType", activationFuncType);
            instance = ctor.Invoke(new object?[] { inputSize, attentionSize, activation });
        }
        else if (genericDef == typeof(GraphAttentionLayer<>))
        {
            // GraphAttentionLayer(int inputFeatures, int outputFeatures, int numHeads = 1, double alpha = 0.2, double dropoutRate = 0.0, IActivationFunction<T>? = null)
            int inputFeatures = inputShape[0];
            int outputFeatures = outputShape[0];
            int numHeads = TryGetInt(additionalParams, "NumHeads") ?? 1;
            double alpha = TryGetDouble(additionalParams, "Alpha") ?? 0.2;
            double dropout = TryGetDouble(additionalParams, "DropoutRate") ?? 0.0;

            var activationFuncType = typeof(IActivationFunction<>).MakeGenericType(typeof(T));
            var ctor = type.GetConstructor(new Type[] { typeof(int), typeof(int), typeof(int), typeof(double), typeof(double), activationFuncType });
            if (ctor is null)
            {
                throw new InvalidOperationException("Cannot find GraphAttentionLayer constructor with expected signature.");
            }
            object? activation = TryCreateActivationInstance(additionalParams, "ScalarActivationType", activationFuncType);
            instance = ctor.Invoke(new object?[] { inputFeatures, outputFeatures, numHeads, alpha, dropout, activation });
        }
        else if (genericDef == typeof(AiDotNet.NeuralNetworks.Attention.FlashAttentionLayer<>))
        {
            instance = CreateFlashAttentionLayer<T>(type, inputShape, additionalParams);
        }
        else if (genericDef == typeof(AiDotNet.Inference.CachedMultiHeadAttention<>))
        {
            instance = CreateCachedMultiHeadAttention<T>(type, inputShape, additionalParams);
        }
        else if (genericDef == typeof(AiDotNet.Inference.PagedCachedMultiHeadAttention<>))
        {
            instance = CreatePagedCachedMultiHeadAttention<T>(type, inputShape, additionalParams);
        }
        else if (genericDef == typeof(AiDotNet.LoRA.Adapters.MultiLoRAAdapter<>))
        {
            instance = CreateMultiLoRAAdapter<T>(type, inputShape, outputShape, additionalParams);
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
            var ctor = type.GetConstructor(new Type[] { typeof(int), typeof(int), typeof(int), typeof(int), typeof(int), typeof(int), typeof(int), activationFuncType });
            if (ctor is null)
            {
                throw new InvalidOperationException($"Cannot find ConvolutionalLayer constructor.");
            }
            instance = ctor.Invoke(new object?[] { inputDepth, outputDepth, kernelSize, inputHeight, inputWidth, stride, padding, null });
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

            var ctor = type.GetConstructor(new Type[] { typeof(int), typeof(int), typeof(int), typeof(int), typeof(int), typeof(PoolingType) });
            if (ctor is null)
            {
                throw new InvalidOperationException($"Cannot find PoolingLayer constructor.");
            }
            instance = ctor.Invoke(new object[] { inputDepth, inputHeight, inputWidth, poolSize, stride, poolingType });
        }
        else if (genericDef == typeof(ActivationLayer<>))
        {
            instance = CreateActivationLayer<T>(type, inputShape, additionalParams);
        }
        else
        {
            // Default: pass inputShape as first parameter
            var ctor = type.GetConstructor(new Type[] { typeof(int[]) });
            if (ctor is null)
            {
                throw new NotSupportedException(
                    $"Layer type {layerType} is not supported for deserialization (no known constructor found).");
            }

            instance = ctor.Invoke(new object[] { inputShape });
        }
        if (instance == null)
        {
            throw new InvalidOperationException($"Failed to create instance of layer type {layerType}.");
        }

        return (ILayer<T>)instance;
    }

    private static object CreateDenseLayer<T>(Type type, int[] inputShape, int[] outputShape, Dictionary<string, object>? additionalParams)
    {
        // DenseLayer(int inputSize, int outputSize, IActivationFunction<T>? activationFunction = null)
        // Use specific constructor to avoid ambiguity with vector activation constructor.
        var activationFuncType = typeof(IActivationFunction<>).MakeGenericType(typeof(T));
        var ctor = type.GetConstructor(new Type[] { typeof(int), typeof(int), activationFuncType });
        if (ctor is null)
        {
            throw new InvalidOperationException("Cannot find DenseLayer constructor with (int, int, IActivationFunction<T>).");
        }

        object? activation = TryCreateActivationInstance(additionalParams, "ScalarActivationType", activationFuncType);
        return ctor.Invoke(new object?[] { inputShape[0], outputShape[0], activation });
    }

    private static object CreateMultiHeadAttentionLayer<T>(Type type, int[] inputShape, Dictionary<string, object>? additionalParams)
    {
        // MultiHeadAttentionLayer(int sequenceLength, int embeddingDimension, int headCount, IActivationFunction<T>? activationFunction = null)
        if (inputShape.Length < 2)
        {
            throw new InvalidOperationException("MultiHeadAttentionLayer requires input shape [sequenceLength, embeddingDimension].");
        }

        int seqLen = inputShape[0];
        int embDim = inputShape[1];
        int headCount = TryGetInt(additionalParams, "HeadCount") ?? ResolveDefaultHeadCount(embDim);

        var activationFuncType = typeof(IActivationFunction<>).MakeGenericType(typeof(T));
        var ctor = type.GetConstructor(new Type[] { typeof(int), typeof(int), typeof(int), activationFuncType });
        if (ctor is null)
        {
            throw new InvalidOperationException("Cannot find MultiHeadAttentionLayer constructor with (int, int, int, IActivationFunction<T>).");
        }

        object? activation = TryCreateActivationInstance(additionalParams, "ScalarActivationType", activationFuncType);
        return ctor.Invoke(new object?[] { seqLen, embDim, headCount, activation });
    }

    private static object CreateFlashAttentionLayer<T>(Type type, int[] inputShape, Dictionary<string, object>? additionalParams)
    {
        // FlashAttentionLayer(int sequenceLength, int embeddingDimension, int headCount, FlashAttentionConfig config, IActivationFunction<T>? activationFunction = null)
        if (inputShape.Length < 2)
        {
            throw new InvalidOperationException("FlashAttentionLayer requires input shape [sequenceLength, embeddingDimension].");
        }

        int seqLen = inputShape[0];
        int embDim = inputShape[1];
        int headCount = TryGetInt(additionalParams, "HeadCount") ?? ResolveDefaultHeadCount(embDim);
        bool useCausal = TryGetBool(additionalParams, "UseCausalMask") ?? false;

        var flashConfig = AiDotNet.NeuralNetworks.Attention.FlashAttentionConfig.Default;
        flashConfig.UseCausalMask = useCausal;

        var activationFuncType = typeof(IActivationFunction<>).MakeGenericType(typeof(T));
        var ctor = type.GetConstructor(new Type[] { typeof(int), typeof(int), typeof(int), typeof(AiDotNet.NeuralNetworks.Attention.FlashAttentionConfig), activationFuncType });
        if (ctor is null)
        {
            throw new InvalidOperationException("Cannot find FlashAttentionLayer constructor with expected signature.");
        }

        object? activation = TryCreateActivationInstance(additionalParams, "ScalarActivationType", activationFuncType);
        return ctor.Invoke(new object?[] { seqLen, embDim, headCount, flashConfig, activation });
    }

    private static object CreateCachedMultiHeadAttention<T>(Type type, int[] inputShape, Dictionary<string, object>? additionalParams)
    {
        // CachedMultiHeadAttention(int sequenceLength, int embeddingDimension, int headCount, bool useFlashAttention, int layerIndex, bool useCausalMask, IActivationFunction<T>? activationFunction = null)
        if (inputShape.Length < 2)
        {
            throw new InvalidOperationException("CachedMultiHeadAttention requires input shape [sequenceLength, embeddingDimension].");
        }

        int seqLen = inputShape[0];
        int embDim = inputShape[1];
        int headCount = TryGetInt(additionalParams, "HeadCount") ?? ResolveDefaultHeadCount(embDim);
        bool useFlash = TryGetBool(additionalParams, "UseFlashAttention") ?? true;
        bool useCausal = TryGetBool(additionalParams, "UseCausalMask") ?? true;

        var activationFuncType = typeof(IActivationFunction<>).MakeGenericType(typeof(T));
        var ctor = type.GetConstructor(new Type[] { typeof(int), typeof(int), typeof(int), typeof(bool), typeof(int), typeof(bool), activationFuncType });
        if (ctor is null)
        {
            throw new InvalidOperationException("Cannot find CachedMultiHeadAttention constructor with expected signature.");
        }

        object? activation = TryCreateActivationInstance(additionalParams, "ScalarActivationType", activationFuncType);
        return ctor.Invoke(new object?[] { seqLen, embDim, headCount, useFlash, 0, useCausal, activation });
    }

    private static object CreatePagedCachedMultiHeadAttention<T>(Type type, int[] inputShape, Dictionary<string, object>? additionalParams)
    {
        // PagedCachedMultiHeadAttention(int sequenceLength, int embeddingDimension, int headCount, bool useCausalMask, IActivationFunction<T>? activationFunction = null)
        if (inputShape.Length < 2)
        {
            throw new InvalidOperationException("PagedCachedMultiHeadAttention requires input shape [sequenceLength, embeddingDimension].");
        }

        int seqLen = inputShape[0];
        int embDim = inputShape[1];
        int headCount = TryGetInt(additionalParams, "HeadCount") ?? ResolveDefaultHeadCount(embDim);
        bool useCausal = TryGetBool(additionalParams, "UseCausalMask") ?? true;

        var activationFuncType = typeof(IActivationFunction<>).MakeGenericType(typeof(T));
        var ctor = type.GetConstructor(new Type[] { typeof(int), typeof(int), typeof(int), typeof(bool), activationFuncType });
        if (ctor is null)
        {
            throw new InvalidOperationException("Cannot find PagedCachedMultiHeadAttention constructor with expected signature.");
        }

        object? activation = TryCreateActivationInstance(additionalParams, "ScalarActivationType", activationFuncType);
        return ctor.Invoke(new object?[] { seqLen, embDim, headCount, useCausal, activation });
    }

    private static object CreateMultiLoRAAdapter<T>(Type type, int[] inputShape, int[] outputShape, Dictionary<string, object>? additionalParams)
    {
        // MultiLoRAAdapter(ILayer<T> baseLayer, string defaultTaskName, int defaultRank, double alpha, bool freezeBaseLayer)
        bool freezeBaseLayer = TryGetBool(additionalParams, "FreezeBaseLayer") ?? true;

        string? encodedBaseLayerId = additionalParams?.TryGetValue("BaseLayerTypeId", out var baseType) == true ? baseType as string : null;
        string baseLayerIdentifier = !string.IsNullOrWhiteSpace(encodedBaseLayerId)
            ? Uri.UnescapeDataString(encodedBaseLayerId)
            : "DenseLayer`1";

        var baseLayer = CreateLayerFromType<T>(baseLayerIdentifier, inputShape, outputShape, null);

        static string[] ParseList(string? raw)
        {
            if (string.IsNullOrWhiteSpace(raw)) return Array.Empty<string>();
            return raw!.Split(new[] { '|' }, StringSplitOptions.RemoveEmptyEntries);
        }

        static int[] ParseIntList(string? raw)
        {
            var parts = ParseList(raw);
            var result = new int[parts.Length];
            for (int i = 0; i < parts.Length; i++)
            {
                result[i] = int.TryParse(parts[i], System.Globalization.NumberStyles.Integer, System.Globalization.CultureInfo.InvariantCulture, out var v) ? v : 1;
            }
            return result;
        }

        static double[] ParseDoubleList(string? raw)
        {
            var parts = ParseList(raw);
            var result = new double[parts.Length];
            for (int i = 0; i < parts.Length; i++)
            {
                result[i] = double.TryParse(parts[i], System.Globalization.NumberStyles.Float, System.Globalization.CultureInfo.InvariantCulture, out var v) ? v : -1;
            }
            return result;
        }

        string? tasksRaw = additionalParams?.TryGetValue("Tasks", out var tasksObj) == true ? tasksObj as string : null;
        var encodedTasks = ParseList(tasksRaw);
        if (encodedTasks.Length == 0)
        {
            encodedTasks = new string[] { "default" };
        }

        var tasks = encodedTasks.Select(Uri.UnescapeDataString).ToArray();
        var ranks = ParseIntList(additionalParams?.TryGetValue("TaskRanks", out var ranksObj) == true ? ranksObj as string : null);
        var alphas = ParseDoubleList(additionalParams?.TryGetValue("TaskAlphas", out var alphasObj) == true ? alphasObj as string : null);

        int defaultRank = ranks.Length > 0 ? ranks[0] : 1;
        double defaultAlpha = alphas.Length > 0 ? alphas[0] : -1;

        var iLayerType = typeof(ILayer<>).MakeGenericType(typeof(T));
        var ctor = type.GetConstructor(new Type[] { iLayerType, typeof(string), typeof(int), typeof(double), typeof(bool) });
        if (ctor is null)
        {
            throw new InvalidOperationException("Cannot find MultiLoRAAdapter constructor with expected signature.");
        }

        var instance = ctor.Invoke(new object[] { baseLayer, tasks[0], defaultRank, defaultAlpha, freezeBaseLayer });
        var multi = (AiDotNet.LoRA.Adapters.MultiLoRAAdapter<T>)instance;

        for (int taskIndex = 1; taskIndex < tasks.Length; taskIndex++)
        {
            int rank = taskIndex < ranks.Length ? ranks[taskIndex] : defaultRank;
            double alpha = taskIndex < alphas.Length ? alphas[taskIndex] : -1;
            multi.AddTask(tasks[taskIndex], rank, alpha);
        }

        if (additionalParams?.TryGetValue("CurrentTask", out var currentTaskObj) == true &&
            currentTaskObj is string currentTaskEncoded)
        {
            string currentTask = Uri.UnescapeDataString(currentTaskEncoded);
            if (!string.IsNullOrWhiteSpace(currentTask))
            {
                multi.SetCurrentTask(currentTask);
            }
        }

        return instance;
    }

    private static object CreateActivationLayer<T>(Type type, int[] inputShape, Dictionary<string, object>? additionalParams)
    {
        // ActivationLayer(int[] inputShape, IActivationFunction<T> activationFunction)
        var scalarActivationType = typeof(IActivationFunction<>).MakeGenericType(typeof(T));
        var vectorActivationType = typeof(IVectorActivationFunction<>).MakeGenericType(typeof(T));

        object? vectorActivation = TryCreateActivationInstance(additionalParams, "VectorActivationType", vectorActivationType);
        object? scalarActivation = TryCreateActivationInstance(additionalParams, "ScalarActivationType", scalarActivationType);

        object? activationFunction = vectorActivation ?? scalarActivation;

        if (activationFunction == null)
        {
            // Back-compat fallback: use enum if available, otherwise default ReLU.
            ActivationFunction activationFunctionEnum = additionalParams?.TryGetValue("ActivationFunction", out var af) == true
                ? (ActivationFunction)af : ActivationFunction.ReLU;

            var factoryType = typeof(ActivationFunctionFactory<>).MakeGenericType(typeof(T));
            var createMethod = factoryType.GetMethod("CreateActivationFunction", BindingFlags.Public | BindingFlags.Static);
            if (createMethod is null)
            {
                throw new InvalidOperationException("Cannot find ActivationFunctionFactory.CreateActivationFunction method.");
            }

            activationFunction = createMethod.Invoke(null, new object[] { activationFunctionEnum });
        }

        if (activationFunction == null)
        {
            throw new InvalidOperationException("Failed to create activation function for ActivationLayer.");
        }

        if (vectorActivationType.IsInstanceOfType(activationFunction))
        {
            var ctor = type.GetConstructor(new Type[] { typeof(int[]), vectorActivationType });
            if (ctor is null)
            {
                throw new InvalidOperationException("Cannot find ActivationLayer constructor with (int[], IVectorActivationFunction<T>).");
            }
            return ctor.Invoke(new object[] { inputShape, activationFunction });
        }

        var scalarCtor = type.GetConstructor(new Type[] { typeof(int[]), scalarActivationType });
        if (scalarCtor is null)
        {
            throw new InvalidOperationException("Cannot find ActivationLayer constructor with (int[], IActivationFunction<T>).");
        }
        return scalarCtor.Invoke(new object[] { inputShape, activationFunction });
    }

    private static bool TryParseLayerTypeIdentifier(
        string identifier,
        out string typeName,
        out Dictionary<string, object> parameters)
    {
        typeName = identifier;
        parameters = new Dictionary<string, object>(StringComparer.Ordinal);

        int sep = identifier.IndexOf(';');
        if (sep < 0)
        {
            return false;
        }

        typeName = identifier.Substring(0, sep);
        var parts = identifier.Substring(sep + 1).Split(new[] { ';' }, StringSplitOptions.RemoveEmptyEntries);
        foreach (var part in parts)
        {
            int eq = part.IndexOf('=');
            if (eq <= 0 || eq == part.Length - 1)
            {
                continue;
            }

            string key = part.Substring(0, eq);
            string value = part.Substring(eq + 1);

            if (int.TryParse(value, System.Globalization.NumberStyles.Integer, System.Globalization.CultureInfo.InvariantCulture, out int i))
            {
                parameters[key] = i;
            }
            else if (long.TryParse(value, System.Globalization.NumberStyles.Integer, System.Globalization.CultureInfo.InvariantCulture, out long l))
            {
                parameters[key] = l;
            }
            else if (double.TryParse(value, System.Globalization.NumberStyles.Float, System.Globalization.CultureInfo.InvariantCulture, out double d))
            {
                parameters[key] = d;
            }
            else if (bool.TryParse(value, out bool b))
            {
                parameters[key] = b;
            }
            else
            {
                parameters[key] = value;
            }
        }

        return true;
    }

    private static Dictionary<string, object> MergeParams(
        Dictionary<string, object>? original,
        Dictionary<string, object> parsed)
    {
        if (original == null || original.Count == 0)
        {
            return parsed;
        }

        foreach (var kvp in parsed)
        {
            original[kvp.Key] = kvp.Value;
        }

        return original;
    }

    private static int? TryGetInt(Dictionary<string, object>? parameters, string key)
    {
        if (parameters != null && parameters.TryGetValue(key, out var value) && value != null)
        {
            if (value is int i)
                return i;
            if (value is long l && l >= int.MinValue && l <= int.MaxValue)
                return (int)l;
            if (int.TryParse(value.ToString() ?? string.Empty, out int parsed))
                return parsed;
        }
        return null;
    }

    private static double? TryGetDouble(Dictionary<string, object>? parameters, string key)
    {
        if (parameters != null && parameters.TryGetValue(key, out var value) && value != null)
        {
            if (value is double d)
                return d;
            if (double.TryParse(value.ToString() ?? string.Empty, System.Globalization.NumberStyles.Float, System.Globalization.CultureInfo.InvariantCulture, out double parsed))
                return parsed;
        }
        return null;
    }

    private static bool? TryGetBool(Dictionary<string, object>? parameters, string key)
    {
        if (parameters != null && parameters.TryGetValue(key, out var value) && value != null)
        {
            if (value is bool b)
                return b;
            if (bool.TryParse(value.ToString() ?? string.Empty, out bool parsed))
                return parsed;
        }
        return null;
    }

    private static object? TryCreateActivationInstance(
        Dictionary<string, object>? parameters,
        string key,
        Type expectedInterface)
    {
        if (parameters == null || !parameters.TryGetValue(key, out var value) || value == null)
        {
            return null;
        }

        string? typeName = value as string ?? value.ToString() ?? string.Empty;
        if (string.IsNullOrWhiteSpace(typeName))
        {
            return null;
        }

        var type = Type.GetType(typeName, throwOnError: false);
        if (type == null)
        {
            return null;
        }

        try
        {
            var instance = Activator.CreateInstance(type);
            if (instance == null)
            {
                return null;
            }

            return expectedInterface.IsInstanceOfType(instance) ? instance : null;
        }
        catch (MissingMethodException)
        {
            return null;
        }
        catch (TargetInvocationException ex) when (ex.InnerException is MissingMethodException)
        {
            return null;
        }
        catch (Exception ex)
        {
            // Best-effort: deserialization should not throw if an optional activation cannot be created.
            System.Diagnostics.Debug.WriteLine($"Unexpected error deserializing activation {typeName}: {ex.Message}");
            return null;
        }
    }

    private static int ResolveDefaultHeadCount(int embeddingDimension)
    {
        // Conservative but practical default: prefer common head counts if divisible, otherwise fall back to 1.
        foreach (var candidate in new[] { 8, 4, 16, 12, 6, 2, 1 })
        {
            if (candidate > 0 && embeddingDimension % candidate == 0)
            {
                return candidate;
            }
        }
        return 1;
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

        try
        {
            return (TInterface?)Activator.CreateInstance(type)
                ?? throw new InvalidOperationException($"Failed to create instance of type {typeName}");
        }
        catch (MissingMethodException)
        {
            // Some implementations require constructor arguments.
            // Treat them as optional on deserialization and let callers provide sensible defaults.
            return null;
        }
        catch (TargetInvocationException ex) when (ex.InnerException is MissingMethodException)
        {
            // Same as above: no parameterless ctor available.
            return null;
        }
        catch (Exception ex)
        {
            throw new InvalidOperationException($"Failed to instantiate type {typeName}", ex);
        }
    }
}
