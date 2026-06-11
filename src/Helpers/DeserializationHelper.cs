global using System.Reflection;

namespace AiDotNet.Helpers;

public static class DeserializationHelper
{
    /// <summary>
    /// Structured marker exception thrown by an explicit-branch constructor
    /// lookup when the expected constructor signature is not present on the
    /// concrete layer type (typically because the layer was refactored away
    /// from that signature). The outer try/catch in
    /// <see cref="CreateLayerFromType{T}"/> uses this as the trigger to fall
    /// through to <see cref="TryConstructByMatchingMetadata{T}"/>. Replaces
    /// the brittle <c>ex.Message.StartsWith("Cannot find ")</c> convention.
    /// </summary>
    private sealed class MissingLayerCtorException : InvalidOperationException
    {
        public MissingLayerCtorException(string message) : base(message) { }
    }

    /// <summary>
    /// Defensive fallback: returns true when an InvalidOperationException's
    /// message matches the legacy "Cannot find &lt;layer name&gt; constructor"
    /// convention. As of #1239 every in-tree "Cannot find ... constructor"
    /// throw site in this file has migrated to
    /// <see cref="MissingLayerCtorException"/> (which inherits from
    /// InvalidOperationException, so existing catch blocks keep working),
    /// but this helper stays in place to handle third-party serialization
    /// paths or test-only layer types that might still surface the legacy
    /// form. Kept narrowly scoped to avoid swallowing unrelated
    /// InvalidOperationExceptions.
    /// </summary>
    private static bool IsMissingCtorMessage(string message)
    {
        return message.StartsWith("Cannot find ", StringComparison.Ordinal)
            && message.Contains("constructor", StringComparison.Ordinal);
    }

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

        // Determine if this is a shape-agnostic layer (layers that don't require specific input shapes)
        // These layers adapt to whatever input shape they receive at runtime
        Type genericDefForValidation = openGenericType.IsGenericTypeDefinition
            ? openGenericType
            : (openGenericType.IsGenericType ? openGenericType.GetGenericTypeDefinition() : openGenericType);

        bool isShapeAgnosticLayer = genericDefForValidation == typeof(AiDotNet.NeuralNetworks.Layers.DropoutLayer<>);

        // Validate input/output shapes (skip for shape-agnostic layers)
        if (!isShapeAgnosticLayer)
        {
            if (inputShape is null || inputShape.Length == 0)
            {
                throw new ArgumentException("Input shape must have at least one dimension.", nameof(inputShape));
            }
            if (outputShape is null || outputShape.Length == 0)
            {
                throw new ArgumentException("Output shape must have at least one dimension.", nameof(outputShape));
            }
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

        // Prepare constructor and parameters based on layer type. The if-chain
        // below is wrapped so that any explicit branch's "Cannot find ...
        // constructor" InvalidOperationException — which signals the layer's
        // ctor was refactored away — falls through to the reflection-driven
        // matcher rather than crashing the deserialization. The matcher then
        // attempts to find any working public constructor.
        object? instance;
        InvalidOperationException? branchFailure = null;
        try {
        if (genericDef == typeof(DenseLayer<>))
        {
            instance = CreateDenseLayer<T>(type, inputShape, outputShape, additionalParams);
        }
        else if (genericDef == typeof(NeuralNetworks.Layers.ReconstructionLayer<>))
        {
            // ReconstructionLayer(int inputDim, int hidden1Dim, int hidden2Dim, int outputDim, ...)
            // Two constructor overloads: scalar (IActivationFunction) vs vector (IVectorActivationFunction)
            int inputDim = inputShape[^1];
            int outputDim = outputShape[^1];
            int hidden1 = TryGetInt(additionalParams, "Hidden1Dim") ?? 512;
            int hidden2 = TryGetInt(additionalParams, "Hidden2Dim") ?? 1024;
            bool useVector = additionalParams != null
                && additionalParams.TryGetValue("UseVectorActivation", out var uvVal)
                && bool.TryParse(uvVal as string, out var uv) && uv;

            var vectorActivationType = typeof(IVectorActivationFunction<>).MakeGenericType(typeof(T));
            var scalarActivationType = typeof(IActivationFunction<>).MakeGenericType(typeof(T));
            var targetActivationType = useVector ? vectorActivationType : scalarActivationType;

            var ctor = type.GetConstructors()
                .FirstOrDefault(c => c.GetParameters().Length >= 4 &&
                    c.GetParameters().Take(4).All(p => p.ParameterType == typeof(int)) &&
                    (c.GetParameters().Length < 5 || c.GetParameters()[4].ParameterType == targetActivationType));
            if (ctor is null)
                throw new MissingLayerCtorException("Cannot find ReconstructionLayer constructor.");
            var args = new object?[ctor.GetParameters().Length];
            args[0] = inputDim;
            args[1] = hidden1;
            args[2] = hidden2;
            args[3] = outputDim;
            for (int pi = 4; pi < args.Length; pi++)
                args[pi] = ctor.GetParameters()[pi].HasDefaultValue ? ctor.GetParameters()[pi].DefaultValue : null;
            instance = ctor.Invoke(args);
        }
        else if (genericDef == typeof(NeuralNetworks.Layers.GlobalPoolingLayer<>))
        {
            // GlobalPoolingLayer(PoolingType poolingType, IActivationFunction<T>?) — lazy ctor.
            var poolingTypeStr = additionalParams != null && additionalParams.TryGetValue("PoolingType", out var ptVal)
                ? ptVal as string : null;
            var poolingType = !string.IsNullOrEmpty(poolingTypeStr) && Enum.TryParse<Enums.PoolingType>(poolingTypeStr, out var pt)
                ? pt : Enums.PoolingType.Average;

            // Restore activation from metadata
            object? activation = TryRestoreActivation<T>(additionalParams);

            var ctor = type.GetConstructors()
                .FirstOrDefault(c => c.GetParameters().Length >= 1 &&
                    c.GetParameters()[0].ParameterType == typeof(Enums.PoolingType));
            if (ctor is null)
                throw new MissingLayerCtorException("Cannot find GlobalPoolingLayer constructor.");
            var args = new object?[ctor.GetParameters().Length];
            args[0] = poolingType;
            if (args.Length > 1) args[1] = activation;
            instance = ctor.Invoke(args);
        }
        else if (genericDef == typeof(InputLayer<>))
        {
            // InputLayer(int inputSize)
            var ctor = type.GetConstructor(new Type[] { typeof(int) });
            if (ctor is null)
            {
                throw new MissingLayerCtorException("Cannot find InputLayer constructor with (int).");
            }

            instance = ctor.Invoke(new object[] { inputShape[0] });
        }
        else if (genericDef == typeof(ReshapeLayer<>))
        {
            // ReshapeLayer(int[] outputShape) — input shape resolved on first forward
            var ctor = type.GetConstructor(new Type[] { typeof(int[]) });
            if (ctor is null)
            {
                throw new MissingLayerCtorException("Cannot find ReshapeLayer constructor with (int[]).");
            }

            instance = ctor.Invoke(new object[] { outputShape });
        }
        else if (genericDef == typeof(FlattenLayer<>))
        {
            // FlattenLayer() — lazy: input shape resolved on first forward
            var ctor = type.GetConstructor(Type.EmptyTypes);
            if (ctor is null)
            {
                throw new MissingLayerCtorException("Cannot find FlattenLayer parameterless constructor.");
            }

            instance = ctor.Invoke(new object[0]);
        }
        else if (genericDef == typeof(TabNetEncoderLayer<>))
        {
            // TabNetEncoderLayer(numFeatures, decisionDim, attentionDim, numSteps,
            // numSharedLayers, numStepSpecificLayers, relaxationFactor, virtualBatchSize,
            // momentum, epsilon). numFeatures comes from the input shape; the rest are
            // persisted by the layer's GetMetadata.
            int tnNumFeatures = inputShape.Length > 0 ? inputShape[inputShape.Length - 1]
                : (TryGetInt(additionalParams, "NumFeatures") ?? 16);
            int tnDecisionDim = TryGetInt(additionalParams, "DecisionDim")
                ?? (outputShape.Length > 0 ? outputShape[outputShape.Length - 1] : 64);
            int tnAttentionDim = TryGetInt(additionalParams, "AttentionDim") ?? tnDecisionDim;
            int tnNumSteps = TryGetInt(additionalParams, "NumSteps") ?? 3;
            int tnNumShared = TryGetInt(additionalParams, "NumSharedLayers") ?? 2;
            int tnNumStep = TryGetInt(additionalParams, "NumStepSpecificLayers") ?? 2;
            double tnRelax = TryGetDouble(additionalParams, "RelaxationFactor") ?? 1.5;
            int tnVbs = TryGetInt(additionalParams, "VirtualBatchSize") ?? 128;
            double tnMomentum = TryGetDouble(additionalParams, "Momentum") ?? 0.02;
            double tnEpsilon = TryGetDouble(additionalParams, "Epsilon") ?? 1e-5;

            var tnCtor = type.GetConstructor(new[]
            {
                typeof(int), typeof(int), typeof(int), typeof(int), typeof(int),
                typeof(int), typeof(double), typeof(int), typeof(double), typeof(double)
            });
            if (tnCtor is null)
            {
                throw new MissingLayerCtorException("Cannot find TabNetEncoderLayer 10-arg constructor.");
            }

            instance = tnCtor.Invoke(new object[]
            {
                tnNumFeatures, tnDecisionDim, tnAttentionDim, tnNumSteps, tnNumShared,
                tnNumStep, tnRelax, tnVbs, tnMomentum, tnEpsilon
            });
        }
        else if (genericDef == typeof(TabMEnsembleLayer<>))
        {
            // TabMEnsembleLayer(numFeatures, int[] hiddenDimensions, outputDim, numMembers).
            // numFeatures from input shape; the rest from the layer's GetMetadata.
            int tmNumFeatures = inputShape.Length > 0 ? inputShape[inputShape.Length - 1]
                : (TryGetInt(additionalParams, "NumFeatures") ?? 16);
            int tmOutputDim = TryGetInt(additionalParams, "OutputDim")
                ?? (outputShape.Length > 0 ? outputShape[outputShape.Length - 1] : 1);
            int tmNumMembers = TryGetInt(additionalParams, "NumMembers") ?? 8;
            int[] tmHidden = TryGetIntArray(additionalParams, "HiddenDimensions") ?? new[] { 256, 256 };

            var tmCtor = type.GetConstructor(new[] { typeof(int), typeof(int[]), typeof(int), typeof(int) });
            if (tmCtor is null)
            {
                throw new MissingLayerCtorException("Cannot find TabMEnsembleLayer(int, int[], int, int) constructor.");
            }

            instance = tmCtor.Invoke(new object[] { tmNumFeatures, tmHidden, tmOutputDim, tmNumMembers });
        }
        else if (genericDef == typeof(FeatureTokenizerLayer<>))
        {
            // FeatureTokenizerLayer(int numFeatures, int embeddingDim) — both constructor dims are
            // the TRAILING two axes of the output shape: [..., numFeatures, embeddingDim]. Reading
            // the trailing axes (not [0],[1]) handles a saved batched shape [batch, F, E] correctly
            // — using the leading axes there would mistake `batch` for `numFeatures`.
            if (outputShape.Length < 2)
            {
                throw new MissingLayerCtorException(
                    "FeatureTokenizerLayer requires an output shape of rank >= 2 ending in "
                    + "[numFeatures, embeddingDim]; got [" + string.Join(",", outputShape) + "].");
            }

            var ctor = type.GetConstructor(new[] { typeof(int), typeof(int) });
            if (ctor is null)
            {
                throw new MissingLayerCtorException("Cannot find FeatureTokenizerLayer(int, int) constructor.");
            }

            instance = ctor.Invoke(new object[] { outputShape[^2], outputShape[^1] });
        }
        else if (genericDef == typeof(GandalfGFLULayer<>))
        {
            // GandalfGFLULayer(int numFeatures, int numStages). numFeatures is the trailing output
            // dim; numStages comes from metadata. The ctor eagerly resolves its sub-layers (probe
            // forward), so the constructed instance already has the right ParameterCount.
            int gfFeatures = outputShape.Length > 0 ? outputShape[^1] : inputShape[^1];
            int gfStages = TryGetInt(additionalParams, "NumStages") ?? 6;
            var gfCtor = type.GetConstructor(new[] { typeof(int), typeof(int) });
            if (gfCtor is null)
            {
                throw new MissingLayerCtorException("Cannot find GandalfGFLULayer(int, int) constructor.");
            }
            instance = gfCtor.Invoke(new object[] { gfFeatures, gfStages });
        }
        else if (genericDef == typeof(NodeEnsembleLayer<>))
        {
            // NodeEnsembleLayer(int numFeatures, int numTrees, int treeDepth, int treeOutputDim).
            // numFeatures is the input width; the rest come from metadata. Output is
            // [numTrees * treeOutputDim], so derive numFeatures from the SAVED input shape.
            int neFeatures = inputShape.Length > 0 ? inputShape[^1]
                : TryGetInt(additionalParams, "NumFeatures") ?? 1;
            int neTrees = TryGetInt(additionalParams, "NumTrees") ?? 20;
            int neDepth = TryGetInt(additionalParams, "TreeDepth") ?? 6;
            int neOutDim = TryGetInt(additionalParams, "TreeOutputDim") ?? 3;
            if (neFeatures <= 0) neFeatures = TryGetInt(additionalParams, "NumFeatures") ?? 1;
            var neCtor = type.GetConstructor(new[] { typeof(int), typeof(int), typeof(int), typeof(int), typeof(double) });
            if (neCtor is null)
            {
                throw new MissingLayerCtorException("Cannot find NodeEnsembleLayer(int, int, int, int, double) constructor.");
            }
            instance = neCtor.Invoke(new object[] { neFeatures, neTrees, neDepth, neOutDim, 0.01 });
        }
        else if (genericDef == typeof(TransposeLayer<>))
        {
            // TransposeLayer(int[] inputShape, int[] permutation)
            //
            // Prefer the permutation persisted in additionalParams (emitted by
            // TransposeLayer.GetMetadata). Fall back to shape inference only
            // when absent — inference is ambiguous whenever multiple input
            // axes share the same extent, and degenerate whenever the
            // permutation leaves the output shape equal to the input shape
            // (e.g. swapping two equal-size axes).
            int n = inputShape.Length;
            int[]? permutation = TryGetIntArray(additionalParams, "Permutation");

            if (permutation is not null)
            {
                if (permutation.Length != n)
                    throw new InvalidOperationException(
                        $"TransposeLayer deserialization: persisted Permutation length {permutation.Length} does not match inputShape rank {n}.");
            }
            else
            {
                if (inputShape.Length != outputShape.Length)
                    throw new InvalidOperationException(
                        $"TransposeLayer requires inputShape and outputShape to have the same rank. Got input rank {inputShape.Length}, output rank {outputShape.Length}.");

                permutation = new int[n];
                var used = new bool[n];
                for (int i = 0; i < n; i++)
                {
                    int found = -1;
                    for (int j = 0; j < n; j++)
                    {
                        if (!used[j] && inputShape[j] == outputShape[i])
                        {
                            found = j;
                            break;
                        }
                    }
                    if (found < 0)
                        throw new InvalidOperationException(
                            $"TransposeLayer deserialization: cannot recover permutation from shapes ({string.Join(",", inputShape)}) -> ({string.Join(",", outputShape)}). Re-serialize the network so the Permutation metadata is persisted.");
                    permutation[i] = found;
                    used[found] = true;
                }
            }

            var ctor = type.GetConstructor(new Type[] { typeof(int[]) });
            if (ctor is null)
            {
                throw new MissingLayerCtorException("Cannot find TransposeLayer constructor with (int[]).");
            }

            instance = ctor.Invoke(new object[] { permutation });
        }
        else if (genericDef == typeof(EmbeddingLayer<>))
        {
            // EmbeddingLayer(int vocabularySize, int embeddingDimension)
            int embeddingDim = outputShape[0];
            // EmbeddingLayer.GetMetadata persists VocabularySize on every
            // serialize call, so any properly-saved network has it. Refuse
            // to deserialize when missing rather than fabricating 256 (the
            // old "byte-level LM default") — a wrong vocab size produces
            // a structurally-incorrect embedding matrix that breaks weight
            // reattachment or silently changes semantics on legacy
            // metadata-less payloads. Surface the bad payload as an error
            // instead.
            int vocabSize = TryGetInt(additionalParams, "VocabularySize")
                ?? TryGetInt(additionalParams, "VocabSize")
                ?? throw new InvalidOperationException(
                    "EmbeddingLayer requires 'VocabularySize' (or legacy 'VocabSize') metadata. " +
                    "Re-serialize the network with the current GetMetadata implementation, " +
                    "or pass the vocab size via additionalParams when calling " +
                    "DeserializationHelper.CreateLayerFromType from a probe path.");

            var ctor = type.GetConstructor(new Type[] { typeof(int), typeof(int) });
            if (ctor is null)
            {
                throw new MissingLayerCtorException("Cannot find EmbeddingLayer constructor with (int, int).");
            }
            instance = ctor.Invoke(new object[] { vocabSize, embeddingDim });
            // Restore config properties set via object-initializer at build time
            // (the ctor only takes vocab/dim). Without this the transformer embedding
            // loses its forced Indices mode and the Vaswani §3.4 sqrt(d) scale on
            // deserialization, so a round-tripped model would behave differently.
            if (instance is EmbeddingLayer<T> embInstance)
            {
                // Restore InputMode / ScaleBySqrtDimension from metadata. A key that is ABSENT
                // keeps the ctor default (older models serialized before these knobs existed). A
                // key that is PRESENT but unparseable is a corrupt/incompatible stream — reject it
                // loudly rather than silently falling back to a default, which would round-trip the
                // model into DIFFERENT behavior (e.g. losing the Vaswani §3.4 sqrt(d) embedding
                // scale) with no error.
                if (additionalParams != null && additionalParams.TryGetValue("InputMode", out var modeObj))
                {
                    var modeStr = modeObj?.ToString();
                    if (!Enum.TryParse<EmbeddingInputMode>(modeStr, out var mode))
                        throw new InvalidOperationException(
                            $"EmbeddingLayer metadata 'InputMode' has an unparseable value '{modeStr}'. " +
                            $"Expected one of: {string.Join(", ", Enum.GetNames(typeof(EmbeddingInputMode)))}.");
                    embInstance.InputMode = mode;
                }
                if (additionalParams != null && additionalParams.TryGetValue("ScaleBySqrtDimension", out var scaleObj))
                {
                    var scaleStr = scaleObj?.ToString();
                    if (!bool.TryParse(scaleStr, out var scaleVal))
                        throw new InvalidOperationException(
                            $"EmbeddingLayer metadata 'ScaleBySqrtDimension' has an unparseable value " +
                            $"'{scaleStr}'. Expected 'true' or 'false'.");
                    embInstance.ScaleBySqrtDimension = scaleVal;
                }
            }
        }
        else if (genericDef == typeof(PatchEmbeddingLayer<>))
        {
            // PatchEmbeddingLayer(int patchSize, int embeddingDim) — lazy ctor;
            // image H/W and channels resolve from first Forward input.
            if (outputShape.Length < 2)
                throw new InvalidOperationException(
                    $"PatchEmbeddingLayer requires output shape [numPatches, embeddingDim] but got {outputShape.Length} dimensions.");

            int patchEmbedDim = outputShape[1];
            int numPatches = outputShape[0];
            // Use the trailing two axes as H/W so both CHW (rank 3) and NCHW
            // (rank 4) deserialize correctly. Hard-coding inputShape[1]/[2]
            // mapped to (C,H) for NCHW input and back-derived patchSize from
            // the wrong axes — broken weight reattachment for any saved
            // model whose recorded inputShape was rank-4.
            int imageHeight = inputShape.Length >= 2 ? inputShape[inputShape.Length - 2] : 0;
            int imageWidth = inputShape.Length >= 1 ? inputShape[inputShape.Length - 1] : 0;

            int patchSize;
            int? metadataPatchSize = TryGetInt(additionalParams, "PatchSize");
            if (metadataPatchSize.HasValue)
            {
                patchSize = metadataPatchSize.Value;
            }
            else if (numPatches > 0 && imageHeight > 0 && imageWidth > 0)
            {
                double sqrtVal = Math.Sqrt((double)numPatches * imageHeight / imageWidth);
                patchSize = sqrtVal > 0 ? (imageHeight / (int)Math.Round(sqrtVal)) : 16;
            }
            else
            {
                // Refuse to silently default — without metadata or shape data
                // we'd reconstruct a PatchEmbeddingLayer at the wrong grid
                // and SetParameters would fail the parameter-count check (or
                // worse, succeed and load weights into a wrong-shape kernel).
                // Surface the missing-info case loudly so the caller knows
                // the saved model needs the PatchSize entry in additionalParams
                // or an inputShape with concrete H/W.
                throw new InvalidOperationException(
                    "Cannot deserialize PatchEmbeddingLayer: PatchSize is missing " +
                    "from layer metadata AND inputShape does not carry concrete " +
                    "image dimensions to back-derive it from numPatches. Re-save " +
                    "the model on a build that emits PatchSize via " +
                    "PatchEmbeddingLayer.GetMetadata, or pass an inputShape with " +
                    "positive height/width when constructing the layer manually.");
            }

            // Prefer the LEGACY 2+activation+init ctor over any newer
            // overload that adds non-nullable trailing parameters. Older
            // saved models don't carry metadata for new args, and falling
            // through to a wider overload would either need that metadata
            // or trip the "no default for non-nullable value type" guard.
            // The legacy ctor signature is (int patchSize, int embeddingDim,
            // IActivationFunction<T>?, IInitializationStrategy<T>?) — both
            // trailing args are nullable reference types.
            var allCtors = type.GetConstructors()
                .Where(c => c.GetParameters().Length >= 2 &&
                    c.GetParameters()[0].ParameterType == typeof(int) &&
                    c.GetParameters()[1].ParameterType == typeof(int))
                .ToArray();
            var ctor = allCtors
                // Score 0 = ideal: only nullable trailing params. Score 1 =
                // has trailing value-type params (newer eager-channel ctor).
                .OrderBy(c => c.GetParameters()
                    .Skip(2)
                    .Any(p => p.ParameterType.IsValueType
                              && Nullable.GetUnderlyingType(p.ParameterType) is null) ? 1 : 0)
                .ThenBy(c => c.GetParameters().Length)  // prefer fewer args
                .FirstOrDefault();
            if (ctor is null)
            {
                throw new MissingLayerCtorException("Cannot find PatchEmbeddingLayer constructor.");
            }
            var ctorParams = ctor.GetParameters();
            var args = new object?[ctorParams.Length];
            args[0] = patchSize;
            args[1] = patchEmbedDim;
            for (int i = 2; i < ctorParams.Length; i++)
            {
                // Use the constructor's declared default value when
                // available, normalising sentinel values that
                // ParameterInfo can return:
                //   - Type.Missing  → ctor.Invoke fails when this is
                //                     forwarded as-is for value types.
                //   - DBNull.Value  → same problem.
                // Both indicate "no real default" and should be treated
                // like the no-default case below. For reference-type /
                // Nullable<T> params we fall back to null; for non-
                // nullable value types we throw because there's no safe
                // value to invent.
                object? defaultValue = ctorParams[i].HasDefaultValue
                    ? ctorParams[i].DefaultValue
                    : null;
                bool isSentinel = ReferenceEquals(defaultValue, Type.Missing)
                                  || defaultValue is DBNull;
                if (ctorParams[i].HasDefaultValue && !isSentinel)
                {
                    args[i] = defaultValue;
                }
                else if (!ctorParams[i].ParameterType.IsValueType ||
                         Nullable.GetUnderlyingType(ctorParams[i].ParameterType) is not null)
                {
                    args[i] = null;
                }
                else
                {
                    throw new InvalidOperationException(
                        $"PatchEmbeddingLayer constructor parameter '{ctorParams[i].Name}' " +
                        $"(type {ctorParams[i].ParameterType.Name}) has no default value and " +
                        "is a non-nullable value type. Cannot deserialize without the explicit " +
                        "value — re-save the model on a build that emits this parameter via " +
                        "GetMetadata.");
                }
            }
            instance = ctor.Invoke(args);
        }
        else if (genericDef == typeof(SpiralConvLayer<>))
        {
            // SpiralConvLayer(int outputChannels, int spiralLength, IActivationFunction<T>?).
            // OutputChannels + SpiralLength come from GetMetadata; InputChannels is lazy
            // and re-derived from the resolved input shape on the first forward. Without
            // these the generic ctor fallback picked a wrong SpiralLength, sizing the lazy
            // weights differently than the original and breaking Clone (#1450).
            int spOut = TryGetInt(additionalParams, "OutputChannels")
                ?? (outputShape is { Length: > 0 } ? outputShape[^1] : throw new InvalidOperationException(
                    "SpiralConvLayer deserialize: missing OutputChannels metadata and no usable output shape."));
            int spLen = TryGetInt(additionalParams, "SpiralLength")
                ?? throw new InvalidOperationException(
                    "SpiralConvLayer deserialize: missing SpiralLength metadata — re-save the model on a build "
                    + "that emits it via GetMetadata.");
            // SpiralConvLayer exposes both a scalar- and a vector-activation
            // constructor. Route the restored activation to the matching ctor so
            // a vector-configured layer round-trips with its real activation
            // instead of silently falling back to scalar behavior.
            var spActObj = TryRestoreActivation<T>(additionalParams);
            if (spActObj is IVectorActivationFunction<T> spVecAct)
            {
                var spVecCtor = type.GetConstructor(new[] { typeof(int), typeof(int), typeof(IVectorActivationFunction<T>) });
                if (spVecCtor is null)
                    throw new MissingLayerCtorException("Cannot find SpiralConvLayer(int, int, IVectorActivationFunction<T>) constructor.");
                instance = spVecCtor.Invoke(new object?[] { spOut, spLen, spVecAct });
            }
            else
            {
                var spAct = spActObj as IActivationFunction<T>;
                var spCtor = type.GetConstructor(new[] { typeof(int), typeof(int), typeof(IActivationFunction<T>) });
                if (spCtor is null)
                    throw new MissingLayerCtorException("Cannot find SpiralConvLayer(int, int, IActivationFunction<T>) constructor.");
                instance = spCtor.Invoke(new object?[] { spOut, spLen, spAct });
            }
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
                throw new MissingLayerCtorException("Cannot find PositionalEncodingLayer constructor with (int, int).");
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
                throw new MissingLayerCtorException("Cannot find DropoutLayer constructor with (double).");
            }
            instance = ctor.Invoke(new object[] { rate });
        }
        else if (genericDef == typeof(LayerNormalizationLayer<>))
        {
            // LayerNormalizationLayer is now lazy: ctor signature is (double epsilon).
            // Feature size is inferred from input.Shape[^1] on first Forward.
            // The fallback ResolveFromShape below the per-branch dispatch picks up
            // the serialized inputShape so SetParameters has concrete shapes to load.
            double epsilon = TryGetDouble(additionalParams, "Epsilon") ?? NumericalStabilityHelper.LargeEpsilon;
            var ctor = type.GetConstructor(new Type[] { typeof(double) });
            if (ctor is null)
            {
                throw new MissingLayerCtorException("Cannot find LayerNormalizationLayer constructor with (double).");
            }
            instance = ctor.Invoke(new object[] { epsilon });
        }
        else if (genericDef == typeof(BatchNormalizationLayer<>))
        {
            // BatchNormalizationLayer is now lazy: ctor signature is (double epsilon, double momentum).
            // Feature size is inferred from input.Shape[^1] on first Forward.
            double epsilon = TryGetDouble(additionalParams, "Epsilon") ?? NumericalStabilityHelper.LargeEpsilon;
            double momentum = TryGetDouble(additionalParams, "Momentum") ?? 0.9;
            var ctor = type.GetConstructor([typeof(double), typeof(double)]);
            if (ctor is null)
            {
                throw new MissingLayerCtorException("Cannot find BatchNormalizationLayer constructor with (double, double).");
            }
            instance = ctor.Invoke([epsilon, momentum]);
        }
        else if (genericDef == typeof(MultiHeadAttentionLayer<>))
        {
            instance = CreateMultiHeadAttentionLayer<T>(type, inputShape, additionalParams);
        }
        else if (genericDef == typeof(TransformerEncoderLayer<>))
        {
            // The current TransformerEncoderLayer<T> constructor is (numHeads, feedForwardDim);
            // the embeddingSize is resolved lazily from the first forward call. Use the
            // saved input shape to pre-resolve the layer so deserialized weights can be
            // reattached without an extra warm-up forward pass.
            int embeddingSize = inputShape[^1];
            int numHeads = TryGetInt(additionalParams, "NumHeads") ?? ResolveDefaultHeadCount(embeddingSize);
            int feedForwardDim = TryGetInt(additionalParams, "FeedForwardDim")
                ?? TryGetInt(additionalParams, "FeedForwardDimension")
                ?? embeddingSize * 4;

            var ctor = type.GetConstructor(new Type[] { typeof(int), typeof(int) });
            if (ctor is null)
            {
                throw new MissingLayerCtorException("Cannot find TransformerEncoderLayer constructor with (int, int).");
            }
            instance = ctor.Invoke(new object[] { numHeads, feedForwardDim });

            // Pre-resolve and allocate sublayer weights from the saved inputShape so
            // SetParameters can populate them. We need full ResolveFromShape (not just
            // ResolveShapesOnly) because TransformerEncoderLayer constructs its
            // sublayers in EnsureInitialized.
            if (instance is LayerBase<T> layerBase && inputShape.Length > 0)
            {
                int[] resolvedShape = embeddingSize > 0
                    ? inputShape.Select(d => d > 0 ? d : 1).ToArray()
                    : (numHeads > 0 ? new[] { 1, numHeads * 64 } : new[] { 1, 64 });
                // Always pin the trailing axis to the embeddingSize the
                // saved-shape declared. The previous form only corrected
                // resolvedShape[^1] when it was <= 0 — a positive-but-stale
                // last dim (e.g. saved shape carries a placeholder 1) would
                // resolve the layer to the wrong embedding width and throw at
                // SetParameters or load weights into a wrong-shape kernel.
                if (embeddingSize > 0)
                    resolvedShape[^1] = embeddingSize;
                else if (resolvedShape[^1] <= 0)
                    resolvedShape[^1] = numHeads > 0 ? numHeads * 64 : 64;
                layerBase.ResolveFromShape(resolvedShape);
            }
        }
        else if (genericDef == typeof(IntersampleAttentionLayer<>))
        {
            // IntersampleAttentionLayer(int embeddingDim, int numHeads = 8, double dropoutRate = 0.1).
            // The embedding dim is the trailing axis of the saved shape; the four FC projections are
            // resolved eagerly in the ctor (probe forward), so the constructed instance already has
            // the right ParameterCount for SetParameters — no post-hoc ResolveFromShape needed.
            int embDim = outputShape.Length > 0 ? outputShape[^1] : inputShape[^1];
            int isaHeads = TryGetInt(additionalParams, "NumHeads") ?? ResolveDefaultHeadCount(embDim);
            double isaDropout = TryGetDouble(additionalParams, "DropoutRate") ?? 0.1;

            var isaCtor = type.GetConstructor(new[] { typeof(int), typeof(int), typeof(double) });
            if (isaCtor is null)
            {
                throw new MissingLayerCtorException("Cannot find IntersampleAttentionLayer(int, int, double) constructor.");
            }
            instance = isaCtor.Invoke(new object[] { embDim, isaHeads, isaDropout });
        }
        else if (genericDef == typeof(TransformerDecoderLayer<>))
        {
            // TransformerDecoderLayer(int numHeads, int feedForwardDim,
            //                          int sequenceLength = 512,
            //                          IActivationFunction<T>? ffnActivation = null)
            // _embeddingSize is resolved lazily from input.Shape[^1] on first Forward
            // (or eagerly via ResolveFromShape after this method returns).
            int embeddingSize = inputShape[^1];
            int numHeads = TryGetInt(additionalParams, "NumHeads") ?? ResolveDefaultHeadCount(embeddingSize);
            int feedForwardDim = TryGetInt(additionalParams, "FeedForwardDim")
                ?? TryGetInt(additionalParams, "FeedForwardDimension")
                ?? embeddingSize * 4;
            // Fallback hierarchy for sequenceLength:
            //   1. explicit "SequenceLength" metadata entry  (paper-faithful)
            //   2. derive from inputShape[0] when input is at least rank-2
            //   3. fallback to 1 for rank-1 feature-only inputs
            // The previous fallback used 512 (the transformer paper default)
            // which silently inflated memory/compute for feature-only inputs
            // that should be treated as a single token, and could OOM on
            // large embeddingSize because attention scores scale O(B·S²).
            // Callers that need a longer sequence must persist
            // "SequenceLength" explicitly in metadata.
            int sequenceLength = TryGetInt(additionalParams, "SequenceLength")
                ?? (inputShape.Length >= 2 ? inputShape[0] : 1);
            // Clamp non-positive sequenceLength to 1. Serialized
            // inputShape[0] can carry 0 or -1 (lazy / placeholder shapes)
            // — those would otherwise produce an invalid decoder
            // configuration that fails later inside the ctor with a less
            // actionable error.
            if (sequenceLength <= 0) sequenceLength = 1;

            var activationFuncType = typeof(IActivationFunction<>).MakeGenericType(typeof(T));
            object? activation = TryCreateActivationInstance(additionalParams, "FfnActivationType", activationFuncType);

            var ctor = type.GetConstructor(new Type[] { typeof(int), typeof(int), typeof(int), activationFuncType });
            if (ctor is null)
            {
                throw new MissingLayerCtorException(
                    "Cannot find TransformerDecoderLayer constructor with (int numHeads, int feedForwardDim, int sequenceLength, IActivationFunction<T>?).");
            }
            instance = ctor.Invoke(new object?[] { numHeads, feedForwardDim, sequenceLength, activation });
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
                throw new MissingLayerCtorException("Cannot find SelfAttentionLayer constructor with (int, int, int, IActivationFunction<T>).");
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
                throw new MissingLayerCtorException("Cannot find AttentionLayer constructor with (int, int, IActivationFunction<T>).");
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
            var initStrategyType = typeof(IInitializationStrategy<>).MakeGenericType(typeof(T));
            // Try 7-param constructor (with IInitializationStrategy) then 6-param fallback
            var ctor = type.GetConstructor(new Type[] { typeof(int), typeof(int), typeof(int), typeof(double), typeof(double), activationFuncType, initStrategyType })
                    ?? type.GetConstructor(new Type[] { typeof(int), typeof(int), typeof(int), typeof(double), typeof(double), activationFuncType });
            if (ctor is null)
            {
                throw new MissingLayerCtorException("Cannot find GraphAttentionLayer constructor with expected signature.");
            }
            object? activation = TryCreateActivationInstance(additionalParams, "ScalarActivationType", activationFuncType);

            // If the activation is LeakyReLU, restore the alpha parameter
            double? leakyAlpha = TryGetDouble(additionalParams, "LeakyReLUAlpha");
            if (leakyAlpha.HasValue && activation != null)
            {
                var leakyType = typeof(LeakyReLUActivation<>).MakeGenericType(typeof(T));
                if (leakyType.IsInstanceOfType(activation))
                {
                    activation = Activator.CreateInstance(leakyType, leakyAlpha.Value);
                }
            }

            instance = ctor.GetParameters().Length == 7
                ? ctor.Invoke(new object?[] { inputFeatures, outputFeatures, numHeads, alpha, dropout, activation, null })
                : ctor.Invoke(new object?[] { inputFeatures, outputFeatures, numHeads, alpha, dropout, activation });
        }
        else if (genericDef == typeof(GraphConvolutionalLayer<>))
        {
            // GraphConvolutionalLayer(int inputFeatures, int outputFeatures, IActivationFunction<T>? activationFunction = null)
            int inputFeatures = inputShape[0];
            int outputFeatures = outputShape[0];

            var activationFuncType = typeof(IActivationFunction<>).MakeGenericType(typeof(T));
            var ctor = type.GetConstructor(new Type[] { typeof(int), typeof(int), activationFuncType });
            if (ctor is null)
            {
                throw new MissingLayerCtorException("Cannot find GraphConvolutionalLayer constructor with expected signature.");
            }
            object? activation = TryCreateActivationInstance(additionalParams, "ScalarActivationType", activationFuncType);
            instance = ctor.Invoke(new object?[] { inputFeatures, outputFeatures, activation });
        }
        else if (genericDef == typeof(GraphSAGELayer<>))
        {
            // GraphSAGELayer(int inputFeatures, int outputFeatures, SAGEAggregatorType,
            //                bool normalize, IActivationFunction<T>?, IInitializationStrategy<T>?)
            // — Hamilton et al. 2017 "Inductive Representation Learning on Large Graphs":
            // GraphSAGE samples and aggregates neighborhood features. Default ctor
            // exposes the paper's per-layer parameters (input/output dim, aggregator
            // type, L2-normalize-output flag) plus AiDotNet's standard activation +
            // init-strategy slots. The init strategy defaults to Eager / Xavier; we
            // pass null so the layer applies its default at construction time.
            // Read feature dim from the LAST axis: serialized graph tensors are
            // [numNodes, features] (rank 2) or [batch, numNodes, features] (rank 3),
            // so axis 0 would be node count or batch — never the feature width.
            int inputFeatures = inputShape[inputShape.Length - 1];
            int outputFeatures = outputShape[outputShape.Length - 1];
            int aggType = TryGetInt(additionalParams, "AggregatorType") ?? 0;
            bool normalize = TryGetBool(additionalParams, "Normalize") ?? true;

            var activationFuncType = typeof(IActivationFunction<>).MakeGenericType(typeof(T));
            var initStrategyType = typeof(IInitializationStrategy<>).MakeGenericType(typeof(T));
            var ctor = type.GetConstructor(new Type[] { typeof(int), typeof(int), typeof(SAGEAggregatorType), typeof(bool), activationFuncType, initStrategyType });
            if (ctor is null)
            {
                throw new MissingLayerCtorException("Cannot find GraphSAGELayer constructor with expected signature.");
            }
            object? activation = TryCreateActivationInstance(additionalParams, "ScalarActivationType", activationFuncType);
            instance = ctor.Invoke(new object?[] { inputFeatures, outputFeatures, (SAGEAggregatorType)aggType, normalize, activation, null });
        }
        else if (genericDef == typeof(GraphIsomorphismLayer<>))
        {
            // GraphIsomorphismLayer(int inputFeatures, int outputFeatures, int mlpHiddenDim,
            //                       bool learnEpsilon, double epsilon,
            //                       IActivationFunction<T>?, IInitializationStrategy<T>?)
            // — Xu et al. 2019, "How Powerful are Graph Neural Networks?" (GIN).
            // GIN updates h_v <- MLP((1+epsilon) * h_v + sum_u h_u). Default ctor
            // exposes the paper's MLP hidden dim, the learnable / fixed epsilon
            // pair, plus the standard activation + init-strategy slots.
            // Read feature dim from the LAST axis: serialized graph tensors are
            // [numNodes, features] (rank 2) or [batch, numNodes, features] (rank 3),
            // so axis 0 would be node count or batch — never the feature width.
            int inputFeatures = inputShape[inputShape.Length - 1];
            int outputFeatures = outputShape[outputShape.Length - 1];
            // Default to -1, matching the constructor default at
            // GraphIsomorphismLayer.cs:163, which then resolves the MLP
            // hidden dim to outputFeatures inside the layer ctor (line 174).
            // Hard-coding 64 here would silently produce a different
            // MLP shape than the original network for any GIN whose
            // outputFeatures != 64, breaking weight reattachment.
            int mlpHiddenDim = TryGetInt(additionalParams, "MlpHiddenDim") ?? -1;
            bool learnEpsilon = TryGetBool(additionalParams, "LearnEpsilon") ?? true;
            double initialEpsilon = TryGetDouble(additionalParams, "InitialEpsilon") ?? 0.0;

            var activationFuncType = typeof(IActivationFunction<>).MakeGenericType(typeof(T));
            var initStrategyType = typeof(IInitializationStrategy<>).MakeGenericType(typeof(T));
            var ctor = type.GetConstructor(new Type[] { typeof(int), typeof(int), typeof(int), typeof(bool), typeof(double), activationFuncType, initStrategyType });
            if (ctor is null)
            {
                throw new MissingLayerCtorException("Cannot find GraphIsomorphismLayer constructor with expected signature.");
            }
            object? activation = TryCreateActivationInstance(additionalParams, "ScalarActivationType", activationFuncType);
            instance = ctor.Invoke(new object?[] { inputFeatures, outputFeatures, mlpHiddenDim, learnEpsilon, initialEpsilon, activation, null });
        }
        else if (genericDef == typeof(AiDotNet.NeuralNetworks.Layers.FlashAttentionLayer<>))
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
        else if (genericDef == typeof(NeuralNetworks.Layers.MemoryReadLayer<>))
        {
            // MemoryReadLayer(int memoryDimension, int outputDimension, IActivationFunction<T>?)
            // memoryDimension is a free parameter — the default MemoryNetwork wires
            // memory == output == embeddingSize, so fall back to output size if the
            // serialized metadata doesn't pin it explicitly. inputDimension is
            // resolved lazily on the first forward (lazy-shape contract).
            //
            // Output dim comes from the LAST axis of the serialized output shape,
            // not Shape[0]. A batched output shape `[batch, features]` would
            // otherwise reconstruct outputDim = batch (which is wrong, would
            // make weights `[memoryDim, batch]`). Picking the last axis matches
            // the MemoryReadLayer Forward contract (output features in the
            // trailing axis) and the BottleneckBlock width-axis fix shipped
            // in e0c78b820.
            int outputDim = outputShape[outputShape.Length - 1];
            int memoryDim = TryGetInt(additionalParams, "MemoryDimension") ?? outputDim;

            var activationFuncType = typeof(IActivationFunction<>).MakeGenericType(typeof(T));
            var ctor = type.GetConstructor(new Type[] { typeof(int), typeof(int), activationFuncType });
            if (ctor is null)
            {
                throw new MissingLayerCtorException("Cannot find MemoryReadLayer constructor with (int memoryDimension, int outputDimension, IActivationFunction<T>?).");
            }
            object? activation = TryCreateActivationInstance(additionalParams, "ScalarActivationType", activationFuncType);
            instance = ctor.Invoke(new object?[] { memoryDim, outputDim, activation });
        }
        else if (genericDef == typeof(NeuralNetworks.Layers.MemoryWriteLayer<>))
        {
            // MemoryWriteLayer(int memoryDimension, IActivationFunction<T>?)
            // inputDimension is resolved lazily on the first forward (lazy-shape contract).
            // Use the LAST axis of the serialized output shape so a batched
            // shape `[batch, memoryDim]` reconstructs the actual feature
            // dim, not the batch count.
            int memoryDim = TryGetInt(additionalParams, "MemoryDimension")
                ?? outputShape[outputShape.Length - 1];

            var activationFuncType = typeof(IActivationFunction<>).MakeGenericType(typeof(T));
            var ctor = type.GetConstructor(new Type[] { typeof(int), activationFuncType });
            if (ctor is null)
            {
                throw new MissingLayerCtorException("Cannot find MemoryWriteLayer constructor with (int memoryDimension, IActivationFunction<T>?).");
            }
            object? activation = TryCreateActivationInstance(additionalParams, "ScalarActivationType", activationFuncType);
            instance = ctor.Invoke(new object?[] { memoryDim, activation });
        }
        else if (genericDef == typeof(Conv1DLayer<>))
        {
            // Conv1DLayer(inputChannels?, outputChannels, kernelSize, dilation,
            // stride, padding?, activation?). The conv hyper-parameters are not
            // recoverable from the input/output shapes, so they come from the
            // metadata written by Conv1DLayer.GetMetadata(). When InputChannels
            // is known we use the eager ctor (concrete ParameterCount → the
            // per-layer Clone fast path and SetParameters both line up); the
            // lazy ctor still works because SetParameters infers inputChannels
            // from the restored parameter-vector length.
            int outputChannels = TryGetInt(additionalParams, "OutputChannels")
                ?? (outputShape.Length > 0 ? outputShape[0] : 1);
            int kernelSize = TryGetInt(additionalParams, "KernelSize") ?? 3;
            int dilation = TryGetInt(additionalParams, "Dilation") ?? 1;
            int stride = TryGetInt(additionalParams, "Stride") ?? 1;
            int padding = TryGetInt(additionalParams, "Padding") ?? ((kernelSize - 1) * dilation / 2);
            int? inputChannels = TryGetInt(additionalParams, "InputChannels");

            var activationFuncType = typeof(IActivationFunction<>).MakeGenericType(typeof(T));
            object? activation = TryCreateActivationInstance(additionalParams, "ScalarActivationType", activationFuncType);
            if (activation is null && additionalParams is not null && additionalParams.ContainsKey("ScalarActivationType"))
                throw new InvalidOperationException($"Failed to deserialize activation function of type '{additionalParams["ScalarActivationType"]}' for Conv1DLayer.");

            instance = (inputChannels.HasValue && inputChannels.Value > 0)
                ? new Conv1DLayer<T>(inputChannels.Value, outputChannels, kernelSize, dilation, stride, padding, activation as IActivationFunction<T>)
                : new Conv1DLayer<T>(outputChannels, kernelSize, dilation, stride, padding, activation as IActivationFunction<T>);
        }
        else if (genericDef == typeof(Conv1DTransposeLayer<>))
        {
            // Conv1DTransposeLayer(inputChannels?, outputChannels, kernelSize, stride,
            // padding?, outputPadding, dilation, activation?). Hyper-parameters come
            // from Conv1DTransposeLayer.GetMetadata(); eager ctor when InputChannels
            // is known, else lazy (SetParameters infers C_in from the vector length).
            int outputChannels = TryGetInt(additionalParams, "OutputChannels")
                ?? (outputShape.Length > 0 ? outputShape[0] : 1);
            int kernelSize = TryGetInt(additionalParams, "KernelSize") ?? 1;
            int stride = TryGetInt(additionalParams, "Stride") ?? 1;
            int outputPadding = TryGetInt(additionalParams, "OutputPadding") ?? 0;
            int dilation = TryGetInt(additionalParams, "Dilation") ?? 1;
            int padding = TryGetInt(additionalParams, "Padding") ?? ((kernelSize - stride) / 2);
            int? inputChannels = TryGetInt(additionalParams, "InputChannels");

            var activationFuncType = typeof(IActivationFunction<>).MakeGenericType(typeof(T));
            object? activation = TryCreateActivationInstance(additionalParams, "ScalarActivationType", activationFuncType);
            if (activation is null && additionalParams is not null && additionalParams.ContainsKey("ScalarActivationType"))
                throw new InvalidOperationException($"Failed to deserialize activation function of type '{additionalParams["ScalarActivationType"]}' for Conv1DTransposeLayer.");

            instance = (inputChannels.HasValue && inputChannels.Value > 0)
                ? new Conv1DTransposeLayer<T>(inputChannels.Value, outputChannels, kernelSize, stride, padding, outputPadding, dilation, activation as IActivationFunction<T>)
                : new Conv1DTransposeLayer<T>(outputChannels, kernelSize, stride, padding, outputPadding, dilation, activation as IActivationFunction<T>);
        }
        else if (genericDef == typeof(HiFiGANResBlockLayer<>))
        {
            // HiFiGANResBlockLayer(channels, kernelSizes?, dilations?) — fully
            // reconstructable from metadata; SetParameters restores the inner convs.
            int channels = TryGetInt(additionalParams, "Channels")
                ?? (outputShape.Length > 0 ? outputShape[0] : 1);
            int[]? kernelSizes = TryGetIntArray(additionalParams, "KernelSizes");
            int[]? dilations = TryGetIntArray(additionalParams, "Dilations");
            instance = new HiFiGANResBlockLayer<T>(channels, kernelSizes, dilations);
        }
        else if (genericDef == typeof(WaveNetResidualBlockLayer<>))
        {
            // WaveNetResidualBlockLayer(channels, kernelSize, dilation) — fully
            // reconstructable from metadata; SetParameters restores the inner convs.
            int channels = TryGetInt(additionalParams, "Channels")
                ?? (outputShape.Length > 0 ? outputShape[0] : 1);
            int kernelSize = TryGetInt(additionalParams, "KernelSize") ?? 3;
            int dilation = TryGetInt(additionalParams, "Dilation") ?? 1;
            instance = new WaveNetResidualBlockLayer<T>(channels, kernelSize, dilation);
        }
        else if (genericDef == typeof(ConvolutionalLayer<>))
        {
            // ConvolutionalLayer(int outputDepth, int kernelSize, int stride, int padding, IActivationFunction<T>?, IInitializationStrategy<T>?)
            // Spatial dims (H/W) and inputDepth are resolved on the first Forward call via OnFirstForward.
            // The serialized inputShape/outputShape arrays are used only to size the weights via SetParameters
            // after construction — they do not feed the constructor anymore.
            int kernelSize = TryGetInt(additionalParams, "FilterSize") ?? 3;
            int stride = TryGetInt(additionalParams, "Stride") ?? 1;
            int padding = TryGetInt(additionalParams, "Padding") ?? 0;
            // outputShape can be rank-4 [batch, depth, height, width] (NCHW) when
            // serialized after a batched forward, OR rank-3 [depth, height, width]
            // when GetOutputShape() returns the layer-only shape (no batch axis).
            // In rank-3, outputShape[1] is HEIGHT, not depth — reading axis 1 in
            // both cases produces wrong-depth weights on Clone() and breaks
            // SetParameters with "Expected N, but got M".
            int outputDepth = outputShape.Length switch
            {
                >= 4 => outputShape[1],
                3 => outputShape[0],
                _ => outputShape[0]
            };

            var activationFuncType = typeof(IActivationFunction<>).MakeGenericType(typeof(T));
            var initStrategyType = typeof(AiDotNet.Initialization.IInitializationStrategy<>).MakeGenericType(typeof(T));
            // Try the 7-param ctor first (added nonlinearityForInit for paper-faithful
            // Conv→BN→LeakyReLU init gain). Fall back to the 6-param ctor for
            // backwards compatibility with older builds.
            var ctor7 = type.GetConstructor(new Type[] { typeof(int), typeof(int), typeof(int), typeof(int), activationFuncType, initStrategyType, activationFuncType });
            var ctor = ctor7 ?? type.GetConstructor(new Type[] { typeof(int), typeof(int), typeof(int), typeof(int), activationFuncType, initStrategyType });
            if (ctor is null)
            {
                throw new MissingLayerCtorException($"Cannot find ConvolutionalLayer constructor.");
            }
            object? activation = TryCreateActivationInstance(additionalParams, "ScalarActivationType", activationFuncType);
            if (activation is null && additionalParams is not null && additionalParams.ContainsKey("ScalarActivationType"))
                throw new InvalidOperationException($"Failed to deserialize activation function of type '{additionalParams["ScalarActivationType"]}' for ConvolutionalLayer.");
            // Pass null for nonlinearityForInit on deserialization — weights are
            // restored from the saved parameter vector, so InitializeWeights
            // never runs and the gain choice is moot for the clone path.
            instance = ctor7 is not null
                ? ctor.Invoke(new object?[] { outputDepth, kernelSize, stride, padding, activation, null, null })
                : ctor.Invoke(new object?[] { outputDepth, kernelSize, stride, padding, activation, null });

            // Pre-resolve the lazy layer using the serialized inputShape so SetParameters
            // sees the correct InputDepth and the kernel/bias counts match the saved
            // parameter vector exactly. Without this the auto-resolve heuristic in
            // ConvolutionalLayer.SetParameters can pick a different InputDepth than the
            // original (especially when outputDepth × kernelSize² happens to factor the
            // saved parameter count more than one way), and Clone()/DeepCopy() throw
            // "Expected N parameters, but got M".
            // Saved inputShape format: [batch, channels, height, width] (NCHW); some
            // legacy paths serialize without the batch dim, so accept rank 3 too.
            // Note: the rank-validation switch below now handles ALL ranks (not
            // just >= 3) — rank-1 and rank-2 payloads fall through to the
            // default branch and throw, instead of silently bypassing the
            // ConvolutionalLayer pre-resolve when inputShape.Length < 3 left
            // the layer in its lazy state (PR #1389 review C8oz1 — gate moved
            // from inputShape.Length >= 3 to inputShape != null so malformed
            // ranks fail fast). The previous `>= 3` guard was a relic from
            // when the switch only had the rank-3/rank-4 cases and rank-1/2
            // would have crashed on `inputShape[2]` — now they're explicitly
            // rejected with a clear error.
            if (instance is ConvolutionalLayer<T> conv && inputShape != null)
            {
                // Saved-record axes: rank-4 = [batch, channels, H, W],
                // rank-3 = [channels, H, W] (legacy unbatched). Any other
                // rank is malformed and must fail fast — silently
                // reinterpreting a rank-5 or rank-6 payload's leading
                // axes as the legacy [C, H, W] layout would deserialize
                // ConvolutionalLayer with the wrong channels/InputDepth
                // and produce a Clone that disagrees with the original
                // model's contract several layers downstream.
                int savedInDepth, savedInH, savedInW;
                switch (inputShape.Length)
                {
                    case 4:
                        savedInDepth = inputShape[1];
                        savedInH = inputShape[2];
                        savedInW = inputShape[3];
                        break;
                    case 3:
                        savedInDepth = inputShape[0];
                        savedInH = inputShape[1];
                        savedInW = inputShape[2];
                        break;
                    default:
                        throw new InvalidOperationException(
                            $"ConvolutionalLayer deserialize: saved inputShape rank must be 3 ([C, H, W]) " +
                            $"or 4 ([N, C, H, W]); got rank {inputShape.Length} ([{string.Join(", ", inputShape)}]). " +
                            "This usually indicates a corrupted layer record or a forward-incompatible " +
                            "newer-format payload — abort deserialize rather than silently misinterpret " +
                            "the trailing axes.");
                }

                // Three branches based on what the saved inputShape resolved
                // by serialize time:
                //
                // (a) InputDepth concrete: pre-resolve so SetParameters sees
                //     the correct InputDepth and the kernel/bias counts match
                //     the saved parameter vector exactly. Without this the
                //     auto-resolve heuristic in ConvolutionalLayer.SetParameters
                //     can pick a different InputDepth than the original
                //     (especially when outputDepth × kernelSize² happens to
                //     factor the saved parameter count more than one way),
                //     and Clone()/DeepCopy() throw "Expected N parameters,
                //     but got M". Spatial dims that were never forwarded
                //     fall back to Math.Max(1, kernelSize) so
                //     ConvolutionalLayer.OnFirstForward's kernel-size
                //     constraint (inH + 2*Padding >= KernelSize) passes —
                //     DCGAN's discriminator (kernel=4, padding=1, needs
                //     inH >= 2) is the canary. The stored OutputShape after
                //     this resolve is a placeholder; the first real Forward
                //     call recomputes the actual output tensor dimensions
                //     from the real input.
                //
                // (b) InputDepth deferred (saved as -1 because the layer
                //     was serialized before its first Forward): skip the
                //     pre-resolve entirely. ConvolutionalLayer.SetParameters
                //     has its own auto-resolve fallback (~ line 1598) that
                //     derives InputDepth from the saved parameter vector's
                //     length — (length - OutputDepth) / (OutputDepth *
                //     KernelSize²) — and that fallback uses KernelSize as
                //     the spatial placeholder. Pre-resolving with a
                //     placeholder InputDepth=1 would have locked
                //     InputDepth=1 into the layer's state, then Forward
                //     with the real RGB-3 input would throw
                //     "Expected input depth 1, but got 3" before the lazy
                //     resolve had a chance to fire. This is the failure
                //     mode that surfaced on DCGAN clones where the
                //     discriminator's layers had never seen the
                //     [3, 64, 64] image input at clone time (the test's
                //     pre-clone Predict only runs the generator).
                //
                // (c) inputShape supplied but malformed (rank < 3 case
                //     handled by the outer guard).
                if (savedInDepth > 0)
                {
                    int spatialFallback = Math.Max(1, kernelSize);
                    int inH = savedInH > 0 ? savedInH : spatialFallback;
                    int inW = savedInW > 0 ? savedInW : spatialFallback;
                    conv.ResolveShapesOnly(new[] { savedInDepth, inH, inW });
                }
            }
        }
        else if (genericDef == typeof(Conv3DLayer<>))
        {
            // Conv3DLayer(int outputChannels, int kernelSize, int stride, int padding, IActivationFunction<T>?)
            // — lazy ctor; spatial dims (D/H/W) and inputChannels resolved on first Forward.
            // outputShape can be:
            //   rank-5 [batch, channels, depth, height, width] (NCDHW after batched forward)
            //   rank-4 [channels, depth, height, width]        (layer-only OutputShape)
            // Reading axis 1 in rank-4 returns DEPTH, not channels — same trap as
            // ConvolutionalLayer. Switch on rank explicitly.
            int outputChannels = outputShape.Length switch
            {
                >= 5 => outputShape[1],
                4 => outputShape[0],
                _ => outputShape.Length > 0 ? outputShape[0] : 1
            };
            int kernelSize = TryGetInt(additionalParams, "KernelSize") ?? 3;
            int stride = TryGetInt(additionalParams, "Stride") ?? 1;
            int padding = TryGetInt(additionalParams, "Padding") ?? 0;

            var activationFuncType = typeof(IActivationFunction<>).MakeGenericType(typeof(T));
            var ctor = type.GetConstructor(new Type[] { typeof(int), typeof(int), typeof(int), typeof(int), activationFuncType });
            if (ctor is null)
                throw new MissingLayerCtorException("Cannot find Conv3DLayer constructor with expected signature.");
            var activation = TryRestoreActivation<T>(additionalParams);
            instance = ctor.Invoke(new object?[] { outputChannels, kernelSize, stride, padding, activation });

            // Pre-resolve from saved inputShape so SetParameters sees the correct
            // InputChannels and matches the saved parameter count.
            // inputShape is rank-4 [channels, depth, height, width] for the layer-only
            // OutputShape, or rank-5 [batch, channels, depth, height, width] post-batch.
            if (instance is Conv3DLayer<T> conv3d && inputShape != null && inputShape.Length >= 4)
            {
                int inC, inD, inH, inW;
                if (inputShape.Length == 5)
                {
                    inC = inputShape[1] > 0 ? inputShape[1] : 1;
                    inD = inputShape[2] > 0 ? inputShape[2] : 1;
                    inH = inputShape[3] > 0 ? inputShape[3] : 1;
                    inW = inputShape[4] > 0 ? inputShape[4] : 1;
                }
                else
                {
                    inC = inputShape[0] > 0 ? inputShape[0] : 1;
                    inD = inputShape[1] > 0 ? inputShape[1] : 1;
                    inH = inputShape[2] > 0 ? inputShape[2] : 1;
                    inW = inputShape[3] > 0 ? inputShape[3] : 1;
                }
                conv3d.ResolveShapesOnly(new[] { inC, inD, inH, inW });
            }
        }
        else if (genericDef == typeof(NeuralNetworks.Layers.DeconvolutionalLayer<>))
        {
            // DeconvolutionalLayer(int outputDepth, int kernelSize, int stride, int padding, IActivationFunction<T>?)
            // — lazy ctor; spatial dims (H/W) and inputDepth resolved on first Forward.
            // Same NCHW vs CHW disambiguation as ConvolutionalLayer above.
            int outputDepth = outputShape.Length switch
            {
                >= 4 => outputShape[1],
                3 => outputShape[0],
                _ => outputShape[0]
            };
            int kernelSize = TryGetInt(additionalParams, "KernelSize") ?? 3;
            int stride = TryGetInt(additionalParams, "Stride") ?? 1;
            int padding = TryGetInt(additionalParams, "Padding") ?? 0;

            var activationFuncType = typeof(IActivationFunction<>).MakeGenericType(typeof(T));
            var ctor = type.GetConstructor(new Type[] { typeof(int), typeof(int), typeof(int), typeof(int), activationFuncType });
            if (ctor is null)
                throw new MissingLayerCtorException("Cannot find DeconvolutionalLayer constructor with expected signature.");
            object? activation = TryCreateActivationInstance(additionalParams, "ScalarActivationType", activationFuncType);
            instance = ctor.Invoke(new object?[] { outputDepth, kernelSize, stride, padding, activation });

            // Pre-resolve from inputShape so SetParameters matches saved counts.
            if (instance is NeuralNetworks.Layers.DeconvolutionalLayer<T> deconv && inputShape != null && inputShape.Length >= 3)
            {
                int inDepth, inH, inW;
                if (inputShape.Length == 4)
                {
                    inDepth = inputShape[1] > 0 ? inputShape[1] : 1;
                    inH = inputShape[2] > 0 ? inputShape[2] : 1;
                    inW = inputShape[3] > 0 ? inputShape[3] : 1;
                }
                else
                {
                    inDepth = inputShape[0] > 0 ? inputShape[0] : 1;
                    inH = inputShape[1] > 0 ? inputShape[1] : 1;
                    inW = inputShape[2] > 0 ? inputShape[2] : 1;
                }
                deconv.ResolveShapesOnly(new[] { inDepth, inH, inW });
            }
        }
        else if (genericDef == typeof(NeuralNetworks.Layers.FullyConnectedLayer<>))
        {
            // FullyConnectedLayer(int outputSize, IActivationFunction<T>? activationFunction = null)
            // — lazy ctor; input feature size resolved from input.Shape[^1] on first Forward.
            int outputSize = outputShape.Length > 0 ? outputShape[^1] : 0;
            if (outputSize <= 0)
                throw new InvalidOperationException(
                    $"FullyConnectedLayer deserialization requires positive outputSize; got {outputSize}.");

            var activationFuncType = typeof(IActivationFunction<>).MakeGenericType(typeof(T));
            object? activation = TryCreateActivationInstance(additionalParams, "ScalarActivationType", activationFuncType);

            var ctor = type.GetConstructor(new Type[] { typeof(int), activationFuncType });
            if (ctor is null)
                throw new MissingLayerCtorException("Cannot find FullyConnectedLayer constructor with (int, IActivationFunction<T>).");
            instance = ctor.Invoke(new object?[] { outputSize, activation });
        }
        else if (genericDef == typeof(NeuralNetworks.Layers.Upsample3DLayer<>))
        {
            // Upsample3DLayer(int scaleDepth, int scaleHeight, int scaleWidth) — preferred lazy ctor.
            // Falls back to (int scaleFactor) when only a single scale was serialized.
            int? scaleD = TryGetInt(additionalParams, "ScaleDepth");
            int? scaleH = TryGetInt(additionalParams, "ScaleHeight");
            int? scaleW = TryGetInt(additionalParams, "ScaleWidth");
            int? scaleF = TryGetInt(additionalParams, "ScaleFactor");

            if (scaleD.HasValue && scaleH.HasValue && scaleW.HasValue)
            {
                var ctor3 = type.GetConstructor(new Type[] { typeof(int), typeof(int), typeof(int) });
                if (ctor3 is null)
                    throw new MissingLayerCtorException("Cannot find Upsample3DLayer constructor with (int, int, int).");
                instance = ctor3.Invoke(new object[] { scaleD.Value, scaleH.Value, scaleW.Value });
            }
            else
            {
                int sf = scaleF ?? 2;
                var ctor1 = type.GetConstructor(new Type[] { typeof(int) });
                if (ctor1 is null)
                    throw new MissingLayerCtorException("Cannot find Upsample3DLayer constructor with (int).");
                instance = ctor1.Invoke(new object[] { sf });
            }
        }
        else if (genericDef == typeof(NeuralNetworks.Layers.MeshPoolLayer<>))
        {
            // MeshPoolLayer(int inputChannels, int targetEdges, int numNeighbors)
            int inputChannels = inputShape[^1];
            int targetEdges = TryGetInt(additionalParams, "TargetEdges") ?? inputChannels;
            int numNeighbors = TryGetInt(additionalParams, "NumNeighbors") ?? 4;

            var ctor = type.GetConstructor(new Type[] { typeof(int), typeof(int), typeof(int) });
            if (ctor is null)
                throw new MissingLayerCtorException("Cannot find MeshPoolLayer constructor.");
            instance = ctor.Invoke(new object[] { inputChannels, targetEdges, numNeighbors });
        }
        else if (genericDef == typeof(NeuralNetworks.Layers.MeshEdgeConvLayer<>))
        {
            // MeshEdgeConvLayer(int inputChannels, int outputChannels, int numNeighbors, IActivationFunction?)
            int inputChannels = inputShape[^1];
            int outputChannels = outputShape[^1];
            int numNeighbors = TryGetInt(additionalParams, "NumNeighbors") ?? 4;
            var activation = TryRestoreActivation<T>(additionalParams);

            var activationFuncType = typeof(IActivationFunction<>).MakeGenericType(typeof(T));
            var ctor = type.GetConstructor(new Type[] { typeof(int), typeof(int), typeof(int), activationFuncType });
            if (ctor is null)
                throw new MissingLayerCtorException("Cannot find MeshEdgeConvLayer constructor.");
            instance = ctor.Invoke(new object?[] { inputChannels, outputChannels, numNeighbors, activation });
        }
        else if (genericDef == typeof(PrimaryCapsuleLayer<>))
        {
            // PrimaryCapsuleLayer(int inputChannels, int capsuleChannels, int capsuleDimension, int kernelSize, int stride, IActivationFunction<T>?)
            int inputChannels = inputShape.Length > 0 ? inputShape[0] : 1;
            int capsuleChannels = TryGetInt(additionalParams, "CapsuleChannels") ?? 32;
            int capsuleDimension = TryGetInt(additionalParams, "CapsuleDimension") ?? 8;
            int kernelSize = TryGetInt(additionalParams, "KernelSize") ?? 9;
            int stride = TryGetInt(additionalParams, "Stride") ?? 2;

            var activationFuncType = typeof(IActivationFunction<>).MakeGenericType(typeof(T));
            var ctor = type.GetConstructor(new Type[] { typeof(int), typeof(int), typeof(int), typeof(int), typeof(int), activationFuncType });
            if (ctor is null)
                throw new MissingLayerCtorException("Cannot find PrimaryCapsuleLayer constructor with expected signature.");
            instance = ctor.Invoke(new object?[] { inputChannels, capsuleChannels, capsuleDimension, kernelSize, stride, null });
        }
        else if (genericDef == typeof(DigitCapsuleLayer<>))
        {
            // DigitCapsuleLayer(int inputCapsules, int inputCapsuleDimension, int numClasses, int outputCapsuleDimension, int routingIterations)
            int inputCapsules = TryGetInt(additionalParams, "InputCapsules") ?? inputShape[0];
            int inputCapsuleDim = TryGetInt(additionalParams, "InputCapsuleDimension") ?? (inputShape.Length > 1 ? inputShape[1] : 8);
            int numClasses = TryGetInt(additionalParams, "NumClasses") ?? outputShape[0];
            int outputCapsuleDim = TryGetInt(additionalParams, "OutputCapsuleDimension") ?? (outputShape.Length > 1 ? outputShape[1] : 16);
            int routingIter = TryGetInt(additionalParams, "RoutingIterations") ?? 3;

            var ctor = type.GetConstructor(new Type[] { typeof(int), typeof(int), typeof(int), typeof(int), typeof(int) });
            if (ctor is null)
                throw new MissingLayerCtorException("Cannot find DigitCapsuleLayer constructor with expected signature.");
            instance = ctor.Invoke(new object[] { inputCapsules, inputCapsuleDim, numClasses, outputCapsuleDim, routingIter });
        }
        else if (genericDef == typeof(ReconstructionLayer<>))
        {
            // ReconstructionLayer(int inputDimension, int hidden1Dimension, int hidden2Dimension, int outputDimension, IActivationFunction<T>?, IActivationFunction<T>?)
            int inputDim = inputShape[0];
            int outputDim = outputShape[0];
            int hidden1 = TryGetInt(additionalParams, "Hidden1Dimension") ?? Math.Max(inputDim / 2, 64);
            int hidden2 = TryGetInt(additionalParams, "Hidden2Dimension") ?? Math.Max(inputDim / 4, 32);

            var activationFuncType = typeof(IActivationFunction<>).MakeGenericType(typeof(T));
            var ctor = type.GetConstructor(new Type[] { typeof(int), typeof(int), typeof(int), typeof(int), activationFuncType, activationFuncType });
            if (ctor is null)
                throw new MissingLayerCtorException("Cannot find ReconstructionLayer constructor.");
            instance = ctor.Invoke(new object?[] { inputDim, hidden1, hidden2, outputDim, null, null });
        }
        else if (genericDef == typeof(MaxPool3DLayer<>))
        {
            // MaxPool3DLayer is now lazy: ctor signature is (int poolSize, int stride = 0).
            // Spatial dims (D/H/W) and channel count resolve on first Forward via OnFirstForward.
            // The fallback ResolveFromShape below the per-branch dispatch picks up the
            // serialized inputShape so the layer's lazy state is materialised before
            // SetParameters runs.
            int poolSize = TryGetInt(additionalParams, "PoolSize") ?? 2;
            int stride = TryGetInt(additionalParams, "Stride") ?? 0;

            var ctor = type.GetConstructor(new Type[] { typeof(int), typeof(int) });
            if (ctor is null)
                throw new MissingLayerCtorException("Cannot find MaxPool3DLayer constructor with (int, int).");
            instance = ctor.Invoke(new object[] { poolSize, stride });
        }
        else if (genericDef == typeof(PoolingLayer<>))
        {
            // PoolingLayer(int inputDepth, int inputHeight, int inputWidth, int poolSize, int stride, PoolingType type)
            int poolSize = TryGetInt(additionalParams, "PoolSize") ?? 2;
            int stride = TryGetInt(additionalParams, "Stride") ?? 2;
            PoolingType poolingType = TryGetEnum<PoolingType>(additionalParams, "PoolingType") ?? PoolingType.Max;
            // inputShape format: [batch, depth, height, width] (NCHW format)
            int inputDepth = inputShape.Length > 1 ? inputShape[1] : inputShape[0];
            int inputHeight = inputShape.Length > 2 ? inputShape[2] : 1;
            int inputWidth = inputShape.Length > 3 ? inputShape[3] : 1;

            var ctor = type.GetConstructor(new Type[] { typeof(int), typeof(int), typeof(int), typeof(int), typeof(int), typeof(PoolingType) });
            if (ctor is null)
            {
                throw new MissingLayerCtorException($"Cannot find PoolingLayer constructor.");
            }
            instance = ctor.Invoke(new object[] { inputDepth, inputHeight, inputWidth, poolSize, stride, poolingType });
        }
        else if (genericDef == typeof(AiDotNet.NeuralNetworks.Layers.UpsamplingLayer<>) ||
                 (openGenericType.FullName != null && openGenericType.FullName.EndsWith(".NeuralNetworks.Layers.UpsamplingLayer`1")))
        {
            // UpsamplingLayer(int scaleFactor) — lazy: input shape resolved on first forward.
            // Keep the ScaleFactor > 0 validation guard; route through the new 1-arg
            // lazy ctor (the legacy 2-arg eager ctor was removed in the lazy-shape-
            // inference migration).
            int scaleFactor = TryGetInt(additionalParams, "ScaleFactor") ?? 2;
            if (scaleFactor <= 0)
            {
                throw new InvalidOperationException(
                    $"Invalid UpsamplingLayer ScaleFactor metadata: {scaleFactor}. ScaleFactor must be positive.");
            }

            var ctor = type.GetConstructor(new Type[] { typeof(int) });
            if (ctor is null)
            {
                throw new MissingLayerCtorException("Cannot find UpsamplingLayer constructor.");
            }
            instance = ctor.Invoke(new object[] { scaleFactor });
        }
        else if (genericDef == typeof(AiDotNet.NeuralNetworks.Layers.MaxPoolingLayer<>) ||
                 (openGenericType.FullName != null && openGenericType.FullName.EndsWith(".NeuralNetworks.Layers.MaxPoolingLayer`1")))
        {
            // MaxPoolingLayer(int poolSize, int strides) — lazy ctor; spatial dims resolved on first Forward.
            int poolSize = TryGetInt(additionalParams, "PoolSize") ?? 2;
            int strides = TryGetInt(additionalParams, "Strides") ?? 2;

            var ctor = type.GetConstructor(new Type[] { typeof(int), typeof(int) });
            if (ctor is null)
            {
                throw new MissingLayerCtorException($"Cannot find MaxPoolingLayer constructor.");
            }
            instance = ctor.Invoke(new object[] { poolSize, strides });
        }
        else if (genericDef == typeof(AiDotNet.NeuralNetworks.Layers.DenseBlock<>) ||
                 (openGenericType.FullName != null && openGenericType.FullName.EndsWith(".NeuralNetworks.Layers.DenseBlock`1")))
        {
            // DenseBlock(int inputChannels, int numLayers, int growthRate, int inputHeight, int inputWidth, double bnMomentum = 0.1)
            int inputChannels = TryGetInt(additionalParams, "InputChannels") ?? (inputShape.Length > 1 ? inputShape[1] : inputShape[0]);
            int numLayers = TryGetInt(additionalParams, "NumLayers") ?? 4;
            int growthRate = TryGetInt(additionalParams, "GrowthRate") ?? 32;
            int inputHeight = inputShape.Length > 2 ? inputShape[2] : 1;
            int inputWidth = inputShape.Length > 3 ? inputShape[3] : 1;
            double bnMomentum = TryGetDouble(additionalParams, "BnMomentum") ?? 0.1;

            var ctor = type.GetConstructor(new Type[] { typeof(int), typeof(int), typeof(int), typeof(int), typeof(int), typeof(double) });
            if (ctor is null)
            {
                throw new MissingLayerCtorException($"Cannot find DenseBlock constructor.");
            }
            instance = ctor.Invoke(new object[] { inputChannels, numLayers, growthRate, inputHeight, inputWidth, bnMomentum });
        }
        else if (genericDef == typeof(AiDotNet.NeuralNetworks.Layers.InvertedResidualBlock<>) ||
                 (openGenericType.FullName != null && openGenericType.FullName.EndsWith(".NeuralNetworks.Layers.InvertedResidualBlock`1")))
        {
            // InvertedResidualBlock(int inChannels, int outChannels, int inputHeight, int inputWidth, int expansionRatio, int stride, bool useSE, int seRatio, IActivationFunction<T>?)
            int inChannels = TryGetInt(additionalParams, "InChannels") ?? (inputShape.Length > 1 ? inputShape[1] : inputShape[0]);
            int outChannels = TryGetInt(additionalParams, "OutChannels") ?? (outputShape.Length > 1 ? outputShape[1] : outputShape[0]);
            int inputHeight = inputShape.Length > 2 ? inputShape[2] : 1;
            int inputWidth = inputShape.Length > 3 ? inputShape[3] : 1;
            int expansionRatio = TryGetInt(additionalParams, "ExpansionRatio") ?? 6;
            int stride = TryGetInt(additionalParams, "Stride") ?? 1;
            bool useSE = TryGetBool(additionalParams, "UseSE") ?? false;
            int seRatio = TryGetInt(additionalParams, "SERatio") ?? 4;

            var activationFuncType = typeof(IActivationFunction<>).MakeGenericType(typeof(T));
            var ctor = type.GetConstructor(new Type[] { typeof(int), typeof(int), typeof(int), typeof(int), typeof(int), typeof(int), typeof(bool), typeof(int), activationFuncType });
            if (ctor is null)
            {
                throw new MissingLayerCtorException($"Cannot find InvertedResidualBlock constructor.");
            }
            object? activation = TryCreateActivationInstance(additionalParams, "ScalarActivationType", activationFuncType);
            instance = ctor.Invoke(new object?[] { inChannels, outChannels, inputHeight, inputWidth, expansionRatio, stride, useSE, seRatio, activation });
        }
        else if (genericDef == typeof(AiDotNet.NeuralNetworks.Layers.BottleneckBlock<>) ||
                 (openGenericType.FullName != null && openGenericType.FullName.EndsWith(".NeuralNetworks.Layers.BottleneckBlock`1")))
        {
            // BottleneckBlock(int inChannels, int baseChannels, int stride, int inputHeight, int inputWidth, bool zeroInitResidual)
            int inChannels = TryGetInt(additionalParams, "InChannels") ?? (inputShape.Length > 0 ? inputShape[0] : 64);
            int outChannels = TryGetInt(additionalParams, "OutChannels")
                              ?? (outputShape.Length > 0 ? outputShape[0] : inChannels);
            // Prefer the explicit BaseChannels from additionalParams when present; fall
            // back to the standard expansion=4 derivation otherwise. Blindly dividing by
            // 4 silently produces wrong constructor arguments when outputShape is empty
            // or when outChannels isn't divisible by 4.
            const int BottleneckExpansion = 4;
            int baseChannels = TryGetInt(additionalParams, "BaseChannels")
                               ?? (outChannels / BottleneckExpansion);
            if (baseChannels <= 0)
                throw new InvalidOperationException(
                    $"BottleneckBlock baseChannels must be positive (derived {baseChannels} " +
                    $"from outChannels={outChannels}); provide 'BaseChannels' in additionalParams.");
            if (baseChannels * BottleneckExpansion != outChannels)
                throw new InvalidOperationException(
                    $"BottleneckBlock expansion mismatch: baseChannels={baseChannels} * 4 " +
                    $"!= outChannels={outChannels}. Populate 'BaseChannels' in additionalParams.");
            int stride = TryGetInt(additionalParams, "Stride") ?? 1;
            int inputHeight = inputShape.Length > 1 ? inputShape[1] : 56;
            int inputWidth = inputShape.Length > 2 ? inputShape[2] : 56;
            bool zeroInitResidual = TryGetBool(additionalParams, "ZeroInitResidual") ?? true;

            var ctor = type.GetConstructor(new Type[] { typeof(int), typeof(int), typeof(int), typeof(int), typeof(int), typeof(bool) });
            if (ctor is null)
                throw new MissingLayerCtorException("Cannot find BottleneckBlock constructor.");
            instance = ctor.Invoke(new object[] { inChannels, baseChannels, stride, inputHeight, inputWidth, zeroInitResidual });
        }
        else if (genericDef == typeof(AiDotNet.NeuralNetworks.Layers.BasicBlock<>) ||
                 (openGenericType.FullName != null && openGenericType.FullName.EndsWith(".NeuralNetworks.Layers.BasicBlock`1")))
        {
            // BasicBlock(int inChannels, int outChannels, int stride, int inputHeight, int inputWidth, bool zeroInitResidual)
            int inChannels = TryGetInt(additionalParams, "InChannels") ?? (inputShape.Length > 0 ? inputShape[0] : 64);
            int outChannels = TryGetInt(additionalParams, "OutChannels") ?? (outputShape.Length > 0 ? outputShape[0] : inChannels);
            int stride = TryGetInt(additionalParams, "Stride") ?? 1;
            int inputHeight = inputShape.Length > 1 ? inputShape[1] : 56;
            int inputWidth = inputShape.Length > 2 ? inputShape[2] : 56;
            bool zeroInitResidual = TryGetBool(additionalParams, "ZeroInitResidual") ?? true;

            var ctor = type.GetConstructor(new Type[] { typeof(int), typeof(int), typeof(int), typeof(int), typeof(int), typeof(bool) });
            if (ctor is null)
                throw new MissingLayerCtorException("Cannot find BasicBlock constructor.");
            instance = ctor.Invoke(new object[] { inChannels, outChannels, stride, inputHeight, inputWidth, zeroInitResidual });
        }
        else if (genericDef == typeof(AiDotNet.NeuralNetworks.Layers.TransitionLayer<>) ||
                 (openGenericType.FullName != null && openGenericType.FullName.EndsWith(".NeuralNetworks.Layers.TransitionLayer`1")))
        {
            // TransitionLayer(int inputChannels, int inputHeight, int inputWidth, double compressionFactor = 0.5)
            int inputChannels = TryGetInt(additionalParams, "InputChannels") ?? (inputShape.Length > 1 ? inputShape[1] : inputShape[0]);
            int inputHeight = inputShape.Length > 2 ? inputShape[2] : 1;
            int inputWidth = inputShape.Length > 3 ? inputShape[3] : 1;
            // Calculate compression factor from input and output channels
            int outputChannels = TryGetInt(additionalParams, "OutputChannels") ?? (outputShape.Length > 1 ? outputShape[1] : outputShape[0]);
            double compressionFactor = inputChannels > 0 ? (double)outputChannels / inputChannels : 0.5;

            var ctor = type.GetConstructor(new Type[] { typeof(int), typeof(int), typeof(int), typeof(double) });
            if (ctor is null)
            {
                throw new MissingLayerCtorException($"Cannot find TransitionLayer constructor.");
            }
            instance = ctor.Invoke(new object[] { inputChannels, inputHeight, inputWidth, compressionFactor });
        }
        else if (genericDef == typeof(AiDotNet.NeuralNetworks.Layers.AdaptiveAveragePoolingLayer<>) ||
                 (openGenericType.FullName != null && openGenericType.FullName.EndsWith(".NeuralNetworks.Layers.AdaptiveAveragePoolingLayer`1")))
        {
            // AdaptiveAveragePoolingLayer(int outputHeight = 1, int outputWidth = 1) — lazy ctor;
            // input C/H/W resolve on first Forward.
            int outputHeight = outputShape.Length > 1 ? Math.Max(1, outputShape[outputShape.Length - 2]) : 1;
            int outputWidth = outputShape.Length > 0 ? Math.Max(1, outputShape[outputShape.Length - 1]) : 1;

            var ctor = type.GetConstructor(new Type[] { typeof(int), typeof(int) });
            if (ctor is null)
            {
                throw new MissingLayerCtorException($"Cannot find AdaptiveAveragePoolingLayer constructor.");
            }
            instance = ctor.Invoke(new object[] { outputHeight, outputWidth });
        }
        else if (genericDef == typeof(ActivationLayer<>))
        {
            instance = CreateActivationLayer<T>(type, inputShape, additionalParams);
        }
        else if (genericDef == typeof(GRULayer<>))
        {
            instance = CreateGRULayer<T>(type, inputShape, outputShape, additionalParams);
        }
        else if (genericDef == typeof(LSTMLayer<>))
        {
            instance = CreateLSTMLayer<T>(type, inputShape, outputShape, additionalParams);
        }
        else if (genericDef == typeof(MixtureOfExpertsLayer<>))
        {
            // Recreate MoE with default expert count and router
            int inputSize = inputShape[0];
            int outputSize = outputShape[0];
            int numExperts = TryGetInt(additionalParams, "NumExperts") ?? 4;
            int topK = TryGetInt(additionalParams, "TopK") ?? 0;

            var experts = new List<ILayer<T>>();
            for (int e = 0; e < numExperts; e++)
                experts.Add(new DenseLayer<T>(outputSize, new IdentityActivation<T>() as IActivationFunction<T>));

            var router = new DenseLayer<T>(numExperts, new SoftmaxActivation<T>() as IActivationFunction<T>);
            instance = new MixtureOfExpertsLayer<T>(experts, router, inputShape, outputShape, topK);
        }
        else if (genericDef == typeof(ReservoirLayer<>))
        {
            int inputSize = inputShape[0];
            int reservoirSize = outputShape[0];
            double connProb = TryGetDouble(additionalParams, "ConnectionProbability") ?? 0.1;
            double specRadius = TryGetDouble(additionalParams, "SpectralRadius") ?? 0.9;
            double inpScaling = TryGetDouble(additionalParams, "InputScaling") ?? 1.0;
            double leakRate = TryGetDouble(additionalParams, "LeakingRate") ?? 1.0;
            instance = new ReservoirLayer<T>(inputSize, reservoirSize, connProb, specRadius, inpScaling, leakRate);
        }
        else if (genericDef == typeof(RBFLayer<>))
        {
            int inputSize = inputShape[0];
            int numCenters = TryGetInt(additionalParams, "NumCenters") ?? outputShape[0];
            instance = new RBFLayer<T>(inputSize, numCenters, new GaussianRBF<T>());
        }
        else if (genericDef == typeof(RecurrentLayer<>))
        {
            int inputSize = inputShape.Length > 0 ? inputShape[^1] : 128;
            int hiddenSize = outputShape.Length > 0 ? outputShape[^1] : 64;
            instance = new RecurrentLayer<T>( hiddenSize, (IActivationFunction<T>?)null);
        }
        else if (genericDef == typeof(SparseLinearLayer<>))
        {
            // SparseLinearLayer(int, int, double sparsity, IActivationFunction<T>?,
            //                   IInitializationStrategy<T>?). Restore sparsity +
            // activation from the saved metadata so a Clone of a layer with a
            // non-default activation (e.g., the IdentityActivation that
            // SparseNeuralNetwork uses on its regression output layer) doesn't
            // silently fall back to the default ReLU and clamp the output to
            // zero — clone_ShouldProduceIdenticalOutput would otherwise see
            // original≠0 vs cloned=0 for any negative pre-activation.
            int inputSize = inputShape[0];
            int outputSize = outputShape[0];
            var activationFuncType = typeof(IActivationFunction<>).MakeGenericType(typeof(T));
            object? activation = TryCreateActivationInstance(additionalParams, "ScalarActivationType", activationFuncType);
            // If metadata named an activation but TryCreateActivationInstance
            // couldn't materialise it (missing type, assembly-load failure,
            // ctor mismatch), the layer would silently fall back to the
            // ctor default — reintroducing the clone/output drift this
            // metadata exists to prevent. Surface the failure instead.
            if (activation is null
                && additionalParams is not null
                && additionalParams.ContainsKey("ScalarActivationType"))
            {
                throw new InvalidOperationException(
                    $"Failed to deserialize activation function of type " +
                    $"'{additionalParams["ScalarActivationType"]}' for SparseLinearLayer. " +
                    $"Ensure the assembly providing this activation is loaded before deserialization.");
            }
            double sparsity = TryGetDouble(additionalParams, "Sparsity") ?? 0.9;
            instance = new SparseLinearLayer<T>(inputSize, outputSize, sparsity,
                (IActivationFunction<T>?)activation);
        }
        else if (genericDef == typeof(OctonionLinearLayer<>))
        {
            // OctonionLinearLayer stores shapes as features*8 in base class
            // Divide by 8 to get actual octonion feature counts
            int inputFeatures = inputShape[0] / 8;
            int outputFeatures = outputShape[0] / 8;
            instance = new OctonionLinearLayer<T>(inputFeatures, outputFeatures);
        }
        else if (genericDef == typeof(HyperbolicLinearLayer<>))
        {
            int inputSize = inputShape[0];
            int outputSize = outputShape[0];
            double curvature = TryGetDouble(additionalParams, "Curvature") ?? -1.0;
            instance = new HyperbolicLinearLayer<T>(inputSize, outputSize, curvature);
        }
        else if (genericDef == typeof(SequenceLastLayer<>))
        {
            int featureSize = inputShape.Length > 0 ? inputShape[^1] : outputShape[0];
            instance = new SequenceLastLayer<T>(featureSize);
        }
        else if (genericDef.Name == "ConcatenateLayer`1" || genericDef.Name == "AddLayer`1" || genericDef.Name == "MultiplyLayer`1")
        {
            // Ctors: (int[][] inputShapes, [int axis,] IActivationFunction).
            // Pass two identical inputShape entries so binary concat / add /
            // multiply have a sane two-operand setup. Pick the constructor
            // by SHAPE — first ctor whose parameters are all int[][] / int /
            // activation / defaulted. Falling back to "highest-arity-with-
            // null-for-everything-else" was unsafe: a future ctor overload
            // taking a non-defaulted reference parameter (e.g. a custom
            // schedule object) would receive null and either NRE inside
            // the ctor or pass validation only to crash on first Forward.
            var defaultAxis = inputShape.Length > 0 ? inputShape.Length - 1 : 0;
            var activationFuncType = typeof(IActivationFunction<>).MakeGenericType(typeof(T));
            var ctorB = type.GetConstructors()
                .Where(c =>
                {
                    var ps = c.GetParameters();
                    foreach (var p in ps)
                    {
                        if (p.ParameterType == typeof(int[][])) continue;
                        if (p.ParameterType == typeof(int)) continue;
                        if (p.ParameterType == activationFuncType) continue;
                        if (p.HasDefaultValue) continue;
                        // Non-defaulted parameter we can't resolve from
                        // (inputShape, additionalParams, activation): reject
                        // this overload.
                        return false;
                    }
                    return ps.Length >= 1; // at minimum needs the inputShapes arg
                })
                .OrderByDescending(c => c.GetParameters().Length)
                .FirstOrDefault();
            if (ctorB is null)
            {
                // No safely-fillable ctor: fall through to the matcher
                // which has its own activation-restoration + defaulting
                // path and surfaces a clear error if it can't fill any
                // ctor either.
                instance = TryConstructByMatchingMetadata<T>(type, inputShape, outputShape, additionalParams, layerType);
                if (instance is null)
                {
                    throw new MissingLayerCtorException(
                        $"Cannot find a {layerType} constructor whose non-defaulted parameters " +
                        $"are all resolvable from (inputShape, additionalParams, activation). " +
                        $"Public ctors: [{string.Join(" | ", type.GetConstructors().Select(c => "(" + string.Join(", ", c.GetParameters().Select(p => p.ParameterType.Name + " " + p.Name)) + ")"))}].");
                }
            }
            else
            {
                var psB = ctorB.GetParameters();
                var argsB = new object?[psB.Length];
                for (int i = 0; i < psB.Length; i++)
                {
                    var p = psB[i];
                    if (p.ParameterType == typeof(int[][])) argsB[i] = new int[][] { inputShape, inputShape };
                    else if (p.ParameterType == typeof(int)) argsB[i] = TryGetInt(additionalParams, "Axis") ?? defaultAxis;
                    else if (p.HasDefaultValue) argsB[i] = p.DefaultValue;
                    // Activation: TryRestoreActivation if metadata holds it,
                    // else null is safe — this ctor signature accepts null
                    // for activation (the layer's IdentityActivation default
                    // path).
                    else if (p.ParameterType == activationFuncType)
                        argsB[i] = TryCreateActivationInstance(additionalParams, "ScalarActivationType", activationFuncType);
                    else argsB[i] = null;
                }
                instance = ctorB.Invoke(argsB);
            }
        }
        else if (genericDef.Name == "ConvLSTMLayer`1")
        {
            // (int[] inputShape, int kernelSize, int filters, int padding,
            //  int strides, IActivationFunction).
            // ConvLSTM expects inputShape rank 4: [time, channels, H, W].
            // ConvLSTMLayer.GetMetadata persists KernelSize / Filters /
            // Padding / Strides — fail fast if any are missing or if the
            // input shape is degenerate, rather than fabricating values
            // that produce a structurally-wrong reconstruction (issue #1239).
            if (inputShape.Length < 4)
            {
                throw new InvalidOperationException(
                    $"ConvLSTMLayer requires serialized inputShape with rank 4 " +
                    $"[time, channels, height, width]; got rank {inputShape.Length} " +
                    $"(shape [{string.Join(",", inputShape)}]). Re-serialize with the " +
                    $"current LayerBase shape persistence to recover.");
            }
            int kernelSize = TryGetInt(additionalParams, "KernelSize")
                ?? throw new InvalidOperationException(
                    "ConvLSTMLayer requires 'KernelSize' metadata (added in #1239). " +
                    "Re-serialize the network with the current GetMetadata implementation.");
            int filters = TryGetInt(additionalParams, "Filters")
                ?? throw new InvalidOperationException(
                    "ConvLSTMLayer requires 'Filters' metadata (added in #1239).");
            int padding = TryGetInt(additionalParams, "Padding")
                ?? throw new InvalidOperationException(
                    "ConvLSTMLayer requires 'Padding' metadata (added in #1239).");
            int strides = TryGetInt(additionalParams, "Strides")
                ?? throw new InvalidOperationException(
                    "ConvLSTMLayer requires 'Strides' metadata (added in #1239).");

            var ctorC = type.GetConstructors().OrderByDescending(c => c.GetParameters().Length).FirstOrDefault() ?? throw new MissingLayerCtorException($"Cannot find any public constructor for {layerType} during deserialization.");
            var psC = ctorC.GetParameters();
            var argsC = new object?[psC.Length];
            for (int i = 0; i < psC.Length; i++)
            {
                var p = psC[i];
                var n = (p.Name ?? "").ToLowerInvariant();
                if (p.ParameterType == typeof(int[])) argsC[i] = inputShape;
                else if (p.ParameterType == typeof(int))
                {
                    argsC[i] = n switch
                    {
                        "kernelsize" => kernelSize,
                        "filters" => filters,
                        "padding" => padding,
                        "strides" => strides,
                        _ => p.HasDefaultValue ? (int)p.DefaultValue! : throw new InvalidOperationException(
                            $"ConvLSTMLayer ctor parameter '{n}' has no metadata mapping " +
                            $"and no default value."),
                    };
                }
                else if (p.HasDefaultValue) argsC[i] = p.DefaultValue;
                else argsC[i] = null;
            }
            instance = ctorC.Invoke(argsC);
        }
        else if (genericDef.Name == "TransformerEncoderBlock`1")
        {
            // Pre-LN transformer encoder block (ctor: hiddenSize, numHeads, ffnDim, dropoutRate).
            // TransformerEncoderBlock.GetMetadata persists all four — fail fast if the
            // dimension metadata is missing rather than fabricating defaults that may
            // violate hiddenSize % numHeads == 0.
            int hsTeb = TryGetInt(additionalParams, "HiddenSize")
                ?? (inputShape.Length > 0 ? inputShape[inputShape.Length - 1] : throw new InvalidOperationException(
                    $"{genericDef.Name} requires 'HiddenSize' metadata or a rank>=1 inputShape."));
            int nhTeb = TryGetInt(additionalParams, "NumHeads")
                ?? throw new InvalidOperationException($"{genericDef.Name} requires 'NumHeads' metadata.");
            int ffTeb = TryGetInt(additionalParams, "FfnDim")
                ?? throw new InvalidOperationException($"{genericDef.Name} requires 'FfnDim' metadata.");
            double drTeb = TryGetDouble(additionalParams, "DropoutRate") ?? 0.0;
            // Validate positivity BEFORE the modulo: a corrupt numHeads of 0 would make
            // (hsTeb % nhTeb) throw DivideByZeroException, and a negative value would pass the
            // modulo (C# % takes the dividend's sign) yet yield a negative per-head dimension.
            if (hsTeb <= 0 || nhTeb <= 0 || ffTeb <= 0)
                throw new InvalidOperationException(
                    $"{genericDef.Name} metadata is corrupt: hiddenSize ({hsTeb}), numHeads ({nhTeb}), " +
                    $"and ffnDim ({ffTeb}) must all be positive.");
            if (hsTeb % nhTeb != 0)
            {
                throw new InvalidOperationException(
                    $"{genericDef.Name} divisibility violation: hiddenSize ({hsTeb}) must be a " +
                    $"multiple of numHeads ({nhTeb}). Serialized metadata is corrupt.");
            }
            var ctorTeb = type.GetConstructors().OrderByDescending(c => c.GetParameters().Length).FirstOrDefault()
                ?? throw new MissingLayerCtorException($"Cannot find any public constructor for {layerType} during deserialization.");
            var psTeb = ctorTeb.GetParameters();
            var argsTeb = new object?[psTeb.Length];
            for (int i = 0; i < psTeb.Length; i++)
            {
                var p = psTeb[i];
                var n = (p.Name ?? "").ToLowerInvariant();
                argsTeb[i] = (p.ParameterType, n) switch
                {
                    (Type t, _) when t == typeof(int) && n == "hiddensize" => hsTeb,
                    (Type t, _) when t == typeof(int) && n == "numheads" => nhTeb,
                    (Type t, _) when t == typeof(int) && n == "ffndim" => ffTeb,
                    (Type t, _) when t == typeof(double) && n == "dropoutrate" => drTeb,
                    _ => p.HasDefaultValue ? p.DefaultValue : null,
                };
            }
            instance = ctorTeb.Invoke(argsTeb);
        }
        else if (genericDef.Name == "TransformerDecoderBlock`1")
        {
            // Pre-LN decoder block (ctor: hiddenSize, numHeads, ffnDim, dropoutRate).
            int hsTdb = TryGetInt(additionalParams, "HiddenSize")
                ?? (inputShape.Length > 0 ? inputShape[inputShape.Length - 1] : throw new InvalidOperationException(
                    $"{genericDef.Name} requires 'HiddenSize' metadata or a rank>=1 inputShape."));
            int nhTdb = TryGetInt(additionalParams, "NumHeads")
                ?? throw new InvalidOperationException($"{genericDef.Name} requires 'NumHeads' metadata.");
            int ffTdb = TryGetInt(additionalParams, "FfnDim")
                ?? throw new InvalidOperationException($"{genericDef.Name} requires 'FfnDim' metadata.");
            double drTdb = TryGetDouble(additionalParams, "DropoutRate") ?? 0.0;
            // Validate positivity before the modulo (numHeads==0 would throw DivideByZeroException;
            // a negative value would slip past % and yield a negative per-head dimension).
            if (hsTdb <= 0 || nhTdb <= 0 || ffTdb <= 0)
                throw new InvalidOperationException(
                    $"{genericDef.Name} metadata is corrupt: hiddenSize ({hsTdb}), numHeads ({nhTdb}), " +
                    $"and ffnDim ({ffTdb}) must all be positive.");
            if (hsTdb % nhTdb != 0)
            {
                throw new InvalidOperationException(
                    $"{genericDef.Name} divisibility violation: hiddenSize ({hsTdb}) must be a " +
                    $"multiple of numHeads ({nhTdb}). Serialized metadata is corrupt.");
            }
            var ctorTdb = type.GetConstructors().OrderByDescending(c => c.GetParameters().Length).FirstOrDefault()
                ?? throw new MissingLayerCtorException($"Cannot find any public constructor for {layerType} during deserialization.");
            var psTdb = ctorTdb.GetParameters();
            var argsTdb = new object?[psTdb.Length];
            for (int i = 0; i < psTdb.Length; i++)
            {
                var p = psTdb[i];
                var n = (p.Name ?? "").ToLowerInvariant();
                argsTdb[i] = (p.ParameterType, n) switch
                {
                    (Type t, _) when t == typeof(int) && n == "hiddensize" => hsTdb,
                    (Type t, _) when t == typeof(int) && n == "numheads" => nhTdb,
                    (Type t, _) when t == typeof(int) && n == "ffndim" => ffTdb,
                    (Type t, _) when t == typeof(double) && n == "dropoutrate" => drTdb,
                    _ => p.HasDefaultValue ? p.DefaultValue : null,
                };
            }
            instance = ctorTdb.Invoke(argsTdb);
        }
        else if (genericDef.Name == "GroupedQueryAttentionLayer`1" || genericDef.Name == "CachedGroupedQueryAttention`1")
        {
            // (int sequenceLength, int embeddingDimension, int numHeads, int numKVHeads, ...).
            // numHeads must be a multiple of numKVHeads, embeddingDimension % numHeads == 0.
            // GroupedQueryAttentionLayer.GetMetadata persists all four
            // dimensions — fail fast if any are missing rather than
            // fabricating defaults that may not satisfy the layer's
            // divisibility constraints (issue #1239).
            int seqLen = TryGetInt(additionalParams, "SequenceLength")
                ?? (inputShape.Length > 0 ? inputShape[0] : throw new InvalidOperationException(
                    $"{genericDef.Name} requires 'SequenceLength' metadata or a rank>=1 inputShape."));
            int embDim = TryGetInt(additionalParams, "EmbeddingDimension")
                ?? (inputShape.Length > 1 ? inputShape[1] : throw new InvalidOperationException(
                    $"{genericDef.Name} requires 'EmbeddingDimension' metadata or a rank>=2 inputShape."));
            int numHeads = TryGetInt(additionalParams, "NumHeads")
                ?? throw new InvalidOperationException(
                    $"{genericDef.Name} requires 'NumHeads' metadata (added in #1239).");
            int numKVHeads = TryGetInt(additionalParams, "NumKVHeads")
                ?? throw new InvalidOperationException(
                    $"{genericDef.Name} requires 'NumKVHeads' metadata (added in #1239).");
            // Enforce divisibility constraints — if violated, the metadata
            // is corrupt rather than something we should silently adjust.
            if (numHeads % numKVHeads != 0)
            {
                throw new InvalidOperationException(
                    $"{genericDef.Name} divisibility violation: numHeads ({numHeads}) must be " +
                    $"a multiple of numKVHeads ({numKVHeads}). Serialized metadata is corrupt.");
            }
            if (embDim % numHeads != 0)
            {
                throw new InvalidOperationException(
                    $"{genericDef.Name} divisibility violation: embeddingDimension ({embDim}) " +
                    $"must be a multiple of numHeads ({numHeads}). Serialized metadata is corrupt.");
            }
            var ctorG = type.GetConstructors().OrderByDescending(c => c.GetParameters().Length).FirstOrDefault() ?? throw new MissingLayerCtorException($"Cannot find any public constructor for {layerType} during deserialization.");
            var psG = ctorG.GetParameters();
            var argsG = new object?[psG.Length];
            for (int i = 0; i < psG.Length; i++)
            {
                var p = psG[i];
                var n = (p.Name ?? "").ToLowerInvariant();
                argsG[i] = (p.ParameterType, n) switch
                {
                    (Type t, _) when t == typeof(int) && n == "sequencelength" => seqLen,
                    (Type t, _) when t == typeof(int) && n == "embeddingdimension" => embDim,
                    (Type t, _) when t == typeof(int) && n == "numheads" => numHeads,
                    (Type t, _) when t == typeof(int) && n == "numkvheads" => numKVHeads,
                    (Type t, _) when t == typeof(int) && n.Contains("layerindex") => 0,
                    (Type t, _) when t == typeof(int) => 1,
                    (Type t, _) when t == typeof(bool) => p.HasDefaultValue ? p.DefaultValue : false,
                    _ => p.HasDefaultValue ? p.DefaultValue : null,
                };
            }
            instance = ctorG.Invoke(argsG);
        }
        else if (genericDef.Name == "MesaNetLayer`1")
        {
            // (sequenceLength, modelDimension, numHeads, regularization, IActivation, IInitStrategy)
            // MesaNetLayer.GetMetadata persists SequenceLength / ModelDimension
            // / NumHeads / Regularization — fail fast if any are missing
            // rather than fabricating defaults that may violate the layer's
            // divisibility / positivity constraints (issue #1239).
            int seqLen = TryGetInt(additionalParams, "SequenceLength")
                ?? (inputShape.Length > 0 ? inputShape[0] : throw new InvalidOperationException(
                    "MesaNetLayer requires 'SequenceLength' metadata or a rank>=1 inputShape."));
            int modelDim = TryGetInt(additionalParams, "ModelDimension")
                ?? throw new InvalidOperationException(
                    "MesaNetLayer requires 'ModelDimension' metadata (added in #1239).");
            int numHeads = TryGetInt(additionalParams, "NumHeads")
                ?? throw new InvalidOperationException(
                    "MesaNetLayer requires 'NumHeads' metadata (added in #1239).");
            if (modelDim % numHeads != 0)
            {
                throw new InvalidOperationException(
                    $"MesaNetLayer divisibility violation: modelDimension ({modelDim}) must be " +
                    $"a multiple of numHeads ({numHeads}). Serialized metadata is corrupt.");
            }
            // MesaNet's ctor validates Regularization > 0; persist it so we
            // don't fabricate the default 1e-3 when the original was different.
            double reg = TryGetDouble(additionalParams, "Regularization")
                ?? throw new InvalidOperationException(
                    "MesaNetLayer requires 'Regularization' metadata (added in #1239).");
            var ctorM = type.GetConstructors().OrderByDescending(c => c.GetParameters().Length).FirstOrDefault() ?? throw new MissingLayerCtorException($"Cannot find any public constructor for {layerType} during deserialization.");
            var psM = ctorM.GetParameters();
            var argsM = new object?[psM.Length];
            for (int i = 0; i < psM.Length; i++)
            {
                var p = psM[i];
                var n = (p.Name ?? "").ToLowerInvariant();
                argsM[i] = (p.ParameterType, n) switch
                {
                    (Type t, _) when t == typeof(int) && n == "sequencelength" => seqLen,
                    (Type t, _) when t == typeof(int) && n == "modeldimension" => modelDim,
                    (Type t, _) when t == typeof(int) && n == "numheads" => numHeads,
                    (Type t, _) when t == typeof(double) => reg,
                    _ => p.HasDefaultValue ? p.DefaultValue : null,
                };
            }
            instance = ctorM.Invoke(argsM);
        }
        else if (genericDef.Name == "NHiTSStackTensor`1")
        {
            // (inputLength, outputLength, hiddenSize, numLayers, numBlocks,
            //  poolingSize, seed)
            // NHiTSStackTensor.GetMetadata persists InputLength / OutputLength
            // / HiddenSize / NumLayers / PoolingSize. numBlocks is vestigial
            // in this implementation and seed advances _random's state at
            // construction time — neither is round-trippable, so they fall
            // back to safe defaults (1, 0). Issue #1239.
            int inputLength = TryGetInt(additionalParams, "InputLength")
                ?? throw new InvalidOperationException(
                    "NHiTSStackTensor requires 'InputLength' metadata (added in #1239).");
            int outputLength = TryGetInt(additionalParams, "OutputLength")
                ?? throw new InvalidOperationException(
                    "NHiTSStackTensor requires 'OutputLength' metadata (added in #1239).");
            int hiddenSize = TryGetInt(additionalParams, "HiddenSize")
                ?? throw new InvalidOperationException(
                    "NHiTSStackTensor requires 'HiddenSize' metadata (added in #1239).");
            int numLayers = TryGetInt(additionalParams, "NumLayers")
                ?? throw new InvalidOperationException(
                    "NHiTSStackTensor requires 'NumLayers' metadata (added in #1239).");
            int poolingSize = TryGetInt(additionalParams, "PoolingSize")
                ?? throw new InvalidOperationException(
                    "NHiTSStackTensor requires 'PoolingSize' metadata (added in #1239).");

            var ctorN = type.GetConstructors().OrderByDescending(c => c.GetParameters().Length).FirstOrDefault() ?? throw new MissingLayerCtorException($"Cannot find any public constructor for {layerType} during deserialization.");
            var psN = ctorN.GetParameters();
            var argsN = new object?[psN.Length];
            for (int i = 0; i < psN.Length; i++)
            {
                var p = psN[i];
                var n = (p.Name ?? "").ToLowerInvariant();
                argsN[i] = n switch
                {
                    "inputlength" => inputLength,
                    "outputlength" => outputLength,
                    "hiddensize" => hiddenSize,
                    "numlayers" => numLayers,
                    // numBlocks is vestigial in NHiTSStackTensor's current
                    // impl (ctor param but doesn't influence internal state);
                    // GetMetadata intentionally doesn't persist it.
                    "numblocks" => 1,
                    "poolingsize" => poolingSize,
                    // seed: same — consumed at ctor to seed _random and
                    // discarded. Post-training the random state has advanced
                    // past the original seed, so persisting is misleading.
                    "seed" => 0,
                    _ when p.HasDefaultValue => p.DefaultValue,
                    _ => throw new InvalidOperationException(
                        $"NHiTSStackTensor ctor parameter '{n}' has no metadata mapping " +
                        $"and no default value."),
                };
            }
            instance = ctorN.Invoke(argsN);
        }
        else if (genericDef.Name == "HybridBlockScheduler`1")
        {
            // (sequenceLength, ILayer<T>[] blocks, bool[] isAttentionBlock,
            //  HybridSchedulePattern, modelDimension, IActivationFunction)
            // HybridBlockScheduler.GetMetadata persists SequenceLength /
            // ModelDimension / NumBlocks / SchedulePattern. Fail fast if
            // missing — issue #1239. The inner blocks now round-trip as
            // proper MambaBlock / GatedLinearAttentionLayer instances based
            // on the AttentionPattern + per-type dimension metadata that
            // GetMetadata persists, so SetParameters can correctly route
            // params into each real sub-block (Jamba/Samba/Zamba Clone tests).
            int seqLen = TryGetInt(additionalParams, "SequenceLength")
                ?? throw new InvalidOperationException(
                    "HybridBlockScheduler requires 'SequenceLength' metadata (added in #1239).");
            int modelDim = TryGetInt(additionalParams, "ModelDimension")
                ?? throw new InvalidOperationException(
                    "HybridBlockScheduler requires 'ModelDimension' metadata (added in #1239).");
            int numBlocks = TryGetInt(additionalParams, "NumBlocks")
                ?? throw new InvalidOperationException(
                    "HybridBlockScheduler requires 'NumBlocks' metadata (added in #1239).");

            // Recover the per-position attention/SSM pattern from metadata.
            // Older models without AttentionPattern metadata fall back to
            // all-SSM (the placeholder regime before block-typed deser).
            var isAttentionPattern = new bool[numBlocks];
            if (additionalParams is not null
                && additionalParams.TryGetValue("AttentionPattern", out var patternObj)
                && patternObj is string patternStr)
            {
                int n = Math.Min(numBlocks, patternStr.Length);
                for (int i = 0; i < n; i++)
                    isAttentionPattern[i] = patternStr[i] == '1';
            }

            // Per-type dimension metadata (sampled from the source model's
            // first SSM / first attention block in HybridBlockScheduler.GetMetadata).
            int mambaStateDim = TryGetInt(additionalParams, "MambaStateDimension") ?? 16;
            int mambaExpand = TryGetInt(additionalParams, "MambaExpandFactor") ?? 2;
            int mambaConv = TryGetInt(additionalParams, "MambaConvKernelSize") ?? 4;
            int mambaDtRank = TryGetInt(additionalParams, "MambaDtRank") ?? -1;
            int attnNumHeads = TryGetInt(additionalParams, "AttentionNumHeads") ?? 8;

            // Construct numBlocks proper inner-block instances using the
            // metadata-driven type + dimensions. This makes the cloned
            // scheduler.ParameterCount match the original, so SetParameters
            // routes params into the correct sub-block shapes instead of
            // failing with a placeholder-vs-real mismatch.
            var blocksArrayType = typeof(ILayer<T>).MakeArrayType();
            var blocksArray = Array.CreateInstance(typeof(ILayer<T>), numBlocks);
            for (int b = 0; b < numBlocks; b++)
            {
                ILayer<T> blockLayer;
                if (isAttentionPattern[b])
                {
                    blockLayer = new NeuralNetworks.Layers.SSM.GatedLinearAttentionLayer<T>(
                        seqLen, modelDim, attnNumHeads);
                }
                else
                {
                    blockLayer = new NeuralNetworks.Layers.SSM.MambaBlock<T>(
                        seqLen, modelDim, mambaStateDim, mambaExpand, mambaConv, mambaDtRank);
                }
                blocksArray.SetValue(blockLayer, b);
            }

            var ctorH = type.GetConstructors().OrderByDescending(c => c.GetParameters().Length).FirstOrDefault() ?? throw new MissingLayerCtorException($"Cannot find any public constructor for {layerType} during deserialization.");
            var psH = ctorH.GetParameters();
            var argsH = new object?[psH.Length];
            for (int i = 0; i < psH.Length; i++)
            {
                var p = psH[i];
                var n = (p.Name ?? "").ToLowerInvariant();
                if (p.ParameterType == blocksArrayType) argsH[i] = blocksArray;
                else if (p.ParameterType == typeof(bool[])) argsH[i] = isAttentionPattern;
                else if (p.ParameterType == typeof(int) && n == "sequencelength") argsH[i] = seqLen;
                else if (p.ParameterType == typeof(int) && n == "modeldimension") argsH[i] = modelDim;
                else if (p.ParameterType == typeof(int)) argsH[i] = 1;
                else if (p.ParameterType.IsEnum) argsH[i] = Enum.GetValues(p.ParameterType).GetValue(0);
                else if (p.HasDefaultValue) argsH[i] = p.DefaultValue;
                else argsH[i] = null;
            }
            instance = ctorH.Invoke(argsH);
        }
        else if (genericDef.Name == "HeterogeneousGraphLayer`1")
        {
            // (HeterogeneousGraphMetadata metadata, int outputFeatures,
            //  bool useBasis, int numBases, IActivationFunction)
            // The metadata type carries node/edge type info. Construct a
            // minimal valid instance with one node type ("default") and one
            // edge type ("default" -> "default") so the layer can be allocated.
            // Real Clone() would round-trip the actual metadata via
            // ILayerSerializationExtras.
            var hgmType = type.Assembly.GetTypes().FirstOrDefault(x => x.Name == "HeterogeneousGraphMetadata")
                ?? throw new InvalidOperationException(
                    $"HeterogeneousGraphLayer reconstruction needs the `HeterogeneousGraphMetadata` " +
                    $"type to be present in {type.Assembly.FullName}, but no such type was found. " +
                    $"This usually means the type was renamed, moved to a different assembly, or " +
                    $"trimmed away by an aggressive linker / AOT compile. Restore the type or " +
                    $"adjust the deser branch to look up the new name.");
            object hgm = BuildPlaceholderHeterogeneousGraphMetadata(hgmType);

            var ctorHg = type.GetConstructors().OrderByDescending(c => c.GetParameters().Length).FirstOrDefault() ?? throw new MissingLayerCtorException($"Cannot find any public constructor for {layerType} during deserialization.");
            var psHg = ctorHg.GetParameters();
            var argsHg = new object?[psHg.Length];
            for (int i = 0; i < psHg.Length; i++)
            {
                var p = psHg[i];
                var n = (p.Name ?? "").ToLowerInvariant();
                if (p.ParameterType == hgmType) argsHg[i] = hgm;
                else if (p.ParameterType == typeof(int) && n.Contains("outputfeature")) argsHg[i] = 64;
                else if (p.ParameterType == typeof(int) && n.Contains("numbase")) argsHg[i] = 4;
                else if (p.ParameterType == typeof(int)) argsHg[i] = 1;
                else if (p.ParameterType == typeof(bool)) argsHg[i] = p.HasDefaultValue ? p.DefaultValue : false;
                else if (p.HasDefaultValue) argsHg[i] = p.DefaultValue;
                else argsHg[i] = null;
            }
            instance = ctorHg.Invoke(argsHg);
        }
        else if (genericDef.Name == "GraphConvolutionalLoRAAdapter`1")
        {
            // (ILayer<T> baseLayer, int rank, double alpha, bool freezeBaseLayer)
            // The base layer must implement IGraphConvolutionLayer<T>; the
            // standard placeholder DenseLayer<T> doesn't, so allocate a real
            // GraphConvolutionalLayer<T> as the placeholder instead. Real
            // Clone() round-trips the actual wrapped graph layer via
            // ILayerSerializationExtras.
            var graphConvType = typeof(NeuralNetworks.Layers.GraphConvolutionalLayer<T>);
            var graphConvCtor = graphConvType.GetConstructor(new[] { typeof(int), typeof(int), typeof(IActivationFunction<T>) });
            var graphPlaceholder = graphConvCtor?.Invoke(new object?[] { 64, 64, null });
            if (graphPlaceholder is null)
                throw new InvalidOperationException("Could not construct GraphConvolutionalLayer placeholder for GraphConvolutionalLoRAAdapter.");

            int gclRank = TryGetInt(additionalParams, "Rank") ?? 4;
            double gclAlpha = TryGetDouble(additionalParams, "Alpha") ?? -1.0;
            bool gclFreeze = TryGetBool(additionalParams, "FreezeBaseLayer") ?? true;
            var gclCtor = type.GetConstructors().First();
            instance = gclCtor.Invoke(new object?[] { graphPlaceholder, gclRank, gclAlpha, gclFreeze });
        }
        else if (IsLoRAAdapterWithSpecificValidation(genericDef))
        {
            // LoRA adapters with extra ctor validation (range checks,
            // index lookups against bank sizes, etc.). My matcher's generic
            // defaults sometimes violate the validation, so we hand-pick
            // safe values for these adapters here.
            instance = ConstructLoRAAdapterWithValidation<T>(type, genericDef, additionalParams, layerType);
            if (instance is null)
            {
                throw new InvalidOperationException(
                    $"Could not construct {layerType} (LoRA adapter with constraint failures).");
            }
        }
        else if (IsLoRAAdapterRequiringSharedMatrices(genericDef))
        {
            // VeRA / TiedLoRA / DVoRA require their static
            // InitializeSharedMatrices to be called once before any adapter
            // instance is constructed. Initialize now using sensible defaults
            // so reconstruction succeeds. This is correct round-trip: real
            // Clone() should also call InitializeSharedMatrices once at the
            // network level, but doing it lazily here is the safer fallback.
            EnsureLoRASharedMatricesInitialized<T>(genericDef);
            instance = TryConstructByMatchingMetadata<T>(type, inputShape, outputShape, additionalParams, layerType);
            if (instance is null)
            {
                throw new InvalidOperationException(
                    $"Could not construct {layerType} via reflection matcher even after initializing shared matrices.");
            }
        }
        else if (genericDef == typeof(SequenceTokenSliceLayer<>))
        {
            // SequenceTokenSliceLayer<T>(Position position) — round-tripped
            // via GetMetadata("Position") as its enum name. Use the shared
            // TryGetEnum<TEnum> helper so already-typed values pass through
            // (the inline Enum.TryParse path silently ignored non-string
            // values), and so the parsing convention stays consistent with
            // the rest of DeserializationHelper.
            //
            // Default to Position.Last only when the key is ABSENT.
            // GetMetadata always writes the enum name, so a present-but-
            // unparseable value means corrupt or incompatible serialized
            // data — surface that as an error rather than silently
            // collapsing to Last and changing layer semantics.
            SequenceTokenSliceLayer<T>.Position position;
            bool positionPresent = additionalParams is not null
                && additionalParams.TryGetValue("Position", out _);
            if (!positionPresent)
            {
                position = SequenceTokenSliceLayer<T>.Position.Last;
            }
            else
            {
                var parsed = TryGetEnum<SequenceTokenSliceLayer<T>.Position>(additionalParams, "Position");
                if (parsed is null)
                {
                    var raw = additionalParams!["Position"];
                    throw new InvalidOperationException(
                        $"Invalid SequenceTokenSliceLayer Position metadata '{raw}' " +
                        $"(type {raw?.GetType().Name ?? "null"}). Expected one of: " +
                        $"{string.Join(", ", Enum.GetNames(typeof(SequenceTokenSliceLayer<T>.Position)))}.");
                }
                position = parsed.Value;
            }
            instance = new SequenceTokenSliceLayer<T>(position);
        }
        else if (genericDef == typeof(RBMLayer<>))
        {
            int visibleUnits = inputShape[0];
            int hiddenUnits = outputShape[0];
            instance = new RBMLayer<T>(visibleUnits, hiddenUnits, (IActivationFunction<T>?)null);
        }
        else if (genericDef == typeof(SpikingLayer<>))
        {
            int inputSize = inputShape[0];
            int outputSize = outputShape[0];
            instance = new SpikingLayer<T>(inputSize, outputSize);
        }
        else if (genericDef == typeof(TemporalMemoryLayer<>))
        {
            int columnCount = inputShape[0];
            int totalCells = outputShape[0];
            int cellsPerColumn = totalCells / Math.Max(1, columnCount);
            instance = new TemporalMemoryLayer<T>(columnCount, cellsPerColumn);
        }
        else if (genericDef == typeof(SpatialPoolerLayer<>))
        {
            int columnCount = outputShape[0];
            double sparsityThreshold = TryGetDouble(additionalParams, "SparsityThreshold") ?? 0.02;
            instance = new SpatialPoolerLayer<T>(columnCount, sparsityThreshold);
        }
        else if (genericDef == typeof(MeasurementLayer<>))
        {
            int size = inputShape[0];
            instance = new MeasurementLayer<T>(size);
        }
        else if (genericDef == typeof(QuantumLayer<>))
        {
            int inputSize = inputShape[0];
            int outputSize = outputShape[0];
            int numQubits = TryGetInt(additionalParams, "NumQubits") ?? Math.Max(4, (int)(Math.Log(Math.Max(inputSize, outputSize)) / Math.Log(2)));
            instance = new QuantumLayer<T>(inputSize, outputSize, numQubits);
        }
        else if (genericDef == typeof(MeanLayer<>) || genericDef == typeof(LogVarianceLayer<>))
        {
            // MeanLayer/LogVarianceLayer are now lazy: ctor signature is (int axis).
            // Resolve from input shape after construction so the deserialized layer
            // has concrete shapes for any subsequent SetParameters call.
            int axis = TryGetInt(additionalParams, "Axis") ?? 0;
            instance = Activator.CreateInstance(type, axis);
            if (instance is NeuralNetworks.Layers.LayerBase<T> meanOrVarLayer
                && !meanOrVarLayer.IsShapeResolved
                && inputShape.Length > 0
                && Array.TrueForAll(inputShape, d => d > 0))
            {
                meanOrVarLayer.ResolveFromShape(inputShape);
            }
        }
        else if (genericDef == typeof(ResidualLayer<>))
        {
            // ResidualLayer wraps an inner DenseLayer. Reconstruct inner layer from metadata.
            int innerInputSize = TryGetInt(additionalParams, "InnerInputSize") ?? inputShape[0];
            int innerOutputSize = TryGetInt(additionalParams, "InnerOutputSize") ?? inputShape[0];

            var activationFuncType = typeof(IActivationFunction<>).MakeGenericType(typeof(T));
            object? innerActivation = TryCreateActivationInstance(additionalParams, "ScalarActivationType", activationFuncType);
            var innerLayer = new DenseLayer<T>(innerOutputSize, innerActivation as IActivationFunction<T>);
            // Eagerly resolve so ValidateInnerLayer sees concrete matching shapes.
            innerLayer.ResolveFromShape(new[] { innerInputSize });

            // Create ResidualLayer directly to avoid constructor ambiguity
            object? residualActivation = TryCreateActivationInstance(additionalParams, "ScalarActivationType", activationFuncType);
            instance = new ResidualLayer<T>(innerLayer, residualActivation as IActivationFunction<T>);
        }
        else if (openGenericType.FullName != null && openGenericType.FullName.EndsWith(".MambaBlock`1"))
        {
            // MambaBlock(int sequenceLength, int modelDimension, int stateDimension, int expandFactor, int convKernelSize, int dtRank)
            int sequenceLength = inputShape.Length > 0 ? inputShape[0] : 1;
            int modelDimension = inputShape.Length > 1 ? inputShape[1] : 256;
            int stateDimension = TryGetInt(additionalParams, "StateDimension") ?? 16;
            int expandFactor = TryGetInt(additionalParams, "ExpandFactor") ?? 2;
            int convKernelSize = TryGetInt(additionalParams, "ConvKernelSize") ?? 4;

            // MambaBlock has a constructor with all optional params after the first:
            // (int sequenceLength, int modelDimension = 256, int stateDimension = 16, int expandFactor = 2, int convKernelSize = 4, int dtRank = -1, IActivationFunction<T>? activationFunction = null)
            // Use reflection to find the constructor and call it with named parameter matching
            var ctors = type.GetConstructors();
            var ctor = ctors.FirstOrDefault(c =>
            {
                var p = c.GetParameters();
                return p.Length >= 1 && p[0].ParameterType == typeof(int);
            });
            if (ctor is not null)
            {
                var parameters = ctor.GetParameters();
                var args = new object?[parameters.Length];
                args[0] = sequenceLength;
                for (int pi = 1; pi < parameters.Length; pi++)
                {
                    if (parameters[pi].Name == "modelDimension") args[pi] = modelDimension;
                    else if (parameters[pi].Name == "stateDimension") args[pi] = stateDimension;
                    else if (parameters[pi].Name == "expandFactor") args[pi] = expandFactor;
                    else if (parameters[pi].Name == "convKernelSize") args[pi] = convKernelSize;
                    else if (parameters[pi].HasDefaultValue) args[pi] = parameters[pi].DefaultValue;
                    else args[pi] = null;
                }
                instance = ctor.Invoke(args);
            }
            else
            {
                // Keep NotSupportedException here (rather than the
                // MissingLayerCtorException used by other branches) so
                // we preserve the original observable behavior: the
                // outer catch in CreateLayerFromType only re-routes
                // MissingLayerCtorException to the metadata matcher;
                // NotSupportedException propagates to the caller, which
                // is the right signal when MambaBlock's expected named-
                // parameter ctor isn't present (a specifically-shaped
                // layer that the generic matcher cannot reconstruct).
                throw new NotSupportedException("Cannot find MambaBlock constructor for deserialization.");
            }
        }
        else if (openGenericType.FullName != null && openGenericType.FullName.EndsWith(".ContinuumMemorySystemLayer`1"))
        {
            // ContinuumMemorySystemLayer(int[] inputShape, int hiddenDim,
            //                            int numFrequencyLevels = 3, ...)
            // numFrequencyLevels controls the count of internal MLP blocks
            // (Hope passes 5 from inContextLearningLevels per Behrouz et al.
            // 2025 §3.2). If the saved value isn't restored, the default 3
            // is used and the deserialized layer's ParameterCount is 3/5 of
            // the saved vector — SetParameters then rejects with
            // "Parameter vector length (X) does not match total parameters (Y)"
            // and Clone fails. Read NumFrequencyLevels from additionalParams
            // (serialized by the layer alongside its weights).
            int hiddenDim = TryGetInt(additionalParams, "HiddenDim")
                ?? (inputShape.Length > 0 ? inputShape[inputShape.Length - 1] : 256);
            int numFreqLevels = TryGetInt(additionalParams, "NumFrequencyLevels") ?? 3;
            var ctor = type.GetConstructors()
                .Where(c => c.GetParameters().Length >= 2 &&
                       c.GetParameters()[0].ParameterType == typeof(int[]) &&
                       c.GetParameters()[1].ParameterType == typeof(int))
                .OrderBy(c => c.GetParameters().Length)
                .FirstOrDefault();
            if (ctor != null)
            {
                var parameters = ctor.GetParameters();
                var args = new object?[parameters.Length];
                args[0] = inputShape;
                args[1] = hiddenDim;
                for (int pi = 2; pi < parameters.Length; pi++)
                {
                    if (parameters[pi].Name == "numFrequencyLevels")
                        args[pi] = numFreqLevels;
                    else
                        args[pi] = parameters[pi].HasDefaultValue ? parameters[pi].DefaultValue : null;
                }
                instance = ctor.Invoke(args);
            }
            else
            {
                // Keep NotSupportedException to preserve original
                // observable behavior — see MambaBlock branch above
                // for rationale.
                throw new NotSupportedException("Cannot find ContinuumMemorySystemLayer constructor for deserialization.");
            }
        }
        else if (openGenericType.FullName != null && openGenericType.FullName.EndsWith(".FeedForwardLayer`1"))
        {
            // FeedForwardLayer is now lazy:
            //   FeedForwardLayer(int outputSize, IActivationFunction<T>? activationFunction = null)
            // Input feature size is resolved from input.Shape[^1] on first Forward.
            int outputSize = outputShape.Length > 0 ? outputShape[^1] : 0;
            if (outputSize <= 0)
                throw new InvalidOperationException(
                    $"FeedForwardLayer deserialization requires positive outputSize; got {outputSize}.");

            var activationFuncType = typeof(IActivationFunction<>).MakeGenericType(typeof(T));
            object? activation = TryCreateActivationInstance(additionalParams, "ScalarActivationType", activationFuncType);

            var ctor = type.GetConstructor(new Type[] { typeof(int), activationFuncType });
            if (ctor is null)
                throw new MissingLayerCtorException(
                    "Cannot find FeedForwardLayer constructor with (int, IActivationFunction<T>).");
            instance = ctor.Invoke(new object?[] { outputSize, activation });
        }
        else if (openGenericType.FullName != null
            && (openGenericType.FullName.EndsWith(".RWKVLayer`1")
                || openGenericType.FullName.EndsWith(".Mamba2Block`1")))
        {
            // RWKVLayer(int sequenceLength, int modelDimension = 256, int numHeads = 8, …)
            // Mamba2Block(int sequenceLength, int modelDimension = 256, int stateDimension = 64,
            //             int numHeads = 8, int expandFactor = 2, int convKernelSize = 4,
            //             int chunkSize = 64, IActivationFunction?, IInitializationStrategy?)
            // Both take (sequenceLength, modelDimension) derived from the 2D input shape,
            // with remaining positional int parameters matched from metadata by parameter name.
            int sequenceLength = inputShape.Length > 0 ? inputShape[0] : 1;
            int modelDimension = inputShape.Length > 1 ? inputShape[1] : 256;

            var ctor = type.GetConstructors()
                .Where(c =>
                {
                    var p = c.GetParameters();
                    return p.Length >= 1 && p[0].ParameterType == typeof(int);
                })
                .OrderByDescending(c => c.GetParameters().Length)
                .FirstOrDefault();
            if (ctor is null)
                throw new MissingLayerCtorException(
                    $"Cannot find {layerType} constructor for deserialization.");

            var parameters = ctor.GetParameters();
            var args = new object?[parameters.Length];
            for (int pi = 0; pi < parameters.Length; pi++)
            {
                string name = parameters[pi].Name ?? string.Empty;
                if (name == "sequenceLength") args[pi] = sequenceLength;
                else if (name == "modelDimension") args[pi] = modelDimension;
                else if (parameters[pi].ParameterType == typeof(int))
                    args[pi] = TryGetInt(additionalParams, char.ToUpperInvariant(name[0]) + name.Substring(1))
                        ?? (parameters[pi].HasDefaultValue ? (int?)parameters[pi].DefaultValue : null);
                else
                    args[pi] = parameters[pi].HasDefaultValue ? parameters[pi].DefaultValue : null;
            }
            instance = ctor.Invoke(args);
        }
        else
        {
            // Default fallback path. Prior behavior was a single
            // type.GetConstructor(new[] { typeof(int[]) }) lookup, which left ~190
            // LayerBase<T> subclasses (the vast majority of layers in this
            // codebase: every Mamba/SSM, every graph layer, every conv variant,
            // every pooling/norm variant, every modern attention layer) crashing
            // Clone() / DeepCopy() with NotSupportedException because they have
            // no (int[]) constructor and no explicit branch above. Tracked as
            // #1235.
            //
            // The new fallback implements direction (1) of #1235's three proposed
            // fixes: a reflection-driven default-ctor matcher. It iterates
            // public constructors by descending arity and invokes the FIRST
            // overload whose parameters can all be resolved from
            // (inputShape, outputShape, additionalParams, default values).
            // This is a longest-fillable-first heuristic, not a true scored
            // match — see TryConstructByMatchingMetadata's remarks for the
            // selection caveat. It handles the overwhelming majority of leaf
            // layers without per-layer maintenance.
            //
            // Layers whose construction genuinely needs a wrapped layer instance
            // (LoRA / PEFT adapters, composite blocks) still cannot be handled
            // here — their reconstruction needs the wrapped layer reference,
            // which the helper API doesn't carry. Those still throw.
            instance = TryConstructByMatchingMetadata<T>(type, inputShape, outputShape, additionalParams, layerType);
            if (instance is null)
            {
                // Last-resort legacy behavior: try the (int[]) catch-all that
                // existed before this fallback. Preserves working semantics for
                // any layer that previously hit this path.
                var ctor = type.GetConstructor(new Type[] { typeof(int[]) });
                if (ctor is null)
                {
                    throw new NotSupportedException(
                        $"Layer type {layerType} is not supported for deserialization (no known constructor found). " +
                        $"Public constructors: [{string.Join(" | ", type.GetConstructors().Select(c => "(" + string.Join(", ", c.GetParameters().Select(p => p.ParameterType.Name + " " + p.Name)) + ")"))}]. " +
                        $"Either add a dedicated branch in DeserializationHelper.CreateLayerFromType, ensure GetMetadata persists the constructor parameters, or give the layer a (int[] inputShape) constructor.");
                }

                instance = ctor.Invoke(new object[] { inputShape });
            }
        }
        }
        catch (MissingLayerCtorException ex)
        {
            // Explicit branch reported a constructor lookup miss via the
            // structured marker exception. This is the preferred signal
            // for the matcher fall-through.
            branchFailure = ex;
            instance = null;
        }
        catch (InvalidOperationException ex) when (IsMissingCtorMessage(ex.Message))
        {
            // Defensive fallback: as of #1239 all 44 in-tree throw sites
            // have been migrated to MissingLayerCtorException, so the
            // catch above handles them via the structured marker. This
            // legacy-convention catch stays in place to handle third-
            // party serialization paths or test-only layer types that
            // might still surface "Cannot find ... constructor" via a
            // plain InvalidOperationException — keeps the matcher
            // fall-through behavior stable for those edge cases.
            branchFailure = ex;
            instance = null;
        }
        if (instance is null && branchFailure is not null)
        {
            instance = TryConstructByMatchingMetadata<T>(
                type,
                inputShape ?? Array.Empty<int>(),
                outputShape ?? Array.Empty<int>(),
                additionalParams,
                layerType);
            if (instance is null)
            {
                // Re-throw the original "Cannot find" so the caller still
                // sees the diagnostic if both paths fail.
                throw branchFailure;
            }
        }
        if (instance == null)
        {
            throw new InvalidOperationException($"Failed to create instance of layer type {layerType}.");
        }

        // Resolve lazy layers from the serialized inputShape so SetParameters
        // can land trained weights. Layers whose lazy SetParameters can self-
        // resolve from the parameter vector (DenseLayer, FullyConnectedLayer,
        // FeedForwardLayer, LayerNormalizationLayer, BatchNormalizationLayer)
        // tolerate ResolveFromShape failures here since SetParameters will
        // recover; for others the layer's first Forward resolves it.
        if (instance is NeuralNetworks.Layers.LayerBase<T> lb
            && !lb.IsShapeResolved
            && inputShape != null
            && inputShape.Length > 0
            && System.Array.TrueForAll(inputShape, d => d > 0))
        {
            try
            {
                lb.ResolveFromShape(inputShape);
            }
            catch (Exception ex) when (ex is ArgumentException || ex is InvalidOperationException)
            {
                // Layer's OnFirstForward expects a different input rank.
                // SetParameters now self-resolves from the parameter vector
                // size for the lazy-migrated layer family, so the trained
                // weights still land. Trace it for telemetry — silent swallow
                // hid #1221 for too long.
                System.Diagnostics.Trace.TraceWarning(
                    $"DeserializationHelper: ResolveFromShape failed for {lb.GetType().Name} " +
                    $"with inputShape [{string.Join(", ", inputShape)}]: {ex.GetType().Name}: {ex.Message}. " +
                    "Layer will resolve via SetParameters or first Forward.");
            }
        }

        return (ILayer<T>)instance;
    }

    private static object CreateDenseLayer<T>(Type type, int[] inputShape, int[] outputShape, Dictionary<string, object>? additionalParams)
    {
        // DenseLayer is now lazy-only:
        //   DenseLayer(int outputSize, IActivationFunction<T>? activationFunction = null,
        //              IInitializationStrategy<T>? initializationStrategy = null)
        // After construction we MUST call ResolveFromShape so SetParameters can load
        // saved weights without hitting the -1 sentinel guard.
        var activationFuncType = typeof(IActivationFunction<>).MakeGenericType(typeof(T));
        var initStrategyType = typeof(AiDotNet.Initialization.IInitializationStrategy<>).MakeGenericType(typeof(T));
        var ctor = type.GetConstructor(new Type[] { typeof(int), activationFuncType, initStrategyType });
        if (ctor is null)
        {
            throw new MissingLayerCtorException("Cannot find DenseLayer constructor with (int, IActivationFunction<T>, IInitializationStrategy<T>).");
        }

        object? activation = TryCreateActivationInstance(additionalParams, "ScalarActivationType", activationFuncType);
        object instance = ctor.Invoke(new object?[] { outputShape[0], activation, null });

        // Resolve lazy shape so SetParameters can populate weights/biases on the deserialized layer.
        if (instance is NeuralNetworks.Layers.LayerBase<T> layerBase && !layerBase.IsShapeResolved && inputShape.Length > 0 && inputShape[0] > 0)
        {
            layerBase.ResolveFromShape(new[] { inputShape[0] });
        }
        return instance;
    }

    private static object CreateMultiHeadAttentionLayer<T>(Type type, int[] inputShape, Dictionary<string, object>? additionalParams)
    {
        // MultiHeadAttentionLayer is now lazy:
        //   MultiHeadAttentionLayer(int headCount, int headDimension,
        //                           IActivationFunction<T>? activationFunction = null,
        //                           IInitializationStrategy<T>? initializationStrategy = null)
        // Sequence length is inferred per-forward; embeddingDimension = headCount * headDimension.
        // Q/K/V/O projection weights are allocated lazily on first Forward.
        //
        // The optional initializationStrategy parameter MUST be included in
        // the reflection signature even though it has a default — Type.GetConstructor
        // matches by exact parameter list, not by "first N + defaults".
        if (inputShape.Length < 2)
        {
            throw new InvalidOperationException("MultiHeadAttentionLayer requires input shape [sequenceLength, embeddingDimension].");
        }

        int embDim = inputShape[1];
        int headCount = TryGetInt(additionalParams, "HeadCount") ?? ResolveDefaultHeadCount(embDim);
        int headDimension = TryGetInt(additionalParams, "HeadDimension") ?? (embDim / headCount);
        if (headDimension * headCount != embDim)
        {
            throw new InvalidOperationException(
                $"MultiHeadAttentionLayer deserialization: embeddingDimension {embDim} is not divisible by headCount {headCount}.");
        }

        var activationFuncType = typeof(IActivationFunction<>).MakeGenericType(typeof(T));
        var initStrategyType = typeof(IInitializationStrategy<>).MakeGenericType(typeof(T));
        var ctor = type.GetConstructor(new Type[] { typeof(int), typeof(int), activationFuncType, initStrategyType });
        if (ctor is null)
        {
            throw new MissingLayerCtorException("Cannot find MultiHeadAttentionLayer constructor with (int, int, IActivationFunction<T>, IInitializationStrategy<T>).");
        }

        object? activation = TryCreateActivationInstance(additionalParams, "ScalarActivationType", activationFuncType);
        return ctor.Invoke(new object?[] { headCount, headDimension, activation, null });
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
            throw new MissingLayerCtorException("Cannot find FlashAttentionLayer constructor with expected signature.");
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
            throw new MissingLayerCtorException("Cannot find CachedMultiHeadAttention constructor with expected signature.");
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
            throw new MissingLayerCtorException("Cannot find PagedCachedMultiHeadAttention constructor with expected signature.");
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
            if (string.IsNullOrWhiteSpace(raw) || raw is null) return Array.Empty<string>();
            return raw.Split(new[] { '|' }, StringSplitOptions.RemoveEmptyEntries);
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
            throw new MissingLayerCtorException("Cannot find MultiLoRAAdapter constructor with expected signature.");
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
                throw new MissingLayerCtorException("Cannot find ActivationFunctionFactory.CreateActivationFunction method.");
            }

            activationFunction = createMethod.Invoke(null, new object[] { activationFunctionEnum });
        }

        if (activationFunction == null)
        {
            throw new InvalidOperationException("Failed to create activation function for ActivationLayer.");
        }

        if (vectorActivationType.IsInstanceOfType(activationFunction))
        {
            var ctor = type.GetConstructor(new Type[] { vectorActivationType });
            if (ctor is null)
            {
                throw new MissingLayerCtorException("Cannot find ActivationLayer constructor with (IVectorActivationFunction<T>).");
            }
            return ctor.Invoke(new object[] { activationFunction });
        }

        var scalarCtor = type.GetConstructor(new Type[] { scalarActivationType });
        if (scalarCtor is null)
        {
            throw new MissingLayerCtorException("Cannot find ActivationLayer constructor with (IActivationFunction<T>).");
        }
        return scalarCtor.Invoke(new object[] { activationFunction });
    }

    /// <summary>
    /// Creates a GRU layer during deserialization.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> GRU (Gated Recurrent Unit) layers process sequences of data
    /// and maintain memory of previous inputs. This method recreates a GRU layer with the
    /// correct input size and hidden size from the serialized shape data.</para>
    /// </remarks>
    private static object CreateGRULayer<T>(Type type, int[] inputShape, int[] outputShape, Dictionary<string, object>? additionalParams)
    {
        // GRULayer(int hiddenSize, bool returnSequences = false, IActivationFunction<T>? activation = null, IActivationFunction<T>? recurrentActivation = null)
        // inputSize is now resolved lazily on first forward (lazy-shape contract from #1220).
        int hiddenSize = outputShape.Length >= 2 ? outputShape[^1] : outputShape[0];
        // ReturnSequences serialization contract: when missing from
        // additionalParams, infer from the persisted output shape rather
        // than hard-coding `true` (which contradicts the GRULayer ctor
        // default of `false` and silently changes output rank for any
        // checkpoint that doesn't pin the value). If the persisted output
        // has the same rank as the input, the layer was emitting full
        // [batch, time, hidden] sequences; otherwise it was returning the
        // last hidden state only.
        bool returnSequences = TryGetBool(additionalParams, "ReturnSequences")
            ?? (outputShape.Length == inputShape.Length);

        var activationFuncType = typeof(IActivationFunction<>).MakeGenericType(typeof(T));
        var ctor = type.GetConstructor(new Type[] { typeof(int), typeof(bool), activationFuncType, activationFuncType });
        if (ctor is null)
        {
            throw new MissingLayerCtorException("Cannot find GRULayer constructor with (int hiddenSize, bool returnSequences, IActivationFunction<T>?, IActivationFunction<T>?).");
        }

        object? activation = TryCreateActivationInstance(additionalParams, "ScalarActivationType", activationFuncType);
        return ctor.Invoke(new object?[] { hiddenSize, returnSequences, activation, null });
    }

    /// <summary>
    /// Creates an LSTM layer during deserialization.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> LSTM (Long Short-Term Memory) layers are a type of recurrent
    /// neural network that can learn long-term dependencies. This method recreates an LSTM layer
    /// with the correct input size, hidden size, and input shape from serialized data.</para>
    /// </remarks>
    private static object CreateLSTMLayer<T>(Type type, int[] inputShape, int[] outputShape, Dictionary<string, object>? additionalParams)
    {
        // LSTMLayer(int hiddenSize, IActivationFunction<T>? activation = null, IActivationFunction<T>? recurrentActivation = null)
        // Lazy layer — _inputSize is resolved from input shape on first forward.
        int hiddenSize = outputShape.Length >= 2 ? outputShape[^1] : outputShape[0];

        var activationFuncType = typeof(IActivationFunction<>).MakeGenericType(typeof(T));
        var ctor = type.GetConstructor(new Type[] { typeof(int), activationFuncType, activationFuncType });
        if (ctor is null)
        {
            throw new MissingLayerCtorException("Cannot find LSTMLayer constructor with (int, IActivationFunction<T>, IActivationFunction<T>).");
        }

        object? activation = TryCreateActivationInstance(additionalParams, "ScalarActivationType", activationFuncType);
        return ctor.Invoke(new object?[] { hiddenSize, activation, null });
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

    /// <summary>
    /// Tries to restore an activation function from serialized metadata.
    /// </summary>
    private static object? TryRestoreActivation<T>(Dictionary<string, object>? additionalParams)
    {
        if (additionalParams == null) return null;

        string? typeName = null;
        if (additionalParams.TryGetValue("ScalarActivationType", out var atVal))
            typeName = atVal as string;

        if (string.IsNullOrEmpty(typeName)) return null;

        var activationType = Type.GetType(typeName);
        if (activationType == null) return null;

        if (activationType.IsGenericTypeDefinition)
            activationType = activationType.MakeGenericType(typeof(T));

        return Activator.CreateInstance(activationType);
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

    private static int[]? TryGetIntArray(Dictionary<string, object>? parameters, string key)
    {
        if (parameters is null || !parameters.TryGetValue(key, out var value) || value is null)
            return null;

        if (value is int[] arr)
            return arr;

        string str = value.ToString() ?? string.Empty;
        if (string.IsNullOrWhiteSpace(str))
            return null;

        var parts = str.Split(new[] { ',', ' ' }, StringSplitOptions.RemoveEmptyEntries);
        var result = new int[parts.Length];
        for (int i = 0; i < parts.Length; i++)
        {
            if (!int.TryParse(parts[i], out result[i]))
                return null;
        }
        return result;
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

    private static TEnum? TryGetEnum<TEnum>(Dictionary<string, object>? parameters, string key) where TEnum : struct, Enum
    {
        if (parameters != null && parameters.TryGetValue(key, out var value) && value != null)
        {
            if (value is TEnum e)
                return e;
            if (Enum.TryParse<TEnum>(value.ToString() ?? string.Empty, out TEnum parsed))
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

        // Recover per-instance scalar parameters that LayerBase.GetMetadata
        // captured under "ScalarActivationAlpha" / "VectorActivationAlpha"
        // (LayerBase.CaptureScalarActivationParameters). Without this, a
        // layer wired with LeakyReLU(0.2) round-trips through clone as
        // LeakyReLU(0.01) because the default Activator.CreateInstance call
        // below uses the constructor's default alpha — the network's
        // negative-input forward then diverges by the slope ratio.
        double? alpha = TryGetDouble(parameters,
            key == "VectorActivationType" ? "VectorActivationAlpha" : "ScalarActivationAlpha");

        try
        {
            object? instance = null;

            // Try the single-double constructor (LeakyReLU, ELU, PReLU,
            // RReLU, SELU all expose `Activation(double alpha = …)`) when
            // we have a saved alpha value. Falls through to the
            // parameterless / defaulted-parameter path otherwise.
            if (alpha.HasValue)
            {
                var doubleCtor = type.GetConstructor(new Type[] { typeof(double) });
                if (doubleCtor is not null)
                {
                    try { instance = doubleCtor.Invoke(new object?[] { alpha.Value }); }
                    catch (TargetInvocationException) { instance = null; }
                }
            }

            if (instance is null)
            {
                try
                {
                    instance = Activator.CreateInstance(type);
                }
                catch (MissingMethodException)
                {
                    // Fall back to constructors with all-optional parameters
                    var ctor = type.GetConstructors()
                        .FirstOrDefault(c => c.GetParameters().All(p => p.HasDefaultValue));
                    instance = ctor?.Invoke(ctor.GetParameters().Select(p => p.DefaultValue).ToArray());
                }
            }

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

    /// <summary>
    /// Sensible-default lookup for int-typed ML hyperparameters whose names
    /// don't match the input/output-shape naming heuristic. Used by
    /// <see cref="TryConstructByMatchingMetadata{T}"/> as a final fallback
    /// before declaring a constructor-parameter unresolvable. These defaults
    /// only fire when the layer's <c>GetMetadata</c> didn't persist the
    /// parameter, which on the real Clone() path it should — but lots of
    /// existing layers don't yet (tracked in #1235). The defaults keep
    /// Clone() / DeepCopy() functional in that case rather than throwing.
    /// </summary>
    private static int? TryDefaultMlIntHyperparameter(string pNameLower)
    {
        // Attention head counts.
        if (pNameLower.Contains("numhead") || pNameLower == "headcount" || pNameLower == "numkvhead") return 4;
        // Convolution / pooling kernels.
        if (pNameLower.Contains("kernelsize") || pNameLower == "kernel") return 3;
        if (pNameLower.Contains("poolsize") || pNameLower == "pool") return 2;
        if (pNameLower.Contains("stride")) return 1;
        if (pNameLower.Contains("dilation")) return 1;
        if (pNameLower.Contains("padding")) return 0;
        if (pNameLower.Contains("upscalefactor") || pNameLower == "upscale") return 2;
        // Time-series / video.
        if (pNameLower.Contains("numframe")) return 1;
        if (pNameLower.Contains("contextlength")) return 16;
        if (pNameLower.Contains("inputlength") || pNameLower.Contains("outputlength")) return 16;
        if (pNameLower.Contains("inputseqlen") || pNameLower.Contains("seqlen")) return 16;
        // Spatial.
        if (pNameLower.Contains("spatialsize") || pNameLower == "height" || pNameLower == "width") return 8;
        if (pNameLower.Contains("inputheight") || pNameLower.Contains("inputwidth")) return 8;
        if (pNameLower.Contains("outputspatialsize") || pNameLower.Contains("inputspatialsize")) return 8;
        // Normalisation / blocks.
        if (pNameLower.Contains("numgroup")) return 1;
        // Channel/dim defaults are 64 not 4: many layers (attention, group-norm,
        // SE, mixers) carry divisibility constraints (channels divisible by
        // numHeads, numChannels divisible by numGroups, etc.) and 64 satisfies
        // 1/2/4/8/16/32/64 head counts and 1/2/4/8 group counts.
        if (pNameLower == "numchannel" || pNameLower == "channels" || pNameLower == "numchannels") return 64;
        if (pNameLower.Contains("inputchannel") || pNameLower.Contains("inchannel")) return 64;
        if (pNameLower.Contains("outputchannel") || pNameLower.Contains("outchannel")) return 64;
        if (pNameLower.Contains("skipchannel")) return 64;
        // numChannels suffix matchers — covers any *Channels naming.
        if (pNameLower.EndsWith("channels", StringComparison.Ordinal)) return 64;
        // numMembers (BatchEnsembleLayer ensemble size).
        if (pNameLower.Contains("nummember")) return 4;
        // numSplits (SplitLayer).
        if (pNameLower.Contains("numsplit")) return 2;
        // chainLength (ChainLoRAAdapter), layerIndex (TiedLoRAAdapter),
        // numberOfExperts / numBasis / quantizationBits etc.
        if (pNameLower.Contains("chainlength")) return 2;
        if (pNameLower.Contains("layerindex")) return 0;
        if (pNameLower.Contains("numberofexpert")) return 4;
        if (pNameLower.Contains("filters")) return 64;
        // model/sequence/intermediate dimensions used by MesaNet etc.
        if (pNameLower.Contains("modeldimension")) return 64;
        if (pNameLower.Contains("sequencelength")) return 16;
        if (pNameLower.Contains("numblock") || pNameLower.Contains("numlayer")) return 1;
        if (pNameLower.Contains("numresblock")) return 1;
        if (pNameLower.Contains("growthrate")) return 8;
        if (pNameLower.Contains("ffnmultiplier") || pNameLower.Contains("expansionfactor") || pNameLower == "mlpratio") return 2;
        // Rank / capacity.
        if (pNameLower == "rank" || pNameLower.Contains("maxrank") || pNameLower.Contains("ttrank")
            || pNameLower.Contains("expertrank") || pNameLower.Contains("weightrank") || pNameLower.Contains("activationrank")) return 4;
        if (pNameLower.Contains("numcore")) return 2;
        if (pNameLower.Contains("numexpert") || pNameLower == "topk") return 4;
        if (pNameLower.Contains("numbasis") || pNameLower.Contains("numbase")) return 4;
        if (pNameLower.Contains("hiddendim") || pNameLower.Contains("hiddensize") || pNameLower == "latentdim"
            || pNameLower.Contains("intermediatesize")) return 64;
        if (pNameLower.Contains("basechannel")) return 64;
        if (pNameLower.Contains("latentchannel")) return 64;
        if (pNameLower.Contains("attentionsize") || pNameLower.Contains("feedforwardsize")) return 64;
        if (pNameLower.Contains("feedforwarddim") || pNameLower.Contains("ffwidth") || pNameLower.Contains("ffdim")
            || pNameLower.Contains("ffnwidth") || pNameLower.Contains("ffnhidden")) return 64;
        if (pNameLower.Contains("memorydim") || pNameLower.Contains("memorysize")
            || pNameLower.Contains("memoryslots") || pNameLower.Contains("controllersize")
            || pNameLower.Contains("vectordim")) return 16;
        if (pNameLower.Contains("activerank")) return 4;
        // Extended context length must be > original context length per
        // LongLoRAAdapter's validation. Pair with originalContextLength=16
        // above and return 32 for extended.
        if (pNameLower.Contains("extendedcontextlength")) return 32;
        if (pNameLower.Contains("contextdim") || pNameLower.Contains("querydim")) return 64;
        if (pNameLower.Contains("embeddingdim") || pNameLower.Contains("embedding") || pNameLower == "embeddim"
            || pNameLower.Contains("modeldim") || pNameLower == "dim") return 64;
        if (pNameLower.Contains("headdim") || pNameLower.Contains("headdimension")) return 4;
        if (pNameLower.Contains("transformdim")) return 4;
        if (pNameLower.Contains("numfeature")) return 64;
        if (pNameLower.Contains("numpatch")) return 4;
        if (pNameLower.Contains("maxsequencelength")) return 16;
        if (pNameLower.Contains("numclasses") || pNameLower == "numclass") return 2;
        // (filters matched above; this duplicate was dead code.)
        if (pNameLower.Contains("numpoint")) return 4;
        if (pNameLower.Contains("numprototype")) return 4;
        if (pNameLower.Contains("numroutingiteration")) return 3;
        if (pNameLower.Contains("numcapsule") || pNameLower.Contains("capsuledimension")) return 4;
        if (pNameLower.Contains("neighborsample")) return 4;
        if (pNameLower.Contains("numalternatingiteration")) return 1;
        if (pNameLower.Contains("powerit")) return 1;
        if (pNameLower.Contains("flashattentionthreshold")) return 256;
        if (pNameLower.Contains("autocorrelationfactor") || pNameLower.Contains("sparsityfactor")
            || pNameLower.Contains("distillingfactor")) return 1;
        if (pNameLower.Contains("movingavgkernel")) return 3;
        // Note: extendedcontextlength is already matched above (returns 32 to
        // satisfy LongLoRA's "extended > original" validation); originalcontextlength
        // and attentionshiftsize land here.
        if (pNameLower.Contains("originalcontextlength") || pNameLower.Contains("attentionshiftsize")) return 16;
        // (layerindex matched above; this duplicate was dead code.)
        if (pNameLower.Contains("restartinterval")) return 100;
        if (pNameLower.Contains("warmupstep")) return 0;
        if (pNameLower.Contains("resamplinginterval") || pNameLower.Contains("pruninginterval")
            || pNameLower.Contains("importanceupdateinterval")) return 100;
        if (pNameLower.Contains("minrank")) return 1;
        if (pNameLower.Contains("maxloadedadapter")) return 4;
        if (pNameLower.Contains("quantizationbit") || pNameLower.Contains("quantizationblocksize") || pNameLower == "groupsize") return 8;
        if (pNameLower.Contains("banksize") || pNameLower.Contains("modes") || pNameLower.Contains("width")) return 8;
        if (pNameLower.Contains("seed")) return 0;
        if (pNameLower.Contains("historycapacity")) return 100;
        // (topk matched above; this duplicate Contains-form was partially
        // unreachable. Single-letter "k" stays here.)
        if (pNameLower == "k") return 4;
        if (pNameLower.Contains("windowsize") || pNameLower.Contains("shiftsize")) return 4;
        if (pNameLower.Contains("spirallength")) return 4;
        if (pNameLower.Contains("timeembeddim")) return 16;
        if (pNameLower.Contains("reductionratio")) return 2;
        if (pNameLower.Contains("totalcell")) return 4;
        if (pNameLower.Contains("columncount")) return 4;
        return null;
    }

    /// <summary>
    /// Builds a minimal valid <c>HeterogeneousGraphMetadata</c> for layer
    /// reconstruction: one node type ("default"), one self-loop edge type
    /// ("default" → "default"), 64-dim node features. Real Clone() round-trips
    /// the actual metadata via ILayerSerializationExtras.
    /// </summary>
    private static object BuildPlaceholderHeterogeneousGraphMetadata(Type hgmType)
    {
        var hgm = Activator.CreateInstance(hgmType)
            ?? throw new InvalidOperationException("Could not allocate HeterogeneousGraphMetadata.");

        hgmType.GetProperty("NodeTypes")!.SetValue(hgm, new[] { "default" });
        hgmType.GetProperty("EdgeTypes")!.SetValue(hgm, new[] { "default" });

        var nodeFeats = new System.Collections.Generic.Dictionary<string, int> { ["default"] = 64 };
        hgmType.GetProperty("NodeTypeFeatures")!.SetValue(hgm, nodeFeats);

        // EdgeTypeSchema is Dictionary<string, (string, string)>. Build via reflection.
        var schemaType = typeof(System.Collections.Generic.Dictionary<,>)
            .MakeGenericType(typeof(string), typeof(ValueTuple<string, string>));
        var schema = Activator.CreateInstance(schemaType);
        var addMethod = schemaType.GetMethod("Add");
        addMethod!.Invoke(schema, new object?[] { "default", ("default", "default") });
        hgmType.GetProperty("EdgeTypeSchema")!.SetValue(hgm, schema);

        return hgm;
    }

    /// <summary>
    /// True when this LoRA adapter's constructor performs validation that the
    /// generic matcher's HP defaults can't satisfy without targeted tweaks.
    /// These need a hand-tuned construction path.
    /// </summary>
    private static bool IsLoRAAdapterWithSpecificValidation(Type genericDef)
    {
        var n = genericDef.Name;
        return n == "ChainLoRAAdapter`1"
            || n == "DeltaLoRAAdapter`1"
            || n == "GLoRAAdapter`1"
            || n == "GraphConvolutionalLoRAAdapter`1"
            || n == "LongLoRAAdapter`1"
            || n == "LoRETTAAdapter`1"
            || n == "NOLAAdapter`1"
            || n == "ReLoRAAdapter`1"
            || n == "XLoRAAdapter`1";
    }

    /// <summary>
    /// Reads InnerLayerTypeName / InnerLayerInputShape / InnerLayerOutputShape
    /// from <paramref name="additionalParams"/> (written by
    /// <see cref="LoRA.Adapters.LoRAAdapterBase{T}.GetMetadata"/>) and
    /// recursively builds the wrapped layer via
    /// <see cref="CreateLayerFromType{T}"/>. Falls back to <c>null</c> when
    /// the metadata is absent so callers can take the legacy placeholder
    /// path. Issue #1239 wrapped-layer round-trip.
    /// </summary>
    private static object? TryConstructInnerLayerFromMetadata<T>(Dictionary<string, object>? additionalParams)
    {
        if (additionalParams is null) return null;

        if (!additionalParams.TryGetValue("InnerLayerTypeName", out var typeNameObj)
            || typeNameObj is not string innerTypeName
            || string.IsNullOrEmpty(innerTypeName))
        {
            return null;
        }

        // Parse shape strings — comma-joined int lists written by
        // LoRAAdapterBase.GetMetadata.
        int[] ParseShape(string key)
        {
            if (!additionalParams.TryGetValue(key, out var sObj) || sObj is not string s || string.IsNullOrEmpty(s))
                return Array.Empty<int>();
            var parts = s.Split(',');
            var result = new int[parts.Length];
            for (int i = 0; i < parts.Length; i++)
            {
                if (!int.TryParse(parts[i], out result[i])) return Array.Empty<int>();
            }
            return result;
        }

        var innerInputShape = ParseShape("InnerLayerInputShape");
        var innerOutputShape = ParseShape("InnerLayerOutputShape");
        if (innerInputShape.Length == 0 || innerOutputShape.Length == 0) return null;

        try
        {
            // Recursive deser. The inner layer's GetMetadata-extras are NOT
            // currently nested inside the wrapper's metadata — that's a
            // limitation: any inner-layer-specific scalar metadata (e.g.,
            // a wrapped MultiHeadAttention's NumHeads) won't be available
            // here. The wrapped types we care about most (DenseLayer,
            // FullyConnectedLayer) don't need such extras since their
            // ctors accept just outputSize. Tracked under #1239.
            var inner = CreateLayerFromType<T>(innerTypeName, innerInputShape, innerOutputShape, additionalParams: null);

            // Force-resolve the inner layer's shape so its weight tensors
            // are allocated immediately. Without this, lazy layers like
            // DenseLayer / LSTM stay at ParameterCount==0, which makes the
            // wrapper's flat-vector SetParameters(208) fail with "Expected
            // 0 parameters, but got 208" because the wrapper's
            // ParameterCount delegates to the unresolved inner.
            if (inner is NeuralNetworks.Layers.LayerBase<T> innerBase
                && !innerBase.IsShapeResolved
                && innerInputShape.All(d => d > 0))
            {
                try { innerBase.ResolveFromShape(innerInputShape); }
                catch (Exception resolveEx)
                {
                    // ResolveFromShape can throw if the shape rank doesn't
                    // match what the layer expects. Trace and continue —
                    // the layer's own first Forward may still resolve it.
                    System.Diagnostics.Trace.TraceWarning(
                        $"DeserializationHelper.TryConstructInnerLayerFromMetadata: " +
                        $"ResolveFromShape on reconstructed '{innerTypeName}' " +
                        $"with shape [{string.Join(",", innerInputShape)}] failed: " +
                        $"{resolveEx.Message}");
                }
            }

            return inner;
        }
        catch (Exception ex)
        {
            // Trace and fall back to placeholder so callers don't crash;
            // user can inspect the trace to see why the inner-layer round-
            // trip didn't apply.
            System.Diagnostics.Trace.TraceWarning(
                $"DeserializationHelper.TryConstructInnerLayerFromMetadata: " +
                $"failed to reconstruct inner layer of type '{innerTypeName}' " +
                $"with shape [{string.Join(",", innerInputShape)}] -> " +
                $"[{string.Join(",", innerOutputShape)}]: {ex.Message}");
            return null;
        }
    }

    /// <summary>
    /// Construct a LoRA adapter with constraint-aware defaults. Each adapter
    /// has its own validation requirements (rank-vs-bank-size, original-vs-
    /// extended context, etc.) which the generic matcher can't infer; this
    /// path picks values known to satisfy each adapter's preconditions.
    /// </summary>
    private static object? ConstructLoRAAdapterWithValidation<T>(
        Type type, Type genericDef,
        Dictionary<string, object>? additionalParams, string layerType)
    {
        var ctor = type.GetConstructors().OrderByDescending(c => c.GetParameters().Length).FirstOrDefault();
        if (ctor is null) return null;
        var ps = ctor.GetParameters();
        var args = new object?[ps.Length];

        // Reconstruct the actual wrapped layer from
        // LoRAAdapterBase.GetMetadata's InnerLayerTypeName + shape if
        // present; otherwise fall back to the DenseLayer placeholder for
        // legacy networks serialized before #1239 added the metadata.
        // The placeholder path stays for backward-compatibility — newly
        // serialized networks take the proper round-trip via the
        // metadata-driven branch.
        var baseLayer = TryConstructInnerLayerFromMetadata<T>(additionalParams);
        if (baseLayer is null && !TryCreatePlaceholderInnerLayer<T>(typeof(ILayer<T>), out baseLayer))
            return null;

        for (int i = 0; i < ps.Length; i++)
        {
            var p = ps[i];
            var pt = p.ParameterType;
            string n = (p.Name ?? "").ToLowerInvariant();

            if (pt.IsGenericType && pt.GetGenericTypeDefinition() == typeof(ILayer<>))
            {
                args[i] = baseLayer;
                continue;
            }

            // Prefer the constructor's compiler-supplied default when one
            // exists — those are the layer author's chosen safe values and
            // already satisfy all internal validation. Only hand-pick values
            // for parameters with no default.
            if (p.HasDefaultValue)
            {
                args[i] = p.DefaultValue;
                continue;
            }
            // Pick a value the layer's constraints will accept.
            object? value = (pt, n) switch
            {
                _ when pt == typeof(int) && n.Contains("rank") => 4,
                _ when pt == typeof(int) && n.Contains("ttrank") => 4,
                _ when pt == typeof(int) && n.Contains("expertrank") => 4,
                _ when pt == typeof(int) && n.Contains("numcore") => 2,
                _ when pt == typeof(int) && n.Contains("numbasis") => 4,
                _ when pt == typeof(int) && n.Contains("numbase") => 4,
                _ when pt == typeof(int) && n.Contains("numberofexpert") => 4,
                _ when pt == typeof(int) && n.Contains("chainlength") => 2,
                _ when pt == typeof(int) && n.Contains("originalcontextlength") => 16,
                _ when pt == typeof(int) && n.Contains("extendedcontextlength") => 32,
                _ when pt == typeof(int) && n.Contains("attentionshiftsize") => 8,
                _ when pt == typeof(int) && n.Contains("layerindex") => 0,
                _ when pt == typeof(int) && n.Contains("seed") => 0,
                _ when pt == typeof(int) && n.Contains("restartinterval") => 100,
                _ when pt == typeof(int) && n.Contains("warmupstep") => 0,
                _ when pt == typeof(int) => 4,
                _ when pt == typeof(double) && n.Contains("alpha") => 1.0,
                _ when pt == typeof(double) && n.Contains("deltascaling") => 0.1,
                _ when pt == typeof(double) => 0.5,
                _ when pt == typeof(bool) && n.Contains("freeze") => true,
                _ when pt == typeof(bool) => false,
                _ => null,
            };
            args[i] = value;
        }

        try { return ctor.Invoke(args); }
        catch (Exception ex)
        {
            // Trace the actual failure reason before falling through to the
            // generic "Could not construct ..." error at the caller. The
            // ctor.Invoke wrap turns validation errors into TargetInvocation-
            // Exceptions; unwrap to keep the message actionable.
            System.Diagnostics.Trace.TraceWarning(
                $"DeserializationHelper: LoRA adapter {type.Name} construction failed: " +
                $"{(ex is TargetInvocationException tie ? tie.InnerException?.Message : ex.Message)}");
            return null;
        }
    }

    /// <summary>
    /// True when the adapter type requires
    /// <c>InitializeSharedMatrices</c> to be called once before any instance
    /// is constructed. These adapters use static shared low-rank factors
    /// across all instances and can't be allocated without the shared
    /// state seeded first.
    /// </summary>
    private static bool IsLoRAAdapterRequiringSharedMatrices(Type genericDef)
    {
        var n = genericDef.Name;
        return n == "VeRAAdapter`1"
            || n == "TiedLoRAAdapter`1"
            || n == "DVoRAAdapter`1";
    }

    /// <summary>
    /// Calls the adapter type's <c>InitializeSharedMatrices(int, int, int)</c>
    /// static method with sensible defaults so reflection-based reconstruction
    /// can construct an adapter instance afterwards.
    /// </summary>
    private static void EnsureLoRASharedMatricesInitialized<T>(Type genericDef)
    {
        var closed = genericDef.MakeGenericType(typeof(T));
        var init = closed.GetMethod("InitializeSharedMatrices", BindingFlags.Public | BindingFlags.Static);
        if (init is null) return;
        var ps = init.GetParameters();
        var args = new object?[ps.Length];
        for (int i = 0; i < ps.Length; i++)
        {
            if (ps[i].ParameterType == typeof(int))
            {
                // (inputSize, outputSize, rank) — the canonical signature.
                args[i] = ps[i].Name?.ToLowerInvariant() switch
                {
                    "rank" => 4,
                    _ => 64,
                };
            }
            else if (ps[i].HasDefaultValue) args[i] = ps[i].DefaultValue;
            else args[i] = null;
        }
        try { init.Invoke(null, args); }
        catch (Exception ex)
        {
            // Surface the failure at Trace level instead of swallowing
            // silently — a failed shared-matrix init would otherwise let
            // the next adapter ctor fall through and report a confusing
            // downstream error (e.g., "shared matrices not initialized")
            // with no breadcrumb back to the actual failure here.
            System.Diagnostics.Trace.TraceWarning(
                $"DeserializationHelper.EnsureLoRASharedMatricesInitialized: " +
                $"InitializeSharedMatrices for {genericDef.Name} failed: " +
                $"{(ex is TargetInvocationException tie ? tie.InnerException?.Message : ex.Message)}");
        }
    }

    /// <summary>
    /// Builds a placeholder inner-layer instance when a constructor parameter
    /// expects an <c>ILayer&lt;T&gt;</c> / <c>LayerBase&lt;T&gt;</c> /
    /// <c>ILayer&lt;T&gt;[]</c> / <c>List&lt;ILayer&lt;T&gt;&gt;</c> /
    /// <c>IEnumerable&lt;ILayer&lt;T&gt;&gt;</c> reference.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>Known limitation (issue #1235 follow-up):</b> the placeholder is
    /// a tiny <c>DenseLayer&lt;T&gt;</c> — enough to satisfy null-check
    /// preconditions in LoRA adapters / composite layers so the matcher's
    /// construction step doesn't crash, but NOT semantically equivalent to
    /// the original wrapped layer. Adapters reconstructed via this path
    /// have the WRONG inner layer (a generic DenseLayer instead of e.g.
    /// the original Conv2D / MultiHeadAttention / etc.). Their structural
    /// metadata round-trips; their wrapped layer's parameters do not.
    /// </para>
    /// <para>
    /// Each call emits a <c>Trace.TraceWarning</c> so the placeholder
    /// usage is visible in diagnostics rather than silent. The proper fix
    /// is for each adapter to embed its wrapped base layer through
    /// <c>ILayerSerializationExtras</c>; until that ships, this fallback
    /// keeps Clone()/DeepCopy() functional for the architecture-only
    /// round-trip cases that don't depend on inner-layer parameter values.
    /// </para>
    /// </remarks>
    private static bool TryCreatePlaceholderInnerLayer<T>(Type pType, out object? placeholder)
    {
        placeholder = null;
        Type denseType = typeof(NeuralNetworks.Layers.DenseLayer<T>);

        object? CreatePlaceholderDense()
        {
            // DenseLayer<T> has two overloads with the same arity differing in
            // their activation type (IActivationFunction vs
            // IVectorActivationFunction); Activator.CreateInstance throws
            // AmbiguousMatchException when handed null for the activation
            // slot. Resolve the unambiguous scalar overload by hand.
            //
            // Output size is 64 (not 1) so wrappers like LoRA adapters that
            // read min(inputSize, outputSize) from the wrapped layer's shape
            // don't reject typical rank defaults (rank=4 etc.). The dense
            // layer's input shape is resolved lazily on first forward, so
            // outputSize is the only ctor arg.
            var ctor = denseType.GetConstructor(new[]
            {
                typeof(int),
                typeof(IActivationFunction<T>),
                typeof(Initialization.IInitializationStrategy<T>),
            });
            var dense = ctor?.Invoke(new object?[] { 64, null, null });
            // Pre-resolve to a 64-input/64-output shape so wrappers can read
            // both dims immediately. Use the layer base's ResolveFromShape if
            // available; otherwise let the layer stay lazy.
            if (dense is NeuralNetworks.Layers.LayerBase<T> lb)
            {
                try { lb.ResolveFromShape(new[] { 64 }); }
                catch { /* lazy-resolve failure is non-fatal — wrapper may still cope */ }
            }
            return dense;
        }

        // ILayer<T> or LayerBase<T> (single layer).
        bool isLayerBase = pType == typeof(NeuralNetworks.Layers.LayerBase<T>);
        bool isILayer = pType.IsGenericType && pType.GetGenericTypeDefinition() == typeof(ILayer<>) && pType.GetGenericArguments()[0] == typeof(T);
        if (isLayerBase || isILayer)
        {
            placeholder = CreatePlaceholderDense();
            // Surface the placeholder usage at Trace level so callers can
            // detect that the reconstructed adapter doesn't have its
            // original wrapped layer. Issue #1235 follow-up will replace
            // this with proper ILayerSerializationExtras round-trip.
            System.Diagnostics.Trace.TraceWarning(
                $"DeserializationHelper.TryCreatePlaceholderInnerLayer: " +
                $"injected DenseLayer<T> placeholder for ctor parameter of type {pType.Name}; " +
                $"reconstructed layer is NOT semantically equivalent to the original (wrapped " +
                $"layer parameters were not round-tripped). Tracked under issue #1235.");
            return placeholder is not null;
        }

        // ILayer<T>[] (array of layers).
        if (pType.IsArray && pType.GetElementType() is Type elem
            && elem.IsGenericType && elem.GetGenericTypeDefinition() == typeof(ILayer<>)
            && elem.GetGenericArguments()[0] == typeof(T))
        {
            var instance = CreatePlaceholderDense();
            var arr = Array.CreateInstance(elem, 1);
            arr.SetValue(instance, 0);
            placeholder = arr;
            return true;
        }

        // List<ILayer<T>> / IEnumerable<ILayer<T>> / IList<ILayer<T>>.
        if (pType.IsGenericType)
        {
            var def = pType.GetGenericTypeDefinition();
            var ga = pType.GetGenericArguments();
            if (ga.Length == 1 && ga[0].IsGenericType
                && ga[0].GetGenericTypeDefinition() == typeof(ILayer<>)
                && ga[0].GetGenericArguments()[0] == typeof(T))
            {
                if (def == typeof(System.Collections.Generic.List<>)
                    || def == typeof(System.Collections.Generic.IList<>)
                    || def == typeof(System.Collections.Generic.IEnumerable<>)
                    || def == typeof(System.Collections.Generic.IReadOnlyList<>)
                    || def == typeof(System.Collections.Generic.IReadOnlyCollection<>)
                    || def == typeof(System.Collections.Generic.ICollection<>))
                {
                    var listType = typeof(System.Collections.Generic.List<>).MakeGenericType(ga[0]);
                    var list = Activator.CreateInstance(listType);
                    var addMethod = listType.GetMethod("Add");
                    var instance = CreatePlaceholderDense();
                    addMethod?.Invoke(list, new[] { instance });
                    placeholder = list;
                    return true;
                }
            }
        }

        return false;
    }

    /// <summary>Sensible-default lookup for double-typed ML hyperparameters
    /// — analog of <see cref="TryDefaultMlIntHyperparameter"/>.</summary>
    private static double? TryDefaultMlDoubleHyperparameter(string pNameLower)
    {
        if (pNameLower.Contains("epsilon") || pNameLower == "eps") return 1e-5;
        if (pNameLower.Contains("dropout") || pNameLower.Contains("dropoutrate")) return 0.0;
        if (pNameLower == "alpha" || pNameLower.Contains("weightalpha") || pNameLower.Contains("activationalpha")) return 1.0;
        if (pNameLower.Contains("momentum") || pNameLower.Contains("decay")) return 0.9;
        if (pNameLower.Contains("threshold") || pNameLower.Contains("anomalythreshold")) return 0.5;
        if (pNameLower.Contains("smoothingfactor")) return 0.1;
        if (pNameLower.Contains("temperature")) return 1.0;
        if (pNameLower.Contains("theta")) return 10000.0;
        if (pNameLower.Contains("learningrateratio")) return 1.0;
        if (pNameLower.Contains("rankinitscale") || pNameLower == "scale") return 1.0;
        if (pNameLower.Contains("sparsethreshold") || pNameLower.Contains("deltascaling")) return 0.0;
        // [0,1)-range hyperparameters: pruning thresholds, EMA factors,
        // momentum factors. AdaLoRA / HRA / DeltaLoRA validate these.
        if (pNameLower.Contains("sparsityratio") || pNameLower.Contains("rankpruningthreshold")
            || pNameLower.Contains("momentumfactor") || pNameLower.Contains("importancescoreema")
            || pNameLower.Contains("importanceema")) return 0.5;
        if (pNameLower.Contains("searchradius") || pNameLower.Contains("radii")) return 1.0;
        if (pNameLower.Contains("bnmomentum")) return 0.99;
        if (pNameLower.Contains("density")) return 0.1;
        if (pNameLower.Contains("regularization")) return 0.0;
        if (pNameLower.Contains("spectralradius")) return 0.9;
        return null;
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
    /// Reflection-driven constructor matcher used as the universal fallback
    /// when no dedicated branch exists for a layer. Walks every public
    /// constructor, classifies how each parameter would be resolved from
    /// (inputShape, outputShape, additionalParams, default values),
    /// computes a per-ctor score weighted toward exact-metadata matches,
    /// and invokes the highest-scoring constructor whose parameters can
    /// all be filled. Returns null if no constructor can be filled —
    /// caller falls back to the legacy (int[]) path or throws.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>Scoring (#1239):</b> previously this matcher ordered constructors
    /// by descending arity and picked the first one that fully resolved.
    /// That heuristic let a broad overload accepting our heuristic /
    /// defaulted arguments beat a narrower overload whose parameters
    /// would have been an exact metadata match — purely because it had
    /// more parameters. The new score ranks candidates by:
    /// </para>
    /// <list type="number">
    ///   <item><b>Metadata matches × 1000</b> — the parameter's name was
    ///     a direct hit in <c>additionalParams</c>. Highest weight because
    ///     the user explicitly persisted that exact value at serialize time.</item>
    ///   <item><b>Shape-derived matches × 100</b> — the parameter's name
    ///     pattern (<c>input*</c> / <c>output*</c> / <c>sequenceLength</c> /
    ///     <c>modelDimension</c> / etc.) let us derive its value from
    ///     <c>inputShape</c> or <c>outputShape</c>. Less authoritative
    ///     than metadata but still informed by the layer's runtime shape.</item>
    ///   <item><b>Default fallback × 0</b> — framework default value, ML-
    ///     domain hyperparameter default, or hardcoded safe fallback.
    ///     These contribute nothing to the score so a ctor with all-defaults
    ///     never beats a ctor with even a single metadata hit.</item>
    ///   <item><b>Arity tie-break</b> — parameter count adds 1 per param,
    ///     so when two ctors are otherwise tied the longer one wins
    ///     (preserves the prior heuristic for layers without metadata).</item>
    /// </list>
    /// <para>
    /// Constructors are scored without invoking, sorted by descending score,
    /// then attempted in order. First successful invoke wins. Failed invokes
    /// are logged via <c>Trace.TraceWarning</c> and don't count against the
    /// score (any ctor whose resolved args don't satisfy a runtime
    /// precondition gets skipped, and the next-best score is tried).
    /// </para>
    /// </remarks>
    /// <remarks>
    /// <para>
    /// Naming conventions used to map parameters from shape arrays:
    /// </para>
    /// <list type="bullet">
    ///   <item><c>int</c> parameters whose name contains "input" / "feature" / "in" / "size" / "dim" / "channel" / "vocab" / "embedding" → first try <c>additionalParams</c>; fall back to <c>inputShape[0]</c> or <c>inputShape[^1]</c>.</item>
    ///   <item><c>int</c> parameters whose name contains "output" / "out" → first try <c>additionalParams</c>; fall back to <c>outputShape[0]</c> or <c>outputShape[^1]</c>.</item>
    ///   <item><c>int[]</c> parameters named like inputShape / outputShape → use the corresponding array directly; otherwise look up in <c>additionalParams</c>.</item>
    ///   <item><c>bool</c> / <c>double</c> / <c>float</c> / <c>string</c> / <c>enum</c> — look up in <c>additionalParams</c> by parameter-name (capitalised first letter); fall back to default value if available.</item>
    ///   <item><c>IActivationFunction&lt;T&gt;</c> / <c>IVectorActivationFunction&lt;T&gt;</c> — restored via <see cref="TryRestoreActivation{T}"/> (already used by the explicit branches).</item>
    ///   <item>Other reference types — default value if available, otherwise <c>null</c>.</item>
    /// </list>
    /// </remarks>
    private static object? TryConstructByMatchingMetadata<T>(
        Type type,
        int[] inputShape,
        int[] outputShape,
        Dictionary<string, object>? additionalParams,
        string layerType)
    {
        var ctors = type.GetConstructors();

        // Score each ctor without invoking. We accumulate a candidate list
        // of (score, ctor, args) for every constructor whose parameters
        // are all resolvable, then sort by descending score and try
        // invokes in order. Pre-#1239 this loop just returned the first
        // fully-resolvable ctor in descending-arity order, which let
        // a 6-param all-defaults ctor beat a 4-param all-metadata-matched
        // ctor purely on arity.
        var candidates = new List<(int score, ConstructorInfo ctor, object?[] args, int arity, string sig)>();

        foreach (var ctor in ctors)
        {
            var parameters = ctor.GetParameters();
            var args = new object?[parameters.Length];
            int metadataMatches = 0;
            int shapeMatches = 0;
            bool allResolved = true;

            for (int pi = 0; pi < parameters.Length; pi++)
            {
                var p = parameters[pi];
                var pType = p.ParameterType;
                string pName = p.Name ?? string.Empty;
                string pNameLower = pName.ToLowerInvariant();
                string capName = pName.Length == 0
                    ? string.Empty
                    : char.ToUpperInvariant(pName[0]) + pName.Substring(1);

                // 1. int-array parameters
                if (pType == typeof(int[]))
                {
                    var arr = TryGetIntArray(additionalParams, capName);
                    if (arr is not null) { args[pi] = arr; metadataMatches++; continue; }
                    if (pNameLower.Contains("input") && pNameLower.Contains("shape")) { args[pi] = inputShape; shapeMatches++; continue; }
                    if (pNameLower.Contains("output") && pNameLower.Contains("shape")) { args[pi] = outputShape; shapeMatches++; continue; }
                    if (pNameLower.Contains("input")) { args[pi] = inputShape; shapeMatches++; continue; }
                    if (pNameLower.Contains("output")) { args[pi] = outputShape; shapeMatches++; continue; }
                    if (p.HasDefaultValue) { args[pi] = p.DefaultValue; continue; }
                    // Safe fallback: a single-element array. Used by params like
                    // padding (PaddingLayer), cropTop/Bottom/Left/Right
                    // (CroppingLayer), spatialDimensions (FourierLayer),
                    // patchSizes (KairosMultiSizePatchLayer), activeRanks
                    // (DyLoRAAdapter), mlpDimensions (SetAbstractionLayer).
                    // Real Clone() always supplies these via metadata; this
                    // path only fires on metadata-less reconstruction.
                    args[pi] = new int[] { 1 };
                    continue;
                }

                // 2. int parameters
                if (pType == typeof(int))
                {
                    var v = TryGetInt(additionalParams, capName);
                    if (v.HasValue) { args[pi] = v.Value; metadataMatches++; continue; }
                    // Common transformer naming
                    if (pNameLower == "sequencelength" && inputShape.Length > 0) { args[pi] = inputShape[0]; shapeMatches++; continue; }
                    if (pNameLower == "modeldimension" && inputShape.Length > 1) { args[pi] = inputShape[1]; shapeMatches++; continue; }
                    bool inputish = pNameLower.Contains("input") || pNameLower.Contains("feature") || pNameLower.Contains("vocab")
                        || pNameLower.Contains("embedding") || pNameLower == "size" || pNameLower == "indim" || pNameLower == "infeatures"
                        || pNameLower.Contains("inputchannel") || pNameLower.Contains("inchannel");
                    bool outputish = pNameLower.Contains("output") || pNameLower == "outdim" || pNameLower == "outfeatures"
                        || pNameLower.Contains("outputchannel") || pNameLower.Contains("outchannel") || pNameLower == "numclass";
                    if (inputish && inputShape.Length > 0) { args[pi] = inputShape[^1]; shapeMatches++; continue; }
                    if (outputish && outputShape.Length > 0) { args[pi] = outputShape[^1]; shapeMatches++; continue; }
                    // ML-domain defaults take priority over any compiler-supplied
                    // default value, because divisibility constraints across
                    // hyperparameters (channels divisible by numHeads, etc.)
                    // commonly invalidate the constructor's per-parameter
                    // defaults when only one of the two is overridden. Defaults
                    // here are chosen to satisfy common cross-parameter
                    // constraints (channels=64 divides 1/2/4/8/16/32/64 head
                    // counts; embeddingDim=64 likewise).
                    var hpDefault = TryDefaultMlIntHyperparameter(pNameLower);
                    if (hpDefault.HasValue) { args[pi] = hpDefault.Value; continue; }
                    if (p.HasDefaultValue) { args[pi] = p.DefaultValue; continue; }
                    allResolved = false; break;
                }

                // 3. bool / double / float / string parameters
                if (pType == typeof(bool))
                {
                    var v = TryGetBool(additionalParams, capName);
                    if (v.HasValue) { args[pi] = v.Value; metadataMatches++; continue; }
                    if (p.HasDefaultValue) { args[pi] = p.DefaultValue; continue; }
                    // Safe default: false. Most ML bool hyperparameters
                    // (freezeBaseLayer, useBias, useFlashAttention, causal,
                    // useDoubleQuantization, etc.) default to false in their
                    // declared constructors, and false is the conservative
                    // round-trip choice when metadata is absent.
                    args[pi] = false;
                    continue;
                }
                if (pType == typeof(double))
                {
                    var v = TryGetDouble(additionalParams, capName);
                    if (v.HasValue) { args[pi] = v.Value; metadataMatches++; continue; }
                    var hpDefault = TryDefaultMlDoubleHyperparameter(pNameLower);
                    if (hpDefault.HasValue) { args[pi] = hpDefault.Value; continue; }
                    if (p.HasDefaultValue) { args[pi] = p.DefaultValue; continue; }
                    args[pi] = 0.0;  // safe metadata-less fallback
                    continue;
                }
                if (pType == typeof(float))
                {
                    var v = TryGetDouble(additionalParams, capName);
                    if (v.HasValue) { args[pi] = (float)v.Value; metadataMatches++; continue; }
                    var hpDefault = TryDefaultMlDoubleHyperparameter(pNameLower);
                    if (hpDefault.HasValue) { args[pi] = (float)hpDefault.Value; continue; }
                    if (p.HasDefaultValue) { args[pi] = p.DefaultValue; continue; }
                    args[pi] = 0.0f;
                    continue;
                }
                if (pType == typeof(string))
                {
                    if (additionalParams != null && additionalParams.TryGetValue(capName, out var sv) && sv is string s) { args[pi] = s; metadataMatches++; continue; }
                    if (p.HasDefaultValue) { args[pi] = p.DefaultValue; continue; }
                    args[pi] = string.Empty;
                    continue;
                }

                // 4. enum parameters — look up by name, parse string (GetMetadata persists enum.ToString())
                if (pType.IsEnum)
                {
                    // .NET Framework 4.7.1 doesn't expose the non-generic
                    // Enum.TryParse(Type, ...) overload, so we route through
                    // Enum.Parse with try/catch to keep both targets happy.
                    // Already-typed enum values (rare but possible) pass
                    // through directly; strings get parsed.
                    if (additionalParams != null && additionalParams.TryGetValue(capName, out var ev) && ev is not null)
                    {
                        if (ev.GetType() == pType)
                        {
                            args[pi] = ev; metadataMatches++; continue;
                        }
                        if (ev is string es)
                        {
                            try
                            {
                                args[pi] = Enum.Parse(pType, es);
                                metadataMatches++;
                                continue;
                            }
                            catch (ArgumentException)
                            {
                                // Fall through to default-value / unresolved
                                // path so the matcher tries the next ctor
                                // overload instead of crashing here.
                            }
                        }
                    }
                    if (p.HasDefaultValue) { args[pi] = p.DefaultValue; continue; }
                    allResolved = false; break;
                }

                // 5. activation functions — reuse the existing restorer
                bool isScalarAct = pType.IsGenericType && pType.GetGenericTypeDefinition() == typeof(IActivationFunction<>);
                bool isVectorAct = pType.IsGenericType && pType.GetGenericTypeDefinition() == typeof(IVectorActivationFunction<>);
                if (isScalarAct || isVectorAct)
                {
                    var act = TryRestoreActivation<T>(additionalParams);
                    if (act != null && pType.IsAssignableFrom(act.GetType())) { args[pi] = act; metadataMatches++; continue; }
                    if (p.HasDefaultValue) { args[pi] = p.DefaultValue; continue; }
                    if (!pType.IsValueType) { args[pi] = null; continue; }  // most activation params are nullable interface refs
                    allResolved = false; break;
                }

                // 6. IEngine — use the ambient process engine.
                if (pType == typeof(AiDotNet.Tensors.Engines.IEngine))
                {
                    args[pi] = AiDotNet.Tensors.Engines.AiDotNetEngine.Current;
                    continue;
                }

                // 7. int[][] (jagged arrays for AddLayer, MultiplyLayer,
                //    ConcatenateLayer "inputShapes") — try metadata, fall back
                //    to a two-element jagged wrap of inputShape (most binary
                //    composite layers expect at least two input shapes).
                if (pType == typeof(int[][]))
                {
                    if (additionalParams != null
                        && additionalParams.TryGetValue(capName, out var jv)
                        && jv is int[][] jarr)
                    {
                        args[pi] = jarr;
                        metadataMatches++;
                        continue;
                    }
                    args[pi] = new int[][] { inputShape, inputShape };
                    continue;
                }

                // 8. ILayer<T> / LayerBase<T> / single-base-layer references —
                //    used by LoRA/PEFT adapters, BidirectionalLayer,
                //    SpectralNormalizationLayer, TimeDistributedLayer.
                //    Per-adapter round-trip via LoRAAdapterBase.GetMetadata
                //    (issue #1239): InnerLayerTypeName / InnerLayerInputShape
                //    / InnerLayerOutputShape lets the deser path reconstruct
                //    the actual wrapped layer instead of a placeholder. Falls
                //    back to the DenseLayer<T> placeholder for legacy
                //    networks serialized before that metadata existed —
                //    those reconstructions are NOT semantically equivalent
                //    (only the wrapper's structural metadata round-trips).
                bool isSingleLayerParam = pType == typeof(NeuralNetworks.Layers.LayerBase<T>)
                    || (pType.IsGenericType
                        && pType.GetGenericTypeDefinition() == typeof(ILayer<>)
                        && pType.GetGenericArguments()[0] == typeof(T));
                if (isSingleLayerParam)
                {
                    var fromMeta = TryConstructInnerLayerFromMetadata<T>(additionalParams);
                    if (fromMeta is not null)
                    {
                        args[pi] = fromMeta;
                        metadataMatches++;
                        continue;
                    }
                }
                if (TryCreatePlaceholderInnerLayer<T>(pType, out var placeholderInner))
                {
                    args[pi] = placeholderInner;
                    continue;
                }

                // 9. other reference types — default value or null. Reject value
                // types we don't know how to fill (otherwise we'd hand the ctor
                // a default(T) that probably violates a precondition).
                if (p.HasDefaultValue) { args[pi] = p.DefaultValue; continue; }
                if (!pType.IsValueType) { args[pi] = null; continue; }
                allResolved = false; break;
            }

            if (allResolved)
            {
                // Score formula (#1239 — see method docstring): metadata
                // hits dominate (×1000), shape-derived hits secondary
                // (×100), arity tie-breaks (+1 per param). Defaults
                // contribute 0 so a ctor with all-defaults can never beat
                // a ctor with even a single metadata hit.
                int score = (metadataMatches * 1000) + (shapeMatches * 100) + parameters.Length;
                // Capture parameter-type FullName signature as the final
                // tie-break key. List<T>.Sort is not guaranteed stable, so
                // when (score, arity) collide (e.g., two overloads of the
                // same arity differing only by IActivationFunction<T> vs
                // IInitializationStrategy<T>) we need a deterministic key
                // to make ctor selection reproducible across runs. Sorting
                // signatures lexicographically gives us cross-process /
                // cross-machine determinism that depends only on the
                // type's metadata, not on reflection iteration order.
                // Use AssemblyQualifiedName when available so two types
                // sharing a simple name in different assemblies (e.g. an
                // internal `MyLayer` in two assemblies linked together)
                // can't collide on the tie-break and silently switch
                // ctor selection. FullName drops the assembly identity
                // and Name additionally drops the namespace — both
                // weaker. ToString() is the deterministic fallback for
                // open generic parameters / dynamic types where AQN
                // returns null.
                string sig = string.Join(",",
                    parameters.Select(p => p.ParameterType.AssemblyQualifiedName ?? p.ParameterType.ToString()));
                candidates.Add((score, ctor, args, parameters.Length, sig));
            }
        }

        // Sort by descending score, then by parameter-type signature
        // ascending. Score already incorporates arity (+1 per parameter
        // in the score formula), so a separate arity tie-break would be
        // redundant — when scores tie, arities tie too. The signature
        // key turns an unstable tie-break into a deterministic one for
        // the case of two same-arity overloads differing only by
        // parameter type (e.g. IActivationFunction<T> vs
        // IInitializationStrategy<T>) — see the candidates.Add call
        // above for the rationale.
        candidates.Sort((a, b) =>
        {
            int byScore = b.score.CompareTo(a.score);
            if (byScore != 0) return byScore;
            return string.CompareOrdinal(a.sig, b.sig);
        });

        foreach (var (_, ctor, args, _, _) in candidates)
        {
            try { return ctor.Invoke(args); }
            catch (Exception ex)
            {
                // Best-effort matcher: any failure means this constructor
                // didn't accept our resolved args — try the next-best score.
                // Trace the rejection so a missing fallback default in
                // TryDefaultMlIntHyperparameter / TryDefaultMlDouble-
                // Hyperparameter is debuggable when reconstruction fails
                // downstream with no breadcrumb back to here.
                System.Diagnostics.Trace.TraceWarning(
                    $"DeserializationHelper.TryConstructByMatchingMetadata: " +
                    $"ctor {ctor} for {type.Name} rejected our resolved args: " +
                    $"{(ex is TargetInvocationException tie ? tie.InnerException?.Message : ex.Message)}");
            }
        }

        return null;
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
