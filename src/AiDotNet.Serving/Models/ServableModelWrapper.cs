using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Models;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Validation;

namespace AiDotNet.Serving.Models;

/// <summary>
/// Generic wrapper that adapts various AiDotNet models to the IServableModel interface.
/// This allows any model with a Predict method to be served via the REST API.
/// </summary>
/// <typeparam name="T">The numeric type used by the model</typeparam>
public class ServableModelWrapper<T> : IServableModel<T>, IServableModelInferenceOptions, IServableGenerativeModel<T>
{
    private readonly Func<Vector<T>, Vector<T>> _predictFunc;
    private readonly Func<Matrix<T>, Matrix<T>>? _predictBatchFunc;

    // Raw token-level forward (tokens -> logits) for autoregressive generation. Set only when
    // the wrapper is constructed from a tensor-to-tensor model (e.g. a transformer language
    // model); null for vector/matrix prediction models, which cannot generate text.
    private readonly Func<Tensor<T>, Tensor<T>>? _tensorForward;

    // Incremental KV-cached generation (#99): an InferenceOptimizer-optimized clone (paged cached
    // attention) + its shared PagedKVCache. Built only for a generative NeuralNetworkBase model
    // with optimizable attention; null => incremental unsupported (callers use stateless Forward).
    private readonly AiDotNet.NeuralNetworks.NeuralNetworkBase<T>? _incrementalModel;
    private readonly AiDotNet.Inference.PagedAttention.PagedKVCache<T>? _incrementalCache;
    private long _nextSequenceId;

    // RadixAttention-style prefix sharing (#99 Stage 2 integration): maps a prompt-prefix key to a
    // base KV-cache sequence whose cache holds exactly that prefix. New requests fork from the
    // longest registered strict-prefix (copy-on-write), reusing the prefix's KV. LRU-capped; evicted
    // bases are freed (existing forks keep shared blocks alive via block ref-counting).
    private const int PrefixRegistryCapacity = 64;
    private readonly object _prefixLock = new();
    private readonly System.Collections.Generic.Dictionary<string, long> _prefixRegistry = new();
    private readonly System.Collections.Generic.LinkedList<string> _prefixLru = new();

    private readonly string _modelName;
    private readonly int _inputDimension;
    private readonly int _outputDimension;
    private readonly int[] _inputShape;
    private readonly int[] _outputShape;
    private readonly DynamicShapeInfo _dynamicShapeInfo;
    private readonly bool _enableBatching;
    private readonly bool _enableSpeculativeDecoding;

    /// <summary>
    /// Initializes a new instance of the ServableModelWrapper with custom prediction functions.
    /// </summary>
    /// <param name="modelName">The name of the model</param>
    /// <param name="inputDimension">The expected number of input features</param>
    /// <param name="outputDimension">The number of output dimensions</param>
    /// <param name="predictFunc">Function to perform single prediction</param>
    /// <param name="predictBatchFunc">Optional function to perform batch prediction. If not provided, batch prediction will use multiple single predictions.</param>
    /// <param name="enableBatching">Whether serving-side batching is enabled for this model.</param>
    /// <param name="enableSpeculativeDecoding">Whether speculative decoding is enabled for this model.</param>
    /// <param name="inputShape">Optional full input shape array. If null, derived from inputDimension.</param>
    /// <param name="outputShape">Optional full output shape array. If null, derived from outputDimension.</param>
    /// <param name="dynamicShapeInfo">Optional dynamic shape information. If null, all dimensions are fixed.</param>
    /// <param name="generationForward">Optional token-level forward (token-IDs tensor -> logits) enabling
    /// autoregressive text generation. When null, the model does not support generation.</param>
    public ServableModelWrapper(
        string modelName,
        int inputDimension,
        int outputDimension,
        Func<Vector<T>, Vector<T>> predictFunc,
        Func<Matrix<T>, Matrix<T>>? predictBatchFunc = null,
        bool enableBatching = true,
        bool enableSpeculativeDecoding = false,
        int[]? inputShape = null,
        int[]? outputShape = null,
        DynamicShapeInfo? dynamicShapeInfo = null,
        Func<Tensor<T>, Tensor<T>>? generationForward = null)
    {
        Guard.NotNullOrWhiteSpace(modelName);
        _modelName = modelName;
        _inputDimension = inputDimension;
        _outputDimension = outputDimension;
        _inputShape = inputShape ?? new[] { inputDimension };
        _outputShape = outputShape ?? new[] { outputDimension };
        _dynamicShapeInfo = dynamicShapeInfo ?? DynamicShapeInfo.None;
        Guard.NotNull(predictFunc);
        _predictFunc = predictFunc;
        _predictBatchFunc = predictBatchFunc;
        _enableBatching = enableBatching;
        _enableSpeculativeDecoding = enableSpeculativeDecoding;
        _tensorForward = generationForward;
    }

    /// <summary>
    /// Initializes a new instance of the ServableModelWrapper from an IRegression model.
    /// </summary>
    /// <param name="modelName">The name of the model</param>
    /// <param name="regressionModel">The regression model to wrap</param>
    /// <param name="inputDimension">The expected number of input features</param>
    /// <param name="enableBatching">Whether serving-side batching is enabled for this model.</param>
    /// <param name="enableSpeculativeDecoding">Whether speculative decoding is enabled for this model.</param>
    public ServableModelWrapper(
        string modelName,
        IRegression<T> regressionModel,
        int inputDimension,
        bool enableBatching = true,
        bool enableSpeculativeDecoding = false)
    {
        Guard.NotNullOrWhiteSpace(modelName);
        _modelName = modelName;
        _inputDimension = inputDimension;
        _outputDimension = 1; // Regression models typically output a single value
        _inputShape = new[] { inputDimension };
        _outputShape = new[] { 1 };
        _dynamicShapeInfo = DynamicShapeInfo.None;
        _enableBatching = enableBatching;
        _enableSpeculativeDecoding = enableSpeculativeDecoding;

        if (regressionModel == null)
        {
            throw new ArgumentNullException(nameof(regressionModel));
        }

        _predictFunc = input =>
        {
            // Regression predict typically takes Matrix and returns Vector
            var inputMatrix = new Matrix<T>(1, input.Length);
            for (int i = 0; i < input.Length; i++)
            {
                inputMatrix[0, i] = input[i];
            }

            var predictions = regressionModel.Predict(inputMatrix);
            return new Vector<T>(new[] { predictions[0] });
        };

        _predictBatchFunc = inputs =>
        {
            var predictions = regressionModel.Predict(inputs);
            var result = new Matrix<T>(inputs.Rows, 1);
            for (int i = 0; i < predictions.Length; i++)
            {
                result[i, 0] = predictions[i];
            }
            return result;
        };
    }

    /// <summary>
    /// Initializes a new instance of the ServableModelWrapper from a Matrix-to-Vector model
    /// (regression, classification, clustering, survival, causal, online learning, time series).
    /// </summary>
    /// <param name="modelName">The name of the model.</param>
    /// <param name="model">The model that accepts Matrix input and returns Vector output.</param>
    /// <param name="enableBatching">Whether serving-side batching is enabled.</param>
    /// <param name="enableSpeculativeDecoding">Whether speculative decoding is enabled.</param>
    public ServableModelWrapper(
        string modelName,
        IFullModel<T, Matrix<T>, Vector<T>> model,
        bool enableBatching = true,
        bool enableSpeculativeDecoding = false)
    {
        Guard.NotNullOrWhiteSpace(modelName);
        Guard.NotNull(model);

        _modelName = modelName;
        _enableBatching = enableBatching;
        _enableSpeculativeDecoding = enableSpeculativeDecoding;

        // Extract shape from IModelShape if available
        if (model is IModelShape shapeModel)
        {
            var inShape = shapeModel.GetInputShape();
            var outShape = shapeModel.GetOutputShape();
            _inputShape = inShape;
            _outputShape = outShape;
            _inputDimension = inShape.Length > 0 ? inShape[inShape.Length - 1] : 0;
            _outputDimension = outShape.Length > 0 ? outShape[outShape.Length - 1] : 0;
            _dynamicShapeInfo = shapeModel.GetDynamicShapeInfo();
        }
        else
        {
            _inputDimension = 0;
            _outputDimension = 0;
            _inputShape = Array.Empty<int>();
            _outputShape = Array.Empty<int>();
            _dynamicShapeInfo = DynamicShapeInfo.None;
        }

        _predictFunc = input =>
        {
            // Wrap single Vector into a 1-row Matrix for Matrix→Vector models
            var inputMatrix = new Matrix<T>(1, input.Length);
            for (int i = 0; i < input.Length; i++)
            {
                inputMatrix[0, i] = input[i];
            }

            return model.Predict(inputMatrix);
        };

        _predictBatchFunc = inputs =>
        {
            // Matrix→Vector models: Predict(Matrix) returns a single Vector for the whole batch.
            // For single-row input this is the output directly; for multi-row, use row-by-row.
            if (inputs.Rows == 1)
            {
                var predictions = model.Predict(inputs);
                var result = new Matrix<T>(1, predictions.Length);
                for (int j = 0; j < predictions.Length; j++)
                {
                    result[0, j] = predictions[j];
                }
                return result;
            }

            // Multi-row: process each row individually to get per-row Vector outputs
            int outputWidth = 0;
            var rowResults = new Vector<T>[inputs.Rows];
            for (int i = 0; i < inputs.Rows; i++)
            {
                var row = inputs.GetRow(i);
                var rowMatrix = new Matrix<T>(1, row.Length);
                for (int j = 0; j < row.Length; j++)
                {
                    rowMatrix[0, j] = row[j];
                }

                rowResults[i] = model.Predict(rowMatrix);
                if (i == 0)
                {
                    outputWidth = rowResults[i].Length;
                }
                else if (rowResults[i].Length != outputWidth)
                {
                    throw new InvalidOperationException(
                        $"Batch row {i} produced output width {rowResults[i].Length} " +
                        $"but row 0 produced {outputWidth}. All rows must have consistent output width.");
                }
            }

            var batchResult = new Matrix<T>(inputs.Rows, outputWidth);
            for (int i = 0; i < inputs.Rows; i++)
            {
                for (int j = 0; j < outputWidth; j++)
                {
                    batchResult[i, j] = rowResults[i][j];
                }
            }
            return batchResult;
        };
    }

    /// <summary>
    /// Initializes a new instance of the ServableModelWrapper from a Tensor-to-Tensor model
    /// (neural networks, diffusion models).
    /// </summary>
    /// <param name="modelName">The name of the model.</param>
    /// <param name="model">The model that accepts Tensor input and returns Tensor output.</param>
    /// <param name="inputShape">The shape to reshape flat input vectors into tensors.</param>
    /// <param name="enableBatching">Whether serving-side batching is enabled.</param>
    /// <param name="enableSpeculativeDecoding">Whether speculative decoding is enabled.</param>
    /// <param name="generationForward">Optional token-level forward (token-IDs tensor -> logits) that
    /// enables autoregressive text generation. Pass <c>model.Predict</c> ONLY for models with
    /// token-to-logits semantics (e.g. a transformer LM). When null (default), the model does not
    /// advertise generation support — not every Tensor-to-Tensor model (e.g. a diffusion model) has
    /// valid next-token-logits semantics, so generation must be opt-in rather than assumed.</param>
    public ServableModelWrapper(
        string modelName,
        IFullModel<T, Tensor<T>, Tensor<T>> model,
        int[] inputShape,
        bool enableBatching = true,
        bool enableSpeculativeDecoding = false,
        Func<Tensor<T>, Tensor<T>>? generationForward = null,
        bool quantizeIncrementalWeights = false)
    {
        Guard.NotNullOrWhiteSpace(modelName);
        Guard.NotNull(model);

        _modelName = modelName;
        _enableBatching = enableBatching;
        _enableSpeculativeDecoding = enableSpeculativeDecoding;

        // Generation is explicit opt-in: only a caller that knows this model maps a [1, seqLen] token
        // tensor to per-position logits supplies generationForward. Wiring model.Predict
        // unconditionally would make SupportsGeneration true for non-generative tensor models and route
        // them into the token-generation path with meaningless logits.
        _tensorForward = generationForward;

        // Use provided inputShape and extract output shape from IModelShape if available
        _inputShape = inputShape ?? Array.Empty<int>();
        _inputDimension = 1;
        foreach (int dim in _inputShape)
        {
            if (dim > 0)
            {
                _inputDimension *= dim;
            }
        }

        if (model is IModelShape shapeModel)
        {
            _outputShape = shapeModel.GetOutputShape();
            _dynamicShapeInfo = shapeModel.GetDynamicShapeInfo();
        }
        else
        {
            _outputShape = Array.Empty<int>();
            _dynamicShapeInfo = DynamicShapeInfo.None;
        }

        _outputDimension = 1;
        foreach (int dim in _outputShape)
        {
            if (dim > 0)
            {
                _outputDimension *= dim;
            }
        }

        _predictFunc = input =>
        {
            // Validate input length matches expected flat dimension before reshape
            if (input.Length != _inputDimension)
            {
                throw new ArgumentException(
                    $"Input vector length {input.Length} does not match expected flat dimension {_inputDimension} " +
                    $"(shape: [{string.Join(", ", _inputShape)}]).",
                    nameof(input));
            }

            // Reshape flat Vector into Tensor using inputShape
            var tensor = new Tensor<T>(_inputShape, input);
            var result = model.Predict(tensor);

            // Flatten Tensor output back to Vector
            var outputVector = new Vector<T>(result.Length);
            for (int i = 0; i < result.Length; i++)
            {
                outputVector[i] = result[i];
            }
            return outputVector;
        };

        _predictBatchFunc = null; // Falls back to row-by-row single predictions

        // Build the incremental (KV-cached) generation path when this is an opt-in generative model
        // (generationForward supplied) backed by an optimizable NeuralNetworkBase. Best-effort: any
        // failure leaves incremental disabled and callers fall back to stateless Forward decoding.
        if (generationForward is not null && model is AiDotNet.NeuralNetworks.NeuralNetworkBase<T> neural)
        {
            try
            {
                var built = BuildIncrementalModel(neural, quantizeIncrementalWeights);
                _incrementalModel = built.Model;
                _incrementalCache = built.Cache;
            }
            catch (Exception ex)
            {
                AiDotNet.Helpers.InferenceDiagnostics.RecordException(
                    area: "Serving.ServableModelWrapper",
                    feature: "IncrementalGeneration",
                    ex: ex,
                    reason: "Failed to build incremental KV-cached model; falling back to stateless decode.");
                _incrementalModel = null;
                _incrementalCache = null;
            }
        }
    }

    /// <summary>
    /// Optimizes a generative model to a paged-KV inference form for incremental decode. Returns
    /// (null, null) when the model has no optimizable attention (incremental unsupported).
    /// </summary>
    private static (AiDotNet.NeuralNetworks.NeuralNetworkBase<T>? Model, AiDotNet.Inference.PagedAttention.PagedKVCache<T>? Cache)
        BuildIncrementalModel(AiDotNet.NeuralNetworks.NeuralNetworkBase<T> source, bool quantizeWeights)
    {
        var config = new AiDotNet.Configuration.InferenceOptimizationConfig
        {
            EnableKVCache = true,
            EnablePagedKVCache = true,
            EnableFlashAttention = false,
            EnableLayerFusion = false,
            AttentionMasking = AiDotNet.Configuration.AttentionMaskingMode.Causal,
            // int8 weight-only quantization shrinks resident weights so more sequences fit; the paged
            // decode stays correct under it (proven by PagedQuantizedDecodeTests).
            InferenceQuantization = quantizeWeights
                ? AiDotNet.Configuration.InferenceQuantizationMode.WeightOnlyInt8
                : AiDotNet.Configuration.InferenceQuantizationMode.None
        };

        var optimizer = new AiDotNet.Inference.InferenceOptimizer<T>(config);
        var (optimized, applied) = optimizer.OptimizeForInference(source, cloneModel: true);
        var cache = optimizer.PagedKVCache;
        if (!applied || cache is null)
        {
            return (null, null);
        }

        // Eval mode + warm up lazy caches (kernel weight cache, etc.) single-threaded before
        // concurrent sessions touch the shared model — keeps the per-call forward race-free.
        optimized.SetTrainingMode(false);
        long warmupId = long.MaxValue;
        if (cache.AllocateSequence(warmupId, 0))
        {
            try
            {
                var probe = new Tensor<T>(new[] { 1, 1 });
                optimized.PredictWithContext(probe, new AiDotNet.Inference.InferenceForwardContext(warmupId, 0));
            }
            finally
            {
                cache.FreeSequence(warmupId);
            }
        }

        return (optimized, cache);
    }

    /// <summary>
    /// Initializes a new instance of the ServableModelWrapper from a Vector-to-Vector model
    /// (reinforcement learning agents).
    /// </summary>
    /// <param name="modelName">The name of the model.</param>
    /// <param name="model">The model that accepts Vector input and returns Vector output.</param>
    /// <param name="enableBatching">Whether serving-side batching is enabled.</param>
    /// <param name="enableSpeculativeDecoding">Whether speculative decoding is enabled.</param>
    public ServableModelWrapper(
        string modelName,
        IFullModel<T, Vector<T>, Vector<T>> model,
        bool enableBatching = true,
        bool enableSpeculativeDecoding = false)
    {
        Guard.NotNullOrWhiteSpace(modelName);
        Guard.NotNull(model);

        _modelName = modelName;
        _enableBatching = enableBatching;
        _enableSpeculativeDecoding = enableSpeculativeDecoding;

        if (model is IModelShape shapeModel)
        {
            var inShape = shapeModel.GetInputShape();
            var outShape = shapeModel.GetOutputShape();
            _inputShape = inShape;
            _outputShape = outShape;
            _inputDimension = inShape.Length > 0 ? inShape[inShape.Length - 1] : 0;
            _outputDimension = outShape.Length > 0 ? outShape[outShape.Length - 1] : 0;
            _dynamicShapeInfo = shapeModel.GetDynamicShapeInfo();
        }
        else
        {
            _inputDimension = 0;
            _outputDimension = 0;
            _inputShape = Array.Empty<int>();
            _outputShape = Array.Empty<int>();
            _dynamicShapeInfo = DynamicShapeInfo.None;
        }

        // Direct pass-through: Vector→Vector models need no adaptation
        _predictFunc = input => model.Predict(input);
        _predictBatchFunc = null; // Falls back to row-by-row single predictions
    }

    /// <summary>
    /// Creates a ServableModelWrapper by automatically detecting the model's input/output type pattern
    /// and selecting the appropriate adapter.
    /// </summary>
    /// <param name="modelName">The name for the servable model.</param>
    /// <param name="model">The model instance (must implement IModelSerializer and one of the IFullModel variants).</param>
    /// <param name="enableBatching">Whether serving-side batching is enabled.</param>
    /// <param name="enableSpeculativeDecoding">Whether speculative decoding is enabled.</param>
    /// <returns>A ServableModelWrapper configured for the detected model type.</returns>
    /// <exception cref="InvalidOperationException">Thrown when the model does not implement a supported IFullModel variant.</exception>
    internal static ServableModelWrapper<T> FromModel(
        string modelName,
        IModelSerializer model,
        bool enableBatching = true,
        bool enableSpeculativeDecoding = false)
    {
        Guard.NotNullOrWhiteSpace(modelName);
        Guard.NotNull(model);

        // Check Vector→Vector first (RL agents) — most specific
        if (model is IFullModel<T, Vector<T>, Vector<T>> vectorModel)
        {
            return new ServableModelWrapper<T>(modelName, vectorModel, enableBatching, enableSpeculativeDecoding);
        }

        // Check Matrix→Vector (regression, classification, clustering, etc.)
        if (model is IFullModel<T, Matrix<T>, Vector<T>> matrixModel)
        {
            return new ServableModelWrapper<T>(modelName, matrixModel, enableBatching, enableSpeculativeDecoding);
        }

        // Check Tensor→Tensor (neural networks, diffusion)
        if (model is IFullModel<T, Tensor<T>, Tensor<T>> tensorModel)
        {
            // Get input shape from IModelShape
            int[] inputShape;
            if (model is IModelShape shapeModel)
            {
                inputShape = shapeModel.GetInputShape();
            }
            else
            {
                throw new InvalidOperationException(
                    $"Model '{modelName}' (type: {model.GetType().Name}) is a Tensor model but does not implement IModelShape. " +
                    "Tensor models must implement IModelShape to provide the input shape for serving.");
            }

            return new ServableModelWrapper<T>(modelName, tensorModel, inputShape, enableBatching, enableSpeculativeDecoding);
        }

        throw new InvalidOperationException(
            $"Model '{modelName}' (type: {model.GetType().Name}) does not implement a supported IFullModel variant. " +
            "Supported patterns: IFullModel<T, Vector<T>, Vector<T>>, IFullModel<T, Matrix<T>, Vector<T>>, " +
            "IFullModel<T, Tensor<T>, Tensor<T>>. Use the constructor overload with custom predict functions instead.");
    }

    /// <summary>
    /// Loads an AIMF model file and creates a ServableModelWrapper in one step.
    /// Combines ModelLoader.Load with FromModel for the common serving use case.
    /// </summary>
    /// <param name="filePath">The path to the AIMF model file.</param>
    /// <param name="modelName">The name for the servable model.</param>
    /// <param name="enableBatching">Whether serving-side batching is enabled.</param>
    /// <param name="enableSpeculativeDecoding">Whether speculative decoding is enabled.</param>
    /// <param name="licenseKey">Optional license key for encrypted AIMF models.</param>
    /// <param name="decryptionToken">Optional server-side decryption token.</param>
    /// <returns>A ServableModelWrapper ready for serving.</returns>
    internal static ServableModelWrapper<T> LoadServable(
        string filePath,
        string modelName,
        bool enableBatching = true,
        bool enableSpeculativeDecoding = false,
        string? licenseKey = null,
        byte[]? decryptionToken = null)
    {
        var model = ModelLoader.Load<T>(filePath, licenseKey, decryptionToken);
        return FromModel(modelName, model, enableBatching, enableSpeculativeDecoding);
    }

    /// <inheritdoc/>
    public string ModelName => _modelName;

    /// <inheritdoc/>
    public int InputDimension => _inputDimension;

    /// <inheritdoc/>
    public int OutputDimension => _outputDimension;

    /// <inheritdoc/>
    public int[] InputShape => (int[])_inputShape.Clone();

    /// <inheritdoc/>
    public int[] OutputShape => (int[])_outputShape.Clone();

    /// <inheritdoc/>
    public DynamicShapeInfo DynamicShapeInfo => _dynamicShapeInfo;

    bool IServableModelInferenceOptions.EnableBatching => _enableBatching;

    bool IServableModelInferenceOptions.EnableSpeculativeDecoding => _enableSpeculativeDecoding;

    /// <inheritdoc/>
    public bool SupportsGeneration => _tensorForward is not null;

    /// <inheritdoc/>
    public Tensor<T> Forward(Tensor<T> inputTokenIds)
    {
        Guard.NotNull(inputTokenIds);

        if (_tensorForward is null)
        {
            throw new NotSupportedException(
                $"Model '{_modelName}' does not support token-level generation. " +
                "Only tensor-to-tensor models (e.g. transformer language models) can generate text.");
        }

        return _tensorForward(inputTokenIds);
    }

    /// <inheritdoc/>
    public bool SupportsIncrementalGeneration => _incrementalModel is not null && _incrementalCache is not null;

    /// <inheritdoc/>
    public IGenerationSession<T> BeginGeneration()
    {
        if (_incrementalModel is null || _incrementalCache is null)
        {
            throw new NotSupportedException(
                $"Model '{_modelName}' does not support incremental generation.");
        }

        long sequenceId = AllocateSessionSequence(_incrementalCache);
        return new GenerationSession(this, _incrementalModel, _incrementalCache, sequenceId, cachedPromptTokens: 0);
    }

    /// <inheritdoc/>
    public IGenerationSession<T> BeginGeneration(System.Collections.Generic.IReadOnlyList<int> promptTokens)
    {
        Guard.NotNull(promptTokens);
        if (_incrementalModel is null || _incrementalCache is null)
        {
            throw new NotSupportedException(
                $"Model '{_modelName}' does not support incremental generation.");
        }

        long sequenceId = AllocateSessionSequence(_incrementalCache);
        int cachedPromptTokens = 0;

        // Reuse the longest registered prompt prefix that is a STRICT prefix of this prompt (so at
        // least the last token is still forwarded, producing the first next-token logits). Fork it
        // copy-on-write so the shared prefix KV is reused and only the suffix allocates new blocks.
        lock (_prefixLock)
        {
            for (int len = promptTokens.Count - 1; len >= 1; len--)
            {
                string key = PrefixKey(promptTokens, len);
                if (_prefixRegistry.TryGetValue(key, out long baseSeqId) &&
                    _incrementalCache.ForkSequence(baseSeqId, sequenceId))
                {
                    cachedPromptTokens = len;
                    TouchPrefix(key);
                    break;
                }
            }
        }

        return new GenerationSession(this, _incrementalModel, _incrementalCache, sequenceId, cachedPromptTokens);
    }

    /// <summary>
    /// Registers a base sequence (forked from <paramref name="sourceSequenceId"/>, which must hold
    /// exactly the prompt's KV) under the full prompt key, so later prompts that extend it can fork.
    /// </summary>
    private void RegisterPrefix(System.Collections.Generic.IReadOnlyList<int> promptTokens, long sourceSequenceId)
    {
        if (_incrementalCache is null || promptTokens.Count == 0)
        {
            return;
        }

        string key = PrefixKey(promptTokens, promptTokens.Count);
        lock (_prefixLock)
        {
            if (_prefixRegistry.ContainsKey(key))
            {
                TouchPrefix(key);
                return;
            }

            long baseSeqId = System.Threading.Interlocked.Increment(ref _nextSequenceId);
            if (!_incrementalCache.ForkSequence(sourceSequenceId, baseSeqId))
            {
                return; // best-effort: skip registration if the fork fails
            }

            _prefixRegistry[key] = baseSeqId;
            _prefixLru.AddLast(key);
            EvictPrefixesIfNeeded();
        }
    }

    // Caller must hold _prefixLock.
    private void TouchPrefix(string key)
    {
        _prefixLru.Remove(key);
        _prefixLru.AddLast(key);
    }

    // Caller must hold _prefixLock.
    private void EvictPrefixesIfNeeded()
    {
        while (_prefixRegistry.Count > PrefixRegistryCapacity && _prefixLru.First is not null)
        {
            string oldest = _prefixLru.First.Value;
            _prefixLru.RemoveFirst();
            if (_prefixRegistry.TryGetValue(oldest, out long evictedBase))
            {
                _prefixRegistry.Remove(oldest);
                _incrementalCache?.FreeSequence(evictedBase);
            }
        }
    }

    private static string PrefixKey(System.Collections.Generic.IReadOnlyList<int> tokens, int length)
    {
        var sb = new System.Text.StringBuilder(length * 4);
        for (int i = 0; i < length; i++)
        {
            sb.Append(tokens[i]);
            sb.Append(',');
        }
        return sb.ToString();
    }

    private long AllocateSessionSequence(AiDotNet.Inference.PagedAttention.PagedKVCache<T> cache)
    {
        for (int attempt = 0; attempt < 4096; attempt++)
        {
            long id = System.Threading.Interlocked.Increment(ref _nextSequenceId);
            if (cache.AllocateSequence(id, 0))
            {
                return id;
            }
        }

        throw new InvalidOperationException(
            $"Unable to allocate a KV-cache sequence for model '{_modelName}' (cache exhausted).");
    }

    /// <summary>
    /// Per-request KV-cached generation session over the shared optimized model + paged cache,
    /// isolated by its own sequence id.
    /// </summary>
    private sealed class GenerationSession : IGenerationSession<T>
    {
        private readonly ServableModelWrapper<T> _owner;
        private readonly AiDotNet.NeuralNetworks.NeuralNetworkBase<T> _model;
        private readonly AiDotNet.Inference.PagedAttention.PagedKVCache<T> _cache;
        private readonly long _sequenceId;
        private int _position;
        private bool _disposed;

        public GenerationSession(
            ServableModelWrapper<T> owner,
            AiDotNet.NeuralNetworks.NeuralNetworkBase<T> model,
            AiDotNet.Inference.PagedAttention.PagedKVCache<T> cache,
            long sequenceId,
            int cachedPromptTokens)
        {
            _owner = owner;
            _model = model;
            _cache = cache;
            _sequenceId = sequenceId;
            // A forked session starts decoding after the shared prefix already in its KV cache.
            _position = cachedPromptTokens;
            CachedPromptTokens = cachedPromptTokens;
        }

        public int CachedPromptTokens { get; }

        public Tensor<T> Forward(Tensor<T> newTokenIds)
        {
            Guard.NotNull(newTokenIds);
            if (_disposed)
            {
                throw new ObjectDisposedException(nameof(GenerationSession));
            }

            int newLength = newTokenIds.Shape[newTokenIds.Shape.Length - 1];
            var context = new AiDotNet.Inference.InferenceForwardContext(_sequenceId, _position);
            var logits = _model.PredictWithContext(newTokenIds, context);
            _position += newLength;
            return logits;
        }

        public void RegisterPromptPrefix(System.Collections.Generic.IReadOnlyList<int> promptTokens)
        {
            if (_disposed)
            {
                throw new ObjectDisposedException(nameof(GenerationSession));
            }

            // Only meaningful right after prefill, when this session's KV holds exactly the prompt.
            if (_position == promptTokens.Count)
            {
                _owner.RegisterPrefix(promptTokens, _sequenceId);
            }
        }

        public void Dispose()
        {
            if (_disposed)
            {
                return;
            }

            _disposed = true;
            _cache.FreeSequence(_sequenceId);
        }
    }

    /// <inheritdoc/>
    public Vector<T> Predict(Vector<T> input)
    {
        if (_inputDimension > 0 && input.Length != _inputDimension)
        {
            throw new ArgumentException(
                $"Input dimension mismatch. Expected {_inputDimension}, got {input.Length}",
                nameof(input));
        }

        return _predictFunc(input);
    }

    /// <inheritdoc/>
    public Matrix<T> PredictBatch(Matrix<T> inputs)
    {
        if (_inputDimension > 0 && inputs.Columns != _inputDimension)
        {
            throw new ArgumentException(
                $"Input dimension mismatch. Expected {_inputDimension}, got {inputs.Columns}",
                nameof(inputs));
        }

        // If a batch prediction function was provided, use it
        if (_predictBatchFunc != null)
        {
            return _predictBatchFunc(inputs);
        }

        // Otherwise, fall back to multiple single predictions
        // Predict first row to discover output dimension when it's unknown (0)
        var firstOutput = Predict(inputs.GetRow(0));
        int outDim = _outputDimension > 0 ? _outputDimension : firstOutput.Length;
        var result = new Matrix<T>(inputs.Rows, outDim);
        for (int j = 0; j < outDim; j++)
        {
            result[0, j] = firstOutput[j];
        }

        for (int i = 1; i < inputs.Rows; i++)
        {
            var inputVector = inputs.GetRow(i);
            var outputVector = Predict(inputVector);
            for (int j = 0; j < outDim; j++)
            {
                result[i, j] = outputVector[j];
            }
        }

        return result;
    }
}
