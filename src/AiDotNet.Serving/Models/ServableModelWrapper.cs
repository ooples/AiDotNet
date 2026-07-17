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
public class ServableModelWrapper<T> : IServableModel<T>, IServableModelInferenceOptions, IServableGenerativeModel<T>, System.IDisposable
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
    private readonly bool _supportsBatchedPrefill;

    // The ONE shared continuous-batching engine for this model (built lazily over the incremental model +
    // paged cache). All generation requests are submitted to it and co-batch; its background loop is the
    // single thread that runs model forwards, so concurrent requests never race per-forward layer scratch.
    private AiDotNet.Serving.ContinuousBatching.ContinuousBatcher<T>? _sharedBatcher;
    private readonly object _batcherInitLock = new();
    // Only one thread drives the engine's Step() at a time. Whoever holds it advances the WHOLE batch, so
    // concurrent requests co-batch; a request returns as soon as its own result task completes. Driving
    // synchronously (no background loop) keeps generation deterministic and serializes model forwards.
    private readonly object _engineLock = new();
    private bool _disposed;

    // The optimized, context-aware model backing incremental decode (writes/reads paged KV per sequence
    // id via PredictWithContext), or null when this model has no incremental path. Exposed to the
    // continuous-batching engine (and its equivalence tests) so ONE shared batcher can drive the same
    // model + paged cache the session path uses — the two must produce byte-identical output before the
    // live serving path routes through the batcher.
    internal AiDotNet.NeuralNetworks.NeuralNetworkBase<T>? IncrementalModel => _incrementalModel;

    // The shared paged KV cache backing every sequence of this model, or null when there is no
    // incremental path. Same instance the GenerationSession path allocates its sequences in.
    internal AiDotNet.Inference.PagedAttention.PagedKVCache<T>? IncrementalCache => _incrementalCache;



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
    /// <param name="quantizeIncrementalWeights">When generation is enabled, whether the incremental
    /// KV-cached model uses int8 weight-only quantization so more sequences stay KV-resident.</param>
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
                if (_incrementalModel is not null && _incrementalCache is not null)
                {
                    _supportsBatchedPrefill = ProbeBatchedPrefill(_incrementalModel, _incrementalCache);
                }
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
    /// Probes whether the optimized model accepts a multi-token forward as genuine PER-POSITION logits,
    /// so the prompt can be prefilled in one pass. Two conditions must hold: (1) the forward does not
    /// throw, and (2) the output preserves the sequence dimension — i.e. an <c>n</c>-token input yields
    /// <c>n</c> output positions (<c>[1, n, vocab]</c>), not a single collapsed row (<c>[1, vocab]</c>).
    /// </summary>
    /// <remarks>
    /// The second condition is essential: a <c>Flatten → Dense</c> language model collapses the
    /// sequence into one fixed-width row, so a multi-token forward does NOT throw — it silently
    /// produces a single position (and a shape-dependent dense layer would even re-fit its weights to
    /// the wider flattened input). Treating that as "batched prefill supported" would feed the model a
    /// multi-token sequence whose Dense input width differs from the single-token decode width, which
    /// is incorrect. Requiring the position dimension to be preserved rejects such models, so they fall
    /// back to the correct per-token prefill path.
    /// </remarks>
    private static bool ProbeBatchedPrefill(
        AiDotNet.NeuralNetworks.NeuralNetworkBase<T> model,
        AiDotNet.Inference.PagedAttention.PagedKVCache<T> cache)
    {
        const int probeTokens = 2;
        const long probeSequenceId = long.MaxValue - 1;
        if (!cache.AllocateSequence(probeSequenceId, 0))
        {
            return false;
        }

        try
        {
            var probe = new Tensor<T>(new[] { 1, probeTokens }); // two tokens in one forward
            var output = model.PredictWithContext(probe, new AiDotNet.Inference.InferenceForwardContext(probeSequenceId, 0));

            // Require the output to keep one logits row PER input token. positions = (total / vocab):
            // [1, 2, vocab] -> 2 (per-position, supported); [1, vocab] -> 1 (collapsed, NOT supported).
            int rank = output.Shape.Length;
            if (rank < 2)
            {
                return RejectBatchedPrefill("OutputRankTooLow");
            }
            int vocab = output.Shape[rank - 1];
            if (vocab <= 0)
            {
                return RejectBatchedPrefill("OutputVocabNonPositive");
            }
            long total = 1;
            for (int d = 0; d < rank; d++) total *= output.Shape[d];
            long positions = total / vocab;
            if (positions != probeTokens)
            {
                return RejectBatchedPrefill($"CollapsedSequence(positions={positions})");
            }
            return true;
        }
        catch (Exception ex)
        {
            // Expected for fixed single-token-step models; record as a capability decision, not an error.
            return RejectBatchedPrefill(ex.GetType().Name);
        }
        finally
        {
            cache.FreeSequence(probeSequenceId);
        }
    }

    private static bool RejectBatchedPrefill(string reason)
    {
        AiDotNet.Helpers.InferenceDiagnostics.RecordDecision(
            area: "Serving.ServableModelWrapper",
            feature: "BatchedPrefill",
            enabled: false,
            reason: reason);
        return false;
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
    /// <param name="enableTextGeneration">Whether to build the KV-cached incremental generation path (tensor LMs only).</param>
    /// <param name="quantizeKvCacheWeights">Whether the incremental generation model uses int8 weight-only quantization.</param>
    /// <returns>A ServableModelWrapper configured for the detected model type.</returns>
    /// <exception cref="InvalidOperationException">Thrown when the model does not implement a supported IFullModel variant.</exception>
    internal static ServableModelWrapper<T> FromModel(
        string modelName,
        IModelSerializer model,
        bool enableBatching = true,
        bool enableSpeculativeDecoding = false,
        bool enableTextGeneration = false,
        bool quantizeKvCacheWeights = false)
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

            // Token-generation models (declared via config) get the KV-cached incremental path; other
            // tensor models (e.g. diffusion) do not advertise generation. model.Predict is the
            // token-to-logits forward only when the operator marks the model generative.
            Func<Tensor<T>, Tensor<T>>? generationForward = enableTextGeneration ? tensorModel.Predict : null;

            return new ServableModelWrapper<T>(
                modelName, tensorModel, inputShape, enableBatching, enableSpeculativeDecoding,
                generationForward: generationForward,
                quantizeIncrementalWeights: enableTextGeneration && quantizeKvCacheWeights);
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
    /// <param name="enableTextGeneration">Whether to build the KV-cached incremental generation path (tensor LMs only).</param>
    /// <param name="quantizeKvCacheWeights">Whether the incremental generation model uses int8 weight-only quantization.</param>
    /// <returns>A ServableModelWrapper ready for serving.</returns>
    internal static ServableModelWrapper<T> LoadServable(
        string filePath,
        string modelName,
        bool enableBatching = true,
        bool enableSpeculativeDecoding = false,
        string? licenseKey = null,
        byte[]? decryptionToken = null,
        bool enableTextGeneration = false,
        bool quantizeKvCacheWeights = false)
    {
        var model = ModelLoader.Load<T>(filePath, licenseKey, decryptionToken);
        return FromModel(modelName, model, enableBatching, enableSpeculativeDecoding, enableTextGeneration, quantizeKvCacheWeights);
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
    public bool SupportsBatchedPrefill => _supportsBatchedPrefill;


    /// <summary>
    /// The ONE shared continuous-batching engine for this model, built lazily over the incremental model
    /// and paged cache. Returns null when the model has no incremental path. Its background loop is the
    /// single thread that runs forwards, so concurrent submitted requests co-batch and never race.
    /// </summary>
    internal AiDotNet.Serving.ContinuousBatching.ContinuousBatcher<T>? EnsureBatcher()
    {
        if (_incrementalModel is not { } model || _incrementalCache is not { } cache)
        {
            return null;
        }

        if (_sharedBatcher is { } existing)
        {
            return existing;
        }

        lock (_batcherInitLock)
        {
            if (_sharedBatcher is null)
            {
                var config = new AiDotNet.Serving.ContinuousBatching.ContinuousBatcherConfig
                {
                    // Requests drive Step() synchronously under the engine lock (no background loop), so
                    // generation is deterministic and forwards are serialized.
                    AutoStart = false,
                    // Per-request GenerationRequest.SpeculationDepth overrides this; a positive default keeps
                    // greedy-exact prompt-lookup speculation available to greedy requests that don't specify one.
                    // Sequence-collapsing models (SupportsBatchedPrefill=false) must prefill/decode one
                    // token at a time; a multi-token forward would re-fit a shape-dependent head.
                    SupportsBatchedPrefill = _supportsBatchedPrefill,
                    EnableSpeculativeDecoding = true,
                    SpeculationDepth = 4
                };
                _sharedBatcher = new AiDotNet.Serving.ContinuousBatching.ContinuousBatcher<T>(config, model, cache);
            }
            return _sharedBatcher;
        }
    }

    /// <summary>
    /// Submits a request to the shared batcher and blocks until it completes, returning the result. The
    /// batcher's background loop drives generation; concurrent callers co-batch on the same engine.
    /// </summary>
    internal AiDotNet.Serving.ContinuousBatching.GenerationResult<T> RunGeneration(
        AiDotNet.Serving.ContinuousBatching.GenerationRequest<T> request,
        System.Threading.CancellationToken cancellationToken)
    {
        Guard.NotNull(request);
        var batcher = EnsureBatcher()
            ?? throw new NotSupportedException($"Model '{_modelName}' does not support incremental generation.");
        var task = batcher.GenerateAsync(request, cancellationToken);
        DriveUntilComplete(batcher, task, request, cancellationToken);
        return task.GetAwaiter().GetResult();
    }

    /// <summary>
    /// Cooperatively drives the shared engine until <paramref name="task"/> completes. Whoever holds the
    /// engine lock advances the whole batch (so concurrent requests co-batch); a caller whose task another
    /// thread is already driving spins until its own task completes. Forwards are serialized by the lock.
    /// </summary>
    private void DriveUntilComplete(
        AiDotNet.Serving.ContinuousBatching.ContinuousBatcher<T> batcher,
        System.Threading.Tasks.Task task,
        AiDotNet.Serving.ContinuousBatching.GenerationRequest<T> request,
        System.Threading.CancellationToken cancellationToken)
    {
        // Generous upper bound on Step() calls for this driver: each Step advances every scheduled sequence
        // (including this request's) by >= 1 token, so this request completes well within it. Guards against
        // a wedged sequence that never makes progress instead of spinning forever.
        long maxSteps = ((long)request.MaxNewTokens + request.PromptTokenIds.Count + 64) * 8L;
        var spin = new System.Threading.SpinWait();

        while (!task.IsCompleted)
        {
            cancellationToken.ThrowIfCancellationRequested();

            if (System.Threading.Monitor.TryEnter(_engineLock))
            {
                try
                {
                    while (!task.IsCompleted)
                    {
                        cancellationToken.ThrowIfCancellationRequested();
                        batcher.Step();
                        if (--maxSteps <= 0)
                        {
                            throw new TimeoutException(
                                $"Generation for model '{_modelName}' did not complete within the step budget.");
                        }
                    }
                }
                finally
                {
                    System.Threading.Monitor.Exit(_engineLock);
                }
            }
            else
            {
                // Another thread is driving (and advancing this request's sequence too); wait for my task.
                spin.SpinOnce();
            }
        }
    }

    /// <summary>
    /// The shared batcher's cumulative speculative-decoding acceptance rate (fraction of drafted tokens
    /// accepted), or null when speculation has not run. Exposed for serving-layer telemetry.
    /// </summary>
    internal double? SpeculationAcceptanceRate => _sharedBatcher?.SpeculationAcceptanceRate;

    /// <summary>
    /// Submits a request to the shared batcher and streams generated token ids as they are produced. The
    /// end-of-sequence token is not yielded. Enumeration ends when generation completes or is cancelled.
    /// </summary>
    internal System.Collections.Generic.IEnumerable<int> StreamGeneration(
        AiDotNet.Serving.ContinuousBatching.GenerationRequest<T> request,
        int eosTokenId,
        System.Threading.CancellationToken cancellationToken)
    {
        Guard.NotNull(request);
        var batcher = EnsureBatcher();
        if (batcher is null)
        {
            yield break;
        }

        // Tokens are captured as the engine produces them and yielded (outside the engine lock) as they
        // become available. The consumer itself helps drive the shared engine one Step at a time.
        var queue = new System.Collections.Concurrent.ConcurrentQueue<int>();
        request.OnTokenGenerated = token => queue.Enqueue(token);
        var task = batcher.GenerateAsync(request, cancellationToken);

        while (true)
        {
            if (cancellationToken.IsCancellationRequested)
            {
                yield break;
            }

            // Advance the batch by one Step if we can grab the engine (another streaming/blocking caller
            // may currently hold it — its Steps still advance this request's sequence).
            if (System.Threading.Monitor.TryEnter(_engineLock))
            {
                try
                {
                    if (!task.IsCompleted)
                    {
                        batcher.Step();
                    }
                }
                finally
                {
                    System.Threading.Monitor.Exit(_engineLock);
                }
            }

            // Drain and yield any tokens produced so far (EOS terminates the stream, not yielded).
            while (queue.TryDequeue(out int token))
            {
                if (token == eosTokenId)
                {
                    yield break;
                }
                yield return token;
            }

            if (task.IsCompleted)
            {
                yield break;
            }
        }
    }

    /// <summary>
    /// Stops the shared batching engine (if started) and releases its resources.
    /// </summary>
    public void Dispose()
    {
        if (_disposed)
        {
            return;
        }
        _disposed = true;
        _sharedBatcher?.Dispose();
        System.GC.SuppressFinalize(this);
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
