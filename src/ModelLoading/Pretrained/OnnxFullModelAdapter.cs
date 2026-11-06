using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using AiDotNet.Agentic.Models.Local;
using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.LossFunctions;
using AiDotNet.Models;
using AiDotNet.Onnx;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.Interfaces;
using AiDotNet.Tensors.LinearAlgebra;
using Microsoft.ML.OnnxRuntime;
using OnnxTensors = Microsoft.ML.OnnxRuntime.Tensors;

namespace AiDotNet.ModelLoading.Pretrained;

/// <summary>
/// Adapts an <see cref="OnnxModel{T}"/> (an <see cref="IOnnxModel{T}"/> that runs an ONNX graph through the
/// ONNX runtime) to the <see cref="IFullModel{T, TInput, TOutput}"/>, <see cref="ILanguageModel{T}"/>, and
/// <see cref="IIncrementalCausalLanguageModel{T}"/> contracts so <c>ConfigureModel(PretrainedSource.Onnx(path))</c>
/// supports full-context and KV-cached causal language model generation behind the <c>AiModelResult</c> facade.
/// </summary>
/// <typeparam name="T">The numeric type used for computation.</typeparam>
[ModelDomain(ModelDomain.General)]
[ModelCategory(ModelCategory.NeuralNetwork)]
[ModelTask(ModelTask.FeatureExtraction)]
[ModelComplexity(ModelComplexity.Medium)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("ONNX: Open Neural Network Exchange", "https://onnx.ai/", Year = 2017, Authors = "ONNX Community")]
public sealed class OnnxFullModelAdapter<T>
    : IFullModel<T, Tensor<T>, Tensor<T>>,
      ILanguageModel<T>,
      IIncrementalCausalLanguageModel<T>
{
    private readonly record struct KvSlotInfo(
        string PastInputName,
        string PresentOutputName,
        int PresentOutputIndex,
        int[] TemplateDims,
        int PastSeqAxis,
        Type ElementType);

    private readonly INumericOperations<T> _numOps = MathHelper.GetNumericOperations<T>();
    private readonly OnnxModelOptions? _options;
    private readonly object _kvLock = new();

    private byte[]? _onnxBytes;
    private string? _modelPath;
    private OnnxModel<T> _model;

    private string? _inputIdsName;
    private Type _inputIdsElementType = typeof(long);
    private string? _attentionMaskName;
    private Type _attentionMaskElementType = typeof(long);
    private string? _positionIdsName;
    private Type _positionIdsElementType = typeof(long);
    private string? _useCacheBranchName;
    private string _logitsOutputName = string.Empty;
    private int _vocabularySize;
    private KvSlotInfo[] _kvSlots = Array.Empty<KvSlotInfo>();
    private string[] _incrementalOutputNames = Array.Empty<string>();

    private IDisposableReadOnlyCollection<DisposableNamedOnnxValue>? _cachedRunResult;
    private int _cachedPastLength;
    private List<int>? _statelessRunningTokens;

    /// <summary>Creates an adapter over the ONNX graph in <paramref name="onnxBytes"/>.</summary>
    /// <param name="onnxBytes">The serialized ONNX model.</param>
    /// <param name="options">Optional runtime options (execution provider, etc.).</param>
    public OnnxFullModelAdapter(byte[] onnxBytes, OnnxModelOptions? options = null)
    {
        Guard.NotNull(onnxBytes);
        _onnxBytes = (byte[])onnxBytes.Clone();
        _options = options;
        _model = new OnnxModel<T>(_onnxBytes, options);
        InitializeCausalMetadata();
    }

    private OnnxFullModelAdapter(string modelPath, OnnxModelOptions? options)
    {
        _modelPath = Path.GetFullPath(modelPath);
        _options = options;
        _model = new OnnxModel<T>(_modelPath, options);
        InitializeCausalMetadata();
    }

    /// <summary>Creates an adapter over the ONNX graph at <paramref name="path"/>.</summary>
    /// <param name="path">Path to the <c>.onnx</c> file.</param>
    /// <param name="options">Optional runtime options.</param>
    public static OnnxFullModelAdapter<T> FromFile(string path, OnnxModelOptions? options = null)
    {
        if (string.IsNullOrWhiteSpace(path))
            throw new ArgumentException("Path must be non-empty.", nameof(path));
        if (!File.Exists(path))
            throw new FileNotFoundException($"ONNX file not found: {path}", path);

        // Open directly from disk so ONNX Runtime resolves companion external weight files (e.g. model.onnx_data)
        // and avoids copying multi-gigabyte checkpoints into a managed byte[] array.
        return new OnnxFullModelAdapter<T>(path, options);
    }

    /// <summary>The wrapped ONNX runtime model.</summary>
    public OnnxModel<T> Model => _model;

    /// <inheritdoc/>
    public int VocabularySize
    {
        get
        {
            if (_vocabularySize > 0)
                return _vocabularySize;

            // Dynamic vocab axis fallback: probe with a 1-token forward pass once.
            var probe = ForwardLogits(new Tensor<T>(new[] { 1, 1 }));
            _vocabularySize = probe.Shape[probe.Shape.Length - 1];
            return _vocabularySize;
        }
    }

    /// <inheritdoc/>
    public Tensor<T> Predict(Tensor<T> input)
    {
        Guard.NotNull(input);
        if (_inputIdsName != null && input.Shape.Length == 2 && _model.Session.InputMetadata.Count > 1)
        {
            return ForwardLogits(input);
        }

        return _model.Run(input);
    }

    /// <inheritdoc/>
    public Tensor<T> ForwardLogits(Tensor<T> tokenIds)
    {
        Guard.NotNull(tokenIds);
        if (tokenIds.Shape.Length != 2)
        {
            throw new ArgumentException(
                $"tokenIds must have shape [batch, seqLen]; got [{string.Join(",", tokenIds.Shape)}].",
                nameof(tokenIds));
        }

        int batch = tokenIds.Shape[0];
        int seqLen = tokenIds.Shape[1];
        if (batch <= 0 || seqLen <= 0)
            throw new ArgumentException("tokenIds batch and sequence length must both be positive.", nameof(tokenIds));

        if (_inputIdsName == null)
        {
            return _model.Run(tokenIds);
        }

        var idsLong = new long[batch * seqLen];
        var maskLong = new long[batch * seqLen];
        var posLong = new long[batch * seqLen];
        for (int b = 0; b < batch; b++)
        {
            for (int s = 0; s < seqLen; s++)
            {
                int idx = b * seqLen + s;
                idsLong[idx] = Convert.ToInt64(_numOps.ToDouble(tokenIds[b, s]));
                maskLong[idx] = 1L;
                posLong[idx] = s;
            }
        }

        var inputs = BuildPrefillOnnxInputs(idsLong, maskLong, posLong, batch, seqLen);
        using var results = _model.Session.Run(inputs, new[] { _logitsOutputName });
        var converted = OnnxTensorConverter.FromOnnxValue<T>(results.First());
        return converted ?? throw new InvalidOperationException("ONNX session returned null logits tensor.");
    }

    /// <inheritdoc/>
    public Vector<T> NextTokenLogits(IReadOnlyList<int> tokenIds)
    {
        Guard.NotNull(tokenIds);
        if (tokenIds.Count == 0)
            throw new ArgumentException("Prompt must contain at least one token.", nameof(tokenIds));

        if (_inputIdsName == null)
        {
            var tensor = new Tensor<T>(new[] { 1, tokenIds.Count });
            for (int i = 0; i < tokenIds.Count; i++)
                tensor[0, i] = _numOps.FromDouble(tokenIds[i]);
            var logitsTensor = ForwardLogits(tensor);
            return ExtractLastLogitsFromAiTensor(logitsTensor);
        }

        int seqLen = tokenIds.Count;
        var idsLong = new long[seqLen];
        var maskLong = new long[seqLen];
        var posLong = new long[seqLen];
        for (int i = 0; i < seqLen; i++)
        {
            idsLong[i] = tokenIds[i];
            maskLong[i] = 1L;
            posLong[i] = i;
        }

        var inputs = BuildPrefillOnnxInputs(idsLong, maskLong, posLong, batch: 1, seqLen);
        using var results = _model.Session.Run(inputs, new[] { _logitsOutputName });
        return ExtractLastPositionVector(results.First());
    }

    /// <inheritdoc/>
    public void ResetCache()
    {
        lock (_kvLock)
        {
            _cachedRunResult?.Dispose();
            _cachedRunResult = null;
            _cachedPastLength = 0;
            _statelessRunningTokens = null;
        }
    }

    /// <inheritdoc/>
    public Vector<T> StartSequence(IReadOnlyList<int> promptTokenIds)
    {
        Guard.NotNull(promptTokenIds);
        if (promptTokenIds.Count == 0)
            throw new ArgumentException("Prompt must contain at least one token.", nameof(promptTokenIds));

        lock (_kvLock)
        {
            _cachedRunResult?.Dispose();
            _cachedRunResult = null;
            _cachedPastLength = 0;
            _statelessRunningTokens = null;

            if (_inputIdsName == null || _kvSlots.Length == 0)
            {
                _statelessRunningTokens = new List<int>(promptTokenIds);
                return NextTokenLogits(_statelessRunningTokens);
            }

            int seqLen = promptTokenIds.Count;
            var idsLong = new long[seqLen];
            var maskLong = new long[seqLen];
            var posLong = new long[seqLen];
            for (int i = 0; i < seqLen; i++)
            {
                idsLong[i] = promptTokenIds[i];
                maskLong[i] = 1L;
                posLong[i] = i;
            }

            var prefillInputs = BuildPrefillOnnxInputs(idsLong, maskLong, posLong, batch: 1, seqLen);
            _cachedRunResult = _model.Session.Run(prefillInputs, _incrementalOutputNames);
            _cachedPastLength = seqLen;
            return ExtractLastPositionVector(_cachedRunResult.First());
        }
    }

    /// <inheritdoc/>
    public Vector<T> AppendToken(int tokenId)
    {
        lock (_kvLock)
        {
            if (_kvSlots.Length == 0 || _inputIdsName == null)
            {
                if (_statelessRunningTokens == null)
                    throw new InvalidOperationException("StartSequence must be called before AppendToken.");
                _statelessRunningTokens.Add(tokenId);
                return NextTokenLogits(_statelessRunningTokens);
            }

            if (_cachedRunResult == null)
                throw new InvalidOperationException("StartSequence must be called before AppendToken.");

            var prevResults = _cachedRunResult;
            var prevArray = prevResults.ToArray();

            var stepInputs = new List<NamedOnnxValue>(4 + _kvSlots.Length)
            {
                CreateIntTensorValue(_inputIdsName, _inputIdsElementType, new[] { (long)tokenId }, new[] { 1, 1 })
            };

            if (_attentionMaskName != null)
            {
                var stepMask = new long[_cachedPastLength + 1];
                for (int i = 0; i < stepMask.Length; i++)
                    stepMask[i] = 1L;
                stepInputs.Add(CreateIntTensorValue(_attentionMaskName, _attentionMaskElementType, stepMask, new[] { 1, _cachedPastLength + 1 }));
            }

            if (_positionIdsName != null)
            {
                stepInputs.Add(CreateIntTensorValue(_positionIdsName, _positionIdsElementType, new[] { (long)_cachedPastLength }, new[] { 1, 1 }));
            }

            if (_useCacheBranchName != null)
            {
                stepInputs.Add(NamedOnnxValue.CreateFromTensor(
                    _useCacheBranchName,
                    new OnnxTensors.DenseTensor<bool>(new[] { true }, new[] { 1 })));
            }

            for (int i = 0; i < _kvSlots.Length; i++)
            {
                var slot = _kvSlots[i];
                var presentValue = prevArray[slot.PresentOutputIndex];
                stepInputs.Add(ForwardCachedSlotValue(slot, presentValue));
            }

            var nextResults = _model.Session.Run(stepInputs, _incrementalOutputNames);
            try
            {
                var logitsVec = ExtractLastPositionVector(nextResults.First());
                prevResults.Dispose();
                _cachedRunResult = nextResults;
                _cachedPastLength++;
                return logitsVec;
            }
            catch
            {
                nextResults.Dispose();
                throw;
            }
        }
    }
    /// <inheritdoc/>
    /// <remarks>ONNX graphs are frozen; training is not supported.</remarks>
    public void Train(Tensor<T> input, Tensor<T> expectedOutput) =>
        throw new NotSupportedException(
            "OnnxFullModelAdapter wraps a frozen ONNX inference graph; training is not supported. " +
            "Train the model in its source framework and re-export to ONNX.");

    /// <inheritdoc/>
    public ModelMetadata<T> GetModelMetadata()
    {
        var meta = new ModelMetadata<T>
        {
            Name = string.IsNullOrEmpty(_model.Metadata.ModelName) ? "onnx-model" : _model.Metadata.ModelName,
            Description = _model.Metadata.Description ?? "ONNX inference graph",
        };
        meta.Properties["ExecutionProvider"] = _model.ExecutionProvider;
        meta.Properties["OpsetVersion"] = _model.Metadata.OpsetVersion;
        return meta;
    }

    /// <inheritdoc/>
    public ILossFunction<T> DefaultLossFunction => new MeanSquaredErrorLoss<T>();

    private byte[] GetOrLoadOnnxBytes()
    {
        if (_onnxBytes != null)
            return _onnxBytes;
        if (_modelPath != null && File.Exists(_modelPath))
            return File.ReadAllBytes(_modelPath);
        throw new InvalidOperationException("ONNX bytes are unavailable because the source model file no longer exists.");
    }

    // ---- serialization: round-trip the raw ONNX bytes ----

    /// <inheritdoc/>
    public byte[] Serialize() => (byte[])GetOrLoadOnnxBytes().Clone();

    /// <inheritdoc/>
    public void Deserialize(byte[] data) => Reload(data);

    /// <inheritdoc/>
    public void SaveModel(string filePath) => File.WriteAllBytes(filePath, GetOrLoadOnnxBytes());

    /// <inheritdoc/>
    public void LoadModel(string filePath)
    {
        if (string.IsNullOrWhiteSpace(filePath))
            throw new ArgumentException("Path must be non-empty.", nameof(filePath));
        ResetCache();
        var fresh = new OnnxModel<T>(filePath, _options);
        _model.Dispose();
        _model = fresh;
        _modelPath = Path.GetFullPath(filePath);
        _onnxBytes = null;
        InitializeCausalMetadata();
    }

    /// <inheritdoc/>
    public void SaveState(Stream stream)
    {
        Guard.NotNull(stream);
        var bytes = GetOrLoadOnnxBytes();
        stream.Write(bytes, 0, bytes.Length);
    }

    /// <inheritdoc/>
    public void LoadState(Stream stream)
    {
        Guard.NotNull(stream);
        using var buffer = new MemoryStream();
        stream.CopyTo(buffer);
        Reload(buffer.ToArray());
    }

    private void Reload(byte[] data)
    {
        Guard.NotNull(data);
        ResetCache();
        var fresh = new OnnxModel<T>(data, _options);
        _model.Dispose();
        _model = fresh;
        _modelPath = null;
        _onnxBytes = (byte[])data.Clone();
        InitializeCausalMetadata();
    }

    // ---- feature importance: not meaningful for an opaque graph ----

    /// <inheritdoc/>
    public IEnumerable<int> GetActiveFeatureIndices() => Enumerable.Empty<int>();

    /// <inheritdoc/>
    public void SetActiveFeatureIndices(IEnumerable<int> featureIndices) =>
        throw new NotSupportedException("ONNX graphs do not expose selectable input features.");

    /// <inheritdoc/>
    public bool IsFeatureUsed(int featureIndex) => true;

    /// <inheritdoc/>
    public Dictionary<string, T> GetFeatureImportance() => new();

    // ---- cloning ----

    /// <inheritdoc/>
    public IFullModel<T, Tensor<T>, Tensor<T>> Clone() =>
        _modelPath != null
            ? new OnnxFullModelAdapter<T>(_modelPath, _options)
            : new OnnxFullModelAdapter<T>(GetOrLoadOnnxBytes(), _options);

    /// <inheritdoc/>
    public IFullModel<T, Tensor<T>, Tensor<T>> DeepCopy() => Clone();

    /// <inheritdoc/>
    public void Dispose()
    {
        ResetCache();
        _model.Dispose();
    }
    // --- Parameter surface -------------------------------------------------------------------
    //
    // The weights belong to the loaded ONNX graph, not to this adapter, so there is no vector to
    // hand out and nothing to write into. Reporting zero and refusing writes is the truthful
    // answer, and it is stated here rather than inherited because this adapter implements
    // IFullModel directly instead of deriving from a model base.

    /// <summary>Always zero: the graph owns its weights, this adapter does not.</summary>
    public long ParameterCount => 0;

    /// <summary>Always false: a frozen ONNX graph cannot be initialized from a flat vector.</summary>
    public bool SupportsParameterInitialization => false;

    /// <summary>An empty vector, matching <see cref="ParameterCount"/>.</summary>
    public Vector<T> GetParameters() => new Vector<T>(0);

    /// <inheritdoc/>
    public Vector<T> SanitizeParameters(Vector<T> parameters) =>
        parameters ?? throw new ArgumentNullException(nameof(parameters));

    /// <summary>
    /// Writes a parameter vector into this adapter. Only an EMPTY vector is accepted.
    /// </summary>
    /// <remarks>
    /// The weights belong to the loaded ONNX graph, so this adapter owns none and reports zero.
    /// Accepting the empty vector that implies -- rather than throwing on any call -- is what keeps
    /// a caller that walks every model uniformly from having to special-case ONNX-backed ones. A
    /// non-empty vector still fails on the length.
    /// </remarks>
    public void SetParameters(Vector<T> parameters)
    {
        if (parameters is null) throw new ArgumentNullException(nameof(parameters));
        if (parameters.Length == 0) return;

        throw new ArgumentException(
            "Expected 0 parameters: an ONNX-backed model's weights belong to the loaded graph, not "
            + $"to this adapter, so none can be written. Got {parameters.Length}. Load the model in "
            + "a native mode to train or restore parameters.",
            nameof(parameters));
    }

    /// <inheritdoc />
    public IFullModel<T, Tensor<T>, Tensor<T>> WithParameters(Vector<T> parameters)
    {
        if (parameters is null) throw new ArgumentNullException(nameof(parameters));
        if (parameters.Length == 0) return this;

        throw new ArgumentException(
            "Expected 0 parameters: an ONNX-backed model cannot be rebuilt from a parameter vector; "
            + $"its weights live in the loaded graph. Got {parameters.Length}.",
            nameof(parameters));
    }
    private void InitializeCausalMetadata()
    {
        var session = _model.Session;
        var inMeta = session.InputMetadata;
        var outMeta = session.OutputMetadata;

        _inputIdsName = null;
        _attentionMaskName = null;
        _positionIdsName = null;
        _useCacheBranchName = null;

        foreach (var kv in inMeta)
        {
            string name = kv.Key;
            if (_inputIdsName == null && name.IndexOf("input_ids", StringComparison.OrdinalIgnoreCase) >= 0)
            {
                _inputIdsName = name;
                _inputIdsElementType = kv.Value.ElementType;
            }
            else if (_attentionMaskName == null && name.IndexOf("attention_mask", StringComparison.OrdinalIgnoreCase) >= 0)
            {
                _attentionMaskName = name;
                _attentionMaskElementType = kv.Value.ElementType;
            }
            else if (_positionIdsName == null && name.IndexOf("position_ids", StringComparison.OrdinalIgnoreCase) >= 0)
            {
                _positionIdsName = name;
                _positionIdsElementType = kv.Value.ElementType;
            }
            else if (_useCacheBranchName == null && string.Equals(name, "use_cache_branch", StringComparison.OrdinalIgnoreCase))
            {
                _useCacheBranchName = name;
            }
        }

        _logitsOutputName = outMeta.Keys.FirstOrDefault(k => string.Equals(k, "logits", StringComparison.OrdinalIgnoreCase))
            ?? outMeta.Keys.FirstOrDefault(k => k.IndexOf("logits", StringComparison.OrdinalIgnoreCase) >= 0)
            ?? outMeta.Keys.First();

        var logitsDims = outMeta[_logitsOutputName].Dimensions;
        _vocabularySize = logitsDims.Length > 0 && logitsDims[logitsDims.Length - 1] > 0
            ? logitsDims[logitsDims.Length - 1]
            : 0;

        var slots = new List<KvSlotInfo>();
        var outputNames = new List<string> { _logitsOutputName };
        foreach (var kv in inMeta)
        {
            string pastName = kv.Key;
            string? candidatePresent = null;
            if (pastName.StartsWith("past_key_values.", StringComparison.Ordinal))
            {
                string suffix = pastName.Substring("past_key_values.".Length);
                string p1 = "present." + suffix;
                string p2 = "present_key_values." + suffix;
                if (outMeta.ContainsKey(p1)) candidatePresent = p1;
                else if (outMeta.ContainsKey(p2)) candidatePresent = p2;
            }
            else if (pastName.StartsWith("past.", StringComparison.Ordinal))
            {
                string p1 = "present." + pastName.Substring("past.".Length);
                if (outMeta.ContainsKey(p1)) candidatePresent = p1;
            }

            if (candidatePresent == null)
                continue;

            int[] rawDims = kv.Value.Dimensions;
            int[] template = new int[rawDims.Length];
            for (int d = 0; d < rawDims.Length; d++)
                template[d] = rawDims[d] > 0 ? rawDims[d] : 1;

            int pastAxis = rawDims.Length >= 4
                ? ((rawDims[1] <= 0 && rawDims[2] > 0) ? 1 : 2)
                : Math.Max(0, rawDims.Length - 2);
            template[pastAxis] = 0;

            int presentIndex = outputNames.Count;
            outputNames.Add(candidatePresent);
            slots.Add(new KvSlotInfo(
                PastInputName: pastName,
                PresentOutputName: candidatePresent,
                PresentOutputIndex: presentIndex,
                TemplateDims: template,
                PastSeqAxis: pastAxis,
                ElementType: kv.Value.ElementType));
        }

        _kvSlots = slots.ToArray();
        _incrementalOutputNames = outputNames.ToArray();
    }

    private List<NamedOnnxValue> BuildPrefillOnnxInputs(long[] ids, long[] mask, long[] pos, int batch, int seqLen)
    {
        var inputs = new List<NamedOnnxValue>(4 + _kvSlots.Length)
        {
            CreateIntTensorValue(_inputIdsName!, _inputIdsElementType, ids, new[] { batch, seqLen })
        };

        if (_attentionMaskName != null)
            inputs.Add(CreateIntTensorValue(_attentionMaskName, _attentionMaskElementType, mask, new[] { batch, seqLen }));
        if (_positionIdsName != null)
            inputs.Add(CreateIntTensorValue(_positionIdsName, _positionIdsElementType, pos, new[] { batch, seqLen }));
        if (_useCacheBranchName != null)
        {
            inputs.Add(NamedOnnxValue.CreateFromTensor(
                _useCacheBranchName,
                new OnnxTensors.DenseTensor<bool>(new[] { false }, new[] { 1 })));
        }

        for (int i = 0; i < _kvSlots.Length; i++)
            inputs.Add(CreateEmptySlotValue(_kvSlots[i], batch));

        return inputs;
    }

    private static NamedOnnxValue CreateIntTensorValue(string name, Type elementType, long[] values, int[] dims)
    {
        if (elementType == typeof(int))
        {
            var intVals = new int[values.Length];
            for (int i = 0; i < values.Length; i++)
                intVals[i] = (int)values[i];
            return NamedOnnxValue.CreateFromTensor(name, new OnnxTensors.DenseTensor<int>(intVals, dims));
        }

        return NamedOnnxValue.CreateFromTensor(name, new OnnxTensors.DenseTensor<long>(values, dims));
    }

    private static NamedOnnxValue CreateEmptySlotValue(KvSlotInfo slot, int batch)
    {
        int[] dims = (int[])slot.TemplateDims.Clone();
        if (dims.Length > 0)
            dims[0] = batch;
        dims[slot.PastSeqAxis] = 0;

        if (slot.ElementType == typeof(Float16))
            return NamedOnnxValue.CreateFromTensor(slot.PastInputName, new OnnxTensors.DenseTensor<Float16>(Array.Empty<Float16>(), dims));
        if (slot.ElementType == typeof(double))
            return NamedOnnxValue.CreateFromTensor(slot.PastInputName, new OnnxTensors.DenseTensor<double>(Array.Empty<double>(), dims));

        return NamedOnnxValue.CreateFromTensor(slot.PastInputName, new OnnxTensors.DenseTensor<float>(Array.Empty<float>(), dims));
    }

    private static NamedOnnxValue ForwardCachedSlotValue(KvSlotInfo slot, DisposableNamedOnnxValue presentValue)
    {
        if (slot.ElementType == typeof(Float16))
            return NamedOnnxValue.CreateFromTensor(slot.PastInputName, presentValue.AsTensor<Float16>());
        if (slot.ElementType == typeof(double))
            return NamedOnnxValue.CreateFromTensor(slot.PastInputName, presentValue.AsTensor<double>());

        return NamedOnnxValue.CreateFromTensor(slot.PastInputName, presentValue.AsTensor<float>());
    }

    private Vector<T> ExtractLastPositionVector(DisposableNamedOnnxValue logitsValue)
    {
        if (logitsValue.ElementType == Microsoft.ML.OnnxRuntime.Tensors.TensorElementType.Float)
        {
            var tensor = logitsValue.AsTensor<float>();
            var dims = tensor.Dimensions;
            int vocab = dims[dims.Length - 1];
            if (_vocabularySize == 0)
                _vocabularySize = vocab;

            var row = new T[vocab];
            if (dims.Length == 3)
            {
                int lastSeq = dims[1] - 1;
                for (int v = 0; v < vocab; v++)
                    row[v] = _numOps.FromDouble(tensor[0, lastSeq, v]);
            }
            else if (dims.Length == 2)
            {
                int lastSeq = dims[0] - 1;
                for (int v = 0; v < vocab; v++)
                    row[v] = _numOps.FromDouble(tensor[lastSeq, v]);
            }
            else
            {
                throw new InvalidOperationException($"Unexpected ONNX logits rank {dims.Length}.");
            }

            return new Vector<T>(row);
        }

        var converted = OnnxTensorConverter.FromOnnxValue<T>(logitsValue)
            ?? throw new InvalidOperationException("Failed to convert ONNX logits output.");
        return ExtractLastLogitsFromAiTensor(converted);
    }

    private Vector<T> ExtractLastLogitsFromAiTensor(Tensor<T> output)
    {
        var shape = output.Shape;
        int vocab = shape[shape.Length - 1];
        if (_vocabularySize == 0)
            _vocabularySize = vocab;

        var row = new T[vocab];
        if (shape.Length == 3)
        {
            int lastPos = shape[1] - 1;
            for (int v = 0; v < vocab; v++)
                row[v] = output[0, lastPos, v];
        }
        else if (shape.Length == 2)
        {
            int lastPos = shape[0] - 1;
            for (int v = 0; v < vocab; v++)
                row[v] = output[lastPos, v];
        }
        else
        {
            throw new InvalidOperationException($"Expected rank-2 or rank-3 logits tensor; got rank {shape.Length}.");
        }

        return new Vector<T>(row);
    }
}
