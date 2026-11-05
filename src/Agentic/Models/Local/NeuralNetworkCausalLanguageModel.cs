using System;
using System.Collections.Generic;
using AiDotNet.Configuration;
using AiDotNet.Helpers;
using AiDotNet.Inference;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tensors;
using AiDotNet.Tensors.Helpers;

namespace AiDotNet.Agentic.Models.Local;

/// <summary>
/// Adapts a trained AiDotNet <see cref="NeuralNetworkBase{T}"/> language model (e.g.,
/// <c>MambaLanguageModel</c>, <c>GLALanguageModel</c>, or a Transformer LM head) to the
/// <see cref="IIncrementalCausalLanguageModel{T}"/> and <see cref="ILanguageModel{T}"/> seams,
/// enabling KV-cached incremental generation as well as full-context forward passes.
/// </summary>
/// <typeparam name="T">The tensor element type of the network (e.g., <see cref="float"/> or <see cref="double"/>).</typeparam>
public sealed class NeuralNetworkCausalLanguageModel<T>
    : IIncrementalCausalLanguageModel<T>,
      ILanguageModel<T>,
      IDisposable
{
    private readonly NeuralNetworkBase<T> _network;
    private readonly int _maxContextTokens;
    private readonly object _sync = new();

    private InferenceOptimizer<T>? _kvOptimizer;
    private NeuralNetworkBase<T>? _kvNetwork;
    private bool _ownsKvNetwork;
    private bool _kvInitAttempted;
    private bool _hasActiveKvCache;
    private List<int>? _runningTokens;
    private bool _disposed;

    /// <summary>
    /// Initializes a new adapter over a network language model.
    /// </summary>
    public NeuralNetworkCausalLanguageModel(NeuralNetworkBase<T> network, int vocabularySize, int? maxContextTokens = null)
    {
        Guard.NotNull(network);
        Guard.Positive(vocabularySize);
        _network = network;
        VocabularySize = vocabularySize;
        _maxContextTokens = maxContextTokens is { } configured && configured > 0 ? configured : int.MaxValue;
    }

    /// <inheritdoc/>
    public int VocabularySize { get; }

    /// <inheritdoc/>
    public Tensor<T> ForwardLogits(Tensor<T> tokenIds)
    {
        Guard.NotNull(tokenIds);
        if (tokenIds.Shape.Length != 2)
            throw new ArgumentException("tokenIds must have shape [batch, seqLen].", nameof(tokenIds));

        lock (_sync)
        {
            _network.ResetState();
            return _network.Predict(tokenIds);
        }
    }

    /// <inheritdoc/>
    public Vector<T> NextTokenLogits(IReadOnlyList<int> tokenIds)
    {
        Guard.NotNull(tokenIds);
        if (tokenIds.Count == 0)
        {
            throw new ArgumentException("The context must contain at least one token.", nameof(tokenIds));
        }

        var start = _maxContextTokens == int.MaxValue ? 0 : Math.Max(0, tokenIds.Count - _maxContextTokens);
        var sequenceLength = tokenIds.Count - start;

        var numOps = MathHelper.GetNumericOperations<T>();
        var input = new Tensor<T>(new[] { 1, sequenceLength });
        for (var position = 0; position < sequenceLength; position++)
        {
            var id = tokenIds[start + position];
            if (id < 0 || id >= VocabularySize)
            {
                throw new ArgumentOutOfRangeException(
                    nameof(tokenIds), id, $"Token id is outside the vocabulary [0, {VocabularySize}).");
            }

            input[new[] { 0, position }] = numOps.FromDouble(id);
        }

        Tensor<T> output;
        lock (_sync)
        {
            _network.ResetState();
            output = _network.Predict(input);
        }

        return ExtractLastPositionLogits(output);
    }

    /// <inheritdoc/>
    public void ResetCache()
    {
        lock (_sync)
        {
            _kvOptimizer?.ClearCache();
            _kvNetwork?.ResetState();
            _runningTokens = null;
        }
    }

    /// <inheritdoc/>
    public Vector<T> StartSequence(IReadOnlyList<int> promptTokenIds)
    {
        Guard.NotNull(promptTokenIds);
        if (promptTokenIds.Count == 0)
            throw new ArgumentException("The prompt must contain at least one token.", nameof(promptTokenIds));

        lock (_sync)
        {
            EnsureKvOptimizerInitializedLocked();
            _kvOptimizer?.ClearCache();
            _kvNetwork?.ResetState();

            var start = _maxContextTokens == int.MaxValue ? 0 : Math.Max(0, promptTokenIds.Count - _maxContextTokens);
            int seqLen = promptTokenIds.Count - start;
            _runningTokens = new List<int>(seqLen + 16);

            var numOps = MathHelper.GetNumericOperations<T>();
            var input = new Tensor<T>(new[] { 1, seqLen });
            for (int i = 0; i < seqLen; i++)
            {
                int id = promptTokenIds[start + i];
                if (id < 0 || id >= VocabularySize)
                    throw new ArgumentOutOfRangeException(nameof(promptTokenIds), id, $"Token id is outside [0, {VocabularySize}).");
                _runningTokens.Add(id);
                input[0, i] = numOps.FromDouble(id);
            }

            if (_hasActiveKvCache && _kvNetwork != null)
            {
                var cachedOut = _kvNetwork.Predict(input);
                return ExtractLastPositionLogits(cachedOut);
            }

            _network.ResetState();
            return ExtractLastPositionLogits(_network.Predict(input));
        }
    }

    /// <inheritdoc/>
    public Vector<T> AppendToken(int tokenId)
    {
        if (tokenId < 0 || tokenId >= VocabularySize)
            throw new ArgumentOutOfRangeException(nameof(tokenId), tokenId, $"Token id is outside [0, {VocabularySize}).");

        lock (_sync)
        {
            if (_runningTokens == null)
                throw new InvalidOperationException("StartSequence must be called before AppendToken.");

            _runningTokens.Add(tokenId);
            if (_hasActiveKvCache && _kvNetwork != null && _runningTokens.Count <= _maxContextTokens)
            {
                var stepInput = new Tensor<T>(new[] { 1, 1 });
                stepInput[0, 0] = MathHelper.GetNumericOperations<T>().FromDouble(tokenId);
                var stepOut = _kvNetwork.Predict(stepInput);
                return ExtractLastPositionLogits(stepOut);
            }

            return NextTokenLogits(_runningTokens);
        }
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        if (_disposed)
            return;

        lock (_sync)
        {
            if (_disposed)
                return;

            _kvOptimizer?.ClearCache();
            if (_ownsKvNetwork && _kvNetwork != null && !ReferenceEquals(_kvNetwork, _network))
            {
                DisposeOnceGuard.TryDispose(_kvNetwork);
            }
            _kvNetwork = null;
            _kvOptimizer = null;
            _runningTokens = null;
            _disposed = true;
        }
    }

    private void EnsureKvOptimizerInitializedLocked()
    {
        if (_kvInitAttempted)
            return;
        _kvInitAttempted = true;

        try
        {
            var cfg = new InferenceOptimizationConfig
            {
                EnableKVCache = true,
                EnablePagedKVCache = true,
                AttentionMasking = AttentionMaskingMode.Causal,
                KVCacheWindowSize = _maxContextTokens == int.MaxValue ? 4096 : _maxContextTokens
            };
            var opt = new InferenceOptimizer<T>(cfg);
            var (optModel, anyApplied) = opt.OptimizeForInference(_network, cloneModel: true);
            bool kvActive = opt.KVCache != null || opt.PagedKVCache != null;
            if (anyApplied && kvActive)
            {
                _kvOptimizer = opt;
                _kvNetwork = optModel;
                _ownsKvNetwork = !ReferenceEquals(optModel, _network);
                _hasActiveKvCache = true;
            }
            else if (!ReferenceEquals(optModel, _network))
            {
                DisposeOnceGuard.TryDispose(optModel);
            }
        }
        catch
        {
            _hasActiveKvCache = false;
        }
    }
    private Vector<T> ExtractLastPositionLogits(Tensor<T> output)
    {
        var shape = output._shape;
        var vocab = shape[shape.Length - 1];
        if (vocab != VocabularySize)
        {
            throw new InvalidOperationException(
                $"Model produced logits of width {vocab}, but vocabulary size is {VocabularySize}.");
        }

        var logits = new T[vocab];
        if (shape.Length == 3)
        {
            var lastPosition = shape[1] - 1;
            for (var v = 0; v < vocab; v++)
            {
                logits[v] = output[new[] { 0, lastPosition, v }];
            }
        }
        else if (shape.Length == 2)
        {
            var lastPosition = shape[0] - 1;
            for (var v = 0; v < vocab; v++)
            {
                logits[v] = output[new[] { lastPosition, v }];
            }
        }
        else
        {
            throw new InvalidOperationException(
                $"Expected logits of rank 2 or 3, but the model returned rank {shape.Length}.");
        }

        return new Vector<T>(logits);
    }
}
