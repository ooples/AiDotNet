using System;
using System.Collections.Generic;
using System.Threading;
using AiDotNet.Agentic.Models.Local;

namespace AiDotNet.Serving.Engine;

/// <summary>
/// The in-process text-generation facade: the one-object surface behind <c>model.Generate(...)</c>. It owns a
/// <see cref="ContinuousBatchingEngine{T}"/> configured with the right runner for the model (paged fast path or
/// recompute fallback) and, when a tokenizer is attached, turns strings into tokens and back. A beginner uses
/// it without ever seeing block managers, schedulers, or sampling internals.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> this is what "just generate text with my model" looks like. Give it a prompt and
/// it returns the continuation — fast, batched, and memory-managed underneath. If you attached a tokenizer it
/// speaks strings; otherwise it speaks token ids (numbers).</para>
/// </remarks>
/// <typeparam name="T">The model's numeric type.</typeparam>
public sealed class TextGenerator<T> : IDisposable
{
    private readonly ContinuousBatchingEngine<T> _engine;
    private readonly IGenerationTokenizer? _tokenizer;
    private readonly SamplingParameters _defaultSampling;
    private long _requestCounter;
    private bool _disposed;

    /// <summary>Builds a generator for a model, auto-selecting the runner and EOS handling.</summary>
    /// <param name="model">The model to serve (must implement <see cref="AiDotNet.Interfaces.ICausalLmModel{T}"/>
    /// or the paged <see cref="ICausalLmRunner{T}"/> capability).</param>
    /// <param name="tokenizer">Optional tokenizer enabling the string overloads. When supplied and the model
    /// declares no EOS, the tokenizer's EOS is used.</param>
    /// <param name="options">Optional engine tuning. Its <see cref="EngineOptions.EosTokenId"/> is filled from
    /// the model/tokenizer when not set.</param>
    /// <param name="defaultSampling">Sampling used when a call passes none. Defaults to greedy, 128 tokens.</param>
    public TextGenerator(
        object model,
        IGenerationTokenizer? tokenizer = null,
        EngineOptions? options = null,
        SamplingParameters? defaultSampling = null)
    {
        if (model is null) throw new ArgumentNullException(nameof(model));
        _tokenizer = tokenizer;
        _defaultSampling = defaultSampling ?? new SamplingParameters { Temperature = 0.0, MaxTokens = 128 };
        _defaultSampling.Validate();

        var selection = ServingRunnerFactory.Create<T>(model);
        int? eos = selection.EosTokenId
            ?? (tokenizer is not null && tokenizer.EosTokenId >= 0 ? tokenizer.EosTokenId : (int?)null);

        var opts = options ?? new EngineOptions();
        if (opts.EosTokenId is null && eos is not null)
            opts = CloneWithEos(opts, eos);

        _engine = new ContinuousBatchingEngine<T>(selection.Runner, opts);
    }

    /// <summary>Generates a continuation for a tokenized prompt, returning the generated token ids.</summary>
    public IReadOnlyList<int> Generate(IReadOnlyList<int> promptTokenIds, SamplingParameters? sampling = null)
    {
        if (promptTokenIds is null) throw new ArgumentNullException(nameof(promptTokenIds));
        if (promptTokenIds.Count == 0) throw new ArgumentException("Prompt must be non-empty.", nameof(promptTokenIds));

        var sp = sampling ?? _defaultSampling;
        string requestId = "gen-" + Interlocked.Increment(ref _requestCounter).ToString();
        _engine.AddRequest(new GenerationRequest(requestId, promptTokenIds, sp));

        // Pump the engine until this request finishes. Other in-flight requests advance too (continuous
        // batching), but this call returns as soon as its own request is complete.
        while (_engine.HasUnfinishedRequests)
        {
            foreach (var output in _engine.Step())
            {
                if (output.RequestId == requestId && output.IsFinished)
                    return output.Outputs.Count > 0 ? output.Outputs[0].TokenIds : Array.Empty<int>();
            }
        }
        return Array.Empty<int>();
    }

    /// <summary>Generates a continuation for a text prompt (requires a tokenizer). Returns the generated text.</summary>
    public string Generate(string prompt, SamplingParameters? sampling = null)
    {
        if (prompt is null) throw new ArgumentNullException(nameof(prompt));
        if (_tokenizer is null)
            throw new InvalidOperationException(
                "Generate(string) requires a tokenizer. Attach one when creating the generator, or call " +
                "Generate(IReadOnlyList<int>) with pre-tokenized ids.");

        var promptIds = _tokenizer.Encode(prompt);
        if (promptIds.Count == 0)
            throw new ArgumentException("Prompt tokenized to zero tokens.", nameof(prompt));

        var generated = Generate(promptIds, sampling);
        return _tokenizer.Decode(generated);
    }

    /// <summary>A snapshot of engine load and KV-cache utilization.</summary>
    public EngineStatistics GetStatistics() => _engine.GetStatistics();

    private static EngineOptions CloneWithEos(EngineOptions o, int? eos) => new()
    {
        MaxNumSequences = o.MaxNumSequences,
        MaxBatchedTokens = o.MaxBatchedTokens,
        BlockSize = o.BlockSize,
        NumKvBlocks = o.NumKvBlocks,
        EosTokenId = eos,
    };

    /// <inheritdoc/>
    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        _engine.Dispose();
    }
}
