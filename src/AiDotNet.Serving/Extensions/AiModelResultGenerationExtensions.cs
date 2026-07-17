using System;
using System.Collections.Generic;
using AiDotNet.Agentic.Models.Local;
using AiDotNet.Configuration;
using AiDotNet.Models.Results;
using AiDotNet.Serving.Engine;

namespace AiDotNet.Serving.Extensions;

/// <summary>
/// The beginner-facing text-generation surface on a built model. These extensions give a trained
/// <see cref="AiModelResult{T, TInput, TOutput}"/> vLLM-class generation with zero configuration: a
/// paged-KV, continuously-batched engine is created for the model automatically, using the fast path when the
/// model advertises it and a correctness fallback otherwise.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> after you build a language model, call <c>model.Generate("your prompt")</c> and
/// you get the continuation — the high-performance serving engine is wired up for you. For repeated calls or
/// many concurrent users, keep a single <see cref="TextGenerator{T}"/> from <see cref="AsTextGenerator"/> (or
/// run a server) instead of calling the one-shot helpers in a loop.</para>
/// </remarks>
public static class AiModelResultGenerationExtensions
{
    /// <summary>
    /// Creates a reusable <see cref="TextGenerator{T}"/> for the built model — the handle to hold when you
    /// generate repeatedly. Auto-selects the paged fast path or the recompute fallback.
    /// </summary>
    public static TextGenerator<T> AsTextGenerator<T, TInput, TOutput>(
        this AiModelResult<T, TInput, TOutput> result,
        IGenerationTokenizer? tokenizer = null,
        EngineOptions? options = null,
        SamplingParameters? defaultSampling = null,
        InferenceOptimizationConfig? config = null)
    {
        if (result is null) throw new ArgumentNullException(nameof(result));
        var model = result.Model
            ?? throw new InvalidOperationException("The model has not been trained/initialized; cannot generate.");
        return new TextGenerator<T>(model, tokenizer, options, defaultSampling, config);
    }

    /// <summary>Generates a continuation for a tokenized prompt, returning the generated token ids.</summary>
    public static IReadOnlyList<int> Generate<T, TInput, TOutput>(
        this AiModelResult<T, TInput, TOutput> result,
        IReadOnlyList<int> promptTokenIds,
        SamplingParameters? sampling = null)
    {
        using var generator = result.AsTextGenerator();
        return generator.Generate(promptTokenIds, sampling);
    }

    /// <summary>
    /// Generates a text continuation for a text prompt using the supplied tokenizer. For repeated calls, prefer
    /// holding a <see cref="TextGenerator{T}"/> from <see cref="AsTextGenerator"/> with the tokenizer attached.
    /// </summary>
    public static string Generate<T, TInput, TOutput>(
        this AiModelResult<T, TInput, TOutput> result,
        string prompt,
        IGenerationTokenizer tokenizer,
        SamplingParameters? sampling = null)
    {
        if (tokenizer is null) throw new ArgumentNullException(nameof(tokenizer));
        using var generator = result.AsTextGenerator(tokenizer);
        return generator.Generate(prompt, sampling);
    }
}
