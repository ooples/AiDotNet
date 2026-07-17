using System;
using System.Collections.Generic;
using AiDotNet.Interfaces;
using AiDotNet.Models.Results;
using AiDotNet.Serving.Engine;
using AiDotNet.Serving.Engine.Speculative;

namespace AiDotNet.Serving.Extensions;

/// <summary>
/// Speculative-decoding surface on a built model: faster greedy generation with the same output. Uses model-free
/// prompt-lookup drafting by default, so a beginner gets the speedup with no extra model and no configuration.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> this generates the same text as <c>Generate</c> but can be faster on repetitive
/// output, because it guesses several tokens ahead and lets the model confirm them in one shot. The result is
/// identical to normal greedy generation — only the speed changes.</para>
/// </remarks>
public static class AiModelResultSpeculativeExtensions
{
    /// <summary>
    /// Creates a reusable <see cref="SpeculativeGenerator{T}"/> for the model (requires the model to implement
    /// <see cref="ICausalLmModel{T}"/>). Defaults to model-free prompt-lookup drafting.
    /// </summary>
    public static SpeculativeGenerator<T> AsSpeculativeGenerator<T, TInput, TOutput>(
        this AiModelResult<T, TInput, TOutput> result,
        ISpeculativeDrafter? drafter = null,
        int maxDraftTokens = 4)
    {
        if (result is null) throw new ArgumentNullException(nameof(result));
        var model = result.Model
            ?? throw new InvalidOperationException("The model has not been trained/initialized; cannot generate.");
        if (model is not ICausalLmModel<T> lm)
            throw new NotSupportedException(
                $"Speculative decoding requires the model to implement ICausalLmModel<{typeof(T).Name}> " +
                "(it verifies drafts via multi-position ForwardLogits).");
        return new SpeculativeGenerator<T>(lm, drafter, maxDraftTokens);
    }

    /// <summary>
    /// Generates a continuation with greedy speculative decoding, returning the generated token ids. The output
    /// is identical to <see cref="AiModelResultGenerationExtensions.Generate{T, TInput, TOutput}(AiModelResult{T, TInput, TOutput}, IReadOnlyList{int}, SamplingParameters)"/>
    /// greedy generation — only faster.
    /// </summary>
    public static IReadOnlyList<int> GenerateSpeculative<T, TInput, TOutput>(
        this AiModelResult<T, TInput, TOutput> result,
        IReadOnlyList<int> promptTokenIds,
        SamplingParameters? sampling = null,
        ISpeculativeDrafter? drafter = null,
        int maxDraftTokens = 4)
    {
        var generator = result.AsSpeculativeGenerator(drafter, maxDraftTokens);
        return generator.Generate(promptTokenIds, sampling ?? new SamplingParameters { Temperature = 0.0, MaxTokens = 128 });
    }
}
