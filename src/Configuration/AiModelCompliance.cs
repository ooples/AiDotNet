using System;
using AiDotNet.Models.Options;
using AiDotNet.Safety;

namespace AiDotNet.Configuration;

/// <summary>
/// Default implementation of <see cref="IAiModelCompliance{T,TInput,TOutput}"/>. Audit-2026-05
/// phase-2a slice 4. Mirrors pre-refactor inline logic verbatim, including the
/// <c>?? new SomethingConfig()</c> fallback patterns for the null-arg overloads.
/// </summary>
public class AiModelCompliance<T, TInput, TOutput> : IAiModelCompliance<T, TInput, TOutput>
{
    /// <inheritdoc/>
    public IBiasDetector<T>? BiasDetector { get; private set; }

    /// <inheritdoc/>
    public IFairnessEvaluator<T>? FairnessEvaluator { get; private set; }

    /// <inheritdoc/>
    public InterpretabilityOptions? InterpretabilityOptions { get; private set; }

    /// <inheritdoc/>
    public AdversarialRobustnessConfiguration<T, TInput, TOutput>? AdversarialRobustnessConfiguration { get; private set; }

    /// <inheritdoc/>
    public SafetyConfig? SafetyPipelineConfig { get; private set; }

    /// <inheritdoc/>
    public void ConfigureBiasDetector(IBiasDetector<T> detector) => BiasDetector = detector;

    /// <inheritdoc/>
    public void ConfigureFairnessEvaluator(IFairnessEvaluator<T> evaluator) => FairnessEvaluator = evaluator;

    /// <inheritdoc/>
    public void ConfigureInterpretability(InterpretabilityOptions? options)
        => InterpretabilityOptions = options ?? new InterpretabilityOptions();

    /// <inheritdoc/>
    public void ConfigureAdversarialRobustness(AdversarialRobustnessConfiguration<T, TInput, TOutput>? configuration)
        => AdversarialRobustnessConfiguration = configuration ?? new AdversarialRobustnessConfiguration<T, TInput, TOutput>();

    /// <inheritdoc/>
    public void ConfigureSafety(Action<SafetyConfig>? configure)
    {
        var config = new SafetyConfig();
        configure?.Invoke(config);
        SafetyPipelineConfig = config;
    }
}
