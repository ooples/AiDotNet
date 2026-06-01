using System;
using AiDotNet.Models.Options;
using AiDotNet.Safety;

namespace AiDotNet.Configuration;

/// <summary>
/// Component that owns the compliance / ethical-AI configuration for an AI model build:
/// bias detection, fairness evaluation, interpretability / explainability, adversarial-robustness
/// hardening, and safety filtering. Extracted from <c>AiModelBuilder</c> as slice 4 of the
/// audit-2026-05 phase-2a DI refactor.
/// </summary>
/// <typeparam name="T">Element numeric type.</typeparam>
/// <typeparam name="TInput">Model input tensor type.</typeparam>
/// <typeparam name="TOutput">Model output tensor type.</typeparam>
public interface IAiModelCompliance<T, TInput, TOutput>
{
    /// <summary>The configured bias detector, or <c>null</c> if not configured.</summary>
    IBiasDetector<T>? BiasDetector { get; }

    /// <summary>The configured fairness evaluator, or <c>null</c> if not configured.</summary>
    IFairnessEvaluator<T>? FairnessEvaluator { get; }

    /// <summary>The configured interpretability options, or <c>null</c> if not configured.</summary>
    InterpretabilityOptions? InterpretabilityOptions { get; }

    /// <summary>The configured adversarial-robustness configuration, or <c>null</c> if not configured.</summary>
    AdversarialRobustnessConfiguration<T, TInput, TOutput>? AdversarialRobustnessConfiguration { get; }

    /// <summary>The configured safety-filter pipeline, or <c>null</c> if not configured.</summary>
    SafetyConfig? SafetyPipelineConfig { get; }

    /// <summary>Sets the bias detector.</summary>
    void ConfigureBiasDetector(IBiasDetector<T> detector);

    /// <summary>Sets the fairness evaluator.</summary>
    void ConfigureFairnessEvaluator(IFairnessEvaluator<T> evaluator);

    /// <summary>Sets the interpretability options. <c>null</c> applies defaults.</summary>
    void ConfigureInterpretability(InterpretabilityOptions? options);

    /// <summary>Sets the adversarial-robustness configuration. <c>null</c> applies defaults.</summary>
    void ConfigureAdversarialRobustness(AdversarialRobustnessConfiguration<T, TInput, TOutput>? configuration);

    /// <summary>Sets the safety pipeline. <c>null</c> action applies an empty config.</summary>
    void ConfigureSafety(Action<SafetyConfig>? configure);
}
