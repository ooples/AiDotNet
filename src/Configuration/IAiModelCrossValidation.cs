namespace AiDotNet.Configuration;

/// <summary>
/// Component that owns the cross-validation configuration for an AI model build. Extracted from
/// <c>AiModelBuilder</c> as slice 3 of the audit-2026-05 phase-2a DI refactor.
/// </summary>
/// <typeparam name="T">Element numeric type.</typeparam>
/// <typeparam name="TInput">Model input tensor type.</typeparam>
/// <typeparam name="TOutput">Model output tensor type.</typeparam>
internal interface IAiModelCrossValidation<T, TInput, TOutput>
{
    /// <summary>The configured cross-validator, or <c>null</c> if not configured.</summary>
    ICrossValidator<T, TInput, TOutput>? CrossValidator { get; }

    /// <summary>Sets the cross-validation strategy.</summary>
    void ConfigureCrossValidation(ICrossValidator<T, TInput, TOutput> crossValidator);
}
