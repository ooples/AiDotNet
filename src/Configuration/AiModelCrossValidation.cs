namespace AiDotNet.Configuration;

/// <summary>
/// Default implementation of <see cref="IAiModelCrossValidation{T,TInput,TOutput}"/>.
/// Audit-2026-05 phase-2a slice 3.
/// </summary>
internal class AiModelCrossValidation<T, TInput, TOutput> : IAiModelCrossValidation<T, TInput, TOutput>
{
    /// <inheritdoc/>
    public ICrossValidator<T, TInput, TOutput>? CrossValidator { get; private set; }

    /// <inheritdoc/>
    public void ConfigureCrossValidation(ICrossValidator<T, TInput, TOutput> crossValidator)
        => CrossValidator = crossValidator;
}
