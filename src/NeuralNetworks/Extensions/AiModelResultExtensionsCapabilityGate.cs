using System;

namespace AiDotNet.Extensions.Capability;

/// <summary>
/// Shared capability-gate helper used by the four family-specific extension files
/// (radiance-field / diffusion / transformer / graph). Consolidates the previously-duplicated
/// null-check + type-check + uniform-error-message pattern into one place so every extension
/// path throws with the same shape of message and dev-time updates only happen once.
/// </summary>
internal static class AiModelResultExtensionsCapabilityGate
{
    /// <summary>
    /// Returns <c>result.Model</c> as <typeparamref name="TCapability"/>, or throws with a
    /// clear message naming the actual model type + pointing at the other extension namespaces.
    /// </summary>
    /// <param name="result">The AiModelResult to gate on.</param>
    /// <param name="extensionName">The extension method's name, used in the error message.</param>
    /// <param name="expectedInterfaceName">
    /// Fully-qualified expected interface / class name (e.g.
    /// <c>"AiDotNet.NeuralRadianceFields.Interfaces.IRadianceField&lt;T&gt;"</c>). Included in the
    /// error message so callers can look up the correct type / namespace.
    /// </param>
    /// <param name="hint">Optional additional hint (e.g. "use ImageTrainingDataLoaders...").</param>
    public static TCapability Require<T, TInput, TOutput, TCapability>(
        AiModelResult<T, TInput, TOutput>? result,
        string extensionName,
        string expectedInterfaceName,
        string? hint = null)
        where TCapability : class
    {
        if (result is null)
        {
            throw new ArgumentNullException(nameof(result));
        }

        if (result.Model is not TCapability cap)
        {
            var actualModelType = result.Model?.GetType().FullName ?? "<no model — result not built yet>";
            var hintSuffix = string.IsNullOrEmpty(hint) ? string.Empty : $" {hint}";
            throw new InvalidOperationException(
                $"AiModelResult.{extensionName} requires the underlying model to implement " +
                $"{expectedInterfaceName}. The result was built with '{actualModelType}'.{hintSuffix} " +
                $"Available extension namespaces: AiDotNet.NeuralRadianceFields.Extensions, " +
                $"AiDotNet.Transformers.Extensions, AiDotNet.Diffusion.Extensions, " +
                $"AiDotNet.Graphs.Extensions.");
        }

        return cap;
    }
}
