using System.Collections.Generic;
using System.Linq;
using AiDotNet.Models;
using AiDotNet.Optimizers;

namespace AiDotNet.Models.Results;

/// <summary>
/// Reports what a trained model's research paper specifies for training, what was actually used,
/// and every difference between the two.
/// </summary>
/// <remarks>
/// <para>
/// Reached through the facade, alongside the family-specific inference extensions from #1836, so a
/// caller who holds only an <see cref="AiModelResult{T, TInput, TOutput}"/> can still ask the
/// question. <c>Model</c> stays internal.
/// </para>
/// <para>
/// The gap this closes: every other library leaves a paper's training recipe in prose — a model
/// card, a README, a training script — detached from the model, not applied, and silently replaced
/// by a generic default. Nothing tells you the model you just trained is an order of magnitude
/// away from its published learning rate. That is exactly what #1928 found across 685 construction
/// sites, and it went unnoticed for the life of the library.
/// </para>
/// </remarks>
public static class AiModelResultTrainingRecipeExtensions
{
    /// <summary>
    /// What the paper specifies, what was applied, and why they differ — one report per component.
    /// </summary>
    /// <returns>
    /// Empty when the model built no optimizer through the paper-recipe path. A model that declares
    /// nothing yields a single report whose <see cref="TrainingRecipeReport.Fidelity"/> is
    /// <see cref="RecipeFidelity.NotDeclared"/>, which is a different statement from "not checked".
    /// </returns>
    /// <example>
    /// <code>
    /// var recipe = result.GetTrainingRecipe().FirstOrDefault();
    /// if (recipe?.Fidelity == RecipeFidelity.Deviated)
    /// {
    ///     foreach (string problem in recipe.Unhonoured) Console.WriteLine(problem);
    /// }
    /// </code>
    /// </example>
    public static IReadOnlyList<TrainingRecipeReport> GetTrainingRecipe<T, TInput, TOutput>(
        this AiModelResult<T, TInput, TOutput> result)
    {
        if (result is null) return [];
        return PaperOptimizerFactory.ReportsFor(result.Model);
    }

    /// <summary>
    /// A one-line summary per component, suitable for logging or a diagnostics panel.
    /// </summary>
    /// <remarks>
    /// Deliberately available without inspecting the structured report: the most common thing a
    /// caller wants is to see, once, that the model is training the way its paper says — or to see
    /// plainly that it is not.
    /// </remarks>
    public static IReadOnlyList<string> DescribeTrainingRecipe<T, TInput, TOutput>(
        this AiModelResult<T, TInput, TOutput> result)
        => result.GetTrainingRecipe().Select(report => report.Describe()).ToArray();

    /// <summary>
    /// True when every component reproduces its paper exactly, with no adaptations and nothing
    /// unhonoured.
    /// </summary>
    /// <remarks>
    /// <see cref="RecipeFidelity.Adapted"/> is deliberately NOT counted as faithful here, even
    /// though an adaptation is usually the more correct choice — a rate stated for batch 4096 is not
    /// the same instruction at batch 32. The distinction is left to the caller rather than decided
    /// for them, because "matches the paper exactly" and "follows the paper, adjusted for my run"
    /// are different claims and only one of them can be made about a reproduction.
    /// </remarks>
    public static bool TrainsExactlyAsPublished<T, TInput, TOutput>(
        this AiModelResult<T, TInput, TOutput> result)
    {
        var reports = result.GetTrainingRecipe();
        return reports.Count > 0 && reports.All(r => r.Fidelity == RecipeFidelity.Exact);
    }
}
