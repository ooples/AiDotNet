using System.Collections.Generic;
using AiDotNet.Enums;

// Deliberately in AiDotNet.Models rather than a Models.Training sub-namespace: a
// `Models.Training` segment shadows the existing top-level AiDotNet.Training for any
// file inside AiDotNet.Models.*, which breaks unqualified references such as
// `Training.Memory.TrainingMemoryConfig` in AiModelResultOptions.
namespace AiDotNet.Models;

/// <summary>
/// How faithfully a model's training configuration reproduces the recipe its paper specifies.
/// </summary>
public enum RecipeFidelity
{
    /// <summary>The model declares no paper recipe, so library defaults are in use.</summary>
    /// <remarks>
    /// Not a failure — most models are simply not declared yet. It is reported rather than hidden
    /// so that "we do not know what this paper specifies" is distinguishable from "we match it".
    /// </remarks>
    NotDeclared = 0,

    /// <summary>Every declared value is applied exactly as the paper states it.</summary>
    Exact,

    /// <summary>
    /// The paper's recipe is applied, with documented adjustments for this run's scale.
    /// </summary>
    /// <remarks>
    /// Adapted is not a lesser form of Exact. A learning rate stated for batch 4096 is not the same
    /// instruction at batch 32, and transplanting the number unchanged would be less faithful, not
    /// more. Every adjustment names the rule that justifies it.
    /// </remarks>
    Adapted,

    /// <summary>Part of the recipe could not be honoured, and is listed in the report.</summary>
    Deviated,
}

/// <summary>One adjustment made to a paper's recipe, and the rule that justifies it.</summary>
/// <param name="Setting">The hyperparameter adjusted, for example <c>LearningRate</c>.</param>
/// <param name="PaperValue">What the paper states.</param>
/// <param name="AppliedValue">What is actually being used.</param>
/// <param name="Rule">The published rule or precedent that justifies the adjustment.</param>
/// <remarks>
/// <see cref="Rule"/> is required by construction rather than optional. An adjustment without a
/// stated justification is indistinguishable from an arbitrary one, and arbitrary adjustment to a
/// published hyperparameter is how a reproduction quietly stops being a reproduction.
/// </remarks>
public sealed record RecipeAdaptation(string Setting, string PaperValue, string AppliedValue, string Rule)
{
    /// <inheritdoc />
    public override string ToString()
        => $"{Setting}: {PaperValue} -> {AppliedValue} ({Rule})";
}

/// <summary>
/// What a model's paper specifies for training, what is actually being used, and every difference
/// between the two.
/// </summary>
/// <remarks>
/// <para>
/// Libraries normally leave a paper's training recipe in prose — a model card, a README, a training
/// script — detached from the model, not applied, and silently replaced by a generic default. The
/// result is that a model can be an order of magnitude away from its published learning rate with
/// nothing anywhere saying so. That is the defect issue #1928 records, across 685 construction
/// sites.
/// </para>
/// <para>
/// This report exists so that never happens silently again. It answers three questions a user
/// otherwise cannot ask: what does the paper say, what am I actually training with, and if those
/// differ, why. A deviation the library cannot honour is listed rather than hidden.
/// </para>
/// <para><b>For Beginners:</b> Research papers describe exactly how they trained a model. This tells
/// you whether your model is training that way — and when it is not, what changed and on what
/// grounds. "Adapted" is normal and usually correct: a learning rate chosen for a batch of 4096
/// images has to be adjusted for a batch of 32, and the adjustment follows a published rule.
/// </para>
/// </remarks>
public sealed class TrainingRecipeReport
{
    /// <summary>The component this report describes, or empty for the model as a whole.</summary>
    /// <remarks>
    /// Papers for composite models state different settings per part — Stable Audio Open gives one
    /// rate for its autoencoder, another for its discriminators and a third for its DiT. A single
    /// per-model recipe cannot express that, so each component reports separately.
    /// </remarks>
    public string Component { get; init; } = string.Empty;

    /// <summary>The optimizer the paper specifies, or <see cref="OptimizerKind.Unspecified"/>.</summary>
    public OptimizerKind PaperOptimizer { get; init; }

    /// <summary>The optimizer actually constructed.</summary>
    public string AppliedOptimizer { get; init; } = string.Empty;

    /// <summary>Where in the paper the recipe comes from.</summary>
    public string Source { get; init; } = string.Empty;

    /// <summary>Adjustments made for this run's scale, each naming its justifying rule.</summary>
    public IReadOnlyList<RecipeAdaptation> Adaptations { get; init; } = [];

    /// <summary>Parts of the recipe that could not be applied, and why.</summary>
    public IReadOnlyList<string> Unhonoured { get; init; } = [];

    /// <summary>How faithfully the applied configuration reproduces the paper.</summary>
    public RecipeFidelity Fidelity
        => PaperOptimizer == OptimizerKind.Unspecified ? RecipeFidelity.NotDeclared
         : Unhonoured.Count > 0 ? RecipeFidelity.Deviated
         : Adaptations.Count > 0 ? RecipeFidelity.Adapted
         : RecipeFidelity.Exact;

    /// <summary>A one-line human-readable summary, suitable for logs and diagnostics.</summary>
    public string Describe()
    {
        if (Fidelity == RecipeFidelity.NotDeclared)
            return "No paper training recipe declared; library defaults in use.";

        string component = Component.Length > 0 ? $"[{Component}] " : string.Empty;
        string summary = $"{component}{Fidelity}: {PaperOptimizer} per {Source}";

        if (Adaptations.Count > 0)
            summary += $"; adapted {string.Join("; ", Adaptations)}";

        if (Unhonoured.Count > 0)
            summary += $"; NOT honoured: {string.Join("; ", Unhonoured)}";

        return summary;
    }

    /// <summary>The report for a model that declares nothing.</summary>
    public static TrainingRecipeReport NotDeclaredFor(string component = "")
        => new() { Component = component };
}
