using System.Collections.Concurrent;
using System.Reflection;
using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Interfaces;

namespace AiDotNet.Optimizers;

/// <summary>
/// Seeds a freshly-constructed optimizer options object from the model's
/// <see cref="PaperOptimizerAttribute"/> declaration, so a model that constructs its optimizer with
/// no options trains at its paper's settings rather than the optimizer class's generic defaults.
/// </summary>
/// <remarks>
/// <para>
/// Issue #1928: 685 construction sites across 592 files are
/// <c>optimizer ?? new XOptimizer&lt;...&gt;(this)</c>, supplying no options at all. Every one of
/// them already passes the model, which is what makes a central fix possible — the defaults can be
/// resolved here instead of editing 592 files.
/// </para>
/// <para>
/// It is not only the learning rate. <c>AdamWOptimizerOptions</c> also contributes
/// <c>WeightDecay = 0.01</c>, <c>Beta1 = 0.9</c>, <c>Beta2 = 0.999</c> and <c>Epsilon = 1e-8</c>.
/// Commit <c>1972a510a</c> fixed exactly that in <c>SpanBasedNERBase</c>, which "silently took
/// AdamW's own default decoupled weight decay of 0.01 and applied it to every parameter on every
/// step" where the paper specifies plain Adam with no such decay.
/// </para>
/// <para>
/// <b>Precedence.</b> Caller-supplied options are returned untouched — this only fills in the
/// no-options case, so <c>ConfigureOptimizer</c> always wins.
/// </para>
/// </remarks>
public static class PaperOptimizerDefaults
{
    /// <summary>
    /// Cached per model type, because the reflection cost would otherwise be paid on every
    /// optimizer construction, and models are constructed in loops during hyperparameter search.
    /// </summary>
    /// <remarks>
    /// Reflection rather than a generated table on purpose: the attribute is the single source of
    /// truth, so reading it directly cannot drift from what the model declares. A generator still
    /// validates the declarations at compile time and reports the undeclared backlog, but it does
    /// not own the values.
    /// </remarks>
    private static readonly ConcurrentDictionary<Type, PaperOptimizerAttribute[]> _byModelType = new();

    /// <summary>
    /// Returns the caller's options when supplied, otherwise a fresh options object seeded from the
    /// model's paper declaration.
    /// </summary>
    /// <typeparam name="TOptions">The concrete optimizer options type.</typeparam>
    /// <param name="model">The model the optimizer was constructed for. May be <c>null</c>.</param>
    /// <param name="callerOptions">Options the caller supplied, or <c>null</c>.</param>
    /// <param name="expected">
    /// The optimizer this call site actually constructs. A declaration naming a different optimizer
    /// is ignored rather than applied, since a learning rate chosen for SGD is not a learning rate
    /// for Adam — transplanting one across optimizers would be worse than the default it replaced.
    /// </param>
    public static TOptions Resolve<TOptions>(object? model, TOptions? callerOptions, OptimizerKind expected)
        where TOptions : class, new()
    {
        if (callerOptions is not null) return callerOptions;

        var options = new TOptions();
        Apply(model, options, expected);
        return options;
    }

    /// <summary>
    /// Applies the model's paper hyperparameters to an already-constructed options object.
    /// </summary>
    /// <returns>The declaration that was applied, or <c>null</c> when nothing applied.</returns>
    public static PaperOptimizerAttribute? Apply(object? model, object options, OptimizerKind expected)
    {
        if (model is null || options is null) return null;

        var declaration = Find(model, expected);
        if (declaration is null) return null;

        // InitialLearningRate is virtual on OptimizationAlgorithmOptions (0.01) and overridden by
        // the Adam-family options (0.001), so setting it by name reaches whichever the model got.
        ApplyScalar(options, "InitialLearningRate", declaration.LearningRate);
        ApplyScalar(options, "WeightDecay", declaration.WeightDecay);
        ApplyScalar(options, "Beta1", declaration.Beta1);
        ApplyScalar(options, "Beta2", declaration.Beta2);
        ApplyScalar(options, "Epsilon", declaration.Epsilon);

        return declaration;
    }

    /// <summary>
    /// The declaration matching this model and optimizer: the variant-specific one when the model
    /// exposes a variant and an entry exists for it, otherwise the unkeyed one.
    /// </summary>
    public static PaperOptimizerAttribute? Find(object? model, OptimizerKind expected)
    {
        if (model is null || expected == OptimizerKind.Unspecified) return null;

        var declarations = _byModelType.GetOrAdd(
            model.GetType(),
            static type => (PaperOptimizerAttribute[])type
                .GetCustomAttributes(typeof(PaperOptimizerAttribute), inherit: true));

        if (declarations.Length == 0) return null;

        string? variant = (model as IPaperOptimizerVariant)?.PaperOptimizerVariant;

        PaperOptimizerAttribute? unkeyed = null;
        foreach (var declaration in declarations)
        {
            if (declaration.Optimizer != expected) continue;
            if (!declaration.DeclaresAnyHyperparameter) continue;

            if (string.IsNullOrEmpty(declaration.Variant))
            {
                unkeyed ??= declaration;
                continue;
            }

            if (!string.IsNullOrEmpty(variant)
                && string.Equals(declaration.Variant, variant, StringComparison.Ordinal))
            {
                // A variant-specific declaration is more specific than the unkeyed fallback, so it
                // wins immediately rather than waiting to see what else is declared.
                return declaration;
            }
        }

        return unkeyed;
    }

    /// <summary>
    /// Assigns a declared value to a settable double property when the options type has one.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Property-name based because the hyperparameters are spread across an options hierarchy
    /// rather than a shared interface: <c>InitialLearningRate</c> is on
    /// <c>OptimizationAlgorithmOptions</c>, <c>Beta1</c>/<c>Beta2</c>/<c>Epsilon</c> on the
    /// Adam-family types, and <c>WeightDecay</c> only on AdamW. Introducing an interface would mean
    /// editing every options class, which is the file-count problem this design exists to avoid.
    /// </para>
    /// <para>
    /// A missing property is silently skipped, not an error: declaring <c>WeightDecay</c> on a model
    /// whose optimizer has no such knob means the paper stated something this optimizer cannot
    /// express, which is information for a reader rather than a runtime failure.
    /// </para>
    /// </remarks>
    private static void ApplyScalar(object options, string propertyName, double value)
    {
        if (double.IsNaN(value)) return;

        PropertyInfo? property = options.GetType().GetProperty(
            propertyName, BindingFlags.Public | BindingFlags.Instance);

        if (property is null || !property.CanWrite || property.PropertyType != typeof(double)) return;

        property.SetValue(options, value);
    }
}
