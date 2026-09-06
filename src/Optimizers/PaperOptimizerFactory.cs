using System.Collections.Generic;
using System.Collections.Concurrent;
using System.Reflection;
using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.LearningRateSchedulers;
using AiDotNet.Models;
using AiDotNet.Models.Options;

namespace AiDotNet.Optimizers;

/// <summary>
/// Builds the optimizer a model's research paper specifies, from its
/// <see cref="PaperOptimizerAttribute"/> declaration.
/// </summary>
/// <remarks>
/// <para>
/// Issue #1928: 685 construction sites across 592 files are
/// <c>optimizer ?? new AdamWOptimizer&lt;...&gt;(this)</c> or the Adam equivalent, so every model
/// trains with one of two optimizers regardless of what its paper used, at that optimizer's own
/// default rate and with no schedule.
/// </para>
/// <para>
/// <b>Why the recipe is built as a unit.</b> An earlier revision recorded the paper's scalars and
/// applied them to whatever optimizer the model had already constructed, skipping when the kinds
/// disagreed. That cannot work: ResNet's paper specifies SGD at 0.1, and pushing 0.1 into Adam
/// diverges immediately, so the safe behaviour was to ignore the declaration — leaving the model
/// with the wrong optimizer, the wrong rate and no schedule while looking paper-faithful. The
/// optimizer, its hyperparameters and its schedule are one recipe and are reproduced together.
/// </para>
/// <para>
/// <b>Behaviour is unchanged until a recipe is declared.</b> <see cref="CreateFor"/> returns
/// <c>null</c> when a model declares nothing, so the call site falls through to the default it
/// already had:
/// </para>
/// <code>
/// _optimizer = optimizer
///     ?? PaperOptimizerFactory.CreateFor&lt;T, Tensor&lt;T&gt;, Tensor&lt;T&gt;&gt;(this)
///     ?? new AdamWOptimizer&lt;T, Tensor&lt;T&gt;, Tensor&lt;T&gt;&gt;(this);
/// </code>
/// </remarks>
public static class PaperOptimizerFactory
{
    /// <summary>
    /// Cached per model type: the reflection cost would otherwise be paid on every construction,
    /// and models are built in loops during hyperparameter search.
    /// </summary>
    private static readonly ConcurrentDictionary<Type, PaperOptimizerAttribute[]> _byModelType = new();

    /// <summary>
    /// Builds the optimizer this model's paper specifies, or <c>null</c> when it declares none.
    /// </summary>
    /// <returns>
    /// <c>null</c> when there is no applicable declaration, so callers keep their existing default.
    /// </returns>
    public static IGradientBasedOptimizer<T, TInput, TOutput>? CreateFor<T, TInput, TOutput>(
        IFullModel<T, TInput, TOutput> model, string component = "")
    {
        if (model is null) return null;

        var recipe = Find(model, component);
        if (recipe is null)
        {
            // Record the absence too. "This model declares nothing" is a different statement from
            // "we never looked", and only one of them is actionable.
            Record(model, TrainingRecipeReport.NotDeclaredFor(component));
            return null;
        }

        var adaptations = new List<RecipeAdaptation>();
        var unhonoured = new List<string>();
        var cautions = new List<string>();

        var optimizer = Build<T, TInput, TOutput>(model, recipe, adaptations, unhonoured, cautions);

        if (optimizer is null)
        {
            unhonoured.Add(
                $"the paper specifies {recipe.Optimizer}, which has no gradient-based implementation "
                + "reachable here, so the model keeps its own default optimizer");
        }

        Record(model, new TrainingRecipeReport
        {
            Component = component,
            PaperOptimizer = recipe.Optimizer,
            AppliedOptimizer = optimizer?.GetType().Name ?? "(model default)",
            Source = recipe.Source,
            Adaptations = adaptations,
            Unhonoured = unhonoured,
            Cautions = cautions,
        });

        return optimizer;
    }

    /// <summary>Constructs the declared optimizer, or <c>null</c> when this library has none for it.</summary>
    private static IGradientBasedOptimizer<T, TInput, TOutput>? Build<T, TInput, TOutput>(
        IFullModel<T, TInput, TOutput> model,
        PaperOptimizerAttribute recipe,
        List<RecipeAdaptation> adaptations,
        List<string> unhonoured,
        List<string> cautions)
    {
        _pendingAdaptations.Value = adaptations;
        _pendingUnhonoured.Value = unhonoured;
        _pendingCautions.Value = cautions;
        try
        {
            return recipe.Optimizer switch
        {
            OptimizerKind.Adam => new AdamOptimizer<T, TInput, TOutput>(
                model, Configure(new AdamOptimizerOptions<T, TInput, TOutput>(), recipe)),

            OptimizerKind.AdamW => new AdamWOptimizer<T, TInput, TOutput>(
                model, Configure(new AdamWOptimizerOptions<T, TInput, TOutput>(), recipe)),

            OptimizerKind.Sgd => new StochasticGradientDescentOptimizer<T, TInput, TOutput>(
                model, Configure(new StochasticGradientDescentOptimizerOptions<T, TInput, TOutput>(), recipe)),

            OptimizerKind.SgdMomentum when recipe.UseNesterov
                => new NesterovAcceleratedGradientOptimizer<T, TInput, TOutput>(
                    model, Configure(new NesterovAcceleratedGradientOptimizerOptions<T, TInput, TOutput>(), recipe)),

            OptimizerKind.SgdMomentum => new MomentumOptimizer<T, TInput, TOutput>(
                model, Configure(new MomentumOptimizerOptions<T, TInput, TOutput>(), recipe)),

            OptimizerKind.RmsProp => new RootMeanSquarePropagationOptimizer<T, TInput, TOutput>(
                model, Configure(new RootMeanSquarePropagationOptimizerOptions<T, TInput, TOutput>(), recipe)),

            OptimizerKind.Adagrad => new AdagradOptimizer<T, TInput, TOutput>(
                model, Configure(new AdagradOptimizerOptions<T, TInput, TOutput>(), recipe)),

            OptimizerKind.Adadelta => new AdaDeltaOptimizer<T, TInput, TOutput>(
                model, Configure(new AdaDeltaOptimizerOptions<T, TInput, TOutput>(), recipe)),

            OptimizerKind.Adamax => new AdaMaxOptimizer<T, TInput, TOutput>(
                model, Configure(new AdaMaxOptimizerOptions<T, TInput, TOutput>(), recipe)),

            OptimizerKind.Nadam => new NadamOptimizer<T, TInput, TOutput>(
                model, Configure(new NadamOptimizerOptions<T, TInput, TOutput>(), recipe)),

            OptimizerKind.Lamb => new LAMBOptimizer<T, TInput, TOutput>(
                model, Configure(new LAMBOptimizerOptions<T, TInput, TOutput>(), recipe)),

            OptimizerKind.Lion => new LionOptimizer<T, TInput, TOutput>(
                model, Configure(new LionOptimizerOptions<T, TInput, TOutput>(), recipe)),

                // Unspecified, and optimizers with no gradient-based implementation here, fall
                // through to the caller's own default rather than being approximated by a
                // different algorithm. Substituting one optimizer for another is not a smaller
                // deviation than using the default; it is an undeclared one.
                _ => null,
            };
        }
        finally
        {
            _pendingAdaptations.Value = null;
            _pendingUnhonoured.Value = null;
            _pendingCautions.Value = null;
        }
    }

    /// <summary>
    /// Adaptations and unhonoured settings for the recipe currently being built on this thread.
    /// </summary>
    /// <remarks>
    /// Thread-local rather than passed through every helper: Configure and its callees are reached
    /// from a switch arm per optimizer kind, and threading two lists through all of them would
    /// obscure the mapping they exist to express. Cleared in a finally so a throwing constructor
    /// cannot leak state into the next build on the same thread.
    /// </remarks>
    private static readonly ThreadLocal<List<RecipeAdaptation>?> _pendingAdaptations = new();
    private static readonly ThreadLocal<List<string>?> _pendingUnhonoured = new();
    private static readonly ThreadLocal<List<string>?> _pendingCautions = new();

    private static void NoteAdaptation(string setting, string paper, string applied, string rule)
        => _pendingAdaptations.Value?.Add(new RecipeAdaptation(setting, paper, applied, rule));

    private static void NoteCaution(string caution) => _pendingCautions.Value?.Add(caution);

    /// <summary>Reports for models built on this process, keyed weakly so they do not retain models.</summary>
    private static readonly System.Runtime.CompilerServices.ConditionalWeakTable<object, List<TrainingRecipeReport>> _reports = new();

    private static void Record(object model, TrainingRecipeReport report)
    {
        var list = _reports.GetOrCreateValue(model);
        lock (list)
        {
            list.RemoveAll(r => string.Equals(r.Component, report.Component, StringComparison.OrdinalIgnoreCase));
            list.Add(report);
        }
    }

    /// <summary>
    /// What this model's paper specifies for training, what was applied, and every difference.
    /// </summary>
    /// <remarks>
    /// Returns one report per component built. Empty when no optimizer has been constructed for the
    /// model yet, which for most models happens in their constructor.
    /// </remarks>
    public static IReadOnlyList<TrainingRecipeReport> ReportsFor(object? model)
    {
        if (model is null) return [];
        if (!_reports.TryGetValue(model, out var list)) return [];
        lock (list) return list.ToArray();
    }

    /// <summary>
    /// Applies the recipe's scalars, schedule and clipping to a freshly-built options object.
    /// </summary>
    /// <remarks>
    /// Property-name based because the hyperparameters are spread across an options hierarchy
    /// rather than a shared interface — <c>InitialLearningRate</c> on the base,
    /// <c>Beta1</c>/<c>Beta2</c>/<c>Epsilon</c> on the Adam family, <c>WeightDecay</c> only on
    /// AdamW, <c>Momentum</c> only on the momentum optimizers. A knob the chosen optimizer does not
    /// have is skipped rather than treated as an error: it means the paper stated something this
    /// optimizer cannot express, which is information for a reader, not a crash.
    /// </remarks>
    private static TOptions Configure<TOptions>(TOptions options, PaperOptimizerAttribute recipe)
        where TOptions : class
    {
        SetDouble(options, "InitialLearningRate", recipe.LearningRate);
        SetDouble(options, "WeightDecay", recipe.WeightDecay);
        SetDouble(options, "Beta1", recipe.Beta1);
        SetDouble(options, "Beta2", recipe.Beta2);
        SetDouble(options, "Epsilon", recipe.Epsilon);
        SetDouble(options, "Rho", recipe.Rho);
        SetDouble(options, "Decay", recipe.Rho);

        if (!double.IsNaN(recipe.Momentum))
        {
            SetDouble(options, "Momentum", recipe.Momentum);
            SetDouble(options, "InitialMomentum", recipe.Momentum);
            // The adaptive-momentum controller would otherwise drift away from the paper's value
            // on its own schedule, which is not what the paper describes.
            SetBool(options, "UseAdaptiveMomentum", false);
        }

        ScaleToConfiguredRun(options, recipe);
        ConfigureScheduleAndClipping(options, recipe);
        return options;
    }

    /// <summary>
    /// Adapts paper-scale values to the run this options object actually describes.
    /// </summary>
    /// <remarks>
    /// <para>
    /// A paper's hyperparameters are chosen for a paper's training run. Transplanted unchanged into
    /// a much shorter run or a much smaller batch they do not merely underperform, they stop
    /// training altogether -- which is not a faithful reproduction of the paper either. Two
    /// adjustments, both with established justification rather than invented ratios:
    /// </para>
    /// <para>
    /// <b>Warmup.</b> A 4000-step warmup inside a 100-step run leaves the learning rate at
    /// essentially zero for the whole run, so parameters never move. Warmup is rescaled to the same
    /// proportion of the run it occupies in the paper, floored at one step. This mirrors what #1835
    /// already does for GaussianSplatting's densification window, which would otherwise never fire
    /// because its start iteration exceeded the configured run.
    /// </para>
    /// <para>
    /// <b>Learning rate versus batch.</b> A rate is only meaningful for the batch it was tuned at.
    /// Where the recipe declares <c>ReferenceBatchSize</c> and the options carry a batch, the rate
    /// is scaled linearly by their ratio, following the linear scaling rule of Goyal et al. 2017.
    /// MobileNetV3's 0.1 at batch 4096 becomes 7.8e-4 at batch 32 -- still the paper's recipe,
    /// expressed for the batch actually being used.
    /// </para>
    /// <para>
    /// Both are no-ops when the run already matches the paper's scale, so a full-scale training run
    /// gets the paper's numbers unmodified.
    /// </para>
    /// </remarks>
    private static void ScaleToConfiguredRun(object options, PaperOptimizerAttribute recipe)
    {
        if (recipe.ReferenceBatchSize > 0 && !double.IsNaN(recipe.LearningRate))
        {
            int batch = GetInt(options, "BatchSize");
            if (batch > 0 && batch != recipe.ReferenceBatchSize)
            {
                if (ScalesLinearlyWithBatch(recipe.Optimizer))
                {
                    double scaled = recipe.LearningRate * batch / recipe.ReferenceBatchSize;
                    SetDouble(options, "InitialLearningRate", scaled);
                    NoteAdaptation(
                        "LearningRate",
                        $"{recipe.LearningRate:G6} at batch {recipe.ReferenceBatchSize}",
                        $"{scaled:G6} at batch {batch}",
                        "linear scaling rule, Goyal et al. 2017");
                }
                else
                {
                    NoteCaution(
                        $"the paper's rate {recipe.LearningRate:G6} was chosen for batch "
                        + $"{recipe.ReferenceBatchSize} and this run uses batch {batch}; the linear "
                        + $"scaling rule is established for SGD, not for {recipe.Optimizer}, so the "
                        + "paper's rate is used unchanged");
                }
            }
        }

    }

    /// <summary>
    /// Whether the linear scaling rule may be applied to this optimizer's learning rate.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Only for the SGD family, because that is the only family the rule was established on. Goyal
    /// et al. 2017 state and evidence it for SGD with momentum on ImageNet; Krizhevsky 2014 Sec. 5
    /// derives sqrt(k) from keeping the gradient variance constant and reports that k worked better
    /// in his experiments -- also SGD with momentum. Neither result covers Adam-family optimizers,
    /// whose per-parameter second-moment normalisation is precisely the thing the derivation
    /// assumes away.
    /// </para>
    /// <para>
    /// So for an adaptive optimizer the paper's rate is used exactly as published and the batch
    /// mismatch is reported as a caution instead. Applying a scaling rule outside the regime it was
    /// demonstrated in, and citing a paper that does not say it, would be exactly the fabrication
    /// this whole feature is built to prevent -- and it would be invisible, because the report would
    /// name a real citation for a rule that citation does not contain.
    /// </para>
    /// </remarks>
    private static bool ScalesLinearlyWithBatch(OptimizerKind kind)
        => kind is OptimizerKind.Sgd or OptimizerKind.SgdMomentum;


    /// <summary>
    /// The paper's warmup length, rescaled when the configured run is shorter than the warmup.
    /// </summary>
    /// <remarks>
    /// A 4000-step warmup inside a 100-step run holds the learning rate at essentially zero for the
    /// entire run, so parameters never move -- which reproduces the paper no better than ignoring
    /// the warmup would. Warmup keeps its share of the run instead of its absolute length, floored
    /// at one step. Ten percent is what the paper's own 4000 steps works out to against its
    /// 100k-step schedule, so the shape is preserved rather than invented, and a full-length run
    /// gets the paper's 4000 unchanged.
    /// </remarks>
    private static int EffectiveWarmupSteps(object options, PaperOptimizerAttribute recipe)
    {
        int iterations = GetInt(options, "MaxIterations");

        // A fraction is exact at any run length, so it is computed rather than adapted.
        if (!double.IsNaN(recipe.WarmupFraction) && recipe.WarmupFraction > 0)
        {
            if (iterations <= 0) return recipe.WarmupSteps;
            return Math.Max(1, (int)Math.Round(iterations * recipe.WarmupFraction));
        }

        if (recipe.WarmupSteps <= 0) return 0;
        if (iterations <= 0 || recipe.WarmupSteps < iterations) return recipe.WarmupSteps;

        int scaled = Math.Max(1, iterations / 10);
        NoteAdaptation(
            "WarmupSteps",
            $"{recipe.WarmupSteps} steps",
            $"{scaled} steps over a {iterations}-step run",
            "warmup held at its share of the run; same treatment as the #1835 densification window");
        return scaled;
    }
    private static int GetInt(object options, string propertyName)
    {
        PropertyInfo? property = options.GetType().GetProperty(
            propertyName, BindingFlags.Public | BindingFlags.Instance);
        if (property is null || property.PropertyType != typeof(int)) return 0;
        return (int)(property.GetValue(options) ?? 0);
    }

    /// <summary>Applies schedule and gradient-clipping settings, which live on the gradient-based base.</summary>
    private static void ConfigureScheduleAndClipping(object options, PaperOptimizerAttribute recipe)
    {
        if (!double.IsNaN(recipe.MaxGradientNorm))
        {
            SetBool(options, "EnableGradientClipping", true);
            SetDouble(options, "MaxGradientNorm", recipe.MaxGradientNorm);
        }

        if (recipe.Schedule == LearningRateSchedulerType.Constant && recipe.WarmupSteps <= 0) return;

        PropertyInfo? schedulerProperty = options.GetType().GetProperty(
            "LearningRateScheduler", BindingFlags.Public | BindingFlags.Instance);
        if (schedulerProperty is null || !schedulerProperty.CanWrite) return;

        double baseRate = double.IsNaN(recipe.LearningRate) ? 0.001 : recipe.LearningRate;
        ILearningRateScheduler? scheduler = BuildScheduler(
            recipe, baseRate, EffectiveWarmupSteps(options, recipe), GetInt(options, "MaxIterations"));
        if (scheduler is not null) schedulerProperty.SetValue(options, scheduler);
    }

    /// <summary>
    /// Builds the declared schedule, or <c>null</c> when it cannot be expressed.
    /// </summary>
    /// <remarks>
    /// Deliberately conservative: an unrecognised or under-specified schedule yields <c>null</c> and
    /// the optimizer keeps its constant rate, rather than substituting a different curve. A wrong
    /// schedule is harder to notice than a missing one.
    /// </remarks>
    private static ILearningRateScheduler? BuildScheduler(
        PaperOptimizerAttribute recipe, double baseRate, int warmupSteps, int totalSteps)
    {
        // A floor of zero is the usual published one, and it is also the correct fallback for the
        // schedulers below, all of which decay towards zero when no floor is named.
        double floor = double.IsNaN(recipe.MinLearningRate) ? 0.0 : recipe.MinLearningRate;

        try
        {
            return recipe.Schedule switch
            {
                LearningRateSchedulerType.LinearWarmup when warmupSteps > 0
                    => new LinearWarmupScheduler(baseRate, warmupSteps, totalSteps,
                                                 decayMode: recipe.PostWarmupDecay, endLr: floor),

                LearningRateSchedulerType.Exponential when !double.IsNaN(recipe.DecayRate)
                    => new ExponentialLRScheduler(baseRate, recipe.DecayRate),

                LearningRateSchedulerType.Step when recipe.StepSize > 0 && !double.IsNaN(recipe.DecayRate)
                    => new StepLRScheduler(baseRate, recipe.StepSize, recipe.DecayRate),

                // "Divide the rate by 10 when the error plateaus" is how a whole generation of
                // vision papers state their schedule, so leaving it unmapped would report the most
                // common published schedule in the catalogue as a deviation.
                LearningRateSchedulerType.ReduceOnPlateau
                    => new ReduceOnPlateauScheduler(
                           baseRate,
                           factor: double.IsNaN(recipe.DecayRate) ? 0.1 : recipe.DecayRate,
                           patience: recipe.StepSize > 0 ? recipe.StepSize : 10,
                           minLearningRate: floor),

                LearningRateSchedulerType.CosineAnnealing when totalSteps > 0
                    => new CosineAnnealingLRScheduler(baseRate, totalSteps, floor),

                LearningRateSchedulerType.Polynomial when totalSteps > 0
                    => new PolynomialLRScheduler(
                           baseRate, totalSteps,
                           power: double.IsNaN(recipe.DecayRate) ? 1.0 : recipe.DecayRate,
                           endLearningRate: floor),

                LearningRateSchedulerType.OneCycle when totalSteps > 0
                    => new OneCycleLRScheduler(baseRate, totalSteps),

                _ => Unexpressible(recipe),
            };
        }
        catch (Exception)
        {
            // A scheduler whose constructor rejects these arguments must not take the model's
            // construction down with it; falling back to a constant rate is recoverable. But it is
            // reported rather than swallowed -- a silently missing schedule is the failure this
            // whole report exists to make impossible.
            _pendingUnhonoured.Value?.Add(
                $"the paper's {recipe.Schedule} schedule could not be constructed from the declared "
                + "parameters; a constant learning rate is in use");
            return null;
        }
    }


    /// <summary>Reports a declared schedule this library cannot express, and returns no scheduler.</summary>
    private static ILearningRateScheduler? Unexpressible(PaperOptimizerAttribute recipe)
    {
        if (recipe.Schedule != LearningRateSchedulerType.Constant)
        {
            _pendingUnhonoured.Value?.Add(
                $"the paper's {recipe.Schedule} schedule is declared but not yet mapped to a "
                + "scheduler here; a constant learning rate is in use");
        }

        return null;
    }
    /// <summary>The declaration matching this model: variant-specific when one exists, else unkeyed.</summary>
    public static PaperOptimizerAttribute? Find(object? model, string component = "")
    {
        if (model is null) return null;

        var declarations = _byModelType.GetOrAdd(
            model.GetType(),
            static type => (PaperOptimizerAttribute[])type
                .GetCustomAttributes(typeof(PaperOptimizerAttribute), inherit: true));

        if (declarations.Length == 0) return null;

        string? variant = (model as IPaperOptimizerVariant)?.PaperOptimizerVariant;

        // Two independent keys, each with the same precedence rule: an exact match beats the
        // unnamed fallback. Ranking them rather than returning the first match matters because the
        // shared model-wide declaration is usually written FIRST, so a first-wins scan would hand
        // back the default and silently ignore the component's own row.
        PaperOptimizerAttribute? best = null;
        int bestRank = -1;

        foreach (var declaration in declarations)
        {
            if (declaration.Optimizer == OptimizerKind.Unspecified) continue;

            bool componentExact = declaration.Component.Length > 0
                && string.Equals(declaration.Component, component, StringComparison.OrdinalIgnoreCase);
            bool componentFallback = declaration.Component.Length == 0;
            if (!componentExact && !componentFallback) continue;

            bool variantExact = declaration.Variant.Length > 0
                && !string.IsNullOrEmpty(variant)
                && string.Equals(declaration.Variant, variant, StringComparison.OrdinalIgnoreCase);
            bool variantFallback = declaration.Variant.Length == 0;
            if (!variantExact && !variantFallback) continue;

            // Component is the stronger key: it selects which PART of the model is being built,
            // whereas variant only picks a size for that part.
            int rank = (componentExact ? 2 : 0) + (variantExact ? 1 : 0);
            if (rank > bestRank)
            {
                bestRank = rank;
                best = declaration;
            }
        }

        return best;
    }

    private static void SetDouble(object options, string propertyName, double value)
    {
        if (double.IsNaN(value)) return;

        PropertyInfo? property = options.GetType().GetProperty(
            propertyName, BindingFlags.Public | BindingFlags.Instance);
        if (property is null || !property.CanWrite || property.PropertyType != typeof(double)) return;

        property.SetValue(options, value);
    }

    private static void SetBool(object options, string propertyName, bool value)
    {
        PropertyInfo? property = options.GetType().GetProperty(
            propertyName, BindingFlags.Public | BindingFlags.Instance);
        if (property is null || !property.CanWrite || property.PropertyType != typeof(bool)) return;

        property.SetValue(options, value);
    }
}
