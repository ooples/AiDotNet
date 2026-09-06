using System.Collections.Concurrent;
using System.Reflection;
using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.LearningRateSchedulers;
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
        IFullModel<T, TInput, TOutput> model)
    {
        if (model is null) return null;

        var recipe = Find(model);
        if (recipe is null) return null;

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

            // Unspecified, and optimizers with no gradient-based implementation here, fall through
            // to the caller's own default rather than being approximated by a different algorithm.
            _ => null,
        };
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
                double scaled = recipe.LearningRate * batch / recipe.ReferenceBatchSize;
                SetDouble(options, "InitialLearningRate", scaled);
            }
        }

    }


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
        if (recipe.WarmupSteps <= 0) return 0;

        int iterations = GetInt(options, "MaxIterations");
        if (iterations <= 0 || recipe.WarmupSteps < iterations) return recipe.WarmupSteps;

        return Math.Max(1, iterations / 10);
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
            recipe, baseRate, EffectiveWarmupSteps(options, recipe));
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
        PaperOptimizerAttribute recipe, double baseRate, int warmupSteps)
    {
        try
        {
            return recipe.Schedule switch
            {
                LearningRateSchedulerType.LinearWarmup when warmupSteps > 0
                    => new LinearWarmupScheduler(baseRate, warmupSteps),

                LearningRateSchedulerType.Exponential when !double.IsNaN(recipe.DecayRate)
                    => new ExponentialLRScheduler(baseRate, recipe.DecayRate),

                LearningRateSchedulerType.Step when recipe.StepSize > 0 && !double.IsNaN(recipe.DecayRate)
                    => new StepLRScheduler(baseRate, recipe.StepSize, recipe.DecayRate),

                _ => null,
            };
        }
        catch (Exception)
        {
            // A scheduler whose constructor rejects these arguments must not take the model's
            // construction down with it; falling back to a constant rate is recoverable, and the
            // declaration is still visible in source for whoever fixes it.
            return null;
        }
    }

    /// <summary>The declaration matching this model: variant-specific when one exists, else unkeyed.</summary>
    public static PaperOptimizerAttribute? Find(object? model)
    {
        if (model is null) return null;

        var declarations = _byModelType.GetOrAdd(
            model.GetType(),
            static type => (PaperOptimizerAttribute[])type
                .GetCustomAttributes(typeof(PaperOptimizerAttribute), inherit: true));

        if (declarations.Length == 0) return null;

        string? variant = (model as IPaperOptimizerVariant)?.PaperOptimizerVariant;

        PaperOptimizerAttribute? unkeyed = null;
        foreach (var declaration in declarations)
        {
            if (declaration.Optimizer == OptimizerKind.Unspecified) continue;

            if (string.IsNullOrEmpty(declaration.Variant))
            {
                unkeyed ??= declaration;
                continue;
            }

            if (!string.IsNullOrEmpty(variant)
                && string.Equals(declaration.Variant, variant, StringComparison.Ordinal))
            {
                return declaration;
            }
        }

        return unkeyed;
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
