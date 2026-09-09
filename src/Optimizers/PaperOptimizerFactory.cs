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
/// <remarks>
/// Internal by design: applications configure optimizers through the model and facade API, and
/// models reach this only to choose their paper-faithful default when the caller supplied none.
/// InternalsVisibleTo already exposes it to the test assembly.
/// </remarks>
internal static class PaperOptimizerFactory
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
    internal static IGradientBasedOptimizer<T, TInput, TOutput>? CreateFor<T, TInput, TOutput>(
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

    /// <summary>
    /// Builds one complete, already-configured recipe without going through attribute lookup.
    /// </summary>
    /// <remarks>
    /// Kept separate from reflection-based selection so every optimizer kind and every scheduler
    /// contract can be exercised directly against a synthetic recipe, rather than only through
    /// whichever models happen to declare one.
    /// </remarks>
    internal static IGradientBasedOptimizer<T, TInput, TOutput> CreateFromRecipe<T, TInput, TOutput>(
        IFullModel<T, TInput, TOutput> model,
        PaperOptimizerAttribute recipe)
    {
        if (model is null) throw new ArgumentNullException(nameof(model));
        if (recipe is null) throw new ArgumentNullException(nameof(recipe));

        var built = Build<T, TInput, TOutput>(
            model, recipe, new List<RecipeAdaptation>(), new List<string>(), new List<string>());

        return built ?? throw new NotSupportedException(
            "The declared paper optimizer " + recipe.Optimizer
                + " has no gradient-based implementation here.");
    }

    /// <summary>Builds the scheduler a recipe declares, for a run whose length is not known.</summary>
    /// <remarks>
    /// The two-argument form, for callers and tests that hold only the recipe. Schedules needing
    /// a horizon degrade exactly as they would in a real short run.
    /// </remarks>
    internal static ILearningRateScheduler? BuildScheduler(PaperOptimizerAttribute recipe, double baseRate)
    {
        if (recipe is null) throw new ArgumentNullException(nameof(recipe));

        ValidateSchedule(recipe);

        int warmupSteps = recipe.WarmupSteps > 0 ? recipe.WarmupSteps : 0;
        var scheduler = BuildScheduler(recipe, baseRate, warmupSteps, totalSteps: 0, modelDimension: 0);
        return ComposeWarmup(scheduler, recipe, baseRate, warmupSteps);
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

        // Typed rather than reflective. Setting a value by name cannot see the flag that governs
        // it, and several optimizers here pair a hyperparameter with an adapt-during-training
        // switch that is ON by default: Adam and AdamW clamp their running betas into
        // [MinBeta1, MaxBeta1] every step under UseAdaptiveBetas, which silently rewrote the
        // beta1 of 0.5 that MelGAN declares to 0.8, while the report still read Exact. A typed
        // switch makes each pairing explicit, and makes a renamed property a compile error
        // rather than a silent no-op.
        TOptions Configured<TOptions>(TOptions options)
            where TOptions : GradientBasedOptimizerOptions<T, TInput, TOutput>
        {
            if (!double.IsNaN(recipe.LearningRate)) options.InitialLearningRate = recipe.LearningRate;

            if (!double.IsNaN(recipe.Momentum))
            {
                options.InitialMomentum = recipe.Momentum;
                options.UseAdaptiveMomentum = false;
            }

            switch (options)
            {
                // Adam8Bit derives from the Adam options, so it must precede the Adam case.
                case Adam8BitOptimizerOptions<T, TInput, TOutput> adam8Bit:
                    ApplyAdamFamily(adam8Bit, recipe);
                    break;
                case AdamOptimizerOptions<T, TInput, TOutput> adam:
                    ApplyAdamFamily(adam, recipe);
                    break;
                // AdamW options do not derive from Adam options, so this cannot share the helper.
                case AdamWOptimizerOptions<T, TInput, TOutput> adamW:
                    if (!double.IsNaN(recipe.Beta1)) adamW.Beta1 = recipe.Beta1;
                    if (!double.IsNaN(recipe.Beta2)) adamW.Beta2 = recipe.Beta2;
                    if (!double.IsNaN(recipe.Epsilon)) adamW.Epsilon = recipe.Epsilon;
                    if (!double.IsNaN(recipe.WeightDecay)) adamW.WeightDecay = recipe.WeightDecay;
                    if (!double.IsNaN(recipe.Beta1) || !double.IsNaN(recipe.Beta2))
                        adamW.UseAdaptiveBetas = false;
                    break;
                case RootMeanSquarePropagationOptimizerOptions<T, TInput, TOutput> rmsProp:
                    if (!double.IsNaN(recipe.Rho)) rmsProp.Decay = recipe.Rho;
                    if (!double.IsNaN(recipe.Epsilon)) rmsProp.Epsilon = recipe.Epsilon;
                    break;
                case AdagradOptimizerOptions<T, TInput, TOutput> adagrad:
                    if (!double.IsNaN(recipe.Epsilon)) adagrad.Epsilon = recipe.Epsilon;
                    break;
                case AdaDeltaOptimizerOptions<T, TInput, TOutput> adadelta:
                    if (!double.IsNaN(recipe.Rho))
                    {
                        adadelta.Rho = recipe.Rho;
                        adadelta.UseAdaptiveRho = false;
                    }
                    if (!double.IsNaN(recipe.Epsilon)) adadelta.Epsilon = recipe.Epsilon;
                    break;
                case AdaMaxOptimizerOptions<T, TInput, TOutput> adamax:
                    if (!double.IsNaN(recipe.Beta1)) adamax.Beta1 = recipe.Beta1;
                    if (!double.IsNaN(recipe.Beta2)) adamax.Beta2 = recipe.Beta2;
                    if (!double.IsNaN(recipe.Epsilon)) adamax.Epsilon = recipe.Epsilon;
                    break;
                case NadamOptimizerOptions<T, TInput, TOutput> nadam:
                    if (!double.IsNaN(recipe.Beta1)) nadam.Beta1 = recipe.Beta1;
                    if (!double.IsNaN(recipe.Beta2)) nadam.Beta2 = recipe.Beta2;
                    if (!double.IsNaN(recipe.Epsilon)) nadam.Epsilon = recipe.Epsilon;
                    break;
                case LAMBOptimizerOptions<T, TInput, TOutput> lamb:
                    if (!double.IsNaN(recipe.Beta1)) lamb.Beta1 = recipe.Beta1;
                    if (!double.IsNaN(recipe.Beta2)) lamb.Beta2 = recipe.Beta2;
                    if (!double.IsNaN(recipe.Epsilon)) lamb.Epsilon = recipe.Epsilon;
                    if (!double.IsNaN(recipe.WeightDecay)) lamb.WeightDecay = recipe.WeightDecay;
                    break;
                case LionOptimizerOptions<T, TInput, TOutput> lion:
                    if (!double.IsNaN(recipe.Beta1))
                    {
                        lion.Beta1 = recipe.Beta1;
                        lion.UseAdaptiveBeta1 = false;
                    }
                    if (!double.IsNaN(recipe.Beta2))
                    {
                        lion.Beta2 = recipe.Beta2;
                        lion.UseAdaptiveBeta2 = false;
                    }
                    if (!double.IsNaN(recipe.WeightDecay)) lion.WeightDecay = recipe.WeightDecay;
                    break;
            }

            ScaleToConfiguredRun(options, recipe);
            ConfigureScheduleAndClipping(options, recipe);
            return options;
        }

        try
        {
            return recipe.Optimizer switch
        {
            OptimizerKind.Adam => new AdamOptimizer<T, TInput, TOutput>(
                model, Configured(new AdamOptimizerOptions<T, TInput, TOutput>())),

            OptimizerKind.AdamW => new AdamWOptimizer<T, TInput, TOutput>(
                model, Configured(new AdamWOptimizerOptions<T, TInput, TOutput>())),

            OptimizerKind.Sgd => new StochasticGradientDescentOptimizer<T, TInput, TOutput>(
                model, Configured(new StochasticGradientDescentOptimizerOptions<T, TInput, TOutput>())),

            OptimizerKind.SgdMomentum when recipe.UseNesterov
                => new NesterovAcceleratedGradientOptimizer<T, TInput, TOutput>(
                    model, Configured(new NesterovAcceleratedGradientOptimizerOptions<T, TInput, TOutput>())),

            OptimizerKind.SgdMomentum => new MomentumOptimizer<T, TInput, TOutput>(
                model, Configured(new MomentumOptimizerOptions<T, TInput, TOutput>())),

            OptimizerKind.RmsProp => new RootMeanSquarePropagationOptimizer<T, TInput, TOutput>(
                model, Configured(new RootMeanSquarePropagationOptimizerOptions<T, TInput, TOutput>())),

            OptimizerKind.Adagrad => new AdagradOptimizer<T, TInput, TOutput>(
                model, Configured(new AdagradOptimizerOptions<T, TInput, TOutput>())),

            OptimizerKind.Adadelta => new AdaDeltaOptimizer<T, TInput, TOutput>(
                model, Configured(new AdaDeltaOptimizerOptions<T, TInput, TOutput>())),

            OptimizerKind.Adamax => new AdaMaxOptimizer<T, TInput, TOutput>(
                model, Configured(new AdaMaxOptimizerOptions<T, TInput, TOutput>())),

            OptimizerKind.Nadam => new NadamOptimizer<T, TInput, TOutput>(
                model, Configured(new NadamOptimizerOptions<T, TInput, TOutput>())),

            OptimizerKind.Adam8Bit => new Adam8BitOptimizer<T, TInput, TOutput>(
                model, Configured(new Adam8BitOptimizerOptions<T, TInput, TOutput>())),

            OptimizerKind.LBfgs => new LBFGSOptimizer<T, TInput, TOutput>(
                model, Configured(new LBFGSOptimizerOptions<T, TInput, TOutput>())),

            OptimizerKind.Lamb => new LAMBOptimizer<T, TInput, TOutput>(
                model, Configured(new LAMBOptimizerOptions<T, TInput, TOutput>())),

            OptimizerKind.Lion => new LionOptimizer<T, TInput, TOutput>(
                model, Configured(new LionOptimizerOptions<T, TInput, TOutput>())),

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
    /// <summary>
    /// Checks an optimizer the model built itself against its declared paper recipe, records the
    /// result, and returns that same optimizer unchanged.
    /// </summary>
    /// <remarks>
    /// <para>
    /// For models that already implement their paper by hand. Some of them are MORE faithful than
    /// the factory could be: Transformer builds a Noam schedule from the architecture's own model
    /// dimension, which a recipe cannot see, and PANNs deliberately turns off the gradient clipping
    /// the options default to because the paper does not clip. Replacing either would lose the
    /// thing that made it correct, so nothing here is replaced.
    /// </para>
    /// <para>
    /// The declaration becomes an assertion instead. Every value the paper states is compared with
    /// what the model actually built, and a difference is reported as a deviation naming both. That
    /// is strictly stronger than the factory path: there the recipe IS the configuration and cannot
    /// disagree with itself, whereas here a later edit that quietly moves a rate away from its
    /// published value is caught and named.
    /// </para>
    /// </remarks>
    internal static IGradientBasedOptimizer<T, TInput, TOutput> VerifyHandBuilt<T, TInput, TOutput>(
        IFullModel<T, TInput, TOutput> model,
        IGradientBasedOptimizer<T, TInput, TOutput> optimizer,
        string component = "")
    {
        if (optimizer is null)
        {
            // Nothing to verify and nothing to hand back; a caller reaching here with no
            // optimizer has a bug worth surfacing rather than a state worth tolerating.
            throw new ArgumentNullException(
                nameof(optimizer), "VerifyHandBuilt checks an optimizer the model already built.");
        }

        if (model is null) return optimizer;

        var recipe = Find(model, component);
        if (recipe is null)
        {
            Record(model, TrainingRecipeReport.NotDeclaredFor(component));
            return optimizer;
        }

        var mismatches = new List<string>();
        object? options = null;
        try { options = optimizer.GetOptions(); }
        catch (Exception)
        {
            // An options accessor that throws must not take the model down; it only means this
            // recipe cannot be checked, which is itself worth reporting.
            mismatches.Add("the built optimizer did not expose its options, so the recipe could not be verified");
        }

        if (options is not null)
        {
            Compare(options, "InitialLearningRate", recipe.LearningRate, "LearningRate", mismatches);
            Compare(options, "Beta1", recipe.Beta1, "Beta1", mismatches);
            Compare(options, "Beta2", recipe.Beta2, "Beta2", mismatches);
            Compare(options, "Epsilon", recipe.Epsilon, "Epsilon", mismatches);
            Compare(options, "WeightDecay", recipe.WeightDecay, "WeightDecay", mismatches);
            Compare(options, "MaxGradientNorm", recipe.MaxGradientNorm, "MaxGradientNorm", mismatches);
        }

        Record(model, new TrainingRecipeReport
        {
            Component = component,
            PaperOptimizer = recipe.Optimizer,
            AppliedOptimizer = optimizer.GetType().Name,
            Source = recipe.Source,
            Unhonoured = mismatches,
        });

        return optimizer;
    }

    /// <summary>Applies the Adam-family moments, and stops them being adapted away.</summary>
    /// <remarks>
    /// UseAdaptiveBetas defaults to true and clamps the running betas into [MinBeta1, MaxBeta1]
    /// on every step, so a paper value outside that band is silently replaced. Declaring a beta
    /// therefore also means declaring that it does not move.
    /// </remarks>
    private static void ApplyAdamFamily<T, TInput, TOutput>(
        AdamOptimizerOptions<T, TInput, TOutput> options, PaperOptimizerAttribute recipe)
    {
        if (!double.IsNaN(recipe.Beta1)) options.Beta1 = recipe.Beta1;
        if (!double.IsNaN(recipe.Beta2)) options.Beta2 = recipe.Beta2;
        if (!double.IsNaN(recipe.Epsilon)) options.Epsilon = recipe.Epsilon;
        if (!double.IsNaN(recipe.Beta1) || !double.IsNaN(recipe.Beta2))
            options.UseAdaptiveBetas = false;
    }

    /// <summary>Reports a hand-built value that disagrees with the paper.</summary>
    /// <remarks>
    /// An unstated value is skipped rather than compared against zero: a paper that says nothing
    /// about weight decay is not a paper that says zero, and treating the two alike would report a
    /// deviation on every model that simply declares less than the whole recipe.
    /// </remarks>
    private static void Compare(
        object options, string propertyName, double declared, string label, List<string> mismatches)
    {
        if (double.IsNaN(declared)) return;

        PropertyInfo? property = options.GetType().GetProperty(
            propertyName, BindingFlags.Public | BindingFlags.Instance);
        if (property is null || property.PropertyType != typeof(double)) return;

        double actual = (double)(property.GetValue(options) ?? 0.0);
        if (Math.Abs(actual - declared) <= Math.Abs(declared) * 1e-9) return;

        mismatches.Add($"the paper states {label} {declared:G6} but this model builds {actual:G6}");
    }

    internal static IReadOnlyList<TrainingRecipeReport> ReportsFor(object? model)
    {
        if (model is null) return [];
        if (!_reports.TryGetValue(model, out var list)) return [];
        lock (list) return list.ToArray();
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
    /// <summary>
    /// Puts the optimizer on a per-step schedule cadence, which is the one every published recipe
    /// here is written in.
    /// </summary>
    /// <param name="options">The optimizer options a schedule was just attached to.</param>
    /// <remarks>
    /// <para>
    /// <c>SchedulerStepMode</c> defaults to <c>StepPerEpoch</c>, and
    /// <c>GradientBasedOptimizerBase.OnBatchEnd</c> advances the schedule only under
    /// <c>StepPerBatch</c> (or during warmup under <c>WarmupThenEpoch</c>). Tape training signals
    /// batch ends and nothing on the <c>Train</c> path raises an epoch, so a schedule attached from
    /// a recipe was installed and then never advanced: the optimizer kept whatever rate the
    /// schedule reports at step 0, for the whole run.
    /// </para>
    /// <para>
    /// That is silent for a decaying schedule, which merely trains at its peak rate throughout, and
    /// destructive for one that ramps up. ECAPA-TDNN declares Cyclic Triangular2 between
    /// MinLearningRate 1e-8 and 1e-3; a cyclic schedule starts at its base, so the model trained at
    /// 1e-8 forever and its memorization probe moved loss from 0.438444 to 0.438423 across 100
    /// steps. Noam and LinearWarmup have the same shape and the same exposure.
    /// </para>
    /// <para>
    /// Every schedule these recipes can express is stated in steps -- StepSize, WarmupSteps, Noam's
    /// t, milestone fractions of a step budget -- so per-batch is the cadence they were written for.
    /// Set only when the factory itself attaches a schedule, so an optimizer configured by hand
    /// keeps the library default.
    /// </para>
    /// </remarks>
    private static void StepPerBatch(object options)
    {
        PropertyInfo? mode = options.GetType().GetProperty(
            "SchedulerStepMode", BindingFlags.Public | BindingFlags.Instance);

        if (mode is not null && mode.CanWrite && mode.PropertyType == typeof(SchedulerStepMode))
        {
            mode.SetValue(options, SchedulerStepMode.StepPerBatch);
        }
    }

    /// <summary>
    /// The half-cycle a cyclic schedule should actually use: the published one, or a proportional
    /// share of the run when the published one is longer than the run itself.
    /// </summary>
    /// <param name="declaredStepSize">The half-cycle the paper states, in steps.</param>
    /// <param name="totalSteps">The run length, or 0 when it is not known.</param>
    /// <remarks>
    /// <para>
    /// A published StepSize is stated in the units of that paper's own training run. ECAPA-TDNN
    /// declares a Triangular2 cycle with StepSize 65000 ramping 1e-8 to 1e-3, which is right for the
    /// run it was measured on and useless for a shorter one: 100 steps into a 65000-step ramp the
    /// rate is about 1.5e-6, roughly 650 times below the declared rate, so the model does not
    /// visibly train. That is what the generated memorization probe caught -- loss moved from
    /// 0.438444 to 0.438423 across 100 steps.
    /// </para>
    /// <para>
    /// Copying 65000 into a 100-step run misapplies the recipe rather than honouring it, so the
    /// half-cycle is scaled to fit: half the run, leaving room for one full up-and-down cycle. The
    /// shape of the schedule, which is the part the paper is actually asserting, is preserved. This
    /// mirrors <see cref="ResolveMilestones"/> and the TriStage hold fraction, both of which already
    /// resolve published positions against the real run length.
    /// </para>
    /// <para>
    /// With no horizon the declared value is kept unchanged: guessing a cycle for an unknown run
    /// length would be inventing a schedule rather than adapting one.
    /// </para>
    /// </remarks>
    private static int FittedHalfCycle(int declaredStepSize, int totalSteps)
    {
        if (totalSteps <= 0 || declaredStepSize <= totalSteps)
        {
            return declaredStepSize;
        }

        return Math.Max(1, totalSteps / 2);
    }

    /// <summary>Turns fractional decay points into the step numbers this run will actually reach.</summary>
    /// <remarks>
    /// Distinct and strictly increasing, because MultiStepLRScheduler requires increasing
    /// milestones and two fractions of a short run can round to the same step -- 0.9 and 0.95 of a
    /// 10-step run are both 9. Rounding them together would throw where the paper simply means
    /// "decay twice, near the end".
    /// </remarks>
    private static int[] ResolveMilestones(double[] fractions, int totalSteps)
    {
        var resolved = new List<int>();
        foreach (double fraction in fractions)
        {
            int step = Math.Max(1, (int)Math.Round(totalSteps * fraction));
            if (resolved.Count > 0 && step <= resolved[resolved.Count - 1]) step = resolved[resolved.Count - 1] + 1;
            if (step < totalSteps) resolved.Add(step);
        }

        return resolved.ToArray();
    }

    /// <summary>Where a warmup ramp starts, so its first step is not a no-op.</summary>
    /// <remarks>
    /// <para>
    /// A ramp computed as <c>step / warmup</c> returns exactly zero on step 0, so the first
    /// optimizer step moves nothing. Over a paper-length run that is one wasted step; over a short
    /// one it can be the whole warmup, and the model then appears not to train at all.
    /// </para>
    /// <para>
    /// The reference implementations avoid this by indexing the ramp from one -- Vaswani's Eq. 3
    /// is 1-indexed, and the fairseq and NeMo warmups both start at one step's worth of the peak.
    /// Starting the ramp there reproduces that shape without changing LinearWarmupScheduler, whose
    /// existing behaviour six test files already pin.
    /// </para>
    /// </remarks>
    private static double FirstWarmupRate(double baseRate, int warmupSteps)
        => warmupSteps > 0 ? baseRate / warmupSteps : baseRate;

    /// <summary>A warmup ramp that holds its peak, for runs too short to decay across.</summary>
    /// <remarks>
    /// Reported rather than silent. Holding the peak is the safe half of the paper's schedule --
    /// the model still trains -- but it is not the whole schedule, and a reproduction that never
    /// anneals is a different result from one that does.
    /// </remarks>
    private static ILearningRateScheduler WarmupWithoutDecay(
        PaperOptimizerAttribute recipe, double baseRate, int warmupSteps, int totalSteps)
    {
        if (recipe.PostWarmupDecay != LinearWarmupScheduler.DecayMode.Constant)
        {
            _pendingUnhonoured.Value?.Add(
                $"the paper decays the rate after warmup, but this run is {totalSteps} steps against "
                + $"{warmupSteps} of warmup, leaving nothing to decay across; the rate holds at its "
                + "peak instead");
        }

        return new LinearWarmupScheduler(
            baseRate, warmupSteps, warmupInitLr: FirstWarmupRate(baseRate, warmupSteps));
    }

    /// <summary>Puts a warmup ramp in front of a schedule that does not have one of its own.</summary>
    /// <remarks>
    /// "Warm up, then decay" is how most papers state a schedule, but only three of the shapes here
    /// contain their own warmup. Without composition a recipe declaring warmup alongside step,
    /// cosine or multi-step decay would build the decay and silently drop the ramp -- and dropping
    /// warmup is not a small loss: for a post-norm transformer it is the difference between
    /// training and diverging at the same rate.
    /// </remarks>
    private static ILearningRateScheduler? ComposeWarmup(
        ILearningRateScheduler? scheduler, PaperOptimizerAttribute recipe,
        double baseRate, int warmupSteps)
    {
        if (scheduler is null || warmupSteps <= 0) return scheduler;

        // These three already ramp; wrapping them would warm up twice.
        if (recipe.Schedule is LearningRateSchedulerType.LinearWarmup
                            or LearningRateSchedulerType.TriStage
                            or LearningRateSchedulerType.Noam)
        {
            return scheduler;
        }

        try
        {
            var ramp = new LinearWarmupScheduler(baseRate, warmupSteps);
            return new SequentialLRScheduler([ramp, scheduler], [warmupSteps]);
        }
        catch (Exception)
        {
            _pendingUnhonoured.Value?.Add(
                $"the paper warms up over {warmupSteps} steps before its {recipe.Schedule} schedule, "
                + "which could not be composed here; the schedule runs without the warmup");
            return scheduler;
        }
    }

    /// <summary>The model dimension, when the options expose one under a name we recognise.</summary>
    /// <remarks>
    /// Only the Noam schedule needs this, and only because its peak rate is derived from the
    /// dimension rather than stated. Returns 0 when nothing matches, which leaves the schedule
    /// reported as unhonoured -- deliberately, since guessing a dimension would silently produce a
    /// peak rate that appears in no paper.
    /// </remarks>
    private static int ModelDimension(object options)
    {
        foreach (string name in new[] { "ModelDimension", "ModelDim", "HiddenDim", "HiddenSize",
                                        "EmbeddingSize", "EmbeddingDimension", "DModel" })
        {
            int value = GetInt(options, name);
            if (value > 0) return value;
        }

        return 0;
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

        ValidateSchedule(recipe);

        double baseRate = double.IsNaN(recipe.LearningRate) ? 0.001 : recipe.LearningRate;
        int warmupSteps = EffectiveWarmupSteps(options, recipe);
        int totalSteps = GetInt(options, "MaxIterations");

        ILearningRateScheduler? scheduler = BuildScheduler(
            recipe, baseRate, warmupSteps, totalSteps, ModelDimension(options));
        scheduler = ComposeWarmup(scheduler, recipe, baseRate, warmupSteps);
        if (scheduler is not null)
        {
            schedulerProperty.SetValue(options, scheduler);
            StepPerBatch(options);
        }
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
        PaperOptimizerAttribute recipe, double baseRate, int warmupSteps, int totalSteps,
        int modelDimension)
    {
        // A floor of zero is the usual published one, and it is also the correct fallback for the
        // schedulers below, all of which decay towards zero when no floor is named.
        double floor = double.IsNaN(recipe.MinLearningRate) ? 0.0 : recipe.MinLearningRate;

        try
        {
            return recipe.Schedule switch
            {
                // The decay needs a horizon. Without one, decaySteps goes negative and every step
                // after warmup returns the floor -- a silent learning rate of zero, so the model
                // trains not at all while every declaration still reads as faithful. Caught by
                // Wav2Vec2's generated Training_ShouldChangeParameters, which failed with exactly
                // "learning rate is 0" the first time a decaying warmup was declared.
                LearningRateSchedulerType.LinearWarmup
                    when warmupSteps > 0 && recipe.PostWarmupDecay != LinearWarmupScheduler.DecayMode.Constant
                         && totalSteps > warmupSteps
                    => new LinearWarmupScheduler(baseRate, warmupSteps, totalSteps,
                                                 warmupInitLr: FirstWarmupRate(baseRate, warmupSteps),
                                                 decayMode: recipe.PostWarmupDecay, endLr: floor),

                // Warmup with no room to decay: keep the ramp, hold the peak, and say so.
                LearningRateSchedulerType.LinearWarmup when warmupSteps > 0
                    => WarmupWithoutDecay(recipe, baseRate, warmupSteps, totalSteps),

                // Needs the run length because every phase is stated as a share of it.
                LearningRateSchedulerType.TriStage when totalSteps > 0 && warmupSteps > 0
                    => new TriStageScheduler(
                           baseRate, warmupSteps,
                           holdSteps: double.IsNaN(recipe.HoldFraction)
                               ? 0 : (int)Math.Round(totalSteps * recipe.HoldFraction),
                           totalSteps: totalSteps, minLearningRate: floor,
                           decayPower: double.IsNaN(recipe.DecayRate) ? 1.0 : recipe.DecayRate),

                // Alone among the schedules here, Noam has no stated peak rate: it is
                // factor * d^-0.5 * min(t^-0.5, t * warmup^-1.5), a function of the model
                // dimension. When that dimension cannot be found the schedule is reported as
                // unhonoured rather than approximated, because the nearest shape this library
                // has decays linearly and Noam decays as the inverse square root.
                LearningRateSchedulerType.Noam when modelDimension > 0 && warmupSteps > 0
                    => new NoamSchedule(modelDimension, warmupSteps),

                // The bounds are the declared rate and floor; StepSize is the half-cycle, scaled to
                // the run when the published one would not fit inside it.
                LearningRateSchedulerType.Cyclic
                    when recipe.StepSize > 0 && !double.IsNaN(recipe.LearningRate)
                    => new CyclicLRScheduler(
                           baseLearningRate: floor, maxLearningRate: recipe.LearningRate,
                           stepSizeUp: FittedHalfCycle(recipe.StepSize, totalSteps),
                           mode: recipe.CyclicPolicy),

                LearningRateSchedulerType.MultiStep
                    when recipe.MilestoneFractions.Length > 0 && totalSteps > 0
                    => new MultiStepLRScheduler(
                           baseRate, ResolveMilestones(recipe.MilestoneFractions, totalSteps),
                           gamma: double.IsNaN(recipe.DecayRate) ? 0.1 : recipe.DecayRate,
                           minLearningRate: floor),

                LearningRateSchedulerType.MultiStep when recipe.Milestones.Length > 0
                    => new MultiStepLRScheduler(
                           baseRate, recipe.Milestones,
                           gamma: double.IsNaN(recipe.DecayRate) ? 0.1 : recipe.DecayRate,
                           minLearningRate: floor),

                LearningRateSchedulerType.Exponential when !double.IsNaN(recipe.DecayRate)
                    => new ExponentialLRScheduler(baseRate, recipe.DecayRate, floor),

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

                // Reached only when an arm could not match because RUNTIME data is missing --
                // no known run length, no discoverable model dimension. A malformed or
                // unmapped declaration was already rejected by ValidateSchedule.
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


    /// <summary>Rejects a declaration this library can never honour, before anything is built.</summary>
    /// <remarks>
    /// Runs outside the construction try/catch on purpose. That catch exists so a scheduler
    /// constructor rejecting its arguments cannot take the model down, but it would equally
    /// swallow these, turning a malformed declaration back into the silent constant rate this is
    /// meant to prevent. Everything checked here depends only on the recipe, never on the run.
    /// </remarks>
    private static void ValidateSchedule(PaperOptimizerAttribute recipe)
    {
        switch (recipe.Schedule)
        {
            case LearningRateSchedulerType.Exponential when double.IsNaN(recipe.DecayRate):
                throw new ArgumentException(
                    "Exponential scheduling requires DecayRate.", nameof(recipe));

            case LearningRateSchedulerType.Step when recipe.StepSize <= 0 || double.IsNaN(recipe.DecayRate):
                throw new ArgumentException(
                    "Step scheduling requires a positive StepSize and a DecayRate.", nameof(recipe));

            case LearningRateSchedulerType.Constant:
            case LearningRateSchedulerType.LinearWarmup:
            case LearningRateSchedulerType.TriStage:
            case LearningRateSchedulerType.Noam:
            case LearningRateSchedulerType.Exponential:
            case LearningRateSchedulerType.Step:
            case LearningRateSchedulerType.MultiStep:
            case LearningRateSchedulerType.CosineAnnealing:
            case LearningRateSchedulerType.Polynomial:
            case LearningRateSchedulerType.OneCycle:
            case LearningRateSchedulerType.Cyclic:
            case LearningRateSchedulerType.ReduceOnPlateau:
                return;

            default:
                throw new NotSupportedException(
                    "Paper optimizer recipes do not express " + recipe.Schedule
                        + "; choose a supported schedule or extend the recipe contract.");
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
    internal static PaperOptimizerAttribute? Find(object? model, string component = "")
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
