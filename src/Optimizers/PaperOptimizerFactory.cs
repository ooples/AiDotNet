using System.Collections.Concurrent;
using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.LearningRateSchedulers;
using AiDotNet.Models.Options;

namespace AiDotNet.Optimizers;

/// <summary>
/// Builds the optimizer and learning-rate policy declared by a model's
/// <see cref="PaperOptimizerAttribute"/>.
/// </summary>
/// <remarks>
/// Internal by design: applications configure optimizers through the public model/facade API.
/// Models use this helper only to choose their paper-faithful default when the caller did not
/// supply one.
/// </remarks>
internal static class PaperOptimizerFactory
{
    private static readonly ConcurrentDictionary<Type, PaperOptimizerAttribute[]> _byModelType = new();

    /// <summary>Builds the applicable declared optimizer, or returns null when none is declared.</summary>
    internal static IGradientBasedOptimizer<T, TInput, TOutput>? CreateFor<T, TInput, TOutput>(
        IFullModel<T, TInput, TOutput>? model)
    {
        if (model is null) return null;

        PaperOptimizerAttribute? recipe = Find(model);
        return recipe is null ? null : CreateFromRecipe(model, recipe);
    }

    /// <summary>
    /// Constructs one complete, already-configured recipe. Kept separate from reflection-based
    /// selection so every enum member and scheduler contract can be tested directly.
    /// </summary>
    internal static IGradientBasedOptimizer<T, TInput, TOutput> CreateFromRecipe<T, TInput, TOutput>(
        IFullModel<T, TInput, TOutput> model,
        PaperOptimizerAttribute recipe)
    {
        if (model is null) throw new ArgumentNullException(nameof(model));
        if (recipe is null) throw new ArgumentNullException(nameof(recipe));

        TOptions ConfigureOptions<TOptions>(TOptions options)
            where TOptions : GradientBasedOptimizerOptions<T, TInput, TOutput>
        {
            if (!double.IsNaN(recipe.LearningRate))
                options.InitialLearningRate = recipe.LearningRate;

            if (!double.IsNaN(recipe.Momentum))
            {
                options.InitialMomentum = recipe.Momentum;
                options.UseAdaptiveMomentum = false;
            }

            if (!double.IsNaN(recipe.MaxGradientNorm))
            {
                options.EnableGradientClipping = true;
                options.MaxGradientNorm = recipe.MaxGradientNorm;
            }

            if (recipe.Schedule != LearningRateSchedulerType.Constant || recipe.WarmupSteps > 0)
                options.LearningRateScheduler = BuildScheduler(recipe, options.InitialLearningRate);

            switch (options)
            {
                // Adam8Bit derives from Adam options, so it must precede the Adam case.
                case Adam8BitOptimizerOptions<T, TInput, TOutput> adam8Bit:
                    ConfigureAdam(adam8Bit);
                    break;
                case AdamOptimizerOptions<T, TInput, TOutput> adam:
                    ConfigureAdam(adam);
                    break;
                case AdamWOptimizerOptions<T, TInput, TOutput> adamW:
                    SetAdamWValues(adamW);
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
                    SetAdamaxValues(adamax);
                    break;
                case NadamOptimizerOptions<T, TInput, TOutput> nadam:
                    SetNadamValues(nadam);
                    break;
                case LAMBOptimizerOptions<T, TInput, TOutput> lamb:
                    SetLambValues(lamb);
                    break;
                case LionOptimizerOptions<T, TInput, TOutput> lion:
                    SetLionValues(lion);
                    break;
            }

            return options;
        }

        void ConfigureAdam(AdamOptimizerOptions<T, TInput, TOutput> options)
        {
            if (!double.IsNaN(recipe.Beta1)) options.Beta1 = recipe.Beta1;
            if (!double.IsNaN(recipe.Beta2)) options.Beta2 = recipe.Beta2;
            if (!double.IsNaN(recipe.Epsilon)) options.Epsilon = recipe.Epsilon;
            if (!double.IsNaN(recipe.Beta1) || !double.IsNaN(recipe.Beta2))
                options.UseAdaptiveBetas = false;
        }

        void SetAdamWValues(AdamWOptimizerOptions<T, TInput, TOutput> options)
        {
            if (!double.IsNaN(recipe.Beta1)) options.Beta1 = recipe.Beta1;
            if (!double.IsNaN(recipe.Beta2)) options.Beta2 = recipe.Beta2;
            if (!double.IsNaN(recipe.Epsilon)) options.Epsilon = recipe.Epsilon;
            if (!double.IsNaN(recipe.WeightDecay)) options.WeightDecay = recipe.WeightDecay;
            if (!double.IsNaN(recipe.Beta1) || !double.IsNaN(recipe.Beta2))
                options.UseAdaptiveBetas = false;
        }

        void SetAdamaxValues(AdaMaxOptimizerOptions<T, TInput, TOutput> options)
        {
            if (!double.IsNaN(recipe.Beta1)) options.Beta1 = recipe.Beta1;
            if (!double.IsNaN(recipe.Beta2)) options.Beta2 = recipe.Beta2;
            if (!double.IsNaN(recipe.Epsilon)) options.Epsilon = recipe.Epsilon;
        }

        void SetNadamValues(NadamOptimizerOptions<T, TInput, TOutput> options)
        {
            if (!double.IsNaN(recipe.Beta1)) options.Beta1 = recipe.Beta1;
            if (!double.IsNaN(recipe.Beta2)) options.Beta2 = recipe.Beta2;
            if (!double.IsNaN(recipe.Epsilon)) options.Epsilon = recipe.Epsilon;
        }

        void SetLambValues(LAMBOptimizerOptions<T, TInput, TOutput> options)
        {
            if (!double.IsNaN(recipe.Beta1)) options.Beta1 = recipe.Beta1;
            if (!double.IsNaN(recipe.Beta2)) options.Beta2 = recipe.Beta2;
            if (!double.IsNaN(recipe.Epsilon)) options.Epsilon = recipe.Epsilon;
            if (!double.IsNaN(recipe.WeightDecay)) options.WeightDecay = recipe.WeightDecay;
        }

        void SetLionValues(LionOptimizerOptions<T, TInput, TOutput> options)
        {
            if (!double.IsNaN(recipe.Beta1))
            {
                options.Beta1 = recipe.Beta1;
                options.UseAdaptiveBeta1 = false;
            }
            if (!double.IsNaN(recipe.Beta2))
            {
                options.Beta2 = recipe.Beta2;
                options.UseAdaptiveBeta2 = false;
            }
            if (!double.IsNaN(recipe.WeightDecay)) options.WeightDecay = recipe.WeightDecay;
        }

        return recipe.Optimizer switch
        {
            OptimizerKind.Adam => new AdamOptimizer<T, TInput, TOutput>(
                model, ConfigureOptions(new AdamOptimizerOptions<T, TInput, TOutput>())),
            OptimizerKind.AdamW => new AdamWOptimizer<T, TInput, TOutput>(
                model, ConfigureOptions(new AdamWOptimizerOptions<T, TInput, TOutput>())),
            OptimizerKind.Sgd => new StochasticGradientDescentOptimizer<T, TInput, TOutput>(
                model, ConfigureOptions(new StochasticGradientDescentOptimizerOptions<T, TInput, TOutput>())),
            OptimizerKind.SgdMomentum when recipe.UseNesterov
                => new NesterovAcceleratedGradientOptimizer<T, TInput, TOutput>(
                    model, ConfigureOptions(new NesterovAcceleratedGradientOptimizerOptions<T, TInput, TOutput>())),
            OptimizerKind.SgdMomentum => new MomentumOptimizer<T, TInput, TOutput>(
                model, ConfigureOptions(new MomentumOptimizerOptions<T, TInput, TOutput>())),
            OptimizerKind.Adam8Bit => new Adam8BitOptimizer<T, TInput, TOutput>(
                model, ConfigureOptions(new Adam8BitOptimizerOptions<T, TInput, TOutput>())),
            OptimizerKind.RmsProp => new RootMeanSquarePropagationOptimizer<T, TInput, TOutput>(
                model, ConfigureOptions(new RootMeanSquarePropagationOptimizerOptions<T, TInput, TOutput>())),
            OptimizerKind.Adagrad => new AdagradOptimizer<T, TInput, TOutput>(
                model, ConfigureOptions(new AdagradOptimizerOptions<T, TInput, TOutput>())),
            OptimizerKind.Adadelta => new AdaDeltaOptimizer<T, TInput, TOutput>(
                model, ConfigureOptions(new AdaDeltaOptimizerOptions<T, TInput, TOutput>())),
            OptimizerKind.Adamax => new AdaMaxOptimizer<T, TInput, TOutput>(
                model, ConfigureOptions(new AdaMaxOptimizerOptions<T, TInput, TOutput>())),
            OptimizerKind.Nadam => new NadamOptimizer<T, TInput, TOutput>(
                model, ConfigureOptions(new NadamOptimizerOptions<T, TInput, TOutput>())),
            OptimizerKind.Lamb => new LAMBOptimizer<T, TInput, TOutput>(
                model, ConfigureOptions(new LAMBOptimizerOptions<T, TInput, TOutput>())),
            OptimizerKind.Lion => new LionOptimizer<T, TInput, TOutput>(
                model, ConfigureOptions(new LionOptimizerOptions<T, TInput, TOutput>())),
            OptimizerKind.LBfgs => new LBFGSOptimizer<T, TInput, TOutput>(
                model, ConfigureOptions(new LBFGSOptimizerOptions<T, TInput, TOutput>())),
            OptimizerKind.Unspecified => throw new InvalidOperationException(
                "An unspecified paper optimizer cannot be constructed."),
            _ => throw new ArgumentOutOfRangeException(
                nameof(recipe), recipe.Optimizer, "The declared paper optimizer is not supported."),
        };
    }

    /// <summary>Builds exactly the scheduler declared by the recipe.</summary>
    internal static ILearningRateScheduler BuildScheduler(PaperOptimizerAttribute recipe, double baseRate)
    {
        if (recipe is null) throw new ArgumentNullException(nameof(recipe));

        if (recipe.Schedule == LearningRateSchedulerType.LinearWarmup && recipe.WarmupSteps <= 0)
            throw new ArgumentException("LinearWarmup requires a positive WarmupSteps value.", nameof(recipe));

        ILearningRateScheduler mainScheduler = recipe.Schedule switch
        {
            LearningRateSchedulerType.Constant or LearningRateSchedulerType.LinearWarmup
                => new ConstantLRScheduler(baseRate),
            LearningRateSchedulerType.Exponential when !double.IsNaN(recipe.DecayRate)
                => new ExponentialLRScheduler(
                    baseRate,
                    recipe.DecayRate,
                    double.IsNaN(recipe.MinLearningRate) ? 0.0 : recipe.MinLearningRate),
            LearningRateSchedulerType.Exponential => throw new ArgumentException(
                "Exponential scheduling requires DecayRate.", nameof(recipe)),
            LearningRateSchedulerType.Step when recipe.StepSize > 0 && !double.IsNaN(recipe.DecayRate)
                => new StepLRScheduler(
                    baseRate,
                    recipe.StepSize,
                    recipe.DecayRate,
                    double.IsNaN(recipe.MinLearningRate) ? 0.0 : recipe.MinLearningRate),
            LearningRateSchedulerType.Step => throw new ArgumentException(
                "Step scheduling requires a positive StepSize and a DecayRate.", nameof(recipe)),
            _ => throw new NotSupportedException(
                $"Paper optimizer recipes do not yet express the parameters required by "
                    + $"{recipe.Schedule}; choose a supported schedule or extend the recipe contract."),
        };

        if (recipe.WarmupSteps <= 0) return mainScheduler;

        var warmup = new LinearWarmupScheduler(baseRate, recipe.WarmupSteps);
        if (recipe.Schedule is LearningRateSchedulerType.Constant or LearningRateSchedulerType.LinearWarmup)
            return warmup;

        return new SequentialLRScheduler(
            new ILearningRateScheduler[] { warmup, mainScheduler },
            new[] { recipe.WarmupSteps });
    }

    /// <summary>The declaration matching this model: variant-specific when one exists, else unkeyed.</summary>
    internal static PaperOptimizerAttribute? Find(object? model)
    {
        if (model is null) return null;

        PaperOptimizerAttribute[] declarations = _byModelType.GetOrAdd(
            model.GetType(),
            static type => (PaperOptimizerAttribute[])type
                .GetCustomAttributes(typeof(PaperOptimizerAttribute), inherit: true));

        string variant = (model as IPaperOptimizerVariant)?.PaperOptimizerVariant ?? string.Empty;
        PaperOptimizerAttribute? matching = null;
        PaperOptimizerAttribute? unkeyed = null;

        foreach (var declaration in declarations.Where(
            declaration => declaration.Optimizer != OptimizerKind.Unspecified))
        {
            if (string.IsNullOrEmpty(declaration.Variant))
            {
                if (unkeyed is not null)
                    throw DuplicateVariant(model.GetType(), "(default)");
                unkeyed = declaration;
                continue;
            }

            if (variant.Length == 0
                || !string.Equals(declaration.Variant, variant, StringComparison.Ordinal))
                continue;

            if (matching is not null)
                throw DuplicateVariant(model.GetType(), variant);
            matching = declaration;
        }

        return matching ?? unkeyed;
    }

    private static InvalidOperationException DuplicateVariant(Type modelType, string variant)
        => new($"{modelType.FullName} declares more than one paper optimizer recipe for variant "
            + $"'{variant}'. Variant keys must be unique across optimizer kinds.");
}
