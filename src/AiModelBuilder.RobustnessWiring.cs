using AiDotNet.AdversarialRobustness.CertifiedRobustness;
using AiDotNet.AdversarialRobustness.Defenses;
using AiDotNet.AdversarialRobustness.Documentation;
using AiDotNet.AdversarialRobustness.Safety;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.Models.Options;
using AiDotNet.Models.Results;
using AiDotNet.Safety;
using AiDotNet.Safety.Benchmarking;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Logging.Abstractions;

namespace AiDotNet;

/// <summary>
/// Routes every group of <see cref="AdversarialRobustnessOptions{T}"/> and
/// <see cref="AdversarialRobustnessConfiguration{T, TInput, TOutput}"/> onto the engine that implements it.
/// </summary>
/// <remarks>
/// <list type="bullet">
/// <item><description>Safety filtering: the <see cref="SafetyFilter{T}"/> of the result.</description></item>
/// <item><description>Adversarial defense: <see cref="AdversarialTraining{T, TInput, TOutput}"/> run on the trained
/// model during the supervised build.</description></item>
/// <item><description>Certification: the <see cref="ICertifiedDefense{T, TInput, TOutput}"/> engine named by
/// <see cref="AdversarialRobustnessOptions{T}.CertificationMethod"/>.</description></item>
/// <item><description>Content moderation and red teaming: the <c>ConfigureSafety</c> pipeline and its safety
/// benchmarks.</description></item>
/// </list>
/// </remarks>
public partial class AiModelBuilder<T, TInput, TOutput>
{
    /// <summary>True when the model was replaced at build by the defended model a defense returned.</summary>
    private bool _adversarialDefenseAppliedToModel;

    /// <summary>The certified defense chosen for this build (explicit, custom, or created from the options).</summary>
    private ICertifiedDefense<T, TInput, TOutput>? _robustnessCertifiedDefense;

    /// <summary>The enabled robustness configuration, or null when none is configured or it is disabled.</summary>
    private AdversarialRobustnessConfiguration<T, TInput, TOutput>? ActiveRobustnessConfiguration =>
        _adversarialRobustnessConfiguration is { Enabled: true } configuration ? configuration : null;

    private ILogger RobustnessLogger => _adversarialRobustnessConfiguration?.Logger ?? NullLogger.Instance;

    private void LogRobustnessStep(string message, params object[] args)
    {
        if (ActiveRobustnessConfiguration?.Options.VerboseLogging == true)
        {
            RobustnessLogger.LogInformation(message, args);
        }
    }

    /// <summary>
    /// The safety filter options for the safety group of <paramref name="options"/>.
    /// </summary>
    internal static SafetyFilterOptions<T> ToSafetyFilterOptions(AdversarialRobustnessOptions<T> options)
    {
        return new SafetyFilterOptions<T>
        {
            SafetyThreshold = options.SafetyThreshold,
            JailbreakSensitivity = options.JailbreakSensitivity,
            EnableInputValidation = options.EnableInputValidation,
            EnableOutputFiltering = options.EnableOutputFiltering,
            HarmfulContentCategories = options.HarmfulContentCategories.ToArray(),
            UseClassifier = options.UseContentClassifier,
            MaxInputLength = options.MaxInputLength,
            LogFilteredContent = options.LogFilteredContent,
            LogFilePath = options.SafetyLogFilePath
        };
    }

    /// <summary>
    /// The adversarial training options for the defense group of <paramref name="options"/>.
    /// </summary>
    internal static AdversarialDefenseOptions<T> ToAdversarialDefenseOptions(AdversarialRobustnessOptions<T> options)
    {
        return new AdversarialDefenseOptions<T>
        {
            AdversarialRatio = options.AdversarialTrainingRatio,
            Epsilon = options.DefenseEpsilon,
            TrainingEpochs = options.EnableAdversarialTraining ? options.AdversarialTrainingEpochs : 0,
            BatchSize = options.BatchSize,
            UsePreprocessing = options.UseInputPreprocessing,
            PreprocessingMethod = options.PreprocessingMethod,
            UseEnsemble = options.UseEnsembleDefense,
            EnsembleSize = options.EnsembleSize,
            AttackMethod = options.TrainingAttackMethod,
            Seed = options.Seed
        };
    }

    /// <summary>
    /// The certified defense options for the certification group of <paramref name="options"/>.
    /// </summary>
    internal static CertifiedDefenseOptions<T> ToCertifiedDefenseOptions(AdversarialRobustnessOptions<T> options)
    {
        return new CertifiedDefenseOptions<T>
        {
            CertificationMethod = options.CertificationMethod,
            NumSamples = options.CertificationSamples,
            NoiseSigma = options.CertificationNoiseSigma,
            ConfidenceLevel = options.CertificationConfidence,
            UseTightBounds = options.UseTightCertificationBounds,
            NormType = options.CertificationNormType,
            BatchSize = options.BatchSize,
            Seed = options.Seed
        };
    }

    /// <summary>
    /// The certification engine named by <see cref="CertifiedDefenseOptions{T}.CertificationMethod"/>.
    /// </summary>
    internal static ICertifiedDefense<T, TInput, TOutput> CreateCertifiedDefense(CertifiedDefenseOptions<T> options)
    {
        return options.CertificationMethod switch
        {
            CertifiedDefenseMethod.RandomizedSmoothing => new RandomizedSmoothing<T, TInput, TOutput>(options),
            CertifiedDefenseMethod.IntervalBoundPropagation => new IntervalBoundPropagation<T, TInput, TOutput>(options),
            CertifiedDefenseMethod.Crown => new CROWNVerification<T, TInput, TOutput>(options),
            CertifiedDefenseMethod.AdaptiveRandomizedSmoothing => new AdaptiveRandomizedSmoothing<T, TInput, TOutput>(options),
            _ => throw new InvalidOperationException(
                $"AdversarialRobustnessOptions.CertificationMethod has the undefined value {(int)options.CertificationMethod}.")
        };
    }

    /// <summary>
    /// The certified defense of this build: <c>ConfigureCertifiedDefense</c>, else the configuration's
    /// <c>CustomCertifiedDefense</c>, else the engine the options name when certification is enabled.
    /// </summary>
    private ICertifiedDefense<T, TInput, TOutput>? EffectiveCertifiedDefense()
    {
        if (_configuredCertifiedDefense is not null)
        {
            return _configuredCertifiedDefense;
        }

        var configuration = ActiveRobustnessConfiguration;
        if (configuration is null)
        {
            return null;
        }

        if (_robustnessCertifiedDefense is null)
        {
            _robustnessCertifiedDefense = configuration.CustomCertifiedDefense
                ?? (configuration.Options.EnableCertifiedRobustness
                    ? CreateCertifiedDefense(ToCertifiedDefenseOptions(configuration.Options))
                    : null);
        }

        return _robustnessCertifiedDefense;
    }

    /// <summary>
    /// The attacks of the build robustness audit: <c>ConfigureAdversarialAttack</c> plus the configuration's
    /// <c>CustomAttacks</c> (when <c>IncludeRobustnessInEvaluation</c> is on).
    /// </summary>
    private List<IAdversarialAttack<T, TInput, TOutput>> RobustnessAuditAttacks()
    {
        var attacks = new List<IAdversarialAttack<T, TInput, TOutput>>();
        if (_configuredAdversarialAttack is not null)
        {
            attacks.Add(_configuredAdversarialAttack);
        }

        var configuration = ActiveRobustnessConfiguration;
        if (configuration is { IncludeRobustnessInEvaluation: true, CustomAttacks: { } custom })
        {
            attacks.AddRange(custom.Where(a => a is not null));
        }

        return attacks;
    }

    /// <summary>
    /// The indices of the samples the robustness audits evaluate: every sample, or with a configuration
    /// round(<c>RobustnessEvaluationSampleRatio</c> * n) (at least one) evenly spaced samples.
    /// </summary>
    private int[] RobustnessEvaluationIndices(int count)
    {
        var configuration = ActiveRobustnessConfiguration;
        double ratio = configuration?.RobustnessEvaluationSampleRatio ?? 1.0;
        if (!(ratio > 0.0 && ratio <= 1.0))
        {
            throw new InvalidOperationException(
                $"AdversarialRobustnessConfiguration.RobustnessEvaluationSampleRatio must lie in (0, 1] (was {ratio}).");
        }

        int take = Math.Max(1, Math.Min(count, (int)Math.Round(ratio * count, MidpointRounding.AwayFromZero)));
        var indices = new int[take];
        for (int i = 0; i < take; i++)
        {
            indices[i] = (int)((long)i * count / take);
        }

        return indices;
    }

    /// <summary>
    /// The perturbation budgets of the robustness curve: the configuration's <c>EvaluationEpsilons</c>, or the
    /// historical 0.25x, 0.5x, 1x and 2x of the attack's own budget.
    /// </summary>
    private double[] RobustnessCurveBudgets(double baseEpsilon)
    {
        var configured = ActiveRobustnessConfiguration?.EvaluationEpsilons;
        if (configured is { Length: > 0 })
        {
            foreach (double epsilon in configured)
            {
                if (!(epsilon > 0 && !double.IsInfinity(epsilon)))
                {
                    throw new InvalidOperationException(
                        $"AdversarialRobustnessConfiguration.EvaluationEpsilons must be positive finite numbers (found {epsilon}).");
                }
            }

            return configured.Distinct().OrderBy(e => e).ToArray();
        }

        return new[] { 0.25 * baseEpsilon, 0.5 * baseEpsilon, baseEpsilon, 2.0 * baseEpsilon };
    }

    /// <summary>
    /// Runs the configured adversarial defense on the trained model (supervised build path).
    /// </summary>
    /// <remarks>
    /// With <c>EnableAdversarialTraining</c> the configuration's <c>CustomDefense</c> (or
    /// <see cref="AdversarialTraining{T, TInput, TOutput}"/> built from the defense options) is applied to the training
    /// samples. When the defense returns a different model, that defended model replaces the result's model, so
    /// <c>Predict</c>, the robustness audits and certification all run through it.
    /// </remarks>
    /// <exception cref="InvalidOperationException">The training data cannot be split into samples.</exception>
    private void ApplyConfiguredAdversarialDefense(AiModelResult<T, TInput, TOutput> result, TInput preparedX, TOutput preparedY)
    {
        var configuration = ActiveRobustnessConfiguration;
        if (configuration is null || !configuration.Options.EnableAdversarialTraining || result.Model is null)
        {
            return;
        }

        if (!TrySplitSamples(preparedX, preparedY, out var inputs, out var labels))
        {
            throw new InvalidOperationException(
                "EnableAdversarialTraining needs per-sample training data: Matrix<T> inputs with Vector<T> labels, or " +
                $"Tensor<T> inputs and labels with a batch axis. {typeof(TInput).Name} / {typeof(TOutput).Name} cannot be split.");
        }

        var defense = configuration.CustomDefense
            ?? new AdversarialTraining<T, TInput, TOutput>(ToAdversarialDefenseOptions(configuration.Options));
        LogRobustnessStep("Adversarial training: {Defense} over {Samples} samples.", defense.GetType().Name, inputs.Length);

        var model = result.Model;
        var defended = defense.ApplyDefense(inputs, labels, model);
        if (!ReferenceEquals(defended, model))
        {
            result.Model = defended;
            _adversarialDefenseAppliedToModel = true;
        }
    }

    /// <summary>
    /// Splits a prepared data set into per-sample inputs and labels: matrix rows with vector entries, or the
    /// leading axis of tensors.
    /// </summary>
    [AiDotNet.Attributes.ScalarKernel("Splitting a data set into samples is one element per sample by definition.")]
    internal static bool TrySplitSamples(TInput x, TOutput y, out TInput[] inputs, out TOutput[] labels)
    {
        inputs = Array.Empty<TInput>();
        labels = Array.Empty<TOutput>();
        if (x is Matrix<T> matrix && y is Vector<T> vector && matrix.Rows == vector.Length)
        {
            int n = matrix.Rows;
            inputs = new TInput[n];
            labels = new TOutput[n];
            for (int i = 0; i < n; i++)
            {
                var row = new Matrix<T>(1, matrix.Columns);
                row.SetRow(0, matrix.GetRow(i));
                inputs[i] = (TInput)(object)row;
                labels[i] = (TOutput)(object)new Vector<T>(new[] { vector[i] });
            }

            return true;
        }

        if (x is Tensor<T> features && y is Tensor<T> targets && features.Rank >= 1 && targets.Rank >= 1
            && features.Shape[0] == targets.Shape[0])
        {
            int n = features.Shape[0];
            inputs = new TInput[n];
            labels = new TOutput[n];
            var engine = AiDotNetEngine.Current;
            for (int i = 0; i < n; i++)
            {
                inputs[i] = (TInput)(object)SliceSample(engine, features, i);
                labels[i] = (TOutput)(object)SliceSample(engine, targets, i);
            }

            return true;
        }

        return false;
    }

    private static Tensor<T> SliceSample(IEngine engine, Tensor<T> tensor, int index)
    {
        var start = new int[tensor.Rank];
        var length = tensor.Shape.ToArray();
        start[0] = index;
        length[0] = 1;
        return engine.TensorSlice(tensor, start, length);
    }

    /// <summary>
    /// Applies the safety, defense, certification, content moderation and red teaming groups to a built result.
    /// Called from every build path by <see cref="AttachAdversarialRobustness"/>.
    /// </summary>
    private void AttachConfiguredRobustness(AiModelResult<T, TInput, TOutput> result)
    {
        var configuration = ActiveRobustnessConfiguration;
        var certifiedDefense = EffectiveCertifiedDefense();
        if (certifiedDefense is not null)
        {
            result.SetCertifiedDefense(certifiedDefense);
        }

        if (configuration is null)
        {
            return;
        }

        var options = configuration.Options;

        // Safety group -> the SafetyFilter of the result.
        if (options.EnableSafetyFiltering)
        {
            result.SetSafetyFilter(configuration.CustomSafetyFilter
                ?? new SafetyFilter<T>(ToSafetyFilterOptions(options), configuration.CustomContentClassifier));
            LogRobustnessStep("Safety filter attached (threshold {Threshold}).", options.SafetyThreshold);
        }
        else
        {
            result.SetSafetyFilter(null);
        }

        result.SetCertifiedInference(new CertifiedInferenceSettings(
            configuration.UseCertifiedInference,
            configuration.MinimumCertifiedRadius,
            configuration.RejectNonRobustPredictions));

        if (options.EnableRedTeaming)
        {
            RunRedTeaming(result, options);
        }

        if (configuration.IncludeRobustnessMetrics)
        {
            WriteRobustnessMetadata(result);
        }

        if (configuration.AutoGenerateModelCardSections)
        {
            result.SetRobustnessModelCard(BuildRobustnessModelCard(result, configuration));
        }
    }

    /// <summary>
    /// Fills the text section of the <c>ConfigureSafety</c> pipeline from the content moderation group. Values the
    /// caller set explicitly through <c>ConfigureSafety</c> are kept.
    /// </summary>
    private void MergeContentModerationIntoSafetyConfig()
    {
        var configuration = ActiveRobustnessConfiguration;
        if (configuration is null || !configuration.Options.EnableContentModeration)
        {
            return;
        }

        var options = configuration.Options;
        var safety = _safetyPipelineConfig ??= new SafetyConfig();
        safety.Text.ToxicityDetection ??= true;
        safety.Text.JailbreakDetection ??= true;
        safety.Text.JailbreakSensitivity ??= options.PromptInjectionSensitivity;
        safety.Text.BlockJailbreaks ??= options.BlockPromptInjections;
        safety.Text.PIIDetection ??= options.FilterPII;
        safety.Text.PIITypes ??= options.PIITypes.ToArray();
        safety.Text.HallucinationDetection ??= options.EnableFactualityChecking;
        safety.Text.HallucinationThreshold ??= options.HallucinationThreshold;
        safety.Text.MaxInputLength ??= options.MaxInputLength;
    }

    /// <summary>
    /// Red teams the result's safety pipeline with <c>RedTeamingPromptCount</c> prompts drawn round-robin from the
    /// safety benchmarks of <c>RedTeamingCategories</c>.
    /// </summary>
    private void RunRedTeaming(AiModelResult<T, TInput, TOutput> result, AdversarialRobustnessOptions<T> options)
    {
        if (options.RedTeamingPromptCount < 1)
        {
            throw new InvalidOperationException(
                $"AdversarialRobustnessOptions.RedTeamingPromptCount must be at least 1 (was {options.RedTeamingPromptCount}).");
        }

        var categories = options.RedTeamingCategories?.Distinct().ToArray() ?? Array.Empty<RedTeamingCategory>();
        if (categories.Length == 0)
        {
            throw new InvalidOperationException("AdversarialRobustnessOptions.RedTeamingCategories must name at least one category.");
        }

        var pools = categories.Select(c => RedTeamingBenchmark(c).TestCases).ToArray();
        var prompts = new List<SafetyBenchmarkCase>(options.RedTeamingPromptCount);
        for (int round = 0; prompts.Count < options.RedTeamingPromptCount; round++)
        {
            bool any = false;
            foreach (var pool in pools)
            {
                if (round < pool.Count && prompts.Count < options.RedTeamingPromptCount)
                {
                    prompts.Add(pool[round]);
                    any = true;
                }
            }

            if (!any)
            {
                break; // every pool is exhausted
            }
        }

        var pipeline = result.SafetyPipeline ?? SafetyPipelineFactory<T>.Create(_safetyPipelineConfig);
        LogRobustnessStep("Red teaming the safety pipeline with {Prompts} prompts.", prompts.Count);
        result.SetRedTeamingReport(new SafetyBenchmarkRunner<T>(pipeline).RunBenchmark(prompts));
    }

    private static SafetyBenchmarkBase<T> RedTeamingBenchmark(RedTeamingCategory category) => category switch
    {
        RedTeamingCategory.Jailbreaking => new JailbreakBenchmark<T>(),
        RedTeamingCategory.HarmfulContent => new ToxicityBenchmark<T>(),
        RedTeamingCategory.Misinformation => new HallucinationBenchmark<T>(),
        RedTeamingCategory.PrivacyViolation => new PIIBenchmark<T>(),
        RedTeamingCategory.BiasExploration => new BiasBenchmark<T>(),
        _ => throw new InvalidOperationException(
            $"AdversarialRobustnessOptions.RedTeamingCategories contains the undefined value {(int)category}.")
    };

    /// <summary>
    /// The robustness figures of the result, by metadata key.
    /// </summary>
    internal static Dictionary<string, double> RobustnessFigures(AiModelResult<T, TInput, TOutput> result)
    {
        var figures = new Dictionary<string, double>();
        if (result.AdversarialRobustness is { } empirical)
        {
            figures["Robustness.CleanAccuracy"] = empirical.CleanAccuracy;
            figures["Robustness.RobustAccuracy"] = empirical.RobustAccuracy;
            figures["Robustness.AttackSuccessRate"] = empirical.AttackSuccessRate;
            figures["Robustness.RobustAccuracyAuc"] = empirical.RobustAccuracyAuc;
        }

        if (result.CertifiedRobustness is { } certified)
        {
            figures["Robustness.CertifiedAccuracy"] = certified.Metrics.CertifiedAccuracy;
            figures["Robustness.CertifiedRadius"] = certified.RadiusUsed;
        }

        if (result.RedTeamingReport is { } redTeam)
        {
            figures["Robustness.RedTeamingRecall"] = redTeam.Recall;
            figures["Robustness.RedTeamingFalsePositiveRate"] = redTeam.FalsePositiveRate;
        }

        return figures;
    }

    private static void WriteRobustnessMetadata(AiModelResult<T, TInput, TOutput> result)
    {
        var metadata = result.GetModelMetadata();
        foreach (var figure in RobustnessFigures(result))
        {
            metadata.SetProperty(figure.Key, figure.Value);
        }
    }

    private static ModelCard BuildRobustnessModelCard(
        AiModelResult<T, TInput, TOutput> result,
        AdversarialRobustnessConfiguration<T, TInput, TOutput> configuration)
    {
        var card = new ModelCard
        {
            ModelName = result.Model?.GetType().Name ?? "Unnamed Model",
            Date = DateTime.UtcNow
        };

        foreach (var figure in RobustnessFigures(result))
        {
            card.RobustnessMetrics[figure.Key.Substring("Robustness.".Length)] = figure.Value;
        }

        var options = configuration.Options;
        if (options.EnableAdversarialTraining)
        {
            card.Recommendations.Add(
                $"Adversarially trained ({options.TrainingAttackMethod}, epsilon {options.DefenseEpsilon}, ratio {options.AdversarialTrainingRatio}).");
        }

        if (result.CertifiedRobustness is null && !options.EnableCertifiedRobustness)
        {
            card.Caveats.Add("No certified robustness guarantee was computed.");
        }

        if (!string.IsNullOrWhiteSpace(configuration.ModelCardRobustnessNotes))
        {
            card.Caveats.Add(configuration.ModelCardRobustnessNotes ?? string.Empty);
        }

        return card;
    }
}
