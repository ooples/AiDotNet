using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using System.Text;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.Configuration;

/// <summary>
/// Ratchet for the model configuration surface (issue #2090).
/// </summary>
/// <remarks>
/// <para>
/// Model options classes are meant to be the only user-facing configuration surface for
/// model-specific parameters. Today most models take their tunable values as defaulted
/// constructor parameters instead, and the options object they are handed is stored,
/// returned by <c>GetOptions()</c>, and never read.
/// </para>
/// <para>
/// This test counts the gap and refuses to let it grow. As each model family is migrated,
/// <see cref="Baseline"/> comes down. It is deliberately a count rather than a whitelist:
/// a whitelist gets appended to without thought, whereas a number that may only ever
/// decrease forces the choice to be explicit.
/// </para>
/// <para>
/// The count is computed by reflection rather than by scanning source, so it needs no
/// documentation to exist and cannot be defeated by deleting doc comments.
/// </para>
/// </remarks>
public class OptionsSurfaceRatchetTests
{
    private readonly Xunit.Abstractions.ITestOutputHelper _output;

    public OptionsSurfaceRatchetTests(Xunit.Abstractions.ITestOutputHelper output)
    {
        _output = output;
    }

    /// <summary>
    /// Number of tunable defaulted constructor parameters that have no correspondingly-named
    /// property on their model's options type.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>Lower this number when you migrate a model family; never raise it.</b> If this test
    /// fails saying the count went up, you added a tunable parameter to a model constructor —
    /// add a property to that model's options class instead.
    /// </para>
    /// <para>
    /// Phase 2 took this from 1067 to 977 (17 sequence models, 90 params); phase 3 to 875
    /// (11 vision-language models, 102 params), then 861 once enum- and string-typed
    /// parameters were migrated too and VisionMambaModel was picked up; phase 4a to 782
    /// (11 embedding and retrieval models, 77 params); phase 4b to 741 (10 GANs, 41 params); phase 5 to 697 (19 Document models, 148 params); phase 6 to 562 (43 Video models, 136 params); phase 7 to 441 (12 audio models, 130 params); phase 8 to 408 (all 8 panoptic segmentation models, 33 params); phase 9 to 392 (3 graph task models, 16 params); phase 10 to 378 (3 graph encoder networks, 14 params); phase 11 to 368 (GraphGenerationModel, 10 params); phase 12 to 350 (3 PINN models, 18 params).
    /// Originally established by this test's first run against master on 2026-09-08. A file-based
    /// estimate of the three areas named in the #2090 spec put it at 806; this reflection
    /// measurement found 1067, because the defect also reaches models the file scan never
    /// examined — Tacotron2Model, TtsModel and VITSModel each carry 16-20 tunable parameters
    /// against a generic OnnxModelOptions, and SpeechEmotionRecognizer takes 11 with no
    /// options parameter at all. Those areas looked healthy when measured by whether their
    /// Options classes declare properties, which is a different question from whether the
    /// models read them.
    /// </para>
    /// <para>
    /// 53 to 0 with the final tail cluster. Zero means no model constructor takes a tunable
    /// parameter whose options class does not already declare an equivalently-named property.
    /// It does NOT mean #2090 is closed: this ratchet credits a NAME match, so it is satisfied
    /// by a property existing, and says nothing about whether the constructor reads it. The
    /// stricter <see cref="ConstructorBaseline"/> still stands at 21, and the unread-property
    /// and Validate-coverage ratchets measure two further defect forms again.
    /// </para>
    /// </remarks>
    private const int Baseline = 0;

    /// <summary>
    /// Number of tunable defaulted constructor parameters still declared by an in-scope model,
    /// giving NO credit for a matching options property.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <see cref="Baseline"/> counts a parameter as covered once the options class has a
    /// property of the same name. That is satisfiable without touching the constructor, so a
    /// model can score as migrated while still taking every parameter it always did —
    /// <c>UnifiedMultimodalNetwork</c> did exactly that for the whole of phase 3, and it was
    /// caught by reading the file rather than by this test.
    /// </para>
    /// <para>
    /// This count admits no such credit, so it can only fall when a constructor actually stops
    /// taking the parameter. Where the two disagree, this one is the truth.
    /// </para>
    /// <para>
    /// Movement, newest last. Both counts falling by the SAME amount is itself diagnostic: it
    /// means none of the migrated options classes declared a property of a matching name, so none
    /// had been drawing name credit.
    /// </para>
    /// <para>
    /// 307 to 151 in one step when the segmentation family moved: 62 models x numClasses,
    /// dropRate and (for 32 of them) modelSize. Both counts fell by the same 156, which is itself
    /// the diagnostic — not one of those 62 options classes declared a property of a matching
    /// name, so none of them had been drawing name credit.
    /// </para>
    /// <para>
    /// 151 to 104 when the maxGradNorm cluster moved: 47 models, one parameter each. Both counts
    /// again fell by the same amount, for the same reason as the segmentation family — none of
    /// those options classes declared a property of a matching name, because 33 of them sat
    /// outside the <c>ModelHyperparameterOptions</c> hierarchy entirely and so did not inherit
    /// <c>MaxGradNorm</c> at all.
    /// </para>
    /// <para>
    /// 104 to 92 with the forecaster numFeatures family: 12 models, one parameter each. The
    /// property went onto TimeSeriesRegressionOptions itself rather than being gained by a
    /// re-parent onto ModelHyperparameterOptions, because RegressionOptions also serves the
    /// classical non-neural regressors.
    /// </para>
    /// <para>
    /// 92 to 74 with the variant/inChannels cluster: 4 detection backbones and 8 diffusion text
    /// conditioners. Unlike the earlier clusters these models had NO options parameter at all, so
    /// one was added and twelve options classes created; the backbone ones are named
    /// <c>ResNetBackboneOptions</c> / <c>EfficientNetBackboneOptions</c> because the bare names
    /// already belong to the separate <c>ResNetNetwork</c> and <c>EfficientNetNetwork</c>
    /// classifiers.
    /// </para>
    /// <para>
    /// 74 to 21 with the final tail: 29 model types across NeuralNetworks, PhysicsInformed,
    /// Document and UncertaintyQuantification. This cluster is where the two ratchets converged —
    /// <see cref="Baseline"/> reached 0 in the same run — which confirms the 21-point spread
    /// between them had been exactly what it was recorded as: constructors still taking a
    /// parameter whose options property already existed, not a detector disagreement.
    /// </para>
    /// <para>
    /// 21 to 0, closing this measure. The 21 reduced to exactly two shapes, both of which read as
    /// working configurability. Four probabilistic forecasters (CSDI, DiffusionTS, ScoreGrad,
    /// TSDiff) reconciled parameter and options with
    /// <c>_numFeatures = numFeatures &gt; 0 ? numFeatures : _options.NumFeatures;</c> — so the
    /// options value applied ONLY when a caller passed zero or less, and the parameter's own
    /// default of 1 shadowed it for everyone else. The other six (DGCNN, PointNet,
    /// PointNetPlusPlus, GaussianSplatting, MeshCNN, SpiralNet) had convenience constructors
    /// forwarding scalars into an options object initializer, applying the parameter copy last.
    /// Every one of those 11 defaults was checked against its options property and matched, so the
    /// removals changed no behaviour. Parameters with no default (numClasses, samplingRates) and
    /// collaborators (lossFunction, optimizer) stayed: a parameter with no default is a required
    /// input, not a duplicated value.
    /// </para>
    /// <para>
    /// Zero here means no in-scope model constructor declares a tunable defaulted parameter. It
    /// does NOT mean issue #2090 is closed — this counts one of six defect forms. The unread
    /// ratchet still stands at 97, <c>UncoveredBaseline</c> at 81, and the forms with no detector
    /// at all (hardcoded literals shadowing an option, constructors disagreeing about a default,
    /// bare optimizers, factories discarding tuned parameters, doc examples that cannot compile)
    /// remain findable only by reading code.
    /// </para>
    /// </remarks>
    private const int ConstructorBaseline = 0;

    /// <summary>
    /// How far the measured count may sit below <see cref="Baseline"/> before the test insists
    /// the baseline be lowered. Keeps ordinary refactoring from being blocked by an
    /// off-by-a-couple drift while still forcing real progress to be recorded.
    /// </summary>
    private const int Slack = 10;

    /// <summary>
    /// Constructor parameters that are collaborators or artifacts rather than hyperparameters.
    /// </summary>
    private static readonly HashSet<string> ExcludedParameterNames =
        new HashSet<string>(StringComparer.OrdinalIgnoreCase)
        {
            "modelIdentity", "modelPath", "name", "seed", "checkpointPath", "weightsPath",
        };

    /// <summary>
    /// Types that are not models in the sense this ratchet cares about: architecture
    /// descriptors, whose topology knobs belong on the architecture per the design, and
    /// compiled-model hosts, whose parameters describe an artifact rather than a model.
    /// </summary>
    private static readonly HashSet<string> ExcludedTypeNames =
        new HashSet<string>(StringComparer.Ordinal)
        {
            "NeuralNetworkArchitecture",
            "TransformerArchitecture",
            "DualStreamArchitecture",
            "TripleStreamArchitecture",
            "AudioTextDualStreamArchitecture",
            "CompiledModelHost",
            "ChainedCompiledModelHost",
        };

    [Fact]
    public void ModelConstructorParametersHaveOptionsEquivalents_DoesNotRegress()
    {
        var gaps = MeasureGaps();
        int count = gaps.Count;

        Assert.True(
            count <= Baseline,
            BuildFailureMessage(
                $"The options-surface gap grew from {Baseline} to {count}.",
                "A model constructor gained a tunable defaulted parameter. Put the value on that "
                    + "model's Options class instead, and have the constructor read it.",
                gaps));

        Assert.True(
            count >= Baseline - Slack,
            BuildFailureMessage(
                $"The options-surface gap fell from {Baseline} to {count}. That is the goal — "
                    + $"now lower the Baseline constant in {nameof(OptionsSurfaceRatchetTests)} to {count}.",
                "The baseline only descends deliberately, so that progress is recorded in the "
                    + "diff rather than silently absorbed.",
                gaps));
    }

    /// <summary>
    /// Pins the fix rather than its shape: a property that exists but is ignored still leaves
    /// the model unconfigurable, which is the exact defect this work addresses.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Enabled in phase 2, when the first model family is wired to read its options. Until
    /// then there is nothing for it to assert against — no model reads a value off its options
    /// object, so every case would fail for the reason the ratchet above already records.
    /// </para>
    /// </remarks>
    [Fact(Skip = "Enabled in phase 2 of #2090, when the first model family reads its options.")]
    public void SettingAnOptionsPropertyChangesTheModel()
    {
        throw new NotImplementedException(
            "Phase 2: for each migrated model, constructing it with a non-default Options value "
                + "must produce a model whose GetOptions() returns that value and whose layer "
                + "stack differs from the default-constructed one.");
    }

    /// <summary>
    /// Reports the current gap, so a migration PR can see exactly what remains.
    /// </summary>
    [Fact]
    public void ReportRemainingGaps()
    {
        var gaps = MeasureGaps();
        var byType = gaps.GroupBy(g => g.TypeName)
            .OrderByDescending(g => g.Count())
            .ThenBy(g => g.Key, StringComparer.Ordinal)
            .ToList();

        // This test exists to make the numbers visible when a migration lands, so it has to
        // actually emit them. It previously computed byType and then asserted only
        // `byType.Count >= 0` — always true, with nothing written anywhere — so it reported
        // nothing at all while reading as though it did.
        _output.WriteLine($"NAME-CREDITED ({nameof(Baseline)}): {gaps.Count} gaps across "
            + $"{byType.Count} model types.");
        foreach (var group in byType)
        {
            _output.WriteLine($"  {group.Key} ({group.Count()}): "
                + string.Join(", ", group.Select(g => g.ParameterName).OrderBy(n => n, StringComparer.Ordinal)));
        }

        // Once the name-credited count reaches zero this report goes silent, which is exactly when
        // the remaining work becomes invisible: the STRICT measure is what still has entries, and
        // it was only ever printed by ConstructorBaseline's failure message — so it could not be
        // read at all while that test was passing. Report both.
        var strict = MeasureRemaining();
        var strictByType = strict.GroupBy(g => g.TypeName)
            .OrderByDescending(g => g.Count())
            .ThenBy(g => g.Key, StringComparer.Ordinal)
            .ToList();

        _output.WriteLine(string.Empty);
        _output.WriteLine($"STRICT ({nameof(ConstructorBaseline)}): {strict.Count} parameters across "
            + $"{strictByType.Count} model types.");
        foreach (var group in strictByType)
        {
            _output.WriteLine($"  {group.Key} ({group.Count()}): "
                + string.Join(", ", group.Select(g => g.ParameterName).OrderBy(n => n, StringComparer.Ordinal)));
        }

        Assert.True(byType.Count >= 0);
    }

    [Fact]
    public void ModelConstructorsDoNotDeclareTunableParameters_DoesNotRegress()
    {
        var remaining = MeasureRemaining();
        int count = remaining.Count;

        Assert.True(
            count <= ConstructorBaseline,
            BuildFailureMessage(
                $"Tunable constructor parameters grew from {ConstructorBaseline} to {count}.",
                "A model constructor gained a tunable defaulted parameter, or one that was "
                    + "supposed to move to Options is still declared. Unlike the count above, "
                    + "this one cannot be satisfied by adding a property.",
                remaining));

        Assert.True(
            count >= ConstructorBaseline - Slack,
            BuildFailureMessage(
                $"Tunable constructor parameters fell from {ConstructorBaseline} to {count}. "
                    + $"Lower the ConstructorBaseline constant in {nameof(OptionsSurfaceRatchetTests)} to {count}.",
                "The baseline only descends deliberately, so progress is recorded in the diff.",
                remaining));
    }

    /// <summary>
    /// Every tunable defaulted constructor parameter on an in-scope model, with no allowance
    /// for a matching options property.
    /// </summary>
    private static List<Gap> MeasureRemaining()
    {
        var remaining = new List<Gap>();

        foreach (var model in GetModelTypes())
        {
            // Resolved in a FIRST pass over every constructor, because most models declare their
            // options parameter LAST. Resolving it lazily while walking parameters in declaration
            // order labelled each preceding tunable "(no options parameter)" even when the model
            // took a perfectly good options object -- FEDformer reported that way while
            // FEDformerOptions<T> declared all twelve of the properties in question. The COUNT was
            // never affected, only the label, but the label is what a reader plans from.
            Type? optionsType = ResolveOptionsType(model);
            var seen = new HashSet<string>(StringComparer.Ordinal);

            foreach (var ctor in model.GetConstructors(BindingFlags.Public | BindingFlags.Instance))
            {
                foreach (var parameter in ctor.GetParameters())
                {
                    var parameterType = Nullable.GetUnderlyingType(parameter.ParameterType)
                        ?? parameter.ParameterType;

                    if (IsOptionsType(parameterType)) { continue; }
                    if (!parameter.HasDefaultValue || parameter.Name == null) continue;
                    if (ExcludedParameterNames.Contains(parameter.Name)) continue;
                    if (!IsTunable(parameterType)) continue;
                    if (!seen.Add(parameter.Name)) continue;

                    remaining.Add(new Gap
                    {
                        TypeName = StripArity(model.Name),
                        ParameterName = parameter.Name,
                        OptionsTypeName = optionsType == null
                            ? "(no options parameter)"
                            : StripArity(optionsType.Name),
                    });
                }
            }
        }

        return remaining;
    }

    private sealed class Gap
    {
        public string TypeName { get; set; } = string.Empty;

        public string ParameterName { get; set; } = string.Empty;

        public string OptionsTypeName { get; set; } = string.Empty;
    }

    /// <summary>
    /// Every concrete model type in the AiDotNet assembly, found by walking the base chain.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Most models do not name <c>NeuralNetworkBase&lt;T&gt;</c> directly: <c>BGE</c> derives
    /// from <c>TransformerEmbeddingNetwork</c>, <c>MambaLanguageModel</c> from
    /// <c>TokenLanguageModelLayoutBase</c>, <c>TrOCR</c> from
    /// <c>DocumentNeuralNetworkBase</c>. A search of the source for the base name finds only a
    /// handful of files, so no naming or path heuristic identifies models correctly — walking
    /// the inheritance chain is the only approach that does.
    /// </para>
    /// </remarks>
    private static IEnumerable<Type> GetModelTypes()
    {
        Type[] types;
        try
        {
            types = typeof(NeuralNetworkBase<>).Assembly.GetTypes();
        }
        catch (ReflectionTypeLoadException ex)
        {
            // A type that fails to load cannot be measured, but the ones that did load still can.
            types = ex.Types.Where(t => t != null).ToArray()!;
        }

        foreach (var type in types)
        {
            if (type == null || type.IsAbstract || type.IsInterface || !type.IsClass) continue;
            if (!type.IsPublic && !type.IsNestedPublic) continue;

            string simpleName = StripArity(type.Name);
            if (ExcludedTypeNames.Contains(simpleName)) continue;

            if (DerivesFromNeuralNetworkBase(type)) yield return type;
        }
    }

    private static bool DerivesFromNeuralNetworkBase(Type type)
    {
        for (var current = type.BaseType; current != null; current = current.BaseType)
        {
            if (current.IsGenericType
                && current.GetGenericTypeDefinition() == typeof(NeuralNetworkBase<>))
            {
                return true;
            }
        }

        return false;
    }

    private static List<Gap> MeasureGaps()
    {
        var gaps = new List<Gap>();

        foreach (var model in GetModelTypes())
        {
            Type? optionsType = null;
            var tunable = new Dictionary<string, ParameterInfo>(StringComparer.Ordinal);

            foreach (var ctor in model.GetConstructors(BindingFlags.Public | BindingFlags.Instance))
            {
                foreach (var parameter in ctor.GetParameters())
                {
                    var parameterType = Nullable.GetUnderlyingType(parameter.ParameterType)
                        ?? parameter.ParameterType;

                    if (IsOptionsType(parameterType))
                    {
                        optionsType ??= parameterType;
                        continue;
                    }

                    if (!parameter.HasDefaultValue) continue;
                    if (parameter.Name == null) continue;
                    if (ExcludedParameterNames.Contains(parameter.Name)) continue;
                    if (!IsTunable(parameterType)) continue;

                    // Union across overloads: the same knob offered by two constructors is one gap.
                    if (!tunable.ContainsKey(parameter.Name)) tunable[parameter.Name] = parameter;
                }
            }

            if (tunable.Count == 0) continue;

            var optionsProperties = optionsType == null
                ? new HashSet<string>(StringComparer.Ordinal)
                : new HashSet<string>(
                    optionsType.GetProperties(BindingFlags.Public | BindingFlags.Instance)
                        .Select(p => p.Name),
                    StringComparer.Ordinal);

            foreach (var parameterName in tunable.Keys)
            {
                if (optionsProperties.Contains(ToPascalCase(parameterName))) continue;

                gaps.Add(new Gap
                {
                    TypeName = StripArity(model.Name),
                    ParameterName = parameterName,
                    OptionsTypeName = optionsType == null ? "(no options parameter)" : StripArity(optionsType.Name),
                });
            }
        }

        return gaps;
    }

    /// <summary>
    /// A parameter carries a hyperparameter when it is a scalar or an enum. Interfaces,
    /// delegates and model classes are collaborators, not configuration.
    /// </summary>
    private static bool IsTunable(Type type)
    {
        if (type.IsEnum) return true;

        return type == typeof(int) || type == typeof(long) || type == typeof(double)
            || type == typeof(float) || type == typeof(bool) || type == typeof(decimal)
            || type == typeof(string);
    }

    /// <summary>
    /// The options type a model accepts, taken from any of its public constructors.
    /// </summary>
    /// <param name="model">The model type.</param>
    /// <returns>The options type, or null when no constructor accepts one.</returns>
    private static Type? ResolveOptionsType(Type model)
    {
        foreach (var ctor in model.GetConstructors(BindingFlags.Public | BindingFlags.Instance))
        {
            foreach (var parameter in ctor.GetParameters())
            {
                var parameterType = Nullable.GetUnderlyingType(parameter.ParameterType)
                    ?? parameter.ParameterType;

                if (IsOptionsType(parameterType)) return parameterType;
            }
        }

        return null;
    }

    private static bool IsOptionsType(Type type)
    {
        if (typeof(ModelOptions).IsAssignableFrom(type)) return true;

        // Some options classes predate the ModelOptions hierarchy and stand alone.
        return type.Name.StartsWith("Options", StringComparison.Ordinal)
            || StripArity(type.Name).EndsWith("Options", StringComparison.Ordinal);
    }

    private static string StripArity(string typeName)
    {
        int tick = typeName.IndexOf('`');
        return tick < 0 ? typeName : typeName.Substring(0, tick);
    }

    private static string ToPascalCase(string parameterName)
    {
        if (parameterName.Length == 0) return parameterName;
        return char.ToUpperInvariant(parameterName[0]) + parameterName.Substring(1);
    }

    private static string BuildFailureMessage(string headline, string guidance, List<Gap> gaps)
    {
        var message = new StringBuilder();
        message.AppendLine(headline);
        message.AppendLine(guidance);
        message.AppendLine();
        message.AppendLine("Largest remaining gaps:");

        foreach (var group in gaps.GroupBy(g => g.TypeName)
                     .OrderByDescending(g => g.Count())
                     .ThenBy(g => g.Key, StringComparer.Ordinal)
                     .Take(15))
        {
            message.AppendLine(
                $"  {group.Key} ({group.Count()}) options={group.First().OptionsTypeName}: "
                    + string.Join(", ", group.Select(g => g.ParameterName).Take(8)));
        }

        return message.ToString();
    }
}
