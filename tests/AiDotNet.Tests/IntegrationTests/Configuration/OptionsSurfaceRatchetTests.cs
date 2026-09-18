using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using System.Text;
using AiDotNet.Enums;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.NeuralNetworks.Layers.SSM;
using AiDotNet.NeuralNetworks.Options;
using Xunit;
using Xunit.Abstractions;

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
    private readonly ITestOutputHelper _output;

    public OptionsSurfaceRatchetTests(ITestOutputHelper output)
    {
        // .NET Framework does not invoke the assembly's module initializer.
        TestModuleInitializer.EnsureInitialized();
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
    [Theory]
    [MemberData(nameof(SequenceConfigurationCases))]
    public void SettingAnOptionsPropertyChangesTheModel(SequenceFamily family, SequenceChange change)
    {
        var baseline = CreateSequenceModel(family, width: 16, depth: 1);
        using var baselineModel = baseline.Model;
        var changed = CreateSequenceModel(family,
            width: change == SequenceChange.Width ? 32 : 16,
            depth: change == SequenceChange.Depth ? 2 : 1);
        using var changedModel = changed.Model;

        var expectedModelType = RequiredSequenceModels[family].MakeGenericType(typeof(float));
        Assert.Equal(expectedModelType, baselineModel.GetType());
        Assert.Equal(expectedModelType, changedModel.GetType());
        Assert.Same(baseline.Options, baselineModel.GetOptions());
        Assert.Same(changed.Options, changedModel.GetOptions());
        AssertMaterializedChange(baselineModel, changedModel, change);
        _output.WriteLine($"{family}/{change}: materialized parameters {baselineModel.ParameterCount} -> "
            + $"{changedModel.ParameterCount}; physical block/layer count {TopologySize(baselineModel)} -> {TopologySize(changedModel)}.");
    }

    [Theory]
    [InlineData(SequenceFamily.Mamba)]
    [InlineData(SequenceFamily.Jamba)]
    public void BehavioralGuard_RejectsOptionsEchoWithoutTopologyChange(SequenceFamily family)
    {
        var baseline = CreateSequenceModel(family, width: 16, depth: 1);
        using var baselineModel = baseline.Model;
        var unchanged = CreateSequenceModel(family, width: 16, depth: 1);
        using var unchangedModel = unchanged.Model;
        // Real models with identical topology, but GetOptions now advertises another depth.
        // This reproduces the false-proof shape without any replacement model or fake layer.
        unchanged.Options.NumLayers = 2;
        Assert.Same(unchanged.Options, unchangedModel.GetOptions());
        Assert.Equal(2, Assert.IsAssignableFrom<SequenceModelOptions>(unchangedModel.GetOptions()).NumLayers);
        var failure = Assert.ThrowsAny<Xunit.Sdk.XunitException>(() =>
            AssertMaterializedChange(baselineModel, unchangedModel, SequenceChange.Depth));
        Assert.Contains("Physical topology does not match", failure.Message);
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

        _output.WriteLine($"Total gaps: {gaps.Count} across {byType.Count} model types; "
            + $"scanned {GetModelTypes().Length} concrete models.");
        foreach (var group in byType)
        {
            _output.WriteLine($"  {group.Key} ({group.Count()}) options={group.First().OptionsTypeName}: "
                + string.Join(", ", group.Select(gap => gap.ParameterName)));
        }
        Assert.Equal(gaps.Count, byType.Sum(group => group.Count()));
    }

    [Fact]
    public void GapReport_WritesTheMeasuredCountsAndEveryGroup()
    {
        var capture = new CapturingOutput();
        new OptionsSurfaceRatchetTests(capture).ReportRemainingGaps();
        var gaps = MeasureGaps();
        var groups = gaps.GroupBy(gap => gap.TypeName).OrderByDescending(group => group.Count())
            .ThenBy(group => group.Key, StringComparer.Ordinal).ToArray();
        Assert.Equal(groups.Length + 1, capture.Lines.Count);
        Assert.Equal($"Total gaps: {gaps.Count} across {groups.Length} model types; "
            + $"scanned {GetModelTypes().Length} concrete models.", capture.Lines[0]);
        for (var index = 0; index < groups.Length; index++)
        {
            Assert.Contains(groups[index].Key, capture.Lines[index + 1]);
            Assert.All(groups[index], gap => Assert.Contains(gap.ParameterName, capture.Lines[index + 1]));
        }
    }

    private sealed class CapturingOutput : ITestOutputHelper
    {
        public List<string> Lines { get; } = new();
        public void WriteLine(string message) => Lines.Add(message);
        public void WriteLine(string format, params object[] args) => Lines.Add(string.Format(format, args));
    }

    public enum SequenceFamily
    {
        Eagle, FalconMamba, Finch, GatedDeltaNet, GLA, Griffin, Hawk, Jamba,
        Mamba2, Mamba, RecurrentGemma, RWKV4, RWKV7, Samba, XLSTM, Zamba2, Zamba
    }

    public enum SequenceChange { Width, Depth }

    public static IEnumerable<object[]> SequenceConfigurationCases =>
        from family in Enum.GetValues(typeof(SequenceFamily)).Cast<SequenceFamily>()
        from change in Enum.GetValues(typeof(SequenceChange)).Cast<SequenceChange>()
        select new object[] { family, change };

    private static readonly IReadOnlyDictionary<SequenceFamily, Type> RequiredSequenceModels = new Dictionary<SequenceFamily, Type>
    {
        [SequenceFamily.Eagle] = typeof(EagleLanguageModel<>),
        [SequenceFamily.FalconMamba] = typeof(FalconMambaLanguageModel<>),
        [SequenceFamily.Finch] = typeof(FinchLanguageModel<>),
        [SequenceFamily.GatedDeltaNet] = typeof(GatedDeltaNetLanguageModel<>),
        [SequenceFamily.GLA] = typeof(GLALanguageModel<>),
        [SequenceFamily.Griffin] = typeof(GriffinLanguageModel<>),
        [SequenceFamily.Hawk] = typeof(HawkLanguageModel<>),
        [SequenceFamily.Jamba] = typeof(JambaLanguageModel<>),
        [SequenceFamily.Mamba2] = typeof(Mamba2LanguageModel<>),
        [SequenceFamily.Mamba] = typeof(MambaLanguageModel<>),
        [SequenceFamily.RecurrentGemma] = typeof(RecurrentGemmaLanguageModel<>),
        [SequenceFamily.RWKV4] = typeof(RWKV4LanguageModel<>),
        [SequenceFamily.RWKV7] = typeof(RWKV7LanguageModel<>),
        [SequenceFamily.Samba] = typeof(SambaLanguageModel<>),
        [SequenceFamily.XLSTM] = typeof(XLSTMLanguageModel<>),
        [SequenceFamily.Zamba2] = typeof(Zamba2LanguageModel<>),
        [SequenceFamily.Zamba] = typeof(ZambaLanguageModel<>)
    };

    [Fact]
    public void SequenceCohort_CoversEveryMigratedOptionsType()
    {
        var optionsTypes = typeof(SequenceModelOptions).Assembly.GetTypes()
            .Where(type => !type.IsAbstract && typeof(SequenceModelOptions).IsAssignableFrom(type))
            .OrderBy(type => type.FullName, StringComparer.Ordinal).ToArray();
        var consumedTypes = RequiredSequenceModels.Values.SelectMany(type => type.GetConstructors())
            .SelectMany(constructor => constructor.GetParameters()).Select(parameter => parameter.ParameterType)
            .Where(type => typeof(SequenceModelOptions).IsAssignableFrom(type)).Distinct()
            .OrderBy(type => type.FullName, StringComparer.Ordinal).ToArray();
        Assert.Equal(17, Enum.GetValues(typeof(SequenceFamily)).Length);
        Assert.Equal(17, RequiredSequenceModels.Count);
        Assert.Equal(Enum.GetValues(typeof(SequenceFamily)).Cast<SequenceFamily>().OrderBy(family => family),
            RequiredSequenceModels.Keys.OrderBy(family => family));
        Assert.Equal(optionsTypes, consumedTypes);
        Assert.Equal(17, optionsTypes.Length);
    }

    private static (NeuralNetworkBase<float> Model, SequenceModelOptions Options) CreateSequenceModel(
        SequenceFamily family, int width, int depth) => family switch
        {
            SequenceFamily.Eagle => Create(new EagleOptions(), (architecture, options) => new EagleLanguageModel<float>(architecture, options), width, depth),
            SequenceFamily.FalconMamba => Create(new FalconMambaOptions(), (architecture, options) => new FalconMambaLanguageModel<float>(architecture, options), width, depth),
            SequenceFamily.Finch => Create(new FinchOptions(), (architecture, options) => new FinchLanguageModel<float>(architecture, options), width, depth),
            SequenceFamily.GatedDeltaNet => Create(new GatedDeltaNetOptions(), (architecture, options) => new GatedDeltaNetLanguageModel<float>(architecture, options), width, depth),
            SequenceFamily.GLA => Create(new GLAOptions(), (architecture, options) => new GLALanguageModel<float>(architecture, options), width, depth),
            SequenceFamily.Griffin => Create(new GriffinOptions { RecurrenceDimension = 16 }, (architecture, options) => new GriffinLanguageModel<float>(architecture, options), width, depth),
            SequenceFamily.Hawk => Create(new HawkOptions { RecurrenceDimension = 16 }, (architecture, options) => new HawkLanguageModel<float>(architecture, options), width, depth),
            SequenceFamily.Jamba => Create(new JambaOptions(), (architecture, options) => new JambaLanguageModel<float>(architecture, options), width, depth),
            SequenceFamily.Mamba2 => Create(new Mamba2Options(), (architecture, options) => new Mamba2LanguageModel<float>(architecture, options), width, depth),
            SequenceFamily.Mamba => Create(new MambaOptions(), (architecture, options) => new MambaLanguageModel<float>(architecture, options), width, depth),
            SequenceFamily.RecurrentGemma => Create(new RecurrentGemmaOptions(), (architecture, options) => new RecurrentGemmaLanguageModel<float>(architecture, options), width, depth),
            SequenceFamily.RWKV4 => Create(new RWKV4Options(), (architecture, options) => new RWKV4LanguageModel<float>(architecture, options), width, depth),
            SequenceFamily.RWKV7 => Create(new RWKV7Options(), (architecture, options) => new RWKV7LanguageModel<float>(architecture, options), width, depth),
            SequenceFamily.Samba => Create(new SambaOptions(), (architecture, options) => new SambaLanguageModel<float>(architecture, options), width, depth),
            SequenceFamily.XLSTM => Create(new XLSTMOptions(), (architecture, options) => new XLSTMLanguageModel<float>(architecture, options), width, depth),
            SequenceFamily.Zamba2 => Create(new Zamba2Options(), (architecture, options) => new Zamba2LanguageModel<float>(architecture, options), width, depth),
            SequenceFamily.Zamba => Create(new ZambaOptions(), (architecture, options) => new ZambaLanguageModel<float>(architecture, options), width, depth),
            _ => throw new ArgumentOutOfRangeException(nameof(family), family, "Unknown sequence family.")
        };

    private static (NeuralNetworkBase<float> Model, SequenceModelOptions Options) Create<TOptions>(TOptions options,
        Func<NeuralNetworkArchitecture<float>, TOptions, NeuralNetworkBase<float>> construct, int width, int depth)
        where TOptions : SequenceModelOptions
    {
        options.Seed = 42;
        options.VocabSize = 32;
        options.ModelDimension = width;
        options.NumLayers = depth;
        options.NumHeads = 2;
        options.StateDimension = 8;
        options.MaxSequenceLength = 4;
        options.AttentionInterval = 2;
        options.ExpandFactor = 2;
        options.FfnMultiplier = 3.5;
        var architecture = new NeuralNetworkArchitecture<float>(InputType.OneDimensional,
            NeuralNetworkTaskType.TextGeneration, inputSize: 32, outputSize: 32);
        return (construct(architecture, options), options);
    }

    private static int TopologySize(NeuralNetworkBase<float> model) => model.Layers.Sum(layer => layer switch
    {
        HybridBlockScheduler<float> scheduler => scheduler.NumBlocks,
        Rwkv7Stack<float> stack => stack.Blocks.Count,
        _ => 1
    });

    private static void AssertMaterializedChange(NeuralNetworkBase<float> baseline,
        NeuralNetworkBase<float> changed, SequenceChange change)
    {
        Assert.InRange(baseline.LayerCount, 4, 8);
        Assert.InRange(changed.LayerCount, 4, 8);
        var baselineEmbedding = Assert.IsType<EmbeddingLayer<float>>(baseline.Layers[0]);
        var changedEmbedding = Assert.IsType<EmbeddingLayer<float>>(changed.Layers[0]);
        // Check an actual parameter tensor, not a model metadata/property echo.
        Assert.Equal(32 * 16, baselineEmbedding.GetParameters().Length);
        Assert.Equal(32 * (change == SequenceChange.Width ? 32 : 16), changedEmbedding.GetParameters().Length);
        Assert.True(TopologySize(baseline) + (change == SequenceChange.Depth ? 1 : 0) == TopologySize(changed),
            "Physical topology does not match the requested options change.");
        // Count first so a broken configuration cannot trigger paper-scale flattening.
        Assert.InRange(baseline.ParameterCount, 1, 1_000_000);
        Assert.InRange(changed.ParameterCount, 1, 1_000_000);
        var baselineParameters = baseline.GetParameters();
        var changedParameters = changed.GetParameters();
        Assert.Equal(baseline.ParameterCount, baselineParameters.Length);
        Assert.Equal(changed.ParameterCount, changedParameters.Length);
        Assert.True(changedParameters.Length > baselineParameters.Length,
            $"Changing {change} must increase materialized parameters, not just the advertised options.");
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
    private static Type[] GetModelTypes(Func<Type[]>? loadTypes = null)
    {
        Type[] types;
        try
        {
            types = (loadTypes ?? typeof(NeuralNetworkBase<>).Assembly.GetTypes)();
        }
        catch (ReflectionTypeLoadException ex)
        {
            throw new InvalidOperationException("Cannot measure the options surface from a partially loaded assembly. "
                + string.Join("; ", ex.LoaderExceptions.Where(error => error != null).Select(error => error?.Message)), ex);
        }

        var models = types.Where(type => type.IsClass && !type.IsAbstract
            && (type.IsPublic || type.IsNestedPublic) && DerivesFromNeuralNetworkBase(type)).ToArray();
        if (models.Length == 0)
            throw new InvalidOperationException("Model discovery returned no concrete neural-network models.");
        var missing = RequiredSequenceModels.Values.Except(models).ToArray();
        if (missing.Length > 0)
            throw new InvalidOperationException("Model discovery omitted migrated sequence models: "
                + string.Join(", ", missing.Select(type => type.FullName)));
        return models;
    }

    [Fact]
    public void ModelDiscovery_RejectsEmptyCensus()
    {
        var exception = Assert.Throws<InvalidOperationException>(() => GetModelTypes(() => Array.Empty<Type>()));
        Assert.Contains("no concrete", exception.Message);
    }

    [Fact]
    public void ModelDiscovery_RejectsIncompleteCohort()
    {
        var exception = Assert.Throws<InvalidOperationException>(() => GetModelTypes(() => new[] { typeof(RWKV7LanguageModel<>) }));
        Assert.Contains(typeof(MambaLanguageModel<>).FullName ?? nameof(MambaLanguageModel<float>), exception.Message);
    }

    [Fact]
    public void ModelDiscovery_DoesNotDiscardLoaderFailures()
    {
        var failure = new ReflectionTypeLoadException(new[] { typeof(RWKV7LanguageModel<>) },
            new Exception[] { new TypeLoadException("Required model dependency could not load.") });
        var exception = Assert.Throws<InvalidOperationException>(() => GetModelTypes(() => throw failure));
        Assert.Same(failure, exception.InnerException);
        Assert.Contains("Required model dependency could not load", exception.Message);
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

                    if (!HasDefaultValue(parameter)) continue;
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

    private static bool HasDefaultValue(ParameterInfo parameter)
    {
        try
        {
            return parameter.HasDefaultValue;
        }
        catch (ArgumentException) when (parameter.ParameterType.IsEnum && parameter.ParameterType.ContainsGenericParameters)
        {
            // The runtime attempts to box a nested enum from an open generic model. That is
            // not a constructible runtime type. Enum defaults are metadata constants, so only
            // this case can use the flag; decimal/DateTime attribute defaults keep the normal path.
            return (parameter.Attributes & ParameterAttributes.HasDefault) != 0;
        }
    }

    [Theory]
    [InlineData(typeof(DefaultMetadataSample<>))]
    [InlineData(typeof(DefaultMetadataSample<int>))]
    public void DefaultMetadata_PreservesRequiredOptionalAndAttributeConstants(Type declaringType)
    {
        var method = declaringType.GetMethod(nameof(DefaultMetadataSample<int>.Sample));
        Assert.NotNull(method);
        var parameters = method.GetParameters();
        Assert.Equal(new[] { false, false, false, true, true, true, true, true, true, true, true, true },
            parameters.Select(HasDefaultValue));
        // The decimal and DateTime controls specifically forbid a blanket HasDefault flag check.
        Assert.Equal(ParameterAttributes.None, parameters[3].Attributes & ParameterAttributes.HasDefault);
        Assert.Equal(ParameterAttributes.None, parameters[6].Attributes & ParameterAttributes.HasDefault);
        Assert.Equal(new DateTime(1234), parameters[3].DefaultValue);
        Assert.Equal(1.25m, parameters[6].DefaultValue);
    }

    public sealed class DefaultMetadataSample<T>
    {
        public enum NestedMode { Standard, Custom }

        public static void Sample(int required,
            [System.Runtime.InteropServices.Optional] int optionalWithoutDefault,
            NestedMode requiredMode,
            [System.Runtime.CompilerServices.DateTimeConstant(1234)] DateTime date,
            NestedMode mode = NestedMode.Custom,
            int number = 3,
            decimal amount = 1.25m,
            string? label = null,
            bool enabled = true,
            float fraction = 0.5f,
            double ratio = 0.25,
            long count = 4)
        {
        }
    }

    [Fact]
    public void ModelDiscovery_ArchitectureAndArtifactExclusionsDoNotChangeTheCensus()
    {
        var nonModels = new[]
        {
            typeof(NeuralNetworkArchitecture<>), typeof(TransformerArchitecture<>),
            typeof(DualStreamArchitecture<>), typeof(TripleStreamArchitecture<>),
            typeof(AudioTextDualStreamArchitecture<>), typeof(CompiledModelHost<>), typeof(ChainedCompiledModelHost<>)
        };
        Assert.All(nonModels, type => Assert.False(DerivesFromNeuralNetworkBase(type)));
        var models = GetModelTypes();
        // Compare with the exact previous name exclusions as a migration control, not a discovery policy.
        var oldExcludedNames = nonModels.Select(type => StripArity(type.Name)).ToArray();
        var oldCensus = models.Where(type => !oldExcludedNames.Contains(StripArity(type.Name))).ToArray();
        Assert.Equal(models, oldCensus);
        _output.WriteLine($"Current and previous-exclusion census: {models.Length} models; excluded model delta: {models.Length - oldCensus.Length}.");
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
