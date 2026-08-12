using System;
using System.Collections.Generic;
using System.Linq;
using AiDotNet.Tests.ModelFamilyTests.Base;
using Xunit;

namespace AiDotNet.Tests.Generators;

/// <summary>
/// End-to-end smoke test for the #1679 float scaffolds. Rather than snapshot raw generator output,
/// this inspects the scaffolds the generator ACTUALLY emitted into this test assembly (namespace
/// <c>AiDotNet.Tests.ModelFamilyTests.Generated</c>) and proves, by reflection on their compiled base
/// types, that the float rewrite fired exactly where intended and nowhere else:
///   * training-perf-bound models (in Fp32TestClassNames / [GenerateFloatTestScaffold]) inherit a
///     <c>&lt;float&gt;</c> test base, and
///   * models in supported generic families inherit a <c>&lt;float&gt;</c> test base regardless of
///     their name's initial, while explicit exclusions still inherit <c>&lt;double&gt;</c>.
/// Because it reads the compiled output, it also confirms the rewritten scaffolds compile — the
/// minimum bar the review asked for.
/// </summary>
public class GeneratedFloatScaffoldSmokeTests
{
    private const string GeneratedNamespace = "AiDotNet.Tests.ModelFamilyTests.Generated";

    private static List<Type> GeneratedScaffolds()
        => typeof(GeneratedFloatScaffoldSmokeTests).Assembly.GetTypes()
            .Where(t => t.IsClass && t.Namespace == GeneratedNamespace)
            .ToList();

    /// <summary>
    /// Walks the base-type chain and returns the element type of the first generic test base:
    /// <c>typeof(float)</c> for a float scaffold, <c>typeof(double)</c> for the default, or null if
    /// the family base is non-generic. Handles the non-generic alias pattern
    /// (<c>Base : Base&lt;double&gt;</c>) because the loop continues into the alias's own base.
    /// </summary>
    private static Type? ScaffoldPrecision(Type scaffold)
    {
        for (var b = scaffold.BaseType; b != null && b != typeof(object); b = b.BaseType)
        {
            if (b.IsGenericType)
            {
                foreach (var arg in b.GetGenericArguments())
                {
                    if (arg == typeof(float)) return typeof(float);
                    if (arg == typeof(double)) return typeof(double);
                }
            }
        }
        return null;
    }

    [Fact]
    public void GeneratedScaffolds_FloatRewrite_FiresForSomeModelsAndNotAll()
    {
        var scaffolds = GeneratedScaffolds();
        Assert.True(scaffolds.Count > 0,
            "The TestScaffoldGenerator produced no scaffolds in the Generated namespace — the generator " +
            "did not run, so the float rewrite cannot be verified.");

        var floatScaffolds = scaffolds.Where(t => ScaffoldPrecision(t) == typeof(float)).ToList();
        var doubleScaffolds = scaffolds.Where(t => ScaffoldPrecision(t) == typeof(double)).ToList();

        // The #1679 float path must actually fire end-to-end for at least one model (and the emitted
        // <float> scaffold must compile, or this assembly would not have built).
        Assert.True(floatScaffolds.Count > 0,
            "No generated scaffold inherits a <float> test base. The #1679 float rewrite did not fire " +
            "for any model — every training-perf-bound model would still run in <double>.");

        // ...and we must NOT have accidentally floated every model.
        Assert.True(doubleScaffolds.Count > 0,
            "No generated scaffold inherits a <double> test base — the float rewrite leaked to all models.");

        // A bare count still passes if the float path regresses to one accidental model. Pin each
        // selection route and every newly eligible family by name, plus stable explicit double
        // exclusions. LayoutGraph is deliberate: its L initial used to keep it at double and proves
        // the obsolete shard-letter gate did not return.
        Assert.Contains(floatScaffolds, t => t.Name == "ABINetTests");
        Assert.Contains(floatScaffolds, t => t.Name == "WhisperLargeV3Tests");
        Assert.Contains(floatScaffolds, t => t.Name == "CIFEncoderTests");
        Assert.Contains(floatScaffolds, t => t.Name == "BasicVSRTests");
        Assert.Contains(floatScaffolds, t => t.Name == "OuteTTSTests");
        Assert.Contains(floatScaffolds, t => t.Name == "AmphionTests");
        Assert.Contains(floatScaffolds, t => t.Name == "CLAPModelTests");
        Assert.Contains(floatScaffolds, t => t.Name == "DiaTests");
        Assert.Contains(floatScaffolds, t => t.Name == "MATCHATests");
        Assert.Contains(floatScaffolds, t => t.Name == "CUPSTests");
        Assert.Contains(floatScaffolds, t => t.Name == "ContextNetTests");
        Assert.Contains(floatScaffolds, t => t.Name == "CodeSwitchingASRTests");
        Assert.Contains(floatScaffolds, t => t.Name == "Chirp2Tests");
        Assert.Contains(floatScaffolds, t => t.Name == "FlowDiffuserTests");
        Assert.Contains(floatScaffolds, t => t.Name == "FLIPTests");
        Assert.Contains(floatScaffolds, t => t.Name == "Gemma3Tests");
        Assert.Contains(floatScaffolds, t => t.Name == "MemFlowTests");
        Assert.Contains(floatScaffolds, t => t.Name == "MiniGPT4Tests");
        Assert.Contains(floatScaffolds, t => t.Name == "SpeechGPTTests");
        Assert.Contains(floatScaffolds, t => t.Name == "PixelLMTests");
        Assert.Contains(floatScaffolds, t => t.Name == "OpenCLIPTests");
        Assert.Contains(floatScaffolds, t => t.Name == "PointTransformerV3Tests");
        Assert.Contains(floatScaffolds, t => t.Name == "PyramidNERTests");
        Assert.Contains(floatScaffolds, t => t.Name == "PerVFITests");
        Assert.Contains(floatScaffolds, t => t.Name == "PIDNetTests");
        Assert.Contains(floatScaffolds, t => t.Name == "LayoutGraphTests");
        Assert.Contains(floatScaffolds, t => t.Name == "ZScoreDetectorTests");
        Assert.Contains(floatScaffolds, t => t.Name == "NeuralCVaRTests");
        Assert.Contains(floatScaffolds, t => t.Name == "BSVDTests");
        Assert.Contains(floatScaffolds, t => t.Name == "AVIDTests");
        Assert.Contains(floatScaffolds, t => t.Name == "DIFRINTTests");
        Assert.Contains(floatScaffolds, t => t.Name == "DQNAgentTests");
        Assert.Contains(floatScaffolds, t => t.Name == "ActivationLayerTests");
        Assert.Contains(floatScaffolds, t => t.Name == "BentIdentityActivationTests");
        Assert.Contains(floatScaffolds, t => t.Name == "CharbonnierLossTests");
        Assert.Contains(floatScaffolds, t => t.Name == "TripletLossTests");
        Assert.Contains(floatScaffolds, t => t.Name == "NoiseContrastiveEstimationLossTests");
        Assert.Contains(floatScaffolds, t => t.Name == "SparseCategoricalCrossEntropyLossTests");
        Assert.Contains(floatScaffolds, t => t.Name == "GraphAttentionLayerTests");
        Assert.Contains(floatScaffolds, t => t.Name == "CrossAttentionLayerTests");
        Assert.Contains(floatScaffolds, t => t.Name == "AddLayerTests");
        Assert.Contains(floatScaffolds, t => t.Name == "ProbabilisticDistillationStrategyTests");
        Assert.Contains(floatScaffolds, t => t.Name == "BALDTests");

        Assert.Contains(doubleScaffolds, t => t.Name == "GraFPrintTests");
        Assert.Contains(doubleScaffolds, t => t.Name == "SambaLanguageModelTests");
        Assert.Contains(doubleScaffolds, t => t.Name == "TabPFNNetworkTests");
    }

    [Fact]
    public void FloatLayerBase_DiscoversFloatActivationImplementations()
    {
        var names = LayerTestBase<float>.DiscoveredActivationNames
            .Select(values => Assert.IsType<string>(values[0]))
            .ToList();

        Assert.NotEmpty(names);
        Assert.Contains("ReLUActivation", names);
    }

    [Fact]
    public void GeneratedFloatScaffolds_AreOnlyEverFloatOrDouble_NeverMalformed()
    {
        // Every generic-family scaffold must resolve to exactly float or double — never some other
        // type argument from a botched rewrite (e.g. a partially-rewritten <flat> or a leaked <T>).
        foreach (var scaffold in GeneratedScaffolds())
        {
            var precision = ScaffoldPrecision(scaffold);
            Assert.True(precision is null || precision == typeof(float) || precision == typeof(double),
                $"Generated scaffold {scaffold.Name} resolved to an unexpected precision '{precision}'.");
        }
    }


}
