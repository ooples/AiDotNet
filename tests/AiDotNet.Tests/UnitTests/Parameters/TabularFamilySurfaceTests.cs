using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks.Tabular;
using AiDotNet.Tensors;
using Xunit;
using System;
using System.Collections.Generic;
using System.Threading.Tasks;

namespace AiDotNet.Tests.UnitTests.Parameters;

/// <summary>
/// Count-equals-vector and lossless-restore contracts for every Tabular family.
/// </summary>
/// <remarks>
/// <para>
/// Eight of the nine Tabular bases reported a <c>ParameterCount</c> and had NO <c>GetParameters</c>
/// and no <c>SetParameters</c> at all, and implemented no interface that would have required them.
/// Their two concrete variants each then OVERRODE the count to add a head. So the count grew, and
/// nothing in the library could read or restore a single one of those weights.
/// </para>
/// <para>
/// That state was invisible to the contract gate: <c>ParameterCount_ShouldMatchGetParameters</c>
/// compares two surfaces, and these models only had one. A count with no vector cannot disagree with
/// anything, so it was never reported wrong — it was unfalsifiable, which is worse. These tests make
/// it falsifiable for all eight at once.
/// </para>
/// <para>
/// FTTransformer is the control: it was already migrated and should behave identically, which is
/// what makes a pass here evidence about the others rather than a description of whatever the new
/// code happens to do.
/// </para>
/// </remarks>
public class TabularFamilySurfaceTests
{
    /// <summary>Every Tabular family, built at the same small size.</summary>
    public static IEnumerable<object[]> Families()
    {
        yield return Row("AutoInt", () => new AutoIntRegression<double>(4));
        yield return Row("GANDALF", () => new GANDALFRegression<double>(4));
        yield return Row("Mambular", () => new MambularRegression<double>(4));
        yield return Row("NODE", () => new NODERegression<double>(4));
        yield return Row("SAINT", () => new SAINTRegression<double>(4));
        yield return Row("TabDPT", () => new TabDPTRegression<double>(4));
        yield return Row("TabPFN", () => new TabPFNRegression<double>(4));
        // TabTransformer is a CATEGORICAL architecture -- its embeddings are per categorical
        // feature, so with none declared it genuinely owns nothing until a forward pass sizes the
        // encoder. Giving it cardinalities exercises the thing this family actually holds.
        yield return Row("TabTransformer", () => new TabTransformerRegression<double>(
            4, 1, new TabTransformerOptions<double> { CategoricalCardinalities = [5, 3] }));
        yield return Row("TabR", () => new TabRRegression<double>(4));
        yield return Row("FTTransformer", () => new FTTransformerRegression<double>(4));
    }

    private static object[] Row(string name, Func<IParameterSource<double>> factory)
        => new object[] { name, factory };

    /// <summary>Regression/classification variants that share the same backbone configuration.</summary>
    public static IEnumerable<object[]> VariantPairs()
    {
        yield return Pair("AutoInt", () => new AutoIntRegression<double>(4), () => new AutoIntClassifier<double>(4, 3));
        yield return Pair("GANDALF", () => new GANDALFRegression<double>(4), () => new GANDALFClassifier<double>(4, 3));
        yield return Pair("Mambular", () => new MambularRegression<double>(4), () => new MambularClassifier<double>(4, 3));
        yield return Pair("NODE", () => new NODERegression<double>(4), () => new NODEClassifier<double>(4, 3));
        yield return Pair("SAINT", () => new SAINTRegression<double>(4), () => new SAINTClassifier<double>(4, 3));
        yield return Pair("TabDPT", () => new TabDPTRegression<double>(4), () => new TabDPTClassifier<double>(4, 3));
        yield return Pair("TabPFN", () => new TabPFNRegression<double>(4), () => new TabPFNClassifier<double>(4, 3));
        yield return Pair("TabR", () => new TabRRegression<double>(4), () => new TabRClassifier<double>(4, 3));
        yield return Pair(
            "TabTransformer",
            () => new TabTransformerRegression<double>(4, 1,
                new TabTransformerOptions<double> { CategoricalCardinalities = [5, 3] }),
            () => new TabTransformerClassifier<double>(4, 3,
                new TabTransformerOptions<double> { CategoricalCardinalities = [5, 3] }));
        yield return Pair("FTTransformer", () => new FTTransformerRegression<double>(4), () => new FTTransformerClassifier<double>(4, 3));
    }

    private static object[] Pair(
        string name,
        Func<IParameterSource<double>> regressionFactory,
        Func<IParameterSource<double>> classifierFactory)
        => new object[] { name, regressionFactory, classifierFactory };

    /// <summary>Families whose declared dimensions are sufficient to run a first prediction.</summary>
    public static IEnumerable<object[]> FamiliesWithWarmup()
    {
        yield return WarmupRow("AutoInt", () => new AutoIntRegression<double>(4),
            model => _ = ((AutoIntRegression<double>)model).Predict(CreateNumericalFeatures()));
        yield return WarmupRow("GANDALF", () => new GANDALFRegression<double>(4),
            model => _ = ((GANDALFRegression<double>)model).Predict(CreateNumericalFeatures()));
        yield return WarmupRow("Mambular", () => new MambularRegression<double>(4),
            model => _ = ((MambularRegression<double>)model).Predict(CreateNumericalFeatures()));
        yield return WarmupRow("NODE", () => new NODERegression<double>(4),
            model => _ = ((NODERegression<double>)model).Predict(CreateNumericalFeatures()));
        yield return WarmupRow("SAINT", () => new SAINTRegression<double>(4),
            model => _ = ((SAINTRegression<double>)model).Predict(CreateNumericalFeatures()));
        yield return WarmupRow("TabDPT", () => new TabDPTRegression<double>(4),
            model => _ = ((TabDPTRegression<double>)model).Predict(CreateNumericalFeatures()));
        yield return WarmupRow("TabPFN", () => new TabPFNRegression<double>(4),
            model => _ = ((TabPFNRegression<double>)model).Predict(CreateNumericalFeatures()));
        yield return WarmupRow("TabR", () => new TabRRegression<double>(4),
            model =>
            {
                var tabR = (TabRRegression<double>)model;
                var features = CreateNumericalFeatures();
                tabR.BuildIndex(features);
                _ = tabR.Predict(features);
            });
        yield return WarmupRow(
            "TabTransformer",
            () => new TabTransformerRegression<double>(4, 1,
                new TabTransformerOptions<double> { CategoricalCardinalities = [5, 3] }),
            model => _ = ((TabTransformerRegression<double>)model).Predict(
                CreateNumericalFeatures(), CreateCategoricalIndices()));
        yield return WarmupRow("FTTransformer", () => new FTTransformerRegression<double>(4),
            model => _ = ((FTTransformerRegression<double>)model).Predict(CreateNumericalFeatures()));
    }

    private static object[] WarmupRow(
        string name,
        Func<IParameterSource<double>> factory,
        Action<IParameterSource<double>> warmup)
        => new object[] { name, factory, warmup };

    private static Tensor<double> CreateNumericalFeatures()
    {
        var features = new Tensor<double>([2, 4]);
        for (int i = 0; i < features.Length; i++) features[i] = (i + 1) * 0.1;
        return features;
    }

    private static Matrix<int> CreateCategoricalIndices()
    {
        var categories = new Matrix<int>(2, 2);
        categories[0, 0] = 1;
        categories[0, 1] = 2;
        categories[1, 0] = 4;
        categories[1, 1] = 0;
        return categories;
    }

    /// <summary>
    /// The count must equal the length of the vector — the contract that could not previously be
    /// stated for these models, because there was no vector.
    /// </summary>
    [Theory(Timeout = 120000)]
    [MemberData(nameof(Families))]
    public async Task CountEqualsVectorLength(string name, Func<IParameterSource<double>> factory)
    {
        await Task.Yield();
        var model = factory();

        Assert.Equal(model.ParameterCount, model.GetParameters().Length);
        Assert.True(model.ParameterCount > 0,
            $"{name} reports no parameters at all, which would make the round trip below vacuous.");
    }

    /// <summary>
    /// Restore must actually land: read, perturb, write, read back, value by value.
    /// </summary>
    /// <remarks>
    /// The strongest of the two, and the one that proves the interior participates. Every attention
    /// block, embedding table and tree tensor previously contributed to a count with no way to read
    /// or write it, so a round trip would have silently dropped all of them.
    /// </remarks>
    [Theory(Timeout = 120000)]
    [MemberData(nameof(Families))]
    public async Task RestoreRoundTripsEveryValue(string name, Func<IParameterSource<double>> factory)
    {
        await Task.Yield();
        var model = factory();

        var original = model.GetParameters();
        var perturbed = new Vector<double>(original.Length);
        for (int i = 0; i < original.Length; i++) perturbed[i] = original[i] + 0.25;

        model.SetParameters(perturbed);
        var readBack = model.GetParameters();

        Assert.Equal(perturbed.Length, readBack.Length);
        for (int i = 0; i < perturbed.Length; i++)
        {
            Assert.Equal(perturbed[i], readBack[i], precision: 10);
        }

        // The count must not have moved either. A surface that reports one size before a restore and
        // another after describes two different models, and the vector a caller saved fits neither.
        Assert.Equal(readBack.Length, model.ParameterCount);
    }

    /// <summary>
    /// A checkpoint produced by one instance must fit a separate instance that has never executed.
    /// Same-instance round trips cannot expose restore paths that depend on a prior forward or read.
    /// </summary>
    [Theory(Timeout = 120000)]
    [MemberData(nameof(Families))]
    public async Task FreshInstanceRestore_RoundTripsEveryValue(
        string name,
        Func<IParameterSource<double>> factory)
    {
        await Task.Yield();
        var source = factory();
        var target = factory();

        int length = source.GetParameters().Length;
        var checkpoint = new Vector<double>(length);
        for (int i = 0; i < length; i++) checkpoint[i] = (i % 97 - 48) / 100.0;

        target.SetParameters(checkpoint);
        var readBack = target.GetParameters();

        Assert.Equal(checkpoint.Length, readBack.Length);
        Assert.Equal(checkpoint.Length, target.ParameterCount);
        for (int i = 0; i < checkpoint.Length; i++)
        {
            Assert.Equal(checkpoint[i], readBack[i], precision: 10);
        }
    }

    /// <summary>
    /// The head must be INSIDE the agreement, not bolted onto the count.
    /// </summary>
    /// <remarks>
    /// Both variants share a backbone and differ only by their final projection, so the classifier
    /// (3 outputs) must expose at least as many parameters as the regressor (1 output) — and its
    /// count must still equal its own vector. If a head were counted but not emitted, the counts
    /// would differ while the vectors matched, which is exactly the state this replaced.
    /// </remarks>
    [Fact(Timeout = 120000)]
    public async Task ClassificationHead_IsInBothTheCountAndTheVector()
    {
        await Task.Yield();

        var classifiers = new IParameterSource<double>[]
        {
            new AutoIntClassifier<double>(4, 3),
            new GANDALFClassifier<double>(4, 3),
            new MambularClassifier<double>(4, 3),
            new NODEClassifier<double>(4, 3),
            new SAINTClassifier<double>(4, 3),
            new TabDPTClassifier<double>(4, 3),
            new TabPFNClassifier<double>(4, 3),
            new TabTransformerClassifier<double>(4, 3,
                new TabTransformerOptions<double> { CategoricalCardinalities = [5, 3] }),
        };

        foreach (var model in classifiers)
        {
            Assert.Equal(model.ParameterCount, model.GetParameters().Length);
        }
    }

    /// <summary>
    /// A larger output head must add actual restorable values, not only a count override (or nothing
    /// at all while the deferred head waits for a first forward).
    /// </summary>
    [Theory(Timeout = 120000)]
    [MemberData(nameof(VariantPairs))]
    public async Task ClassificationHead_StrictlyIncreasesTheParameterVector(
        string name,
        Func<IParameterSource<double>> regressionFactory,
        Func<IParameterSource<double>> classifierFactory)
    {
        await Task.Yield();
        var regression = regressionFactory();
        var classifier = classifierFactory();

        int regressionLength = regression.GetParameters().Length;
        int classifierLength = classifier.GetParameters().Length;

        Assert.True(classifierLength > regressionLength,
            $"{name}'s three-output classification head exposed {classifierLength} values, " +
            $"but its one-output regression sibling exposed {regressionLength}. The head is absent " +
            "from the pre-forward parameter surface if those lengths do not increase.");
    }

    /// <summary>
    /// When the constructor already knows every feature and hidden width, a first prediction must not
    /// reveal parameters that were missing from the checkpoint surface at construction time.
    /// </summary>
    [Theory(Timeout = 120000)]
    [MemberData(nameof(FamiliesWithWarmup))]
    public async Task KnownArchitecture_CountDoesNotGrowAfterFirstPrediction(
        string name,
        Func<IParameterSource<double>> factory,
        Action<IParameterSource<double>> warmup)
    {
        await Task.Yield();
        var model = factory();
        long before = model.ParameterCount;

        warmup(model);

        Assert.Equal(before, model.ParameterCount);
        Assert.Equal(before, model.GetParameters().Length);
    }

    [Fact]
    public void SAINTLayerNorms_ArePartOfTheParameterSurface()
    {
        var withoutNorm = new SAINTRegression<double>(4, 1, new SAINTOptions<double>
        {
            EmbeddingDimension = 8,
            NumHeads = 2,
            NumLayers = 1,
            UseIntersampleAttention = false,
            UseLayerNorm = false,
            MLPHiddenDimensions = [4]
        });
        var withNorm = new SAINTRegression<double>(4, 1, new SAINTOptions<double>
        {
            EmbeddingDimension = 8,
            NumHeads = 2,
            NumLayers = 1,
            UseIntersampleAttention = false,
            UseLayerNorm = true,
            MLPHiddenDimensions = [4]
        });

        // One block owns two layer norms; each owns gamma and beta of EmbeddingDimension values.
        const int expectedNormParameters = 2 * 2 * 8;
        Assert.Equal(withoutNorm.ParameterCount + expectedNormParameters, withNorm.ParameterCount);
        Assert.Equal(withNorm.ParameterCount, withNorm.GetParameters().Length);
    }
}
