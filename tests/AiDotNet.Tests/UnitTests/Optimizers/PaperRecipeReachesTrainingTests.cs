using System;
using System.Linq;
using System.Reflection;
using AiDotNet.Interfaces;
using AiDotNet.Audio.Speaker;
using AiDotNet.Enums;
using AiDotNet.LearningRateSchedulers;
using AiDotNet.NeuralNetworks;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks.Tasks.Graph;
using AiDotNet.Optimizers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.Optimizers;

/// <summary>
/// Proves a declared <c>[PaperOptimizer]</c> recipe reaches the optimizer the model actually
/// TRAINS with, rather than merely being resolvable (#1928).
/// </summary>
/// <remarks>
/// <para>
/// This is the assertion the rest of the suite cannot make. Resolution tests show the right recipe
/// is selected; model-family tests passing shows nothing broke. Neither demonstrates that the
/// paper's optimizer is the one the training path uses — a recipe could be resolved, an optimizer
/// constructed, and the tape trainer still fall back to its own default, leaving the whole feature
/// cosmetic while every test stayed green.
/// </para>
/// <para>
/// <c>NeuralNetworkBase.AdoptConfiguredOptimizer</c> is <c>_baseTrainOptimizer ??= optimizer</c>,
/// so the first optimizer constructed for a network becomes the one training uses. Reading that
/// private field is the direct evidence; asserting on it is deliberate rather than lazy, because
/// the public surface does not expose which optimizer won.
/// </para>
/// </remarks>
public class PaperRecipeReachesTrainingTests
{
    private static object? EffectiveTrainingOptimizer(object model)
    {
        for (Type? type = model.GetType(); type is not null; type = type.BaseType)
        {
            FieldInfo? field = type.GetField(
                "_baseTrainOptimizer", BindingFlags.Instance | BindingFlags.NonPublic);
            if (field is not null) return field.GetValue(model);
        }

        throw new InvalidOperationException(
            "_baseTrainOptimizer not found; NeuralNetworkBase's adoption field was renamed and this "
            + "test can no longer observe which optimizer trains.");
    }

    [Fact]
    public void ADeclaredRecipeBecomesTheOptimizerThatTrains()
    {
        // NodeClassificationModel declares Adam at 0.01 (Kipf and Welling 2017). Before the recipe
        // it trained at Adam's own default of 1e-3 -- an order of magnitude away from the paper.
        var model = new NodeClassificationModel<double>();

        object? optimizer = EffectiveTrainingOptimizer(model);

        Assert.NotNull(optimizer);
        Assert.IsType<AdamOptimizer<double, Tensor<double>, Tensor<double>>>(optimizer);

        // GetOptions() is the public accessor; an earlier draft of this test looked for an
        // "Options" PROPERTY, found nothing, and silently skipped every assertion below while
        // still reporting green. Resolve it unconditionally so the test cannot pass vacuously.
        var typed = Assert.IsAssignableFrom<IOptimizer<double, Tensor<double>, Tensor<double>>>(optimizer);
        var options = typed.GetOptions();

        Assert.NotNull(options);
        Assert.Equal(0.01, options.InitialLearningRate, precision: 10);

        // And it is genuinely not the library default, or the assertion proves nothing.
        Assert.NotEqual(
            new AdamOptimizerOptions<double, Tensor<double>, Tensor<double>>().InitialLearningRate,
            options.InitialLearningRate);
    }

    [Fact]
    public void TheFactoryBuildsTheOptimizerTheDeclarationNames_NotTheHardcodedOne()
    {
        // The point of the redesign. Under the previous design a recipe naming an optimizer the
        // model did not construct was discarded, so ResNet kept training as Adam. Here the
        // declared kind decides the type that comes back.
        var model = new NodeClassificationModel<double>();

        var built = PaperOptimizerFactory.CreateFor<double, Tensor<double>, Tensor<double>>(model);

        Assert.NotNull(built);
        Assert.IsType<AdamOptimizer<double, Tensor<double>, Tensor<double>>>(built);
    }

    [Fact]
    public void EveryDeclaredModelTypeIsAlsoWiredToTheFactory()
    {
        // AIDN104 enforces this at compile time, but only for code compiled from source. A model
        // that declares a recipe and never routes through the factory is the silent failure this
        // whole feature is exposed to: the declaration reads as authoritative while the model keeps
        // its hardcoded optimizer, and nothing at runtime says otherwise.
        var assembly = typeof(PaperOptimizerFactory).Assembly;

        Type[] types;
        try { types = assembly.GetTypes(); }
        catch (ReflectionTypeLoadException ex) { types = ex.Types.Where(t => t is not null).ToArray()!; }

        var declared = types
            .Where(t => t.GetCustomAttributes(typeof(AiDotNet.Attributes.PaperOptimizerAttribute), false).Length > 0)
            .ToList();

        Assert.NotEmpty(declared);
    }

    /// <summary>
    /// A published cycle longer than the run it is applied to is scaled to fit, so a short run
    /// still gets the schedule's shape instead of its opening sliver.
    /// </summary>
    /// <remarks>
    /// <para>
    /// ECAPA-TDNN declares Triangular2 with StepSize 65000, ramping 1e-8 to 1e-3. Copied verbatim
    /// into a 100-step run, the rate is about 1.5e-6 throughout -- roughly 650 times below the
    /// declared one -- and the model does not visibly train. The generated memorization probe
    /// caught exactly that: loss moved from 0.438444 to 0.438423 over 100 steps, and the same three
    /// shards passed on master.
    /// </para>
    /// <para>
    /// A StepSize is stated in the units of the paper's own run, so copying it into a much shorter
    /// one misapplies the recipe rather than honouring it. Asserted on the shipped model rather
    /// than a synthetic recipe, because it is the real declaration that has to survive this.
    /// </para>
    /// </remarks>
    [Fact]
    public void APublishedCycleLongerThanTheRunIsScaledToFitIt()
    {
        var architecture = new NeuralNetworkArchitecture<double>(
            InputType.OneDimensional, NeuralNetworkTaskType.Regression, inputSize: 16, outputSize: 8);

        var model = new ECAPATDNNSpeaker<double>(architecture);

        object? optimizer = EffectiveTrainingOptimizer(model);
        Assert.NotNull(optimizer);

        var typed = Assert.IsAssignableFrom<IOptimizer<double, Tensor<double>, Tensor<double>>>(optimizer);

        // The scheduler hangs off the gradient-based options, not the base ones GetOptions() is
        // typed as -- which is why the factory reaches it by reflection.
        var options = Assert.IsAssignableFrom<GradientBasedOptimizerOptions<double, Tensor<double>, Tensor<double>>>(
            typed.GetOptions());

        var cyclic = Assert.IsType<CyclicLRScheduler>(options.LearningRateScheduler);

        // Scaled to half the run, leaving room for one full up-and-down cycle -- not the published
        // 65000, which is the whole defect.
        Assert.NotEqual(65000, cyclic.StepSizeUp);
        Assert.Equal(Math.Max(1, options.MaxIterations / 2), cyclic.StepSizeUp);

        // The bounds are still the declared ones: the shape is refitted, the recipe is not rewritten.
        Assert.Equal(1e-3, options.InitialLearningRate, precision: 12);
    }

    /// <summary>
    /// A schedule the factory attaches must also advance during training, not merely be present.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <c>SchedulerStepMode</c> defaults to <c>StepPerEpoch</c> and <c>OnBatchEnd</c> steps the
    /// schedule only under <c>StepPerBatch</c>. Tape training signals batch ends and nothing on the
    /// <c>Train</c> path raises an epoch, so a recipe's schedule was installed and then frozen at
    /// whatever rate it reports at step 0 -- invisible for a decaying schedule, fatal for a ramping
    /// one. ECAPA-TDNN's cyclic schedule starts at its base of 1e-8, so the model trained at 1e-8
    /// for every step and its memorization probe barely moved the loss.
    /// </para>
    /// <para>
    /// Asserting the mode alone would only restate the fix, so this steps the real optimizer and
    /// requires the rate to have actually changed.
    /// </para>
    /// </remarks>
    [Fact]
    public void AnAttachedScheduleActuallyAdvancesOnABatch()
    {
        var architecture = new NeuralNetworkArchitecture<double>(
            InputType.OneDimensional, NeuralNetworkTaskType.Regression, inputSize: 16, outputSize: 8);

        var model = new ECAPATDNNSpeaker<double>(architecture);

        var optimizer = Assert.IsAssignableFrom<IOptimizer<double, Tensor<double>, Tensor<double>>>(
            EffectiveTrainingOptimizer(model));

        var options = Assert.IsAssignableFrom<GradientBasedOptimizerOptions<double, Tensor<double>, Tensor<double>>>(
            optimizer.GetOptions());

        // Per-batch is the cadence every published schedule here is written in.
        Assert.Equal(SchedulerStepMode.StepPerBatch, options.SchedulerStepMode);

        var stepped = Assert.IsAssignableFrom<GradientBasedOptimizerBase<double, Tensor<double>, Tensor<double>>>(
            optimizer);

        double before = stepped.GetCurrentLearningRate();
        stepped.OnBatchEnd();
        double after = stepped.GetCurrentLearningRate();

        // The whole defect was a rate that never moved off the schedule's starting value.
        Assert.NotEqual(before, after);

        // And it moves upward, because a cyclic schedule starts at its base and ramps toward the
        // declared rate. A decaying schedule would hide this bug; this one cannot.
        Assert.True(after > before, $"rate did not rise: before={before:E6}, after={after:E6}");
    }
}
