using System;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.NeuralNetworks;
using Xunit;

namespace AiDotNet.Tests.UnitTests.NeuralNetworks;

/// <summary>
/// The two regimes of <see cref="CloneInitialization"/>, and the property that makes the independent one
/// worth having: it is independent AND reproducible AND order-independent at the same time.
/// </summary>
/// <remarks>
/// These cannot run against the pre-fix commit — the parameter does not exist there — so the fails-before
/// evidence for the defect itself lives in <c>CloneInitializationBehaviourTests</c>, which compiles on both.
/// </remarks>
public class CloneInitializationModeTests
{
    internal static NeuralNetworkArchitecture<double> SeededArchitecture(int seed, int outputs = 2)
    {
        var architecture = new NeuralNetworkArchitecture<double>(
            inputType: InputType.OneDimensional,
            taskType: NeuralNetworkTaskType.Regression,
            inputSize: 4,
            outputSize: outputs)
        {
            RandomSeed = seed,
        };

        architecture.Layers.AddRange(LayerHelper<double>.CreateDefaultLayers(
            architecture, hiddenLayerCount: 2, hiddenLayerSize: 8, outputSize: outputs));
        return architecture;
    }

    internal static double MaxAbsoluteDifference(Vector<double> a, Vector<double> b)
    {
        if (a.Length != b.Length) return double.PositiveInfinity;
        double worst = 0.0;
        for (int i = 0; i < a.Length; i++) worst = Math.Max(worst, Math.Abs(a[i] - b[i]));
        return worst;
    }

    [Fact]
    public void Identical_reproduces_the_sources_initialization()
    {
        var architecture = SeededArchitecture(seed: 2024);
        using var source = new NeuralNetwork<double>(architecture);
        using var twin = new NeuralNetwork<double>(
            architecture.CloneForModelConstruction(CloneInitialization.Identical));

        double difference = MaxAbsoluteDifference(source.GetParameters(), twin.GetParameters());

        Assert.True(difference < 1e-12,
            $"Identical did not reproduce the source's weights (max |difference| = {difference:E3}).");
    }

    [Fact]
    public void Independent_produces_different_weights_from_the_source()
    {
        var architecture = SeededArchitecture(seed: 2024);
        using var source = new NeuralNetwork<double>(architecture);
        using var other = new NeuralNetwork<double>(
            architecture.CloneForModelConstruction(CloneInitialization.Independent));

        Assert.True(MaxAbsoluteDifference(source.GetParameters(), other.GetParameters()) > 1e-12,
            "Independent produced the source's weights.");
    }

    [Fact]
    public void The_two_regimes_disagree_with_each_other()
    {
        // If both modes produced the same thing the parameter would be decorative.
        var architecture = SeededArchitecture(seed: 2024);
        using var twin = new NeuralNetwork<double>(
            architecture.CloneForModelConstruction(CloneInitialization.Identical));
        using var other = new NeuralNetwork<double>(
            architecture.CloneForModelConstruction(CloneInitialization.Independent));

        Assert.True(MaxAbsoluteDifference(twin.GetParameters(), other.GetParameters()) > 1e-12,
            "Identical and Independent produced the same weights.");
    }

    [Fact]
    public void Independent_is_reproducible_run_to_run()
    {
        // THE property that exceeds the frameworks surveyed. Keras gives independence with no seed
        // discipline, so clones are not reproducible; PyTorch gives independence keyed to a global stream,
        // so weights depend on construction ORDER. Splitting deterministically gives all three at once:
        // the clone differs from its source, repeats exactly across runs, and does not depend on how many
        // unrelated models were built first.
        using var firstRun = new NeuralNetwork<double>(
            SeededArchitecture(seed: 555).CloneForModelConstruction());

        // A whole unrelated model built in between: under a global-stream scheme this would shift the draw.
        using var interference = new NeuralNetwork<double>(SeededArchitecture(seed: 999, outputs: 3));

        using var secondRun = new NeuralNetwork<double>(
            SeededArchitecture(seed: 555).CloneForModelConstruction());

        double difference = MaxAbsoluteDifference(firstRun.GetParameters(), secondRun.GetParameters());

        Assert.True(difference < 1e-12,
            $"the independent clone is not reproducible (max |difference| = {difference:E3}); identical "
            + "inputs produced different weights, so a seeded run cannot be repeated.");
    }

    [Fact]
    public void Successive_independent_clones_split_to_different_seeds()
    {
        var architecture = SeededArchitecture(seed: 31337);
        using var a = new NeuralNetwork<double>(architecture.CloneForModelConstruction());
        using var b = new NeuralNetwork<double>(architecture.CloneForModelConstruction());
        using var c = new NeuralNetwork<double>(architecture.CloneForModelConstruction());

        Assert.True(MaxAbsoluteDifference(a.GetParameters(), b.GetParameters()) > 1e-12, "clones 1 and 2 match");
        Assert.True(MaxAbsoluteDifference(b.GetParameters(), c.GetParameters()) > 1e-12, "clones 2 and 3 match");
        Assert.True(MaxAbsoluteDifference(a.GetParameters(), c.GetParameters()) > 1e-12, "clones 1 and 3 match");
    }
}

/// <summary>
/// Isolates the tests that mutate the process-wide seed override, so they cannot leak into tests running
/// in parallel.
/// </summary>
[CollectionDefinition("GlobalSeedOverride", DisableParallelization = true)]
public class GlobalSeedOverrideCollection
{
}

/// <summary>
/// The seed does not have to live on the architecture: <c>RandomSeed</c> falls back to a process-wide
/// override, and a clone is just as duplicated when determinism arrives that way.
/// </summary>
[Collection("GlobalSeedOverride")]
public class CloneInitializationGlobalOverrideTests
{
    [Fact]
    public void An_independent_clone_splits_even_when_the_seed_comes_from_the_global_override()
    {
        // Reading the _randomSeed FIELD instead of the RandomSeed PROPERTY would leave the defect intact
        // here — and this is the configuration a deterministic test harness is most likely to create.
        int? previous = NeuralNetworkArchitecture<double>.DefaultRandomSeedOverride;
        try
        {
            NeuralNetworkArchitecture<double>.DefaultRandomSeedOverride = 8675309;

            var architecture = new NeuralNetworkArchitecture<double>(
                inputType: InputType.OneDimensional,
                taskType: NeuralNetworkTaskType.Regression,
                inputSize: 4,
                outputSize: 2);
            architecture.Layers.AddRange(LayerHelper<double>.CreateDefaultLayers(
                architecture, hiddenLayerCount: 2, hiddenLayerSize: 8, outputSize: 2));

            using var source = new NeuralNetwork<double>(architecture);
            using var clone = new NeuralNetwork<double>(architecture.CloneForModelConstruction());

            double difference = CloneInitializationModeTests.MaxAbsoluteDifference(
                source.GetParameters(), clone.GetParameters());

            Assert.True(difference > 1e-12,
                $"the clone is identical to its source (max |difference| = {difference:E3}) when the seed "
                + "comes from DefaultRandomSeedOverride; the split is reading the field, not the effective seed.");
        }
        finally
        {
            NeuralNetworkArchitecture<double>.DefaultRandomSeedOverride = previous;
        }
    }
}
