using System;
using System.Collections.Generic;
using AiDotNet.ActivationFunctions;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks;
using Xunit;

namespace AiDotNet.Tests.UnitTests.NeuralNetworks;

/// <summary>
/// Cloning an architecture rebuilds its layers through their constructors, so the clone is INITIALIZED
/// rather than copied into. This pins what it initializes to.
/// </summary>
/// <remarks>
/// <para>
/// Every test here uses only the no-argument <c>CloneForModelConstruction()</c>, so the whole file compiles
/// and runs against the pre-fix commit — where it fails. The tests for the new
/// <c>CloneInitialization</c> parameter live in a separate file, because they cannot compile without it.
/// </para>
/// <para>
/// The architectures are seeded EXPLICITLY here rather than through an agent. Seeding is what makes the
/// defect appear, and relying on some other component to supply it would make these tests hostage to that
/// component's behaviour.
/// </para>
/// </remarks>
public class CloneInitializationBehaviourTests
{
    private static NeuralNetworkArchitecture<double> SeededArchitecture(int seed, int inputs = 4, int outputs = 2)
    {
        var architecture = new NeuralNetworkArchitecture<double>(
            inputType: InputType.OneDimensional,
            taskType: NeuralNetworkTaskType.Regression,
            inputSize: inputs,
            outputSize: outputs)
        {
            RandomSeed = seed,
        };

        architecture.Layers.AddRange(LayerHelper<double>.CreateDefaultLayers(
            architecture, hiddenLayerCount: 2, hiddenLayerSize: 8, outputSize: outputs));
        return architecture;
    }

    private static double MaxAbsoluteDifference(Vector<double> a, Vector<double> b)
    {
        if (a.Length != b.Length) return double.PositiveInfinity;
        double worst = 0.0;
        for (int i = 0; i < a.Length; i++) worst = Math.Max(worst, Math.Abs(a[i] - b[i]));
        return worst;
    }

    [Fact]
    public void A_seeded_clone_does_not_start_from_its_sources_weights()
    {
        // The defect: the clone copied the source's seed, so a SEEDED clone re-initialized from the same
        // seed and came out bit-identical. Unseeded clones were independent — the semantics flipped on a
        // setting unrelated to cloning, and flipped toward duplication exactly when a run was made
        // reproducible. Seeding PyTorch and constructing two modules gives DIFFERENT weights; this rewound.
        var architecture = SeededArchitecture(seed: 1234);
        using var source = new NeuralNetwork<double>(architecture);
        using var clone = new NeuralNetwork<double>(architecture.CloneForModelConstruction());

        double difference = MaxAbsoluteDifference(source.GetParameters(), clone.GetParameters());

        Assert.True(difference > 1e-12,
            $"the clone is bit-identical to its source (max |difference| = {difference:E3}); a seeded clone "
            + "is a duplicate rather than an independent draw.");
    }

    [Fact]
    public void Successive_clones_of_one_source_differ_from_each_other()
    {
        // Splitting to a single derived seed would move the defect one level down instead of fixing it:
        // the clones would differ from the source but be identical siblings. SAC takes THREE clones of one
        // critic architecture, so this is the ordinary case.
        var architecture = SeededArchitecture(seed: 4321);
        using var first = new NeuralNetwork<double>(architecture.CloneForModelConstruction());
        using var second = new NeuralNetwork<double>(architecture.CloneForModelConstruction());

        double difference = MaxAbsoluteDifference(first.GetParameters(), second.GetParameters());

        Assert.True(difference > 1e-12,
            $"two clones of one architecture are identical to each other (max |difference| = {difference:E3}).");
    }

    [Fact]
    public void Twin_critics_built_by_cloning_do_not_collapse_to_one_estimate()
    {
        // The failure this exists to prevent, in the shape it actually took: a twin critic whose two
        // networks are bit-identical makes min(Q1, Q2) equal Q1, so the pessimism that twin critics exist
        // to provide is silently absent. Nothing throws and the training curves look ordinary.
        var criticArchitecture = SeededArchitecture(seed: 99, inputs: 5, outputs: 1);

        using var q1 = new NeuralNetwork<double>(criticArchitecture);
        using var q2 = new NeuralNetwork<double>(criticArchitecture.CloneForModelConstruction());

        var input = new Tensor<double>(new[] { 1, 5 });
        for (int i = 0; i < 5; i++) input[0, i] = 0.2 * (i + 1);

        double v1 = q1.Predict(input).ToVector()[0];
        double v2 = q2.Predict(input).ToVector()[0];

        Assert.True(MaxAbsoluteDifference(q1.GetParameters(), q2.GetParameters()) > 1e-12,
            "the twin critics were initialised identically, so min(Q1, Q2) is just Q1.");
        Assert.True(Math.Abs(v1 - v2) > 1e-12,
            $"both critics return the same value ({v1:R}); they are not independent estimates.");
    }

    [Fact]
    public void An_unseeded_clone_is_still_independent()
    {
        // The half that already worked must keep working: with no seed there is nothing to pin, and each
        // rebuild draws from the shared generator. This guards against "fixing" the seeded path by making
        // the unseeded one deterministic.
        var architecture = new NeuralNetworkArchitecture<double>(
            inputType: InputType.OneDimensional,
            taskType: NeuralNetworkTaskType.Regression,
            inputSize: 4,
            outputSize: 2);
        architecture.Layers.AddRange(LayerHelper<double>.CreateDefaultLayers(
            architecture, hiddenLayerCount: 2, hiddenLayerSize: 8, outputSize: 2));

        using var source = new NeuralNetwork<double>(architecture);
        using var clone = new NeuralNetwork<double>(architecture.CloneForModelConstruction());

        Assert.True(MaxAbsoluteDifference(source.GetParameters(), clone.GetParameters()) > 1e-12,
            "an unseeded clone came out identical to its source.");
    }

    [Fact]
    public void A_clone_still_owns_its_layers_rather_than_sharing_them()
    {
        // Independence of VALUES must not cost independence of STORAGE — the property the architecture
        // sharing guard already protects. Training one model must still leave the other alone.
        var architecture = SeededArchitecture(seed: 7);
        using var online = new NeuralNetwork<double>(architecture);
        using var clone = new NeuralNetwork<double>(architecture.CloneForModelConstruction());

        var before = clone.GetParameters().Clone();
        var bumped = online.GetParameters();
        for (int i = 0; i < bumped.Length; i++) bumped[i] += 1.0;
        online.UpdateParameters(bumped);

        Assert.Equal(0.0, MaxAbsoluteDifference(before, clone.GetParameters()), 12);
    }
}
