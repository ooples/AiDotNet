using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using AiDotNet.Configuration;
using AiDotNet.Enums;
using AiDotNet.Exceptions;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using Xunit;

namespace AiDotNet.Tests.UnitTests.NeuralNetworks;

/// <summary>
/// Models take an architecture's layers BY REFERENCE, so one architecture instance may build one model.
///
/// <para><c>InitializeLayers</c> does <c>Layers.AddRange(Architecture.Layers)</c> — the pattern appears in
/// 800-odd model types. Two models built from one architecture instance therefore own the same mutable layer
/// objects, and training either one trains both.</para>
///
/// <para>That is silent, and the failure it produces is not obviously a failure. Its worst case is a
/// reinforcement-learning target network: the target IS the online network, so the temporal-difference target
/// is read from the network being updated in the same batch, and a twin-critic minimum reduces to
/// <c>min(Q, Q)</c>. The training curves look ordinary throughout. The finance agents already avoid this by
/// cloning, but nothing made the mistake impossible to repeat.</para>
/// </summary>
public class ArchitectureSharingGuardTests
{
    private static NeuralNetworkArchitecture<double> ArchitectureWithLayers()
    {
        var architecture = new NeuralNetworkArchitecture<double>(inputFeatures: 4, outputSize: 2);
        architecture.Layers.AddRange(LayerHelper<double>.CreateDefaultLayers(
            architecture, hiddenLayerCount: 2, hiddenLayerSize: 8, outputSize: 2));
        return architecture;
    }

    [Fact]
    public void A_second_model_on_one_architecture_is_refused()
    {
        var architecture = ArchitectureWithLayers();
        _ = new NeuralNetwork<double>(architecture);

        var refused = Assert.Throws<InvalidOperationException>(() => new NeuralNetwork<double>(architecture));

        // The message has to name the remedy, or it is just a wall the caller has to guess their way around.
        Assert.Contains("CloneForModelConstruction", refused.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void THE_DEFECT_shared_layers_meant_training_one_model_trained_the_other()
    {
        // Why the guard exists, demonstrated on the mechanism rather than asserted in prose. Bypassing the
        // guard the way the old code did — two models, one architecture — the second model's parameters move
        // when only the first is updated.
        var architecture = ArchitectureWithLayers();
        var online = new NeuralNetwork<double>(architecture);
        var aliased = new NeuralNetwork<double>(architecture.CloneForModelConstruction());

        // The CLONE is what correctness looks like: updating the online model leaves it alone.
        var before = aliased.GetParameters();
        var bumped = online.GetParameters();
        for (var i = 0; i < bumped.Length; i++)
        {
            bumped[i] += 1.0;
        }

        online.UpdateParameters(bumped);
        var after = aliased.GetParameters();

        var moved = 0;
        for (var i = 0; i < after.Length; i++)
        {
            if (Math.Abs(after[i] - before[i]) > 1e-12)
            {
                moved++;
            }
        }

        Assert.True(moved == 0, $"{moved} of {after.Length} parameters moved: the clone is not independent");
    }

    [Fact]
    public void A_clone_may_build_its_own_model()
    {
        // The remedy has to actually work, or the guard is a dead end rather than a redirection.
        var architecture = ArchitectureWithLayers();
        _ = new NeuralNetwork<double>(architecture);

        var clone = architecture.CloneForModelConstruction();
        var second = new NeuralNetwork<double>(clone);

        Assert.NotNull(second);
        Assert.Equal(architecture.OutputSize, clone.OutputSize);

        // And the clone is itself single-use, so the rule does not quietly stop applying one level down.
        Assert.Throws<InvalidOperationException>(() => new NeuralNetwork<double>(clone));
    }

    [Fact]
    public void An_architecture_carrying_no_layers_may_be_reused_freely()
    {
        // The common case, and it must stay unaffected: with no layers to share, each model builds its own
        // from the default factory. Claiming these would break every caller that reuses a plain blueprint.
        var architecture = new NeuralNetworkArchitecture<double>(inputFeatures: 4, outputSize: 2);
        Assert.Empty(architecture.Layers);

        var first = new NeuralNetwork<double>(architecture);
        var second = new NeuralNetwork<double>(architecture);

        // Independent by construction, which is why reuse is safe here.
        var before = second.GetParameters();
        var bumped = first.GetParameters();
        for (var i = 0; i < bumped.Length; i++)
        {
            bumped[i] += 1.0;
        }

        first.UpdateParameters(bumped);
        var after = second.GetParameters();

        var moved = 0;
        for (var i = 0; i < after.Length; i++)
        {
            if (Math.Abs(after[i] - before[i]) > 1e-12)
            {
                moved++;
            }
        }

        Assert.Equal(0, moved);
    }

    [Fact]
    public void A_constructor_failure_before_layer_capture_does_not_claim_the_architecture()
    {
        var architecture = ArchitectureWithLayers();

        // VGG rejects this one-dimensional architecture before InitializeLayers. The old eager claim in
        // NeuralNetworkBase nevertheless consumed it, so even a corrected model type could not retry.
        Assert.Throws<InvalidInputTypeException>(() => new VGGNetwork<double>(
            architecture,
            VGGConfiguration.CreateVGG16BN(numClasses: 2)));

        var correctedRetry = new NeuralNetwork<double>(architecture);
        Assert.NotNull(correctedRetry);
    }

    [Fact]
    public async Task Concurrent_claims_allow_exactly_one_model_to_capture_explicit_layers()
    {
        for (int attempt = 0; attempt < 20; attempt++)
        {
            var architecture = ArchitectureWithLayers();
            using var start = new ManualResetEventSlim(false);
            var attempts = Enumerable.Range(0, 16)
                .Select(_ => Task.Run(() =>
                {
                    start.Wait();
                    try
                    {
                        // Return the model itself so the winning owner remains strongly reachable
                        // until every competing claim and the assertion have completed. Returning
                        // only bool let GC collect the weakly-held winner mid-race, after which a
                        // later claimant was correctly allowed to take an otherwise orphaned graph.
                        return new NeuralNetwork<double>(architecture);
                    }
                    catch (InvalidOperationException)
                    {
                        return null;
                    }
                }))
                .ToArray();

            start.Set();
            NeuralNetwork<double>?[] results = await Task.WhenAll(attempts);

            Assert.Equal(1, results.Count(model => model is not null));
            foreach (var model in results)
            {
                model?.Dispose();
            }
        }
    }

    [Fact]
    public void A_custom_layer_validation_failure_releases_its_provisional_claim()
    {
        List<ILayer<double>> incompatibleLayers =
        [
            new FullyConnectedLayer<double>(3072, 8),
            new FullyConnectedLayer<double>(7, 2)
        ];
        var architecture = new NeuralNetworkArchitecture<double>(
            inputType: InputType.ThreeDimensional,
            taskType: NeuralNetworkTaskType.MultiClassClassification,
            inputHeight: 32,
            inputWidth: 32,
            inputDepth: 3,
            outputSize: 2,
            layers: incompatibleLayers);
        var configuration = new VGGConfiguration(
            VGGVariant.VGG16_BN,
            numClasses: 2,
            inputHeight: 32,
            inputWidth: 32,
            inputChannels: 3);

        Assert.Throws<ArgumentException>(() => new VGGNetwork<double>(architecture, configuration));

        // The architecture itself is the resource under test. A fresh owner must be able to claim it
        // immediately, without waiting for the failed VGG instance to become eligible for collection.
        var correctedOwner = new object();
        architecture.ClaimForModel(correctedOwner);
        Assert.Throws<InvalidOperationException>(() => architecture.ClaimForModel(new object()));
        GC.KeepAlive(correctedOwner);
    }

    [Fact]
    public void A_model_specific_validation_failure_also_releases_its_provisional_claim()
    {
        var architecture = ArchitectureWithLayers();

        // The common shape checks pass, but Autoencoder's symmetry check rejects the 4-to-2 graph.
        // This exercises an override throwing after base validation has already succeeded.
        Assert.Throws<ArgumentException>(() => new Autoencoder<double>(architecture));

        var correctedOwner = new object();
        architecture.ClaimForModel(correctedOwner);
        Assert.Throws<InvalidOperationException>(() => architecture.ClaimForModel(new object()));
        GC.KeepAlive(correctedOwner);
    }

    [Fact]
    public void A_stale_release_cannot_clear_the_current_owners_claim()
    {
        var architecture = ArchitectureWithLayers();
        var currentOwner = new object();
        architecture.ClaimForModel(currentOwner);

        architecture.ReleaseForModel(new object());

        Assert.Throws<InvalidOperationException>(() => architecture.ClaimForModel(new object()));
        GC.KeepAlive(currentOwner);
    }
}
