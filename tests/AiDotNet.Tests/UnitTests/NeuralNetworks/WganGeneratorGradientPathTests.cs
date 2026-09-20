using System;
using System.Collections.Generic;
using System.Linq;
using AiDotNet.Enums;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.NeuralNetworks;

/// <summary>
/// Reachability regression for WGAN's generator update.
/// </summary>
/// <remarks>
/// <para>
/// WGAN's generator minimises -mean(Critic(G(z))). Reading the critic through <c>Predict</c> runs it
/// inside a <c>NoGradScope</c>, so the score arrives tape-detached, the loss is a constant, and the
/// generator never learns to reduce the Wasserstein distance at all.
/// </para>
/// <para>
/// This defect was found by the AIDN101 analyzer rather than by any test: no WGAN fixture existed,
/// and the model family it derives from (ImageGeneratorModelLayoutBase) does not inherit the GAN
/// base's per-component checks. Even if it had, a liveness check would not have caught it -- the
/// measured behaviour on the sibling ConditionalGAN defect is that "did this component change"
/// passes while the weights receive no gradient, because registered non-gradient state still moves
/// the parameter vector.
/// </para>
/// <para>
/// The assertion is that the generator's own parameters are reachable from its loss. If the critic
/// score is detached, the loss is constant with respect to the generator and NOTHING is reachable.
/// </para>
/// </remarks>
public class WganGeneratorGradientPathTests
{
    private const int NoiseSize = 16;
    private const int ImageSize = 32;

    private static List<Tensor<float>> TrainableTensors(NeuralNetworkBase<float> network)
    {
        var tensors = new List<Tensor<float>>();
        foreach (var chunk in network.GetParameterStateChunks())
        {
            if (chunk.Tensor is not null && chunk.Tensor.Length > 0)
            {
                tensors.Add(chunk.Tensor);
            }
        }

        return tensors;
    }

    /// <summary>
    /// Reference-identity membership. The probe keys on tensor identity, and Tensor&lt;T&gt; may define
    /// value equality, so a default-comparer lookup could silently match the wrong instance.
    /// </summary>
    private static int CountReached(
        IReadOnlyCollection<Tensor<float>> reached, IReadOnlyList<Tensor<float>> wanted)
        => wanted.Count(w => reached.Any(r => ReferenceEquals(r, w)));

    private static Tensor<float> RandomTensor(int[] shape, Random rng)
    {
        var tensor = new Tensor<float>(shape);
        for (int i = 0; i < tensor.Length; i++)
        {
            tensor.SetFlat(i, (float)(rng.NextDouble() * 2.0 - 1.0));
        }

        return tensor;
    }

    [Fact]
    public void Generator_parameters_must_be_reachable_from_the_wasserstein_loss()
    {
        var rng = RandomHelper.CreateSeededRandom(23);

        // The ctor requires generator output size == critic input size, and critic output size == 1
        // (the Wasserstein score is a scalar, not a probability).
        var generatorArchitecture = new NeuralNetworkArchitecture<float>(
            InputType.OneDimensional,
            NeuralNetworkTaskType.Generative,
            NetworkComplexity.Simple,
            inputSize: NoiseSize,
            outputSize: ImageSize);

        var criticArchitecture = new NeuralNetworkArchitecture<float>(
            InputType.OneDimensional,
            NeuralNetworkTaskType.BinaryClassification,
            NetworkComplexity.Simple,
            inputSize: ImageSize,
            outputSize: 1);

        // criticIterations: 1 keeps the step cheap; the default 5 only adds critic passes, which are
        // filtered out below anyway.
        using var wgan = new WGAN<float>(
            generatorArchitecture,
            criticArchitecture,
            InputType.OneDimensional,
            criticIterations: 1);

        var generatorTensors = TrainableTensors(wgan.Generator);
        var criticTensors = TrainableTensors(wgan.Critic);

        Assert.True(generatorTensors.Count > 0, "The generator exposed no parameter tensors to probe.");

        var realImages = RandomTensor([1, ImageSize], rng);
        var noise = RandomTensor([1, NoiseSize], rng);

        var probed = generatorTensors.Concat(criticTensors).ToList();
        using var probe = TapeReachabilityProbe<float>.Arm(probed);

        wgan.TrainStep(realImages, noise);

        // The critic trains BEFORE the generator and legitimately reaches its own parameters in its
        // own backward pass. Inspect only the generator-owned pass, or that unrelated update would
        // mask a severed generator objective.
        var generatorPasses = probe.Observations
            .Where(o => ReferenceEquals(o.Owner, wgan.Generator))
            .ToList();

        Assert.True(
            generatorPasses.Count > 0,
            "The generator's training step never ran, so nothing was measured and this test proves "
            + "nothing about reachability.");

        var generatorPass = generatorPasses[generatorPasses.Count - 1];

        int generatorReached = CountReached(generatorPass.Reached, generatorTensors);
        int criticReached = CountReached(generatorPass.Reached, criticTensors);

        Assert.True(
            generatorReached > 0,
            $"None of the generator's {generatorTensors.Count} parameter tensors were reachable from "
            + "-mean(Critic(G(z))), so the generator receives no gradient and cannot reduce the "
            + "Wasserstein distance. Read the critic with ForwardForTraining rather than Predict, "
            + "which runs inside a NoGradScope. "
            + $"(For context, {criticReached} of {criticTensors.Count} critic tensors were reachable; "
            + "that number is not asserted on, because the generator's gradient needs the critic's "
            + "weights as values rather than as tape sources.)");
    }
}
