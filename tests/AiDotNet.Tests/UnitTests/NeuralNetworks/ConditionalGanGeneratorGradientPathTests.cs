using System;
using System.Collections.Generic;
using System.Linq;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.NeuralNetworks;

/// <summary>
/// Reachability regression for ConditionalGAN's generator update.
/// </summary>
/// <remarks>
/// <para>
/// The generator's objective is BCE(D(G(z) | c), real). Two separate defects each severed it, and
/// either one alone left the generator training on nothing:
/// </para>
/// <para>
/// 1. <c>ConcatenateFlattenedImageAndCondition</c> rented a fresh tensor and filled it element by
/// element, which detaches the result from the tape -- so the generator's own output was already a
/// constant before the discriminator was ever consulted.
/// 2. The discriminator was read through <c>Predict</c>, which runs inside a <c>NoGradScope</c>.
/// </para>
/// <para>
/// This asserts the property that is violated in both cases and satisfied only when the whole chain
/// is on the tape: the generator's OWN parameters must be reachable from its adversarial loss. If any
/// link is severed the loss is constant with respect to the generator and NOTHING is reachable. On the
/// unfixed code this test reports 0 of 6; with the chain intact it reports 6 of 6.
/// </para>
/// <para>
/// It deliberately does NOT require the DISCRIMINATOR's parameters to be reachable. The generator's
/// gradient is dL/dG = dL/dD_out * dD_out/dG, which needs the discriminator's weights as VALUES, not
/// as tape sources -- so a correct implementation can leave D's weights out of the graph entirely
/// while the generator still receives an exactly correct adversarial gradient. Asserting on them
/// would fail on correct code. The count is reported as context only.
/// </para>
/// <para>
/// Note that a per-component liveness check cannot stand in for this: it passes on the unfixed code,
/// because registered non-gradient state still moves the parameter vector while the weights receive
/// no gradient at all.
/// </para>
/// </remarks>
public class ConditionalGanGeneratorGradientPathTests
{
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
    public void Generator_parameters_must_be_reachable_from_the_adversarial_loss()
    {
        var rng = RandomHelper.CreateSeededRandom(17);
        using var gan = new ConditionalGAN<float>();

        var generatorTensors = TrainableTensors(gan.Generator);
        var discriminatorTensors = TrainableTensors(gan.Discriminator);

        Assert.True(generatorTensors.Count > 0, "The generator exposed no parameter tensors to probe.");

        // Shapes follow the ConditionalGANTests fixture: noise-only input, conditions added internally.
        var noise = RandomTensor([1, 100], rng);
        var target = RandomTensor([1, 784], rng);

        var probed = generatorTensors.Concat(discriminatorTensors).ToList();
        using var probe = TapeReachabilityProbe<float>.Arm(probed);

        gan.Train(noise, target);

        // Train() updates the discriminator BEFORE the generator, and the discriminator's own step
        // legitimately reaches its parameters. Inspect ONLY the generator-owned backward pass, or that
        // unrelated update would mask a severed generator objective.
        var generatorPasses = probe.Observations
            .Where(o => ReferenceEquals(o.Owner, gan.Generator))
            .ToList();

        Assert.True(
            generatorPasses.Count > 0,
            "The generator's training step never ran, so nothing was measured and this test proves "
            + "nothing about reachability.");

        var generatorPass = generatorPasses[generatorPasses.Count - 1];

        int generatorReached = CountReached(generatorPass.Reached, generatorTensors);
        int discriminatorReached = CountReached(generatorPass.Reached, discriminatorTensors);

        Assert.True(
            generatorReached > 0,
            $"None of the generator's {generatorTensors.Count} parameter tensors were reachable from "
            + "its own adversarial loss, so the generator receives no gradient and is not learning to "
            + "fool the discriminator. Some link in BCE(D(G(z)|c), real) is off the tape: check that "
            + "the image/condition concatenation uses engine ops rather than filling a fresh tensor "
            + "element by element, and that the discriminator is read with ForwardForTraining rather "
            + "than Predict, which runs inside a NoGradScope. "
            + $"(For context, {discriminatorReached} of {discriminatorTensors.Count} discriminator "
            + "tensors were reachable; that number is expected to be 0 and is not asserted on.)");
    }
}
