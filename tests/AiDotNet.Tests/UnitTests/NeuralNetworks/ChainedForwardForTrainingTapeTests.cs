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
/// Framework-level check: does a CHAIN of cross-network <c>ForwardForTraining</c> calls stay on the
/// tape, so the trained network still receives gradient through them?
/// </summary>
/// <remarks>
/// <para>
/// SAC forwards ONE other network inside its actor's loss (the critic) and its actor is demonstrably
/// reachable. Dreamer forwards TWO in sequence -- dynamics, then value, with the first's output as the
/// second's input -- and nothing in that chain is reachable at all: not the actor, not the dynamics
/// head, not even the value head's own output bias. A bias always receives gradient from a loss that
/// is not constant, so that pattern says the tape recorded nothing rather than that any head is dead.
/// </para>
/// <para>
/// This reproduces exactly that shape at minimal scale and with no agent involved, to separate a
/// framework limitation from a defect in any one model. If the chained case fails here while the
/// single case passes, the problem is the hand-off between two ForwardForTraining calls and affects
/// every model that composes a loss that way.
/// </para>
/// </remarks>
public class ChainedForwardForTrainingTapeTests
{
    private const int InputDim = 4;
    private const int TrainedOutDim = 2;
    private const int MiddleOutDim = 3;

    private static NeuralNetwork<double> Net(int inputSize, int outputSize, ActivationFunction tail)
    {
        var architecture = new NeuralNetworkArchitecture<double>(
            inputType: InputType.OneDimensional,
            taskType: NeuralNetworkTaskType.Regression,
            complexity: NetworkComplexity.Simple,
            inputSize: inputSize,
            outputSize: outputSize);

        var network = new NeuralNetwork<double>(architecture);
        network.AddLayer(LayerType.Dense, 8, ActivationFunction.ReLU);
        network.AddLayer(LayerType.Dense, outputSize, tail);
        return network;
    }

    private static List<Tensor<double>> TrainableTensors(NeuralNetworkBase<double> network)
    {
        var tensors = new List<Tensor<double>>();
        foreach (var chunk in network.GetParameterStateChunks())
        {
            if (chunk.Tensor is not null && chunk.Tensor.Length > 0) tensors.Add(chunk.Tensor);
        }

        return tensors;
    }

    private static int CountReached(
        IReadOnlyCollection<Tensor<double>> reached, IReadOnlyList<Tensor<double>> wanted)
        => wanted.Count(w => reached.Any(r => ReferenceEquals(r, w)));

    private static Tensor<double> RandomInput(Random rng)
    {
        var t = new Tensor<double>([1, InputDim]);
        for (int i = 0; i < t.Length; i++) t.SetFlat(i, rng.NextDouble() * 2.0 - 1.0);
        return t;
    }

    [Fact]
    public void One_cross_network_forward_keeps_the_trained_network_reachable()
    {
        var rng = RandomHelper.CreateSeededRandom(31);
        using var trained = Net(InputDim, TrainedOutDim, ActivationFunction.Tanh);
        using var other = Net(InputDim + TrainedOutDim, 1, ActivationFunction.Linear);

        var trainedTensors = TrainableTensors(trained);
        var otherTensors = TrainableTensors(other);
        var input = RandomInput(rng);

        using var probe = TapeReachabilityProbe<double>.Arm(trainedTensors.Concat(otherTensors).ToList());

        var engine = AiDotNet.Tensors.Engines.AiDotNetEngine.Current;
        trained.TrainWithCustomLoss(input, output =>
        {
            var joined = engine.TensorConcatenate([input, output], axis: 1);
            var scored = other.ForwardForTraining(joined);
            var flat = engine.ReduceSum(scored, new[] { 1 }, keepDims: false);
            return engine.ReduceMean(flat, new[] { 0 }, keepDims: false);
        });

        var pass = probe.Observations.LastOrDefault(o => ReferenceEquals(o.Owner, trained));
        Assert.True(pass is not null, "The trained network ran no backward pass.");

        Assert.True(
            CountReached(pass!.Reached, trainedTensors) > 0,
            $"SINGLE cross-network forward: none of the trained network's {trainedTensors.Count} "
            + "tensors were reachable. This is the shape SAC uses successfully, so a failure here "
            + "means the probe or the engine op is at fault rather than any chaining.");
    }

    [Fact]
    public void Two_chained_cross_network_forwards_keep_the_trained_network_reachable()
    {
        var rng = RandomHelper.CreateSeededRandom(31);
        using var trained = Net(InputDim, TrainedOutDim, ActivationFunction.Tanh);
        using var middle = Net(InputDim + TrainedOutDim, MiddleOutDim, ActivationFunction.Linear);
        using var head = Net(MiddleOutDim, 1, ActivationFunction.Linear);

        var trainedTensors = TrainableTensors(trained);
        var middleTensors = TrainableTensors(middle);
        var headTensors = TrainableTensors(head);
        var input = RandomInput(rng);

        var probed = trainedTensors.Concat(middleTensors).Concat(headTensors).ToList();
        using var probe = TapeReachabilityProbe<double>.Arm(probed);

        var engine = AiDotNet.Tensors.Engines.AiDotNetEngine.Current;
        trained.TrainWithCustomLoss(input, output =>
        {
            // Exactly Dreamer's shape: concat the training input with the trained output, forward one
            // network, then feed THAT output into a second network.
            var joined = engine.TensorConcatenate([input, output], axis: 1);
            var imagined = middle.ForwardForTraining(joined);
            var scored = head.ForwardForTraining(imagined);
            var flat = engine.ReduceSum(scored, new[] { 1 }, keepDims: false);
            return engine.ReduceMean(flat, new[] { 0 }, keepDims: false);
        });

        var pass = probe.Observations.LastOrDefault(o => ReferenceEquals(o.Owner, trained));
        Assert.True(pass is not null, "The trained network ran no backward pass.");

        int trainedReached = CountReached(pass!.Reached, trainedTensors);
        int middleReached = CountReached(pass.Reached, middleTensors);
        int headReached = CountReached(pass.Reached, headTensors);

        Assert.True(
            trainedReached > 0,
            $"CHAINED cross-network forwards: none of the trained network's {trainedTensors.Count} "
            + $"tensors were reachable (middle {middleReached}/{middleTensors.Count}, head "
            + $"{headReached}/{headTensors.Count}). If the single-forward test above passes while this "
            + "one fails, feeding one ForwardForTraining output into another severs the tape, which "
            + "affects every model that composes a loss across more than two networks.");
    }
}
