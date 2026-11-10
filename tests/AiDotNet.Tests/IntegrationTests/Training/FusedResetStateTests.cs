using System;
using AiDotNet.Enums;
using AiDotNet.LossFunctions;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.Optimizers;
using AiDotNet.Tensors.Engines.Optimization;
using AiDotNet.Training;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.Training;

/// <summary>
/// <c>ResetState</c> clears recurrent state, not optimizer state. A committed fused plan holds the optimizer's moments,
/// so it must keep them (the eager optimizer keeps its own) and stay committed: an uncommitted session with a live plan
/// would fall back to the eager tape on its next failed step, continuing from the eager optimizer's older moments
/// without a word.
/// </summary>
[Collection("FusedTrainingSerial")]
public class FusedResetStateTests
{
    private const int Steps = 10, ResetAfter = 5;

    private static FeedForwardNeuralNetwork<float> Build()
    {
        NeuralNetworkArchitecture<float>.DefaultRandomSeedOverride = 1234;
        var architecture = new NeuralNetworkArchitecture<float>(
            inputType: InputType.OneDimensional,
            taskType: NeuralNetworkTaskType.Regression,
            complexity: NetworkComplexity.Simple,
            inputSize: 8,
            outputSize: 3);
        var adam = new AdamOptimizer<float, Tensor<float>, Tensor<float>>(
            null, new AdamOptimizerOptions<float, Tensor<float>, Tensor<float>> { InitialLearningRate = 1e-2 });
        var network = new FeedForwardNeuralNetwork<float>(architecture, adam, new MeanSquaredErrorLoss<float>());
        network.SetTrainingMode(true);
        return network;
    }

    private static (Tensor<float> X, Tensor<float> Y) Data()
    {
        var x = new Tensor<float>(new[] { 4, 8 });
        var y = new Tensor<float>(new[] { 4, 3 });
        for (int b = 0; b < 4; b++)
        {
            for (int f = 0; f < 8; f++) x[b, f] = (((b * 8 + f) % 7) - 3) * 0.1f;
            for (int o = 0; o < 3; o++) y[b, o] = (((b * 3 + o) % 5) - 2) * 0.2f;
        }
        return (x, y);
    }

    // Trains fused for Steps steps, calling ResetState after ResetAfter when asked; returns the parameters, the fused
    // step count and whether the session was still committed right after the reset.
    private static (float[] Parameters, long FusedSteps, bool CommittedAfterReset) TrainFused(
        FeedForwardNeuralNetwork<float> network, float[] init, bool resetState)
    {
        network.UpdateParameters(new Vector<float>(init));
        var (x, y) = Data();
        var saved = TensorCodecOptions.Current;
        bool committedAfterReset = false;
        try
        {
            TensorCodecOptions.SetCurrent(new TensorCodecOptions { EnableCompilation = true });
            CompiledTapeTrainingStep<float>.Invalidate();
            CompiledTapeTrainingStep<float>.ResetFusedStepCount();
            for (int step = 1; step <= Steps; step++)
            {
                network.Train(x, y);
                if (resetState && step == ResetAfter)
                {
                    network.ResetState();
                    committedAfterReset = network.FusedSession.IsCommitted;
                }
            }

            return (network.GetParameters().ToArray(), CompiledTapeTrainingStep<float>.GetFusedStepCount(), committedAfterReset);
        }
        finally
        {
            TensorCodecOptions.SetCurrent(saved);
            CompiledTapeTrainingStep<float>.Invalidate();
        }
    }

    /// <summary>
    /// A feed-forward network has no recurrent state, so a ResetState mid-run must leave the fused trajectory exactly
    /// where it would be without one. Dropping the plan's Adam moments there (restarting at t = 1, an update of
    /// lr * sign(g)) moves the weights by about the learning rate.
    /// </summary>
    [Fact(Timeout = 120_000)]
    public async System.Threading.Tasks.Task ResetState_KeepsTheCommittedPlanAndItsMoments()
    {
        await System.Threading.Tasks.Task.Yield();
        var reference = Build();
        float[] init = reference.GetParameters().ToArray();

        var (withoutReset, referenceSteps, _) = TrainFused(reference, init, resetState: false);
        var (withReset, fusedSteps, committedAfterReset) = TrainFused(Build(), init, resetState: true);

        Assert.Equal(Steps, referenceSteps);
        Assert.Equal(Steps, fusedSteps);
        Assert.True(committedAfterReset, "ResetState uncommitted a fused session whose plan still holds the optimizer's moments.");
        Assert.Equal(withoutReset, withReset);
    }

    /// <summary>An uncommitted session that a prior fallback disabled is re-enabled by ResetState.</summary>
    [Fact(Timeout = 120_000)]
    public async System.Threading.Tasks.Task ResetState_ReEnablesAnUncommittedSession()
    {
        await System.Threading.Tasks.Task.Yield();
        var network = Build();
        network.FusedSession.IsDisabled = true;
        Assert.False(network.FusedSession.IsCommitted);

        network.ResetState();

        Assert.False(network.FusedSession.IsDisabled);
    }
}
