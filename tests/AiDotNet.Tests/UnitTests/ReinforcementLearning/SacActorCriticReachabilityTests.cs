using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using AiDotNet.LinearAlgebra;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.ReinforcementLearning.Agents.SAC;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.ReinforcementLearning;

/// <summary>
/// Reachability regression for SAC's actor update.
/// </summary>
/// <remarks>
/// <para>
/// SAC's actor minimizes <c>alpha * log pi - min(Q1, Q2)</c>. Reading the critics through
/// <c>Predict</c> runs them inside a <c>NoGradScope</c>, so the Q term arrives as a tape-detached
/// constant: the actor is then trained by the entropy term alone and never follows the reward.
/// </para>
/// <para>
/// No parameter-movement test can catch this. The entropy term still carries real gradient, so the
/// actor's weights change every single step and every liveness assertion passes while the agent is
/// silently broken. The only question that separates the two cases is whether the critic parameters
/// were REACHABLE from the actor's loss — the question <c>torch.autograd.grad(loss, params)</c>
/// answers by raising when a parameter is unused in the graph.
/// </para>
/// </remarks>
public class SacActorCriticReachabilityTests
{
    private const int StateDim = 4;
    private const int ActionDim = 2;
    private const int Batch = 8;

    private static NeuralNetworkBase<double> PrivateNetwork(SACAgent<double> agent, string fieldName)
    {
        var field = typeof(SACAgent<double>).GetField(
            fieldName, BindingFlags.Instance | BindingFlags.NonPublic);
        if (field is null)
        {
            throw new InvalidOperationException(
                $"SACAgent<double> no longer has a private field '{fieldName}'. This test reaches the "
                + "actor and critics directly because the agent does not expose them.");
        }

        if (field.GetValue(agent) is not NeuralNetworkBase<double> network)
        {
            throw new InvalidOperationException(
                $"Field '{fieldName}' is not a NeuralNetworkBase<double>, so its parameter tensors "
                + "cannot be enumerated for a reachability check.");
        }

        return network;
    }

    private static List<Tensor<double>> TrainableTensors(NeuralNetworkBase<double> network)
    {
        var tensors = new List<Tensor<double>>();
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
        IReadOnlyCollection<Tensor<double>> reached, IReadOnlyList<Tensor<double>> wanted)
        => wanted.Count(w => reached.Any(r => ReferenceEquals(r, w)));

    private static void FillBuffer(SACAgent<double> agent, int count)
    {
        var rng = RandomHelper.CreateSeededRandom(11);
        for (int i = 0; i < count; i++)
        {
            var state = new Vector<double>(StateDim);
            var nextState = new Vector<double>(StateDim);
            for (int j = 0; j < StateDim; j++)
            {
                state[j] = rng.NextDouble() * 2.0 - 1.0;
                nextState[j] = rng.NextDouble() * 2.0 - 1.0;
            }

            var action = new Vector<double>(ActionDim);
            for (int k = 0; k < ActionDim; k++)
            {
                action[k] = rng.NextDouble() * 2.0 - 1.0;
            }

            // Reward correlated with the first action dimension so the objective is not degenerate.
            agent.StoreExperience(state, action, action[0] > 0.0 ? 1.0 : -1.0, nextState, false);
        }
    }

    [Fact]
    public void Sac_actor_update_must_reach_the_critic_parameters()
    {
        var options = new SACOptions<double>
        {
            StateSize = StateDim,
            ActionSize = ActionDim,
            BatchSize = Batch,
            ReplayBufferSize = 256,
            // The defaults (WarmupSteps = 10000, BatchSize = 256) put the gradient update out of
            // reach of any test budget; the update itself is unchanged.
            WarmupSteps = 1,
            GradientSteps = 1,
            AutoTuneTemperature = false,
        };

        var agent = new SACAgent<double>(options);
        var actor = PrivateNetwork(agent, "_policyNetwork");
        var critic1 = PrivateNetwork(agent, "_q1Network");
        var critic2 = PrivateNetwork(agent, "_q2Network");

        var actorTensors = TrainableTensors(actor);
        var criticTensors = TrainableTensors(critic1).Concat(TrainableTensors(critic2)).ToList();

        Assert.True(actorTensors.Count > 0, "The actor exposed no parameter tensors to probe.");
        Assert.True(criticTensors.Count > 0, "The critics exposed no parameter tensors to probe.");

        FillBuffer(agent, Batch * 8);

        // Arm with BOTH sets. The critics train before the actor in the same Train() call, so the
        // probe records each backward pass separately and only the actor-owned pass is inspected.
        var probed = actorTensors.Concat(criticTensors).ToList();
        using var probe = TapeReachabilityProbe<double>.Arm(probed);

        agent.Train();

        var actorPasses = probe.Observations.Where(o => ReferenceEquals(o.Owner, actor)).ToList();
        Assert.True(
            actorPasses.Count > 0,
            "SAC's actor update never ran, so nothing was measured. The warmup/batch gate in Train() "
            + "was not cleared and this test proves nothing about reachability.");

        var actorPass = actorPasses[actorPasses.Count - 1];

        // Positive control FIRST. The actor's own parameters are the sources of this very backward
        // pass, so they must be reachable. If they are not, the probe is not seeing the live tensors
        // (for example, if the parameter chunks handed out copies) and the critic result below would
        // be a false alarm rather than evidence.
        int actorReached = CountReached(actorPass.Reached, actorTensors);
        Assert.True(
            actorReached > 0,
            $"Instrument check failed: none of the actor's {actorTensors.Count} parameter tensors "
            + "were reachable from the actor's own loss. The probe is not observing the tensors the "
            + "tape differentiates, so no conclusion about the critics can be drawn from this run.");

        int criticReached = CountReached(actorPass.Reached, criticTensors);
        Assert.True(
            criticReached > 0,
            $"SAC's actor update reached {actorReached} actor tensors but 0 of "
            + $"{criticTensors.Count} critic tensors. The min(Q1,Q2) term is detached from the tape, "
            + "so the actor is trained by the entropy term alone and the policy cannot follow the "
            + "reward. Read the critics with ForwardForTraining instead of Predict, which runs "
            + "inside a NoGradScope.");
    }
}
