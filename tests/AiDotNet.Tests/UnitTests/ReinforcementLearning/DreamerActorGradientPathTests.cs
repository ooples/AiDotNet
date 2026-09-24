using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using AiDotNet.LinearAlgebra;
using AiDotNet.NeuralNetworks;
using AiDotNet.ReinforcementLearning.Agents.Dreamer;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.ReinforcementLearning;

/// <summary>
/// Reachability regression for Dreamer's actor update.
/// </summary>
/// <remarks>
/// <para>
/// Dreamer improves its policy against an IMAGINED value: q(z,a) = gamma * V(dynamics(z,a)). The
/// original implementation estimated dq/da by central finite differences and then fitted the actor to
/// a nudged copy of its own output, which degenerates silently -- when the value head is locally flat
/// the estimated gradient is exactly zero, the regression target equals the actor's current output,
/// and the supervised step applies no update at all.
/// </para>
/// <para>
/// The per-component invariant caught that as "6 of 36 components received no update in 800 steps",
/// every one of them in _actorNetwork, while the world model, reward, continue and value heads all
/// trained normally. This test asks the precise question underneath that symptom, so a regression
/// reports the cause rather than the symptom: are the actor's own parameters REACHABLE from the loss
/// that its training step differentiates?
/// </para>
/// <para>
/// It is deliberately narrow and fast: one Train() call, not an 800-step loop.
/// </para>
/// </remarks>
public class DreamerActorGradientPathTests
{
    private static NeuralNetworkBase<double> PrivateNetwork(DreamerAgent<double> agent, string fieldName)
    {
        var field = typeof(DreamerAgent<double>).GetField(
            fieldName, BindingFlags.Instance | BindingFlags.NonPublic);
        if (field is null)
        {
            throw new InvalidOperationException(
                $"DreamerAgent<double> no longer has a private field '{fieldName}'. This test reaches "
                + "the actor directly because the agent does not expose it.");
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

    [Fact]
    public void Dreamer_actor_must_be_reachable_from_the_imagined_value()
    {
        var rng = RandomHelper.CreateSeededRandom(29);
        using var agent = new DreamerAgent<double>();

        var actor = PrivateNetwork(agent, "_actorNetwork");
        var dynamics = PrivateNetwork(agent, "_dynamicsNetwork");
        var value = PrivateNetwork(agent, "_valueNetwork");

        var actorTensors = TrainableTensors(actor);
        var dynamicsTensors = TrainableTensors(dynamics);
        var valueTensors = TrainableTensors(value);
        Assert.True(actorTensors.Count > 0, "The actor exposed no parameter tensors to probe.");

        int stateDim = agent.FeatureCount;
        Assert.True(stateDim > 0, "The agent reported a non-positive FeatureCount.");

        Vector<double> RandomState()
        {
            var v = new Vector<double>(stateDim);
            for (int i = 0; i < stateDim; i++) v[i] = rng.NextDouble() * 2.0 - 1.0;
            return v;
        }

        // Take the action width from the agent itself rather than assuming it.
        int actionDim = agent.SelectAction(RandomState(), false).Length;
        Assert.True(actionDim > 0, "The agent produced an empty action.");

        // Fill the replay buffer past the agent's own batch gate, with a reward that actually varies
        // so the value head has something to fit.
        for (int i = 0; i < 512; i++)
        {
            var state = RandomState();
            var nextState = RandomState();
            var action = new Vector<double>(actionDim);
            for (int k = 0; k < actionDim; k++) action[k] = rng.NextDouble() * 2.0 - 1.0;
            agent.StoreExperience(state, action, action[0] > 0.0 ? 1.0 : -1.0, nextState, i % 64 == 63);
        }

        // WARM UP FIRST, then re-read the tensors. The probe keys on reference identity, and these
        // heads are Predict-ed and Train-ed before the actor update, so a network that materializes
        // or replaces its parameter tensors on first forward would hand back instances that are not
        // the ones on the tape -- reporting a uniform 0 of N for every group and looking exactly like
        // a severed tape. One throwaway step removes that ambiguity.
        agent.Train();
        actorTensors = TrainableTensors(actor);
        dynamicsTensors = TrainableTensors(dynamics);
        valueTensors = TrainableTensors(value);

        // Arm the WHOLE chain, not just the actor. The loss is gamma * V(dynamics(z, a)), so which
        // groups come back reached localises where the tape stops:
        //   dynamics+value reached, actor not -> the break is between the actor's output and the
        //                                        concatenated [z|a] handed to the dynamics head;
        //   dynamics reached, value not       -> the break is the dynamics -> value hand-off, i.e.
        //                                        one ForwardForTraining output feeding another;
        //   nothing reached                   -> the loss is constant, so the heads are dead or
        //                                        saturated and the defect is their initialisation.
        var probed = actorTensors.Concat(dynamicsTensors).Concat(valueTensors).ToList();
        using var probe = TapeReachabilityProbe<double>.Arm(probed);

        agent.Train();

        var actorPasses = probe.Observations
            .Where(o => ReferenceEquals(o.Owner, actor))
            .ToList();

        Assert.True(
            actorPasses.Count > 0,
            "The actor's training step never ran a backward pass, so nothing was measured. Either "
            + "Train() returned before the behaviour-learning block (its replay buffer gate was not "
            + "cleared) or the actor is not trained through the tape at all.");

        var actorPass = actorPasses[actorPasses.Count - 1];
        int reached = CountReached(actorPass.Reached, actorTensors);
        int dynamicsReached = CountReached(actorPass.Reached, dynamicsTensors);
        int valueReached = CountReached(actorPass.Reached, valueTensors);

        // Ask the actor DIRECTLY, bypassing the probe entirely. The probe arms tensors from
        // GetParameterStateChunks() while the tape differentiates whatever CollectModelTrainableTensors()
        // returns; if those are different instances for this model, every group reads 0 and the probe
        // is mis-keyed rather than the actor being dead. A non-zero published gradient here separates
        // "my instrument cannot see it" from "there is genuinely no gradient".
        double maxPublishedGradient = 0.0;
        string publishedGradientNote;
        try
        {
            var published = actor.GetParameterGradients();
            for (int i = 0; i < published.Length; i++)
                maxPublishedGradient = Math.Max(maxPublishedGradient, Math.Abs(published[i]));
            publishedGradientNote =
                $"actor published {published.Length} gradients, max |g| = {maxPublishedGradient:E3}";
        }
        catch (NotSupportedException)
        {
            publishedGradientNote = "actor does not publish a gradient surface";
        }

        Assert.True(
            reached > 0,
            $"None of the actor's {actorTensors.Count} parameter tensors were reachable from "
            + "gamma * V(dynamics(z, a)), so the actor receives no gradient and its policy cannot "
            + $"improve. Chain reachability in the same backward pass: dynamics {dynamicsReached} of "
            + $"{dynamicsTensors.Count}, value {valueReached} of {valueTensors.Count}. "
            + $"Independent of the probe: {publishedGradientNote}. Non-zero counts "
            + "downstream with a zero actor count mean the tape reaches the world model but not back "
            + "through the actor's output; all-zero counts mean the loss is constant with respect to "
            + "everything, so the value/dynamics heads are dead or saturated and their initialisation "
            + "is the defect rather than the gradient path.");
    }
}
