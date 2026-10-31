using System;
using System.IO;
using System.Linq;
using System.Text.Json;
using System.Threading.Tasks;
using AiDotNet.Finance.Trading.Agents;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.ReinforcementLearning;

/// <summary>
/// FinancialPPOAgent's update against a PyTorch port of OpenAI Baselines' ppo1 iteration (the code of Schulman et al.
/// 2017; tools/reference-data/ppo_reference.py): the same initial weights and rollout give the same mean loss and the
/// same weights after three epochs of minibatch Adam, for a discrete policy with CleanRL's options (entropy bonus, value
/// weight 0.5, global gradient clipping) and for a Gaussian policy with its state-independent log standard deviation.
/// </summary>
public class PpoReferenceTests
{
    private static readonly Lazy<JsonElement> Data = new(() =>
    {
        const string name = "ppo_reference.json";
        string relative = Path.Combine("ReinforcementLearning", "ReferenceData", name);
        var dir = new DirectoryInfo(AppContext.BaseDirectory);
        while (dir is not null && !File.Exists(Path.Combine(dir.FullName, "tests", "AiDotNet.Tests", relative))) dir = dir.Parent;
        string path = Path.Combine(dir?.FullName ?? throw new FileNotFoundException(name), "tests", "AiDotNet.Tests", relative);
        return JsonDocument.Parse(File.ReadAllText(path)).RootElement;
    });

    private static double[] Values(JsonElement e) => e.EnumerateArray().Select(v => v.GetDouble()).ToArray();

    [Fact(Timeout = 120000)]
    public async Task TheSeededGenerator_IsDotNetsSystemRandom()
    {
        await Task.Yield();
        // The fixture's minibatch shuffles come from a port of System.Random(seed); this pins that the agent's generator
        // draws the same sequence, so the two runs shuffle alike.
        var random = RandomHelper.CreateSeededRandom(11);
        var expected = Data.Value.GetProperty("discrete").GetProperty("random_probe").EnumerateArray().Select(v => v.GetInt32()).ToArray();
        Assert.Equal(expected, Enumerable.Range(0, expected.Length).Select(_ => random.Next(1000)).ToArray());
    }

    [Theory(Timeout = 300000)]
    [InlineData("discrete")]
    [InlineData("continuous")]
    public async Task OneUpdate_MatchesBaselinesPpo1(string name)
    {
        await Task.Yield();
        var fixture = Data.Value.GetProperty(name);
        var c = fixture.GetProperty("case");
        int s = c.GetProperty("state_size").GetInt32(), a = c.GetProperty("action_size").GetInt32(), t = c.GetProperty("rollout").GetInt32();
        bool continuous = c.GetProperty("continuous").GetBoolean();
        var options = new FinancialPPOAgentOptions<double>
        {
            StateSize = s,
            ActionSize = a,
            ContinuousActions = continuous,
            HiddenLayers = c.GetProperty("hidden").EnumerateArray().Select(v => v.GetInt32()).ToArray(),
            RolloutLength = t,
            MiniBatchSize = c.GetProperty("minibatch").GetInt32(),
            NumEpochs = c.GetProperty("epochs").GetInt32(),
            LearningRate = c.GetProperty("lr").GetDouble(),
            AdamEpsilon = c.GetProperty("adam_eps").GetDouble(),
            DiscountFactor = c.GetProperty("gamma").GetDouble(),
            GAELambda = c.GetProperty("lambda").GetDouble(),
            PPOClipRange = c.GetProperty("clip").GetDouble(),
            EntropyCoefficient = c.GetProperty("ent_coef").GetDouble(),
            ValueCoefficient = c.GetProperty("vf_coef").GetDouble(),
            MaxGradientNorm = c.GetProperty("max_grad_norm").ValueKind == JsonValueKind.Null ? null : c.GetProperty("max_grad_norm").GetDouble(),
            Seed = c.GetProperty("seed").GetInt32(),
        };
        using var agent = new FinancialPPOAgent<double>(
            new NeuralNetworkArchitecture<double>(inputFeatures: s, outputSize: a),
            new NeuralNetworkArchitecture<double>(inputFeatures: s, outputSize: 1), options);

        var states = Values(fixture.GetProperty("states"));
        var actions = Values(fixture.GetProperty("actions"));
        var rewards = Values(fixture.GetProperty("rewards"));
        var dones = fixture.GetProperty("dones").EnumerateArray().Select(v => v.GetBoolean()).ToArray();
        Vector<double> State(int i) => new(states.Skip(i * s).Take(s).ToArray());

        // Materialize both networks, then load the reference's initial weights in the agent's own component order, each
        // dense layer as its [in, out] weights then its biases.
        agent.SelectAction(State(0), training: false);
        var initial = c.GetProperty("initial");
        agent.SetParameters(new Vector<double>(InLayout(agent,
            continuous ? Values(c.GetProperty("logstd")) : Array.Empty<double>(),
            Values(initial.GetProperty("actor")), Values(initial.GetProperty("critic")))));

        for (int i = 0; i < t; i++)
        {
            var next = i == t - 1 ? new Vector<double>(Values(fixture.GetProperty("next_last"))) : State(i + 1);
            agent.StoreExperience(State(i), new Vector<double>(actions.Skip(i * a).Take(a).ToArray()), rewards[i], next, dones[i]);
        }
        double loss = agent.Train();

        Assert.True(Math.Abs(loss - fixture.GetProperty("mean_loss").GetDouble()) < 1e-9,
            $"{name}: mean loss {loss} vs {fixture.GetProperty("mean_loss").GetDouble()}");
        var expected = InLayout(agent, Values(fixture.GetProperty("logstd")), Values(fixture.GetProperty("actor")), Values(fixture.GetProperty("critic")));
        var actual = agent.GetParameters().ToArray();
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++)
            Assert.True(Math.Abs(expected[i] - actual[i]) < 1e-9, $"{name}: parameter {i}: {expected[i]} vs {actual[i]}");
    }

    // The agent's flat parameter vector, component by component as GetParameterStateChunks lays it out.
    private static double[] InLayout(FinancialPPOAgent<double> agent, double[] logStd, double[] actor, double[] critic)
    {
        var parts = agent.GetParameterStateChunks().GroupBy(chunk => chunk.StableId.Contains("::_actor") ? "actor"
            : chunk.StableId.Contains("::_critic") ? "critic" : chunk.StableId.Contains("::_logStd") ? "logstd" : chunk.StableId)
            .Select(group => group.Key).Distinct().ToList();
        return parts.SelectMany(part => part switch
        {
            "actor" => actor,
            "critic" => critic,
            "logstd" => logStd,
            _ => throw new InvalidOperationException($"Unexpected parameter component {part}."),
        }).ToArray();
    }
}
