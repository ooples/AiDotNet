using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text.Json;
using System.Threading.Tasks;
using AiDotNet.Enums;
using AiDotNet.Finance.Trading.Agents;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.ReinforcementLearning;

/// <summary>
/// FinancialSACAgent against a float64 PyTorch port of Soft Actor-Critic's gradient steps
/// (tools/reference-data/sac_reference.py): three steps from the same weights, replay contents and random streams give
/// the same losses and the same weights, targets included. The cases are the 2018a paper (value network, fixed
/// temperature), the 2018b paper (learned temperature, stepped on α), and 2018b with the authors' code's choices
/// (simultaneous steps, the mean of the Q-functions, log α, a state-independent log-std).
/// </summary>
public class SacReferenceTests
{
    private static readonly Lazy<JsonElement> Data = new(() =>
    {
        const string name = "sac_reference.json";
        string relative = Path.Combine("ReinforcementLearning", "ReferenceData", name);
        var dir = new DirectoryInfo(AppContext.BaseDirectory);
        while (dir is not null && !File.Exists(Path.Combine(dir.FullName, "tests", "AiDotNet.Tests", relative))) dir = dir.Parent;
        string path = Path.Combine(dir?.FullName ?? throw new FileNotFoundException(name), "tests", "AiDotNet.Tests", relative);
        return JsonDocument.Parse(File.ReadAllText(path)).RootElement;
    });

    private static double[] Values(JsonElement e) => e.EnumerateArray().Select(v => v.GetDouble()).ToArray();

    [Theory(Timeout = 300000)]
    [InlineData("v1")]
    [InlineData("v2")]
    [InlineData("v2_code")]
    public async Task ThreeSteps_MatchTheReference(string name)
    {
        await Task.Yield();
        var fixture = Data.Value.GetProperty(name);
        var c = fixture.GetProperty("case");
        int s = c.GetProperty("state_size").GetInt32(), a = c.GetProperty("action_size").GetInt32();
        bool v1 = c.GetProperty("variant").GetString() == "v1";
        bool stateHead = c.GetProperty("state_head").GetBoolean();
        var options = v1 ? new FinancialSACAgentOptions<double>() : FinancialSACAgentOptions<double>.Sac2018b();
        options.StateSize = s;
        options.ActionSize = a;
        options.ContinuousActions = true;
        options.HiddenLayers = c.GetProperty("hidden").EnumerateArray().Select(v => v.GetInt32()).ToArray();
        options.BatchSize = c.GetProperty("batch").GetInt32();
        options.ReplayBufferSize = 100;
        options.WarmupSteps = 0;
        options.Seed = c.GetProperty("seed").GetInt32();
        options.LearningRate = c.GetProperty("lr").GetDouble();
        options.DiscountFactor = c.GetProperty("gamma").GetDouble();
        options.Tau = c.GetProperty("tau").GetDouble();
        options.MaxPositionSize = c.GetProperty("max_position").GetDouble();
        options.SACAlpha = c.GetProperty("alpha").GetDouble();
        options.AutoTuneAlpha = c.GetProperty("auto_alpha").GetBoolean();
        options.InitialLogAlpha = c.GetProperty("initial_log_alpha").GetDouble();
        options.ActorHead = stateHead ? SacActorHead.StateConditionedGaussian : SacActorHead.StateIndependentLogStd;
        options.PolicyCriticAggregate = c.GetProperty("aggregate").GetString() switch
        {
            "mean" => SacCriticAggregate.Mean,
            "first" => SacCriticAggregate.FirstCritic,
            _ => SacCriticAggregate.Minimum,
        };
        options.SimultaneousUpdates = c.GetProperty("simultaneous").GetBoolean();
        options.TemperatureParameter = c.GetProperty("temperature").GetString() == "log_alpha"
            ? SacTemperatureParameter.LogAlpha : SacTemperatureParameter.Alpha;

        using var agent = new FinancialSACAgent<double>(
            new NeuralNetworkArchitecture<double>(inputFeatures: s, outputSize: stateHead ? 2 * a : a),
            new NeuralNetworkArchitecture<double>(inputFeatures: s + a, outputSize: 1), options);

        var states = Values(fixture.GetProperty("states"));
        Vector<double> Row(double[] data, int i, int width) => new(data.Skip(i * width).Take(width).ToArray());
        agent.SelectAction(Row(states, 0, s), training: false);

        // The reference's initial weights, targets equal to what they track, in the agent's component order.
        var initial = fixture.GetProperty("initial");
        double initialTemperature = options.TemperatureParameter == SacTemperatureParameter.LogAlpha
            ? options.InitialLogAlpha : Math.Exp(options.InitialLogAlpha);
        var start = new Dictionary<string, double[]>
        {
            ["actor"] = Values(initial.GetProperty("actor")),
            ["critic1"] = Values(initial.GetProperty("critic1")),
            ["critic2"] = Values(initial.GetProperty("critic2")),
            ["value"] = Values(initial.GetProperty("value")),
            ["targetValue"] = Values(initial.GetProperty("value")),
            ["targetCritic1"] = Values(initial.GetProperty("critic1")),
            ["targetCritic2"] = Values(initial.GetProperty("critic2")),
            ["logStd"] = Values(initial.GetProperty("logstd")),
            ["temperature"] = options.AutoTuneAlpha ? new[] { initialTemperature } : Array.Empty<double>(),
        };
        agent.SetParameters(new Vector<double>(InLayout(agent, start)));

        var actions = Values(fixture.GetProperty("actions"));
        var rewards = Values(fixture.GetProperty("rewards"));
        var nextStates = Values(fixture.GetProperty("next_states"));
        var dones = fixture.GetProperty("dones").EnumerateArray().Select(v => v.GetBoolean()).ToArray();
        for (int i = 0; i < dones.Length; i++)
            agent.StoreExperience(Row(states, i, s), Row(actions, i, a), rewards[i], Row(nextStates, i, s), dones[i]);

        var losses = Values(fixture.GetProperty("losses"));
        for (int step = 0; step < losses.Length; step++)
        {
            double loss = agent.Train();
            Assert.True(Math.Abs(loss - losses[step]) < 1e-9, $"{name} step {step}: loss {loss} vs {losses[step]}");
        }

        var final = fixture.GetProperty("final");
        var end = new Dictionary<string, double[]>
        {
            ["actor"] = Values(final.GetProperty("actor")),
            ["critic1"] = Values(final.GetProperty("critic1")),
            ["critic2"] = Values(final.GetProperty("critic2")),
            ["value"] = Values(final.GetProperty("value")),
            ["targetValue"] = Values(final.GetProperty("target_value")),
            ["targetCritic1"] = Values(final.GetProperty("target_critic1")),
            ["targetCritic2"] = Values(final.GetProperty("target_critic2")),
            ["logStd"] = Values(final.GetProperty("logstd")),
            ["temperature"] = Values(final.GetProperty("temperature")),
        };
        var expected = InLayout(agent, end);
        var actual = agent.GetParameters().ToArray();
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++)
            Assert.True(Math.Abs(expected[i] - actual[i]) < 1e-9, $"{name}: parameter {i}: {expected[i]} vs {actual[i]}");
    }

    // The agent's flat parameter vector, component by component as GetParameterStateChunks lays it out.
    private static double[] InLayout(FinancialSACAgent<double> agent, Dictionary<string, double[]> parts)
    {
        var order = agent.GetParameterStateChunks().Select(chunk => chunk.StableId.Substring(chunk.StableId.IndexOf("::_", StringComparison.Ordinal) + 3))
            .Select(id => id.Split('/', '.', ':')[0]).Distinct().ToList();
        return order.SelectMany(part => parts.TryGetValue(part, out var values) ? values
            : throw new InvalidOperationException($"Unexpected parameter component {part}.")).ToArray();
    }
}
