using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text.Json;
using System.Threading.Tasks;
using AiDotNet.Models.Options;
using AiDotNet.ReinforcementLearning.Agents.QMIX;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.ReinforcementLearning;

/// <summary>
/// One QMIX update against pymarl, the authors' code (tools/reference-data/qmix_reference.py): its RNNAgent and QMixer
/// in float64 from the same weights, trained on the same two episodes (one terminated, one cut at a time limit). The
/// "paper" case is the paper's configuration (single-layer hypernetworks, max target, summed loss); "pymarl" is
/// qmix.yaml (two-layer hypernetworks, double Q, mean loss).
/// </summary>
public class QmixReferenceTests
{
    private const int Agents = 2, Observation = 3, Actions = 3, State = 4;

    private static readonly Lazy<JsonElement> Data = new(() =>
    {
        const string name = "qmix_reference.json";
        string relative = Path.Combine("ReinforcementLearning", "ReferenceData", name);
        var dir = new DirectoryInfo(AppContext.BaseDirectory);
        while (dir is not null && !File.Exists(Path.Combine(dir.FullName, "tests", "AiDotNet.Tests", relative))) dir = dir.Parent;
        string path = Path.Combine(dir?.FullName ?? throw new FileNotFoundException(name), "tests", "AiDotNet.Tests", relative);
        return JsonDocument.Parse(File.ReadAllText(path)).RootElement;
    });

    private static double[] Values(JsonElement e) => e.EnumerateArray().Select(v => v.GetDouble()).ToArray();

    private static Dictionary<string, double[]> StateDict(JsonElement e)
        => e.EnumerateObject().ToDictionary(p => p.Name, p => Values(p.Value));

    private static QMIXAgent<double> Make(JsonElement c) => new(new QMIXOptions<double>
    {
        NumAgents = Agents,
        StateSize = Observation,
        ActionSize = Actions,
        GlobalStateSize = State,
        RnnHiddenDim = 5,
        MixingEmbedDim = 4,
        HypernetLayers = c.GetProperty("hypernet_layers").GetInt32(),
        HypernetEmbed = 6,
        DoubleQ = c.GetProperty("double_q").GetBoolean(),
        MeanLoss = c.GetProperty("mean").GetBoolean(),
        BatchSize = 2,
        ReplayBufferSize = 2,
        LearningRate = 5e-4,
        DiscountFactor = 0.99,
        Seed = 1,
    });

    [Theory(Timeout = 300000)]
    [InlineData("paper")]
    [InlineData("pymarl")]
    public async Task OneUpdate_MatchesPymarl(string name)
    {
        await Task.Yield();
        var fixture = Data.Value.GetProperty(name);
        var c = fixture.GetProperty("case");
        using var agent = Make(c);
        agent.LoadTorchWeights(StateDict(fixture.GetProperty("initial").GetProperty("agent")),
            StateDict(fixture.GetProperty("initial").GetProperty("mixer")));

        var lengths = fixture.GetProperty("lengths").EnumerateArray().Select(v => v.GetInt32()).ToArray();
        var terminated = fixture.GetProperty("terminated").EnumerateArray().Select(v => v.GetBoolean()).ToArray();
        var obs = Values(fixture.GetProperty("obs"));
        var state = Values(fixture.GetProperty("state"));
        var actions = fixture.GetProperty("actions").EnumerateArray().Select(v => v.GetInt32()).ToArray();
        var reward = Values(fixture.GetProperty("reward"));
        int steps = lengths.Max();

        List<Vector<double>> Observations(int b, int t) => Enumerable.Range(0, Agents)
            .Select(i => new Vector<double>(obs.Skip(((b * (steps + 1) + t) * Agents + i) * Observation).Take(Observation).ToArray())).ToList();
        Vector<double> GlobalState(int b, int t) => new(state.Skip((b * (steps + 1) + t) * State).Take(State).ToArray());

        for (int b = 0; b < lengths.Length; b++)
        {
            for (int t = 0; t < lengths[b]; t++)
            {
                var chosen = Enumerable.Range(0, Agents).Select(i =>
                {
                    var v = new Vector<double>(Actions);
                    v[actions[(b * steps + t) * Agents + i]] = 1.0;
                    return v;
                }).ToList();
                bool last = t == lengths[b] - 1;
                agent.StoreMultiAgentExperience(Observations(b, t), chosen, reward[b * steps + t], Observations(b, t + 1),
                    GlobalState(b, t), GlobalState(b, t + 1), done: last && terminated[b], truncated: last && !terminated[b]);
            }
            if (b == 0) Assert.Equal(0.0, agent.Train()); // one episode is not yet a batch
        }

        double loss = agent.Train();
        double expectedLoss = fixture.GetProperty("loss").GetDouble();
        Assert.True(Math.Abs(loss - expectedLoss) < 1e-9, $"{name}: loss {loss} vs {expectedLoss}");
        Assert.Equal(0.0, agent.Train()); // no new episode, no update

        using var expected = Make(c);
        expected.LoadTorchWeights(StateDict(fixture.GetProperty("final").GetProperty("agent")),
            StateDict(fixture.GetProperty("final").GetProperty("mixer")));
        var (expectedAgent, expectedMixer) = expected.LiveNetworkParameters();
        var (actualAgent, actualMixer) = agent.LiveNetworkParameters();
        AssertClose(name + " agent", expectedAgent, actualAgent);
        AssertClose(name + " mixer", expectedMixer, actualMixer);
    }

    private static void AssertClose(string what, Vector<double> expected, Vector<double> actual)
    {
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++)
            Assert.True(Math.Abs(expected[i] - actual[i]) < 1e-9, $"{what} parameter {i}: {expected[i]} vs {actual[i]}");
    }
}
