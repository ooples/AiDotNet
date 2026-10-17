using System;
using System.IO;
using System.Linq;
using System.Text.Json;
using System.Threading.Tasks;
using AiDotNet.Models.Options;
using AiDotNet.Optimizers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.Optimizers;

/// <summary>
/// D-Adaptation Adam reproduces the reference optimizer step for step (audiocraft/optim/dadam.py, float64;
/// <c>tools/reference-data/dadapt_adam_reference.py</c>): the parameters and the distance estimate d after each of a
/// fixed sequence of gradients, with MusicGen's settings (decoupled weight decay 0.1, β = (0.9, 0.95)) and with
/// coupled weight decay and a learning-rate multiplier.
/// </summary>
public class DAdaptAdamReferenceTests
{
    private static JsonElement Cases()
    {
        const string name = "dadapt_adam_reference.json";
        string relative = Path.Combine("UnitTests", "Optimizers", "ReferenceData", name);
        string path = Path.Combine(AppContext.BaseDirectory, relative);
        if (!File.Exists(path))
        {
            var dir = new DirectoryInfo(AppContext.BaseDirectory);
            while (dir is not null && !File.Exists(Path.Combine(dir.FullName, "tests", "AiDotNet.Tests", relative))) dir = dir.Parent;
            path = Path.Combine(dir?.FullName ?? throw new FileNotFoundException(name), "tests", "AiDotNet.Tests", relative);
        }
        return JsonDocument.Parse(File.ReadAllText(path)).RootElement.GetProperty("cases");
    }

    private static double[] Values(JsonElement e) => e.EnumerateArray().Select(x => x.GetDouble()).ToArray();

    [Theory(Timeout = 30000)]
    [InlineData(0)]
    [InlineData(1)]
    public async Task EveryStep_MatchesTheReference(int index)
    {
        await Task.Yield();
        var c = Cases()[index];
        var betas = Values(c.GetProperty("betas"));
        var optimizer = new DAdaptAdamOptimizer<double, Tensor<double>, Tensor<double>>(null,
            new DAdaptAdamOptimizerOptions<double, Tensor<double>, Tensor<double>>
            {
                InitialLearningRate = c.GetProperty("lr").GetDouble(),
                Beta1 = betas[0],
                Beta2 = betas[1],
                Epsilon = c.GetProperty("eps").GetDouble(),
                WeightDecay = c.GetProperty("weight_decay").GetDouble(),
                DecoupleWeightDecay = c.GetProperty("decouple").GetBoolean(),
                InitialDistance = c.GetProperty("d0").GetDouble(),
            });
        var parameters = new Vector<double>(Values(c.GetProperty("start")));
        var gradients = c.GetProperty("gradients").EnumerateArray().Select(Values).ToArray();
        var trajectory = c.GetProperty("trajectory").EnumerateArray().ToArray();
        for (int step = 0; step < gradients.Length; step++)
        {
            parameters = optimizer.UpdateParameters(parameters, new Vector<double>(gradients[step]));
            var expected = Values(trajectory[step].GetProperty("params"));
            double d = trajectory[step].GetProperty("d").GetDouble();
            Assert.Equal(d, optimizer.Distance, 12);
            for (int i = 0; i < expected.Length; i++)
                Assert.True(Math.Abs(expected[i] - parameters[i]) <= 1e-12 * Math.Max(1.0, Math.Abs(expected[i])),
                    $"{c.GetProperty("name").GetString()} step {step}, parameter {i}: {parameters[i]} vs {expected[i]}");
        }
        // The estimate grew from d0, so the comparison covers the adaptation, not just Adam with a tiny step.
        Assert.True(optimizer.Distance > c.GetProperty("d0").GetDouble());
    }
}
