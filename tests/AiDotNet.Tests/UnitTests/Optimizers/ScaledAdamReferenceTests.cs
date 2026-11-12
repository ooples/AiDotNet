using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using AiDotNet.Models.Options;
using AiDotNet.Optimizers;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.LinearAlgebra;
using Newtonsoft.Json.Linq;
using Xunit;

namespace AiDotNet.Tests.UnitTests.Optimizers;

/// <summary>
/// ScaledAdam against GPT-SoVITS's AR/modules/optim.py (icefall's) in float64 (tools/reference-data/scaled_adam_reference.py):
/// a matrix, a vector, two stacked scalars and a column over 30 steps with clipping and scale learning engaged.
/// </summary>
public class ScaledAdamReferenceTests
{
    private static JObject Fixture()
    {
        const string relative = "tests/AiDotNet.Tests/UnitTests/Optimizers/ReferenceData/scaled_adam_reference.json";
        string path = Path.Combine(AppContext.BaseDirectory, "UnitTests", "Optimizers", "ReferenceData", "scaled_adam_reference.json");
        for (var dir = new DirectoryInfo(AppContext.BaseDirectory); !File.Exists(path) && dir is not null; dir = dir.Parent)
            path = Path.Combine(dir.FullName, relative);
        return JObject.Parse(File.ReadAllText(path));
    }

    private static double[] Doubles(JToken t) => t.Select(v => (double)v).ToArray();

    [Fact]
    public void TapeSteps_MatchTheReference()
    {
        var f = Fixture();
        var names = f["names"]!.Select(n => (string)n!).ToArray();
        var options = new ScaledAdamOptimizerOptions<double, Matrix<double>, Vector<double>>
        {
            InitialLearningRate = 0.01, Beta1 = 0.9, Beta2 = 0.95, ClippingScale = 2.0, ClippingUpdatePeriod = 5,
        };
        var optimizer = new ScaledAdamOptimizer<double, Matrix<double>, Vector<double>>(null, options);
        var parameters = names.Select(n => new Tensor<double>(f["shapes"]![n]!.Select(v => (int)v).ToArray(),
            new Vector<double>(Doubles(f["initial"]![n]!)))).ToArray();
        var steps = f["steps"]!.ToArray();
        var gradients = f["gradients"]!.ToArray();
        for (int k = 0; k < steps.Length; k++)
        {
            var grads = new Dictionary<Tensor<double>, Tensor<double>>();
            for (int i = 0; i < names.Length; i++)
                grads[parameters[i]] = new Tensor<double>(parameters[i].Shape.ToArray(), new Vector<double>(Doubles(gradients[k][names[i]]!)));
            optimizer.Step(new TapeStepContext<double>(parameters, grads, 0.0));
            for (int i = 0; i < names.Length; i++)
            {
                var expected = Doubles(steps[k][names[i]]!);
                var actual = parameters[i].ToArray();
                for (int j = 0; j < expected.Length; j++)
                    Assert.True(Math.Abs(expected[j] - actual[j]) <= 1e-10 * Math.Max(1, Math.Abs(expected[j])),
                        $"{names[i]} step {k + 1}[{j}]: {expected[j]} vs {actual[j]}");
            }
        }
    }
}
