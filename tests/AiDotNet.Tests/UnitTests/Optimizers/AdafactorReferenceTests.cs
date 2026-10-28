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
/// Adafactor against transformers' implementation (tools/reference-data/adafactor_reference.py): four steps of the
/// paper's defaults (factored, relative step), PaLM's fine-tuning (a constant 5e-5 with parameter scaling and beta1 0.9)
/// and an unfactored second moment, through the tape step and, for the unfactored case, the vector update.
/// </summary>
public class AdafactorReferenceTests
{
    private static JObject Fixture()
    {
        const string relative = "tests/AiDotNet.Tests/UnitTests/Optimizers/ReferenceData/adafactor_reference.json";
        string path = Path.Combine(AppContext.BaseDirectory, "UnitTests", "Optimizers", "ReferenceData", "adafactor_reference.json");
        for (var dir = new DirectoryInfo(AppContext.BaseDirectory); !File.Exists(path) && dir is not null; dir = dir.Parent)
            path = Path.Combine(dir.FullName, relative);
        return JObject.Parse(File.ReadAllText(path));
    }

    private static double[] Doubles(JToken t) => t.Select(v => (double)v).ToArray();

    private static AdafactorOptimizer<double, Matrix<double>, Vector<double>> Create(string name)
    {
        var o = new AdafactorOptimizerOptions<double, Matrix<double>, Vector<double>>();
        switch (name)
        {
            case "palm":
                o.UseRelativeStepSize = false;
                o.InitialLearningRate = 5e-5;
                o.MultiplyByParameterScale = true;
                o.Beta1 = 0.9;
                break;
            case "unfactored":
                o.Factored = false;
                o.Beta1 = 0.9;
                break;
        }
        return new AdafactorOptimizer<double, Matrix<double>, Vector<double>>(null, o);
    }

    private static void Close(double[] expected, double[] actual, string what)
    {
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++)
            Assert.True(Math.Abs(expected[i] - actual[i]) <= 1e-12 * Math.Max(1, Math.Abs(expected[i])), $"{what}[{i}]: {expected[i]} vs {actual[i]}");
    }

    [Theory]
    [InlineData("relative")]
    [InlineData("palm")]
    [InlineData("unfactored")]
    public void TapeStep_MatchesTransformers(string name)
    {
        var f = Fixture();
        var optimizer = Create(name);
        var parameter = new Tensor<double>(new[] { 4, 6 }, new Vector<double>(Doubles(f["initial"]!)));
        var gradients = f["gradients"]!.Select(g => Doubles(g)).ToArray();
        for (int k = 0; k < gradients.Length; k++)
        {
            var gradient = new Tensor<double>(new[] { 4, 6 }, new Vector<double>(gradients[k]));
            optimizer.Step(new TapeStepContext<double>(new[] { parameter },
                new Dictionary<Tensor<double>, Tensor<double>> { [parameter] = gradient }, 0.0));
            Close(Doubles(f[name]![k]!), parameter.ToArray(), $"{name} step {k + 1}");
        }
    }

    [Fact]
    public void VectorUpdate_MatchesTransformersUnfactored()
    {
        var f = Fixture();
        var optimizer = Create("unfactored");
        var parameters = new Vector<double>(Doubles(f["initial"]!));
        var gradients = f["gradients"]!.Select(g => Doubles(g)).ToArray();
        for (int k = 0; k < gradients.Length; k++)
        {
            parameters = optimizer.UpdateParameters(parameters, new Vector<double>(gradients[k]));
            Close(Doubles(f["unfactored"]![k]!), parameters.ToArray(), $"unfactored step {k + 1}");
        }
    }
}
