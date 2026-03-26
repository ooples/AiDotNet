using AiDotNet.NeuralNetworks.Layers.SSM;
using AiDotNet.Tensors;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.NeuralNetworks;

/// <summary>
/// Targeted gradient check for MambaBlock to isolate which sublayer gradient is wrong.
/// Tests parameters from different positions in the parameter vector.
/// </summary>
public class MambaBlockGradientTest
{
    [Theory]
    [InlineData(0, "inputProjectionWeight[0]")]   // First input projection weight
    [InlineData(5, "inputProjectionWeight[5]")]   // Another input projection weight
    public void MambaBlock_GradientCheck_ByParameter(int paramIdx, string paramName)
    {
        var layer = new MambaBlock<double>(4, 16, 4);
        layer.SetTrainingMode(true);

        var rng = RandomHelper.CreateSeededRandom(42);
        var input = new Tensor<double>(new[] { 4, 16 });
        for (int i = 0; i < input.Length; i++)
            input[i] = rng.NextDouble() * 2.0 - 1.0;

        double epsilon = 1e-5;

        // Random projection loss
        var rngLoss = RandomHelper.CreateSeededRandom(12345);

        // Forward + backward for analytical gradient
        layer.ClearGradients();
        var output = layer.Forward(input);
        var outputGrad = new Tensor<double>(output.Shape.ToArray());
        var projW = new double[output.Length];
        for (int i = 0; i < output.Length; i++)
        {
            double w = rngLoss.NextDouble() * 2.0 - 1.0;
            outputGrad[i] = w;
            projW[i] = w;
        }
        layer.Backward(outputGrad);
        var analyticalGradients = layer.GetParameterGradients();
        double analytical = analyticalGradients[paramIdx];

        // Numerical gradient
        var parameters = layer.GetParameters();

        var paramsPlus = parameters.Clone();
        paramsPlus[paramIdx] += epsilon;
        layer.SetParameters(paramsPlus);
        layer.ResetState();
        var outputPlus = layer.Forward(input);
        var rng2 = RandomHelper.CreateSeededRandom(12345);
        double lossPlus = 0;
        for (int i = 0; i < outputPlus.Length; i++)
            lossPlus += (rng2.NextDouble() * 2.0 - 1.0) * outputPlus[i];

        var paramsMinus = parameters.Clone();
        paramsMinus[paramIdx] -= epsilon;
        layer.SetParameters(paramsMinus);
        layer.ResetState();
        var outputMinus = layer.Forward(input);
        var rng3 = RandomHelper.CreateSeededRandom(12345);
        double lossMinus = 0;
        for (int i = 0; i < outputMinus.Length; i++)
            lossMinus += (rng3.NextDouble() * 2.0 - 1.0) * outputMinus[i];

        double numerical = (lossPlus - lossMinus) / (2.0 * epsilon);

        layer.SetParameters(parameters);

        double absMax = Math.Max(Math.Abs(numerical), Math.Abs(analytical));
        double relErr = absMax < 1e-7 ? 0 : Math.Abs(numerical - analytical) / (absMax + 1e-8);

        Assert.True(relErr <= 0.05,
            $"Gradient check failed for {paramName} (idx={paramIdx}): " +
            $"analytical={analytical:G6} numerical={numerical:G6} relErr={relErr:G4}");
    }

    [Fact]
    public void MambaBlock_OutputProjectionWeightGradient_IsCorrect()
    {
        // Test the LAST parameter group (output projection weights)
        // These should be correct since they're closest to the loss
        var layer = new MambaBlock<double>(4, 16, 4);
        layer.SetTrainingMode(true);

        var rng = RandomHelper.CreateSeededRandom(42);
        var input = new Tensor<double>(new[] { 4, 16 });
        for (int i = 0; i < input.Length; i++)
            input[i] = rng.NextDouble() * 2.0 - 1.0;

        double epsilon = 1e-5;

        // Get parameter count breakdown
        // Order: inputProj, inputBias, conv, convBias, xProj, dtProj, dtBias, aLog, dParam, outProj, outBias
        var parameters = layer.GetParameters();
        int totalParams = parameters.Length;

        // Test last few params (output bias — simplest gradient)
        int outBiasStart = totalParams - 16; // last 16 = output bias

        layer.ClearGradients();
        var output = layer.Forward(input);
        var outputGrad = new Tensor<double>(output.Shape.ToArray());
        var rngLoss = RandomHelper.CreateSeededRandom(12345);
        var projW = new double[output.Length];
        for (int i = 0; i < output.Length; i++)
        {
            double w = rngLoss.NextDouble() * 2.0 - 1.0;
            outputGrad[i] = w;
            projW[i] = w;
        }
        layer.Backward(outputGrad);
        var analyticalGradients = layer.GetParameterGradients();

        int failCount = 0;
        var errors = new System.Text.StringBuilder();
        for (int p = outBiasStart; p < Math.Min(outBiasStart + 5, totalParams); p++)
        {
            double analytical = analyticalGradients[p];

            var paramsPlus = parameters.Clone();
            paramsPlus[p] += epsilon;
            layer.SetParameters(paramsPlus);
            layer.ResetState();
            var outputPlus = layer.Forward(input);
            var rng2 = RandomHelper.CreateSeededRandom(12345);
            double lossPlus = 0;
            for (int i = 0; i < outputPlus.Length; i++)
                lossPlus += (rng2.NextDouble() * 2.0 - 1.0) * outputPlus[i];

            var paramsMinus = parameters.Clone();
            paramsMinus[p] -= epsilon;
            layer.SetParameters(paramsMinus);
            layer.ResetState();
            var outputMinus = layer.Forward(input);
            var rng3 = RandomHelper.CreateSeededRandom(12345);
            double lossMinus = 0;
            for (int i = 0; i < outputMinus.Length; i++)
                lossMinus += (rng3.NextDouble() * 2.0 - 1.0) * outputMinus[i];

            double numerical = (lossPlus - lossMinus) / (2.0 * epsilon);
            layer.SetParameters(parameters);

            double absMax = Math.Max(Math.Abs(numerical), Math.Abs(analytical));
            if (absMax < 1e-7) continue;
            double relErr = Math.Abs(numerical - analytical) / (absMax + 1e-8);
            if (relErr > 0.01)
            {
                failCount++;
                errors.Append($"outBias[{p - outBiasStart}](idx={p}): a={analytical:G6} n={numerical:G6} e={relErr:G4} | ");
            }
        }

        Assert.True(failCount == 0,
            $"Output bias gradient failed for {failCount} params. {errors}");
    }
}
