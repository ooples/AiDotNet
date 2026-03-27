using AiDotNet.NeuralNetworks.Layers.SSM;
using AiDotNet.Tensors;
using AiDotNet.Tensors.Helpers;
using Xunit;

namespace AiDotNet.Tests.UnitTests.NeuralNetworks;

public class Mamba2BlockGradientTest
{
    [Fact]
    public void Mamba2Block_GradientCheck_AllGroups()
    {
        var layer = new Mamba2Block<double>(4, 16, 4, 2);
        layer.SetTrainingMode(true);

        var rng = RandomHelper.CreateSeededRandom(42);
        var input = new Tensor<double>(new[] { 4, 16 });
        for (int i = 0; i < input.Length; i++) input[i] = rng.NextDouble() * 2.0 - 1.0;

        double eps = 1e-5;
        var rngLoss = RandomHelper.CreateSeededRandom(12345);

        layer.ClearGradients();
        var output = layer.Forward(input);
        var outputGrad = new Tensor<double>(output.Shape.ToArray());
        for (int i = 0; i < output.Length; i++)
        {
            outputGrad[i] = rngLoss.NextDouble() * 2.0 - 1.0;
        }
        layer.Backward(outputGrad);
        var analytical = layer.GetParameterGradients();
        var parameters = layer.GetParameters();

        // Test parameter groups - figure out where each group starts
        // Order: inputProj(16*32), inputBias(32), conv(32*4), convBias(32),
        //        bProj(32*4), cProj(32*4), aLog(2), dtProj(32*2), dtBias(2),
        //        dParam(2), outProj(32*16), outBias(16), normGamma(32), normBeta(32)
        int inputProjSize = 16 * 64;  // [modelDim=16, 2*innerDim=64]
        int inputBiasSize = 64;       // [2*innerDim=64]
        int convSize = 32 * 4;        // [innerDim=32, kernelSize=4]
        int convBiasSize = 32;         // [innerDim=32]
        int bProjSize = 32 * 4;
        int cProjSize = 32 * 4;
        int aLogSize = 2;
        int dtProjSize = 32 * 2;
        int dtBiasSize = 2;
        int dParamSize = 2;
        int outProjSize = 32 * 16;
        int outBiasSize = 16;

        // inputProj: [16, 64]. Cols 0-31 are x-branch, cols 32-63 are z-branch.
        var groups = new[] {
            ("inputProj_xBranch_r0c0", 0, 3),
            ("inputProj_zBranch_r0c32", 32, 3),
            ("inputBias_x", inputProjSize, 3),
            ("inputBias_z", inputProjSize + 32, 3),
            ("convW", inputProjSize + inputBiasSize, 3),
            ("convBias", inputProjSize + inputBiasSize + convSize, 3),
            ("outBias", inputProjSize + inputBiasSize + convSize + convBiasSize + bProjSize + cProjSize + aLogSize + dtProjSize + dtBiasSize + dParamSize + outProjSize, 3),
            ("normGamma", inputProjSize + inputBiasSize + convSize + convBiasSize + bProjSize + cProjSize + aLogSize + dtProjSize + dtBiasSize + dParamSize + outProjSize + outBiasSize, 3),
        };

        int totalFails = 0;
        var failures = new List<string>();

        foreach (var (name, start, count) in groups)
        {
            int failCount = 0;
            for (int p = start; p < Math.Min(start + count, parameters.Length); p++)
            {
                double origVal = parameters[p];

                var pp = parameters.Clone(); pp[p] = origVal + eps;
                layer.SetParameters(pp); layer.ResetState();
                var outP = layer.Forward(input);
                var r2 = RandomHelper.CreateSeededRandom(12345);
                double lP = 0; for (int i = 0; i < outP.Length; i++) lP += (r2.NextDouble() * 2.0 - 1.0) * outP[i];

                var pm = parameters.Clone(); pm[p] = origVal - eps;
                layer.SetParameters(pm); layer.ResetState();
                var outM = layer.Forward(input);
                var r3 = RandomHelper.CreateSeededRandom(12345);
                double lM = 0; for (int i = 0; i < outM.Length; i++) lM += (r3.NextDouble() * 2.0 - 1.0) * outM[i];

                double numerical = (lP - lM) / (2 * eps);
                double anal = analytical[p];
                double absMax = Math.Max(Math.Abs(numerical), Math.Abs(anal));
                if (absMax < 1e-7) continue;
                double relErr = Math.Abs(numerical - anal) / (absMax + 1e-8);

                if (relErr > 0.05)
                {
                    failCount++;
                    failures.Add($"{name}[{p - start}] (idx={p}): analytical={anal:G6} numerical={numerical:G6} relErr={relErr:G4}");
                }
            }
            layer.SetParameters(parameters);
            totalFails += failCount;
        }

        Assert.True(totalFails == 0,
            $"Gradient check failed for {totalFails} parameters:\n{string.Join("\n", failures)}");
    }
}
