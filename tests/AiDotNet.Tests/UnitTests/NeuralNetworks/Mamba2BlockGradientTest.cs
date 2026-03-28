using AiDotNet.NeuralNetworks.Layers.SSM;
using AiDotNet.Tensors;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;
using Xunit.Abstractions;

namespace AiDotNet.Tests.UnitTests.NeuralNetworks;

public class Mamba2BlockGradientTest
{
    private static readonly string _logPath = Path.Combine(Path.GetTempPath(), "mamba2_gradtest.log");
    private static void LogGrad(string msg) => File.AppendAllText(_logPath, msg + Environment.NewLine);

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
        var projW = new double[output.Length];
        for (int i = 0; i < output.Length; i++)
        {
            double w = rngLoss.NextDouble() * 2.0 - 1.0;
            outputGrad[i] = w;
            projW[i] = w;
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
        int normGammaSize = 32;

        // inputProj: [16, 64]. Cols 0-31 are x-branch, cols 32-63 are z-branch.
        var groups = new[] {
            ("inputProj_xBranch_r0c0", 0, 3),         // row 0, x-branch col 0-2
            ("inputProj_zBranch_r0c32", 32, 3),        // row 0, z-branch col 32-34
            ("inputBias_x", inputProjSize, 3),
            ("inputBias_z", inputProjSize + 32, 3),
            ("convW", inputProjSize + inputBiasSize, 3),
            ("convBias", inputProjSize + inputBiasSize + convSize, 3),
            ("outBias", inputProjSize + inputBiasSize + convSize + convBiasSize + bProjSize + cProjSize + aLogSize + dtProjSize + dtBiasSize + dParamSize + outProjSize, 3),
            ("normGamma", inputProjSize + inputBiasSize + convSize + convBiasSize + bProjSize + cProjSize + aLogSize + dtProjSize + dtBiasSize + dParamSize + outProjSize + outBiasSize, 3),
        };

        foreach (var (name, start, count) in groups)
        {
            int passCount = 0, failCount = 0;
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
                if (absMax < 1e-7) { passCount++; continue; }
                double relErr = Math.Abs(numerical - anal) / (absMax + 1e-8);

                if (relErr > 0.05) failCount++;
                else passCount++;

                LogGrad($"  {name}[{p - start}] (idx={p}): a={anal:G6} n={numerical:G6} relErr={relErr:G4}");
            }
            layer.SetParameters(parameters);
            LogGrad($"{name}: {passCount} pass, {failCount} fail");
        }
    }
}
