using System;
using System.Linq;
using System.Text;
using AiDotNet.ComputerVision.Segmentation.Efficient;
using AiDotNet.Enums;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tensors;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;
using Xunit.Abstractions;

namespace AiDotNet.Tests.Diag;

// DIAG (diag/efficientsam-intel): at the 32x32 fixture EfficientSAM's third stage normalizes 8 BatchNorms over
// 2x2 = 4 values per channel. Is its tape gradient a descent direction at small enough steps, and does the
// usable step grow with the input size (more values per channel)?
public class EfficientSAMDescentDiag
{
    private readonly ITestOutputHelper _output;
    public EfficientSAMDescentDiag(ITestOutputHelper output) => _output = output;

    [Theory]
    [InlineData(32)]
    [InlineData(64)]
    [InlineData(128)]
    public void TapeGradient_DescendsAtSmallSteps(int size)
    {
        var rng = new Random(11);
        var arch = new NeuralNetworkArchitecture<double>(
            inputType: InputType.ThreeDimensional, taskType: NeuralNetworkTaskType.ImageSegmentation,
            inputHeight: size, inputWidth: size, inputDepth: 3);
        using var net = new EfficientSAM<double>(arch);
        var input = new Tensor<double>(new[] { 1, 3, size, size });
        for (int i = 0; i < input.Length; i++) input[i] = rng.NextDouble() * 2 - 1;
        net.SetTrainingMode(true);
        var shapeProbe = net.ForwardForTraining(input);
        var target = new Tensor<double>(shapeProbe.Shape.ToArray());
        for (int i = 0; i < target.Length; i++) target[i] = rng.NextDouble() < 0.5 ? 0 : 1;

        double l0 = net.EvaluateTrainingObjective(input, target);
        var g = net.ComputeGradients(input, target);
        var theta = net.GetParameters();
        double gg = 0; for (int k = 0; k < g.Length; k++) gg += g[k] * g[k];
        double gNorm = Math.Sqrt(gg);
        var sb = new StringBuilder($"DESCENT size={size} out=[{string.Join(",", shapeProbe.Shape.ToArray())}] l0={l0:G9} gNorm={gNorm:G6}");
        foreach (double stepLen in new[] { 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3 })
        {
            double alpha = stepLen / gNorm;
            var moved = new Vector<double>(theta.Length);
            for (int k = 0; k < theta.Length; k++) moved[k] = theta[k] - alpha * g[k];
            net.SetParameters(moved);
            double l = net.EvaluateTrainingObjective(input, target);
            sb.Append($" | step={stepLen:G1} dL={l - l0:G4} pred={-alpha * gg:G4} ratio={(l - l0) / (-alpha * gg):G4}");
        }
        net.SetParameters(theta);
        _output.WriteLine(sb.ToString());
        var path = Environment.GetEnvironmentVariable("AIDOTNET_DIAG_LOSS_TRACE");
        if (path is not null) System.IO.File.AppendAllText(path + ".descent", sb + Environment.NewLine);
    }
}
