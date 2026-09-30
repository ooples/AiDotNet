using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using AiDotNet.ActivationFunctions;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.LossFunctions;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;
using Xunit.Abstractions;

namespace AiDotNet.Tests.Diag;

// DIAG (diag/efficientsam-intel): does the training-mode BatchNorm gradient match finite differences for a
// single-sample convolutional activation? EfficientSAM's tape gradient is not a descent direction of its own
// training objective; this isolates whether batch-statistics BatchNorm is the cause.
public class BatchNormTrainingGradientDiag
{
    private readonly ITestOutputHelper _output;
    public BatchNormTrainingGradientDiag(ITestOutputHelper output) => _output = output;

    private static NeuralNetwork<double> Build(bool withBatchNorm, bool bnFirst)
    {
        IActivationFunction<double> relu = new ReLUActivation<double>();
        IActivationFunction<double> identity = new IdentityActivation<double>();
        var layers = new List<ILayer<double>>();
        if (bnFirst) layers.Add(new BatchNormalizationLayer<double>());
        layers.Add(new ConvolutionalLayer<double>(6, 3, 1, 1, relu));
        if (withBatchNorm) layers.Add(new BatchNormalizationLayer<double>());
        layers.Add(new ConvolutionalLayer<double>(4, 1, 1, 0, identity));
        var arch = new NeuralNetworkArchitecture<double>(
            inputType: InputType.ThreeDimensional, taskType: NeuralNetworkTaskType.Regression,
            inputHeight: 8, inputWidth: 8, inputDepth: 3, outputSize: 4 * 8 * 8, layers: layers);
        return new NeuralNetwork<double>(arch, lossFunction: new MeanSquaredErrorLoss<double>());
    }

    [Theory]
    [InlineData("conv-bn-conv b1", true, false, 1)]
    [InlineData("conv-conv b1 (no bn)", false, false, 1)]
    [InlineData("conv-bn-conv b2", true, false, 2)]
    [InlineData("bn-conv-conv b1", false, true, 1)]
    public void TrainingModeGradient_MatchesFiniteDifference(string label, bool withBn, bool bnFirst, int batch)
    {
        var rng = new Random(7);
        using var net = Build(withBn, bnFirst);
        var input = new Tensor<double>(new[] { batch, 3, 8, 8 });
        for (int i = 0; i < input.Length; i++) input[i] = rng.NextDouble() * 2 - 1;
        var target = new Tensor<double>(new[] { batch, 4, 8, 8 });
        for (int i = 0; i < target.Length; i++) target[i] = rng.NextDouble() * 2 - 1;
        var loss = new MeanSquaredErrorLoss<double>();

        net.SetTrainingMode(true);
        _ = net.EvaluateTrainingObjective(input, target, loss); // materialize lazy layers
        var analytical = net.ComputeGradients(input, target, loss);
        var theta = net.GetParameters();
        var sb = new StringBuilder($"GRADFD {label}: params={theta.Length} grads={analytical.Length}");
        double worstRel = 0; int worstIdx = -1; double worstA = 0, worstN = 0; double sumA2 = 0, sumN2 = 0, dot = 0;
        const double h = 1e-5;
        for (int i = 0; i < theta.Length; i++)
        {
            var plus = theta.Clone(); plus[i] += h; net.SetParameters(plus);
            double lp = net.EvaluateTrainingObjective(input, target, loss);
            var minus = theta.Clone(); minus[i] -= h; net.SetParameters(minus);
            double lm = net.EvaluateTrainingObjective(input, target, loss);
            double numeric = (lp - lm) / (2 * h);
            double a = i < analytical.Length ? analytical[i] : double.NaN;
            sumA2 += a * a; sumN2 += numeric * numeric; dot += a * numeric;
            double rel = Math.Abs(a - numeric) / Math.Max(1e-8, Math.Abs(a) + Math.Abs(numeric));
            if (rel > worstRel) { worstRel = rel; worstIdx = i; worstA = a; worstN = numeric; }
        }
        net.SetParameters(theta);
        double cosine = dot / Math.Max(1e-300, Math.Sqrt(sumA2) * Math.Sqrt(sumN2));
        sb.Append($" |a|={Math.Sqrt(sumA2):G6} |fd|={Math.Sqrt(sumN2):G6} cos={cosine:G6} worstRel={worstRel:G4} at {worstIdx} (a={worstA:G6} fd={worstN:G6})");
        // Per-chunk cosine, to name the layer whose gradient disagrees.
        int offset = 0;
        foreach (var chunk in net.GetParameterStateChunks())
        {
            int len = chunk.Tensor.Length; double cd = 0, ca = 0, cn = 0;
            for (int i = offset; i < offset + len && i < theta.Length; i++)
            {
                var plus = theta.Clone(); plus[i] += h; net.SetParameters(plus);
                double lp = net.EvaluateTrainingObjective(input, target, loss);
                var minus = theta.Clone(); minus[i] -= h; net.SetParameters(minus);
                double lm = net.EvaluateTrainingObjective(input, target, loss);
                double numeric = (lp - lm) / (2 * h);
                double a = analytical[i];
                cd += a * numeric; ca += a * a; cn += numeric * numeric;
            }
            sb.Append($" | {chunk.StableId}[{offset}..{offset + len}) cos={cd / Math.Max(1e-300, Math.Sqrt(ca * cn)):G4} |a|={Math.Sqrt(ca):G4} |fd|={Math.Sqrt(cn):G4}");
            offset += len;
        }
        net.SetParameters(theta);
        _output.WriteLine(sb.ToString());
        var path = Environment.GetEnvironmentVariable("AIDOTNET_DIAG_LOSS_TRACE");
        if (path is not null) System.IO.File.AppendAllText(path + ".gradfd", sb + Environment.NewLine);
        Assert.True(cosine > 0.999, sb.ToString());
    }
}
