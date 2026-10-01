using System;
using System.Linq;
using System.Text;
using AiDotNet.ComputerVision.Segmentation.Foundation;
using AiDotNet.Enums;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors;
using Xunit;
using Xunit.Abstractions;

namespace AiDotNet.Tests.Diag;

// DIAG: SAMTests.MoreData fails with a NaN loss on the UNTRAINED model right after its BatchNorm statistics are
// recalibrated. Where does the NaN first appear: the recalibrated statistics, or a layer's eval output?
public class SAMRecalibrationDiag
{
    private readonly ITestOutputHelper _output;
    public SAMRecalibrationDiag(ITestOutputHelper output) => _output = output;

    private static string Stats(Tensor<double> t)
    {
        var a = t.ToArray();
        int nan = a.Count(double.IsNaN), inf = a.Count(double.IsInfinity), neg = a.Count(v => v < 0);
        var finite = a.Where(v => !double.IsNaN(v) && !double.IsInfinity(v)).ToArray();
        string range = finite.Length > 0 ? $"min={finite.Min():G4} max={finite.Max():G4}" : "no finite";
        return $"n={a.Length} nan={nan} inf={inf} neg={neg} {range}";
    }

    [Fact]
    public void Recalibration_FirstNaN()
    {
        var arch = new NeuralNetworkArchitecture<double>(
            inputType: InputType.ThreeDimensional, taskType: NeuralNetworkTaskType.Regression,
            inputHeight: 112, inputWidth: 112, inputDepth: 3, outputSize: 1);
        using var net = new SAM<double>(arch, options: new SAMOptions
        { NumClasses = 1, ModelSize = SAMModelSize.ViTBase, DropRate = 0.0, LearningRate = 1e-5 });
        var rng = new Random(5);
        var input = new Tensor<double>(new[] { 1, 3, 112, 112 });
        for (int i = 0; i < input.Length; i++) input[i] = rng.NextDouble();
        var sb = new StringBuilder();
        sb.AppendLine($"SAMDIAG predict-before: {Stats(net.Predict(input))}");

        var bns = net.Layers.OfType<BatchNormalizationLayer<double>>().ToList();
        net.SetTrainingMode(false);
        foreach (var bn in bns) bn.OverwriteRunningStatistics = true;
        var recal = net.Predict(input);
        foreach (var bn in bns) bn.OverwriteRunningStatistics = false;
        sb.AppendLine($"SAMDIAG recalibration-pass output: {Stats(recal)} bnCount={bns.Count}");
        for (int i = 0; i < bns.Count; i++)
            sb.AppendLine($"SAMDIAG bn{i} mean[{Stats(bns[i].GetRunningMean())}] var[{Stats(bns[i].GetRunningVariance())}]");

        sb.AppendLine($"SAMDIAG predict-after: {Stats(net.Predict(input))}");
        var h = input;
        for (int li = 0; li < net.Layers.Count; li++)
        {
            try
            {
                h = net.Layers[li].Forward(h);
                sb.AppendLine($"SAMDIAG L{li:D2} {net.Layers[li].GetType().Name} shape=[{string.Join(",", h.Shape.ToArray())}] {Stats(h)}");
            }
            catch (Exception ex) { sb.AppendLine($"SAMDIAG L{li:D2} CHAIN-BREAK {ex.GetType().Name}: {ex.Message}"); break; }
        }
        _output.WriteLine(sb.ToString());
        var path = Environment.GetEnvironmentVariable("AIDOTNET_DIAG_LOSS_TRACE");
        if (path is not null) System.IO.File.AppendAllText(path + ".sam", sb.ToString());
    }
}
