using AiDotNet.ActivationFunctions;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.LossFunctions;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNetTestConsole;

/// <summary>
/// Minimal repro for the ACEStep eager-tape NaN (see ACEStepNanDiag): the
/// full model's ComputeGradients returns NaN for exactly the LAST
/// decoder-block subgraph (MHA -> LN -> Dense -> Dense -> LN -> Dense head)
/// with sibling tensors at exact zero, while everything upstream is clean.
/// This probe rebuilds that subgraph at toy dims over the same rank-3
/// [batch=1, seq=2, features] input and prints per-layer gradient health,
/// at seq lengths 2 and 4, to pin the failing op.
/// </summary>
internal static class AceTailMinRepro
{
    public static void Run()
    {
        foreach (int seq in new[] { 2, 4 })
        {
            Console.WriteLine($"==== seq={seq} ====");
            RunCase(seq);
        }
    }

    private static void RunCase(int seq)
    {
        const int feat = 16;
        const int dim = 8;
        var layers = new List<ILayer<double>>
        {
            new DenseLayer<double>(dim, (IActivationFunction<double>)new GELUActivation<double>()),
            new LayerNormalizationLayer<double>(),
            new MultiHeadAttentionLayer<double>(2, dim / 2),
            new LayerNormalizationLayer<double>(),
            new DenseLayer<double>(dim * 4, (IActivationFunction<double>)new GELUActivation<double>()),
            new DenseLayer<double>(dim, (IActivationFunction<double>)new IdentityActivation<double>()),
            new LayerNormalizationLayer<double>(),
            new DenseLayer<double>(4, (IActivationFunction<double>?)null),
        };

        var architecture = new NeuralNetworkArchitecture<double>(
            inputType: InputType.OneDimensional,
            taskType: NeuralNetworkTaskType.Regression,
            inputSize: feat,
            outputSize: 4,
            layers: layers);
        var net = new NeuralNetwork<double>(architecture, lossFunction: new MeanSquaredErrorLoss<double>());

        var rng = new Random(7);
        var input = new Tensor<double>([1, seq, feat]);
        for (int i = 0; i < input.Length; i++) input[i] = rng.NextDouble() * 2 - 1;

        var warm = net.Predict(input);
        Console.WriteLine($"output shape [{string.Join(",", DimsOf(warm))}], nonFinite={CountNonFinite(warm)}/{warm.Length}");

        var targetDims = DimsOf(warm);
        var target = new Tensor<double>(targetDims);
        for (int i = 0; i < target.Length; i++) target[i] = rng.NextDouble() * 2 - 1;

        var grads = net.ComputeGradients(input, target);
        int offset = 0, layerIdx = 0;
        foreach (var layer in net.Layers)
        {
            int count = checked((int)layer.ParameterCount);
            int bad = 0, zero = 0;
            double maxAbs = 0;
            for (int i = offset; i < offset + count && i < grads.Length; i++)
            {
                double g = grads[i];
                if (double.IsNaN(g) || double.IsInfinity(g)) bad++;
                else if (g == 0) zero++;
                else { double a = Math.Abs(g); if (a > maxAbs) maxAbs = a; }
            }
            Console.WriteLine($"layer {layerIdx} ({layer.GetType().Name}): params={count}, " +
                              $"nonFinite={bad}, exactZero={zero}, max|g|={maxAbs:G4}");
            offset += count;
            layerIdx++;
        }
    }

    private static int[] DimsOf(Tensor<double> t)
    {
        var dims = new int[t.Rank];
        for (int i = 0; i < t.Rank; i++) dims[i] = t.Shape[i];
        return dims;
    }

    private static int CountNonFinite(Tensor<double> t)
    {
        int bad = 0;
        for (int i = 0; i < t.Length; i++)
            if (double.IsNaN(t[i]) || double.IsInfinity(t[i])) bad++;
        return bad;
    }
}
