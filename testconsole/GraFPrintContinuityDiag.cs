using AiDotNet.Audio.Fingerprinting;
using AiDotNet.Enums;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNetTestConsole;

public static class GraFPrintContinuityDiag
{
    public static void Run()
    {
        var t = typeof(GraFPrint<>).MakeGenericType(typeof(double));
        var ctor = t.GetConstructor(new[] {
            typeof(NeuralNetworkArchitecture<double>),
            typeof(GraFPrintOptions),
            typeof(AiDotNet.Interfaces.IGradientBasedOptimizer<double, Tensor<double>, Tensor<double>>)
        });

        NeuralNetworkArchitecture<double> Arch() => new(
            inputType: InputType.TwoDimensional,
            taskType: NeuralNetworkTaskType.Regression,
            inputHeight: 64, inputWidth: 32, inputDepth: 1, outputSize: 4);

        // Match the test's TensorArena scope — some tensors get arena-pooled
        // when an arena is active, which could affect determinism / state.
        using var arena = AiDotNet.Tensors.Helpers.TensorArena.Create();

        // Match the test's CreateSeededRandom path so we exercise the same seed.
        var rng = AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(42);
        dynamic model = ctor!.Invoke(new object?[] { Arch(), null, null });
        model.SetTrainingMode(false);

        var input1 = new Tensor<double>(new[] { 1, 64, 32 });
        for (int i = 0; i < input1.Length; i++) input1[i] = rng.NextDouble();

        var input2 = new Tensor<double>(new[] { 1, 64, 32 });
        for (int i = 0; i < input2.Length; i++) input2[i] = input1[i] + 1e-6;

        Tensor<double> emb1 = model.Predict(input1);
        Tensor<double> emb2 = model.Predict(input2);

        double dot = 0, norm1 = 0, norm2 = 0;
        int n = Math.Min(emb1.Length, emb2.Length);
        for (int i = 0; i < n; i++)
        {
            dot += emb1[i] * emb2[i];
            norm1 += emb1[i] * emb1[i];
            norm2 += emb2[i] * emb2[i];
        }
        double cos = (norm1 > 1e-15 && norm2 > 1e-15) ? dot / (Math.Sqrt(norm1) * Math.Sqrt(norm2)) : 0;
        Console.WriteLine($"Cosine sim = {cos:F4}");
        Console.WriteLine($"||emb1|| = {Math.Sqrt(norm1):F6}, ||emb2|| = {Math.Sqrt(norm2):F6}");
        Console.WriteLine($"emb1: {string.Join(",", Enumerable.Range(0, n).Select(i => emb1[i].ToString("F6")))}");
        Console.WriteLine($"emb2: {string.Join(",", Enumerable.Range(0, n).Select(i => emb2[i].ToString("F6")))}");
        double maxAbsDiff = 0;
        for (int i = 0; i < n; i++)
            maxAbsDiff = Math.Max(maxAbsDiff, Math.Abs(emb1[i] - emb2[i]));
        Console.WriteLine($"Max |Δ embedding| = {maxAbsDiff:E6}");
        double inputMaxDiff = 1e-6; // by construction
        Console.WriteLine($"Amplification factor = {maxAbsDiff / inputMaxDiff:E3}");
    }
}
