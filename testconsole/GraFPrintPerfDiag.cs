using System.Diagnostics;
using AiDotNet.Audio.Fingerprinting;
using AiDotNet.Enums;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNetTestConsole;

public static class GraFPrintPerfDiag
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

        // Hook System.Diagnostics.Trace to print to console so we can see
        // why the fused training path fails (TryStepWithFusedOptimizer
        // logs the exception via Trace.TraceWarning before returning false).
        var listener = new TextWriterTraceListener(Console.Out);
        Trace.Listeners.Add(listener);
        Trace.AutoFlush = true;

        // Invalidate any cached fused-plan state so this run starts fresh.
        AiDotNet.Training.CompiledTapeTrainingStep<double>.Invalidate();

        using var arena = AiDotNet.Tensors.Helpers.TensorArena.Create();
        // Test with NO DROPOUT to see if dropout is blocking fused path.
        var opts = new GraFPrintOptions { DropoutRate = 0.0 };
        dynamic model = ctor!.Invoke(new object?[] { Arch(), opts, null });

        var input = new Tensor<double>(new[] { 1, 64, 32 });
        var rng = new Random(42);
        for (int i = 0; i < input.Length; i++) input[i] = rng.NextDouble();

        // Determine target shape from one Predict (post-squeeze rank-2 [1, 4])
        Tensor<double> warmupOut = model.Predict(input);
        Console.WriteLine($"Predict output shape: [{string.Join(",", warmupOut.Shape)}] length={warmupOut.Length}");

        var shapeArr = new int[warmupOut.Shape.Length];
        for (int i = 0; i < warmupOut.Shape.Length; i++) shapeArr[i] = warmupOut.Shape[i];
        var target = new Tensor<double>(shapeArr);
        for (int i = 0; i < target.Length; i++) target[i] = 0.5;

        // Check if fused path is engaged. _fusedTrainingCommitted flips
        // to true on the first successful fused Train step.
        var commitField = typeof(NeuralNetworkBase<double>).GetField(
            "_fusedTrainingCommitted",
            System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance);
        var disabledField = typeof(NeuralNetworkBase<double>).GetField(
            "_fusedTrainingDisabled",
            System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance);

        Console.WriteLine($"Before first Train: fusedCommitted={commitField?.GetValue(model)}, fusedDisabled={disabledField?.GetValue(model)}");
        Console.WriteLine($"EnableCompilation: {AiDotNet.Tensors.Engines.Optimization.TensorCodecOptions.Current.EnableCompilation}");

        // Reset the global fused step counter, run 1 Train, see if it engaged.
        AiDotNet.Training.CompiledTapeTrainingStep<double>.ResetFusedStepCount();
        try { model.Train(input, target); }
        catch (Exception ex) { Console.WriteLine($"Train threw: {ex.Message}"); }
        Console.WriteLine($"After  first Train: fusedCommitted={commitField?.GetValue(model)}, fusedDisabled={disabledField?.GetValue(model)}");
        Console.WriteLine($"Global fused step count after Train 1: {AiDotNet.Training.CompiledTapeTrainingStep<double>.GetFusedStepCount()}");

        // Warm 5 iters
        for (int i = 0; i < 5; i++) model.Train(input, target);

        // Time 10 iters
        var sw = Stopwatch.StartNew();
        for (int i = 0; i < 10; i++) model.Train(input, target);
        sw.Stop();
        double perIterMs = sw.Elapsed.TotalMilliseconds / 10.0;
        Console.WriteLine($"GraFPrint Train: {perIterMs:F1} ms/iter ⇒ {perIterMs * 250 / 1000.0:F1}s for 250 iters (120s budget)");

        // Check if parameters actually changed across training steps
        var p0 = model.GetParameters() as Vector<double>;
        double[] before = new double[Math.Min(10, p0!.Length)];
        for (int i = 0; i < before.Length; i++) before[i] = p0[i];
        for (int i = 0; i < 20; i++) model.Train(input, target);
        var p1 = model.GetParameters() as Vector<double>;
        double maxParamDelta = 0;
        for (int i = 0; i < before.Length; i++)
            maxParamDelta = Math.Max(maxParamDelta, Math.Abs(p1![i] - before[i]));
        Console.WriteLine($"After 20 fused Train: first-10-params max |Δ|={maxParamDelta:E6}");

        // Time Predict only
        var sw2 = Stopwatch.StartNew();
        for (int i = 0; i < 10; i++) model.Predict(input);
        sw2.Stop();
        double predictPerIter = sw2.Elapsed.TotalMilliseconds / 10.0;
        Console.WriteLine($"GraFPrint Predict: {predictPerIter:F1} ms/iter");
    }
}
