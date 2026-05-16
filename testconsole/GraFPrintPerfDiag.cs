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
        // Match the test's default DropoutRate=0.1 path to see whether
        // dropout's presence is what's making the test "pass" (by falling
        // back to eager, which propagates params), while no-dropout fused
        // silently no-ops the parameter writes.
        var opts = new GraFPrintOptions(); // defaults: DropoutRate=0.1
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

        // Per-layer-type profiling
        Console.WriteLine("\n=== Per-layer profiling ===");
        int convCount = 0, bnCount = 0, actCount = 0, dropCount = 0, poolCount = 0;
        foreach (var l in model.Layers)
        {
            var name = ((object)l).GetType().Name;
            if (name.StartsWith("ConvolutionalLayer")) convCount++;
            else if (name.StartsWith("BatchNormalizationLayer")) bnCount++;
            else if (name.StartsWith("ActivationLayer")) actCount++;
            else if (name.StartsWith("DropoutLayer")) dropCount++;
            else if (name.StartsWith("GlobalPoolingLayer")) poolCount++;
        }
        Console.WriteLine($"Layer counts: Conv={convCount}, BN={bnCount}, Activation={actCount}, Dropout={dropCount}, GlobalPool={poolCount}");

        // Verify the fused path actually updates the layer-owned parameters.
        // We check BOTH paths because they read from different sources:
        //  * GetParameters() walks Layers, calls layer.GetParameters() (which
        //    in ConvolutionalLayer reads from its private _kernels / _biases
        //    fields), then COPIES into a flat Vector<T>.
        //  * GetParameterChunks() walks CollectTrainableLayers recursively
        //    and yields each layer's GetTrainableParameters() = _registeredTensors
        //    by reference (zero-copy).
        // If the fused path replaces _registeredTensors with buffer views via
        // SetTrainableParameters but leaves _kernels / _biases pointing at the
        // pre-fuse tensors, the chunks path will see updates while the flat
        // GetParameters() path won't. That's the divergence the test suite
        // catches as a coherence bug (Training_ShouldChangeParameters reads
        // chunks; SimilarInputs / Predict-driven tests read whatever the
        // layer's Forward path consumes).
        Console.WriteLine("\n=== Fused param propagation check ===");

        // (a) Flat GetParameters() snapshot — reads from layer-private fields.
        var pBefore = model.GetParameters() as Vector<double>;
        double[] before = new double[Math.Min(20, pBefore!.Length)];
        for (int i = 0; i < before.Length; i++) before[i] = pBefore[i];

        // (b) Chunks snapshot — sample first chunk's first 1024 values, like
        // the actual Training_ShouldChangeParameters test does.
        var chunksBefore = new List<double[]>();
        foreach (Tensor<double> chunk in model.GetParameterChunks())
        {
            if (chunksBefore.Count >= 4) break;
            int n = Math.Min(chunk.Length, 1024);
            var arr = new double[n];
            for (int j = 0; j < n; j++) arr[j] = chunk[j];
            chunksBefore.Add(arr);
        }

        for (int i = 0; i < 20; i++) model.Train(input, target);

        var pAfter = model.GetParameters() as Vector<double>;
        double maxFlatDelta = 0;
        int firstChanged = -1;
        for (int i = 0; i < pBefore.Length; i++)
        {
            double d = Math.Abs(pAfter![i] - pBefore[i]);
            if (d > maxFlatDelta) { maxFlatDelta = d; firstChanged = i; }
        }
        Console.WriteLine($"[flat GetParameters()]   After 20 Train: max |Δ|={maxFlatDelta:E6} at index {firstChanged}");
        Console.WriteLine($"  First 5 BEFORE: {string.Join(",", Enumerable.Range(0, 5).Select(i => pBefore[i].ToString("F8")))}");
        Console.WriteLine($"  First 5 AFTER:  {string.Join(",", Enumerable.Range(0, 5).Select(i => pAfter![i].ToString("F8")))}");

        // Compare against chunks (matches test's Training_ShouldChangeParameters).
        double maxChunkDelta = 0;
        int chunkIdx = 0;
        foreach (Tensor<double> chunk in model.GetParameterChunks())
        {
            if (chunkIdx >= chunksBefore.Count) break;
            var prev = chunksBefore[chunkIdx];
            int n = Math.Min(prev.Length, chunk.Length);
            for (int j = 0; j < n; j++)
            {
                double d = Math.Abs(prev[j] - chunk[j]);
                if (d > maxChunkDelta) maxChunkDelta = d;
            }
            chunkIdx++;
        }
        Console.WriteLine($"[GetParameterChunks()]  After 20 Train: max |Δ|={maxChunkDelta:E6} across {chunkIdx} chunks");
        if (maxFlatDelta == 0 && maxChunkDelta > 0)
        {
            Console.WriteLine("  *** DIVERGENCE: flat=0, chunks>0. Fused writes to _registeredTensors but layer-private fields stale. ***");
        }

        // Time Predict only
        var sw2 = Stopwatch.StartNew();
        for (int i = 0; i < 10; i++) model.Predict(input);
        sw2.Stop();
        double predictPerIter = sw2.Elapsed.TotalMilliseconds / 10.0;
        Console.WriteLine($"GraFPrint Predict: {predictPerIter:F1} ms/iter");
    }
}
