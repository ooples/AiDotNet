using AiDotNet.Audio.Fingerprinting;
using AiDotNet.Enums;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNetTestConsole;

public static class GraFPrintCloneDiag
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

        dynamic original = ctor!.Invoke(new object?[] { Arch(), null, null });
        original.SetTrainingMode(false);

        // Deterministic input
        var input = new Tensor<double>(new[] { 1, 64, 32 });
        for (int i = 0; i < input.Length; i++)
            input[i] = (i % 7) * 0.13 - 0.5;

        Tensor<double> originalOut = original.Predict(input);
        // Same model, same input, second call — should be deterministic in eval mode.
        Tensor<double> originalOut2 = original.Predict(input);
        double selfDiff = 0;
        for (int i = 0; i < originalOut.Length; i++)
            selfDiff = Math.Max(selfDiff, Math.Abs(originalOut[i] - originalOut2[i]));
        Console.WriteLine($"Self-determinism check: original→original max |Δ| = {selfDiff:E6}");
        Console.WriteLine($"  call 1: {string.Join(",", Enumerable.Range(0, originalOut.Length).Select(i => originalOut[i].ToString("F8")))}");
        Console.WriteLine($"  call 2: {string.Join(",", Enumerable.Range(0, originalOut2.Length).Select(i => originalOut2[i].ToString("F8")))}");

        // Print first few params of original
        var origParams = original.GetParameters() as Vector<double>;
        Console.WriteLine($"Original params total: {origParams!.Length}");
        Console.WriteLine($"Original params [0..9]: {string.Join(",", Enumerable.Range(0, Math.Min(10, origParams.Length)).Select(i => origParams[i].ToString("F8")))}");
        Console.WriteLine($"Original output: {string.Join(",", Enumerable.Range(0, originalOut.Length).Select(i => originalOut[i].ToString("F8")))}");

        // Inspect BN running stats on first BN layer (index 1 — stem BN)
        var bn0 = original.Layers[1] as BatchNormalizationLayer<double>;
        if (bn0 is not null)
        {
            var rm = bn0.GetRunningMean();
            var rv = bn0.GetRunningVariance();
            Console.WriteLine($"Original BN0 running mean [0..4]: {string.Join(",", Enumerable.Range(0, Math.Min(5, rm.Length)).Select(i => rm[i].ToString("F8")))}");
            Console.WriteLine($"Original BN0 running var  [0..4]: {string.Join(",", Enumerable.Range(0, Math.Min(5, rv.Length)).Select(i => rv[i].ToString("F8")))}");
        }

        // Clone
        dynamic cloned = original.Clone();
        cloned.SetTrainingMode(false);

        var clonedParams = cloned.GetParameters() as Vector<double>;
        Console.WriteLine($"Cloned   params total: {clonedParams!.Length}");
        Console.WriteLine($"Cloned   params [0..9]: {string.Join(",", Enumerable.Range(0, Math.Min(10, clonedParams.Length)).Select(i => clonedParams[i].ToString("F8")))}");

        // Compare params
        int firstDiffIdx = -1;
        double maxAbsDiff = 0;
        for (int i = 0; i < Math.Min(origParams.Length, clonedParams.Length); i++)
        {
            double d = Math.Abs(origParams[i] - clonedParams[i]);
            if (d > maxAbsDiff) { maxAbsDiff = d; firstDiffIdx = i; }
        }
        Console.WriteLine($"\nParam max |Δ| = {maxAbsDiff:E6} at index {firstDiffIdx}");

        var bn0c = cloned.Layers[1] as BatchNormalizationLayer<double>;
        if (bn0c is not null)
        {
            var rmc = bn0c.GetRunningMean();
            var rvc = bn0c.GetRunningVariance();
            Console.WriteLine($"Cloned   BN0 running mean [0..4]: {string.Join(",", Enumerable.Range(0, Math.Min(5, rmc.Length)).Select(i => rmc[i].ToString("F8")))}");
            Console.WriteLine($"Cloned   BN0 running var  [0..4]: {string.Join(",", Enumerable.Range(0, Math.Min(5, rvc.Length)).Select(i => rvc[i].ToString("F8")))}");
        }

        // Check cloned layer count + types match original
        Console.WriteLine($"\nOriginal layer count: {original.Layers.Count}, Cloned: {cloned.Layers.Count}");
        for (int i = 0; i < Math.Min(original.Layers.Count, cloned.Layers.Count); i++)
        {
            var ot = ((object)original.Layers[i]).GetType().Name;
            var ct = ((object)cloned.Layers[i]).GetType().Name;
            if (ot != ct)
                Console.WriteLine($"  [{i}] MISMATCH: original={ot}, cloned={ct}");
        }
        if (original.Layers.Count != cloned.Layers.Count)
            Console.WriteLine("LAYER COUNT MISMATCH!");

        // Check IsTrainingMode (field on LayerBase, protected)
        var modeField = typeof(LayerBase<double>).GetField("IsTrainingMode",
            System.Reflection.BindingFlags.Public | System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance);
        bool origAnyTraining = false, cloneAnyTraining = false;
        for (int i = 0; i < original.Layers.Count; i++)
        {
            bool om = (bool)modeField!.GetValue(original.Layers[i])!;
            bool cm = (bool)modeField!.GetValue(cloned.Layers[i])!;
            if (om) origAnyTraining = true;
            if (cm) cloneAnyTraining = true;
            if (om != cm)
                Console.WriteLine($"  Layer[{i}] training mode: original={om}, cloned={cm}");
        }
        Console.WriteLine($"Any layer in training mode? original={origAnyTraining}, cloned={cloneAnyTraining}");

        // Layer-by-layer trace: run forward on each pair of corresponding
        // layers from a freshly-promoted input and compare outputs.
        Console.WriteLine("\n=== Layer-by-layer divergence trace ===");
        var inputBatched = new Tensor<double>(new[] { 1, 1, 64, 32 });
        for (int i = 0; i < input.Length; i++) inputBatched[i] = input[i];

        Tensor<double> oCur = inputBatched, cCur = inputBatched;
        for (int li = 0; li < original.Layers.Count; li++)
        {
            var oLayer = original.Layers[li];
            var cLayer = cloned.Layers[li];
            Tensor<double> oOut = ((dynamic)oLayer).Forward(oCur);
            Tensor<double> cOut = ((dynamic)cLayer).Forward(cCur);
            double mx = 0;
            int ln = Math.Min(oOut.Length, cOut.Length);
            for (int k = 0; k < ln; k++)
                mx = Math.Max(mx, Math.Abs(oOut[k] - cOut[k]));
            if (mx > 1e-8)
            {
                string lt = ((object)oLayer).GetType().Name;
                Console.WriteLine($"  Layer[{li}] {lt}: max |Δ|={mx:E6} (o.shape=[{string.Join(",", oOut.Shape)}])");
                if (li < 5 || mx > 1.0) // print first samples
                {
                    Console.WriteLine($"    orig[0..3]: {string.Join(",", Enumerable.Range(0, Math.Min(3, oOut.Length)).Select(k => oOut[k].ToString("F8")))}");
                    Console.WriteLine($"    clon[0..3]: {string.Join(",", Enumerable.Range(0, Math.Min(3, cOut.Length)).Select(k => cOut[k].ToString("F8")))}");
                }
            }
            oCur = oOut;
            cCur = cOut;
        }

        Tensor<double> clonedOut = cloned.Predict(input);
        Tensor<double> clonedOut2 = cloned.Predict(input);
        double cloneSelfDiff = 0;
        for (int i = 0; i < clonedOut.Length; i++)
            cloneSelfDiff = Math.Max(cloneSelfDiff, Math.Abs(clonedOut[i] - clonedOut2[i]));
        Console.WriteLine($"\nCloned self-determinism: max |Δ| = {cloneSelfDiff:E6}");
        Console.WriteLine($"\nCloned   output: {string.Join(",", Enumerable.Range(0, clonedOut.Length).Select(i => clonedOut[i].ToString("F8")))}");

        // After Predict, sample weights and BN stats again to see if any
        // state changed during the forward.
        var origParamsPost = original.GetParameters() as Vector<double>;
        var clonedParamsPost = cloned.GetParameters() as Vector<double>;
        double paramDiffPostA = 0, paramDiffPostB = 0;
        for (int i = 0; i < origParams!.Length; i++)
        {
            paramDiffPostA = Math.Max(paramDiffPostA, Math.Abs(origParams[i] - origParamsPost![i]));
            paramDiffPostB = Math.Max(paramDiffPostB, Math.Abs(clonedParams[i] - clonedParamsPost![i]));
        }
        Console.WriteLine($"\nPost-Predict param drift: original→original-post |Δ|={paramDiffPostA:E6}");
        Console.WriteLine($"                          cloned→cloned-post     |Δ|={paramDiffPostB:E6}");

        // Compare original-post vs cloned-post directly
        double postOrigVsClone = 0;
        for (int i = 0; i < origParamsPost!.Length; i++)
            postOrigVsClone = Math.Max(postOrigVsClone, Math.Abs(origParamsPost[i] - clonedParamsPost![i]));
        Console.WriteLine($"Post-Predict original vs cloned params |Δ|={postOrigVsClone:E6}");

        // Sample a few conv layers' first kernel weight before/after
        Console.WriteLine($"\nLayer[0] (stem Conv) sample params (first 3):");
        var l0_orig = ((dynamic)original.Layers[0]).GetParameters() as Vector<double>;
        var l0_clone = ((dynamic)cloned.Layers[0]).GetParameters() as Vector<double>;
        Console.WriteLine($"  original: {string.Join(",", Enumerable.Range(0, 3).Select(i => l0_orig![i].ToString("F8")))}");
        Console.WriteLine($"  cloned:   {string.Join(",", Enumerable.Range(0, 3).Select(i => l0_clone![i].ToString("F8")))}");

        double maxOutDiff = 0;
        for (int i = 0; i < Math.Min(originalOut.Length, clonedOut.Length); i++)
        {
            double d = Math.Abs(originalOut[i] - clonedOut[i]);
            if (d > maxOutDiff) maxOutDiff = d;
        }
        Console.WriteLine($"Output max |Δ| = {maxOutDiff:E6}");
    }
}
