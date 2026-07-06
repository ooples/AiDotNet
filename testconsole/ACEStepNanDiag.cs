using AiDotNet.Audio.Generation;
using AiDotNet.Enums;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tensors.Engines.Optimization;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNetTestConsole;

/// <summary>
/// Localizes the ACEStepTests NaN (ForwardPass_ShouldBeFinite_AfterTraining:
/// "Output[0] is NaN after 10 training iterations"). Mirrors the test
/// scaffold's exact configuration ([1, 2, 44100] stereo input, paper-default
/// ACEStepOptions, double precision), trains one iteration at a time, and
/// after each iteration scans every layer's parameters and replays the
/// forward layer-by-layer to report WHERE the first non-finite value
/// appears — layer index, layer type, and whether it is in the weights or
/// the activations.
/// </summary>
internal static class ACEStepNanDiag
{
    public static void Run()
    {
        var architecture = new NeuralNetworkArchitecture<double>(
            inputType: InputType.OneDimensional,
            taskType: NeuralNetworkTaskType.Regression,
            inputSize: 44100,
            outputSize: 44100);
        var options = new ACEStepOptions();
        // Optional scale knob for bisecting size-dependent kernel dispatch:
        // ACE_UNETDIM=64 shrinks the transformer width while keeping the
        // exact model wiring.
        var dimEnv = Environment.GetEnvironmentVariable("ACE_UNETDIM");
        if (int.TryParse(dimEnv, out int dimOverride) && dimOverride > 0)
        {
            options.UNetDim = dimOverride;
            Console.WriteLine($"UNetDim override: {dimOverride}");
        }
        // ACE_DISABLE_COMPILATION=1: force the pure eager tape path so the
        // compiled-plan step/parameter-sync can be ruled in or out as the
        // source of the iter-2 zero-output + NaN-update corruption.
        if (Environment.GetEnvironmentVariable("ACE_DISABLE_COMPILATION") == "1")
        {
            TensorCodecOptions.SetCurrent(new TensorCodecOptions { EnableCompilation = false });
            Console.WriteLine("Compilation disabled (pure eager tape).");
        }
        var model = new ACEStep<double>(architecture, options);

        var rng = new Random(42);
        var input = RandomTensor([1, 2, 44100], rng);

        // Match the test base: target shaped like the model's actual output.
        var warm = model.Predict(input);
        Console.WriteLine($"warm-up output shape: [{string.Join(",", warm.Shape)}], " +
                          $"finite={CountNonFinite(warm)} non-finite of {warm.Length}");
        var targetDims = new int[warm.Rank];
        for (int i = 0; i < warm.Rank; i++) targetDims[i] = warm.Shape[i];
        var target = RandomTensor(targetDims, rng);

        // ACE_PURE_TRAIN=1: skip the mid-loop ComputeGradients probe — that
        // call is itself a tape pass and may be the state-poisoner; the real
        // CI test only calls Train.
        bool pureTrain = Environment.GetEnvironmentVariable("ACE_PURE_TRAIN") == "1";

        // ACE_SKIP_PARAM_BUFFER=1: disable the TrainWithTape ParameterBuffer
        // view-swap via the same private flags the foundation-scale skip path
        // sets, to cleanly attribute the zero-forward corruption to the
        // buffer/view machinery vs the tape-mode layer forward.
        if (Environment.GetEnvironmentVariable("ACE_SKIP_PARAM_BUFFER") == "1")
        {
            var t = typeof(NeuralNetworkBase<double>);
            var skipField = t.GetField("_skipParameterBuffer",
                System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance);
            var verField = t.GetField("_skipParameterBufferVersion",
                System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance);
            var structVerField = t.GetField("_layerStructureVersion",
                System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance);
            if (skipField is not null && verField is not null && structVerField is not null)
            {
                skipField.SetValue(model, true);
                verField.SetValue(model, structVerField.GetValue(model));
                Console.WriteLine("ParameterBuffer swap disabled via reflection.");
            }
            else
            {
                Console.WriteLine("WARNING: could not find skip fields; buffer swap still active.");
            }
        }

        // L2 norm of the first Dense layer's first trainable tensor — the
        // value the next tape forward will actually read through the layer
        // field (post-swap this is a ParameterBuffer view).
        double FirstWeightL2()
        {
            foreach (var layer in model.Layers)
            {
                if (layer is AiDotNet.Interfaces.ITrainableLayer<double> t)
                {
                    var ps = t.GetTrainableParameters();
                    if (ps.Count == 0) continue;
                    double s = 0;
                    var w = ps[0];
                    for (int i = 0; i < w.Length; i++) { double v = w[i]; if (!double.IsNaN(v) && !double.IsInfinity(v)) s += v * v; }
                    return Math.Sqrt(s);
                }
            }
            return -1;
        }

        for (int iter = 1; iter <= 10; iter++)
        {
            Console.WriteLine($"iter {iter}: PRE-Train  firstWeight L2={FirstWeightL2():G8}");
            model.Train(input, target);
            Console.WriteLine($"iter {iter}: POST-Train firstWeight L2={FirstWeightL2():G8}");
            Console.WriteLine($"iter {iter}: LastLoss={model.GetLastLoss()}");

            // ACE_TAPE_WALK=1: after each Train, replicate TrainWithTape's
            // forward conditions (training mode + active GradientTape) and
            // walk the layers manually, printing the first layer whose
            // output collapses — pinpoints WHERE the tape-mode forward
            // diverges from the eager one.
            if (Environment.GetEnvironmentVariable("ACE_TAPE_WALK") == "1")
            {
                model.SetTrainingMode(true);
                try
                {
                    using var walkTape = new AiDotNet.Tensors.Engines.Autodiff.GradientTape<double>();
                    var cur = input;
                    int li = 0;
                    foreach (var layer in model.Layers)
                    {
                        cur = layer.Forward(cur);
                        double l2 = 0; int bad = 0;
                        for (int i = 0; i < cur.Length; i++)
                        {
                            double v = cur[i];
                            if (double.IsNaN(v) || double.IsInfinity(v)) bad++;
                            else l2 += v * v;
                        }
                        l2 = Math.Sqrt(l2);
                        if (l2 < 1e-12 || bad > 0 || li >= model.Layers.Count - 2)
                            Console.WriteLine($"iter {iter}: tape-walk layer {li} ({layer.GetType().Name}): outL2={l2:G6}, nonFinite={bad}");
                        if (bad > 0 && li == 0)
                        {
                            // First NaN at the very first layer: dump positions and
                            // check the layer's FIELD-based parameter export for NaN.
                            var positions = new List<int>();
                            for (int i = 0; i < cur.Length && positions.Count < 20; i++)
                                if (double.IsNaN(cur[i]) || double.IsInfinity(cur[i])) positions.Add(i);
                            Console.WriteLine($"iter {iter}: layer0 NaN cells: {string.Join(",", positions)}");
                            var fieldParams = layer.GetParameters();
                            int fpBad = 0;
                            for (int i = 0; i < fieldParams.Length; i++)
                                if (double.IsNaN(fieldParams[i]) || double.IsInfinity(fieldParams[i])) fpBad++;
                            Console.WriteLine($"iter {iter}: layer0 GetParameters nonFinite={fpBad}/{fieldParams.Length}");

                            // Manually recompute the first NaN cell: GetParameters layout
                            // for DenseLayer is [inputSize*outputSize weights..., biases].
                            // out[row, col] = sum_k input[row,k] * W[k,col] + b[col].
                            int cell = positions[0];
                            const int outSize = 64, inSize = 44100;
                            int row = cell / outSize, col = cell % outSize;
                            double acc = 0;
                            for (int kk = 0; kk < inSize; kk++)
                                acc += input[row * inSize + kk] * fieldParams[kk * outSize + col];
                            acc += fieldParams[inSize * outSize + col];
                            var gelu = new AiDotNet.ActivationFunctions.GELUActivation<double>();
                            double act = gelu.Activate(acc);
                            Console.WriteLine($"iter {iter}: cell {cell} (r{row},c{col}) naive preAct={acc:G8}, scalarGELU={act:G8}, tapeValue={cur[cell]}");
                            break;
                        }
                        if (l2 < 1e-12 && bad == 0) break;
                        li++;
                    }
                }
                finally { model.SetTrainingMode(false); }
            }

            if (pureTrain)
            {
                int badLayerIdx = 0;
                bool anyBad = false;
                foreach (var layer in model.Layers)
                {
                    var p = layer.GetParameters();
                    for (int i = 0; i < p.Length; i++)
                        if (double.IsNaN(p[i]) || double.IsInfinity(p[i])) { anyBad = true; break; }
                    if (anyBad)
                    {
                        Console.WriteLine($"iter {iter}: PARAMS non-finite first in layer {badLayerIdx} ({layer.GetType().Name})");
                        break;
                    }
                    badLayerIdx++;
                }
                var outNow = model.Predict(input);
                double maxNow = 0;
                int badNow = 0;
                for (int i = 0; i < outNow.Length; i++)
                {
                    double v = outNow[i];
                    if (double.IsNaN(v) || double.IsInfinity(v)) badNow++;
                    else if (Math.Abs(v) > maxNow) maxNow = Math.Abs(v);
                }
                Console.WriteLine($"iter {iter}: eager Predict max|x|={maxNow:G6}, nonFinite={badNow}");
                if (anyBad) break;
                continue;
            }

            // 0) Gradient health for the NEXT step: compute (without applying)
            //    and report per-layer non-finite counts + max |g| so the layer
            //    whose backward produces the first Inf/NaN is identified BEFORE
            //    the global clip poisons every parameter.
            var grads = model.ComputeGradients(input, target);
            long totalParams = 0;
            foreach (var layer in model.Layers) totalParams += layer.ParameterCount;
            int firstBad = -1;
            for (int i = 0; i < grads.Length; i++)
            {
                if (double.IsNaN(grads[i]) || double.IsInfinity(grads[i])) { firstBad = i; break; }
            }
            Console.WriteLine($"iter {iter}: grads.Length={grads.Length}, sum(ParameterCount)={totalParams}, " +
                              $"firstNonFiniteIdx={firstBad}" +
                              (firstBad >= 0 ? $" value={grads[firstBad]}" : string.Empty));
            int offset = 0;
            int gLayerIdx = 0;
            foreach (var layer in model.Layers)
            {
                int count = checked((int)layer.ParameterCount);
                int bad = 0;
                double maxAbsG = 0;
                for (int i = offset; i < offset + count && i < grads.Length; i++)
                {
                    double g = grads[i];
                    if (double.IsNaN(g) || double.IsInfinity(g)) bad++;
                    else { double a = Math.Abs(g); if (a > maxAbsG) maxAbsG = a; }
                }
                int zeros = 0;
                for (int i = offset; i < offset + count && i < grads.Length; i++)
                    if (grads[i] == 0) zeros++;
                // Print bad layers, plus the block-boundary region (layers 40+)
                // unconditionally so zero-vs-nonzero "clean" layers are
                // distinguishable (tape-graph disconnect shows as all-zero).
                if (bad > 0 || maxAbsG > 1e12 || gLayerIdx >= 40 || gLayerIdx <= 2)
                {
                    Console.WriteLine($"iter {iter}: GRAD layer {gLayerIdx} ({layer.GetType().Name}): " +
                                      $"nonFinite={bad}/{count}, exactZero={zeros}, max finite |g|={maxAbsG:G5}");
                }
                offset += count;
                gLayerIdx++;
            }

            // 1) Which layer's PARAMETERS went non-finite first?
            int layerIdx = 0;
            bool paramNan = false;
            foreach (var layer in model.Layers)
            {
                var p = layer.GetParameters();
                int bad = 0;
                for (int i = 0; i < p.Length; i++)
                    if (double.IsNaN(p[i]) || double.IsInfinity(p[i])) bad++;
                if (bad > 0)
                {
                    Console.WriteLine($"iter {iter}: PARAMS non-finite in layer {layerIdx} " +
                                      $"({layer.GetType().Name}): {bad}/{p.Length}");
                    paramNan = true;
                }
                layerIdx++;
            }

            // 2) Replay forward layer-by-layer to find first non-finite ACTIVATION.
            var current = input;
            layerIdx = 0;
            foreach (var layer in model.Layers)
            {
                current = layer.Forward(current);
                int bad = CountNonFinite(current);
                if (bad > 0)
                {
                    double maxAbs = 0;
                    for (int i = 0; i < current.Length; i++)
                    {
                        double v = Math.Abs(current[i]);
                        if (!double.IsNaN(v) && !double.IsInfinity(v) && v > maxAbs) maxAbs = v;
                    }
                    Console.WriteLine($"iter {iter}: ACTIVATION non-finite first at layer {layerIdx} " +
                                      $"({layer.GetType().Name}): {bad}/{current.Length}, " +
                                      $"max finite |x|={maxAbs:G5}");
                    break;
                }
                layerIdx++;
            }

            if (CountNonFinite(current) == 0 && !paramNan)
            {
                double maxAbs = 0;
                for (int i = 0; i < current.Length; i++)
                {
                    double v = Math.Abs(current[i]);
                    if (v > maxAbs) maxAbs = v;
                }
                Console.WriteLine($"iter {iter}: all finite, output max |x|={maxAbs:G5}");
            }

            if (paramNan) break;
        }
    }

    private static Tensor<double> RandomTensor(int[] shape, Random rng)
    {
        var t = new Tensor<double>(shape);
        for (int i = 0; i < t.Length; i++) t[i] = rng.NextDouble() * 2 - 1;
        return t;
    }

    private static int CountNonFinite(Tensor<double> t)
    {
        int bad = 0;
        for (int i = 0; i < t.Length; i++)
            if (double.IsNaN(t[i]) || double.IsInfinity(t[i])) bad++;
        return bad;
    }
}
