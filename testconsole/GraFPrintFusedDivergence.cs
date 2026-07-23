using System.Diagnostics;
using System.Reflection;
using AiDotNet.Audio.Fingerprinting;
using AiDotNet.Enums;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNetTestConsole;

/// <summary>
/// Step-by-step eager-vs-fused comparison on full paper-faithful GraFPrint.
/// Same architecture, same init seed, same input, same target. Print loss
/// and per-layer parameter L2-norm after each Train step. Find the
/// divergence point between the two trajectories.
///
/// Used to chase down the "fused-Adam explodes loss to ~4000x initial
/// while eager-Adam decreases it" bug surfaced when the smart-gate change
/// (let fused engage on industry-default Adam) is on.
/// </summary>
public static class GraFPrintFusedDivergence
{
    public static void Run()
    {
        var listener = new TextWriterTraceListener(Console.Out);
        Trace.Listeners.Add(listener);
        Trace.AutoFlush = true;

        var t = typeof(GraFPrint<>).MakeGenericType(typeof(double));
        var ctor = t.GetConstructor(new[] {
            typeof(NeuralNetworkArchitecture<double>),
            typeof(GraFPrintOptions),
            typeof(AiDotNet.Interfaces.IGradientBasedOptimizer<double, Tensor<double>, Tensor<double>>)
        });

        // No RandomSeed support on this NeuralNetworkArchitecture ctor — both
        // models will init from RandomHelper.CreateSecureRandom() so their
        // initial params differ. That's OK: we look at the loss-trajectory
        // SHAPE, not bit-exact equality. A monotonic-decreasing trajectory
        // vs. an exploding trajectory at the same depth + LR is the signal.
        NeuralNetworkArchitecture<double> Arch() => new(
            inputType: InputType.TwoDimensional,
            taskType: NeuralNetworkTaskType.Regression,
            inputHeight: 64, inputWidth: 32, inputDepth: 1, outputSize: 4);

        var input = new Tensor<double>(new[] { 1, 64, 32 });
        var rng = new Random(42);
        for (int i = 0; i < input.Length; i++) input[i] = rng.NextDouble();

        // Both runs share identical inputs and targets.
        var commitField = typeof(NeuralNetworkBase<double>).GetField(
            "_fusedTrainingCommitted", BindingFlags.NonPublic | BindingFlags.Instance);

        Console.WriteLine("=== Building eager model + capturing target ===");
        AiDotNet.Training.CompiledTapeTrainingStep<double>.Invalidate();
        dynamic eagerModel = ctor!.Invoke(new object?[] { Arch(), null, null });

        // Use the eager model's output shape as the target shape (paper-faithful path).
        Tensor<double> warmup = eagerModel.Predict(input);
        var shapeArr = new int[warmup.Shape.Length];
        for (int i = 0; i < warmup.Shape.Length; i++) shapeArr[i] = warmup.Shape[i];
        var target = new Tensor<double>(shapeArr);
        var trng = new Random(99);
        for (int i = 0; i < target.Length; i++) target[i] = trng.NextDouble();

        Console.WriteLine($"target shape: [{string.Join(",", shapeArr)}], target [0..3] = {target[0]:F4},{target[1]:F4},{target[2]:F4},{target[3]:F4}");

        // ----- EAGER trajectory -----
        // Force eager by swapping TensorCodecOptions.Current to one with
        // EnableCompilation=false. Setting Current.EnableCompilation directly
        // is a no-op because Current returns a snapshot.
        var savedOptions = AiDotNet.Tensors.Engines.Optimization.TensorCodecOptions.Current;
        var eagerOpts = new AiDotNet.Tensors.Engines.Optimization.TensorCodecOptions
        {
            EnableCompilation = false,
            EnableDataflowFusion = savedOptions.EnableDataflowFusion,
            EnableAlgebraicBackward = savedOptions.EnableAlgebraicBackward,
            EnableSpectralDecomposition = savedOptions.EnableSpectralDecomposition,
            SpectralErrorTolerance = savedOptions.SpectralErrorTolerance,
            DataflowFusionMaxHidden = savedOptions.DataflowFusionMaxHidden,
            EnableConvBnFusion = savedOptions.EnableConvBnFusion,
            EnableAttentionFusion = savedOptions.EnableAttentionFusion,
            EnablePointwiseFusion = savedOptions.EnablePointwiseFusion,
            EnableForwardCSE = savedOptions.EnableForwardCSE,
            EnableBlasBatch = savedOptions.EnableBlasBatch,
            EnableMixedPrecision = savedOptions.EnableMixedPrecision,
        };
        AiDotNet.Tensors.Engines.Optimization.TensorCodecOptions.SetCurrent(eagerOpts);
        Console.WriteLine($"\n=== EAGER trajectory (EnableCompilation forced to false) ===");
        TraceTrajectory(eagerModel, input, target, 8, "EAGER", commitField);

        // ----- FUSED trajectory -----
        AiDotNet.Tensors.Engines.Optimization.TensorCodecOptions.SetCurrent(savedOptions);
        AiDotNet.Training.CompiledTapeTrainingStep<double>.Invalidate();
        AiDotNet.Training.CompiledTapeTrainingStep<double>.ResetFusedStepCount();
        Console.WriteLine($"\n=== FUSED trajectory (smart-gate engaged, EnableCompilation={savedOptions.EnableCompilation}) ===");
        dynamic fusedModel = ctor!.Invoke(new object?[] { Arch(), null, null });
        TraceTrajectory(fusedModel, input, target, 8, "FUSED", commitField);
    }

    private static void TraceTrajectory(
        dynamic model, Tensor<double> input, Tensor<double> target,
        int steps, string tag, FieldInfo? commitField)
    {
        Tensor<double> initOut = model.Predict(input);
        double initLoss = Mse(initOut, target);
        Vector<double> initParams = model.GetParameters();
        double initParamL2 = L2(initParams);
        Console.WriteLine($"  [{tag}] step 0 (pre-Train): loss={initLoss,12:F4}  paramL2={initParamL2,10:F4}");

        Vector<double> prevParams = initParams;
        for (int s = 1; s <= steps; s++)
        {
            try { model.Train(input, target); }
            catch (Exception ex) { Console.WriteLine($"  [{tag}] step {s} Train THREW: {ex.Message}"); return; }

            Tensor<double> outAfter = model.Predict(input);
            double loss = Mse(outAfter, target);

            Vector<double> currParams = model.GetParameters();
            double paramL2 = L2(currParams);
            double deltaParamL2 = L2Diff(currParams, prevParams);
            prevParams = currParams;

            bool committed = false;
            if (commitField is not null)
            {
                var v = commitField.GetValue(model);
                if (v is bool b) committed = b;
            }
            long fusedSteps = AiDotNet.Training.CompiledTapeTrainingStep<double>.GetFusedStepCount();
            Console.WriteLine($"  [{tag}] step {s}: loss={loss,12:F4}  paramL2={paramL2,10:F4}  ||Δparams||={deltaParamL2,10:F6}  fusedCommitted={committed}  fusedSteps={fusedSteps}");
        }
    }

    private static double L2(Vector<double> v)
    {
        double s = 0;
        for (int i = 0; i < v.Length; i++) s += v[i] * v[i];
        return Math.Sqrt(s);
    }

    private static double L2Diff(Vector<double> a, Vector<double> b)
    {
        int n = Math.Min(a.Length, b.Length);
        double s = 0;
        for (int i = 0; i < n; i++) { double d = a[i] - b[i]; s += d * d; }
        return Math.Sqrt(s);
    }

    private static double Mse(Tensor<double> a, Tensor<double> b)
    {
        int n = Math.Min(a.Length, b.Length);
        double sum = 0;
        for (int i = 0; i < n; i++)
        {
            double d = a[i] - b[i];
            sum += d * d;
        }
        return n > 0 ? sum / n : 0;
    }
}
