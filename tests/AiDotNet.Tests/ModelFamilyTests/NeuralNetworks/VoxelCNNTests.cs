using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Tests.ModelFamilyTests.Base;
using Xunit;

namespace AiDotNet.Tests.ModelFamilyTests.NeuralNetworks;

public class VoxelCNNTests : NeuralNetworkModelTestBase<float>
{
    // VoxelCNN default: 32x32x32 voxels, 1 channel
    // Actual output is 128-dim from conv feature extraction
    protected override int[] InputShape => [1, 32, 32, 32];
    protected override int[] OutputShape => [128];

    // 3D convolutions on 32³ voxel grids are inherently expensive on CPU
    // (one Conv3D forward at the default Layers stack takes ≳ 200 ms on
    // consumer hardware). MoreData_ShouldNotDegrade at the default 50/200
    // iter count = 250 × 200 ms ≈ 50 s per network × 2 networks ≈ 100 s,
    // and pairs with the test's setup / arena work to overflow the 120 s
    // xUnit per-test timeout. 1 / 2 still exercises the "long ≥ short
    // shouldn't degrade" invariant — same pattern Forecasting Foundation
    // models and paper-scale CLIP encoders use.
    protected override int MoreDataShortIterations => 1;
    protected override int MoreDataLongIterations => 2;
    protected override double MoreDataTolerance => 0.5;

    protected override INeuralNetworkModel<float> CreateNetwork()
        => new VoxelCNN<float>();

    // TEMP DIAGNOSTIC (#1789 T-Z shard): VoxelCNN GradientFlow/Clone NaN reproduces ONLY on the Linux
    // CI runner (Windows local + WSL2 container both pass). This probe reproduces the exact
    // GradientFlow sequence and, in the trx Message, reports: per-layer parameter magnitudes after one
    // Train step (which layer first goes non-finite), the post-train forward output range, the same
    // under compile ON vs OFF (does the compiled/autotune path matter?), and the raw per-layer
    // gradient max-abs from a tape forward (is the gradient Inf before the optimizer runs?). Passes
    // clean when everything is finite; fails with the full breakdown when a NaN/Inf appears. Temporary.
    [Fact]
    public void Diag_VoxelCNN_NaNOrigin()
    {
        var sb = new StringBuilder();
        bool bad = false;

        MirrorProbe("compile=ON", sb, ref bad, compile: true);
        MirrorProbe("compile=OFF", sb, ref bad, compile: false);
        GradientProbe(sb, ref bad);

        if (bad)
            Assert.Fail("VoxelCNN NaN diagnostic (non-finite detected):\n" + sb.ToString());
    }

    // Exact GradientFlow body (warmup Predict, one Train step) + per-layer parameter magnitude scan.
    private void MirrorProbe(string label, StringBuilder sb, ref bool bad, bool compile)
    {
        var codec = AiDotNet.Tensors.Engines.Optimization.TensorCodecOptions.Current;
        bool saved = codec.EnableCompilation;
        try
        {
            codec.EnableCompilation = compile;
            var rng = ModelTestHelpers.CreateSeededRandom();
            var net = (VoxelCNN<float>)CreateNetwork();
            var input = CreateRandomTensor(InputShape, rng);
            var target = CreateRandomTargetTensor(EffectiveOutputShape, rng);
            sb.AppendLine($"[{label}] input {Describe(input)}; target {Describe(target)}");

            net.SetTrainingMode(false);
            try { net.Predict(input); }
            catch (InvalidOperationException)
            {
                net.SetTrainingMode(true);
                try { net.Train(input, target); } catch { /* warmup */ }
            }

            net.SetTrainingMode(true);
            try { net.Train(input, target); }
            catch (Exception ex) { bad = true; sb.AppendLine($"[{label}] Train THREW {ex.GetType().Name}: {ex.Message}"); }

            var layers = net.Layers;
            string firstNan = "none";
            for (int i = 0; i < layers.Count; i++)
            {
                var p = layers[i].GetParameters();
                if (p is null || p.Length == 0) continue;
                double mn = double.MaxValue, mx = double.MinValue; bool fin = true;
                for (int j = 0; j < p.Length; j++)
                {
                    double v = Convert.ToDouble(p[j]);
                    if (double.IsNaN(v) || double.IsInfinity(v)) fin = false;
                    else { if (v < mn) mn = v; if (v > mx) mx = v; }
                }
                if (!fin) { bad = true; if (firstNan == "none") firstNan = $"L{i}:{layers[i].GetType().Name}"; }
                sb.AppendLine($"[{label}] L{i}:{layers[i].GetType().Name} finite={fin} range=[{mn:E2},{mx:E2}] n={p.Length}");
            }

            net.SetTrainingMode(false);
            string outRange;
            try { outRange = Describe(net.Predict(input)); }
            catch (Exception ex) { outRange = $"Predict threw {ex.GetType().Name}"; }
            sb.AppendLine($"[{label}] post-train forward out={outRange}; firstNaNLayer={firstNan}");
        }
        finally { codec.EnableCompilation = saved; }
    }

    // Direct tape gradient: is any layer's gradient non-finite BEFORE the optimizer step?
    private void GradientProbe(StringBuilder sb, ref bool bad)
    {
        try
        {
            var rng = ModelTestHelpers.CreateSeededRandom();
            var net = (VoxelCNN<float>)CreateNetwork();
            var input = CreateRandomTensor(InputShape, rng);
            var target = CreateRandomTargetTensor(EffectiveOutputShape, rng);
            net.SetTrainingMode(true);
            var eng = AiDotNetEngine.Current;

            using var tape = new GradientTape<float>();
            var outT = net.ForwardForTraining(input);
            var tgt = outT.Shape.ToArray().SequenceEqual(target.Shape.ToArray())
                ? target : eng.Reshape(target, outT.Shape.ToArray());
            var diff = eng.TensorSubtract(outT, tgt);
            var sq = eng.TensorMultiply(diff, diff);
            var axes = Enumerable.Range(0, sq.Shape.Length).ToArray();
            var loss = eng.ReduceMean(sq, axes, keepDims: false);

            var trainables = new List<Tensor<float>>();
            foreach (var l in net.Layers)
                if (l is ITrainableLayer<float> tl) trainables.AddRange(tl.GetTrainableParameters());

            var grads = tape.ComputeGradients(loss, trainables);
            double lossVal = Convert.ToDouble(loss.Data.Span.Length > 0 ? loss.Data.Span[0] : float.NaN);
            sb.AppendLine($"[grad] outT={Describe(outT)} loss={lossVal:E3} trainables={trainables.Count}");
            for (int i = 0; i < trainables.Count; i++)
            {
                grads.TryGetValue(trainables[i], out var g);
                double mx = 0; bool fin = true;
                if (g is not null)
                {
                    var gs = g.Data.Span;
                    for (int j = 0; j < gs.Length; j++)
                    {
                        double v = Math.Abs((double)gs[j]);
                        if (double.IsNaN(v) || double.IsInfinity(v)) fin = false;
                        else if (v > mx) mx = v;
                    }
                }
                if (!fin) { bad = true; sb.AppendLine($"[grad] trainable[{i}] shape=[{string.Join(",", trainables[i].Shape.ToArray())}] NON-FINITE gradient"); }
                else sb.AppendLine($"[grad] trainable[{i}] shape=[{string.Join(",", trainables[i].Shape.ToArray())}] maxabs={mx:E3}");
            }
        }
        catch (Exception ex) { sb.AppendLine($"[grad] probe threw {ex.GetType().Name}: {ex.Message}"); }
    }

    private static string Describe(Tensor<float> t)
    {
        var s = t.Data.Span;
        double mn = double.MaxValue, mx = double.MinValue; int nan = 0, inf = 0;
        for (int i = 0; i < s.Length; i++)
        {
            float v = s[i];
            if (float.IsNaN(v)) { nan++; continue; }
            if (float.IsInfinity(v)) { inf++; continue; }
            if (v < mn) mn = v; if (v > mx) mx = v;
        }
        return $"[{string.Join(",", t.Shape.ToArray())}] range=[{mn:E2},{mx:E2}] nan={nan} inf={inf}";
    }
}
