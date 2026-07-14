using System;
using System.Text;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks;
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

    // TEMP DIAGNOSTIC (#1789 T-Z shard): VoxelCNN GradientFlow/Clone NaN reproduces ONLY on the
    // Linux CI runner (Windows local passes the whole T-Z shard 252/252). This test reuses the
    // exact harness conditions and localizes WHERE the NaN first appears: forward activations at
    // init, or a specific layer's parameters after one Train step. It PASSES when everything is
    // finite (Windows) and FAILS with the pinpoint location when a NaN is present (Linux CI),
    // so the trx Message carries the diagnosis back. Delete once the root cause is fixed.
    [Fact]
    public void Diag_VoxelCNN_NaNOrigin()
    {
        var rng = ModelTestHelpers.CreateSeededRandom();
        var net = (VoxelCNN<float>)CreateNetwork();
        var input = CreateRandomTensor(InputShape, rng);
        var target = CreateRandomTargetTensor(EffectiveOutputShape, rng);

        var sb = new StringBuilder();
        bool bad = false;

        sb.AppendLine($"input {Describe(input)}; target {Describe(target)}");

        // 1) Forward at init — is any layer activation already non-finite?
        net.SetTrainingMode(false);
        try
        {
            var pred = net.Predict(input);
            bool predFin = AllFinite(pred);
            bad |= !predFin;
            sb.AppendLine($"[pre-train] Predict finite={predFin} {Describe(pred)}");
        }
        catch (Exception ex) { bad = true; sb.AppendLine($"[pre-train] Predict THREW {ex.GetType().Name}: {ex.Message}"); }

        ScanActivations(net, input, "pre-train", sb, ref bad);
        ScanLayerParams(net, "pre-train", sb, ref bad);

        // 2) One training step (matches GradientFlow: single Train call).
        net.SetTrainingMode(true);
        try { net.Train(input, target); }
        catch (Exception ex) { bad = true; sb.AppendLine($"[train] Train THREW {ex.GetType().Name}: {ex.Message}"); }

        // 3) Which layer's parameters went non-finite after the update?
        ScanLayerParams(net, "post-train", sb, ref bad);
        // 4) And is the forward now non-finite (and at which layer)?
        ScanActivations(net, input, "post-train", sb, ref bad);

        if (bad)
            Assert.Fail("VoxelCNN NaN diagnostic (non-finite detected):\n" + sb.ToString());
    }

    private static void ScanActivations(VoxelCNN<float> net, Tensor<float> input, string phase, StringBuilder sb, ref bool bad)
    {
        try
        {
            var acts = net.GetNamedLayerActivations(input);
            foreach (var kv in acts)
            {
                bool fin = AllFinite(kv.Value);
                if (!fin) { bad = true; sb.AppendLine($"[{phase}] act[{kv.Key}] NON-FINITE {Describe(kv.Value)}"); }
            }
        }
        catch (Exception ex) { bad = true; sb.AppendLine($"[{phase}] GetNamedLayerActivations THREW {ex.GetType().Name}: {ex.Message}"); }
    }

    private static void ScanLayerParams(VoxelCNN<float> net, string phase, StringBuilder sb, ref bool bad)
    {
        var layers = net.Layers;
        for (int i = 0; i < layers.Count; i++)
        {
            try
            {
                var p = layers[i].GetParameters();
                if (p is null || p.Length == 0) continue;
                bool fin = true; double mn = double.MaxValue, mx = double.MinValue;
                for (int j = 0; j < p.Length; j++)
                {
                    double v = Convert.ToDouble(p[j]);
                    if (double.IsNaN(v) || double.IsInfinity(v)) fin = false;
                    if (v < mn) mn = v; if (v > mx) mx = v;
                }
                if (!fin) { bad = true; sb.AppendLine($"[{phase}] L{i}:{layers[i].GetType().Name} params NON-FINITE range=[{mn:E3},{mx:E3}] count={p.Length}"); }
            }
            catch (Exception ex) { bad = true; sb.AppendLine($"[{phase}] L{i} GetParameters THREW {ex.GetType().Name}: {ex.Message}"); }
        }
    }

    private static bool AllFinite(Tensor<float> t)
    {
        var s = t.Data.Span;
        for (int i = 0; i < s.Length; i++)
            if (float.IsNaN(s[i]) || float.IsInfinity(s[i])) return false;
        return true;
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
        return $"[{string.Join(",", t.Shape.ToArray())}] range=[{mn:E3},{mx:E3}] nan={nan} inf={inf}";
    }
}
