using System;
using System.Collections.Generic;
using AiDotNet.ActivationFunctions;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;
using Xunit.Abstractions;

namespace AiDotNet.Tests.UnitTests.NeuralNetworks;

/// <summary>
/// Per-op breakdown of the zero-alloc CNN inference stem (option C: find the
/// non-conv headroom). Times each step the parity CNN's <c>TryFusedConvStemPredict</c>
/// runs — both Conv2DInto kernels, both bias/ReLU epilogues, both MaxPool2DInto, the
/// flatten Reshape, and the dense FusedLinear — individually, vs the whole Predict, at
/// bs1 and bs128. Reveals where the time goes once the conv kernel is fixed.
/// Env-gated (AIDOTNET_RUN_JIT_PERF=1). Uses the exact parity shapes
/// (1×28×28 → Conv16 → Pool → Conv32 → Pool → Flatten → Dense10).
/// </summary>
public class CnnStemOpBreakdownBench
{
    private readonly ITestOutputHelper _output;
    public CnnStemOpBreakdownBench(ITestOutputHelper output) => _output = output;

    [Fact]
    public void Breakdown_CnnStem_PerOp()
    {
        if (Environment.GetEnvironmentVariable("AIDOTNET_RUN_JIT_PERF") != "1") return;
        Run(batch: 1);
        Run(batch: 128);
    }

    private void Run(int batch)
    {
        var conv1 = new ConvolutionalLayer<float>(16, 3, 1, 1, activationFunction: new ReLUActivation<float>());
        var pool1 = new MaxPoolingLayer<float>(2, 2);
        var conv2 = new ConvolutionalLayer<float>(32, 3, 1, 1, activationFunction: new ReLUActivation<float>());
        var pool2 = new MaxPoolingLayer<float>(2, 2);
        var flatten = new FlattenLayer<float>();
        var dense = new DenseLayer<float>(10, activationFunction: (IActivationFunction<float>?)null);
        var layers = new List<ILayer<float>> { conv1, pool1, conv2, pool2, flatten, dense };
        var arch = new NeuralNetworkArchitecture<float>(InputType.ThreeDimensional,
            NeuralNetworkTaskType.MultiClassClassification, inputHeight: 28, inputWidth: 28, inputDepth: 1,
            outputSize: 10, layers: layers);
        var model = new ConvolutionalNeuralNetwork<float>(arch);

        var x = Rand(new[] { batch, 1, 28, 28 }, 1);
        for (int i = 0; i < 50; i++) { var _ = model.Predict(x); }   // warm: materialize weights, prepack, JIT

        var cpu = new CpuEngine();
        // Scratch buffers (exactly the stem's geometry).
        var b1 = new Tensor<float>(new[] { batch, 16, 28, 28 });   // conv1 out
        var b2 = new Tensor<float>(new[] { batch, 16, 14, 14 });   // pool1 out
        var b3 = new Tensor<float>(new[] { batch, 32, 14, 14 });   // conv2 out
        var b4 = new Tensor<float>(new[] { batch, 32, 7, 7 });     // pool2 out
        var k1 = conv1.GetFilters(); var bias1 = (float[])(object)conv1.GetBiases().GetDataArray();
        var k2 = conv2.GetFilters(); var bias2 = (float[])(object)conv2.GetBiases().GetDataArray();
        var w = dense.GetWeights(); var db = dense.GetBiases();
        var b1d = (float[])(object)b1.GetDataArray();
        var b3d = (float[])(object)b3.GetDataArray();
        var stride = new[] { 1, 1 }; var pad = new[] { 1, 1 }; var dil = new[] { 1, 1 };
        var flat = new Tensor<float>(new[] { batch, 32 * 7 * 7 });

        double tConv1 = B(() => cpu.Conv2DInto(b1, x, k1, stride, pad, dil));
        double tAct1 = B(() => CpuFusedOperations.ApplyBiasActivationNCHWInPlace(b1d, bias1, batch, 16, 28, 28, FusedActivationType.ReLU));
        double tPool1 = B(() => cpu.MaxPool2DInto(b2, b1, 2, 2));
        // Isolate kernel vs engine-wrapper: raw inline 2x2 maxpool over the same arrays.
        var b2d = (float[])(object)b2.GetDataArray();
        int poolC = 16, inH = 28, inW = 28, outHp = 14, outWp = 14;
        double tPool1Raw = B(() =>
        {
            for (int bc = 0; bc < batch * poolC; bc++)
            {
                int inBase = bc * inH * inW, outBase = bc * outHp * outWp;
                for (int oh = 0; oh < outHp; oh++)
                {
                    int r0 = inBase + oh * 2 * inW, r1 = r0 + inW, dst = outBase + oh * outWp;
                    for (int ow = 0; ow < outWp; ow++)
                    {
                        int iw = ow * 2;
                        float m = b1d[r0 + iw];
                        float v = b1d[r0 + iw + 1]; if (v > m) m = v;
                        v = b1d[r1 + iw]; if (v > m) m = v;
                        v = b1d[r1 + iw + 1]; if (v > m) m = v;
                        b2d[dst + ow] = m;
                    }
                }
            }
        });
        _output.WriteLine($"  [pool1 raw-inline {tPool1Raw*1000:F2}us  engine {tPool1*1000:F2}us  b1.IsContiguous={b1.IsContiguous}]");
        double tConv2 = B(() => cpu.Conv2DInto(b3, b2, k2, stride, pad, dil));
        double tAct2 = B(() => CpuFusedOperations.ApplyBiasActivationNCHWInPlace(b3d, bias2, batch, 32, 14, 14, FusedActivationType.ReLU));
        double tPool2 = B(() => cpu.MaxPool2DInto(b4, b3, 2, 2));
        double tFlat = B(() => { var r = cpu.Reshape(b4, new[] { batch, 32 * 7 * 7 }); });
        double tDense = B(() => { var r = cpu.FusedLinear(flat, w, db, FusedActivationType.None); });

        // Manual full chain in ONE timed region — disambiguates real per-call
        // orchestration overhead from the sum-of-per-op-minimums artifact.
        double tChain = B(() =>
        {
            cpu.Conv2DInto(b1, x, k1, stride, pad, dil);
            CpuFusedOperations.ApplyBiasActivationNCHWInPlace(b1d, bias1, batch, 16, 28, 28, FusedActivationType.ReLU);
            cpu.MaxPool2DInto(b2, b1, 2, 2);
            cpu.Conv2DInto(b3, b2, k2, stride, pad, dil);
            CpuFusedOperations.ApplyBiasActivationNCHWInPlace(b3d, bias2, batch, 32, 14, 14, FusedActivationType.ReLU);
            cpu.MaxPool2DInto(b4, b3, 2, 2);
            var fl = cpu.Reshape(b4, new[] { batch, 32 * 7 * 7 });
            var r = cpu.FusedLinear(fl, w, db, FusedActivationType.None);
        });
        double tPredict = B(() => { var _ = model.Predict(x); });

        double sum = tConv1 + tAct1 + tPool1 + tConv2 + tAct2 + tPool2 + tFlat + tDense;
        _output.WriteLine($"  [MaxDOP={AiDotNet.Tensors.Helpers.CpuParallelSettings.MaxDegreeOfParallelism}]");
        _output.WriteLine($"=== CNN stem breakdown, batch={batch} (us, best-of) ===");
        _output.WriteLine($"  conv1  Conv2DInto   {tConv1*1000,8:F2}");
        _output.WriteLine($"  conv1  bias+ReLU    {tAct1*1000,8:F2}");
        _output.WriteLine($"  pool1  MaxPool2DInto{tPool1*1000,8:F2}");
        _output.WriteLine($"  conv2  Conv2DInto   {tConv2*1000,8:F2}");
        _output.WriteLine($"  conv2  bias+ReLU    {tAct2*1000,8:F2}");
        _output.WriteLine($"  pool2  MaxPool2DInto{tPool2*1000,8:F2}");
        _output.WriteLine($"  flat   Reshape      {tFlat*1000,8:F2}");
        _output.WriteLine($"  dense  FusedLinear  {tDense*1000,8:F2}");
        _output.WriteLine($"  ---- sum of ops      {sum*1000,8:F2}  (sum-of-minimums; lower bound)");
        _output.WriteLine($"  ==== manual chain    {tChain*1000,8:F2}  (real op cost, one region)");
        _output.WriteLine($"  ==== full Predict    {tPredict*1000,8:F2}  (Predict overhead vs chain = {(tPredict-tChain)*1000:F2})");
    }

    private static double B(Action f)
    {
        for (int i = 0; i < 30; i++) f();
        double best = double.MaxValue;
        for (int r = 0; r < 400; r++)
        {
            var sw = System.Diagnostics.Stopwatch.StartNew();
            f();
            sw.Stop();
            best = Math.Min(best, sw.Elapsed.TotalMilliseconds);
        }
        return best;
    }

    private static Tensor<float> Rand(int[] shape, int seed)
    {
        var rng = new Random(seed); var t = new Tensor<float>(shape); var s = t.AsWritableSpan();
        for (int i = 0; i < s.Length; i++) s[i] = (float)(rng.NextDouble() - 0.5);
        return t;
    }
}
