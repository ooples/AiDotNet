// Copyright (c) CrowdTrainer LLC. All rights reserved.

using System;
using System.Collections.Generic;
using System.Linq;
using AiDotNet.ActivationFunctions;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Optimizers;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.NeuralNetworks;

/// <summary>
/// The eager tape step (TrainWithTape, the path a model takes when the compiled fused step does not apply) on the GPU
/// engine must follow the CPU engine's trajectory. Its parameters live in one ParameterBuffer, every parameter a view of
/// a flat array. Three bugs made the GPU step train on wrong values: the device weight cache served every view the
/// buffer cached for the whole flat array (the first parameter's data), the post-step invalidation dropped the device
/// optimizer's update, and the device optimizer was off unless an environment variable opted in. (Measured on a
/// 27M-parameter LM: GPU loss 10.91 -> 9.55 over five steps while the CPU engine went 10.91 -> 5.81.)
/// </summary>
[Collection("TrainingDiagnosticsSequential")]
public class GpuEagerTapeStepMatchesCpuTests
{
    private const int Inputs = 16, Hidden = 32, Outputs = 4, Batch = 32, Steps = 6;

    // Warm eager steps the readback test measures, and the reads each may make: three device norm records (network
    // clip, optimizer clip, finiteness check) and the loss. A record's bytes are those of the pooled buffer it was
    // rented into, so they are fixed per model but not pinned here.
    private const int MeasuredSteps = 3, ScalarReadsPerStep = 4;

    private sealed class ClipProbe : AdamWOptimizer<float, Tensor<float>, Tensor<float>>
    {
        public ClipProbe() : base(null) { }
        public static void Clip(TapeStepContext<float> context) => ApplyTapeGlobalNormGradientClipping(context, 1.0);
    }

    [SkippableFact]
    public void Gpu_norm_overflow_falls_back_to_double_precision_clipping()
    {
        using var gpu = new DirectGpuTensorEngine();
        if (Environment.GetEnvironmentVariable("AIDOTNET_REQUIRE_GPU_TESTS") == "1")
            Assert.True(gpu.IsGpuAvailable, "The required GPU backend failed to initialize.");
        Skip.If(!gpu.IsGpuAvailable, "No DirectGpuTensorEngine available.");
        var previous = AiDotNetEngine.Current;
        AiDotNetEngine.Current = gpu;
        try
        {
            var param = new Tensor<float>(new[] { 1 });
            var grad = gpu.TensorMultiplyScalar(new Tensor<float>(new[] { 1 }, new Vector<float>(new[] { 3e19f })), 1f);
            ClipProbe.Clip(new TapeStepContext<float>(new[] { param },
                new Dictionary<Tensor<float>, Tensor<float>> { [param] = grad }, 0f));
            Assert.InRange(grad.ToArray()[0], 0.99999f, 1.00001f);
        }
        finally { AiDotNetEngine.Current = previous; }
    }

    /// <summary>A network that always takes the eager tape step.</summary>
    private sealed class EagerNetwork : NeuralNetwork<float>
    {
        public EagerNetwork(NeuralNetworkArchitecture<float> architecture, IGradientBasedOptimizer<float, Tensor<float>, Tensor<float>> optimizer)
            : base(architecture, optimizer) { }

        protected override bool SupportsFusedCompiledTraining => false;
    }

    private static EagerNetwork BuildNetwork(int hidden = Hidden)
    {
        var layers = new List<ILayer<float>>
        {
            new DenseLayer<float>(hidden, (IActivationFunction<float>)new ReLUActivation<float>()),
            // A norm gain read through the device weight path, at a nonzero offset of the flat parameter array.
            new LayerNormalizationLayer<float>(hidden),
            new DenseLayer<float>(Outputs, (IActivationFunction<float>?)null),
        };
        var architecture = new NeuralNetworkArchitecture<float>(
            inputType: InputType.OneDimensional, taskType: NeuralNetworkTaskType.Regression,
            inputSize: Inputs, outputSize: Outputs, layers: layers);
        var options = new AdamWOptimizerOptions<float, Tensor<float>, Tensor<float>> { InitialLearningRate = 1e-2 };
        return new EagerNetwork(architecture, new AdamWOptimizer<float, Tensor<float>, Tensor<float>>(null, options));
    }

    private static (Tensor<float> X, Tensor<float> Y) Data()
    {
        var rng = AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(20261102);
        var x = new Tensor<float>(new[] { Batch, Inputs });
        var y = new Tensor<float>(new[] { Batch, Outputs });
        for (int r = 0; r < Batch; r++)
        {
            for (int i = 0; i < Inputs; i++) x[r, i] = (float)(rng.NextDouble() * 2 - 1);
            for (int o = 0; o < Outputs; o++) y[r, o] = (float)(rng.NextDouble() * 2 - 1);
        }
        return (x, y);
    }

    /// <summary>
    /// The eager path's own decision for models it cannot pack into one flat buffer (over the parameter threshold, or
    /// with tied weights): every parameter is then its own tensor, and the post-step flush must still keep its device
    /// update.
    /// </summary>
    private static void SkipParameterBuffer(EagerNetwork net)
    {
        var flags = System.Reflection.BindingFlags.Instance | System.Reflection.BindingFlags.NonPublic;
        var baseType = typeof(NeuralNetworkBase<float>);
        System.Reflection.FieldInfo Field(string name) => baseType.GetField(name, flags)
            ?? throw new InvalidOperationException($"NeuralNetworkBase has no field {name}.");
        int version = Field("_layerStructureVersion").GetValue(net) is int v
            ? v : throw new InvalidOperationException("_layerStructureVersion is not an int.");
        Field("_skipParameterBuffer").SetValue(net, true);
        Field("_skipParameterBufferVersion").SetValue(net, version);
    }

    /// <summary>
    /// The #2433 transfer contract: a warm eager GPU step with the default global-norm clipping and the anomaly guard
    /// reads back no gradient or weight. Those host passes downloaded every device gradient each step (27M-parameter
    /// LM: 164 readbacks, 538 MB per step), so their bytes grew with the model; the device step reads only scalars, the
    /// same count and bytes at any width. A 16x wider hidden layer is the control that would expose a per-parameter
    /// download.
    /// </summary>
    [SkippableFact]
    public void Gpu_eager_step_reads_back_no_gradients_at_any_width()
    {
        using var gpu = new DirectGpuTensorEngine();
        if (Environment.GetEnvironmentVariable("AIDOTNET_REQUIRE_GPU_TESTS") == "1")
            Assert.True(gpu.SupportsGpu && gpu.IsGpuAvailable, "The required GPU backend failed to initialize.");
        Skip.If(!gpu.SupportsGpu || !gpu.IsGpuAvailable, "No DirectGpuTensorEngine available.");

        var narrow = ReadbackTotalsOverWarmSteps(gpu, Hidden);
        var wide = ReadbackTotalsOverWarmSteps(gpu, Hidden * 16);
        string Show((long Count, long Bytes, int Parameters, string Sites) r) =>
            $"{r.Count} readbacks, {r.Bytes} B over {MeasuredSteps} steps and {r.Parameters} parameters ({r.Sites})";
        // The loss is always read back, so a zero count means the probe saw nothing and the checks below are vacuous.
        Assert.True(narrow.Count > 0 && wide.Count > 0,
            $"the readback probe recorded nothing: narrow {Show(narrow)}; wide {Show(wide)}");
        // Totals, not per-step averages, so a step that splits or adds a read cannot hide behind integer division.
        Assert.True(wide.Count <= ScalarReadsPerStep * MeasuredSteps,
            $"a wide step made more than the {ScalarReadsPerStep} scalar reads per step: {Show(wide)}");
        // Each read is a loss or a norm record, never a gradient-sized buffer.
        Assert.True(wide.Bytes < (long)wide.Parameters * sizeof(float),
            $"a wide step read back gradient-sized data: {Show(wide)}");
        Assert.True(wide.Count == narrow.Count && wide.Bytes == narrow.Bytes,
            $"readbacks grew with the model: narrow {Show(narrow)}; wide {Show(wide)}");
    }

    private static (long Count, long Bytes, int Parameters, string Sites) ReadbackTotalsOverWarmSteps(IEngine engine, int hidden)
    {
        var previous = AiDotNetEngine.Current;
        bool priorCapture = GpuLaunchProbe.CaptureReadbackSites;
        AiDotNetEngine.Current = engine;
        try
        {
            var net = BuildNetwork(hidden);
            var (x, y) = Data();
            net.SetTrainingMode(true);
            net.Predict(x);
            net.SetTrainingMode(true);
            for (int s = 0; s < 3; s++) net.Train(x, y);      // warm: device state, kernels and caches exist
            int parameters = net.GetParameters().Length;     // before the probe: it reads every weight

            GpuLaunchProbe.CaptureReadbackSites = true;
            GpuLaunchProbe.Reset();
            for (int s = 0; s < MeasuredSteps; s++)
            {
                long beforeDispatch = GpuLaunchProbe.AdamWDispatches;
                net.Train(x, y);
                Assert.True(GpuLaunchProbe.AdamWDispatches > beforeDispatch, "eager training did not dispatch GPU AdamW");
            }
            return (GpuLaunchProbe.Readbacks, GpuLaunchProbe.ReadbackBytes,
                parameters, string.Join("; ", GpuLaunchProbe.ReadbackSites));
        }
        finally
        {
            GpuLaunchProbe.CaptureReadbackSites = priorCapture;
            AiDotNetEngine.Current = previous;
        }
    }

    private static List<double> Trajectory(IEngine engine, float[] initialParameters, bool withoutParameterBuffer)
    {
        var previous = AiDotNetEngine.Current;
        AiDotNetEngine.Current = engine;
        try
        {
            var net = BuildNetwork();
            var (x, y) = Data();
            net.SetTrainingMode(true);
            net.Predict(x);                                   // materialize the lazy layers
            net.SetTrainingMode(true);
            net.UpdateParameters(new Vector<float>((float[])initialParameters.Clone()));
            if (withoutParameterBuffer) SkipParameterBuffer(net);
            var losses = new List<double>();
            for (int s = 0; s < Steps; s++)
            {
                long beforeDispatch = GpuLaunchProbe.AdamWDispatches;
                net.Train(x, y);
                if (engine is DirectGpuTensorEngine)
                    Assert.True(GpuLaunchProbe.AdamWDispatches > beforeDispatch,
                        $"step {s}: eager training did not dispatch GPU AdamW");
                losses.Add(Convert.ToDouble(net.GetLastLoss()));
            }
            return losses;
        }
        finally
        {
            AiDotNetEngine.Current = previous;
        }
    }

    [SkippableTheory]
    [InlineData(false)]
    [InlineData(true)]
    public void Gpu_eager_step_follows_the_cpu_trajectory(bool withoutParameterBuffer)
    {
        var gpu = new DirectGpuTensorEngine();
        try
        {
            if (Environment.GetEnvironmentVariable("AIDOTNET_REQUIRE_GPU_TESTS") == "1")
                Assert.True(gpu.SupportsGpu && gpu.IsGpuAvailable, "The required GPU backend failed to initialize.");
            Skip.If(!gpu.SupportsGpu || !gpu.IsGpuAvailable, "No DirectGpuTensorEngine available.");

            // One shared initialization for both engines.
            float[] init;
            var previous = AiDotNetEngine.Current;
            AiDotNetEngine.Current = new CpuEngine();
            try
            {
                var seedNet = BuildNetwork();
                seedNet.Predict(Data().X);
                init = seedNet.GetParameters().ToArray();
            }
            finally
            {
                AiDotNetEngine.Current = previous;
            }

            var cpu = Trajectory(new CpuEngine(), init, withoutParameterBuffer);
            var gpuLosses = Trajectory(gpu, init, withoutParameterBuffer);
            string Show(List<double> l) => string.Join(" -> ", l.Select(v => v.ToString("G5")));
            Assert.True(cpu[^1] < 0.9 * cpu[0], $"the CPU reference did not learn: {Show(cpu)}");
            for (int s = 0; s < Steps; s++)
                Assert.True(Math.Abs(gpuLosses[s] - cpu[s]) <= 2e-3 * Math.Max(1.0, Math.Abs(cpu[s])),
                    $"step {s}: GPU {Show(gpuLosses)} vs CPU {Show(cpu)}");
        }
        finally
        {
            gpu.Dispose();
        }
    }
}
