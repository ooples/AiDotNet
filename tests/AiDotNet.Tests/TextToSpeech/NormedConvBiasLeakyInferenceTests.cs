using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA;
using Xunit;

namespace AiDotNet.Tests.TextToSpeech;

[Collection("FusedTrainingSerial")]
public sealed class NormedConvBiasLeakyInferenceTests
{
    private static NormedConv1DLayer<T> Create<T>()
    {
        var layer = new NormedConv1DLayer<T>(2, 3, 3, 1, 1, 1, 1, false, ConvolutionNormalization.Weight);
        layer.LoadTorchWeights(Enumerable.Range(0, 18).Select(i => .1 + .2 * Math.Sin(i * .7)).ToArray(),
            new[] { 1.0, 1.2, .9 }, new[] { -.25, .125, .5 });
        layer.SetTrainingMode(false);
        return layer;
    }

    private static Tensor<T> Input<T>()
    {
        var ops = AiDotNet.Tensors.Helpers.MathHelper.GetNumericOperations<T>();
        return new Tensor<T>(Enumerable.Range(0, 28).Select(i => ops.FromDouble(Math.Sin(i * .17) * .5)).ToArray(), new[] { 2, 2, 7 });
    }

    private static void CpuFallback<T>()
    {
        var previous = AiDotNetEngine.Current;
        AiDotNetEngine.Current = new CpuEngine();
        try
        {
            using var layer = Create<T>();
            using var input = Input<T>();
            T alpha = AiDotNet.Tensors.Helpers.MathHelper.GetNumericOperations<T>().FromDouble(.1);
            var expected = AiDotNetEngine.Current.LeakyReLU(layer.Forward(input), alpha).ToArray();
            Assert.Equal(expected, layer.ForwardLeakyInference(input, alpha).ToArray());
        }
        finally { AiDotNetEngine.Current = previous; }
    }

    [Fact] public void Float_cpu_keeps_the_ordinary_public_forward() => CpuFallback<float>();
    [Fact] public void Double_cpu_keeps_the_ordinary_public_forward() => CpuFallback<double>();

    private static float[][] GradientReplay(bool explicitEntry, bool compile)
    {
        using var layer = Create<float>();
        using var input = Input<float>();
        var parameters = layer.GetTrainableParameters().ToArray();
        var engine = AiDotNetEngine.Current;
        Tensor<float> Objective() => engine.ReduceSum(explicitEntry ? layer.ForwardLeakyInference(input, .1f)
            : engine.LeakyReLU(layer.Forward(input), .1f));
        if (!compile)
        {
            using var tape = new GradientTape<float>();
            var loss = Objective();
            var gradients = tape.ComputeGradients(loss, parameters);
            return new[] { loss.ToArray() }.Concat(parameters.Select(parameter => gradients[parameter].ToArray())).ToArray();
        }
        CompiledTrainingPlan<float> plan;
        using (var scope = GraphMode.EnableTraining(parameters)) plan = scope.CompileTraining(parameters, Objective());
        using (plan)
        {
            plan.ConfigureOptimizer(OptimizerType.SGD, learningRate: .01);
            var values = new List<float[]>();
            for (int step = 0; step < 3; ++step)
            {
                values.Add(plan.Step().ToArray());
                values.AddRange(plan.Gradients.Select(gradient => gradient.ToArray()));
                values.AddRange(parameters.Select(parameter => parameter.ToArray()));
            }
            return values.ToArray();
        }
    }

    [Theory] [InlineData(false)] [InlineData(true)]
    public void Tape_and_compiled_graph_preserve_all_gradients_and_updates(bool compile)
    {
        var previous = AiDotNetEngine.Current;
        AiDotNetEngine.Current = new CpuEngine();
        try
        {
            var expected = GradientReplay(false, compile);
            var actual = GradientReplay(true, compile);
            Assert.Equal(expected.Length, actual.Length);
            for (int index = 0; index < actual.Length; ++index) Assert.Equal(expected[index], actual[index]);
        }
        finally { AiDotNetEngine.Current = previous; }
    }

    private sealed class CustomLayer : NormedConv1DLayer<float>
    {
        internal int Calls;
        internal CustomLayer() : base(2, 3, 3, 1, 1, 1, 1, false, ConvolutionNormalization.None) { }
        protected override Tensor<float> ForwardTraced(Tensor<float> input) { ++Calls; return base.ForwardTraced(input); }
    }

    [Fact]
    public void Custom_layer_override_is_called()
    {
        var previous = AiDotNetEngine.Current;
        AiDotNetEngine.Current = new CpuEngine();
        try
        {
            using var layer = new CustomLayer(); layer.SetTrainingMode(false);
            using var input = Input<float>();
            layer.ForwardLeakyInference(input, .1f);
            Assert.Equal(1, layer.Calls);
        }
        finally { AiDotNetEngine.Current = previous; }
    }

    [SkippableFact]
    public void Cuda_removes_one_launch_and_keeps_outputs_after_parameter_updates()
    {
        using var gpu = new DirectGpuTensorEngine();
        Skip.IfNot(gpu.IsGpuAvailable && gpu.GetBackend() is CudaBackend, "CUDA unavailable.");
        var previous = AiDotNetEngine.Current;
        AiDotNetEngine.Current = gpu;
        try
        {
            using var layer = Create<float>();
            using var input = Input<float>();
            for (int iteration = 0; iteration < 3; ++iteration)
            {
                if (iteration > 0)
                    foreach (var parameter in layer.GetTrainableParameters())
                    {
                        parameter[0] += .03125f;
                        gpu.InvalidatePersistentTensor(parameter);
                    }
                // Both measured paths must see already-normalized parameters and warm native kernels.
                using (var warmed = gpu.LeakyReLU(layer.Forward(input), .1f)) warmed.ToArray();
                using (var warmedFused = layer.ForwardLeakyInference(input, .1f)) warmedFused.ToArray();
                long before = GpuLaunchProbe.Count;
                using var ordinary = gpu.LeakyReLU(layer.Forward(input), .1f);
                long ordinaryLaunches = GpuLaunchProbe.Count - before;
                float[] expected = ordinary.ToArray();
                before = GpuLaunchProbe.Count;
                using var fused = layer.ForwardLeakyInference(input, .1f);
                long fusedLaunches = GpuLaunchProbe.Count - before;
                Assert.Equal(ordinaryLaunches - 1, fusedLaunches);
                Assert.Equal(expected.Select(BitConverter.SingleToInt32Bits), fused.ToArray().Select(BitConverter.SingleToInt32Bits));
                using (GraphCaptureCompatibility.Enter())
                {
                    before = GpuLaunchProbe.Count;
                    using var guarded = layer.ForwardLeakyInference(input, .1f);
                    Assert.Equal(ordinaryLaunches, GpuLaunchProbe.Count - before);
                    Assert.Equal(expected.Select(BitConverter.SingleToInt32Bits), guarded.ToArray().Select(BitConverter.SingleToInt32Bits));
                }
            }
        }
        finally { AiDotNetEngine.Current = previous; }
    }
}
