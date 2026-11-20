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
public sealed class NormedConvResidualInferenceTests
{
    private static NormedConv1DLayer<T> Create<T>()
    {
        var layer = new NormedConv1DLayer<T>(2, 3, 3, 1, 1, 1, 1, false, ConvolutionNormalization.Weight);
        layer.LoadTorchWeights(Enumerable.Range(0, 18).Select(i => .1 + .2 * Math.Sin(i * .7)).ToArray(),
            new[] { 1.0, 1.2, .9 }, new[] { -.25, .125, .5 });
        layer.SetTrainingMode(false);
        return layer;
    }

    private static Tensor<T> Values<T>(int channels)
    {
        var ops = AiDotNet.Tensors.Helpers.MathHelper.GetNumericOperations<T>();
        return new Tensor<T>(Enumerable.Range(0, 14 * channels).Select(i => ops.FromDouble(Math.Sin(i * .17) * .5)).ToArray(),
            new[] { 2, channels, 7 });
    }

    private static void CpuFallback<T>(bool writeActivation)
    {
        var previous = AiDotNetEngine.Current;
        AiDotNetEngine.Current = new CpuEngine();
        try
        {
            using var layer = Create<T>();
            using var input = Values<T>(2);
            using var residual = Values<T>(3);
            T alpha = AiDotNet.Tensors.Helpers.MathHelper.GetNumericOperations<T>().FromDouble(.1);
            var expected = AiDotNetEngine.Current.TensorAdd(layer.Forward(input), residual);
            var actual = layer.ForwardResidualInference(input, residual, alpha, writeActivation, out var activated);
            Assert.Equal(expected.ToArray(), actual.ToArray());
            if (writeActivation)
                Assert.Equal(AiDotNetEngine.Current.LeakyReLU(expected, alpha).ToArray(), Assert.IsType<Tensor<T>>(activated).ToArray());
            else Assert.Null(activated);
        }
        finally { AiDotNetEngine.Current = previous; }
    }

    [Theory] [InlineData(false)] [InlineData(true)]
    public void Float_cpu_preserves_ordinary_composition(bool writeActivation) => CpuFallback<float>(writeActivation);
    [Theory] [InlineData(false)] [InlineData(true)]
    public void Double_cpu_preserves_ordinary_composition(bool writeActivation) => CpuFallback<double>(writeActivation);

    private static float[][] GradientReplay(bool explicitEntry, bool compile)
    {
        using var layer = Create<float>();
        using var input = Values<float>(2);
        using var residual = Values<float>(3);
        var parameters = layer.GetTrainableParameters().ToArray();
        var engine = AiDotNetEngine.Current;
        Tensor<float> Objective()
        {
            Tensor<float> sum, activated;
            if (explicitEntry)
            {
                sum = layer.ForwardResidualInference(input, residual, .1f, true, out var carry);
                activated = Assert.IsType<Tensor<float>>(carry);
            }
            else
            {
                sum = engine.TensorAdd(layer.Forward(input), residual);
                activated = engine.LeakyReLU(sum, .1f);
            }
            return engine.ReduceSum(engine.TensorAdd(sum, activated));
        }
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
    public void Tape_and_compiled_graph_keep_both_output_gradients_and_updates(bool compile)
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

    [Theory] [InlineData(false)] [InlineData(true)]
    public void Cuda_saves_launches_and_preserves_unread_carries_across_parameter_updates(bool writeActivation)
    {
        using var gpu = new DirectGpuTensorEngine();
        Assert.True(gpu.IsGpuAvailable && gpu.GetBackend() is CudaBackend, "Actual CUDA required.");
        var previous = AiDotNetEngine.Current;
        AiDotNetEngine.Current = gpu;
        try
        {
            using var layer = Create<float>();
            using var input = Values<float>(2);
            using var residual = Values<float>(3);
            var originalInput = input.ToArray();
            var originalResidual = residual.ToArray();
            var unread = new List<(Tensor<float> Sum, Tensor<float>? Activation, float[] Expected, float[]? ExpectedActivation)>();
            for (int iteration = 0; iteration < 3; ++iteration)
            {
                if (iteration > 0)
                    foreach (var parameter in layer.GetTrainableParameters())
                    {
                        parameter[0] += .03125f;
                        gpu.InvalidatePersistentTensor(parameter);
                    }
                using var warm = gpu.TensorAdd(layer.Forward(input), residual);
                warm.ToArray();
                using var warmFused = layer.ForwardResidualInference(input, residual, .1f, writeActivation, out var warmActivation);
                warmFused.ToArray();
                warmActivation?.Dispose();
                long before = GpuLaunchProbe.Count;
                using var ordinary = gpu.TensorAdd(layer.Forward(input), residual);
                using var ordinaryActivation = writeActivation ? gpu.LeakyReLU(ordinary, .1f) : null;
                long ordinaryLaunches = GpuLaunchProbe.Count - before;
                float[] expected = ordinary.ToArray();
                float[]? expectedActivation = ordinaryActivation?.ToArray();
                before = GpuLaunchProbe.Count;
                var sum = layer.ForwardResidualInference(input, residual, .1f, writeActivation, out var activated);
                bool hasFused = typeof(DirectGpuTensorEngine).GetMethod("TryChannelBiasResidualInference",
                    System.Reflection.BindingFlags.Instance | System.Reflection.BindingFlags.NonPublic) is not null;
                long expectedLaunches = hasFused ? ordinaryLaunches - (writeActivation ? 2 : 1) : ordinaryLaunches;
                Assert.Equal(expectedLaunches, GpuLaunchProbe.Count - before);
                if (activated is not null) Assert.False(sum.SharesStorageWith(activated));
                unread.Add((sum, activated, expected, expectedActivation));
                using (GraphCaptureCompatibility.Enter())
                {
                    before = GpuLaunchProbe.Count;
                    using var guarded = layer.ForwardResidualInference(input, residual, .1f, writeActivation, out var guardedActivation);
                    Assert.Equal(ordinaryLaunches, GpuLaunchProbe.Count - before);
                    Assert.Equal(expected, guarded.ToArray());
                    guardedActivation?.Dispose();
                }
            }
            GC.Collect(); GC.WaitForPendingFinalizers(); GC.Collect();
            foreach (var item in unread)
            {
                Assert.Equal(item.Expected.Select(BitConverter.SingleToInt32Bits), item.Sum.ToArray().Select(BitConverter.SingleToInt32Bits));
                if (item.Activation is not null)
                    Assert.Equal(item.ExpectedActivation!.Select(BitConverter.SingleToInt32Bits), item.Activation.ToArray().Select(BitConverter.SingleToInt32Bits));
                item.Sum.Dispose(); item.Activation?.Dispose();
            }
            Assert.Equal(originalInput, input.ToArray());
            Assert.Equal(originalResidual, residual.ToArray());
        }
        finally { AiDotNetEngine.Current = previous; }
    }
}
