using AiDotNet.ActivationFunctions;
using AiDotNet.Engines;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.Engines.Diagnostics;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.Engines.DirectGpu.OpenCL;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.Training;

[Collection("NonParallelIntegration")]
public class RemainingLstmIntegrationTests
{
    // The production LSTM cell (LstmGateDispatch -> IEngine.LstmCellFromGates) that replaced the legacy
    // graph LSTMCell. x = [input, h, c, W, U, b].
    private static (Tensor<TNum> H, Tensor<TNum> C) Cell<TNum>(IEngine e, Tensor<TNum>[] x)
    {
        var gates = e.TensorAdd(e.TensorAdd(e.TensorMatMul(x[0], x[3]), e.TensorMatMul(x[1], x[4])), x[5]);
        var h = AiDotNet.Helpers.LstmGateDispatch<TNum>.Forward(e, gates, x[2], out var c);
        return (h, c);
    }
    [Fact]
    [Trait("Category", "GPU")]
    public void OpenCl_FusedCellKeepsBothStateAdjointsOnDevice()
    {
        using var backend = new OpenClBackend(); Assert.True(backend.IsAvailable, backend.InitializationException?.ToString());
        using var direct = new DirectGpuEngine(backend); using var gpu = new DirectGpuTensorEngine(direct);
        var prior = AiDotNetEngine.Current;
        try
        {
            var values = new[] { new[] { 2, 3 }, new[] { 2, 2 }, new[] { 2, 2 }, new[] { 3, 8 }, new[] { 2, 8 }, new[] { 8 } }
                .Select(shape => { var x = new Tensor<float>(shape); for (int i = 0; i < x.Length; i++) x[i] = MathF.Sin(i + .3f) * .4f; return x; }).ToArray();
            Dictionary<Tensor<float>, Tensor<float>> Run(IEngine e)
            {
                using var tape = new GradientTape<float>();
                var (h, c) = Cell(e, values);
                return tape.ComputeGradients(e.ReduceSum(e.TensorAdd(h, c), null), values);
            }
            var cpu = new CpuEngine(); AiDotNetEngine.Current = cpu; var expected = Run(cpu);
            AiDotNetEngine.Current = gpu;
            Dictionary<Tensor<float>, Tensor<float>> actual;
            using (var scope = GpuResidencyScope.Begin(captureOperations: true))
            {
                actual = Run(gpu);
                Assert.True(scope.Downloads == 0, string.Join("\n", scope.Events.Where(x => x.Kind == GpuTransferKind.DeviceToHost)));
            }
            for (int j = 0; j < values.Length; j++)
                for (int i = 0; i < values[j].Length; i++)
                    Assert.InRange(MathF.Abs(expected[values[j]][i] - actual[values[j]][i]), 0, 2e-5f);
        }
        finally { AiDotNetEngine.Current = prior; }
    }
    [Theory] [InlineData(0)] [InlineData(1)] [InlineData(2)] [InlineData(3)] [InlineData(4)] [InlineData(5)]
    [Trait("Category", "GPU")]
    public void OpenCl_ConsumersTrainWithoutHostReadbacks(int kind)
    {
        using var backend = new OpenClBackend(); Assert.True(backend.IsAvailable, backend.InitializationException?.ToString());
        using var direct = new DirectGpuEngine(backend); using var gpu = new DirectGpuTensorEngine(direct);
        var prior = AiDotNetEngine.Current; var cpu = new CpuEngine(); AiDotNetEngine.Current = cpu;
        try
        {
            LayerBase<float> layer = kind == 0 ? new AiDotNet.NeuralNetworks.Layers.SSM.ExtendedLSTMLayer<float>(4, 4, 2)
                : kind == 1 ? new LSTMLayer<float>(2, (IActivationFunction<float>)new ReLUActivation<float>())
                : kind == 4 ? new LSTMLayer<float>(2)
                : kind == 5 ? new AiDotNet.NeuralNetworks.Layers.SSM.MinLSTMLayer<float>(4, 4)
                : new ConvLSTMLayer<float>(new[] { 2, 3, 3, 2 }, 3, 2, 1, 1,
                    kind == 2 ? (IActivationFunction<float>)new ReLUActivation<float>() : null);
            var input = new Tensor<float>(kind == 2 || kind == 3 ? new[] { 1, 2, 3, 3, 2 } : new[] { 2, 3, 4 });
            for (int i = 0; i < input.Length; i++) input[i] = MathF.Sin(i * .4f) * .2f;
            layer.Forward(input);
            var sources = new[] { input }.Concat(layer.GetTrainableParameters()).ToArray();
            Tensor<float> reference; Dictionary<Tensor<float>, Tensor<float>> expected;
            using (var tape = new GradientTape<float>())
            {
                reference = layer.Forward(input);
                expected = tape.ComputeGradients(cpu.ReduceSum(cpu.TensorMultiply(reference, reference), null), sources);
            }
            AiDotNetEngine.Current = gpu;
            Tensor<float> output; Dictionary<Tensor<float>, Tensor<float>> gradients;
            using (var scope = GpuResidencyScope.Begin(captureOperations: true))
            {
                using var tape = new GradientTape<float>();
                output = layer.Forward(input);
                gradients = tape.ComputeGradients(gpu.ReduceSum(gpu.TensorMultiply(output, output), null), sources);
                Assert.True(scope.Downloads == 0, string.Join("\n", scope.Events.Where(x => x.Kind == GpuTransferKind.DeviceToHost)));
            }
            void Compare(Tensor<float> a, Tensor<float> b)
            { for (int i = 0; i < a.Length; i++) Assert.InRange(MathF.Abs(a[i] - b[i]), 0, 8e-4f * (1 + MathF.Abs(a[i]))); }
            Compare(reference, output);
            foreach (var source in sources) Compare(expected[source], gradients[source]);
        }
        finally { AiDotNetEngine.Current = prior; }
    }
    private static Tensor<double> Values(params int[] shape)
    {
        var x = new Tensor<double>(shape);
        for (int i = 0; i < x.Length; i++) x[i] = Math.Sin(i + .3) * .4;
        return x;
    }
    private static void Near(Tensor<double> a, Tensor<double> b)
    {
        Assert.Equal(a.Shape, b.Shape);
        for (int i = 0; i < a.Length; i++) Assert.InRange(Math.Abs(a[i] - b[i]), 0, 1e-9);
    }
    [Theory] [InlineData(1)] [InlineData(2)] [InlineData(3)]
    public void FusedCell_CombinesIndependentStateGradients(int mask)
    {
        var prior = AiDotNetEngine.Current; var e = new CpuEngine(); AiDotNetEngine.Current = e;
        try
        {
            var x = new[] { Values(2, 3), Values(2, 2), Values(2, 2), Values(3, 8), Values(2, 8), Values(8) };
            Tensor<double> root; Dictionary<Tensor<double>, Tensor<double>> fused;
            using (var fusedTape = new GradientTape<double>())
            {
                var (h, c) = Cell(e, x);
                root = mask == 1 ? h : mask == 2 ? c : e.TensorAdd(h, c);
                fused = fusedTape.ComputeGradients(e.ReduceSum(root, null), x);
            }
            using var tape = new GradientTape<double>();
            var gates = e.TensorAdd(e.TensorAdd(e.TensorMatMul(x[0], x[3]), e.TensorMatMul(x[1], x[4])), x[5]);
            Tensor<double> Gate(int j) => e.TensorNarrow(gates, 1, 2 * j, 2);
            var rc = e.TensorAdd(e.TensorMultiply(e.Sigmoid(Gate(1)), x[2]), e.TensorMultiply(e.Sigmoid(Gate(0)), e.Tanh(Gate(2))));
            var rh = e.TensorMultiply(e.Sigmoid(Gate(3)), e.Tanh(rc));
            var output = mask == 1 ? rh : mask == 2 ? rc : e.TensorAdd(rh, rc);
            var grads = tape.ComputeGradients(e.ReduceSum(output, null), x);
            Near(output, root);
            for (int j = 0; j < x.Length; j++) Near(grads[x[j]], fused[x[j]]);
        }
        finally { AiDotNetEngine.Current = prior; }
    }
    [Fact]
    public void CustomLstm_UsesConfiguredActivationAndCorrectGradients()
    {
        var prior = AiDotNetEngine.Current; var e = new CpuEngine(); AiDotNetEngine.Current = e;
        try
        {
            var layer = new LSTMLayer<double>(1, (IActivationFunction<double>)new IdentityActivation<double>(), new IdentityActivation<double>());
            var input = Values(1, 3, 1);
            layer.Forward(input);
            var parameters = layer.GetParameters(); for (int i = 0; i < parameters.Length; i++) parameters[i] = .1;
            layer.SetParameters(parameters);
            using var tape = new GradientTape<double>();
            var output = layer.Forward(input);
            double h = 0, c = 0;
            for (int t = 0; t < 3; t++)
            {
                double gate = .1 * input[t] + .1 * h + .1;
                c = gate * c + gate * gate; h = gate * c;
                Assert.InRange(Math.Abs(h - output[t]), 0, 1e-12);
            }
            var sources = new[] { input }.Concat(layer.GetTrainableParameters()).ToArray();
            var gradients = tape.ComputeGradients(e.ReduceSum(output, null), sources);
            foreach (var source in sources)
            {
                double old = source[0], eps = 1e-6;
                source[0] = old + eps; double plus = e.ReduceSum(layer.Forward(input), null)[0];
                source[0] = old - eps; double minus = e.ReduceSum(layer.Forward(input), null)[0];
                source[0] = old;
                Assert.InRange(Math.Abs((plus - minus) / (2 * eps) - gradients[source][0]), 0, 1e-8);
            }
        }
        finally { AiDotNetEngine.Current = prior; }
    }

    [Theory]
    [InlineData(false)] [InlineData(true)]
    [Trait("Category", "GPU")]
    public void OpenCl_MemoryControllersKeepCellGradientsOnDevice(bool dnc)
    {
        using var backend = new OpenClBackend(); Assert.True(backend.IsAvailable);
        using var direct = new DirectGpuEngine(backend); using var gpu = new DirectGpuTensorEngine(direct);
        var prior = AiDotNetEngine.Current;
        try
        {
            var cpu = new CpuEngine(); AiDotNetEngine.Current = cpu;
            Tensor<float> Make(params int[] shape)
            {
                var t = new Tensor<float>(shape);
                for (int i = 0; i < t.Length; i++) t[i] = MathF.Sin(i + .3f) * .2f;
                return t;
            }
            Func<Tensor<float>> forward;
            Tensor<float>[] sources;
            var flags = System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance;
            if (dnc)
            {
                var model = new AiDotNet.NeuralNetworks.DifferentiableNeuralComputer<float>(
                    new AiDotNet.NeuralNetworks.NeuralNetworkArchitecture<float>(
                        inputType: AiDotNet.Enums.InputType.OneDimensional, taskType: AiDotNet.Enums.NeuralNetworkTaskType.Regression,
                        inputSize: 2, outputSize: 1), memorySize: 3, memoryWordSize: 2, controllerSize: 2, readHeads: 1,
                    activationFunction: (IActivationFunction<float>?)null);
                var method = model.GetType().GetMethod("ApplyControllerCell", flags)!;
                var state = model.GetType().GetField("_controllerCell", flags)!;
                var gates = Make(1, 8); var old = Make(1, 2); sources = new[] { gates };
                forward = () =>
                {
                    state.SetValue(model, old);
                    var first = (Tensor<float>)method.Invoke(model, new object[] { gates })!;
                    var second = (Tensor<float>)method.Invoke(model, new object[] { gates })!;
                    return AiDotNetEngine.Current.TensorAdd(AiDotNetEngine.Current.TensorAdd(first, second),
                        (Tensor<float>)state.GetValue(model)!);
                };
            }
            else
            {
                var controller = new AiDotNet.MetaLearning.Algorithms.LSTMNTMController<float, Matrix<float>, Vector<float>>(
                    new AiDotNet.MetaLearning.Options.NTMOptions<float, Matrix<float>, Vector<float>>(
                        new AiDotNet.Regression.SimpleRegression<float>())
                    { ControllerHiddenSize = 3, MemoryWidth = 2, NumReadHeads = 1, RandomSeed = 42 });
                var input = Make(4); var read = new List<Tensor<float>>(); controller.Forward(input, read);
                sources = new[] { input }.Concat(new[] { "_weightsInput", "_weightsHidden", "_biases" }
                    .Select(name => (Tensor<float>)controller.GetType().GetField(name, flags)!.GetValue(controller)!)).ToArray();
                forward = () =>
                {
                    controller.Reset(); var first = controller.Forward(input, read); var second = controller.Forward(input, read);
                    return AiDotNetEngine.Current.TensorConcatenate(new[] { first, second }, 0);
                };
            }
            (Tensor<float> Output, Dictionary<Tensor<float>, Tensor<float>> Gradients) Run()
            {
                using var tape = new GradientTape<float>(); var output = forward();
                var loss = AiDotNetEngine.Current.ReduceSum(output, null);
                return (output, tape.ComputeGradients(loss, sources));
            }
            var expected = Run(); AiDotNetEngine.Current = gpu;
            (Tensor<float> Output, Dictionary<Tensor<float>, Tensor<float>> Gradients) actual;
            using (var scope = GpuResidencyScope.Begin(captureOperations: true))
            {
                actual = Run();
                Assert.True(scope.Downloads == 0, string.Join("\n", scope.Events));
            }
            void Compare(Tensor<float> a, Tensor<float> b)
            { for (int i = 0; i < a.Length; i++) Assert.InRange(MathF.Abs(a[i] - b[i]), 0, 5e-5f); }
            Compare(expected.Output, actual.Output);
            foreach (var source in sources) Compare(expected.Gradients[source], actual.Gradients[source]);
        }
        finally { AiDotNetEngine.Current = prior; }
    }
}
