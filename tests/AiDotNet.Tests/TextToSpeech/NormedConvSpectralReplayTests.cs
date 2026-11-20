using System.Reflection;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.TextToSpeech;

[Collection("FusedTrainingSerial")]
public sealed class NormedConvSpectralReplayTests
{
    [Fact]
    public void Float_cpu_probes_preserve_state_and_replay_the_eager_trajectory()
        => Check<float>(new CpuEngine(), 5e-5);

    [Fact]
    public void Double_cpu_probes_preserve_state_and_replay_the_eager_trajectory()
        => Check<double>(new CpuEngine(), 1e-9);

    [Fact]
    [Trait("Category", "GPU")]
    public void Cuda_probes_preserve_state_and_replay_the_eager_trajectory()
    {
        using var engine = new DirectGpuTensorEngine();
        Assert.True(engine.IsGpuAvailable && engine.GetBackend() is CudaBackend, "Actual CUDA required.");
        Check<float>(engine, 5e-5);
    }

    private static void Check<T>(IEngine engine, double tolerance)
    {
        var previous = AiDotNetEngine.Current;
        AiDotNetEngine.Current = engine;
        try
        {
            var ops = MathHelper.GetNumericOperations<T>();
            Tensor<T> Field(NormedConv1DLayer<T> layer, string name) =>
                (Tensor<T>)PrivateState.Required(typeof(NormedConv1DLayer<T>), name, layer);
            NormedConv1DLayer<T> Create()
            {
                var layer = new NormedConv1DLayer<T>(2, 2, 1, 1, 1, 1, 0, false, ConvolutionNormalization.Spectral);
                layer.SetTrainingMode(true);
                foreach (var (name, values) in new[]
                {
                    ("_direction", new[] { 1.0, 2.0, 3.0, -1.0 }),
                    ("_u", new[] { 1.0, 0.0 }), ("_v", new[] { 0.0, 1.0 })
                })
                {
                    var field = Field(layer, name);
                    for (int i = 0; i < values.Length; i++) field[i] = ops.FromDouble(values[i]);
                    engine.InvalidatePersistentTensor(field);
                }
                return layer;
            }

            using var eagerFirst = Create();
            using var eagerSecond = Create();
            using var compiledFirst = Create();
            using var compiledSecond = Create();
            var parameters = new[] { Field(compiledFirst, "_direction"), Field(compiledSecond, "_direction") };
            var eagerParameters = new[] { Field(eagerFirst, "_direction"), Field(eagerSecond, "_direction") };
            Tensor<T> Objective(NormedConv1DLayer<T> first, NormedConv1DLayer<T> second)
            {
                // The second layer and repeated first layer expose position drift after a state update.
                var a = engine.ReduceSum(first.Kernel(), null, false);
                var b = engine.ReduceSum(second.Kernel(), null, false);
                var c = engine.ReduceSum(first.Kernel(), null, false);
                return engine.TensorAdd(engine.TensorAdd(a, b), c);
            }
            Tensor<T>[] State() => new[] { compiledFirst, compiledSecond }
                .SelectMany(layer => new[] { Field(layer, "_direction"), Field(layer, "_u"), Field(layer, "_v") }).ToArray();

            using var recording = ClosureObjectiveGuard<T>.BeginRecording();
            CompiledTrainingPlan<T> plan;
            using (var scope = GraphMode.EnableTraining(parameters))
                plan = scope.CompileTraining(parameters, Objective(compiledFirst, compiledSecond));
            using (plan)
            {
                var guard = ClosureObjectiveGuard<T>.TakeRecorded(out var reason);
                Assert.True(guard is not null, reason);
                PrivateState.NotNull(guard, "the replay guard").AttachTo(plan);
                plan.DisableGraphStep();
                plan.ConfigureOptimizer(OptimizerType.SGD, learningRate: .01);
                for (int step = 0; step < 3; step++)
                {
                    var before = State().Select(tensor => tensor.ToArray()).ToArray();
                    for (int probe = 0; probe < 2; probe++)
                    {
                        Assert.True(guard.TryRebind(() => Objective(compiledFirst, compiledSecond), parameters, out _) == ClosureGuardResult.Matched,
                            guard.LastReason);
                        var after = State();
                        for (int i = 0; i < before.Length; i++) Assert.Equal(before[i], after[i].ToArray());
                    }

                    T expected;
                    Dictionary<Tensor<T>, Tensor<T>> gradients;
                    using (var tape = new GradientTape<T>())
                    {
                        var loss = Objective(eagerFirst, eagerSecond);
                        expected = loss.ToArray()[0];
                        gradients = tape.ComputeGradients(loss, sources: eagerParameters);
                    }
                    var actual = plan.Step().ToArray()[0];
                    Assert.InRange(Math.Abs(ops.ToDouble(actual) - ops.ToDouble(expected)), 0, tolerance);
                    foreach (var parameter in eagerParameters)
                    {
                        var gradient = gradients[parameter].ToArray();
                        for (int i = 0; i < gradient.Length; i++)
                            parameter[i] = ops.Subtract(parameter[i], ops.Multiply(ops.FromDouble(.01), gradient[i]));
                        engine.InvalidatePersistentTensor(parameter);
                    }
                    foreach (var pair in new[] { (eagerFirst, compiledFirst), (eagerSecond, compiledSecond) })
                        foreach (var name in new[] { "_direction", "_u", "_v" })
                        {
                            var a = Field(pair.Item1, name).ToArray();
                            var b = Field(pair.Item2, name).ToArray();
                            for (int i = 0; i < a.Length; i++)
                                Assert.InRange(Math.Abs(ops.ToDouble(a[i]) - ops.ToDouble(b[i])), 0, tolerance);
                        }
                }
            }
        }
        finally { AiDotNetEngine.Current = previous; }
    }
}
