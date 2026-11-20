using System.Collections;
using System.Reflection;
using AiDotNet.Interfaces;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Optimizers;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Training;
using Xunit;

namespace AiDotNet.Tests.UnitTests.Training;

[Collection("FoundationScaleSerial")]
public sealed class FusedPersistenceOwnershipTests
{
    private const BindingFlags PrivateStatic = BindingFlags.Static | BindingFlags.NonPublic;

    private static StochasticGradientDescentOptimizer<float, Tensor<float>, Tensor<float>> Sgd()
        => new(null, new StochasticGradientDescentOptimizerOptions<float, Tensor<float>, Tensor<float>>
        { InitialLearningRate = .01, UseAdaptiveLearningRate = false });

    private static Tensor<float> Loss(Tensor<float> actual, Tensor<float> expected)
    {
        var e = AiDotNetEngine.Current;
        var delta = e.TensorSubtract(actual, expected);
        return e.TensorMultiplyScalar(e.ReduceSum(e.TensorMultiply(delta, delta), null), 1f / delta.Length);
    }

    private static long NativeSequence() => (long)StaticField(typeof(GpuMemoryTracker), "s_seq");

    private static object[] LiveProbeAllocations(long afterSequence)
    {
        var live = (IEnumerable)StaticField(typeof(GpuMemoryTracker), "s_live");
        var records = new List<object>();
        foreach (object pair in live)
        {
            object record = Member(pair, "Value");
            long sequence = (long)Member(record, "Seq");
            string site = (string)Member(record, "Site");
            // Sampling may upload a caller-owned parameter for the first time. Its input buffer
            // stays live; only the newly computed copy belongs to the persistence probe.
            if (sequence > afterSequence && site.Contains(".SampleParameters")
                && site.Contains("AllocateFullyWrittenOutputBuffer")) records.Add(record);
        }
        return records.ToArray();
    }

    private static object StaticField(Type type, string name)
    {
        var field = type.GetField(name, PrivateStatic)
            ?? throw new Xunit.Sdk.XunitException(type.Name + " has no private static field '" + name + "'.");
        return field.GetValue(null) ?? throw new Xunit.Sdk.XunitException(type.Name + "." + name + " is null.");
    }

    // The value of a public field or property that must be present.
    private static object Member(object owner, string name)
    {
        var type = owner.GetType();
        object? value;
        if (type.GetField(name) is { } field) value = field.GetValue(owner);
        else if (type.GetProperty(name) is { } property) value = property.GetValue(owner);
        else throw new Xunit.Sdk.XunitException(type.Name + " has no public member '" + name + "'.");
        return value ?? throw new Xunit.Sdk.XunitException(type.Name + "." + name + " is null.");
    }

    private static MethodInfo SessionMethod(string name) =>
        typeof(FusedTrainingSession<float>).GetMethod(name, PrivateStatic)
        ?? throw new Xunit.Sdk.XunitException("FusedTrainingSession has no private static method '" + name + "'.");

    private static object Call(MethodInfo method, object[] arguments) =>
        method.Invoke(null, arguments) ?? throw new Xunit.Sdk.XunitException(method.Name + " returned null.");

    private static T Present<T>(T? value, string what) where T : class =>
        value ?? throw new Xunit.Sdk.XunitException("Expected " + what + " to be present.");

    private static object? Read(object? obj, string name)
    {
        if (obj is null) return null;
        const BindingFlags flags = BindingFlags.Instance | BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.DeclaredOnly;
        for (Type? type = obj.GetType(); type is not null; type = type.BaseType)
        {
            if (type.GetField(name, flags) is { } field) return field.GetValue(obj);
            if (type.GetProperty(name, flags) is { } property) return property.GetValue(obj);
        }
        return null;
    }

    private static Tensor<float> Snapshot(object sample)
    {
        var values = (Array)Present(Read(sample, "Samples"), "the parameter samples");
        object first = Present(values.GetValue(0), "the first parameter sample");
        return (Tensor<float>)Member(first, "Item3");
    }

    private static void AssertNoProbeAllocations(long sequence, string metadata = "")
    {
        var records = LiveProbeAllocations(sequence);
        var sites = records.Select(r => (string)Member(r, "Site"));
        Assert.True(records.Length == 0, string.Join(Environment.NewLine, sites) + metadata);
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void Probe_owns_snapshots_preserves_sources_and_releases_native_storage(bool gpu)
    {
        var previous = AiDotNetEngine.Current;
        using var device = gpu ? new DirectGpuTensorEngine() : null;
        if (gpu) Assert.True(Present(device, "the GPU engine").IsGpuAvailable, "Native GPU required; this test must execute.");
        AiDotNetEngine.Current = gpu ? Present(device, "the GPU engine") : new CpuEngine();
        try
        {
            if (gpu) Assert.True(GpuMemoryTracker.Enabled, "Enable native allocation tracking before test process startup.");
            long sequence = gpu ? NativeSequence() : 0;
            using var source = new Tensor<float>(new[] { 1f, 2f, 3f, 4f }, new[] { 4 });
            var request = new FusedTrainingStepRequest<float>
            {
                Layers = Array.Empty<ITrainableLayer<float>>(), Input = source, Target = source,
                Forward = x => x, ComputeLoss = Loss, Optimizer = Sgd(), Selection = new[] { source },
            };
            var sample = Call(SessionMethod("SampleParameters"), new object[] { request });
            using var owner = Assert.IsAssignableFrom<IDisposable>(sample);
            var snapshot = Snapshot(sample);
            var buffer = Read(snapshot, "_gpuBuffer");
            var changed = SessionMethod("AnySampleChanged");
            Assert.False((bool)Call(changed, new[] { sample }));
            source[0] = 9f;
            Assert.True((bool)Call(changed, new[] { sample }));
            owner.Dispose();
            owner.Dispose();
            Assert.Equal(new[] { 9f, 2f, 3f, 4f }, source.ToArray());
            if (gpu)
            {
                Assert.Null(Read(Read(snapshot.DataVector, "_deviceState"), "Buffer"));
                Assert.Null(Read(snapshot, "_gpuBuffer"));
                // The native allocation can remain cached; ownership must return to the reuse pool.
                Assert.Equal(1, (int)Present(Read(buffer, "_poolState"), "the pool state"));
                long warmedSequence = NativeSequence();
                for (int repeat = 0; repeat < 6; repeat++)
                {
                    using var next = (IDisposable)Call(SessionMethod("SampleParameters"), new object[] { request });
                    Assert.False((bool)Call(changed, new object[] { next }));
                }
                Assert.Equal(warmedSequence, NativeSequence());
            }
        }
        finally { AiDotNetEngine.Current = previous; }
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void Completed_fused_steps_release_probe_storage_and_match_eager_updates(bool gpu)
    {
        var previous = AiDotNetEngine.Current;
        using var device = gpu ? new DirectGpuTensorEngine() : null;
        if (gpu) Assert.True(Present(device, "the GPU engine").IsGpuAvailable, "Native GPU required; this test must execute.");
        AiDotNetEngine.Current = gpu ? Present(device, "the GPU engine") : new CpuEngine();
        try
        {
            if (gpu) Assert.True(GpuMemoryTracker.Enabled);
            using var input = new Tensor<float>(Enumerable.Range(0, 20).Select(i => (i - 10) * .05f).ToArray(), new[] { 5, 4 });
            using var target = new Tensor<float>(Enumerable.Range(0, 15).Select(i => (i - 7) * .03f).ToArray(), new[] { 5, 3 });
            var layer = new DenseLayer<float>(3);
            var reference = new DenseLayer<float>(3);
            using (var warm = layer.Forward(input)) { }
            using (var warm = reference.Forward(input)) { }
            reference.SetParameters(layer.GetParameters());
            var stepper = new TapeTrainingStepper<float>(new object());
            var optimizer = Sgd();
            var referenceOptimizer = Sgd();
            long sequence = gpu ? NativeSequence() : 0;
            FusedTrainingStepRequest<float> Request(DenseLayer<float> l, IGradientBasedOptimizer<float, Tensor<float>, Tensor<float>> o)
                => new() { Layers = new ITrainableLayer<float>[] { l }, Input = input, Target = target,
                    Forward = l.Forward, ComputeLoss = Loss, Optimizer = o };
            for (int step = 0; step < 4; step++)
            {
                float actual = stepper.Step(Request(layer, optimizer));
                Assert.True(stepper.LastStepFused, stepper.Session.LastMissReason);
                if (gpu && step == 0) sequence = NativeSequence();
                if (gpu && step > 0) AssertNoProbeAllocations(sequence);
                float expected = TapeTrainingStepper<float>.EagerStep(Request(reference, referenceOptimizer));
                Assert.True(Math.Abs(actual - expected) <= 1e-4f * Math.Max(1f, Math.Abs(expected)));
                var values = layer.GetParameters().ToArray();
                var referenceValues = reference.GetParameters().ToArray();
                for (int i = 0; i < values.Length; i++) Assert.True(Math.Abs(values[i] - referenceValues[i]) <= 1e-4f);
            }
        }
        finally { AiDotNetEngine.Current = previous; }
    }
}
