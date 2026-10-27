using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Models.Parameters;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.Training;

[Collection("NonParallelIntegration")]
public class FoundationDispatchTests
{
    private sealed class SpyEngine : CpuEngine
    {
        public int Calls, StateCalls;
        public bool Decline;
        public Tensor<double>? Packed;
        public override Tensor<T> LstmSequenceForward<T>(Tensor<T> input, Tensor<T>? h0, Tensor<T>? c0,
            Tensor<T> wIh, Tensor<T> wHh, Tensor<T>? bIh, Tensor<T>? bHh, bool returnSequences = false)
        {
            Calls++;
            if (typeof(T) == typeof(double)) Packed = (Tensor<double>)(object)wIh;
            if (Decline) throw new NotSupportedException();
            return base.LstmSequenceForward(input, h0, c0, wIh, wHh, bIh, bHh, returnSequences);
        }
        public override Tensor<T> LstmSequenceForward<T>(Tensor<T> input, Tensor<T>? h0, Tensor<T>? c0,
            Tensor<T> wIh, Tensor<T> wHh, Tensor<T>? bIh, Tensor<T>? bHh,
            out Tensor<T> finalHidden, out Tensor<T> finalCell, bool returnSequences = false)
        {
            StateCalls++;
            if (Decline) throw new NotSupportedException();
            return base.LstmSequenceForward(input, h0, c0, wIh, wHh, bIh, bHh,
                out finalHidden, out finalCell, returnSequences);
        }
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void DoubleSequence_DispatchAndGradientsMatchDecomposed(bool training)
    {
        var previous = AiDotNetEngine.Current;
        var engine = new SpyEngine();
        AiDotNetEngine.Current = engine;
        try
        {
            var layer = new LSTMLayer<double>(3);
            layer.SetTrainingMode(training);
            var input = new Tensor<double>(new[] { 2, 4, 2 });
            for (int i = 0; i < input.Length; i++) input[i] = 0.03 * (i - 5);
            layer.Forward(input); // initialize outside the tape
            var parameters = new[] { input }.Concat(layer.GetTrainableParameters()).ToArray();
            Tensor<double> fused;
            Tensor<double>[] fusedGradients;
            using (var tape = new GradientTape<double>())
            {
                fused = layer.Forward(input);
                Assert.NotNull(fused.GradFn);
                var gradients = tape.ComputeGradients(engine.ReduceSum(fused, null), parameters);
                fusedGradients = parameters.Select(parameter => gradients[parameter]).ToArray();
            }
            Assert.True(engine.Calls > 0);
            engine.Decline = true;
            using var referenceTape = new GradientTape<double>();
            var reference = layer.Forward(input);
            var referenceGradients = referenceTape.ComputeGradients(engine.ReduceSum(reference, null), parameters);
            Near(reference, fused, 1e-10);
            for (int i = 0; i < parameters.Length; i++)
                Near(referenceGradients[parameters[i]], fusedGradients[i], 1e-10);
        }
        finally { AiDotNetEngine.Current = previous; }
    }

    [Fact]
    public void FloatSequence_UsesPrimitiveUnderTape()
    {
        var previous = AiDotNetEngine.Current;
        var engine = new SpyEngine();
        AiDotNetEngine.Current = engine;
        try
        {
            var layer = new LSTMLayer<float>(2);
            var input = new Tensor<float>(new[] { 1, 3, 2 });
            layer.Forward(input);
            using var tape = new GradientTape<float>();
            Assert.NotNull(layer.Forward(input).GradFn);
            Assert.True(engine.Calls > 0);
        }
        finally { AiDotNetEngine.Current = previous; }
    }

    [Fact]
    public void InferenceCache_ReusesPackingAndInvalidatesOnRestoreAndEngineChange()
    {
        var previous = AiDotNetEngine.Current;
        var engine = new SpyEngine();
        AiDotNetEngine.Current = engine;
        try
        {
            var layer = new LSTMLayer<double>(3);
            layer.SetTrainingMode(false);
            var input = new Tensor<double>(new[] { 1, 4, 2 });
            var expected = layer.Forward(input);
            var packed = engine.Packed;
            layer.Forward(input);
            Assert.Same(packed, engine.Packed);
            layer.SetParameters(layer.GetParameters());
            Near(expected, layer.Forward(input), 1e-12);
            Assert.NotSame(packed, engine.Packed);
            var other = new SpyEngine();
            AiDotNetEngine.Current = other;
            Near(expected, layer.Forward(input), 1e-12);
            Assert.NotSame(engine.Packed, other.Packed);
            layer.ForwardFromState(input, null, null, out var h, out var c);
            Assert.Equal(1, other.StateCalls);
            Assert.Equal(new[] { 1, 3 }, h.Shape.ToArray());
            Assert.Equal(h.Shape, c.Shape);
        }
        finally { AiDotNetEngine.Current = previous; }
    }

    [Fact]
    public void RegistryRestore_PreflightsAndHandlesReorderedLiveSources()
    {
        var a = new Tensor<double>(new[] { 2 }); a[0] = 1; a[1] = 2;
        var b = new Tensor<double>(new[] { 2 }); b[0] = 3; b[1] = 4;
        var registry = new ParameterComponentRegistry<double>();
        registry.Register("a", new TensorFieldParameterSource<double>(() => a));
        registry.Register("b", new TensorFieldParameterSource<double>(() => b));
        Assert.False(registry.TrySetParameterChunks(new[] { b, new Tensor<double>(new[] { 3 }) }));
        Assert.Equal(1, a[0]);
        int before = 0;
        Assert.True(registry.TrySetParameterChunks(new[] { b, a }, () => before++));
        Assert.Equal(1, before);
        Assert.Equal(new double[] { 3, 4 }, a.ToArray());
        Assert.Equal(new double[] { 1, 2 }, b.ToArray());
    }

    private sealed class CustomSource : IParameterSource<double>, IParameterChunkSource<double>
    {
        public Tensor<double> Weight = new(new[] { 2 });
        public long ParameterCount => 2;
        public Vector<double> GetParameters() => Weight.ToVector();
        public void SetParameters(Vector<double> values) => throw new InvalidOperationException("custom validation");
        public IEnumerable<ParameterChunk<double>> GetParameterStateChunks()
        { yield return new ParameterChunk<double>("$", ParameterSlotRole.Trainable, Weight); }
    }

    [Fact]
    public void RegistryRestore_DoesNotBypassNestedCustomSetter()
    {
        var registry = new ParameterComponentRegistry<double>();
        var custom = new CustomSource();
        registry.Register("child", custom);
        Assert.False(registry.TrySetParameterChunks(new[] { custom.Weight }));
        Assert.Throws<InvalidOperationException>(() => registry.SetParameters(new Vector<double>(2)));
    }

    [Fact]
    public void CheckpointRestore_PreservesCopyOnWriteClone()
    {
        var original = new Tensor<double>(new[] { 2 }); original[0] = 7;
        var clone = (Tensor<double>)original.CloneShared();
        var replacement = new Tensor<double>(new[] { 2 }); replacement[0] = 9;
        AiDotNet.Training.TensorCheckpoint<double>.CopyValues(replacement, clone);
        Assert.Equal(7, original[0]);
        Assert.Equal(9, clone[0]);
    }

    private class RestoreOwner
    {
        public readonly ParameterComponentRegistry<double> Registry = new();
        public readonly Tensor<double> Weight = new(new[] { 2 });
        public int Restored, Restoring;
        public RestoreOwner() => Registry.Register("weight", new TensorFieldParameterSource<double>(() => Weight));
        public virtual Vector<double> GetParameters() => Registry.GetParameters();
        public virtual void SetParameters(Vector<double> values)
        { Restoring++; Registry.SetParameters(values); Restored++; }
        public void Restore(IEnumerable<Tensor<double>> chunks) => ParameterChunkRestore<double>.Restore(
            this, typeof(RestoreOwner), chunks, Registry, SetParameters, () => Restored++, () => Restoring++);
    }
    private sealed class CustomOwner : RestoreOwner
    {
        public int CustomCalls;
        public override void SetParameters(Vector<double> values) { CustomCalls++; base.SetParameters(values); }
    }

    [Fact]
    public void SharedRestore_InvokesHooksOnceAndPreservesCustomSetters()
    {
        var source = new Tensor<double>(new[] { 2 }); source[0] = 5;
        var ordinary = new RestoreOwner(); ordinary.Restore(new[] { source });
        Assert.Equal(5, ordinary.Weight[0]);
        Assert.Equal(1, ordinary.Restoring); Assert.Equal(1, ordinary.Restored);
        var custom = new CustomOwner(); custom.Restore(new[] { source });
        Assert.Equal(5, custom.Weight[0]);
        Assert.Equal(1, custom.CustomCalls);
        Assert.Equal(1, custom.Restoring); Assert.Equal(1, custom.Restored);
    }

    [Fact]
    public void RestoreHook_RebindingTargetsDoesNotWriteDetachedStorage()
    {
        var old = new Tensor<double>(new[] { 2 }); old[0] = 4;
        var current = old;
        var registry = new ParameterComponentRegistry<double>();
        registry.Register("weight", new TensorFieldParameterSource<double>(() => current));
        var source = new Tensor<double>(new[] { 2 }); source[0] = 8;
        Assert.True(registry.TrySetParameterChunks(new[] { source }, () => current = new Tensor<double>(new[] { 2 })));
        Assert.Equal(4, old[0]); Assert.Equal(8, current[0]);
    }

    [Fact]
    public void DeferredField_ChunkRestoreDeclinesWithoutWritingThenFlatRestoreResolvesShape()
    {
        Tensor<double>? weight = new(new[] { 2, 0 });
        var registry = new ParameterComponentRegistry<double>();
        registry.Register("weight", new ResizableTensorFieldParameterSource<double>(() => weight, value => weight = value));
        var values = new Tensor<double>(new double[] { 1, 2, 3, 4, 5, 6 }, new[] { 2, 3 });
        Assert.False(registry.TrySetParameterChunks(new[] { values }));
        Assert.Equal(new[] { 2, 0 }, weight.Shape.ToArray());
        registry.SetParameters(new Vector<double>(values.ToArray()));
        Assert.Equal(new[] { 2, 3 }, weight.Shape.ToArray());
        Assert.Equal(values.ToArray(), weight.ToArray());
    }

    [Fact]
    public void DeepARSequence_MatchesStepRecurrenceForBatchedInputs()
    {
        var cell = new AiDotNet.TimeSeries.DeepARLstmCellTape<double>(2, 3, seed: 51);
        var engine = AiDotNetEngine.Current;
        var input = new Tensor<double>(new[] { 2, 4, 2 });
        for (int i = 0; i < input.Length; i++) input[i] = Math.Sin(i) * 0.3;
        var actual = cell.Sequence(input);
        var h = new Tensor<double>(new[] { 3, 2 });
        var c = new Tensor<double>(new[] { 3, 2 });
        for (int t = 0; t < 4; t++)
        {
            var x = engine.TensorTranspose(engine.TensorSliceAxis(input, 1, t));
            (h, c) = cell.Step(x, h, c);
            Near(engine.TensorTranspose(h), engine.TensorSliceAxis(actual, 1, t), 1e-12);
        }
    }

    [Fact]
    public void DecimalDeepArSequence_GradientMatchesFiniteDifference()
    {
        var prior = AiDotNetEngine.Current;
        var engine = new CpuEngine(); AiDotNetEngine.Current = engine;
        try
        {
            var cell = new AiDotNet.TimeSeries.DeepARLstmCellTape<decimal>(2, 3, seed: 51);
            var input = new Tensor<decimal>(new decimal[] { .1m, .2m, .3m, .4m, .5m, .6m }, new[] { 1, 3, 2 });
            Tensor<decimal> Loss() => engine.ReduceSum(cell.Sequence(input), null);
            decimal actual;
            using (var tape = new AiDotNet.Tensors.Engines.Autodiff.GradientTape<decimal>())
                actual = tape.ComputeGradients(Loss(), new[] { input })[input][0];
            const decimal epsilon = .0001m;
            input[0] += epsilon; decimal high = Loss()[0];
            input[0] -= 2 * epsilon; decimal low = Loss()[0];
            Assert.InRange(Math.Abs(actual - (high - low) / (2 * epsilon)), 0m, .00001m);
        }
        finally { AiDotNetEngine.Current = prior; }
    }

    [Fact]
    public void StatefulTraining_DispatchesAndPreservesAllStateGradients()
    {
        var previous = AiDotNetEngine.Current;
        var engine = new SpyEngine(); AiDotNetEngine.Current = engine;
        try
        {
            var layer = new LSTMLayer<double>(3);
            var input = new Tensor<double>(new[] { 2, 4, 2 });
            for (int i = 0; i < input.Length; i++) input[i] = Math.Sin(i) * 0.2;
            var h0 = new Tensor<double>(new[] { 2, 3 }); h0[0] = 0.2;
            var c0 = new Tensor<double>(new[] { 2, 3 }); c0[0] = -0.3;
            layer.Forward(input);
            var sources = new[] { input, h0, c0 }.Concat(layer.GetTrainableParameters()).ToArray();
            Tensor<double> Loss()
            {
                var y = layer.ForwardFromState(input, h0, c0, out var h, out var c);
                return engine.TensorAdd(engine.ReduceSum(y, null),
                    engine.ReduceSum(engine.TensorAdd(h, c), null));
            }
            Tensor<double> actual;
            Dictionary<Tensor<double>, Tensor<double>> gradients;
            using (var tape = new GradientTape<double>())
            { actual = Loss(); gradients = tape.ComputeGradients(actual, sources); }
            Assert.Equal(1, engine.StateCalls);
            engine.Decline = true;
            using var referenceTape = new GradientTape<double>();
            var expected = Loss(); var reference = referenceTape.ComputeGradients(expected, sources);
            Near(expected, actual, 1e-10);
            foreach (var parameter in sources) Near(reference[parameter], gradients[parameter], 1e-10);
        }
        finally { AiDotNetEngine.Current = previous; }
    }

    private static void Near(Tensor<double> expected, Tensor<double> actual, double tolerance)
    {
        Assert.Equal(expected.Shape, actual.Shape);
        for (int i = 0; i < expected.Length; i++)
            Assert.True(Math.Abs(expected[i] - actual[i]) <= tolerance, $"index {i}: {expected[i]} vs {actual[i]}");
    }

    [Fact]
    public void LayerCopy_PreservesIdentityAndRejectsInvalidLaterChunkBeforeWriting()
    {
        var source = new DenseLayer<double>(2);
        var target = new DenseLayer<double>(2);
        source.Forward(new Tensor<double>(new[] { 1, 3 })); target.Forward(new Tensor<double>(new[] { 1, 3 }));
        var chunks = source.GetParameterStateChunks("$").Select(c => c.Tensor).ToArray();
        var destinations = target.GetParameterStateChunks("$").Select(c => c.Tensor).ToArray();
        Assert.True(chunks.Length > 1);
        chunks[0][0] = 7;
        ParameterTransfer.CopyLayer(source, target);
        Assert.Same(destinations[0], target.GetParameterStateChunks("$").First().Tensor);
        Assert.Equal(7, destinations[0][0]);
        chunks[0][0] = 9;
        Assert.Equal(7, destinations[0][0]);
        var invalid = chunks.ToArray(); invalid[^1] = new Tensor<double>(new[] { chunks[^1].Length + 1 });
        Assert.Throws<ArgumentException>(() => target.TryRestoreParameterChunks(invalid));
        Assert.Equal(7, destinations[0][0]);
    }

    [Fact]
    [Trait("Category", "GPU")]
    public void RegisteredLayerRestore_OpenClDoesNotDownloadWeights()
    {
        using var backend = new AiDotNet.Tensors.Engines.DirectGpu.OpenCL.OpenClBackend();
        Assert.True(backend.IsAvailable, backend.InitializationException?.ToString());
        using var direct = new AiDotNet.Tensors.Engines.DirectGpu.DirectGpuEngine(backend);
        using var gpu = new AiDotNet.Tensors.Engines.DirectGpuTensorEngine(direct);
        var prior = AiDotNetEngine.Current;
        try
        {
            AiDotNetEngine.Current = gpu;
            var source = new DenseLayer<float>(2);
            var target = new DenseLayer<float>(2);
            source.Forward(new Tensor<float>(new[] { 1, 3 })); target.Forward(new Tensor<float>(new[] { 1, 3 }));
            var sourceChunks = source.GetParameterStateChunks("$").Select(c => c.Tensor).ToArray();
            var expected = sourceChunks.Select(c => c.ToArray()).ToArray();
            var registry = new ParameterComponentRegistry<float>(); registry.Register("layer", target);
            var probe = new Tensor<float>(new[] { 1, 3 }); probe[0] = .3f; probe[1] = -.2f;
            var referenceOutput = source.Forward(probe).ToArray();
            Tensor<float>? actualOutput = null;
            int hooks = 0;
            using (var scope = AiDotNet.Tensors.Engines.Diagnostics.GpuResidencyScope.Begin(captureOperations: true))
            {
                var snapshot = AiDotNet.Training.TensorCheckpoint<float>.Capture(sourceChunks);
                Assert.True(registry.TrySetParameterChunks(snapshot, () => hooks++));
                ParameterTransfer.CopyLayer(source, target);
                actualOutput = target.Forward(probe);
                Assert.True(scope.Downloads == 0, string.Join("\n", scope.Events.Where(e => e.Kind == AiDotNet.Tensors.Engines.Diagnostics.GpuTransferKind.DeviceToHost)));
            }
            Assert.Equal(referenceOutput, actualOutput!.ToArray());
            Assert.Equal(1, hooks);
            var restored = target.GetParameterStateChunks("$").ToArray();
            for (int i = 0; i < restored.Length; i++) Assert.Equal(expected[i], restored[i].Tensor.ToArray());
        }
        finally { AiDotNetEngine.Current = prior; }
    }

    [Fact]
    public void NestedBidirectionalConsumer_UsesTwoSequenceKernels()
    {
        var prior = AiDotNetEngine.Current; var engine = new SpyEngine(); AiDotNetEngine.Current = engine;
        try
        {
            var layer = new BidirectionalLayer<double>(new LSTMLayer<double>(3), activationFunction: (IActivationFunction<double>?)null);
            var input = new Tensor<double>(new[] { 2, 4, 2 });
            layer.Forward(input); engine.Calls = 0;
            using var tape = new GradientTape<double>();
            var output = layer.Forward(input);
            var gradients = tape.ComputeGradients(engine.ReduceSum(output, null), new[] { input });
            Assert.Equal(2, engine.Calls);
            Assert.Equal(input.Shape, gradients[input].Shape);
        }
        finally { AiDotNetEngine.Current = prior; }
    }

    [Fact]
    public void TimeGradHistory_UsesOneSequenceKernelPerLayerAndPreservesCarriedState()
    {
        var prior = AiDotNetEngine.Current; var engine = new SpyEngine(); AiDotNetEngine.Current = engine;
        try
        {
            var network = new AiDotNet.Finance.Forecasting.Foundation.TimeGradNetwork<double>(2, 3, 0, 2, 1, 2, 1, 2, 2);
            var input = new Tensor<double>(new[] { 2, 4, 2 });
            for (int i = 0; i < input.Length; i++) input[i] = Math.Sin(i) * .2;
            var expected = network.EncodeHistory(input);
            Assert.Equal(2, engine.Calls);
            var hidden = new Tensor<double>?[2]; var cell = new Tensor<double>?[2];
            var first = engine.TensorNarrow(input, 1, 0, 2); var second = engine.TensorNarrow(input, 1, 2, 2);
            network.AdvanceHistory(engine, first, hidden, cell);
            var actual = network.AdvanceHistory(engine, second, hidden, cell);
            Assert.Equal(4, engine.StateCalls);
            Near(engine.Reshape(engine.TensorNarrow(expected, 1, 3, 1), new[] { 2, 3 }), actual, 1e-10);
        }
        finally { AiDotNetEngine.Current = prior; }
    }

    private sealed class CustomDense : DenseLayer<double>
    {
        internal int SetterCalls;
        internal CustomDense() : base(2) { }
        public override void SetParameters(Vector<double> values) { SetterCalls++; base.SetParameters(values); }
    }

    [Fact]
    public void LayerCopy_PreservesCustomSetterContract()
    {
        var source = new DenseLayer<double>(2); var target = new CustomDense();
        var input = new Tensor<double>(new[] { 1, 3 }); source.Forward(input); target.Forward(input);
        ParameterTransfer.CopyLayer(source, target);
        Assert.Equal(1, target.SetterCalls);
        Assert.Equal(source.GetParameters().ToArray(), target.GetParameters().ToArray());
    }
}
