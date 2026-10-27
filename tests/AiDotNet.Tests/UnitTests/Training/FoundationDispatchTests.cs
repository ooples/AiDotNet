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
            Assert.Equal(new[] { 1, 3 }, h.Shape);
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

    private static void Near(Tensor<double> expected, Tensor<double> actual, double tolerance)
    {
        Assert.Equal(expected.Shape, actual.Shape);
        for (int i = 0; i < expected.Length; i++)
            Assert.True(Math.Abs(expected[i] - actual[i]) <= tolerance, $"index {i}: {expected[i]} vs {actual[i]}");
    }
}
