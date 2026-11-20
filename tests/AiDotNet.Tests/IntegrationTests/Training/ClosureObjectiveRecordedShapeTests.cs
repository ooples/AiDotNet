using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Training;
using Xunit;

namespace AiDotNetTests.IntegrationTests.Training;

[Collection("FusedTrainingSerial")]
public sealed class ClosureObjectiveRecordedShapeTests
{
    [Fact]
    public void MultiPlanProbeUsesRecordedShapesAndStillRequiresTheWholeSignature()
    {
        var engine = AiDotNetEngine.Current;
        var parameter = new Tensor<float>(new[] { 1 }, new Vector<float>(new[] { .7f }));
        var parameters = new[] { parameter };
        Tensor<float> Data(int count, float start) => new(new[] { count },
            new Vector<float>(Enumerable.Range(0, count).Select(i => start + i * .1f)));
        var data = Data(2, .2f);
        float scale = 1;
        Tensor<float> Objective() => engine.ReduceSum(engine.TensorMax(
            engine.TensorMultiplyScalar(engine.TensorMultiply(parameter, data), scale), .25f), null);
        var state = new ClosureObjectiveState<float>();
        ClosureObjectiveState<float>.Entry AddPlan()
        {
            using var recording = ClosureObjectiveGuard<float>.BeginRecording();
            using var scope = GraphMode.EnableTraining(parameters);
            var loss = Objective();
            var signature = ObjectiveSignature.Build(scope.Nodes, parameters, out var leaves, loss);
            Assert.True(signature.Signable, signature.UnsignableReason);
            var shapes = scope.CompiledOpShapes.Snapshot();
            var plan = scope.CompileTraining(parameters, loss);
            var guard = ClosureObjectiveGuard<float>.TakeRecorded(out var reason);
            Assert.NotNull(guard);
            var entry = new ClosureObjectiveState<float>.Entry(plan, guard, leaves, shapes);
            state.Add(entry, keepAlive: null);
            return entry;
        }
        try
        {
            var first = AddPlan();
            data = Data(3, .4f);
            var secondData = data;
            var second = AddPlan();
            var untouched = secondData.ToArray();
            data = Data(3, .8f);
            Assert.Equal(ClosureGuardResult.Matched,
                state.Probe(Objective, parameters, out var signature, out _, out var reason));
            Assert.NotNull(signature);
            Assert.Same(second, state.Find(signature));
            Assert.Equal(untouched, secondData.ToArray());

            // Native scalar operations do not affect the fallback's output shape. The full objective signature
            // must still reject the changed coefficient before any plan's leaves are copied or its optimizer runs.
            scale = 2;
            Assert.Equal(ClosureGuardResult.Matched,
                state.Probe(Objective, parameters, out signature, out _, out reason));
            Assert.NotNull(signature);
            Assert.Null(state.Find(signature));
            Assert.Equal(untouched, secondData.ToArray());

            // A shape from another plan cannot substitute for this input geometry.
            scale = 1;
            data = Data(4, .2f);
            Assert.Equal(ClosureGuardResult.Aborted,
                state.Probe(Objective, parameters, out _, out _, out reason));
            Assert.Contains("no matching trace-time shape", reason);
            Assert.Equal(untouched, secondData.ToArray());
            Assert.NotSame(first, second);
        }
        finally { state.Clear(keepAlive: null); }
    }
}
