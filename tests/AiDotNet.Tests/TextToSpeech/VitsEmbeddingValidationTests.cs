using System;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.TextToSpeech;

[Collection("EngineCurrentGlobalState")]
public sealed class VitsEmbeddingValidationTests
{
    private static Tensor<double> Initialize(EmbeddingLayer<double> layer)
    {
        var table = layer.GetMaterializedEmbeddingTable();
        for (int row = 0; row < 5; row++)
            for (int col = 0; col < 3; col++) table[row, col] = row * 10 + col + 0.5;
        return table;
    }

    [Fact]
    public void Compiled_embedding_revalidates_current_ids_and_updates_only_the_selected_rows()
    {
        var previous = AiDotNetEngine.Current;
        try
        {
            var engine = new CpuEngine();
            AiDotNetEngine.Current = engine;
            using var layer = new EmbeddingLayer<double>(5, 3);
            var table = Initialize(layer);
            var expected = table.ToArray();
            var ids = new Tensor<double>(new[] { 0.0, 1.0 }, new[] { 2 });
            using var recording = ClosureObjectiveGuard<double>.BeginRecording();
            CompiledTrainingPlan<double> plan;
            using (var scope = GraphMode.EnableTraining(new[] { table }))
            {
                var loss = engine.ReduceSum(layer.Forward(ids), null);
                plan = scope.CompileTraining(new[] { table }, loss);
            }
            using (plan)
            {
                var guard = ClosureObjectiveGuard<double>.TakeRecorded(out var reason);
                Assert.True(guard != null, reason);
                guard!.AttachTo(plan);
                const float rate = 0.125f;
                plan.ConfigureOptimizer(OptimizerType.SGD, learningRate: rate);
                foreach (var selected in new[] { new[] { 0, 1 }, new[] { 3, 3 }, new[] { 4, 2 }, new[] { 1, 0 } })
                {
                    for (int i = 0; i < selected.Length; i++) ids[i] = selected[i];
                    ids.IncrementVersion();
                    Assert.Equal(ClosureGuardResult.Matched,
                        guard.TryRebind(() => engine.ReduceSum(layer.Forward(ids), null), new[] { table }, out _));
                    double expectedLoss = 0;
                    foreach (int row in selected)
                        for (int col = 0; col < 3; col++) expectedLoss += expected[row * 3 + col];
                    Assert.Equal(expectedLoss, plan.Step().ToArray()[0], 10);
                    foreach (int row in selected)
                        for (int col = 0; col < 3; col++) expected[row * 3 + col] -= rate;
                    Assert.Equal(expected, table.ToArray());
                    var gradients = plan.Gradients[0].ToArray();
                    for (int row = 0; row < 5; row++)
                    {
                        int uses = Array.FindAll(selected, id => id == row).Length;
                        for (int col = 0; col < 3; col++) Assert.Equal((double)uses, gradients[row * 3 + col]);
                    }
                }
            }
        }
        finally { AiDotNetEngine.Current = previous; }
    }

    [Theory]
    [InlineData(-1.0)]
    [InlineData(1.5)]
    [InlineData(5.0)]
    [InlineData(1e20)]
    [InlineData(double.NaN)]
    [InlineData(double.PositiveInfinity)]
    [InlineData(double.NegativeInfinity)]
    public void Invalid_ids_after_successful_compiled_steps_are_rejected_before_any_parameter_update(double invalid)
    {
        var previous = AiDotNetEngine.Current;
        try
        {
            var engine = new CpuEngine();
            AiDotNetEngine.Current = engine;
            using var layer = new EmbeddingLayer<double>(5, 3);
            var table = Initialize(layer);
            var ids = new Tensor<double>(new[] { 0.0, 1.0 }, new[] { 2 });
            CompiledTrainingPlan<double> plan;
            using (var scope = GraphMode.EnableTraining(new[] { table }))
            {
                var loss = engine.ReduceSum(layer.Forward(ids), null);
                plan = scope.CompileTraining(new[] { table }, loss);
            }
            using (plan)
            {
                plan.ConfigureOptimizer(OptimizerType.SGD, learningRate: 0.125f);
                plan.Step();
                ids[0] = 3.0;
                ids.IncrementVersion();
                plan.Step();
                var beforeRejection = table.ToArray();
                ids[0] = invalid;
                ids.IncrementVersion();
                var error = Assert.ThrowsAny<ArgumentException>(() => plan.Step());
                Assert.Contains("token indices", error.Message);
                Assert.Equal(beforeRejection, table.ToArray());
            }
        }
        finally { AiDotNetEngine.Current = previous; }
    }
}
