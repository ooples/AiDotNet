using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.SyntheticData;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Optimization;
using Xunit;
using Xunit.Abstractions;

namespace AiDotNet.Tests.IntegrationTests.SyntheticData;

/// <summary>
/// The fused (compiled) training step must update every tensor the eager step updates. TabDDPM keeps its timestep
/// projection and both output heads outside <c>Layers</c>; the fused path compiled with only
/// <c>CollectParameters(Layers)</c>, so those three layers were constants in the plan and never trained on the GPU
/// path (measured: max weight change 0 over a batch, while the eager step moved them by 0.02-0.05).
/// </summary>
[Collection("DirectGpuSerial")]
public class FusedTrainableSetTests
{
    private static readonly string[] AuxFields = { "_timestepProjection", "_numericalOutputHead", "_categoricalOutputHead" };
    private readonly ITestOutputHelper _out;
    public FusedTrainableSetTests(ITestOutputHelper output) => _out = output;

    [Fact]
    public void TabDDPM_eager_step_updates_every_layer() => AssertEveryLayerTrains(fused: false);

    [SkippableFact]
    public void TabDDPM_fused_step_updates_every_layer_the_eager_step_does()
    {
        DirectGpuTensorEngine? gpu = null;
        try { gpu = new DirectGpuTensorEngine(); } catch (Exception) { }
        Skip.IfNot(gpu is not null && gpu.SupportsGpu, "The fused multi-slot path needs a GPU engine.");
        var prior = AiDotNetEngine.Current;
        var codec = TensorCodecOptions.Current;
        bool saved = codec.EnableCompilation;
        AiDotNetEngine.Current = gpu!;
        codec.EnableCompilation = true;
        try { AssertEveryLayerTrains(fused: true); }
        finally
        {
            codec.EnableCompilation = saved;
            AiDotNetEngine.Current = prior;
            gpu!.Dispose();
        }
    }

    private void AssertEveryLayerTrains(bool fused)
    {
        var random = new Random(7);
        var data = new Matrix<float>(60, 5);
        for (int i = 0; i < 60; i++)
        {
            data[i, 0] = (float)(50 + 10 * random.NextDouble());
            data[i, 1] = (float)(100 + 20 * random.NextDouble());
            data[i, 2] = (float)random.NextDouble();
            data[i, 3] = random.Next(3);
            data[i, 4] = random.Next(2);
        }
        var columns = new List<ColumnMetadata>
        {
            new("F1", ColumnDataType.Continuous, columnIndex: 0),
            new("F2", ColumnDataType.Continuous, columnIndex: 1),
            new("F3", ColumnDataType.Continuous, columnIndex: 2),
            new("C1", ColumnDataType.Categorical, new[] { "A", "B", "C" }, columnIndex: 3),
            new("C2", ColumnDataType.Categorical, new[] { "Y", "N" }, columnIndex: 4),
        };
        var generator = new TabDDPMGenerator<float>(new NeuralNetworkArchitecture<float>(5, 5, NetworkComplexity.Simple),
            new TabDDPMOptions<float> { Seed = 3, MLPDimensions = [32, 32], NumTimesteps = 10, BatchSize = 20 });
        generator.Fit(data, columns, 1);   // builds the layers for the data's actual widths

        // Fit rebuilds its layers, so compare ONE training call on the fitted model: TrainBatch over its own data.
        var flags = BindingFlags.NonPublic | BindingFlags.Instance;
        var type = typeof(TabDDPMGenerator<float>);
        var layers = AuxFields.Select(f => (name: f, layer: (ILayer<float>)type.GetField(f, flags)!.GetValue(generator)!))
            .Concat(generator.Layers.Select((l, i) => (name: $"Layers[{i}]", layer: l)))
            .ToArray();
        var before = layers.Select(x => x.layer.GetParameters().ToArray()).ToArray();

        var prepared = type.GetMethod("PreprocessData", flags)!
            .Invoke(generator, new[] { data, type.GetField("_columns", flags)!.GetValue(generator) })!;
        generator.SetTrainingMode(true);
        long fusedStepsBefore = FusedOptimizerStepCount(generator);
        type.GetMethod("TrainBatch", flags)!.Invoke(generator, new object[]
        {
            prepared.GetType().GetField("Item1")!.GetValue(prepared)!,
            prepared.GetType().GetField("Item2")!.GetValue(prepared)!,
            0, 60, 0.001f,
        });

        if (fused)
        {
            // The weight checks below pass for an eager fallback too, so first prove THIS call ran the fused plan.
            long fusedStepsAfter = FusedOptimizerStepCount(generator);
            Assert.True(fusedStepsAfter > fusedStepsBefore,
                $"TrainBatch did not advance the retained fused plan (optimizer step {fusedStepsBefore} -> {fusedStepsAfter}); "
                + "it fell back to the eager step, so the fused trainable set was not exercised.");
        }

        for (int i = 0; i < layers.Length; i++)
        {
            var after = layers[i].layer.GetParameters().ToArray();
            double change = before[i].Zip(after, (a, b) => Math.Abs(a - b)).DefaultIfEmpty(0).Max();
            _out.WriteLine($"[{(fused ? "fused" : "eager")}] {layers[i].name}: max change {change:G4}");
            Assert.True(change > 0, $"{(fused ? "fused" : "eager")} step left {layers[i].name} untrained");
        }
    }

    /// <summary>The retained fused plan's optimizer step count, or -1 when no fused step or plan exists.</summary>
    private static long FusedOptimizerStepCount(TabDDPMGenerator<float> generator)
    {
        var flags = BindingFlags.NonPublic | BindingFlags.Instance;
        var step = typeof(TabDDPMGenerator<float>).GetField("_fusedMultiSlotStep", flags)?.GetValue(generator);
        var plan = step?.GetType().GetField("_plan", flags)?.GetValue(step);
        var counter = plan?.GetType().GetField("_optimizerStep", flags)?.GetValue(plan);
        return counter is null ? -1 : Convert.ToInt64(counter);
    }
}
