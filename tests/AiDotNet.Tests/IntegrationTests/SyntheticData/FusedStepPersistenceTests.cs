using System;
using System.Collections.Generic;
using System.Reflection;
using AiDotNet.Enums;
using AiDotNet.LinearAlgebra;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.SyntheticData;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Optimization;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.SyntheticData;

/// <summary>
/// The fused (MultiSlotFusedStep) training path must keep ONE compiled plan - and with it the optimizer's
/// moments and step counter - across the batches of a fit. TabDDPM, FinDiff, AutoDiffTab and TabSyn built
/// the step object inside the per-batch method and CSDI inside every Train call, so each batch re-traced,
/// recompiled and restarted Adam at t = 1 (an update of ~lr*sign(g) instead of Adam's).
/// </summary>
[Collection("DirectGpuSerial")]
public class FusedStepPersistenceTests
{
    private const int Rows = 100, BatchSize = 25, Epochs = 2;

    [SkippableFact]
    public void TabDDPM_fused_plan_keeps_its_optimizer_step_across_batches_and_epochs()
    {
        DirectGpuTensorEngine? gpu = null;
        try { gpu = new DirectGpuTensorEngine(); } catch (Exception) { }
        Skip.IfNot(gpu is not null && gpu.SupportsGpu, "The fused multi-slot path needs a GPU engine.");
        var priorEngine = AiDotNetEngine.Current;
        var codec = TensorCodecOptions.Current;
        bool savedCompilation = codec.EnableCompilation;
        AiDotNetEngine.Current = gpu!;
        codec.EnableCompilation = true;
        try
        {
            var random = new Random(7);
            var data = new Matrix<float>(Rows, 5);
            for (int i = 0; i < Rows; i++)
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
            var generator = new TabDDPMGenerator<float>(
                new NeuralNetworkArchitecture<float>(5, 5, NetworkComplexity.Simple),
                new TabDDPMOptions<float> { Seed = 3, MLPDimensions = [32, 32], NumTimesteps = 10, BatchSize = BatchSize });

            generator.Fit(data, columns, Epochs);

            var step = typeof(TabDDPMGenerator<float>)
                .GetField("_fusedMultiSlotStep", BindingFlags.NonPublic | BindingFlags.Instance)!
                .GetValue(generator);
            Skip.If(step is null, "The fused path did not engage on this engine (eager fallback).");
            var plan = step!.GetType().GetField("_plan", BindingFlags.NonPublic | BindingFlags.Instance)!.GetValue(step);
            Assert.NotNull(plan);
            int optimizerStep = (int)plan!.GetType()
                .GetField("_optimizerStep", BindingFlags.NonPublic | BindingFlags.Instance)!.GetValue(plan)!;

            // One plan.Step per row per epoch. A step object rebuilt per batch left the counter at one
            // batch's worth (<= BatchSize) with the moments reset each time.
            Assert.True(optimizerStep > BatchSize,
                $"fused optimizer step counter is {optimizerStep} after {Epochs} epochs of {Rows} rows in batches of " +
                $"{BatchSize}: the plan's optimizer state was restarted between batches.");
            int skipped = (int)plan.GetType().GetProperty("NonFiniteStepsSkipped")!.GetValue(plan)!;
            var graphExec = (IntPtr)plan.GetType().GetField("_stepGraphExec", BindingFlags.NonPublic | BindingFlags.Instance)!.GetValue(plan)!;
            Console.WriteLine($"TabDDPM fused step: optimizer steps {optimizerStep}, whole-step CUDA graph engaged: {graphExec != IntPtr.Zero}");
            Assert.True(skipped == 0, $"{skipped} fused step(s) were discarded for non-finite gradients");
            Assert.Equal(Rows * Epochs, optimizerStep);
        }
        finally
        {
            codec.EnableCompilation = savedCompilation;
            AiDotNetEngine.Current = priorEngine;
            gpu?.Dispose();
        }
    }
}
