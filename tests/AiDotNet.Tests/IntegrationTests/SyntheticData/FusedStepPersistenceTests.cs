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
        try { gpu = new DirectGpuTensorEngine(); } catch (System.Exception bestEffortError) { System.Diagnostics.Trace.TraceWarning($"FusedStepPersistenceTests: a best-effort step failed ({bestEffortError.GetType().Name}: {bestEffortError.Message})"); }
        Skip.IfNot(gpu is not null && gpu.SupportsGpu, "The fused multi-slot path needs a GPU engine.");
        var priorEngine = AiDotNetEngine.Current;
        var codec = TensorCodecOptions.Current;
        bool savedCompilation = codec.EnableCompilation;
        AiDotNetEngine.Current = gpu;
        codec.EnableCompilation = true;
        try
        {
            var random = AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(7);
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

            var step = (typeof(TabDDPMGenerator<float>)
                .GetField("_fusedMultiSlotStep", BindingFlags.NonPublic | BindingFlags.Instance) ?? throw new InvalidOperationException("typeof(TabDDPMGenerator<float>) .GetField(\"_fusedMultiSlotStep\", BindingFlags... is null where it is required."))
                .GetValue(generator);
            // On a GPU engine with compilation on, the fused path must engage; an eager fallback is the regression
            // this test exists to catch, not a reason to skip it.
            Assert.NotNull(step);
            var multiSlot = (AiDotNet.Training.MultiSlotFusedStep<float>)step;

            // One fused step per row per epoch, none declined to the eager tape.
            Assert.Equal(0, multiSlot.DeclinedStepCount);
            Assert.Equal(Rows * Epochs, multiSlot.FusedStepCount);

            // The plan's own optimizer step: one per row per epoch. A plan rebuilt per batch restarted it (and the Adam
            // moments) at every batch, leaving at most one batch's worth. The multi-slot step trains through the
            // shared fused session, whose configured plan is the compiled step's.
            var plan = (typeof(AiDotNet.Training.CompiledTapeTrainingStep<float>)
                .GetProperty("_configuredPlan", BindingFlags.NonPublic | BindingFlags.Static) ?? throw new InvalidOperationException("typeof(AiDotNet.Training.CompiledTapeTrainingStep<float>) .GetProperty(\"_conf... is null where it is required.")).GetValue(null);
            Assert.NotNull(plan);
            int optimizerStep = (int)((plan.GetType()
                .GetField("_optimizerStep", BindingFlags.NonPublic | BindingFlags.Instance) ?? throw new InvalidOperationException("plan.GetType() .GetField(\"_optimizerStep\", BindingFlags.NonPublic | BindingFl... is null where it is required.")).GetValue(plan) ?? throw new InvalidOperationException("(plan.GetType() .GetField(\"_optimizerStep\", BindingFlags.NonPublic | BindingF... is null where it is required."));
            Assert.True(optimizerStep > BatchSize,
                $"fused optimizer step counter is {optimizerStep} after {Epochs} epochs of {Rows} rows in batches of " +
                $"{BatchSize}: the plan's optimizer state was restarted between batches.");
            int skipped = (int)((plan.GetType().GetProperty("NonFiniteStepsSkipped") ?? throw new InvalidOperationException("plan.GetType().GetProperty(\"NonFiniteStepsSkipped\") is null where it is required.")).GetValue(plan) ?? throw new InvalidOperationException("(plan.GetType().GetProperty(\"NonFiniteStepsSkipped\") ?? throw new InvalidOper... is null where it is required."));
            var graphExec = (IntPtr)((plan.GetType().GetField("_stepGraphExec", BindingFlags.NonPublic | BindingFlags.Instance) ?? throw new InvalidOperationException("plan.GetType().GetField(\"_stepGraphExec\", BindingFlags.NonPublic | BindingFla... is null where it is required.")).GetValue(plan) ?? throw new InvalidOperationException("(plan.GetType().GetField(\"_stepGraphExec\", BindingFlags.NonPublic | BindingFl... is null where it is required."));
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
