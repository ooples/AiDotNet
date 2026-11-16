using System;
using System.Collections.Generic;
using System.Linq;
using AiDotNet.Interfaces;
using AiDotNet.ActivationFunctions;
using AiDotNet.Enums;
using AiDotNet.MixedPrecision;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Optimizers;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.Engines.Optimization;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Training;
using Xunit;

namespace AiDotNet.Tests.TextToSpeech;

[Collection("FusedOptimizerGlobalState")]
public sealed class RetainedGenerationHandoffTests
{
    [Fact]
    public void Shared_plain_sgd_reports_its_successful_update_before_publishing()
    {
        var previous = AiDotNetEngine.Current;
        try
        {
            var engine = new CpuEngine();
            AiDotNetEngine.Current = engine;
            var weight = new Tensor<double>(new[] { 2.0 }, new[] { 1 });
            Tensor<double> Objective()
            {
                var generated = engine.TensorMultiplyScalar(weight, 3.0);
                GradientRecording.RetainOutput("generation", engine.StopGradient(generated));
                return engine.ReduceSum(generated, null);
            }
            using var handoff = RetainedOutputCapture<double>.Begin();
            Assert.Equal(6.0, TapeTrainingStepper<double>.EagerSgdObjectiveStep(() => new[] { weight }, Objective, 0.1));
            Assert.Equal(1.7, weight.ToArray()[0], 10);
            handoff.Complete();
            Assert.Equal(6.0, handoff.Get("generation").ToArray()[0]);
        }
        finally { AiDotNetEngine.Current = previous; }
    }

    [Fact]
    public void Failed_outer_objective_cannot_publish_or_leak_into_the_next_valid_step()
    {
        var previous = AiDotNetEngine.Current;
        try
        {
            var engine = new CpuEngine();
            AiDotNetEngine.Current = engine;
            var weight = new Tensor<double>(new[] { 2.0 }, new[] { 1 });
            using (var failed = RetainedOutputCapture<double>.Begin())
            {
                Tensor<double> FailingObjective()
                {
                    GradientRecording.RetainOutput("generation", engine.StopGradient(engine.TensorMultiplyScalar(weight, 99.0)));
                    throw new InvalidOperationException("The outer objective failed after retaining its output.");
                }
                Assert.Throws<InvalidOperationException>(() => TapeTrainingStepper<double>.EagerSgdObjectiveStep(
                    () => new[] { weight }, FailingObjective, 0.1));
                Assert.Equal(2.0, weight.ToArray()[0]);
                Assert.Throws<InvalidOperationException>(() => failed.Complete());
                Assert.Throws<InvalidOperationException>(() => failed.Get("generation"));
            }
            using var valid = RetainedOutputCapture<double>.Begin();
            Tensor<double> ValidObjective()
            {
                var generated = engine.TensorMultiplyScalar(weight, 3.0);
                GradientRecording.RetainOutput("generation", engine.StopGradient(generated));
                return engine.ReduceSum(generated, null);
            }
            Assert.Equal(6.0, TapeTrainingStepper<double>.EagerSgdObjectiveStep(() => new[] { weight }, ValidObjective, 0.1));
            valid.Complete();
            Assert.Equal(6.0, valid.Get("generation").ToArray()[0]);
            Assert.Equal(1.7, weight.ToArray()[0], 10);
        }
        finally { AiDotNetEngine.Current = previous; }
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void Native_eager_or_hybrid_nonfinite_guard_cannot_publish_a_skipped_producer(bool hybrid)
        => AssertNativeNonfiniteSkip(compile: hybrid, hybrid: hybrid);

    [Fact]
    public void Native_fused_nonfinite_guard_cannot_publish_a_skipped_producer()
        => AssertNativeNonfiniteSkip(compile: true, hybrid: false);

    private static void AssertNativeNonfiniteSkip(bool compile, bool hybrid)
    {
        var previousEngine = AiDotNetEngine.Current;
        var previousOptions = TensorCodecOptions.Current;
        try
        {
            var engine = new CpuEngine();
            AiDotNetEngine.Current = engine;
            TensorCodecOptions.SetCurrent(new TensorCodecOptions { EnableCompilation = compile });
            var weight = new Tensor<float>(new[] { 1.0f }, new[] { 1 });
            var placeholder = new Tensor<float>(new[] { 1 });
            var optimizer = new AdamOptimizer<float, Tensor<float>, Tensor<float>>(null,
                new AdamOptimizerOptions<float, Tensor<float>, Tensor<float>>
                { InitialLearningRate = 0.01, EnableGradientClipping = false });
            var stepper = TapeTrainingStepper<float>.ForOwner(optimizer);
            Tensor<float> Objective()
            {
                if (GradientRecording.IsRecording<float>())
                    GradientRecording.RetainOutput("generation", engine.StopGradient(engine.TensorAddScalar(weight, 2.0f)));
                return engine.ReduceSum(engine.TensorSqrt(weight), null);
            }
            FusedTrainingStepRequest<float> Request(bool declaredBreak) => new()
            {
                Layers = Array.Empty<ITrainableLayer<float>>(), Input = placeholder, Target = placeholder,
                Forward = static x => x, ComputeLoss = (_, _) => Objective(), ClosureObjective = Objective,
                Optimizer = optimizer, ExtraParameters = new[] { weight },
                GraphBreakReason = declaredBreak ? "This producer explicitly uses the native eager-gradient path." : null,
                ClosureGraphBreakPolicy = declaredBreak
                    ? ClosureGraphBreakPolicy.UseConfiguredOptimizer : ClosureGraphBreakPolicy.Disallow,
            };
            try
            {
                using (var first = RetainedOutputCapture<float>.Begin())
                {
                    Assert.Equal(1.0f, stepper.Step(Request(false)));
                    first.Complete();
                    Assert.Equal(3.0f, first.Get("generation").ToArray()[0]);
                }
                weight[0] = 0; // Finite sqrt loss, nonfinite derivative; invoke the optimizer's actual guard.
                using var skipped = RetainedOutputCapture<float>.Begin();
                Assert.Equal(0.0f, stepper.Step(Request(hybrid)));
                Assert.Equal(hybrid, stepper.LastStepHybrid);
                Assert.Equal(compile && !hybrid, stepper.LastStepFused);
                Assert.Equal(compile ? (hybrid ? 1 : 2) : 0, stepper.FusedStepCount);
                Assert.Equal(compile ? 0 : 2, stepper.EagerStepCount);
                Assert.Equal(hybrid ? 1 : 0, stepper.HybridStepCount);
                Assert.Equal(0.0f, weight.ToArray()[0]);
                Assert.Throws<InvalidOperationException>(() => skipped.Complete());
                Assert.Throws<InvalidOperationException>(() => skipped.Get("generation"));
            }
            finally { stepper.Session.Reset(stickyDisable: false); }
        }
        finally
        {
            TensorCodecOptions.SetCurrent(previousOptions);
            AiDotNetEngine.Current = previousEngine;
        }
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void Public_custom_objective_hands_off_waveform_after_its_step_arena_has_ended(bool compile)
    {
        var previousEngine = AiDotNetEngine.Current;
        var previousOptions = TensorCodecOptions.Current;
        try
        {
            var engine = new CpuEngine();
            AiDotNetEngine.Current = engine;
            TensorCodecOptions.SetCurrent(new TensorCodecOptions { EnableCompilation = compile });
            using var model = new ObjectiveHarness(new NeuralNetworkArchitecture<double>(
                InputType.OneDimensional, NeuralNetworkTaskType.Regression, inputSize: 3, outputSize: 3,
                layers: new List<ILayer<double>>
                {
                    new InputLayer<double>(3), new DenseLayer<double>(3, activationFunction: new IdentityActivation<double>()),
                }));
            model.Predict(new Tensor<double>(new[] { 0.0, 0.0, 0.0 }, new[] { 3 })); // Initialize the lazy dense weights.
            var weight = model.CollectTrainableParameters()[0];
            Assert.Equal(9, weight.Length);
            weight.AsWritableSpan().Fill(0.25);
            var optimizer = new AdamOptimizer<double, Tensor<double>, Tensor<double>>(null,
                new AdamOptimizerOptions<double, Tensor<double>, Tensor<double>>
                { InitialLearningRate = 0.01, EnableGradientClipping = false });
            using var callerArena = TensorArena.Create();
            for (int step = 0; step < 3; step++)
            {
                var input = new Tensor<double>(Enumerable.Repeat(step + 1.0, 96).ToArray(), new[] { 32, 3 });
                var before = weight.ToArray();
                var expected = new double[3];
                for (int channel = 0; channel < 3; channel++)
                    for (int feature = 0; feature < 3; feature++)
                        expected[channel] += (step + 1.0) * before[feature * 3 + channel];
                Tensor<double> generated;
                using (var outputs = RetainedOutputCapture<double>.Begin())
                {
                    model.TrainObjective(input, input, (_, _) =>
                    {
                        var waveform = engine.TensorMatMul(input, weight);
                        if (GradientRecording.IsRecording<double>())
                            GradientRecording.RetainOutput("generation", engine.StopGradient(waveform));
                        return engine.ReduceMean(waveform, null, keepDims: false);
                    }, optimizer);
                    outputs.Complete();
                    generated = outputs.Get("generation");
                }
                object storage = generated._storage;
                callerArena.Reset();
                callerArena.TryRentTensor<double>(96, new[] { 96 })!.AsWritableSpan().Fill(-99.0);
                Assert.Equal(new[] { 32, 3 }, generated.Shape.ToArray());
                Assert.Same(storage, generated._storage);
                var values = generated.ToArray();
                for (int row = 0; row < 32; row++)
                    for (int channel = 0; channel < 3; channel++)
                        Assert.Equal(expected[channel], values[row * 3 + channel], 10);
                Assert.Equal(compile ? step + 1 : 0, model.CustomFusedStepCount);
                Assert.Equal(compile ? 0 : step + 1, model.CustomEagerStepCount);
                Assert.Equal(0, model.CustomOutsidePlanStepCount);
            }
        }
        finally
        {
            TensorCodecOptions.SetCurrent(previousOptions);
            AiDotNetEngine.Current = previousEngine;
        }
    }

    [Theory]
    [InlineData("success")]
    [InlineData("overflow")]
    [InlineData("update-failure")]
    public void Single_amp_handoff_requires_an_actual_successful_update(string outcome)
    {
        var previousEngine = AiDotNetEngine.Current;
        try
        {
            var engine = new CpuEngine();
            AiDotNetEngine.Current = engine;
            using var model = new NeuralNetwork<float>(new NeuralNetworkArchitecture<float>(
                InputType.OneDimensional, NeuralNetworkTaskType.Regression, inputSize: 1, outputSize: 1));
            model.EnableMixedPrecision(new AiDotNet.MixedPrecision.MixedPrecisionConfig
            { PrecisionType = MixedPrecisionType.FP16, InitialLossScale = 1, EnableDynamicScaling = false });
            var weight = new Tensor<float>(new[] { 2.0f }, new[] { 1 });
            var placeholder = new Tensor<float>(new[] { 1 });
            var optimizer = new AdamOptimizer<float, Tensor<float>, Tensor<float>>(null,
                new AdamOptimizerOptions<float, Tensor<float>, Tensor<float>>
                { InitialLearningRate = 0.01, EnableGradientClipping = false });
            int updates = 0;
            Tensor<float> Objective()
            {
                var generated = engine.TensorMultiplyScalar(weight, 3.0f);
                GradientRecording.RetainOutput("generation", engine.StopGradient(generated));
                return engine.ReduceSum(engine.TensorMultiplyScalar(generated,
                    outcome == "overflow" ? float.PositiveInfinity : 1.0f), null);
            }
            using var handoff = RetainedOutputCapture<float>.Begin();
            void Train() => TapeTrainingStepper<float>.EagerTapeStep(Objective, () => new[] { weight }, context =>
            {
                if (outcome == "update-failure") throw new InvalidOperationException("injected update failure");
                optimizer.Step(context);
                updates++;
            }, placeholder, placeholder, (_, _) => Objective(), (loss, _) => loss,
                new TapeTrainingStepper<float>.EagerStepHooks
                {
                    Owner = model,
                    ReportsRetainedOutputOutcome = TapeTrainingStepper<float>.ReportsRetainedOutputOutcome(optimizer),
                });
            if (outcome == "update-failure") Assert.Throws<InvalidOperationException>(Train);
            else Train();
            if (outcome == "success")
            {
                Assert.Equal(1, updates);
                Assert.Equal(1.99f, weight.ToArray()[0], 5);
                handoff.Complete();
                Assert.Equal(6.0f, handoff.Get("generation").ToArray()[0]);
            }
            else
            {
                Assert.Equal(0, updates);
                Assert.Equal(2.0f, weight.ToArray()[0]);
                Assert.Throws<InvalidOperationException>(() => handoff.Complete());
                Assert.Throws<InvalidOperationException>(() => handoff.Get("generation"));
            }
        }
        finally { AiDotNetEngine.Current = previousEngine; }
    }

    [Fact]
    public void Unsupported_summed_objective_handoff_refuses_before_forward_or_update()
    {
        int forwards = 0, updates = 0;
        using var handoff = RetainedOutputCapture<double>.Begin();
        var placeholder = new Tensor<double>(new[] { 1 });
        Tensor<double> Objective() { forwards++; return placeholder; }
        Assert.Throws<NotSupportedException>(() => TapeTrainingStepper<double>.EagerTapeStep(Objective,
            () => new[] { placeholder }, _ => updates++, placeholder, placeholder,
            (_, _) => Objective(), (loss, _) => loss,
            new TapeTrainingStepper<double>.EagerStepHooks { ReportsRetainedOutputOutcome = true },
            objectiveTerms: new Func<Tensor<double>>[] { Objective }));
        Assert.Equal(0, forwards);
        Assert.Equal(0, updates);
        Assert.Throws<InvalidOperationException>(() => handoff.Complete());
    }

    [Fact]
    public void Unknown_updater_handoff_refuses_before_forward_or_mutation_but_ordinary_training_is_unchanged()
    {
        var previous = AiDotNetEngine.Current;
        try
        {
            var engine = new CpuEngine();
            AiDotNetEngine.Current = engine;
            var weight = new Tensor<double>(new[] { 2.0 }, new[] { 1 });
            var placeholder = new Tensor<double>(new[] { 1 });
            int forwards = 0, updates = 0;
            Tensor<double> Objective()
            {
                forwards++;
                return engine.ReduceSum(engine.TensorMultiply(weight, weight), null);
            }
            void Train() => TapeTrainingStepper<double>.EagerTapeStep(Objective, () => new[] { weight }, context =>
            {
                updates++;
                weight[0] -= 0.1 * context.Gradients[weight].ToArray()[0];
            }, placeholder, placeholder, (_, _) => Objective(), (loss, _) => loss);
            using (var handoff = RetainedOutputCapture<double>.Begin())
            {
                Assert.Throws<NotSupportedException>(Train);
                Assert.Equal(0, forwards);
                Assert.Equal(0, updates);
                Assert.Equal(2.0, weight.ToArray()[0]);
                Assert.Throws<InvalidOperationException>(() => handoff.Complete());
            }
            Train();
            Assert.Equal(1, forwards);
            Assert.Equal(1, updates);
            Assert.Equal(1.6, weight.ToArray()[0], 10);
        }
        finally { AiDotNetEngine.Current = previous; }
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void Critic_consumes_current_preupdate_generation_across_A_B_A_and_eager_or_hybrid_steps(bool hybrid)
    {
        var previousEngine = AiDotNetEngine.Current;
        var previousOptions = TensorCodecOptions.Current;
        try
        {
            AiDotNetEngine.Current = new CpuEngine();
            var eager = Run(compile: false, hybrid: false);
            var compiled = Run(compile: true, hybrid: hybrid);
            Assert.Equal(hybrid ? 3 : 4, compiled.Fused);
            Assert.Equal(hybrid ? 1 : 0, compiled.Hybrid);
            Assert.Equal(2, compiled.Plans);
            Assert.Same(compiled.Outputs[0], compiled.Outputs[2]);
            Assert.NotSame(compiled.Outputs[0], compiled.Outputs[1]);
            for (int step = 0; step < eager.Generations.Length; step++)
            {
                Assert.Equal(eager.Generations[step], compiled.Generations[step], 7);
                Assert.Equal(eager.Weights[step], compiled.Weights[step], 7);
            }
        }
        finally
        {
            TensorCodecOptions.SetCurrent(previousOptions);
            AiDotNetEngine.Current = previousEngine;
        }
    }

    private sealed record Result(double[] Generations, double[] Weights, Tensor<double>[] Outputs,
        int Fused, int Hybrid, int Plans);

    private static Result Run(bool compile, bool hybrid)
    {
        TensorCodecOptions.SetCurrent(new TensorCodecOptions { EnableCompilation = compile });
        var engine = AiDotNetEngine.Current;
        var weight = new Tensor<double>(new[] { 2.0 }, new[] { 1, 1 });
        var criticWeight = new Tensor<double>(new[] { 1.0 }, new[] { 1, 1 });
        var optimizer = new AdamOptimizer<double, Tensor<double>, Tensor<double>>(null,
            new AdamOptimizerOptions<double, Tensor<double>, Tensor<double>>
            { InitialLearningRate = 0.01, EnableGradientClipping = false });
        var stepper = TapeTrainingStepper<double>.ForOwner(optimizer);
        var generations = new List<double>();
        var weights = new List<double>();
        var outputs = new List<Tensor<double>>();
        var input = new Tensor<double>(new[] { 4, 1 });
        Tensor<double> Objective()
        {
            var generated = engine.TensorMultiply(weight, input);
            if (GradientRecording.IsRecording<double>())
                GradientRecording.RetainOutput("generation", engine.StopGradient(generated));
            return engine.ReduceMean(generated, null, keepDims: false);
        }
        var placeholder = new Tensor<double>(new[] { 1 });
        double expectedWeight = 2, moment = 0, variance = 0, expectedCritic = 1;
        object? planAStorage = null;
        int plans = 0;
        try
        {
            for (int step = 0; step < 4; step++)
            {
                int rows = step % 2 == 0 ? 4 : 2;
                var data = Enumerable.Range(0, rows).Select(i => (step + 1.0) * (i + 1) / 4).ToArray();
                input = new Tensor<double>(data, new[] { rows, 1 });
                double inputMean = data.Average();
                double expectedGeneration = expectedWeight * inputMean;
                Tensor<double> generated;
                using (var handoff = RetainedOutputCapture<double>.Begin())
                {
                    var request = new FusedTrainingStepRequest<double>
                    {
                        Layers = Array.Empty<ITrainableLayer<double>>(),
                        Input = placeholder, Target = placeholder,
                        Forward = static x => x, ComputeLoss = (_, _) => Objective(),
                        ClosureObjective = Objective, Optimizer = optimizer, ExtraParameters = new[] { weight },
                        GraphBreakReason = hybrid && step == 3 ? "An explicit eager producer step." : null,
                        ClosureGraphBreakPolicy = hybrid && step == 3
                            ? ClosureGraphBreakPolicy.UseConfiguredOptimizer : ClosureGraphBreakPolicy.Disallow,
                    };
                    Assert.Equal(expectedGeneration, stepper.Step(request), 7);
                    handoff.Complete();
                    generated = handoff.Get("generation");
                    if (compile) plans = CompiledTapeTrainingStep<double>.LastClosurePlanCount;
                }
                outputs.Add(generated);
                if (compile && step == 0) planAStorage = generated._storage;
                if (compile && step == 2) Assert.Same(planAStorage, generated._storage);
                Assert.Equal(rows, generated.Length);
                for (int i = 0; i < rows; i++) Assert.Equal(expectedWeight * data[i], generated.ToArray()[i], 7);
                generations.Add(generated.ToArray().Average());

                moment = 0.9 * moment + 0.1 * inputMean;
                variance = 0.999 * variance + 0.001 * inputMean * inputMean;
                expectedWeight -= 0.01 * (moment / (1 - Math.Pow(0.9, step + 1)))
                    / (Math.Sqrt(variance / (1 - Math.Pow(0.999, step + 1))) + 1e-8);
                Assert.Equal(expectedWeight, weight.ToArray()[0], 7);
                weights.Add(weight.ToArray()[0]);

                // A separate tape consumes the producer's borrowed output AFTER the producer's arena/tape ended.
                // Its gradient is the pre-update generation, not a re-generation from the updated weight.
                double producerAfterUpdate = weight.ToArray()[0];
                TapeTrainingStepper<double>.EagerSgdObjectiveStep(() => new[] { criticWeight },
                    () => engine.ReduceMean(engine.TensorMultiply(criticWeight, generated), null, keepDims: false), 0.125);
                expectedCritic -= 0.125 * expectedGeneration;
                Assert.Equal(expectedCritic, criticWeight.ToArray()[0], 7);
                Assert.Equal(producerAfterUpdate, weight.ToArray()[0]);
                Assert.Equal(expectedGeneration, generated.ToArray().Average(), 7);
            }
            return new Result(generations.ToArray(), weights.ToArray(), outputs.ToArray(),
                stepper.FusedStepCount, stepper.HybridStepCount, plans);
        }
        finally { stepper.Session.Reset(stickyDisable: false); }
    }

    private sealed class ObjectiveHarness : NeuralNetwork<double>
    {
        public ObjectiveHarness(NeuralNetworkArchitecture<double> architecture) : base(architecture) { }

        public double TrainObjective(Tensor<double> input, Tensor<double> expected,
            Func<Tensor<double>, Tensor<double>, Tensor<double>> objective,
            IGradientBasedOptimizer<double, Tensor<double>, Tensor<double>> optimizer)
            => TrainWithCustomObjective(input, expected, objective, optimizer);
    }
}
