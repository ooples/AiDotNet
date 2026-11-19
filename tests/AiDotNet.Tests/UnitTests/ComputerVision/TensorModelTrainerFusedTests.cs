using AiDotNet.ComputerVision;
using AiDotNet.Interfaces;
using AiDotNet.LossFunctions;
using AiDotNet.Models;
using AiDotNet.Models.Options;
using AiDotNet.Models.Parameters;
using AiDotNet.Optimizers;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Optimization;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Training;
using Xunit;

namespace AiDotNet.Tests.UnitTests.ComputerVision;

/// <summary>
/// TensorModelTrainer.Step, the step every vision model's Train goes through, runs on the shared fused session when
/// it has an optimizer and a tensor target (#1804): the compiled plan must take the eager step, over steps whose
/// input and target change.
/// </summary>
[Collection("FusedOptimizerGlobalState")]
public class TensorModelTrainerFusedTests
{
    private sealed class LinearModel : ModelBase<double, Tensor<double>, Tensor<double>>
    {
        public Tensor<double> Weight { get; } = new(new[] { 3, 2 }, new Vector<double>(new[] { 0.2, -0.1, 0.4, 0.3, -0.2, 0.1 }));
        public override ILossFunction<double> DefaultLossFunction => new MeanSquaredErrorLoss<double>();
        protected override void RegisterComponents() =>
            RegisterParameterComponent("weight", new TensorListParameterSource<double>(() => new[] { Weight }));
        public override Tensor<double> Predict(Tensor<double> input) => AiDotNetEngine.Current.TensorMatMul(input, Weight);
        public override void Train(Tensor<double> input, Tensor<double> expectedOutput) => throw new NotSupportedException();
        public override IFullModel<double, Tensor<double>, Tensor<double>> WithParameters(Vector<double> parameters) =>
            throw new NotSupportedException();
    }

    private static (double[] Weights, int FusedSteps) Run(bool compile)
    {
        var original = TensorCodecOptions.Current;
        try
        {
            TensorCodecOptions.SetCurrent(new TensorCodecOptions { EnableCompilation = compile });
            CompiledTapeTrainingStep<double>.Invalidate();
            var model = new LinearModel();
            int fusedBefore = TensorModelTrainer<double>.FusedStepCount;
            var optimizer = new AdamOptimizer<double, Tensor<double>, Tensor<double>>(null,
                new AdamOptimizerOptions<double, Tensor<double>, Tensor<double>> { InitialLearningRate = 0.05, EnableGradientClipping = false });
            for (int step = 0; step < 3; step++)
            {
                var input = new Tensor<double>([4, 3]);
                var target = new Tensor<double>([4, 2]);
                for (int i = 0; i < input.Length; i++) input[i] = Math.Sin(0.4 * i + step);
                for (int i = 0; i < target.Length; i++) target[i] = Math.Cos(0.6 * i - step);
                TensorModelTrainer<double>.Step(model, input, target, 0.05, model.Predict, optimizer: optimizer);
            }
            return (model.Weight.ToArray(), TensorModelTrainer<double>.FusedStepCount - fusedBefore);
        }
        finally
        {
            TensorCodecOptions.SetCurrent(original);
            CompiledTapeTrainingStep<double>.Invalidate();
        }
    }

    [Fact(Timeout = 120000)]
    public async Task FusedStep_TakesTheEagerStep_AcrossChangingBatches()
    {
        await Task.Yield();
        var (eager, eagerFused) = Run(compile: false);
        var (fused, fusedSteps) = Run(compile: true);
        Assert.Equal(0, eagerFused);
        Assert.Equal(3, fusedSteps);
        Assert.NotEqual(new[] { 0.2, -0.1, 0.4, 0.3, -0.2, 0.1 }, eager);
        for (int i = 0; i < eager.Length; i++)
            Assert.True(Math.Abs(eager[i] - fused[i]) <= 1e-6, $"weight {i}: eager {eager[i]}, fused {fused[i]}");
    }

    private sealed class HostBreakableLinearModel : ModelBase<double, Tensor<double>, Tensor<double>>
    {
        public Tensor<double> Weight { get; } = new(new[] { 3, 2 }, new Vector<double>(new[] { 0.2, -0.1, 0.4, 0.3, -0.2, 0.1 }));
        public bool ReadTracedActivationOnHost { get; set; }
        public override ILossFunction<double> DefaultLossFunction => new MeanSquaredErrorLoss<double>();
        protected override void RegisterComponents() =>
            RegisterParameterComponent("weight", new TensorListParameterSource<double>(() => new[] { Weight }));

        public override Tensor<double> Predict(Tensor<double> input)
        {
            var projected = AiDotNetEngine.Current.TensorMatMul(input, Weight);
            if (ReadTracedActivationOnHost)
            {
                _ = projected[0, 0];
            }

            return projected;
        }

        public override void Train(Tensor<double> input, Tensor<double> expectedOutput) => throw new NotSupportedException();
        public override IFullModel<double, Tensor<double>, Tensor<double>> WithParameters(Vector<double> parameters) =>
            throw new NotSupportedException();
    }

    private static (double[] Weights, long OwnerFusedSteps, long OwnerEagerSteps) RunPlainSgd(bool compile)
    {
        var original = TensorCodecOptions.Current;
        try
        {
            TensorCodecOptions.SetCurrent(new TensorCodecOptions { EnableCompilation = compile });
            CompiledTapeTrainingStep<double>.Invalidate();
            var model = new LinearModel();
            for (int step = 0; step < 3; step++)
            {
                var input = new Tensor<double>([4, 3]);
                var target = new Tensor<double>([4, 2]);
                for (int i = 0; i < input.Length; i++) input[i] = Math.Sin(0.35 * i + step);
                for (int i = 0; i < target.Length; i++) target[i] = Math.Cos(0.55 * i - step);
                TensorModelTrainer<double>.Step(model, input, target, 0.05, model.Predict);
            }

            var stepper = TapeTrainingStepper<double>.ForOwner(model);
            return (model.Weight.ToArray(), stepper.FusedStepCount, stepper.EagerStepCount);
        }
        finally
        {
            TensorCodecOptions.SetCurrent(original);
            CompiledTapeTrainingStep<double>.Invalidate();
        }
    }

    [Fact(Timeout = 120000)]
    public async Task PlainSgdStep_PreservesCompiledOwnerRoutingAndContinuity_AcrossChangingBatches()
    {
        await Task.Yield();
        var (eager, eagerFusedSteps, _) = RunPlainSgd(compile: false);
        var (compiled, compiledFusedSteps, compiledEagerSteps) = RunPlainSgd(compile: true);

        Assert.Equal(0, eagerFusedSteps);
        Assert.Equal(3, compiledFusedSteps);
        Assert.Equal(0, compiledEagerSteps);
        Assert.NotEqual(new[] { 0.2, -0.1, 0.4, 0.3, -0.2, 0.1 }, eager);
        for (int i = 0; i < eager.Length; i++)
        {
            Assert.True(Math.Abs(eager[i] - compiled[i]) <= 1e-10, $"weight {i}: eager {eager[i]}, compiled {compiled[i]}");
        }
    }

    private sealed class CountingScalarModel : ModelBase<double, Tensor<double>, Tensor<double>>
    {
        public Tensor<double> Weight { get; } = new(new[] { 1 }, new Vector<double>(new[] { 2.0 }));
        public int ForwardCalls { get; private set; }
        public override ILossFunction<double> DefaultLossFunction => new MeanSquaredErrorLoss<double>();
        protected override void RegisterComponents() =>
            RegisterParameterComponent("weight", new TensorListParameterSource<double>(() => new[] { Weight }));

        public override Tensor<double> Predict(Tensor<double> input)
        {
            ForwardCalls++;
            return AiDotNetEngine.Current.TensorMultiply(Weight, input);
        }

        public override void Train(Tensor<double> input, Tensor<double> expectedOutput) => throw new NotSupportedException();
        public override IFullModel<double, Tensor<double>, Tensor<double>> WithParameters(Vector<double> parameters) =>
            throw new NotSupportedException();
    }

    [Fact(Timeout = 120000)]
    public async Task PlainSgdStep_SingleWarmupLifecycle_EngagesOwnerCompiledClosureInThreeForwards()
    {
        await Task.Yield();
        var original = TensorCodecOptions.Current;
        try
        {
            TensorCodecOptions.SetCurrent(new TensorCodecOptions { EnableCompilation = true });
            CompiledTapeTrainingStep<double>.Invalidate();

            var model = new CountingScalarModel();
            var input = new Tensor<double>(new[] { 1 }, new Vector<double>(new[] { 1.0 }));
            var target = new Tensor<double>(new[] { 1 });

            double firstLoss = TensorModelTrainer<double>.Step(model, input, target, 0.1, model.Predict);
            double secondLoss = TensorModelTrainer<double>.Step(model, input, target, 0.1, model.Predict);

            var stepper = TapeTrainingStepper<double>.ForOwner(model);
            Assert.Equal(4.0, firstLoss, 12);
            Assert.Equal(2.56, secondLoss, 12);
            Assert.Equal(1.28, model.Weight[0], 12);
            Assert.Equal(3, model.ForwardCalls);
            Assert.Equal(2, stepper.FusedStepCount);
            Assert.Equal(0, stepper.EagerStepCount);
        }
        finally
        {
            TensorCodecOptions.SetCurrent(original);
            CompiledTapeTrainingStep<double>.Invalidate();
        }
    }

    [Fact(Timeout = 120000)]
    public async Task StepWithTargets_TypedTargetPlainSgd_EngagesOwnerCompiledClosureAndMatchesEager()
    {
        await Task.Yield();
        static (double[] Weights, long FusedSteps) RunTypedTarget(bool compile)
        {
            var saved = TensorCodecOptions.Current;
            try
            {
                TensorCodecOptions.SetCurrent(new TensorCodecOptions { EnableCompilation = compile });
                CompiledTapeTrainingStep<double>.Invalidate();
                var model = new LinearModel();
                for (int step = 0; step < 3; step++)
                {
                    var input = new Tensor<double>([4, 3]);
                    var targetTensor = new Tensor<double>([4, 2]);
                    for (int i = 0; i < input.Length; i++) input[i] = Math.Sin(0.25 * i + step);
                    for (int i = 0; i < targetTensor.Length; i++) targetTensor[i] = Math.Cos(0.45 * i - step);
                    var typedTarget = (Expected: targetTensor, Scale: 0.5);
                    TensorModelTrainer<double>.StepWithTargets(
                        model,
                        input,
                        typedTarget,
                        0.05,
                        model.Predict,
                        static (pred, tgt) =>
                        {
                            var eng = AiDotNetEngine.Current;
                            var diff = eng.TensorSubtract(pred, tgt.Expected);
                            return eng.TensorMultiplyScalar(eng.ReduceSum(eng.TensorMultiply(diff, diff), null), tgt.Scale / diff.Length);
                        });
                }

                return (model.Weight.ToArray(), TapeTrainingStepper<double>.ForOwner(model).FusedStepCount);
            }
            finally
            {
                TensorCodecOptions.SetCurrent(saved);
                CompiledTapeTrainingStep<double>.Invalidate();
            }
        }

        var (eager, eagerFused) = RunTypedTarget(compile: false);
        var (compiled, compiledFused) = RunTypedTarget(compile: true);

        Assert.Equal(0, eagerFused);
        Assert.Equal(3, compiledFused);
        for (int i = 0; i < eager.Length; i++)
        {
            Assert.True(Math.Abs(eager[i] - compiled[i]) <= 1e-10, $"weight {i}: eager {eager[i]}, compiled {compiled[i]}");
        }
    }

    [Fact(Timeout = 120000)]
    public async Task PlainSgdStep_CommittedOwnerSession_RefusesMidRunUnreplayableClosureBreak()
    {
        await Task.Yield();
        var original = TensorCodecOptions.Current;
        try
        {
            TensorCodecOptions.SetCurrent(new TensorCodecOptions { EnableCompilation = true });
            CompiledTapeTrainingStep<double>.Invalidate();

            var model = new HostBreakableLinearModel();
            var input = new Tensor<double>([2, 3], new Vector<double>(new[] { 0.5, -0.25, 1.0, -0.5, 0.75, 0.25 }));
            var target = new Tensor<double>([2, 2], new Vector<double>(new[] { 0.1, -0.2, 0.3, 0.4 }));

            // Step 1 compiles and commits the closure plan on owner: model.
            TensorModelTrainer<double>.Step(model, input, target, 0.05, model.Predict);
            var stepper = TapeTrainingStepper<double>.ForOwner(model);
            Assert.Equal(1, stepper.FusedStepCount);
            Assert.True(stepper.Session.IsCommitted);

            // Step 2 performs a host read of a traced activation on a committed session and must be refused,
            // never silently downgraded to eager execution.
            model.ReadTracedActivationOnHost = true;
            var committedWeights = model.Weight.ToArray();
            var refusal = Assert.Throws<InvalidOperationException>(() =>
                TensorModelTrainer<double>.Step(model, input, target, 0.05, model.Predict));
            Assert.Contains("Fused compiled training has already run successfully", refusal.Message);
            Assert.Equal(committedWeights, model.Weight.ToArray());
            Assert.Equal(1, stepper.FusedStepCount);
        }
        finally
        {
            TensorCodecOptions.SetCurrent(original);
            CompiledTapeTrainingStep<double>.Invalidate();
        }
    }
}