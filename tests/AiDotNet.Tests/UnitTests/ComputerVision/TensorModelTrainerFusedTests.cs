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
}