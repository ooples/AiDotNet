using AiDotNet.Models.Options;
using AiDotNet.Optimizers;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.Optimizers;

[Collection("EngineCurrentGlobalState")]
public class TapeOptimizerGpuReviewRegressionTests
{
    private sealed class MutableMomentum : MomentumOptimizer<float, Tensor<float>, Tensor<float>>
    {
        public MutableMomentum() : base(null, new MomentumOptimizerOptions<float, Tensor<float>, Tensor<float>>
        {
            InitialLearningRate = 0.1, InitialMomentum = 0.9, EnableGradientClipping = false
        }) { }

        public void SetRate(float rate) => CurrentLearningRate = rate;
    }

    [SkippableTheory]
    [InlineData(false)]
    [InlineData(true)]
    public void DeviceSteps_PreserveLearningRateAndStateAcrossHostFallback(bool momentum)
    {
        using var gpu = new DirectGpuTensorEngine();
        Skip.If(!gpu.IsGpuAvailable, "GPU engine unavailable.");
        var saved = AiDotNetEngine.Current;
        try
        {
            AiDotNetEngine.Current = gpu;
            var parameter = gpu.TensorMultiplyScalar(new Tensor<float>(new[] { 1f, 2f }, new[] { 2 }), 1f);
            var optimizer = momentum ? (AiDotNet.Interfaces.IGradientBasedOptimizer<float, Tensor<float>, Tensor<float>>)new MutableMomentum()
                : new AdaDeltaOptimizer<float, Tensor<float>, Tensor<float>>(null,
                    new AdaDeltaOptimizerOptions<float, Tensor<float>, Tensor<float>>
                    { InitialLearningRate = 0.01, Rho = 0.9, Epsilon = 1e-6, EnableGradientClipping = false });
            var expected = new[] { 1.0, 2.0 };
            var velocity = new double[2];
            var accumulatedGrad = new double[2];
            var accumulatedUpdate = new double[2];
            for (int step = 0; step < 3; step++)
            {
                float rate = momentum ? new[] { 0.1f, 0.01f, 0.2f }[step] : 0.01f;
                if (optimizer is MutableMomentum mutable) mutable.SetRate(rate);
                if (step == 2) AiDotNetEngine.Current = new CpuEngine();
                var gradient = new Tensor<float>(new[] { 0.5f, -0.25f }, new[] { 2 });
                if (step < 2) gradient = gpu.TensorMultiplyScalar(gradient, 1f);
                optimizer.Step(new TapeStepContext<float>(new[] { parameter },
                    new Dictionary<Tensor<float>, Tensor<float>> { [parameter] = gradient }, 0f));
                for (int i = 0; i < 2; i++)
                {
                    double g = i == 0 ? 0.5 : -0.25;
                    if (momentum)
                    {
                        velocity[i] = 0.9 * velocity[i] + rate * g;
                        expected[i] -= velocity[i];
                    }
                    else
                    {
                        accumulatedGrad[i] = 0.9 * accumulatedGrad[i] + 0.1 * g * g;
                        double update = Math.Sqrt(accumulatedUpdate[i] + 1e-6) / Math.Sqrt(accumulatedGrad[i] + 1e-6) * g;
                        accumulatedUpdate[i] = 0.9 * accumulatedUpdate[i] + 0.1 * update * update;
                        expected[i] -= rate * update;
                    }
                }
            }
            var actual = parameter.ToArray();
            for (int i = 0; i < actual.Length; i++) Assert.InRange(Math.Abs(actual[i] - expected[i]), 0, 1e-5);
        }
        finally { AiDotNetEngine.Current = saved; }
    }
}
