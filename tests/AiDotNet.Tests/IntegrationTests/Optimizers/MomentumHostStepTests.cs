using AiDotNet.Models.Options;
using AiDotNet.Optimizers;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.Optimizers;

/// <summary>
/// On the CPU, the momentum step updates velocity and parameter in one pass over the arrays. The tensor-op form it
/// replaces allocated three parameter-sized tensors per parameter per step (680 MB over five FasterRCNN steps in an
/// allocation trace). The pass must compute velocity = momentum * velocity + lr * grad and parameter -= velocity exactly,
/// on the vector lanes and on the tail.
/// </summary>
[Collection("EngineCurrentGlobalState")]
public class MomentumHostStepTests
{
    private sealed class MutableMomentum : MomentumOptimizer<float, Tensor<float>, Tensor<float>>
    {
        public MutableMomentum() : base(null, new MomentumOptimizerOptions<float, Tensor<float>, Tensor<float>>
        {
            InitialLearningRate = 0.1, InitialMomentum = 0.9, EnableGradientClipping = false, UseAdaptiveMomentum = false,
        }) { }

        public void SetRate(float rate) => CurrentLearningRate = rate;
    }

    [Fact]
    public void HostStep_FollowsTheMomentumUpdate_OnLanesAndTail()
    {
        var saved = AiDotNetEngine.Current;
        try
        {
            AiDotNetEngine.Current = new CpuEngine();
            const int length = 1003; // not a multiple of any vector width
            var random = new Random(5);
            var initial = new float[length];
            for (int i = 0; i < length; i++) initial[i] = (float)(random.NextDouble() * 2 - 1);
            var parameter = new Tensor<float>((float[])initial.Clone(), new[] { length });
            var optimizer = new MutableMomentum();

            var expected = initial.Select(v => (double)v).ToArray();
            var velocity = new double[length];
            foreach (float rate in new[] { 0.1f, 0.0f, 0.05f })
            {
                optimizer.SetRate(rate);
                var gradientValues = new float[length];
                for (int i = 0; i < length; i++) gradientValues[i] = (float)(random.NextDouble() * 2 - 1);
                var gradient = new Tensor<float>(gradientValues, new[] { length });
                optimizer.Step(new TapeStepContext<float>(new[] { parameter },
                    new Dictionary<Tensor<float>, Tensor<float>> { [parameter] = gradient }, 0f));
                for (int i = 0; i < length; i++)
                {
                    velocity[i] = 0.9 * velocity[i] + rate * gradientValues[i];
                    expected[i] -= velocity[i];
                }
            }

            var actual = parameter.ToArray();
            for (int i = 0; i < length; i++) Assert.InRange(Math.Abs(actual[i] - expected[i]), 0, 1e-5);
        }
        finally { AiDotNetEngine.Current = saved; }
    }

    [Fact]
    public void HostStep_DoesNotAllocateParameterSizedTensors()
    {
        var saved = AiDotNetEngine.Current;
        try
        {
            AiDotNetEngine.Current = new CpuEngine();
            const int length = 1 << 20; // 4 MB of float
            var parameter = new Tensor<float>(new float[length], new[] { length });
            var gradient = new Tensor<float>(Enumerable.Repeat(0.5f, length).ToArray(), new[] { length });
            var optimizer = new MutableMomentum();
            var context = new TapeStepContext<float>(new[] { parameter },
                new Dictionary<Tensor<float>, Tensor<float>> { [parameter] = gradient }, 0f);
            optimizer.Step(context); // the first step creates the velocity buffer

            long before = GC.GetTotalAllocatedBytes(true);
            optimizer.Step(context);
            long allocated = GC.GetTotalAllocatedBytes(true) - before;

            // The tensor-op form allocated three 4 MB tensors here.
            Assert.True(allocated < length * sizeof(float) / 4, $"A momentum step allocated {allocated:N0} bytes.");
        }
        finally { AiDotNetEngine.Current = saved; }
    }
}
