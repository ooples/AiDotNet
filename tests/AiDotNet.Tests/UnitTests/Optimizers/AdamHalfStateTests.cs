using AiDotNet.Models.Options;
using AiDotNet.Optimizers;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.LinearAlgebra;
using System.Reflection;
using Xunit;

namespace AiDotNet.Tests.UnitTests.Optimizers;

[Collection("EngineCurrentGlobalState")]
public sealed class AdamHalfStateTests
{
    private static AdamOptimizer<Half, Matrix<Half>, Vector<Half>> Adam(bool amsgrad = false)
        => new(null, new AdamOptimizerOptions<Half, Matrix<Half>, Vector<Half>>
        {
            InitialLearningRate = 0.01, UseAdaptiveLearningRate = false,
            UseAMSGrad = amsgrad, Epsilon = 1e-8,
        });

    private static Vector<Half> Values(params double[] values)
        => new(values.Select(value => (Half)value).ToArray());

    private static Vector<Half> Gradient(int step)
        => (step % 4) switch
        {
            0 => Values(1e-4, -2e-4, 5e-5),
            1 => Values(-3e-5, 1e-5, 0),
            2 => Values(0, 0, 0),
            _ => Values(2e-5, -4e-5, -8e-5),
        };

    private static void OnCpu(Action action)
    {
        var previous = AiDotNetEngine.Current;
        try { AiDotNetEngine.Current = new CpuEngine(); action(); }
        finally { AiDotNetEngine.Current = previous; }
    }

    [Fact]
    public void DefaultEpsilon_ZeroGradientsLeaveFiniteParametersUnchanged()
        => OnCpu(() =>
        {
            var optimizer = Adam();
            var original = Values(0.125, -0.25, 0.375);
            var current = original;
            for (int step = 0; step < 3; step++)
            {
                current = optimizer.UpdateParameters(current, Values(0, 0, 0));
                Assert.Equal(original.ToArray(), current.ToArray());
            }
        });

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void TinyGradients_RetainFp32MomentsAcrossSteps(bool amsgrad)
        => OnCpu(() =>
        {
            var optimizer = Adam(amsgrad);
            var actual = Values(0.125, -0.25, 0.375);
            var expected = actual.ToArray();
            var master = expected.Select(value => (float)value).ToArray();
            var m = new float[3]; var v = new float[3]; var maximum = new float[3];
            for (int step = 1; step <= 16; step++)
            {
                var gradient = Gradient(step - 1);
                // Independent scalar FP32 reference, quantizing only the updated parameters.
                for (int i = 0; i < expected.Length; i++)
                {
                    float g = (float)gradient[i];
                    m[i] = 0.9f * m[i] + 0.1f * g;
                    v[i] = 0.999f * v[i] + 0.001f * (g * g);
                    maximum[i] = Math.Max(maximum[i], v[i]);
                    float mHat = m[i] / (float)(1 - Math.Pow(0.9, step));
                    float vHat = (amsgrad ? maximum[i] : v[i]) / (float)(1 - Math.Pow(0.999, step));
                    float update = (float)(Half)0.01 * (mHat / ((float)Math.Sqrt(vHat) + 1e-8f));
                    master[i] -= update;
                    expected[i] = (Half)master[i];
                }
                actual = optimizer.UpdateParameters(actual, gradient);
                Assert.Equal(expected, actual.ToArray());
            }
        });

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void CheckpointCopies_ContinueIndependentMomentState(bool amsgrad)
        => OnCpu(() =>
        {
            var original = Adam(amsgrad);
            var parameters = Values(0.125, -0.25, 0.375);
            for (int step = 0; step < 5; step++)
                parameters = original.UpdateParameters(parameters, Gradient(step));
            var bytes = original.Serialize();
            var restored = Adam(amsgrad); restored.Deserialize(bytes);
            var copy = Adam(amsgrad); copy.Deserialize(bytes);
            var expected = parameters; var resumed = parameters; var copied = parameters;
            for (int step = 5; step < 12; step++)
            {
                expected = original.UpdateParameters(expected, Gradient(step));
                resumed = restored.UpdateParameters(resumed, Gradient(step));
                copied = copy.UpdateParameters(copied, Gradient(step));
                Assert.Equal(expected.ToArray(), resumed.ToArray());
                Assert.Equal(expected.ToArray(), copied.ToArray());
            }
            // Advancing one copy must not mutate another copy's saved moments.
            original.UpdateParameters(expected, Values(0.5, -0.25, 0.125));
            Assert.Equal(restored.UpdateParameters(resumed, Gradient(12)).ToArray(),
                copy.UpdateParameters(copied, Gradient(12)).ToArray());
        });

    [Fact]
    public void ResetAndShapeChange_ClearMoments_AndReverseSurvivesCheckpoint()
        => OnCpu(() =>
        {
            var optimizer = Adam();
            var initial = Values(0.125, -0.25, 0.375);
            var first = optimizer.UpdateParameters(initial, Gradient(0));
            var copy = Adam(); copy.Deserialize(optimizer.Serialize());
            Assert.Equal(initial.ToArray(), copy.ReverseUpdate(first, Gradient(0)).ToArray());
            // ReverseUpdate returns weights; like the existing API it does not roll back optimizer time.
            Assert.Equal(optimizer.UpdateParameters(first, Gradient(1)).ToArray(),
                copy.UpdateParameters(first, Gradient(1)).ToArray());
            optimizer.Reset();
            Assert.Equal(Adam().UpdateParameters(initial, Gradient(0)).ToArray(),
                optimizer.UpdateParameters(initial, Gradient(0)).ToArray());
            var smaller = Values(0.125, -0.25);
            Assert.Equal(Adam().UpdateParameters(smaller, Values(1e-4, -2e-4)).ToArray(),
                optimizer.UpdateParameters(smaller, Values(1e-4, -2e-4)).ToArray());
        });

    [Fact]
    public void MatrixUpdates_UseTheSamePersistentFp32StateAsVectorUpdates()
        => OnCpu(() =>
        {
            var vectorOptimizer = Adam(); var matrixOptimizer = Adam();
            var vector = Values(0.125, -0.25, 0.375, -0.125);
            var matrix = new Matrix<Half>(new[,] { { vector[0], vector[1] }, { vector[2], vector[3] } });
            var gradient = Values(1e-4, -2e-4, 5e-5, -3e-5);
            var matrixGradient = new Matrix<Half>(new[,] { { gradient[0], gradient[1] }, { gradient[2], gradient[3] } });
            for (int step = 0; step < 5; step++)
            {
                vector = vectorOptimizer.UpdateParameters(vector, gradient);
                matrix = matrixOptimizer.UpdateParameters(matrix, matrixGradient);
                Assert.Equal(vector.ToArray(), matrix.ToRowVector().ToArray());
            }
        });

    [Fact]
    public void SubHalfUlpUpdates_AccumulateThroughCheckpointMatrixAndReverseCalls()
        => OnCpu(() =>
        {
            var optimizer = Adam(); var matrixOptimizer = Adam(); var restored = Adam();
            var parameters = Values(1000);
            var matrix = new Matrix<Half>(new[,] { { (Half)1000 } });
            var gradient = Values(1e-4);
            var matrixGradient = new Matrix<Half>(new[,] { { gradient[0] } });
            var resumed = parameters;
            float master = 1000, m = 0, v = 0, g = (float)gradient[0];
            for (int step = 1; step <= 64; step++)
            {
                m = 0.9f * m + 0.1f * g;
                v = 0.999f * v + 0.001f * (g * g);
                float mHat = m / (float)(1 - Math.Pow(0.9, step));
                float vHat = v / (float)(1 - Math.Pow(0.999, step));
                master -= (float)(Half)0.01 * (mHat / ((float)Math.Sqrt(vHat) + 1e-8f));
                var previous = parameters;
                parameters = optimizer.UpdateParameters(parameters, gradient);
                matrix = matrixOptimizer.UpdateParameters(matrix, matrixGradient);
                Assert.Equal((Half)master, parameters[0]);
                Assert.Equal(parameters[0], matrix[0, 0]);
                // Reverse returns the old published weights, while the optimizer retains the
                // current step's hidden FP32 residual and time for the next forward update.
                Assert.Equal(previous.ToArray(), optimizer.ReverseUpdate(parameters, gradient).ToArray());
                if (step == 17)
                {
                    restored.Deserialize(optimizer.Serialize());
                    resumed = parameters;
                    Assert.Equal(previous.ToArray(), restored.ReverseUpdate(resumed, gradient).ToArray());
                }
                else if (step > 17)
                {
                    resumed = restored.UpdateParameters(resumed, gradient);
                    Assert.Equal(parameters.ToArray(), resumed.ToArray());
                }
            }
            Assert.NotEqual((Half)1000, parameters[0]);
        });

    [Fact]
    public void CallerWeightReplacement_RebindsTheMaster_AndRetainsMoments()
        => OnCpu(() =>
        {
            var optimizer = Adam();
            var parameters = Values(1000);
            float m = 0, v = 0;
            for (int step = 1; step <= 6; step++)
            {
                var gradient = Values(step == 6 ? -4e-5 : 1e-4);
                float g = (float)gradient[0];
                m = 0.9f * m + 0.1f * g;
                v = 0.999f * v + 0.001f * (g * g);
                if (step == 6)
                    parameters = Values(10);
                parameters = optimizer.UpdateParameters(parameters, gradient);
                if (step == 6)
                {
                    float mHat = m / (float)(1 - Math.Pow(0.9, step));
                    float vHat = v / (float)(1 - Math.Pow(0.999, step));
                    float expected = 10 - (float)(Half)0.01 * (mHat / ((float)Math.Sqrt(vHat) + 1e-8f));
                    Assert.Equal((Half)expected, parameters[0]);
                }
            }
        });

    [Theory]
    [InlineData("legacy")]
    [InlineData("master")]
    [InlineData("second-moment")]
    public void UnsupportedOrIncompleteState_IsRejectedWithoutMutation(string state)
        => OnCpu(() =>
        {
            var optimizer = Adam();
            var parameters = Values(0.125, -0.25, 0.375);
            const BindingFlags flags = BindingFlags.Instance | BindingFlags.NonPublic;
            var type = optimizer.GetType();
            if (state == "legacy")
            {
                // The materialized fields of an older Half checkpoint: only T moments and time.
                type.GetField("_m", flags)!.SetValue(optimizer, Values(1e-4, -2e-4, 5e-5));
                type.GetField("_v", flags)!.SetValue(optimizer, Values(1e-4, 2e-4, 5e-5));
                type.GetField("_t", flags)!.SetValue(optimizer, 3);
            }
            else
            {
                parameters = optimizer.UpdateParameters(parameters, Gradient(0));
                type.GetField(state == "master" ? "_halfMaster" : "_halfV", flags)!.SetValue(optimizer, null);
            }
            var restored = Adam();
            restored.Deserialize(optimizer.Serialize());
            var before = restored.Serialize();
            var weights = parameters.ToArray();
            if (state == "legacy")
            {
                Assert.Throws<NotSupportedException>(() => restored.UpdateParameters(parameters, Gradient(1)));
                Assert.Throws<NotSupportedException>(() => restored.ReverseUpdate(parameters, Gradient(1)));
            }
            else
            {
                Assert.Throws<InvalidOperationException>(() => restored.UpdateParameters(parameters, Gradient(1)));
                Assert.Throws<InvalidOperationException>(() => restored.ReverseUpdate(parameters, Gradient(1)));
            }
            Assert.Equal(before, restored.Serialize());
            Assert.Equal(weights, parameters.ToArray());
        });

    [SkippableTheory]
    [InlineData(false)]
    [InlineData(true)]
    public void GpuUpdates_CommitNativeState_ForImmediateCheckpointAndCpuResume(bool amsgrad)
    {
        var previous = AiDotNetEngine.Current;
        using var gpu = new DirectGpuTensorEngine();
        if (Environment.GetEnvironmentVariable("AIDOTNET_REQUIRE_GPU_TESTS") == "1")
            Assert.True(gpu.IsGpuAvailable, "The required GPU backend failed to initialize.");
        Skip.If(!gpu.IsGpuAvailable, "No DirectGpuTensorEngine is available.");
        try
        {
            var optimizer = Adam(amsgrad);
            var parameters = Values(1000, 1000, 1000);
            AiDotNetEngine.Current = new CpuEngine();
            var reference = Adam(amsgrad);
            var expected = parameters;
            AiDotNetEngine.Current = gpu;
            Vector<Half> prior = parameters;
            for (int step = 0; step < 19; step++)
            {
                prior = parameters;
                long launchesBefore = GpuLaunchProbe.TotalEver;
                parameters = optimizer.UpdateParameters(parameters, Gradient(step));
                Assert.True(GpuLaunchProbe.TotalEver > launchesBefore,
                    "The FP32 optimizer state update executed no GPU work.");
                AiDotNetEngine.Current = new CpuEngine();
                expected = reference.UpdateParameters(expected, Gradient(step));
                Assert.Equal(expected.ToArray(), parameters.ToArray());
                AiDotNetEngine.Current = gpu;
            }
            // Serialize immediately after a device update; the native arrays must already be authoritative.
            var checkpoint = optimizer.Serialize();
            AiDotNetEngine.Current = new CpuEngine();
            var restored = Adam(amsgrad); restored.Deserialize(checkpoint);
            if (!amsgrad)
                Assert.Equal(prior.ToArray(), restored.ReverseUpdate(parameters, Gradient(18)).ToArray());
            else
                Assert.Throws<NotSupportedException>(() => restored.ReverseUpdate(parameters, Gradient(18)));
            for (int step = 19; step < 64; step++)
            {
                parameters = restored.UpdateParameters(parameters, Gradient(step));
                expected = reference.UpdateParameters(expected, Gradient(step));
                Assert.Equal(expected.ToArray(), parameters.ToArray());
            }
        }
        finally { AiDotNetEngine.Current = previous; }
    }
}
