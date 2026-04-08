// Copyright (c) AiDotNet. All rights reserved.
// Correctness tests for DirectGpu backend operations.
using System;
using Xunit;
using Xunit.Abstractions;
#if !NET462
using AiDotNet.Tensors.Engines.DirectGpu;
#endif

namespace AiDotNet.Tests;

public class DirectGpuCorrectnessTests
{
    private readonly ITestOutputHelper _output;

    public DirectGpuCorrectnessTests(ITestOutputHelper output)
    {
        _output = output;
    }

#if !NET462
    [Fact]
    [Trait("Category", "GPU")]
    public void DirectGpu_ElementwiseOps_MatchCpu()
    {
        if (!TryGetBackend(out var engine, out var backend))
            return;
        using (engine)
        {
            float[] a = { 1.5f, -2.0f, 3.25f, 4.0f, -5.5f, 6.25f };
            float[] b = { 0.5f, 4.0f, -1.25f, 2.0f, 1.5f, -3.0f };

            using var bufA = backend.AllocateBuffer(a);
            using var bufB = backend.AllocateBuffer(b);
            using var bufC = backend.AllocateBuffer(a.Length);

            backend.Add(bufA, bufB, bufC, a.Length);
            AssertAllClose(AddCpu(a, b), DownloadAfterSync(backend, bufC));

            backend.Subtract(bufA, bufB, bufC, a.Length);
            AssertAllClose(SubCpu(a, b), DownloadAfterSync(backend, bufC));

            backend.Multiply(bufA, bufB, bufC, a.Length);
            AssertAllClose(MulCpu(a, b), DownloadAfterSync(backend, bufC));

            backend.Divide(bufA, bufB, bufC, a.Length);
            AssertAllClose(DivCpu(a, b), DownloadAfterSync(backend, bufC), 1e-4f);

            backend.Min(bufA, bufB, bufC, a.Length);
            AssertAllClose(MinCpu(a, b), DownloadAfterSync(backend, bufC));

            backend.Max(bufA, bufB, bufC, a.Length);
            AssertAllClose(MaxCpu(a, b), DownloadAfterSync(backend, bufC));

            backend.Scale(bufA, bufC, 2.5f, a.Length);
            AssertAllClose(ScaleCpu(a, 2.5f), DownloadAfterSync(backend, bufC));

            backend.Power(bufA, bufC, 2.0f, a.Length);
            AssertAllClose(PowCpu(a, 2.0f), DownloadAfterSync(backend, bufC), 1e-4f);

            backend.Abs(bufA, bufC, a.Length);
            AssertAllClose(AbsCpu(a), DownloadAfterSync(backend, bufC));

            backend.Sign(bufA, bufC, a.Length);
            AssertAllClose(SignCpu(a), DownloadAfterSync(backend, bufC));
        }
    }

    [Fact]
    [Trait("Category", "GPU")]
    public void DirectGpu_UnaryOps_MatchCpu()
    {
        if (!TryGetBackend(out var engine, out var backend))
            return;
        using (engine)
        {
            float[] a = { 0.1f, 0.5f, 1.25f, 2.0f, 3.5f, 5.0f };
            using var bufA = backend.AllocateBuffer(a);
            using var bufB = backend.AllocateBuffer(a.Length);

            backend.Exp(bufA, bufB, a.Length);
            AssertAllClose(ExpCpu(a), DownloadAfterSync(backend, bufB), 1e-3f);

            backend.Exp2(bufA, bufB, a.Length);
            AssertAllClose(Exp2Cpu(a), DownloadAfterSync(backend, bufB), 1e-3f);

            backend.Exp10(bufA, bufB, a.Length);
            AssertAllClose(Exp10Cpu(a), DownloadAfterSync(backend, bufB), 1e-2f);

            backend.ExpM1(bufA, bufB, a.Length);
            AssertAllClose(ExpM1Cpu(a), DownloadAfterSync(backend, bufB), 1e-3f);

            backend.Log(bufA, bufB, a.Length);
            AssertAllClose(LogCpu(a), DownloadAfterSync(backend, bufB), 1e-3f);

            backend.Log2(bufA, bufB, a.Length);
            AssertAllClose(Log2Cpu(a), DownloadAfterSync(backend, bufB), 1e-3f);

            backend.Log1P(bufA, bufB, a.Length);
            AssertAllClose(Log1PCpu(a), DownloadAfterSync(backend, bufB), 1e-3f);

            backend.Sqrt(bufA, bufB, a.Length);
            AssertAllClose(SqrtCpu(a), DownloadAfterSync(backend, bufB), 1e-3f);
        }
    }

    [Fact]
    [Trait("Category", "GPU")]
    public void DirectGpu_Activations_MatchCpu()
    {
        if (!TryGetBackend(out var engine, out var backend))
            return;
        using (engine)
        {
            float[] a = { -3.0f, -1.0f, -0.1f, 0.0f, 0.5f, 2.5f };
            using var bufA = backend.AllocateBuffer(a);
            using var bufB = backend.AllocateBuffer(a.Length);

            backend.Relu(bufA, bufB, a.Length);
            AssertAllClose(ReluCpu(a), DownloadAfterSync(backend, bufB));

            backend.Sigmoid(bufA, bufB, a.Length);
            AssertAllClose(SigmoidCpu(a), DownloadAfterSync(backend, bufB), 1e-3f);

            backend.Tanh(bufA, bufB, a.Length);
            AssertAllClose(TanhCpu(a), DownloadAfterSync(backend, bufB), 1e-3f);

            backend.Gelu(bufA, bufB, a.Length);
            AssertAllClose(GeluCpu(a), DownloadAfterSync(backend, bufB), 1e-3f);
        }
    }

    [Fact]
    [Trait("Category", "GPU")]
    public void DirectGpu_Reductions_MatchCpu()
    {
        if (!TryGetBackend(out var engine, out var backend))
            return;
        using (engine)
        {
            float[] a = { 1.5f, -2.0f, 3.25f, 4.0f, -5.5f, 6.25f };
            using var bufA = backend.AllocateBuffer(a);

            float sum = backend.Sum(bufA, a.Length);
            float sumExpected = SumCpu(a);
            Assert.InRange(sum, sumExpected - 1e-3f, sumExpected + 1e-3f);

            float max = backend.Max(bufA, a.Length);
            float maxExpected = MaxCpu(a);
            Assert.InRange(max, maxExpected - 1e-3f, maxExpected + 1e-3f);

            float[] axisInput = { 1f, 2f, 3f, 4f, 5f, 6f };
            using var bufAxisIn = backend.AllocateBuffer(axisInput);
            using var bufAxisOut = backend.AllocateBuffer(3);
            backend.SumAxis(bufAxisIn, bufAxisOut, outerSize: 3, reduceSize: 2);
            AssertAllClose(new[] { 3f, 7f, 11f }, DownloadAfterSync(backend, bufAxisOut));
        }
    }

    [Fact]
    [Trait("Category", "GPU")]
    public void DirectGpu_Softmax_MatchCpu()
    {
        if (!TryGetBackend(out var engine, out var backend))
            return;
        using (engine)
        {
            int batch = 2;
            int features = 3;
            float[] input = { 1.0f, 2.0f, 3.0f, -1.0f, 0.5f, 0.25f };
            using var bufA = backend.AllocateBuffer(input);
            using var bufB = backend.AllocateBuffer(input.Length);

            backend.Softmax(bufA, bufB, batch, features);
            var result = DownloadAfterSync(backend, bufB);
            var expected = SoftmaxCpu(input, batch, features);
            AssertAllClose(expected, result, 1e-4f);
        }
    }
#else
    [Fact]
    public void DirectGpuCorrectness_NotAvailableOnNet462()
    {
        _output.WriteLine("DirectGpu correctness tests not available on .NET Framework 4.6.2");
        Assert.True(true);
    }
#endif

#if !NET462
    private bool TryGetBackend(out DirectGpuEngine engine, out IDirectGpuBackend backend)
    {
        engine = new DirectGpuEngine();
        if (!engine.IsAvailable || engine.Backend == null)
        {
            _output.WriteLine("DirectGpu not available - skipping test.");
            engine.Dispose();
            backend = null!;
            return false;
        }

        backend = engine.Backend;
        _output.WriteLine($"DirectGpu backend: {engine.BackendName} on {engine.DeviceName}");
        return true;
    }

    private static float[] DownloadAfterSync(IDirectGpuBackend backend, IGpuBuffer buffer)
    {
        backend.Synchronize();
        return backend.DownloadBuffer(buffer);
    }

    private static void AssertAllClose(float[] expected, float[] actual, float tol = 1e-3f)
    {
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++)
        {
            float e = expected[i];
            float a = actual[i];
            if (float.IsNaN(e))
            {
                Assert.True(float.IsNaN(a), $"Index {i}: expected NaN, got {a}");
                continue;
            }
            Assert.InRange(a, e - tol, e + tol);
        }
    }

    private static float[] AddCpu(float[] a, float[] b)
    {
        var result = new float[a.Length];
        for (int i = 0; i < a.Length; i++)
            result[i] = a[i] + b[i];
        return result;
    }

    private static float[] SubCpu(float[] a, float[] b)
    {
        var result = new float[a.Length];
        for (int i = 0; i < a.Length; i++)
            result[i] = a[i] - b[i];
        return result;
    }

    private static float[] MulCpu(float[] a, float[] b)
    {
        var result = new float[a.Length];
        for (int i = 0; i < a.Length; i++)
            result[i] = a[i] * b[i];
        return result;
    }

    private static float[] DivCpu(float[] a, float[] b)
    {
        var result = new float[a.Length];
        for (int i = 0; i < a.Length; i++)
            result[i] = a[i] / b[i];
        return result;
    }

    private static float[] MinCpu(float[] a, float[] b)
    {
        var result = new float[a.Length];
        for (int i = 0; i < a.Length; i++)
            result[i] = a[i] < b[i] ? a[i] : b[i];
        return result;
    }

    private static float[] MaxCpu(float[] a, float[] b)
    {
        var result = new float[a.Length];
        for (int i = 0; i < a.Length; i++)
            result[i] = a[i] > b[i] ? a[i] : b[i];
        return result;
    }

    private static float[] ScaleCpu(float[] a, float scalar)
    {
        var result = new float[a.Length];
        for (int i = 0; i < a.Length; i++)
            result[i] = a[i] * scalar;
        return result;
    }

    private static float[] PowCpu(float[] a, float exponent)
    {
        var result = new float[a.Length];
        for (int i = 0; i < a.Length; i++)
            result[i] = MathF.Pow(a[i], exponent);
        return result;
    }

    private static float[] AbsCpu(float[] a)
    {
        var result = new float[a.Length];
        for (int i = 0; i < a.Length; i++)
            result[i] = MathF.Abs(a[i]);
        return result;
    }

    private static float[] SignCpu(float[] a)
    {
        var result = new float[a.Length];
        for (int i = 0; i < a.Length; i++)
        {
            if (a[i] > 0)
                result[i] = 1.0f;
            else if (a[i] < 0)
                result[i] = -1.0f;
            else
                result[i] = 0.0f;
        }
        return result;
    }

    private static float[] ExpCpu(float[] a)
    {
        var result = new float[a.Length];
        for (int i = 0; i < a.Length; i++)
            result[i] = MathF.Exp(a[i]);
        return result;
    }

    private static float[] Exp2Cpu(float[] a)
    {
        var result = new float[a.Length];
        for (int i = 0; i < a.Length; i++)
            result[i] = MathF.Pow(2.0f, a[i]);
        return result;
    }

    private static float[] Exp10Cpu(float[] a)
    {
        var result = new float[a.Length];
        for (int i = 0; i < a.Length; i++)
            result[i] = MathF.Pow(10.0f, a[i]);
        return result;
    }

    private static float[] ExpM1Cpu(float[] a)
    {
        var result = new float[a.Length];
        for (int i = 0; i < a.Length; i++)
            result[i] = MathF.Exp(a[i]) - 1.0f;
        return result;
    }

    private static float[] LogCpu(float[] a)
    {
        var result = new float[a.Length];
        for (int i = 0; i < a.Length; i++)
            result[i] = MathF.Log(a[i]);
        return result;
    }

    private static float[] Log2Cpu(float[] a)
    {
        const float invLog2 = 1.4426950409f;
        var result = new float[a.Length];
        for (int i = 0; i < a.Length; i++)
            result[i] = MathF.Log(a[i]) * invLog2;
        return result;
    }

    private static float[] Log1PCpu(float[] a)
    {
        var result = new float[a.Length];
        for (int i = 0; i < a.Length; i++)
            result[i] = MathF.Log(1.0f + a[i]);
        return result;
    }

    private static float[] SqrtCpu(float[] a)
    {
        var result = new float[a.Length];
        for (int i = 0; i < a.Length; i++)
            result[i] = MathF.Sqrt(a[i]);
        return result;
    }

    private static float[] ReluCpu(float[] a)
    {
        var result = new float[a.Length];
        for (int i = 0; i < a.Length; i++)
            result[i] = a[i] > 0 ? a[i] : 0.0f;
        return result;
    }

    private static float[] SigmoidCpu(float[] a)
    {
        var result = new float[a.Length];
        for (int i = 0; i < a.Length; i++)
            result[i] = 1.0f / (1.0f + MathF.Exp(-a[i]));
        return result;
    }

    private static float[] TanhCpu(float[] a)
    {
        var result = new float[a.Length];
        for (int i = 0; i < a.Length; i++)
            result[i] = MathF.Tanh(a[i]);
        return result;
    }

    private static float[] GeluCpu(float[] a)
    {
        const float sqrt2OverPi = 0.7978845608f;
        const float coeff = 0.044715f;
        var result = new float[a.Length];
        for (int i = 0; i < a.Length; i++)
        {
            float x = a[i];
            float x3 = x * x * x;
            float inner = sqrt2OverPi * (x + coeff * x3);
            result[i] = 0.5f * x * (1.0f + MathF.Tanh(inner));
        }
        return result;
    }

    private static float SumCpu(float[] a)
    {
        float sum = 0.0f;
        for (int i = 0; i < a.Length; i++)
            sum += a[i];
        return sum;
    }

    private static float MaxCpu(float[] a)
    {
        float max = float.MinValue;
        for (int i = 0; i < a.Length; i++)
            if (a[i] > max) max = a[i];
        return max;
    }

    private static float[] SoftmaxCpu(float[] input, int batch, int features)
    {
        var output = new float[input.Length];
        for (int b = 0; b < batch; b++)
        {
            int baseIdx = b * features;
            float max = float.MinValue;
            for (int f = 0; f < features; f++)
            {
                float v = input[baseIdx + f];
                if (v > max) max = v;
            }

            float sum = 0.0f;
            for (int f = 0; f < features; f++)
            {
                float expVal = MathF.Exp(input[baseIdx + f] - max);
                output[baseIdx + f] = expVal;
                sum += expVal;
            }

            float invSum = 1.0f / sum;
            for (int f = 0; f < features; f++)
                output[baseIdx + f] *= invSum;
        }

        return output;
    }
#endif
}
