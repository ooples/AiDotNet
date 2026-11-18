using System;
using System.IO;
using System.Security.Cryptography;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;
using Xunit.Abstractions;

namespace AiDotNet.Tests.Helpers;

/// <summary>Strict capability and execution checks for required GPU tests. Unavailability fails.</summary>
internal sealed class RequiredGpuTestContract : IDisposable
{
    private readonly ITestOutputHelper _output;
    private bool _disposed;
    public GpuEngine Engine { get; }
    // Vector/matrix GPU methods explicitly implement IEngine; concrete dispatch can execute CPU code.
    public IEngine Dispatch => Engine;

    public RequiredGpuTestContract(ITestOutputHelper output)
    {
        _output = output;
        Engine = new GpuEngine();
        try
        {
            _output.WriteLine("Engine: " + Engine.Name);
            Assert.True(Engine.SupportsGpu, "Required GPU unavailable: " + GpuEngine.GetDiagnosticReport());
            var backend = Assert.IsType<CudaBackend>(Engine.GetBackend());
            Assert.True(backend.IsAvailable);
            _output.WriteLine("RequiredGpuContract: backend=CUDA; device=" + backend.DeviceName);
            foreach (var assembly in new[] { typeof(RequiredGpuTestContract).Assembly, typeof(AiDotNet.NeuralNetworks.NeuralNetworkBase<float>).Assembly, typeof(GpuEngine).Assembly })
            {
                using var stream = File.OpenRead(assembly.Location);
                using var sha = SHA256.Create();
                _output.WriteLine($"Assembly: {assembly.FullName}; SHA256={BitConverter.ToString(sha.ComputeHash(stream)).Replace("-", "")}; path={assembly.Location}");
            }
            GpuLaunchProbe.Reset();
        }
        catch { Engine.Dispose(); throw; }
    }

    public void AssertExecuted()
    {
        Engine.SynchronizeStream();
        _output.WriteLine($"GPU launches={GpuLaunchProbe.Count}; kernel misses={GpuLaunchProbe.KernelMisses}; fallbacks={string.Join("; ", GpuLaunchProbe.Fallbacks)}");
        Assert.True(GpuLaunchProbe.Count > 0, "Required GPU test performed no device work.");
        Assert.Equal(0, GpuLaunchProbe.KernelMisses);
        Assert.Empty(GpuLaunchProbe.Fallbacks);
    }

    public static void CheckAddition(IEngine engine, int seed)
    {
        var left = new Vector<float>(1024);
        var right = new Vector<float>(1024);
        for (int i = 0; i < left.Length; i++) { left[i] = i * 0.125f + seed; right[i] = seed - i * 0.25f; }
        var result = engine.Add(left, right);
        Assert.Equal(left.Length, result.Length);
        for (int i = 0; i < left.Length; i++)
        {
            Assert.Equal(i * 0.125f + seed, left[i]);
            Assert.Equal(seed - i * 0.25f, right[i]);
            Assert.Equal(left[i] + right[i], result[i], precision: 5);
        }
    }

    public static void CheckMatrix(IEngine engine, int seed)
    {
        var left = new Matrix<float>(16, 16);
        var right = new Matrix<float>(16, 16);
        for (int i = 0; i < 16; i++)
        for (int j = 0; j < 16; j++) { left[i, j] = (i - j + seed) * 0.125f; right[i, j] = (i + j - seed) * 0.0625f; }
        var result = engine.MatrixMultiply(left, right);
        // The current GPU matrix contract returns a deferred device result. The CPU fallback
        // returns host data, even when an earlier failed cuBLAS call incremented the launch probe.
        // Check this result's own registration before materializing it; global counters alone
        // cannot attribute a dispatch when the other workers are running concurrently.
        var backing = result.GetBackingArrayUnsafe();
        Assert.NotNull(backing);
        Assert.True(DeferredArrayMaterializer.IsPending(backing), "MatrixMultiply returned a CPU result in the required GPU lane.");
        Assert.Equal(16, result.Rows);
        Assert.Equal(16, result.Columns);
        for (int i = 0; i < 16; i++)
        for (int j = 0; j < 16; j++)
        {
            Assert.Equal((i - j + seed) * 0.125f, left[i, j]);
            Assert.Equal((i + j - seed) * 0.0625f, right[i, j]);
            float expected = 0;
            for (int k = 0; k < 16; k++) expected += ((i - k + seed) * 0.125f) * ((k + j - seed) * 0.0625f);
            Assert.Equal(expected, result[i, j], precision: 4);
        }
    }

    public void Dispose()
    {
        if (_disposed) return;
        Engine.Dispose();
        _disposed = true;
    }
}
