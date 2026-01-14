#if NET8_0_OR_GREATER
using System;
using System.Diagnostics;
using AiDotNet.Tensors;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.Engines.Gpu;

namespace AiDotNetBenchmarkTests;

internal static class GpuResidentQuickHarness
{
    private const int MatrixSize = 512;
    private const int VectorSize = 1_000_000;

    public static void Run(int warmup, int iterations)
    {
        warmup = Math.Max(0, warmup);
        iterations = Math.Max(1, iterations);

        Console.WriteLine("GPU-resident quick harness");

        if (!AiDotNetEngine.AutoDetectAndConfigureGpu())
        {
            Console.WriteLine("GPU not available; skipping.");
            return;
        }

        if (AiDotNetEngine.Current is not DirectGpuTensorEngine gpuEngine || !gpuEngine.SupportsGpu)
        {
            Console.WriteLine("GPU engine not available; skipping.");
            return;
        }

        var backend = gpuEngine.GetBackend();
        if (backend == null)
        {
            Console.WriteLine("GPU backend not available; skipping.");
            return;
        }

        var options = new GpuExecutionOptions
        {
            ForceGpu = true,
            EnableGpuResidency = true
        };

        using var ctx = gpuEngine.BeginGpuContext(options);
        if (ctx == null)
        {
            Console.WriteLine("GPU context not available; skipping.");
            return;
        }

        Console.WriteLine($"Engine: {gpuEngine.Name}");
        Console.WriteLine($"Warmup: {warmup}, Iterations: {iterations}");
        Console.WriteLine($"MatrixSize: {MatrixSize}, VectorSize: {VectorSize}");

        var matrixA = new Tensor<float>(CreateData(MatrixSize * MatrixSize, MatrixSize), new[] { MatrixSize, MatrixSize });
        var matrixB = new Tensor<float>(CreateData(MatrixSize * MatrixSize, MatrixSize + 1337), new[] { MatrixSize, MatrixSize });
        var vectorA = new Tensor<float>(CreateData(VectorSize, VectorSize), new[] { VectorSize });
        var vectorB = new Tensor<float>(CreateData(VectorSize, VectorSize + 999), new[] { VectorSize });

        var gpuMatrixA = gpuEngine.UploadToContext(matrixA, GpuTensorRole.Activation);
        var gpuMatrixB = gpuEngine.UploadToContext(matrixB, GpuTensorRole.Activation);
        var gpuVectorA = gpuEngine.UploadToContext(vectorA, GpuTensorRole.Activation);
        var gpuVectorB = gpuEngine.UploadToContext(vectorB, GpuTensorRole.Activation);

        if (gpuMatrixA == null || gpuMatrixB == null || gpuVectorA == null || gpuVectorB == null)
        {
            Console.WriteLine("GPU upload failed; skipping.");
            return;
        }

        var addOutput = backend.AllocateBuffer(VectorSize);
        var multiplyOutput = backend.AllocateBuffer(VectorSize);

        var convInput = CreateConvInput();
        var convKernel = CreateConvKernel();
        var gpuConvInput = gpuEngine.UploadToContext(convInput, GpuTensorRole.Activation);

        if (gpuConvInput == null)
        {
            Console.WriteLine("GPU conv upload failed; skipping conv2d.");
        }

        try
        {
            RunTimed("MatMul (GPU resident)", warmup, iterations, ctx, () =>
                gpuEngine.MatMulGpuTensors(gpuMatrixA, gpuMatrixB));

            RunTimed("Add (GPU resident)", warmup, iterations, ctx, () =>
                gpuEngine.AddGpu(gpuVectorA, gpuVectorB));

            RunTimed("Add (reused output)", warmup, iterations, ctx, () =>
            {
                backend.Add(gpuVectorA.Buffer, gpuVectorB.Buffer, addOutput, VectorSize);
                return null;
            });

            RunTimed("Multiply (GPU resident)", warmup, iterations, ctx, () =>
                gpuEngine.MultiplyGpu(gpuVectorA, gpuVectorB));

            RunTimed("Multiply (reused output)", warmup, iterations, ctx, () =>
            {
                backend.Multiply(gpuVectorA.Buffer, gpuVectorB.Buffer, multiplyOutput, VectorSize);
                return null;
            });

            RunTimed("ReLU (GPU resident)", warmup, iterations, ctx, () =>
                gpuEngine.ActivationGpu(gpuVectorA, FusedActivationType.ReLU));

            RunTimed("Sigmoid (GPU resident)", warmup, iterations, ctx, () =>
                gpuEngine.ActivationGpu(gpuVectorA, FusedActivationType.Sigmoid));

            RunTimed("Sum (GPU resident)", warmup, iterations, ctx, () =>
                gpuEngine.SumAxisGpu(gpuVectorA, 0));

            RunTimed("Mean (GPU resident)", warmup, iterations, ctx, () =>
            {
                var sum = gpuEngine.SumAxisGpu(gpuVectorA, 0);
                var mean = gpuEngine.DivideScalarGpu(sum, VectorSize);
                return new CompositeDisposable(sum, mean);
            });

            if (gpuConvInput != null)
            {
                RunTimed("Conv2D (GPU resident)", warmup, iterations, ctx, () =>
                    gpuEngine.FusedConv2DGpu(gpuConvInput, convKernel, null, 1, 1, 1, 1, 1, 1,
                        FusedActivationType.None));
            }
        }
        finally
        {
            addOutput.Dispose();
            multiplyOutput.Dispose();
            gpuMatrixA.Dispose();
            gpuMatrixB.Dispose();
            gpuVectorA.Dispose();
            gpuVectorB.Dispose();
            gpuConvInput?.Dispose();
        }

        Console.WriteLine("GPU-resident harness complete.");
    }

    private static Tensor<float> CreateConvInput()
    {
        const int batch = 1;
        const int inChannels = 16;
        const int height = 64;
        const int width = 64;
        var inputData = CreateData(batch * inChannels * height * width, 7777);
        return new Tensor<float>(inputData, new[] { batch, inChannels, height, width });
    }

    private static Tensor<float> CreateConvKernel()
    {
        const int outChannels = 32;
        const int inChannels = 16;
        const int kernelSize = 3;
        var kernelData = CreateData(outChannels * inChannels * kernelSize * kernelSize, 9999);
        return new Tensor<float>(kernelData, new[] { outChannels, inChannels, kernelSize, kernelSize });
    }

    private static void RunTimed(string name, int warmup, int iterations, GpuExecutionContext ctx, Func<IDisposable?> action)
    {
        ctx.Synchronize();
        for (int i = 0; i < warmup; i++)
        {
            using var result = action();
            ctx.Synchronize();
        }

        ctx.Synchronize();
        var sw = Stopwatch.StartNew();
        for (int i = 0; i < iterations; i++)
        {
            using var result = action();
            ctx.Synchronize();
        }
        sw.Stop();

        var avgMs = sw.Elapsed.TotalMilliseconds / Math.Max(1, iterations);
        Console.WriteLine($"{name}: {avgMs:F3} ms avg");
    }

    private static float[] CreateData(int length, int seedOffset)
    {
        var data = new float[length];
        for (int i = 0; i < length; i++)
        {
            data[i] = DeterministicValue(i + seedOffset);
        }

        return data;
    }

    private static float DeterministicValue(int i)
    {
        unchecked
        {
            uint x = (uint)(i * 1664525 + 1013904223);
            return (x & 0x00FFFFFF) / 16777216f;
        }
    }

    private sealed class CompositeDisposable : IDisposable
    {
        private readonly IDisposable _first;
        private readonly IDisposable _second;

        public CompositeDisposable(IDisposable first, IDisposable second)
        {
            _first = first;
            _second = second;
        }

        public void Dispose()
        {
            _second.Dispose();
            _first.Dispose();
        }
    }
}
#endif
