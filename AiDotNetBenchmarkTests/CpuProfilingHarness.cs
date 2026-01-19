using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNetBenchmarkTests;

internal enum CpuProfilingMode
{
    All,
    MatMul,
    Conv2D
}

internal sealed class CpuProfilingOptions
{
    public CpuProfilingMode Mode { get; set; } = CpuProfilingMode.All;
    public int Iterations { get; set; } = 25;
}

internal static class CpuProfilingHarness
{
    private static readonly int[] MatMulSizes = [256, 512];

    private static volatile float _sink;

    public static bool TryParseArgs(string[] args, out CpuProfilingOptions options, out string[] remainingArgs)
    {
        options = new CpuProfilingOptions();
        var remaining = new List<string>(args.Length);

        var mode = CpuProfilingMode.All;
        var iterations = 25;
        var enabled = false;

        for (int i = 0; i < args.Length; i++)
        {
            var arg = args[i];
            if (arg.StartsWith("--profileCpu", StringComparison.OrdinalIgnoreCase))
            {
                enabled = true;
                string? value = null;
                var parts = arg.Split(new[] { '=' }, 2);
                if (parts.Length == 2)
                {
                    value = parts[1];
                }
                else if (i + 1 < args.Length && !args[i + 1].StartsWith("-", StringComparison.Ordinal))
                {
                    value = args[++i];
                }

                if (!string.IsNullOrWhiteSpace(value))
                {
                    mode = ParseMode(value);
                }

                continue;
            }

            if (arg.StartsWith("--profileIterations", StringComparison.OrdinalIgnoreCase))
            {
                enabled = true;
                if (TryParseIntArg(arg, args, ref i, out var parsed))
                {
                    iterations = parsed;
                }

                continue;
            }

            remaining.Add(arg);
        }

        remainingArgs = remaining.ToArray();

        options = new CpuProfilingOptions
        {
            Mode = mode,
            Iterations = Math.Max(1, iterations)
        };

        return enabled;
    }

    public static void Run(CpuProfilingOptions options)
    {
        AiDotNetEngine.Current = new CpuEngine();

        var (matA, matB) = BuildMatMulInputs();
        var (convInput, convKernel, stride, padding, dilation) = BuildConv2DInputs();

        // Warm-up to avoid JIT noise in the trace.
        _ = AiDotNetEngine.Current.TensorMatMul(matA[MatMulSizes[0]], matB[MatMulSizes[0]]);
        _ = AiDotNetEngine.Current.Conv2D(convInput, convKernel, stride, padding, dilation);

        if (options.Mode is CpuProfilingMode.All or CpuProfilingMode.MatMul)
        {
            for (int iter = 0; iter < options.Iterations; iter++)
            {
                foreach (var size in MatMulSizes)
                {
                    var result = AiDotNetEngine.Current.TensorMatMul(matA[size], matB[size]);
                    _sink = result.GetFlat(0);
                }
            }
        }

        if (options.Mode is CpuProfilingMode.All or CpuProfilingMode.Conv2D)
        {
            for (int iter = 0; iter < options.Iterations; iter++)
            {
                var result = AiDotNetEngine.Current.Conv2D(convInput, convKernel, stride, padding, dilation);
                _sink = result.GetFlat(0);
            }
        }
    }

    private static (Dictionary<int, Tensor<float>> a, Dictionary<int, Tensor<float>> b) BuildMatMulInputs()
    {
        var matricesA = new Dictionary<int, Tensor<float>>();
        var matricesB = new Dictionary<int, Tensor<float>>();

        foreach (var size in MatMulSizes)
        {
            var dataA = CreateData(size * size, seedOffset: size);
            var dataB = CreateData(size * size, seedOffset: size + 13_337);

            matricesA[size] = new Tensor<float>(dataA, new[] { size, size });
            matricesB[size] = new Tensor<float>(dataB, new[] { size, size });
        }

        return (matricesA, matricesB);
    }

    private static (Tensor<float> input, Tensor<float> kernel, int[] stride, int[] padding, int[] dilation) BuildConv2DInputs()
    {
        const int batch = 1;
        const int inChannels = 16;
        const int height = 64;
        const int width = 64;
        const int outChannels = 32;
        const int kernelSize = 3;

        var stride = new[] { 1, 1 };
        var padding = new[] { 1, 1 };
        var dilation = new[] { 1, 1 };

        var inputData = CreateData(batch * inChannels * height * width, seedOffset: 7_777);
        var kernelData = CreateData(outChannels * inChannels * kernelSize * kernelSize, seedOffset: 9_999);

        var input = new Tensor<float>(inputData, new[] { batch, inChannels, height, width });
        var kernel = new Tensor<float>(kernelData, new[] { outChannels, inChannels, kernelSize, kernelSize });

        return (input, kernel, stride, padding, dilation);
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
            int value = i * 1_664_525 + 1_013_904_223;
            return (value & 0x00FFFFFF) / (float)0x01000000;
        }
    }

    private static CpuProfilingMode ParseMode(string value)
    {
        if (value.Equals("matmul", StringComparison.OrdinalIgnoreCase))
        {
            return CpuProfilingMode.MatMul;
        }

        if (value.Equals("conv2d", StringComparison.OrdinalIgnoreCase))
        {
            return CpuProfilingMode.Conv2D;
        }

        return CpuProfilingMode.All;
    }

    private static bool TryParseIntArg(string arg, string[] args, ref int index, out int value)
    {
        value = 0;
        var parts = arg.Split(new[] { '=' }, 2);
        if (parts.Length == 2)
        {
            return int.TryParse(parts[1], out value);
        }

        if (index + 1 >= args.Length)
        {
            return false;
        }

        if (int.TryParse(args[index + 1], out value))
        {
            index++;
            return true;
        }

        return false;
    }
}
