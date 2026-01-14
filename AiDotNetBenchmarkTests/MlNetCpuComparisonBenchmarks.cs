#if NET8_0_OR_GREATER
using AiDotNet.Tensors.Engines;
using System.Reflection;
using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Engines;
using BenchmarkDotNet.Jobs;
using Microsoft.ML;
using Microsoft.ML.Data;

namespace AiDotNetBenchmarkTests;

[MemoryDiagnoser]
[SimpleJob(RuntimeMoniker.Net10_0, launchCount: 1, warmupCount: 3, iterationCount: 5)]
public class MlNetCpuComparisonBenchmarks
{
    private static readonly int[] VectorSizes = [100_000, 1_000_000];

    private delegate void MulElementWiseDelegate(ref VBuffer<float> src, ref VBuffer<float> dst);

    private static readonly Type VectorUtilsType = ResolveVectorUtilsType();
    private static readonly Action<float[], float[]> AddVectors = CreateAddDelegate();
    private static readonly Func<float[], float> SumVector = CreateSumDelegate();
    private static readonly MulElementWiseDelegate MulElementWise = CreateMulElementWiseDelegate();

    private readonly Dictionary<int, Tensor<float>> _aiVectorsA = new();
    private readonly Dictionary<int, Tensor<float>> _aiVectorsB = new();
    private readonly Dictionary<int, float[]> _mlVectorsA = new();
    private readonly Dictionary<int, float[]> _mlVectorsB = new();
    private readonly Dictionary<int, float[]> _mlAddTargets = new();
    private readonly Dictionary<int, float[]> _mlMultiplyTargets = new();
    private readonly Dictionary<int, VBuffer<float>> _mlVBufferA = new();
    private readonly Consumer _consumer = new();

    [GlobalSetup]
    public void Setup()
    {
        AiDotNetEngine.Current = new CpuEngine();

        foreach (var size in VectorSizes)
        {
            var dataA = CreateData(size, seedOffset: size);
            var dataB = CreateData(size, seedOffset: size + 13_337);

            _aiVectorsA[size] = new Tensor<float>(dataA, new[] { size });
            _aiVectorsB[size] = new Tensor<float>(dataB, new[] { size });

            _mlVectorsA[size] = dataA;
            _mlVectorsB[size] = dataB;
            _mlAddTargets[size] = new float[size];
            _mlMultiplyTargets[size] = new float[size];
            _mlVBufferA[size] = new VBuffer<float>(size, dataA);
        }

        Warmup();
    }

    private void Warmup()
    {
        _ = AiDotNetEngine.Current.TensorAdd(_aiVectorsA[VectorSizes[0]], _aiVectorsB[VectorSizes[0]]);
        var addDst = _mlAddTargets[VectorSizes[0]];
        Array.Copy(_mlVectorsB[VectorSizes[0]], addDst, VectorSizes[0]);
        AddVectors(_mlVectorsA[VectorSizes[0]], addDst);

        _ = AiDotNetEngine.Current.TensorMultiply(_aiVectorsA[VectorSizes[0]], _aiVectorsB[VectorSizes[0]]);
        var mulDst = _mlMultiplyTargets[VectorSizes[0]];
        Array.Copy(_mlVectorsB[VectorSizes[0]], mulDst, VectorSizes[0]);
        var mulSrc = _mlVBufferA[VectorSizes[0]];
        var mulVBuffer = new VBuffer<float>(VectorSizes[0], mulDst);
        MulElementWise(ref mulSrc, ref mulVBuffer);

        _ = AiDotNetEngine.Current.TensorSum(_aiVectorsA[VectorSizes[0]]);
        _ = SumVector(_mlVectorsA[VectorSizes[0]]);

        _ = AiDotNetEngine.Current.TensorMean(_aiVectorsA[VectorSizes[0]]);
        _ = SumVector(_mlVectorsA[VectorSizes[0]]) / VectorSizes[0];
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

    [Benchmark]
    [Arguments(100_000)]
    [Arguments(1_000_000)]
    public Tensor<float> AiDotNet_TensorAdd(int size)
    {
        return AiDotNetEngine.Current.TensorAdd(_aiVectorsA[size], _aiVectorsB[size]);
    }

    [Benchmark]
    [Arguments(100_000)]
    [Arguments(1_000_000)]
    public void MlNet_Add(int size)
    {
        var dst = _mlAddTargets[size];
        Array.Copy(_mlVectorsB[size], dst, size);
        AddVectors(_mlVectorsA[size], dst);
        _consumer.Consume(dst);
    }

    [Benchmark]
    [Arguments(100_000)]
    [Arguments(1_000_000)]
    public Tensor<float> AiDotNet_TensorMultiply(int size)
    {
        return AiDotNetEngine.Current.TensorMultiply(_aiVectorsA[size], _aiVectorsB[size]);
    }

    [Benchmark]
    [Arguments(100_000)]
    [Arguments(1_000_000)]
    public void MlNet_Multiply(int size)
    {
        var dstValues = _mlMultiplyTargets[size];
        Array.Copy(_mlVectorsB[size], dstValues, size);
        var src = _mlVBufferA[size];
        var dst = new VBuffer<float>(size, dstValues);
        MulElementWise(ref src, ref dst);
        _consumer.Consume(dstValues);
    }

    [Benchmark]
    public float AiDotNet_TensorSum()
    {
        return AiDotNetEngine.Current.TensorSum(_aiVectorsA[VectorSizes[1]]);
    }

    [Benchmark]
    public float MlNet_Sum()
    {
        return SumVector(_mlVectorsA[VectorSizes[1]]);
    }

    [Benchmark]
    public float AiDotNet_TensorMean()
    {
        return AiDotNetEngine.Current.TensorMean(_aiVectorsA[VectorSizes[1]]);
    }

    [Benchmark]
    public float MlNet_Mean()
    {
        return SumVector(_mlVectorsA[VectorSizes[1]]) / VectorSizes[1];
    }

    private static Type ResolveVectorUtilsType()
    {
        var type = typeof(MLContext).Assembly.GetType("Microsoft.ML.Numeric.VectorUtils");
        if (type == null)
        {
            throw new InvalidOperationException("Microsoft.ML.Numeric.VectorUtils was not found. ML.NET vector utilities are unavailable.");
        }

        return type;
    }

    private static MethodInfo GetVectorUtilsMethod(string name, params Type[] parameters)
    {
        var method = VectorUtilsType.GetMethod(
            name,
            BindingFlags.Static | BindingFlags.Public | BindingFlags.NonPublic,
            binder: null,
            types: parameters,
            modifiers: null);

        if (method == null)
        {
            throw new InvalidOperationException($"VectorUtils method '{name}' was not found with the expected signature.");
        }

        return method;
    }

    private static Action<float[], float[]> CreateAddDelegate()
    {
        var method = GetVectorUtilsMethod("Add", typeof(float[]), typeof(float[]));
        return (Action<float[], float[]>)method.CreateDelegate(typeof(Action<float[], float[]>));
    }

    private static Func<float[], float> CreateSumDelegate()
    {
        var method = GetVectorUtilsMethod("Sum", typeof(float[]));
        return (Func<float[], float>)method.CreateDelegate(typeof(Func<float[], float>));
    }

    private static MulElementWiseDelegate CreateMulElementWiseDelegate()
    {
        var method = GetVectorUtilsMethod(
            "MulElementWise",
            typeof(VBuffer<float>).MakeByRefType(),
            typeof(VBuffer<float>).MakeByRefType());
        return (MulElementWiseDelegate)method.CreateDelegate(typeof(MulElementWiseDelegate));
    }
}
#endif
