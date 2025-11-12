using AiDotNet.LinearAlgebra;
using AiDotNet.Serialization;
using Newtonsoft.Json;
using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Jobs;

namespace AiDotNetBenchmarkTests.BenchmarkTests;

/// <summary>
/// Comprehensive benchmarks for Serialization infrastructure
/// Tests Matrix, Vector, and Tensor JSON serialization/deserialization performance
/// </summary>
[MemoryDiagnoser]
[SimpleJob(RuntimeMoniker.Net462, baseline: true)]
[SimpleJob(RuntimeMoniker.Net60)]
[SimpleJob(RuntimeMoniker.Net70)]
[SimpleJob(RuntimeMoniker.Net80)]
public class SerializationBenchmarks
{
    [Params(10, 100)]
    public int MatrixSize { get; set; }

    [Params(100, 1000)]
    public int VectorSize { get; set; }

    private Matrix<double> _matrix = null!;
    private Vector<double> _vector = null!;
    private Tensor<double> _tensor2D = null!;
    private Tensor<double> _tensor3D = null!;
    private string _serializedMatrix = null!;
    private string _serializedVector = null!;
    private string _serializedTensor2D = null!;
    private string _serializedTensor3D = null!;
    private JsonSerializerSettings _settings = null!;

    [GlobalSetup]
    public void Setup()
    {
        var random = new Random(42);

        // Initialize Matrix
        _matrix = new Matrix<double>(MatrixSize, MatrixSize);
        for (int i = 0; i < MatrixSize; i++)
        {
            for (int j = 0; j < MatrixSize; j++)
            {
                _matrix[i, j] = random.NextDouble() * 2 - 1;
            }
        }

        // Initialize Vector
        _vector = new Vector<double>(VectorSize);
        for (int i = 0; i < VectorSize; i++)
        {
            _vector[i] = random.NextDouble() * 2 - 1;
        }

        // Initialize 2D Tensor
        _tensor2D = new Tensor<double>(new[] { MatrixSize, MatrixSize });
        for (int i = 0; i < MatrixSize; i++)
        {
            for (int j = 0; j < MatrixSize; j++)
            {
                _tensor2D[i, j] = random.NextDouble() * 2 - 1;
            }
        }

        // Initialize 3D Tensor
        _tensor3D = new Tensor<double>(new[] { 10, 10, 10 });
        for (int i = 0; i < 10; i++)
        {
            for (int j = 0; j < 10; j++)
            {
                for (int k = 0; k < 10; k++)
                {
                    _tensor3D[i, j, k] = random.NextDouble() * 2 - 1;
                }
            }
        }

        // Setup JSON serializer with custom converters
        _settings = new JsonSerializerSettings();
        JsonConverterRegistry.RegisterCustomConverters(_settings);

        // Pre-serialize for deserialization benchmarks
        _serializedMatrix = JsonConvert.SerializeObject(_matrix, _settings);
        _serializedVector = JsonConvert.SerializeObject(_vector, _settings);
        _serializedTensor2D = JsonConvert.SerializeObject(_tensor2D, _settings);
        _serializedTensor3D = JsonConvert.SerializeObject(_tensor3D, _settings);
    }

    #region Matrix Serialization

    [Benchmark(Baseline = true)]
    public string Serialization01_Matrix_Serialize()
    {
        return JsonConvert.SerializeObject(_matrix, _settings);
    }

    [Benchmark]
    public Matrix<double> Serialization02_Matrix_Deserialize()
    {
        return JsonConvert.DeserializeObject<Matrix<double>>(_serializedMatrix, _settings)!;
    }

    [Benchmark]
    public Matrix<double> Serialization03_Matrix_RoundTrip()
    {
        var serialized = JsonConvert.SerializeObject(_matrix, _settings);
        return JsonConvert.DeserializeObject<Matrix<double>>(serialized, _settings)!;
    }

    #endregion

    #region Vector Serialization

    [Benchmark]
    public string Serialization04_Vector_Serialize()
    {
        return JsonConvert.SerializeObject(_vector, _settings);
    }

    [Benchmark]
    public Vector<double> Serialization05_Vector_Deserialize()
    {
        return JsonConvert.DeserializeObject<Vector<double>>(_serializedVector, _settings)!;
    }

    [Benchmark]
    public Vector<double> Serialization06_Vector_RoundTrip()
    {
        var serialized = JsonConvert.SerializeObject(_vector, _settings);
        return JsonConvert.DeserializeObject<Vector<double>>(serialized, _settings)!;
    }

    #endregion

    #region Tensor 2D Serialization

    [Benchmark]
    public string Serialization07_Tensor2D_Serialize()
    {
        return JsonConvert.SerializeObject(_tensor2D, _settings);
    }

    [Benchmark]
    public Tensor<double> Serialization08_Tensor2D_Deserialize()
    {
        return JsonConvert.DeserializeObject<Tensor<double>>(_serializedTensor2D, _settings)!;
    }

    [Benchmark]
    public Tensor<double> Serialization09_Tensor2D_RoundTrip()
    {
        var serialized = JsonConvert.SerializeObject(_tensor2D, _settings);
        return JsonConvert.DeserializeObject<Tensor<double>>(serialized, _settings)!;
    }

    #endregion

    #region Tensor 3D Serialization

    [Benchmark]
    public string Serialization10_Tensor3D_Serialize()
    {
        return JsonConvert.SerializeObject(_tensor3D, _settings);
    }

    [Benchmark]
    public Tensor<double> Serialization11_Tensor3D_Deserialize()
    {
        return JsonConvert.DeserializeObject<Tensor<double>>(_serializedTensor3D, _settings)!;
    }

    [Benchmark]
    public Tensor<double> Serialization12_Tensor3D_RoundTrip()
    {
        var serialized = JsonConvert.SerializeObject(_tensor3D, _settings);
        return JsonConvert.DeserializeObject<Tensor<double>>(serialized, _settings)!;
    }

    #endregion

    #region Converter Registration

    [Benchmark]
    public JsonSerializerSettings Serialization13_RegisterCustomConverters()
    {
        var settings = new JsonSerializerSettings();
        JsonConverterRegistry.RegisterCustomConverters(settings);
        return settings;
    }

    #endregion

    #region Multiple Objects Serialization

    [Benchmark]
    public string Serialization14_SerializeMultipleObjects()
    {
        var data = new
        {
            Matrix = _matrix,
            Vector = _vector,
            Tensor = _tensor2D
        };
        return JsonConvert.SerializeObject(data, _settings);
    }

    [Benchmark]
    public (Matrix<double>, Vector<double>, Tensor<double>) Serialization15_DeserializeMultipleObjects()
    {
        var json = JsonConvert.SerializeObject(new
        {
            Matrix = _matrix,
            Vector = _vector,
            Tensor = _tensor2D
        }, _settings);

        var obj = JsonConvert.DeserializeObject<dynamic>(json, _settings);
        return (
            JsonConvert.DeserializeObject<Matrix<double>>(obj.Matrix.ToString(), _settings)!,
            JsonConvert.DeserializeObject<Vector<double>>(obj.Vector.ToString(), _settings)!,
            JsonConvert.DeserializeObject<Tensor<double>>(obj.Tensor.ToString(), _settings)!
        );
    }

    #endregion

    #region Float vs Double Serialization

    [Benchmark]
    public string Serialization16_Matrix_Float_Serialize()
    {
        var floatMatrix = new Matrix<float>(MatrixSize, MatrixSize);
        for (int i = 0; i < MatrixSize; i++)
        {
            for (int j = 0; j < MatrixSize; j++)
            {
                floatMatrix[i, j] = (float)_matrix[i, j];
            }
        }
        return JsonConvert.SerializeObject(floatMatrix, _settings);
    }

    [Benchmark]
    public string Serialization17_Vector_Float_Serialize()
    {
        var floatVector = new Vector<float>(VectorSize);
        for (int i = 0; i < VectorSize; i++)
        {
            floatVector[i] = (float)_vector[i];
        }
        return JsonConvert.SerializeObject(floatVector, _settings);
    }

    #endregion
}
