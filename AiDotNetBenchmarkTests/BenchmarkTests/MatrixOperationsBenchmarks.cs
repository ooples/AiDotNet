using AiDotNet.LinearAlgebra;
using BenchmarkDotNet.Attributes;
using Accord.Math;
using MLMatrix = Microsoft.ML.Data.VBuffer<double>;

namespace AiDotNetBenchmarkTests.BenchmarkTests
{
    /// <summary>
    /// Benchmarks for matrix operations comparing AiDotNet with Accord.NET and ML.NET.
    /// Measures performance of core linear algebra operations.
    /// </summary>
    [MemoryDiagnoser]
    [RankColumn]
    public class MatrixOperationsBenchmarks
    {
        private Matrix<double> _aiMatrix1;
        private Matrix<double> _aiMatrix2;
        private double[,] _accordMatrix1;
        private double[,] _accordMatrix2;

        [Params(10, 100, 500)]
        public int Size { get; set; }

        [GlobalSetup]
        public void Setup()
        {
            // Initialize AiDotNet matrices
            _aiMatrix1 = new Matrix<double>(Size, Size);
            _aiMatrix2 = new Matrix<double>(Size, Size);

            // Initialize Accord matrices
            _accordMatrix1 = new double[Size, Size];
            _accordMatrix2 = new double[Size, Size];

            // Fill with random data
            var random = new Random(42);
            for (int i = 0; i < Size; i++)
            {
                for (int j = 0; j < Size; j++)
                {
                    var value = random.NextDouble();
                    _aiMatrix1[i, j] = value;
                    _accordMatrix1[i, j] = value;

                    value = random.NextDouble();
                    _aiMatrix2[i, j] = value;
                    _accordMatrix2[i, j] = value;
                }
            }
        }

        [Benchmark(Baseline = true)]
        public Matrix<double> AiDotNet_MatrixMultiplication()
        {
            return _aiMatrix1 * _aiMatrix2;
        }

        [Benchmark]
        public double[,] Accord_MatrixMultiplication()
        {
            return _accordMatrix1.Dot(_accordMatrix2);
        }

        [Benchmark]
        public Matrix<double> AiDotNet_MatrixTranspose()
        {
            return _aiMatrix1.Transpose();
        }

        [Benchmark]
        public double[,] Accord_MatrixTranspose()
        {
            return _accordMatrix1.Transpose();
        }

        [Benchmark]
        public Matrix<double> AiDotNet_MatrixAddition()
        {
            return _aiMatrix1 + _aiMatrix2;
        }

        [Benchmark]
        public double[,] Accord_MatrixAddition()
        {
            return _accordMatrix1.Add(_accordMatrix2);
        }

        [Benchmark]
        public double AiDotNet_MatrixDeterminant()
        {
            return _aiMatrix1.Determinant();
        }

        [Benchmark]
        public double Accord_MatrixDeterminant()
        {
            return _accordMatrix1.Determinant();
        }

        [Benchmark]
        public Matrix<double> AiDotNet_MatrixInverse()
        {
            return _aiMatrix1.Inverse();
        }

        [Benchmark]
        public double[,] Accord_MatrixInverse()
        {
            return _accordMatrix1.Inverse();
        }
    }

    /// <summary>
    /// Benchmarks for vector operations.
    /// </summary>
    [MemoryDiagnoser]
    [RankColumn]
    public class VectorOperationsBenchmarks
    {
        private Vector<double> _aiVector1;
        private Vector<double> _aiVector2;
        private double[] _accordVector1;
        private double[] _accordVector2;

        [Params(100, 1000, 10000)]
        public int Size { get; set; }

        [GlobalSetup]
        public void Setup()
        {
            // Initialize vectors
            _aiVector1 = new Vector<double>(Size);
            _aiVector2 = new Vector<double>(Size);
            _accordVector1 = new double[Size];
            _accordVector2 = new double[Size];

            // Fill with random data
            var random = new Random(42);
            for (int i = 0; i < Size; i++)
            {
                var value1 = random.NextDouble();
                var value2 = random.NextDouble();

                _aiVector1[i] = value1;
                _accordVector1[i] = value1;

                _aiVector2[i] = value2;
                _accordVector2[i] = value2;
            }
        }

        [Benchmark(Baseline = true)]
        public double AiDotNet_DotProduct()
        {
            return _aiVector1.DotProduct(_aiVector2);
        }

        [Benchmark]
        public double Accord_DotProduct()
        {
            return _accordVector1.Dot(_accordVector2);
        }

        [Benchmark]
        public Vector<double> AiDotNet_VectorAddition()
        {
            return _aiVector1 + _aiVector2;
        }

        [Benchmark]
        public double[] Accord_VectorAddition()
        {
            return _accordVector1.Add(_accordVector2);
        }

        [Benchmark]
        public double AiDotNet_VectorMagnitude()
        {
            return _aiVector1.Magnitude();
        }

        [Benchmark]
        public double Accord_VectorMagnitude()
        {
            return _accordVector1.Euclidean();
        }

        [Benchmark]
        public Vector<double> AiDotNet_VectorNormalize()
        {
            return _aiVector1.Normalize();
        }

        [Benchmark]
        public double[] Accord_VectorNormalize()
        {
            return _accordVector1.Divide(_accordVector1.Euclidean());
        }

        [Benchmark]
        public double AiDotNet_EuclideanDistance()
        {
            return _aiVector1.EuclideanDistance(_aiVector2);
        }

        [Benchmark]
        public double Accord_EuclideanDistance()
        {
            return Accord.Math.Distance.Euclidean(_accordVector1, _accordVector2);
        }

        [Benchmark]
        public double AiDotNet_CosineSimilarity()
        {
            return _aiVector1.CosineSimilarity(_aiVector2);
        }

        [Benchmark]
        public double Accord_CosineSimilarity()
        {
            return Accord.Math.Distance.Cosine(_accordVector1, _accordVector2);
        }
    }
}
