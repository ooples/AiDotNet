using AiDotNet.LinearAlgebra;
using AiDotNet.Regression;
using AiDotNet.Statistics;
using BenchmarkDotNet.Attributes;
using Accord.Statistics.Models.Regression.Linear;
using MLContext = Microsoft.ML.MLContext;
using Microsoft.ML.Data;

namespace AiDotNetBenchmarkTests.BenchmarkTests
{
    /// <summary>
    /// Benchmarks for regression models comparing AiDotNet with ML.NET and Accord.NET.
    /// </summary>
    [MemoryDiagnoser]
    [RankColumn]
    public class SimpleRegressionBenchmarks
    {
        private Vector<double> _x;
        private Vector<double> _y;
        private double[] _xArray;
        private double[] _yArray;

        [Params(100, 1000, 10000)]
        public int SampleSize { get; set; }

        [GlobalSetup]
        public void Setup()
        {
            _x = new Vector<double>(SampleSize);
            _y = new Vector<double>(SampleSize);
            _xArray = new double[SampleSize];
            _yArray = new double[SampleSize];

            var random = new Random(42);
            for (int i = 0; i < SampleSize; i++)
            {
                var x = random.NextDouble() * 100;
                var y = 2.5 * x + 10.0 + (random.NextDouble() - 0.5) * 5; // y = 2.5x + 10 with noise

                _x[i] = x;
                _y[i] = y;
                _xArray[i] = x;
                _yArray[i] = y;
            }
        }

        [Benchmark(Baseline = true)]
        public void AiDotNet_SimpleRegression_Fit()
        {
            var regression = new SimpleRegression<double>();
            regression.Fit(_x, _y);
        }

        [Benchmark]
        public void Accord_SimpleRegression_Fit()
        {
            var regression = new SimpleLinearRegression();
            regression.Regress(_xArray, _yArray);
        }

        [Benchmark]
        public void AiDotNet_SimpleRegression_FitAndPredict()
        {
            var regression = new SimpleRegression<double>();
            regression.Fit(_x, _y);

            var testX = new Vector<double>(10);
            for (int i = 0; i < 10; i++)
            {
                testX[i] = i * 10.0;
            }
            var predictions = regression.Predict(testX);
        }

        [Benchmark]
        public void Accord_SimpleRegression_FitAndPredict()
        {
            var regression = new SimpleLinearRegression();
            regression.Regress(_xArray, _yArray);

            var predictions = new double[10];
            for (int i = 0; i < 10; i++)
            {
                predictions[i] = regression.Compute(i * 10.0);
            }
        }
    }

    /// <summary>
    /// Benchmarks for statistical operations comparing AiDotNet with Accord.NET.
    /// </summary>
    [MemoryDiagnoser]
    [RankColumn]
    public class StatisticsBenchmarks
    {
        private Vector<double> _data1;
        private Vector<double> _data2;
        private double[] _accordData1;
        private double[] _accordData2;

        [Params(1000, 10000, 100000)]
        public int Size { get; set; }

        [GlobalSetup]
        public void Setup()
        {
            _data1 = new Vector<double>(Size);
            _data2 = new Vector<double>(Size);
            _accordData1 = new double[Size];
            _accordData2 = new double[Size];

            var random = new Random(42);
            for (int i = 0; i < Size; i++)
            {
                var value1 = random.NextDouble() * 100;
                var value2 = random.NextDouble() * 100;

                _data1[i] = value1;
                _data2[i] = value2;
                _accordData1[i] = value1;
                _accordData2[i] = value2;
            }
        }

        [Benchmark(Baseline = true)]
        public double AiDotNet_Mean()
        {
            return StatisticsHelper.Mean(_data1);
        }

        [Benchmark]
        public double Accord_Mean()
        {
            return Accord.Statistics.Measures.Mean(_accordData1);
        }

        [Benchmark]
        public double AiDotNet_Variance()
        {
            return StatisticsHelper.Variance(_data1);
        }

        [Benchmark]
        public double Accord_Variance()
        {
            return Accord.Statistics.Measures.Variance(_accordData1);
        }

        [Benchmark]
        public double AiDotNet_StandardDeviation()
        {
            return StatisticsHelper.StandardDeviation(_data1);
        }

        [Benchmark]
        public double Accord_StandardDeviation()
        {
            return Accord.Statistics.Measures.StandardDeviation(_accordData1);
        }

        [Benchmark]
        public double AiDotNet_Median()
        {
            return StatisticsHelper.Median(_data1);
        }

        [Benchmark]
        public double Accord_Median()
        {
            return Accord.Statistics.Measures.Median(_accordData1);
        }

        [Benchmark]
        public double AiDotNet_Correlation()
        {
            return StatisticsHelper.Correlation(_data1, _data2);
        }

        [Benchmark]
        public double Accord_Correlation()
        {
            return Accord.Statistics.Measures.Correlation(_accordData1, _accordData2);
        }

        [Benchmark]
        public double AiDotNet_Covariance()
        {
            return StatisticsHelper.Covariance(_data1, _data2);
        }

        [Benchmark]
        public double Accord_Covariance()
        {
            return Accord.Statistics.Measures.Covariance(_accordData1, _accordData2);
        }
    }

    /// <summary>
    /// Benchmarks for multiple regression models.
    /// </summary>
    [MemoryDiagnoser]
    [RankColumn]
    public class MultipleRegressionBenchmarks
    {
        private Matrix<double> _X;
        private Vector<double> _y;
        private double[][] _accordX;
        private double[] _accordY;

        [Params(100, 1000)]
        public int Samples { get; set; }

        [Params(5, 20)]
        public int Features { get; set; }

        [GlobalSetup]
        public void Setup()
        {
            _X = new Matrix<double>(Samples, Features);
            _y = new Vector<double>(Samples);
            _accordX = new double[Samples][];
            _accordY = new double[Samples];

            var random = new Random(42);

            // Generate synthetic data: y = sum(features) + noise
            for (int i = 0; i < Samples; i++)
            {
                _accordX[i] = new double[Features];
                double sum = 0;

                for (int j = 0; j < Features; j++)
                {
                    var value = random.NextDouble() * 10;
                    _X[i, j] = value;
                    _accordX[i][j] = value;
                    sum += value;
                }

                var y = sum + (random.NextDouble() - 0.5) * 2; // Add noise
                _y[i] = y;
                _accordY[i] = y;
            }
        }

        [Benchmark(Baseline = true)]
        public void AiDotNet_MultipleRegression_Fit()
        {
            var regression = new MultipleRegression<double>();
            regression.Fit(_X, _y);
        }

        [Benchmark]
        public void Accord_MultipleRegression_Fit()
        {
            var regression = new MultivariateLinearRegression();
            regression.Regress(_accordX, _accordY);
        }
    }

    /// <summary>
    /// Benchmarks for polynomial regression.
    /// </summary>
    [MemoryDiagnoser]
    [RankColumn]
    public class PolynomialRegressionBenchmarks
    {
        private Vector<double> _x;
        private Vector<double> _y;

        [Params(100, 1000, 5000)]
        public int SampleSize { get; set; }

        [Params(2, 3, 5)]
        public int Degree { get; set; }

        [GlobalSetup]
        public void Setup()
        {
            _x = new Vector<double>(SampleSize);
            _y = new Vector<double>(SampleSize);

            var random = new Random(42);
            for (int i = 0; i < SampleSize; i++)
            {
                var x = (random.NextDouble() - 0.5) * 20;
                var y = Math.Pow(x, Degree) + (random.NextDouble() - 0.5) * 10;

                _x[i] = x;
                _y[i] = y;
            }
        }

        [Benchmark]
        public void AiDotNet_PolynomialRegression_Fit()
        {
            var regression = new PolynomialRegression<double>(degree: Degree);
            regression.Fit(_x, _y);
        }
    }
}
