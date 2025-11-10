using AiDotNet.ActivationFunctions;
using AiDotNet.LinearAlgebra;
using BenchmarkDotNet.Attributes;
using Accord.Neuro;

namespace AiDotNetBenchmarkTests.BenchmarkTests
{
    /// <summary>
    /// Benchmarks for activation functions comparing AiDotNet with Accord.NET.
    /// Measures performance of forward and backward passes.
    /// </summary>
    [MemoryDiagnoser]
    [RankColumn]
    public class ActivationFunctionsBenchmarks
    {
        private Tensor<double> _input;
        private double[] _accordInput;

        [Params(100, 1000, 10000)]
        public int Size { get; set; }

        [GlobalSetup]
        public void Setup()
        {
            _input = new Tensor<double>(new[] { Size });
            _accordInput = new double[Size];

            var random = new Random(42);
            for (int i = 0; i < Size; i++)
            {
                var value = (random.NextDouble() - 0.5) * 4; // Range: -2 to 2
                _input[i] = value;
                _accordInput[i] = value;
            }
        }

        // ReLU Benchmarks
        [Benchmark]
        public Tensor<double> AiDotNet_ReLU_Forward()
        {
            var relu = new ReLUActivation<double>();
            return relu.Forward(_input);
        }

        [Benchmark]
        public double[] Accord_ReLU_Forward()
        {
            var output = new double[Size];
            for (int i = 0; i < Size; i++)
            {
                output[i] = Math.Max(0, _accordInput[i]);
            }
            return output;
        }

        // Sigmoid Benchmarks
        [Benchmark]
        public Tensor<double> AiDotNet_Sigmoid_Forward()
        {
            var sigmoid = new SigmoidActivation<double>();
            return sigmoid.Forward(_input);
        }

        [Benchmark]
        public double[] Accord_Sigmoid_Forward()
        {
            var sigmoid = new SigmoidFunction();
            var output = new double[Size];
            for (int i = 0; i < Size; i++)
            {
                output[i] = sigmoid.Function(_accordInput[i]);
            }
            return output;
        }

        // Tanh Benchmarks
        [Benchmark]
        public Tensor<double> AiDotNet_Tanh_Forward()
        {
            var tanh = new TanhActivation<double>();
            return tanh.Forward(_input);
        }

        [Benchmark]
        public double[] Accord_Tanh_Forward()
        {
            var tanh = new BipolarSigmoidFunction();
            var output = new double[Size];
            for (int i = 0; i < Size; i++)
            {
                output[i] = tanh.Function(_accordInput[i]);
            }
            return output;
        }

        // LeakyReLU Benchmarks
        [Benchmark]
        public Tensor<double> AiDotNet_LeakyReLU_Forward()
        {
            var leakyRelu = new LeakyReLUActivation<double>(alpha: 0.01);
            return leakyRelu.Forward(_input);
        }

        [Benchmark]
        public double[] Manual_LeakyReLU_Forward()
        {
            var alpha = 0.01;
            var output = new double[Size];
            for (int i = 0; i < Size; i++)
            {
                output[i] = _accordInput[i] > 0 ? _accordInput[i] : alpha * _accordInput[i];
            }
            return output;
        }

        // ELU Benchmarks
        [Benchmark]
        public Tensor<double> AiDotNet_ELU_Forward()
        {
            var elu = new ELUActivation<double>(alpha: 1.0);
            return elu.Forward(_input);
        }

        [Benchmark]
        public double[] Manual_ELU_Forward()
        {
            var alpha = 1.0;
            var output = new double[Size];
            for (int i = 0; i < Size; i++)
            {
                output[i] = _accordInput[i] > 0 ? _accordInput[i] : alpha * (Math.Exp(_accordInput[i]) - 1);
            }
            return output;
        }

        // GELU Benchmarks
        [Benchmark]
        public Tensor<double> AiDotNet_GELU_Forward()
        {
            var gelu = new GELUActivation<double>();
            return gelu.Forward(_input);
        }

        [Benchmark]
        public double[] Manual_GELU_Forward()
        {
            var output = new double[Size];
            for (int i = 0; i < Size; i++)
            {
                // GELU(x) = x * Φ(x) where Φ is the CDF of standard normal distribution
                // Approximation: 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
                var x = _accordInput[i];
                var cube = x * x * x;
                output[i] = 0.5 * x * (1.0 + Math.Tanh(Math.Sqrt(2.0 / Math.PI) * (x + 0.044715 * cube)));
            }
            return output;
        }

        // Softmax Benchmarks
        [Benchmark]
        public Tensor<double> AiDotNet_Softmax_Forward()
        {
            var softmax = new SoftmaxActivation<double>();
            return softmax.Forward(_input);
        }

        [Benchmark]
        public double[] Manual_Softmax_Forward()
        {
            var output = new double[Size];
            var max = _accordInput.Max();
            var sum = 0.0;

            // Subtract max for numerical stability
            for (int i = 0; i < Size; i++)
            {
                output[i] = Math.Exp(_accordInput[i] - max);
                sum += output[i];
            }

            // Normalize
            for (int i = 0; i < Size; i++)
            {
                output[i] /= sum;
            }

            return output;
        }
    }

    /// <summary>
    /// Benchmarks for activation function backward passes (gradients).
    /// </summary>
    [MemoryDiagnoser]
    [RankColumn]
    public class ActivationFunctionGradientBenchmarks
    {
        private Tensor<double> _input;
        private Tensor<double> _outputGradient;
        private ReLUActivation<double> _relu;
        private SigmoidActivation<double> _sigmoid;
        private TanhActivation<double> _tanh;

        [Params(1000, 10000)]
        public int Size { get; set; }

        [GlobalSetup]
        public void Setup()
        {
            _input = new Tensor<double>(new[] { Size });
            _outputGradient = new Tensor<double>(new[] { Size });

            var random = new Random(42);
            for (int i = 0; i < Size; i++)
            {
                _input[i] = (random.NextDouble() - 0.5) * 4;
                _outputGradient[i] = random.NextDouble();
            }

            _relu = new ReLUActivation<double>();
            _sigmoid = new SigmoidActivation<double>();
            _tanh = new TanhActivation<double>();

            // Warm up with forward pass
            _relu.Forward(_input);
            _sigmoid.Forward(_input);
            _tanh.Forward(_input);
        }

        [Benchmark]
        public Tensor<double> AiDotNet_ReLU_Backward()
        {
            return _relu.Backward(_outputGradient);
        }

        [Benchmark]
        public Tensor<double> AiDotNet_Sigmoid_Backward()
        {
            return _sigmoid.Backward(_outputGradient);
        }

        [Benchmark]
        public Tensor<double> AiDotNet_Tanh_Backward()
        {
            return _tanh.Backward(_outputGradient);
        }
    }
}
