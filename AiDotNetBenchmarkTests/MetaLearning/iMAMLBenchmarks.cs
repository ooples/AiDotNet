using AiDotNet.Benchmarks.Helpers;
using AiDotNet.Data.Structures;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.MetaLearning.Algorithms;
using AiDotNet.MetaLearning.Data;
using AiDotNet.MetaLearning.Options;
using BenchmarkDotNet.Attributes;

namespace AiDotNet.Benchmarks.MetaLearning;

/// <summary>
/// Performance benchmarks for iMAML algorithm comparing different configurations.
/// </summary>
[MemoryDiagnoser]
[SimpleJob]
public class iMAMLBenchmarks
{
    private MockNeuralNetwork<double, Matrix<double>, Vector<double>>? _mockModel;
    private MockLossFunction<double> _mockLossFunction = null!;
    private TaskBatch<double, Matrix<double>, Vector<double>>? _taskBatch;

    [GlobalSetup]
    public void Setup()
    {
        // Setup mock model and loss function
        _mockModel = new MockNeuralNetwork<double, Matrix<double>, Vector<double>>(100, 10);
        _mockLossFunction = new MockLossFunction<double>();

        // Create test tasks using MetaLearningTask
        var tasks = new List<IMetaLearningTask<double, Matrix<double>, Vector<double>>>();
        for (int i = 0; i < 10; i++)
        {
            var supportInput = Matrix<double>.CreateRandom(5, 100, -1, 1);
            var supportOutput = Vector<double>.CreateRandom(5, -1, 1);
            var queryInput = Matrix<double>.CreateRandom(15, 100, -1, 1);
            var queryOutput = Vector<double>.CreateRandom(15, -1, 1);

            tasks.Add(new MetaLearningTask<double, Matrix<double>, Vector<double>>
            {
                SupportSetX = supportInput,
                SupportSetY = supportOutput,
                QuerySetX = queryInput,
                QuerySetY = queryOutput,
                NumWays = 5,
                NumShots = 1,
                NumQueryPerClass = 3,
                Name = $"task-{i}"
            });
        }
        _taskBatch = new TaskBatch<double, Matrix<double>, Vector<double>>(tasks.ToArray());
    }

    #region Regularization Strength Comparison

    [Benchmark(Baseline = true)]
    public double MetaTrain_Lambda1()
    {
        var options = new iMAMLOptions<double, Matrix<double>, Vector<double>>(_mockModel!)
        {
            LossFunction = _mockLossFunction!,
            InnerLearningRate = 0.01,
            OuterLearningRate = 0.001,
            AdaptationSteps = 5,
            LambdaRegularization = 1.0,
            ConjugateGradientIterations = 10
        };
        var algorithm = new iMAMLAlgorithm<double, Matrix<double>, Vector<double>>(options);
        return algorithm.MetaTrain(_taskBatch!);
    }

    [Benchmark]
    public double MetaTrain_Lambda05()
    {
        var options = new iMAMLOptions<double, Matrix<double>, Vector<double>>(_mockModel!)
        {
            LossFunction = _mockLossFunction!,
            InnerLearningRate = 0.01,
            OuterLearningRate = 0.001,
            AdaptationSteps = 5,
            LambdaRegularization = 0.5,
            ConjugateGradientIterations = 10
        };
        var algorithm = new iMAMLAlgorithm<double, Matrix<double>, Vector<double>>(options);
        return algorithm.MetaTrain(_taskBatch!);
    }

    [Benchmark]
    public double MetaTrain_Lambda2()
    {
        var options = new iMAMLOptions<double, Matrix<double>, Vector<double>>(_mockModel!)
        {
            LossFunction = _mockLossFunction!,
            InnerLearningRate = 0.01,
            OuterLearningRate = 0.001,
            AdaptationSteps = 5,
            LambdaRegularization = 2.0,
            ConjugateGradientIterations = 10
        };
        var algorithm = new iMAMLAlgorithm<double, Matrix<double>, Vector<double>>(options);
        return algorithm.MetaTrain(_taskBatch!);
    }

    #endregion

    #region CG Iterations Comparison

    [Benchmark]
    public double MetaTrain_CG5()
    {
        var options = new iMAMLOptions<double, Matrix<double>, Vector<double>>(_mockModel!)
        {
            LossFunction = _mockLossFunction!,
            InnerLearningRate = 0.01,
            OuterLearningRate = 0.001,
            AdaptationSteps = 5,
            LambdaRegularization = 1.0,
            ConjugateGradientIterations = 5
        };
        var algorithm = new iMAMLAlgorithm<double, Matrix<double>, Vector<double>>(options);
        return algorithm.MetaTrain(_taskBatch!);
    }

    [Benchmark]
    public double MetaTrain_CG20()
    {
        var options = new iMAMLOptions<double, Matrix<double>, Vector<double>>(_mockModel!)
        {
            LossFunction = _mockLossFunction!,
            InnerLearningRate = 0.01,
            OuterLearningRate = 0.001,
            AdaptationSteps = 5,
            LambdaRegularization = 1.0,
            ConjugateGradientIterations = 20
        };
        var algorithm = new iMAMLAlgorithm<double, Matrix<double>, Vector<double>>(options);
        return algorithm.MetaTrain(_taskBatch!);
    }

    #endregion

    #region Neumann Approximation Comparison

    [Benchmark]
    public double MetaTrain_Neumann()
    {
        var options = new iMAMLOptions<double, Matrix<double>, Vector<double>>(_mockModel!)
        {
            LossFunction = _mockLossFunction!,
            InnerLearningRate = 0.01,
            OuterLearningRate = 0.001,
            AdaptationSteps = 5,
            LambdaRegularization = 1.0,
            UseNeumannApproximation = true,
            NeumannSeriesTerms = 5
        };
        var algorithm = new iMAMLAlgorithm<double, Matrix<double>, Vector<double>>(options);
        return algorithm.MetaTrain(_taskBatch!);
    }

    #endregion

    #region Adaptation Steps Comparison

    [Benchmark]
    public double MetaTrain_Steps10()
    {
        var options = new iMAMLOptions<double, Matrix<double>, Vector<double>>(_mockModel!)
        {
            LossFunction = _mockLossFunction!,
            InnerLearningRate = 0.01,
            OuterLearningRate = 0.001,
            AdaptationSteps = 10,
            LambdaRegularization = 1.0,
            ConjugateGradientIterations = 10
        };
        var algorithm = new iMAMLAlgorithm<double, Matrix<double>, Vector<double>>(options);
        return algorithm.MetaTrain(_taskBatch!);
    }

    [Benchmark]
    public double MetaTrain_Steps20()
    {
        var options = new iMAMLOptions<double, Matrix<double>, Vector<double>>(_mockModel!)
        {
            LossFunction = _mockLossFunction!,
            InnerLearningRate = 0.01,
            OuterLearningRate = 0.001,
            AdaptationSteps = 20,
            LambdaRegularization = 1.0,
            ConjugateGradientIterations = 10
        };
        var algorithm = new iMAMLAlgorithm<double, Matrix<double>, Vector<double>>(options);
        return algorithm.MetaTrain(_taskBatch!);
    }

    #endregion
}

/// <summary>
/// Memory usage benchmarks for iMAML compared to MAML.
/// </summary>
[MemoryDiagnoser]
[SimpleJob]
public class iMAMLMemoryBenchmarks
{
    private MockNeuralNetwork<double, Matrix<double>, Vector<double>>? _mockModel;
    private MockLossFunction<double> _mockLossFunction = null!;
    private TaskBatch<double, Matrix<double>, Vector<double>>? _taskBatch;

    [GlobalSetup]
    public void Setup()
    {
        _mockModel = new MockNeuralNetwork<double, Matrix<double>, Vector<double>>(1000, 100);
        _mockLossFunction = new MockLossFunction<double>();

        // Create large tasks to test memory usage
        var tasks = new List<IMetaLearningTask<double, Matrix<double>, Vector<double>>>();
        for (int i = 0; i < 10; i++)
        {
            var supportInput = Matrix<double>.CreateRandom(50, 1000, -1, 1);
            var supportOutput = Vector<double>.CreateRandom(50, -1, 1);
            var queryInput = Matrix<double>.CreateRandom(150, 1000, -1, 1);
            var queryOutput = Vector<double>.CreateRandom(150, -1, 1);

            tasks.Add(new MetaLearningTask<double, Matrix<double>, Vector<double>>
            {
                SupportSetX = supportInput,
                SupportSetY = supportOutput,
                QuerySetX = queryInput,
                QuerySetY = queryOutput,
                NumWays = 10,
                NumShots = 5,
                NumQueryPerClass = 15,
                Name = $"large-task-{i}"
            });
        }
        _taskBatch = new TaskBatch<double, Matrix<double>, Vector<double>>(tasks.ToArray());
    }

    [Benchmark]
    public double MetaTrain_iMAML_50Steps()
    {
        var options = new iMAMLOptions<double, Matrix<double>, Vector<double>>(_mockModel!)
        {
            LossFunction = _mockLossFunction!,
            InnerLearningRate = 0.01,
            OuterLearningRate = 0.001,
            AdaptationSteps = 50,
            LambdaRegularization = 1.0,
            ConjugateGradientIterations = 20
        };
        var imaml = new iMAMLAlgorithm<double, Matrix<double>, Vector<double>>(options);
        return imaml.MetaTrain(_taskBatch!);
    }

    [Benchmark]
    public double MetaTrain_MAML_50Steps()
    {
        var options = new MAMLOptions<double, Matrix<double>, Vector<double>>(_mockModel!)
        {
            LossFunction = _mockLossFunction!,
            InnerLearningRate = 0.01,
            OuterLearningRate = 0.001,
            AdaptationSteps = 50
        };
        var maml = new MAMLAlgorithm<double, Matrix<double>, Vector<double>>(options);
        return maml.MetaTrain(_taskBatch!);
    }
}

/// <summary>
/// Convergence benchmarks for iMAML.
/// </summary>
[SimpleJob]
public class iMAMLConvergenceBenchmarks
{
    private MockNeuralNetwork<double, Matrix<double>, Vector<double>>? _mockModel;
    private MockLossFunction<double> _mockLossFunction = null!;
    private TaskBatch<double, Matrix<double>, Vector<double>>? _taskBatch;

    [GlobalSetup]
    public void Setup()
    {
        _mockModel = new MockNeuralNetwork<double, Matrix<double>, Vector<double>>(50, 10);
        _mockLossFunction = new MockLossFunction<double>();

        // Create tasks for convergence testing
        var tasks = new List<IMetaLearningTask<double, Matrix<double>, Vector<double>>>();
        for (int i = 0; i < 5; i++)
        {
            var supportInput = Matrix<double>.CreateRandom(5, 50, -1, 1);
            var supportOutput = Vector<double>.CreateRandom(5, -1, 1);
            var queryInput = Matrix<double>.CreateRandom(15, 50, -1, 1);
            var queryOutput = Vector<double>.CreateRandom(15, -1, 1);

            tasks.Add(new MetaLearningTask<double, Matrix<double>, Vector<double>>
            {
                SupportSetX = supportInput,
                SupportSetY = supportOutput,
                QuerySetX = queryInput,
                QuerySetY = queryOutput,
                NumWays = 5,
                NumShots = 1,
                NumQueryPerClass = 3,
                Name = $"convergence-task-{i}"
            });
        }
        _taskBatch = new TaskBatch<double, Matrix<double>, Vector<double>>(tasks.ToArray());
    }

    [Benchmark]
    public void Convergence_FullCG()
    {
        var options = new iMAMLOptions<double, Matrix<double>, Vector<double>>(_mockModel!)
        {
            LossFunction = _mockLossFunction!,
            InnerLearningRate = 0.01,
            OuterLearningRate = 0.001,
            AdaptationSteps = 5,
            LambdaRegularization = 1.0,
            UseNeumannApproximation = false,
            ConjugateGradientIterations = 10
        };

        var algorithm = new iMAMLAlgorithm<double, Matrix<double>, Vector<double>>(options);

        // Train for multiple epochs
        for (int epoch = 0; epoch < 10; epoch++)
        {
            algorithm.MetaTrain(_taskBatch!);
        }
    }

    [Benchmark]
    public void Convergence_NeumannApprox()
    {
        var options = new iMAMLOptions<double, Matrix<double>, Vector<double>>(_mockModel!)
        {
            LossFunction = _mockLossFunction!,
            InnerLearningRate = 0.01,
            OuterLearningRate = 0.001,
            AdaptationSteps = 5,
            LambdaRegularization = 1.0,
            UseNeumannApproximation = true,
            NeumannSeriesTerms = 5
        };

        var algorithm = new iMAMLAlgorithm<double, Matrix<double>, Vector<double>>(options);

        // Train for multiple epochs
        for (int epoch = 0; epoch < 10; epoch++)
        {
            algorithm.MetaTrain(_taskBatch!);
        }
    }

    [Benchmark]
    public void Convergence_FirstOrder()
    {
        var options = new iMAMLOptions<double, Matrix<double>, Vector<double>>(_mockModel!)
        {
            LossFunction = _mockLossFunction!,
            InnerLearningRate = 0.01,
            OuterLearningRate = 0.001,
            AdaptationSteps = 5,
            LambdaRegularization = 1.0,
            UseFirstOrder = true
        };

        var algorithm = new iMAMLAlgorithm<double, Matrix<double>, Vector<double>>(options);

        // Train for multiple epochs
        for (int epoch = 0; epoch < 10; epoch++)
        {
            algorithm.MetaTrain(_taskBatch!);
        }
    }
}
