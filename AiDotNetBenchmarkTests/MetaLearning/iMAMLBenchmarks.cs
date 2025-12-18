using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.MetaLearning.Algorithms;
using AiDotNet.MetaLearning.Data;
using AiDotNet.Models;
using AiDotNet.Models.Options;
using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Running;

namespace AiDotNet.Benchmarks.MetaLearning;

/// <summary>
/// Performance benchmarks for iMAML algorithm comparing different configurations.
/// </summary>
[MemoryDiagnoser]
[SimpleJob]
public class iMAMLBenchmarks
{
    private iMAMLAlgorithm<double, Matrix<double>, Vector<double>>? _imamlFiniteDifferences;
    private iMAMLAlgorithm<double, Matrix<double>, Vector<double>>? _imamlAutomaticDiff;
    private iMAMLAlgorithm<double, Matrix<double>, Vector<double>>? _imamlWithPreconditioning;
    private iMAMLAlgorithm<double, Matrix<double>, Vector<double>>? _imamlWithLineSearch;
    private MockNeuralNetwork<double, Matrix<double>, Vector<double>>? _mockModel;
    private MockLossFunction<double> _mockLossFunction = null!;
    private TaskBatch<double, Matrix<double>, Vector<double>>? _taskBatch;

    [GlobalSetup]
    public void Setup()
    {
        // Setup mock model and loss function
        _mockModel = new MockNeuralNetwork<double, Matrix<double>, Vector<double>>(100, 10);
        _mockLossFunction = new MockLossFunction<double>();

        // Create test tasks
        var tasks = new List<ITask<double, Matrix<double>, Vector<double>>>();
        for (int i = 0; i < 10; i++)
        {
            var supportInput = Matrix<double>.Random(5, 100, -1, 1);
            var supportOutput = Vector<double>.Random(5, -1, 1);
            var queryInput = Matrix<double>.Random(15, 100, -1, 1);
            var queryOutput = Vector<double>.Random(15, -1, 1);

            tasks.Add(new Task<double, Matrix<double>, Vector<double>>(
                supportInput, supportOutput, queryInput, queryOutput,
                numWays: 5, numShots: 1, numQueryPerClass: 3,
                taskId: $"task-{i}"));
        }
        _taskBatch = new TaskBatch<double, Matrix<double>, Vector<double>>(tasks);

        // Setup iMAML with finite differences
        var options = new iMAMLAlgorithmOptions<double, Matrix<double>, Vector<double>>
        {
            BaseModel = _mockModel,
            LossFunction = _mockLossFunction,
            InnerLearningRate = 0.01,
            OuterLearningRate = 0.001,
            AdaptationSteps = 5,
            LambdaRegularization = 1.0,
            ConjugateGradientIterations = 10,
            ConjugateGradientTolerance = 1e-8,
            HessianVectorProductMethod = HessianVectorProductMethod.FiniteDifferences,
            CGPreconditioningMethod = CGPreconditioningMethod.None,
            FiniteDifferencesEpsilon = 1e-5,
            UseAdaptiveInnerLearningRate = false,
            EnableLineSearch = false
        };

        _imamlFiniteDifferences = new iMAMLAlgorithm<double, Matrix<double>, Vector<double>>(options);

        // Setup iMAML with automatic differentiation
        options.HessianVectorProductMethod = HessianVectorProductMethod.AutomaticDifferentiation;
        _imamlAutomaticDiff = new iMAMLAlgorithm<double, Matrix<double>, Vector<double>>(options);

        // Setup iMAML with Jacobi preconditioning
        options.HessianVectorProductMethod = HessianVectorProductMethod.FiniteDifferences;
        options.CGPreconditioningMethod = CGPreconditioningMethod.Jacobi;
        _imamlWithPreconditioning = new iMAMLAlgorithm<double, Matrix<double>, Vector<double>>(options);

        // Setup iMAML with line search
        options.CGPreconditioningMethod = CGPreconditioningMethod.None;
        options.EnableLineSearch = true;
        options.UseAdaptiveInnerLearningRate = true;
        _imamlWithLineSearch = new iMAMLAlgorithm<double, Matrix<double>, Vector<double>>(options);
    }

    [Benchmark]
    [Arguments(1)]
    [Arguments(5)]
    [Arguments(10)]
    public double MetaTrain_FiniteDifferences(int adaptationSteps)
    {
        var options = new iMAMLAlgorithmOptions<double, Matrix<double>, Vector<double>>
        {
            BaseModel = _mockModel!,
            LossFunction = _mockLossFunction!,
            InnerLearningRate = 0.01,
            OuterLearningRate = 0.001,
            AdaptationSteps = adaptationSteps,
            LambdaRegularization = 1.0,
            ConjugateGradientIterations = 10,
            HessianVectorProductMethod = HessianVectorProductMethod.FiniteDifferences
        };

        var algorithm = new iMAMLAlgorithm<double, Matrix<double>, Vector<double>>(options);
        return algorithm.MetaTrain(_taskBatch!);
    }

    [Benchmark]
    [Arguments(1)]
    [Arguments(5)]
    [Arguments(10)]
    public double MetaTrain_AutomaticDifferentiation(int adaptationSteps)
    {
        var options = new iMAMLAlgorithmOptions<double, Matrix<double>, Vector<double>>
        {
            BaseModel = _mockModel!,
            LossFunction = _mockLossFunction!,
            InnerLearningRate = 0.01,
            OuterLearningRate = 0.001,
            AdaptationSteps = adaptationSteps,
            LambdaRegularization = 1.0,
            ConjugateGradientIterations = 10,
            HessianVectorProductMethod = HessianVectorProductMethod.AutomaticDifferentiation
        };

        var algorithm = new iMAMLAlgorithm<double, Matrix<double>, Vector<double>>(options);
        return algorithm.MetaTrain(_taskBatch!);
    }

    [Benchmark]
    [Arguments(5)]
    [Arguments(10)]
    [Arguments(20)]
    public double MetaTrain_CGIterations(int cgIterations)
    {
        var options = new iMAMLAlgorithmOptions<double, Matrix<double>, Vector<double>>
        {
            BaseModel = _mockModel!,
            LossFunction = _mockLossFunction!,
            InnerLearningRate = 0.01,
            OuterLearningRate = 0.001,
            AdaptationSteps = 5,
            LambdaRegularization = 1.0,
            ConjugateGradientIterations = cgIterations,
            HessianVectorProductMethod = HessianVectorProductMethod.FiniteDifferences,
            CGPreconditioningMethod = CGPreconditioningMethod.None
        };

        var algorithm = new iMAMLAlgorithm<double, Matrix<double>, Vector<double>>(options);
        return algorithm.MetaTrain(_taskBatch!);
    }

    [Benchmark]
    public void Adapt_SingleTask()
    {
        var task = _taskBatch!.Tasks.First();
        _imamlFiniteDifferences!.Adapt(task);
    }

    [Benchmark]
    public void Adapt_WithAdaptiveLearningRate()
    {
        var task = _taskBatch!.Tasks.First();
        _imamlWithLineSearch!.Adapt(task);
    }

    [Benchmark]
    public void Adapt_WithLineSearch()
    {
        var task = _taskBatch!.Tasks.First();
        _imamlWithLineSearch!.Adapt(task);
    }

    [Benchmark]
    public void MetaTrain_WithPreconditioning()
    {
        _imamlWithPreconditioning!.MetaTrain(_taskBatch!);
    }

    [Benchmark]
    public void MetaTrain_BatchSize1()
    {
        var singleTaskBatch = new TaskBatch<double, Matrix<double>, Vector<double>>(
            new[] { _taskBatch!.Tasks.First() });
        _imamlFiniteDifferences!.MetaTrain(singleTaskBatch);
    }

    [Benchmark]
    public void MetaTrain_BatchSize10()
    {
        _imamlFiniteDifferences!.MetaTrain(_taskBatch!);
    }
}

/// <summary>
/// Memory usage benchmarks for iMAML compared to MAML.
/// </summary>
[MemoryDiagnoser]
[SimpleJob]
public class iMAMLMemoryBenchmarks
{
    private iMAMLAlgorithm<double, Matrix<double>, Vector<double>>? _imaml;
    private MAMLAlgorithm<double, Matrix<double>, Vector<double>>? _maml;
    private MockNeuralNetwork<double, Matrix<double>, Vector<double>>? _mockModel;
    private MockLossFunction<double> _mockLossFunction = null!;
    private TaskBatch<double, Matrix<double>, Vector<double>>? _taskBatch;

    [GlobalSetup]
    public void Setup()
    {
        _mockModel = new MockNeuralNetwork<double, Matrix<double>, Vector<double>>(1000, 100);
        _mockLossFunction = new MockLossFunction<double>();

        // Create large tasks to test memory usage
        var tasks = new List<ITask<double, Matrix<double>, Vector<double>>>();
        for (int i = 0; i < 10; i++)
        {
            var supportInput = Matrix<double>.Random(50, 1000, -1, 1);
            var supportOutput = Vector<double>.Random(50, -1, 1);
            var queryInput = Matrix<double>.Random(150, 1000, -1, 1);
            var queryOutput = Vector<double>.Random(150, -1, 1);

            tasks.Add(new Task<double, Matrix<double>, Vector<double>>(
                supportInput, supportOutput, queryInput, queryOutput,
                numWays: 10, numShots: 5, numQueryPerClass: 15,
                taskId: $"large-task-{i}"));
        }
        _taskBatch = new TaskBatch<double, Matrix<double>, Vector<double>>(tasks);

        // Setup iMAML with high adaptation steps
        var iMamlOptions = new iMAMLAlgorithmOptions<double, Matrix<double>, Vector<double>>
        {
            BaseModel = _mockModel,
            LossFunction = _mockLossFunction,
            InnerLearningRate = 0.01,
            OuterLearningRate = 0.001,
            AdaptationSteps = 50,
            LambdaRegularization = 1.0,
            ConjugateGradientIterations = 20,
            HessianVectorProductMethod = HessianVectorProductMethod.FiniteDifferences
        };
        _imaml = new iMAMLAlgorithm<double, Matrix<double>, Vector<double>>(iMamlOptions);

        // Setup MAML with same adaptation steps
        var mamlOptions = new MAMLAlgorithmOptions<double, Matrix<double>, Vector<double>>
        {
            BaseModel = _mockModel,
            LossFunction = _mockLossFunction,
            InnerLearningRate = 0.01,
            OuterLearningRate = 0.001,
            AdaptationSteps = 50
        };
        _maml = new MAMLAlgorithm<double, Matrix<double>, Vector<double>>(mamlOptions);
    }

    [Benchmark]
    public double MetaTrain_iMAML_50Steps()
    {
        return _imaml!.MetaTrain(_taskBatch!);
    }

    [Benchmark]
    public double MetaTrain_MAML_50Steps()
    {
        return _maml!.MetaTrain(_taskBatch!);
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

    [GlobalSetup]
    public void Setup()
    {
        _mockModel = new MockNeuralNetwork<double, Matrix<double>, Vector<double>>(50, 10);
        _mockLossFunction = new MockLossFunction<double>();
    }

    [Benchmark]
    public void Convergence_WithAdaptiveLearningRate()
    {
        var options = new iMAMLAlgorithmOptions<double, Matrix<double>, Vector<double>>
        {
            BaseModel = _mockModel!,
            LossFunction = _mockLossFunction!,
            InnerLearningRate = 0.01,
            OuterLearningRate = 0.001,
            AdaptationSteps = 5,
            UseAdaptiveInnerLearningRate = true,
            LambdaRegularization = 0.5
        };

        var algorithm = new iMAMLAlgorithm<double, Matrix<double>, Vector<double>>(options);

        // Train for multiple epochs
        for (int epoch = 0; epoch < 10; epoch++)
        {
            var tasks = new List<ITask<double, Matrix<double>, Vector<double>>>();
            for (int i = 0; i < 5; i++)
            {
                var supportInput = Matrix<double>.Random(5, 50, -1, 1);
                var supportOutput = Vector<double>.Random(5, -1, 1);
                var queryInput = Matrix<double>.Random(15, 50, -1, 1);
                var queryOutput = Vector<double>.Random(15, -1, 1);

                tasks.Add(new Task<double, Matrix<double>, Vector<double>>(
                    supportInput, supportOutput, queryInput, queryOutput,
                    numWays: 5, numShots: 1, numQueryPerClass: 3,
                    taskId: $"task-{i}"));
            }
            var batch = new TaskBatch<double, Matrix<double>, Vector<double>>(tasks);
            algorithm.MetaTrain(batch);
        }
    }

    [Benchmark]
    public void Convergence_WithFixedLearningRate()
    {
        var options = new iMAMLAlgorithmOptions<double, Matrix<double>, Vector<double>>
        {
            BaseModel = _mockModel!,
            LossFunction = _mockLossFunction!,
            InnerLearningRate = 0.01,
            OuterLearningRate = 0.001,
            AdaptationSteps = 5,
            UseAdaptiveInnerLearningRate = false,
            LambdaRegularization = 0.5
        };

        var algorithm = new iMAMLAlgorithm<double, Matrix<double>, Vector<double>>(options);

        // Train for multiple epochs
        for (int epoch = 0; epoch < 10; epoch++)
        {
            var tasks = new List<ITask<double, Matrix<double>, Vector<double>>>();
            for (int i = 0; i < 5; i++)
            {
                var supportInput = Matrix<double>.Random(5, 50, -1, 1);
                var supportOutput = Vector<double>.Random(5, -1, 1);
                var queryInput = Matrix<double>.Random(15, 50, -1, 1);
                var queryOutput = Vector<double>.Random(15, -1, 1);

                tasks.Add(new Task<double, Matrix<double>, Vector<double>>(
                    supportInput, supportOutput, queryInput, queryOutput,
                    numWays: 5, numShots: 1, numQueryPerClass: 3,
                    taskId: $"task-{i}"));
            }
            var batch = new TaskBatch<double, Matrix<double>, Vector<double>>(tasks);
            algorithm.MetaTrain(batch);
        }
    }
}

/// <summary>
/// Entry point for running iMAML benchmarks.
/// </summary>
public class iMAMLBenchmarkRunner
{
    public static void Main(string[] args)
    {
        var summary = BenchmarkRunner.Run(
            typeof(iMAMLBenchmarks),
            typeof(iMAMLMemoryBenchmarks),
            typeof(iMAMLConvergenceBenchmarks));
    }
}