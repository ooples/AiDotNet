using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using AiDotNet.Data.Structures;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.MetaLearning;
using AiDotNet.MetaLearning.Data;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.MetaLearning;

public class MetaLearnerBaseIntegrationTests
{
    private static MetaLearningTask<double, Matrix<double>, Vector<double>> CreateVectorTask(int seed)
    {
        const int numWays = 2;
        const int numShots = 2;
        const int numQueryPerClass = 2;
        const int featureCount = 3;

        int supportRows = numWays * numShots;
        int queryRows = numWays * numQueryPerClass;

        var supportX = new Matrix<double>(supportRows, featureCount);
        var supportY = new Vector<double>(supportRows);
        var queryX = new Matrix<double>(queryRows, featureCount);
        var queryY = new Vector<double>(queryRows);

        var random = new Random(seed);

        for (int i = 0; i < supportRows; i++)
        {
            for (int j = 0; j < featureCount; j++)
            {
                supportX[i, j] = random.NextDouble() - 0.5;
            }
            supportY[i] = i % numWays;
        }

        for (int i = 0; i < queryRows; i++)
        {
            for (int j = 0; j < featureCount; j++)
            {
                queryX[i, j] = random.NextDouble() - 0.5;
            }
            queryY[i] = i % numWays;
        }

        return new MetaLearningTask<double, Matrix<double>, Vector<double>>
        {
            SupportSetX = supportX,
            SupportSetY = supportY,
            QuerySetX = queryX,
            QuerySetY = queryY,
            NumWays = numWays,
            NumShots = numShots,
            NumQueryPerClass = numQueryPerClass,
            Name = $"task-{seed}"
        };
    }

    private static TaskBatch<double, Matrix<double>, Vector<double>> CreateTaskBatch(int count)
    {
        var tasks = Enumerable.Range(0, count)
            .Select(i => CreateVectorTask(100 + i))
            .Cast<IMetaLearningTask<double, Matrix<double>, Vector<double>>>()
            .ToArray();

        return new TaskBatch<double, Matrix<double>, Vector<double>>(tasks);
    }

    [Fact]
    public void Constructor_NullArguments_Throws()
    {
        var options = new MetaLearnerOptionsBase<double>();

        Assert.Throws<ArgumentNullException>(() => new TestMetaLearner(null!, options));
        Assert.Throws<ArgumentNullException>(() => new TestMetaLearner(new LinearVectorModel(3), null!));
    }

    [Fact]
    public void Constructor_InvalidOptions_Throws()
    {
        var options = new MetaLearnerOptionsBase<double> { InnerLearningRate = 0 };

        Assert.Throws<ArgumentException>(() => new TestMetaLearner(new LinearVectorModel(3), options));
    }

    [Fact]
    public void Evaluate_NullBatch_Throws()
    {
        var learner = new TestMetaLearner(new LinearVectorModel(3), new MetaLearnerOptionsBase<double>());

        Assert.Throws<ArgumentNullException>(() => learner.Evaluate(null!));
    }

    [Fact]
    public void Evaluate_ComputesAverageLoss()
    {
        var learner = new TestMetaLearner(new LinearVectorModel(3), new MetaLearnerOptionsBase<double>());
        var batch = CreateTaskBatch(2);

        var expectedLosses = new List<double>();
        foreach (var task in batch.Tasks)
        {
            var adapted = learner.Adapt(task);
            var predictions = adapted.Predict(task.QueryInput);
            expectedLosses.Add(learner.CallComputeLossFromOutput(predictions, task.QueryOutput));
        }

        var expected = expectedLosses.Average();
        var actual = learner.Evaluate(batch);

        Assert.Equal(expected, actual, precision: 6);
    }

    [Fact]
    public void MetaTrainStep_InvalidBatchSize_Throws()
    {
        var learner = new TestMetaLearner(new LinearVectorModel(3), new MetaLearnerOptionsBase<double>());

        Assert.Throws<ArgumentOutOfRangeException>(() => learner.MetaTrainStep(0));
    }

    [Fact]
    public void MetaTrainStep_WithoutDataLoader_Throws()
    {
        var learner = new TestMetaLearner(new LinearVectorModel(3), new MetaLearnerOptionsBase<double>());

        Assert.Throws<InvalidOperationException>(() => learner.MetaTrainStep(1));
    }

    [Fact]
    public void MetaTrainStep_WithDataLoader_IncrementsIteration()
    {
        var tasks = new[] { CreateVectorTask(1) };
        var dataLoader = new TestEpisodicDataLoader<double, Matrix<double>, Vector<double>>(tasks, 2, 2, 2, 2);
        var learner = new TestMetaLearner(new LinearVectorModel(3), new MetaLearnerOptionsBase<double>(), dataLoader);

        var result = learner.MetaTrainStep(1);

        Assert.Equal(1, result.Iteration);
        Assert.Equal(1, result.NumTasks);
        Assert.Equal(1, learner.CurrentIteration);
    }

    [Fact]
    public void Train_WithoutDataLoader_Throws()
    {
        var learner = new TestMetaLearner(new LinearVectorModel(3), new MetaLearnerOptionsBase<double>());

        Assert.Throws<InvalidOperationException>(() => learner.Train());
    }

    [Fact]
    public void Train_RunsConfiguredIterations()
    {
        var tasks = new[] { CreateVectorTask(2) };
        var dataLoader = new TestEpisodicDataLoader<double, Matrix<double>, Vector<double>>(tasks, 2, 2, 2, 2);
        var options = new MetaLearnerOptionsBase<double>
        {
            NumMetaIterations = 3,
            MetaBatchSize = 1,
            EnableCheckpointing = false
        };

        var learner = new TestMetaLearner(new LinearVectorModel(3), options, dataLoader);
        var result = learner.Train();

        Assert.Equal(3, result.TotalIterations);
        Assert.Equal(3, result.LossHistory.Length);
        Assert.Equal(3, result.AccuracyHistory.Length);
        Assert.Equal(3, learner.CurrentIteration);
    }

    [Fact]
    public void Evaluate_WithNumTasks_UsesDataLoader()
    {
        var tasks = new[] { CreateVectorTask(3), CreateVectorTask(4) };
        var dataLoader = new TestEpisodicDataLoader<double, Matrix<double>, Vector<double>>(tasks, 2, 2, 2, 2);
        var learner = new TestMetaLearner(new LinearVectorModel(3), new MetaLearnerOptionsBase<double>(), dataLoader);

        var result = learner.Evaluate(3);

        Assert.Equal(3, result.NumTasks);
        Assert.Equal(3, result.PerTaskAccuracies.Length);
        Assert.Equal(3, result.PerTaskLosses.Length);
    }

    [Fact]
    public void AdaptAndEvaluate_ReturnsMetrics()
    {
        var learner = new TestMetaLearner(new LinearVectorModel(3), new MetaLearnerOptionsBase<double>());
        var task = CreateVectorTask(5);

        var result = learner.AdaptAndEvaluate(task);

        Assert.Equal(learner.AdaptationSteps, result.AdaptationSteps);
        Assert.True(result.QueryLoss >= 0);
        Assert.True(result.SupportLoss >= 0);
        Assert.InRange(result.QueryAccuracy, 0.0, 1.0);
        Assert.InRange(result.SupportAccuracy, 0.0, 1.0);
    }

    [Fact]
    public void SetMetaModel_ResetsOptimizers()
    {
        var metaOptimizer = new ResetTrackingOptimizer();
        var innerOptimizer = new ResetTrackingOptimizer();
        var learner = new TestMetaLearner(
            new LinearVectorModel(3),
            new MetaLearnerOptionsBase<double>(),
            metaOptimizer: metaOptimizer,
            innerOptimizer: innerOptimizer);

        learner.SetMetaModel(new LinearVectorModel(3));

        Assert.Equal(1, metaOptimizer.ResetCount);
        Assert.Equal(1, innerOptimizer.ResetCount);
    }

    [Fact]
    public void Reset_ClearsIteration()
    {
        var tasks = new[] { CreateVectorTask(6) };
        var dataLoader = new TestEpisodicDataLoader<double, Matrix<double>, Vector<double>>(tasks, 2, 2, 2, 2);
        var learner = new TestMetaLearner(new LinearVectorModel(3), new MetaLearnerOptionsBase<double>(), dataLoader);

        learner.MetaTrainStep(1);
        Assert.Equal(1, learner.CurrentIteration);

        learner.Reset();
        Assert.Equal(0, learner.CurrentIteration);
    }

    [Fact]
    public void ComputeAccuracy_HandlesOneHotAndClassIndexCases()
    {
        var learner = new TestMetaLearner(new LinearVectorModel(3), new MetaLearnerOptionsBase<double>());

        var oneHotPred = new Vector<double>(new[] { 0.1, 0.9, 0.0 });
        var oneHotLabel = new Vector<double>(new[] { 0.0, 1.0, 0.0 });
        Assert.Equal(1.0, learner.CallComputeAccuracy(oneHotPred, oneHotLabel), precision: 6);

        var classPred = new Vector<double>(new[] { 2.0, 1.0, 2.0 });
        var classLabel = new Vector<double>(new[] { 2.0, 0.0, 2.0 });
        Assert.Equal(2.0 / 3.0, learner.CallComputeAccuracy(classPred, classLabel), precision: 6);

        var mismatchPred = new Vector<double>(new[] { 0.0, 1.0 });
        var mismatchLabel = new Vector<double>(new[] { 0.0, 1.0, 2.0 });
        Assert.Equal(0.0, learner.CallComputeAccuracy(mismatchPred, mismatchLabel), precision: 6);
    }

    [Fact]
    public void ComputeLossFromOutput_ReturnsExpectedValue()
    {
        var learner = new TestMetaLearner(new LinearVectorModel(3), new MetaLearnerOptionsBase<double>());

        var predictions = new Vector<double>(new[] { 1.0, 2.0 });
        var expected = new Vector<double>(new[] { 1.0, 3.0 });
        var loss = learner.CallComputeLossFromOutput(predictions, expected);

        Assert.Equal(0.5, loss, precision: 6);
    }

    [Fact]
    public void ConvertToVector_ReturnsSameVector()
    {
        var learner = new TestMetaLearner(new LinearVectorModel(3), new MetaLearnerOptionsBase<double>());
        var vector = new Vector<double>(new[] { 1.0, 2.0 });

        var converted = learner.CallConvertToVector(vector);

        Assert.NotNull(converted);
        Assert.Equal(vector.Length, converted!.Length);
        Assert.Equal(vector[0], converted[0]);
        Assert.Equal(vector[1], converted[1]);
    }

    [Fact]
    public void ComputeMean_HandlesEmptyAndNonEmpty()
    {
        var learner = new TestMetaLearner(new LinearVectorModel(3), new MetaLearnerOptionsBase<double>());

        Assert.Equal(0.0, learner.CallComputeMean(new List<double>()), precision: 6);
        Assert.Equal(2.0, learner.CallComputeMean(new List<double> { 1.0, 2.0, 3.0 }), precision: 6);
    }

    [Fact]
    public void ApplyGradients_UpdatesParameters()
    {
        var learner = new TestMetaLearner(new LinearVectorModel(2), new MetaLearnerOptionsBase<double>());
        var parameters = new Vector<double>(new[] { 1.0, 2.0 });
        var gradients = new Vector<double>(new[] { 0.5, 1.0 });

        var updated = learner.CallApplyGradients(parameters, gradients, 0.1);

        Assert.Equal(0.95, updated[0], precision: 6);
        Assert.Equal(1.9, updated[1], precision: 6);
    }

    [Fact]
    public void ClipGradients_RespectsThreshold()
    {
        var learner = new TestMetaLearner(new LinearVectorModel(2), new MetaLearnerOptionsBase<double>());
        var gradients = new Vector<double>(new[] { 3.0, 4.0 });

        var clipped = learner.CallClipGradients(gradients, 2.5);

        Assert.Equal(1.5, clipped[0], precision: 6);
        Assert.Equal(2.0, clipped[1], precision: 6);
    }

    [Fact]
    public void CreateTaskBatch_WrapsTasks()
    {
        var learner = new TestMetaLearner(new LinearVectorModel(3), new MetaLearnerOptionsBase<double>());
        var tasks = new List<MetaLearningTask<double, Matrix<double>, Vector<double>>>
        {
            CreateVectorTask(7),
            CreateVectorTask(8)
        };

        var batch = learner.CallCreateTaskBatch(tasks);

        Assert.Equal(2, batch.BatchSize);
        Assert.Equal(tasks[0].NumWays, batch.NumWays);
        Assert.Equal(tasks[0].NumShots, batch.NumShots);
        Assert.Equal(tasks[0].NumQueryPerClass, batch.NumQueryPerClass);
    }

    [Fact]
    public void ToMetaLearningTask_PreservesData()
    {
        var learner = new TestMetaLearner(new LinearVectorModel(3), new MetaLearnerOptionsBase<double>());
        var task = CreateVectorTask(9);

        var wrapped = learner.CallToMetaLearningTask(task);

        Assert.Equal(task.SupportSetX, wrapped.SupportInput);
        Assert.Equal(task.SupportSetY, wrapped.SupportOutput);
        Assert.Equal(task.QuerySetX, wrapped.QueryInput);
        Assert.Equal(task.QuerySetY, wrapped.QueryOutput);
    }

    [Fact]
    public void CloneModel_ReturnsIndependentCopy()
    {
        var learner = new TestMetaLearner(new LinearVectorModel(3), new MetaLearnerOptionsBase<double>());

        var cloned = learner.CallCloneModel();

        Assert.NotSame(learner.BaseModel, cloned);
        var original = learner.BaseModel.GetParameters();
        var clonedParams = cloned.GetParameters();
        Assert.Equal(original.Length, clonedParams.Length);
        for (int i = 0; i < original.Length; i++)
        {
            Assert.Equal(original[i], clonedParams[i], precision: 6);
        }
    }

    [Fact]
    public void ComputeSecondOrderGradients_UsesSecondOrderModel()
    {
        var learner = new TestMetaLearner(new LinearVectorModel(3), new MetaLearnerOptionsBase<double>());
        var model = new SecondOrderMatrixModel(3);

        var steps = new List<(Matrix<double>, Vector<double>)>
        {
            (CreateVectorTask(10).SupportSetX, CreateVectorTask(10).SupportSetY)
        };

        var gradients = learner.CallComputeSecondOrderGradients(
            model,
            steps,
            CreateVectorTask(11).QuerySetX,
            CreateVectorTask(11).QuerySetY,
            0.1);

        Assert.Equal(model.ParameterCount, gradients.Length);
        Assert.Equal(0.05, gradients[0], precision: 6);
    }

    [Fact]
    public void ComputeSecondOrderGradients_FallsBackToFirstOrder()
    {
        var learner = new TestMetaLearner(new LinearVectorModel(3), new MetaLearnerOptionsBase<double>());
        var model = new LinearVectorModel(3);
        var task = CreateVectorTask(12);

        var steps = new List<(Matrix<double>, Vector<double>)>
        {
            (task.SupportSetX, task.SupportSetY)
        };

        var gradients = learner.CallComputeSecondOrderGradients(
            model,
            steps,
            task.QuerySetX,
            task.QuerySetY,
            0.1);

        Assert.Equal(model.ParameterCount, gradients.Length);
        Assert.True(gradients.Any(value => Math.Abs(value) > 1e-12));
    }

    [Fact]
    public void SaveAndLoad_RestoresIteration()
    {
        var task = CreateVectorTask(13);
        var dataLoader = new TestEpisodicDataLoader<double, Matrix<double>, Vector<double>>(new[] { task }, 2, 2, 2, 2);
        var learner = new TestMetaLearner(new LinearVectorModel(3), new MetaLearnerOptionsBase<double>(), dataLoader);

        learner.MetaTrainStep(1);
        Assert.Equal(1, learner.CurrentIteration);

        var tempPath = Path.Combine(Path.GetTempPath(), $"meta_{Guid.NewGuid():N}.bin");
        try
        {
            learner.Save(tempPath);
            learner.Reset();
            Assert.Equal(0, learner.CurrentIteration);

            learner.Load(tempPath);
            Assert.Equal(1, learner.CurrentIteration);
        }
        finally
        {
            if (File.Exists(tempPath))
            {
                File.Delete(tempPath);
            }
            if (File.Exists(tempPath + ".meta"))
            {
                File.Delete(tempPath + ".meta");
            }
        }
    }
}
