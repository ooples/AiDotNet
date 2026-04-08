using System;
using AiDotNet.Data.Structures;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.MetaLearning;
using AiDotNet.MetaLearning.Algorithms;
using AiDotNet.MetaLearning.Data;
using AiDotNet.MetaLearning.Options;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.MetaLearning;

/// <summary>
/// Integration tests for Foundation Model Era meta-learning algorithms (Phase 2).
/// Verifies: finite loss, parameter change, adaptation, and multi-step stability.
/// </summary>
public class FoundationModelMetaTests
{
    private static MetaLearningTask<double, Matrix<double>, Vector<double>> CreateTask(int seed)
    {
        var supportX = new Matrix<double>(4, 3);
        var supportY = new Vector<double>(4);
        var queryX = new Matrix<double>(4, 3);
        var queryY = new Vector<double>(4);
        var rng = new Random(seed);
        for (int i = 0; i < 4; i++)
        {
            for (int j = 0; j < 3; j++)
            {
                supportX[i, j] = rng.NextDouble() - 0.5;
                queryX[i, j] = rng.NextDouble() - 0.5;
            }
            supportY[i] = i % 2;
            queryY[i] = i % 2;
        }
        return new MetaLearningTask<double, Matrix<double>, Vector<double>>
        {
            SupportSetX = supportX, SupportSetY = supportY,
            QuerySetX = queryX, QuerySetY = queryY,
            NumWays = 2, NumShots = 2, NumQueryPerClass = 2
        };
    }

    private static bool ParamsChanged(Vector<double> before, Vector<double> after)
    {
        if (before.Length != after.Length) return true;
        for (int i = 0; i < before.Length; i++)
            if (Math.Abs(before[i] - after[i]) > 1e-15) return true;
        return false;
    }

    [Fact]
    public void MetaLoRA_FiniteLossAndParamChange()
    {
        var model = new LinearVectorModel(3);
        var options = new MetaLoRAOptions<double, Matrix<double>, Vector<double>>(model)
        { InnerLearningRate = 0.01, OuterLearningRate = 0.01 };
        var algorithm = new MetaLoRAAlgorithm<double, Matrix<double>, Vector<double>>(options);
        var paramsBefore = model.GetParameters();

        var task = CreateTask(1876);
        var batch = new TaskBatch<double, Matrix<double>, Vector<double>>(new[] { task });

        var loss = algorithm.MetaTrain(batch);
        Assert.False(double.IsNaN(loss), "MetaLoRA loss is NaN");
        Assert.False(double.IsInfinity(loss), "MetaLoRA loss is infinite");
        Assert.True(ParamsChanged(paramsBefore, model.GetParameters()), "MetaLoRA params unchanged");
        Assert.Equal(MetaLearningAlgorithmType.MetaLoRA, algorithm.AlgorithmType);
        Assert.NotNull(algorithm.Adapt(task).Predict(task.QuerySetX));
    }

    [Fact]
    public void LoRARecycle_FiniteLossAndParamChange()
    {
        var model = new LinearVectorModel(3);
        var options = new LoRARecycleOptions<double, Matrix<double>, Vector<double>>(model)
        { InnerLearningRate = 0.01, OuterLearningRate = 0.01 };
        var algorithm = new LoRARecycleAlgorithm<double, Matrix<double>, Vector<double>>(options);
        var paramsBefore = model.GetParameters();

        var task = CreateTask(6177);
        var batch = new TaskBatch<double, Matrix<double>, Vector<double>>(new[] { task });

        var loss = algorithm.MetaTrain(batch);
        Assert.False(double.IsNaN(loss), "LoRARecycle loss is NaN");
        Assert.False(double.IsInfinity(loss), "LoRARecycle loss is infinite");
        Assert.True(ParamsChanged(paramsBefore, model.GetParameters()), "LoRARecycle params unchanged");
        Assert.Equal(MetaLearningAlgorithmType.LoRARecycle, algorithm.AlgorithmType);
        Assert.NotNull(algorithm.Adapt(task).Predict(task.QuerySetX));
    }

    [Fact]
    public void ICMFusion_FiniteLossAndParamChange()
    {
        var model = new LinearVectorModel(3);
        var options = new ICMFusionOptions<double, Matrix<double>, Vector<double>>(model)
        { InnerLearningRate = 0.01, OuterLearningRate = 0.01 };
        var algorithm = new ICMFusionAlgorithm<double, Matrix<double>, Vector<double>>(options);
        var paramsBefore = model.GetParameters();

        var task = CreateTask(6625);
        var batch = new TaskBatch<double, Matrix<double>, Vector<double>>(new[] { task });

        var loss = algorithm.MetaTrain(batch);
        Assert.False(double.IsNaN(loss), "ICMFusion loss is NaN");
        Assert.False(double.IsInfinity(loss), "ICMFusion loss is infinite");
        Assert.True(ParamsChanged(paramsBefore, model.GetParameters()), "ICMFusion params unchanged");
        Assert.Equal(MetaLearningAlgorithmType.ICMFusion, algorithm.AlgorithmType);
        Assert.NotNull(algorithm.Adapt(task).Predict(task.QuerySetX));
    }

    [Fact]
    public void MetaLoRABank_FiniteLossAndParamChange()
    {
        var model = new LinearVectorModel(3);
        var options = new MetaLoRABankOptions<double, Matrix<double>, Vector<double>>(model)
        { InnerLearningRate = 0.01, OuterLearningRate = 0.01 };
        var algorithm = new MetaLoRABankAlgorithm<double, Matrix<double>, Vector<double>>(options);
        var paramsBefore = model.GetParameters();

        var task = CreateTask(1037);
        var batch = new TaskBatch<double, Matrix<double>, Vector<double>>(new[] { task });

        var loss = algorithm.MetaTrain(batch);
        Assert.False(double.IsNaN(loss), "MetaLoRABank loss is NaN");
        Assert.False(double.IsInfinity(loss), "MetaLoRABank loss is infinite");
        Assert.True(ParamsChanged(paramsBefore, model.GetParameters()), "MetaLoRABank params unchanged");
        Assert.Equal(MetaLearningAlgorithmType.MetaLoRABank, algorithm.AlgorithmType);
        Assert.NotNull(algorithm.Adapt(task).Predict(task.QuerySetX));
    }

    [Fact]
    public void AutoLoRA_FiniteLossAndParamChange()
    {
        var model = new LinearVectorModel(3);
        var options = new AutoLoRAOptions<double, Matrix<double>, Vector<double>>(model)
        { InnerLearningRate = 0.01, OuterLearningRate = 0.01 };
        var algorithm = new AutoLoRAAlgorithm<double, Matrix<double>, Vector<double>>(options);
        var paramsBefore = model.GetParameters();

        var task = CreateTask(4997);
        var batch = new TaskBatch<double, Matrix<double>, Vector<double>>(new[] { task });

        var loss = algorithm.MetaTrain(batch);
        Assert.False(double.IsNaN(loss), "AutoLoRA loss is NaN");
        Assert.False(double.IsInfinity(loss), "AutoLoRA loss is infinite");
        Assert.True(ParamsChanged(paramsBefore, model.GetParameters()), "AutoLoRA params unchanged");
        Assert.Equal(MetaLearningAlgorithmType.AutoLoRA, algorithm.AlgorithmType);
        Assert.NotNull(algorithm.Adapt(task).Predict(task.QuerySetX));
    }

    [Fact]
    public void MetaDiff_FiniteLossAndParamChange()
    {
        var model = new LinearVectorModel(3);
        var options = new MetaDiffOptions<double, Matrix<double>, Vector<double>>(model)
        { InnerLearningRate = 0.01, OuterLearningRate = 0.01 };
        var algorithm = new MetaDiffAlgorithm<double, Matrix<double>, Vector<double>>(options);
        var paramsBefore = model.GetParameters();

        var task = CreateTask(1809);
        var batch = new TaskBatch<double, Matrix<double>, Vector<double>>(new[] { task });

        var loss = algorithm.MetaTrain(batch);
        Assert.False(double.IsNaN(loss), "MetaDiff loss is NaN");
        Assert.False(double.IsInfinity(loss), "MetaDiff loss is infinite");
        Assert.True(ParamsChanged(paramsBefore, model.GetParameters()), "MetaDiff params unchanged");
        Assert.Equal(MetaLearningAlgorithmType.MetaDiff, algorithm.AlgorithmType);
        Assert.NotNull(algorithm.Adapt(task).Predict(task.QuerySetX));
    }

    [Fact]
    public void MetaDM_FiniteLossAndParamChange()
    {
        var model = new LinearVectorModel(3);
        var options = new MetaDMOptions<double, Matrix<double>, Vector<double>>(model)
        { InnerLearningRate = 0.01, OuterLearningRate = 0.01 };
        var algorithm = new MetaDMAlgorithm<double, Matrix<double>, Vector<double>>(options);
        var paramsBefore = model.GetParameters();

        var task = CreateTask(4932);
        var batch = new TaskBatch<double, Matrix<double>, Vector<double>>(new[] { task });

        var loss = algorithm.MetaTrain(batch);
        Assert.False(double.IsNaN(loss), "MetaDM loss is NaN");
        Assert.False(double.IsInfinity(loss), "MetaDM loss is infinite");
        Assert.True(ParamsChanged(paramsBefore, model.GetParameters()), "MetaDM params unchanged");
        Assert.Equal(MetaLearningAlgorithmType.MetaDM, algorithm.AlgorithmType);
        Assert.NotNull(algorithm.Adapt(task).Predict(task.QuerySetX));
    }

    [Fact]
    public void MetaDDPM_FiniteLossAndParamChange()
    {
        var model = new LinearVectorModel(3);
        var options = new MetaDDPMOptions<double, Matrix<double>, Vector<double>>(model)
        { InnerLearningRate = 0.01, OuterLearningRate = 0.01 };
        var algorithm = new MetaDDPMAlgorithm<double, Matrix<double>, Vector<double>>(options);
        var paramsBefore = model.GetParameters();

        var task = CreateTask(9068);
        var batch = new TaskBatch<double, Matrix<double>, Vector<double>>(new[] { task });

        var loss = algorithm.MetaTrain(batch);
        Assert.False(double.IsNaN(loss), "MetaDDPM loss is NaN");
        Assert.False(double.IsInfinity(loss), "MetaDDPM loss is infinite");
        Assert.True(ParamsChanged(paramsBefore, model.GetParameters()), "MetaDDPM params unchanged");
        Assert.Equal(MetaLearningAlgorithmType.MetaDDPM, algorithm.AlgorithmType);
        Assert.NotNull(algorithm.Adapt(task).Predict(task.QuerySetX));
    }

    [Fact]
    public void MetaLoRA_MultiStep_StableLoss()
    {
        var model = new LinearVectorModel(3);
        var options = new MetaLoRAOptions<double, Matrix<double>, Vector<double>>(model)
        { InnerLearningRate = 0.01, OuterLearningRate = 0.001 };
        var algorithm = new MetaLoRAAlgorithm<double, Matrix<double>, Vector<double>>(options);

        var task = CreateTask(42);
        var batch = new TaskBatch<double, Matrix<double>, Vector<double>>(new[] { task });

        int finiteCount = 0;
        for (int i = 0; i < 5; i++)
        {
            var loss = algorithm.MetaTrain(batch);
            if (!double.IsNaN(loss) && !double.IsInfinity(loss)) finiteCount++;
        }
        Assert.True(finiteCount >= 4, $"Only {finiteCount}/5 MetaLoRA steps produced finite loss");
    }
}
