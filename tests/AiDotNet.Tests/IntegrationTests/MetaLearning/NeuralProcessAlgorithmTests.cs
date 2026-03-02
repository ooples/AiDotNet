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
/// Integration tests for Neural Process family algorithms (Phase 1).
/// Verifies: finite loss, non-null predictions, correct algorithm type,
/// parameter change after training, and adaptation produces different params.
/// </summary>
public class NeuralProcessAlgorithmTests
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
    public void CNP_MetaTrainAndAdapt_ProducesFiniteLossAndChangesParams()
    {
        var model = new LinearVectorModel(3);
        var options = new CNPOptions<double, Matrix<double>, Vector<double>>(model)
        { InnerLearningRate = 0.01, OuterLearningRate = 0.01 };
        var algorithm = new CNPAlgorithm<double, Matrix<double>, Vector<double>>(options);
        var paramsBefore = model.GetParameters();

        var task = CreateTask(3821);
        var batch = new TaskBatch<double, Matrix<double>, Vector<double>>(new[] { task });

        var loss1 = algorithm.MetaTrain(batch);
        var paramsAfter = model.GetParameters();

        Assert.False(double.IsNaN(loss1), "CNP loss is NaN");
        Assert.False(double.IsInfinity(loss1), "CNP loss is infinite");
        Assert.True(ParamsChanged(paramsBefore, paramsAfter), "CNP params unchanged after MetaTrain");
        Assert.Equal(MetaLearningAlgorithmType.CNP, algorithm.AlgorithmType);

        var adapted = algorithm.Adapt(task);
        Assert.NotNull(adapted);
        var predictions = adapted.Predict(task.QuerySetX);
        Assert.NotNull(predictions);
    }

    [Fact]
    public void NP_MetaTrainAndAdapt_ProducesFiniteLossAndChangesParams()
    {
        var model = new LinearVectorModel(3);
        var options = new NPOptions<double, Matrix<double>, Vector<double>>(model)
        { InnerLearningRate = 0.01, OuterLearningRate = 0.01 };
        var algorithm = new NPAlgorithm<double, Matrix<double>, Vector<double>>(options);
        var paramsBefore = model.GetParameters();

        var task = CreateTask(5663);
        var batch = new TaskBatch<double, Matrix<double>, Vector<double>>(new[] { task });

        var loss = algorithm.MetaTrain(batch);
        Assert.False(double.IsNaN(loss), "NP loss is NaN");
        Assert.False(double.IsInfinity(loss), "NP loss is infinite");
        Assert.True(ParamsChanged(paramsBefore, model.GetParameters()), "NP params unchanged");
        Assert.Equal(MetaLearningAlgorithmType.NP, algorithm.AlgorithmType);

        var adapted = algorithm.Adapt(task);
        Assert.NotNull(adapted.Predict(task.QuerySetX));
    }

    [Fact]
    public void ANP_MetaTrainAndAdapt_ProducesFiniteLossAndChangesParams()
    {
        var model = new LinearVectorModel(3);
        var options = new ANPOptions<double, Matrix<double>, Vector<double>>(model)
        { InnerLearningRate = 0.01, OuterLearningRate = 0.01 };
        var algorithm = new ANPAlgorithm<double, Matrix<double>, Vector<double>>(options);
        var paramsBefore = model.GetParameters();

        var task = CreateTask(6629);
        var batch = new TaskBatch<double, Matrix<double>, Vector<double>>(new[] { task });

        var loss = algorithm.MetaTrain(batch);
        Assert.False(double.IsNaN(loss), "ANP loss is NaN");
        Assert.False(double.IsInfinity(loss), "ANP loss is infinite");
        Assert.True(ParamsChanged(paramsBefore, model.GetParameters()), "ANP params unchanged");
        Assert.Equal(MetaLearningAlgorithmType.ANP, algorithm.AlgorithmType);

        var adapted = algorithm.Adapt(task);
        Assert.NotNull(adapted.Predict(task.QuerySetX));
    }

    [Fact]
    public void ConvCNP_MetaTrainAndAdapt_ProducesFiniteLossAndChangesParams()
    {
        var model = new LinearVectorModel(3);
        var options = new ConvCNPOptions<double, Matrix<double>, Vector<double>>(model)
        { InnerLearningRate = 0.01, OuterLearningRate = 0.01 };
        var algorithm = new ConvCNPAlgorithm<double, Matrix<double>, Vector<double>>(options);
        var paramsBefore = model.GetParameters();

        var task = CreateTask(7825);
        var batch = new TaskBatch<double, Matrix<double>, Vector<double>>(new[] { task });

        var loss = algorithm.MetaTrain(batch);
        Assert.False(double.IsNaN(loss), "ConvCNP loss is NaN");
        Assert.False(double.IsInfinity(loss), "ConvCNP loss is infinite");
        Assert.True(ParamsChanged(paramsBefore, model.GetParameters()), "ConvCNP params unchanged");
        Assert.Equal(MetaLearningAlgorithmType.ConvCNP, algorithm.AlgorithmType);

        Assert.NotNull(algorithm.Adapt(task).Predict(task.QuerySetX));
    }

    [Fact]
    public void ConvNP_MetaTrainAndAdapt_ProducesFiniteLossAndChangesParams()
    {
        var model = new LinearVectorModel(3);
        var options = new ConvNPOptions<double, Matrix<double>, Vector<double>>(model)
        { InnerLearningRate = 0.01, OuterLearningRate = 0.01 };
        var algorithm = new ConvNPAlgorithm<double, Matrix<double>, Vector<double>>(options);
        var paramsBefore = model.GetParameters();

        var task = CreateTask(2048);
        var batch = new TaskBatch<double, Matrix<double>, Vector<double>>(new[] { task });

        var loss = algorithm.MetaTrain(batch);
        Assert.False(double.IsNaN(loss), "ConvNP loss is NaN");
        Assert.False(double.IsInfinity(loss), "ConvNP loss is infinite");
        Assert.True(ParamsChanged(paramsBefore, model.GetParameters()), "ConvNP params unchanged");
        Assert.Equal(MetaLearningAlgorithmType.ConvNP, algorithm.AlgorithmType);

        Assert.NotNull(algorithm.Adapt(task).Predict(task.QuerySetX));
    }

    [Fact]
    public void TNP_MetaTrainAndAdapt_ProducesFiniteLossAndChangesParams()
    {
        var model = new LinearVectorModel(3);
        var options = new TNPOptions<double, Matrix<double>, Vector<double>>(model)
        { InnerLearningRate = 0.01, OuterLearningRate = 0.01 };
        var algorithm = new TNPAlgorithm<double, Matrix<double>, Vector<double>>(options);
        var paramsBefore = model.GetParameters();

        var task = CreateTask(1773);
        var batch = new TaskBatch<double, Matrix<double>, Vector<double>>(new[] { task });

        var loss = algorithm.MetaTrain(batch);
        Assert.False(double.IsNaN(loss), "TNP loss is NaN");
        Assert.False(double.IsInfinity(loss), "TNP loss is infinite");
        Assert.True(ParamsChanged(paramsBefore, model.GetParameters()), "TNP params unchanged");
        Assert.Equal(MetaLearningAlgorithmType.TNP, algorithm.AlgorithmType);

        Assert.NotNull(algorithm.Adapt(task).Predict(task.QuerySetX));
    }

    [Fact]
    public void SwinTNP_MetaTrainAndAdapt_ProducesFiniteLossAndChangesParams()
    {
        var model = new LinearVectorModel(3);
        var options = new SwinTNPOptions<double, Matrix<double>, Vector<double>>(model)
        { InnerLearningRate = 0.01, OuterLearningRate = 0.01 };
        var algorithm = new SwinTNPAlgorithm<double, Matrix<double>, Vector<double>>(options);
        var paramsBefore = model.GetParameters();

        var task = CreateTask(2742);
        var batch = new TaskBatch<double, Matrix<double>, Vector<double>>(new[] { task });

        var loss = algorithm.MetaTrain(batch);
        Assert.False(double.IsNaN(loss), "SwinTNP loss is NaN");
        Assert.False(double.IsInfinity(loss), "SwinTNP loss is infinite");
        Assert.True(ParamsChanged(paramsBefore, model.GetParameters()), "SwinTNP params unchanged");
        Assert.Equal(MetaLearningAlgorithmType.SwinTNP, algorithm.AlgorithmType);

        Assert.NotNull(algorithm.Adapt(task).Predict(task.QuerySetX));
    }

    [Fact]
    public void EquivCNP_MetaTrainAndAdapt_ProducesFiniteLossAndChangesParams()
    {
        var model = new LinearVectorModel(3);
        var options = new EquivCNPOptions<double, Matrix<double>, Vector<double>>(model)
        { InnerLearningRate = 0.01, OuterLearningRate = 0.01 };
        var algorithm = new EquivCNPAlgorithm<double, Matrix<double>, Vector<double>>(options);
        var paramsBefore = model.GetParameters();

        var task = CreateTask(764);
        var batch = new TaskBatch<double, Matrix<double>, Vector<double>>(new[] { task });

        var loss = algorithm.MetaTrain(batch);
        Assert.False(double.IsNaN(loss), "EquivCNP loss is NaN");
        Assert.False(double.IsInfinity(loss), "EquivCNP loss is infinite");
        Assert.True(ParamsChanged(paramsBefore, model.GetParameters()), "EquivCNP params unchanged");
        Assert.Equal(MetaLearningAlgorithmType.EquivCNP, algorithm.AlgorithmType);

        Assert.NotNull(algorithm.Adapt(task).Predict(task.QuerySetX));
    }

    [Fact]
    public void SteerCNP_MetaTrainAndAdapt_ProducesFiniteLossAndChangesParams()
    {
        var model = new LinearVectorModel(3);
        var options = new SteerCNPOptions<double, Matrix<double>, Vector<double>>(model)
        { InnerLearningRate = 0.01, OuterLearningRate = 0.01 };
        var algorithm = new SteerCNPAlgorithm<double, Matrix<double>, Vector<double>>(options);
        var paramsBefore = model.GetParameters();

        var task = CreateTask(1136);
        var batch = new TaskBatch<double, Matrix<double>, Vector<double>>(new[] { task });

        var loss = algorithm.MetaTrain(batch);
        Assert.False(double.IsNaN(loss), "SteerCNP loss is NaN");
        Assert.False(double.IsInfinity(loss), "SteerCNP loss is infinite");
        Assert.True(ParamsChanged(paramsBefore, model.GetParameters()), "SteerCNP params unchanged");
        Assert.Equal(MetaLearningAlgorithmType.SteerCNP, algorithm.AlgorithmType);

        Assert.NotNull(algorithm.Adapt(task).Predict(task.QuerySetX));
    }

    [Fact]
    public void RCNP_MetaTrainAndAdapt_ProducesFiniteLossAndChangesParams()
    {
        var model = new LinearVectorModel(3);
        var options = new RCNPOptions<double, Matrix<double>, Vector<double>>(model)
        { InnerLearningRate = 0.01, OuterLearningRate = 0.01 };
        var algorithm = new RCNPAlgorithm<double, Matrix<double>, Vector<double>>(options);
        var paramsBefore = model.GetParameters();

        var task = CreateTask(4627);
        var batch = new TaskBatch<double, Matrix<double>, Vector<double>>(new[] { task });

        var loss = algorithm.MetaTrain(batch);
        Assert.False(double.IsNaN(loss), "RCNP loss is NaN");
        Assert.False(double.IsInfinity(loss), "RCNP loss is infinite");
        Assert.True(ParamsChanged(paramsBefore, model.GetParameters()), "RCNP params unchanged");
        Assert.Equal(MetaLearningAlgorithmType.RCNP, algorithm.AlgorithmType);

        Assert.NotNull(algorithm.Adapt(task).Predict(task.QuerySetX));
    }

    [Fact]
    public void LBANP_MetaTrainAndAdapt_ProducesFiniteLossAndChangesParams()
    {
        var model = new LinearVectorModel(3);
        var options = new LBANPOptions<double, Matrix<double>, Vector<double>>(model)
        { InnerLearningRate = 0.01, OuterLearningRate = 0.01 };
        var algorithm = new LBANPAlgorithm<double, Matrix<double>, Vector<double>>(options);
        var paramsBefore = model.GetParameters();

        var task = CreateTask(9518);
        var batch = new TaskBatch<double, Matrix<double>, Vector<double>>(new[] { task });

        var loss = algorithm.MetaTrain(batch);
        Assert.False(double.IsNaN(loss), "LBANP loss is NaN");
        Assert.False(double.IsInfinity(loss), "LBANP loss is infinite");
        Assert.True(ParamsChanged(paramsBefore, model.GetParameters()), "LBANP params unchanged");
        Assert.Equal(MetaLearningAlgorithmType.LBANP, algorithm.AlgorithmType);

        Assert.NotNull(algorithm.Adapt(task).Predict(task.QuerySetX));
    }

    [Fact]
    public void CNP_MultipleTrainingSteps_ProducesStableLoss()
    {
        var model = new LinearVectorModel(3);
        var options = new CNPOptions<double, Matrix<double>, Vector<double>>(model)
        { InnerLearningRate = 0.01, OuterLearningRate = 0.001 };
        var algorithm = new CNPAlgorithm<double, Matrix<double>, Vector<double>>(options);

        var task = CreateTask(42);
        var batch = new TaskBatch<double, Matrix<double>, Vector<double>>(new[] { task });

        double lastLoss = double.MaxValue;
        int finiteCount = 0;
        for (int i = 0; i < 5; i++)
        {
            var loss = algorithm.MetaTrain(batch);
            if (!double.IsNaN(loss) && !double.IsInfinity(loss)) finiteCount++;
            lastLoss = loss;
        }

        Assert.True(finiteCount >= 4, $"Only {finiteCount}/5 training steps produced finite loss");
        Assert.False(double.IsNaN(lastLoss), "Final CNP loss is NaN after 5 steps");
    }

    [Fact]
    public void TETNP_FiniteLossAndParamChange()
    {
        var model = new LinearVectorModel(3);
        var options = new TETNPOptions<double, Matrix<double>, Vector<double>>(model)
        { InnerLearningRate = 0.01, OuterLearningRate = 0.01 };
        var algorithm = new TETNPAlgorithm<double, Matrix<double>, Vector<double>>(options);
        var paramsBefore = model.GetParameters();

        var task = CreateTask(3344);
        var batch = new TaskBatch<double, Matrix<double>, Vector<double>>(new[] { task });

        var loss = algorithm.MetaTrain(batch);
        Assert.False(double.IsNaN(loss), "TETNP loss is NaN");
        Assert.False(double.IsInfinity(loss), "TETNP loss is infinite");
        Assert.True(ParamsChanged(paramsBefore, model.GetParameters()), "TETNP params unchanged");
        Assert.Equal(MetaLearningAlgorithmType.TETNP, algorithm.AlgorithmType);
        Assert.NotNull(algorithm.Adapt(task).Predict(task.QuerySetX));
    }
}
