using System;
using AiDotNet.Data.Structures;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.MetaLearning;
using AiDotNet.MetaLearning.Algorithms;
using AiDotNet.MetaLearning.Data;
using AiDotNet.MetaLearning.Options;
using Xunit;
using System.Threading.Tasks;

namespace AiDotNet.Tests.IntegrationTests.MetaLearning;

/// <summary>
/// Integration tests for Task Augmentation, Transductive, and Hypernetwork algorithms (Phase 7).
/// Verifies: finite loss, parameter change, adaptation, and multi-step stability.
/// </summary>
public class AdvancedMetaTests
{
    private static MetaLearningTask<double, Matrix<double>, Vector<double>> CreateTask(int seed)
    {
        var supportX = new Matrix<double>(4, 3);
        var supportY = new Vector<double>(4);
        var queryX = new Matrix<double>(4, 3);
        var queryY = new Vector<double>(4);
        var rng = AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(seed);
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

    // ── Phase 7a: Task Augmentation ──

    [Fact(Timeout = 120000)]
    public async Task MetaTask_FiniteLossAndParamChange()
    {
        var model = new LinearVectorModel(3);
        var options = new MetaTaskOptions<double, Matrix<double>, Vector<double>>(model)
        { InnerLearningRate = 0.01, OuterLearningRate = 0.01 };
        var algorithm = new MetaTaskAlgorithm<double, Matrix<double>, Vector<double>>(options);
        var paramsBefore = model.GetParameters();

        var task = CreateTask(1899);
        var batch = new TaskBatch<double, Matrix<double>, Vector<double>>(new[] { task });

        var loss = algorithm.MetaTrain(batch);
        Assert.False(double.IsNaN(loss), "MetaTask loss is NaN");
        Assert.False(double.IsInfinity(loss), "MetaTask loss is infinite");
        Assert.True(ParamsChanged(paramsBefore, model.GetParameters()), "MetaTask params unchanged");
        Assert.Equal(MetaLearningAlgorithmType.MetaTask, algorithm.AlgorithmType);
        Assert.NotNull(algorithm.Adapt(task).Predict(task.QuerySetX));
    }

    [Fact(Timeout = 120000)]
    public async Task ATAML_FiniteLossAndParamChange()
    {
        var model = new LinearVectorModel(3);
        var options = new ATAMLOptions<double, Matrix<double>, Vector<double>>(model)
        { InnerLearningRate = 0.01, OuterLearningRate = 0.01 };
        var algorithm = new ATAMLAlgorithm<double, Matrix<double>, Vector<double>>(options);
        var paramsBefore = model.GetParameters();

        var task = CreateTask(3592);
        var batch = new TaskBatch<double, Matrix<double>, Vector<double>>(new[] { task });

        var loss = algorithm.MetaTrain(batch);
        Assert.False(double.IsNaN(loss), "ATAML loss is NaN");
        Assert.False(double.IsInfinity(loss), "ATAML loss is infinite");
        Assert.True(ParamsChanged(paramsBefore, model.GetParameters()), "ATAML params unchanged");
        Assert.Equal(MetaLearningAlgorithmType.ATAML, algorithm.AlgorithmType);
        Assert.NotNull(algorithm.Adapt(task).Predict(task.QuerySetX));
    }

    [Fact(Timeout = 120000)]
    public async Task MPTS_FiniteLossAndParamChange()
    {
        var model = new LinearVectorModel(3);
        var options = new MPTSOptions<double, Matrix<double>, Vector<double>>(model)
        { InnerLearningRate = 0.01, OuterLearningRate = 0.01 };
        var algorithm = new MPTSAlgorithm<double, Matrix<double>, Vector<double>>(options);
        var paramsBefore = model.GetParameters();

        var task = CreateTask(9546);
        var batch = new TaskBatch<double, Matrix<double>, Vector<double>>(new[] { task });

        var loss = algorithm.MetaTrain(batch);
        Assert.False(double.IsNaN(loss), "MPTS loss is NaN");
        Assert.False(double.IsInfinity(loss), "MPTS loss is infinite");
        Assert.True(ParamsChanged(paramsBefore, model.GetParameters()), "MPTS params unchanged");
        Assert.Equal(MetaLearningAlgorithmType.MPTS, algorithm.AlgorithmType);
        Assert.NotNull(algorithm.Adapt(task).Predict(task.QuerySetX));
    }

    [Fact(Timeout = 120000)]
    public async Task DynamicTaskSampling_FiniteLossAndParamChange()
    {
        var model = new LinearVectorModel(3);
        var options = new DynamicTaskSamplingOptions<double, Matrix<double>, Vector<double>>(model)
        { InnerLearningRate = 0.01, OuterLearningRate = 0.01 };
        var algorithm = new DynamicTaskSamplingAlgorithm<double, Matrix<double>, Vector<double>>(options);
        var paramsBefore = model.GetParameters();

        var task = CreateTask(2439);
        var batch = new TaskBatch<double, Matrix<double>, Vector<double>>(new[] { task });

        var loss = algorithm.MetaTrain(batch);
        Assert.False(double.IsNaN(loss), "DynamicTaskSampling loss is NaN");
        Assert.False(double.IsInfinity(loss), "DynamicTaskSampling loss is infinite");
        Assert.True(ParamsChanged(paramsBefore, model.GetParameters()), "DynamicTaskSampling params unchanged");
        Assert.Equal(MetaLearningAlgorithmType.DynamicTaskSampling, algorithm.AlgorithmType);
        Assert.NotNull(algorithm.Adapt(task).Predict(task.QuerySetX));
    }

    [Fact(Timeout = 120000)]
    public async Task UnsupervisedMetaLearn_FiniteLossAndParamChange()
    {
        var model = new LinearVectorModel(3);
        var options = new UnsupervisedMetaLearnOptions<double, Matrix<double>, Vector<double>>(model)
        { InnerLearningRate = 0.01, OuterLearningRate = 0.01 };
        var algorithm = new UnsupervisedMetaLearnAlgorithm<double, Matrix<double>, Vector<double>>(options);
        var paramsBefore = model.GetParameters();

        var task = CreateTask(4459);
        var batch = new TaskBatch<double, Matrix<double>, Vector<double>>(new[] { task });

        var loss = algorithm.MetaTrain(batch);
        Assert.False(double.IsNaN(loss), "UnsupervisedMetaLearn loss is NaN");
        Assert.False(double.IsInfinity(loss), "UnsupervisedMetaLearn loss is infinite");
        Assert.True(ParamsChanged(paramsBefore, model.GetParameters()), "UnsupervisedMetaLearn params unchanged");
        Assert.Equal(MetaLearningAlgorithmType.UnsupervisedMetaLearn, algorithm.AlgorithmType);
        Assert.NotNull(algorithm.Adapt(task).Predict(task.QuerySetX));
    }

    // ── Phase 7b: Transductive ──

    [Fact(Timeout = 120000)]
    public async Task LFT_FiniteLossAndParamChange()
    {
        var model = new LinearVectorModel(3);
        var options = new LFTOptions<double, Matrix<double>, Vector<double>>(model)
        { InnerLearningRate = 0.01, OuterLearningRate = 0.01, FeatureDimension = 1 };
        var algorithm = new LFTAlgorithm<double, Matrix<double>, Vector<double>>(options);
        var paramsBefore = model.GetParameters();

        var task = CreateTask(9004);
        var batch = new TaskBatch<double, Matrix<double>, Vector<double>>(new[] { task });

        var loss = algorithm.MetaTrain(batch);
        Assert.False(double.IsNaN(loss), "LFT loss is NaN");
        Assert.False(double.IsInfinity(loss), "LFT loss is infinite");
        Assert.True(ParamsChanged(paramsBefore, model.GetParameters()), "LFT params unchanged");
        Assert.Equal(MetaLearningAlgorithmType.LFT, algorithm.AlgorithmType);
        Assert.NotNull(algorithm.Adapt(task).Predict(task.QuerySetX));
    }

    [Fact(Timeout = 120000)]
    public async Task BayTransProto_FiniteLossAndParamChange()
    {
        var model = new LinearVectorModel(3);
        var options = new BayTransProtoOptions<double, Matrix<double>, Vector<double>>(model)
        { InnerLearningRate = 0.01, OuterLearningRate = 0.01 };
        var algorithm = new BayTransProtoAlgorithm<double, Matrix<double>, Vector<double>>(options);
        var paramsBefore = model.GetParameters();

        var task = CreateTask(7354);
        var batch = new TaskBatch<double, Matrix<double>, Vector<double>>(new[] { task });

        var loss = algorithm.MetaTrain(batch);
        Assert.False(double.IsNaN(loss), "BayTransProto loss is NaN");
        Assert.False(double.IsInfinity(loss), "BayTransProto loss is infinite");
        Assert.True(ParamsChanged(paramsBefore, model.GetParameters()), "BayTransProto params unchanged");
        Assert.Equal(MetaLearningAlgorithmType.BayTransProto, algorithm.AlgorithmType);
        Assert.NotNull(algorithm.Adapt(task).Predict(task.QuerySetX));
    }

    [Theory]
    [InlineData(0)]
    [InlineData(-1)]
    public void BayTransProto_MetaTrain_RejectsNonpositiveSampleCountBeforeUpdatingTheModel(int samples)
    {
        var model = new LinearVectorModel(3);
        var options = new BayTransProtoOptions<double, Matrix<double>, Vector<double>>(model)
        { InnerLearningRate = 0.01, OuterLearningRate = 0.01 };
        var algorithm = new BayTransProtoAlgorithm<double, Matrix<double>, Vector<double>>(options);
        options.NumPosteriorSamples = samples;
        var paramsBefore = model.GetParameters().Clone();

        var batch = new TaskBatch<double, Matrix<double>, Vector<double>>(new[] { CreateTask(7354) });
        Assert.Throws<InvalidOperationException>(() => algorithm.MetaTrain(batch));
        Assert.Equal(paramsBefore.ToArray(), model.GetParameters().ToArray());
    }

    [Fact(Timeout = 120000)]
    public async Task BayTransProto_MetaTrain_LearnsPosteriorLogVarianceTowardKLOptimum()
    {
        // q(theta) = N(mean, diag exp(logVar)) is regularized by KL(q || N(theta_init, I)), whose per-dimension
        // log-variance part 0.5 * (exp(lv) - lv) is minimized at lv = 0. With a dominant KL weight and lv = 3, the
        // SPSA step on the variational objective must lower that part. Three parameters (two features plus a
        // bias) keep every Rademacher direction's component sum odd, so the KL directional derivative is never 0.
        var model = new LinearVectorModel(2);
        var options = new BayTransProtoOptions<double, Matrix<double>, Vector<double>>(model)
        {
            InnerLearningRate = 0.01, OuterLearningRate = 0.01,
            InitialLogVar = 3.0, KLWeight = 1000.0, RandomSeed = 42
        };
        var algorithm = new BayTransProtoAlgorithm<double, Matrix<double>, Vector<double>>(options);
        var before = algorithm.PosteriorLogVariance.Clone();

        var x = new Matrix<double>(4, 2);
        var y = new Vector<double>(4);
        var rng = AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(7355);
        for (int i = 0; i < 4; i++)
        {
            x[i, 0] = rng.NextDouble() - 0.5;
            x[i, 1] = rng.NextDouble() - 0.5;
            y[i] = i % 2;
        }

        var task = new MetaLearningTask<double, Matrix<double>, Vector<double>>
        {
            SupportSetX = x, SupportSetY = y, QuerySetX = x, QuerySetY = y,
            NumWays = 2, NumShots = 2, NumQueryPerClass = 2
        };
        var loss = algorithm.MetaTrain(new TaskBatch<double, Matrix<double>, Vector<double>>(new[] { task }));
        var after = algorithm.PosteriorLogVariance;

        Assert.False(double.IsNaN(loss) || double.IsInfinity(loss), "BayTransProto loss is not finite");
        Assert.True(ParamsChanged(before, after), "The posterior log-variance got no update signal");
        double KlPart(Vector<double> logVar)
        {
            double sum = 0;
            for (int d = 0; d < logVar.Length; d++) sum += Math.Exp(logVar[d]) - logVar[d];
            return sum;
        }

        Assert.True(KlPart(after) < KlPart(before),
            $"KL log-variance term should decrease: before {KlPart(before)}, after {KlPart(after)}");
    }

    [Fact(Timeout = 120000)]
    public async Task JMP_FiniteLossAndParamChange()
    {
        var model = new LinearVectorModel(3);
        var options = new JMPOptions<double, Matrix<double>, Vector<double>>(model)
        { InnerLearningRate = 0.01, OuterLearningRate = 0.01 };
        var algorithm = new JMPAlgorithm<double, Matrix<double>, Vector<double>>(options);
        var paramsBefore = model.GetParameters();

        var task = CreateTask(8249);
        var batch = new TaskBatch<double, Matrix<double>, Vector<double>>(new[] { task });

        var loss = algorithm.MetaTrain(batch);
        Assert.False(double.IsNaN(loss), "JMP loss is NaN");
        Assert.False(double.IsInfinity(loss), "JMP loss is infinite");
        Assert.True(ParamsChanged(paramsBefore, model.GetParameters()), "JMP params unchanged");
        Assert.Equal(MetaLearningAlgorithmType.JMP, algorithm.AlgorithmType);
        Assert.NotNull(algorithm.Adapt(task).Predict(task.QuerySetX));
    }

    [Fact(Timeout = 120000)]
    public async Task ETPN_FiniteLossAndParamChange()
    {
        var model = new LinearVectorModel(3);
        var options = new ETPNOptions<double, Matrix<double>, Vector<double>>(model)
        { InnerLearningRate = 0.01, OuterLearningRate = 0.01 };
        var algorithm = new ETPNAlgorithm<double, Matrix<double>, Vector<double>>(options);
        var paramsBefore = model.GetParameters();

        var task = CreateTask(8774);
        var batch = new TaskBatch<double, Matrix<double>, Vector<double>>(new[] { task });

        var loss = algorithm.MetaTrain(batch);
        Assert.False(double.IsNaN(loss), "ETPN loss is NaN");
        Assert.False(double.IsInfinity(loss), "ETPN loss is infinite");
        Assert.True(ParamsChanged(paramsBefore, model.GetParameters()), "ETPN params unchanged");
        Assert.Equal(MetaLearningAlgorithmType.ETPN, algorithm.AlgorithmType);
        Assert.NotNull(algorithm.Adapt(task).Predict(task.QuerySetX));
    }

    [Fact(Timeout = 120000)]
    public async Task ActiveTransFSL_FiniteLossAndParamChange()
    {
        var model = new LinearVectorModel(3);
        var options = new ActiveTransFSLOptions<double, Matrix<double>, Vector<double>>(model)
        { InnerLearningRate = 0.01, OuterLearningRate = 0.01 };
        var algorithm = new ActiveTransFSLAlgorithm<double, Matrix<double>, Vector<double>>(options);
        var paramsBefore = model.GetParameters();

        var task = CreateTask(1582);
        var batch = new TaskBatch<double, Matrix<double>, Vector<double>>(new[] { task });

        var loss = algorithm.MetaTrain(batch);
        Assert.False(double.IsNaN(loss), "ActiveTransFSL loss is NaN");
        Assert.False(double.IsInfinity(loss), "ActiveTransFSL loss is infinite");
        Assert.True(ParamsChanged(paramsBefore, model.GetParameters()), "ActiveTransFSL params unchanged");
        Assert.Equal(MetaLearningAlgorithmType.ActiveTransFSL, algorithm.AlgorithmType);
        Assert.NotNull(algorithm.Adapt(task).Predict(task.QuerySetX));
    }

    // ── Phase 7c: Hypernetwork ──

    [Fact(Timeout = 120000)]
    public async Task TaskCondHyperNet_FiniteLossAndParamChange()
    {
        var model = new LinearVectorModel(3);
        var options = new TaskCondHyperNetOptions<double, Matrix<double>, Vector<double>>(model)
        { InnerLearningRate = 0.01, OuterLearningRate = 0.01 };
        var algorithm = new TaskCondHyperNetAlgorithm<double, Matrix<double>, Vector<double>>(options);
        var paramsBefore = model.GetParameters();

        var task = CreateTask(2856);
        var batch = new TaskBatch<double, Matrix<double>, Vector<double>>(new[] { task });

        var loss = algorithm.MetaTrain(batch);
        Assert.False(double.IsNaN(loss), "TaskCondHyperNet loss is NaN");
        Assert.False(double.IsInfinity(loss), "TaskCondHyperNet loss is infinite");
        Assert.True(ParamsChanged(paramsBefore, model.GetParameters()), "TaskCondHyperNet params unchanged");
        Assert.Equal(MetaLearningAlgorithmType.TaskCondHyperNet, algorithm.AlgorithmType);
        Assert.NotNull(algorithm.Adapt(task).Predict(task.QuerySetX));
    }

    [Fact(Timeout = 120000)]
    public async Task HyperCLIP_FiniteLossAndParamChange()
    {
        var model = new LinearVectorModel(3);
        var options = new HyperCLIPOptions<double, Matrix<double>, Vector<double>>(model)
        { InnerLearningRate = 0.01, OuterLearningRate = 0.01 };
        var algorithm = new HyperCLIPAlgorithm<double, Matrix<double>, Vector<double>>(options);
        var paramsBefore = model.GetParameters();

        var task = CreateTask(8270);
        var batch = new TaskBatch<double, Matrix<double>, Vector<double>>(new[] { task });

        var loss = algorithm.MetaTrain(batch);
        Assert.False(double.IsNaN(loss), "HyperCLIP loss is NaN");
        Assert.False(double.IsInfinity(loss), "HyperCLIP loss is infinite");
        Assert.True(ParamsChanged(paramsBefore, model.GetParameters()), "HyperCLIP params unchanged");
        Assert.Equal(MetaLearningAlgorithmType.HyperCLIP, algorithm.AlgorithmType);
        Assert.NotNull(algorithm.Adapt(task).Predict(task.QuerySetX));
    }

    [Fact(Timeout = 120000)]
    public async Task RecurrentHyperNet_FiniteLossAndParamChange()
    {
        var model = new LinearVectorModel(3);
        var options = new RecurrentHyperNetOptions<double, Matrix<double>, Vector<double>>(model)
        { InnerLearningRate = 0.01, OuterLearningRate = 0.01 };
        var algorithm = new RecurrentHyperNetAlgorithm<double, Matrix<double>, Vector<double>>(options);
        var paramsBefore = model.GetParameters();

        var task = CreateTask(9617);
        var batch = new TaskBatch<double, Matrix<double>, Vector<double>>(new[] { task });

        var loss = algorithm.MetaTrain(batch);
        Assert.False(double.IsNaN(loss), "RecurrentHyperNet loss is NaN");
        Assert.False(double.IsInfinity(loss), "RecurrentHyperNet loss is infinite");
        Assert.True(ParamsChanged(paramsBefore, model.GetParameters()), "RecurrentHyperNet params unchanged");
        Assert.Equal(MetaLearningAlgorithmType.RecurrentHyperNet, algorithm.AlgorithmType);
        Assert.NotNull(algorithm.Adapt(task).Predict(task.QuerySetX));
    }

    [Fact(Timeout = 120000)]
    public async Task HyperNeRFMeta_FiniteLossAndParamChange()
    {
        var model = new LinearVectorModel(3);
        var options = new HyperNeRFMetaOptions<double, Matrix<double>, Vector<double>>(model)
        { InnerLearningRate = 0.01, OuterLearningRate = 0.01 };
        var algorithm = new HyperNeRFMetaAlgorithm<double, Matrix<double>, Vector<double>>(options);
        var paramsBefore = model.GetParameters();

        var task = CreateTask(2785);
        var batch = new TaskBatch<double, Matrix<double>, Vector<double>>(new[] { task });

        var loss = algorithm.MetaTrain(batch);
        Assert.False(double.IsNaN(loss), "HyperNeRFMeta loss is NaN");
        Assert.False(double.IsInfinity(loss), "HyperNeRFMeta loss is infinite");
        Assert.True(ParamsChanged(paramsBefore, model.GetParameters()), "HyperNeRFMeta params unchanged");
        Assert.Equal(MetaLearningAlgorithmType.HyperNeRFMeta, algorithm.AlgorithmType);
        Assert.NotNull(algorithm.Adapt(task).Predict(task.QuerySetX));
    }

    // ── Multi-step stability tests ──

    [Fact(Timeout = 120000)]
    public async Task LFT_MultiStep_StableLoss()
    {
        var model = new LinearVectorModel(3);
        var options = new LFTOptions<double, Matrix<double>, Vector<double>>(model)
        { InnerLearningRate = 0.01, OuterLearningRate = 0.001, FeatureDimension = 1 };
        var algorithm = new LFTAlgorithm<double, Matrix<double>, Vector<double>>(options);

        var task = CreateTask(42);
        var batch = new TaskBatch<double, Matrix<double>, Vector<double>>(new[] { task });

        int finiteCount = 0;
        for (int i = 0; i < 5; i++)
        {
            var loss = algorithm.MetaTrain(batch);
            if (!double.IsNaN(loss) && !double.IsInfinity(loss)) finiteCount++;
        }
        Assert.True(finiteCount >= 4, $"Only {finiteCount}/5 LFT steps produced finite loss");
    }

    [Fact(Timeout = 120000)]
    public async Task TaskCondHyperNet_MultiStep_StableLoss()
    {
        var model = new LinearVectorModel(3);
        var options = new TaskCondHyperNetOptions<double, Matrix<double>, Vector<double>>(model)
        { InnerLearningRate = 0.01, OuterLearningRate = 0.001 };
        var algorithm = new TaskCondHyperNetAlgorithm<double, Matrix<double>, Vector<double>>(options);

        var task = CreateTask(42);
        var batch = new TaskBatch<double, Matrix<double>, Vector<double>>(new[] { task });

        int finiteCount = 0;
        for (int i = 0; i < 5; i++)
        {
            var loss = algorithm.MetaTrain(batch);
            if (!double.IsNaN(loss) && !double.IsInfinity(loss)) finiteCount++;
        }
        Assert.True(finiteCount >= 4, $"Only {finiteCount}/5 TaskCondHyperNet steps produced finite loss");
    }

    [Fact(Timeout = 120000)]
    public async Task RecurrentHyperNet_MultiStep_StableLoss()
    {
        var model = new LinearVectorModel(3);
        var options = new RecurrentHyperNetOptions<double, Matrix<double>, Vector<double>>(model)
        { InnerLearningRate = 0.01, OuterLearningRate = 0.001 };
        var algorithm = new RecurrentHyperNetAlgorithm<double, Matrix<double>, Vector<double>>(options);

        var task = CreateTask(42);
        var batch = new TaskBatch<double, Matrix<double>, Vector<double>>(new[] { task });

        int finiteCount = 0;
        for (int i = 0; i < 5; i++)
        {
            var loss = algorithm.MetaTrain(batch);
            if (!double.IsNaN(loss) && !double.IsInfinity(loss)) finiteCount++;
        }
        Assert.True(finiteCount >= 4, $"Only {finiteCount}/5 RecurrentHyperNet steps produced finite loss");
    }
}
