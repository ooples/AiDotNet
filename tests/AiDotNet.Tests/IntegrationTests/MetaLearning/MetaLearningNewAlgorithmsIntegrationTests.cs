using System;
using AiDotNet.Data.Structures;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.MetaLearning;
using AiDotNet.MetaLearning.Algorithms;
using AiDotNet.MetaLearning.Data;
using AiDotNet.MetaLearning.Options;
using AiDotNet.Tensors;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.MetaLearning;

/// <summary>
/// Integration tests for all new meta-learning algorithms added in the expanded meta-learning PR.
/// Each test exercises the full pipeline: construct options -> create algorithm -> MetaTrain -> Adapt -> Predict.
/// </summary>
public class MetaLearningNewAlgorithmsIntegrationTests
{
    private static MetaLearningTask<double, Matrix<double>, Vector<double>> CreateVectorTask(
        int seed,
        int supportRows = 4,
        int queryRows = 4,
        int featureCount = 3,
        int numWays = 2)
    {
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
            supportY[i] = i % Math.Max(1, numWays);
        }

        for (int i = 0; i < queryRows; i++)
        {
            for (int j = 0; j < featureCount; j++)
            {
                queryX[i, j] = random.NextDouble() - 0.5;
            }
            queryY[i] = i % Math.Max(1, numWays);
        }

        int numShots = Math.Max(1, supportRows / Math.Max(1, numWays));
        int numQueryPerClass = Math.Max(1, queryRows / Math.Max(1, numWays));

        return new MetaLearningTask<double, Matrix<double>, Vector<double>>
        {
            SupportSetX = supportX,
            SupportSetY = supportY,
            QuerySetX = queryX,
            QuerySetY = queryY,
            NumWays = numWays,
            NumShots = numShots,
            NumQueryPerClass = numQueryPerClass,
            Name = $"vector-task-{seed}"
        };
    }

    private static IMetaLearningTask<double, Matrix<double>, Tensor<double>> CreateTensorLabelTask(
        int seed,
        int supportRows = 4,
        int queryRows = 4,
        int featureCount = 3,
        int numWays = 2)
    {
        var supportX = new Matrix<double>(supportRows, featureCount);
        var queryX = new Matrix<double>(queryRows, featureCount);
        var supportY = new Tensor<double>(new[] { supportRows });
        var queryY = new Tensor<double>(new[] { queryRows });

        var random = new Random(seed);

        for (int i = 0; i < supportRows; i++)
        {
            for (int j = 0; j < featureCount; j++)
            {
                supportX[i, j] = random.NextDouble() - 0.5;
            }
            supportY[new[] { i }] = i % Math.Max(1, numWays);
        }

        for (int i = 0; i < queryRows; i++)
        {
            for (int j = 0; j < featureCount; j++)
            {
                queryX[i, j] = random.NextDouble() - 0.5;
            }
            queryY[new[] { i }] = i % Math.Max(1, numWays);
        }

        int numShots = Math.Max(1, supportRows / Math.Max(1, numWays));
        int numQueryPerClass = Math.Max(1, queryRows / Math.Max(1, numWays));

        return new TestMetaLearningTask<double, Matrix<double>, Tensor<double>>(
            supportX,
            supportY,
            queryX,
            queryY,
            numWays,
            numShots,
            numQueryPerClass,
            $"tensor-task-{seed}");
    }

    // ========================================================================
    // Group 1: MAML Variants
    // ========================================================================

    [Fact]
    public void MAMLPlusPlus_MetaTrainAndAdapt_Run()
    {
        var model = new LinearVectorModel(3);
        var options = new MAMLPlusPlusOptions<double, Matrix<double>, Vector<double>>(model)
        {
            InnerLearningRate = 0.01,
            OuterLearningRate = 0.01,
            AdaptationSteps = 2,
            UseMultiStepLoss = true,
            UsePerStepLearningRates = true,
            UseDerivativeOrderAnnealing = false,
            UsePerStepBatchNorm = false
        };

        var algorithm = new MAMLPlusPlusAlgorithm<double, Matrix<double>, Vector<double>>(options);
        var task = CreateVectorTask(100);
        var batch = new TaskBatch<double, Matrix<double>, Vector<double>>(new[] { task });

        var loss = algorithm.MetaTrain(batch);
        var adapted = algorithm.Adapt(task);
        var predictions = adapted.Predict(task.QuerySetX);

        Assert.False(double.IsNaN(loss), "MAMLPlusPlus MetaTrain returned NaN loss");
        Assert.NotNull(predictions);
        Assert.Equal(MetaLearningAlgorithmType.MAMLPlusPlus, algorithm.AlgorithmType);
    }

    [Fact]
    public void OpenMAML_MetaTrainAndAdapt_Run()
    {
        var model = new LinearVectorModel(3);
        var options = new OpenMAMLOptions<double, Matrix<double>, Vector<double>>(model)
        {
            InnerLearningRate = 0.01,
            OuterLearningRate = 0.01,
            AdaptationSteps = 2
        };

        var algorithm = new OpenMAMLAlgorithm<double, Matrix<double>, Vector<double>>(options);
        var task = CreateVectorTask(101);
        var batch = new TaskBatch<double, Matrix<double>, Vector<double>>(new[] { task });

        var loss = algorithm.MetaTrain(batch);
        var adapted = algorithm.Adapt(task);
        var predictions = adapted.Predict(task.QuerySetX);

        Assert.False(double.IsNaN(loss), "OpenMAML MetaTrain returned NaN loss");
        Assert.NotNull(predictions);
        Assert.Equal(MetaLearningAlgorithmType.OpenMAML, algorithm.AlgorithmType);
    }

    [Fact]
    public void HyperMAML_MetaTrainAndAdapt_Run()
    {
        var model = new LinearVectorModel(3);
        var options = new HyperMAMLOptions<double, Matrix<double>, Vector<double>>(model)
        {
            InnerLearningRate = 0.01,
            OuterLearningRate = 0.01,
            AdaptationSteps = 2
        };

        var algorithm = new HyperMAMLAlgorithm<double, Matrix<double>, Vector<double>>(options);
        var task = CreateVectorTask(102);
        var batch = new TaskBatch<double, Matrix<double>, Vector<double>>(new[] { task });

        var loss = algorithm.MetaTrain(batch);
        var adapted = algorithm.Adapt(task);
        var predictions = adapted.Predict(task.QuerySetX);

        Assert.False(double.IsNaN(loss), "HyperMAML MetaTrain returned NaN loss");
        Assert.NotNull(predictions);
        Assert.Equal(MetaLearningAlgorithmType.HyperMAML, algorithm.AlgorithmType);
    }

    // ========================================================================
    // Group 2: Context/Parameter Adaptation Methods
    // ========================================================================

    [Fact]
    public void CAVIA_MetaTrainAndAdapt_Run()
    {
        var model = new LinearVectorModel(3);
        var options = new CAVIAOptions<double, Matrix<double>, Vector<double>>(model)
        {
            ContextDimension = 4,
            AdaptationSteps = 2,
            InnerLearningRate = 0.01,
            OuterLearningRate = 0.01,
            ContextInjectionMode = CAVIAContextInjectionMode.Concatenation,
            UseContextRegularization = false
        };

        var algorithm = new CAVIAAlgorithm<double, Matrix<double>, Vector<double>>(options);
        var task = CreateVectorTask(103);
        var batch = new TaskBatch<double, Matrix<double>, Vector<double>>(new[] { task });

        var loss = algorithm.MetaTrain(batch);
        var adapted = algorithm.Adapt(task);
        var predictions = adapted.Predict(task.QuerySetX);

        Assert.False(double.IsNaN(loss), "CAVIA MetaTrain returned NaN loss");
        Assert.NotNull(predictions);
        Assert.Equal(MetaLearningAlgorithmType.CAVIA, algorithm.AlgorithmType);
    }

    [Fact]
    public void WarpGrad_MetaTrainAndAdapt_Run()
    {
        var model = new LinearVectorModel(3);
        var options = new WarpGradOptions<double, Matrix<double>, Vector<double>>(model)
        {
            InnerLearningRate = 0.01,
            OuterLearningRate = 0.01,
            AdaptationSteps = 2,
            NumWarpLayers = 1,
            WarpLayerHiddenDim = 4,
            WarpLearningRate = 0.001,
            WarpInitScale = 0.01,
            WarpRegularization = 0.0001
        };

        var algorithm = new WarpGradAlgorithm<double, Matrix<double>, Vector<double>>(options);
        var task = CreateVectorTask(104);
        var batch = new TaskBatch<double, Matrix<double>, Vector<double>>(new[] { task });

        var loss = algorithm.MetaTrain(batch);
        var adapted = algorithm.Adapt(task);
        var predictions = adapted.Predict(task.QuerySetX);

        Assert.False(double.IsNaN(loss), "WarpGrad MetaTrain returned NaN loss");
        Assert.NotNull(predictions);
        Assert.Equal(MetaLearningAlgorithmType.WarpGrad, algorithm.AlgorithmType);
    }

    // ========================================================================
    // Group 3: Closed-Form / Metric Methods
    // ========================================================================

    [Fact]
    public void R2D2_MetaTrainAndAdapt_Run()
    {
        var model = new LinearVectorModel(3);
        var options = new R2D2Options<double, Matrix<double>, Vector<double>>(model)
        {
            OuterLearningRate = 0.01,
            Lambda = 1.0,
            LearnLambda = true,
            UseWoodburyIdentity = true
        };

        var algorithm = new R2D2Algorithm<double, Matrix<double>, Vector<double>>(options);
        var task = CreateVectorTask(105);
        var batch = new TaskBatch<double, Matrix<double>, Vector<double>>(new[] { task });

        var loss = algorithm.MetaTrain(batch);
        var adapted = algorithm.Adapt(task);
        var predictions = adapted.Predict(task.QuerySetX);

        Assert.False(double.IsNaN(loss), "R2D2 MetaTrain returned NaN loss");
        Assert.NotNull(predictions);
        Assert.Equal(MetaLearningAlgorithmType.R2D2, algorithm.AlgorithmType);
    }

    [Fact]
    public void MetaBaseline_MetaTrainAndAdapt_Run()
    {
        var model = new LinearVectorModel(3);
        var options = new MetaBaselineOptions<double, Matrix<double>, Vector<double>>(model)
        {
            OuterLearningRate = 0.01
        };

        var algorithm = new MetaBaselineAlgorithm<double, Matrix<double>, Vector<double>>(options);
        var task = CreateVectorTask(106);
        var batch = new TaskBatch<double, Matrix<double>, Vector<double>>(new[] { task });

        var loss = algorithm.MetaTrain(batch);
        var adapted = algorithm.Adapt(task);
        var predictions = adapted.Predict(task.QuerySetX);

        Assert.False(double.IsNaN(loss), "MetaBaseline MetaTrain returned NaN loss");
        Assert.NotNull(predictions);
        Assert.Equal(MetaLearningAlgorithmType.MetaBaseline, algorithm.AlgorithmType);
    }

    [Fact]
    public void SimpleShot_MetaTrainAndAdapt_Run()
    {
        var model = new LinearVectorModel(3);
        var options = new SimpleShotOptions<double, Matrix<double>, Vector<double>>(model)
        {
            OuterLearningRate = 0.01,
            NormalizationType = "CL2N",
            DistanceMetric = "cosine"
        };

        var algorithm = new SimpleShotAlgorithm<double, Matrix<double>, Vector<double>>(options);
        var task = CreateVectorTask(107);
        var batch = new TaskBatch<double, Matrix<double>, Vector<double>>(new[] { task });

        var loss = algorithm.MetaTrain(batch);
        var adapted = algorithm.Adapt(task);
        var predictions = adapted.Predict(task.QuerySetX);

        Assert.False(double.IsNaN(loss), "SimpleShot MetaTrain returned NaN loss");
        Assert.NotNull(predictions);
        Assert.Equal(MetaLearningAlgorithmType.SimpleShot, algorithm.AlgorithmType);
    }

    [Fact]
    public void LaplacianShot_MetaTrainAndAdapt_Run()
    {
        var model = new LinearVectorModel(3);
        var options = new LaplacianShotOptions<double, Matrix<double>, Vector<double>>(model)
        {
            OuterLearningRate = 0.01
        };

        var algorithm = new LaplacianShotAlgorithm<double, Matrix<double>, Vector<double>>(options);
        var task = CreateVectorTask(108);
        var batch = new TaskBatch<double, Matrix<double>, Vector<double>>(new[] { task });

        var loss = algorithm.MetaTrain(batch);
        var adapted = algorithm.Adapt(task);
        var predictions = adapted.Predict(task.QuerySetX);

        Assert.False(double.IsNaN(loss), "LaplacianShot MetaTrain returned NaN loss");
        Assert.NotNull(predictions);
        Assert.Equal(MetaLearningAlgorithmType.LaplacianShot, algorithm.AlgorithmType);
    }

    // ========================================================================
    // Group 4: Transductive / Inference-Time Methods
    // ========================================================================

    [Fact]
    public void TIM_MetaTrainAndAdapt_Run()
    {
        var model = new LinearVectorModel(3);
        var options = new TIMOptions<double, Matrix<double>, Vector<double>>(model)
        {
            OuterLearningRate = 0.01,
            TransductiveIterations = 5,
            ConditionalEntropyWeight = 1.0,
            MarginalEntropyWeight = 1.0,
            Temperature = 15.0
        };

        var algorithm = new TIMAlgorithm<double, Matrix<double>, Vector<double>>(options);
        var task = CreateVectorTask(109);
        var batch = new TaskBatch<double, Matrix<double>, Vector<double>>(new[] { task });

        var loss = algorithm.MetaTrain(batch);
        var adapted = algorithm.Adapt(task);
        var predictions = adapted.Predict(task.QuerySetX);

        Assert.False(double.IsNaN(loss), "TIM MetaTrain returned NaN loss");
        Assert.NotNull(predictions);
        Assert.Equal(MetaLearningAlgorithmType.TIM, algorithm.AlgorithmType);
    }

    [Fact]
    public void SIB_MetaTrainAndAdapt_Run()
    {
        var model = new LinearVectorModel(3);
        var options = new SIBOptions<double, Matrix<double>, Vector<double>>(model)
        {
            OuterLearningRate = 0.01
        };

        var algorithm = new SIBAlgorithm<double, Matrix<double>, Vector<double>>(options);
        var task = CreateVectorTask(110);
        var batch = new TaskBatch<double, Matrix<double>, Vector<double>>(new[] { task });

        var loss = algorithm.MetaTrain(batch);
        var adapted = algorithm.Adapt(task);
        var predictions = adapted.Predict(task.QuerySetX);

        Assert.False(double.IsNaN(loss), "SIB MetaTrain returned NaN loss");
        Assert.NotNull(predictions);
        Assert.Equal(MetaLearningAlgorithmType.SIB, algorithm.AlgorithmType);
    }

    [Fact]
    public void PTMAP_MetaTrainAndAdapt_Run()
    {
        var model = new LinearVectorModel(3);
        var options = new PTMAPOptions<double, Matrix<double>, Vector<double>>(model)
        {
            OuterLearningRate = 0.01
        };

        var algorithm = new PTMAPAlgorithm<double, Matrix<double>, Vector<double>>(options);
        var task = CreateVectorTask(111);
        var batch = new TaskBatch<double, Matrix<double>, Vector<double>>(new[] { task });

        var loss = algorithm.MetaTrain(batch);
        var adapted = algorithm.Adapt(task);
        var predictions = adapted.Predict(task.QuerySetX);

        Assert.False(double.IsNaN(loss), "PTMAP MetaTrain returned NaN loss");
        Assert.NotNull(predictions);
        Assert.Equal(MetaLearningAlgorithmType.PTMAP, algorithm.AlgorithmType);
    }

    // ========================================================================
    // Group 5: Amortized / Feed-Forward Methods
    // ========================================================================

    [Fact]
    public void VERSA_MetaTrainAndAdapt_Run()
    {
        var model = new LinearVectorModel(3);
        var options = new VERSAOptions<double, Matrix<double>, Vector<double>>(model)
        {
            OuterLearningRate = 0.01,
            AmortizationHiddenDim = 8,
            AmortizationNumLayers = 1
        };

        var algorithm = new VERSAAlgorithm<double, Matrix<double>, Vector<double>>(options);
        var task = CreateVectorTask(112);
        var batch = new TaskBatch<double, Matrix<double>, Vector<double>>(new[] { task });

        var loss = algorithm.MetaTrain(batch);
        var adapted = algorithm.Adapt(task);
        var predictions = adapted.Predict(task.QuerySetX);

        Assert.False(double.IsNaN(loss), "VERSA MetaTrain returned NaN loss");
        Assert.NotNull(predictions);
        Assert.Equal(MetaLearningAlgorithmType.VERSA, algorithm.AlgorithmType);
    }

    [Fact]
    public void SNAIL_MetaTrainAndAdapt_Run()
    {
        var model = new LinearVectorModel(3);
        var options = new SNAILOptions<double, Matrix<double>, Vector<double>>(model)
        {
            OuterLearningRate = 0.01,
            NumAttentionHeads = 1,
            AttentionKeyDim = 4,
            AttentionValueDim = 4,
            NumTCFilters = 4,
            NumBlocks = 1,
            MaxSequenceLength = 20,
            DropoutRate = 0.0
        };

        var algorithm = new SNAILAlgorithm<double, Matrix<double>, Vector<double>>(options);
        var task = CreateVectorTask(113);
        var batch = new TaskBatch<double, Matrix<double>, Vector<double>>(new[] { task });

        var loss = algorithm.MetaTrain(batch);
        var adapted = algorithm.Adapt(task);
        var predictions = adapted.Predict(task.QuerySetX);

        Assert.False(double.IsNaN(loss), "SNAIL MetaTrain returned NaN loss");
        Assert.NotNull(predictions);
        Assert.Equal(MetaLearningAlgorithmType.SNAIL, algorithm.AlgorithmType);
    }

    // ========================================================================
    // Group 6: Kernel / GP Methods
    // ========================================================================

    [Fact]
    public void DKT_MetaTrainAndAdapt_Run()
    {
        var model = new LinearVectorModel(3);
        var options = new DKTOptions<double, Matrix<double>, Vector<double>>(model)
        {
            OuterLearningRate = 0.01,
            KernelType = "rbf",
            KernelLengthScale = 1.0,
            NoiseVariance = 0.1
        };

        var algorithm = new DKTAlgorithm<double, Matrix<double>, Vector<double>>(options);
        var task = CreateVectorTask(114);
        var batch = new TaskBatch<double, Matrix<double>, Vector<double>>(new[] { task });

        var loss = algorithm.MetaTrain(batch);
        var adapted = algorithm.Adapt(task);
        var predictions = adapted.Predict(task.QuerySetX);

        Assert.False(double.IsNaN(loss), "DKT MetaTrain returned NaN loss");
        Assert.NotNull(predictions);
        Assert.Equal(MetaLearningAlgorithmType.DKT, algorithm.AlgorithmType);
    }

    // ========================================================================
    // Group 7: Feature Transformation Methods
    // ========================================================================

    [Fact]
    public void FEAT_MetaTrainAndAdapt_Run()
    {
        var model = new LinearVectorModel(3);
        var options = new FEATOptions<double, Matrix<double>, Vector<double>>(model)
        {
            OuterLearningRate = 0.01,
            NumTransformerHeads = 1,
            NumTransformerLayers = 1,
            ContrastiveWeight = 0.5,
            Temperature = 64.0
        };

        var algorithm = new FEATAlgorithm<double, Matrix<double>, Vector<double>>(options);
        var task = CreateVectorTask(115);
        var batch = new TaskBatch<double, Matrix<double>, Vector<double>>(new[] { task });

        var loss = algorithm.MetaTrain(batch);
        var adapted = algorithm.Adapt(task);
        var predictions = adapted.Predict(task.QuerySetX);

        Assert.False(double.IsNaN(loss), "FEAT MetaTrain returned NaN loss");
        Assert.NotNull(predictions);
        Assert.Equal(MetaLearningAlgorithmType.FEAT, algorithm.AlgorithmType);
    }

    [Fact]
    public void DeepEMD_MetaTrainAndAdapt_Run()
    {
        var model = new LinearVectorModel(3);
        var options = new DeepEMDOptions<double, Matrix<double>, Vector<double>>(model)
        {
            OuterLearningRate = 0.01,
            NumNodes = 3,
            SinkhornIterations = 5,
            SinkhornRegularization = 0.01,
            Temperature = 12.5
        };

        var algorithm = new DeepEMDAlgorithm<double, Matrix<double>, Vector<double>>(options);
        var task = CreateVectorTask(116);
        var batch = new TaskBatch<double, Matrix<double>, Vector<double>>(new[] { task });

        var loss = algorithm.MetaTrain(batch);
        var adapted = algorithm.Adapt(task);
        var predictions = adapted.Predict(task.QuerySetX);

        Assert.False(double.IsNaN(loss), "DeepEMD MetaTrain returned NaN loss");
        Assert.NotNull(predictions);
        Assert.Equal(MetaLearningAlgorithmType.DeepEMD, algorithm.AlgorithmType);
    }

    [Fact]
    public void FRN_MetaTrainAndAdapt_Run()
    {
        var model = new LinearVectorModel(3);
        var options = new FRNOptions<double, Matrix<double>, Vector<double>>(model)
        {
            OuterLearningRate = 0.01
        };

        var algorithm = new FRNAlgorithm<double, Matrix<double>, Vector<double>>(options);
        var task = CreateVectorTask(117);
        var batch = new TaskBatch<double, Matrix<double>, Vector<double>>(new[] { task });

        var loss = algorithm.MetaTrain(batch);
        var adapted = algorithm.Adapt(task);
        var predictions = adapted.Predict(task.QuerySetX);

        Assert.False(double.IsNaN(loss), "FRN MetaTrain returned NaN loss");
        Assert.NotNull(predictions);
        Assert.Equal(MetaLearningAlgorithmType.FRN, algorithm.AlgorithmType);
    }

    // ========================================================================
    // Group 8: Graph / Distribution Methods
    // ========================================================================

    [Fact]
    public void DPGN_MetaTrainAndAdapt_Run()
    {
        var model = new LinearVectorModel(3);
        var options = new DPGNOptions<double, Matrix<double>, Vector<double>>(model)
        {
            OuterLearningRate = 0.01
        };

        var algorithm = new DPGNAlgorithm<double, Matrix<double>, Vector<double>>(options);
        var task = CreateVectorTask(118);
        var batch = new TaskBatch<double, Matrix<double>, Vector<double>>(new[] { task });

        var loss = algorithm.MetaTrain(batch);
        var adapted = algorithm.Adapt(task);
        var predictions = adapted.Predict(task.QuerySetX);

        Assert.False(double.IsNaN(loss), "DPGN MetaTrain returned NaN loss");
        Assert.NotNull(predictions);
        Assert.Equal(MetaLearningAlgorithmType.DPGN, algorithm.AlgorithmType);
    }

    [Fact]
    public void ConstellationNet_MetaTrainAndAdapt_Run()
    {
        var model = new LinearVectorModel(3);
        var options = new ConstellationNetOptions<double, Matrix<double>, Vector<double>>(model)
        {
            OuterLearningRate = 0.01
        };

        var algorithm = new ConstellationNetAlgorithm<double, Matrix<double>, Vector<double>>(options);
        var task = CreateVectorTask(119);
        var batch = new TaskBatch<double, Matrix<double>, Vector<double>>(new[] { task });

        var loss = algorithm.MetaTrain(batch);
        var adapted = algorithm.Adapt(task);
        var predictions = adapted.Predict(task.QuerySetX);

        Assert.False(double.IsNaN(loss), "ConstellationNet MetaTrain returned NaN loss");
        Assert.NotNull(predictions);
        Assert.Equal(MetaLearningAlgorithmType.ConstellationNet, algorithm.AlgorithmType);
    }

    // ========================================================================
    // Group 9: Set / Embedding Methods
    // ========================================================================

    [Fact]
    public void SetFeat_MetaTrainAndAdapt_Run()
    {
        var model = new LinearVectorModel(3);
        var options = new SetFeatOptions<double, Matrix<double>, Vector<double>>(model)
        {
            OuterLearningRate = 0.01
        };

        var algorithm = new SetFeatAlgorithm<double, Matrix<double>, Vector<double>>(options);
        var task = CreateVectorTask(120);
        var batch = new TaskBatch<double, Matrix<double>, Vector<double>>(new[] { task });

        var loss = algorithm.MetaTrain(batch);
        var adapted = algorithm.Adapt(task);
        var predictions = adapted.Predict(task.QuerySetX);

        Assert.False(double.IsNaN(loss), "SetFeat MetaTrain returned NaN loss");
        Assert.NotNull(predictions);
        Assert.Equal(MetaLearningAlgorithmType.SetFeat, algorithm.AlgorithmType);
    }

    [Fact]
    public void FewTURE_MetaTrainAndAdapt_Run()
    {
        var model = new LinearVectorModel(3);
        var options = new FewTUREOptions<double, Matrix<double>, Vector<double>>(model)
        {
            OuterLearningRate = 0.01
        };

        var algorithm = new FewTUREAlgorithm<double, Matrix<double>, Vector<double>>(options);
        var task = CreateVectorTask(121);
        var batch = new TaskBatch<double, Matrix<double>, Vector<double>>(new[] { task });

        var loss = algorithm.MetaTrain(batch);
        var adapted = algorithm.Adapt(task);
        var predictions = adapted.Predict(task.QuerySetX);

        Assert.False(double.IsNaN(loss), "FewTURE MetaTrain returned NaN loss");
        Assert.NotNull(predictions);
        Assert.Equal(MetaLearningAlgorithmType.FewTURE, algorithm.AlgorithmType);
    }

    [Fact]
    public void HyperShot_MetaTrainAndAdapt_Run()
    {
        var model = new LinearVectorModel(3);
        var options = new HyperShotOptions<double, Matrix<double>, Vector<double>>(model)
        {
            OuterLearningRate = 0.01
        };

        var algorithm = new HyperShotAlgorithm<double, Matrix<double>, Vector<double>>(options);
        var task = CreateVectorTask(122);
        var batch = new TaskBatch<double, Matrix<double>, Vector<double>>(new[] { task });

        var loss = algorithm.MetaTrain(batch);
        var adapted = algorithm.Adapt(task);
        var predictions = adapted.Predict(task.QuerySetX);

        Assert.False(double.IsNaN(loss), "HyperShot MetaTrain returned NaN loss");
        Assert.NotNull(predictions);
        Assert.Equal(MetaLearningAlgorithmType.HyperShot, algorithm.AlgorithmType);
    }

    // ========================================================================
    // Group 10: Embedding Propagation / Bayesian Methods
    // ========================================================================

    [Fact]
    public void EPNet_MetaTrainAndAdapt_Run()
    {
        var model = new LinearVectorModel(3);
        var options = new EPNetOptions<double, Matrix<double>, Vector<double>>(model)
        {
            OuterLearningRate = 0.01
        };

        var algorithm = new EPNetAlgorithm<double, Matrix<double>, Vector<double>>(options);
        var task = CreateVectorTask(123);
        var batch = new TaskBatch<double, Matrix<double>, Vector<double>>(new[] { task });

        var loss = algorithm.MetaTrain(batch);
        var adapted = algorithm.Adapt(task);
        var predictions = adapted.Predict(task.QuerySetX);

        Assert.False(double.IsNaN(loss), "EPNet MetaTrain returned NaN loss");
        Assert.NotNull(predictions);
        Assert.Equal(MetaLearningAlgorithmType.EPNet, algorithm.AlgorithmType);
    }

    [Fact]
    public void NPBML_MetaTrainAndAdapt_Run()
    {
        var model = new LinearVectorModel(3);
        var options = new NPBMLOptions<double, Matrix<double>, Vector<double>>(model)
        {
            OuterLearningRate = 0.01
        };

        var algorithm = new NPBMLAlgorithm<double, Matrix<double>, Vector<double>>(options);
        var task = CreateVectorTask(124);
        var batch = new TaskBatch<double, Matrix<double>, Vector<double>>(new[] { task });

        var loss = algorithm.MetaTrain(batch);
        var adapted = algorithm.Adapt(task);
        var predictions = adapted.Predict(task.QuerySetX);

        Assert.False(double.IsNaN(loss), "NPBML MetaTrain returned NaN loss");
        Assert.NotNull(predictions);
        Assert.Equal(MetaLearningAlgorithmType.NPBML, algorithm.AlgorithmType);
    }

    // ========================================================================
    // Group 11: Contrastive / Pre-trained Methods
    // ========================================================================

    [Fact]
    public void MCL_MetaTrainAndAdapt_Run()
    {
        var model = new LinearVectorModel(3);
        var options = new MCLOptions<double, Matrix<double>, Vector<double>>(model)
        {
            OuterLearningRate = 0.01
        };

        var algorithm = new MCLAlgorithm<double, Matrix<double>, Vector<double>>(options);
        var task = CreateVectorTask(125);
        var batch = new TaskBatch<double, Matrix<double>, Vector<double>>(new[] { task });

        var loss = algorithm.MetaTrain(batch);
        var adapted = algorithm.Adapt(task);
        var predictions = adapted.Predict(task.QuerySetX);

        Assert.False(double.IsNaN(loss), "MCL MetaTrain returned NaN loss");
        Assert.NotNull(predictions);
        Assert.Equal(MetaLearningAlgorithmType.MCL, algorithm.AlgorithmType);
    }

    [Fact]
    public void PMF_MetaTrainAndAdapt_Run()
    {
        var model = new LinearVectorModel(3);
        var options = new PMFOptions<double, Matrix<double>, Vector<double>>(model)
        {
            OuterLearningRate = 0.01
        };

        var algorithm = new PMFAlgorithm<double, Matrix<double>, Vector<double>>(options);
        var task = CreateVectorTask(126);
        var batch = new TaskBatch<double, Matrix<double>, Vector<double>>(new[] { task });

        var loss = algorithm.MetaTrain(batch);
        var adapted = algorithm.Adapt(task);
        var predictions = adapted.Predict(task.QuerySetX);

        Assert.False(double.IsNaN(loss), "PMF MetaTrain returned NaN loss");
        Assert.NotNull(predictions);
        Assert.Equal(MetaLearningAlgorithmType.PMF, algorithm.AlgorithmType);
    }

    [Fact]
    public void CAML_MetaTrainAndAdapt_Run()
    {
        var model = new LinearVectorModel(3);
        var options = new CAMLOptions<double, Matrix<double>, Vector<double>>(model)
        {
            OuterLearningRate = 0.01
        };

        var algorithm = new CAMLAlgorithm<double, Matrix<double>, Vector<double>>(options);
        var task = CreateVectorTask(127);
        var batch = new TaskBatch<double, Matrix<double>, Vector<double>>(new[] { task });

        var loss = algorithm.MetaTrain(batch);
        var adapted = algorithm.Adapt(task);
        var predictions = adapted.Predict(task.QuerySetX);

        Assert.False(double.IsNaN(loss), "CAML MetaTrain returned NaN loss");
        Assert.NotNull(predictions);
        Assert.Equal(MetaLearningAlgorithmType.CAML, algorithm.AlgorithmType);
    }

    // ========================================================================
    // Group 12: Multi-task Batch Tests (multiple tasks per batch)
    // ========================================================================

    [Fact]
    public void MAMLPlusPlus_MultipleTasks_Run()
    {
        var model = new LinearVectorModel(3);
        var options = new MAMLPlusPlusOptions<double, Matrix<double>, Vector<double>>(model)
        {
            InnerLearningRate = 0.01,
            OuterLearningRate = 0.01,
            AdaptationSteps = 2,
            UseMultiStepLoss = true,
            UsePerStepLearningRates = true,
            UseDerivativeOrderAnnealing = false
        };

        var algorithm = new MAMLPlusPlusAlgorithm<double, Matrix<double>, Vector<double>>(options);
        var taskA = CreateVectorTask(200);
        var taskB = CreateVectorTask(201);
        var batch = new TaskBatch<double, Matrix<double>, Vector<double>>(new[] { taskA, taskB });

        var loss = algorithm.MetaTrain(batch);

        Assert.False(double.IsNaN(loss), "MAMLPlusPlus multi-task MetaTrain returned NaN loss");
    }

    [Fact]
    public void WarpGrad_MultipleTasks_Run()
    {
        var model = new LinearVectorModel(3);
        var options = new WarpGradOptions<double, Matrix<double>, Vector<double>>(model)
        {
            InnerLearningRate = 0.01,
            OuterLearningRate = 0.01,
            AdaptationSteps = 2,
            NumWarpLayers = 1,
            WarpLayerHiddenDim = 4
        };

        var algorithm = new WarpGradAlgorithm<double, Matrix<double>, Vector<double>>(options);
        var taskA = CreateVectorTask(202);
        var taskB = CreateVectorTask(203);
        var batch = new TaskBatch<double, Matrix<double>, Vector<double>>(new[] { taskA, taskB });

        var loss = algorithm.MetaTrain(batch);

        Assert.False(double.IsNaN(loss), "WarpGrad multi-task MetaTrain returned NaN loss");
    }

    [Fact]
    public void CAVIA_MultipleTasks_Run()
    {
        var model = new LinearVectorModel(3);
        var options = new CAVIAOptions<double, Matrix<double>, Vector<double>>(model)
        {
            ContextDimension = 4,
            AdaptationSteps = 2,
            InnerLearningRate = 0.01,
            OuterLearningRate = 0.01
        };

        var algorithm = new CAVIAAlgorithm<double, Matrix<double>, Vector<double>>(options);
        var taskA = CreateVectorTask(204);
        var taskB = CreateVectorTask(205);
        var batch = new TaskBatch<double, Matrix<double>, Vector<double>>(new[] { taskA, taskB });

        var loss = algorithm.MetaTrain(batch);

        Assert.False(double.IsNaN(loss), "CAVIA multi-task MetaTrain returned NaN loss");
    }

    [Fact]
    public void DeepEMD_MultipleTasks_Run()
    {
        var model = new LinearVectorModel(3);
        var options = new DeepEMDOptions<double, Matrix<double>, Vector<double>>(model)
        {
            OuterLearningRate = 0.01,
            NumNodes = 3,
            SinkhornIterations = 3,
            SinkhornRegularization = 0.01,
            Temperature = 12.5
        };

        var algorithm = new DeepEMDAlgorithm<double, Matrix<double>, Vector<double>>(options);
        var taskA = CreateVectorTask(206);
        var taskB = CreateVectorTask(207);
        var batch = new TaskBatch<double, Matrix<double>, Vector<double>>(new[] { taskA, taskB });

        var loss = algorithm.MetaTrain(batch);

        Assert.False(double.IsNaN(loss), "DeepEMD multi-task MetaTrain returned NaN loss");
    }

    // ========================================================================
    // Group 13: Parameter change verification (training actually updates params)
    // ========================================================================

    [Fact]
    public void CAVIA_MetaTrain_UpdatesParameters()
    {
        var model = new LinearVectorModel(3);
        var options = new CAVIAOptions<double, Matrix<double>, Vector<double>>(model)
        {
            ContextDimension = 4,
            AdaptationSteps = 2,
            InnerLearningRate = 0.01,
            OuterLearningRate = 0.01
        };

        var algorithm = new CAVIAAlgorithm<double, Matrix<double>, Vector<double>>(options);
        var initial = model.GetParameters().ToArray();

        var task = CreateVectorTask(300);
        var batch = new TaskBatch<double, Matrix<double>, Vector<double>>(new[] { task });
        algorithm.MetaTrain(batch);

        var updated = model.GetParameters().ToArray();
        bool changed = false;
        for (int i = 0; i < updated.Length; i++)
        {
            if (Math.Abs(updated[i] - initial[i]) > 1e-12)
            {
                changed = true;
                break;
            }
        }
        Assert.True(changed, "CAVIA MetaTrain did not update model parameters");
    }

    [Fact]
    public void R2D2_MetaTrain_UpdatesParameters()
    {
        var model = new LinearVectorModel(3);
        var options = new R2D2Options<double, Matrix<double>, Vector<double>>(model)
        {
            OuterLearningRate = 0.01,
            Lambda = 1.0,
            LearnLambda = false
        };

        var algorithm = new R2D2Algorithm<double, Matrix<double>, Vector<double>>(options);
        var initial = model.GetParameters().ToArray();

        var task = CreateVectorTask(301);
        var batch = new TaskBatch<double, Matrix<double>, Vector<double>>(new[] { task });
        algorithm.MetaTrain(batch);

        var updated = model.GetParameters().ToArray();
        bool changed = false;
        for (int i = 0; i < updated.Length; i++)
        {
            if (Math.Abs(updated[i] - initial[i]) > 1e-12)
            {
                changed = true;
                break;
            }
        }
        Assert.True(changed, "R2D2 MetaTrain did not update model parameters");
    }

    [Fact]
    public void WarpGrad_MetaTrain_UpdatesParameters()
    {
        var model = new LinearVectorModel(3);
        var options = new WarpGradOptions<double, Matrix<double>, Vector<double>>(model)
        {
            InnerLearningRate = 0.01,
            OuterLearningRate = 0.01,
            AdaptationSteps = 2,
            NumWarpLayers = 1,
            WarpLayerHiddenDim = 4
        };

        var algorithm = new WarpGradAlgorithm<double, Matrix<double>, Vector<double>>(options);
        var initial = model.GetParameters().ToArray();

        var task = CreateVectorTask(302);
        var batch = new TaskBatch<double, Matrix<double>, Vector<double>>(new[] { task });
        algorithm.MetaTrain(batch);

        var updated = model.GetParameters().ToArray();
        bool changed = false;
        for (int i = 0; i < updated.Length; i++)
        {
            if (Math.Abs(updated[i] - initial[i]) > 1e-12)
            {
                changed = true;
                break;
            }
        }
        Assert.True(changed, "WarpGrad MetaTrain did not update model parameters");
    }

    [Fact]
    public void MAMLPlusPlus_MetaTrain_UpdatesParameters()
    {
        var model = new LinearVectorModel(3);
        var options = new MAMLPlusPlusOptions<double, Matrix<double>, Vector<double>>(model)
        {
            InnerLearningRate = 0.01,
            OuterLearningRate = 0.01,
            AdaptationSteps = 2,
            UseMultiStepLoss = true,
            UsePerStepLearningRates = false,
            UseDerivativeOrderAnnealing = false
        };

        var algorithm = new MAMLPlusPlusAlgorithm<double, Matrix<double>, Vector<double>>(options);
        var initial = model.GetParameters().ToArray();

        var task = CreateVectorTask(303);
        var batch = new TaskBatch<double, Matrix<double>, Vector<double>>(new[] { task });
        algorithm.MetaTrain(batch);

        var updated = model.GetParameters().ToArray();
        bool changed = false;
        for (int i = 0; i < updated.Length; i++)
        {
            if (Math.Abs(updated[i] - initial[i]) > 1e-12)
            {
                changed = true;
                break;
            }
        }
        Assert.True(changed, "MAMLPlusPlus MetaTrain did not update model parameters");
    }

    // ========================================================================
    // Group 14: Adapted model interface verification (IAdaptedMetaModel)
    // ========================================================================

    [Fact]
    public void DeepEMD_AdaptedModel_ImplementsIAdaptedMetaModel()
    {
        var model = new LinearVectorModel(3);
        var options = new DeepEMDOptions<double, Matrix<double>, Vector<double>>(model)
        {
            OuterLearningRate = 0.01,
            NumNodes = 3,
            SinkhornIterations = 3,
            Temperature = 12.5
        };

        var algorithm = new DeepEMDAlgorithm<double, Matrix<double>, Vector<double>>(options);
        var task = CreateVectorTask(400);
        var adapted = algorithm.Adapt(task);

        Assert.NotNull(adapted);
        var adaptedMeta = adapted as IAdaptedMetaModel<double>;
        Assert.NotNull(adaptedMeta);
        // Verify the interface properties are accessible
        var features = adaptedMeta.AdaptedSupportFeatures;
        var modulation = adaptedMeta.ParameterModulationFactors;

        var predictions = adapted.Predict(task.QuerySetX);
        Assert.NotNull(predictions);
    }

    [Fact]
    public void VERSA_AdaptedModel_ImplementsIAdaptedMetaModel()
    {
        var model = new LinearVectorModel(3);
        var options = new VERSAOptions<double, Matrix<double>, Vector<double>>(model)
        {
            OuterLearningRate = 0.01,
            AmortizationHiddenDim = 8,
            AmortizationNumLayers = 1
        };

        var algorithm = new VERSAAlgorithm<double, Matrix<double>, Vector<double>>(options);
        var task = CreateVectorTask(401);
        var adapted = algorithm.Adapt(task);

        Assert.NotNull(adapted);
        var adaptedMeta = adapted as IAdaptedMetaModel<double>;
        Assert.NotNull(adaptedMeta);
        var features = adaptedMeta.AdaptedSupportFeatures;
        var modulation = adaptedMeta.ParameterModulationFactors;

        var predictions = adapted.Predict(task.QuerySetX);
        Assert.NotNull(predictions);
    }

    [Fact]
    public void FEAT_AdaptedModel_ImplementsIAdaptedMetaModel()
    {
        var model = new LinearVectorModel(3);
        var options = new FEATOptions<double, Matrix<double>, Vector<double>>(model)
        {
            OuterLearningRate = 0.01,
            NumTransformerHeads = 1,
            NumTransformerLayers = 1,
            Temperature = 64.0
        };

        var algorithm = new FEATAlgorithm<double, Matrix<double>, Vector<double>>(options);
        var task = CreateVectorTask(402);
        var adapted = algorithm.Adapt(task);

        Assert.NotNull(adapted);
        var adaptedMeta = adapted as IAdaptedMetaModel<double>;
        Assert.NotNull(adaptedMeta);
        var features = adaptedMeta.AdaptedSupportFeatures;
        var modulation = adaptedMeta.ParameterModulationFactors;

        var predictions = adapted.Predict(task.QuerySetX);
        Assert.NotNull(predictions);
    }

    // ========================================================================
    // Group 15: Consecutive MetaTrain calls (stability test)
    // ========================================================================

    [Fact]
    public void R2D2_ConsecutiveMetaTrain_StaysFinite()
    {
        var model = new LinearVectorModel(3);
        var options = new R2D2Options<double, Matrix<double>, Vector<double>>(model)
        {
            OuterLearningRate = 0.001,
            Lambda = 1.0,
            LearnLambda = false
        };

        var algorithm = new R2D2Algorithm<double, Matrix<double>, Vector<double>>(options);

        for (int i = 0; i < 5; i++)
        {
            var task = CreateVectorTask(500 + i);
            var batch = new TaskBatch<double, Matrix<double>, Vector<double>>(new[] { task });
            var loss = algorithm.MetaTrain(batch);
            Assert.False(double.IsNaN(loss), $"R2D2 MetaTrain returned NaN at iteration {i}");
            Assert.False(double.IsInfinity(loss), $"R2D2 MetaTrain returned Infinity at iteration {i}");
        }
    }

    [Fact]
    public void VERSA_ConsecutiveMetaTrain_StaysFinite()
    {
        var model = new LinearVectorModel(3);
        var options = new VERSAOptions<double, Matrix<double>, Vector<double>>(model)
        {
            OuterLearningRate = 0.001,
            AmortizationHiddenDim = 8,
            AmortizationNumLayers = 1
        };

        var algorithm = new VERSAAlgorithm<double, Matrix<double>, Vector<double>>(options);

        for (int i = 0; i < 5; i++)
        {
            var task = CreateVectorTask(510 + i);
            var batch = new TaskBatch<double, Matrix<double>, Vector<double>>(new[] { task });
            var loss = algorithm.MetaTrain(batch);
            Assert.False(double.IsNaN(loss), $"VERSA MetaTrain returned NaN at iteration {i}");
            Assert.False(double.IsInfinity(loss), $"VERSA MetaTrain returned Infinity at iteration {i}");
        }
    }

    [Fact]
    public void SNAIL_ConsecutiveMetaTrain_StaysFinite()
    {
        var model = new LinearVectorModel(3);
        var options = new SNAILOptions<double, Matrix<double>, Vector<double>>(model)
        {
            OuterLearningRate = 0.001,
            NumAttentionHeads = 1,
            AttentionKeyDim = 4,
            AttentionValueDim = 4,
            NumTCFilters = 4,
            NumBlocks = 1,
            MaxSequenceLength = 20,
            DropoutRate = 0.0
        };

        var algorithm = new SNAILAlgorithm<double, Matrix<double>, Vector<double>>(options);

        for (int i = 0; i < 5; i++)
        {
            var task = CreateVectorTask(520 + i);
            var batch = new TaskBatch<double, Matrix<double>, Vector<double>>(new[] { task });
            var loss = algorithm.MetaTrain(batch);
            Assert.False(double.IsNaN(loss), $"SNAIL MetaTrain returned NaN at iteration {i}");
            Assert.False(double.IsInfinity(loss), $"SNAIL MetaTrain returned Infinity at iteration {i}");
        }
    }

    [Fact]
    public void DKT_ConsecutiveMetaTrain_StaysFinite()
    {
        var model = new LinearVectorModel(3);
        var options = new DKTOptions<double, Matrix<double>, Vector<double>>(model)
        {
            OuterLearningRate = 0.001,
            KernelType = "rbf",
            KernelLengthScale = 1.0,
            NoiseVariance = 0.1
        };

        var algorithm = new DKTAlgorithm<double, Matrix<double>, Vector<double>>(options);

        for (int i = 0; i < 5; i++)
        {
            var task = CreateVectorTask(530 + i);
            var batch = new TaskBatch<double, Matrix<double>, Vector<double>>(new[] { task });
            var loss = algorithm.MetaTrain(batch);
            Assert.False(double.IsNaN(loss), $"DKT MetaTrain returned NaN at iteration {i}");
            Assert.False(double.IsInfinity(loss), $"DKT MetaTrain returned Infinity at iteration {i}");
        }
    }

    [Fact]
    public void FEAT_ConsecutiveMetaTrain_StaysFinite()
    {
        var model = new LinearVectorModel(3);
        var options = new FEATOptions<double, Matrix<double>, Vector<double>>(model)
        {
            OuterLearningRate = 0.001,
            NumTransformerHeads = 1,
            NumTransformerLayers = 1,
            Temperature = 64.0
        };

        var algorithm = new FEATAlgorithm<double, Matrix<double>, Vector<double>>(options);

        for (int i = 0; i < 5; i++)
        {
            var task = CreateVectorTask(540 + i);
            var batch = new TaskBatch<double, Matrix<double>, Vector<double>>(new[] { task });
            var loss = algorithm.MetaTrain(batch);
            Assert.False(double.IsNaN(loss), $"FEAT MetaTrain returned NaN at iteration {i}");
            Assert.False(double.IsInfinity(loss), $"FEAT MetaTrain returned Infinity at iteration {i}");
        }
    }
}
