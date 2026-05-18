using AiDotNet.Data.Loaders;
using AiDotNet.Interfaces;
using AiDotNet.Models.Options;
using AiDotNet.Regression;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.Configuration;

/// <summary>
/// Regression test for AiDotNet#1361 — <c>ConfigureTrainingPipeline</c>
/// was stored but never consumed by any Build path. Calling
/// <c>ConfigureTrainingPipeline(...)</c> followed by <c>BuildAsync</c>
/// previously had no observable effect; the stages were stored on the
/// builder but never executed.
///
/// <para>The fix executes <c>Stages</c> sequentially right after
/// <c>ConfigureFineTuning</c> runs (and after main training completes),
/// before metric finalization. Each enabled stage's
/// <c>CustomTrainingFunction</c> is invoked with the previous stage's
/// output model and that stage's <c>TrainingData</c>; the returned
/// model feeds the next stage. The final stage's output replaces
/// <c>optimizationResult.BestSolution</c>.</para>
///
/// <para>StageType / FineTuningMethod auto-dispatch (e.g. "build an SFT
/// trainer from FineTuningMethodType.SFT and run it") is documented as
/// not-yet-implemented — until that factory lands, each enabled stage
/// must provide its own <c>CustomTrainingFunction</c> delegate or it
/// throws a clearly-worded <c>InvalidOperationException</c>.</para>
/// </summary>
public class ConfigureTrainingPipelineWiringTests
{
    private static (Matrix<double> x, Vector<double> y) BuildDataset(int rows = 20, int features = 3)
    {
        var rng = new Random(123);
        var xData = new double[rows, features];
        var yData = new double[rows];
        for (int r = 0; r < rows; r++)
        {
            double sum = 0;
            for (int c = 0; c < features; c++)
            {
                xData[r, c] = rng.NextDouble() * 2 - 1;
                sum += xData[r, c];
            }
            yData[r] = sum;
        }
        return (new Matrix<double>(xData), new Vector<double>(yData));
    }

    private static FineTuningData<double, Matrix<double>, Vector<double>> BuildSFTData()
    {
        var (x, y) = BuildDataset(rows: 4);
        return new FineTuningData<double, Matrix<double>, Vector<double>>
        {
            Inputs = new[] { x },
            Outputs = new[] { y }
        };
    }

    [Fact(Timeout = 120000)]
    public async Task ConfigureTrainingPipeline_ExecutesEnabledStagesInOrder()
    {
        var (x, y) = BuildDataset();
        var executionLog = new List<string>();

        var stage1 = new TrainingStage<double, Matrix<double>, Vector<double>>
        {
            Name = "stage-1",
            Enabled = true,
            TrainingData = BuildSFTData(),
            CustomTrainingFunction = (model, data, ct) =>
            {
                executionLog.Add("stage-1");
                return Task.FromResult(model);
            }
        };
        var stage2 = new TrainingStage<double, Matrix<double>, Vector<double>>
        {
            Name = "stage-2",
            Enabled = true,
            TrainingData = BuildSFTData(),
            CustomTrainingFunction = (model, data, ct) =>
            {
                executionLog.Add("stage-2");
                return Task.FromResult(model);
            }
        };
        var stage3 = new TrainingStage<double, Matrix<double>, Vector<double>>
        {
            Name = "stage-3",
            Enabled = true,
            TrainingData = BuildSFTData(),
            CustomTrainingFunction = (model, data, ct) =>
            {
                executionLog.Add("stage-3");
                return Task.FromResult(model);
            }
        };

        var pipeline = new TrainingPipelineConfiguration<double, Matrix<double>, Vector<double>>
        {
            Name = "test-pipeline",
            Stages = new List<TrainingStage<double, Matrix<double>, Vector<double>>>
            {
                stage1, stage2, stage3
            }
        };

        await new AiModelBuilder<double, Matrix<double>, Vector<double>>()
            .ConfigureDataLoader(DataLoaders.FromMatrixVector(x, y))
            .ConfigureModel(new RidgeRegression<double>())
            .ConfigureTrainingPipeline(pipeline)
            .BuildAsync();

        Assert.Equal(new[] { "stage-1", "stage-2", "stage-3" }, executionLog);
    }

    [Fact(Timeout = 120000)]
    public async Task ConfigureTrainingPipeline_StageOutputFeedsNextStage()
    {
        var (x, y) = BuildDataset();
        IFullModel<double, Matrix<double>, Vector<double>>? observedAtStage2 = null;
        IFullModel<double, Matrix<double>, Vector<double>>? returnedFromStage1 = null;

        var stage1 = new TrainingStage<double, Matrix<double>, Vector<double>>
        {
            Name = "stage-1-replace",
            Enabled = true,
            TrainingData = BuildSFTData(),
            CustomTrainingFunction = (model, data, ct) =>
            {
                // Return a NEW model instance to verify it flows to stage 2.
                var replacement = new RidgeRegression<double>();
                returnedFromStage1 = replacement;
                return Task.FromResult<IFullModel<double, Matrix<double>, Vector<double>>>(replacement);
            }
        };
        var stage2 = new TrainingStage<double, Matrix<double>, Vector<double>>
        {
            Name = "stage-2-observe",
            Enabled = true,
            TrainingData = BuildSFTData(),
            CustomTrainingFunction = (model, data, ct) =>
            {
                observedAtStage2 = model;
                return Task.FromResult(model);
            }
        };

        await new AiModelBuilder<double, Matrix<double>, Vector<double>>()
            .ConfigureDataLoader(DataLoaders.FromMatrixVector(x, y))
            .ConfigureModel(new RidgeRegression<double>())
            .ConfigureTrainingPipeline(new TrainingPipelineConfiguration<double, Matrix<double>, Vector<double>>
            {
                Stages = new() { stage1, stage2 }
            })
            .BuildAsync();

        Assert.NotNull(returnedFromStage1);
        Assert.Same(returnedFromStage1, observedAtStage2);
    }

    [Fact(Timeout = 120000)]
    public async Task ConfigureTrainingPipeline_DisabledStages_AreSkipped()
    {
        var (x, y) = BuildDataset();
        var executionLog = new List<string>();

        TrainingStage<double, Matrix<double>, Vector<double>> Stage(string name, bool enabled)
            => new()
            {
                Name = name,
                Enabled = enabled,
                TrainingData = BuildSFTData(),
                CustomTrainingFunction = (m, d, ct) =>
                {
                    executionLog.Add(name);
                    return Task.FromResult(m);
                }
            };

        await new AiModelBuilder<double, Matrix<double>, Vector<double>>()
            .ConfigureDataLoader(DataLoaders.FromMatrixVector(x, y))
            .ConfigureModel(new RidgeRegression<double>())
            .ConfigureTrainingPipeline(new TrainingPipelineConfiguration<double, Matrix<double>, Vector<double>>
            {
                Stages = new()
                {
                    Stage("on-1", enabled: true),
                    Stage("OFF",   enabled: false),
                    Stage("on-2", enabled: true),
                }
            })
            .BuildAsync();

        Assert.Equal(new[] { "on-1", "on-2" }, executionLog);
    }

    [Fact(Timeout = 120000)]
    public async Task ConfigureTrainingPipeline_RunConditionFalse_SkipsStage()
    {
        var (x, y) = BuildDataset();
        var executionLog = new List<string>();

        var conditionalStage = new TrainingStage<double, Matrix<double>, Vector<double>>
        {
            Name = "conditional",
            Enabled = true,
            TrainingData = BuildSFTData(),
            RunCondition = _ => false, // never runs
            CustomTrainingFunction = (m, d, ct) =>
            {
                executionLog.Add("conditional");
                return Task.FromResult(m);
            }
        };
        var afterStage = new TrainingStage<double, Matrix<double>, Vector<double>>
        {
            Name = "after",
            Enabled = true,
            TrainingData = BuildSFTData(),
            CustomTrainingFunction = (m, d, ct) =>
            {
                executionLog.Add("after");
                return Task.FromResult(m);
            }
        };

        await new AiModelBuilder<double, Matrix<double>, Vector<double>>()
            .ConfigureDataLoader(DataLoaders.FromMatrixVector(x, y))
            .ConfigureModel(new RidgeRegression<double>())
            .ConfigureTrainingPipeline(new TrainingPipelineConfiguration<double, Matrix<double>, Vector<double>>
            {
                Stages = new() { conditionalStage, afterStage }
            })
            .BuildAsync();

        Assert.Equal(new[] { "after" }, executionLog);
    }

    [Fact(Timeout = 60000)]
    public async Task ConfigureTrainingPipeline_StageWithoutCustomFunction_Throws()
    {
        var (x, y) = BuildDataset();
        var stage = new TrainingStage<double, Matrix<double>, Vector<double>>
        {
            Name = "unwired-stage",
            Enabled = true,
            TrainingData = BuildSFTData(),
            CustomTrainingFunction = null, // explicitly missing
        };

        var builder = new AiModelBuilder<double, Matrix<double>, Vector<double>>()
            .ConfigureDataLoader(DataLoaders.FromMatrixVector(x, y))
            .ConfigureModel(new RidgeRegression<double>())
            .ConfigureTrainingPipeline(new TrainingPipelineConfiguration<double, Matrix<double>, Vector<double>>
            {
                Stages = new() { stage }
            });

        var ex = await Assert.ThrowsAsync<InvalidOperationException>(
            async () => await builder.BuildAsync());
        Assert.Contains("CustomTrainingFunction", ex.Message);
    }

    [Fact(Timeout = 60000)]
    public async Task ConfigureTrainingPipeline_StageWithoutTrainingData_Throws()
    {
        var (x, y) = BuildDataset();
        var stage = new TrainingStage<double, Matrix<double>, Vector<double>>
        {
            Name = "no-data-stage",
            Enabled = true,
            TrainingData = null,
            CustomTrainingFunction = (m, d, ct) => Task.FromResult(m),
        };

        var builder = new AiModelBuilder<double, Matrix<double>, Vector<double>>()
            .ConfigureDataLoader(DataLoaders.FromMatrixVector(x, y))
            .ConfigureModel(new RidgeRegression<double>())
            .ConfigureTrainingPipeline(new TrainingPipelineConfiguration<double, Matrix<double>, Vector<double>>
            {
                Stages = new() { stage }
            });

        var ex = await Assert.ThrowsAsync<InvalidOperationException>(
            async () => await builder.BuildAsync());
        Assert.Contains("TrainingData", ex.Message);
    }

    [Fact(Timeout = 120000)]
    public async Task ConfigureTrainingPipeline_EvaluationOnlyStage_NoData_AllowedAndRuns()
    {
        var (x, y) = BuildDataset();
        bool stageRan = false;

        var stage = new TrainingStage<double, Matrix<double>, Vector<double>>
        {
            Name = "eval-only",
            Enabled = true,
            IsEvaluationOnly = true,
            TrainingData = null, // allowed when IsEvaluationOnly = true
            CustomTrainingFunction = (m, d, ct) =>
            {
                stageRan = true; // For IsEvaluationOnly stages the training fn is skipped per contract.
                return Task.FromResult(m);
            }
        };

        await new AiModelBuilder<double, Matrix<double>, Vector<double>>()
            .ConfigureDataLoader(DataLoaders.FromMatrixVector(x, y))
            .ConfigureModel(new RidgeRegression<double>())
            .ConfigureTrainingPipeline(new TrainingPipelineConfiguration<double, Matrix<double>, Vector<double>>
            {
                Stages = new() { stage }
            })
            .BuildAsync();

        // IsEvaluationOnly stages should NOT invoke CustomTrainingFunction (no training).
        Assert.False(stageRan,
            "Evaluation-only stages must not call CustomTrainingFunction — they exist for evaluation hooks only.");
    }

    [Fact(Timeout = 120000)]
    public async Task BuildAsync_WithoutConfigureTrainingPipeline_NoStagesRun()
    {
        var (x, y) = BuildDataset();
        // Sanity: when the builder is not configured with a pipeline, BuildAsync
        // completes normally and no pipeline machinery runs.
        var result = await new AiModelBuilder<double, Matrix<double>, Vector<double>>()
            .ConfigureDataLoader(DataLoaders.FromMatrixVector(x, y))
            .ConfigureModel(new RidgeRegression<double>())
            .BuildAsync();

        Assert.NotNull(result);
        Assert.NotNull(result.Model);
    }
}
