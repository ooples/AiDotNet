using System;
using System.Collections.Generic;
using System.Threading.Tasks;
using AiDotNet;
using AiDotNet.Data.Loaders;
using AiDotNet.Interfaces;
using AiDotNet.Models.Options;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.TimeSeries;
using Xunit;

namespace AiDotNet.Tests.UnitTests.TimeSeries;

/// <summary>
/// Covers per-epoch reporting and early stopping for the TimeSeries family.
/// </summary>
/// <remarks>
/// These models run their own epoch loop inside Train(), which the facade calls once. That made
/// them invisible to ConfigureTrainingCallback: callers got a single synthetic epoch reported
/// after training had already finished, so no callback could observe real progress or stop a run.
/// DLinear is used as the representative model because it is by far the cheapest to fit.
/// </remarks>
public class TimeSeriesEarlyStoppingTests
{
    /// <summary>A short deterministic series with enough structure for the loss to move.</summary>
    private static (Matrix<double> X, Vector<double> Y) BuildSeries(int rows = 60, int cols = 8)
    {
        var x = new Matrix<double>(rows, cols);
        var y = new Vector<double>(rows);
        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < cols; j++)
            {
                x[i, j] = Math.Sin((i + j) * 0.3) + (i * 0.01);
            }

            y[i] = Math.Sin((i + cols) * 0.3) + (i * 0.01);
        }

        return (x, y);
    }

    private static DLinearModel<double> BuildModel(Action<DLinearOptions<double>>? configure = null)
    {
        var options = new DLinearOptions<double>
        {
            LookbackWindow = 4,
            ForecastHorizon = 1,
            Epochs = 40,
        };
        configure?.Invoke(options);
        return new DLinearModel<double>(options);
    }

    [Fact(Timeout = 60000)]
    public async Task ReportsEveryEpoch_NotJustOne()
    {
        var (x, y) = BuildSeries();
        var model = BuildModel(o => o.Epochs = 12);
        var observed = new List<int>();

        ((ITrainingEpochReporter<double>)model).TrainingEpochCallback = progress =>
        {
            observed.Add(progress.Epoch);
            return true;
        };

        await Task.Run(() => model.Train(x, y));

        // The whole point of the fix: one report per epoch, in order — not a single post-hoc epoch.
        Assert.Equal(12, observed.Count);
        Assert.Equal(0, observed[0]);
        Assert.Equal(11, observed[^1]);
    }

    [Fact(Timeout = 60000)]
    public async Task CallbackReturningFalse_StopsTrainingEarly()
    {
        var (x, y) = BuildSeries();
        var model = BuildModel(o => o.Epochs = 50);
        int epochsSeen = 0;

        ((ITrainingEpochReporter<double>)model).TrainingEpochCallback = _ =>
        {
            epochsSeen++;
            return epochsSeen < 3; // veto after the third epoch
        };

        await Task.Run(() => model.Train(x, y));

        Assert.Equal(3, epochsSeen);
        Assert.Contains("callback requested abort", model.TrainingStopReason);
    }

    [Fact(Timeout = 60000)]
    public async Task EarlyStopping_StopsOnceLossPlateaus()
    {
        var (x, y) = BuildSeries();
        // A zero learning rate guarantees a flat loss, so patience must expire on schedule. This
        // asserts the stopping rule itself rather than a particular convergence trajectory.
        var model = BuildModel(o =>
        {
            o.Epochs = 100;
            o.LearningRate = 0.0;
            o.UseEarlyStopping = true;
            o.EarlyStoppingPatience = 3;
        });

        int epochsSeen = 0;
        ((ITrainingEpochReporter<double>)model).TrainingEpochCallback = _ =>
        {
            epochsSeen++;
            return true;
        };

        await Task.Run(() => model.Train(x, y));

        // Epoch 0 sets the baseline; epochs 1-3 fail to improve and exhaust patience.
        Assert.Equal(4, epochsSeen);
        Assert.Contains("early stopping", model.TrainingStopReason);
    }

    [Fact(Timeout = 60000)]
    public async Task EarlyStoppingDisabled_RunsFullEpochBudget()
    {
        var (x, y) = BuildSeries();
        var model = BuildModel(o =>
        {
            o.Epochs = 20;
            o.LearningRate = 0.0; // flat loss would trip patience if early stopping were on
            o.UseEarlyStopping = false;
        });

        int epochsSeen = 0;
        ((ITrainingEpochReporter<double>)model).TrainingEpochCallback = _ =>
        {
            epochsSeen++;
            return true;
        };

        await Task.Run(() => model.Train(x, y));

        // Default must preserve the exact-iteration-count contract.
        Assert.Equal(20, epochsSeen);
        Assert.Null(model.TrainingStopReason);
    }

    [Fact(Timeout = 60000)]
    public async Task EarlyStoppedModel_IsStillTrainedAndPredicts()
    {
        var (x, y) = BuildSeries();
        var model = BuildModel(o =>
        {
            o.Epochs = 50;
            o.UseEarlyStopping = true;
            o.EarlyStoppingPatience = 2;
        });

        ((ITrainingEpochReporter<double>)model).TrainingEpochCallback = p => p.Epoch < 2;

        await Task.Run(() => model.Train(x, y));

        // An early stop must leave a usable model, not a half-initialized one.
        var prediction = model.Predict(x);
        Assert.Equal(x.Rows, prediction.Length);
        for (int i = 0; i < prediction.Length; i++)
        {
            Assert.False(
                double.IsNaN(prediction[i]) || double.IsInfinity(prediction[i]),
                $"prediction[{i}] was not finite");
        }
    }

    [Fact(Timeout = 120000)]
    public async Task Facade_ConfigureTrainingCallback_ObservesRealEpochs()
    {
        var (x, y) = BuildSeries();
        var model = BuildModel(o => o.Epochs = 10);
        var losses = new List<double>();

        var loader = new InMemoryDataLoader<double, Matrix<double>, Vector<double>>(x, y);
        await new AiModelBuilder<double, Matrix<double>, Vector<double>>()
            .ConfigureModel(model)
            .ConfigureDataLoader(loader)
            .ConfigureTrainingCallback(progress =>
            {
                losses.Add(progress.Loss);
                return true;
            })
            .BuildAsync();

        // Before the fix this saw exactly one epoch, reported after training had finished — so a
        // caller could neither watch progress nor implement early stopping through the facade. The model is
        // configured for exactly 10 epochs with early stopping off, so the facade must observe all 10.
        Assert.Equal(10, losses.Count);
    }

    [Fact(Timeout = 120000)]
    public async Task Facade_CallbackReturningFalse_StopsTrainingEarly()
    {
        var (x, y) = BuildSeries();
        var model = BuildModel(o => o.Epochs = 60);
        int observed = 0;

        var loader = new InMemoryDataLoader<double, Matrix<double>, Vector<double>>(x, y);
        await new AiModelBuilder<double, Matrix<double>, Vector<double>>()
            .ConfigureModel(model)
            .ConfigureDataLoader(loader)
            .ConfigureTrainingCallback(_ =>
            {
                observed++;
                return observed < 4;
            })
            .BuildAsync();

        // The veto has to reach the model's own loop, not just be recorded by the facade.
        Assert.Equal(4, observed);
    }
}
