using System;
using System.Collections.Generic;
using System.Linq;
using AiDotNet.Models.Options;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.TimeSeries;
using Xunit;

namespace AiDotNet.Tests.UnitTests.TimeSeries;

/// <summary>
/// Chronos trains through the shared tape step inside the TensorArena that TimeSeriesModelBase.Train opens. Every step
/// must be exactly the plain SGD step on the mean next-token gradient, step after step: the arena recycles memory between
/// steps, and AiDotNet.Tensors 0.136.0 let TensorPool keep arena-backed tensors, so from the second step on the
/// backward of TensorNarrow was overwritten (this replay was off by 6e-4 after four steps; Tensors #1275).
/// </summary>
public sealed class ChronosSharedStepTrainingTests
{
    private static ChronosOptions<double> Options(int epochs) => new()
    {
        Seed = 11, ContextLength = 12, ForecastHorizon = 1, EmbeddingDim = 16, NumLayers = 2, NumHeads = 2,
        Epochs = epochs, UseEarlyStopping = false,
    };

    [Fact(Timeout = 120000)]
    public async System.Threading.Tasks.Task TrainingMatchesAHandReplayOfTheSgdSteps()
    {
        await System.Threading.Tasks.Task.Yield();

        // 30 rows give 29 training windows, one mini-batch, so every epoch is one full-batch step and the replay does
        // not depend on the shuffle order.
        const int rows = 30, epochs = 4;
        var x = new Matrix<double>(rows, 1);
        var y = new Vector<double>(rows);
        for (int i = 0; i < rows; i++) { x[i, 0] = i; y[i] = Math.Sin(i * 0.2) + 0.01 * i; }

        var model = new ChronosFoundationModel<double>(Options(epochs));
        model.Train(x, y);

        var replay = new ChronosFoundationModel<double>(Options(epochs));
        var groups = new Dictionary<int, (List<Vector<double>> Windows, List<double> Targets)>();
        for (int idx = 1; idx < rows; idx++)
        {
            int start = Math.Max(0, idx - 12);
            var window = new Vector<double>(idx - start);
            for (int i = start; i < idx; i++) window[i - start] = y[i];
            if (!groups.TryGetValue(window.Length, out var group))
                groups[window.Length] = group = (new List<Vector<double>>(), new List<double>());
            group.Windows.Add(window);
            group.Targets.Add(y[idx]);
        }

        var parameters = replay.NamedTrainableTensors().ToList();
        double learningRate = Options(epochs).LearningRate;
        for (int step = 0; step < epochs; step++)
        {
            var sum = new Dictionary<string, double[]>();
            foreach (var group in groups.Values)
                foreach (var gradient in replay.ComputeBatchGradientsTape(group.Windows, group.Targets))
                {
                    if (!sum.TryGetValue(gradient.Key, out var acc))
                        sum[gradient.Key] = acc = new double[gradient.Value.Length];
                    for (int i = 0; i < acc.Length; i++) acc[i] += gradient.Value[i];
                }
            foreach (var (key, parameter) in parameters)
            {
                if (!sum.TryGetValue(key, out var total)) continue;
                for (int i = 0; i < parameter.Length; i++)
                    parameter[i] -= learningRate * total[i] / (rows - 1);
                parameter.IncrementVersion();
            }
        }

        var trained = model.NamedTrainableTensors().ToDictionary(p => p.Key, p => p.Param);
        foreach (var (key, expected) in parameters)
        {
            var actual = trained[key];
            for (int i = 0; i < expected.Length; i++)
                Assert.True(Math.Abs(actual[i] - expected[i]) <= 1e-12,
                    $"{key}[{i}] after {epochs} steps: trained {actual[i]:R}, replayed {expected[i]:R}.");
        }
    }
}
