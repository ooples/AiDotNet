using System;
using System.Threading.Tasks;
using AiDotNet;
using AiDotNet.Data.Loaders;
using AiDotNet.DriftDetection;
using AiDotNet.Regression;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.Training;

/// <summary>
/// Covers the DriftMonitor's two-lens attribution and its wiring into Build via ConfigureDriftDetection.
/// </summary>
/// <remarks>
/// The monitor watches concept drift (the error stream) and covariate drift (the prediction
/// distribution) in a single pass and attributes which one fired. Before this wiring
/// ConfigureDriftDetection stored a field nothing read, so no drift monitoring ran.
/// </remarks>
public class DriftMonitorTests
{
    private static Vector<double> Vec(params double[] v) => new(v);

    [Fact]
    public void Monitor_NoShift_ReportsNoDrift()
    {
        var monitor = new DriftMonitor<double>(new DDMDriftDetector<double>(), windowSize: 10);
        // Reference: predictions ~1.0, small errors.
        var (p, a) = Steady(120, prediction: 1.0, error: 0.02, seed: 1);
        monitor.Prime(p, a);
        var (pc, ac) = Steady(120, prediction: 1.0, error: 0.02, seed: 2);

        var report = monitor.Check(pc, ac);

        Assert.False(report.DriftDetected);
        Assert.Equal(DriftSource.None, report.Source);
    }

    [Fact]
    public void Monitor_PredictionMeanShifts_AttributesToCovariate()
    {
        var monitor = new DriftMonitor<double>(new DDMDriftDetector<double>(), windowSize: 10);
        var (p, a) = Steady(150, prediction: 1.0, error: 0.02, seed: 3);
        monitor.Prime(p, a);

        // Errors stay tiny (no concept drift) but the prediction level jumps to 5.0 (covariate shift).
        var (pc, ac) = Steady(150, prediction: 5.0, error: 0.02, seed: 4);
        var report = monitor.Check(pc, ac);

        Assert.True(report.DriftDetected);
        Assert.True(report.Source is DriftSource.Covariate or DriftSource.Both);
    }

    [Fact]
    public void Monitor_ErrorRateRises_AttributesToConcept()
    {
        var monitor = new DriftMonitor<double>(new DDMDriftDetector<double>(), windowSize: 10);
        // Reference: prediction level 1.0, tiny errors.
        var (p, a) = Steady(200, prediction: 1.0, error: 0.02, seed: 5);
        monitor.Prime(p, a);

        // Prediction level stays 1.0 (no covariate shift) but errors blow up to 2.0 (concept drift):
        // the model's outputs look the same, yet they are now far from the targets.
        var (pc, ac) = Steady(200, prediction: 1.0, error: 2.0, seed: 6);
        var report = monitor.Check(pc, ac);

        Assert.True(report.DriftDetected);
        Assert.True(report.Source is DriftSource.Concept or DriftSource.Both);
    }

    [Fact(Timeout = 120000)]
    public async Task ConfigureDriftDetection_IsWiredIntoBuild()
    {
        var (x, y) = BuildData();

        var result = await new AiModelBuilder<double, Matrix<double>, Vector<double>>()
            .ConfigureModel(new MultipleRegression<double>())
            .ConfigureDriftDetection(new DDMDriftDetector<double>())
            .ConfigureDataLoader(new InMemoryDataLoader<double, Matrix<double>, Vector<double>>(x, y))
            .BuildAsync();

        // The report and the live monitor are surfaced whenever a detector is configured.
        Assert.NotNull(result.DriftReport);
        Assert.NotNull(result.DriftMonitor);
        Assert.True((result.DriftReport?.ObservationsChecked ?? 0) > 0);
    }

    [Fact(Timeout = 120000)]
    public async Task NoDriftDetector_LeavesResultDriftNull()
    {
        var (x, y) = BuildData();

        var result = await new AiModelBuilder<double, Matrix<double>, Vector<double>>()
            .ConfigureModel(new MultipleRegression<double>())
            .ConfigureDataLoader(new InMemoryDataLoader<double, Matrix<double>, Vector<double>>(x, y))
            .BuildAsync();

        Assert.Null(result.DriftReport);
        Assert.Null(result.DriftMonitor);
    }

    private static (Vector<double> predicted, Vector<double> actual) Steady(int n, double prediction, double error, int seed)
    {
        var rng = new Random(seed);
        var p = new double[n];
        var a = new double[n];
        for (int i = 0; i < n; i++)
        {
            p[i] = prediction;
            // Alternate the error sign so |error| is steady but actual varies around the prediction.
            a[i] = prediction + ((i % 2 == 0 ? 1 : -1) * error) + ((rng.NextDouble() - 0.5) * 1e-6);
        }

        return (new Vector<double>(p), new Vector<double>(a));
    }

    private static (Matrix<double> X, Vector<double> Y) BuildData(int rows = 120, int cols = 3)
    {
        var x = new Matrix<double>(rows, cols);
        var y = new Vector<double>(rows);
        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < cols; j++) x[i, j] = Math.Sin((i + j) * 0.15) + (i * 0.01);
            y[i] = Math.Sin((i + cols) * 0.15) + (i * 0.01);
        }

        return (x, y);
    }
}
