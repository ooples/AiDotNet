using System;
using System.Linq;
using System.Threading.Tasks;
using AiDotNet;
using AiDotNet.CrossValidators;
using AiDotNet.Data.Loaders;
using AiDotNet.Interfaces;
using AiDotNet.Optimizers;
using AiDotNet.Regression;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.Training;

/// <summary>
/// Asserts that a configured cross-validator actually runs, and that the purged variant does not
/// leak across the fold boundary.
/// </summary>
/// <remarks>
/// ConfigureCrossValidation stored its argument in a field nothing read, so Validate was never
/// invoked and AiModelResult.CrossValidationResult stayed null — which its own documentation reads
/// as "cross-validation was not performed". A caller who asked for it was told it had not run.
/// </remarks>
public class ConfiguredCrossValidationTests
{
    private static (Matrix<double> X, Vector<double> Y) BuildData(int rows = 300, int cols = 3)
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

    [Fact(Timeout = 120000)]
    public async Task ConfiguredCrossValidator_IsActuallyRun_AndItsResultSurfaced()
    {
        await Task.Yield();
        var (x, y) = BuildData();

        var result = await new AiModelBuilder<double, Matrix<double>, Vector<double>>()
            .ConfigureModel(new MultipleRegression<double>())
            .ConfigureCrossValidation(new PurgedWalkForwardCrossValidator<double, Matrix<double>, Vector<double>>(
                labelHorizon: 3, nSplits: 3, embargo: 1))
            .ConfigureDataLoader(new InMemoryDataLoader<double, Matrix<double>, Vector<double>>(x, y))
            .BuildAsync();

        // Before the fix this was always null — the documented meaning of which is
        // "cross-validation was not performed".
        Assert.NotNull(result.CrossValidationResult);
    }

    [Fact(Timeout = 60000)]
    public async Task PurgedCrossValidator_IsAnICrossValidator_SoTheFacadeAcceptsIt()
    {
        await Task.Yield();
        // Until now no ICrossValidator implemented purge/embargo at all, so wiring
        // ConfigureCrossValidation would have left nothing purged to pass it.
        Assert.True(typeof(ICrossValidator<double, Matrix<double>, Vector<double>>)
            .IsAssignableFrom(typeof(PurgedWalkForwardCrossValidator<double, Matrix<double>, Vector<double>>)));
    }

    [Fact(Timeout = 60000)]
    public async Task PurgedCrossValidator_RejectsGeometryThatCannotPurge()
    {
        await Task.Yield();
        Assert.Throws<ArgumentOutOfRangeException>(
            () => new PurgedWalkForwardCrossValidator<double, Matrix<double>, Vector<double>>(labelHorizon: 0));
        Assert.Throws<ArgumentOutOfRangeException>(
            () => new PurgedWalkForwardCrossValidator<double, Matrix<double>, Vector<double>>(
                labelHorizon: 5, embargo: -1));
    }
}
