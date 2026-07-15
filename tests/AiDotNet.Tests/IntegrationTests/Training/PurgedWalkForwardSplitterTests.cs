using System;
using System.Linq;
using System.Threading.Tasks;
using AiDotNet.Preprocessing.DataPreparation;
using AiDotNet.Preprocessing.DataPreparation.Splitting.TimeSeries;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.Training;

/// <summary>
/// Covers the purged, embargoed walk-forward splitter — the finance-grade fold geometry, now
/// reachable through IDataSplitter and therefore through ConfigureDataSplitter.
/// </summary>
/// <remarks>
/// The logic already existed in PurgedWalkForwardValidator, but as a static class it could never be
/// passed to the facade. These tests assert the property that makes it worth having: training rows
/// whose label window reaches into the test fold are removed, so the score is not computed partly on
/// data the model effectively saw.
/// </remarks>
public class PurgedWalkForwardSplitterTests
{
    private static Matrix<double> BuildMatrix(int rows)
    {
        var x = new Matrix<double>(rows, 2);
        for (int i = 0; i < rows; i++)
        {
            x[i, 0] = i;
            x[i, 1] = i * 0.5;
        }

        return x;
    }

    [Fact(Timeout = 60000)]
    public async Task IsAnIDataSplitter_SoTheFacadeCanAcceptIt()
    {
        // The whole point of reshaping the static validator: it must satisfy the contract the
        // builder's ConfigureDataSplitter takes.
        Assert.True(typeof(IDataSplitter<double>).IsAssignableFrom(typeof(PurgedWalkForwardSplitter<double>)));
        await Task.CompletedTask;
    }

    [Fact(Timeout = 60000)]
    public async Task TrainAndTestIndicesNeverOverlap()
    {
        var splitter = new PurgedWalkForwardSplitter<double>(labelHorizon: 5, nSplits: 3, embargo: 2);

        foreach (var fold in splitter.GetSplits(BuildMatrix(120)))
        {
            Assert.Empty(fold.TrainIndices.Intersect(fold.TestIndices));
        }

        await Task.CompletedTask;
    }

    [Fact(Timeout = 60000)]
    public async Task PurgesTrainingRowsWhoseLabelWindowReachesTheTestFold()
    {
        // This is the leak a plain chronological split leaves behind: with a 10-step label horizon,
        // a training row at testStart-3 has a label derived from rows inside the test fold.
        const int labelHorizon = 10;
        var splitter = new PurgedWalkForwardSplitter<double>(labelHorizon, nSplits: 3, embargo: 0);

        foreach (var fold in splitter.GetSplits(BuildMatrix(150)))
        {
            int testStart = fold.TestIndices.Min();

            foreach (int trainIdx in fold.TrainIndices.Where(i => i < testStart))
            {
                Assert.True(
                    trainIdx + labelHorizon <= testStart,
                    $"training row {trainIdx} has a {labelHorizon}-step label window reaching into the " +
                    $"test fold starting at {testStart}; it should have been purged");
            }
        }

        await Task.CompletedTask;
    }

    [Fact(Timeout = 60000)]
    public async Task EmbargoWidensTheGapAfterTheTestFold()
    {
        var noEmbargo = new PurgedWalkForwardSplitter<double>(labelHorizon: 5, nSplits: 3, embargo: 0);
        var withEmbargo = new PurgedWalkForwardSplitter<double>(labelHorizon: 5, nSplits: 3, embargo: 10);

        int trainRowsWithout = noEmbargo.GetSplits(BuildMatrix(150)).Sum(f => f.TrainIndices.Count());
        int trainRowsWith = withEmbargo.GetSplits(BuildMatrix(150)).Sum(f => f.TrainIndices.Count());

        // An embargo can only ever remove rows; if it removed none it isn't being applied.
        Assert.True(
            trainRowsWith < trainRowsWithout,
            $"embargo removed no training rows ({trainRowsWith} vs {trainRowsWithout})");
        await Task.CompletedTask;
    }

    [Fact(Timeout = 60000)]
    public async Task RejectsGeometryThatCannotPurge()
    {
        // A label horizon below 1 means "no forward window", which makes purging meaningless — the
        // parameter would be silently useless rather than merely wrong.
        Assert.Throws<ArgumentOutOfRangeException>(
            () => new PurgedWalkForwardSplitter<double>(labelHorizon: 0));
        Assert.Throws<ArgumentOutOfRangeException>(
            () => new PurgedWalkForwardSplitter<double>(labelHorizon: 5, embargo: -1));
        await Task.CompletedTask;
    }

    [Fact(Timeout = 60000)]
    public async Task FoldsAreChronological()
    {
        var splitter = new PurgedWalkForwardSplitter<double>(labelHorizon: 5, nSplits: 3, embargo: 1);

        foreach (var fold in splitter.GetSplits(BuildMatrix(120)))
        {
            // Walk-forward means train precedes test; shuffling would destroy the ordering that
            // purge and embargo are defined against.
            Assert.True(fold.TrainIndices.Max() < fold.TestIndices.Min());
        }

        await Task.CompletedTask;
    }
}
