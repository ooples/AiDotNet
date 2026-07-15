using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using AiDotNet;
using AiDotNet.Data.Loaders;
using AiDotNet.Preprocessing.DataPreparation;
using AiDotNet.Preprocessing.DataPreparation.Splitting.TimeSeries;
using AiDotNet.Regression;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.Training;

/// <summary>
/// Asserts that a splitter passed to ConfigureDataSplitter actually partitions the training data.
/// </summary>
/// <remarks>
/// The configured splitter was stored in a field nothing read, so the built-in 0.7/0.15 ratio split
/// ran regardless. That also left every implementation under Preprocessing/DataPreparation/Splitting
/// unreachable from the facade — walk-forward, purged k-fold and combinatorial purged among them —
/// despite all of them already deriving DataSplitterBase&lt;T&gt;.
/// </remarks>
public class ConfiguredDataSplitterTests
{
    private static (Matrix<double> X, Vector<double> Y) BuildData(int rows = 60, int cols = 3)
    {
        var x = new Matrix<double>(rows, cols);
        var y = new Vector<double>(rows);
        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < cols; j++) x[i, j] = (i * 0.1) + j;
            y[i] = i * 0.1;
        }

        return (x, y);
    }

    /// <summary>Records the partition sizes the pipeline actually trained on.</summary>
    private sealed class RecordingSplitter : DataSplitterBase<double>
    {
        public int SplitCallCount { get; private set; }

        public override int NumSplits => 1;

        public override bool RequiresLabels => false;

        public override bool SupportsValidation => true;

        public override string Description => "recording-splitter";

        public override DataSplitResult<double> Split(Matrix<double> X, Vector<double>? y = null)
        {
            SplitCallCount++;
            LastTrainRows = X.Rows / 2;

            // A deliberately lopsided split, unlike the built-in 0.7/0.15, so an assertion on the
            // partition sizes can only pass if THIS splitter shaped the data.
            int valRows = X.Rows / 4;
            var indices = GetIndices(X.Rows);

            return BuildResult(
                X, y,
                trainIndices: indices.Take(LastTrainRows).ToArray(),
                testIndices: indices.Skip(LastTrainRows + valRows).ToArray(),
                validationIndices: indices.Skip(LastTrainRows).Take(valRows).ToArray());
        }

        public int LastTrainRows { get; private set; }
    }

    [Fact(Timeout = 120000)]
    public async Task ConfiguredSplitter_IsActuallyUsed()
    {
        var (x, y) = BuildData();
        var splitter = new RecordingSplitter();

        await new AiModelBuilder<double, Matrix<double>, Vector<double>>()
            .ConfigureModel(new MultipleRegression<double>())
            .ConfigureDataSplitter(splitter)
            .ConfigureDataLoader(new InMemoryDataLoader<double, Matrix<double>, Vector<double>>(x, y))
            .BuildAsync();

        // Before the fix this was 0: the configured splitter was never consulted.
        Assert.True(
            splitter.SplitCallCount > 0,
            "the configured splitter was never called — the built-in ratio split ran instead");
    }

    [Fact(Timeout = 120000)]
    public async Task PurgedSplitter_ReachesTheFacade()
    {
        // The point of the whole exercise: a purged, embargoed splitter is usable from the builder.
        // It already derived DataSplitterBase<T>; only the dead field kept it unreachable.
        var (x, y) = BuildData(rows: 80);
        var purged = new PurgedKFoldSplitter<double>(k: 3, purgeSize: 2, embargoSize: 2);

        var result = await new AiModelBuilder<double, Matrix<double>, Vector<double>>()
            .ConfigureModel(new MultipleRegression<double>())
            .ConfigureDataSplitter(purged)
            .ConfigureDataLoader(new InMemoryDataLoader<double, Matrix<double>, Vector<double>>(x, y))
            .BuildAsync();

        Assert.NotNull(result);
    }
}
