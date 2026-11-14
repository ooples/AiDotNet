using System;
using System.Linq;
using System.Threading.Tasks;
using AiDotNet.Preprocessing;
using AiDotNet.Preprocessing.FeatureSelection;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.Preprocessing;

/// <summary>
/// Reference tests for the feature-selector helpers: class, group, row and pair counts must be exact past 2048 in
/// Half (a 0/1 sum accumulated in Half stops growing at 2048), and the nearest-neighbour mutual information must
/// match scikit-learn on a large sample.
/// </summary>
public class SelectorTensorOpsReferenceTests
{
    // 3072 rows of class 0 and 1024 of class 1: every expected value below is exact in Half.
    private const int ClassZeroRows = 3072;
    private const int ClassOneRows = 1024;
    private const int Rows = ClassZeroRows + ClassOneRows;

    private static int[] ClassIndex() =>
        Enumerable.Range(0, Rows).Select(i => i < ClassZeroRows ? 0 : 1).ToArray();

    private static Tensor<Half> ClassOneColumn(double value) =>
        new Tensor<Half>(ClassIndex().Select(c => c == 1 ? (Half)value : (Half)0.0).ToArray(), new[] { Rows, 1 });

    [Fact(Timeout = 60000)]
    public async Task GroupMeans_Half_CountsGroupSizesPast2048()
    {
        await Task.CompletedTask;
        var membership = PreprocessingTensorOps.ClassOneHot<Half>(ClassIndex(), 2);

        SelectorTensorOps.GroupMeans(membership, ClassOneColumn(1.0), out var counts);

        Assert.Equal((double)ClassZeroRows, (double)counts.ToArray()[0]);
        Assert.Equal((double)ClassOneRows, (double)counts.ToArray()[1]);
    }

    // scikit-learn chi2: observed [0, 64], expected = 64 * [0.75, 0.25] = [48, 16], so
    // chi2 = 48^2 / 48 + 48^2 / 16 = 192. With the class-0 size stuck at 2048 the statistic came out 176.
    [Fact(Timeout = 60000)]
    public async Task Chi2_Half_UsesExactClassSizes()
    {
        await Task.CompletedTask;
        var target = new Vector<Half>(ClassIndex().Select(c => (Half)(double)c).ToArray());

        var scores = SelectorTensorOps.Chi2(ClassOneColumn(0.0625), target, out var pValues);

        Assert.Equal(192.0, (double)scores.ToArray()[0]);
        Assert.True((double)pValues.ToArray()[0] < 1e-3);
    }

    // minRows = 3000 is met by the 4096 masked rows; a count stuck at 2048 failed it and zeroed the correlation.
    [Fact(Timeout = 60000)]
    public async Task MaskedPearson_Half_CountsRowsPast2048()
    {
        await Task.CompletedTask;
        var values = Enumerable.Range(0, Rows).Select(i => (Half)(0.0625 * (i % 2))).ToArray();
        var x = new Tensor<Half>(values, new[] { Rows, 1 });
        var y = new Tensor<Half>((Half[])values.Clone(), new[] { Rows, 1 });
        var mask = new Tensor<Half>(Enumerable.Repeat((Half)1.0, Rows).ToArray(), new[] { Rows, 1 });

        var r = SelectorTensorOps.MaskedPearson(x, y, mask, 3000);

        Assert.Equal(1.0, (double)r.ToArray()[0], 2);
    }

    // x = y = 0..99: every one of the 4950 pairs is concordant, so tau-a = tau-b = 1. The 9900 ordered-pair sums
    // stopped at 2048 in Half and the tau-b denominator product overflowed Half.
    [Fact(Timeout = 60000)]
    public async Task KendallTau_Half_CountsPairsPast2048()
    {
        await Task.CompletedTask;
        const int n = 100;
        var values = Enumerable.Range(0, n).Select(i => (Half)(double)i).ToArray();
        var x = new Tensor<Half>(values, new[] { n, 1 });
        var y = new Tensor<Half>((Half[])values.Clone(), new[] { n });

        var tauB = SelectorTensorOps.KendallTau(x, y, tauB: true);
        var tauA = SelectorTensorOps.KendallTau(x, y, tauB: false);

        Assert.Equal(1.0, (double)tauB.ToArray()[0], 2);
        Assert.Equal(1.0, (double)tauA.ToArray()[0], 2);
    }

    // 3001 samples (one in a single-member class, which is ignored). Expected values are
    // sklearn.feature_selection._mutual_info._compute_mi_cd(column, labels, n_neighbors=3) (scikit-learn 1.9.1) on
    // the same tie-free columns, i.e. mutual_info_classif without its tie-breaking noise.
    [Fact(Timeout = 120000)]
    public async Task MutualInfoClassif_LargeSample_MatchesScikitLearn()
    {
        await Task.CompletedTask;
        const int n = 3000;
        var data = new double[(n + 1) * 2];
        var labels = new double[n + 1];
        for (int i = 0; i < n; i++)
        {
            data[2 * i] = (i % 3) * 0.5 + (i * 0.6180339887498949) % 1.0;
            data[2 * i + 1] = (i * 0.4142135623730951) % 1.0 + (i % 3) * 0.3;
            labels[i] = i % 3;
        }

        data[2 * n] = 0.25;
        data[2 * n + 1] = 0.75;
        labels[n] = 7;

        var mi = SelectorTensorOps.MutualInfoClassif(new Tensor<double>(data, new[] { n + 1, 2 }), new Vector<double>(labels));

        Assert.Equal(0.4909923771721847, mi.ToArray()[0], 10);
        Assert.Equal(0.20556793494774284, mi.ToArray()[1], 10);
    }
}
