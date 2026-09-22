using System;
using AiDotNet.Clustering.Base;
using AiDotNet.Clustering.Hierarchical;
using AiDotNet.Clustering.Partitioning;
using AiDotNet.Clustering.Probabilistic;
using AiDotNet.Clustering.Spectral;
using AiDotNet.Clustering.Options;
using AiDotNet.Regression;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.Clustering;

/// <summary>
/// Covers <c>AiModelBuilder.Build(features)</c>, the overload for models that have nothing to learn
/// from labels.
/// </summary>
/// <remarks>
/// <para>
/// Clustering has no labels — that is what makes it clustering — so the two-argument
/// <c>Build(features, labels)</c> forced every caller to construct an argument the model ignores and
/// then explain it. A parameter that exists only to be discarded is worth removing rather than
/// documenting.
/// </para>
/// </remarks>
public class UnsupervisedBuildOverloadTests
{
    /// <summary>Two well-separated groups, so an assignment is something to check rather than noise.</summary>
    private static Matrix<double> Data()
    {
        double[,] rows =
        {
            { 1.0, 2.0 }, { 1.5, 1.8 }, { 1.0, 0.6 }, { 1.2, 1.4 },
            { 8.0, 8.0 }, { 9.0, 11.0 }, { 8.5, 9.5 }, { 9.2, 10.1 }
        };
        var m = new Matrix<double>(rows.GetLength(0), rows.GetLength(1));
        for (int i = 0; i < rows.GetLength(0); i++)
        {
            for (int j = 0; j < rows.GetLength(1); j++)
            {
                m[i, j] = rows[i, j];
            }
        }

        return m;
    }

    /// <summary>
    /// The overload's whole job: reach the same fit the ignored-label call reached, without the caller
    /// having to construct the thing being ignored.
    /// </summary>
    [Fact]
    public void ItFitsWhatThePlaceholderCallFits()
    {
        var data = Data();

        var viaOverload = new AiModelBuilder<double, Matrix<double>, Vector<double>>()
            .ConfigureModel(new KMeans<double>(new KMeansOptions<double> { NumClusters = 2 }))
            .Build(data);

        var viaPlaceholder = new AiModelBuilder<double, Matrix<double>, Vector<double>>()
            .ConfigureModel(new KMeans<double>(new KMeansOptions<double> { NumClusters = 2 }))
            .Build(data, new Vector<double>(data.Rows));

        var a = viaOverload.GetClusterLabels();
        var b = viaPlaceholder.GetClusterLabels();

        Assert.NotNull(a);
        Assert.NotNull(b);
        Assert.Equal(b!.Length, a!.Length);
        for (int i = 0; i < b.Length; i++)
        {
            Assert.Equal(b[i], a[i]);
        }
    }

    /// <summary>
    /// The point of clustering: an assignment per row, and the two obvious groups in this fixture kept
    /// apart. A build that silently did nothing would still return a result, so this checks the output.
    /// </summary>
    [Fact]
    public void ItAssignsEveryRowAndSeparatesTheGroups()
    {
        var data = Data();

        var result = new AiModelBuilder<double, Matrix<double>, Vector<double>>()
            .ConfigureModel(new KMeans<double>(new KMeansOptions<double> { NumClusters = 2 }))
            .Build(data);

        var labels = result.GetClusterLabels();

        Assert.NotNull(labels);
        Assert.Equal(data.Rows, labels!.Length);

        // Rows 0-3 are one group and 4-7 the other; whichever index each got, they must not mix.
        Assert.Equal(labels[0], labels[1]);
        Assert.Equal(labels[0], labels[2]);
        Assert.Equal(labels[4], labels[5]);
        Assert.Equal(labels[4], labels[6]);
        Assert.NotEqual(labels[0], labels[4]);
    }

    [Fact]
    public void ARowTheModelHasNotSeen_IsPlacedByPredict()
    {
        var data = Data();

        var result = new AiModelBuilder<double, Matrix<double>, Vector<double>>()
            .ConfigureModel(new KMeans<double>(new KMeansOptions<double> { NumClusters = 2 }))
            .Build(data);

        var newRows = new Matrix<double>(1, 2);
        newRows[0, 0] = 8.8;
        newRows[0, 1] = 9.9;

        var placed = result.Predict(newRows);

        Assert.Equal(1, placed.Length);
    }

    [Fact]
    public void ASupervisedModel_IsRejectedRatherThanHavingItsLabelsDiscarded()
    {
        var ex = Assert.Throws<InvalidOperationException>(() =>
            new AiModelBuilder<double, Matrix<double>, Vector<double>>()
                .ConfigureModel(new SimpleRegression<double>())
                .Build(Data()));

        Assert.Contains("unsupervised", ex.Message, StringComparison.Ordinal);
        Assert.Contains(nameof(SimpleRegression<double>), ex.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void NoModelConfigured_SaysWhatToConfigure()
    {
        var ex = Assert.Throws<InvalidOperationException>(() =>
            new AiModelBuilder<double, Matrix<double>, Vector<double>>().Build(Data()));

        Assert.Contains("KMeans", ex.Message, StringComparison.Ordinal);
    }

    [Theory]
    [InlineData("KMeans")]
    [InlineData("BIRCH")]
    [InlineData("CURE")]
    [InlineData("CLARANS")]
    [InlineData("GaussianMixtureModel")]
    [InlineData("SpectralClustering")]
    public void EveryClusteringModel_BuildsFromDataAlone(string model)
    {
        ClusteringBase<double> Build() => model switch
        {
            "KMeans" => new KMeans<double>(new KMeansOptions<double> { NumClusters = 2 }),
            "BIRCH" => new BIRCH<double>(new BIRCHOptions<double> { NumClusters = 2 }),
            "CURE" => new CURE<double>(new CUREOptions<double> { NumClusters = 2 }),
            "CLARANS" => new CLARANS<double>(new CLARANSOptions<double> { NumClusters = 2 }),
            "GaussianMixtureModel" => new GaussianMixtureModel<double>(
                new GMMOptions<double> { NumComponents = 2 }),
            "SpectralClustering" => new SpectralClustering<double>(
                new SpectralOptions<double> { NumClusters = 2 }),
            _ => throw new ArgumentOutOfRangeException(nameof(model))
        };

        var data = Data();

        var result = new AiModelBuilder<double, Matrix<double>, Vector<double>>()
            .ConfigureModel(Build())
            .Build(data);

        var labels = result.GetClusterLabels();

        Assert.NotNull(labels);
        Assert.Equal(data.Rows, labels!.Length);
    }
}
