using System;
using System.Collections.Generic;
using System.Linq;
using AiDotNet.Enums;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks.SyntheticData;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNetTests.IntegrationTests.SyntheticData;

/// <summary>
/// Reference tests for <see cref="BayesianNetworkSynthGenerator{T}"/> against PrivBayes
/// (Zhang, Cormode, Procopiuc, Srivastava, Xiao; SIGMOD 2014 / TODS 2017).
/// </summary>
public class PrivBayesPaperFidelityTests
{
    private const double Tol = 1e-12;

    /// <summary>
    /// Lemma 1: the binary-domain sensitivity is (1/n) log2 n + ((n-1)/n) log2(n/(n-1)); the general
    /// form is (2/n) log2((n+1)/2) + ((n-1)/n) log2((n+1)/(n-1)). Values for n = 100 computed by hand
    /// from those formulas.
    /// </summary>
    [Fact]
    public void Sensitivity_MatchesLemma1_BinaryAndGeneral()
    {
        Assert.Equal(0.08079313589591126, BayesianNetworkSynthGenerator<double>.MutualInformationSensitivity(100, true), Tol);
        Assert.Equal(0.14173054370049903, BayesianNetworkSynthGenerator<double>.MutualInformationSensitivity(100, false), Tol);
    }

    /// <summary>
    /// Mutual information of a known table: X = [0,1,2,0,1,2,0,0], Pi = [0,0,1,1,0,1,1,0] has
    /// I(X; Pi) = 0.5 bits (sklearn mutual_info_score gives 0.5 ln 2 nats for the same labels).
    /// </summary>
    [Fact]
    public void MutualInformation_MatchesReferenceTable()
    {
        var child = new Tensor<int>(new[] { 0, 1, 2, 0, 1, 2, 0, 0 }, new[] { 8 });
        var parent = new Tensor<int>(new[] { 0, 0, 1, 1, 0, 1, 1, 0 }, new[] { 8 });
        var counts = BayesianNetworkSynthGenerator<double>.JointCounts(child, 3, parent, 2);

        Assert.Equal(new[] { 2.0, 2.0, 0.0, 2.0, 0.0, 2.0 }, counts.ToArray());
        Assert.Equal(0.5, BayesianNetworkSynthGenerator<double>.MutualInformation(counts, 8).GetFlat(0), Tol);
    }

    /// <summary>
    /// GreedyBayes considers parent sets of EXACTLY min(k, |V|) attached attributes, not every size up
    /// to k.
    /// </summary>
    [Fact]
    public void ParentSets_HaveExactlyRequestedSize()
    {
        var sets = BayesianNetworkSynthGenerator<double>.EnumerateParentSets(new[] { 4, 1, 7 }, 2)
            .Select(s => string.Join(",", s)).ToList();
        Assert.Equal(new[] { "4,1", "4,7", "1,7" }, sets);

        var empty = BayesianNetworkSynthGenerator<double>.EnumerateParentSets(new[] { 4, 1, 7 }, 0).ToList();
        Assert.Single(empty);
        Assert.Empty(empty[0]);
    }

    /// <summary>
    /// Theta-usefulness (Definition 4.5, binary domains): n eps / ((d-k) 2^(k+3)) &gt;= theta with
    /// eps = 2 eps2. For n = 10000, eps2 = 0.5, d = 10, theta = 4: k = 6 gives 4.88, k = 7 gives 3.26,
    /// so k = 6.
    /// </summary>
    [Fact]
    public void Degree_FollowsThetaUsefulness()
    {
        Assert.Equal(6, BayesianNetworkSynthGenerator<double>.SelectDegree(10000, 10, 2, 0.5, 4.0, 10));
        Assert.Equal(3, BayesianNetworkSynthGenerator<double>.SelectDegree(10000, 10, 2, 0.5, 4.0, 3));
        Assert.Equal(9, BayesianNetworkSynthGenerator<double>.SelectDegree(10000, 10, 2, 0.5, 0.0, 20));
        Assert.Equal(0, BayesianNetworkSynthGenerator<double>.SelectDegree(10, 10, 2, 0.5, 4.0, 3));
    }

    private static (Matrix<double> Data, List<ColumnMetadata> Columns) CategoricalData(int rows, int cols)
    {
        var rng = AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(5);
        var data = new Matrix<double>(rows, cols);
        var values = new[] { 3.0, 7.0, 11.0 };
        for (int i = 0; i < rows; i++)
        {
            int a = rng.Next(3);
            data[i, 0] = values[a];
            for (int j = 1; j < cols; j++)
            {
                data[i, j] = values[rng.NextDouble() < 0.8 ? a : rng.Next(3)];
            }
        }

        var columns = Enumerable.Range(0, cols)
            .Select(j => new ColumnMetadata($"c{j}", ColumnDataType.Categorical)).ToList();
        return (data, columns);
    }

    /// <summary>
    /// The network has degree k: attribute i (in attachment order) has min(i, k) parents, the phase-2
    /// budget pays for exactly d - k noisy joints, and every conditional table covers the whole parent
    /// domain, including configurations that never occur in the data.
    /// </summary>
    [Fact]
    public void Network_HasDegreeK_AndNoisesDMinusKFullTables()
    {
        var (data, columns) = CategoricalData(300, 4);
        var generator = new BayesianNetworkSynthGenerator<double>(new BayesianNetworkSynthOptions<double>
        {
            Seed = 17,
            MaxParents = 2,
            UsefulnessThreshold = 0.0
        });
        generator.Fit(data, columns, 1);

        Assert.Equal(2, generator.Degree);
        Assert.Equal(2, generator.NoisedMarginalCount);
        var sizes = generator.Parents.Select(p => p.Count).OrderBy(c => c).ToArray();
        Assert.Equal(new[] { 0, 1, 2, 2 }, sizes);
        for (int j = 0; j < 4; j++)
        {
            int expectedRows = (int)Math.Pow(3, generator.Parents[j].Count);
            Assert.Equal(new[] { expectedRows, 3 }, generator.Conditionals[j].Shape.ToArray());
        }
    }

    /// <summary>
    /// With a negligible-noise budget the root's conditional (derived from the (k+1)-th noisy joint)
    /// matches the empirical marginal of that column.
    /// </summary>
    [Fact]
    public void FirstKConditionals_AreMarginalsOfTheAnchorJoint()
    {
        var (data, columns) = CategoricalData(300, 4);
        var generator = new BayesianNetworkSynthGenerator<double>(new BayesianNetworkSynthOptions<double>
        {
            Seed = 3,
            MaxParents = 2,
            UsefulnessThreshold = 0.0,
            PrivacyBudget = 1e9
        });
        generator.Fit(data, columns, 1);

        int root = Enumerable.Range(0, 4).Single(j => generator.Parents[j].Count == 0);
        var expected = new[] { 3.0, 7.0, 11.0 }
            .Select(v => Enumerable.Range(0, 300).Count(i => data[i, root] == v) / 300.0).ToArray();
        var actual = generator.Conditionals[root].ToArray();
        for (int b = 0; b < 3; b++) Assert.Equal(expected[b], actual[b], 1e-4);
    }

    /// <summary>Categorical columns are generated as their own category values, never as interpolated numbers.</summary>
    [Fact]
    public void CategoricalColumns_GenerateCategoryValues()
    {
        var (data, columns) = CategoricalData(200, 3);
        var generator = new BayesianNetworkSynthGenerator<double>(new BayesianNetworkSynthOptions<double> { Seed = 9 });
        generator.Fit(data, columns, 1);
        var m = generator.Generate(150);

        var allowed = new HashSet<double> { 3.0, 7.0, 11.0 };
        for (int i = 0; i < m.Rows; i++)
            for (int j = 0; j < m.Columns; j++)
                Assert.Contains(m[i, j], allowed);
    }

    /// <summary>
    /// Public domain bounds set the bins: with bounds [0, 10] and 10 bins, data in [0, 1) all lands in
    /// bin 0, so the non-private, unsmoothed model only generates values in [0, 1).
    /// </summary>
    [Fact]
    public void DomainBounds_DefineTheBins()
    {
        var rng = AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(1);
        var data = new Matrix<double>(50, 2);
        for (int i = 0; i < 50; i++) { data[i, 0] = rng.NextDouble() * 0.99; data[i, 1] = rng.NextDouble() * 0.99; }
        var columns = new List<ColumnMetadata>
        {
            new ColumnMetadata("a", ColumnDataType.Continuous),
            new ColumnMetadata("b", ColumnDataType.Continuous)
        };

        var generator = new BayesianNetworkSynthGenerator<double>(new BayesianNetworkSynthOptions<double>
        {
            Seed = 2,
            NumBins = 10,
            EnableDifferentialPrivacy = false,
            LaplaceSmoothing = 0.0,
            DomainLowerBounds = new Vector<double>(new[] { 0.0, 0.0 }),
            DomainUpperBounds = new Vector<double>(new[] { 10.0, 10.0 })
        });
        generator.Fit(data, columns, 1);
        var m = generator.Generate(100);

        for (int i = 0; i < m.Rows; i++)
            for (int j = 0; j < m.Columns; j++)
                Assert.InRange(m[i, j], 0.0, 1.0);
    }

    [Fact]
    public void DomainBounds_WithWrongLength_Throw()
    {
        var (data, columns) = CategoricalData(20, 3);
        var generator = new BayesianNetworkSynthGenerator<double>(new BayesianNetworkSynthOptions<double>
        {
            DomainLowerBounds = new Vector<double>(new[] { 0.0 })
        });
        Assert.Throws<ArgumentException>(() => generator.Fit(data, columns, 1));
    }
}
