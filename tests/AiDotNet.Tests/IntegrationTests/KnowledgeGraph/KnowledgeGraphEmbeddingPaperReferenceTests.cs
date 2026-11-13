using System;
using AiDotNet.RetrievalAugmentedGeneration.Graph.Embeddings;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.KnowledgeGraph;

/// <summary>
/// One scoring / SGD step of the knowledge-graph embedding models checked against their papers.
/// </summary>
public class KnowledgeGraphEmbeddingPaperReferenceTests
{
    private const double Tolerance = 1e-12;

    /// <summary>
    /// Bordes et al. 2013: loss = [γ + d(h + r, t) - d(h' + r, t')]_+ with d the L2 norm.
    /// h = t = 0, r = (3, 0), t' = (3, 3.5): d_pos = 3, d_neg = 3.5, so the loss is 1 + 3 - 3.5 = 0.5.
    /// Master used squared distances (1 + 9 - 12.25 &lt; 0) and skipped the update.
    /// The L2 gradient is the unit residual, so every embedding moves by exactly lr along it.
    /// </summary>
    [Fact]
    public void TransE_Step_UsesL2NormMarginLoss()
    {
        var model = new TransEEmbedding<double>();
        var entities = new[] { new[] { 0.0, 0.0 }, new[] { 0.0, 0.0 }, new[] { 3.0, 3.5 } };
        var relations = new[] { new[] { 3.0, 0.0 } };
        model.LoadEmbeddings(entities, relations, 2);

        double loss = model.TrainingStep(0, 0, 1, 0, 2, 0.1, new KGEmbeddingOptions { Margin = 1.0 });

        Assert.Equal(0.5, loss, Tolerance);

        // pos unit residual (1, 0), neg unit residual (0, -1); entity 0 is both heads.
        AssertRow(new[] { -0.1, -0.1 }, entities[0]);
        AssertRow(new[] { 0.1, 0.0 }, entities[1]);
        AssertRow(new[] { 3.0, 3.6 }, entities[2]);
        AssertRow(new[] { 2.9, -0.1 }, relations[0]);
    }

    /// <summary>
    /// Sun et al. 2019, Eq. 3: d_r(h, t) = ||h ∘ r - t||_1 = Σ_k |h_k r_k - t_k|. With h = (1, i),
    /// r = (e^{iπ/2}, 1), t = 0 both moduli are 1, so the distance is 2; master returned the L2 norm √2.
    /// </summary>
    [Fact]
    public void RotatE_Score_IsSumOfComplexModuli()
    {
        var model = new RotatEEmbedding<double>();
        model.LoadEmbeddings(Entities(), new[] { new[] { Math.PI / 2.0, 0.0 } }, 2);

        Assert.Equal(2.0, model.ScoreTriple("e0", "r0", "e1"), Tolerance);
    }

    /// <summary>
    /// Sun et al. 2019, Eq. 4 with one uniformly weighted negative:
    /// L = -log σ(γ - d_pos) - log σ(d_neg - γ). With d_pos = d_neg = 2 and γ = 1 that is
    /// softplus(1) + softplus(-1); master's margin loss returned max(0, 1 + 4 - 4) = 1.
    /// </summary>
    [Fact]
    public void RotatE_Step_UsesLogSigmoidLoss()
    {
        var model = new RotatEEmbedding<double>();
        model.LoadEmbeddings(Entities(), new[] { new[] { Math.PI / 2.0, 0.0 } }, 2);

        double loss = model.TrainingStep(0, 0, 1, 0, 1, 0.0, new KGEmbeddingOptions { Margin = 1.0 });

        double expected = Math.Log(1.0 + Math.E) + Math.Log(1.0 + Math.Exp(-1.0));
        Assert.Equal(expected, loss, Tolerance);
    }

    // [re_0, re_1, im_0, im_1]: h = (1, i), t = 0
    private static double[][] Entities() => new[] { new[] { 1.0, 0.0, 0.0, 1.0 }, new[] { 0.0, 0.0, 0.0, 0.0 } };

    private static void AssertRow(double[] expected, double[] actual)
    {
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++)
        {
            Assert.Equal(expected[i], actual[i], Tolerance);
        }
    }
}
