using System;

namespace AiDotNet.RetrievalAugmentedGeneration.Graph.Embeddings;

/// <summary>
/// DistMult embedding model: bilinear diagonal scoring with Σ(h_k · r_k · t_k).
/// </summary>
/// <typeparam name="T">The numeric type used for embedding calculations.</typeparam>
/// <remarks>
/// <para>
/// DistMult (Yang et al., 2015) uses a diagonal bilinear form to score triples.
/// Score: Σ_k (h_k · r_k · t_k). Higher scores indicate more plausible triples.
/// Training uses logistic loss with L2 regularization on embeddings.
/// </para>
/// <para>
/// DistMult is inherently symmetric: score(h, r, t) = score(t, r, h), making it
/// best suited for symmetric relations like "similar_to" or "married_to".
/// </para>
/// <para><b>For Beginners:</b> DistMult is the simplest bilinear model:
/// - Each entity and relation gets a vector
/// - Score = element-wise product of head, relation, and tail vectors, summed up
/// - Higher score = more likely to be a true fact
/// - Works best for symmetric relations where direction doesn't matter
/// </para>
/// </remarks>
public class DistMultEmbedding<T> : KGEmbeddingBase<T>
{
    /// <inheritdoc />
    public override bool IsDistanceBased => false;

    private protected override T ScoreTripleInternal(int headIdx, int relationIdx, int tailIdx)
    {
        var h = _entityEmbeddings[headIdx];
        var r = _relationEmbeddings[relationIdx];
        var t = _entityEmbeddings[tailIdx];
        int dim = EmbeddingDimension;

        // Σ(h_k · r_k · t_k)
        T score = NumOps.Zero;
        for (int d = 0; d < dim; d++)
        {
            score = NumOps.Add(score, NumOps.Multiply(NumOps.Multiply(h[d], r[d]), t[d]));
        }

        return score;
    }

    private protected override double ComputeLossAndUpdateGradients(
        int posHead, int relation, int posTail,
        int negHead, int negTail,
        double learningRate, KGEmbeddingOptions options)
    {
        int dim = EmbeddingDimension;

        double posScore = NumOps.ToDouble(ScoreTripleInternal(posHead, relation, posTail));
        double negScore = NumOps.ToDouble(ScoreTripleInternal(negHead, relation, negTail));

        // Logistic loss: -log(σ(posScore)) - log(σ(-negScore))
        double posLoss = Math.Log(1.0 + Math.Exp(-posScore));
        double negLoss = Math.Log(1.0 + Math.Exp(negScore));
        double loss = posLoss + negLoss;

        // Gradient scales
        double posGradScale = -1.0 / (1.0 + Math.Exp(posScore));
        double negGradScale = 1.0 / (1.0 + Math.Exp(-negScore));

        // Update positive triple
        UpdateTripleGradients(posHead, relation, posTail, dim, learningRate * posGradScale);
        // Update negative triple
        UpdateTripleGradients(negHead, relation, negTail, dim, learningRate * negGradScale);

        return loss;
    }

    private void UpdateTripleGradients(int headIdx, int relationIdx, int tailIdx, int dim, double step)
    {
        var h = _entityEmbeddings[headIdx];
        var r = _relationEmbeddings[relationIdx];
        var t = _entityEmbeddings[tailIdx];

        for (int d = 0; d < dim; d++)
        {
            double hd = NumOps.ToDouble(h[d]);
            double rd = NumOps.ToDouble(r[d]);
            double td = NumOps.ToDouble(t[d]);

            // Gradients: d/dh = r*t, d/dr = h*t, d/dt = h*r
            h[d] = NumOps.FromDouble(hd - step * rd * td);
            r[d] = NumOps.FromDouble(rd - step * hd * td);
            t[d] = NumOps.FromDouble(td - step * hd * rd);
        }
    }
}
