using System;

namespace AiDotNet.RetrievalAugmentedGeneration.Graph.Embeddings;

/// <summary>
/// TransE embedding model: entities and relations are vectors in the same space,
/// with the scoring function d(h, r, t) = ||h + r - t||.
/// </summary>
/// <typeparam name="T">The numeric type used for embedding calculations.</typeparam>
/// <remarks>
/// <para>
/// TransE (Bordes et al., 2013) models relations as translations in embedding space.
/// For a valid triple (h, r, t), the model learns embeddings such that h + r ≈ t.
/// Training uses margin-based ranking loss: max(0, margin + d_pos - d_neg).
/// </para>
/// <para><b>For Beginners:</b> Imagine entities as points on a map and relations as directions.
/// "Paris" + "capital_of" should point to "France". If the model learns good vectors,
/// you can predict that "Berlin" + "capital_of" ≈ "Germany".
/// </para>
/// </remarks>
public class TransEEmbedding<T> : KGEmbeddingBase<T>
{
    /// <inheritdoc />
    public override bool IsDistanceBased => true;

    private protected override T ScoreTripleInternal(int headIdx, int relationIdx, int tailIdx)
    {
        var h = _entityEmbeddings[headIdx];
        var r = _relationEmbeddings[relationIdx];
        var t = _entityEmbeddings[tailIdx];
        int dim = EmbeddingDimension;

        // L2 distance: ||h + r - t||
        T sumSq = NumOps.Zero;
        for (int d = 0; d < dim; d++)
        {
            T diff = NumOps.Subtract(NumOps.Add(h[d], r[d]), t[d]);
            sumSq = NumOps.Add(sumSq, NumOps.Multiply(diff, diff));
        }

        return NumOps.Sqrt(sumSq);
    }

    private protected override double ComputeLossAndUpdateGradients(
        int posHead, int relation, int posTail,
        int negHead, int negTail,
        double learningRate, KGEmbeddingOptions options)
    {
        double margin = options.GetEffectiveMargin();
        int dim = EmbeddingDimension;

        var h = _entityEmbeddings[posHead];
        var r = _relationEmbeddings[relation];
        var t = _entityEmbeddings[posTail];
        var nh = _entityEmbeddings[negHead];
        var nt = _entityEmbeddings[negTail];

        // Compute positive distance: ||h + r - t||²
        double posDist = 0.0;
        for (int d = 0; d < dim; d++)
        {
            double diff = NumOps.ToDouble(NumOps.Subtract(NumOps.Add(h[d], r[d]), t[d]));
            posDist += diff * diff;
        }

        // Compute negative distance: ||nh + r - nt||²
        double negDist = 0.0;
        for (int d = 0; d < dim; d++)
        {
            double diff = NumOps.ToDouble(NumOps.Subtract(NumOps.Add(nh[d], r[d]), nt[d]));
            negDist += diff * diff;
        }

        double loss = Math.Max(0.0, margin + posDist - negDist);
        if (loss <= 0.0) return 0.0;

        // Gradient update: push positive closer, negative farther
        // Per Bordes et al. 2013: loss = [margin + d(h+r,t) - d(h'+r,t')]_+
        // Relation r appears in both positive and negative distances,
        // so its gradient is: 2(h+r-t) - 2(h'+r-t')
        T lr = NumOps.FromDouble(learningRate);
        T two = NumOps.FromDouble(2.0);

        for (int d = 0; d < dim; d++)
        {
            // Positive gradient: 2(h + r - t)
            T posGrad = NumOps.Subtract(NumOps.Add(h[d], r[d]), t[d]);
            T scaledPosGrad = NumOps.Multiply(two, posGrad);

            // Negative gradient: 2(nh + r - nt)
            T negGrad = NumOps.Subtract(NumOps.Add(nh[d], r[d]), nt[d]);
            T scaledNegGrad = NumOps.Multiply(two, negGrad);

            // Update positive triple embeddings (decrease distance)
            h[d] = NumOps.Subtract(h[d], NumOps.Multiply(lr, scaledPosGrad));
            t[d] = NumOps.Add(t[d], NumOps.Multiply(lr, scaledPosGrad));

            // Relation gradient accounts for both positive and negative triples:
            // grad_r = 2(h+r-t) - 2(h'+r-t')
            T rGrad = NumOps.Subtract(scaledPosGrad, scaledNegGrad);
            r[d] = NumOps.Subtract(r[d], NumOps.Multiply(lr, rGrad));

            // Update negative triple embeddings (increase distance)
            nh[d] = NumOps.Add(nh[d], NumOps.Multiply(lr, scaledNegGrad));
            nt[d] = NumOps.Subtract(nt[d], NumOps.Multiply(lr, scaledNegGrad));
        }

        return loss;
    }

    /// <summary>
    /// After each epoch, normalize entity embeddings to the unit ball (TransE constraint).
    /// </summary>
    private protected override void OnPostEpoch(int epoch)
    {
        foreach (var emb in _entityEmbeddings)
        {
            NormalizeL2(emb);
        }
    }
}
