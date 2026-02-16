using System;

namespace AiDotNet.RetrievalAugmentedGeneration.Graph.Embeddings;

/// <summary>
/// ComplEx embedding model: uses complex-valued embeddings with Hermitian dot product scoring.
/// </summary>
/// <typeparam name="T">The numeric type used for embedding calculations.</typeparam>
/// <remarks>
/// <para>
/// ComplEx (Trouillon et al., 2016) represents entities and relations as complex vectors.
/// Score: Re(⟨h, r, conj(t)⟩) = Σ(hRe·rRe·tRe + hRe·rImag·tImag + hImag·rRe·tImag - hImag·rImag·tRe).
/// Training uses logistic loss with optional N3 regularization.
/// Higher scores indicate more plausible triples.
/// </para>
/// <para>
/// Complex numbers are represented as paired real/imaginary T[] arrays (length 2*dim each)
/// to maintain generic T type parameter compatibility.
/// </para>
/// <para><b>For Beginners:</b> ComplEx is particularly good at modeling:
/// - Symmetric relations: "married_to" (score(A,r,B) = score(B,r,A))
/// - Antisymmetric relations: "parent_of" (score(A,r,B) ≠ score(B,r,A))
/// It achieves this by using complex numbers, which naturally distinguish direction.
/// </para>
/// </remarks>
public class ComplExEmbedding<T> : KGEmbeddingBase<T>
{
    /// <inheritdoc />
    public override bool IsDistanceBased => false;

    private protected override int GetEntityEmbeddingSize() => EmbeddingDimension * 2;

    private protected override int GetRelationEmbeddingSize() => EmbeddingDimension * 2;

    // No OnInitialize needed — base class allocates entity embeddings at GetEntityEmbeddingSize() (2*dim)
    // and relation embeddings at GetRelationEmbeddingSize() (2*dim) automatically.

    private protected override T ScoreTripleInternal(int headIdx, int relationIdx, int tailIdx)
    {
        var h = _entityEmbeddings[headIdx];
        var r = _relationEmbeddings[relationIdx];
        var t = _entityEmbeddings[tailIdx];
        int dim = EmbeddingDimension;

        // Re(⟨h, r, conj(t)⟩) per Trouillon et al. (2016)
        // = Σ(hRe*rRe*tRe + hRe*rIm*tIm + hIm*rRe*tIm - hIm*rIm*tRe)
        double score = 0.0;
        for (int d = 0; d < dim; d++)
        {
            double hRe = NumOps.ToDouble(h[d]);
            double hIm = NumOps.ToDouble(h[d + dim]);
            double rRe = NumOps.ToDouble(r[d]);
            double rIm = NumOps.ToDouble(r[d + dim]);
            double tRe = NumOps.ToDouble(t[d]);
            double tIm = NumOps.ToDouble(t[d + dim]);

            score += hRe * rRe * tRe + hRe * rIm * tIm + hIm * rRe * tIm - hIm * rIm * tRe;
        }

        return NumOps.FromDouble(score);
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
        // = log(1 + exp(-posScore)) + log(1 + exp(negScore))
        double posLoss = Math.Log(1.0 + Math.Exp(-posScore));
        double negLoss = Math.Log(1.0 + Math.Exp(negScore));
        double loss = posLoss + negLoss;

        // Gradient of logistic loss:
        // d/dx log(1+exp(-x)) = -σ(-x) = -(1/(1+exp(x)))
        // d/dx log(1+exp(x)) = σ(x) = 1/(1+exp(-x))
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
            double hRe = NumOps.ToDouble(h[d]);
            double hIm = NumOps.ToDouble(h[d + dim]);
            double rRe = NumOps.ToDouble(r[d]);
            double rIm = NumOps.ToDouble(r[d + dim]);
            double tRe = NumOps.ToDouble(t[d]);
            double tIm = NumOps.ToDouble(t[d + dim]);

            // Gradients of score w.r.t. each component
            // score = hRe*rRe*tRe + hRe*rIm*tIm + hIm*rRe*tIm - hIm*rIm*tRe
            double gradHRe = rRe * tRe + rIm * tIm;
            double gradHIm = rRe * tIm - rIm * tRe;
            double gradRRe = hRe * tRe + hIm * tIm;
            double gradRIm = hRe * tIm - hIm * tRe;
            double gradTRe = hRe * rRe - hIm * rIm;
            double gradTIm = hRe * rIm + hIm * rRe;

            h[d] = NumOps.FromDouble(hRe - step * gradHRe);
            h[d + dim] = NumOps.FromDouble(hIm - step * gradHIm);
            r[d] = NumOps.FromDouble(rRe - step * gradRRe);
            r[d + dim] = NumOps.FromDouble(rIm - step * gradRIm);
            t[d] = NumOps.FromDouble(tRe - step * gradTRe);
            t[d + dim] = NumOps.FromDouble(tIm - step * gradTIm);
        }
    }
}
