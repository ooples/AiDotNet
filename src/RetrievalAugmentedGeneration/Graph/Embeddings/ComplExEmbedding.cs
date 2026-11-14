using System;
using AiDotNet.Attributes;
using AiDotNet.Enums;

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
[ComponentType(ComponentType.DocumentStore)]
[PipelineStage(PipelineStage.Indexing)]
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
        // Re(⟨h, r, conj(t)⟩) per Trouillon et al. (2016)
        // = Σ(hRe*rRe*tRe + hRe*rIm*tIm + hIm*rRe*tIm - hIm*rIm*tRe)
        var p = new ComplexParts(this, headIdx, relationIdx, tailIdx);
        return Engine.Sum(Engine.Add(
            Engine.Multiply(p.HeadReal, p.GradHeadReal()), Engine.Multiply(p.HeadImag, p.GradHeadImag())));
    }

    private protected override double ComputeLossAndUpdateGradients(
        int posHead, int relation, int posTail,
        int negHead, int negTail,
        double learningRate, KGEmbeddingOptions options)
    {
        double posScore = NumOps.ToDouble(ScoreTripleInternal(posHead, relation, posTail));
        double negScore = NumOps.ToDouble(ScoreTripleInternal(negHead, relation, negTail));

        // Logistic loss: -log(σ(posScore)) - log(σ(-negScore))
        // Using numerically stable softplus: log(1 + exp(x)) = x + log(1 + exp(-x)) when x > 0
        double posLoss = StableSoftplus(-posScore);
        double negLoss = StableSoftplus(negScore);
        double loss = posLoss + negLoss;

        // Gradient scales using stable sigmoid
        double posGradScale = -StableSigmoid(-posScore);
        double negGradScale = StableSigmoid(negScore);

        // Update positive triple
        UpdateTripleGradients(posHead, relation, posTail, learningRate * posGradScale);
        // Update negative triple
        UpdateTripleGradients(negHead, relation, negTail, learningRate * negGradScale);

        return loss;
    }

    private void UpdateTripleGradients(int headIdx, int relationIdx, int tailIdx, double step)
    {
        var p = new ComplexParts(this, headIdx, relationIdx, tailIdx);
        T scale = NumOps.FromDouble(step);

        // Gradients of score = hRe*rRe*tRe + hRe*rIm*tIm + hIm*rRe*tIm - hIm*rIm*tRe w.r.t. each part,
        // all formed from the pre-update embeddings.
        var gradHRe = Engine.Multiply(p.GradHeadReal(), scale);
        var gradHIm = Engine.Multiply(p.GradHeadImag(), scale);
        var gradRRe = Engine.Multiply(Engine.Add(
            Engine.Multiply(p.HeadReal, p.TailReal), Engine.Multiply(p.HeadImag, p.TailImag)), scale);
        var gradRIm = Engine.Multiply(Engine.Subtract(
            Engine.Multiply(p.HeadReal, p.TailImag), Engine.Multiply(p.HeadImag, p.TailReal)), scale);
        var gradTRe = Engine.Multiply(Engine.Subtract(
            Engine.Multiply(p.HeadReal, p.RelationReal), Engine.Multiply(p.HeadImag, p.RelationImag)), scale);
        var gradTIm = Engine.Multiply(Engine.Add(
            Engine.Multiply(p.HeadReal, p.RelationImag), Engine.Multiply(p.HeadImag, p.RelationReal)), scale);

        int dim = EmbeddingDimension;
        var h = _entityEmbeddings[headIdx];
        var r = _relationEmbeddings[relationIdx];
        var t = _entityEmbeddings[tailIdx];
        SubtractFrom(h, gradHRe, 0);
        SubtractFrom(h, gradHIm, dim);
        SubtractFrom(r, gradRRe, 0);
        SubtractFrom(r, gradRIm, dim);
        SubtractFrom(t, gradTRe, 0);
        SubtractFrom(t, gradTIm, dim);
    }

    /// <summary>Views of the real and imaginary halves of one triple's embeddings.</summary>
    private sealed class ComplexParts
    {
        public ComplexParts(ComplExEmbedding<T> model, int headIdx, int relationIdx, int tailIdx)
        {
            int dim = model.EmbeddingDimension;
            var h = model._entityEmbeddings[headIdx];
            var r = model._relationEmbeddings[relationIdx];
            var t = model._entityEmbeddings[tailIdx];
            HeadReal = Row(h, 0, dim);
            HeadImag = Row(h, dim, dim);
            RelationReal = Row(r, 0, dim);
            RelationImag = Row(r, dim, dim);
            TailReal = Row(t, 0, dim);
            TailImag = Row(t, dim, dim);
        }

        public Vector<T> HeadReal { get; }
        public Vector<T> HeadImag { get; }
        public Vector<T> RelationReal { get; }
        public Vector<T> RelationImag { get; }
        public Vector<T> TailReal { get; }
        public Vector<T> TailImag { get; }

        /// <summary>d score / d hRe = rRe*tRe + rIm*tIm.</summary>
        public Vector<T> GradHeadReal()
            => Engine.Add(Engine.Multiply(RelationReal, TailReal), Engine.Multiply(RelationImag, TailImag));

        /// <summary>d score / d hIm = rRe*tIm - rIm*tRe.</summary>
        public Vector<T> GradHeadImag()
            => Engine.Subtract(Engine.Multiply(RelationReal, TailImag), Engine.Multiply(RelationImag, TailReal));
    }

    /// <summary>
    /// Numerically stable softplus: log(1 + exp(x)).
    /// </summary>
    private static double StableSoftplus(double x)
    {
        if (x > 20.0) return x;
        if (x < -20.0) return Math.Exp(x);
        return Math.Log(1.0 + Math.Exp(x));
    }

    /// <summary>
    /// Numerically stable sigmoid: 1 / (1 + exp(-x)).
    /// </summary>
    private static double StableSigmoid(double x)
    {
        if (x >= 0.0)
        {
            double ez = Math.Exp(-x);
            return 1.0 / (1.0 + ez);
        }
        else
        {
            double ez = Math.Exp(x);
            return ez / (1.0 + ez);
        }
    }
}
