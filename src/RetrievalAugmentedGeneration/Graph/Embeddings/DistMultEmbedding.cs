using System;
using AiDotNet.Attributes;
using AiDotNet.Enums;

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
[ComponentType(ComponentType.DocumentStore)]
[PipelineStage(PipelineStage.Indexing)]
public class DistMultEmbedding<T> : KGEmbeddingBase<T>
{
    /// <inheritdoc />
    public override bool IsDistanceBased => false;

    private protected override T ScoreTripleInternal(int headIdx, int relationIdx, int tailIdx)
        {
        // Σ(h_k · r_k · t_k)
        return Engine.Sum(Engine.Multiply(
            Engine.Multiply(Row(_entityEmbeddings[headIdx]), Row(_relationEmbeddings[relationIdx])),
            Row(_entityEmbeddings[tailIdx])));
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
        var h = _entityEmbeddings[headIdx];
        var r = _relationEmbeddings[relationIdx];
        var t = _entityEmbeddings[tailIdx];
        T scale = NumOps.FromDouble(step);

        // Gradients: d/dh = r*t, d/dr = h*t, d/dt = h*r, all from the pre-update embeddings.
        var gradH = Engine.Multiply(Engine.Multiply(Row(r), Row(t)), scale);
        var gradR = Engine.Multiply(Engine.Multiply(Row(h), Row(t)), scale);
        var gradT = Engine.Multiply(Engine.Multiply(Row(h), Row(r)), scale);

        SubtractFrom(h, gradH);
        SubtractFrom(r, gradR);
        SubtractFrom(t, gradT);
    }

    /// <summary>
    /// Numerically stable softplus: log(1 + exp(x)).
    /// For large positive x, returns x directly to avoid overflow.
    /// For large negative x, returns exp(x) to avoid log(1+0) precision loss.
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
