using System;
using AiDotNet.Attributes;
using AiDotNet.Enums;

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
/// Training uses the margin-based ranking loss max(0, margin + d_pos - d_neg) with d the L2 norm,
/// relations normalized once at initialization and entities renormalized before every mini-batch.
/// </para>
/// <para><b>For Beginners:</b> Imagine entities as points on a map and relations as directions.
/// "Paris" + "capital_of" should point to "France". If the model learns good vectors,
/// you can predict that "Berlin" + "capital_of" ≈ "Germany".
/// </para>
/// </remarks>
[ComponentType(ComponentType.DocumentStore)]
[PipelineStage(PipelineStage.Indexing)]
public class TransEEmbedding<T> : KGEmbeddingBase<T>
{
    /// <inheritdoc />
    public override bool IsDistanceBased => true;

    private protected override T ScoreTripleInternal(int headIdx, int relationIdx, int tailIdx)
        => Engine.Norm(Residual(headIdx, relationIdx, tailIdx));

    /// <summary>
    /// Relations are normalized once at initialization (Bordes et al. 2013, Algorithm 1, line 2).
    /// </summary>
    private protected override void OnInitialize(KGEmbeddingOptions options, Random rng, KnowledgeGraph<T> graph)
    {
        foreach (var emb in _relationEmbeddings)
        {
            NormalizeL2(emb);
        }
    }

    /// <summary>
    /// Entities are renormalized at the start of every mini-batch (Bordes et al. 2013, Algorithm 1, line 5).
    /// </summary>
    private protected override void OnBatchStart()
    {
        foreach (var emb in _entityEmbeddings)
        {
            NormalizeL2(emb);
        }
    }

    private protected override double ComputeLossAndUpdateGradients(
        int posHead, int relation, int posTail,
        int negHead, int negTail,
        double learningRate, KGEmbeddingOptions options)
    {
        double margin = options.GetEffectiveMargin();

        // Bordes et al. 2013: loss = [margin + d(h + r, t) - d(h' + r, t')]_+ with d the L2 norm.
        var posDiff = Residual(posHead, relation, posTail);
        var negDiff = Residual(negHead, relation, negTail);
        T posDist = Engine.Norm(posDiff);
        T negDist = Engine.Norm(negDiff);

        double loss = Math.Max(0.0, margin + NumOps.ToDouble(posDist) - NumOps.ToDouble(negDist));
        if (loss <= 0.0) return 0.0;

        // d||x||/dx = x / ||x||; every step is formed from the pre-update embeddings.
        T lr = NumOps.FromDouble(learningRate);
        var posStep = Engine.Multiply(UnitDirection(posDiff, posDist), lr);
        var negStep = Engine.Multiply(UnitDirection(negDiff, negDist), lr);

        SubtractFrom(_entityEmbeddings[posHead], posStep);
        AddTo(_entityEmbeddings[posTail], posStep);
        SubtractFrom(_relationEmbeddings[relation], Engine.Subtract(posStep, negStep));
        AddTo(_entityEmbeddings[negHead], negStep);
        SubtractFrom(_entityEmbeddings[negTail], negStep);

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

    /// <summary>h + r - t.</summary>
    private Vector<T> Residual(int headIdx, int relationIdx, int tailIdx)
        => Engine.Subtract(
            Engine.Add(Row(_entityEmbeddings[headIdx]), Row(_relationEmbeddings[relationIdx])),
            Row(_entityEmbeddings[tailIdx]));
}
