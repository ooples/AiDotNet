using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.RetrievalAugmentedGeneration.Retrievers;

/// <summary>
/// Token-level embedder contract for ColBERT-style late interaction.
/// Concrete implementations wrap a pretrained ColBERT / ColBERTv2 / PLAID
/// model (typically loaded from ONNX) and expose two routines: one that
/// embeds the query into <c>[queryTokens, embedDim]</c> and one that
/// embeds a document into <c>[docTokens, embedDim]</c>. ColBERTRetriever
/// then computes the MaxSim score per query token against the document
/// token bank, summing across query tokens for the final relevance score
/// (Khattab &amp; Zaharia 2020 §3.2).
/// </summary>
/// <typeparam name="T">Numeric type for tensor data.</typeparam>
/// <remarks>
/// <para>
/// Splitting query and document embedding into separate methods follows
/// the original ColBERT paper §3.2, which inserts distinct
/// <c>[Q]</c> / <c>[D]</c> marker tokens into the input so the
/// pretrained encoder can specialise its representations for the two
/// roles. Implementations should embed L2-normalised per-token vectors so
/// downstream cosine similarity reduces to a plain dot product.
/// </para>
/// </remarks>
public interface IColBertEmbedder<T>
{
    /// <summary>Embeds a query into <c>[queryTokens, embedDim]</c> L2-normalised vectors.</summary>
    Tensor<T> EmbedQuery(string query);

    /// <summary>Embeds a document into <c>[docTokens, embedDim]</c> L2-normalised vectors.</summary>
    Tensor<T> EmbedDocument(string document);
}
