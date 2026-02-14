using System.Collections.Generic;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Interfaces;

/// <summary>
/// Exposes token embedding lookup for models that maintain a token embedding table.
/// </summary>
/// <typeparam name="T">The numeric type used for computations.</typeparam>
[AiDotNet.Configuration.YamlConfigurable("TokenEmbedding")]
public interface ITokenEmbedding<T>
{
    /// <summary>
    /// Retrieves embeddings for the provided token IDs.
    /// </summary>
    /// <param name="tokenIds">Token IDs to lookup.</param>
    /// <returns>A matrix where each row corresponds to a token embedding.</returns>
    Matrix<T> GetTokenEmbeddings(IReadOnlyList<int> tokenIds);
}
