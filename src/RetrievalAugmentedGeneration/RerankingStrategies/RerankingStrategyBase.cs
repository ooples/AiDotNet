using System;
using AiDotNet.RetrievalAugmentedGeneration.Models;
using System.Collections.Generic;
using AiDotNet.Helpers;

namespace AiDotNet.RetrievalAugmentedGeneration.RerankingStrategies
{
    /// <summary>
    /// Base class for reranking strategies.
    /// </summary>
    /// <typeparam name="T">The numeric type for vector operations.</typeparam>
    public abstract class RerankingStrategyBase<T>
    {
        protected readonly INumericOperations<T> NumOps;

        /// <summary>
        /// Initializes a new instance of the <see cref="RerankingStrategyBase{T}"/> class.
        /// </summary>
        /// <param name="numericOperations">The numeric operations for type T.</param>
        protected RerankingStrategyBase(INumericOperations<T> numericOperations)
        {
            NumOps = numericOperations ?? throw new ArgumentNullException(nameof(numericOperations));
        }

        /// <summary>
        /// Reranks a list of documents based on their relevance to the query.
        /// </summary>
        /// <param name="query">The query string.</param>
        /// <param name="documents">The documents to rerank.</param>
        /// <param name="topK">The number of top documents to return.</param>
        /// <returns>A reranked list of documents.</returns>
        public abstract List<Document<T>> Rerank(string query, List<Document<T>> documents, int topK);
    }
}
