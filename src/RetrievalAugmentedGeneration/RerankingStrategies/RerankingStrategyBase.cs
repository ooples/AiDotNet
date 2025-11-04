using System;
using AiDotNet.RetrievalAugmentedGeneration.Models;
using System.Collections.Generic;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;

namespace AiDotNet.RetrievalAugmentedGeneration.RerankingStrategies
{
    /// <summary>
    /// Base class for reranking strategies.
    /// </summary>
    /// <typeparam name="T">The numeric type for vector operations.</typeparam>
    public abstract class RerankingStrategyBase<T>
    {
        /// <summary>
        /// Provides mathematical operations for the numeric type T.
        /// </summary>
        protected static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

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
