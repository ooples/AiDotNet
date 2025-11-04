using AiDotNet.LinearAlgebra;
using AiDotNet.RetrievalAugmentedGeneration.Models;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;

namespace AiDotNet.RetrievalAugmentedGeneration.RerankingStrategies
{
    /// <summary>
    /// Reranking that places most relevant documents in middle positions to avoid "lost in the middle" effect
    /// </summary>
    /// <typeparam name="T">The numeric type for scoring</typeparam>
    public class LostInTheMiddleReranker<T> : RerankingStrategyBase<T> where T : struct, IComparable, IComparable<T>, IConvertible, IEquatable<T>, IFormattable
    {
        protected override Task<List<Document<T>>> RerankCoreAsync(string query, List<Document<T>> documents, int topK)
        {
            if (documents == null || documents.Count == 0)
                return Task.FromResult(new List<Document<T>>());

            var docsToUse = documents.Take(topK).ToList();
            if (docsToUse.Count <= 2)
                return Task.FromResult(docsToUse);

            var reordered = new List<Document<T>>();
            var remaining = new List<Document<T>>(docsToUse);

            var positions = new List<int>();
            for (int i = 0; i < docsToUse.Count; i++)
            {
                if (i % 2 == 0)
                {
                    positions.Add(i);
                }
                else
                {
                    positions.Insert(0, i);
                }
            }

            foreach (var pos in positions)
            {
                if (pos < docsToUse.Count)
                {
                    reordered.Add(docsToUse[pos]);
                }
            }

            return Task.FromResult(reordered);
        }
    }
}
