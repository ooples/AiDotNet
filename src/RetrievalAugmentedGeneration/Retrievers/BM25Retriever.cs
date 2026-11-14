
using System;
using System.Collections.Generic;
using System.Linq;
using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.RetrievalAugmentedGeneration.Models;

namespace AiDotNet.RetrievalAugmentedGeneration.Retrievers
{
    /// <summary>
    /// BM25 (Best Matching 25) retrieval algorithm for sparse keyword-based search.
    /// </summary>
    /// <typeparam name="T">The numeric type for vector operations.</typeparam>
    [ComponentType(ComponentType.Retriever)]
    [PipelineStage(PipelineStage.Retrieval)]
    public class BM25Retriever<T> : RetrieverBase<T>
    {
        private readonly IDocumentStore<T> _documentStore;
        private readonly T _k1;
        private readonly T _b;

        public BM25Retriever(IDocumentStore<T> documentStore, int defaultTopK = 5, double k1 = 1.5, double b = 0.75)
            : base(defaultTopK)
        {
            if (documentStore == null)
                throw new ArgumentNullException(nameof(documentStore));

            _documentStore = documentStore;
            _k1 = NumOps.FromDouble(k1);
            _b = NumOps.FromDouble(b);
        }

        /// <inheritdoc />
        protected override IEnumerable<Document<T>> RetrieveCore(string query, int topK, Dictionary<string, object> metadataFilters)
        {
            var candidates = _documentStore.GetAll().ToList();
            if (candidates.Count == 0)
                return Enumerable.Empty<Document<T>>();

            var scores = ScoreDocuments(Tokenize(query), candidates.Select(d => Tokenize(d.Content)).ToList());

            return candidates
                .Select((doc, index) => (doc, score: scores[index]))
                .Where(x => MatchesFilters(x.doc, metadataFilters))
                .OrderByDescending(x => x.score)
                .Take(topK)
                .Select(x =>
                {
                    x.doc.RelevanceScore = x.score;
                    x.doc.HasRelevanceScore = true;
                    return x.doc;
                })
                .ToList();
        }

        /// <summary>
        /// Okapi BM25 (Robertson and Zaragoza 2009, Eq. 3.15) of every document for the query:
        /// Σ_t qtf_t · idf_t · tf (k1 + 1) / (tf + k1 (1 - b + b · dl / avgdl)),
        /// with idf_t = ln((N - df_t + 0.5) / (df_t + 0.5)) and repeated query terms counted per occurrence.
        /// </summary>
        internal T[] ScoreDocuments(List<string> queryTerms, List<List<string>> documentTerms)
        {
            int documents = documentTerms.Count;
            var vocabulary = queryTerms.Distinct().ToList();
            int terms = vocabulary.Count;
            if (documents == 0)
                return Array.Empty<T>();
            if (terms == 0)
                return new T[documents];

            var termIndex = vocabulary.Select((term, i) => (term, i)).ToDictionary(x => x.term, x => x.i);
            var termCounts = new int[documents * terms];
            var documentFrequency = new int[terms];
            var lengths = new int[documents];
            for (int d = 0; d < documents; d++)
            {
                lengths[d] = documentTerms[d].Count;
                foreach (var term in documentTerms[d])
                {
                    if (termIndex.TryGetValue(term, out var t))
                    {
                        if (termCounts[d * terms + t]++ == 0)
                            documentFrequency[t]++;
                    }
                }
            }

            var queryCounts = new int[terms];
            foreach (var term in queryTerms)
                queryCounts[termIndex[term]]++;

            // idf_t = ln((N - df + 0.5) / (df + 0.5)), skipped (0) for terms no document contains.
            var half = NumOps.FromDouble(0.5);
            var df = new Tensor<int>(documentFrequency, [terms]).Cast<T>();
            var idf = Engine.TensorLog(Engine.TensorDivide(
                Engine.ScalarMinusTensor(NumOps.Add(NumOps.FromDouble(documents), half), df),
                Engine.TensorAddScalar(df, half)));
            idf = Engine.TensorWhere(Engine.TensorGreaterThan(df, NumOps.Zero), idf, new Tensor<T>([terms]));
            var termWeights = Engine.TensorMultiply(idf, new Tensor<int>(queryCounts, [terms]).Cast<T>());

            // Length normalisation k1 (1 - b + b dl / avgdl); avgdl falls back to 1 for an all-empty corpus.
            var dl = new Tensor<int>(lengths, [documents]).Cast<T>();
            var avgdl = Engine.TensorMean(dl);
            if (!NumOps.GreaterThan(avgdl, NumOps.Zero))
                avgdl = NumOps.One;
            var lengthNorm = Engine.TensorMultiplyScalar(
                Engine.TensorAddScalar(
                    Engine.TensorMultiplyScalar(Engine.TensorDivideScalar(dl, avgdl), _b), NumOps.Subtract(NumOps.One, _b)),
                _k1);

            var tf = new Tensor<int>(termCounts, [documents, terms]).Cast<T>();
            var saturation = Engine.TensorDivide(
                Engine.TensorMultiplyScalar(tf, NumOps.Add(_k1, NumOps.One)),
                Engine.TensorAdd(tf, Engine.TensorBroadcastTo(Engine.Reshape(lengthNorm, [documents, 1]), [documents, terms])));

            return Engine.TensorMatMul(saturation, Engine.Reshape(termWeights, [terms, 1])).ToArray();
        }

        internal static List<string> Tokenize(string text)
        {
            if (string.IsNullOrWhiteSpace(text))
                return new List<string>();

            return text.ToLowerInvariant()
                .Split(new[] { ' ', '\t', '\n', '\r', ',', '.', '!', '?' }, StringSplitOptions.RemoveEmptyEntries)
                .ToList();
        }

        private bool MatchesFilters(Document<T> document, Dictionary<string, object> filters)
        {
            if (filters.Count == 0)
                return true;

            foreach (var filter in filters)
            {
                if (!document.Metadata.TryGetValue(filter.Key, out var value))
                    return false;

                if (!filter.Value.Equals(value))
                    return false;
            }

            return true;
        }
    }
}
