using System;
using System.Collections.Generic;
using System.Linq;
using AiDotNet.Helpers;

namespace AiDotNet.RetrievalAugmentedGeneration.Retrievers
{
    /// <summary>
    /// TF-IDF (Term Frequency-Inverse Document Frequency) retrieval strategy.
    /// </summary>
    /// <typeparam name="T">The numeric type for vector operations.</typeparam>
    public class TFIDFRetriever<T> : RetrieverBase<T>
    {
        private readonly INumericOperations<T> _numOps;
        private readonly List<Document<T>> _documents = new List<Document<T>>();
        private readonly Dictionary<string, Dictionary<string, T>> _tfidf = new Dictionary<string, Dictionary<string, T>>();
        private readonly Dictionary<string, T> _idf = new Dictionary<string, T>();

        /// <summary>
        /// Initializes a new instance of the <see cref="TFIDFRetriever{T}"/> class.
        /// </summary>
        /// <param name="numericOperations">The numeric operations for type T.</param>
        public TFIDFRetriever(INumericOperations<T> numericOperations) : base(numericOperations)
        {
            _numOps = numericOperations ?? throw new ArgumentNullException(nameof(numericOperations));
        }

        /// <summary>
        /// Adds a document to the retriever and updates TF-IDF scores.
        /// </summary>
        /// <param name="document">The document to add.</param>
        public void AddDocument(Document<T> document)
        {
            if (document == null) throw new ArgumentNullException(nameof(document));

            _documents.Add(document);
            ComputeTFIDF(document);
        }

        /// <summary>
        /// Retrieves documents using TF-IDF scoring.
        /// </summary>
        /// <param name="query">The query string.</param>
        /// <param name="topK">The number of documents to retrieve.</param>
        /// <returns>A list of the most relevant documents.</returns>
        public override List<Document<T>> Retrieve(string query, int topK)
        {
            if (string.IsNullOrEmpty(query)) throw new ArgumentNullException(nameof(query));
            if (topK <= 0) throw new ArgumentOutOfRangeException(nameof(topK));

            var queryTerms = Tokenize(query);
            var scores = new Dictionary<string, T>();

            foreach (var doc in _documents)
            {
                var score = _numOps.Zero;

                if (_tfidf.ContainsKey(doc.Id))
                {
                    foreach (var term in queryTerms)
                    {
                        if (_tfidf[doc.Id].ContainsKey(term))
                        {
                            score = _numOps.Add(score, _tfidf[doc.Id][term]);
                        }
                    }
                }

                scores[doc.Id] = score;
            }

            var results = _documents
                .Where(d => scores.ContainsKey(d.Id))
                .OrderByDescending(d => _numOps.ToDouble(scores[d.Id]))
                .Take(topK)
                .ToList();

            foreach (var doc in results)
            {
                doc.RelevanceScore = scores[doc.Id];
            }

            return results;
        }

        private void ComputeTFIDF(Document<T> document)
        {
            var terms = Tokenize(document.Content);
            var termFrequency = new Dictionary<string, int>();

            foreach (var term in terms)
            {
                if (!termFrequency.ContainsKey(term))
                {
                    termFrequency[term] = 0;
                }
                termFrequency[term]++;
            }

            _tfidf[document.Id] = new Dictionary<string, T>();

            foreach (var kvp in termFrequency)
            {
                if (!_idf.ContainsKey(kvp.Key))
                {
                    var docCount = 0;
                    foreach (var doc in _documents)
                    {
                        if (Tokenize(doc.Content).Contains(kvp.Key))
                        {
                            docCount++;
                        }
                    }
                    _idf[kvp.Key] = _numOps.FromDouble(Math.Log((double)_documents.Count / (docCount + 1)));
                }

                var tf = _numOps.FromDouble((double)kvp.Value / terms.Count);
                _tfidf[document.Id][kvp.Key] = _numOps.Multiply(tf, _idf[kvp.Key]);
            }
        }

        private List<string> Tokenize(string text)
        {
            if (string.IsNullOrEmpty(text)) return new List<string>();

            return text.ToLowerInvariant()
                .Split(new[] { ' ', '\t', '\n', '\r', '.', ',', '!', '?' }, StringSplitOptions.RemoveEmptyEntries)
                .ToList();
        }
    }
}
