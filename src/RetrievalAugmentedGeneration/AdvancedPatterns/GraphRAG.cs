using AiDotNet.Interfaces;
using AiDotNet.RetrievalAugmentedGeneration.Interfaces;
using AiDotNet.RetrievalAugmentedGeneration.Retrievers;
using AiDotNet.RetrievalAugmentedGeneration.Models;
using System.Collections.Generic;
using System.Linq;

namespace AiDotNet.RetrievalAugmentedGeneration.AdvancedPatterns
{
    public class GraphRAG<T> : RetrieverBase<T> where T : struct
    {
        private readonly IRetriever<T> _baseRetriever;
        private readonly Dictionary<string, List<string>> _documentGraph;
        private readonly int _traversalDepth;
        
        protected override INumericOperations<T> NumOps { get; }

        public GraphRAG(IRetriever<T> baseRetriever, int traversalDepth = 2)
        {
            _baseRetriever = baseRetriever ?? throw new System.ArgumentNullException(nameof(baseRetriever));
            _documentGraph = new Dictionary<string, List<string>>();
            _traversalDepth = traversalDepth > 0 ? traversalDepth : throw new System.ArgumentOutOfRangeException(nameof(traversalDepth));
            NumOps = NumericOperationsFactory.GetOperations<T>();
        }

        public void AddEdge(string sourceDocId, string targetDocId)
        {
            if (!_documentGraph.ContainsKey(sourceDocId))
            {
                _documentGraph[sourceDocId] = new List<string>();
            }
            
            if (!_documentGraph[sourceDocId].Contains(targetDocId))
            {
                _documentGraph[sourceDocId].Add(targetDocId);
            }
        }

        protected override List<Document<T>> RetrieveCore(string query, int topK)
        {
            var initialResults = _baseRetriever.Retrieve(query, topK);
            var expandedResults = new Dictionary<string, Document<T>>();

            foreach (var doc in initialResults)
            {
                expandedResults[doc.Id] = doc;
                TraverseGraph(doc.Id, _traversalDepth, expandedResults);
            }

            return expandedResults.Values
                .OrderByDescending(d => NumOps.ToDouble(d.RelevanceScore))
                .Take(topK)
                .ToList();
        }

        private void TraverseGraph(string docId, int depth, Dictionary<string, Document<T>> results)
        {
            if (depth <= 0 || !_documentGraph.ContainsKey(docId))
            {
                return;
            }

            foreach (var neighborId in _documentGraph[docId])
            {
                if (!results.ContainsKey(neighborId))
                {
                    var neighborDocs = _baseRetriever.Retrieve(neighborId, 1);
                    if (neighborDocs.Count > 0)
                    {
                        results[neighborId] = neighborDocs[0];
                        TraverseGraph(neighborId, depth - 1, results);
                    }
                }
            }
        }
    }
}

