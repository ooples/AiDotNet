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
    /// Routes each query to the single most appropriate underlying retriever (data source). With a real
    /// text generator the routing is an LLM classification over the source descriptions; without one it
    /// falls back to description/query token overlap. Mirrors the LangChain/LlamaIndex router retriever.
    /// </summary>
    /// <typeparam name="T">The numeric type for vector operations.</typeparam>
    [ComponentType(ComponentType.Retriever)]
    [PipelineStage(PipelineStage.Retrieval)]
    public class QueryRoutingRetriever<T> : RetrieverBase<T>
    {
        /// <summary>A named, described data source backed by a retriever.</summary>
        public sealed class Route
        {
            public string Name { get; }
            public string Description { get; }
            public IRetriever<T> Retriever { get; }

            public Route(string name, string description, IRetriever<T> retriever)
            {
                if (string.IsNullOrWhiteSpace(name)) throw new ArgumentException("Route name is required.", nameof(name));
                Name = name;
                Description = description ?? string.Empty;
                Retriever = retriever ?? throw new ArgumentNullException(nameof(retriever));
            }
        }

        private readonly IReadOnlyList<Route> _routes;
        private readonly ITextGenerator? _generator;

        public QueryRoutingRetriever(IReadOnlyList<Route> routes, ITextGenerator? generator = null, int defaultTopK = 5)
            : base(defaultTopK)
        {
            if (routes == null) throw new ArgumentNullException(nameof(routes));
            if (routes.Count == 0) throw new ArgumentException("At least one route is required.", nameof(routes));
            _routes = routes;
            _generator = generator;
        }

        protected override IEnumerable<Document<T>> RetrieveCore(string query, int topK, Dictionary<string, object> metadataFilters)
            => ChooseRoute(query).Retriever.Retrieve(query, topK, metadataFilters);

        /// <summary>Exposed for testing/inspection: which route a query resolves to.</summary>
        public Route ChooseRoute(string query)
        {
            if (_generator != null)
            {
                var sources = string.Join("\n", _routes.Select(r => $"- {r.Name}: {r.Description}"));
                var prompt =
                    $"Choose the single best data source for the query. Reply with ONLY the source name, exactly " +
                    $"as written, and nothing else.\n\nSources:\n{sources}\n\nQuery: {query}\n\nBest source name:";
                var reply = _generator.Generate(prompt);
                if (!string.IsNullOrWhiteSpace(reply))
                {
                    var match = _routes.FirstOrDefault(r =>
                        reply.IndexOf(r.Name, StringComparison.OrdinalIgnoreCase) >= 0);
                    if (match != null) return match;
                }
                // Unrecognized reply: fall through to the lexical heuristic.
            }

            return ChooseByOverlap(query);
        }

        // Fallback: the route whose name+description shares the most tokens with the query.
        private Route ChooseByOverlap(string query)
        {
            var q = Tokenize(query);
            Route best = _routes[0];
            int bestScore = -1;
            foreach (var route in _routes)
            {
                var tokens = Tokenize(route.Name + " " + route.Description);
                int overlap = tokens.Count(q.Contains);
                if (overlap > bestScore)
                {
                    bestScore = overlap;
                    best = route;
                }
            }

            return best;
        }

        private static HashSet<string> Tokenize(string text)
        {
            if (string.IsNullOrEmpty(text)) return new HashSet<string>();
            return new HashSet<string>(
                text.ToLowerInvariant().Split(new[] { ' ', '\t', '\n', '\r', '.', ',', '!', '?', ';', ':' },
                    StringSplitOptions.RemoveEmptyEntries));
        }
    }
}
