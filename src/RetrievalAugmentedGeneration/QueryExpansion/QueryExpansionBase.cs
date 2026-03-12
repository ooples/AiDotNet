using System.Collections.Generic;

namespace AiDotNet.RetrievalAugmentedGeneration.QueryExpansion
{
    /// <summary>
    /// Base class for query expansion strategies.
    /// </summary>
    public abstract class QueryExpansionBase
    {
        /// <summary>
        /// Expands a query into multiple variations.
        /// </summary>
        /// <param name="query">The original query.</param>
        /// <returns>A list of expanded queries.</returns>
        public abstract List<string> ExpandQuery(string query);
    }
}
