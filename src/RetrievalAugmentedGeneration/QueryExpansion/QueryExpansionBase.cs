using System.Collections.Generic;
using AiDotNet.Attributes;

namespace AiDotNet.RetrievalAugmentedGeneration.QueryExpansion
{
    /// <summary>
    /// Base class for query expansion strategies.
    /// </summary>
    [ModelDomain(ModelDomain.Language)]
    [ModelCategory(ModelCategory.NeuralNetwork)]
    [ModelTask(ModelTask.Generation)]
    [ModelComplexity(ModelComplexity.Low)]
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
