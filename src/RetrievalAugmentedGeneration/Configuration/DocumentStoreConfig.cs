using System.Collections.Generic;

namespace AiDotNet.RetrievalAugmentedGeneration.Configuration
{
    /// <summary>
    /// Configuration for document store components.
    /// </summary>
    public class DocumentStoreConfig
    {
        /// <summary>
        /// Gets or sets the type of document store.
        /// </summary>
        public string Type { get; set; } = string.Empty;

        /// <summary>
        /// Gets or sets the parameters for the document store.
        /// </summary>
        public Dictionary<string, object> Parameters { get; set; } = new Dictionary<string, object>();
    }
}
