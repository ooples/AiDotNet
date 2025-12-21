using System.Collections.Generic;
using Newtonsoft.Json;

namespace AiDotNet.AutoML
{
    /// <summary>
    /// Data transfer object for architecture JSON serialization.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This class provides a JSON-friendly structure for serializing architectures.
    /// Tuples and complex types are converted to simple objects for better interoperability.
    /// </para>
    /// </remarks>
    public class ArchitectureDto
    {
        /// <summary>
        /// Gets or sets the number of nodes in the architecture.
        /// </summary>
        [JsonProperty("nodeCount")]
        public int NodeCount { get; set; }

        /// <summary>
        /// Gets or sets the list of operations in the architecture.
        /// </summary>
        [JsonProperty("operations")]
        public List<OperationDto> Operations { get; set; } = new List<OperationDto>();

        /// <summary>
        /// Gets or sets the node channel mappings.
        /// </summary>
        /// <remarks>
        /// Keys are node indices as strings (for JSON compatibility).
        /// </remarks>
        [JsonProperty("nodeChannels")]
        public Dictionary<string, int> NodeChannels { get; set; } = new Dictionary<string, int>();
    }
}
