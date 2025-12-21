using Newtonsoft.Json;

namespace AiDotNet.AutoML
{
    /// <summary>
    /// Data transfer object for a single operation in the architecture.
    /// </summary>
    public class OperationDto
    {
        /// <summary>
        /// Gets or sets the destination node index.
        /// </summary>
        [JsonProperty("toNode")]
        public int ToNode { get; set; }

        /// <summary>
        /// Gets or sets the source node index.
        /// </summary>
        [JsonProperty("fromNode")]
        public int FromNode { get; set; }

        /// <summary>
        /// Gets or sets the operation type (e.g., "conv3x3", "skip", "max_pool").
        /// </summary>
        [JsonProperty("operation")]
        public string Operation { get; set; } = string.Empty;
    }
}
