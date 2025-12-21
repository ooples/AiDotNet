using System;
using System.Collections.Generic;
using System.IO;
using Newtonsoft.Json;

namespace AiDotNet.AutoML
{
    /// <summary>
    /// Represents a neural network architecture discovered through NAS.
    /// </summary>
    /// <typeparam name="T">The numeric type for calculations.</typeparam>
    /// <remarks>
    /// <para>
    /// This class captures the structure of a neural network discovered through Neural Architecture Search (NAS).
    /// It includes the operations connecting nodes and optional channel information for cost estimation.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> Think of this as a blueprint for a neural network. NAS algorithms explore
    /// many possible blueprints and find the best one for your task. This class stores that blueprint
    /// so you can:
    /// <list type="bullet">
    /// <item><description>Save it to disk for later use</description></item>
    /// <item><description>Share it with others</description></item>
    /// <item><description>Load it to recreate the same network structure</description></item>
    /// </list>
    /// </para>
    /// </remarks>
    public class Architecture<T>
    {
        /// <summary>
        /// Operations in the architecture: (to_node, from_node, operation)
        /// </summary>
        [JsonIgnore]
        public List<(int ToNode, int FromNode, string Operation)> Operations { get; set; } = new List<(int, int, string)>();

        /// <summary>
        /// Optional per-node channel counts (typically output channels) used for cost estimation.
        /// </summary>
        /// <remarks>
        /// <para>
        /// When provided, hardware cost models can use these values to scale operation costs more accurately
        /// (e.g., accounting for channel expansion/reduction across layers). If not provided, cost models may
        /// fall back to assuming a uniform channel count.
        /// </para>
        /// <para><b>For Beginners:</b> This is a simple mapping like: node 0 has 16 channels, node 1 has 32 channels, etc.
        /// Some operations change how many features (channels) flow through the network, which affects compute cost.
        /// </para>
        /// </remarks>
        public Dictionary<int, int> NodeChannels { get; set; } = new Dictionary<int, int>();

        /// <summary>
        /// Number of nodes in the architecture
        /// </summary>
        public int NodeCount { get; set; }

        /// <summary>
        /// Adds an operation to the architecture
        /// </summary>
        public void AddOperation(int toNode, int fromNode, string operation)
        {
            Operations.Add((toNode, fromNode, operation));
            NodeCount = System.Math.Max(NodeCount, System.Math.Max(toNode, fromNode) + 1);
        }

        /// <summary>
        /// Gets a description of the architecture.
        /// </summary>
        public string GetDescription()
        {
            var lines = new List<string>();
            lines.Add($"Architecture with {NodeCount} nodes:");

            foreach (var (toNode, fromNode, operation) in Operations)
            {
                lines.Add($"  Node {toNode} <- {operation} <- Node {fromNode}");
            }

            return string.Join("\n", lines);
        }

        #region Serialization

        /// <summary>
        /// Serializes the architecture to a JSON string.
        /// </summary>
        /// <param name="indented">Whether to use indented formatting for readability.</param>
        /// <returns>A JSON string representation of the architecture.</returns>
        /// <remarks>
        /// <para>
        /// <b>For Beginners:</b> JSON is a text format that's easy for both humans and computers to read.
        /// Use this method to save your architecture in a format you can inspect in a text editor.
        /// </para>
        /// </remarks>
        public string ToJson(bool indented = true)
        {
            var dto = ToSerializable();
            var formatting = indented ? Formatting.Indented : Formatting.None;
            return JsonConvert.SerializeObject(dto, formatting);
        }

        /// <summary>
        /// Deserializes an architecture from a JSON string.
        /// </summary>
        /// <param name="json">The JSON string to deserialize.</param>
        /// <returns>The deserialized architecture.</returns>
        /// <exception cref="ArgumentNullException">Thrown when json is null.</exception>
        /// <exception cref="JsonException">Thrown when the JSON is invalid.</exception>
        public static Architecture<T> FromJson(string json)
        {
            if (string.IsNullOrEmpty(json))
                throw new ArgumentNullException(nameof(json));

            var dto = JsonConvert.DeserializeObject<ArchitectureDto>(json);
            if (dto == null)
                throw new JsonException("Failed to deserialize architecture JSON.");
            return FromSerializable(dto);
        }

        /// <summary>
        /// Saves the architecture to a JSON file.
        /// </summary>
        /// <param name="filePath">The path to save the JSON file.</param>
        /// <param name="indented">Whether to use indented formatting.</param>
        /// <exception cref="ArgumentNullException">Thrown when filePath is null.</exception>
        /// <exception cref="IOException">Thrown when the file cannot be written.</exception>
        public void SaveToFile(string filePath, bool indented = true)
        {
            if (string.IsNullOrEmpty(filePath))
                throw new ArgumentNullException(nameof(filePath));

            var json = ToJson(indented);
            File.WriteAllText(filePath, json);
        }

        /// <summary>
        /// Loads an architecture from a JSON file.
        /// </summary>
        /// <param name="filePath">The path to the JSON file.</param>
        /// <returns>The loaded architecture.</returns>
        /// <exception cref="ArgumentNullException">Thrown when filePath is null.</exception>
        /// <exception cref="FileNotFoundException">Thrown when the file doesn't exist.</exception>
        /// <exception cref="JsonException">Thrown when the JSON is invalid.</exception>
        public static Architecture<T> LoadFromFile(string filePath)
        {
            if (string.IsNullOrEmpty(filePath))
                throw new ArgumentNullException(nameof(filePath));

            if (!File.Exists(filePath))
                throw new FileNotFoundException($"Architecture file not found: {filePath}", filePath);

            var json = File.ReadAllText(filePath);
            return FromJson(json);
        }

        /// <summary>
        /// Serializes the architecture to a binary byte array.
        /// </summary>
        /// <returns>A byte array containing the serialized architecture.</returns>
        /// <remarks>
        /// <para>
        /// <b>For Beginners:</b> Binary format is more compact than JSON and faster to read/write,
        /// but you can't inspect it in a text editor. Use this for production deployments.
        /// </para>
        /// </remarks>
        public byte[] ToBytes()
        {
            using (var stream = new MemoryStream())
            using (var writer = new BinaryWriter(stream))
            {
                // Write version for future compatibility
                writer.Write((byte)1);

                // Write node count
                writer.Write(NodeCount);

                // Write operations
                writer.Write(Operations.Count);
                foreach (var (toNode, fromNode, operation) in Operations)
                {
                    writer.Write(toNode);
                    writer.Write(fromNode);
                    writer.Write(operation);
                }

                // Write node channels
                writer.Write(NodeChannels.Count);
                foreach (var kvp in NodeChannels)
                {
                    writer.Write(kvp.Key);
                    writer.Write(kvp.Value);
                }

                return stream.ToArray();
            }
        }

        /// <summary>
        /// Deserializes an architecture from a binary byte array.
        /// </summary>
        /// <param name="data">The byte array to deserialize.</param>
        /// <returns>The deserialized architecture.</returns>
        /// <exception cref="ArgumentNullException">Thrown when data is null.</exception>
        /// <exception cref="InvalidDataException">Thrown when the data format is invalid.</exception>
        public static Architecture<T> FromBytes(byte[] data)
        {
            if (data == null || data.Length == 0)
                throw new ArgumentNullException(nameof(data));

            using (var stream = new MemoryStream(data))
            using (var reader = new BinaryReader(stream))
            {
                // Read and validate version
                var version = reader.ReadByte();
                if (version != 1)
                    throw new InvalidDataException($"Unsupported architecture format version: {version}");

                var architecture = new Architecture<T>();

                // Read node count
                architecture.NodeCount = reader.ReadInt32();

                // Read operations
                var opCount = reader.ReadInt32();
                for (int i = 0; i < opCount; i++)
                {
                    var toNode = reader.ReadInt32();
                    var fromNode = reader.ReadInt32();
                    var operation = reader.ReadString();
                    architecture.Operations.Add((toNode, fromNode, operation));
                }

                // Read node channels
                var channelCount = reader.ReadInt32();
                for (int i = 0; i < channelCount; i++)
                {
                    var nodeIdx = reader.ReadInt32();
                    var channels = reader.ReadInt32();
                    architecture.NodeChannels[nodeIdx] = channels;
                }

                return architecture;
            }
        }

        /// <summary>
        /// Converts the architecture to a serializable DTO.
        /// </summary>
        internal ArchitectureDto ToSerializable()
        {
            var dto = new ArchitectureDto
            {
                NodeCount = NodeCount,
                Operations = new List<OperationDto>(),
                NodeChannels = new Dictionary<string, int>()
            };

            foreach (var (toNode, fromNode, operation) in Operations)
            {
                dto.Operations.Add(new OperationDto
                {
                    ToNode = toNode,
                    FromNode = fromNode,
                    Operation = operation
                });
            }

            foreach (var kvp in NodeChannels)
            {
                dto.NodeChannels[kvp.Key.ToString()] = kvp.Value;
            }

            return dto;
        }

        /// <summary>
        /// Creates an architecture from a serializable DTO.
        /// </summary>
        internal static Architecture<T> FromSerializable(ArchitectureDto dto)
        {
            var architecture = new Architecture<T>
            {
                NodeCount = dto.NodeCount
            };

            if (dto.Operations != null)
            {
                foreach (var op in dto.Operations)
                {
                    architecture.Operations.Add((op.ToNode, op.FromNode, op.Operation));
                }
            }

            if (dto.NodeChannels != null)
            {
                foreach (var kvp in dto.NodeChannels)
                {
                    if (int.TryParse(kvp.Key, out int nodeIdx))
                    {
                        architecture.NodeChannels[nodeIdx] = kvp.Value;
                    }
                }
            }

            return architecture;
        }

        #endregion
    }
}
