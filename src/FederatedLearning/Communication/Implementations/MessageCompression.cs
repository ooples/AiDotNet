using System;
using System.IO;
using System.IO.Compression;
using System.Text;
using System.Text.Json;
using System.Threading.Tasks;
using AiDotNet.FederatedLearning.Communication.Interfaces;
using AiDotNet.FederatedLearning.Communication.Models;

namespace AiDotNet.FederatedLearning.Communication.Implementations
{
    /// <summary>
    /// Production-ready message compression implementation using GZip
    /// </summary>
    public class MessageCompression : IMessageCompression
    {
        private readonly CompressionLevel _compressionLevel;

        /// <summary>
        /// Initializes a new instance of MessageCompression
        /// </summary>
        /// <param name="compressionLevel">Level of compression to use</param>
        public MessageCompression(CompressionLevel compressionLevel = CompressionLevel.Optimal)
        {
            _compressionLevel = compressionLevel;
        }

        /// <summary>
        /// Compress a federated message
        /// </summary>
        /// <param name="message">Message to compress</param>
        /// <returns>Compressed message</returns>
        public async Task<FederatedMessage> CompressMessageAsync(FederatedMessage message)
        {
            if (message == null)
                throw new ArgumentNullException(nameof(message));

            if (message.IsCompressed)
                return message; // Already compressed

            try
            {
                // Serialize the parameters to JSON
                var json = System.Text.Json.JsonSerializer.Serialize(message.Parameters);
                var jsonBytes = Encoding.UTF8.GetBytes(json);

                // Compress using GZip
                using var outputStream = new MemoryStream();
                using (var gzipStream = new GZipStream(outputStream, _compressionLevel))
                {
                    await gzipStream.WriteAsync(jsonBytes, 0, jsonBytes.Length);
                }

                var compressedData = outputStream.ToArray();

                // Update message
                message.CompressedData = compressedData;
                message.IsCompressed = true;
                message.CompressionType = "gzip";
                message.Parameters = null; // Clear original parameters to save memory

                // Log compression ratio
                var compressionRatio = (double)compressedData.Length / jsonBytes.Length;
                Console.WriteLine($"Compression ratio: {compressionRatio:P2} (from {jsonBytes.Length} to {compressedData.Length} bytes)");

                return message;
            }
            catch (Exception ex)
            {
                throw new InvalidOperationException("Failed to compress message", ex);
            }
        }

        /// <summary>
        /// Decompress a federated message
        /// </summary>
        /// <param name="message">Compressed message</param>
        /// <returns>Decompressed message</returns>
        public async Task<FederatedMessage> DecompressMessageAsync(FederatedMessage message)
        {
            if (message == null)
                throw new ArgumentNullException(nameof(message));

            if (!message.IsCompressed || message.CompressedData == null)
                return message; // Not compressed

            if (message.CompressionType != "gzip")
                throw new NotSupportedException($"Compression type '{message.CompressionType}' is not supported");

            try
            {
                // Decompress using GZip
                using var inputStream = new MemoryStream(message.CompressedData);
                using var outputStream = new MemoryStream();
                using (var gzipStream = new GZipStream(inputStream, CompressionMode.Decompress))
                {
                    await gzipStream.CopyToAsync(outputStream);
                }

                var decompressedBytes = outputStream.ToArray();
                var json = Encoding.UTF8.GetString(decompressedBytes);

                // Deserialize parameters
                message.Parameters = System.Text.Json.JsonSerializer.Deserialize<Dictionary<string, LinearAlgebra.Vector<double>>>(json);
                message.IsCompressed = false;
                message.CompressedData = null; // Clear compressed data to save memory

                return message;
            }
            catch (Exception ex)
            {
                throw new InvalidOperationException("Failed to decompress message", ex);
            }
        }
    }
}