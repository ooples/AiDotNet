using System.Threading.Tasks;
using AiDotNet.FederatedLearning.Communication.Models;

namespace AiDotNet.FederatedLearning.Communication.Interfaces
{
    /// <summary>
    /// Interface for message compression in federated learning communication
    /// </summary>
    public interface IMessageCompression
    {
        /// <summary>
        /// Compress a federated message
        /// </summary>
        /// <param name="message">Message to compress</param>
        /// <returns>Compressed message</returns>
        Task<FederatedMessage> CompressMessageAsync(FederatedMessage message);

        /// <summary>
        /// Decompress a federated message
        /// </summary>
        /// <param name="message">Compressed message</param>
        /// <returns>Decompressed message</returns>
        Task<FederatedMessage> DecompressMessageAsync(FederatedMessage message);
    }
}