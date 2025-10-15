using System.Threading.Tasks;
using AiDotNet.FederatedLearning.Communication.Models;

namespace AiDotNet.FederatedLearning.Communication.Interfaces
{
    /// <summary>
    /// Interface for message encryption in federated learning communication
    /// </summary>
    public interface IMessageEncryption
    {
        /// <summary>
        /// Encrypt a federated message
        /// </summary>
        /// <param name="message">Message to encrypt</param>
        /// <returns>Encrypted message</returns>
        Task<FederatedMessage> EncryptMessageAsync(FederatedMessage message);

        /// <summary>
        /// Decrypt a federated message
        /// </summary>
        /// <param name="message">Encrypted message</param>
        /// <returns>Decrypted message</returns>
        Task<FederatedMessage> DecryptMessageAsync(FederatedMessage message);
    }
}