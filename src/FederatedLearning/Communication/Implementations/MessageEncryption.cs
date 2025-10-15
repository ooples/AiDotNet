using System;
using System.IO;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using System.Threading.Tasks;
using AiDotNet.FederatedLearning.Communication.Interfaces;
using AiDotNet.FederatedLearning.Communication.Models;

namespace AiDotNet.FederatedLearning.Communication.Implementations
{
    /// <summary>
    /// Production-ready message encryption implementation using AES
    /// </summary>
    public class MessageEncryption : IMessageEncryption, IDisposable
    {
        private readonly byte[] _key;
        private readonly byte[] _iv;
        private readonly Aes _aes;

        /// <summary>
        /// Initializes a new instance of MessageEncryption
        /// </summary>
        /// <param name="key">Encryption key (32 bytes for AES-256)</param>
        /// <param name="iv">Initialization vector (16 bytes)</param>
        public MessageEncryption(byte[]? key = null, byte[]? iv = null)
        {
            _aes = Aes.Create();
            _aes.Mode = CipherMode.CBC;
            _aes.Padding = PaddingMode.PKCS7;

            if (key == null || iv == null)
            {
                // Generate secure random key and IV
                _aes.GenerateKey();
                _aes.GenerateIV();
                _key = _aes.Key;
                _iv = _aes.IV;
            }
            else
            {
                if (key.Length != 32) // 256 bits
                    throw new ArgumentException("Key must be 32 bytes for AES-256", nameof(key));
                if (iv.Length != 16) // 128 bits
                    throw new ArgumentException("IV must be 16 bytes", nameof(iv));

                _key = key;
                _iv = iv;
                _aes.Key = _key;
                _aes.IV = _iv;
            }
        }

        /// <summary>
        /// Encrypt a federated message
        /// </summary>
        /// <param name="message">Message to encrypt</param>
        /// <returns>Encrypted message</returns>
        public async Task<FederatedMessage> EncryptMessageAsync(FederatedMessage message)
        {
            if (message == null)
                throw new ArgumentNullException(nameof(message));

            if (message.IsEncrypted)
                return message; // Already encrypted

            try
            {
                // Serialize the entire message content
                var messageContent = new
                {
                    message.Parameters,
                    message.Metadata,
                    message.CompressedData
                };

                var json = System.Text.Json.JsonSerializer.Serialize(messageContent);
                var plainBytes = Encoding.UTF8.GetBytes(json);

                // Encrypt
                using var encryptor = _aes.CreateEncryptor();
                using var msEncrypt = new MemoryStream();
                using (var csEncrypt = new CryptoStream(msEncrypt, encryptor, CryptoStreamMode.Write))
                {
                    await csEncrypt.WriteAsync(plainBytes, 0, plainBytes.Length);
                    csEncrypt.FlushFinalBlock();
                }

                var encryptedData = msEncrypt.ToArray();

                // Update message
                message.EncryptedData = encryptedData;
                message.IsEncrypted = true;
                message.EncryptionType = "AES-256-CBC";
                
                // Clear sensitive data
                message.Parameters = null;
                message.Metadata = null;
                message.CompressedData = null;

                return message;
            }
            catch (Exception ex)
            {
                throw new InvalidOperationException("Failed to encrypt message", ex);
            }
        }

        /// <summary>
        /// Decrypt a federated message
        /// </summary>
        /// <param name="message">Encrypted message</param>
        /// <returns>Decrypted message</returns>
        public async Task<FederatedMessage> DecryptMessageAsync(FederatedMessage message)
        {
            if (message == null)
                throw new ArgumentNullException(nameof(message));

            if (!message.IsEncrypted || message.EncryptedData == null)
                return message; // Not encrypted

            if (message.EncryptionType != "AES-256-CBC")
                throw new NotSupportedException($"Encryption type '{message.EncryptionType}' is not supported");

            try
            {
                // Decrypt
                using var decryptor = _aes.CreateDecryptor();
                using var msDecrypt = new MemoryStream(message.EncryptedData);
                using var csDecrypt = new CryptoStream(msDecrypt, decryptor, CryptoStreamMode.Read);
                using var msPlain = new MemoryStream();
                
                await csDecrypt.CopyToAsync(msPlain);
                var plainBytes = msPlain.ToArray();
                var json = Encoding.UTF8.GetString(plainBytes);

                // Deserialize message content
                var messageContent = System.Text.Json.JsonSerializer.Deserialize<JsonElement>(json);
                
                if (messageContent.TryGetProperty("Parameters", out var parameters))
                {
                    message.Parameters = System.Text.Json.JsonSerializer.Deserialize<Dictionary<string, LinearAlgebra.Vector<double>>>(
                        parameters.GetRawText());
                }

                if (messageContent.TryGetProperty("Metadata", out var metadata))
                {
                    message.Metadata = System.Text.Json.JsonSerializer.Deserialize<Dictionary<string, object>>(
                        metadata.GetRawText());
                }

                if (messageContent.TryGetProperty("CompressedData", out var compressedData))
                {
                    message.CompressedData = System.Text.Json.JsonSerializer.Deserialize<byte[]>(
                        compressedData.GetRawText());
                }

                message.IsEncrypted = false;
                message.EncryptedData = null; // Clear encrypted data

                return message;
            }
            catch (Exception ex)
            {
                throw new InvalidOperationException("Failed to decrypt message", ex);
            }
        }

        /// <summary>
        /// Get the encryption key (for secure key exchange)
        /// </summary>
        /// <returns>Encryption key</returns>
        public byte[] GetKey() => (byte[])_key.Clone();

        /// <summary>
        /// Get the initialization vector (for secure key exchange)
        /// </summary>
        /// <returns>Initialization vector</returns>
        public byte[] GetIV() => (byte[])_iv.Clone();

        /// <summary>
        /// Dispose of resources
        /// </summary>
        public void Dispose()
        {
            _aes?.Dispose();
        }
    }
}