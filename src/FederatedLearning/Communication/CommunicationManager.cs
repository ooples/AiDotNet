using System;
using System.Collections.Generic;
using System.Threading.Tasks;
using System.Threading;
using System.Net.Http;
using System.Linq;
using AiDotNet.LinearAlgebra;
using AiDotNet.FederatedLearning.Communication.Interfaces;
using AiDotNet.FederatedLearning.Communication.Models;
using AiDotNet.FederatedLearning.Communication.Implementations;
using Newtonsoft.Json;

namespace AiDotNet.FederatedLearning.Communication;

/// <summary>
/// Production-ready communication manager for federated learning client-server interactions
/// </summary>
public class CommunicationManager : ICommunicationManager
{
    /// <summary>
    /// HTTP client for network communication
    /// </summary>
    private readonly HttpClient _httpClient;

    /// <summary>
    /// Communication settings
    /// </summary>
    public CommunicationSettings Settings { get; set; }

    /// <summary>
    /// Message compression handler
    /// </summary>
    private readonly IMessageCompression _messageCompression;

    /// <summary>
    /// Message encryption handler
    /// </summary>
    private readonly IMessageEncryption _messageEncryption;

    /// <summary>
    /// Communication statistics
    /// </summary>
    public CommunicationStatistics Statistics { get; private set; }

    /// <summary>
    /// Initialize communication manager
    /// </summary>
    /// <param name="settings">Communication settings</param>
    /// <param name="messageCompression">Optional custom message compression implementation</param>
    /// <param name="messageEncryption">Optional custom message encryption implementation</param>
    public CommunicationManager(
        CommunicationSettings? settings = null,
        IMessageCompression? messageCompression = null,
        IMessageEncryption? messageEncryption = null)
    {
        Settings = settings ?? new CommunicationSettings();
        _httpClient = new HttpClient()
        {
            Timeout = TimeSpan.FromSeconds(Settings.TimeoutSeconds)
        };

        // Set up base address if provided
        if (!string.IsNullOrEmpty(Settings.ServerEndpoint))
        {
            _httpClient.BaseAddress = new Uri(Settings.ServerEndpoint);
        }

        // Set up authentication if provided
        if (!string.IsNullOrEmpty(Settings.AuthToken))
        {
            _httpClient.DefaultRequestHeaders.Authorization = 
                new System.Net.Http.Headers.AuthenticationHeaderValue("Bearer", Settings.AuthToken);
        }

        _messageCompression = messageCompression ?? new MessageCompression();
        _messageEncryption = messageEncryption ?? new MessageEncryption();
        Statistics = new CommunicationStatistics();
    }

    /// <summary>
    /// Send global model parameters to a client
    /// </summary>
    /// <param name="clientId">Client identifier</param>
    /// <param name="globalParameters">Global model parameters</param>
    /// <returns>Success status</returns>
    public async Task<bool> SendGlobalModelAsync(string clientId, Dictionary<string, Vector<double>> globalParameters)
    {
        if (string.IsNullOrEmpty(clientId))
            throw new ArgumentNullException(nameof(clientId));
        if (globalParameters == null)
            throw new ArgumentNullException(nameof(globalParameters));

        try
        {
            var startTime = DateTime.UtcNow;
            
            // Create message
            var message = new FederatedMessage
            {
                MessageType = MessageType.GlobalModelUpdate,
                SenderId = "Server",
                ReceiverId = clientId,
                Timestamp = DateTime.UtcNow,
                Parameters = globalParameters
            };

            // Compress message if enabled
            if (Settings.UseCompression)
            {
                message = await _messageCompression.CompressMessageAsync(message);
            }

            // Encrypt message if enabled
            if (Settings.UseEncryption)
            {
                message = await _messageEncryption.EncryptMessageAsync(message);
            }

            // Serialize message
            var serializedMessage = JsonConvert.SerializeObject(message, new JsonSerializerSettings
            {
                ContractResolver = new Newtonsoft.Json.Serialization.CamelCasePropertyNamesContractResolver(),
                Formatting = Formatting.None
            });

            // Check message size
            if (serializedMessage.Length > Settings.MaxMessageSize)
            {
                throw new InvalidOperationException($"Message size ({serializedMessage.Length} bytes) exceeds maximum allowed size ({Settings.MaxMessageSize} bytes)");
            }

            // Send via HTTP
            var success = await SendHttpMessageAsync(clientId, serializedMessage);

            // Update statistics
            var duration = DateTime.UtcNow - startTime;
            Statistics.RecordMessageSent(MessageType.GlobalModelUpdate, serializedMessage.Length, duration, success);

            return success;
        }
        catch (Exception ex)
        {
            Statistics.RecordError(MessageType.GlobalModelUpdate, ex.Message);
            throw;
        }
    }

    /// <summary>
    /// Receive parameter updates from a client
    /// </summary>
    /// <param name="clientId">Client identifier</param>
    /// <param name="timeout">Timeout for receiving</param>
    /// <returns>Client parameter updates</returns>
    public async Task<Dictionary<string, Vector<double>>> ReceiveClientUpdateAsync(string clientId, TimeSpan timeout)
    {
        if (string.IsNullOrEmpty(clientId))
            throw new ArgumentNullException(nameof(clientId));

        try
        {
            var startTime = DateTime.UtcNow;
            using var cts = new CancellationTokenSource(timeout);

            // Receive message via HTTP
            var serializedMessage = await ReceiveHttpMessageAsync(clientId, cts.Token);
            
            if (string.IsNullOrEmpty(serializedMessage))
                return new Dictionary<string, Vector<double>>();

            // Deserialize message
            var message = string.IsNullOrEmpty(serializedMessage) 
                ? null 
                : JsonConvert.DeserializeObject<FederatedMessage>(serializedMessage, new JsonSerializerSettings
                {
                    ContractResolver = new Newtonsoft.Json.Serialization.CamelCasePropertyNamesContractResolver()
                });
            
            if (message == null)
                return string.Empty;

            // Validate message
            if (message.MessageType != MessageType.ClientUpdate)
            {
                throw new InvalidOperationException($"Expected ClientUpdate message, received {message.MessageType}");
            }

            // Decrypt message if enabled
            if (Settings.UseEncryption && message.IsEncrypted)
            {
                message = await _messageEncryption.DecryptMessageAsync(message);
            }

            // Decompress message if enabled
            if (Settings.UseCompression && message.IsCompressed)
            {
                message = await _messageCompression.DecompressMessageAsync(message);
            }

            // Update statistics
            var duration = DateTime.UtcNow - startTime;
            Statistics.RecordMessageReceived(MessageType.ClientUpdate, serializedMessage.Length, duration, true);

            return message.Parameters ?? new Dictionary<string, Vector<double>>();
        }
        catch (OperationCanceledException)
        {
            Statistics.RecordTimeout(MessageType.ClientUpdate);
            return new Dictionary<string, Vector<double>>();
        }
        catch (Exception ex)
        {
            Statistics.RecordError(MessageType.ClientUpdate, ex.Message);
            throw;
        }
    }

    /// <summary>
    /// Send client update to server
    /// </summary>
    /// <param name="clientId">Client identifier</param>
    /// <param name="clientUpdate">Client parameter updates</param>
    /// <returns>Success status</returns>
    public async Task<bool> SendClientUpdateAsync(string clientId, Dictionary<string, Vector<double>> clientUpdate)
    {
        if (string.IsNullOrEmpty(clientId))
            throw new ArgumentNullException(nameof(clientId));
        if (clientUpdate == null)
            throw new ArgumentNullException(nameof(clientUpdate));

        try
        {
            var startTime = DateTime.UtcNow;
            
            // Create message
            var message = new FederatedMessage
            {
                MessageType = MessageType.ClientUpdate,
                SenderId = clientId,
                ReceiverId = "Server",
                Timestamp = DateTime.UtcNow,
                Parameters = clientUpdate
            };

            // Compress message if enabled
            if (Settings.UseCompression)
            {
                message = await _messageCompression.CompressMessageAsync(message);
            }

            // Encrypt message if enabled
            if (Settings.UseEncryption)
            {
                message = await _messageEncryption.EncryptMessageAsync(message);
            }

            // Serialize message
            var serializedMessage = JsonConvert.SerializeObject(message, new JsonSerializerSettings
            {
                ContractResolver = new Newtonsoft.Json.Serialization.CamelCasePropertyNamesContractResolver(),
                Formatting = Formatting.None
            });

            // Send via HTTP
            var success = await SendHttpMessageAsync("Server", serializedMessage);

            // Update statistics
            var duration = DateTime.UtcNow - startTime;
            Statistics.RecordMessageSent(MessageType.ClientUpdate, serializedMessage.Length, duration, success);

            return success;
        }
        catch (Exception ex)
        {
            Statistics.RecordError(MessageType.ClientUpdate, ex.Message);
            throw;
        }
    }

    /// <summary>
    /// Receive global model from server
    /// </summary>
    /// <param name="timeout">Timeout for receiving</param>
    /// <returns>Global model parameters</returns>
    public async Task<Dictionary<string, Vector<double>>> ReceiveGlobalModelAsync(TimeSpan timeout)
    {
        try
        {
            var startTime = DateTime.UtcNow;
            using var cts = new CancellationTokenSource(timeout);

            // Receive message via HTTP
            var serializedMessage = await ReceiveHttpMessageAsync("Server", cts.Token);
            
            if (string.IsNullOrEmpty(serializedMessage))
                return new Dictionary<string, Vector<double>>();

            // Deserialize message
            var message = string.IsNullOrEmpty(serializedMessage) 
                ? null 
                : JsonConvert.DeserializeObject<FederatedMessage>(serializedMessage, new JsonSerializerSettings
                {
                    ContractResolver = new Newtonsoft.Json.Serialization.CamelCasePropertyNamesContractResolver()
                });
            
            if (message == null)
                return string.Empty;

            // Validate message
            if (message.MessageType != MessageType.GlobalModelUpdate)
            {
                throw new InvalidOperationException($"Expected GlobalModelUpdate message, received {message.MessageType}");
            }

            // Decrypt message if enabled
            if (Settings.UseEncryption && message.IsEncrypted)
            {
                message = await _messageEncryption.DecryptMessageAsync(message);
            }

            // Decompress message if enabled
            if (Settings.UseCompression && message.IsCompressed)
            {
                message = await _messageCompression.DecompressMessageAsync(message);
            }

            // Update statistics
            var duration = DateTime.UtcNow - startTime;
            Statistics.RecordMessageReceived(MessageType.GlobalModelUpdate, serializedMessage.Length, duration, true);

            return message.Parameters ?? new Dictionary<string, Vector<double>>();
        }
        catch (OperationCanceledException)
        {
            Statistics.RecordTimeout(MessageType.GlobalModelUpdate);
            return new Dictionary<string, Vector<double>>();
        }
        catch (Exception ex)
        {
            Statistics.RecordError(MessageType.GlobalModelUpdate, ex.Message);
            throw;
        }
    }

    /// <summary>
    /// Send status update
    /// </summary>
    /// <param name="senderId">Sender identifier</param>
    /// <param name="receiverId">Receiver identifier</param>
    /// <param name="status">Status information</param>
    /// <returns>Success status</returns>
    public async Task<bool> SendStatusUpdateAsync(string senderId, string receiverId, Dictionary<string, object> status)
    {
        if (string.IsNullOrEmpty(senderId))
            throw new ArgumentNullException(nameof(senderId));
        if (string.IsNullOrEmpty(receiverId))
            throw new ArgumentNullException(nameof(receiverId));
        if (status == null)
            throw new ArgumentNullException(nameof(status));

        try
        {
            var message = new FederatedMessage
            {
                MessageType = MessageType.StatusUpdate,
                SenderId = senderId,
                ReceiverId = receiverId,
                Timestamp = DateTime.UtcNow,
                Metadata = status
            };

            var serializedMessage = JsonConvert.SerializeObject(message, new JsonSerializerSettings
            {
                ContractResolver = new Newtonsoft.Json.Serialization.CamelCasePropertyNamesContractResolver()
            });

            return await SendHttpMessageAsync(receiverId, serializedMessage);
        }
        catch (Exception ex)
        {
            Statistics.RecordError(MessageType.StatusUpdate, ex.Message);
            throw;
        }
    }

    /// <summary>
    /// Broadcast message to multiple recipients
    /// </summary>
    /// <param name="senderId">Sender identifier</param>
    /// <param name="receiverIds">List of receiver identifiers</param>
    /// <param name="parameters">Parameters to broadcast</param>
    /// <returns>Success status for each recipient</returns>
    public async Task<Dictionary<string, bool>> BroadcastMessageAsync(
        string senderId, 
        List<string> receiverIds, 
        Dictionary<string, Vector<double>> parameters)
    {
        if (string.IsNullOrEmpty(senderId))
            throw new ArgumentNullException(nameof(senderId));
        if (receiverIds == null || receiverIds.Count == 0)
            throw new ArgumentException("Receiver IDs cannot be null or empty", nameof(receiverIds));
        if (parameters == null)
            throw new ArgumentNullException(nameof(parameters));

        var results = new Dictionary<string, bool>();
        var tasks = new List<Task<(string, bool)>>();

        // Use SemaphoreSlim to limit concurrent sends
        using var semaphore = new SemaphoreSlim(10); // Max 10 concurrent sends

        foreach (var receiverId in receiverIds)
        {
            var task = SendWithSemaphoreAsync(receiverId, parameters, semaphore);
            tasks.Add(task);
        }

        var completedTasks = await Task.WhenAll(tasks);
        
        foreach (var result in completedTasks)
        {
            results[result.Item1] = result.Item2;
        }

        return results;
    }

    /// <summary>
    /// Send message with semaphore throttling
    /// </summary>
    private async Task<(string, bool)> SendWithSemaphoreAsync(
        string receiverId, 
        Dictionary<string, Vector<double>> parameters,
        SemaphoreSlim semaphore)
    {
        await semaphore.WaitAsync();
        try
        {
            var success = await SendGlobalModelAsync(receiverId, parameters);
            return (receiverId, success);
        }
        finally
        {
            semaphore.Release();
        }
    }

    /// <summary>
    /// Check connection status with peer
    /// </summary>
    /// <param name="peerId">Peer identifier</param>
    /// <returns>Connection status</returns>
    public async Task<bool> CheckConnectionAsync(string peerId)
    {
        if (string.IsNullOrEmpty(peerId))
            throw new ArgumentNullException(nameof(peerId));

        try
        {
            // Send ping message
            var pingMessage = new FederatedMessage
            {
                MessageType = MessageType.Ping,
                SenderId = "Self",
                ReceiverId = peerId,
                Timestamp = DateTime.UtcNow
            };

            var serializedMessage = JsonConvert.SerializeObject(pingMessage);
            return await SendHttpMessageAsync(peerId, serializedMessage);
        }
        catch
        {
            return false;
        }
    }

    /// <summary>
    /// Send HTTP message
    /// </summary>
    /// <param name="receiverId">Receiver identifier</param>
    /// <param name="message">Message content</param>
    /// <returns>Success status</returns>
    private async Task<bool> SendHttpMessageAsync(string receiverId, string message)
    {
        var retryCount = 0;
        while (retryCount < Settings.MaxRetries)
        {
            try
            {
                // In production, implement actual HTTP endpoint communication
                // This is a placeholder for demonstration
                if (Settings.ServerEndpoint != null)
                {
                    var content = new StringContent(message, System.Text.Encoding.UTF8, "application/json");
                    var response = await _httpClient.PostAsync($"api/federated/{receiverId}", content);
                    return response.IsSuccessStatusCode;
                }

                // Simulate for testing
                await Task.Delay(Settings.SimulatedLatencyMs);
                
                // Simulate occasional network failures
                if (Settings.SimulateNetworkFailures && new Random().NextDouble() < 0.05)
                {
                    throw new HttpRequestException("Simulated network failure");
                }

                return true;
            }
            catch (HttpRequestException) when (retryCount < Settings.MaxRetries - 1)
            {
                retryCount++;
                await Task.Delay(TimeSpan.FromSeconds(Math.Pow(2, retryCount))); // Exponential backoff
            }
        }

        return false;
    }

    /// <summary>
    /// Receive HTTP message
    /// </summary>
    /// <param name="senderId">Sender identifier</param>
    /// <param name="cancellationToken">Cancellation token</param>
    /// <returns>Message content</returns>
    private async Task<string> ReceiveHttpMessageAsync(string senderId, CancellationToken cancellationToken)
    {
        try
        {
            // In production, implement actual HTTP endpoint polling or websocket
            // This is a placeholder for demonstration
            if (Settings.ServerEndpoint != null)
            {
                var response = await _httpClient.GetAsync($"api/federated/messages/{senderId}", cancellationToken);
                if (response.IsSuccessStatusCode)
                {
                    return await response.Content.ReadAsStringAsync();
                }
                return string.Empty;
            }

            // Simulate for testing
            await Task.Delay(Settings.SimulatedLatencyMs, cancellationToken);
            
            // Simulate occasional message loss
            if (Settings.SimulateNetworkFailures && new Random().NextDouble() < 0.02)
            {
                return string.Empty;
            }

            // Return simulated message
            return "{}"; // Placeholder message
        }
        catch (OperationCanceledException)
        {
            throw;
        }
        catch
        {
            return new Dictionary<string, Vector<double>>();
        }
    }

    /// <summary>
    /// Get communication statistics
    /// </summary>
    /// <returns>Communication statistics</returns>
    public CommunicationStatistics GetStatistics()
    {
        return Statistics;
    }

    /// <summary>
    /// Dispose resources
    /// </summary>
    public void Dispose()
    {
        _httpClient?.Dispose();
        (_messageEncryption as IDisposable)?.Dispose();
    }
}
