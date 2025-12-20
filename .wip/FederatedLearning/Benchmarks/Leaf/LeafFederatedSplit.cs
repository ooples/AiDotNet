using AiDotNet.Models;

namespace AiDotNet.FederatedLearning.Benchmarks.Leaf;

/// <summary>
/// Represents a single LEAF split (train/test) as per-client datasets.
/// </summary>
/// <typeparam name="TInput">The client feature type.</typeparam>
/// <typeparam name="TOutput">The client label type.</typeparam>
/// <remarks>
/// <para>
/// LEAF is a federated learning benchmark suite where each user corresponds to one client.
/// This type preserves that structure by storing one dataset per user.
/// </para>
/// <para><b>For Beginners:</b> Instead of one big dataset for everyone, federated learning uses
/// many small datasets — one per device or organization. This class stores that "one dataset per client"
/// view of the data.
/// </para>
/// </remarks>
public sealed class LeafFederatedSplit<TInput, TOutput>
{
    /// <summary>
    /// Initializes a new instance of the <see cref="LeafFederatedSplit{TInput, TOutput}"/> class.
    /// </summary>
    /// <param name="userIds">Ordered list of user IDs in this split.</param>
    /// <param name="userData">Mapping from user ID to that user's local dataset.</param>
    public LeafFederatedSplit(
        IReadOnlyList<string> userIds,
        IReadOnlyDictionary<string, FederatedClientDataset<TInput, TOutput>> userData)
    {
        UserIds = userIds ?? throw new ArgumentNullException(nameof(userIds));
        UserData = userData ?? throw new ArgumentNullException(nameof(userData));

        if (UserIds.Count == 0)
        {
            throw new ArgumentException("LEAF split must contain at least one user.", nameof(userIds));
        }

        foreach (var userId in UserIds)
        {
            if (string.IsNullOrWhiteSpace(userId))
            {
                throw new ArgumentException("User IDs cannot be null/empty.", nameof(userIds));
            }

            if (!UserData.ContainsKey(userId))
            {
                throw new ArgumentException($"Missing user_data entry for user '{userId}'.", nameof(userData));
            }
        }
    }

    /// <summary>
    /// Gets the ordered list of user IDs in this split.
    /// </summary>
    public IReadOnlyList<string> UserIds { get; }

    /// <summary>
    /// Gets the per-user datasets in this split.
    /// </summary>
    public IReadOnlyDictionary<string, FederatedClientDataset<TInput, TOutput>> UserData { get; }

    /// <summary>
    /// Gets the number of clients/users in this split.
    /// </summary>
    public int ClientCount => UserIds.Count;

    /// <summary>
    /// Converts this split into the <c>Dictionary&lt;int, FederatedClientDataset&gt;</c> form used by trainers.
    /// </summary>
    /// <param name="clientIdToUserId">Mapping from assigned client IDs (0..N-1) to LEAF user IDs.</param>
    /// <returns>Per-client datasets keyed by stable sequential client IDs.</returns>
    public Dictionary<int, FederatedClientDataset<TInput, TOutput>> ToClientIdDictionary(
        out IReadOnlyDictionary<int, string> clientIdToUserId)
    {
        var mapping = new Dictionary<int, string>(UserIds.Count);
        var result = new Dictionary<int, FederatedClientDataset<TInput, TOutput>>(UserIds.Count);

        for (int i = 0; i < UserIds.Count; i++)
        {
            var userId = UserIds[i];
            mapping[i] = userId;
            result[i] = UserData[userId];
        }

        clientIdToUserId = mapping;
        return result;
    }
}

