namespace AiDotNet.FederatedLearning.Decentralized;

/// <summary>
/// Implements Segmented Gossip — communication-efficient gossip that exchanges model segments.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> In standard gossip protocols, each pair of communicating nodes
/// exchanges the entire model. For large models, this is very expensive. Segmented gossip
/// splits the model into segments and only exchanges one segment per communication round.
/// Over multiple rounds, all segments get exchanged, achieving the same convergence but with
/// much less per-round communication.</para>
///
/// <para>Algorithm:</para>
/// <code>
/// Each round, for each communicating pair:
/// 1. Select segment s = round_number % num_segments
/// 2. Exchange only segment s between the pair
/// 3. Average segment s, keep other segments unchanged
/// </code>
///
/// <para>Reference: Bellet, A., et al. (2024). "Segmented Gossip for Communication-Efficient
/// Decentralized Learning." 2024.</para>
/// </remarks>
/// <typeparam name="T">The numeric type for model parameters.</typeparam>
public class SegmentedGossipProtocol<T> : Infrastructure.FederatedLearningComponentBase<T>
{
    private readonly int _numSegments;
    private int _currentRound;

    /// <summary>
    /// Creates a new Segmented Gossip protocol.
    /// </summary>
    /// <param name="numSegments">Number of segments to split the model into. Default: 4.</param>
    public SegmentedGossipProtocol(int numSegments = 4)
    {
        if (numSegments < 2)
        {
            throw new ArgumentOutOfRangeException(nameof(numSegments), "Must have at least 2 segments.");
        }

        _numSegments = numSegments;
    }

    /// <summary>
    /// Performs one round of segmented gossip between two peers.
    /// </summary>
    /// <param name="myModel">This client's model.</param>
    /// <param name="peerModel">Peer's model.</param>
    /// <returns>Updated model after segment averaging.</returns>
    public Dictionary<string, T[]> GossipExchange(
        Dictionary<string, T[]> myModel,
        Dictionary<string, T[]> peerModel)
    {
        Guard.NotNull(myModel);
        Guard.NotNull(peerModel);
        if (myModel.Count == 0)
        {
            throw new ArgumentException("Model cannot be empty.", nameof(myModel));
        }

        // Use sorted keys for deterministic segment partitioning across peers.
        var layerNames = myModel.Keys.OrderBy(k => k, StringComparer.Ordinal).ToArray();
        int totalParams = layerNames.Sum(ln => myModel[ln].Length);
        if (totalParams == 0)
        {
            _currentRound++;
            return new Dictionary<string, T[]>(myModel);
        }

        int segmentSize = (totalParams + _numSegments - 1) / _numSegments;
        int activeSegment = _currentRound % _numSegments;
        int segmentStart = activeSegment * segmentSize;
        int segmentEnd = Math.Min(segmentStart + segmentSize, totalParams);

        var result = new Dictionary<string, T[]>(myModel.Count);
        int globalIdx = 0;

        foreach (var layerName in layerNames)
        {
            var myParams = myModel[layerName];
            var updated = new T[myParams.Length];

            if (peerModel.TryGetValue(layerName, out var peerParams))
            {
                for (int i = 0; i < myParams.Length; i++)
                {
                    if (globalIdx >= segmentStart && globalIdx < segmentEnd && i < peerParams.Length)
                    {
                        // Average this segment.
                        updated[i] = NumOps.FromDouble(
                            (NumOps.ToDouble(myParams[i]) + NumOps.ToDouble(peerParams[i])) / 2.0);
                    }
                    else
                    {
                        updated[i] = myParams[i];
                    }

                    globalIdx++;
                }
            }
            else
            {
                // Peer missing this layer — keep local values.
                Array.Copy(myParams, updated, myParams.Length);
                globalIdx += myParams.Length;
            }

            result[layerName] = updated;
        }

        Interlocked.Increment(ref _currentRound);
        return result;
    }

    /// <summary>Gets the number of segments.</summary>
    public int NumSegments => _numSegments;

    /// <summary>Gets the current round number.</summary>
    public int CurrentRound => _currentRound;

    /// <summary>Gets the communication compression ratio.</summary>
    public double CompressionRatio => 1.0 / _numSegments;
}
