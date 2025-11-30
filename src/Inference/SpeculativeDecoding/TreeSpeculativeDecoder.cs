namespace AiDotNet.Inference.SpeculativeDecoding;

/// <summary>
/// Tree-based speculative decoding for higher acceptance rates.
/// </summary>
/// <remarks>
/// <para>
/// Tree speculation extends standard speculative decoding by generating
/// multiple candidate continuations in a tree structure. This increases
/// the probability that at least one path will be accepted.
/// </para>
/// <para><b>For Beginners:</b> Instead of guessing one sequence of words,
/// tree speculation guesses multiple possible sequences at once.
///
/// Example:
/// Input: "The cat"
/// Standard draft: "sat on the mat"
/// Tree draft:
///   - "sat on the mat"
///   - "sat on the bed"
///   - "ran to the door"
///
/// If "sat on the mat" is wrong but "ran to" is right, we still get speedup!
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type.</typeparam>
public class TreeSpeculativeDecoder<T>
{
    private readonly IDraftModel<T> _draftModel;
    private readonly Func<int[][], float[][][]> _batchTargetForward;
    private readonly TreeSpeculativeConfig _config;
    private readonly Random _random;

    // Statistics
    private long _totalTokensGenerated;
    private long _totalTreeNodes;
    private long _acceptedNodes;

    /// <summary>
    /// Gets the configuration.
    /// </summary>
    public TreeSpeculativeConfig Config => _config;

    /// <summary>
    /// Gets the node acceptance rate.
    /// </summary>
    public double AcceptanceRate => _totalTreeNodes > 0
        ? (double)_acceptedNodes / _totalTreeNodes
        : 0;

    /// <summary>
    /// Creates a tree speculative decoder.
    /// </summary>
    /// <param name="draftModel">The draft model.</param>
    /// <param name="batchTargetForward">Batch target forward function.
    /// Takes array of sequences, returns probabilities for each.</param>
    /// <param name="config">Configuration.</param>
    public TreeSpeculativeDecoder(
        IDraftModel<T> draftModel,
        Func<int[][], float[][][]> batchTargetForward,
        TreeSpeculativeConfig? config = null)
    {
        _draftModel = draftModel ?? throw new ArgumentNullException(nameof(draftModel));
        _batchTargetForward = batchTargetForward ?? throw new ArgumentNullException(nameof(batchTargetForward));
        _config = config ?? new TreeSpeculativeConfig();
        _random = _config.Seed.HasValue ? new Random(_config.Seed.Value) : new Random();
    }

    /// <summary>
    /// Generates tokens using tree-based speculative decoding.
    /// </summary>
    public async Task<TreeSpeculativeResult> GenerateAsync(
        int[] inputTokens,
        int maxNewTokens,
        float temperature = 1.0f,
        int? eosToken = null,
        CancellationToken cancellationToken = default)
    {
        var tokens = new List<int>(inputTokens);
        int generated = 0;
        var stepStats = new List<TreeStepStatistics>();

        while (generated < maxNewTokens)
        {
            cancellationToken.ThrowIfCancellationRequested();

            // Build speculation tree
            var tree = BuildSpeculationTree([.. tokens], temperature);
            _totalTreeNodes += tree.TotalNodes;

            // Get all paths through the tree
            var paths = tree.GetAllPaths();

            // Build batch for verification
            var batchSequences = paths.Select(p =>
            {
                var seq = new int[tokens.Count + p.Length];
                tokens.CopyTo(seq, 0);
                Array.Copy(p, 0, seq, tokens.Count, p.Length);
                return seq;
            }).ToArray();

            // Verify all paths in parallel
            var allTargetProbs = await Task.Run(() => _batchTargetForward(batchSequences), cancellationToken);

            // Find best accepted path
            int bestPathIdx = -1;
            int bestAcceptedLength = 0;

            for (int pathIdx = 0; pathIdx < paths.Count; pathIdx++)
            {
                var path = paths[pathIdx];
                var targetProbs = allTargetProbs[pathIdx];
                var draftProbs = tree.GetPathProbabilities(pathIdx);

                int accepted = VerifyPath(path, draftProbs, targetProbs, tokens.Count, temperature);

                if (accepted > bestAcceptedLength)
                {
                    bestAcceptedLength = accepted;
                    bestPathIdx = pathIdx;
                }
            }

            _acceptedNodes += bestAcceptedLength;

            // Apply best path
            if (bestPathIdx >= 0 && bestAcceptedLength > 0)
            {
                var bestPath = paths[bestPathIdx];
                for (int i = 0; i < bestAcceptedLength; i++)
                {
                    tokens.Add(bestPath[i]);
                    generated++;

                    if (eosToken.HasValue && bestPath[i] == eosToken.Value)
                        goto done;
                }

                // If all accepted, add bonus token
                if (bestAcceptedLength == bestPath.Length && generated < maxNewTokens)
                {
                    var targetProbs = allTargetProbs[bestPathIdx];
                    int bonusPos = tokens.Count - 1;
                    if (bonusPos < targetProbs.Length)
                    {
                        int bonusToken = SampleFromDistribution(
                            ApplyTemperature(targetProbs[bonusPos], temperature));
                        tokens.Add(bonusToken);
                        generated++;

                        if (eosToken.HasValue && bonusToken == eosToken.Value)
                            goto done;
                    }
                }
            }
            else
            {
                // No path accepted - sample from target distribution
                var targetProbs = allTargetProbs[0];
                int pos = tokens.Count - 1;
                if (pos >= 0 && pos < targetProbs.Length)
                {
                    int fallbackToken = SampleFromDistribution(
                        ApplyTemperature(targetProbs[pos], temperature));
                    tokens.Add(fallbackToken);
                    generated++;

                    if (eosToken.HasValue && fallbackToken == eosToken.Value)
                        goto done;
                }
            }

            stepStats.Add(new TreeStepStatistics
            {
                TreeNodes = tree.TotalNodes,
                PathsExplored = paths.Count,
                BestPathLength = bestAcceptedLength
            });
        }

        done:
        _totalTokensGenerated += generated;

        return new TreeSpeculativeResult
        {
            Tokens = [.. tokens],
            NewTokens = [.. tokens.Skip(inputTokens.Length)],
            NumGenerated = generated,
            AcceptanceRate = AcceptanceRate,
            StepStatistics = stepStats
        };
    }

    /// <summary>
    /// Synchronous generation.
    /// </summary>
    public TreeSpeculativeResult Generate(
        int[] inputTokens,
        int maxNewTokens,
        float temperature = 1.0f,
        int? eosToken = null)
    {
        return GenerateAsync(inputTokens, maxNewTokens, temperature, eosToken).GetAwaiter().GetResult();
    }

    private SpeculationTree BuildSpeculationTree(int[] context, float temperature)
    {
        var tree = new SpeculationTree(_config.BranchFactor, _config.MaxDepth);

        // Root node
        var root = tree.Root;
        root.Context = context;

        // BFS to build tree
        var queue = new Queue<TreeNode>();
        queue.Enqueue(root);

        while (queue.Count > 0 && tree.TotalNodes < _config.MaxNodes)
        {
            var node = queue.Dequeue();
            if (node.Depth >= _config.MaxDepth)
                continue;

            // Generate draft continuations for this node
            var nodeContext = GetNodeContext(context, node);
            int numBranches = Math.Min(_config.BranchFactor, _config.MaxNodes - tree.TotalNodes);

            for (int b = 0; b < numBranches; b++)
            {
                var draft = _draftModel.GenerateDraft(nodeContext, 1, temperature);
                if (draft.NumTokens == 0) continue;

                var child = new TreeNode
                {
                    Token = draft.Tokens[0],
                    Probability = draft.TokenProbabilities[0],
                    Depth = node.Depth + 1,
                    Parent = node
                };

                node.Children.Add(child);
                tree.TotalNodes++;

                if (child.Depth < _config.MaxDepth)
                {
                    queue.Enqueue(child);
                }
            }
        }

        return tree;
    }

    private static int[] GetNodeContext(int[] baseContext, TreeNode node)
    {
        var pathTokens = new List<int>();
        var current = node;
        while (current.Parent != null)
        {
            pathTokens.Insert(0, current.Token);
            current = current.Parent;
        }

        var fullContext = new int[baseContext.Length + pathTokens.Count];
        Array.Copy(baseContext, fullContext, baseContext.Length);
        pathTokens.CopyTo(fullContext, baseContext.Length);

        return fullContext;
    }

    private int VerifyPath(
        int[] path,
        float[] draftProbs,
        float[][] targetProbs,
        int contextLength,
        float temperature)
    {
        int accepted = 0;

        for (int i = 0; i < path.Length; i++)
        {
            int token = path[i];
            int targetPos = contextLength + i - 1;

            if (targetPos < 0 || targetPos >= targetProbs.Length)
                break;

            float pTarget = targetProbs[targetPos][token];
            float pDraft = i < draftProbs.Length ? draftProbs[i] : 0.01f;

            if (pDraft <= 0)
            {
                if (pTarget > 0)
                    accepted++;
                else
                    break;
            }
            else
            {
                float acceptProb = Math.Min(1.0f, pTarget / pDraft);
                if ((float)_random.NextDouble() < acceptProb)
                    accepted++;
                else
                    break;
            }
        }

        return accepted;
    }

    private static float[] ApplyTemperature(float[] dist, float temperature)
    {
        if (Math.Abs(temperature - 1.0f) < 0.001f)
            return dist;

        var result = new float[dist.Length];
        float sum = 0;

        for (int i = 0; i < dist.Length; i++)
        {
            result[i] = MathF.Pow(dist[i], 1.0f / temperature);
            sum += result[i];
        }

        if (sum > 0)
        {
            for (int i = 0; i < result.Length; i++)
                result[i] /= sum;
        }

        return result;
    }

    private int SampleFromDistribution(float[] distribution)
    {
        float r = (float)_random.NextDouble();
        float cumulative = 0;

        for (int i = 0; i < distribution.Length; i++)
        {
            cumulative += distribution[i];
            if (r <= cumulative)
                return i;
        }

        return distribution.Length - 1;
    }
}
