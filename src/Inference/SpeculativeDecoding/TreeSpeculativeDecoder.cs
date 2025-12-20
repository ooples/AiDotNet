using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

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
internal class TreeSpeculativeDecoder<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    private readonly IDraftModel<T> _draftModel;
    private readonly Func<List<Vector<int>>, List<Matrix<T>>> _batchTargetForward;
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
    /// Takes list of sequences, returns probabilities for each as matrices [seq_len, vocab_size].</param>
    /// <param name="config">Configuration.</param>
    public TreeSpeculativeDecoder(
        IDraftModel<T> draftModel,
        Func<List<Vector<int>>, List<Matrix<T>>> batchTargetForward,
        TreeSpeculativeConfig? config = null)
    {
        _draftModel = draftModel ?? throw new ArgumentNullException(nameof(draftModel));
        _batchTargetForward = batchTargetForward ?? throw new ArgumentNullException(nameof(batchTargetForward));
        _config = config ?? new TreeSpeculativeConfig();
        _random = _config.Seed.HasValue
            ? RandomHelper.CreateSeededRandom(_config.Seed.Value)
            : RandomHelper.CreateSecureRandom();
    }

    /// <summary>
    /// Generates tokens using tree-based speculative decoding.
    /// </summary>
    /// <param name="inputTokens">Initial input tokens.</param>
    /// <param name="maxNewTokens">Maximum number of new tokens to generate.</param>
    /// <param name="temperature">Sampling temperature.</param>
    /// <param name="eosToken">End-of-sequence token ID (optional).</param>
    /// <param name="cancellationToken">Cancellation token.</param>
    /// <returns>Tree speculative result with tokens and statistics.</returns>
    public async Task<TreeSpeculativeResult> GenerateAsync(
        Vector<int> inputTokens,
        int maxNewTokens,
        T temperature,
        int? eosToken = null,
        CancellationToken cancellationToken = default)
    {
        var tokens = new List<int>();
        for (int i = 0; i < inputTokens.Length; i++)
        {
            tokens.Add(inputTokens[i]);
        }

        int generated = 0;
        var stepStats = new List<TreeStepStatistics>();

        while (generated < maxNewTokens)
        {
            cancellationToken.ThrowIfCancellationRequested();

            // Build speculation tree
            var currentContext = new Vector<int>(tokens.ToArray());
            var tree = BuildSpeculationTree(currentContext, temperature);
            _totalTreeNodes += tree.TotalNodes;

            // Get all paths through the tree
            var paths = tree.GetAllPaths();

            // Build batch for verification
            var batchSequences = new List<Vector<int>>();
            foreach (var path in paths)
            {
                var seq = new Vector<int>(tokens.Count + path.Length);
                for (int i = 0; i < tokens.Count; i++)
                {
                    seq[i] = tokens[i];
                }
                for (int i = 0; i < path.Length; i++)
                {
                    seq[tokens.Count + i] = path[i];
                }
                batchSequences.Add(seq);
            }

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
                    if (bonusPos < targetProbs.Rows)
                    {
                        var targetDist = targetProbs.GetRow(bonusPos);
                        int bonusToken = SampleFromDistribution(ApplyTemperature(targetDist, temperature));
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
                if (allTargetProbs.Count > 0)
                {
                    var targetProbs = allTargetProbs[0];
                    int pos = tokens.Count - 1;
                    if (pos >= 0 && pos < targetProbs.Rows)
                    {
                        var targetDist = targetProbs.GetRow(pos);
                        int fallbackToken = SampleFromDistribution(ApplyTemperature(targetDist, temperature));
                        tokens.Add(fallbackToken);
                        generated++;

                        if (eosToken.HasValue && fallbackToken == eosToken.Value)
                            goto done;
                    }
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

        var resultTokens = new Vector<int>(tokens.ToArray());
        var newTokens = new Vector<int>(generated);
        for (int i = 0; i < generated; i++)
        {
            newTokens[i] = tokens[inputTokens.Length + i];
        }

        return new TreeSpeculativeResult
        {
            Tokens = resultTokens,
            NewTokens = newTokens,
            NumGenerated = generated,
            AcceptanceRate = AcceptanceRate,
            StepStatistics = stepStats
        };
    }

    /// <summary>
    /// Synchronous generation.
    /// </summary>
    public TreeSpeculativeResult Generate(
        Vector<int> inputTokens,
        int maxNewTokens,
        T temperature,
        int? eosToken = null)
    {
        return GenerateAsync(inputTokens, maxNewTokens, temperature, eosToken).GetAwaiter().GetResult();
    }

    /// <summary>
    /// Resets generation statistics.
    /// </summary>
    public void ResetStatistics()
    {
        _totalTokensGenerated = 0;
        _totalTreeNodes = 0;
        _acceptedNodes = 0;
        _draftModel.Reset();
    }

    private SpeculationTree<T> BuildSpeculationTree(Vector<int> context, T temperature)
    {
        var tree = new SpeculationTree<T>(_config.BranchFactor, _config.MaxDepth);

        // Root node
        var root = tree.Root;
        root.Context = context;

        // BFS to build tree
        var queue = new Queue<TreeNode<T>>();
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

                var child = new TreeNode<T>
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

    private static Vector<int> GetNodeContext(Vector<int> baseContext, TreeNode<T> node)
    {
        var pathTokens = new List<int>();
        var current = node;
        while (current.Parent != null)
        {
            pathTokens.Insert(0, current.Token);
            current = current.Parent;
        }

        var fullContext = new Vector<int>(baseContext.Length + pathTokens.Count);
        for (int i = 0; i < baseContext.Length; i++)
        {
            fullContext[i] = baseContext[i];
        }
        for (int i = 0; i < pathTokens.Count; i++)
        {
            fullContext[baseContext.Length + i] = pathTokens[i];
        }

        return fullContext;
    }

    private int VerifyPath(
        Vector<int> path,
        Vector<T> draftProbs,
        Matrix<T> targetProbs,
        int contextLength,
        T temperature)
    {
        int accepted = 0;

        for (int i = 0; i < path.Length; i++)
        {
            int token = path[i];
            int targetPos = contextLength + i - 1;

            if (targetPos < 0 || targetPos >= targetProbs.Rows)
                break;

            T pTarget = targetProbs[targetPos, token];
            T pDraft = i < draftProbs.Length ? draftProbs[i] : NumOps.FromDouble(0.01);

            if (NumOps.LessThanOrEquals(pDraft, NumOps.Zero))
            {
                if (NumOps.GreaterThan(pTarget, NumOps.Zero))
                    accepted++;
                else
                    break;
            }
            else
            {
                T ratio = NumOps.Divide(pTarget, pDraft);
                T acceptProb = NumOps.LessThan(ratio, NumOps.One) ? ratio : NumOps.One;
                if (_random.NextDouble() < NumOps.ToDouble(acceptProb))
                    accepted++;
                else
                    break;
            }
        }

        return accepted;
    }

    private Vector<T> ApplyTemperature(Vector<T> dist, T temperature)
    {
        T one = NumOps.One;
        if (NumOps.Equals(temperature, one))
            return dist;

        var result = new Vector<T>(dist.Length);
        T sum = NumOps.Zero;
        T invTemp = NumOps.Divide(one, temperature);

        for (int i = 0; i < dist.Length; i++)
        {
            result[i] = NumOps.Power(dist[i], invTemp);
            sum = NumOps.Add(sum, result[i]);
        }

        if (NumOps.GreaterThan(sum, NumOps.Zero))
        {
            for (int i = 0; i < result.Length; i++)
            {
                result[i] = NumOps.Divide(result[i], sum);
            }
        }

        return result;
    }

    private int SampleFromDistribution(Vector<T> distribution)
    {
        T r = NumOps.FromDouble(_random.NextDouble());
        T cumulative = NumOps.Zero;

        for (int i = 0; i < distribution.Length; i++)
        {
            cumulative = NumOps.Add(cumulative, distribution[i]);
            if (NumOps.LessThanOrEquals(r, cumulative))
                return i;
        }

        return distribution.Length - 1;
    }
}
