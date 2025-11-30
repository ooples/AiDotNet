using AiDotNet.JitCompiler.IR;
using AiDotNet.JitCompiler.IR.Operations;

namespace AiDotNet.JitCompiler.Optimizations;

/// <summary>
/// Optimization pass that fuses multiple operations into single combined operations.
/// </summary>
/// <remarks>
/// <para>
/// Operation fusion is a critical optimization that combines multiple operations into
/// a single fused operation. This provides several benefits:
/// - Reduces memory traffic (intermediate results don't need to be written/read)
/// - Better cache utilization
/// - Kernel launch overhead reduction (for GPU execution)
/// - Opportunity for specialized implementations
/// </para>
/// <para><b>For Beginners:</b> This combines multiple steps into a single optimized step.
///
/// Think of it like cooking:
/// - Original: "Chop onions. Put onions in pan. Add oil to pan. Heat pan."
/// - Fused: "Sauté onions in oil" (one combined step instead of four!)
///
/// Why this helps:
/// - Fewer operations to execute
/// - Intermediate results don't need to be stored
/// - Can use specialized fast implementations
/// - Much better performance!
///
/// Common fusion patterns in neural networks:
/// 1. MatMul + Add → Linear layer (matrix multiply then add bias)
/// 2. Linear + ReLU → Fused linear activation
/// 3. Conv2D + BatchNorm → Fused convolution
/// 4. Add + Activation → Fused element-wise operation
///
/// Example:
/// Before:
///   t2 = MatMul(input, weights)
///   t3 = Add(t2, bias)
///   t4 = ReLU(t3)
///
/// After:
///   t4 = FusedDenseLayer(input, weights, bias, activation="ReLU")
///
/// This is ONE operation instead of THREE! Much faster and uses less memory.
/// </para>
/// </remarks>
public class OperationFusionPass : IOptimizationPass
{
    /// <summary>
    /// Gets the name of this optimization pass.
    /// </summary>
    public string Name => "Operation Fusion";

    /// <summary>
    /// Applies operation fusion optimization to an IR graph.
    /// </summary>
    public IRGraph Optimize(IRGraph graph)
    {
        // Copy operations to working list
        var operations = new List<IROp>(graph.Operations);
        var fusedOps = new HashSet<IROp>();
        var tensorMapping = new Dictionary<int, int>();

        // Apply fusion patterns (multiple passes to catch chained fusions)
        int fusionCount = 0;
        bool changed = true;
        int maxPasses = 5;
        int passCount = 0;

        while (changed && passCount < maxPasses)
        {
            changed = false;
            int beforeCount = fusionCount;

            // Pattern 1: MatMul + Add + Activation → FusedDenseLayer (3-op fusion first!)
            fusionCount += FuseMatMulAddActivation(operations, fusedOps, tensorMapping);

            // Pattern 2: MatMul + Add → FusedLinear
            fusionCount += FuseMatMulAdd(operations, fusedOps, tensorMapping);

            // Pattern 3: FusedLinear + Activation → FusedLinearActivation
            fusionCount += FuseLinearActivation(operations, fusedOps, tensorMapping);

            // Pattern 4: Add/Mul/etc + Activation → FusedElementwiseActivation
            fusionCount += FuseElementwiseActivation(operations, fusedOps, tensorMapping);

            // Pattern 5: Conv2D + BatchNorm → FusedConvBatchNorm
            fusionCount += FuseConvBatchNorm(operations, fusedOps, tensorMapping);

            // Pattern 6: Conv2D + Add (bias) → Conv2D with bias
            fusionCount += FuseConv2DAdd(operations, fusedOps, tensorMapping);

            // Pattern 7: Add (residual) + Activation → FusedResidualBlock
            fusionCount += FuseResidualActivation(operations, fusedOps, tensorMapping);

            // Pattern 8: BatchNorm + Activation → FusedBatchNormActivation
            fusionCount += FuseBatchNormActivation(operations, fusedOps, tensorMapping);

            // Pattern 9: LayerNorm + Add → FusedLayerNormAdd
            fusionCount += FuseLayerNormAdd(operations, fusedOps, tensorMapping);

            // Pattern 10: Multiple consecutive element-wise ops → FusedElementwiseChain
            fusionCount += FuseElementwiseChain(operations, fusedOps, tensorMapping);

            // Pattern 11: Attention pattern (MatMul + Softmax + MatMul)
            fusionCount += FuseAttentionPattern(operations, fusedOps, tensorMapping);

            // Pattern 12: GELU approximation pattern
            fusionCount += FuseGELUPattern(operations, fusedOps, tensorMapping);

            // Pattern 13: Conv2D + BatchNorm + Activation
            fusionCount += FuseConvBatchNormActivation(operations, fusedOps, tensorMapping);

            // Pattern 14: Add + LayerNorm (common in transformers)
            fusionCount += FuseAddLayerNorm(operations, fusedOps, tensorMapping);

            changed = (fusionCount > beforeCount);
            passCount++;
        }

        // Build optimized graph
        var optimizedGraph = new IRGraph
        {
            InputIds = new List<int>(graph.InputIds),
            OutputIds = new List<int>(graph.OutputIds),
            TensorShapes = new Dictionary<int, int[]>(graph.TensorShapes),
            Metadata = new Dictionary<string, object>(graph.Metadata)
        };

        // Add non-fused operations
        foreach (var op in operations)
        {
            if (!fusedOps.Contains(op))
            {
                // Remap input tensor IDs if they were fused
                var remappedInputs = op.InputIds.Select(id =>
                    tensorMapping.TryGetValue(id, out var newId) ? newId : id).ToArray();
                op.InputIds = remappedInputs;
                optimizedGraph.Operations.Add(op);
            }
        }

        // Add metadata
        if (fusionCount > 0)
        {
            optimizedGraph.Metadata["Fusion_Count"] = fusionCount;
            optimizedGraph.Metadata["Fusion_OriginalOps"] = graph.Operations.Count;
            optimizedGraph.Metadata["Fusion_OptimizedOps"] = optimizedGraph.Operations.Count;
        }

        return optimizedGraph;
    }

    private int FuseMatMulAdd(List<IROp> operations, HashSet<IROp> fusedOps, Dictionary<int, int> tensorMapping)
    {
        int count = 0;

        for (int i = 0; i < operations.Count - 1; i++)
        {
            if (fusedOps.Contains(operations[i])) continue;
            if (operations[i] is not MatMulOp matmul) continue;

            var matmulOutput = matmul.OutputId;

            // Find Add using MatMul output
            for (int j = i + 1; j < operations.Count; j++)
            {
                if (fusedOps.Contains(operations[j])) continue;
                if (operations[j] is not AddOp add) continue;
                if (!add.InputIds.Contains(matmulOutput)) continue;

                // Check that MatMul output is only used by this Add (single consumer)
                if (CountUsages(operations, matmulOutput, fusedOps) != 1) continue;

                // Create fused operation
                var fusedOp = new FusedLinearOp
                {
                    OutputId = add.OutputId,
                    InputIds = new[] { matmul.InputIds[0], matmul.InputIds[1], add.InputIds[0] == matmulOutput ? add.InputIds[1] : add.InputIds[0] },
                    OutputType = add.OutputType,
                    OutputShape = add.OutputShape
                };

                operations[i] = fusedOp;
                fusedOps.Add(matmul);
                fusedOps.Add(add);
                tensorMapping[matmulOutput] = add.OutputId;
                count++;
                break;
            }
        }

        return count;
    }

    private int FuseLinearActivation(List<IROp> operations, HashSet<IROp> fusedOps, Dictionary<int, int> tensorMapping)
    {
        int count = 0;

        for (int i = 0; i < operations.Count - 1; i++)
        {
            if (fusedOps.Contains(operations[i])) continue;
            if (operations[i] is not FusedLinearOp linear) continue;

            var linearOutput = linear.OutputId;

            // Find activation using Linear output
            for (int j = i + 1; j < operations.Count; j++)
            {
                if (fusedOps.Contains(operations[j])) continue;

                string? activationName = operations[j] switch
                {
                    ReLUOp => "ReLU",
                    SigmoidOp => "Sigmoid",
                    TanhOp => "Tanh",
                    _ => null
                };

                if (activationName == null) continue;
                if (operations[j].InputIds.Length != 1 || operations[j].InputIds[0] != linearOutput) continue;
                if (CountUsages(operations, linearOutput, fusedOps) != 1) continue;

                // Create fused operation
                var fusedOp = new FusedLinearActivationOp
                {
                    OutputId = operations[j].OutputId,
                    InputIds = linear.InputIds,
                    OutputType = operations[j].OutputType,
                    OutputShape = operations[j].OutputShape,
                    ActivationName = activationName
                };

                operations[i] = fusedOp;
                fusedOps.Add(linear);
                fusedOps.Add(operations[j]);
                tensorMapping[linearOutput] = operations[j].OutputId;
                count++;
                break;
            }
        }

        return count;
    }

    private int FuseMatMulAddActivation(List<IROp> operations, HashSet<IROp> fusedOps, Dictionary<int, int> tensorMapping)
    {
        int count = 0;

        for (int i = 0; i < operations.Count - 2; i++)
        {
            if (fusedOps.Contains(operations[i])) continue;
            if (operations[i] is not MatMulOp matmul) continue;

            var matmulOutput = matmul.OutputId;

            // Find Add using MatMul output
            for (int j = i + 1; j < operations.Count; j++)
            {
                if (fusedOps.Contains(operations[j])) continue;
                if (operations[j] is not AddOp add) continue;
                if (!add.InputIds.Contains(matmulOutput)) continue;
                if (CountUsages(operations, matmulOutput, fusedOps) != 1) continue;

                var addOutput = add.OutputId;

                // Find activation using Add output
                for (int k = j + 1; k < operations.Count; k++)
                {
                    if (fusedOps.Contains(operations[k])) continue;

                    string? activationName = operations[k] switch
                    {
                        ReLUOp => "ReLU",
                        SigmoidOp => "Sigmoid",
                        TanhOp => "Tanh",
                        _ => null
                    };

                    if (activationName == null) continue;
                    if (operations[k].InputIds.Length != 1 || operations[k].InputIds[0] != addOutput) continue;
                    if (CountUsages(operations, addOutput, fusedOps) != 1) continue;

                    // Create fused 3-operation operation!
                    var fusedOp = new FusedDenseLayerOp
                    {
                        OutputId = operations[k].OutputId,
                        InputIds = new[] { matmul.InputIds[0], matmul.InputIds[1], add.InputIds[0] == matmulOutput ? add.InputIds[1] : add.InputIds[0] },
                        OutputType = operations[k].OutputType,
                        OutputShape = operations[k].OutputShape,
                        ActivationName = activationName
                    };

                    operations[i] = fusedOp;
                    fusedOps.Add(matmul);
                    fusedOps.Add(add);
                    fusedOps.Add(operations[k]);
                    tensorMapping[matmulOutput] = operations[k].OutputId;
                    tensorMapping[addOutput] = operations[k].OutputId;
                    count++;
                    break;
                }
            }
        }

        return count;
    }

    private int FuseElementwiseActivation(List<IROp> operations, HashSet<IROp> fusedOps, Dictionary<int, int> tensorMapping)
    {
        int count = 0;

        for (int i = 0; i < operations.Count - 1; i++)
        {
            if (fusedOps.Contains(operations[i])) continue;

            string? elementwiseOp = operations[i] switch
            {
                AddOp => "Add",
                SubtractOp => "Subtract",
                ElementwiseMultiplyOp => "Multiply",
                DivideOp => "Divide",
                _ => null
            };

            if (elementwiseOp == null) continue;
            if (operations[i].InputIds.Length != 2) continue;

            var elemwiseOutput = operations[i].OutputId;

            // Find activation
            for (int j = i + 1; j < operations.Count; j++)
            {
                if (fusedOps.Contains(operations[j])) continue;

                string? activationName = operations[j] switch
                {
                    ReLUOp => "ReLU",
                    SigmoidOp => "Sigmoid",
                    TanhOp => "Tanh",
                    _ => null
                };

                if (activationName == null) continue;
                if (operations[j].InputIds.Length != 1 || operations[j].InputIds[0] != elemwiseOutput) continue;
                if (CountUsages(operations, elemwiseOutput, fusedOps) != 1) continue;

                // Create fused operation
                var fusedOp = new FusedElementwiseActivationOp
                {
                    OutputId = operations[j].OutputId,
                    InputIds = operations[i].InputIds,
                    OutputType = operations[j].OutputType,
                    OutputShape = operations[j].OutputShape,
                    ElementwiseOp = elementwiseOp,
                    ActivationName = activationName
                };

                operations[i] = fusedOp;
                fusedOps.Add(operations[i]);
                fusedOps.Add(operations[j]);
                tensorMapping[elemwiseOutput] = operations[j].OutputId;
                count++;
                break;
            }
        }

        return count;
    }

    private int FuseConvBatchNorm(List<IROp> operations, HashSet<IROp> fusedOps, Dictionary<int, int> tensorMapping)
    {
        int count = 0;

        for (int i = 0; i < operations.Count - 1; i++)
        {
            if (fusedOps.Contains(operations[i])) continue;
            if (operations[i] is not Conv2DOp conv) continue;

            var convOutput = conv.OutputId;

            // Find BatchNorm using Conv output
            for (int j = i + 1; j < operations.Count; j++)
            {
                if (fusedOps.Contains(operations[j])) continue;
                if (operations[j] is not BatchNormOp bn) continue;
                if (bn.InputIds.Length < 1 || bn.InputIds[0] != convOutput) continue;
                if (CountUsages(operations, convOutput, fusedOps) != 1) continue;

                // Create fused operation
                var fusedOp = new FusedConvBatchNormOp
                {
                    OutputId = bn.OutputId,
                    InputIds = new[] { conv.InputIds[0], conv.InputIds[1], bn.InputIds[1], bn.InputIds[2], bn.InputIds[3], bn.InputIds[4] },
                    OutputType = bn.OutputType,
                    OutputShape = bn.OutputShape,
                    Stride = conv.Stride,
                    Padding = conv.Padding,
                    Epsilon = bn.Epsilon,
                    Momentum = bn.Momentum
                };

                operations[i] = fusedOp;
                fusedOps.Add(conv);
                fusedOps.Add(bn);
                tensorMapping[convOutput] = bn.OutputId;
                count++;
                break;
            }
        }

        return count;
    }

    private int FuseConv2DAdd(List<IROp> operations, HashSet<IROp> fusedOps, Dictionary<int, int> tensorMapping)
    {
        int count = 0;

        for (int i = 0; i < operations.Count - 1; i++)
        {
            if (fusedOps.Contains(operations[i])) continue;
            if (operations[i] is not Conv2DOp conv) continue;
            if (conv.HasBias) continue;

            var convOutput = conv.OutputId;

            // Find Add using Conv output
            for (int j = i + 1; j < operations.Count; j++)
            {
                if (fusedOps.Contains(operations[j])) continue;
                if (operations[j] is not AddOp add) continue;
                if (!add.InputIds.Contains(convOutput)) continue;
                if (CountUsages(operations, convOutput, fusedOps) != 1) continue;

                // Modify conv to include bias
                conv.HasBias = true;
                conv.InputIds = new[] { conv.InputIds[0], conv.InputIds[1], add.InputIds[0] == convOutput ? add.InputIds[1] : add.InputIds[0] };
                conv.OutputId = add.OutputId;
                conv.OutputShape = add.OutputShape;

                fusedOps.Add(add);
                tensorMapping[convOutput] = add.OutputId;
                count++;
                break;
            }
        }

        return count;
    }

    private int FuseResidualActivation(List<IROp> operations, HashSet<IROp> fusedOps, Dictionary<int, int> tensorMapping)
    {
        int count = 0;

        for (int i = 0; i < operations.Count - 1; i++)
        {
            if (fusedOps.Contains(operations[i])) continue;
            if (operations[i] is not AddOp add) continue;

            var addOutput = add.OutputId;

            // Find activation using Add output
            for (int j = i + 1; j < operations.Count; j++)
            {
                if (fusedOps.Contains(operations[j])) continue;

                string? activationName = operations[j] switch
                {
                    ReLUOp => "ReLU",
                    SigmoidOp => "Sigmoid",
                    TanhOp => "Tanh",
                    _ => null
                };

                if (activationName == null) continue;
                if (operations[j].InputIds.Length != 1 || operations[j].InputIds[0] != addOutput) continue;
                if (CountUsages(operations, addOutput, fusedOps) != 1) continue;

                // Check if this looks like a residual connection
                // (both inputs to Add should come from different operations)
                bool looksLikeResidual = add.InputIds[0] != add.InputIds[1];

                if (!looksLikeResidual) continue;

                // Create fused residual block
                var fusedOp = new FusedResidualBlockOp
                {
                    OutputId = operations[j].OutputId,
                    InputIds = add.InputIds,
                    OutputType = operations[j].OutputType,
                    OutputShape = operations[j].OutputShape,
                    ActivationName = activationName
                };

                operations[i] = fusedOp;
                fusedOps.Add(add);
                fusedOps.Add(operations[j]);
                tensorMapping[addOutput] = operations[j].OutputId;
                count++;
                break;
            }
        }

        return count;
    }

    /// <summary>
    /// Counts how many operations use a given tensor as input.
    /// </summary>
    private int CountUsages(List<IROp> operations, int tensorId, HashSet<IROp> fusedOps)
    {
        int count = 0;
        foreach (var op in operations)
        {
            if (fusedOps.Contains(op)) continue;
            if (op.InputIds.Contains(tensorId)) count++;
        }
        return count;
    }

    private int FuseBatchNormActivation(List<IROp> operations, HashSet<IROp> fusedOps, Dictionary<int, int> tensorMapping)
    {
        int count = 0;

        for (int i = 0; i < operations.Count - 1; i++)
        {
            if (fusedOps.Contains(operations[i])) continue;
            if (operations[i] is not BatchNormOp bn) continue;

            var bnOutput = bn.OutputId;

            // Find activation using BatchNorm output
            for (int j = i + 1; j < operations.Count; j++)
            {
                if (fusedOps.Contains(operations[j])) continue;

                string? activationName = operations[j] switch
                {
                    ReLUOp => "ReLU",
                    SigmoidOp => "Sigmoid",
                    TanhOp => "Tanh",
                    _ => null
                };

                if (activationName == null) continue;
                if (operations[j].InputIds.Length != 1 || operations[j].InputIds[0] != bnOutput) continue;
                if (CountUsages(operations, bnOutput, fusedOps) != 1) continue;

                // Create fused operation
                var fusedOp = new FusedBatchNormActivationOp
                {
                    OutputId = operations[j].OutputId,
                    InputIds = bn.InputIds,
                    OutputType = operations[j].OutputType,
                    OutputShape = operations[j].OutputShape,
                    ActivationName = activationName,
                    Epsilon = bn.Epsilon,
                    Momentum = bn.Momentum
                };

                operations[i] = fusedOp;
                fusedOps.Add(bn);
                fusedOps.Add(operations[j]);
                tensorMapping[bnOutput] = operations[j].OutputId;
                count++;
                break;
            }
        }

        return count;
    }

    private int FuseLayerNormAdd(List<IROp> operations, HashSet<IROp> fusedOps, Dictionary<int, int> tensorMapping)
    {
        int count = 0;

        for (int i = 0; i < operations.Count - 1; i++)
        {
            if (fusedOps.Contains(operations[i])) continue;
            if (operations[i] is not LayerNormOp ln) continue;

            var lnOutput = ln.OutputId;

            // Find Add using LayerNorm output (for residual connections)
            for (int j = i + 1; j < operations.Count; j++)
            {
                if (fusedOps.Contains(operations[j])) continue;
                if (operations[j] is not AddOp add) continue;
                if (!add.InputIds.Contains(lnOutput)) continue;
                if (CountUsages(operations, lnOutput, fusedOps) != 1) continue;

                // Get the other input to Add (the residual)
                var residualId = add.InputIds[0] == lnOutput ? add.InputIds[1] : add.InputIds[0];

                // Create fused operation
                var fusedOp = new FusedLayerNormAddOp
                {
                    OutputId = add.OutputId,
                    InputIds = new[] { ln.InputIds[0], ln.InputIds[1], ln.InputIds[2], residualId },
                    OutputType = add.OutputType,
                    OutputShape = add.OutputShape,
                    NormalizedShape = ln.NormalizedShape,
                    Epsilon = ln.Epsilon
                };

                operations[i] = fusedOp;
                fusedOps.Add(ln);
                fusedOps.Add(add);
                tensorMapping[lnOutput] = add.OutputId;
                count++;
                break;
            }
        }

        return count;
    }

    private int FuseElementwiseChain(List<IROp> operations, HashSet<IROp> fusedOps, Dictionary<int, int> tensorMapping)
    {
        int count = 0;
        const int MAX_CHAIN_LENGTH = 4;

        for (int i = 0; i < operations.Count - 1; i++)
        {
            if (fusedOps.Contains(operations[i])) continue;
            if (!IsElementWiseOp(operations[i])) continue;

            // Build a chain of element-wise operations
            var chain = new List<IROp> { operations[i] };
            var chainOps = new List<string> { GetElementWiseOpName(operations[i]) };
            var currentOutput = operations[i].OutputId;

            for (int j = i + 1; j < operations.Count && chain.Count < MAX_CHAIN_LENGTH; j++)
            {
                if (fusedOps.Contains(operations[j])) continue;
                if (!IsElementWiseOp(operations[j])) break;
                if (operations[j].InputIds.Length != 1 || operations[j].InputIds[0] != currentOutput) break;
                if (CountUsages(operations, currentOutput, fusedOps) != 1) break;

                chain.Add(operations[j]);
                chainOps.Add(GetElementWiseOpName(operations[j]));
                currentOutput = operations[j].OutputId;
            }

            // Only fuse if we have 3+ operations
            if (chain.Count >= 3)
            {
                var fusedOp = new FusedElementwiseChainOp
                {
                    OutputId = chain[^1].OutputId,
                    InputIds = chain[0].InputIds,
                    OutputType = chain[^1].OutputType,
                    OutputShape = chain[^1].OutputShape,
                    OperationNames = chainOps
                };

                operations[i] = fusedOp;
                foreach (var op in chain)
                {
                    fusedOps.Add(op);
                    if (op != chain[^1])
                    {
                        tensorMapping[op.OutputId] = chain[^1].OutputId;
                    }
                }
                count++;
            }
        }

        return count;
    }

    private int FuseAttentionPattern(List<IROp> operations, HashSet<IROp> fusedOps, Dictionary<int, int> tensorMapping)
    {
        int count = 0;

        // Look for pattern: MatMul(Q, K^T) -> Softmax -> MatMul(_, V)
        for (int i = 0; i < operations.Count - 2; i++)
        {
            if (fusedOps.Contains(operations[i])) continue;
            if (operations[i] is not MatMulOp matmul1) continue;

            var matmul1Output = matmul1.OutputId;

            // Find Softmax using MatMul output
            for (int j = i + 1; j < operations.Count; j++)
            {
                if (fusedOps.Contains(operations[j])) continue;
                if (operations[j] is not SoftmaxOp softmax) continue;
                if (softmax.InputIds.Length != 1 || softmax.InputIds[0] != matmul1Output) continue;
                if (CountUsages(operations, matmul1Output, fusedOps) != 1) continue;

                var softmaxOutput = softmax.OutputId;

                // Find second MatMul using Softmax output
                for (int k = j + 1; k < operations.Count; k++)
                {
                    if (fusedOps.Contains(operations[k])) continue;
                    if (operations[k] is not MatMulOp matmul2) continue;
                    if (!matmul2.InputIds.Contains(softmaxOutput)) continue;
                    if (CountUsages(operations, softmaxOutput, fusedOps) != 1) continue;

                    // Found attention pattern!
                    var vId = matmul2.InputIds[0] == softmaxOutput ? matmul2.InputIds[1] : matmul2.InputIds[0];

                    var fusedOp = new FusedAttentionOp
                    {
                        OutputId = matmul2.OutputId,
                        InputIds = new[] { matmul1.InputIds[0], matmul1.InputIds[1], vId }, // Q, K, V
                        OutputType = matmul2.OutputType,
                        OutputShape = matmul2.OutputShape,
                        SoftmaxAxis = softmax.Axis
                    };

                    operations[i] = fusedOp;
                    fusedOps.Add(matmul1);
                    fusedOps.Add(softmax);
                    fusedOps.Add(matmul2);
                    tensorMapping[matmul1Output] = matmul2.OutputId;
                    tensorMapping[softmaxOutput] = matmul2.OutputId;
                    count++;
                    break;
                }
            }
        }

        return count;
    }

    private int FuseGELUPattern(List<IROp> operations, HashSet<IROp> fusedOps, Dictionary<int, int> tensorMapping)
    {
        // GELU approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
        // This is complex to detect, so we look for simpler patterns
        int count = 0;

        for (int i = 0; i < operations.Count - 1; i++)
        {
            if (fusedOps.Contains(operations[i])) continue;

            // Look for Mul -> Sigmoid pattern (simplified GELU/Swish)
            if (operations[i] is ElementwiseMultiplyOp mul)
            {
                var mulOutput = mul.OutputId;

                // Check if this is x * sigmoid(x) pattern (Swish/SiLU)
                for (int j = 0; j < operations.Count; j++)
                {
                    if (i == j) continue;
                    if (fusedOps.Contains(operations[j])) continue;
                    if (operations[j] is not SigmoidOp sigmoid) continue;

                    // Check if sigmoid input and one mul input are the same
                    if (sigmoid.InputIds[0] == mul.InputIds[0] || sigmoid.InputIds[0] == mul.InputIds[1])
                    {
                        var otherMulInput = sigmoid.InputIds[0] == mul.InputIds[0] ? mul.InputIds[1] : mul.InputIds[0];
                        if (otherMulInput == sigmoid.OutputId)
                        {
                            // Found x * sigmoid(x) = Swish pattern
                            var fusedOp = new FusedSwishOp
                            {
                                OutputId = mul.OutputId,
                                InputIds = new[] { sigmoid.InputIds[0] },
                                OutputType = mul.OutputType,
                                OutputShape = mul.OutputShape
                            };

                            operations[i] = fusedOp;
                            fusedOps.Add(mul);
                            fusedOps.Add(sigmoid);
                            count++;
                            break;
                        }
                    }
                }
            }
        }

        return count;
    }

    private int FuseConvBatchNormActivation(List<IROp> operations, HashSet<IROp> fusedOps, Dictionary<int, int> tensorMapping)
    {
        int count = 0;

        for (int i = 0; i < operations.Count - 2; i++)
        {
            if (fusedOps.Contains(operations[i])) continue;
            if (operations[i] is not FusedConvBatchNormOp convBn) continue;

            var convBnOutput = convBn.OutputId;

            // Find activation using FusedConvBatchNorm output
            for (int j = i + 1; j < operations.Count; j++)
            {
                if (fusedOps.Contains(operations[j])) continue;

                string? activationName = operations[j] switch
                {
                    ReLUOp => "ReLU",
                    SigmoidOp => "Sigmoid",
                    TanhOp => "Tanh",
                    _ => null
                };

                if (activationName == null) continue;
                if (operations[j].InputIds.Length != 1 || operations[j].InputIds[0] != convBnOutput) continue;
                if (CountUsages(operations, convBnOutput, fusedOps) != 1) continue;

                // Create fused operation
                var fusedOp = new FusedConvBatchNormActivationOp
                {
                    OutputId = operations[j].OutputId,
                    InputIds = convBn.InputIds,
                    OutputType = operations[j].OutputType,
                    OutputShape = operations[j].OutputShape,
                    Stride = convBn.Stride,
                    Padding = convBn.Padding,
                    Epsilon = convBn.Epsilon,
                    Momentum = convBn.Momentum,
                    ActivationName = activationName
                };

                operations[i] = fusedOp;
                fusedOps.Add(convBn);
                fusedOps.Add(operations[j]);
                tensorMapping[convBnOutput] = operations[j].OutputId;
                count++;
                break;
            }
        }

        return count;
    }

    private int FuseAddLayerNorm(List<IROp> operations, HashSet<IROp> fusedOps, Dictionary<int, int> tensorMapping)
    {
        int count = 0;

        for (int i = 0; i < operations.Count - 1; i++)
        {
            if (fusedOps.Contains(operations[i])) continue;
            if (operations[i] is not AddOp add) continue;

            var addOutput = add.OutputId;

            // Find LayerNorm using Add output
            for (int j = i + 1; j < operations.Count; j++)
            {
                if (fusedOps.Contains(operations[j])) continue;
                if (operations[j] is not LayerNormOp ln) continue;
                if (ln.InputIds.Length < 1 || ln.InputIds[0] != addOutput) continue;
                if (CountUsages(operations, addOutput, fusedOps) != 1) continue;

                // Create fused operation (Add + LayerNorm)
                var fusedOp = new FusedAddLayerNormOp
                {
                    OutputId = ln.OutputId,
                    InputIds = new[] { add.InputIds[0], add.InputIds[1], ln.InputIds[1], ln.InputIds[2] },
                    OutputType = ln.OutputType,
                    OutputShape = ln.OutputShape,
                    NormalizedShape = ln.NormalizedShape,
                    Epsilon = ln.Epsilon
                };

                operations[i] = fusedOp;
                fusedOps.Add(add);
                fusedOps.Add(ln);
                tensorMapping[addOutput] = ln.OutputId;
                count++;
                break;
            }
        }

        return count;
    }

    private bool IsElementWiseOp(IROp op)
    {
        return op is AddOp or SubtractOp or ElementwiseMultiplyOp or DivideOp or
               NegateOp or ReLUOp or SigmoidOp or TanhOp or ExpOp or LogOp or SqrtOp;
    }

    private string GetElementWiseOpName(IROp op)
    {
        return op switch
        {
            AddOp => "Add",
            SubtractOp => "Subtract",
            ElementwiseMultiplyOp => "Multiply",
            DivideOp => "Divide",
            NegateOp => "Negate",
            ReLUOp => "ReLU",
            SigmoidOp => "Sigmoid",
            TanhOp => "Tanh",
            ExpOp => "Exp",
            LogOp => "Log",
            SqrtOp => "Sqrt",
            _ => "Unknown"
        };
    }

    /// <summary>
    /// Identifies fusion opportunities in a graph without applying them (for analysis).
    /// </summary>
    public List<string> IdentifyFusionOpportunities(IRGraph graph)
    {
        var opportunities = new List<string>();
        var operations = graph.Operations;

        for (int i = 0; i < operations.Count - 1; i++)
        {
            var op1 = operations[i];

            for (int j = i + 1; j < operations.Count; j++)
            {
                var op2 = operations[j];

                // Check if op2 uses op1's output
                if (op2.InputIds.Contains(op1.OutputId))
                {
                    // Check for known patterns
                    if (op1 is MatMulOp && op2 is AddOp)
                    {
                        opportunities.Add($"MatMul+Add fusion: t{op1.OutputId} → t{op2.OutputId}");
                    }
                    else if (op1 is Conv2DOp && op2 is AddOp)
                    {
                        opportunities.Add($"Conv2D+Add fusion: t{op1.OutputId} → t{op2.OutputId}");
                    }
                    else if (op1 is Conv2DOp && op2 is BatchNormOp)
                    {
                        opportunities.Add($"Conv2D+BatchNorm fusion: t{op1.OutputId} → t{op2.OutputId}");
                    }
                    else if ((op1 is AddOp or SubtractOp or ElementwiseMultiplyOp) &&
                             (op2 is ReLUOp or SigmoidOp or TanhOp))
                    {
                        opportunities.Add($"{op1.OpType}+{op2.OpType} fusion: t{op1.OutputId} → t{op2.OutputId}");
                    }
                }
            }
        }

        return opportunities;
    }
}
