using System;
using System.Threading.Tasks;
using AiDotNet.LinearAlgebra;
using AiDotNet.Tensors.Engines.Simd;

namespace AiDotNet.InferenceOptimization.Kernels
{
    /// <summary>
    /// Fused attention kernel for transformer models
    /// Implements optimized scaled dot-product attention: softmax(QK^T/sqrt(d_k))V
    /// </summary>
    public class AttentionKernel : ICustomOperator<float>
    {
        public string Name => "FusedAttention";
        public string Version => "1.0.0";
        public int Priority => 100;

        public AttentionKernel() { }

        public bool IsSupported()
        {
            return true;
        }

        public double EstimatedSpeedup()
        {
            // Fused attention reduces memory traffic significantly
            return 2.5;
        }

        public Tensor<float> Execute(params Tensor<float>[] inputs)
        {
            if (inputs == null || inputs.Length < 3)
                throw new ArgumentException("Attention requires Q, K, V tensors");

            var q = inputs[0]; // [batch_size, seq_len_q, d_k]
            var k = inputs[1]; // [batch_size, seq_len_k, d_k]
            var v = inputs[2]; // [batch_size, seq_len_v, d_v]

            bool useMask = inputs.Length > 3;
            Tensor<float>? mask = useMask ? inputs[3] : null;

            return ExecuteInternal(q, k, v, mask, maskBatchModulo: q.Shape.Length == 3 ? q.Shape[0] : 0);
        }

        private Tensor<float> ExecuteInternal(
            Tensor<float> q,
            Tensor<float> k,
            Tensor<float> v,
            Tensor<float>? mask,
            int maskBatchModulo)
        {
            if (q.Shape.Length != 3 || k.Shape.Length != 3 || v.Shape.Length != 3)
                throw new ArgumentException("Attention requires 3D tensors [batch, seq_len, features]");

            int batchSize = q.Shape[0];
            int seqLenQ = q.Shape[1];
            int seqLenK = k.Shape[1];
            int dK = q.Shape[2];
            int dV = v.Shape[2];

            if (k.Shape[0] != batchSize || v.Shape[0] != batchSize)
                throw new ArgumentException("Q, K, and V must have the same batch size");

            if (k.Shape[2] != dK)
                throw new ArgumentException("Q and K must have same feature dimension");

            if (v.Shape[1] != seqLenK)
                throw new ArgumentException("K and V must have same sequence length");

            if (mask != null)
            {
                if (mask.Shape.Length != 3)
                    throw new ArgumentException("Attention mask must be a 3D tensor [batch, seq_len_q, seq_len_k]");

                if (mask.Shape[1] != seqLenQ || mask.Shape[2] != seqLenK)
                    throw new ArgumentException("Attention mask must match [batch, seq_len_q, seq_len_k]");

                if (maskBatchModulo <= 0)
                {
                    if (mask.Shape[0] != batchSize)
                        throw new ArgumentException("Attention mask must have the same batch size as Q when used in Execute()");
                }
                else
                {
                    if (mask.Shape[0] != maskBatchModulo)
                        throw new ArgumentException("Attention mask batch dimension must match the provided maskBatchModulo");
                }
            }

            var result = new Tensor<float>(new[] { batchSize, seqLenQ, dV });

            // Process each batch in parallel
            Parallel.For(0, batchSize, b =>
            {
                ProcessBatch(q, k, v, mask, result, b, seqLenQ, seqLenK, dK, dV, maskBatchModulo);
            });

            return result;
        }

        private void ProcessBatch(
            Tensor<float> q, Tensor<float> k, Tensor<float> v,
            Tensor<float>? mask, Tensor<float> result,
            int batchIdx, int seqLenQ, int seqLenK, int dK, int dV,
            int maskBatchModulo)
        {
            float scale = 1.0f / MathF.Sqrt(dK);

            // Extract batch slices
            int qOffset = batchIdx * seqLenQ * dK;
            int kOffset = batchIdx * seqLenK * dK;
            int vOffset = batchIdx * seqLenK * dV;
            int outOffset = batchIdx * seqLenQ * dV;

            // Compute attention scores: QK^T
            var scores = new float[seqLenQ * seqLenK];

            for (int i = 0; i < seqLenQ; i++)
            {
                int qRowOffset = qOffset + i * dK;
                var qRow = q.Data.AsSpan(qRowOffset, dK);

                for (int j = 0; j < seqLenK; j++)
                {
                    int kRowOffset = kOffset + j * dK;
                    var kRow = k.Data.AsSpan(kRowOffset, dK);
                    float score = SimdKernels.DotProduct(qRow, kRow) * scale;

                    // Apply mask if provided
                    if (mask != null)
                    {
                        int effectiveMaskBatch = maskBatchModulo > 0 ? (batchIdx % maskBatchModulo) : batchIdx;
                        int maskIdx = effectiveMaskBatch * seqLenQ * seqLenK + i * seqLenK + j;
                        // Use epsilon-based comparison for floating point equality
                        if (MathF.Abs(mask.Data[maskIdx]) < 1e-6f)
                        {
                            score = float.NegativeInfinity;
                        }
                    }

                    scores[i * seqLenK + j] = score;
                }
            }

            // Apply softmax over each row
            ApplySoftmax(scores, seqLenQ, seqLenK);

            // Compute weighted sum: attention_weights * V
            for (int i = 0; i < seqLenQ; i++)
            {
                var outRow = result.Data.AsSpan(outOffset + i * dV, dV);
                outRow.Clear();

                // Accumulate weighted values
                for (int j = 0; j < seqLenK; j++)
                {
                    float weight = scores[i * seqLenK + j];
                    if (weight <= 0f)
                    {
                        continue;
                    }

                    var vRow = v.Data.AsSpan(vOffset + j * dV, dV);
                    SimdKernels.ScalarMultiplyAdd(outRow, vRow, weight, outRow);
                }
            }
        }

        private void ApplySoftmax(float[] data, int rows, int cols)
        {
            for (int i = 0; i < rows; i++)
            {
                int rowOffset = i * cols;

                // Find max for numerical stability
                float maxVal = float.NegativeInfinity;
                for (int j = 0; j < cols; j++)
                {
                    float v = data[rowOffset + j];
                    if (v > maxVal)
                    {
                        maxVal = v;
                    }
                }

                // Compute exp and sum
                float sum = 0.0f;
                for (int j = 0; j < cols; j++)
                {
                    int idx = rowOffset + j;
                    float v = data[idx];
                    if (float.IsNegativeInfinity(v))
                    {
                        data[idx] = 0.0f;
                        continue;
                    }

                    float ev = MathF.Exp(v - maxVal);
                    data[idx] = ev;
                    sum += ev;
                }

                // Normalize
                if (sum > 0.0f)
                {
                    float invSum = 1.0f / sum;
                    for (int j = 0; j < cols; j++)
                    {
                        data[rowOffset + j] *= invSum;
                    }
                }
            }
        }

        /// <summary>
        /// Multi-head attention variant
        /// </summary>
        public Tensor<float> MultiHeadAttention(
            Tensor<float> q, Tensor<float> k, Tensor<float> v,
            int numHeads, Tensor<float>? mask = null)
        {
            if (q.Shape.Length != 3 || k.Shape.Length != 3 || v.Shape.Length != 3)
                throw new ArgumentException("Multi-head attention requires 3D tensors");

            int batchSize = q.Shape[0];
            int dModel = q.Shape[2];

            if (k.Shape[0] != batchSize || v.Shape[0] != batchSize)
                throw new ArgumentException("Q, K, and V must have the same batch size");

            if (dModel % numHeads != 0)
                throw new ArgumentException("d_model must be divisible by num_heads");

            int dK = dModel / numHeads;

            if (k.Shape[2] != dModel || v.Shape[2] != dModel)
                throw new ArgumentException("Q, K, and V must have the same feature dimension (d_model)");

            if (v.Shape[1] != k.Shape[1])
                throw new ArgumentException("K and V must have the same sequence length");

            // Reshape to [batch * num_heads, seq_len, d_k]
            var qReshaped = ReshapeForMultiHead(q, numHeads, dK);
            var kReshaped = ReshapeForMultiHead(k, numHeads, dK);
            var vReshaped = ReshapeForMultiHead(v, numHeads, dK);

            // Apply attention
            Tensor<float> attended;
            if (mask is null)
            {
                attended = ExecuteInternal(qReshaped, kReshaped, vReshaped, mask: null, maskBatchModulo: 0);
            }
            else
            {
                int expectedPerHeadBatch = batchSize * numHeads;
                if (mask.Shape.Length != 3)
                    throw new ArgumentException("Multi-head attention mask must be a 3D tensor");

                if (mask.Shape[1] != q.Shape[1] || mask.Shape[2] != k.Shape[1])
                    throw new ArgumentException("Multi-head attention mask must match [batch, seq_len_q, seq_len_k]");

                // Accept either per-batch mask [B, SQ, SK] (broadcast across heads) or per-head mask [B*H, SQ, SK].
                int maskBatchModulo = mask.Shape[0] switch
                {
                    int b when b == expectedPerHeadBatch => 0,
                    int b when b == batchSize => batchSize,
                    _ => throw new ArgumentException("Multi-head attention mask must have batch dimension B or B*numHeads"),
                };

                attended = ExecuteInternal(qReshaped, kReshaped, vReshaped, mask, maskBatchModulo);
            }

            // Reshape back to [batch, seq_len, d_model]
            return ReshapeFromMultiHead(attended, batchSize, q.Shape[1], dModel);
        }

        private Tensor<float> ReshapeForMultiHead(Tensor<float> input, int numHeads, int dK)
        {
            int batchSize = input.Shape[0];
            int seqLen = input.Shape[1];
            var reshaped = new Tensor<float>(new[] { batchSize * numHeads, seqLen, dK });

            for (int b = 0; b < batchSize; b++)
            {
                for (int h = 0; h < numHeads; h++)
                {
                    for (int s = 0; s < seqLen; s++)
                    {
                        for (int d = 0; d < dK; d++)
                        {
                            int srcIdx = b * seqLen * numHeads * dK + s * numHeads * dK + h * dK + d;
                            int dstIdx = (b * numHeads + h) * seqLen * dK + s * dK + d;
                            reshaped.Data[dstIdx] = input.Data[srcIdx];
                        }
                    }
                }
            }

            return reshaped;
        }

        private Tensor<float> ReshapeFromMultiHead(Tensor<float> input, int batchSize, int seqLen, int dModel)
        {
            var reshaped = new Tensor<float>(new[] { batchSize, seqLen, dModel });
            int numHeads = input.Shape[0] / batchSize;
            int dK = input.Shape[2];

            for (int b = 0; b < batchSize; b++)
            {
                for (int h = 0; h < numHeads; h++)
                {
                    for (int s = 0; s < seqLen; s++)
                    {
                        for (int d = 0; d < dK; d++)
                        {
                            int srcIdx = (b * numHeads + h) * seqLen * dK + s * dK + d;
                            int dstIdx = b * seqLen * dModel + s * dModel + h * dK + d;
                            reshaped.Data[dstIdx] = input.Data[srcIdx];
                        }
                    }
                }
            }

            return reshaped;
        }
    }
}
