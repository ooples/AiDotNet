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
        private readonly GemmKernel _gemmKernel;

        public string Name => "FusedAttention";
        public string Version => "1.0.0";
        public int Priority => 100;

        public AttentionKernel()
        {
            _gemmKernel = new GemmKernel();
        }

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

            if (q.Shape.Length != 3 || k.Shape.Length != 3 || v.Shape.Length != 3)
                throw new ArgumentException("Attention requires 3D tensors [batch, seq_len, features]");

            int batchSize = q.Shape[0];
            int seqLenQ = q.Shape[1];
            int seqLenK = k.Shape[1];
            int dK = q.Shape[2];
            int dV = v.Shape[2];

            if (k.Shape[2] != dK)
                throw new ArgumentException("Q and K must have same feature dimension");

            if (v.Shape[1] != seqLenK)
                throw new ArgumentException("K and V must have same sequence length");

            var result = new Tensor<float>(new[] { batchSize, seqLenQ, dV });

            // Process each batch in parallel
            Parallel.For(0, batchSize, b =>
            {
                ProcessBatch(q, k, v, mask, result, b, seqLenQ, seqLenK, dK, dV);
            });

            return result;
        }

        private unsafe void ProcessBatch(
            Tensor<float> q, Tensor<float> k, Tensor<float> v,
            Tensor<float>? mask, Tensor<float> result,
            int batchIdx, int seqLenQ, int seqLenK, int dK, int dV)
        {
            float scale = 1.0f / MathF.Sqrt(dK);

            // Extract batch slices
            int qOffset = batchIdx * seqLenQ * dK;
            int kOffset = batchIdx * seqLenK * dK;
            int vOffset = batchIdx * seqLenK * dV;
            int outOffset = batchIdx * seqLenQ * dV;

            // Compute attention scores: QK^T
            var scores = new float[seqLenQ * seqLenK];

            fixed (float* pQ = q.Data, pK = k.Data, pScores = scores)
            {
                for (int i = 0; i < seqLenQ; i++)
                {
                    for (int j = 0; j < seqLenK; j++)
                    {
                        float* qRow = pQ + qOffset + i * dK;
                        float* kRow = pK + kOffset + j * dK;
                        float score = SimdKernels.DotProduct(qRow, kRow, dK) * scale;

                        // Apply mask if provided
                        if (mask != null)
                        {
                            int maskIdx = batchIdx * seqLenQ * seqLenK + i * seqLenK + j;
                            if (mask.Data[maskIdx] == 0.0f)
                            {
                                score = float.NegativeInfinity;
                            }
                        }

                        pScores[i * seqLenK + j] = score;
                    }
                }
            }

            // Apply softmax over each row
            ApplySoftmax(scores, seqLenQ, seqLenK);

            // Compute weighted sum: attention_weights * V
            fixed (float* pScores = scores, pV = v.Data, pOut = result.Data)
            {
                for (int i = 0; i < seqLenQ; i++)
                {
                    float* outRow = pOut + outOffset + i * dV;

                    // Initialize output row to zero
                    for (int j = 0; j < dV; j++)
                    {
                        outRow[j] = 0.0f;
                    }

                    // Accumulate weighted values
                    for (int j = 0; j < seqLenK; j++)
                    {
                        float weight = pScores[i * seqLenK + j];
                        float* vRow = pV + vOffset + j * dV;

                        // outRow += weight * vRow
                        SimdKernels.ScalarMultiplyAdd(outRow, vRow, weight, outRow, dV);
                    }
                }
            }
        }

        private unsafe void ApplySoftmax(float[] data, int rows, int cols)
        {
            fixed (float* pData = data)
            {
                for (int i = 0; i < rows; i++)
                {
                    float* row = pData + i * cols;

                    // Find max for numerical stability
                    float maxVal = float.NegativeInfinity;
                    for (int j = 0; j < cols; j++)
                    {
                        if (row[j] > maxVal)
                            maxVal = row[j];
                    }

                    // Compute exp and sum
                    float sum = 0.0f;
                    for (int j = 0; j < cols; j++)
                    {
                        if (float.IsNegativeInfinity(row[j]))
                        {
                            row[j] = 0.0f;
                        }
                        else
                        {
                            row[j] = MathF.Exp(row[j] - maxVal);
                            sum += row[j];
                        }
                    }

                    // Normalize
                    if (sum > 0.0f)
                    {
                        float invSum = 1.0f / sum;
                        for (int j = 0; j < cols; j++)
                        {
                            row[j] *= invSum;
                        }
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
            if (q.Shape.Length != 3)
                throw new ArgumentException("Multi-head attention requires 3D tensors");

            int batchSize = q.Shape[0];
            int seqLen = q.Shape[1];
            int dModel = q.Shape[2];

            if (dModel % numHeads != 0)
                throw new ArgumentException("d_model must be divisible by num_heads");

            int dK = dModel / numHeads;

            // Reshape to [batch * num_heads, seq_len, d_k]
            var qReshaped = ReshapeForMultiHead(q, numHeads, dK);
            var kReshaped = ReshapeForMultiHead(k, numHeads, dK);
            var vReshaped = ReshapeForMultiHead(v, numHeads, dK);

            // Apply attention
            var attended = mask is not null
                ? Execute(qReshaped, kReshaped, vReshaped, mask)
                : Execute(qReshaped, kReshaped, vReshaped);

            // Reshape back to [batch, seq_len, d_model]
            return ReshapeFromMultiHead(attended, batchSize, seqLen, dModel);
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
