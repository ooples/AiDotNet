using System;

namespace AiDotNet.RetrievalAugmentedGeneration.Graph.Embeddings;

/// <summary>
/// RotatE embedding model: models relations as rotations in complex vector space.
/// </summary>
/// <typeparam name="T">The numeric type used for embedding calculations.</typeparam>
/// <remarks>
/// <para>
/// RotatE (Sun et al., 2019) represents entities as complex vectors and relations as
/// element-wise rotations: t = h ∘ r, where |r_i| = 1 (unit modulus). This captures
/// symmetric, antisymmetric, inversion, and composition patterns.
/// </para>
/// <para>
/// Complex numbers are represented as paired real/imaginary T[] arrays to keep the generic T
/// type parameter (System.Numerics.Complex is double-only).
/// </para>
/// <para><b>For Beginners:</b> Instead of translation (TransE), RotatE rotates entity vectors.
/// Each relation is a rotation angle per dimension. This handles more relation patterns:
/// - Symmetric: "married_to" (A→B and B→A) uses 180° rotation
/// - Antisymmetric: "parent_of" (if A→B, then NOT B→A)
/// - Composition: "grandparent_of" = "parent_of" + "parent_of" (rotations compose)
/// </para>
/// </remarks>
public class RotatEEmbedding<T> : KGEmbeddingBase<T>
{
    /// <inheritdoc />
    public override bool IsDistanceBased => true;

    // Entity embeddings use paired [real, imag] layout: entityEmbeddings[i] has length 2*dim
    // Relation embeddings store phase angles: relationEmbeddings[i] has length dim
    // where r_real[d] = cos(phase[d]), r_imag[d] = sin(phase[d])

    private protected override int GetEntityEmbeddingSize() => EmbeddingDimension * 2;

    private protected override int GetRelationEmbeddingSize() => EmbeddingDimension;

    private protected override void OnInitialize(KGEmbeddingOptions options, Random rng, KnowledgeGraph<T> graph)
    {
        // Entity embeddings are already allocated at 2*dim by base via GetEntityEmbeddingSize().
        // Relation embeddings need reinitialization: phase angles in [−π, π] instead of base's uniform scale.
        int dim = options.GetEffectiveEmbeddingDimension();
        for (int i = 0; i < _relationEmbeddings.Length; i++)
        {
            for (int d = 0; d < dim; d++)
            {
                _relationEmbeddings[i][d] = NumOps.FromDouble((rng.NextDouble() * 2.0 - 1.0) * Math.PI);
            }
        }
    }

    private protected override T ScoreTripleInternal(int headIdx, int relationIdx, int tailIdx)
    {
        var h = _entityEmbeddings[headIdx];
        var rPhase = _relationEmbeddings[relationIdx];
        var t = _entityEmbeddings[tailIdx];
        int dim = EmbeddingDimension;

        // Score: ||h ∘ r - t||  where ∘ is complex multiplication with unit-modulus r
        T sumSq = NumOps.Zero;
        for (int d = 0; d < dim; d++)
        {
            double phase = NumOps.ToDouble(rPhase[d]);
            double rReal = Math.Cos(phase);
            double rImag = Math.Sin(phase);

            double hReal = NumOps.ToDouble(h[d]);
            double hImag = NumOps.ToDouble(h[d + dim]);

            // Complex multiply: (hReal + i*hImag) * (rReal + i*rImag)
            double hrReal = hReal * rReal - hImag * rImag;
            double hrImag = hReal * rImag + hImag * rReal;

            double tReal = NumOps.ToDouble(t[d]);
            double tImag = NumOps.ToDouble(t[d + dim]);

            double diffReal = hrReal - tReal;
            double diffImag = hrImag - tImag;

            sumSq = NumOps.Add(sumSq, NumOps.FromDouble(diffReal * diffReal + diffImag * diffImag));
        }

        return NumOps.Sqrt(sumSq);
    }

    private protected override double ComputeLossAndUpdateGradients(
        int posHead, int relation, int posTail,
        int negHead, int negTail,
        double learningRate, KGEmbeddingOptions options)
    {
        double margin = options.GetEffectiveMargin();
        int dim = EmbeddingDimension;

        double posDist = ComputeDistanceSq(posHead, relation, posTail, dim);
        double negDist = ComputeDistanceSq(negHead, relation, negTail, dim);

        double loss = Math.Max(0.0, margin + posDist - negDist);
        if (loss <= 0.0) return 0.0;

        // Gradient updates for positive triple
        UpdateComplexGradients(posHead, relation, posTail, dim, learningRate, isPositive: true);
        // Gradient updates for negative triple
        UpdateComplexGradients(negHead, relation, negTail, dim, learningRate, isPositive: false);

        return loss;
    }

    private double ComputeDistanceSq(int headIdx, int relationIdx, int tailIdx, int dim)
    {
        var h = _entityEmbeddings[headIdx];
        var rPhase = _relationEmbeddings[relationIdx];
        var t = _entityEmbeddings[tailIdx];

        double sumSq = 0.0;
        for (int d = 0; d < dim; d++)
        {
            double phase = NumOps.ToDouble(rPhase[d]);
            double rReal = Math.Cos(phase);
            double rImag = Math.Sin(phase);

            double hReal = NumOps.ToDouble(h[d]);
            double hImag = NumOps.ToDouble(h[d + dim]);

            double hrReal = hReal * rReal - hImag * rImag;
            double hrImag = hReal * rImag + hImag * rReal;

            double diffReal = hrReal - NumOps.ToDouble(t[d]);
            double diffImag = hrImag - NumOps.ToDouble(t[d + dim]);

            sumSq += diffReal * diffReal + diffImag * diffImag;
        }

        return sumSq;
    }

    private void UpdateComplexGradients(int headIdx, int relationIdx, int tailIdx, int dim, double lr, bool isPositive)
    {
        var h = _entityEmbeddings[headIdx];
        var rPhase = _relationEmbeddings[relationIdx];
        var t = _entityEmbeddings[tailIdx];

        double sign = isPositive ? 1.0 : -1.0;
        double step = 2.0 * lr * sign;

        for (int d = 0; d < dim; d++)
        {
            double phase = NumOps.ToDouble(rPhase[d]);
            double rReal = Math.Cos(phase);
            double rImag = Math.Sin(phase);

            double hReal = NumOps.ToDouble(h[d]);
            double hImag = NumOps.ToDouble(h[d + dim]);

            double hrReal = hReal * rReal - hImag * rImag;
            double hrImag = hReal * rImag + hImag * rReal;

            double tReal = NumOps.ToDouble(t[d]);
            double tImag = NumOps.ToDouble(t[d + dim]);

            double diffReal = hrReal - tReal;
            double diffImag = hrImag - tImag;

            // Gradient w.r.t. h (chain through complex multiply)
            double gradHReal = step * (diffReal * rReal + diffImag * rImag);
            double gradHImag = step * (-diffReal * rImag + diffImag * rReal);

            h[d] = NumOps.FromDouble(hReal - gradHReal);
            h[d + dim] = NumOps.FromDouble(hImag - gradHImag);

            // Gradient w.r.t. t
            t[d] = NumOps.FromDouble(tReal + step * diffReal);
            t[d + dim] = NumOps.FromDouble(tImag + step * diffImag);

            // Gradient w.r.t. phase θ: dL/dθ = 2·(diffReal·d(hrRe)/dθ + diffImag·d(hrIm)/dθ)
            // where d(hrRe)/dθ = -(hRe·sinθ + hIm·cosθ) and d(hrIm)/dθ = hRe·cosθ - hIm·sinθ
            double gradPhase = step * (diffReal * (-hReal * rImag - hImag * rReal) +
                                        diffImag * (hReal * rReal - hImag * rImag));
            rPhase[d] = NumOps.FromDouble(phase - gradPhase);
        }
    }
}
