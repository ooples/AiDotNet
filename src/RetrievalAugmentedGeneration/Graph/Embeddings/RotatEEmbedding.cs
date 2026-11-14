using System;
using AiDotNet.Attributes;
using AiDotNet.Enums;

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
[ComponentType(ComponentType.DocumentStore)]
[PipelineStage(PipelineStage.Indexing)]
public class RotatEEmbedding<T> : KGEmbeddingBase<T>
{
    /// <inheritdoc />
    public override bool IsDistanceBased => true;

    // Entity embeddings use paired [real, imag] layout: entityEmbeddings[i] has length 2*dim
    // Relation embeddings store phase angles: relationEmbeddings[i] has length dim
    // where r_real[d] = cos(phase[d]), r_imag[d] = sin(phase[d])
    //
    // Sun et al. 2019: d_r(h, t) = ||h ∘ r - t||_1 = Σ_k |h_k r_k - t_k| (complex modulus per dimension),
    // trained with L = -log σ(γ - d_r(h, t)) - log σ(d_r(h', t') - γ) (Eq. 4, uniform negative weights).

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
        => Engine.TensorSum(Residual(headIdx, relationIdx, tailIdx).Modulus);

    private protected override double ComputeLossAndUpdateGradients(
        int posHead, int relation, int posTail,
        int negHead, int negTail,
        double learningRate, KGEmbeddingOptions options)
    {
        double gamma = options.GetEffectiveMargin();
        int dim = EmbeddingDimension;

        var pos = Residual(posHead, relation, posTail);
        var neg = Residual(negHead, relation, negTail);
        double posDist = NumOps.ToDouble(Engine.TensorSum(pos.Modulus));
        double negDist = NumOps.ToDouble(Engine.TensorSum(neg.Modulus));

        // -log σ(x) = softplus(-x)
        double loss = Softplus(posDist - gamma) + Softplus(gamma - negDist);

        // dL/dd_pos = σ(d_pos - γ); dL/dd_neg = -σ(γ - d_neg).
        T posScale = NumOps.FromDouble(learningRate * Sigmoid(posDist - gamma));
        T negScale = NumOps.FromDouble(-learningRate * Sigmoid(gamma - negDist));

        // Both gradients use the pre-update embeddings; the shared phase gets their sum.
        var posGrad = DistanceGradient(pos);
        var negGrad = DistanceGradient(neg);

        ApplyEntityStep(_entityEmbeddings[posHead], _entityEmbeddings[posTail], posGrad, posScale, dim);
        ApplyEntityStep(_entityEmbeddings[negHead], _entityEmbeddings[negTail], negGrad, negScale, dim);

        var phase = Tensor<T>.FromVector(Row(_relationEmbeddings[relation]));
        var phaseStep = Engine.TensorAdd(
            Engine.TensorMultiplyScalar(posGrad.Phase, posScale),
            Engine.TensorMultiplyScalar(negGrad.Phase, negScale));
        Store(WrapPhase(Engine.TensorSubtract(phase, phaseStep)).ToVector(), _relationEmbeddings[relation]);

        return loss;
    }

    private RotationResidual Residual(int headIdx, int relationIdx, int tailIdx)
    {
        int dim = EmbeddingDimension;
        var head = _entityEmbeddings[headIdx];
        var tail = _entityEmbeddings[tailIdx];
        var phase = Tensor<T>.FromVector(Row(_relationEmbeddings[relationIdx]));
        var cos = Engine.TensorCos(phase);
        var sin = Engine.TensorSin(phase);

        var hReal = Part(head, 0, dim);
        var hImag = Part(head, dim, dim);

        // (hReal + i hImag)(cos + i sin)
        var rotatedReal = Engine.TensorSubtract(Engine.TensorMultiply(hReal, cos), Engine.TensorMultiply(hImag, sin));
        var rotatedImag = Engine.TensorAdd(Engine.TensorMultiply(hReal, sin), Engine.TensorMultiply(hImag, cos));
        var diffReal = Engine.TensorSubtract(rotatedReal, Part(tail, 0, dim));
        var diffImag = Engine.TensorSubtract(rotatedImag, Part(tail, dim, dim));
        var modulus = Engine.TensorSqrt(
            Engine.TensorAdd(Engine.TensorSquare(diffReal), Engine.TensorSquare(diffImag)));

        return new RotationResidual(rotatedReal, rotatedImag, diffReal, diffImag, cos, sin, modulus);
    }

    /// <summary>
    /// Gradient of d = Σ_k |z_k| with z = h ∘ r - t. With u = z / |z| (0 where |z| = 0):
    /// dd/dh = conj(r) u, dd/dt = -u, dd/dθ = Im(conj(h ∘ r) u).
    /// </summary>
    private static DistanceGradients DistanceGradient(RotationResidual r)
    {
        var positive = Engine.TensorGreaterThan(r.Modulus, NumOps.Zero);
        var safeModulus = Engine.TensorWhere(positive, r.Modulus, Tensor<T>.CreateDefault(r.Modulus._shape, NumOps.One));
        var unitReal = Engine.TensorDivide(r.DiffReal, safeModulus);
        var unitImag = Engine.TensorDivide(r.DiffImag, safeModulus);

        var headReal = Engine.TensorAdd(Engine.TensorMultiply(unitReal, r.Cos), Engine.TensorMultiply(unitImag, r.Sin));
        var headImag = Engine.TensorSubtract(Engine.TensorMultiply(unitImag, r.Cos), Engine.TensorMultiply(unitReal, r.Sin));
        var phase = Engine.TensorSubtract(
            Engine.TensorMultiply(unitImag, r.RotatedReal), Engine.TensorMultiply(unitReal, r.RotatedImag));

        return new DistanceGradients(headReal, headImag, unitReal, unitImag, phase);
    }

    /// <summary>h -= scale * dd/dh; t -= scale * dd/dt, i.e. t += scale * u.</summary>
    private static void ApplyEntityStep(T[] head, T[] tail, DistanceGradients grad, T scale, int dim)
    {
        SubtractFrom(head, Engine.TensorMultiplyScalar(grad.HeadReal, scale).ToVector(), 0);
        SubtractFrom(head, Engine.TensorMultiplyScalar(grad.HeadImag, scale).ToVector(), dim);
        AddTo(tail, Engine.TensorMultiplyScalar(grad.UnitReal, scale).ToVector(), 0);
        AddTo(tail, Engine.TensorMultiplyScalar(grad.UnitImag, scale).ToVector(), dim);
    }

    /// <summary>Wraps phases to [-π, π).</summary>
    private static Tensor<T> WrapPhase(Tensor<T> phase)
    {
        var pi = NumOps.FromDouble(Math.PI);
        var period = Tensor<T>.CreateDefault(phase._shape, NumOps.FromDouble(2.0 * Math.PI));
        return Engine.TensorSubtractScalar(Engine.TensorRemainder(Engine.TensorAddScalar(phase, pi), period), pi);
    }

    private static Tensor<T> Part(T[] row, int offset, int length) => Tensor<T>.FromVector(Row(row, offset, length));

    private static double Sigmoid(double x) => x >= 0 ? 1.0 / (1.0 + Math.Exp(-x)) : Math.Exp(x) / (1.0 + Math.Exp(x));

    private static double Softplus(double x) => Math.Max(x, 0.0) + Math.Log(1.0 + Math.Exp(-Math.Abs(x)));

    private sealed class RotationResidual
    {
        public RotationResidual(Tensor<T> rotatedReal, Tensor<T> rotatedImag, Tensor<T> diffReal, Tensor<T> diffImag,
            Tensor<T> cos, Tensor<T> sin, Tensor<T> modulus)
        {
            RotatedReal = rotatedReal;
            RotatedImag = rotatedImag;
            DiffReal = diffReal;
            DiffImag = diffImag;
            Cos = cos;
            Sin = sin;
            Modulus = modulus;
        }

        public Tensor<T> RotatedReal { get; }
        public Tensor<T> RotatedImag { get; }
        public Tensor<T> DiffReal { get; }
        public Tensor<T> DiffImag { get; }
        public Tensor<T> Cos { get; }
        public Tensor<T> Sin { get; }

        /// <summary>|h_k r_k - t_k| per dimension.</summary>
        public Tensor<T> Modulus { get; }
    }

    private sealed class DistanceGradients
    {
        public DistanceGradients(Tensor<T> headReal, Tensor<T> headImag, Tensor<T> unitReal, Tensor<T> unitImag,
            Tensor<T> phase)
        {
            HeadReal = headReal;
            HeadImag = headImag;
            UnitReal = unitReal;
            UnitImag = unitImag;
            Phase = phase;
        }

        public Tensor<T> HeadReal { get; }
        public Tensor<T> HeadImag { get; }
        public Tensor<T> UnitReal { get; }
        public Tensor<T> UnitImag { get; }
        public Tensor<T> Phase { get; }
    }
}
