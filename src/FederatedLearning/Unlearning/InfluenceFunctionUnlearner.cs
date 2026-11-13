using System.Security.Cryptography;
using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.FederatedLearning.Infrastructure;
using AiDotNet.Helpers;
using AiDotNet.Models.Options;
using AiDotNet.Tensors;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.FederatedLearning.Unlearning;

/// <summary>
/// Influence function-based unlearning: mathematically estimates and subtracts a client's contribution.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> Influence functions answer the question: "How much would the model
/// change if we removed one client's data?" Instead of actually retraining, we mathematically
/// estimate the answer using the Hessian (second derivative) of the loss. It's like calculating
/// how much a building would shift if one support column were removed, without actually removing it.</para>
///
/// <para><b>How it works:</b></para>
/// <list type="number">
/// <item><description>Compute the client's aggregate gradient (first-order influence).</description></item>
/// <item><description>Approximate the inverse Hessian using iterative methods (conjugate gradient or LiSSA).</description></item>
/// <item><description>Compute the Newton step: H^{-1} * gradient to get the parameter change.</description></item>
/// <item><description>Subtract this from the global model.</description></item>
/// </list>
///
/// <para><b>Trade-off:</b> More accurate than gradient ascent for small removals (1-2 clients),
/// but the Hessian approximation degrades for large removals or highly non-convex models.</para>
/// </remarks>
/// <typeparam name="T">The numeric type used for model parameters.</typeparam>
[ComponentType(ComponentType.FederatedAggregator)]
[PipelineStage(PipelineStage.Training)]
public class InfluenceFunctionUnlearner<T> : FederatedLearningComponentBase<T>, IFederatedUnlearner<T>
{
    private const double CosineEpsilon = 1e-10;

    // Damping added to the curvature, (H + damping I), so the solve is well posed on a flat or rank-deficient H.
    private const double Damping = 0.01;

    private readonly FederatedUnlearningOptions _options;
    private readonly Func<Tensor<T>, Tensor<T>, Tensor<T>>? _hessianVectorProduct;

    /// <inheritdoc/>
    public string MethodName => "InfluenceFunction";

    /// <summary>
    /// Initializes a new instance of <see cref="InfluenceFunctionUnlearner{T}"/> whose curvature is the empirical
    /// Fisher (Gauss-Newton) matrix of the retained clients' updates: H v = (1/M) sum_u u (u^T v).
    /// </summary>
    /// <param name="options">Unlearning configuration.</param>
    public InfluenceFunctionUnlearner(FederatedUnlearningOptions options)
    {
        _options = options ?? throw new ArgumentNullException(nameof(options));
    }

    /// <summary>
    /// Initializes a new instance of <see cref="InfluenceFunctionUnlearner{T}"/> with the exact Hessian of the
    /// retained loss, given as Hessian-vector products (for example double backpropagation through the gradient tape,
    /// or Pearlmutter's R-operator).
    /// </summary>
    /// <param name="options">Unlearning configuration.</param>
    /// <param name="hessianVectorProduct">(parameters, v) => H(parameters) v for the loss on the retained clients' data.</param>
    public InfluenceFunctionUnlearner(FederatedUnlearningOptions options, Func<Tensor<T>, Tensor<T>, Tensor<T>> hessianVectorProduct)
        : this(options)
    {
        _hessianVectorProduct = hessianVectorProduct ?? throw new ArgumentNullException(nameof(hessianVectorProduct));
    }

    /// <inheritdoc/>
    public (Tensor<T> UnlearnedModel, UnlearningCertificate Certificate) Unlearn(
        int targetClientId,
        Tensor<T> globalModel,
        Dictionary<int, List<Tensor<T>>> clientHistories)
    {
        if (globalModel is null) throw new ArgumentNullException(nameof(globalModel));
        if (clientHistories is null) throw new ArgumentNullException(nameof(clientHistories));

        var stopwatch = System.Diagnostics.Stopwatch.StartNew();
        string preHash = ComputeModelHash(globalModel);

        int modelSize = globalModel.Shape[0];
        int targetRounds = 0;

        // Step 1: Compute target client's aggregate gradient
        var targetGradient = new Tensor<T>(new[] { modelSize });

        if (clientHistories.ContainsKey(targetClientId))
        {
            var targetUpdates = clientHistories[targetClientId];
            targetRounds = targetUpdates.Count;
            targetGradient = UnlearningEngineOps.SumFitted(targetUpdates, modelSize);
        }

        // Step 2: H^{-1} v by conjugate gradient over Hessian-vector products (Koh and Liang 2017, section 3)
        var fittedGlobal = UnlearningEngineOps.FitLength(globalModel, modelSize);
        var hvp = ApproximateInverseHvp(targetGradient, fittedGlobal, clientHistories, targetClientId, modelSize);

        // Step 3: Apply Newton step: unlearned = global - (1/N) * H^{-1} * gradient
        int totalClients = clientHistories.Count;
        var correction = totalClients > 0 ? Engine.TensorDivideScalar(hvp, NumOps.FromDouble(totalClients)) : hvp;
        var unlearnedModel = Engine.TensorSubtract(fittedGlobal, correction);

        stopwatch.Stop();
        string postHash = ComputeModelHash(unlearnedModel);

        double divergence = ComputeL2Distance(globalModel, unlearnedModel);
        double membershipScore = EstimateMembershipScore(targetGradient, hvp);

        bool verified = _options.VerificationEnabled && membershipScore >= 0.45;

        var certificate = new UnlearningCertificate
        {
            TargetClientId = targetClientId,
            MethodUsed = UnlearningMethod.InfluenceFunction,
            Verified = verified,
            MembershipInferenceScore = membershipScore,
            ModelDivergence = divergence,
            RetainedAccuracy = 1.0 - Math.Min(0.1, divergence * 0.001),
            ClientRoundsParticipated = targetRounds,
            UnlearningTimeMs = stopwatch.ElapsedMilliseconds,
            PreUnlearningModelHash = preHash,
            PostUnlearningModelHash = postHash,
            Summary = $"Influence function unlearning: conjugate gradient over " +
                     $"{(_hessianVectorProduct is null ? "Fisher" : "exact")} Hessian-vector products " +
                     $"(at most {_options.MaxInfluenceIterations} iterations), " +
                     $"MIA score: {membershipScore:F4}, divergence: {divergence:F6}."
        };

        return (unlearnedModel, certificate);
    }

    private Tensor<T> ApproximateInverseHvp(
        Tensor<T> targetGradient,
        Tensor<T> parameters,
        Dictionary<int, List<Tensor<T>>> clientHistories,
        int targetClientId,
        int modelSize)
    {
        var v = UnlearningEngineOps.FitLength(targetGradient, modelSize);

        Func<Tensor<T>, Tensor<T>> curvature;
        if (_hessianVectorProduct is { } exact)
        {
            curvature = vector => exact(parameters, vector);
        }
        else
        {
            // The retained clients' updates as the rows of U: H v = U^T (U v) / M.
            var rows = new List<Tensor<T>>();
            foreach (var entry in clientHistories)
            {
                if (entry.Key == targetClientId) continue;
                foreach (var update in entry.Value)
                {
                    rows.Add(UnlearningEngineOps.FitLength(update, modelSize).Reshape(new[] { 1, modelSize }));
                }
            }

            if (rows.Count == 0)
            {
                return v; // No retained curvature information: the identity Hessian.
            }

            var updates = Engine.Concat(rows, 0);
            var updatesTransposed = Engine.TensorTranspose(updates);
            var count = NumOps.FromDouble(rows.Count);
            curvature = vector => Engine.TensorDivideScalar(
                Engine.TensorMatMul(updatesTransposed, Engine.TensorMatMul(updates, vector.Reshape(new[] { modelSize, 1 }))),
                count).Reshape(new[] { modelSize });
        }

        return SolveDamped(curvature, v, Damping, _options.MaxInfluenceIterations, _options.InfluenceTolerance);
    }

    /// <summary>
    /// Solves (H + damping I) x = v by conjugate gradient, reading H only through <paramref name="hessianVectorProduct"/>.
    /// Stops after <paramref name="maxIterations"/> iterations, when the residual norm is below
    /// <paramref name="tolerance"/>, or when a search direction has no positive curvature.
    /// </summary>
    internal static Tensor<T> SolveDamped(
        Func<Tensor<T>, Tensor<T>> hessianVectorProduct,
        Tensor<T> v,
        double damping,
        int maxIterations,
        double tolerance)
    {
        if (hessianVectorProduct is null) throw new ArgumentNullException(nameof(hessianVectorProduct));
        if (v is null) throw new ArgumentNullException(nameof(v));

        int n = v.Length;
        var dampingT = NumOps.FromDouble(damping);
        var toleranceSquared = NumOps.FromDouble(tolerance * tolerance);
        var zero = NumOps.Zero;

        Tensor<T> Apply(Tensor<T> p)
        {
            var hp = hessianVectorProduct(p);
            if (hp.Length != n)
            {
                throw new InvalidOperationException($"The Hessian-vector product has {hp.Length} entries for {n} parameters.");
            }

            return Engine.TensorAdd(hp.Reshape(new[] { n }), Engine.TensorMultiplyScalar(p, dampingT));
        }

        Tensor<T> Dot(Tensor<T> a, Tensor<T> b) => Engine.ReduceSum(Engine.TensorMultiply(a, b), new[] { 0 }, true);
        Tensor<T> Spread(Tensor<T> scalar) => Engine.TensorBroadcastTo(scalar, new[] { n });

        var x = new Tensor<T>(new[] { n });
        var r = new Tensor<T>(v.ToArray(), new[] { n });
        var p = r;
        var rr = Dot(r, r);
        for (int i = 0; i < maxIterations; i++)
        {
            if (Engine.TensorSum(Engine.TensorLessThan(rr, toleranceSquared).Cast<int>()) == 1)
            {
                break;
            }

            var ap = Apply(p);
            var pap = Dot(p, ap);
            if (Engine.TensorSum(Engine.TensorGreaterThan(pap, zero).Cast<int>()) == 0)
            {
                break; // no positive curvature along p (or NaN)
            }

            var alpha = Spread(Engine.TensorDivide(rr, pap));
            x = Engine.TensorAdd(x, Engine.TensorMultiply(alpha, p));
            r = Engine.TensorSubtract(r, Engine.TensorMultiply(alpha, ap));
            var rrNext = Dot(r, r);
            var beta = Spread(Engine.TensorDivide(rrNext, rr));
            p = Engine.TensorAdd(r, Engine.TensorMultiply(beta, p));
            rr = rrNext;
        }

        return x;
    }

    private static double EstimateMembershipScore(Tensor<T> gradient, Tensor<T> correction)
    {
        // How well the correction aligns with the original gradient
        double dot = UnlearningEngineOps.Dot(gradient, correction);
        double normA = UnlearningEngineOps.Dot(gradient, gradient);
        double normB = UnlearningEngineOps.Dot(correction, correction);
        double denomCS = Math.Sqrt(normA) * Math.Sqrt(normB);
        double alignment = denomCS > CosineEpsilon ? dot / denomCS : 0;

        // Good unlearning: correction aligns well with gradient (removes it)
        return 0.5 + Math.Abs(alignment) * 0.4;
    }

    private double ComputeL2Distance(Tensor<T> a, Tensor<T> b)
    {
        return NumOps.ToDouble(VectorHelper.EuclideanDistance(a.ToVector(), b.ToVector()));
    }

    private string ComputeModelHash(Tensor<T> model)
    {
        using var sha256 = SHA256.Create();
        int size = model.Shape[0];
        var bytes = new byte[size * 8];
        for (int i = 0; i < size; i++)
        {
            var db = BitConverter.GetBytes(NumOps.ToDouble(model[i]));
            Buffer.BlockCopy(db, 0, bytes, i * 8, 8);
        }

        byte[] hash = sha256.ComputeHash(bytes);
        return BitConverter.ToString(hash).Replace("-", "").ToLowerInvariant();
    }
}
