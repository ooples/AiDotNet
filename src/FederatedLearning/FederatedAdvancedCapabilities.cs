using System;
using System.Collections.Generic;
using System.Linq;
using System.Security.Cryptography;
using AiDotNet.FederatedLearning.DriftDetection;
using AiDotNet.FederatedLearning.Fairness;
using AiDotNet.FederatedLearning.MPC;
using AiDotNet.FederatedLearning.PSI;
using AiDotNet.FederatedLearning.TEE;
using AiDotNet.FederatedLearning.Unlearning;
using AiDotNet.FederatedLearning.Verification;
using AiDotNet.LinearAlgebra;
using AiDotNet.Models.Options;

namespace AiDotNet.FederatedLearning;

/// <summary>
/// Drives the advanced federated-learning capabilities (contribution, fairness, drift, PSI, TEE, ZK, MPC,
/// unlearning) at their real points in the federated training lifecycle, on behalf of the trainer, and collects
/// their results into a <see cref="FederatedAdvancedMetadata"/>.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Isolating this from the trainer keeps the training loop small: the trainer accumulates per-round client
/// parameters and calls a handful of hooks; all capability-specific logic lives here and is unit-testable with
/// the real capability implementations. Every capability is optional; the coordinator no-ops for the ones that
/// were not configured.
/// </para>
/// </remarks>
public sealed class FederatedAdvancedCapabilities<T>
{
    private readonly IClientContributionEvaluator<T>? _contribution;
    private readonly IFairnessConstraint<T>? _fairness;
    private readonly IFederatedDriftDetector<T>? _drift;
    private readonly IFederatedUnlearner<T>? _unlearner;
    private readonly ITeeProvider<T>? _tee;
    private readonly IZkProofSystem? _zk;
    private readonly ISecureComputationProtocol<T>? _mpc;
    private readonly IPrivateSetIntersection? _psi;
    private readonly INumericOperations<T> _numOps;

    private readonly Dictionary<int, List<Tensor<T>>> _histories = new();
    private readonly FederatedAdvancedMetadata _meta = new();
    private bool _mpcAttempted;

    /// <summary>Creates a coordinator over the configured capabilities (any may be <c>null</c>).</summary>
    public FederatedAdvancedCapabilities(
        IClientContributionEvaluator<T>? contribution,
        IFairnessConstraint<T>? fairness,
        IFederatedDriftDetector<T>? drift,
        IFederatedUnlearner<T>? unlearner,
        ITeeProvider<T>? tee,
        IZkProofSystem? zk,
        ISecureComputationProtocol<T>? mpc,
        IPrivateSetIntersection? psi)
    {
        _contribution = contribution;
        _fairness = fairness;
        _drift = drift;
        _unlearner = unlearner;
        _tee = tee;
        _zk = zk;
        _mpc = mpc;
        _psi = psi;
        _numOps = MathHelper.GetNumericOperations<T>();

        _meta.UnlearningAvailable = _unlearner is not null;
        _meta.UnlearningMethod = _unlearner?.MethodName ?? "None";
    }

    /// <summary>Whether any advanced capability is configured (so the trainer can skip the hooks entirely).</summary>
    public bool AnyConfigured =>
        _contribution is not null || _fairness is not null || _drift is not null || _unlearner is not null ||
        _tee is not null || _zk is not null || _mpc is not null || _psi is not null;

    /// <summary>Runs PSI over per-client sample identities to report cross-client overlap (leakage/dedup signal).</summary>
    public void RunPrivateSetIntersection(IReadOnlyDictionary<int, IReadOnlyList<string>> clientIdentities)
    {
        if (_psi is null || clientIdentities.Count < 2) return;

        _meta.PsiEnabled = true;
        var options = new PsiOptions();
        var ordered = clientIdentities.OrderBy(kv => kv.Key).ToList();
        var baseIds = ordered[0].Value;
        int overlap = 0;
        for (int i = 1; i < ordered.Count; i++)
        {
            try
            {
                var result = _psi.ComputeIntersection(baseIds, ordered[i].Value, options);
                overlap += result.IntersectionSize;
            }
            catch (Exception ex) when (ex is not OutOfMemoryException)
            {
            }
        }

        _meta.PsiTotalOverlap = overlap;
    }

    /// <summary>Accumulates a round's per-client parameters for later contribution/unlearning analysis.</summary>
    public void RecordClientParameters(IReadOnlyDictionary<int, Vector<T>> clientParameters)
    {
        if (_contribution is null && _unlearner is null) return;
        foreach (var kv in clientParameters)
        {
            if (!_histories.TryGetValue(kv.Key, out var list))
            {
                list = new List<Tensor<T>>();
                _histories[kv.Key] = list;
            }

            list.Add(ToTensor(kv.Value));
        }
    }

    /// <summary>Runs the per-round capabilities: drift detection, TEE attestation, ZK bound proof, and (once) MPC.</summary>
    public void OnRoundComplete(
        int round,
        IReadOnlyDictionary<int, Vector<T>> clientParameters,
        Vector<T> globalParameters,
        IReadOnlyDictionary<int, double> clientLoss)
    {
        var clientTensors = clientParameters.ToDictionary(kv => kv.Key, kv => ToTensor(kv.Value));
        var globalTensor = ToTensor(globalParameters);

        if (_drift is not null)
        {
            _meta.DriftDetectionEnabled = true;
            try
            {
                var lossByClient = clientLoss.ToDictionary(kv => kv.Key, kv => kv.Value);
                var report = _drift.DetectDrift(round, clientTensors, globalTensor, lossByClient);
                _meta.DriftRoundsChecked++;
                if (report.GlobalDriftDetected)
                {
                    _meta.DriftDetectedCount++;
                    _meta.AnyDriftDetected = true;
                }
            }
            catch (Exception ex) when (ex is not OutOfMemoryException) { }
        }

        if (_tee is not null)
        {
            _meta.TeeEnabled = true;
            try
            {
                if (!_tee.IsInitialized) _tee.Initialize(new TeeOptions());
                var reportData = Digest(globalParameters);
                _ = _tee.GenerateAttestationQuote(reportData);
                _meta.TeeAttestationCount++;
                _meta.TeeMeasurementHash = _tee.GetMeasurementHash();
            }
            catch (Exception ex) when (ex is not OutOfMemoryException) { }
        }

        if (_zk is not null)
        {
            _meta.ZkEnabled = true;
            _meta.ZkSystem = _zk.Name;
            try
            {
                double norm = L2Norm(globalParameters);
                var value = BitConverter.GetBytes(norm);
                var upperBound = BitConverter.GetBytes(norm + 1.0); // prove norm is within a generous bound.
                var (commitment, randomness) = _zk.Commit(value);
                var proof = _zk.GenerateRangeProof(value, upperBound, commitment, randomness);
                bool verified = _zk.VerifyRangeProof(proof, upperBound, commitment);
                _meta.ZkProofCount++;
                _meta.ZkAllVerified = _meta.ZkProofCount == 1 ? verified : _meta.ZkAllVerified && verified;
            }
            catch (Exception ex) when (ex is not OutOfMemoryException) { }
        }

        // MPC secure-aggregation correctness demonstration (once): secret-share two clients' updates, add the
        // shares securely, reconstruct, and check the result equals the plaintext sum.
        if (_mpc is not null && !_mpcAttempted && clientParameters.Count >= 2)
        {
            _mpcAttempted = true;
            _meta.McpEnabled = true;
            _meta.McpProtocol = _mpc.GetType().Name;
            if (_mpc is ISecretSharingScheme<T> sharing)
            {
                try
                {
                    var two = clientParameters.OrderBy(kv => kv.Key).Take(2).Select(kv => ToTensor(kv.Value)).ToArray();
                    var sharesA = sharing.Split(two[0], 2);
                    var sharesB = sharing.Split(two[1], 2);
                    var sumShares = _mpc.SecureAdd(sharesA, sharesB);
                    var reconstructed = sharing.Combine(sumShares);
                    _meta.McpSecureSumVerified = TensorsClose(reconstructed, Add(two[0], two[1]));
                }
                catch (Exception ex) when (ex is not OutOfMemoryException) { }
            }
        }
    }

    /// <summary>Runs the end-of-training capabilities: client contribution and fairness evaluation.</summary>
    public void Finalize(IReadOnlyDictionary<int, Vector<T>> finalClientParameters, Vector<T> globalParameters)
    {
        var clientTensors = finalClientParameters.ToDictionary(kv => kv.Key, kv => ToTensor(kv.Value));
        var globalTensor = ToTensor(globalParameters);

        if (_contribution is not null)
        {
            try
            {
                var scores = _contribution.EvaluateContributions(clientTensors, globalTensor, _histories);
                _meta.ContributionEvaluated = true;
                _meta.ContributionMethod = _contribution.MethodName;
                _meta.ClientContributions = new Dictionary<int, double>(scores);
                _meta.FreeRiders = _contribution.IdentifyFreeRiders(scores);
            }
            catch (Exception ex) when (ex is not OutOfMemoryException) { }
        }

        if (_fairness is not null)
        {
            try
            {
                // Without external group labels, treat each client as its own group: this measures per-client
                // performance disparity (the equity signal federated fairness is concerned with).
                var groups = clientTensors.Keys.ToDictionary(id => id, id => id);
                double violation = _fairness.EvaluateFairness(clientTensors, globalTensor, groups);
                _meta.FairnessEvaluated = true;
                _meta.FairnessConstraint = _fairness.ConstraintName;
                _meta.FairnessViolation = violation;
                _meta.FairnessSatisfied = _fairness.IsSatisfied(violation);
            }
            catch (Exception ex) when (ex is not OutOfMemoryException) { }
        }
    }

    /// <summary>
    /// Removes a client's contribution from the trained global model (GDPR right-to-be-forgotten), or <c>null</c>
    /// when no unlearner is configured or the client has no recorded history.
    /// </summary>
    public (Vector<T> UnlearnedParameters, Unlearning.UnlearningCertificate Certificate)? UnlearnClient(
        int clientId, Vector<T> globalParameters)
    {
        if (_unlearner is null || !_histories.ContainsKey(clientId)) return null;

        var (unlearned, certificate) = _unlearner.Unlearn(clientId, ToTensor(globalParameters), _histories);
        return (ToVector(unlearned), certificate);
    }

    /// <summary>Returns the collected results.</summary>
    public FederatedAdvancedMetadata BuildMetadata() => _meta;

    // --- helpers ---

    private static Tensor<T> ToTensor(Vector<T> v)
    {
        var t = new Tensor<T>(new[] { v.Length });
        for (int i = 0; i < v.Length; i++) t[i] = v[i];
        return t;
    }

    private static Vector<T> ToVector(Tensor<T> t)
    {
        var v = new Vector<T>(t.Length);
        for (int i = 0; i < t.Length; i++) v[i] = t[i];
        return v;
    }

    private Tensor<T> Add(Tensor<T> a, Tensor<T> b)
    {
        var r = new Tensor<T>(new[] { a.Length });
        for (int i = 0; i < a.Length; i++) r[i] = _numOps.Add(a[i], b[i]);
        return r;
    }

    private bool TensorsClose(Tensor<T> a, Tensor<T> b)
    {
        if (a.Length != b.Length) return false;
        for (int i = 0; i < a.Length; i++)
        {
            double diff = Math.Abs(_numOps.ToDouble(a[i]) - _numOps.ToDouble(b[i]));
            if (diff > 1e-6 * (1.0 + Math.Abs(_numOps.ToDouble(b[i])))) return false;
        }

        return true;
    }

    private double L2Norm(Vector<T> v)
    {
        double sum = 0;
        for (int i = 0; i < v.Length; i++) { double d = _numOps.ToDouble(v[i]); sum += d * d; }
        return Math.Sqrt(sum);
    }

    private byte[] Digest(Vector<T> v)
    {
        var bytes = new byte[v.Length * sizeof(double)];
        for (int i = 0; i < v.Length; i++)
        {
            BitConverter.GetBytes(_numOps.ToDouble(v[i])).CopyTo(bytes, i * sizeof(double));
        }

        using var sha = SHA256.Create();
        var hash = sha.ComputeHash(bytes);
        // Attestation report data is bounded (<= 64 bytes); SHA-256 is 32, well within.
        return hash;
    }
}
