using System;
using AiDotNet.LossFunctions;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;
using Xunit.Abstractions;

namespace AiDotNet.Tests.IntegrationTests.Optimizers;

/// <summary>
/// Loss-normalization consistency probe for issue #1380's residual.
///
/// <para>
/// The existing 8-arm diagnostic
/// (<see cref="BuildAsyncResidualModeCollapseTests"/>) Arm 6 compares the
/// analytic gradient from <c>NeuralNetworkBase.ComputeGradients</c> (which
/// internally invokes <c>LossFunctionBase{T}.ComputeTapeLoss</c>) against
/// a numeric finite-difference gradient computed from the test-helper
/// <c>ScalarLoss</c> (which calls <c>ILossFunction{T}.CalculateLoss(Vector, Vector)</c>
/// and divides by <c>totalTargetElements</c>).
/// </para>
///
/// <para>
/// On a <c>[B, V]</c> rank-2 target (e.g. SequenceClassification one-hot
/// after the Transformer's <c>SequenceTokenSliceLayer</c> collapses the
/// sequence axis), <see cref="CategoricalCrossEntropyLoss{T}.ComputeTapeLoss"/>
/// reduces with <c>ReduceSum(over class axis)</c> then
/// <c>ReduceMean(over batch axis)</c> — divisor <c>B</c>.
/// </para>
///
/// <para>
/// But <c>ScalarLoss</c> in the test file divides
/// <c>CalculateLoss(Vector, Vector)</c>'s raw sum by
/// <c>totalTargetElements = B*V</c>. So <c>ScalarLoss</c> reports a value
/// that is <c>1/V</c> of what <c>ComputeTapeLoss</c> reports, which means
/// the finite-difference gradient comes out <c>V</c> times smaller than
/// the analytic gradient and Arm 6's assertion ALWAYS fails on this
/// fixture (V=16 → analytic ≈ 16× numeric; observed ratio ~23.6×
/// including FP error from the eps=1e-3 central difference).
/// </para>
///
/// <para>
/// This is a real bug — but the BUG is in the test helper, not the
/// production loss function. <c>ComputeTapeLoss</c>'s mean-over-batch
/// (sum-over-classes) normalization is the canonical PyTorch
/// <c>nn.CrossEntropyLoss(reduction='mean')</c> convention and is what
/// the loss function intends.
/// </para>
///
/// <para>
/// This test pins the production behavior so the test-helper fix in the
/// same PR can be verified against a stable reference: for a
/// <c>[B, V]</c> CategoricalCrossEntropyLoss, <c>ComputeTapeLoss</c>
/// must equal <c>CalculateLoss(flat, flat) / B</c> (not / B*V) — the
/// exact relationship the test helper was getting wrong.
/// </para>
/// </summary>
public class LossNormalizationConsistencyIssue1380Tests
{
    private readonly ITestOutputHelper _output;

    public LossNormalizationConsistencyIssue1380Tests(ITestOutputHelper output)
    {
        _output = output;
    }

    [Fact]
    public void CategoricalCrossEntropyLoss_ComputeTapeLoss_DividesByBatchOnly_OnRank2Target()
    {
        const int B = 4;
        const int V = 8;

        // Build deterministic prediction and target. Prediction is a
        // softmax-like probability distribution (each row sums to ~1);
        // target is one-hot.
        var predicted = new Tensor<float>([B, V]);
        var target = new Tensor<float>([B, V]);

        for (int b = 0; b < B; b++)
        {
            // Prediction: simple ramp normalised so row sums to 1.
            float rowSum = 0;
            for (int v = 0; v < V; v++)
            {
                predicted[b, v] = 0.1f + (b * 0.05f) + (v * 0.02f);
                rowSum += predicted[b, v];
            }
            for (int v = 0; v < V; v++)
            {
                predicted[b, v] /= rowSum;
            }

            // Target: one-hot at class = (b * 3) mod V (arbitrary).
            int trueClass = (b * 3) % V;
            target[b, trueClass] = 1.0f;
        }

        var loss = new CategoricalCrossEntropyLoss<float>();

        // Compute via ComputeTapeLoss — the production path used by
        // NeuralNetworkBase.ComputeGradients.
        var lossTensor = loss.ComputeTapeLoss(predicted, target);
        float computeTapeLossValue = lossTensor[0];

        // Compute via the test-helper formula (raw sum from CalculateLoss
        // divided by totalTargetElements).
        var predFlat = predicted.ToVector();
        var tgtFlat = target.ToVector();
        float rawSum = loss.CalculateLoss(predFlat, tgtFlat);
        int totalTargetElements = B * V;
        float scalarLossValueOverAllElements = rawSum / totalTargetElements;

        // The CORRECT comparator: raw sum divided by batch size only —
        // this matches ComputeTapeLoss's actual reduction
        // (ReduceSum over classes, then ReduceMean over batch).
        float scalarLossValueOverBatchOnly = rawSum / B;

        _output.WriteLine($"B={B}, V={V}");
        _output.WriteLine($"ComputeTapeLoss          = {computeTapeLossValue:F6}");
        _output.WriteLine($"rawSum / (B*V) = {totalTargetElements}-divided = {scalarLossValueOverAllElements:F6}  (test helper's denominator — WRONG by 1/V)");
        _output.WriteLine($"rawSum / B = {B}-divided = {scalarLossValueOverBatchOnly:F6}  (matches ComputeTapeLoss)");
        _output.WriteLine($"Ratio ComputeTape / OverAllElements = {computeTapeLossValue / scalarLossValueOverAllElements:F4}  (should be V = {V})");
        _output.WriteLine($"Ratio ComputeTape / OverBatchOnly   = {computeTapeLossValue / scalarLossValueOverBatchOnly:F4}  (should be 1.0)");

        // Production behavior pin: ComputeTapeLoss divides by B only.
        // Tolerance accommodates the +1e-7 numerical-stability shift
        // inside ComputeTapeLoss that CalculateLoss's SafeLog handles
        // slightly differently.
        Assert.True(
            Math.Abs(computeTapeLossValue - scalarLossValueOverBatchOnly) < 0.01f,
            $"ComputeTapeLoss ({computeTapeLossValue:F6}) must equal rawSum/B ({scalarLossValueOverBatchOnly:F6}) " +
            "to within numerical-stability tolerance — the mean-over-batch reduction is the PyTorch " +
            "nn.CrossEntropyLoss(reduction='mean') convention this loss is documented to follow.");

        // Bug pin: the test-helper formula (divide by total elements)
        // is OFF BY A FACTOR OF V from the production normalization.
        // When this assertion holds, the existing
        // BuildAsyncResidualModeCollapseTests.ScalarLoss is using the
        // wrong denominator and its Arm 6 finite-difference probe
        // is comparing analytic gradient against numeric gradient
        // that is 1/V too small.
        float ratio = computeTapeLossValue / scalarLossValueOverAllElements;
        Assert.True(
            ratio > V * 0.9f && ratio < V * 1.1f,
            $"Expected ComputeTape / (rawSum/(B*V)) ≈ V (= {V}), got {ratio:F4}. " +
            "The test-helper ScalarLoss denominator hypothesis is invalidated.");
    }
}
