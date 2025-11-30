using AiDotNet.Autodiff.Testing;
using Xunit;

namespace AiDotNet.Tests.UnitTests.Autodiff;

/// <summary>
/// Tests for TensorOperationsVerification to ensure autodiff gradients match numerical gradients.
/// </summary>
/// <remarks>
/// <para>
/// These tests verify that the gradient implementations in TensorOperations produce
/// results that match numerical differentiation (finite differences).
/// </para>
/// <para><b>For Beginners:</b> These tests ensure our automatic differentiation is correct.
///
/// Each test:
/// 1. Runs an operation (like ReLU) using autodiff
/// 2. Computes gradients numerically (slow but always correct)
/// 3. Compares them - they should match!
///
/// If a test fails, it means our gradient implementation has a bug.
/// </para>
/// </remarks>
public class TensorOperationsVerificationTests
{
    #region Float Tests

    [Fact]
    public void VerifyReLU_Float_Passes()
    {
        var verifier = new TensorOperationsVerification<float>();
        var result = verifier.VerifyReLU();

        Assert.True(result.Passed, $"ReLU gradient verification failed: {result}");
    }

    [Fact]
    public void VerifySigmoid_Float_Passes()
    {
        var verifier = new TensorOperationsVerification<float>();
        var result = verifier.VerifySigmoid();

        Assert.True(result.Passed, $"Sigmoid gradient verification failed: {result}");
    }

    [Fact]
    public void VerifyTanh_Float_Passes()
    {
        var verifier = new TensorOperationsVerification<float>();
        var result = verifier.VerifyTanh();

        Assert.True(result.Passed, $"Tanh gradient verification failed: {result}");
    }

    [Fact]
    public void VerifyNegate_Float_Passes()
    {
        var verifier = new TensorOperationsVerification<float>();
        var result = verifier.VerifyNegate();

        Assert.True(result.Passed, $"Negate gradient verification failed: {result}");
    }

    [Fact]
    public void VerifyExp_Float_Passes()
    {
        var verifier = new TensorOperationsVerification<float>();
        var result = verifier.VerifyExp();

        Assert.True(result.Passed, $"Exp gradient verification failed: {result}");
    }

    [Fact]
    public void VerifyLog_Float_Passes()
    {
        var verifier = new TensorOperationsVerification<float>();
        var result = verifier.VerifyLog();

        Assert.True(result.Passed, $"Log gradient verification failed: {result}");
    }

    [Fact]
    public void VerifySqrt_Float_Passes()
    {
        var verifier = new TensorOperationsVerification<float>();
        var result = verifier.VerifySqrt();

        Assert.True(result.Passed, $"Sqrt gradient verification failed: {result}");
    }

    [Fact]
    public void VerifySquare_Float_Passes()
    {
        var verifier = new TensorOperationsVerification<float>();
        var result = verifier.VerifySquare();

        Assert.True(result.Passed, $"Square gradient verification failed: {result}");
    }

    [Fact]
    public void VerifyLeakyReLU_Float_Passes()
    {
        var verifier = new TensorOperationsVerification<float>();
        var result = verifier.VerifyLeakyReLU();

        Assert.True(result.Passed, $"LeakyReLU gradient verification failed: {result}");
    }

    [Fact]
    public void VerifyAdd_Float_BothInputs_Pass()
    {
        var verifier = new TensorOperationsVerification<float>();
        var (result1, result2) = verifier.VerifyAdd();

        Assert.True(result1.Passed, $"Add gradient (input1) verification failed: {result1}");
        Assert.True(result2.Passed, $"Add gradient (input2) verification failed: {result2}");
    }

    [Fact]
    public void VerifySubtract_Float_BothInputs_Pass()
    {
        var verifier = new TensorOperationsVerification<float>();
        var (result1, result2) = verifier.VerifySubtract();

        Assert.True(result1.Passed, $"Subtract gradient (input1) verification failed: {result1}");
        Assert.True(result2.Passed, $"Subtract gradient (input2) verification failed: {result2}");
    }

    [Fact]
    public void VerifyElementwiseMultiply_Float_BothInputs_Pass()
    {
        var verifier = new TensorOperationsVerification<float>();
        var (result1, result2) = verifier.VerifyElementwiseMultiply();

        Assert.True(result1.Passed, $"Multiply gradient (input1) verification failed: {result1}");
        Assert.True(result2.Passed, $"Multiply gradient (input2) verification failed: {result2}");
    }

    [Fact]
    public void VerifyElementwiseDivide_Float_BothInputs_Pass()
    {
        var verifier = new TensorOperationsVerification<float>();
        var (result1, result2) = verifier.VerifyElementwiseDivide();

        Assert.True(result1.Passed, $"Divide gradient (input1) verification failed: {result1}");
        Assert.True(result2.Passed, $"Divide gradient (input2) verification failed: {result2}");
    }

    [Fact]
    public void VerifyAllOperations_Float_AllPass()
    {
        var verifier = new TensorOperationsVerification<float>();
        var summary = verifier.VerifyAllOperations();

        Assert.True(summary.AllPassed, $"Some operations failed:\n{summary}");
    }

    #endregion

    #region Double Tests

    [Fact]
    public void VerifyReLU_Double_Passes()
    {
        var verifier = new TensorOperationsVerification<double>();
        var result = verifier.VerifyReLU();

        Assert.True(result.Passed, $"ReLU gradient verification failed: {result}");
    }

    [Fact]
    public void VerifySigmoid_Double_Passes()
    {
        var verifier = new TensorOperationsVerification<double>();
        var result = verifier.VerifySigmoid();

        Assert.True(result.Passed, $"Sigmoid gradient verification failed: {result}");
    }

    [Fact]
    public void VerifyTanh_Double_Passes()
    {
        var verifier = new TensorOperationsVerification<double>();
        var result = verifier.VerifyTanh();

        Assert.True(result.Passed, $"Tanh gradient verification failed: {result}");
    }

    [Fact]
    public void VerifyAllOperations_Double_AllPass()
    {
        var verifier = new TensorOperationsVerification<double>();
        var summary = verifier.VerifyAllOperations();

        Assert.True(summary.AllPassed, $"Some operations failed:\n{summary}");
    }

    #endregion

    #region Configuration Tests

    [Fact]
    public void CustomConfiguration_UsesCorrectTolerances()
    {
        var config = new TensorOperationsVerification<float>.VerificationConfig
        {
            Epsilon = 1e-4,
            RelativeTolerance = 1e-3,
            AbsoluteTolerance = 1e-5,
            RandomSeed = 123
        };

        var verifier = new TensorOperationsVerification<float>(config);
        var result = verifier.VerifyReLU();

        // With looser tolerances, should still pass
        Assert.True(result.Passed, $"ReLU with custom config failed: {result}");
    }

    [Theory]
    [InlineData(new int[] { 5 })]
    [InlineData(new int[] { 2, 3 })]
    [InlineData(new int[] { 2, 2, 2 })]
    public void VerifyReLU_DifferentShapes_AllPass(int[] shape)
    {
        var verifier = new TensorOperationsVerification<float>();
        var result = verifier.VerifyReLU(shape);

        Assert.True(result.Passed, $"ReLU with shape [{string.Join(", ", shape)}] failed: {result}");
    }

    #endregion

    #region NumericalGradient Utility Tests

    [Fact]
    public void NumericalGradient_ComputeForScalarFunction_CorrectForSquare()
    {
        // f(x) = sum(x^2), df/dx = 2x
        var input = new AiDotNet.Tensors.LinearAlgebra.Tensor<float>(new[] { 3 });
        input[0] = 1.0f;
        input[1] = 2.0f;
        input[2] = 3.0f;

        var gradient = NumericalGradient<float>.ComputeForScalarFunction(
            input,
            x =>
            {
                float sum = 0;
                for (int i = 0; i < x.Length; i++)
                    sum += x[i] * x[i];
                return sum;
            });

        // Expected gradients: 2*1=2, 2*2=4, 2*3=6
        Assert.True(Math.Abs(gradient[0] - 2.0f) < 1e-3f, $"Expected 2.0, got {gradient[0]}");
        Assert.True(Math.Abs(gradient[1] - 4.0f) < 1e-3f, $"Expected 4.0, got {gradient[1]}");
        Assert.True(Math.Abs(gradient[2] - 6.0f) < 1e-3f, $"Expected 6.0, got {gradient[2]}");
    }

    [Fact]
    public void NumericalGradient_Compare_IdenticalTensors_Passes()
    {
        var tensor1 = new AiDotNet.Tensors.LinearAlgebra.Tensor<float>(new[] { 3 });
        var tensor2 = new AiDotNet.Tensors.LinearAlgebra.Tensor<float>(new[] { 3 });

        tensor1[0] = 1.0f; tensor2[0] = 1.0f;
        tensor1[1] = 2.0f; tensor2[1] = 2.0f;
        tensor1[2] = 3.0f; tensor2[2] = 3.0f;

        var result = NumericalGradient<float>.Compare(tensor1, tensor2);

        Assert.True(result.Passed);
        Assert.Equal(0.0, result.MaxRelativeError);
    }

    [Fact]
    public void NumericalGradient_Compare_DifferentTensors_FailsWithDetails()
    {
        var expected = new AiDotNet.Tensors.LinearAlgebra.Tensor<float>(new[] { 3 });
        var actual = new AiDotNet.Tensors.LinearAlgebra.Tensor<float>(new[] { 3 });

        expected[0] = 1.0f; actual[0] = 1.0f;
        expected[1] = 2.0f; actual[1] = 3.0f;  // Different!
        expected[2] = 3.0f; actual[2] = 3.0f;

        var result = NumericalGradient<float>.Compare(expected, actual, relativeTolerance: 1e-5);

        Assert.False(result.Passed);
        Assert.True(result.FailedElements > 0);
        Assert.True(result.Errors.Count > 0);
    }

    [Fact]
    public void NumericalGradient_Compare_ShapeMismatch_Fails()
    {
        var tensor1 = new AiDotNet.Tensors.LinearAlgebra.Tensor<float>(new[] { 3 });
        var tensor2 = new AiDotNet.Tensors.LinearAlgebra.Tensor<float>(new[] { 4 });

        var result = NumericalGradient<float>.Compare(tensor1, tensor2);

        Assert.False(result.Passed);
        Assert.Contains("Shape mismatch", result.Errors[0]);
    }

    #endregion
}
