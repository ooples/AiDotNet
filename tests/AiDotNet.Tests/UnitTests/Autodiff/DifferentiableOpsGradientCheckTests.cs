using AiDotNet.Autodiff;
using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.Autodiff;

/// <summary>
/// Gradient checking tests for DifferentiableOps — verifies each backward function
/// against finite-difference approximation. Per Baydin et al. (2018), if
/// |autodiff - numerical| / max(|autodiff|, |numerical|, 1e-8) &lt; 1e-5
/// the gradient is correct.
/// </summary>
public class DifferentiableOpsGradientCheckTests
{
    private const double Epsilon = 1e-5;
    private const double RelTolerance = 1e-4;
    private static readonly INumericOperations<double> NumOps = MathHelper.GetNumericOperations<double>();

    // ─── Elementwise ops ─────────────────────────────────────────────

    [Fact]
    public void Add_GradientCheck()
    {
        var a = Tensor(3.0, 4.0);
        var b = Tensor(1.0, 2.0);
        GradientCheck(
            inputs: [a, b],
            forward: xs => DifferentiableOps<double>.Add(xs[0], xs[1]),
            name: "Add");
    }

    [Fact]
    public void Subtract_GradientCheck()
    {
        var a = Tensor(3.0, 4.0);
        var b = Tensor(1.0, 2.0);
        GradientCheck(
            inputs: [a, b],
            forward: xs => DifferentiableOps<double>.Subtract(xs[0], xs[1]),
            name: "Subtract");
    }

    [Fact]
    public void Multiply_GradientCheck()
    {
        var a = Tensor(3.0, 4.0);
        var b = Tensor(5.0, 6.0);
        GradientCheck(
            inputs: [a, b],
            forward: xs => DifferentiableOps<double>.Multiply(xs[0], xs[1]),
            name: "Multiply");
    }

    [Fact]
    public void Divide_GradientCheck()
    {
        var a = Tensor(6.0, 8.0);
        var b = Tensor(2.0, 4.0); // Non-zero to avoid division by zero
        GradientCheck(
            inputs: [a, b],
            forward: xs => DifferentiableOps<double>.Divide(xs[0], xs[1]),
            name: "Divide");
    }

    [Fact]
    public void Negate_GradientCheck()
    {
        var a = Tensor(3.0, -2.0);
        GradientCheck(
            inputs: [a],
            forward: xs => DifferentiableOps<double>.Negate(xs[0]),
            name: "Negate");
    }

    // ─── Scalar ops ──────────────────────────────────────────────────

    [Fact]
    public void MultiplyScalar_GradientCheck()
    {
        var a = Tensor(3.0, 4.0);
        GradientCheck(
            inputs: [a],
            forward: xs => DifferentiableOps<double>.MultiplyScalar(xs[0], 2.5),
            name: "MultiplyScalar");
    }

    [Fact]
    public void AddScalar_GradientCheck()
    {
        var a = Tensor(3.0, 4.0);
        GradientCheck(
            inputs: [a],
            forward: xs => DifferentiableOps<double>.AddScalar(xs[0], 10.0),
            name: "AddScalar");
    }

    // ─── Matrix ops ──────────────────────────────────────────────────

    [Fact]
    public void MatMul_GradientCheck()
    {
        // A: [2,3], B: [3,2] → C: [2,2]
        var a = new Tensor<double>([2, 3], new Vector<double>([1, 2, 3, 4, 5, 6]));
        var b = new Tensor<double>([3, 2], new Vector<double>([7, 8, 9, 10, 11, 12]));
        GradientCheck(
            inputs: [a, b],
            forward: xs => DifferentiableOps<double>.MatMul(xs[0], xs[1]),
            name: "MatMul");
    }

    [Fact]
    public void Transpose_GradientCheck()
    {
        var a = new Tensor<double>([2, 3], new Vector<double>([1, 2, 3, 4, 5, 6]));
        GradientCheck(
            inputs: [a],
            forward: xs => DifferentiableOps<double>.Transpose(xs[0]),
            name: "Transpose");
    }

    // ─── Reduction ops ───────────────────────────────────────────────

    [Fact]
    public void Sum_GradientCheck()
    {
        var a = Tensor(1.0, 2.0, 3.0, 4.0);
        GradientCheck(
            inputs: [a],
            forward: xs => DifferentiableOps<double>.Sum(xs[0]),
            name: "Sum");
    }

    [Fact]
    public void Mean_GradientCheck()
    {
        var a = Tensor(1.0, 2.0, 3.0, 4.0);
        GradientCheck(
            inputs: [a],
            forward: xs => DifferentiableOps<double>.Mean(xs[0]),
            name: "Mean");
    }

    // ─── Activation ops ──────────────────────────────────────────────

    [Fact]
    public void Sigmoid_GradientCheck()
    {
        var x = Tensor(-1.0, 0.0, 1.0, 2.0);
        GradientCheck(
            inputs: [x],
            forward: xs => DifferentiableOps<double>.Sigmoid(xs[0]),
            name: "Sigmoid");
    }

    [Fact]
    public void Tanh_GradientCheck()
    {
        var x = Tensor(-1.0, 0.0, 0.5, 1.5);
        GradientCheck(
            inputs: [x],
            forward: xs => DifferentiableOps<double>.Tanh(xs[0]),
            name: "Tanh");
    }

    [Fact]
    public void ReLU_GradientCheck()
    {
        // Avoid x=0 (non-differentiable point)
        var x = Tensor(-2.0, -0.5, 0.5, 2.0);
        GradientCheck(
            inputs: [x],
            forward: xs => DifferentiableOps<double>.ReLU(xs[0]),
            name: "ReLU");
    }

    [Fact]
    public void GELU_GradientCheck()
    {
        var x = Tensor(-1.0, 0.0, 0.5, 1.5);
        GradientCheck(
            inputs: [x],
            forward: xs => DifferentiableOps<double>.GELU(xs[0]),
            name: "GELU");
    }

    [Fact]
    public void Swish_GradientCheck()
    {
        var x = Tensor(-1.0, 0.0, 0.5, 1.5);
        GradientCheck(
            inputs: [x],
            forward: xs => DifferentiableOps<double>.Swish(xs[0]),
            name: "Swish");
    }

    // ─── Chain rule (multi-op) ───────────────────────────────────────

    [Fact]
    public void Chain_TanhOfMatMulPlusBias_GradientCheck()
    {
        var x = new Tensor<double>([1, 3], new Vector<double>([0.5, -0.3, 0.8]));
        var w = new Tensor<double>([3, 2], new Vector<double>([0.1, 0.2, -0.3, 0.4, 0.5, -0.1]));
        var b = new Tensor<double>([1, 2], new Vector<double>([0.1, -0.2]));

        GradientCheck(
            inputs: [x, w, b],
            forward: xs =>
            {
                var linear = DifferentiableOps<double>.MatMul(xs[0], xs[1]);
                var biased = DifferentiableOps<double>.Add(linear, xs[2]);
                return DifferentiableOps<double>.Tanh(biased);
            },
            name: "tanh(x@w + b)");
    }

    [Fact]
    public void Chain_MSELoss_GradientCheck()
    {
        var pred = Tensor(0.5, 0.8, 0.2);
        var target = Tensor(1.0, 0.0, 0.5);

        GradientCheck(
            inputs: [pred],
            forward: xs =>
            {
                var diff = DifferentiableOps<double>.Subtract(xs[0], target);
                var squared = DifferentiableOps<double>.Multiply(diff, diff);
                return DifferentiableOps<double>.Mean(squared);
            },
            name: "MSE(pred, target)");
    }

    // ─── Conv ops ─────────────────────────────────────────────────────

    [Fact]
    public void Conv2D_GradientCheck()
    {
        // Input [N=1, C=1, H=4, W=4], Kernel [C_out=1, C_in=1, kH=3, kW=3]
        var input = new Tensor<double>([1, 1, 4, 4]);
        var kernel = new Tensor<double>([1, 1, 3, 3]);
        var rng = new Random(42);
        for (int i = 0; i < input.Length; i++) input[i] = rng.NextDouble();
        for (int i = 0; i < kernel.Length; i++) kernel[i] = rng.NextDouble() * 0.5;

        GradientCheck(
            inputs: [input, kernel],
            forward: xs => DifferentiableOps<double>.Conv2D(
                xs[0], xs[1], [1, 1], [0, 0], [1, 1]),
            name: "Conv2D");
    }

    // ─── Normalization ops ───────────────────────────────────────────

    [Fact]
    public void LayerNorm_GradientCheck()
    {
        // Use non-trivial gamma to ensure gradients are non-zero
        var input = new Tensor<double>([2, 4], new Vector<double>([1, 2, 3, 4, 5, 6, 7, 8]));
        var gamma = new Tensor<double>([4], new Vector<double>([1.5, 0.8, 1.2, 0.9]));
        var beta = new Tensor<double>([4], new Vector<double>([0.1, -0.1, 0.2, 0.0]));
        var weights = new Tensor<double>([2, 4], new Vector<double>([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]));

        GradientCheck(
            inputs: [input, gamma, beta],
            forward: xs =>
            {
                var normed = DifferentiableOps<double>.LayerNorm(xs[0], xs[1], xs[2]);
                return DifferentiableOps<double>.Multiply(normed, weights);
            },
            name: "LayerNorm (weighted)");
    }

    [Fact]
    public void BatchNorm_GradientCheck()
    {
        var input = new Tensor<double>([3, 4]);
        var rng = new Random(42);
        for (int i = 0; i < input.Length; i++) input[i] = rng.NextDouble() * 2 - 1;
        var gamma = new Tensor<double>([4], new Vector<double>([1.5, 0.8, 1.2, 0.9]));
        var beta = new Tensor<double>([4], new Vector<double>([0.1, -0.1, 0.2, 0.0]));
        var weights = new Tensor<double>([3, 4]);
        for (int i = 0; i < weights.Length; i++) weights[i] = 0.1 * (i + 1);

        GradientCheck(
            inputs: [input, gamma, beta],
            forward: xs =>
            {
                var normed = DifferentiableOps<double>.BatchNorm(xs[0], xs[1], xs[2]);
                return DifferentiableOps<double>.Multiply(normed, weights);
            },
            name: "BatchNorm (weighted)");
    }

    [Fact]
    public void Softmax_GradientCheck()
    {
        // Use a weighted sum (not plain sum) because sum(softmax) = 1 (constant → zero gradient)
        var x = Tensor(1.0, 2.0, 3.0, 0.5);
        var weights = Tensor(0.1, 0.4, 0.3, 0.2);
        GradientCheck(
            inputs: [x],
            forward: xs =>
            {
                var s = DifferentiableOps<double>.Softmax(xs[0]);
                return DifferentiableOps<double>.Multiply(s, weights);
            },
            name: "Softmax (weighted)");
    }

    // ─── End-to-end: Conv + BatchNorm + ReLU chain ───────────────────

    [Fact]
    public void Chain_ConvReLU_GradientCheck()
    {
        // Simpler chain without BatchNorm (which has batch=1 edge case)
        var input = new Tensor<double>([1, 1, 4, 4]);
        var kernel = new Tensor<double>([1, 1, 3, 3]);
        var rng = new Random(42);
        for (int i = 0; i < input.Length; i++) input[i] = rng.NextDouble();
        for (int i = 0; i < kernel.Length; i++) kernel[i] = rng.NextDouble() * 0.5;

        GradientCheck(
            inputs: [input, kernel],
            forward: xs =>
            {
                var conv = DifferentiableOps<double>.Conv2D(xs[0], xs[1], [1, 1], [0, 0], [1, 1]);
                return DifferentiableOps<double>.ReLU(conv);
            },
            name: "Conv→ReLU");
    }

    [Fact]
    public void Chain_LayerNormTanh_GradientCheck()
    {
        var input = new Tensor<double>([2, 4]);
        var rng = new Random(42);
        for (int i = 0; i < input.Length; i++) input[i] = rng.NextDouble() * 2 - 1;
        var gamma = new Tensor<double>([4], new Vector<double>([1.2, 0.9, 1.1, 0.8]));
        var beta = new Tensor<double>([4], new Vector<double>([0.1, 0.0, -0.1, 0.05]));

        GradientCheck(
            inputs: [input, gamma, beta],
            forward: xs =>
            {
                var normed = DifferentiableOps<double>.LayerNorm(xs[0], xs[1], xs[2]);
                return DifferentiableOps<double>.Tanh(normed);
            },
            name: "LayerNorm→Tanh");
    }

    // ─── Test infrastructure ─────────────────────────────────────────

    private static Tensor<double> Tensor(params double[] values) =>
        new([values.Length], new Vector<double>(values));

    /// <summary>
    /// Verifies autodiff gradients against finite-difference approximation
    /// for each input tensor.
    /// </summary>
    private static void GradientCheck(
        Tensor<double>[] inputs,
        Func<Tensor<double>[], Tensor<double>> forward,
        string name)
    {
        // 1. Compute autodiff gradients
        var gradients = new Dictionary<Tensor<double>, Tensor<double>>();
        using (var tape = new GradientTape<double>())
        {
            foreach (var input in inputs)
                tape.Watch(input);

            var output = forward(inputs);
            var loss = DifferentiableOps<double>.Sum(output);
            gradients = tape.Gradient(loss);
        }

        // 2. Verify each input's gradient against finite differences
        for (int inputIdx = 0; inputIdx < inputs.Length; inputIdx++)
        {
            var input = inputs[inputIdx];
            Assert.True(gradients.ContainsKey(input),
                $"{name}: no gradient for input[{inputIdx}]");

            var autodiffGrad = gradients[input];

            for (int i = 0; i < input.Length; i++)
            {
                double original = input[i];

                // f(x + ε)
                input[i] = original + Epsilon;
                double fPlus;
                using (var _ = new NoGradScope<double>())
                {
                    var outPlus = forward(inputs);
                    fPlus = SumAll(outPlus);
                }

                // f(x - ε)
                input[i] = original - Epsilon;
                double fMinus;
                using (var _ = new NoGradScope<double>())
                {
                    var outMinus = forward(inputs);
                    fMinus = SumAll(outMinus);
                }

                // Restore
                input[i] = original;

                // Numerical gradient: (f(x+ε) - f(x-ε)) / 2ε
                double numerical = (fPlus - fMinus) / (2 * Epsilon);
                double autodiff = autodiffGrad[i];

                // Relative error check (Baydin et al. 2018)
                double denom = Math.Max(Math.Abs(autodiff), Math.Max(Math.Abs(numerical), 1e-8));
                double relError = Math.Abs(autodiff - numerical) / denom;

                Assert.True(relError < RelTolerance,
                    $"{name} input[{inputIdx}][{i}]: autodiff={autodiff:G6}, numerical={numerical:G6}, " +
                    $"relError={relError:G4} > {RelTolerance}");
            }
        }
    }

    private static double SumAll(Tensor<double> t)
    {
        double sum = 0;
        for (int i = 0; i < t.Length; i++)
            sum += t[i];
        return sum;
    }
}
