using System;
using AiDotNet.ActivationFunctions;
using AiDotNet.Enums;
using AiDotNet.Factories;
using AiDotNet.Interfaces;
using Xunit;
using System.Threading.Tasks;

namespace AiDotNet.Tests.ActivationFunctions
{
    public class ActivationFunctionBehaviorTests
    {
        private static void AssertClose(double actual, double expected, double tol = 1e-6)
        {
            Assert.True(Math.Abs(actual - expected) <= tol, $"Actual {actual} != Expected {expected}");
        }

        [Fact(Timeout = 60000)]
        public async Task ReLU_Activate_And_Derivative()
        {
            var fn = ActivationFunctionFactory<double>.CreateActivationFunction(ActivationFunction.ReLU);
            AssertClose(fn.Activate(-1.0), 0.0);
            AssertClose(fn.Activate(2.5), 2.5);
            AssertClose(fn.Derivative(-1.0), 0.0);
        }

        [Fact(Timeout = 60000)]
        public async Task Sigmoid_Activate_And_Derivative()
        {
            var fn = ActivationFunctionFactory<double>.CreateActivationFunction(ActivationFunction.Sigmoid);
            var y = fn.Activate(0.0);
            AssertClose(y, 0.5);
            var dy = fn.Derivative(0.0);
            Assert.True(dy > 0.0 && dy < 0.3);
        }

        [Fact(Timeout = 60000)]
        public async Task Tanh_Activate_And_Derivative()
        {
            var fn = ActivationFunctionFactory<double>.CreateActivationFunction(ActivationFunction.Tanh);
            AssertClose(fn.Activate(0.0), 0.0);
        }

        [Fact(Timeout = 60000)]
        public async Task Identity_Activate()
        {
            var fn = ActivationFunctionFactory<double>.CreateActivationFunction(ActivationFunction.Linear);
            AssertClose(fn.Activate(3.14), 3.14);
        }

        [Fact(Timeout = 60000)]
        public async Task LeakyRelu_Activate()
        {
            var fn = ActivationFunctionFactory<double>.CreateActivationFunction(ActivationFunction.LeakyReLU);
            Assert.True(fn.Activate(-2.0) < 0 && fn.Activate(2.0) > 0);
        }

        [Fact(Timeout = 60000)]
        public async Task ELU_SELU_Activate()
        {
            var elu = ActivationFunctionFactory<double>.CreateActivationFunction(ActivationFunction.ELU);
            var selu = ActivationFunctionFactory<double>.CreateActivationFunction(ActivationFunction.SELU);
            Assert.True(elu.Activate(-1.0) < 0.0);
            Assert.True(selu.Activate(-1.0) < 0.0);
        }

        [Fact(Timeout = 60000)]
        public async Task Softplus_SoftSign_Swish_GELU()
        {
            var sp = ActivationFunctionFactory<double>.CreateActivationFunction(ActivationFunction.Softplus);
            var ss = ActivationFunctionFactory<double>.CreateActivationFunction(ActivationFunction.SoftSign);
            var sw = ActivationFunctionFactory<double>.CreateActivationFunction(ActivationFunction.Swish);
            var ge = ActivationFunctionFactory<double>.CreateActivationFunction(ActivationFunction.GELU);
            Assert.True(sp.Activate(1.0) > 0.0);
            Assert.True(ss.Activate(1.0) > 0.0 && ss.Activate(1.0) <= 1.0);
            Assert.True(sw.Activate(1.0) > 0.0);
            Assert.True(ge.Activate(1.0) > 0.0);
        }

        [Fact(Timeout = 60000)]
        public async Task Vector_Softmax_Factory()
        {
            var vfn = ActivationFunctionFactory<double>.CreateVectorActivationFunction(ActivationFunction.Softmax);
            Assert.IsAssignableFrom<IVectorActivationFunction<double>>(vfn);
        }

        // ----------------------------------------------------------------
        // IdentityActivation.Activate(Tensor<T>) — PR change coverage
        // ----------------------------------------------------------------

        [Fact(Timeout = 60000)]
        public async Task IdentityActivation_TensorActivate_ReturnsSameReference()
        {
            // The new override must return the exact same object (no allocation).
            // This is required to preserve the autodiff tape chain.
            var fn = new IdentityActivation<double>();
            var input = new Tensor<double>([3, 4]);
            for (int i = 0; i < input.Length; i++)
                input.SetFlat(i, i * 1.5);

            var output = fn.Activate(input);

            Assert.True(ReferenceEquals(input, output),
                "IdentityActivation.Activate(Tensor) must return the exact same tensor reference — no copy.");
        }

        [Fact(Timeout = 60000)]
        public async Task IdentityActivation_TensorActivate_With1DTensor_ReturnsSameReference()
        {
            var fn = new IdentityActivation<double>();
            var input = new Tensor<double>([8]);
            input.SetFlat(0, 42.0);

            var output = fn.Activate(input);

            Assert.True(ReferenceEquals(input, output),
                "IdentityActivation.Activate(Tensor) must return the same reference for 1-D tensors.");
        }

        [Fact(Timeout = 60000)]
        public async Task IdentityActivation_TensorActivate_With3DTensor_ReturnsSameReference()
        {
            var fn = new IdentityActivation<double>();
            var input = new Tensor<double>([2, 3, 4]);
            for (int i = 0; i < input.Length; i++)
                input.SetFlat(i, (double)i);

            var output = fn.Activate(input);

            Assert.True(ReferenceEquals(input, output),
                "IdentityActivation.Activate(Tensor) must return the same reference for 3-D tensors.");
        }

        [Fact(Timeout = 60000)]
        public async Task IdentityActivation_TensorActivate_DataIsUnchanged()
        {
            // Even though it's the same reference, verify values are untouched.
            var fn = new IdentityActivation<double>();
            var data = new double[] { -5.0, 0.0, 3.14, double.MaxValue, double.MinValue };
            var input = new Tensor<double>(data, [data.Length]);

            var output = fn.Activate(input);

            for (int i = 0; i < data.Length; i++)
                Assert.Equal(data[i], output.GetFlat(i));
        }

        [Fact(Timeout = 60000)]
        public async Task IdentityActivation_TensorActivate_WithFloatType_ReturnsSameReference()
        {
            var fn = new IdentityActivation<float>();
            var input = new Tensor<float>([4]);
            input.SetFlat(0, 1.0f);
            input.SetFlat(1, -1.0f);

            var output = fn.Activate(input);

            Assert.True(ReferenceEquals(input, output),
                "IdentityActivation<float>.Activate(Tensor) must return the same reference.");
        }
    }
}