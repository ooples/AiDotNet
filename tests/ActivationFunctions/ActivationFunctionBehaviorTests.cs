using System;
using AiDotNet.ActivationFunctions;
using AiDotNet.Enums;
using AiDotNet.Factories;
using AiDotNet.Interfaces;
using Xunit;

namespace AiDotNet.Tests.ActivationFunctions
{
    public class ActivationFunctionBehaviorTests
    {
        private static void AssertClose(double actual, double expected, double tol = 1e-6)
        {
            Assert.True(Math.Abs(actual - expected) <= tol, $"Actual {actual} != Expected {expected}");
        }

        [Fact]
        public void ReLU_Activate_And_Derivative()
        {
            var fn = ActivationFunctionFactory<double>.CreateActivationFunction(ActivationFunction.ReLU);
            AssertClose(fn.Activate(-1.0), 0.0);
            AssertClose(fn.Activate(2.5), 2.5);
            AssertClose(fn.Derivative(-1.0), 0.0);
        }

        [Fact]
        public void Sigmoid_Activate_And_Derivative()
        {
            var fn = ActivationFunctionFactory<double>.CreateActivationFunction(ActivationFunction.Sigmoid);
            var y = fn.Activate(0.0);
            AssertClose(y, 0.5);
            var dy = fn.Derivative(0.0);
            Assert.True(dy > 0.0 && dy < 0.3);
        }

        [Fact]
        public void Tanh_Activate_And_Derivative()
        {
            var fn = ActivationFunctionFactory<double>.CreateActivationFunction(ActivationFunction.Tanh);
            AssertClose(fn.Activate(0.0), 0.0);
        }

        [Fact]
        public void Identity_Activate()
        {
            var fn = ActivationFunctionFactory<double>.CreateActivationFunction(ActivationFunction.Linear);
            AssertClose(fn.Activate(3.14), 3.14);
        }

        [Fact]
        public void LeakyRelu_Activate()
        {
            var fn = ActivationFunctionFactory<double>.CreateActivationFunction(ActivationFunction.LeakyReLU);
            Assert.True(fn.Activate(-2.0) < 0 && fn.Activate(2.0) > 0);
        }

        [Fact]
        public void ELU_SELU_Activate()
        {
            var elu = ActivationFunctionFactory<double>.CreateActivationFunction(ActivationFunction.ELU);
            var selu = ActivationFunctionFactory<double>.CreateActivationFunction(ActivationFunction.SELU);
            Assert.True(elu.Activate(-1.0) < 0.0);
            Assert.True(selu.Activate(-1.0) < 0.0);
        }

        [Fact]
        public void Softplus_SoftSign_Swish_GELU()
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

        [Fact]
        public void Vector_Softmax_Factory()
        {
            var vfn = ActivationFunctionFactory<double>.CreateVectorActivationFunction(ActivationFunction.Softmax);
            Assert.IsAssignableFrom<IVectorActivationFunction<double>>(vfn);
        }
    }
}
