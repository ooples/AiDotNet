using System;
using System.Linq;
using AiDotNet.Enums;
using AiDotNet.Factories;
using AiDotNet.Interfaces;
using Xunit;
using System.Threading.Tasks;

namespace AiDotNet.Tests.Factories
{
    public class ActivationFunctionFactoryTests
    {
        [Fact(Timeout = 60000)]
        public async Task CreateActivationFunction_Returns_For_Scalar_Compatible()
        {
            // Scalar-compatible functions in current enum
            var scalarValues = new[]
            {
                ActivationFunction.ReLU,
                ActivationFunction.Sigmoid,
                ActivationFunction.Tanh,
                ActivationFunction.Linear,
                ActivationFunction.LeakyReLU,
                ActivationFunction.ELU,
                ActivationFunction.SELU,
                ActivationFunction.Softplus,
                ActivationFunction.SoftSign,
                ActivationFunction.Swish,
                ActivationFunction.GELU,
                ActivationFunction.Identity
            };

            foreach (var af in scalarValues)
            {
                var fn = ActivationFunctionFactory<double>.CreateActivationFunction(af);
                Assert.NotNull(fn);
                Assert.IsAssignableFrom<IActivationFunction<double>>(fn);
            }
        }

        [Fact(Timeout = 60000)]
        public async Task CreateActivationFunction_Throws_For_Softmax_Scalar()
        {
            Assert.Throws<NotSupportedException>(() =>
                ActivationFunctionFactory<double>.CreateActivationFunction(ActivationFunction.Softmax));
        }

        [Fact(Timeout = 60000)]
        public async Task CreateVectorActivationFunction_Returns_For_Vector_Compatible()
        {
            // Vector-compatible functions in current enum (includes Softmax)
            var vectorValues = new[]
            {
                ActivationFunction.Softmax,
                ActivationFunction.ReLU,
                ActivationFunction.Sigmoid,
                ActivationFunction.Tanh,
                ActivationFunction.Linear,
                ActivationFunction.LeakyReLU,
                ActivationFunction.ELU,
                ActivationFunction.SELU,
                ActivationFunction.Softplus,
                ActivationFunction.SoftSign,
                ActivationFunction.Swish,
                ActivationFunction.GELU,
                ActivationFunction.Identity
            };

            foreach (var af in vectorValues)
            {
                var fn = ActivationFunctionFactory<double>.CreateVectorActivationFunction(af);
                Assert.NotNull(fn);
                Assert.IsAssignableFrom<IVectorActivationFunction<double>>(fn);
            }
        }
    }
}

