using System;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Validation;

namespace AiDotNet.PhysicsInformed.PINNs
{
    /// <summary>
    /// Standard energy densities for <see cref="DeepRitzMethod{T}"/>, batched over the quadrature points.
    /// </summary>
    /// <typeparam name="T">The numeric type.</typeparam>
    public static class DeepRitzEnergy<T>
    {
        private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

        /// <summary>
        /// The Poisson energy density 1/2 |grad u|^2 - f u of E and Yu (2018), Eq. 4, for -laplacian u = f with a
        /// constant source.
        /// </summary>
        /// <param name="source">The constant source f.</param>
        public static Func<Tensor<T>, Tensor<T>, Tensor<T>, Tensor<T>> Poisson(T source)
        {
            return (x, u, gradU) =>
            {
                var engine = AiDotNetEngine.Current;
                return Density(engine, u, gradU, engine.TensorMultiplyScalar(PinnAutodiff<T>.Column(engine, u, 0), source));
            };
        }

        /// <summary>
        /// The Poisson energy density 1/2 |grad u|^2 - f(x) u for -laplacian u = f.
        /// </summary>
        /// <param name="source">The batched source: points [N, d] to f [N].</param>
        public static Func<Tensor<T>, Tensor<T>, Tensor<T>, Tensor<T>> Poisson(Func<Tensor<T>, Tensor<T>> source)
        {
            Guard.NotNull(source);
            return (x, u, gradU) =>
            {
                var engine = AiDotNetEngine.Current;
                var column = PinnAutodiff<T>.Column(engine, u, 0);
                var f = engine.Reshape(source(x), new[] { column.Shape[0] });
                return Density(engine, u, gradU, engine.TensorMultiply(f, column));
            };
        }

        private static Tensor<T> Density(IEngine engine, Tensor<T> u, Tensor<T> gradU, Tensor<T> work)
        {
            var squared = engine.ReduceSum(engine.TensorMultiply(gradU, gradU), new[] { 1, 2 }, keepDims: false);
            return engine.TensorSubtract(engine.TensorMultiplyScalar(squared, NumOps.FromDouble(0.5)), work);
        }
    }
}
