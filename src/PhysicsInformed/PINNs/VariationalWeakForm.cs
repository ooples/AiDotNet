using System;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Validation;

namespace AiDotNet.PhysicsInformed.PINNs
{
    /// <summary>
    /// Standard weak-form integrands for <see cref="VariationalPINN{T}"/>, batched over the quadrature points.
    /// </summary>
    /// <typeparam name="T">The numeric type.</typeparam>
    public static class VariationalWeakForm<T>
    {
        /// <summary>
        /// The Poisson integrand grad u . grad v - f v of Kharazmi et al. (2019), Eq. 6, for -laplacian u = f with a
        /// constant source.
        /// </summary>
        /// <param name="source">The constant source f.</param>
        public static Func<Tensor<T>, Tensor<T>, Tensor<T>, Tensor<T>, Tensor<T>, Tensor<T>> Poisson(T source)
        {
            return (x, u, gradU, v, gradV) =>
            {
                var engine = AiDotNetEngine.Current;
                return engine.TensorSubtract(Stiffness(engine, gradU, gradV), engine.TensorMultiplyScalar(v, source));
            };
        }

        /// <summary>
        /// The Poisson integrand grad u . grad v - f(x) v for -laplacian u = f.
        /// </summary>
        /// <param name="source">The batched source: points [N, d] to f [N].</param>
        public static Func<Tensor<T>, Tensor<T>, Tensor<T>, Tensor<T>, Tensor<T>, Tensor<T>> Poisson(Func<Tensor<T>, Tensor<T>> source)
        {
            Guard.NotNull(source);
            return (x, u, gradU, v, gradV) =>
            {
                var engine = AiDotNetEngine.Current;
                var f = engine.Reshape(source(x), new[] { v.Shape[0] });
                return engine.TensorSubtract(Stiffness(engine, gradU, gradV), engine.TensorMultiply(f, v));
            };
        }

        private static Tensor<T> Stiffness(IEngine engine, Tensor<T> gradU, Tensor<T> gradV)
        {
            int n = gradU.Shape[0];
            int d = gradU.Shape[2];
            var first = engine.Reshape(engine.TensorSlice(gradU, new[] { 0, 0, 0 }, new[] { n, 1, d }), new[] { n, d });
            return engine.ReduceSum(engine.TensorMultiply(first, gradV), new[] { 1 }, keepDims: false);
        }
    }
}
