using System.Reflection;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Helpers;

/// <summary>
/// Exposes the tensor library's dense linear-algebra kernels (<see cref="Linalg"/>) to code whose element
/// type <typeparamref name="T"/> is unconstrained. <see cref="Linalg"/> constrains its element type to
/// <c>unmanaged</c>, which a generic decomposition cannot state; each kernel is bound once per
/// <typeparamref name="T"/> by closing the generic method over the runtime type, so the call stays generic
/// and runs the library's own (device-dispatched) implementation.
/// </summary>
internal static class LinalgDispatch<T>
{
    private static Func<Tensor<T>, bool, (Tensor<T>, Tensor<int>)>? _choleskyEx;
    private static Func<Tensor<T>, Tensor<T>>? _det;
    private static Func<Tensor<T>, Tensor<T>>? _inv;
    private static Func<Tensor<T>, bool, (Tensor<T>, Tensor<T>)>? _eigh;
    private static Func<Tensor<T>, (Tensor<T>, Tensor<T>)>? _eig;
    private static Func<Tensor<T>, Tensor<T>, (Tensor<T>, Tensor<int>)>? _solveEx;
    private static Func<Tensor<T>, string, (Tensor<T>, Tensor<T>)>? _qr;
    private static Func<Tensor<T>, (Tensor<T>, Tensor<T>, Tensor<T>)>? _lu;
    private static Func<Tensor<T>, Tensor<T>, bool, bool, Tensor<T>>? _solveTriangular;
    private static Func<Tensor<T>, bool, (Tensor<T>, Tensor<T>, Tensor<T>)>? _svd;
    private static Func<Tensor<T>, int, int, int, Tensor<T>>? _diagonal;
    private static Func<Tensor<T>, Tensor<T>, bool, Tensor<T>>? _choleskySolve;
    private static Func<Tensor<T>, bool, Tensor<T>>? _choleskyInverse;
    private static Func<Tensor<T>, Tensor<T>, Tensor<T>>? _solve;
    private static Func<Tensor<T>, (Tensor<T>, Tensor<int>)>? _luFactor;
    private static Func<Tensor<T>, Tensor<int>, Tensor<T>, Tensor<T>>? _luSolve;

    // The library's kernels run in float and double only (Cholesky, for one, rejects Decimal at run time). Every other
    // element type is computed in double and converted back, as the element loops these calls replaced handled any T.
    private static readonly bool Native = typeof(T) == typeof(float) || typeof(T) == typeof(double);

    private static Tensor<double> Wide(Tensor<T> value) => value.Cast<double>();

    private static Tensor<T> Narrow(Tensor<double> value) => value.Cast<T>();

    /// <summary>Cholesky factor with per-matrix failure info (<c>Linalg.CholeskyEx</c>).</summary>
    internal static (Tensor<T> L, Tensor<int> Info) CholeskyEx(Tensor<T> input, bool upper = false)
    {
        if (Native)
            return (_choleskyEx ??= Bind<Func<Tensor<T>, bool, (Tensor<T>, Tensor<int>)>>(
                nameof(Linalg.CholeskyEx), 2))(input, upper);
        var (l, info) = LinalgDispatch<double>.CholeskyEx(Wide(input), upper);
        return (Narrow(l), info);
    }

    /// <summary>Determinant (<c>Linalg.Det</c>).</summary>
    internal static Tensor<T> Det(Tensor<T> input)
        => Native
            ? (_det ??= Bind<Func<Tensor<T>, Tensor<T>>>(nameof(Linalg.Det), 1))(input)
            : Narrow(LinalgDispatch<double>.Det(Wide(input)));

    /// <summary>Matrix inverse (<c>Linalg.Inv</c>).</summary>
    internal static Tensor<T> Inv(Tensor<T> input)
        => Native
            ? (_inv ??= Bind<Func<Tensor<T>, Tensor<T>>>(nameof(Linalg.Inv), 1))(input)
            : Narrow(LinalgDispatch<double>.Inv(Wide(input)));

    /// <summary>Symmetric eigendecomposition (<c>Linalg.Eigh</c>).</summary>
    internal static (Tensor<T> Values, Tensor<T> Vectors) Eigh(Tensor<T> input, bool upper = false)
    {
        if (Native)
            return (_eigh ??= Bind<Func<Tensor<T>, bool, (Tensor<T>, Tensor<T>)>>(
                nameof(Linalg.Eigh), 2))(input, upper);
        var (values, vectors) = LinalgDispatch<double>.Eigh(Wide(input), upper);
        return (Narrow(values), Narrow(vectors));
    }

    /// <summary>General eigendecomposition (<c>Linalg.Eig</c>).</summary>
    internal static (Tensor<T> Values, Tensor<T> Vectors) Eig(Tensor<T> input)
    {
        if (Native)
            return (_eig ??= Bind<Func<Tensor<T>, (Tensor<T>, Tensor<T>)>>(nameof(Linalg.Eig), 1))(input);
        var (values, vectors) = LinalgDispatch<double>.Eig(Wide(input));
        return (Narrow(values), Narrow(vectors));
    }

    /// <summary>Linear solve with per-system failure info (<c>Linalg.SolveEx</c>).</summary>
    internal static (Tensor<T> Result, Tensor<int> Info) SolveEx(Tensor<T> a, Tensor<T> b)
    {
        if (Native)
            return (_solveEx ??= Bind<Func<Tensor<T>, Tensor<T>, (Tensor<T>, Tensor<int>)>>(
                nameof(Linalg.SolveEx), 2))(a, b);
        var (result, info) = LinalgDispatch<double>.SolveEx(Wide(a), Wide(b));
        return (Narrow(result), info);
    }

    /// <summary>QR decomposition (<c>Linalg.QR</c>); <paramref name="mode"/> is "reduced" or "complete".</summary>
    internal static (Tensor<T> Q, Tensor<T> R) QR(Tensor<T> input, string mode = "reduced")
    {
        if (Native)
            return (_qr ??= Bind<Func<Tensor<T>, string, (Tensor<T>, Tensor<T>)>>(nameof(Linalg.QR), 2))(input, mode);
        var (q, r) = LinalgDispatch<double>.QR(Wide(input), mode);
        return (Narrow(q), Narrow(r));
    }

    /// <summary>LU decomposition with partial pivoting (<c>Linalg.LU</c>).</summary>
    internal static (Tensor<T> P, Tensor<T> L, Tensor<T> U) LU(Tensor<T> input)
    {
        if (Native)
            return (_lu ??= Bind<Func<Tensor<T>, (Tensor<T>, Tensor<T>, Tensor<T>)>>(nameof(Linalg.LU), 1))(input);
        var (permutation, lower, upper) = LinalgDispatch<double>.LU(Wide(input));
        return (Narrow(permutation), Narrow(lower), Narrow(upper));
    }

    /// <summary>Triangular solve (<c>Linalg.SolveTriangular</c>).</summary>
    internal static Tensor<T> SolveTriangular(Tensor<T> a, Tensor<T> b, bool upper, bool unitDiagonal = false)
        => Native
            ? (_solveTriangular ??= Bind<Func<Tensor<T>, Tensor<T>, bool, bool, Tensor<T>>>(
                nameof(Linalg.SolveTriangular), 4))(a, b, upper, unitDiagonal)
            : Narrow(LinalgDispatch<double>.SolveTriangular(Wide(a), Wide(b), upper, unitDiagonal));

    /// <summary>Singular value decomposition (<c>Linalg.Svd</c>).</summary>
    internal static (Tensor<T> U, Tensor<T> S, Tensor<T> Vh) Svd(Tensor<T> input, bool fullMatrices = true)
    {
        if (Native)
            return (_svd ??= Bind<Func<Tensor<T>, bool, (Tensor<T>, Tensor<T>, Tensor<T>)>>(
                nameof(Linalg.Svd), 2))(input, fullMatrices);
        var (u, singular, vh) = LinalgDispatch<double>.Svd(Wide(input), fullMatrices);
        return (Narrow(u), Narrow(singular), Narrow(vh));
    }

    /// <summary>Diagonal extraction (<c>Linalg.Diagonal</c>).</summary>
    internal static Tensor<T> Diagonal(Tensor<T> input, int offset = 0, int dim1 = -2, int dim2 = -1)
        => Native
            ? (_diagonal ??= Bind<Func<Tensor<T>, int, int, int, Tensor<T>>>(
                nameof(Linalg.Diagonal), 4))(input, offset, dim1, dim2)
            : Narrow(LinalgDispatch<double>.Diagonal(Wide(input), offset, dim1, dim2));

    /// <summary>Solve from a Cholesky factor (<c>Linalg.CholeskySolve</c>).</summary>
    internal static Tensor<T> CholeskySolve(Tensor<T> b, Tensor<T> factor, bool upper = false)
        => Native
            ? (_choleskySolve ??= Bind<Func<Tensor<T>, Tensor<T>, bool, Tensor<T>>>(
                nameof(Linalg.CholeskySolve), 3))(b, factor, upper)
            : Narrow(LinalgDispatch<double>.CholeskySolve(Wide(b), Wide(factor), upper));

    /// <summary>Inverse from a Cholesky factor (<c>Linalg.CholeskyInverse</c>).</summary>
    internal static Tensor<T> CholeskyInverse(Tensor<T> factor, bool upper = false)
        => Native
            ? (_choleskyInverse ??= Bind<Func<Tensor<T>, bool, Tensor<T>>>(
                nameof(Linalg.CholeskyInverse), 2))(factor, upper)
            : Narrow(LinalgDispatch<double>.CholeskyInverse(Wide(factor), upper));

    /// <summary>Linear solve (<c>Linalg.Solve</c>).</summary>
    internal static Tensor<T> Solve(Tensor<T> a, Tensor<T> b)
        => Native
            ? (_solve ??= Bind<Func<Tensor<T>, Tensor<T>, Tensor<T>>>(nameof(Linalg.Solve), 2))(a, b)
            : Narrow(LinalgDispatch<double>.Solve(Wide(a), Wide(b)));

    /// <summary>Packed LU factor with its pivots (<c>Linalg.LuFactor</c>, LAPACK getrf form); the
    /// diagonal of the packed factor is the diagonal of U.</summary>
    internal static (Tensor<T> LU, Tensor<int> Pivots) LuFactor(Tensor<T> input)
    {
        if (Native)
            return (_luFactor ??= Bind<Func<Tensor<T>, (Tensor<T>, Tensor<int>)>>(nameof(Linalg.LuFactor), 1))(input);
        var (lu, pivots) = LinalgDispatch<double>.LuFactor(Wide(input));
        return (Narrow(lu), pivots);
    }

    /// <summary>Solve <c>AX = B</c> from a <see cref="LuFactor"/> result (<c>Linalg.LuSolve</c>).</summary>
    internal static Tensor<T> LuSolve(Tensor<T> lu, Tensor<int> pivots, Tensor<T> b)
        => Native
            ? (_luSolve ??= Bind<Func<Tensor<T>, Tensor<int>, Tensor<T>, Tensor<T>>>(
                nameof(Linalg.LuSolve), 3))(lu, pivots, b)
            : Narrow(LinalgDispatch<double>.LuSolve(Wide(lu), pivots, Wide(b)));

    private static TDelegate Bind<TDelegate>(string name, int parameterCount) where TDelegate : Delegate
    {
        var method = typeof(Linalg).GetMethods(BindingFlags.Public | BindingFlags.Static)
            .FirstOrDefault(m => m.Name == name && m.IsGenericMethodDefinition
                && m.GetGenericArguments().Length == 1 && m.GetParameters().Length == parameterCount);
        if (method is null)
            throw new MissingMethodException(typeof(Linalg).FullName, name);

        MethodInfo closed;
        try
        {
            closed = method.MakeGenericMethod(typeof(T));
        }
        catch (ArgumentException ex)
        {
            throw new NotSupportedException(
                $"Linalg.{name} requires an unmanaged numeric element type; {typeof(T).Name} is not supported.", ex);
        }

        return (TDelegate)Delegate.CreateDelegate(typeof(TDelegate), closed);
    }
}
