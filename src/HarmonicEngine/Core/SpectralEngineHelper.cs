using AiDotNet.LinearAlgebra;
using AiDotNet.Tensors.Engines;

namespace AiDotNet.HarmonicEngine.Core;

/// <summary>
/// Helper for converting between Vector/Complex types and Tensor types
/// for Engine-accelerated spectral operations.
/// </summary>
internal static class SpectralEngineHelper
{
    private static IEngine Engine => AiDotNetEngine.Current;

    /// <summary>Convert Vector&lt;T&gt; to Tensor&lt;T&gt; for Engine operations.</summary>
    public static Tensor<T> ToTensor<T>(Vector<T> vector)
    {
        var tensor = new Tensor<T>([vector.Length]);
        for (int i = 0; i < vector.Length; i++) tensor[i] = vector[i];
        return tensor;
    }

    /// <summary>Convert Tensor&lt;T&gt; back to Vector&lt;T&gt;.</summary>
    public static Vector<T> ToVector<T>(Tensor<T> tensor)
    {
        var vector = new Vector<T>(tensor.Length);
        for (int i = 0; i < tensor.Length; i++) vector[i] = tensor[i];
        return vector;
    }

    /// <summary>Convert Vector&lt;Complex&lt;T&gt;&gt; to Tensor&lt;Complex&lt;T&gt;&gt;.</summary>
    public static Tensor<Complex<T>> ToComplexTensor<T>(Vector<Complex<T>> vector)
    {
        var tensor = new Tensor<Complex<T>>([vector.Length]);
        for (int i = 0; i < vector.Length; i++) tensor[i] = vector[i];
        return tensor;
    }

    /// <summary>Convert Tensor&lt;Complex&lt;T&gt;&gt; back to Vector&lt;Complex&lt;T&gt;&gt;.</summary>
    public static Vector<Complex<T>> ToComplexVector<T>(Tensor<Complex<T>> tensor)
    {
        var vector = new Vector<Complex<T>>(tensor.Length);
        for (int i = 0; i < tensor.Length; i++) vector[i] = tensor[i];
        return vector;
    }

    /// <summary>FFT via Engine (returns Tensor&lt;Complex&lt;T&gt;&gt;).</summary>
    public static Tensor<Complex<T>> FFT<T>(Vector<T> signal)
        => Engine.NativeComplexFFT(ToTensor(signal));

    /// <summary>IFFT via Engine (returns Vector&lt;T&gt;).</summary>
    public static Vector<T> IFFTReal<T>(Tensor<Complex<T>> spectrum)
        => ToVector(Engine.NativeComplexIFFTReal(spectrum));

    /// <summary>Cross-spectral density via Engine: X * conj(Y).</summary>
    public static Tensor<Complex<T>> CrossSpectral<T>(Tensor<Complex<T>> x, Tensor<Complex<T>> y)
        => Engine.NativeComplexCrossSpectral(x, y);

    /// <summary>Magnitude squared via Engine: re^2 + im^2.</summary>
    public static Tensor<T> MagnitudeSquared<T>(Tensor<Complex<T>> spectrum)
        => Engine.NativeComplexMagnitudeSquared(spectrum);

    /// <summary>Magnitude via Engine: sqrt(re^2 + im^2).</summary>
    public static Tensor<T> Magnitude<T>(Tensor<Complex<T>> spectrum)
        => Engine.NativeComplexMagnitude(spectrum);

    /// <summary>Complex multiply via Engine.</summary>
    public static Tensor<Complex<T>> ComplexMultiply<T>(Tensor<Complex<T>> a, Tensor<Complex<T>> b)
        => Engine.NativeComplexMultiply(a, b);

    /// <summary>Complex conjugate via Engine.</summary>
    public static Tensor<Complex<T>> Conjugate<T>(Tensor<Complex<T>> a)
        => Engine.NativeComplexConjugate(a);

    /// <summary>TopK by magnitude via Engine.</summary>
    public static Tensor<Complex<T>> TopK<T>(Tensor<Complex<T>> spectrum, int k)
        => Engine.NativeComplexTopK(spectrum, k);

    /// <summary>Per-row softmax via Engine.</summary>
    public static Tensor<T> SoftmaxRows<T>(Tensor<T> matrix)
        => Engine.TensorSoftmaxRows(matrix);
}
