namespace AiDotNet.Factories;

public static class WindowFunctionFactory
{
    public static IWindowFunction<T> CreateWindowFunction<T>(WindowFunctionType type)
    {
        return type switch
        {
            WindowFunctionType.Rectangular => new RectangularWindow<T>(),
            WindowFunctionType.Hanning => new HanningWindow<T>(),
            WindowFunctionType.Hamming => new HammingWindow<T>(),
            WindowFunctionType.Blackman => new BlackmanWindow<T>(),
            WindowFunctionType.Kaiser => new KaiserWindow<T>(),
            WindowFunctionType.Bartlett => new BartlettWindow<T>(),
            WindowFunctionType.Gaussian => new GaussianWindow<T>(),
            WindowFunctionType.BartlettHann => new BartlettHannWindow<T>(),
            WindowFunctionType.Bohman => new BohmanWindow<T>(),
            WindowFunctionType.Lanczos => new LanczosWindow<T>(),
            WindowFunctionType.Parzen => new ParzenWindow<T>(),
            WindowFunctionType.Poisson => new PoissonWindow<T>(),
            WindowFunctionType.Nuttall => new NuttallWindow<T>(),
            WindowFunctionType.Triangular => new TriangularWindow<T>(),
            WindowFunctionType.BlackmanHarris => new BlackmanHarrisWindow<T>(),
            WindowFunctionType.FlatTop => new FlatTopWindow<T>(),
            WindowFunctionType.Welch => new WelchWindow<T>(),
            WindowFunctionType.BlackmanNuttall => new BlackmanNuttallWindow<T>(),
            WindowFunctionType.Cosine => new CosineWindow<T>(),
            WindowFunctionType.Tukey => new TukeyWindow<T>(),
            _ => throw new ArgumentException("Unsupported window function type", nameof(type)),
        };
    }
}