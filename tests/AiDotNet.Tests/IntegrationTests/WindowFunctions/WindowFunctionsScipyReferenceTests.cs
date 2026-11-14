using AiDotNet.Interfaces;
using AiDotNet.WindowFunctions;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.WindowFunctions;

/// <summary>
/// Reference-value tests: every window is compared sample by sample with scipy.signal.windows (scipy 1.18,
/// sym=True). The arrays below were produced by scipy, not by this library.
/// </summary>
public class WindowFunctionsScipyReferenceTests
{
    private const double Tolerance = 1e-9;

    private static readonly double[] Hamming8 = { 0.08000000000000007, 0.25319469114498266, 0.6423596296199047, 0.9544456792351128, 0.9544456792351128, 0.6423596296199047, 0.25319469114498266, 0.08000000000000007 };
    private static readonly double[] Hann8 = { 0.0, 0.18825509907063326, 0.6112604669781572, 0.9504844339512095, 0.9504844339512095, 0.6112604669781572, 0.18825509907063326, 0.0 };
    private static readonly double[] Blackman8 = { -1.3877787807814457e-17, 0.09045342435412808, 0.45918295754596367, 0.9203636180999082, 0.9203636180999082, 0.45918295754596367, 0.09045342435412808, -1.3877787807814457e-17 };
    private static readonly double[] BlackmanHarris8 = { 6.0000000000001025e-05, 0.0333917234781512, 0.33283350429856506, 0.8893697722232838, 0.8893697722232838, 0.33283350429856506, 0.0333917234781512, 6.0000000000001025e-05 };
    private static readonly double[] ScipyNuttall8 = { 0.0003628000000000381, 0.03777576895352028, 0.34272761996881956, 0.8918518610776603, 0.8918518610776603, 0.34272761996881956, 0.03777576895352028, 0.0003628000000000381 };
    private static readonly double[] FlatTop8 = { -0.0004210510000000013, -0.03684078115492349, 0.010703716716153475, 0.78087391493877, 0.78087391493877, 0.010703716716153475, -0.03684078115492349, -0.0004210510000000013 };
    private static readonly double[] Bartlett8 = { 0.0, 0.2857142857142857, 0.5714285714285714, 0.8571428571428571, 0.8571428571428572, 0.5714285714285714, 0.2857142857142858, 0.0 };
    private static readonly double[] BartHann8 = { 0.0, 0.21164530386510985, 0.6017008120462566, 0.9280824555172049, 0.9280824555172051, 0.6017008120462566, 0.21164530386511002, 0.0 };
    private static readonly double[] Bohman8 = { 0.0, 0.07072474681109339, 0.43748401216760524, 0.9103685132461523, 0.9103685132461525, 0.4374840121676056, 0.07072474681109348, 0.0 };
    private static readonly double[] Kaiser8Beta5 = { 0.036710892271286676, 0.2706944178894165, 0.6517382352453626, 0.9552473164564366, 0.9552473164564366, 0.6517382352453626, 0.2706944178894165, 0.036710892271286676 };
    private static readonly double[] Kaiser7Beta5 = { 0.036710892271286676, 0.3282019573723212, 0.7753221044454067, 1.0, 0.7753221044454067, 0.3282019573723212, 0.036710892271286676 };
    private static readonly double[] Lanczos8 = { 3.8981718325193755e-17, 0.3484105662790243, 0.7241014497826596, 0.9667663853085522, 0.9667663853085522, 0.7241014497826596, 0.3484105662790243, 3.8981718325193755e-17 };
    private static readonly double[] Parzen8 = { 0.00390625, 0.10546875, 0.47265625, 0.91796875, 0.91796875, 0.47265625, 0.10546875, 0.00390625 };
    private static readonly double[] Parzen9 = { 0.00274348422496571, 0.0740740740740741, 0.34156378600823034, 0.7695473251028806, 1.0, 0.7695473251028806, 0.34156378600823034, 0.0740740740740741, 0.00274348422496571 };
    private static readonly double[] Tukey9Alpha05 = { 0.0, 0.5, 1.0, 1.0, 1.0, 1.0, 1.0, 0.5, 0.0 };
    private static readonly double[] Tukey8Alpha15 = { 0.0, 0.18825509907063326, 0.6112604669781572, 0.9504844339512095, 0.9504844339512095, 0.6112604669781572, 0.18825509907063326, 0.0 };
    private static readonly double[] NuttallCfd8 = { -2.42861286636753e-17, 0.031142736797915634, 0.32641680590864264, 0.8876284572934416, 0.8876284572934416, 0.32641680590864264, 0.031142736797915634, -2.42861286636753e-17 };
    private static readonly double[] Gaussian8Sigma05 = { 0.1353352832366127, 0.360447788597821, 0.6925693242051977, 0.9600054412854777, 0.9600054412854777, 0.6925693242051977, 0.360447788597821, 0.1353352832366127 };
    private static readonly double[] Poisson8Alpha2 = { 0.1353352832366127, 0.2396510364417758, 0.42437284567695, 0.751477293075286, 0.751477293075286, 0.42437284567695, 0.2396510364417758, 0.1353352832366127 };

    private static void AssertMatches(double[] expected, IWindowFunction<double> window)
    {
        var actual = window.Create(expected.Length);
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++)
        {
            Assert.True(System.Math.Abs(expected[i] - actual[i]) <= Tolerance,
                $"{window.GetType().Name}[{i}] = {actual[i]}, scipy = {expected[i]}");
        }
    }

    [Fact]
    public void Hamming_MatchesScipy() => AssertMatches(Hamming8, new HammingWindow<double>());

    [Fact]
    public void Hann_MatchesScipy() => AssertMatches(Hann8, new HanningWindow<double>());

    [Fact]
    public void Blackman_MatchesScipy() => AssertMatches(Blackman8, new BlackmanWindow<double>());

    [Fact]
    public void BlackmanHarris_MatchesScipy() => AssertMatches(BlackmanHarris8, new BlackmanHarrisWindow<double>());

    [Fact]
    public void BlackmanNuttall_MatchesScipyNuttall() => AssertMatches(ScipyNuttall8, new BlackmanNuttallWindow<double>());

    [Fact]
    public void FlatTop_MatchesScipy() => AssertMatches(FlatTop8, new FlatTopWindow<double>());

    [Fact]
    public void Bartlett_MatchesScipy() => AssertMatches(Bartlett8, new BartlettWindow<double>());

    /// <summary>Master subtracted the 0.38 cos term; scipy (and Ha and Pann 1989) add it, giving zero end points.</summary>
    [Fact]
    public void BartlettHann_MatchesScipy() => AssertMatches(BartHann8, new BartlettHannWindow<double>());

    [Fact]
    public void Bohman_MatchesScipy() => AssertMatches(Bohman8, new BohmanWindow<double>());

    /// <summary>scipy divides by I0(beta); master divided by the sample maximum, which is wrong for even lengths.</summary>
    [Fact]
    public void Kaiser_EvenLength_MatchesScipy() => AssertMatches(Kaiser8Beta5, new KaiserWindow<double>(5.0));

    [Fact]
    public void Kaiser_OddLength_MatchesScipy() => AssertMatches(Kaiser7Beta5, new KaiserWindow<double>(5.0));

    [Fact]
    public void Lanczos_MatchesScipy() => AssertMatches(Lanczos8, new LanczosWindow<double>());

    /// <summary>scipy's Parzen uses |n| / (M/2); master used (M-1)/2, which forced zero end points.</summary>
    [Fact]
    public void Parzen_EvenLength_MatchesScipy() => AssertMatches(Parzen8, new ParzenWindow<double>());

    [Fact]
    public void Parzen_OddLength_MatchesScipy() => AssertMatches(Parzen9, new ParzenWindow<double>());

    [Fact]
    public void Tukey_MatchesScipy() => AssertMatches(Tukey9Alpha05, new TukeyWindow<double>(0.5));

    /// <summary>scipy returns the Hann window for alpha &gt;= 1; master extrapolated the taper formulas.</summary>
    [Fact]
    public void Tukey_AlphaAboveOne_IsHann() => AssertMatches(Tukey8Alpha15, new TukeyWindow<double>(1.5));

    /// <summary>
    /// Nuttall's continuous-first-derivative window: a0 - a1 cos + a2 cos - a3 cos
    /// (scipy.signal.windows.general_cosine). Master flipped the signs of the last three terms.
    /// </summary>
    [Fact]
    public void Nuttall_MatchesScipyGeneralCosine() => AssertMatches(NuttallCfd8, new NuttallWindow<double>());

    /// <summary>sigma is relative to the half width: scipy std = sigma * (M-1)/2.</summary>
    [Fact]
    public void Gaussian_MatchesScipy() => AssertMatches(Gaussian8Sigma05, new GaussianWindow<double>(0.5));

    /// <summary>alpha is relative to the half width: scipy exponential tau = ((M-1)/2) / alpha.</summary>
    [Fact]
    public void Poisson_MatchesScipyExponential() => AssertMatches(Poisson8Alpha2, new PoissonWindow<double>(2.0));

    /// <summary>scipy returns [1] for M = 1; master divided by M-1 = 0 and returned NaN for these windows.</summary>
    [Fact]
    public void LengthOne_IsSingleOne_ForEveryWindow()
    {
        var windows = new IWindowFunction<double>[]
        {
            new BartlettWindow<double>(), new BartlettHannWindow<double>(), new BlackmanWindow<double>(),
            new BlackmanHarrisWindow<double>(), new BlackmanNuttallWindow<double>(), new BohmanWindow<double>(),
            new CosineWindow<double>(), new FlatTopWindow<double>(), new GaussianWindow<double>(),
            new HammingWindow<double>(), new HanningWindow<double>(), new KaiserWindow<double>(),
            new LanczosWindow<double>(), new NuttallWindow<double>(), new ParzenWindow<double>(),
            new PoissonWindow<double>(), new TriangularWindow<double>(), new TukeyWindow<double>(),
            new WelchWindow<double>(),
        };

        foreach (var window in windows)
        {
            var result = window.Create(1);
            Assert.Equal(1, result.Length);
            Assert.True(result[0] == 1.0, $"{window.GetType().Name} size 1 = {result[0]}, scipy = 1");
            Assert.Equal(0, window.Create(0).Length);
        }
    }

    [Fact]
    public void Float_MatchesDouble()
    {
        var single = new HanningWindow<float>().Create(8);
        for (int i = 0; i < Hann8.Length; i++)
        {
            Assert.True(System.Math.Abs(Hann8[i] - single[i]) <= 1e-6, $"Hann<float>[{i}] = {single[i]}");
        }
    }
}
