using System;
using System.Linq;
using AiDotNet.Classification.TimeSeries;
using AiDotNet.Models.Options;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.Classification;

/// <summary>
/// Reference checks for ROCKET (Dempster et al. 2020) and MiniRocket (Dempster et al. 2021) against the sktime /
/// scikit-learn implementations they are specified by.
/// </summary>
public class RocketReferenceTests
{
    private const double Tolerance = 1e-8;

    [Fact]
    [Trait("category", "unit")]
    public void Ridge_head_matches_scikit_learn_binary()
    {
        // make_pipeline(StandardScaler(with_mean=False), RidgeClassifierCV(alphas=np.logspace(-3, 3, 10))),
        // coefficients mapped back to the raw features (coef_ / scale_). n > p: the covariance route.
        var x = new Matrix<double>(new[,]
        {
            { 1.0, 2.0, 0.5, 3.0 }, { 2.0, 1.0, 1.5, 2.5 }, { 0.5, 0.3, 2.0, 1.0 }, { 3.0, 2.5, 0.1, 0.2 },
            { 1.5, 1.8, 1.1, 2.2 }, { 2.2, 0.4, 2.6, 0.9 }, { 0.9, 2.9, 0.7, 1.6 }
        });
        var y = new Vector<double>(new[] { 0.0, 1, 0, 1, 1, 0, 1 });
        var classes = new Vector<double>(new[] { 0.0, 1.0 });

        var parameters = RidgeClassifierCv<double>.Fit(x, y, classes, RidgeClassifierCv<double>.DefaultAlphas);

        // alpha_ = 46.41588833612773
        double[] expected =
        {
            0.05834184275130455, 0.07570114855018902, -0.07071358796360958, 0.0003449503151781822,
            0.017770870051587034
        };
        Assert.Equal(expected.Length, parameters.Length);
        for (int i = 0; i < expected.Length; i++)
        {
            Assert.Equal(expected[i], parameters[i], Tolerance);
        }
    }

    [Fact]
    [Trait("category", "unit")]
    public void Ridge_head_matches_scikit_learn_multiclass()
    {
        // Three classes, n < p: the Gram route; one +1/-1 column per class.
        var x = new Matrix<double>(new[,]
        {
            { 1.0, 0.2, 0.3, 2.0, 1.1, 0.5, 0.9, 1.4 }, { 0.1, 1.5, 0.7, 0.2, 2.2, 1.0, 0.3, 0.8 },
            { 2.1, 0.4, 1.9, 0.6, 0.3, 1.7, 1.2, 0.2 }, { 0.6, 2.2, 0.1, 1.3, 0.9, 0.4, 2.0, 1.1 },
            { 1.4, 0.9, 1.0, 0.8, 1.6, 2.1, 0.5, 0.7 }
        });
        var y = new Vector<double>(new[] { 2.0, 0, 1, 0, 2 });
        var classes = new Vector<double>(new[] { 0.0, 1.0, 2.0 });

        var parameters = RidgeClassifierCv<double>.Fit(x, y, classes, RidgeClassifierCv<double>.DefaultAlphas);

        // alpha_ = 1000.0
        double[] expected =
        {
            -0.005826915894195859, 0.00594457729664273, -0.003913465462655213, -0.0023913706626579965,
            0.0031358797894707967, -0.003909878479848425, 0.0018824816411817827, 0.0026094285320358147,
            0.004458991998540872, -0.0023311712310544533, 0.00540224487754924, -0.0019341358482475035,
            -0.004388484072613503, 0.0024597452206045647, 0.001231376523556245, -0.007742155004967576,
            0.001367923895654986, -0.0036134060655882783, -0.001488779414894026, 0.0043255065109055,
            0.0012526042831427077, 0.0014501332592438611, -0.0031138581647380294, 0.005132726472931762,
            -0.19805331609041116, -0.5967927741407558, -0.20515390976883308
        };
        Assert.Equal(expected.Length, parameters.Length);
        for (int i = 0; i < expected.Length; i++)
        {
            Assert.Equal(expected[i], parameters[i], Tolerance);
        }
    }

    [Fact]
    [Trait("category", "unit")]
    public void Rocket_features_match_the_reference_kernel_application()
    {
        const int samples = 4, length = 24;
        var sequences = new Tensor<double>(new[] { samples, length });
        for (int s = 0; s < samples; s++)
        {
            for (int t = 0; t < length; t++)
            {
                sequences[s, t] = Math.Sin(0.4 * t + s) + (0.1 * s * t);
            }
        }

        var classifier = new RocketClassifier<double>(new RocketOptions<double>
        {
            NumKernels = 40, KernelLengths = new[] { 3, 5 }, RandomSeed = 7
        });
        classifier.TrainOnSequences(sequences, new Vector<double>(new[] { 0.0, 1, 0, 1 }));
        var kernels = classifier.KernelSpecifications;
        var features = classifier.TransformSequences(sequences);

        // Dilations are floor(2^x), x ~ U(0, log2((L - 1) / (l - 1))); padding is 0 or ((l - 1) * d) / 2.
        Assert.Contains(kernels, k => k.Padding > 0);
        Assert.Contains(kernels, k => k.Padding == 0);
        foreach (var kernel in kernels)
        {
            int l = kernel.Weights.Length;
            Assert.InRange(kernel.Dilation, 1, (length - 1) / (l - 1));
            Assert.True(kernel.Padding == 0 || kernel.Padding == ((l - 1) * kernel.Dilation) / 2);
        }

        for (int s = 0; s < samples; s++)
        {
            for (int k = 0; k < kernels.Count; k++)
            {
                var (max, ppv) = ApplyKernelReference(sequences, s, length, kernels[k]);
                Assert.Equal(max, features[s, 2 * k], Tolerance);
                Assert.Equal(ppv, features[s, (2 * k) + 1], Tolerance);
            }
        }
    }

    [Theory]
    [Trait("category", "unit")]
    [InlineData(50, 9996, new[] { 1, 2, 3, 4, 5, 6 }, new[] { 45, 27, 19, 14, 11, 3 })]
    [InlineData(20, 840, new[] { 1, 2 }, new[] { 8, 2 })]
    public void MiniRocket_dilations_match_the_reference_fit(int length, int numFeatures, int[] dilations, int[] perDilation)
    {
        const int samples = 6;
        var sequences = new Tensor<double>(new[] { samples, length });
        for (int s = 0; s < samples; s++)
        {
            for (int t = 0; t < length; t++)
            {
                sequences[s, t] = Math.Cos(0.3 * t * (s + 1)) + s;
            }
        }

        var classifier = new MiniRocketClassifier<double>(new MiniRocketOptions<double> { Seed = 3, NumFeatures = numFeatures });
        classifier.TrainOnSequences(sequences, new Vector<double>(new[] { 0.0, 1, 0, 1, 0, 1 }));

        var plan = classifier.DilationPlan;
        Assert.Equal(dilations, plan.Dilations);
        Assert.Equal(perDilation, plan.FeaturesPerDilation);
        Assert.Equal(84 * perDilation.Sum(), classifier.NumFeatures);
    }

    // sktime's apply_kernel: zero padding, output = bias + sum(w_j * x[i + j * d]); (max, proportion > 0).
    private static (double Max, double Ppv) ApplyKernelReference(
        Tensor<double> sequences, int sample, int length, (double[] Weights, int Dilation, double Bias, int Padding) kernel)
    {
        int l = kernel.Weights.Length;
        int outputLength = length + (2 * kernel.Padding) - ((l - 1) * kernel.Dilation);
        int end = length + kernel.Padding - ((l - 1) * kernel.Dilation);
        double max = double.NegativeInfinity;
        int positive = 0;
        for (int i = -kernel.Padding; i < end; i++)
        {
            double sum = kernel.Bias;
            int index = i;
            for (int j = 0; j < l; j++)
            {
                if (index > -1 && index < length)
                {
                    sum += kernel.Weights[j] * sequences[sample, index];
                }

                index += kernel.Dilation;
            }

            max = Math.Max(max, sum);
            if (sum > 0)
            {
                positive++;
            }
        }

        return (max, positive / (double)outputLength);
    }
}
