using AiDotNet.AnomalyDetection;
using AiDotNet.AnomalyDetection.NeuralNetwork;
using AiDotNet.LinearAlgebra;
using Xunit;
using System.Threading.Tasks;

namespace AiDotNet.Tests.IntegrationTests.AnomalyDetection;

/// <summary>
/// Integration tests for neural network anomaly detection classes.
/// Verifies that each detector correctly identifies known outliers.
/// </summary>
public class NeuralNetworkAnomalyDetectionTests
{
    private const int OutlierIndex = 29; // Last row

    private static Matrix<double> CreateTestData()
    {
        int n = 30;
        var data = new double[n, 4];
        for (int i = 0; i < n - 1; i++)
        {
            data[i, 0] = 1.0 + 0.1 * (i % 5);
            data[i, 1] = 2.0 + 0.1 * (i % 7);
            data[i, 2] = 0.5 + 0.05 * (i % 3);
            data[i, 3] = 1.5 + 0.02 * i;
        }

        data[n - 1, 0] = 100.0;
        data[n - 1, 1] = 100.0;
        data[n - 1, 2] = 100.0;
        data[n - 1, 3] = 100.0;

        return new Matrix<double>(data);
    }

    private static void AssertOutlierScoresHighest(Vector<double> scores, int outlierIdx)
    {
        double outlierScore = scores[outlierIdx];
        for (int i = 0; i < scores.Length; i++)
        {
            if (i == outlierIdx) continue;
            Assert.True(outlierScore > scores[i],
                $"Outlier score ({outlierScore:F4}) at index {outlierIdx} should be higher than " +
                $"inlier score ({scores[i]:F4}) at index {i}");
        }
    }

    private static void AssertPredictClassifiesCorrectly(Vector<double> predictions, int outlierIdx)
    {
        Assert.Equal(-1.0, predictions[outlierIdx]);

        int normalCount = 0;
        int inlierCount = 0;
        for (int i = 0; i < predictions.Length; i++)
        {
            if (i == outlierIdx) continue;
            inlierCount++;
            if (predictions[i] == 1.0) normalCount++;
        }

        Assert.True(normalCount >= inlierCount * 0.8,
            $"Expected at least {inlierCount * 0.8} inliers classified as normal, got {normalCount}/{inlierCount}");
    }

    #region AutoencoderDetector Tests

    [Fact(Timeout = 120000)]
    public async Task Autoencoder_Construction_NotFittedByDefault()
    {
        var detector = new AutoencoderDetector<double>();
        Assert.False(detector.IsFitted);
    }

    [Fact(Timeout = 120000)]
    public async Task Autoencoder_OutlierGetsHighestScore()
    {
        var detector = new AutoencoderDetector<double>(epochs: 10);
        var data = CreateTestData();
        detector.Fit(data);
        Assert.True(detector.IsFitted);
        var scores = detector.ScoreAnomalies(data);
        Assert.Equal(data.Rows, scores.Length);
        AssertOutlierScoresHighest(scores, OutlierIndex);
    }

    [Fact(Timeout = 120000)]
    public async Task Autoencoder_PredictClassifiesOutlierAsAnomaly()
    {
        var detector = new AutoencoderDetector<double>(epochs: 10);
        var data = CreateTestData();
        detector.Fit(data);
        var predictions = detector.Predict(data);
        AssertPredictClassifiesCorrectly(predictions, OutlierIndex);
    }

    #endregion

    #region VAEDetector Tests

    [Fact(Timeout = 120000)]
    public async Task VAE_OutlierGetsHighestScore()
    {
        var detector = new VAEDetector<double>(epochs: 10);
        var data = CreateTestData();
        detector.Fit(data);
        var scores = detector.ScoreAnomalies(data);
        Assert.Equal(data.Rows, scores.Length);
        AssertOutlierScoresHighest(scores, OutlierIndex);
    }

    [Fact(Timeout = 120000)]
    public async Task VAE_PredictClassifiesOutlierAsAnomaly()
    {
        var detector = new VAEDetector<double>(epochs: 10);
        var data = CreateTestData();
        detector.Fit(data);
        var predictions = detector.Predict(data);
        AssertPredictClassifiesCorrectly(predictions, OutlierIndex);
    }

    #endregion

    #region DAGMMDetector Tests

    [Fact(Timeout = 120000)]
    public async Task DAGMM_OutlierGetsHighestScore()
    {
        var detector = new DAGMMDetector<double>(epochs: 10);
        var data = CreateTestData();
        detector.Fit(data);
        var scores = detector.ScoreAnomalies(data);
        Assert.Equal(data.Rows, scores.Length);
        AssertOutlierScoresHighest(scores, OutlierIndex);
    }

    [Fact(Timeout = 120000)]
    public async Task DAGMM_PredictClassifiesOutlierAsAnomaly()
    {
        var detector = new DAGMMDetector<double>(epochs: 10);
        var data = CreateTestData();
        detector.Fit(data);
        var predictions = detector.Predict(data);
        AssertPredictClassifiesCorrectly(predictions, OutlierIndex);
    }

    #endregion

    #region DeepSVDDDetector Tests

    [Fact(Timeout = 120000)]
    public async Task DeepSVDD_OutlierGetsHighestScore()
    {
        var detector = new DeepSVDDDetector<double>(epochs: 10);
        var data = CreateTestData();
        detector.Fit(data);
        var scores = detector.ScoreAnomalies(data);
        Assert.Equal(data.Rows, scores.Length);
        AssertOutlierScoresHighest(scores, OutlierIndex);
    }

    [Fact(Timeout = 120000)]
    public async Task DeepSVDD_PredictClassifiesOutlierAsAnomaly()
    {
        var detector = new DeepSVDDDetector<double>(epochs: 10);
        var data = CreateTestData();
        detector.Fit(data);
        var predictions = detector.Predict(data);
        AssertPredictClassifiesCorrectly(predictions, OutlierIndex);
    }

    #endregion

    #region DevNetDetector Tests

    [Fact(Timeout = 120000)]
    public async Task DevNet_OutlierGetsHighestScore()
    {
        var detector = new DevNetDetector<double>(epochs: 10);
        var data = CreateTestData();
        detector.Fit(data);
        var scores = detector.ScoreAnomalies(data);
        Assert.Equal(data.Rows, scores.Length);
        AssertOutlierScoresHighest(scores, OutlierIndex);
    }

    [Fact(Timeout = 120000)]
    public async Task DevNet_PredictClassifiesOutlierAsAnomaly()
    {
        var detector = new DevNetDetector<double>(epochs: 10);
        var data = CreateTestData();
        detector.Fit(data);
        var predictions = detector.Predict(data);
        AssertPredictClassifiesCorrectly(predictions, OutlierIndex);
    }

    #endregion

    #region GANomalyDetector Tests

    [Fact(Timeout = 120000)]
    public async Task GAnomaly_OutlierGetsHighestScore()
    {
        var detector = new GANomalyDetector<double>(epochs: 10);
        var data = CreateTestData();
        detector.Fit(data);
        var scores = detector.ScoreAnomalies(data);
        Assert.Equal(data.Rows, scores.Length);
        AssertOutlierScoresHighest(scores, OutlierIndex);
    }

    [Fact(Timeout = 120000)]
    public async Task GAnomaly_PredictClassifiesOutlierAsAnomaly()
    {
        var detector = new GANomalyDetector<double>(epochs: 10);
        var data = CreateTestData();
        detector.Fit(data);
        var predictions = detector.Predict(data);
        AssertPredictClassifiesCorrectly(predictions, OutlierIndex);
    }

    private static AiDotNet.Tensors.LinearAlgebra.Tensor<double> GanTensor(int[] shape, params double[] values)
        => new AiDotNet.Tensors.LinearAlgebra.Tensor<double>(values, shape);

    [Fact(Timeout = 120000)]
    public async Task GAnomaly_LossesMatchAkcayEquationFourByHand()
    {
        // Encoder, decoder and re-encoder weights zero except the encoder output bias (1, 0), so z = (1, 0),
        // x' = 0 and z' = 0 for every row. Discriminator W1 = I, W2 = (1, 0)^T, biases 0. For x = [[1, -2],
        // [3, 0]]: f(x) = LeakyReLU(x) = [[1, -.4], [3, 0]], f(x') = 0, so L_adv = (1 + .16 + 9 + 0) / 4 = 2.54,
        // L_con = mean|x| = 1.5, L_enc = (1 + 0 + 1 + 0) / 4 = .5 and L_G = 1 * 2.54 + 50 * 1.5 + 1 * .5 = 78.04.
        // D logits: real (1, 3), fake 0, so L_D = (mean(ln(1 + e^-1), ln(1 + e^-3)) + ln 2) / 2.
        var detector = new GANomalyDetector<double>(latentDim: 2, hiddenDim: 2, epochs: 1);
        detector.Fit(new Matrix<double>(new double[,] { { 0, 1 }, { 1, 0 }, { 2, 3 }, { 3, 2 } }));

        var square = new[] { 2, 2 };
        var pair = new[] { 2 };
        var zeros = GanTensor(square, 0, 0, 0, 0);
        var noBias = GanTensor(pair, 0, 0);
        detector.SetTrainingParameters(new[]
        {
            zeros, noBias, zeros, GanTensor(pair, 1, 0),
            zeros, noBias, zeros, noBias,
            zeros, noBias, zeros, noBias,
            GanTensor(square, 1, 0, 0, 1), noBias, GanTensor(new[] { 2, 1 }, 1, 0), GanTensor(new[] { 1 }, 0)
        });

        var x = GanTensor(square, 1, -2, 3, 0);
        Assert.Equal(78.04, detector.GeneratorLoss(x), 1e-12);
        double expectedD = ((Math.Log(1 + Math.Exp(-1)) + Math.Log(1 + Math.Exp(-3))) / 2 + Math.Log(2)) / 2;
        Assert.Equal(expectedD, detector.DiscriminatorLoss(x), 1e-12);
    }

    [Fact(Timeout = 120000)]
    public async Task GAnomaly_GradientsMatchFiniteDifferencesOfTheLosses()
    {
        // The generator gradients (including the adversarial feature-matching term through the discriminator)
        // and the discriminator gradients agree with central differences of GeneratorLoss / DiscriminatorLoss.
        var detector = new GANomalyDetector<double>(latentDim: 2, hiddenDim: 3, epochs: 2);
        var data = new Matrix<double>(6, 2);
        for (int i = 0; i < 6; i++)
        {
            data[i, 0] = Math.Sin(i + 1);
            data[i, 1] = Math.Cos(2 * i + 0.5);
        }

        detector.Fit(data);
        var x = AiDotNet.Tensors.LinearAlgebra.Tensor<double>.FromMatrix(data);
        var parameters = detector.TrainingParameters();
        var gradients = detector.TrainingGradients(x);
        const double h = 1e-6;
        for (int index = 0; index < parameters.Length; index++)
        {
            Func<AiDotNet.Tensors.LinearAlgebra.Tensor<double>, double> loss = index < 12
                ? detector.GeneratorLoss
                : detector.DiscriminatorLoss;
            var values = parameters[index].ToArray();
            var analytic = gradients[index].ToArray();
            for (int element = 0; element < values.Length; element++)
            {
                double up = LossWith(detector, parameters, index, element, h, loss, x);
                double down = LossWith(detector, parameters, index, element, -h, loss, x);
                Assert.Equal((up - down) / (2 * h), analytic[element], 1e-5);
            }
        }
    }

    private static double LossWith(GANomalyDetector<double> detector,
        AiDotNet.Tensors.LinearAlgebra.Tensor<double>[] parameters, int index, int element, double delta,
        Func<AiDotNet.Tensors.LinearAlgebra.Tensor<double>, double> loss,
        AiDotNet.Tensors.LinearAlgebra.Tensor<double> x)
    {
        var values = (double[])parameters[index].ToArray().Clone();
        values[element] += delta;
        var perturbed = (AiDotNet.Tensors.LinearAlgebra.Tensor<double>[])parameters.Clone();
        perturbed[index] = new AiDotNet.Tensors.LinearAlgebra.Tensor<double>(values, parameters[index].Shape.ToArray());
        detector.SetTrainingParameters(perturbed);
        double result = loss(x);
        detector.SetTrainingParameters(parameters);
        return result;
    }

    #endregion

    #region AnoGANDetector Tests

    [Fact(Timeout = 120000)]
    public async Task AnoGAN_OutlierGetsHighestScore()
    {
        var detector = new AnoGANDetector<double>(epochs: 10);
        var data = CreateTestData();
        detector.Fit(data);
        var scores = detector.ScoreAnomalies(data);
        Assert.Equal(data.Rows, scores.Length);
        AssertOutlierScoresHighest(scores, OutlierIndex);
    }

    [Fact(Timeout = 120000)]
    public async Task AnoGAN_PredictClassifiesOutlierAsAnomaly()
    {
        var detector = new AnoGANDetector<double>(epochs: 10);
        var data = CreateTestData();
        detector.Fit(data);
        var predictions = detector.Predict(data);
        AssertPredictClassifiesCorrectly(predictions, OutlierIndex);
    }

    #endregion

    #region Cross-Detector Tests

    [Fact(Timeout = 120000)]
    public async Task AllNeuralNetworkDetectors_PredictBeforeFit_Throws()
    {
        var detectors = new AnomalyDetectorBase<double>[]
        {
            new AutoencoderDetector<double>(),
            new VAEDetector<double>(),
            new DAGMMDetector<double>(),
            new DeepSVDDDetector<double>(),
            new DevNetDetector<double>(),
            new GANomalyDetector<double>(),
            new AnoGANDetector<double>(),
        };

        var data = CreateTestData();
        foreach (var detector in detectors)
        {
            Assert.Throws<InvalidOperationException>(() => detector.Predict(data));
        }
    }

    #endregion
}
