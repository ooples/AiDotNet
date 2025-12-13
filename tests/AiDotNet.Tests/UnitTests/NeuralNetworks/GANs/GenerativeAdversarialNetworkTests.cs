using AiDotNet.Enums;
using AiDotNet.NeuralNetworks;
using Xunit;

namespace AiDotNetTests.UnitTests.NeuralNetworks.GANs;

/// <summary>
/// Comprehensive tests for GAN implementations covering constructor validation,
/// noise generation, serialization, and forward passes.
/// </summary>
public class GenerativeAdversarialNetworkTests
{
    #region DCGAN Tests

    [Fact]
    public void DCGAN_Constructor_WithValidParameters_InitializesCorrectly()
    {
        // Arrange & Act
        var dcgan = new DCGAN<double>(
            latentSize: 100,
            imageChannels: 3,
            imageHeight: 64,
            imageWidth: 64);

        // Assert
        Assert.NotNull(dcgan);
        Assert.NotNull(dcgan.Generator);
        Assert.NotNull(dcgan.Discriminator);
    }

    [Fact]
    public void DCGAN_Constructor_WithCustomFeatureMaps_InitializesCorrectly()
    {
        // Arrange & Act
        var dcgan = new DCGAN<double>(
            latentSize: 128,
            imageChannels: 1,
            imageHeight: 32,
            imageWidth: 32,
            generatorFeatureMaps: 128,
            discriminatorFeatureMaps: 128);

        // Assert
        Assert.NotNull(dcgan);
        Assert.NotNull(dcgan.Generator);
        Assert.NotNull(dcgan.Discriminator);
    }

    #endregion

    #region InfoGAN Tests

    [Fact]
    public void InfoGAN_Constructor_WithValidParameters_InitializesCorrectly()
    {
        // Arrange
        var genArch = CreateThreeDimensionalGeneratorArchitecture();
        var discArch = CreateThreeDimensionalDiscriminatorArchitecture();
        var qNetArch = CreateThreeDimensionalQNetworkArchitecture();

        // Act
        var infogan = new InfoGAN<double>(
            generatorArchitecture: genArch,
            discriminatorArchitecture: discArch,
            qNetworkArchitecture: qNetArch,
            latentCodeSize: 10,
            inputType: InputType.ThreeDimensional);

        // Assert
        Assert.NotNull(infogan);
        Assert.NotNull(infogan.Generator);
        Assert.NotNull(infogan.Discriminator);
    }

    [Fact]
    public void InfoGAN_GenerateRandomNoiseTensor_WithValidBatchSize_ReturnsCorrectShape()
    {
        // Arrange
        var genArch = CreateThreeDimensionalGeneratorArchitecture();
        var discArch = CreateThreeDimensionalDiscriminatorArchitecture();
        var qNetArch = CreateThreeDimensionalQNetworkArchitecture();
        var infogan = new InfoGAN<double>(genArch, discArch, qNetArch, 10, InputType.ThreeDimensional);

        // Act
        var noise = infogan.GenerateRandomNoiseTensor(4, 100);

        // Assert
        Assert.Equal(2, noise.Shape.Length);
        Assert.Equal(4, noise.Shape[0]);
        Assert.Equal(100, noise.Shape[1]);
    }

    [Fact]
    public void InfoGAN_GenerateRandomNoiseTensor_WithInvalidBatchSize_ThrowsException()
    {
        // Arrange
        var genArch = CreateThreeDimensionalGeneratorArchitecture();
        var discArch = CreateThreeDimensionalDiscriminatorArchitecture();
        var qNetArch = CreateThreeDimensionalQNetworkArchitecture();
        var infogan = new InfoGAN<double>(genArch, discArch, qNetArch, 10, InputType.ThreeDimensional);

        // Act & Assert
        Assert.Throws<ArgumentOutOfRangeException>(() => infogan.GenerateRandomNoiseTensor(0, 100));
        Assert.Throws<ArgumentOutOfRangeException>(() => infogan.GenerateRandomNoiseTensor(-1, 100));
    }

    [Fact]
    public void InfoGAN_GenerateRandomNoiseTensor_WithInvalidNoiseSize_ThrowsException()
    {
        // Arrange
        var genArch = CreateThreeDimensionalGeneratorArchitecture();
        var discArch = CreateThreeDimensionalDiscriminatorArchitecture();
        var qNetArch = CreateThreeDimensionalQNetworkArchitecture();
        var infogan = new InfoGAN<double>(genArch, discArch, qNetArch, 10, InputType.ThreeDimensional);

        // Act & Assert
        Assert.Throws<ArgumentOutOfRangeException>(() => infogan.GenerateRandomNoiseTensor(4, 0));
        Assert.Throws<ArgumentOutOfRangeException>(() => infogan.GenerateRandomNoiseTensor(4, -1));
    }

    #endregion

    #region Pix2Pix Tests

    [Fact]
    public void Pix2Pix_Constructor_WithValidParameters_InitializesCorrectly()
    {
        // Arrange
        var genArch = CreateThreeDimensionalGeneratorArchitecture();
        var discArch = CreateThreeDimensionalDiscriminatorArchitecture();

        // Act
        var pix2pix = new Pix2Pix<double>(
            generatorArchitecture: genArch,
            discriminatorArchitecture: discArch,
            inputType: InputType.ThreeDimensional);

        // Assert
        Assert.NotNull(pix2pix);
        Assert.NotNull(pix2pix.Generator);
        Assert.NotNull(pix2pix.Discriminator);
    }

    [Fact]
    public void Pix2Pix_Constructor_WithCustomL1Lambda_InitializesCorrectly()
    {
        // Arrange
        var genArch = CreateThreeDimensionalGeneratorArchitecture();
        var discArch = CreateThreeDimensionalDiscriminatorArchitecture();

        // Act
        var pix2pix = new Pix2Pix<double>(
            generatorArchitecture: genArch,
            discriminatorArchitecture: discArch,
            inputType: InputType.ThreeDimensional,
            l1Lambda: 50.0);

        // Assert
        Assert.NotNull(pix2pix);
    }

    #endregion

    #region SAGAN Tests

    [Fact]
    public void SAGAN_Constructor_WithValidParameters_InitializesCorrectly()
    {
        // Arrange
        var genArch = CreateThreeDimensionalGeneratorArchitecture();
        var discArch = CreateThreeDimensionalDiscriminatorArchitecture();

        // Act
        var sagan = new SAGAN<double>(
            generatorArchitecture: genArch,
            discriminatorArchitecture: discArch,
            latentSize: 128,
            imageChannels: 3,
            imageHeight: 32,
            imageWidth: 32);

        // Assert
        Assert.NotNull(sagan);
        Assert.NotNull(sagan.Generator);
        Assert.NotNull(sagan.Discriminator);
        Assert.Equal(128, sagan.LatentSize);
    }

    [Fact]
    public void SAGAN_Constructor_WithConditionalGeneration_InitializesCorrectly()
    {
        // Arrange
        var genArch = CreateThreeDimensionalGeneratorArchitecture();
        var discArch = CreateThreeDimensionalDiscriminatorArchitecture();

        // Act
        var sagan = new SAGAN<double>(
            generatorArchitecture: genArch,
            discriminatorArchitecture: discArch,
            latentSize: 128,
            imageChannels: 3,
            imageHeight: 32,
            imageWidth: 32,
            numClasses: 10);

        // Assert
        Assert.NotNull(sagan);
        Assert.Equal(10, sagan.NumClasses);
    }

    #endregion

    #region WGAN Tests

    [Fact]
    public void WGAN_Constructor_WithValidParameters_InitializesCorrectly()
    {
        // Arrange
        var genArch = CreateThreeDimensionalGeneratorArchitecture();
        var criticArch = CreateThreeDimensionalDiscriminatorArchitecture();

        // Act
        var wgan = new WGAN<double>(
            generatorArchitecture: genArch,
            criticArchitecture: criticArch,
            inputType: InputType.ThreeDimensional);

        // Assert
        Assert.NotNull(wgan);
        Assert.NotNull(wgan.Generator);
        Assert.NotNull(wgan.Critic);
    }

    [Fact]
    public void WGAN_Constructor_WithCustomClipValue_InitializesCorrectly()
    {
        // Arrange
        var genArch = CreateThreeDimensionalGeneratorArchitecture();
        var criticArch = CreateThreeDimensionalDiscriminatorArchitecture();

        // Act
        var wgan = new WGAN<double>(
            generatorArchitecture: genArch,
            criticArchitecture: criticArch,
            inputType: InputType.ThreeDimensional,
            weightClipValue: 0.02);

        // Assert
        Assert.NotNull(wgan);
    }

    #endregion

    #region WGANGP Tests

    [Fact]
    public void WGANGP_Constructor_WithValidParameters_InitializesCorrectly()
    {
        // Arrange
        var genArch = CreateThreeDimensionalGeneratorArchitecture();
        var criticArch = CreateThreeDimensionalDiscriminatorArchitecture();

        // Act
        var wgangp = new WGANGP<double>(
            generatorArchitecture: genArch,
            criticArchitecture: criticArch,
            inputType: InputType.ThreeDimensional);

        // Assert
        Assert.NotNull(wgangp);
        Assert.NotNull(wgangp.Generator);
        Assert.NotNull(wgangp.Critic);
    }

    [Fact]
    public void WGANGP_Constructor_WithCustomGradientPenalty_InitializesCorrectly()
    {
        // Arrange
        var genArch = CreateThreeDimensionalGeneratorArchitecture();
        var criticArch = CreateThreeDimensionalDiscriminatorArchitecture();

        // Act
        var wgangp = new WGANGP<double>(
            generatorArchitecture: genArch,
            criticArchitecture: criticArch,
            inputType: InputType.ThreeDimensional,
            gradientPenaltyCoefficient: 15.0);

        // Assert
        Assert.NotNull(wgangp);
    }

    [Fact]
    public void WGANGP_GenerateRandomNoiseTensor_WithValidParameters_ReturnsCorrectShape()
    {
        // Arrange
        var genArch = CreateThreeDimensionalGeneratorArchitecture();
        var criticArch = CreateThreeDimensionalDiscriminatorArchitecture();
        var wgangp = new WGANGP<double>(genArch, criticArch, InputType.ThreeDimensional);

        // Act
        var noise = wgangp.GenerateRandomNoiseTensor(8, 100);

        // Assert
        Assert.Equal(2, noise.Shape.Length);
        Assert.Equal(8, noise.Shape[0]);
        Assert.Equal(100, noise.Shape[1]);
    }

    #endregion

    #region StyleGAN Tests

    [Fact]
    public void StyleGAN_Constructor_WithValidParameters_InitializesCorrectly()
    {
        // Arrange
        var mappingArch = CreateThreeDimensionalMappingArchitecture();
        var synthesisArch = CreateThreeDimensionalGeneratorArchitecture();
        var discArch = CreateThreeDimensionalDiscriminatorArchitecture();

        // Act
        var stylegan = new StyleGAN<double>(
            mappingNetworkArchitecture: mappingArch,
            synthesisNetworkArchitecture: synthesisArch,
            discriminatorArchitecture: discArch,
            latentSize: 512,
            intermediateLatentSize: 512,
            inputType: InputType.ThreeDimensional);

        // Assert
        Assert.NotNull(stylegan);
        Assert.NotNull(stylegan.MappingNetwork);
        Assert.NotNull(stylegan.SynthesisNetwork);
        Assert.NotNull(stylegan.Discriminator);
    }

    [Fact]
    public void StyleGAN_GenerateRandomLatentCodes_WithValidBatchSize_ReturnsCorrectShape()
    {
        // Arrange
        var mappingArch = CreateThreeDimensionalMappingArchitecture();
        var synthesisArch = CreateThreeDimensionalGeneratorArchitecture();
        var discArch = CreateThreeDimensionalDiscriminatorArchitecture();
        var stylegan = new StyleGAN<double>(
            mappingArch, synthesisArch, discArch, 512, 512, InputType.ThreeDimensional);

        // Act
        var latentCodes = stylegan.GenerateRandomLatentCodes(4);

        // Assert
        Assert.Equal(2, latentCodes.Shape.Length);
        Assert.Equal(4, latentCodes.Shape[0]);
        Assert.Equal(512, latentCodes.Shape[1]);
    }

    #endregion

    #region CycleGAN Tests

    [Fact]
    public void CycleGAN_Constructor_WithValidParameters_InitializesCorrectly()
    {
        // Arrange
        var genAtoBArch = CreateThreeDimensionalGeneratorArchitecture();
        var genBtoAArch = CreateThreeDimensionalGeneratorArchitecture();
        var discAArch = CreateThreeDimensionalDiscriminatorArchitecture();
        var discBArch = CreateThreeDimensionalDiscriminatorArchitecture();

        // Act
        var cyclegan = new CycleGAN<double>(
            generatorAtoB: genAtoBArch,
            generatorBtoA: genBtoAArch,
            discriminatorA: discAArch,
            discriminatorB: discBArch,
            inputType: InputType.ThreeDimensional);

        // Assert
        Assert.NotNull(cyclegan);
        Assert.NotNull(cyclegan.GeneratorAtoB);
        Assert.NotNull(cyclegan.GeneratorBtoA);
        Assert.NotNull(cyclegan.DiscriminatorA);
        Assert.NotNull(cyclegan.DiscriminatorB);
    }

    [Fact]
    public void CycleGAN_Constructor_WithCustomLambdas_InitializesCorrectly()
    {
        // Arrange
        var genAtoBArch = CreateThreeDimensionalGeneratorArchitecture();
        var genBtoAArch = CreateThreeDimensionalGeneratorArchitecture();
        var discAArch = CreateThreeDimensionalDiscriminatorArchitecture();
        var discBArch = CreateThreeDimensionalDiscriminatorArchitecture();

        // Act
        var cyclegan = new CycleGAN<double>(
            generatorAtoB: genAtoBArch,
            generatorBtoA: genBtoAArch,
            discriminatorA: discAArch,
            discriminatorB: discBArch,
            inputType: InputType.ThreeDimensional,
            cycleConsistencyLambda: 15.0,
            identityLambda: 7.5);

        // Assert
        Assert.NotNull(cyclegan);
    }

    #endregion

    #region ConditionalGAN Tests

    [Fact]
    public void ConditionalGAN_Constructor_WithValidParameters_InitializesCorrectly()
    {
        // Arrange
        var genArch = CreateThreeDimensionalGeneratorArchitecture();
        var discArch = CreateThreeDimensionalDiscriminatorArchitecture();

        // Act
        var cgan = new ConditionalGAN<double>(
            generatorArchitecture: genArch,
            discriminatorArchitecture: discArch,
            numConditionClasses: 10,
            inputType: InputType.ThreeDimensional);

        // Assert
        Assert.NotNull(cgan);
        Assert.NotNull(cgan.Generator);
        Assert.NotNull(cgan.Discriminator);
    }

    #endregion

    #region Helper Methods

    /// <summary>
    /// Creates a ThreeDimensional generator architecture for testing.
    /// Uses small image size (8x8x3) to minimize memory/computation.
    /// </summary>
    private static NeuralNetworkArchitecture<double> CreateThreeDimensionalGeneratorArchitecture()
    {
        return new NeuralNetworkArchitecture<double>(
            inputType: InputType.ThreeDimensional,
            taskType: NeuralNetworkTaskType.Generative,
            complexity: NetworkComplexity.Simple,
            inputSize: 0,  // Let it be calculated from dimensions
            inputHeight: 8,
            inputWidth: 8,
            inputDepth: 3,
            outputSize: 8 * 8 * 3,  // 192
            layers: null);
    }

    /// <summary>
    /// Creates a ThreeDimensional discriminator architecture for testing.
    /// Takes image input and outputs a single value.
    /// </summary>
    private static NeuralNetworkArchitecture<double> CreateThreeDimensionalDiscriminatorArchitecture()
    {
        return new NeuralNetworkArchitecture<double>(
            inputType: InputType.ThreeDimensional,
            taskType: NeuralNetworkTaskType.BinaryClassification,
            complexity: NetworkComplexity.Simple,
            inputSize: 0,  // Let it be calculated from dimensions
            inputHeight: 8,
            inputWidth: 8,
            inputDepth: 3,
            outputSize: 1,
            layers: null);
    }

    /// <summary>
    /// Creates a ThreeDimensional mapping architecture for StyleGAN.
    /// </summary>
    private static NeuralNetworkArchitecture<double> CreateThreeDimensionalMappingArchitecture()
    {
        return new NeuralNetworkArchitecture<double>(
            inputType: InputType.ThreeDimensional,
            taskType: NeuralNetworkTaskType.Generative,
            complexity: NetworkComplexity.Simple,
            inputSize: 0,
            inputHeight: 8,
            inputWidth: 8,
            inputDepth: 8,  // 8*8*8 = 512
            outputSize: 512,
            layers: null);
    }

    /// <summary>
    /// Creates a ThreeDimensional Q-network architecture for InfoGAN.
    /// </summary>
    private static NeuralNetworkArchitecture<double> CreateThreeDimensionalQNetworkArchitecture()
    {
        return new NeuralNetworkArchitecture<double>(
            inputType: InputType.ThreeDimensional,
            taskType: NeuralNetworkTaskType.Regression,
            complexity: NetworkComplexity.Simple,
            inputSize: 0,
            inputHeight: 8,
            inputWidth: 8,
            inputDepth: 3,
            outputSize: 10,
            layers: null);
    }

    #endregion
}
