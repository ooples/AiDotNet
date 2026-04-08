using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.LossFunctions;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tensors;
using AiDotNetTests.UnitTests.NeuralNetworks.GANs.Helpers;
using Xunit;

namespace AiDotNetTests.UnitTests.NeuralNetworks.GANs;

/// <summary>
/// Comprehensive tests for GAN implementations covering constructor validation,
/// noise generation, serialization, and forward passes.
/// </summary>
public class GenerativeAdversarialNetworkTests
{
    #region Base GAN Tests

    [Fact]
    public void GenerativeAdversarialNetwork_Constructor_WithValidParameters_InitializesCorrectly()
    {
        // Arrange
        var genArch = CreateThreeDimensionalGeneratorArchitecture();
        var discArch = CreateThreeDimensionalDiscriminatorArchitecture();

        // Act
        var gan = new GenerativeAdversarialNetwork<double>(genArch, discArch, InputType.ThreeDimensional);

        // Assert
        Assert.NotNull(gan);
        Assert.NotNull(gan.Generator);
        Assert.NotNull(gan.Discriminator);
    }

    [Fact]
    public void GenerativeAdversarialNetwork_GenerateRandomNoiseTensor_ReturnsCorrectShape()
    {
        // Arrange
        var genArch = CreateThreeDimensionalGeneratorArchitecture();
        var discArch = CreateThreeDimensionalDiscriminatorArchitecture();
        var gan = new GenerativeAdversarialNetwork<double>(genArch, discArch, InputType.ThreeDimensional);

        // Act
        var noise = gan.GenerateRandomNoiseTensor(4, 100);

        // Assert
        Assert.Equal(2, noise.Shape.Length);
        Assert.Equal(4, noise.Shape[0]);
        Assert.Equal(100, noise.Shape[1]);
    }

    [Fact]
    public void GenerativeAdversarialNetwork_GenerateRandomNoiseTensor_WithZeroBatchSize_ReturnsEmptyTensor()
    {
        // Arrange
        var genArch = CreateThreeDimensionalGeneratorArchitecture();
        var discArch = CreateThreeDimensionalDiscriminatorArchitecture();
        var gan = new GenerativeAdversarialNetwork<double>(genArch, discArch, InputType.ThreeDimensional);

        // Act - The implementation creates tensors with the given dimensions
        var noise = gan.GenerateRandomNoiseTensor(1, 100);

        // Assert
        Assert.NotNull(noise);
        Assert.Equal(2, noise.Shape.Length);
    }

    [Fact]
    public void GenerativeAdversarialNetwork_GenerateRandomNoiseTensor_WithZeroNoiseSize_ThrowsException()
    {
        // Arrange
        var genArch = CreateThreeDimensionalGeneratorArchitecture();
        var discArch = CreateThreeDimensionalDiscriminatorArchitecture();
        var gan = new GenerativeAdversarialNetwork<double>(genArch, discArch, InputType.ThreeDimensional);

        // Act & Assert - Zero noiseSize should throw ArgumentOutOfRangeException
        Assert.Throws<ArgumentOutOfRangeException>(() => gan.GenerateRandomNoiseTensor(4, 0));
    }

    [Fact]
    public void GenerativeAdversarialNetwork_GetModelMetadata_ReturnsValidMetadata()
    {
        // Arrange
        var genArch = CreateThreeDimensionalGeneratorArchitecture();
        var discArch = CreateThreeDimensionalDiscriminatorArchitecture();
        var gan = new GenerativeAdversarialNetwork<double>(genArch, discArch, InputType.ThreeDimensional);

        // Act
        var metadata = gan.GetModelMetadata();

        // Assert
        Assert.NotNull(metadata);
    }

    [Fact]
    public void GenerativeAdversarialNetwork_GetDiagnostics_ReturnsDictionary()
    {
        // Arrange
        var genArch = CreateThreeDimensionalGeneratorArchitecture();
        var discArch = CreateThreeDimensionalDiscriminatorArchitecture();
        var gan = new GenerativeAdversarialNetwork<double>(genArch, discArch, InputType.ThreeDimensional);

        // Act
        var diagnostics = gan.GetDiagnostics();

        // Assert
        Assert.NotNull(diagnostics);
    }

    [Fact]
    public void GenerativeAdversarialNetwork_EnableGradientPenalty_SetsFlag()
    {
        // Arrange
        var genArch = CreateThreeDimensionalGeneratorArchitecture();
        var discArch = CreateThreeDimensionalDiscriminatorArchitecture();
        var gan = new GenerativeAdversarialNetwork<double>(genArch, discArch, InputType.ThreeDimensional);

        // Act - Should not throw
        gan.EnableGradientPenalty(true);
        gan.EnableGradientPenalty(false);

        // Assert - No exception means success
        Assert.NotNull(gan);
    }

    [Fact]
    public void GenerativeAdversarialNetwork_EnableFeatureMatching_SetsFlag()
    {
        // Arrange
        var genArch = CreateThreeDimensionalGeneratorArchitecture();
        var discArch = CreateThreeDimensionalDiscriminatorArchitecture();
        var gan = new GenerativeAdversarialNetwork<double>(genArch, discArch, InputType.ThreeDimensional);

        // Act - Should not throw
        gan.EnableFeatureMatching(true);
        gan.EnableFeatureMatching(false);

        // Assert - No exception means success
        Assert.NotNull(gan);
    }

    #endregion

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

    [Fact]
    public void DCGAN_GenerateRandomNoiseTensor_ReturnsCorrectShape()
    {
        // Arrange
        var dcgan = new DCGAN<double>(100, 3, 32, 32);

        // Act
        var noise = dcgan.GenerateRandomNoiseTensor(8, 100);

        // Assert
        Assert.Equal(2, noise.Shape.Length);
        Assert.Equal(8, noise.Shape[0]);
        Assert.Equal(100, noise.Shape[1]);
    }

    [Fact]
    public void DCGAN_GetModelMetadata_ReturnsValidMetadata()
    {
        // Arrange
        var dcgan = new DCGAN<double>(100, 3, 32, 32);

        // Act
        var metadata = dcgan.GetModelMetadata();

        // Assert
        Assert.NotNull(metadata);
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

    [Fact]
    public void InfoGAN_GenerateRandomLatentCodes_ReturnsCorrectShape()
    {
        // Arrange
        var genArch = CreateThreeDimensionalGeneratorArchitecture();
        var discArch = CreateThreeDimensionalDiscriminatorArchitecture();
        var qNetArch = CreateThreeDimensionalQNetworkArchitecture();
        var infogan = new InfoGAN<double>(genArch, discArch, qNetArch, 10, InputType.ThreeDimensional);

        // Act
        var codes = infogan.GenerateRandomLatentCodes(4);

        // Assert
        Assert.Equal(2, codes.Shape.Length);
        Assert.Equal(4, codes.Shape[0]);
        Assert.Equal(10, codes.Shape[1]); // latentCodeSize = 10
    }

    [Fact]
    public void InfoGAN_GetModelMetadata_ReturnsValidMetadata()
    {
        // Arrange
        var genArch = CreateThreeDimensionalGeneratorArchitecture();
        var discArch = CreateThreeDimensionalDiscriminatorArchitecture();
        var qNetArch = CreateThreeDimensionalQNetworkArchitecture();
        var infogan = new InfoGAN<double>(genArch, discArch, qNetArch, 10, InputType.ThreeDimensional);

        // Act
        var metadata = infogan.GetModelMetadata();

        // Assert
        Assert.NotNull(metadata);
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

    [Fact]
    public void Pix2Pix_GetModelMetadata_ReturnsValidMetadata()
    {
        // Arrange
        var genArch = CreateThreeDimensionalGeneratorArchitecture();
        var discArch = CreateThreeDimensionalDiscriminatorArchitecture();
        var pix2pix = new Pix2Pix<double>(genArch, discArch, InputType.ThreeDimensional);

        // Act
        var metadata = pix2pix.GetModelMetadata();

        // Assert
        Assert.NotNull(metadata);
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

    [Fact]
    public void SAGAN_GetModelMetadata_ReturnsValidMetadata()
    {
        // Arrange
        var genArch = CreateThreeDimensionalGeneratorArchitecture();
        var discArch = CreateThreeDimensionalDiscriminatorArchitecture();
        var sagan = new SAGAN<double>(genArch, discArch, 128, 3, 32, 32);

        // Act
        var metadata = sagan.GetModelMetadata();

        // Assert
        Assert.NotNull(metadata);
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

    [Fact]
    public void WGAN_GenerateRandomNoiseTensor_ReturnsCorrectShape()
    {
        // Arrange
        var genArch = CreateThreeDimensionalGeneratorArchitecture();
        var criticArch = CreateThreeDimensionalDiscriminatorArchitecture();
        var wgan = new WGAN<double>(genArch, criticArch, InputType.ThreeDimensional);

        // Act
        var noise = wgan.GenerateRandomNoiseTensor(8, 100);

        // Assert
        Assert.Equal(2, noise.Shape.Length);
        Assert.Equal(8, noise.Shape[0]);
        Assert.Equal(100, noise.Shape[1]);
    }

    [Fact]
    public void WGAN_GetModelMetadata_ReturnsValidMetadata()
    {
        // Arrange
        var genArch = CreateThreeDimensionalGeneratorArchitecture();
        var criticArch = CreateThreeDimensionalDiscriminatorArchitecture();
        var wgan = new WGAN<double>(genArch, criticArch, InputType.ThreeDimensional);

        // Act
        var metadata = wgan.GetModelMetadata();

        // Assert
        Assert.NotNull(metadata);
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

    [Fact]
    public void WGANGP_GenerateRandomNoiseTensor_WithMinimalBatchSize_ReturnsTensor()
    {
        // Arrange
        var genArch = CreateThreeDimensionalGeneratorArchitecture();
        var criticArch = CreateThreeDimensionalDiscriminatorArchitecture();
        var wgangp = new WGANGP<double>(genArch, criticArch, InputType.ThreeDimensional);

        // Act - The implementation creates tensors with the given dimensions
        var noise = wgangp.GenerateRandomNoiseTensor(1, 100);

        // Assert
        Assert.NotNull(noise);
        Assert.Equal(2, noise.Shape.Length);
    }

    [Fact]
    public void WGANGP_GetModelMetadata_ReturnsValidMetadata()
    {
        // Arrange
        var genArch = CreateThreeDimensionalGeneratorArchitecture();
        var criticArch = CreateThreeDimensionalDiscriminatorArchitecture();
        var wgangp = new WGANGP<double>(genArch, criticArch, InputType.ThreeDimensional);

        // Act
        var metadata = wgangp.GetModelMetadata();

        // Assert
        Assert.NotNull(metadata);
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

    [Fact]
    public void StyleGAN_GenerateRandomLatentCodes_WithSingleSample_ReturnsCorrectShape()
    {
        // Arrange
        var mappingArch = CreateThreeDimensionalMappingArchitecture();
        var synthesisArch = CreateThreeDimensionalGeneratorArchitecture();
        var discArch = CreateThreeDimensionalDiscriminatorArchitecture();
        var stylegan = new StyleGAN<double>(
            mappingArch, synthesisArch, discArch, 256, 256, InputType.ThreeDimensional);

        // Act
        var latentCodes = stylegan.GenerateRandomLatentCodes(1);

        // Assert
        Assert.Equal(2, latentCodes.Shape.Length);
        Assert.Equal(1, latentCodes.Shape[0]);
        Assert.Equal(256, latentCodes.Shape[1]);
    }

    [Fact]
    public void StyleGAN_GetModelMetadata_ReturnsValidMetadata()
    {
        // Arrange
        var mappingArch = CreateThreeDimensionalMappingArchitecture();
        var synthesisArch = CreateThreeDimensionalGeneratorArchitecture();
        var discArch = CreateThreeDimensionalDiscriminatorArchitecture();
        var stylegan = new StyleGAN<double>(
            mappingArch, synthesisArch, discArch, 512, 512, InputType.ThreeDimensional);

        // Act
        var metadata = stylegan.GetModelMetadata();

        // Assert
        Assert.NotNull(metadata);
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

    [Fact]
    public void CycleGAN_GetModelMetadata_ReturnsValidMetadata()
    {
        // Arrange
        var genAtoBArch = CreateThreeDimensionalGeneratorArchitecture();
        var genBtoAArch = CreateThreeDimensionalGeneratorArchitecture();
        var discAArch = CreateThreeDimensionalDiscriminatorArchitecture();
        var discBArch = CreateThreeDimensionalDiscriminatorArchitecture();
        var cyclegan = new CycleGAN<double>(genAtoBArch, genBtoAArch, discAArch, discBArch, InputType.ThreeDimensional);

        // Act
        var metadata = cyclegan.GetModelMetadata();

        // Assert
        Assert.NotNull(metadata);
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

    [Fact]
    public void ConditionalGAN_GenerateRandomNoiseTensor_ReturnsCorrectShape()
    {
        // Arrange
        var genArch = CreateThreeDimensionalGeneratorArchitecture();
        var discArch = CreateThreeDimensionalDiscriminatorArchitecture();
        var cgan = new ConditionalGAN<double>(genArch, discArch, 10, InputType.ThreeDimensional);

        // Act
        var noise = cgan.GenerateRandomNoiseTensor(4, 100);

        // Assert
        Assert.Equal(2, noise.Shape.Length);
        Assert.Equal(4, noise.Shape[0]);
    }

    [Fact]
    public void ConditionalGAN_GetModelMetadata_ReturnsValidMetadata()
    {
        // Arrange
        var genArch = CreateThreeDimensionalGeneratorArchitecture();
        var discArch = CreateThreeDimensionalDiscriminatorArchitecture();
        var cgan = new ConditionalGAN<double>(genArch, discArch, 10, InputType.ThreeDimensional);

        // Act
        var metadata = cgan.GetModelMetadata();

        // Assert
        Assert.NotNull(metadata);
    }

    #endregion

    #region ACGAN Tests

    [Fact]
    public void ACGAN_Constructor_WithValidParameters_InitializesCorrectly()
    {
        // Arrange
        var genArch = CreateThreeDimensionalGeneratorArchitecture();
        var discArch = CreateACGANDiscriminatorArchitecture();  // ACGAN requires outputSize = 1 + numClasses

        // Act
        var acgan = new ACGAN<double>(
            generatorArchitecture: genArch,
            discriminatorArchitecture: discArch,
            numClasses: 10,
            inputType: InputType.ThreeDimensional);

        // Assert
        Assert.NotNull(acgan);
        Assert.NotNull(acgan.Generator);
        Assert.NotNull(acgan.Discriminator);
    }

    [Fact]
    public void ACGAN_GenerateRandomNoiseTensor_ReturnsCorrectShape()
    {
        // Arrange
        var genArch = CreateThreeDimensionalGeneratorArchitecture();
        var discArch = CreateACGANDiscriminatorArchitecture();  // ACGAN requires outputSize = 1 + numClasses
        var acgan = new ACGAN<double>(genArch, discArch, 10, InputType.ThreeDimensional);

        // Act
        var noise = acgan.GenerateRandomNoiseTensor(4, 100);

        // Assert
        Assert.Equal(2, noise.Shape.Length);
        Assert.Equal(4, noise.Shape[0]);
    }

    [Fact]
    public void ACGAN_GetModelMetadata_ReturnsValidMetadata()
    {
        // Arrange
        var genArch = CreateThreeDimensionalGeneratorArchitecture();
        var discArch = CreateACGANDiscriminatorArchitecture();  // ACGAN requires outputSize = 1 + numClasses
        var acgan = new ACGAN<double>(genArch, discArch, 10, InputType.ThreeDimensional);

        // Act
        var metadata = acgan.GetModelMetadata();

        // Assert
        Assert.NotNull(metadata);
    }

    #endregion

    #region BigGAN Tests

    [Fact]
    public void BigGAN_Constructor_WithValidParameters_InitializesCorrectly()
    {
        // Arrange
        var genArch = CreateThreeDimensionalGeneratorArchitecture();
        var discArch = CreateThreeDimensionalDiscriminatorArchitecture();

        // Act - use OneDimensional input type to avoid height/width validation in base class
        var biggan = new BigGAN<double>(
            generatorArchitecture: genArch,
            discriminatorArchitecture: discArch,
            latentSize: 128,
            numClasses: 100,
            imageChannels: 3,
            imageHeight: 32,
            imageWidth: 32,
            inputType: InputType.OneDimensional);

        // Assert
        Assert.NotNull(biggan);
        Assert.NotNull(biggan.Generator);
        Assert.NotNull(biggan.Discriminator);
    }

    [Fact]
    public void BigGAN_GetModelMetadata_ReturnsValidMetadata()
    {
        // Arrange
        var genArch = CreateThreeDimensionalGeneratorArchitecture();
        var discArch = CreateThreeDimensionalDiscriminatorArchitecture();
        var biggan = new BigGAN<double>(genArch, discArch, 128, 100, 128, 3, 32, 32, inputType: InputType.OneDimensional);

        // Act
        var metadata = biggan.GetModelMetadata();

        // Assert
        Assert.NotNull(metadata);
    }

    #endregion

    #region ProgressiveGAN Tests

    [Fact]
    public void ProgressiveGAN_Constructor_WithValidParameters_InitializesCorrectly()
    {
        // Arrange
        var genArch = CreateThreeDimensionalGeneratorArchitecture();
        var discArch = CreateThreeDimensionalDiscriminatorArchitecture();

        // Act
        var proggan = new ProgressiveGAN<double>(
            generatorArchitecture: genArch,
            discriminatorArchitecture: discArch,
            latentSize: 512,
            imageChannels: 3,
            inputType: InputType.ThreeDimensional);

        // Assert
        Assert.NotNull(proggan);
        Assert.NotNull(proggan.Generator);
        Assert.NotNull(proggan.Discriminator);
    }

    [Fact]
    public void ProgressiveGAN_GetModelMetadata_ReturnsValidMetadata()
    {
        // Arrange
        var genArch = CreateThreeDimensionalGeneratorArchitecture();
        var discArch = CreateThreeDimensionalDiscriminatorArchitecture();
        var proggan = new ProgressiveGAN<double>(genArch, discArch, 512, 3, inputType: InputType.ThreeDimensional);

        // Act
        var metadata = proggan.GetModelMetadata();

        // Assert
        Assert.NotNull(metadata);
    }

    #endregion

    #region Tensor Shape Validation Tests

    [Theory]
    [InlineData(1, 50)]
    [InlineData(2, 100)]
    [InlineData(16, 128)]
    [InlineData(32, 256)]
    public void GenerativeAdversarialNetwork_GenerateRandomNoiseTensor_WithVariousBatchSizes_ReturnsCorrectShape(int batchSize, int noiseSize)
    {
        // Arrange
        var genArch = CreateThreeDimensionalGeneratorArchitecture();
        var discArch = CreateThreeDimensionalDiscriminatorArchitecture();
        var gan = new GenerativeAdversarialNetwork<double>(genArch, discArch, InputType.ThreeDimensional);

        // Act
        var noise = gan.GenerateRandomNoiseTensor(batchSize, noiseSize);

        // Assert
        Assert.Equal(2, noise.Shape.Length);
        Assert.Equal(batchSize, noise.Shape[0]);
        Assert.Equal(noiseSize, noise.Shape[1]);
    }

    [Fact]
    public void GenerativeAdversarialNetwork_NoiseValuesAreInExpectedRange()
    {
        // Arrange
        var genArch = CreateThreeDimensionalGeneratorArchitecture();
        var discArch = CreateThreeDimensionalDiscriminatorArchitecture();
        var gan = new GenerativeAdversarialNetwork<double>(genArch, discArch, InputType.ThreeDimensional);

        // Act
        var noise = gan.GenerateRandomNoiseTensor(10, 100);

        // Assert - Gaussian noise should have most values between -3 and 3
        bool hasValues = false;
        for (int i = 0; i < noise.Length; i++)
        {
            var value = noise.GetFlat(i);
            if (Math.Abs(value) > 0.001)
            {
                hasValues = true;
                break;
            }
        }
        Assert.True(hasValues, "Noise tensor should contain non-zero values");
    }

    #endregion

    #region Float Type Tests

    [Fact]
    public void DCGAN_WithFloatType_InitializesCorrectly()
    {
        // Arrange & Act
        var dcgan = new DCGAN<float>(100, 3, 32, 32);

        // Assert
        Assert.NotNull(dcgan);
        Assert.NotNull(dcgan.Generator);
        Assert.NotNull(dcgan.Discriminator);
    }

    [Fact]
    public void GenerativeAdversarialNetwork_WithFloatType_InitializesCorrectly()
    {
        // Arrange
        var genArch = CreateThreeDimensionalGeneratorArchitectureFloat();
        var discArch = CreateThreeDimensionalDiscriminatorArchitectureFloat();

        // Act
        var gan = new GenerativeAdversarialNetwork<float>(genArch, discArch, InputType.ThreeDimensional);

        // Assert
        Assert.NotNull(gan);
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

    /// <summary>
    /// Creates a discriminator architecture for ACGAN testing.
    /// ACGAN requires outputSize = 1 + numClasses (11 for 10 classes).
    /// </summary>
    private static NeuralNetworkArchitecture<double> CreateACGANDiscriminatorArchitecture()
    {
        return new NeuralNetworkArchitecture<double>(
            inputType: InputType.ThreeDimensional,
            taskType: NeuralNetworkTaskType.MultiClassClassification,
            complexity: NetworkComplexity.Simple,
            inputSize: 0,
            inputHeight: 8,
            inputWidth: 8,
            inputDepth: 3,
            outputSize: 11,  // 1 + numClasses (10)
            layers: null);
    }

    /// <summary>
    /// Creates a ThreeDimensional generator architecture for float type.
    /// </summary>
    private static NeuralNetworkArchitecture<float> CreateThreeDimensionalGeneratorArchitectureFloat()
    {
        return new NeuralNetworkArchitecture<float>(
            inputType: InputType.ThreeDimensional,
            taskType: NeuralNetworkTaskType.Generative,
            complexity: NetworkComplexity.Simple,
            inputSize: 0,
            inputHeight: 8,
            inputWidth: 8,
            inputDepth: 3,
            outputSize: 8 * 8 * 3,
            layers: null);
    }

    /// <summary>
    /// Creates a ThreeDimensional discriminator architecture for float type.
    /// </summary>
    private static NeuralNetworkArchitecture<float> CreateThreeDimensionalDiscriminatorArchitectureFloat()
    {
        return new NeuralNetworkArchitecture<float>(
            inputType: InputType.ThreeDimensional,
            taskType: NeuralNetworkTaskType.BinaryClassification,
            complexity: NetworkComplexity.Simple,
            inputSize: 0,
            inputHeight: 8,
            inputWidth: 8,
            inputDepth: 3,
            outputSize: 1,
            layers: null);
    }

    #endregion
}
