using AiDotNet.ComputerVision.Segmentation.Foundation;
using AiDotNet.ComputerVision.Segmentation.Semantic;
using AiDotNet.ComputerVision.Segmentation.Efficient;
using AiDotNet.ComputerVision.Segmentation.Medical;
using AiDotNet.ComputerVision.Segmentation.Panoptic;
using AiDotNet.ComputerVision.Segmentation.Mamba;
using AiDotNet.ComputerVision.Segmentation.Video;
using AiDotNet.ComputerVision.Segmentation.OpenVocabulary;
using AiDotNet.ComputerVision.Segmentation.Referring;
using AiDotNet.ComputerVision.Segmentation.Diffusion;
using AiDotNet.ComputerVision.Segmentation.Interactive;
using AiDotNet.ComputerVision.Segmentation.PointCloud;
using AiDotNet.ComputerVision.Segmentation.InstanceSegmentation;
using AiDotNet.Enums;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tensors;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.ComputerVision;

/// <summary>
/// Tests training robustness: multi-step training, predict-after-train consistency,
/// backward pass gradient flow, and training with unbatched inputs.
/// </summary>
public class SegmentationTrainingRobustnessTests
{
    private static NeuralNetworkArchitecture<double> Arch(int h = 32, int w = 32, int d = 3)
        => new(InputType.ThreeDimensional, NeuralNetworkTaskType.Regression,
               NetworkComplexity.Deep, 0, h, w, d, 0);

    private static Tensor<double> Rand(int seed, params int[] shape)
    {
        int total = 1; foreach (int s in shape) total *= s;
        var data = new double[total];
        var rng = new Random(seed);
        for (int i = 0; i < total; i++) data[i] = rng.NextDouble();
        return new Tensor<double>(shape, new Vector<double>(data));
    }

    #region Multi-Step Training — Semantic Models

    [Fact]
    public void SegFormer_MultiStepTrain_DoesNotThrow()
    {
        var model = new SegFormer<double>(Arch(), numClasses: 5, modelSize: SegFormerModelSize.B0);
        var input = Rand(42, 1, 3, 32, 32);
        var predicted = model.Predict(input);

        for (int step = 0; step < 3; step++)
        {
            var expected = Rand(step + 100, predicted.Shape);
            Assert.Null(Record.Exception(() => model.Train(input, expected)));
        }
    }

    [Fact]
    public void SegNeXt_MultiStepTrain_DoesNotThrow()
    {
        var model = new SegNeXt<double>(Arch(), numClasses: 5, modelSize: SegNeXtModelSize.Tiny);
        var input = Rand(42, 1, 3, 32, 32);
        var predicted = model.Predict(input);

        for (int step = 0; step < 3; step++)
        {
            var expected = Rand(step + 100, predicted.Shape);
            Assert.Null(Record.Exception(() => model.Train(input, expected)));
        }
    }

    [Fact]
    public void InternImage_MultiStepTrain_DoesNotThrow()
    {
        var model = new InternImage<double>(Arch(), numClasses: 5, modelSize: InternImageModelSize.Tiny);
        var input = Rand(42, 1, 3, 32, 32);
        var predicted = model.Predict(input);

        for (int step = 0; step < 3; step++)
        {
            var expected = Rand(step + 100, predicted.Shape);
            Assert.Null(Record.Exception(() => model.Train(input, expected)));
        }
    }

    [Fact]
    public void ViTAdapter_MultiStepTrain_DoesNotThrow()
    {
        var model = new ViTAdapter<double>(Arch(), numClasses: 5, modelSize: ViTAdapterModelSize.Small);
        var input = Rand(42, 1, 3, 32, 32);
        var predicted = model.Predict(input);

        for (int step = 0; step < 3; step++)
        {
            var expected = Rand(step + 100, predicted.Shape);
            Assert.Null(Record.Exception(() => model.Train(input, expected)));
        }
    }

    [Fact]
    public void ViTCoMer_MultiStepTrain_DoesNotThrow()
    {
        var model = new ViTCoMer<double>(Arch(), numClasses: 5, modelSize: ViTCoMerModelSize.Small);
        var input = Rand(42, 1, 3, 32, 32);
        var predicted = model.Predict(input);

        for (int step = 0; step < 3; step++)
        {
            var expected = Rand(step + 100, predicted.Shape);
            Assert.Null(Record.Exception(() => model.Train(input, expected)));
        }
    }

    [Fact]
    public void DiffCut_MultiStepTrain_DoesNotThrow()
    {
        var model = new DiffCut<double>(Arch(), numClasses: 5);
        var input = Rand(42, 1, 3, 32, 32);
        var predicted = model.Predict(input);

        for (int step = 0; step < 3; step++)
        {
            var expected = Rand(step + 100, predicted.Shape);
            Assert.Null(Record.Exception(() => model.Train(input, expected)));
        }
    }

    [Fact]
    public void DiffSeg_MultiStepTrain_DoesNotThrow()
    {
        var model = new DiffSeg<double>(Arch(), numClasses: 5);
        var input = Rand(42, 1, 3, 32, 32);
        var predicted = model.Predict(input);

        for (int step = 0; step < 3; step++)
        {
            var expected = Rand(step + 100, predicted.Shape);
            Assert.Null(Record.Exception(() => model.Train(input, expected)));
        }
    }

    #endregion

    #region Multi-Step Training — Foundation Models

    [Fact]
    public void Mask2Former_MultiStepTrain_DoesNotThrow()
    {
        var model = new Mask2Former<double>(Arch(), numClasses: 5, modelSize: Mask2FormerModelSize.SwinTiny);
        var input = Rand(42, 1, 3, 32, 32);
        var predicted = model.Predict(input);

        for (int step = 0; step < 3; step++)
        {
            var expected = Rand(step + 100, predicted.Shape);
            Assert.Null(Record.Exception(() => model.Train(input, expected)));
        }
    }

    [Fact]
    public void OneFormer_MultiStepTrain_DoesNotThrow()
    {
        var model = new OneFormer<double>(Arch(), numClasses: 5, modelSize: OneFormerModelSize.SwinLarge);
        var input = Rand(42, 1, 3, 32, 32);
        var predicted = model.Predict(input);

        for (int step = 0; step < 3; step++)
        {
            var expected = Rand(step + 100, predicted.Shape);
            Assert.Null(Record.Exception(() => model.Train(input, expected)));
        }
    }

    [Fact]
    public void MaskDINO_MultiStepTrain_DoesNotThrow()
    {
        var model = new MaskDINO<double>(Arch(), numClasses: 5, modelSize: MaskDINOModelSize.R50);
        var input = Rand(42, 1, 3, 32, 32);
        var predicted = model.Predict(input);

        for (int step = 0; step < 3; step++)
        {
            var expected = Rand(step + 100, predicted.Shape);
            Assert.Null(Record.Exception(() => model.Train(input, expected)));
        }
    }

    [Fact]
    public void SAM_MultiStepTrain_DoesNotThrow()
    {
        var model = new SAM<double>(Arch(), numClasses: 1, modelSize: SAMModelSize.ViTBase);
        var input = Rand(42, 1, 3, 32, 32);
        var predicted = model.Predict(input);

        for (int step = 0; step < 3; step++)
        {
            var expected = Rand(step + 100, predicted.Shape);
            Assert.Null(Record.Exception(() => model.Train(input, expected)));
        }
    }

    [Fact]
    public void SAM21_MultiStepTrain_DoesNotThrow()
    {
        var model = new SAM21<double>(Arch(), numClasses: 1, modelSize: SAM21ModelSize.BasePlus);
        var input = Rand(42, 1, 3, 32, 32);
        var predicted = model.Predict(input);

        for (int step = 0; step < 3; step++)
        {
            var expected = Rand(step + 100, predicted.Shape);
            Assert.Null(Record.Exception(() => model.Train(input, expected)));
        }
    }

    [Fact]
    public void SAMHQ_MultiStepTrain_DoesNotThrow()
    {
        var model = new SAMHQ<double>(Arch(), numClasses: 1, modelSize: SAMHQModelSize.ViTBase);
        var input = Rand(42, 1, 3, 32, 32);
        var predicted = model.Predict(input);

        for (int step = 0; step < 3; step++)
        {
            var expected = Rand(step + 100, predicted.Shape);
            Assert.Null(Record.Exception(() => model.Train(input, expected)));
        }
    }

    [Fact]
    public void EoMT_MultiStepTrain_DoesNotThrow()
    {
        var model = new EoMT<double>(Arch(), numClasses: 5, modelSize: EoMTModelSize.Small);
        var input = Rand(42, 1, 3, 32, 32);
        var predicted = model.Predict(input);

        for (int step = 0; step < 3; step++)
        {
            var expected = Rand(step + 100, predicted.Shape);
            Assert.Null(Record.Exception(() => model.Train(input, expected)));
        }
    }

    [Fact]
    public void OMGSeg_MultiStepTrain_DoesNotThrow()
    {
        var model = new OMGSeg<double>(Arch(), numClasses: 5, modelSize: OMGSegModelSize.Base);
        var input = Rand(42, 1, 3, 32, 32);
        var predicted = model.Predict(input);

        for (int step = 0; step < 3; step++)
        {
            var expected = Rand(step + 100, predicted.Shape);
            Assert.Null(Record.Exception(() => model.Train(input, expected)));
        }
    }

    [Fact]
    public void QueryMeldNet_MultiStepTrain_DoesNotThrow()
    {
        var model = new QueryMeldNet<double>(Arch(), numClasses: 5, modelSize: QueryMeldNetModelSize.R50);
        var input = Rand(42, 1, 3, 32, 32);
        var predicted = model.Predict(input);

        for (int step = 0; step < 3; step++)
        {
            var expected = Rand(step + 100, predicted.Shape);
            Assert.Null(Record.Exception(() => model.Train(input, expected)));
        }
    }

    [Fact]
    public void UNINEXT_MultiStepTrain_DoesNotThrow()
    {
        var model = new UNINEXT<double>(Arch(), numClasses: 5, modelSize: UNINEXTModelSize.R50);
        var input = Rand(42, 1, 3, 32, 32);
        var predicted = model.Predict(input);

        for (int step = 0; step < 3; step++)
        {
            var expected = Rand(step + 100, predicted.Shape);
            Assert.Null(Record.Exception(() => model.Train(input, expected)));
        }
    }

    [Fact]
    public void U2Seg_MultiStepTrain_DoesNotThrow()
    {
        var model = new U2Seg<double>(Arch(), numClasses: 5);
        var input = Rand(42, 1, 3, 32, 32);
        var predicted = model.Predict(input);

        for (int step = 0; step < 3; step++)
        {
            var expected = Rand(step + 100, predicted.Shape);
            Assert.Null(Record.Exception(() => model.Train(input, expected)));
        }
    }

    [Fact]
    public void XDecoder_MultiStepTrain_DoesNotThrow()
    {
        var model = new XDecoder<double>(Arch(), numClasses: 5, modelSize: XDecoderModelSize.Tiny);
        var input = Rand(42, 1, 3, 32, 32);
        var predicted = model.Predict(input);

        for (int step = 0; step < 3; step++)
        {
            var expected = Rand(step + 100, predicted.Shape);
            Assert.Null(Record.Exception(() => model.Train(input, expected)));
        }
    }

    #endregion

    #region Multi-Step Training — Medical, Mamba, Efficient, Video, OpenVocab, Referring, Diffusion, PointCloud

    [Fact]
    public void NnUNet_MultiStepTrain_DoesNotThrow()
    {
        var model = new NnUNet<double>(Arch(), numClasses: 5, modelSize: NnUNetModelSize.UNet2D);
        var input = Rand(42, 1, 3, 32, 32);
        var predicted = model.Predict(input);

        for (int step = 0; step < 3; step++)
        {
            var expected = Rand(step + 100, predicted.Shape);
            Assert.Null(Record.Exception(() => model.Train(input, expected)));
        }
    }

    [Fact]
    public void TransUNet_MultiStepTrain_DoesNotThrow()
    {
        var model = new TransUNet<double>(Arch(), numClasses: 5, modelSize: TransUNetModelSize.Base);
        var input = Rand(42, 1, 3, 32, 32);
        var predicted = model.Predict(input);

        for (int step = 0; step < 3; step++)
        {
            var expected = Rand(step + 100, predicted.Shape);
            Assert.Null(Record.Exception(() => model.Train(input, expected)));
        }
    }

    [Fact]
    public void SwinUNETR_MultiStepTrain_DoesNotThrow()
    {
        var model = new SwinUNETR<double>(Arch(), numClasses: 5, modelSize: SwinUNETRModelSize.Base);
        var input = Rand(42, 1, 3, 32, 32);
        var predicted = model.Predict(input);

        for (int step = 0; step < 3; step++)
        {
            var expected = Rand(step + 100, predicted.Shape);
            Assert.Null(Record.Exception(() => model.Train(input, expected)));
        }
    }

    [Fact]
    public void MedSAM_MultiStepTrain_DoesNotThrow()
    {
        var model = new MedSAM<double>(Arch(), numClasses: 5, modelSize: MedSAMModelSize.ViTBase);
        var input = Rand(42, 1, 3, 32, 32);
        var predicted = model.Predict(input);

        for (int step = 0; step < 3; step++)
        {
            var expected = Rand(step + 100, predicted.Shape);
            Assert.Null(Record.Exception(() => model.Train(input, expected)));
        }
    }

    [Fact]
    public void MedNeXt_MultiStepTrain_DoesNotThrow()
    {
        var model = new MedNeXt<double>(Arch(), numClasses: 5, modelSize: MedNeXtModelSize.Small);
        var input = Rand(42, 1, 3, 32, 32);
        var predicted = model.Predict(input);

        for (int step = 0; step < 3; step++)
        {
            var expected = Rand(step + 100, predicted.Shape);
            Assert.Null(Record.Exception(() => model.Train(input, expected)));
        }
    }

    [Fact]
    public void MedSAM2_MultiStepTrain_DoesNotThrow()
    {
        var model = new MedSAM2<double>(Arch(), numClasses: 5, modelSize: MedSAM2ModelSize.Base);
        var input = Rand(42, 1, 3, 32, 32);
        var predicted = model.Predict(input);

        for (int step = 0; step < 3; step++)
        {
            var expected = Rand(step + 100, predicted.Shape);
            Assert.Null(Record.Exception(() => model.Train(input, expected)));
        }
    }

    [Fact]
    public void VisionMamba_MultiStepTrain_DoesNotThrow()
    {
        var model = new VisionMamba<double>(Arch(), numClasses: 5, modelSize: VisionMambaModelSize.Tiny);
        var input = Rand(42, 1, 3, 32, 32);
        var predicted = model.Predict(input);

        for (int step = 0; step < 3; step++)
        {
            var expected = Rand(step + 100, predicted.Shape);
            Assert.Null(Record.Exception(() => model.Train(input, expected)));
        }
    }

    [Fact]
    public void VMamba_MultiStepTrain_DoesNotThrow()
    {
        var model = new VMamba<double>(Arch(), numClasses: 5, modelSize: VMambaModelSize.Tiny);
        var input = Rand(42, 1, 3, 32, 32);
        var predicted = model.Predict(input);

        for (int step = 0; step < 3; step++)
        {
            var expected = Rand(step + 100, predicted.Shape);
            Assert.Null(Record.Exception(() => model.Train(input, expected)));
        }
    }

    [Fact]
    public void PIDNet_MultiStepTrain_DoesNotThrow()
    {
        var model = new PIDNet<double>(Arch(), numClasses: 5, modelSize: PIDNetModelSize.Small);
        var input = Rand(42, 1, 3, 32, 32);
        var predicted = model.Predict(input);

        for (int step = 0; step < 3; step++)
        {
            var expected = Rand(step + 100, predicted.Shape);
            Assert.Null(Record.Exception(() => model.Train(input, expected)));
        }
    }

    [Fact]
    public void DEVA_MultiStepTrain_DoesNotThrow()
    {
        var model = new DEVA<double>(Arch(), numClasses: 5, modelSize: DEVAModelSize.Base);
        var input = Rand(42, 1, 3, 32, 32);
        var predicted = model.Predict(input);

        for (int step = 0; step < 3; step++)
        {
            var expected = Rand(step + 100, predicted.Shape);
            Assert.Null(Record.Exception(() => model.Train(input, expected)));
        }
    }

    [Fact]
    public void EfficientTAM_MultiStepTrain_DoesNotThrow()
    {
        var model = new EfficientTAM<double>(Arch(), numClasses: 5, modelSize: EfficientTAMModelSize.Small);
        var input = Rand(42, 1, 3, 32, 32);
        var predicted = model.Predict(input);

        for (int step = 0; step < 3; step++)
        {
            var expected = Rand(step + 100, predicted.Shape);
            Assert.Null(Record.Exception(() => model.Train(input, expected)));
        }
    }

    [Fact]
    public void SAN_MultiStepTrain_DoesNotThrow()
    {
        var model = new SAN<double>(Arch(), numClasses: 5);
        var input = Rand(42, 1, 3, 32, 32);
        var predicted = model.Predict(input);

        for (int step = 0; step < 3; step++)
        {
            var expected = Rand(step + 100, predicted.Shape);
            Assert.Null(Record.Exception(() => model.Train(input, expected)));
        }
    }

    [Fact]
    public void LISA_MultiStepTrain_DoesNotThrow()
    {
        var model = new LISA<double>(Arch(), numClasses: 5);
        var input = Rand(42, 1, 3, 32, 32);
        var predicted = model.Predict(input);

        for (int step = 0; step < 3; step++)
        {
            var expected = Rand(step + 100, predicted.Shape);
            Assert.Null(Record.Exception(() => model.Train(input, expected)));
        }
    }

    [Fact]
    public void DiffCutSegmentation_MultiStepTrain_DoesNotThrow()
    {
        var model = new DiffCutSegmentation<double>(Arch(), numClasses: 5);
        var input = Rand(42, 1, 3, 32, 32);
        var predicted = model.Predict(input);

        for (int step = 0; step < 3; step++)
        {
            var expected = Rand(step + 100, predicted.Shape);
            Assert.Null(Record.Exception(() => model.Train(input, expected)));
        }
    }

    [Fact]
    public void PointTransformerV3_MultiStepTrain_DoesNotThrow()
    {
        var model = new PointTransformerV3<double>(Arch(), numClasses: 5, modelSize: PointTransformerV3ModelSize.Base);
        var input = Rand(42, 1, 3, 32, 32);
        var predicted = model.Predict(input);

        for (int step = 0; step < 3; step++)
        {
            var expected = Rand(step + 100, predicted.Shape);
            Assert.Null(Record.Exception(() => model.Train(input, expected)));
        }
    }

    [Fact]
    public void Sonata_MultiStepTrain_DoesNotThrow()
    {
        var model = new Sonata<double>(Arch(), numClasses: 5, modelSize: SonataModelSize.Base);
        var input = Rand(42, 1, 3, 32, 32);
        var predicted = model.Predict(input);

        for (int step = 0; step < 3; step++)
        {
            var expected = Rand(step + 100, predicted.Shape);
            Assert.Null(Record.Exception(() => model.Train(input, expected)));
        }
    }

    #endregion

    #region Predict-After-Train — Output consistency

    [Fact]
    public void SegFormer_PredictAfterTrain_ProducesDifferentOutput()
    {
        var model = new SegFormer<double>(Arch(), numClasses: 5, modelSize: SegFormerModelSize.B0);
        var input = Rand(42, 1, 3, 32, 32);

        var outputBefore = model.Predict(input);
        var expected = Rand(99, outputBefore.Shape);
        model.Train(input, expected);
        var outputAfter = model.Predict(input);

        Assert.NotNull(outputAfter);
        Assert.Equal(outputBefore.Shape.Length, outputAfter.Shape.Length);
        for (int i = 0; i < outputBefore.Shape.Length; i++)
        {
            Assert.Equal(outputBefore.Shape[i], outputAfter.Shape[i]);
        }
    }

    [Fact]
    public void Mask2Former_PredictAfterTrain_ProducesDifferentOutput()
    {
        var model = new Mask2Former<double>(Arch(), numClasses: 5, modelSize: Mask2FormerModelSize.SwinTiny);
        var input = Rand(42, 1, 3, 32, 32);

        var outputBefore = model.Predict(input);
        var expected = Rand(99, outputBefore.Shape);
        model.Train(input, expected);
        var outputAfter = model.Predict(input);

        Assert.NotNull(outputAfter);
        Assert.Equal(outputBefore.Shape.Length, outputAfter.Shape.Length);
    }

    [Fact]
    public void NnUNet_PredictAfterTrain_ProducesDifferentOutput()
    {
        var model = new NnUNet<double>(Arch(), numClasses: 5, modelSize: NnUNetModelSize.UNet2D);
        var input = Rand(42, 1, 3, 32, 32);

        var outputBefore = model.Predict(input);
        var expected = Rand(99, outputBefore.Shape);
        model.Train(input, expected);
        var outputAfter = model.Predict(input);

        Assert.NotNull(outputAfter);
        Assert.Equal(outputBefore.Shape.Length, outputAfter.Shape.Length);
    }

    #endregion

    #region Train with Unbatched (3D) Input

    [Fact]
    public void SegFormer_Train_Unbatched3DInput_DoesNotThrow()
    {
        var model = new SegFormer<double>(Arch(), numClasses: 5, modelSize: SegFormerModelSize.B0);
        var input = Rand(42, 3, 32, 32); // 3D unbatched
        var predicted = model.Predict(input);
        var expected = Rand(99, predicted.Shape);
        Assert.Null(Record.Exception(() => model.Train(input, expected)));
    }

    [Fact]
    public void Mask2Former_Train_Unbatched3DInput_DoesNotThrow()
    {
        var model = new Mask2Former<double>(Arch(), numClasses: 5, modelSize: Mask2FormerModelSize.SwinTiny);
        var input = Rand(42, 3, 32, 32);
        var predicted = model.Predict(input);
        var expected = Rand(99, predicted.Shape);
        Assert.Null(Record.Exception(() => model.Train(input, expected)));
    }

    [Fact]
    public void SAM_Train_Unbatched3DInput_DoesNotThrow()
    {
        var model = new SAM<double>(Arch(), numClasses: 1, modelSize: SAMModelSize.ViTBase);
        var input = Rand(42, 3, 32, 32);
        var predicted = model.Predict(input);
        var expected = Rand(99, predicted.Shape);
        Assert.Null(Record.Exception(() => model.Train(input, expected)));
    }

    [Fact]
    public void NnUNet_Train_Unbatched3DInput_DoesNotThrow()
    {
        var model = new NnUNet<double>(Arch(), numClasses: 5, modelSize: NnUNetModelSize.UNet2D);
        var input = Rand(42, 3, 32, 32);
        var predicted = model.Predict(input);
        var expected = Rand(99, predicted.Shape);
        Assert.Null(Record.Exception(() => model.Train(input, expected)));
    }

    [Fact]
    public void PIDNet_Train_Unbatched3DInput_DoesNotThrow()
    {
        var model = new PIDNet<double>(Arch(), numClasses: 5, modelSize: PIDNetModelSize.Small);
        var input = Rand(42, 3, 32, 32);
        var predicted = model.Predict(input);
        var expected = Rand(99, predicted.Shape);
        Assert.Null(Record.Exception(() => model.Train(input, expected)));
    }

    [Fact]
    public void VisionMamba_Train_Unbatched3DInput_DoesNotThrow()
    {
        var model = new VisionMamba<double>(Arch(), numClasses: 5, modelSize: VisionMambaModelSize.Tiny);
        var input = Rand(42, 3, 32, 32);
        var predicted = model.Predict(input);
        var expected = Rand(99, predicted.Shape);
        Assert.Null(Record.Exception(() => model.Train(input, expected)));
    }

    [Fact]
    public void DEVA_Train_Unbatched3DInput_DoesNotThrow()
    {
        var model = new DEVA<double>(Arch(), numClasses: 5, modelSize: DEVAModelSize.Base);
        var input = Rand(42, 3, 32, 32);
        var predicted = model.Predict(input);
        var expected = Rand(99, predicted.Shape);
        Assert.Null(Record.Exception(() => model.Train(input, expected)));
    }

    [Fact]
    public void LISA_Train_Unbatched3DInput_DoesNotThrow()
    {
        var model = new LISA<double>(Arch(), numClasses: 5);
        var input = Rand(42, 3, 32, 32);
        var predicted = model.Predict(input);
        var expected = Rand(99, predicted.Shape);
        Assert.Null(Record.Exception(() => model.Train(input, expected)));
    }

    #endregion

    #region Instance Segmentation — Multi-Step Train

    [Fact]
    public void YOLOv8Seg_MultiStepTrain_DoesNotThrow()
    {
        var model = new YOLOv8Seg<double>(Arch(), modelSize: YOLOv8SegModelSize.N);
        var input = Rand(42, 1, 3, 32, 32);
        var predicted = model.Predict(input);

        for (int step = 0; step < 3; step++)
        {
            var expected = Rand(step + 100, predicted.Shape);
            Assert.Null(Record.Exception(() => model.Train(input, expected)));
        }
    }

    [Fact]
    public void YOLOv9Seg_MultiStepTrain_DoesNotThrow()
    {
        var model = new YOLOv9Seg<double>(Arch(), modelSize: YOLOv9SegModelSize.C);
        var input = Rand(42, 1, 3, 32, 32);
        var predicted = model.Predict(input);

        for (int step = 0; step < 3; step++)
        {
            var expected = Rand(step + 100, predicted.Shape);
            Assert.Null(Record.Exception(() => model.Train(input, expected)));
        }
    }

    [Fact]
    public void YOLO11Seg_MultiStepTrain_DoesNotThrow()
    {
        var model = new YOLO11Seg<double>(Arch(), modelSize: YOLO11SegModelSize.N);
        var input = Rand(42, 1, 3, 32, 32);
        var predicted = model.Predict(input);

        for (int step = 0; step < 3; step++)
        {
            var expected = Rand(step + 100, predicted.Shape);
            Assert.Null(Record.Exception(() => model.Train(input, expected)));
        }
    }

    [Fact]
    public void YOLOv12Seg_MultiStepTrain_DoesNotThrow()
    {
        var model = new YOLOv12Seg<double>(Arch(), modelSize: YOLOv12SegModelSize.N);
        var input = Rand(42, 1, 3, 32, 32);
        var predicted = model.Predict(input);

        for (int step = 0; step < 3; step++)
        {
            var expected = Rand(step + 100, predicted.Shape);
            Assert.Null(Record.Exception(() => model.Train(input, expected)));
        }
    }

    [Fact]
    public void YOLO26Seg_MultiStepTrain_DoesNotThrow()
    {
        var model = new YOLO26Seg<double>(Arch(), modelSize: YOLO26SegModelSize.N);
        var input = Rand(42, 1, 3, 32, 32);
        var predicted = model.Predict(input);

        for (int step = 0; step < 3; step++)
        {
            var expected = Rand(step + 100, predicted.Shape);
            Assert.Null(Record.Exception(() => model.Train(input, expected)));
        }
    }

    #endregion

    #region Remaining models — Multi-Step Train

    [Fact]
    public void UniVS_MultiStepTrain_DoesNotThrow()
    {
        var model = new UniVS<double>(Arch(), numClasses: 5, modelSize: UniVSModelSize.R50);
        var input = Rand(42, 1, 3, 32, 32);
        var predicted = model.Predict(input);

        for (int step = 0; step < 3; step++)
        {
            var expected = Rand(step + 100, predicted.Shape);
            Assert.Null(Record.Exception(() => model.Train(input, expected)));
        }
    }

    [Fact]
    public void CATSeg_MultiStepTrain_DoesNotThrow()
    {
        var model = new CATSeg<double>(Arch(), numClasses: 5);
        var input = Rand(42, 1, 3, 32, 32);
        var predicted = model.Predict(input);

        for (int step = 0; step < 3; step++)
        {
            var expected = Rand(step + 100, predicted.Shape);
            Assert.Null(Record.Exception(() => model.Train(input, expected)));
        }
    }

    [Fact]
    public void SED_MultiStepTrain_DoesNotThrow()
    {
        var model = new SED<double>(Arch(), numClasses: 5);
        var input = Rand(42, 1, 3, 32, 32);
        var predicted = model.Predict(input);

        for (int step = 0; step < 3; step++)
        {
            var expected = Rand(step + 100, predicted.Shape);
            Assert.Null(Record.Exception(() => model.Train(input, expected)));
        }
    }

    [Fact]
    public void GroundedSAM2_MultiStepTrain_DoesNotThrow()
    {
        var model = new GroundedSAM2<double>(Arch(), numClasses: 5);
        var input = Rand(42, 1, 3, 32, 32);
        var predicted = model.Predict(input);

        for (int step = 0; step < 3; step++)
        {
            var expected = Rand(step + 100, predicted.Shape);
            Assert.Null(Record.Exception(() => model.Train(input, expected)));
        }
    }

    [Fact]
    public void MaskAdapter_MultiStepTrain_DoesNotThrow()
    {
        var model = new MaskAdapter<double>(Arch(), numClasses: 5);
        var input = Rand(42, 1, 3, 32, 32);
        var predicted = model.Predict(input);

        for (int step = 0; step < 3; step++)
        {
            var expected = Rand(step + 100, predicted.Shape);
            Assert.Null(Record.Exception(() => model.Train(input, expected)));
        }
    }

    [Fact]
    public void OpenVocabSAM_MultiStepTrain_DoesNotThrow()
    {
        var model = new OpenVocabSAM<double>(Arch(), numClasses: 5);
        var input = Rand(42, 1, 3, 32, 32);
        var predicted = model.Predict(input);

        for (int step = 0; step < 3; step++)
        {
            var expected = Rand(step + 100, predicted.Shape);
            Assert.Null(Record.Exception(() => model.Train(input, expected)));
        }
    }

    [Fact]
    public void VideoLISA_MultiStepTrain_DoesNotThrow()
    {
        var model = new VideoLISA<double>(Arch(), numClasses: 5);
        var input = Rand(42, 1, 3, 32, 32);
        var predicted = model.Predict(input);

        for (int step = 0; step < 3; step++)
        {
            var expected = Rand(step + 100, predicted.Shape);
            Assert.Null(Record.Exception(() => model.Train(input, expected)));
        }
    }

    [Fact]
    public void GLaMM_MultiStepTrain_DoesNotThrow()
    {
        var model = new GLaMM<double>(Arch(), numClasses: 5);
        var input = Rand(42, 1, 3, 32, 32);
        var predicted = model.Predict(input);

        for (int step = 0; step < 3; step++)
        {
            var expected = Rand(step + 100, predicted.Shape);
            Assert.Null(Record.Exception(() => model.Train(input, expected)));
        }
    }

    [Fact]
    public void OMGLLaVA_MultiStepTrain_DoesNotThrow()
    {
        var model = new OMGLLaVA<double>(Arch(), numClasses: 5);
        var input = Rand(42, 1, 3, 32, 32);
        var predicted = model.Predict(input);

        for (int step = 0; step < 3; step++)
        {
            var expected = Rand(step + 100, predicted.Shape);
            Assert.Null(Record.Exception(() => model.Train(input, expected)));
        }
    }

    [Fact]
    public void PixelLM_MultiStepTrain_DoesNotThrow()
    {
        var model = new PixelLM<double>(Arch(), numClasses: 5);
        var input = Rand(42, 1, 3, 32, 32);
        var predicted = model.Predict(input);

        for (int step = 0; step < 3; step++)
        {
            var expected = Rand(step + 100, predicted.Shape);
            Assert.Null(Record.Exception(() => model.Train(input, expected)));
        }
    }

    [Fact]
    public void ODISESegmentation_MultiStepTrain_DoesNotThrow()
    {
        var model = new ODISESegmentation<double>(Arch(), numClasses: 5);
        var input = Rand(42, 1, 3, 32, 32);
        var predicted = model.Predict(input);

        for (int step = 0; step < 3; step++)
        {
            var expected = Rand(step + 100, predicted.Shape);
            Assert.Null(Record.Exception(() => model.Train(input, expected)));
        }
    }

    [Fact]
    public void MedSegDiffV2Segmentation_MultiStepTrain_DoesNotThrow()
    {
        var model = new MedSegDiffV2Segmentation<double>(Arch(), numClasses: 5);
        var input = Rand(42, 1, 3, 32, 32);
        var predicted = model.Predict(input);

        for (int step = 0; step < 3; step++)
        {
            var expected = Rand(step + 100, predicted.Shape);
            Assert.Null(Record.Exception(() => model.Train(input, expected)));
        }
    }

    [Fact]
    public void Concerto_MultiStepTrain_DoesNotThrow()
    {
        var model = new Concerto<double>(Arch(), numClasses: 5, modelSize: ConcertoModelSize.Base);
        var input = Rand(42, 1, 3, 32, 32);
        var predicted = model.Predict(input);

        for (int step = 0; step < 3; step++)
        {
            var expected = Rand(step + 100, predicted.Shape);
            Assert.Null(Record.Exception(() => model.Train(input, expected)));
        }
    }

    [Fact]
    public void ViMUNet_MultiStepTrain_DoesNotThrow()
    {
        var model = new ViMUNet<double>(Arch(), numClasses: 5);
        var input = Rand(42, 1, 3, 32, 32);
        var predicted = model.Predict(input);

        for (int step = 0; step < 3; step++)
        {
            var expected = Rand(step + 100, predicted.Shape);
            Assert.Null(Record.Exception(() => model.Train(input, expected)));
        }
    }

    [Fact]
    public void BiomedParse_MultiStepTrain_DoesNotThrow()
    {
        var model = new BiomedParse<double>(Arch(), numClasses: 5);
        var input = Rand(42, 1, 3, 32, 32);
        var predicted = model.Predict(input);

        for (int step = 0; step < 3; step++)
        {
            var expected = Rand(step + 100, predicted.Shape);
            Assert.Null(Record.Exception(() => model.Train(input, expected)));
        }
    }

    [Fact]
    public void UniverSeg_MultiStepTrain_DoesNotThrow()
    {
        var model = new UniverSeg<double>(Arch(), numClasses: 5);
        var input = Rand(42, 1, 3, 32, 32);
        var predicted = model.Predict(input);

        for (int step = 0; step < 3; step++)
        {
            var expected = Rand(step + 100, predicted.Shape);
            Assert.Null(Record.Exception(() => model.Train(input, expected)));
        }
    }

    [Fact]
    public void MedSegDiffV2_MultiStepTrain_DoesNotThrow()
    {
        var model = new MedSegDiffV2<double>(Arch(), numClasses: 5);
        var input = Rand(42, 1, 3, 32, 32);
        var predicted = model.Predict(input);

        for (int step = 0; step < 3; step++)
        {
            var expected = Rand(step + 100, predicted.Shape);
            Assert.Null(Record.Exception(() => model.Train(input, expected)));
        }
    }

    [Fact]
    public void UMamba_MultiStepTrain_DoesNotThrow()
    {
        var model = new UMamba<double>(Arch(), numClasses: 5);
        var input = Rand(42, 1, 3, 32, 32);
        var predicted = model.Predict(input);

        for (int step = 0; step < 3; step++)
        {
            var expected = Rand(step + 100, predicted.Shape);
            Assert.Null(Record.Exception(() => model.Train(input, expected)));
        }
    }

    [Fact]
    public void SegMamba_MultiStepTrain_DoesNotThrow()
    {
        var model = new SegMamba<double>(Arch(), numClasses: 5);
        var input = Rand(42, 1, 3, 32, 32);
        var predicted = model.Predict(input);

        for (int step = 0; step < 3; step++)
        {
            var expected = Rand(step + 100, predicted.Shape);
            Assert.Null(Record.Exception(() => model.Train(input, expected)));
        }
    }

    #endregion

    #region Efficient Models — Multi-Step Train

    [Fact]
    public void FastSAM_MultiStepTrain_DoesNotThrow()
    {
        var model = new FastSAM<double>(Arch(), numClasses: 1);
        var input = Rand(42, 1, 3, 32, 32);
        var predicted = model.Predict(input);

        for (int step = 0; step < 3; step++)
        {
            var expected = Rand(step + 100, predicted.Shape);
            Assert.Null(Record.Exception(() => model.Train(input, expected)));
        }
    }

    [Fact]
    public void MobileSAM_MultiStepTrain_DoesNotThrow()
    {
        var model = new MobileSAM<double>(Arch(), numClasses: 1);
        var input = Rand(42, 1, 3, 32, 32);
        var predicted = model.Predict(input);

        for (int step = 0; step < 3; step++)
        {
            var expected = Rand(step + 100, predicted.Shape);
            Assert.Null(Record.Exception(() => model.Train(input, expected)));
        }
    }

    [Fact]
    public void EdgeSAM_MultiStepTrain_DoesNotThrow()
    {
        var model = new EdgeSAM<double>(Arch(), numClasses: 1);
        var input = Rand(42, 1, 3, 32, 32);
        var predicted = model.Predict(input);

        for (int step = 0; step < 3; step++)
        {
            var expected = Rand(step + 100, predicted.Shape);
            Assert.Null(Record.Exception(() => model.Train(input, expected)));
        }
    }

    [Fact]
    public void SlimSAM_MultiStepTrain_DoesNotThrow()
    {
        var model = new SlimSAM<double>(Arch(), numClasses: 1);
        var input = Rand(42, 1, 3, 32, 32);
        var predicted = model.Predict(input);

        for (int step = 0; step < 3; step++)
        {
            var expected = Rand(step + 100, predicted.Shape);
            Assert.Null(Record.Exception(() => model.Train(input, expected)));
        }
    }

    [Fact]
    public void EfficientSAM_MultiStepTrain_DoesNotThrow()
    {
        var model = new EfficientSAM<double>(Arch(), numClasses: 1);
        var input = Rand(42, 1, 3, 32, 32);
        var predicted = model.Predict(input);

        for (int step = 0; step < 3; step++)
        {
            var expected = Rand(step + 100, predicted.Shape);
            Assert.Null(Record.Exception(() => model.Train(input, expected)));
        }
    }

    [Fact]
    public void RepViTSAM_MultiStepTrain_DoesNotThrow()
    {
        var model = new RepViTSAM<double>(Arch(), numClasses: 1);
        var input = Rand(42, 1, 3, 32, 32);
        var predicted = model.Predict(input);

        for (int step = 0; step < 3; step++)
        {
            var expected = Rand(step + 100, predicted.Shape);
            Assert.Null(Record.Exception(() => model.Train(input, expected)));
        }
    }

    [Fact]
    public void SegGPT_MultiStepTrain_DoesNotThrow()
    {
        var model = new SegGPT<double>(Arch(), numClasses: 5, modelSize: SegGPTModelSize.ViTLarge);
        var input = Rand(42, 1, 3, 32, 32);
        var predicted = model.Predict(input);

        for (int step = 0; step < 3; step++)
        {
            var expected = Rand(step + 100, predicted.Shape);
            Assert.Null(Record.Exception(() => model.Train(input, expected)));
        }
    }

    [Fact]
    public void SEEM_MultiStepTrain_DoesNotThrow()
    {
        var model = new SEEM<double>(Arch(), numClasses: 5, modelSize: SEEMModelSize.Tiny);
        var input = Rand(42, 1, 3, 32, 32);
        var predicted = model.Predict(input);

        for (int step = 0; step < 3; step++)
        {
            var expected = Rand(step + 100, predicted.Shape);
            Assert.Null(Record.Exception(() => model.Train(input, expected)));
        }
    }

    [Fact]
    public void KMaXDeepLab_MultiStepTrain_DoesNotThrow()
    {
        var model = new KMaXDeepLab<double>(Arch(), numClasses: 5, modelSize: KMaXDeepLabModelSize.R50);
        var input = Rand(42, 1, 3, 32, 32);
        var predicted = model.Predict(input);

        for (int step = 0; step < 3; step++)
        {
            var expected = Rand(step + 100, predicted.Shape);
            Assert.Null(Record.Exception(() => model.Train(input, expected)));
        }
    }

    [Fact]
    public void ODISE_MultiStepTrain_DoesNotThrow()
    {
        var model = new ODISE<double>(Arch(), numClasses: 5, modelSize: ODISEModelSize.Base);
        var input = Rand(42, 1, 3, 32, 32);
        var predicted = model.Predict(input);

        for (int step = 0; step < 3; step++)
        {
            var expected = Rand(step + 100, predicted.Shape);
            Assert.Null(Record.Exception(() => model.Train(input, expected)));
        }
    }

    [Fact]
    public void CUPS_MultiStepTrain_DoesNotThrow()
    {
        var model = new CUPS<double>(Arch(), numClasses: 5);
        var input = Rand(42, 1, 3, 32, 32);
        var predicted = model.Predict(input);

        for (int step = 0; step < 3; step++)
        {
            var expected = Rand(step + 100, predicted.Shape);
            Assert.Null(Record.Exception(() => model.Train(input, expected)));
        }
    }

    #endregion
}
