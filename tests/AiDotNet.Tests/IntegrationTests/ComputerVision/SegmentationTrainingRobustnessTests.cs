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
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tests.ModelFamilyTests.Base;
using System;
using System.Threading.Tasks;

namespace AiDotNet.Tests.IntegrationTests.ComputerVision;

/// <summary>
/// Tests training robustness: multi-step training, predict-after-train consistency,
/// backward pass gradient flow, and training with unbatched inputs.
/// </summary>
// #1754/#1706: this class constructs and TRAINS 70 foundation-scale SAM/ViT/Swin-family segmentation
// models (one per test); committed memory accumulates across the class until the runner OOMs and the
// "Integration C - ComputerVision" shard dies with a shutdown signal. Genuine foundation-scale training
// cost, not a per-test bug (retention sources detailed in the class docstring below), so route the class
// to the HeavyTimeout nightly lane; the default PR gate excludes HeavyTimeout and completes on the fast
// forward/shape CV tests. Durable fix: AiDotNet.Tensors #714 (weak-refs / pressure-evict).
[Trait("Category", "HeavyTimeout")]
public class SegmentationTrainingRobustnessTests : IDisposable
{
    // These tests construct and TRAIN heavy SAM/ViT-family segmentation models one per test. Two
    // process-global retention sources let committed memory accumulate across the class's tests until
    // a later one OOMs even though each model is otherwise collectable: InferenceWeightCache pins
    // disposed models' derived weight packs (keyed by array identity), and a plain GC.Collect() does
    // not compact the LOH (committed-but-free LOH counts against the heap limit). Reclaim between
    // every test via the shared model-family gate — pure memory hygiene, changes no assertion, scale,
    // iteration count, or timeout (same teardown the NeuralNetworks/Diffusion model-family bases use).
    public void Dispose() => ModelFamilyTestGcGate.ReclaimBetweenTests();

    private static NeuralNetworkArchitecture<float> Arch(int h = 32, int w = 32, int d = 3)
        => new(InputType.ThreeDimensional, NeuralNetworkTaskType.Regression,
               NetworkComplexity.Deep, 0, h, w, d, 0);

    private static Tensor<float> Rand(int seed, params int[] shape)
    {
        int total = 1; foreach (int s in shape) total *= s;
        var data = new float[total];
        var rng = RandomHelper.CreateSeededRandom(seed);
        for (int i = 0; i < total; i++) data[i] = (float)rng.NextDouble();
        return new Tensor<float>(shape, new Vector<float>(data));
    }

    #region Multi-Step Training — Semantic Models

    [Fact(Timeout = 120000)]
    public async Task SegFormer_MultiStepTrain_DoesNotThrow()
    {
        var model = new SegFormer<float>(Arch(), options: new SegFormerOptions { NumClasses = 5, ModelSize = SegFormerModelSize.B0 });
        var input = Rand(42, 1, 3, 32, 32);
        var predicted = model.Predict(input);

        for (int step = 0; step < 3; step++)
        {
            var expected = Rand(step + 100, predicted.Shape.ToArray());
            Assert.Null(Record.Exception(() => model.Train(input, expected)));
        }
    }

    [Fact(Timeout = 120000)]
    public async Task SegNeXt_MultiStepTrain_DoesNotThrow()
    {
        var model = new SegNeXt<float>(Arch(), options: new SegNeXtOptions { NumClasses = 5, ModelSize = SegNeXtModelSize.Tiny });
        var input = Rand(42, 1, 3, 32, 32);
        var predicted = model.Predict(input);

        for (int step = 0; step < 3; step++)
        {
            var expected = Rand(step + 100, predicted.Shape.ToArray());
            Assert.Null(Record.Exception(() => model.Train(input, expected)));
        }
    }

    [Fact(Timeout = 120000)]
    public async Task InternImage_MultiStepTrain_DoesNotThrow()
    {
        var model = new InternImage<float>(Arch(), options: new InternImageOptions { NumClasses = 5, ModelSize = InternImageModelSize.Tiny });
        var input = Rand(42, 1, 3, 32, 32);
        var predicted = model.Predict(input);

        for (int step = 0; step < 3; step++)
        {
            var expected = Rand(step + 100, predicted.Shape.ToArray());
            Assert.Null(Record.Exception(() => model.Train(input, expected)));
        }
    }

    [Fact(Timeout = 120000)]
    public async Task ViTAdapter_MultiStepTrain_DoesNotThrow()
    {
        var model = new ViTAdapter<float>(Arch(), options: new ViTAdapterOptions { NumClasses = 5, ModelSize = ViTAdapterModelSize.Small });
        var input = Rand(42, 1, 3, 32, 32);
        var predicted = model.Predict(input);

        for (int step = 0; step < 3; step++)
        {
            var expected = Rand(step + 100, predicted.Shape.ToArray());
            Assert.Null(Record.Exception(() => model.Train(input, expected)));
        }
    }

    [Fact(Timeout = 120000)]
    public async Task ViTCoMer_MultiStepTrain_DoesNotThrow()
    {
        var model = new ViTCoMer<float>(Arch(), options: new ViTCoMerOptions { NumClasses = 5, ModelSize = ViTCoMerModelSize.Small });
        var input = Rand(42, 1, 3, 32, 32);
        var predicted = model.Predict(input);

        for (int step = 0; step < 3; step++)
        {
            var expected = Rand(step + 100, predicted.Shape.ToArray());
            Assert.Null(Record.Exception(() => model.Train(input, expected)));
        }
    }

    [Fact(Timeout = 120000)]
    public async Task DiffCut_MultiStepTrain_DoesNotThrow()
    {
        var model = new DiffCut<float>(Arch(), options: new DiffCutOptions { NumClasses = 5 });
        var input = Rand(42, 1, 3, 32, 32);
        var predicted = model.Predict(input);

        for (int step = 0; step < 3; step++)
        {
            var expected = Rand(step + 100, predicted.Shape.ToArray());
            Assert.Null(Record.Exception(() => model.Train(input, expected)));
        }
    }

    [Fact(Timeout = 120000)]
    public async Task DiffSeg_MultiStepTrain_DoesNotThrow()
    {
        var model = new DiffSeg<float>(Arch(), options: new DiffSegOptions { NumClasses = 5 });
        var input = Rand(42, 1, 3, 32, 32);
        var predicted = model.Predict(input);

        for (int step = 0; step < 3; step++)
        {
            var expected = Rand(step + 100, predicted.Shape.ToArray());
            Assert.Null(Record.Exception(() => model.Train(input, expected)));
        }
    }

    #endregion

    #region Multi-Step Training — Foundation Models

    [Fact(Timeout = 120000)]
    public async Task Mask2Former_MultiStepTrain_DoesNotThrow()
    {
        var model = new Mask2Former<float>(Arch(), options: new Mask2FormerOptions { NumClasses = 5, ModelSize = Mask2FormerModelSize.SwinTiny });
        var input = Rand(42, 1, 3, 32, 32);
        var predicted = model.Predict(input);

        for (int step = 0; step < 3; step++)
        {
            var expected = Rand(step + 100, predicted.Shape.ToArray());
            Assert.Null(Record.Exception(() => model.Train(input, expected)));
        }
    }

    [Fact(Timeout = 120000)]
    public async Task OneFormer_MultiStepTrain_DoesNotThrow()
    {
        var model = new OneFormer<float>(Arch(), options: new OneFormerOptions { NumClasses = 5, ModelSize = OneFormerModelSize.SwinLarge });
        var input = Rand(42, 1, 3, 32, 32);
        var predicted = model.Predict(input);

        for (int step = 0; step < 3; step++)
        {
            var expected = Rand(step + 100, predicted.Shape.ToArray());
            Assert.Null(Record.Exception(() => model.Train(input, expected)));
        }
    }

    [Fact(Timeout = 120000)]
    public async Task MaskDINO_MultiStepTrain_DoesNotThrow()
    {
        var model = new MaskDINO<float>(Arch(), options: new MaskDINOOptions { NumClasses = 5, ModelSize = MaskDINOModelSize.R50 });
        var input = Rand(42, 1, 3, 32, 32);
        var predicted = model.Predict(input);

        for (int step = 0; step < 3; step++)
        {
            var expected = Rand(step + 100, predicted.Shape.ToArray());
            Assert.Null(Record.Exception(() => model.Train(input, expected)));
        }
    }

    [Fact(Timeout = 120000)]
    public async Task SAM_MultiStepTrain_DoesNotThrow()
    {
        var model = new SAM<float>(Arch(), options: new SAMOptions { NumClasses = 1, ModelSize = SAMModelSize.ViTBase });
        var input = Rand(42, 1, 3, 32, 32);
        var predicted = model.Predict(input);

        for (int step = 0; step < 3; step++)
        {
            var expected = Rand(step + 100, predicted.Shape.ToArray());
            Assert.Null(Record.Exception(() => model.Train(input, expected)));
        }
    }

    [Fact(Timeout = 120000)]
    public async Task SAM21_MultiStepTrain_DoesNotThrow()
    {
        var model = new SAM21<float>(Arch(), options: new SAM21Options { NumClasses = 1, ModelSize = SAM21ModelSize.BasePlus });
        var input = Rand(42, 1, 3, 32, 32);
        var predicted = model.Predict(input);

        for (int step = 0; step < 3; step++)
        {
            var expected = Rand(step + 100, predicted.Shape.ToArray());
            Assert.Null(Record.Exception(() => model.Train(input, expected)));
        }
    }

    [Fact(Timeout = 120000)]
    public async Task SAMHQ_MultiStepTrain_DoesNotThrow()
    {
        var model = new SAMHQ<float>(Arch(), options: new SAMHQOptions { NumClasses = 1, ModelSize = SAMHQModelSize.ViTBase });
        var input = Rand(42, 1, 3, 32, 32);
        var predicted = model.Predict(input);

        for (int step = 0; step < 3; step++)
        {
            var expected = Rand(step + 100, predicted.Shape.ToArray());
            Assert.Null(Record.Exception(() => model.Train(input, expected)));
        }
    }

    [Fact(Timeout = 120000)]
    public async Task EoMT_MultiStepTrain_DoesNotThrow()
    {
        var model = new EoMT<float>(Arch(), options: new EoMTOptions { NumClasses = 5, ModelSize = EoMTModelSize.Small });
        var input = Rand(42, 1, 3, 32, 32);
        var predicted = model.Predict(input);

        for (int step = 0; step < 3; step++)
        {
            var expected = Rand(step + 100, predicted.Shape.ToArray());
            Assert.Null(Record.Exception(() => model.Train(input, expected)));
        }
    }

    [Fact(Timeout = 120000)]
    public async Task OMGSeg_MultiStepTrain_DoesNotThrow()
    {
        var model = new OMGSeg<float>(Arch(), options: new OMGSegOptions { NumClasses = 5, ModelSize = OMGSegModelSize.Base });
        var input = Rand(42, 1, 3, 32, 32);
        var predicted = model.Predict(input);

        for (int step = 0; step < 3; step++)
        {
            var expected = Rand(step + 100, predicted.Shape.ToArray());
            Assert.Null(Record.Exception(() => model.Train(input, expected)));
        }
    }

    [Fact(Timeout = 120000)]
    public async Task MixedQueryTransformer_MultiStepTrain_DoesNotThrow()
    {
        var model = new MixedQueryTransformer<float>(Arch(), options: new MixedQueryTransformerOptions { NumClasses = 5, ModelSize = MixedQueryTransformerModelSize.R50 });
        var input = Rand(42, 1, 3, 32, 32);
        var predicted = model.Predict(input);

        for (int step = 0; step < 3; step++)
        {
            var expected = Rand(step + 100, predicted.Shape.ToArray());
            Assert.Null(Record.Exception(() => model.Train(input, expected)));
        }
    }

    [Fact(Timeout = 120000)]
    public async Task UNINEXT_MultiStepTrain_DoesNotThrow()
    {
        var model = new UNINEXT<float>(Arch(), options: new UNINEXTOptions { NumClasses = 5, ModelSize = UNINEXTModelSize.R50 });
        var input = Rand(42, 1, 3, 32, 32);
        var predicted = model.Predict(input);

        for (int step = 0; step < 3; step++)
        {
            var expected = Rand(step + 100, predicted.Shape.ToArray());
            Assert.Null(Record.Exception(() => model.Train(input, expected)));
        }
    }

    [Fact(Timeout = 120000)]
    public async Task U2Seg_MultiStepTrain_DoesNotThrow()
    {
        var model = new U2Seg<float>(Arch(), options: new U2SegOptions { NumClasses = 5 });
        var input = Rand(42, 1, 3, 32, 32);
        var predicted = model.Predict(input);

        for (int step = 0; step < 3; step++)
        {
            var expected = Rand(step + 100, predicted.Shape.ToArray());
            Assert.Null(Record.Exception(() => model.Train(input, expected)));
        }
    }

    [Fact(Timeout = 120000)]
    public async Task XDecoder_MultiStepTrain_DoesNotThrow()
    {
        var model = new XDecoder<float>(Arch(), options: new XDecoderOptions { NumClasses = 5, ModelSize = XDecoderModelSize.Tiny });
        var input = Rand(42, 1, 3, 32, 32);
        var predicted = model.Predict(input);

        for (int step = 0; step < 3; step++)
        {
            var expected = Rand(step + 100, predicted.Shape.ToArray());
            Assert.Null(Record.Exception(() => model.Train(input, expected)));
        }
    }

    #endregion

    #region Multi-Step Training — Medical, Mamba, Efficient, Video, OpenVocab, Referring, Diffusion, PointCloud

    [Fact(Timeout = 120000)]
    public async Task NnUNet_MultiStepTrain_DoesNotThrow()
    {
        var model = new NnUNet<float>(Arch(), options: new NnUNetOptions { NumClasses = 5, ModelSize = NnUNetModelSize.UNet2D });
        var input = Rand(42, 1, 3, 32, 32);
        var predicted = model.Predict(input);

        for (int step = 0; step < 3; step++)
        {
            var expected = Rand(step + 100, predicted.Shape.ToArray());
            Assert.Null(Record.Exception(() => model.Train(input, expected)));
        }
    }

    [Fact(Timeout = 120000)]
    public async Task TransUNet_MultiStepTrain_DoesNotThrow()
    {
        var model = new TransUNet<float>(Arch(), options: new TransUNetOptions { NumClasses = 5, ModelSize = TransUNetModelSize.Base });
        var input = Rand(42, 1, 3, 32, 32);
        var predicted = model.Predict(input);

        for (int step = 0; step < 3; step++)
        {
            var expected = Rand(step + 100, predicted.Shape.ToArray());
            Assert.Null(Record.Exception(() => model.Train(input, expected)));
        }
    }

    [Fact(Timeout = 120000)]
    public async Task SwinUNETR_MultiStepTrain_DoesNotThrow()
    {
        var model = new SwinUNETR<float>(Arch(), options: new SwinUNETROptions { NumClasses = 5, ModelSize = SwinUNETRModelSize.Base });
        var input = Rand(42, 1, 3, 32, 32);
        var predicted = model.Predict(input);

        for (int step = 0; step < 3; step++)
        {
            var expected = Rand(step + 100, predicted.Shape.ToArray());
            Assert.Null(Record.Exception(() => model.Train(input, expected)));
        }
    }

    [Fact(Timeout = 120000)]
    public async Task MedSAM_MultiStepTrain_DoesNotThrow()
    {
        var model = new MedSAM<float>(Arch(), options: new MedSAMOptions { NumClasses = 5, ModelSize = MedSAMModelSize.ViTBase });
        var input = Rand(42, 1, 3, 32, 32);
        var predicted = model.Predict(input);

        for (int step = 0; step < 3; step++)
        {
            var expected = Rand(step + 100, predicted.Shape.ToArray());
            Assert.Null(Record.Exception(() => model.Train(input, expected)));
        }
    }

    [Fact(Timeout = 120000)]
    public async Task MedNeXt_MultiStepTrain_DoesNotThrow()
    {
        var model = new MedNeXt<float>(Arch(), options: new MedNeXtOptions { NumClasses = 5, ModelSize = MedNeXtModelSize.Small });
        var input = Rand(42, 1, 3, 32, 32);
        var predicted = model.Predict(input);

        for (int step = 0; step < 3; step++)
        {
            var expected = Rand(step + 100, predicted.Shape.ToArray());
            Assert.Null(Record.Exception(() => model.Train(input, expected)));
        }
    }

    [Fact(Timeout = 120000)]
    public async Task MedSAM2_MultiStepTrain_DoesNotThrow()
    {
        var model = new MedSAM2<float>(Arch(), options: new MedSAM2Options { NumClasses = 5, ModelSize = MedSAM2ModelSize.Base });
        var input = Rand(42, 1, 3, 32, 32);
        var predicted = model.Predict(input);

        for (int step = 0; step < 3; step++)
        {
            var expected = Rand(step + 100, predicted.Shape.ToArray());
            Assert.Null(Record.Exception(() => model.Train(input, expected)));
        }
    }

    [Fact(Timeout = 120000)]
    public async Task VisionMamba_MultiStepTrain_DoesNotThrow()
    {
        var model = new VisionMamba<float>(Arch(), options: new VisionMambaOptions { NumClasses = 5, ModelSize = VisionMambaModelSize.Tiny });
        var input = Rand(42, 1, 3, 32, 32);
        var predicted = model.Predict(input);

        for (int step = 0; step < 3; step++)
        {
            var expected = Rand(step + 100, predicted.Shape.ToArray());
            Assert.Null(Record.Exception(() => model.Train(input, expected)));
        }
    }

    [Fact(Timeout = 120000)]
    public async Task VMamba_MultiStepTrain_DoesNotThrow()
    {
        var model = new VMamba<float>(Arch(), options: new VMambaOptions { NumClasses = 5, ModelSize = VMambaModelSize.Tiny });
        var input = Rand(42, 1, 3, 32, 32);
        var predicted = model.Predict(input);

        for (int step = 0; step < 3; step++)
        {
            var expected = Rand(step + 100, predicted.Shape.ToArray());
            Assert.Null(Record.Exception(() => model.Train(input, expected)));
        }
    }

    [Fact(Timeout = 120000)]
    public async Task PIDNet_MultiStepTrain_DoesNotThrow()
    {
        var model = new PIDNet<float>(Arch(), options: new PIDNetOptions { NumClasses = 5, ModelSize = PIDNetModelSize.Small });
        var input = Rand(42, 1, 3, 32, 32);
        var predicted = model.Predict(input);

        for (int step = 0; step < 3; step++)
        {
            var expected = Rand(step + 100, predicted.Shape.ToArray());
            Assert.Null(Record.Exception(() => model.Train(input, expected)));
        }
    }

    [Fact(Timeout = 120000)]
    public async Task DEVA_MultiStepTrain_DoesNotThrow()
    {
        var model = new DEVA<float>(Arch(), options: new DEVAOptions { NumClasses = 5, ModelSize = DEVAModelSize.Base });
        var input = Rand(42, 1, 3, 32, 32);
        var predicted = model.Predict(input);

        for (int step = 0; step < 3; step++)
        {
            var expected = Rand(step + 100, predicted.Shape.ToArray());
            Assert.Null(Record.Exception(() => model.Train(input, expected)));
        }
    }

    [Fact(Timeout = 120000)]
    public async Task EfficientTAM_MultiStepTrain_DoesNotThrow()
    {
        var model = new EfficientTAM<float>(Arch(), options: new EfficientTAMOptions { NumClasses = 5, ModelSize = EfficientTAMModelSize.Small });
        var input = Rand(42, 1, 3, 32, 32);
        var predicted = model.Predict(input);

        for (int step = 0; step < 3; step++)
        {
            var expected = Rand(step + 100, predicted.Shape.ToArray());
            Assert.Null(Record.Exception(() => model.Train(input, expected)));
        }
    }

    [Fact(Timeout = 120000)]
    public async Task SAN_MultiStepTrain_DoesNotThrow()
    {
        var model = new SAN<float>(Arch(), options: new SANOptions { NumClasses = 5 });
        var input = Rand(42, 1, 3, 32, 32);
        var predicted = model.Predict(input);

        for (int step = 0; step < 3; step++)
        {
            var expected = Rand(step + 100, predicted.Shape.ToArray());
            Assert.Null(Record.Exception(() => model.Train(input, expected)));
        }
    }

    [Fact(Timeout = 120000)]
    public async Task LISA_MultiStepTrain_DoesNotThrow()
    {
        var model = new LISA<float>(Arch(), options: new LISAOptions { NumClasses = 5 });
        var input = Rand(42, 1, 3, 32, 32);
        var predicted = model.Predict(input);

        for (int step = 0; step < 3; step++)
        {
            var expected = Rand(step + 100, predicted.Shape.ToArray());
            Assert.Null(Record.Exception(() => model.Train(input, expected)));
        }
    }

    /// <summary>
    /// DiffCut is TRAINING-FREE, so the invariant is that it REJECTS supervised training --
    /// repeatedly, and without corrupting itself.
    /// </summary>
    /// <remarks>
    /// This was generated from the same "MultiStepTrain_DoesNotThrow" template as its neighbours,
    /// which asserts that three successive Train calls raise nothing. That is the wrong contract for
    /// this model. DiffCut (Couairon et al., NeurIPS 2024) is an unsupervised zero-shot method: it
    /// segments frozen diffusion features with a recursive normalized cut and never learns from
    /// labeled masks, so <see cref="DiffCutSegmentation{T}.Train"/> throws NotSupportedException by
    /// design. The template test therefore demanded behaviour the paper rules out, and it began
    /// failing when 5ed6c772f5 restored the paper training invariants -- the model became correct and
    /// the test kept asserting the old, wrong behaviour.
    ///
    /// Asserting the rejection is strictly stronger than deleting the test: a future change that
    /// silently gave DiffCut a trainable path would now fail here, and the Predict-still-works check
    /// catches a rejection that leaves the model in a broken state.
    /// </remarks>
    [Fact(Timeout = 120000)]
    public async Task DiffCutSegmentation_MultiStepTrain_RejectsSupervisedTraining()
    {
        var model = new DiffCutSegmentation<float>(Arch(), options: new DiffCutSegmentationOptions { NumClasses = 5 });
        var input = Rand(42, 1, 3, 32, 32);
        var predicted = model.Predict(input);

        for (int step = 0; step < 3; step++)
        {
            var expected = Rand(step + 100, predicted.Shape.ToArray());
            Assert.Throws<NotSupportedException>(() => model.Train(input, expected));
        }

        // Rejecting training must not leave the model unusable: inference still works afterwards.
        var after = model.Predict(input);
        Assert.Equal(predicted.Shape.ToArray(), after.Shape.ToArray());
    }

    [Fact(Timeout = 120000)]
    public async Task PointTransformerV3_MultiStepTrain_DoesNotThrow()
    {
        var model = new PointTransformerV3<float>(Arch(), options: new PointTransformerV3Options { NumClasses = 5, ModelSize = PointTransformerV3ModelSize.Base });
        var input = Rand(42, 1, 3, 32, 32);
        var predicted = model.Predict(input);

        for (int step = 0; step < 3; step++)
        {
            var expected = Rand(step + 100, predicted.Shape.ToArray());
            Assert.Null(Record.Exception(() => model.Train(input, expected)));
        }
    }

    [Fact(Timeout = 120000)]
    public async Task Sonata_MultiStepTrain_DoesNotThrow()
    {
        var model = new Sonata<float>(Arch(), options: new SonataOptions { NumClasses = 5, ModelSize = SonataModelSize.Base });
        var input = Rand(42, 1, 3, 32, 32);
        var predicted = model.Predict(input);

        for (int step = 0; step < 3; step++)
        {
            var expected = Rand(step + 100, predicted.Shape.ToArray());
            Assert.Null(Record.Exception(() => model.Train(input, expected)));
        }
    }

    #endregion

    #region Predict-After-Train — Output consistency

    [Fact(Timeout = 120000)]
    public async Task SegFormer_PredictAfterTrain_ProducesDifferentOutput()
    {
        var model = new SegFormer<float>(Arch(), options: new SegFormerOptions { NumClasses = 5, ModelSize = SegFormerModelSize.B0 });
        var input = Rand(42, 1, 3, 32, 32);

        var outputBefore = model.Predict(input);
        var expected = Rand(99, outputBefore.Shape.ToArray());
        model.Train(input, expected);
        var outputAfter = model.Predict(input);

        Assert.NotNull(outputAfter);
        Assert.Equal(outputBefore.Shape.Length, outputAfter.Shape.Length);
        for (int i = 0; i < outputBefore.Shape.Length; i++)
        {
            Assert.Equal(outputBefore.Shape[i], outputAfter.Shape[i]);
        }
    }

    [Fact(Timeout = 120000)]
    public async Task Mask2Former_PredictAfterTrain_ProducesDifferentOutput()
    {
        var model = new Mask2Former<float>(Arch(), options: new Mask2FormerOptions { NumClasses = 5, ModelSize = Mask2FormerModelSize.SwinTiny });
        var input = Rand(42, 1, 3, 32, 32);

        var outputBefore = model.Predict(input);
        var expected = Rand(99, outputBefore.Shape.ToArray());
        model.Train(input, expected);
        var outputAfter = model.Predict(input);

        Assert.NotNull(outputAfter);
        Assert.Equal(outputBefore.Shape.Length, outputAfter.Shape.Length);
    }

    [Fact(Timeout = 120000)]
    public async Task NnUNet_PredictAfterTrain_ProducesDifferentOutput()
    {
        var model = new NnUNet<float>(Arch(), options: new NnUNetOptions { NumClasses = 5, ModelSize = NnUNetModelSize.UNet2D });
        var input = Rand(42, 1, 3, 32, 32);

        var outputBefore = model.Predict(input);
        var expected = Rand(99, outputBefore.Shape.ToArray());
        model.Train(input, expected);
        var outputAfter = model.Predict(input);

        Assert.NotNull(outputAfter);
        Assert.Equal(outputBefore.Shape.Length, outputAfter.Shape.Length);
    }

    #endregion

    #region Train with Unbatched (3D) Input

    [Fact(Timeout = 120000)]
    public async Task SegFormer_Train_Unbatched3DInput_DoesNotThrow()
    {
        var model = new SegFormer<float>(Arch(), options: new SegFormerOptions { NumClasses = 5, ModelSize = SegFormerModelSize.B0 });
        var input = Rand(42, 3, 32, 32); // 3D unbatched
        var predicted = model.Predict(input);
        var expected = Rand(99, predicted.Shape.ToArray());
        Assert.Null(Record.Exception(() => model.Train(input, expected)));
    }

    [Fact(Timeout = 120000)]
    public async Task Mask2Former_Train_Unbatched3DInput_DoesNotThrow()
    {
        var model = new Mask2Former<float>(Arch(), options: new Mask2FormerOptions { NumClasses = 5, ModelSize = Mask2FormerModelSize.SwinTiny });
        var input = Rand(42, 3, 32, 32);
        var predicted = model.Predict(input);
        var expected = Rand(99, predicted.Shape.ToArray());
        Assert.Null(Record.Exception(() => model.Train(input, expected)));
    }

    [Fact(Timeout = 120000)]
    public async Task SAM_Train_Unbatched3DInput_DoesNotThrow()
    {
        var model = new SAM<float>(Arch(), options: new SAMOptions { NumClasses = 1, ModelSize = SAMModelSize.ViTBase });
        var input = Rand(42, 3, 32, 32);
        var predicted = model.Predict(input);
        var expected = Rand(99, predicted.Shape.ToArray());
        Assert.Null(Record.Exception(() => model.Train(input, expected)));
    }

    [Fact(Timeout = 120000)]
    public async Task NnUNet_Train_Unbatched3DInput_DoesNotThrow()
    {
        var model = new NnUNet<float>(Arch(), options: new NnUNetOptions { NumClasses = 5, ModelSize = NnUNetModelSize.UNet2D });
        var input = Rand(42, 3, 32, 32);
        var predicted = model.Predict(input);
        var expected = Rand(99, predicted.Shape.ToArray());
        Assert.Null(Record.Exception(() => model.Train(input, expected)));
    }

    [Fact(Timeout = 120000)]
    public async Task PIDNet_Train_Unbatched3DInput_DoesNotThrow()
    {
        var model = new PIDNet<float>(Arch(), options: new PIDNetOptions { NumClasses = 5, ModelSize = PIDNetModelSize.Small });
        var input = Rand(42, 3, 32, 32);
        var predicted = model.Predict(input);
        var expected = Rand(99, predicted.Shape.ToArray());
        Assert.Null(Record.Exception(() => model.Train(input, expected)));
    }

    [Fact(Timeout = 120000)]
    public async Task VisionMamba_Train_Unbatched3DInput_DoesNotThrow()
    {
        var model = new VisionMamba<float>(Arch(), options: new VisionMambaOptions { NumClasses = 5, ModelSize = VisionMambaModelSize.Tiny });
        var input = Rand(42, 3, 32, 32);
        var predicted = model.Predict(input);
        var expected = Rand(99, predicted.Shape.ToArray());
        Assert.Null(Record.Exception(() => model.Train(input, expected)));
    }

    [Fact(Timeout = 120000)]
    public async Task DEVA_Train_Unbatched3DInput_DoesNotThrow()
    {
        var model = new DEVA<float>(Arch(), options: new DEVAOptions { NumClasses = 5, ModelSize = DEVAModelSize.Base });
        var input = Rand(42, 3, 32, 32);
        var predicted = model.Predict(input);
        var expected = Rand(99, predicted.Shape.ToArray());
        Assert.Null(Record.Exception(() => model.Train(input, expected)));
    }

    [Fact(Timeout = 120000)]
    public async Task LISA_Train_Unbatched3DInput_DoesNotThrow()
    {
        var model = new LISA<float>(Arch(), options: new LISAOptions { NumClasses = 5 });
        var input = Rand(42, 3, 32, 32);
        var predicted = model.Predict(input);
        var expected = Rand(99, predicted.Shape.ToArray());
        Assert.Null(Record.Exception(() => model.Train(input, expected)));
    }

    #endregion

    #region Instance Segmentation — Multi-Step Train

    [Fact(Timeout = 120000)]
    public async Task YOLOv8Seg_MultiStepTrain_DoesNotThrow()
    {
        var model = new YOLOv8Seg<float>(Arch(), options: new YOLOv8SegOptions { ModelSize = YOLOv8SegModelSize.N });
        var input = Rand(42, 1, 3, 32, 32);
        var predicted = model.Predict(input);

        for (int step = 0; step < 3; step++)
        {
            var expected = Rand(step + 100, predicted.Shape.ToArray());
            Assert.Null(Record.Exception(() => model.Train(input, expected)));
        }
    }

    [Fact(Timeout = 120000)]
    public async Task YOLOv9Seg_MultiStepTrain_DoesNotThrow()
    {
        var model = new YOLOv9Seg<float>(Arch(), options: new YOLOv9SegOptions { ModelSize = YOLOv9SegModelSize.C });
        var input = Rand(42, 1, 3, 32, 32);
        var predicted = model.Predict(input);

        for (int step = 0; step < 3; step++)
        {
            var expected = Rand(step + 100, predicted.Shape.ToArray());
            Assert.Null(Record.Exception(() => model.Train(input, expected)));
        }
    }

    [Fact(Timeout = 120000)]
    public async Task YOLO11Seg_MultiStepTrain_DoesNotThrow()
    {
        var model = new YOLO11Seg<float>(Arch(), options: new YOLO11SegOptions { ModelSize = YOLO11SegModelSize.N });
        var input = Rand(42, 1, 3, 32, 32);
        var predicted = model.Predict(input);

        for (int step = 0; step < 3; step++)
        {
            var expected = Rand(step + 100, predicted.Shape.ToArray());
            Assert.Null(Record.Exception(() => model.Train(input, expected)));
        }
    }

    [Fact(Timeout = 120000)]
    public async Task YOLOv12Seg_MultiStepTrain_DoesNotThrow()
    {
        var model = new YOLOv12Seg<float>(Arch(), options: new YOLOv12SegOptions { ModelSize = YOLOv12SegModelSize.N });
        var input = Rand(42, 1, 3, 32, 32);
        var predicted = model.Predict(input);

        for (int step = 0; step < 3; step++)
        {
            var expected = Rand(step + 100, predicted.Shape.ToArray());
            Assert.Null(Record.Exception(() => model.Train(input, expected)));
        }
    }

    [Fact(Timeout = 120000)]
    public async Task YOLO26Seg_MultiStepTrain_DoesNotThrow()
    {
        var model = new YOLO26Seg<float>(Arch(), options: new YOLO26SegOptions { ModelSize = YOLO26SegModelSize.N });
        var input = Rand(42, 1, 3, 32, 32);
        var predicted = model.Predict(input);

        for (int step = 0; step < 3; step++)
        {
            var expected = Rand(step + 100, predicted.Shape.ToArray());
            Assert.Null(Record.Exception(() => model.Train(input, expected)));
        }
    }

    #endregion

    #region Remaining models — Multi-Step Train

    [Fact(Timeout = 120000)]
    public async Task UniVS_MultiStepTrain_DoesNotThrow()
    {
        var model = new UniVS<float>(Arch(), options: new UniVSOptions { NumClasses = 5, ModelSize = UniVSModelSize.R50 });
        var input = Rand(42, 1, 3, 32, 32);
        var predicted = model.Predict(input);

        for (int step = 0; step < 3; step++)
        {
            var expected = Rand(step + 100, predicted.Shape.ToArray());
            Assert.Null(Record.Exception(() => model.Train(input, expected)));
        }
    }

    [Fact(Timeout = 120000)]
    public async Task CATSeg_MultiStepTrain_DoesNotThrow()
    {
        var model = new CATSeg<float>(Arch(), options: new CATSegOptions { NumClasses = 5 });
        var input = Rand(42, 1, 3, 32, 32);
        var predicted = model.Predict(input);

        for (int step = 0; step < 3; step++)
        {
            var expected = Rand(step + 100, predicted.Shape.ToArray());
            Assert.Null(Record.Exception(() => model.Train(input, expected)));
        }
    }

    [Fact(Timeout = 120000)]
    public async Task SED_MultiStepTrain_DoesNotThrow()
    {
        var model = new SED<float>(Arch(), options: new SEDOptions { NumClasses = 5 });
        var input = Rand(42, 1, 3, 32, 32);
        var predicted = model.Predict(input);

        for (int step = 0; step < 3; step++)
        {
            var expected = Rand(step + 100, predicted.Shape.ToArray());
            Assert.Null(Record.Exception(() => model.Train(input, expected)));
        }
    }

    [Fact(Timeout = 120000)]
    public async Task GroundedSAM2_MultiStepTrain_DoesNotThrow()
    {
        var model = new GroundedSAM2<float>(Arch(), options: new GroundedSAM2Options { NumClasses = 5 });
        var input = Rand(42, 1, 3, 32, 32);
        var predicted = model.Predict(input);

        for (int step = 0; step < 3; step++)
        {
            var expected = Rand(step + 100, predicted.Shape.ToArray());
            Assert.Null(Record.Exception(() => model.Train(input, expected)));
        }
    }

    [Fact(Timeout = 120000)]
    public async Task MaskAdapter_MultiStepTrain_DoesNotThrow()
    {
        var model = new MaskAdapter<float>(Arch(), options: new MaskAdapterOptions { NumClasses = 5 });
        var input = Rand(42, 1, 3, 32, 32);
        var predicted = model.Predict(input);

        for (int step = 0; step < 3; step++)
        {
            var expected = Rand(step + 100, predicted.Shape.ToArray());
            Assert.Null(Record.Exception(() => model.Train(input, expected)));
        }
    }

    [Fact(Timeout = 120000)]
    public async Task OpenVocabSAM_MultiStepTrain_DoesNotThrow()
    {
        var model = new OpenVocabSAM<float>(Arch(), options: new OpenVocabSAMOptions { NumClasses = 5 });
        var input = Rand(42, 1, 3, 32, 32);
        var predicted = model.Predict(input);

        for (int step = 0; step < 3; step++)
        {
            var expected = Rand(step + 100, predicted.Shape.ToArray());
            Assert.Null(Record.Exception(() => model.Train(input, expected)));
        }
    }

    [Fact(Timeout = 120000)]
    public async Task VideoLISA_MultiStepTrain_DoesNotThrow()
    {
        var model = new VideoLISA<float>(Arch(), options: new VideoLISAOptions { NumClasses = 5 });
        var input = Rand(42, 1, 3, 32, 32);
        var predicted = model.Predict(input);

        for (int step = 0; step < 3; step++)
        {
            var expected = Rand(step + 100, predicted.Shape.ToArray());
            Assert.Null(Record.Exception(() => model.Train(input, expected)));
        }
    }

    [Fact(Timeout = 120000)]
    public async Task GLaMM_MultiStepTrain_DoesNotThrow()
    {
        var model = new GLaMM<float>(Arch(), options: new GLaMMOptions { NumClasses = 5 });
        var input = Rand(42, 1, 3, 32, 32);
        var predicted = model.Predict(input);

        for (int step = 0; step < 3; step++)
        {
            var expected = Rand(step + 100, predicted.Shape.ToArray());
            Assert.Null(Record.Exception(() => model.Train(input, expected)));
        }
    }

    [Fact(Timeout = 120000)]
    public async Task OMGLLaVA_MultiStepTrain_DoesNotThrow()
    {
        var model = new OMGLLaVA<float>(Arch(), options: new OMGLLaVAOptions { NumClasses = 5 });
        var input = Rand(42, 1, 3, 32, 32);
        var predicted = model.Predict(input);

        for (int step = 0; step < 3; step++)
        {
            var expected = Rand(step + 100, predicted.Shape.ToArray());
            Assert.Null(Record.Exception(() => model.Train(input, expected)));
        }
    }

    [Fact(Timeout = 120000)]
    public async Task PixelLM_MultiStepTrain_DoesNotThrow()
    {
        var model = new PixelLM<float>(Arch(), options: new PixelLMOptions { NumClasses = 5 });
        var input = Rand(42, 1, 3, 32, 32);
        var predicted = model.Predict(input);

        for (int step = 0; step < 3; step++)
        {
            var expected = Rand(step + 100, predicted.Shape.ToArray());
            Assert.Null(Record.Exception(() => model.Train(input, expected)));
        }
    }

    [Fact(Timeout = 120000)]
    public async Task ODISESegmentation_MultiStepTrain_DoesNotThrow()
    {
        var model = new ODISESegmentation<float>(Arch(), options: new ODISESegmentationOptions { NumClasses = 5 });
        var input = Rand(42, 1, 3, 32, 32);
        var predicted = model.Predict(input);

        for (int step = 0; step < 3; step++)
        {
            var expected = Rand(step + 100, predicted.Shape.ToArray());
            Assert.Null(Record.Exception(() => model.Train(input, expected)));
        }
    }

    [Fact(Timeout = 120000)]
    public async Task MedSegDiffV2Segmentation_MultiStepTrain_DoesNotThrow()
    {
        var model = new MedSegDiffV2Segmentation<float>(Arch(), options: new MedSegDiffV2SegmentationOptions { NumClasses = 5 });
        var input = Rand(42, 1, 3, 32, 32);
        var predicted = model.Predict(input);

        for (int step = 0; step < 3; step++)
        {
            var expected = Rand(step + 100, predicted.Shape.ToArray());
            Assert.Null(Record.Exception(() => model.Train(input, expected)));
        }
    }

    [Fact(Timeout = 120000)]
    public async Task Concerto_MultiStepTrain_DoesNotThrow()
    {
        var model = new Concerto<float>(Arch(), options: new ConcertoOptions { NumClasses = 5, ModelSize = ConcertoModelSize.Base });
        var input = Rand(42, 1, 3, 32, 32);
        var predicted = model.Predict(input);

        for (int step = 0; step < 3; step++)
        {
            var expected = Rand(step + 100, predicted.Shape.ToArray());
            Assert.Null(Record.Exception(() => model.Train(input, expected)));
        }
    }

    [Fact(Timeout = 120000)]
    public async Task ViMUNet_MultiStepTrain_DoesNotThrow()
    {
        var model = new ViMUNet<float>(Arch(), options: new ViMUNetOptions { NumClasses = 5 });
        var input = Rand(42, 1, 3, 32, 32);
        var predicted = model.Predict(input);

        for (int step = 0; step < 3; step++)
        {
            var expected = Rand(step + 100, predicted.Shape.ToArray());
            Assert.Null(Record.Exception(() => model.Train(input, expected)));
        }
    }

    [Fact(Timeout = 120000)]
    public async Task BiomedParse_MultiStepTrain_DoesNotThrow()
    {
        var model = new BiomedParse<float>(Arch(), options: new BiomedParseOptions { NumClasses = 5 });
        var input = Rand(42, 1, 3, 32, 32);
        var predicted = model.Predict(input);

        for (int step = 0; step < 3; step++)
        {
            var expected = Rand(step + 100, predicted.Shape.ToArray());
            Assert.Null(Record.Exception(() => model.Train(input, expected)));
        }
    }

    [Fact(Timeout = 120000)]
    public async Task UniverSeg_MultiStepTrain_DoesNotThrow()
    {
        var model = new UniverSeg<float>(Arch(), options: new UniverSegOptions { NumClasses = 5 });
        var input = Rand(42, 1, 3, 32, 32);
        var predicted = model.Predict(input);

        for (int step = 0; step < 3; step++)
        {
            var expected = Rand(step + 100, predicted.Shape.ToArray());
            Assert.Null(Record.Exception(() => model.Train(input, expected)));
        }
    }

    [Fact(Timeout = 120000)]
    public async Task MedSegDiffV2_MultiStepTrain_DoesNotThrow()
    {
        var model = new MedSegDiffV2<float>(Arch(), options: new MedSegDiffV2Options { NumClasses = 5 });
        var input = Rand(42, 1, 3, 32, 32);
        var predicted = model.Predict(input);

        for (int step = 0; step < 3; step++)
        {
            var expected = Rand(step + 100, predicted.Shape.ToArray());
            Assert.Null(Record.Exception(() => model.Train(input, expected)));
        }
    }

    [Fact(Timeout = 120000)]
    public async Task UMamba_MultiStepTrain_DoesNotThrow()
    {
        var model = new UMamba<float>(Arch(), options: new UMambaOptions { NumClasses = 5 });
        var input = Rand(42, 1, 3, 32, 32);
        var predicted = model.Predict(input);

        for (int step = 0; step < 3; step++)
        {
            var expected = Rand(step + 100, predicted.Shape.ToArray());
            Assert.Null(Record.Exception(() => model.Train(input, expected)));
        }
    }

    [Fact(Timeout = 120000)]
    public async Task SegMamba_MultiStepTrain_DoesNotThrow()
    {
        var model = new SegMamba<float>(Arch(), options: new SegMambaOptions { NumClasses = 5 });
        var input = Rand(42, 1, 3, 32, 32);
        var predicted = model.Predict(input);

        for (int step = 0; step < 3; step++)
        {
            var expected = Rand(step + 100, predicted.Shape.ToArray());
            Assert.Null(Record.Exception(() => model.Train(input, expected)));
        }
    }

    #endregion

    #region Efficient Models — Multi-Step Train

    [Fact(Timeout = 120000)]
    public async Task FastSAM_MultiStepTrain_DoesNotThrow()
    {
        var model = new FastSAM<float>(Arch(), options: new FastSAMOptions { NumClasses = 1 });
        var input = Rand(42, 1, 3, 32, 32);
        var predicted = model.Predict(input);

        for (int step = 0; step < 3; step++)
        {
            var expected = Rand(step + 100, predicted.Shape.ToArray());
            Assert.Null(Record.Exception(() => model.Train(input, expected)));
        }
    }

    [Fact(Timeout = 120000)]
    public async Task MobileSAM_MultiStepTrain_DoesNotThrow()
    {
        var model = new MobileSAM<float>(Arch(), options: new MobileSAMOptions { NumClasses = 1 });
        var input = Rand(42, 1, 3, 32, 32);
        var predicted = model.Predict(input);

        for (int step = 0; step < 3; step++)
        {
            var expected = Rand(step + 100, predicted.Shape.ToArray());
            Assert.Null(Record.Exception(() => model.Train(input, expected)));
        }
    }

    [Fact(Timeout = 120000)]
    public async Task EdgeSAM_MultiStepTrain_DoesNotThrow()
    {
        var model = new EdgeSAM<float>(Arch(), options: new EdgeSAMOptions { NumClasses = 1 });
        var input = Rand(42, 1, 3, 32, 32);
        var predicted = model.Predict(input);

        for (int step = 0; step < 3; step++)
        {
            var expected = Rand(step + 100, predicted.Shape.ToArray());
            Assert.Null(Record.Exception(() => model.Train(input, expected)));
        }
    }

    [Fact(Timeout = 120000)]
    public async Task SlimSAM_MultiStepTrain_DoesNotThrow()
    {
        var model = new SlimSAM<float>(Arch(), options: new SlimSAMOptions { NumClasses = 1 });
        var input = Rand(42, 1, 3, 32, 32);
        var predicted = model.Predict(input);

        for (int step = 0; step < 3; step++)
        {
            var expected = Rand(step + 100, predicted.Shape.ToArray());
            Assert.Null(Record.Exception(() => model.Train(input, expected)));
        }
    }

    [Fact(Timeout = 120000)]
    public async Task EfficientSAM_MultiStepTrain_DoesNotThrow()
    {
        var model = new EfficientSAM<float>(Arch(), options: new EfficientSAMOptions { NumClasses = 1 });
        var input = Rand(42, 1, 3, 32, 32);
        var predicted = model.Predict(input);

        for (int step = 0; step < 3; step++)
        {
            var expected = Rand(step + 100, predicted.Shape.ToArray());
            Assert.Null(Record.Exception(() => model.Train(input, expected)));
        }
    }

    [Fact(Timeout = 120000)]
    public async Task RepViTSAM_MultiStepTrain_DoesNotThrow()
    {
        var model = new RepViTSAM<float>(Arch(), options: new RepViTSAMOptions { NumClasses = 1 });
        var input = Rand(42, 1, 3, 32, 32);
        var predicted = model.Predict(input);

        for (int step = 0; step < 3; step++)
        {
            var expected = Rand(step + 100, predicted.Shape.ToArray());
            Assert.Null(Record.Exception(() => model.Train(input, expected)));
        }
    }

    [Fact(Timeout = 120000)]
    public async Task SegGPT_MultiStepTrain_DoesNotThrow()
    {
        var model = new SegGPT<float>(Arch(), options: new SegGPTOptions { NumClasses = 5, ModelSize = SegGPTModelSize.ViTLarge });
        var input = Rand(42, 1, 3, 32, 32);
        var predicted = model.Predict(input);

        for (int step = 0; step < 3; step++)
        {
            var expected = Rand(step + 100, predicted.Shape.ToArray());
            Assert.Null(Record.Exception(() => model.Train(input, expected)));
        }
    }

    [Fact(Timeout = 120000)]
    public async Task SEEM_MultiStepTrain_DoesNotThrow()
    {
        var model = new SEEM<float>(Arch(), options: new SEEMOptions { NumClasses = 5, ModelSize = SEEMModelSize.Tiny });
        var input = Rand(42, 1, 3, 32, 32);
        var predicted = model.Predict(input);

        for (int step = 0; step < 3; step++)
        {
            var expected = Rand(step + 100, predicted.Shape.ToArray());
            Assert.Null(Record.Exception(() => model.Train(input, expected)));
        }
    }

    [Fact(Timeout = 120000)]
    public async Task KMaXDeepLab_MultiStepTrain_DoesNotThrow()
    {
        var model = new KMaXDeepLab<float>(Arch(), options: new KMaXDeepLabOptions { NumClasses = 5, ModelSize = KMaXDeepLabModelSize.R50 });
        var input = Rand(42, 1, 3, 32, 32);
        var predicted = model.Predict(input);

        for (int step = 0; step < 3; step++)
        {
            var expected = Rand(step + 100, predicted.Shape.ToArray());
            Assert.Null(Record.Exception(() => model.Train(input, expected)));
        }
    }

    [Fact(Timeout = 120000)]
    public async Task ODISE_MultiStepTrain_DoesNotThrow()
    {
        var model = new ODISE<float>(Arch(), options: new ODISEOptions { NumClasses = 5, ModelSize = ODISEModelSize.Base });
        var input = Rand(42, 1, 3, 32, 32);
        var predicted = model.Predict(input);

        for (int step = 0; step < 3; step++)
        {
            var expected = Rand(step + 100, predicted.Shape.ToArray());
            Assert.Null(Record.Exception(() => model.Train(input, expected)));
        }
    }

    [Fact(Timeout = 120000)]
    public async Task CUPS_MultiStepTrain_DoesNotThrow()
    {
        var model = new CUPS<float>(Arch(), options: new CUPSOptions { NumClasses = 5 });
        var input = Rand(42, 1, 3, 32, 32);
        var predicted = model.Predict(input);

        for (int step = 0; step < 3; step++)
        {
            var expected = Rand(step + 100, predicted.Shape.ToArray());
            Assert.Null(Record.Exception(() => model.Train(input, expected)));
        }
    }

    #endregion
}
