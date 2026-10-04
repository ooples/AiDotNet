using System;
using System.Collections.Generic;
using System.Threading.Tasks;
using AiDotNet.Diffusion.Conditioning;
using AiDotNet.Enums;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tokenization;
using Xunit;

namespace AiDotNet.Tests.UnitTests.Diffusion.Conditioning;

/// <summary>
/// Regression for #2151: TextConditioningBase used to build its layer stack only on the first forward,
/// so before one every conditioner reported no layers and no parameters, and serialization, cloning and
/// named activations all saw an empty model. Every concrete conditioner now builds its stack in its own
/// constructor (weights stay lazy, so large variants still cost nothing until used).
/// </summary>
public class TextConditionerEagerLayersTests
{
    public static IEnumerable<object[]> Conditioners()
    {
        Func<NeuralNetworkBase<double>>[] factories =
        {
            () => new CLIPTextConditioner<double>(ClipTokenizerFactory.CreateSimple()),
            () => new SigLIPTextConditioner<double>(ClipTokenizerFactory.CreateSimple()),
            () => new SigLIP2TextConditioner<double>(ClipTokenizerFactory.CreateSimple()),
            () => new T5TextConditioner<double>(LanguageModelTokenizerFactory.CreateForBackbone(LanguageModelBackbone.FlanT5)),
            () => new DistilledT5TextConditioner<double>(LanguageModelTokenizerFactory.CreateForBackbone(LanguageModelBackbone.FlanT5)),
            () => new GemmaTextConditioner<double>(LanguageModelTokenizerFactory.CreateForBackbone(LanguageModelBackbone.LLaMA)),
            () => new Qwen2TextConditioner<double>(LanguageModelTokenizerFactory.CreateForBackbone(LanguageModelBackbone.Qwen)),
            () => new ChatGLM3TextConditioner<double>(LanguageModelTokenizerFactory.CreateForBackbone(LanguageModelBackbone.LLaMA)),
        };
        foreach (var factory in factories) yield return new object[] { factory };
    }

    [Theory(Timeout = 300000)]
    [MemberData(nameof(Conditioners))]
    public async Task Conditioner_HasItsLayerStackBeforeAnyForward(Func<NeuralNetworkBase<double>> create)
    {
        await Task.Yield();
        var conditioner = create();
        string name = conditioner.GetType().Name;

        Assert.True(conditioner.LayerCount > 0, $"{name} has no layers until a forward runs.");
    }
}
