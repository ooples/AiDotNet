using System;
using System.Collections.Generic;
using AiDotNet.Enums;
using AiDotNet.Factories;
using AiDotNet.Interfaces;
using AiDotNet.PromptEngineering.FewShot;
using AiDotNet.PromptEngineering.Templates;
using Xunit;

namespace AiDotNet.Tests.Factories
{
    public class PromptTemplateFactoryTests
    {
        [Fact]
        public void Create_Throws_For_FewShot_Without_Typed_Selector()
        {
            var ex = Assert.Throws<ArgumentException>(() =>
                PromptTemplateFactory.Create(PromptTemplateType.FewShot, "Base: {examples}\n{query}"));

            Assert.Equal("templateType", ex.ParamName);
        }

        [Fact]
        public void Create_Generic_FewShot_Works_For_NonDouble_Types()
        {
            var selector = new FixedExampleSelector<float>();
            selector.AddExample(new FewShotExample { Input = "Hello", Output = "Hola" });
            selector.AddExample(new FewShotExample { Input = "Goodbye", Output = "Adios" });

            var template = PromptTemplateFactory.Create<float>(
                PromptTemplateType.FewShot,
                "Translate:\n\n{examples}\n\nNow: {query}",
                selector,
                exampleCount: 1);

            Assert.IsType<FewShotPromptTemplate<float>>(template);

            var prompt = template.Format(new Dictionary<string, string>
            {
                { "query", "Good morning" }
            });

            Assert.Contains("Hello", prompt);
            Assert.Contains("Hola", prompt);
            Assert.Contains("Good morning", prompt);
        }
    }
}
