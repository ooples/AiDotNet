using System.Reflection;
using AiDotNet.Models;
using AiDotNet.ProgramSynthesis.Enums;
using AiDotNet.ProgramSynthesis.Models;
using Xunit;
using System.Threading.Tasks;

namespace AiDotNet.Serving.Tests.ProgramSynthesis;

public sealed class ServingHeuristicCodeModelTests
{
    [Fact(Timeout = 60000)]
    public async Task GetModelMetadata_IncludesExpectedFields()
    {
        var model = CreateDefaultModel(ProgramLanguage.Generic);
        var metadata = (ModelMetadata<double>)(Invoke(model, "GetModelMetadata") ?? throw new InvalidOperationException("Invoke(model, \"GetModelMetadata\") is null where it is required."));

        Assert.NotNull(metadata);
        Assert.Equal("ServingHeuristicCodeModel", metadata.AdditionalInfo["ModelName"]);
        Assert.Equal(ProgramLanguage.Generic.ToString(), metadata.AdditionalInfo["TargetLanguage"]);
    }

    [Fact(Timeout = 60000)]
    public async Task Deserialize_MismatchedArchitecture_Throws()
    {
        using (AiDotNet.Helpers.ModelPersistenceGuard.InternalOperation())
        {
            var modelA = CreateDefaultModel(ProgramLanguage.Generic);
            var payload = (byte[])(Invoke(modelA, "Serialize") ?? throw new InvalidOperationException("Invoke(modelA, \"Serialize\") is null where it is required."));

            var modelB = CreateDefaultModel(ProgramLanguage.Python);

            var ex = Assert.Throws<TargetInvocationException>(() => Invoke(modelB, "Deserialize", payload));
            Assert.IsType<InvalidOperationException>(ex.InnerException);
        }
    }

    private static object CreateDefaultModel(ProgramLanguage language)
    {
        var type = Type.GetType("AiDotNet.Serving.ProgramSynthesis.ServingHeuristicCodeModel, AiDotNet.Serving");
        Assert.NotNull(type);

        var factory = type.GetMethod("CreateDefault", BindingFlags.Public | BindingFlags.Static);
        Assert.NotNull(factory);

        return (factory.Invoke(null, new object[] { language }) ?? throw new InvalidOperationException("factory.Invoke(null, new object[] { language }) is null where it is required."));
    }

    private static object? Invoke(object instance, string methodName, params object[] args)
    {
        var method = instance.GetType().GetMethod(methodName, BindingFlags.Public | BindingFlags.Instance);
        Assert.NotNull(method);
        return method.Invoke(instance, args);
    }
}
