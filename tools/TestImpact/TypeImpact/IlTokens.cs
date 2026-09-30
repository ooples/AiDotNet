using System.Reflection;
using System.Reflection.Emit;
using System.Reflection.Metadata;
using System.Reflection.Metadata.Ecma335;

namespace AiDotNet.TestImpact.TypeImpact;

/// <summary>
/// Enumerates the metadata tokens a method body's IL refers to. Only the operand shapes are
/// needed, so the opcode table is built once from <see cref="OpCodes"/> instead of being
/// hand-maintained.
/// </summary>
internal static class IlTokens
{
    private static readonly OperandType?[] OneByte = new OperandType?[256];
    private static readonly OperandType?[] TwoByte = new OperandType?[256];

    static IlTokens()
    {
        foreach (var field in typeof(OpCodes).GetFields(BindingFlags.Public | BindingFlags.Static))
        {
            if (field.GetValue(null) is not OpCode code)
            {
                continue;
            }

            ushort value = unchecked((ushort)code.Value);
            if (code.Size == 1)
            {
                OneByte[value & 0xFF] = code.OperandType;
            }
            else
            {
                TwoByte[value & 0xFF] = code.OperandType;
            }
        }
    }

    /// <summary>Every token operand in <paramref name="body"/>, in IL order.</summary>
    public static IEnumerable<EntityHandle> Scan(MethodBodyBlock body)
    {
        var reader = body.GetILReader();
        var tokens = new List<EntityHandle>();
        while (reader.RemainingBytes > 0)
        {
            byte first = reader.ReadByte();
            OperandType? operand = first == 0xFE
                ? TwoByte[reader.ReadByte()]
                : OneByte[first];
            if (operand is null)
            {
                throw new BadImageFormatException($"unknown IL opcode 0x{first:X2} at offset {reader.Offset - 1}");
            }

            switch (operand.Value)
            {
                case OperandType.InlineNone:
                    break;
                case OperandType.ShortInlineBrTarget:
                case OperandType.ShortInlineI:
                case OperandType.ShortInlineVar:
                    reader.Offset += 1;
                    break;
                case OperandType.InlineVar:
                    reader.Offset += 2;
                    break;
                case OperandType.InlineBrTarget:
                case OperandType.InlineI:
                case OperandType.ShortInlineR:
                case OperandType.InlineString:
                case OperandType.InlineSig:
                    reader.Offset += 4;
                    break;
                case OperandType.InlineI8:
                case OperandType.InlineR:
                    reader.Offset += 8;
                    break;
                case OperandType.InlineSwitch:
                    int count = reader.ReadInt32();
                    reader.Offset += count * 4;
                    break;
                case OperandType.InlineField:
                case OperandType.InlineMethod:
                case OperandType.InlineTok:
                case OperandType.InlineType:
                    int token = reader.ReadInt32();
                    var handle = MetadataTokens.EntityHandle(token);
                    if (!handle.IsNil)
                    {
                        tokens.Add(handle);
                    }

                    break;
                default:
                    throw new BadImageFormatException($"unhandled IL operand type {operand.Value}");
            }
        }

        return tokens;
    }
}
