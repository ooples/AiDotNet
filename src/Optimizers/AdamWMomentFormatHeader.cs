using System;
using System.IO;
using AiDotNet.Enums;

namespace AiDotNet.Optimizers;

internal static class AdamWMomentFormatHeader
{
    private const string Marker = "AdamWMomentFormat";
    private const int Version = 1;

    internal static void Write(BinaryWriter writer, OptimizerMomentStorage storage, int blockSize, int minLength)
    {
        writer.Write(Marker);
        writer.Write(Version);
        writer.Write((int)storage);
        writer.Write(blockSize);
        writer.Write(minLength);
    }

    internal static void Validate(BinaryReader reader, OptimizerMomentStorage storage, int blockSize, int minLength)
    {
        try
        {
            long remaining = reader.BaseStream.Length - reader.BaseStream.Position;
            if (remaining == 0)
            {
                if (storage != OptimizerMomentStorage.Float32)
                    throw new InvalidOperationException("AdamW checkpoint has no compressed-moment format header.");
                return;
            }
            if (remaining < 0 || reader.ReadString() != Marker || reader.ReadInt32() != Version)
                throw new InvalidOperationException("Unknown AdamW moment-format checkpoint header.");
            var savedStorage = (OptimizerMomentStorage)reader.ReadInt32();
            int savedBlockSize = reader.ReadInt32();
            int savedMinLength = reader.ReadInt32();
            if (savedStorage != storage || savedBlockSize != blockSize || savedMinLength != minLength)
                throw new InvalidOperationException("AdamW checkpoint moment format does not match the target optimizer options.");
            if (reader.BaseStream.Position < reader.BaseStream.Length)
                throw new InvalidOperationException("Unexpected trailing AdamW checkpoint data.");
        }
        catch (IOException error)
        {
            throw new InvalidOperationException("AdamW moment-format checkpoint header is truncated or corrupted.", error);
        }
    }
}
