#pragma once

// Fixed diagnostic ABI. Callers hold CounterMutex while recording or copying these counters.
// The first seven values of SnapshotV2 are the original ABI1 snapshot, copied under the SAME lock.
namespace NumericArrays
{
    using Count = unsigned long long;
    constexpr Count MaxCount = ~static_cast<Count>(0);
    constexpr unsigned Version = 2;
    constexpr unsigned MaxRank = 32;
    constexpr unsigned DTypes = 14;
    enum Slot : unsigned
    {
        AbiVersion = 7, SnapshotLength, PrimitiveArrays, PayloadValidArrays, PayloadBytes, PrimitiveObjectBytes,
        ArrayInfoFailures, RankFailures, PayloadOverflows, PayloadExceedsObject, FailedPayloadObjectBytes,
        ReferenceArrays, ReferenceObjectBytes, UnsupportedArrays, UnsupportedObjectBytes,
        ClassificationFailures, ClassificationFailureObjectBytes, CounterOverflows, SizeFailures,
        BinsBegin
    };
    constexpr unsigned Length = BinsBegin + DTypes * MaxRank * 3;
    static Count Counters[Length] = {};

    enum class Kind { NonArray, Primitive, Reference, Unsupported, Unknown };
    enum class Failure { None, ArrayInfo, Rank, Overflow, ExceedsObject };
    struct Observation
    {
        Kind kind = Kind::Unknown;
        Failure failure = Failure::None;
        unsigned dtype = 0;
        ULONG rank = 0;
        Count payload = 0;
        bool valid = false;
    };

    inline void Add(unsigned slot, Count value)
    {
        if (MaxCount - Counters[slot] < value)
        {
            Counters[slot] = MaxCount;
            Counters[CounterOverflows] = 1;
        }
        else Counters[slot] += value;
    }

    // Includes bool/char: masks and character/byte workspaces must not disappear from a storage upper bound.
    // Value-type arrays (Half, Decimal, Complex, enums, structs) remain explicit conservative candidates.
    inline unsigned PrimitiveWidth(CorElementType type, unsigned& dtype)
    {
        switch (type)
        {
        case ELEMENT_TYPE_BOOLEAN: dtype = 0; return 1;
        case ELEMENT_TYPE_CHAR: dtype = 1; return 2;
        case ELEMENT_TYPE_I1: dtype = 2; return 1;
        case ELEMENT_TYPE_U1: dtype = 3; return 1;
        case ELEMENT_TYPE_I2: dtype = 4; return 2;
        case ELEMENT_TYPE_U2: dtype = 5; return 2;
        case ELEMENT_TYPE_I4: dtype = 6; return 4;
        case ELEMENT_TYPE_U4: dtype = 7; return 4;
        case ELEMENT_TYPE_I8: dtype = 8; return 8;
        case ELEMENT_TYPE_U8: dtype = 9; return 8;
        case ELEMENT_TYPE_R4: dtype = 10; return 4;
        case ELEMENT_TYPE_R8: dtype = 11; return 8;
        case ELEMENT_TYPE_I: dtype = 12; return static_cast<unsigned>(sizeof(INT_PTR));
        case ELEMENT_TYPE_U: dtype = 13; return static_cast<unsigned>(sizeof(UINT_PTR));
        default: return 0;
        }
    }

    inline bool ReferenceElements(CorElementType type)
    {
        return type == ELEMENT_TYPE_CLASS || type == ELEMENT_TYPE_OBJECT || type == ELEMENT_TYPE_STRING
            || type == ELEMENT_TYPE_ARRAY || type == ELEMENT_TYPE_SZARRAY;
    }

    // All CLR queries occur before the counter lock. Stack metadata only; never read/store the returned
    // payload pointer or retain ObjectID/ClassID past ObjectAllocated (objects can move after the callback).
    inline Observation Observe(ICorProfilerInfo4* info, ObjectID object, HRESULT classification,
        CorElementType elementType, ULONG rank, HRESULT sized, SIZE_T objectBytes)
    {
        Observation result;
        if (classification == S_FALSE) { result.kind = Kind::NonArray; return result; }
        if (classification != S_OK) return result;
        unsigned width = PrimitiveWidth(elementType, result.dtype);
        result.rank = rank;
        if (width == 0)
        {
            result.kind = ReferenceElements(elementType) ? Kind::Reference : Kind::Unsupported;
            return result;
        }
        result.kind = Kind::Primitive;
        if (sized != S_OK || objectBytes == 0) return result;
        if (rank == 0 || rank > MaxRank) { result.failure = Failure::Rank; return result; }
        ULONG32 dimensions[MaxRank] = {};
        int lowerBounds[MaxRank] = {};
        BYTE* unusedPayload = nullptr;
        if (info->GetArrayObjectInfo(object, rank, dimensions, lowerBounds, &unusedPayload) != S_OK)
        {
            result.failure = Failure::ArrayInfo;
            return result;
        }
        Count elements = 1;
        for (ULONG dimension = 0; dimension < rank; ++dimension)
        {
            const Count length = dimensions[dimension];
            if (length != 0 && elements > MaxCount / length)
            {
                result.failure = Failure::Overflow;
                return result;
            }
            elements *= length;
        }
        if (elements > MaxCount / width)
        {
            result.failure = Failure::Overflow;
            return result;
        }
        result.payload = elements * width;
        if (result.payload > static_cast<Count>(objectBytes))
        {
            result.failure = Failure::ExceedsObject;
            return result;
        }
        result.valid = true;
        return result;
    }

    inline void Record(const Observation& value, HRESULT sized, SIZE_T objectBytes)
    {
        const bool sizedExactly = sized == S_OK && objectBytes != 0;
        const Count size = sizedExactly ? static_cast<Count>(objectBytes) : 0;
        if (value.kind != Kind::NonArray && !sizedExactly) Add(SizeFailures, 1);
        switch (value.kind)
        {
        case Kind::NonArray: return;
        case Kind::Unknown:
            Add(ClassificationFailures, 1); Add(ClassificationFailureObjectBytes, size); return;
        case Kind::Reference:
            Add(ReferenceArrays, 1); Add(ReferenceObjectBytes, size); return;
        case Kind::Unsupported:
            Add(UnsupportedArrays, 1); Add(UnsupportedObjectBytes, size); return;
        case Kind::Primitive: break;
        }
        Add(PrimitiveArrays, 1);
        Add(PrimitiveObjectBytes, size);
        if (value.valid)
        {
            Add(PayloadValidArrays, 1); Add(PayloadBytes, value.payload);
            const unsigned bin = BinsBegin + (value.dtype * MaxRank + value.rank - 1) * 3;
            Add(bin, 1); Add(bin + 1, value.payload); Add(bin + 2, size);
            return;
        }
        Add(FailedPayloadObjectBytes, size);
        switch (value.failure)
        {
        case Failure::ArrayInfo: Add(ArrayInfoFailures, 1); break;
        case Failure::Rank: Add(RankFailures, 1); break;
        case Failure::Overflow: Add(PayloadOverflows, 1); break;
        case Failure::ExceedsObject: Add(PayloadExceedsObject, 1); break;
        case Failure::None: break; // GetObjectSize2 failure is already explicit in the legacy slots.
        }
    }
}
