// Diagnostic-only CLR allocation callbacks. No sampling, stack walking, IL rewrite or managed calls.
#include <windows.h>
#include <cor.h>
#include <corprof.h>
#include <atomic>
#include <cstdio>
#include <mutex>
#include <new>

// {5B735E8D-5547-4DB5-B407-E9AF3E2A9D15}
static const CLSID ProfilerClsid = {0x5b735e8d, 0x5547, 0x4db5, {0xb4, 0x07, 0xe9, 0xaf, 0x3e, 0x2a, 0x9d, 0x15}};
static constexpr DWORD AllocationFlags = COR_PRF_MONITOR_OBJECT_ALLOCATED | COR_PRF_ENABLE_OBJECT_ALLOCATED;
static std::mutex CounterMutex;
// ABI 1: callbacks, arrays count/bytes, unknown class count/bytes, size failures, initialized.
static unsigned long long Counters[7] = {};

extern "C" __declspec(dllexport) void __cdecl ArrayProfilerSnapshot(unsigned long long* output, unsigned int length)
{
    if (output == nullptr || length != 7) return;
    std::lock_guard<std::mutex> guard(CounterMutex);
    for (unsigned int i = 0; i < length; ++i) output[i] = Counters[i];
}

static void WriteFinalReceipt()
{
    wchar_t path[32768];
    const DWORD length = GetEnvironmentVariableW(L"AIDOTNET_ARRAY_PROFILER_OUTPUT", path, 32768);
    if (length == 0 || length >= 32768) return;
    FILE* file = nullptr;
    if (_wfopen_s(&file, path, L"wb") != 0 || file == nullptr) return;
    unsigned long long values[7];
    ArrayProfilerSnapshot(values, 7);
    std::fprintf(file,
        "{\"schema\":\"clr-array-profiler-v1\",\"pid\":%lu,\"shutdown_complete\":true,"
        "\"event_mask\":%lu,\"snapshot\":[%llu,%llu,%llu,%llu,%llu,%llu,%llu],"
        "\"scope\":\"all_managed_arrays_including_headers_unknown_classes_conservatively_included\","
        "\"sampling\":false,\"unmanaged_coverage\":false}\n",
        GetCurrentProcessId(), static_cast<unsigned long>(AllocationFlags),
        values[0], values[1], values[2], values[3], values[4], values[5], values[6]);
    std::fclose(file);
}

class ArrayProfiler final : public ICorProfilerCallback2
{
    std::atomic<ULONG> references{1};
    ICorProfilerInfo4* info = nullptr;
public:
    ~ArrayProfiler() { if (info != nullptr) info->Release(); }
    HRESULT STDMETHODCALLTYPE QueryInterface(REFIID iid, void** value) override
    {
        if (value == nullptr) return E_POINTER;
        *value = nullptr;
        if (iid == __uuidof(IUnknown) || iid == __uuidof(ICorProfilerCallback) || iid == __uuidof(ICorProfilerCallback2))
        {
            *value = static_cast<ICorProfilerCallback2*>(this);
            AddRef();
            return S_OK;
        }
        return E_NOINTERFACE;
    }
    ULONG STDMETHODCALLTYPE AddRef() override { return ++references; }
    ULONG STDMETHODCALLTYPE Release() override
    {
        const ULONG count = --references;
        if (count == 0) delete this;
        return count;
    }
    HRESULT STDMETHODCALLTYPE Initialize(IUnknown* runtime) override
    {
        // Launch-only profiling is required: an attach profiler misses earlier allocations.
        HRESULT result = runtime->QueryInterface(__uuidof(ICorProfilerInfo4), reinterpret_cast<void**>(&info));
        if (FAILED(result)) return result;
        result = info->SetEventMask(AllocationFlags);
        if (SUCCEEDED(result))
        {
            std::lock_guard<std::mutex> guard(CounterMutex);
            Counters[6] = 1;
        }
        return result;
    }
    HRESULT STDMETHODCALLTYPE ObjectAllocated(ObjectID objectId, ClassID classId) override
    {
        CorElementType elementType;
        ClassID elementClass;
        ULONG rank;
        const HRESULT classification = info->IsArrayClass(classId, &elementType, &elementClass, &rank);
        // S_FALSE is a proven non-array. A not-yet-loaded class must not cause array undercounting:
        // include its full object size conservatively, and expose this separate count in every receipt.
        SIZE_T size = 0;
        const HRESULT sized = classification == S_FALSE ? S_OK : info->GetObjectSize2(objectId, &size);
        std::lock_guard<std::mutex> guard(CounterMutex);
        ++Counters[0];
        if (FAILED(sized)) ++Counters[5];
        else if (classification == S_OK) { ++Counters[1]; Counters[2] += size; }
        else if (classification != S_FALSE) { ++Counters[3]; Counters[4] += size; }
        return S_OK;
    }
    HRESULT STDMETHODCALLTYPE Shutdown() override { WriteFinalReceipt(); return S_OK; }
#include "CallbackStubs.inc"
};

class ProfilerFactory final : public IClassFactory
{
    std::atomic<ULONG> references{1};
public:
    HRESULT STDMETHODCALLTYPE QueryInterface(REFIID iid, void** value) override
    {
        if (value == nullptr) return E_POINTER;
        *value = nullptr;
        if (iid != __uuidof(IUnknown) && iid != __uuidof(IClassFactory)) return E_NOINTERFACE;
        *value = static_cast<IClassFactory*>(this); AddRef(); return S_OK;
    }
    ULONG STDMETHODCALLTYPE AddRef() override { return ++references; }
    ULONG STDMETHODCALLTYPE Release() override
    {
        const ULONG count = --references;
        if (count == 0) delete this;
        return count;
    }
    HRESULT STDMETHODCALLTYPE CreateInstance(IUnknown* outer, REFIID iid, void** value) override
    {
        if (outer != nullptr) return CLASS_E_NOAGGREGATION;
        auto* profiler = new (std::nothrow) ArrayProfiler();
        if (profiler == nullptr) return E_OUTOFMEMORY;
        const HRESULT result = profiler->QueryInterface(iid, value);
        profiler->Release();
        return result;
    }
    HRESULT STDMETHODCALLTYPE LockServer(BOOL) override { return S_OK; }
};

STDAPI DllGetClassObject(REFCLSID clsid, REFIID iid, void** value)
{
    if (clsid != ProfilerClsid) return CLASS_E_CLASSNOTAVAILABLE;
    auto* factory = new (std::nothrow) ProfilerFactory();
    if (factory == nullptr) return E_OUTOFMEMORY;
    const HRESULT result = factory->QueryInterface(iid, value);
    factory->Release();
    return result;
}
STDAPI DllCanUnloadNow() { return S_FALSE; }
