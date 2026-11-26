// Generated from dotnet/runtime 60629d14374c56f1cb51819049ad1fa529307f8d corprof.h.
    HRESULT STDMETHODCALLTYPE AppDomainCreationStarted(AppDomainID appDomainId) override { return S_OK; }
    HRESULT STDMETHODCALLTYPE AppDomainCreationFinished(AppDomainID appDomainId, HRESULT hrStatus) override { return S_OK; }
    HRESULT STDMETHODCALLTYPE AppDomainShutdownStarted(AppDomainID appDomainId) override { return S_OK; }
    HRESULT STDMETHODCALLTYPE AppDomainShutdownFinished(AppDomainID appDomainId, HRESULT hrStatus) override { return S_OK; }
    HRESULT STDMETHODCALLTYPE AssemblyLoadStarted(AssemblyID assemblyId) override { return S_OK; }
    HRESULT STDMETHODCALLTYPE AssemblyLoadFinished(AssemblyID assemblyId, HRESULT hrStatus) override { return S_OK; }
    HRESULT STDMETHODCALLTYPE AssemblyUnloadStarted(AssemblyID assemblyId) override { return S_OK; }
    HRESULT STDMETHODCALLTYPE AssemblyUnloadFinished(AssemblyID assemblyId, HRESULT hrStatus) override { return S_OK; }
    HRESULT STDMETHODCALLTYPE ModuleLoadStarted(ModuleID moduleId) override { return S_OK; }
    HRESULT STDMETHODCALLTYPE ModuleLoadFinished(ModuleID moduleId, HRESULT hrStatus) override { return S_OK; }
    HRESULT STDMETHODCALLTYPE ModuleUnloadStarted(ModuleID moduleId) override { return S_OK; }
    HRESULT STDMETHODCALLTYPE ModuleUnloadFinished(ModuleID moduleId, HRESULT hrStatus) override { return S_OK; }
    HRESULT STDMETHODCALLTYPE ModuleAttachedToAssembly(ModuleID moduleId, AssemblyID AssemblyId) override { return S_OK; }
    HRESULT STDMETHODCALLTYPE ClassLoadStarted(ClassID classId) override { return S_OK; }
    HRESULT STDMETHODCALLTYPE ClassLoadFinished(ClassID classId, HRESULT hrStatus) override { return S_OK; }
    HRESULT STDMETHODCALLTYPE ClassUnloadStarted(ClassID classId) override { return S_OK; }
    HRESULT STDMETHODCALLTYPE ClassUnloadFinished(ClassID classId, HRESULT hrStatus) override { return S_OK; }
    HRESULT STDMETHODCALLTYPE FunctionUnloadStarted(FunctionID functionId) override { return S_OK; }
    HRESULT STDMETHODCALLTYPE JITCompilationStarted(FunctionID functionId, BOOL fIsSafeToBlock) override { return S_OK; }
    HRESULT STDMETHODCALLTYPE JITCompilationFinished(FunctionID functionId, HRESULT hrStatus, BOOL fIsSafeToBlock) override { return S_OK; }
    HRESULT STDMETHODCALLTYPE JITCachedFunctionSearchStarted(FunctionID functionId, BOOL *pbUseCachedFunction) override { return S_OK; }
    HRESULT STDMETHODCALLTYPE JITCachedFunctionSearchFinished(FunctionID functionId, COR_PRF_JIT_CACHE result) override { return S_OK; }
    HRESULT STDMETHODCALLTYPE JITFunctionPitched(FunctionID functionId) override { return S_OK; }
    HRESULT STDMETHODCALLTYPE JITInlining(FunctionID callerId, FunctionID calleeId, BOOL *pfShouldInline) override { return S_OK; }
    HRESULT STDMETHODCALLTYPE ThreadCreated(ThreadID threadId) override { return S_OK; }
    HRESULT STDMETHODCALLTYPE ThreadDestroyed(ThreadID threadId) override { return S_OK; }
    HRESULT STDMETHODCALLTYPE ThreadAssignedToOSThread(ThreadID managedThreadId, DWORD osThreadId) override { return S_OK; }
    HRESULT STDMETHODCALLTYPE RemotingClientInvocationStarted(void) override { return S_OK; }
    HRESULT STDMETHODCALLTYPE RemotingClientSendingMessage(GUID *pCookie, BOOL fIsAsync) override { return S_OK; }
    HRESULT STDMETHODCALLTYPE RemotingClientReceivingReply(GUID *pCookie, BOOL fIsAsync) override { return S_OK; }
    HRESULT STDMETHODCALLTYPE RemotingClientInvocationFinished(void) override { return S_OK; }
    HRESULT STDMETHODCALLTYPE RemotingServerReceivingMessage(GUID *pCookie, BOOL fIsAsync) override { return S_OK; }
    HRESULT STDMETHODCALLTYPE RemotingServerInvocationStarted(void) override { return S_OK; }
    HRESULT STDMETHODCALLTYPE RemotingServerInvocationReturned(void) override { return S_OK; }
    HRESULT STDMETHODCALLTYPE RemotingServerSendingReply(GUID *pCookie, BOOL fIsAsync) override { return S_OK; }
    HRESULT STDMETHODCALLTYPE UnmanagedToManagedTransition(FunctionID functionId, COR_PRF_TRANSITION_REASON reason) override { return S_OK; }
    HRESULT STDMETHODCALLTYPE ManagedToUnmanagedTransition(FunctionID functionId, COR_PRF_TRANSITION_REASON reason) override { return S_OK; }
    HRESULT STDMETHODCALLTYPE RuntimeSuspendStarted(COR_PRF_SUSPEND_REASON suspendReason) override { return S_OK; }
    HRESULT STDMETHODCALLTYPE RuntimeSuspendFinished(void) override { return S_OK; }
    HRESULT STDMETHODCALLTYPE RuntimeSuspendAborted(void) override { return S_OK; }
    HRESULT STDMETHODCALLTYPE RuntimeResumeStarted(void) override { return S_OK; }
    HRESULT STDMETHODCALLTYPE RuntimeResumeFinished(void) override { return S_OK; }
    HRESULT STDMETHODCALLTYPE RuntimeThreadSuspended(ThreadID threadId) override { return S_OK; }
    HRESULT STDMETHODCALLTYPE RuntimeThreadResumed(ThreadID threadId) override { return S_OK; }
    HRESULT STDMETHODCALLTYPE MovedReferences(ULONG cMovedObjectIDRanges, ObjectID oldObjectIDRangeStart[ ], ObjectID newObjectIDRangeStart[ ], ULONG cObjectIDRangeLength[ ]) override { return S_OK; }
    HRESULT STDMETHODCALLTYPE ObjectsAllocatedByClass(ULONG cClassCount, ClassID classIds[ ], ULONG cObjects[ ]) override { return S_OK; }
    HRESULT STDMETHODCALLTYPE ObjectReferences(ObjectID objectId, ClassID classId, ULONG cObjectRefs, ObjectID objectRefIds[ ]) override { return S_OK; }
    HRESULT STDMETHODCALLTYPE RootReferences(ULONG cRootRefs, ObjectID rootRefIds[ ]) override { return S_OK; }
    HRESULT STDMETHODCALLTYPE ExceptionThrown(ObjectID thrownObjectId) override { return S_OK; }
    HRESULT STDMETHODCALLTYPE ExceptionSearchFunctionEnter(FunctionID functionId) override { return S_OK; }
    HRESULT STDMETHODCALLTYPE ExceptionSearchFunctionLeave(void) override { return S_OK; }
    HRESULT STDMETHODCALLTYPE ExceptionSearchFilterEnter(FunctionID functionId) override { return S_OK; }
    HRESULT STDMETHODCALLTYPE ExceptionSearchFilterLeave(void) override { return S_OK; }
    HRESULT STDMETHODCALLTYPE ExceptionSearchCatcherFound(FunctionID functionId) override { return S_OK; }
    HRESULT STDMETHODCALLTYPE ExceptionOSHandlerEnter(UINT_PTR __unused) override { return S_OK; }
    HRESULT STDMETHODCALLTYPE ExceptionOSHandlerLeave(UINT_PTR __unused) override { return S_OK; }
    HRESULT STDMETHODCALLTYPE ExceptionUnwindFunctionEnter(FunctionID functionId) override { return S_OK; }
    HRESULT STDMETHODCALLTYPE ExceptionUnwindFunctionLeave(void) override { return S_OK; }
    HRESULT STDMETHODCALLTYPE ExceptionUnwindFinallyEnter(FunctionID functionId) override { return S_OK; }
    HRESULT STDMETHODCALLTYPE ExceptionUnwindFinallyLeave(void) override { return S_OK; }
    HRESULT STDMETHODCALLTYPE ExceptionCatcherEnter(FunctionID functionId, ObjectID objectId) override { return S_OK; }
    HRESULT STDMETHODCALLTYPE ExceptionCatcherLeave(void) override { return S_OK; }
    HRESULT STDMETHODCALLTYPE COMClassicVTableCreated(ClassID wrappedClassId, REFGUID implementedIID, void *pVTable, ULONG cSlots) override { return S_OK; }
    HRESULT STDMETHODCALLTYPE COMClassicVTableDestroyed(ClassID wrappedClassId, REFGUID implementedIID, void *pVTable) override { return S_OK; }
    HRESULT STDMETHODCALLTYPE ExceptionCLRCatcherFound(void) override { return S_OK; }
    HRESULT STDMETHODCALLTYPE ExceptionCLRCatcherExecute(void) override { return S_OK; }
    HRESULT STDMETHODCALLTYPE ThreadNameChanged(ThreadID threadId, ULONG cchName, _In_reads_opt_(cchName) WCHAR name[ ]) override { return S_OK; }
    HRESULT STDMETHODCALLTYPE GarbageCollectionStarted(int cGenerations, BOOL generationCollected[ ], COR_PRF_GC_REASON reason) override { return S_OK; }
    HRESULT STDMETHODCALLTYPE SurvivingReferences(ULONG cSurvivingObjectIDRanges, ObjectID objectIDRangeStart[ ], ULONG cObjectIDRangeLength[ ]) override { return S_OK; }
    HRESULT STDMETHODCALLTYPE GarbageCollectionFinished(void) override { return S_OK; }
    HRESULT STDMETHODCALLTYPE FinalizeableObjectQueued(DWORD finalizerFlags, ObjectID objectID) override { return S_OK; }
    HRESULT STDMETHODCALLTYPE RootReferences2(ULONG cRootRefs, ObjectID rootRefIds[ ], COR_PRF_GC_ROOT_KIND rootKinds[ ], COR_PRF_GC_ROOT_FLAGS rootFlags[ ], UINT_PTR rootIds[ ]) override { return S_OK; }
    HRESULT STDMETHODCALLTYPE HandleCreated(GCHandleID handleId, ObjectID initialObjectId) override { return S_OK; }
    HRESULT STDMETHODCALLTYPE HandleDestroyed(GCHandleID handleId) override { return S_OK; }
