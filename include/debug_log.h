#pragma once
#include <iostream>

// Debug logging macros. By default (when ENABLE_DEBUG_LOG is *not* defined) all
// invocations of DLOG() are compiled out completely and introduce zero runtime
// overhead.
//
// Usage:
//   DLOG(std::cout << "value=" << x << std::endl);
//
// If you need more granular control (e.g. different levels) you can introduce
// additional macros similar to DLOG.
#ifdef ENABLE_DEBUG_LOG
    #define DLOG(code) do { code; } while(false)
#else
    #define DLOG(code) do { } while(false)
#endif
