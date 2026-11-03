#pragma once
// Live status line on stderr, rewritten in place. Silent when stderr is not a terminal.
#include <cstdio>
#include <cstdarg>
#include <unistd.h>

static inline bool progress_on()
{
    static int on = -1;
    if (on < 0) on = isatty(STDERR_FILENO) ? 1 : 0;
    return on == 1;
}

static inline void progress(const char *fmt, ...)
{
    if (!progress_on()) return;
    fflush(stdout);
    va_list ap;
    va_start(ap, fmt);
    fputs("\r\033[K", stderr);
    vfprintf(stderr, fmt, ap);
    va_end(ap);
    fflush(stderr);
}

// clear the status line before printing a result line
static inline void progress_clear()
{
    if (!progress_on()) return;
    fputs("\r\033[K", stderr);
    fflush(stderr);
}
