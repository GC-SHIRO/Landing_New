#define _GNU_SOURCE

#include <dlfcn.h>
#include <unistd.h>

typedef long (*sysconf_fn)(int);

long sysconf(int name)
{
    if (name == _SC_NPROCESSORS_ONLN || name == _SC_NPROCESSORS_CONF)
        return 32;

    static sysconf_fn real_sysconf = NULL;
    if (real_sysconf == NULL)
        real_sysconf = (sysconf_fn)dlsym(RTLD_NEXT, "sysconf");

    return real_sysconf(name);
}
