// Fonctions auxiliaires
#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#else
#include <sys/time.h>
#include <time.h>
#endif

/**
 * Generate Random number in single precision.
 */
float random_float(int emin, int emax, int pos_neg){
    double tmp;
    unsigned int i, val;
    int e;

    val = (rand() & 0x000000ff);
    for(i=0; i<(sizeof(int)); i++){
	val = val << 8;
	val += (rand() & 0x000000ff ); /* we keep only 8 bits */
    }
    e = emin + (int)( (double)rand()*(emax-emin)/(double)RAND_MAX);
    tmp = ldexp(1.0 + (double)val / UINT_MAX, e);
    if ((pos_neg) && (rand() > (RAND_MAX/2)))		tmp *= -1;

    return (float)tmp;
}

double getclock()
{
#ifdef _WIN32
    LARGE_INTEGER li;
    QueryPerformanceFrequency(&li);

    double PCFreq = (double)li.QuadPart;
    QueryPerformanceCounter(&li);
    __int64 timerStart = li.QuadPart;
    return ((double)li.QuadPart)/PCFreq;
#else
	struct timeval tv;
	gettimeofday(&tv, NULL);
	return tv.tv_sec + double(tv.tv_usec) / 1000000.;
#endif
}
