// Header file for input output functions
#include<iostream>
#include <mpfr.h>
#include <functional>
#include <math.h>
#include <list>

using namespace std;


int main()
{
    // register constants
    mpfr_t pi, one, e, result;

    function<double(double, double)> convtime = [](clock_t begin, clock_t end)
    {
        return double(end - begin) / CLOCKS_PER_SEC;
    };

    // Add
    int refinement_iterations = 18;
    for (int i = 3; i < refinement_iterations; i++){
        int precision = pow(2, i);
        mpfr_init2(pi, precision);
        mpfr_init2(one, precision);
        mpfr_init2(e, precision);
        mpfr_init2(result, precision);
        mpfr_free_cache();

        clock_t begin = clock();
        mpfr_const_pi(pi, GMP_RNDN);
        mpfr_exp(e, one, GMP_RNDN);
        mpfr_add(result, e, pi, GMP_RNDN);
        clock_t end = clock();

        mpfr_clear(pi);
        mpfr_clear(one);
        mpfr_clear(e);
        mpfr_clear(result);

        printf("%.2g sec at precision %d\n", convtime(begin, end), precision);
    }


    printf("\n***===***\n");

    // Sub
    for (int i = 3; i < refinement_iterations; i++){
        int precision = pow(2, i);
        mpfr_init2(pi, precision);
        mpfr_init2(one, precision);
        mpfr_init2(e, precision);
        mpfr_init2(result, precision);
        mpfr_free_cache();

        clock_t begin = clock();
        mpfr_const_pi(pi, GMP_RNDN);
        mpfr_exp(e, one, GMP_RNDN);
        mpfr_sub(result, e, pi, GMP_RNDN);
        clock_t end = clock();

        mpfr_clear(pi);
        mpfr_clear(one);
        mpfr_clear(e);
        mpfr_clear(result);

        printf("%.2g sec at precision %d\n", convtime(begin, end), precision);
    }
    printf("\n***===***\n");

    // Mul
    for (int i = 3; i < refinement_iterations; i++){
        int precision = pow(2, i);
        mpfr_init2(pi, precision);
        mpfr_init2(one, precision);
        mpfr_init2(e, precision);
        mpfr_init2(result, precision);
        mpfr_free_cache();

        clock_t begin = clock();
        mpfr_const_pi(pi, GMP_RNDN);
        mpfr_exp(e, one, GMP_RNDN);
        mpfr_mul(result, e, pi, GMP_RNDN);
        clock_t end = clock();

        mpfr_clear(pi);
        mpfr_clear(one);
        mpfr_clear(e);
        mpfr_clear(result);

        printf("%.2g sec at precision %d\n", convtime(begin, end), precision);
    }
    printf("\n***===***\n");

    // Div
    for (int i = 3; i < refinement_iterations; i++){
        int precision = pow(2, i);
        mpfr_init2(pi, precision);
        mpfr_init2(one, precision);
        mpfr_init2(e, precision);
        mpfr_init2(result, precision);
        mpfr_free_cache();

        clock_t begin = clock();
        mpfr_const_pi(pi, GMP_RNDN);
        mpfr_exp(e, one, GMP_RNDN);
        mpfr_div(result, e, pi, GMP_RNDN);
        clock_t end = clock();

        mpfr_clear(pi);
        mpfr_clear(one);
        mpfr_clear(e);
        mpfr_clear(result);

        printf("%.2g sec at precision %d\n", convtime(begin, end), precision);
    }
}