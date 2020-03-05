import bigfloat as bf


def interval_sin(interval, context_down, context_up):
    def sin_monotone_i(interval):
        lower, upper = interval
        return bf.sin(lower, context_down), bf.sin(upper, context_up)
    lower, upper = interval

    # Start-off knowing nothing about the interval
    out_interval = [-1, 1]

    # [dout_lower/ din_lower, dout_upper/ din_lower], [dout_lower/ din_upper, dout_upper/ din_upper]
    derivs = [[0, 0], [0, 0]]

    if bf.sub(upper, lower, context_up) < 3:
        # Compute derivatives
        lower_deriv_down, lower_deriv_up = bf.cos(lower, context_down), bf.cos(lower, context_up)
        upper_deriv_down, upper_deriv_up = bf.cos(upper, context_down), bf.cos(upper, context_up)

        # Check derivative signs to identify what monotonic region lower and upper lie in
        # set the output intervals as appropriate.
        if not (lower_deriv_down < 0) and not (upper_deriv_down < 0):
            out_interval = sin_monotone_i(interval)
            derivs = [lower_deriv_down, 0], [0, upper_deriv_up]
        elif not (lower_deriv_up > 0) and not (upper_deriv_up > 0):
            out_interval = sin_monotone_i([interval[1], interval[0]])
            derivs = [0, lower_deriv_up], [upper_deriv_down, 0]
        elif not (lower_deriv_down < 0) and not (upper_deriv_up > 0):
            out_interval = [min(bf.sin(lower, context_down), bf.sin(upper, context_down)), 1]
            if bf.sin(lower, context_down) < bf.sin(upper, context_down):
                derivs[0] = [lower_deriv_down, 0]
            else:
                derivs[1] = [upper_deriv_down, 0]
        elif not (lower_deriv_down > 0) and not (upper_deriv_down < 0):
            out_interval = [-1, max(bf.sin(lower, context_up), bf.sin(upper, context_up))]
            if bf.sin(lower, context_down) > bf.sin(upper, context_down):
                derivs[0] = [0, lower_deriv_up]
            else:
                derivs[1] = [0, upper_deriv_up]

    return out_interval, derivs


def interval_cos(interval, context_down, context_up):
    def cos_monotone_i(interval):
        lower, upper = interval
        return bf.cos(lower, context_down), bf.cos(upper, context_up)
    lower, upper = interval

    # Start-off knowing nothing about the interval
    out_interval = [-1, 1]

    # [dout_lower/ din_lower, dout_upper/ din_lower], [dout_lower/ din_upper, dout_upper/ din_upper]
    derivs = [[0, 0], [0, 0]]

    if bf.sub(upper, lower, context_up) < 3:
        # Compute derivatives
        lower_deriv_down, lower_deriv_up = -bf.sin(lower, context_down), -bf.sin(lower, context_up)
        upper_deriv_down, upper_deriv_up = -bf.sin(upper, context_down), -bf.sin(upper, context_up)

        # Check derivative signs to identify what monotonic region lower and upper lie in
        # set the output intervals as appropriate.
        if not (lower_deriv_down < 0) and not (upper_deriv_down < 0):
            out_interval = cos_monotone_i(interval)
            derivs = [lower_deriv_down, 0], [0, upper_deriv_up]
        elif not (lower_deriv_up > 0) and not (upper_deriv_up > 0):
            out_interval = cos_monotone_i([interval[1], interval[0]])
            derivs = [0, lower_deriv_up], [upper_deriv_down, 0]
        elif not (lower_deriv_down < 0) and not (upper_deriv_up > 0):
            out_interval = [min(bf.cos(lower, context_down), bf.cos(upper, context_down)), 1]
            if bf.cos(lower, context_down) < bf.cos(upper, context_down):
                derivs[0] = [lower_deriv_down, 0]
            else:
                derivs[1] = [upper_deriv_down, 0]
        elif not (lower_deriv_down > 0) and not (upper_deriv_down < 0):
            out_interval = [-1, max(bf.cos(lower, context_up), bf.cos(upper, context_up))]
            if bf.cos(lower, context_down) > bf.cos(upper, context_down):
                derivs[0] = [0, lower_deriv_up]
            else:
                derivs[1] = [0, upper_deriv_up]
    return out_interval, derivs


# TODO pow
def interval_sqrt(interval, context_down, context_up):
    lower, upper = interval
    sqrt_down, sqrt_up = bf.sqrt(lower, context_down), bf.sqrt(upper, context_up)
    out_interval = [sqrt_down, sqrt_up]
    derivs = [1 / (2 * sqrt_down), 0], [0, 1 / (2 * sqrt_up)]
    return out_interval, derivs


def interval_cuberoot(interval, context_down, context_up):
    lower, upper = interval
    sqrt_down, sqrt_up = bf.pow(lower, 1/3, context_down), bf.pow(upper, 1/3, context_up)
    out_interval = [sqrt_down, sqrt_up]
    derivs = [bf.pow(lower, -2/3, context_down) / 3, 0], [0, bf.pow(upper, -2/3, context_up) / 3]
    return out_interval, derivs


def interval_fifthroot(interval, context_down, context_up):
    lower, upper = interval
    fifthrt_down, fifthrt_up = bf.pow(lower, 0.2, context_down), bf.pow(upper, 0.2, context_up)
    out_interval = [fifthrt_down, fifthrt_up]
    derivs = [bf.pow(lower, -0.8, context_down) / 5, 0], [0, bf.pow(upper, -0.8, context_up) / 5]
    return out_interval, derivs


def interval_log(interval, context_down, context_up):
    lower, upper = interval
    out_interval = [bf.log(lower, context_down), bf.log(upper, context_up)]
    derivs = [1 / lower, 0], [0, 1 / upper]
    return out_interval, derivs


def interval_exp(interval, context_down, context_up):
    lower, upper = interval
    out_lower, out_upper = [bf.exp(lower, context_down), bf.exp(upper, context_up)]
    out_interval = [out_lower, out_upper]
    derivs = [out_lower, 0], [0, out_upper]
    return out_interval, derivs


def interval_pow10(interval, context_down, context_up):
    lower, upper = interval
    ten_down, ten_up = bf.BigFloat("10", context_down), bf.BigFloat("10", context_up)
    out_lower, out_upper = [bf.pow(ten_down, lower, context_down), bf.pow(ten_up, upper, context_up)]
    out_interval = [out_lower, out_upper]
    derivs = [out_lower, 0], [0, out_upper]
    return out_interval, derivs


def interval_atan(interval, context_down, context_up):
    lower, upper = interval
    out_interval = [bf.atan(lower, context_down), bf.atan(upper, context_up)]
    derivs = [1 / (1 + lower**2), 0], [0, 1 / (1 + upper**2)]
    return out_interval, derivs

# def interval_pow(interval, context_down, context_up):
#     lower, upper = interval
#     out_interval = [bf.log(lower, context_down), bf.log(upper, context_up)]
#     derivs = [1 / lower, 0], [0, 1 / upper]


if __name__ == '__main__':
    context_down = bf.precision(100) + bf.RoundTowardNegative
    context_up = bf.precision(100) + bf.RoundTowardPositive

    pi_down = bf.const_pi(context_down)
    pi_up = bf.const_pi(context_up)

    # Testing sin
    (lower, upper), derivs = interval_sin([5*pi_down/6, 7*pi_down/4], context_down, context_up)
    assert lower == -1 and 0.4 < upper < 0.6
    assert derivs[0][0] == derivs[1][0] == derivs[1][1] and derivs[0][1] < -0.5

    (lower, upper), derivs = interval_sin([5*pi_down/4, 11*pi_down/6], context_down, context_up)
    assert lower == -1 and -0.6 < upper < -0.4
    assert derivs[0][0] == derivs[1][0] == derivs[0][1] and 0.8 < derivs[1][1] < 0.9

    (lower, upper), derivs = interval_sin([pi_down/3, 5*pi_down/6], context_down, context_up)
    assert 0.4 < lower < 0.6 and upper == 1
    assert derivs[0][0] == derivs[0][1] == derivs[1][1] and derivs[1][0] < -0.5

    (lower, upper), derivs = interval_sin([pi_down/6, 2*pi_down/3], context_down, context_up)
    assert 0.4 < lower < 0.6 and upper == 1
    assert derivs[1][0] == derivs[0][1] == derivs[1][1] and 0.8 < derivs[0][0] < 0.9

    (lower, upper), derivs = interval_sin([0.01, pi_down / 2 - 0.01], context_down, context_up)
    assert -0.1 < lower < 0.1 and 0.9 < upper
    assert derivs[0][1] == derivs[1][0] == 0 and derivs[0][0] > 0.9 and -0.1 < derivs[1][1] < 0.1

    (lower, upper), derivs = interval_sin([5*pi_down/6, 7*pi_down/6], context_down, context_up)
    assert -0.6 < lower < -0.4 and 0.4 < upper < 0.6
    assert derivs[0][0] == derivs[1][1] == 0 and -0.9 < derivs[0][1] < -0.7 and -0.9 < derivs[1][0] < -0.7

    # Over pi away
    (lower, upper), derivs = interval_sin([0, 3 * pi_down / 2], context_down, context_up)
    assert upper - lower >= 2

    # Testing cos
    (lower, upper), derivs = interval_cos([5*pi_down/6, 7*pi_down/4], context_down, context_up)
    assert lower == -1 and 0.7 < upper < 0.9
    assert derivs[0][0] == derivs[1][0] == derivs[0][1] == 0 and 0.7 < derivs[1][1] < 0.8

    (lower, upper), derivs = interval_cos([5*pi_down/4, 11*pi_down/6], context_down, context_up)
    assert -0.8 < lower < -0.7 and 0.8 < upper < 0.9
    assert derivs[1][0] == derivs[0][1] == 0 and 0.4 < derivs[1][1] < 0.5 and 0.7 < derivs[0][0] < 0.8

    (lower, upper), derivs = interval_cos([pi_down/3, 5*pi_down/6], context_down, context_up)
    assert -0.9 < lower < -0.8 and 0.4 < upper < 0.9
    assert derivs[0][0] == derivs[1][1] == 0 and -0.9 < derivs[0][1] < -0.8 and -0.6 < derivs[1][0] < -0.4

    (lower, upper), derivs = interval_cos([pi_down/6, 2*pi_down/3], context_down, context_up)
    assert -0.6 < lower < -0.4 and 0.8 < upper < 0.9
    assert derivs[0][0] == derivs[1][1] == 0 and -0.9 < derivs[1][0] < -0.8 and -0.6 < derivs[0][1] < -0.4

    (lower, upper), derivs = interval_cos([pi_down / 2 + 0.01, pi_down + 0.01], context_down, context_up)
    assert lower < -0.9 and -0.1 < upper < 0.1
    assert derivs[1][1] == derivs[1][0] == derivs[0][0] == 0 and -1 < derivs[0][1] < -0.9

    (lower, upper), derivs = interval_cos([5*pi_down/6, 5*pi_down/3], context_down, context_up)
    assert lower == -1 and 0.4 < upper < 0.6
    assert derivs[0][0] == derivs[1][0] == derivs[0][1] == 0 and 0.8 < derivs[1][1] < 0.9

    # Over pi away
    (lower, upper), derivs = interval_cos([0, 3 * pi_down / 2], context_down, context_up)
    assert upper - lower >= 2
