import bigfloat as bf


def interval_sin(interval, context_down, context_up):
    def sin_monotone_i(interval):
        lower, upper = interval
        return bf.sin(lower, context_down), bf.sin(upper, context_up)
    lower, upper = interval

    # Start-off knowing nothing about the interval
    out_interval = [-1, 1]

    # [dout_lower/ din_lower, dout_upper/ din_lower], [dout_lower/ din_upper, dout_upper/ din_upper]
    derivs = [0, 0], [0, 0]

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
            derivs = [0, upper_deriv_up], [lower_deriv_down, 0]
        elif not (lower_deriv_down < 0) and not (upper_deriv_up > 0):
            out_interval = [min(bf.sin(lower, context_down), bf.sin(upper, context_down)), 1]
            derivs[0] = [lower_deriv_down, 0] if bf.sin(lower, context_down) < bf.sin(upper, context_down) else [0, upper_deriv_down]
        elif not (lower_deriv_down > 0) and not (upper_deriv_down < 0):
            out_interval = [-1, max(bf.sin(lower, context_up), bf.sin(upper, context_up))]
            derivs[1] = [lower_deriv_up, 0] if bf.sin(lower, context_down) > bf.sin(upper, context_down) else [0, upper_deriv_up]

    return out_interval, derivs


def interval_cos(interval, context_down, context_up):
    def cos_monotone_i(interval):
        lower, upper = interval
        return bf.sin(lower, context_down), bf.sin(upper, context_up)
    lower, upper = interval

    # Start-off knowing nothing about the interval
    out_interval = [-1, 1]

    if bf.sub(upper, lower, context_up) < 3:
        # Compute derivatives
        lower_deriv_down, lower_deriv_up = -bf.sin(lower, context_down),-bf.sin(lower, context_up)
        upper_deriv_down, upper_deriv_up = -bf.sin(upper, context_down),-bf.sin(upper, context_up)

        # Check derivative signs to identify what monotonic region lower and upper lie in
        # set the output intervals as appropriate.
        if not (lower_deriv_down < 0) and not (upper_deriv_down < 0):
            out_interval = cos_monotone_i(interval)
        elif not (lower_deriv_up > 0) and not (upper_deriv_up > 0):
            out_interval = cos_monotone_i([interval[1], interval[0]])
        elif not (lower_deriv_down < 0) and not (upper_deriv_up > 0):
            out_interval = [min(bf.cos(lower, context_down), bf.cos(upper, context_down)), 1]
        elif not (lower_deriv_down > 0) and not (upper_deriv_down < 0):
            out_interval = [-1, max(bf.cos(lower, context_up), bf.cos(upper, context_up))]

    return out_interval


if __name__ == '__main__':
    context_down = bf.precision(100) + bf.RoundTowardNegative
    context_up = bf.precision(100) + bf.RoundTowardPositive

    pi_down = bf.const_pi(context_down)
    pi_up = bf.const_pi(context_up)

    # Testing sin
    # [0, pi/2]
    lower, upper = interval_sin([0, pi_down / 2 - 0.01], context_down, context_up)
    assert upper - lower < 1 and lower >= 0

    # [pi, 3*pi/2]
    lower, upper = interval_sin([pi_down, 3 * pi_up / 2 - 0.01], context_down, context_up)
    assert upper - lower < 1 and  -0.75 > lower >= -1

    # Over pi away
    lower, upper = interval_sin([0, 3 * pi_down / 2], context_down, context_up)
    assert upper - lower >= 2

    # Near pi
    lower, upper = interval_sin([pi_down, pi_up], context_down, context_up)
    assert lower < 0 and upper > 0

    # Testing cos
    # [0, pi/2]
    lower, upper = interval_cos([0, pi_down / 2 - 0.01], context_down, context_up)
    assert upper - lower < 1 and lower >= 0

    # [pi, 3*pi/2]
    lower, upper = interval_cos([pi_down, 3 * pi_up / 2 - 0.01], context_down, context_up)
    assert upper - lower < 1 and  -0.75 > lower >= -1

    # Over pi away
    lower, upper = interval_cos([0, 3 * pi_down / 2], context_down, context_up)
    assert upper - lower >= 2

    # Near pi
    lower, upper = interval_cos([pi_down, pi_up], context_down, context_up)
    assert upper - lower < 0.001 and lower == -1