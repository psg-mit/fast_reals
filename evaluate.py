from typing import List, Set
import bigfloat as bf
import numpy as np
from datetime import timedelta
from timeit import default_timer as timer

from exact_real_program import ExactRealProgram


def evaluate(program: ExactRealProgram,
             precision_bound: float,
             ad: bool = False,
             initial_precision: int = 50) -> ExactRealProgram:
    """ Uniformly refine the program until the given precision bound is met. """
    precision = initial_precision
    if ad:
        program.lower_grad, program.upper_grad = 1, -1
    program.apply(reset_ad_children)
    program.evaluate(precision, ad)
    print("prec", precision)

    # max precision setting rounding up of the log bound
    error_precision = 1 + int(bf.abs(bf.log2(precision_bound)))
    context = bf.precision(error_precision) + bf.RoundTowardPositive
    program.grad_precision = error_precision

    error = bf.sub(program.upper, program.lower, context)
    refinement_steps = 0

    iteration_times = []
    while error > precision_bound:
        program.apply(reset_ad_children)
        start_time = timer()
        program.evaluate(precision, ad)
        end_time = timer()
        error = bf.sub(program.upper, program.lower, context)
        iteration_times.append(timedelta(seconds=end_time - start_time))
        precision = precision + int(50 * 1.25**(refinement_steps))
        refinement_steps += 1
        # print(refinement_steps, precision)
    return refinement_steps, iteration_times


def reset_ad_children(program: ExactRealProgram):
    program.ad_lower_children, program.ad_upper_children = [], []


def evaluate_using_derivatives(program: ExactRealProgram,
                               precision_bound: float,
                               initial_precisions: List[int]) -> ExactRealProgram:
    """ Uniformly refine the program until the given precision bound is met. """
    precisions = initial_precisions
    program.lower_grad, program.upper_grad = 1, -1
    program.apply(reset_ad_children)
    program.evaluate_at(precisions, ad=True)

    error_precision = 1 + int(bf.abs(bf.log2(precision_bound)))
    context = bf.precision(error_precision) + bf.RoundTowardPositive
    program.grad_precision = error_precision
    error = bf.sub(program.upper, program.lower, context)

    all_precisions = []
    iteration_times = []
    critical_path = set()
    refinement_steps = 0
    while error > precision_bound:
        grads = []
        program.apply(lambda program: grads.append(program.grad()))
        precisions, critical_path = precision_from_grads(program, precisions, grads, refinement_steps)
        program.apply(reset_ad_children)
        start_time = timer()
        program.evaluate_at(precisions, ad=True)
        end_time = timer()
        all_precisions.append(precisions)
        error = bf.sub(program.upper, program.lower, context)
        iteration_times.append(timedelta(seconds=end_time - start_time))
        refinement_steps += 1
    program.apply(reset_ad_children)
    return refinement_steps, iteration_times, all_precisions


def precision_from_grads(program: ExactRealProgram,
                         prev_precisions: List[int],
                         grads: List[float],
                         t: int) -> List[int]:
    critical_path = max_gradient(program, grads)
    precs = [prec + int(50 * 1.25**t + 50 * 1.25**(t + 1)) if i in critical_path else prec + int(50 * 1.25**t)
            for i, prec in enumerate(prev_precisions)]
    # print(precs)
    # Refine the largest precision by an extra step
    return precs, critical_path


def max_gradient(program: ExactRealProgram, grads: List[float]) -> Set[ExactRealProgram]:
    nodes = []
    program.apply(lambda node: nodes.append(node))
    grad_context = bf.precision(program.grad_precision) + bf.RoundAwayFromZero
    widths = [bf.sub(node.upper, node.lower, grad_context) for node in nodes]

    # Find the maximum gradient node in the computation graph
    # compute the (lower_grad - upper_grad), which should be positive
    reversed_grads = list(reversed([(i + 1, bf.mul(bf.sub(grad[0], grad[1], grad_context), widths[i + 1], grad_context))
                                    for i, grad in enumerate(grads[1:])]))
    argmax = max(reversed_grads, key=lambda x: x[1])[0]
    # For debugging ad
    # assert sum([(grad[0] - grad[1]) < 0 for grad in grads]) == 0  # All False

    # Get the set of nodes on the path from the maximum gradient node to the root
    max_gradient_node = nodes[argmax]

    critical_path = set()
    node_to_ind = {node: i for i, node in enumerate(nodes)}
    node = max_gradient_node
    while node is not None:
        critical_path.add(node_to_ind[node])
        node.color = 'red'
        node = node.parent
    return critical_path


def clamped_prop(program: ExactRealProgram, grads: List[float],
                 critical_paths: List[frozenset]) -> Set[ExactRealProgram]:
    # Find the maximum gradient node in the computation graph
    # compute the (lower_grad - upper_grad), which should be positive
    reversed_grads = list(reversed([(i + 1, grad[0] - grad[1]) for i, grad in enumerate(grads[1:])]))
    argmax = max(reversed_grads, key=lambda x: x[1])[0]
    # For debugging ad
    # assert sum([(grad[0] - grad[1]) < 0 for grad in grads]) == 0  # All False

    # Get the set of nodes on the path from the maximum gradient node to the root
    nodes = []
    program.apply(lambda node: nodes.append(node))
    max_gradient_node = nodes[argmax]

    critical_path = set()
    node_to_ind = {node: i for i, node in enumerate(nodes)}
    node = max_gradient_node
    while node is not None:
        critical_path.add(node_to_ind[node])
        node.color = 'red'
        node = node.parent
    return frozenset(critical_path)
