from typing import List, Set

from exact_real_program import ExactRealProgram


def evaluate(program: ExactRealProgram,
             precision_bound: float,
             ad: bool = False,
             initial_precision: int = 30) -> ExactRealProgram:
    """ Uniformly refine the program until the given precision bound is met. """
    precision = initial_precision
    if ad:
        program.lower_grad, program.upper_grad = 1, -1
    program.apply(reset_ad_children)
    program.evaluate(precision, ad)

    error = program.upper - program.lower
    refinement_steps = 0

    iteration_times = []
    from datetime import timedelta
    from timeit import default_timer as timer
    while error > precision_bound:
        start_time = timer()
        precision += 3
        program.apply(reset_ad_children)
        program.evaluate(precision, ad)
        error = program.upper - program.lower
        refinement_steps += 1
        end_time = timer()
        iteration_times.append(timedelta(seconds=end_time - start_time))
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

    error = program.upper - program.lower
    prev_error = error
    critical_path = set()
    refinement_steps = 0
    while error > precision_bound:
        grads = []
        program.apply(lambda program: grads.append(program.grad()))
        precisions, critical_path = precision_from_grads(program, precisions, grads, critical_path)
        program.apply(reset_ad_children)
        program.evaluate_at(precisions, ad=True)
        prev_error = error
        error = program.upper - program.lower
        refinement_steps += 1
    program.apply(reset_ad_children)
    return refinement_steps


def precision_from_grads(program: ExactRealProgram,
                         prev_precisions: List[int],
                         grads: List[float],
                         critical_paths) -> List[int]:
    critical_path = max_gradient(program, grads)
    # Refine the largest precision by an extra step
    return [prec + 6 if i in critical_path else prec + 3
            for i, prec in enumerate(prev_precisions)], critical_path


def max_gradient(program: ExactRealProgram, grads: List[float]) -> Set[ExactRealProgram]:
    nodes = []
    program.apply(lambda node: nodes.append(node))
    widths = [float(node.upper) - float(node.lower) for node in nodes]

    # Find the maximum gradient node in the computation graph
    # compute the (lower_grad - upper_grad), which should be positive
    reversed_grads = list(reversed([(i + 1, (grad[0] - grad[1]) * widths[i + 1]**2) for i, grad in enumerate(grads[1:])]))
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


def refinesiveness(program: ExactRealProgram, grads: List[float],
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
