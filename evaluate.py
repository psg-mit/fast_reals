from typing import List

from exact_real_program import ExactRealProgram


def evaluate(program: ExactRealProgram,
             precision_bound: float,
             ad: bool = False,
             initial_precision: int = 30) -> ExactRealProgram:
    """ Uniformly refine the program until the given precision bound is met. """
    precision = initial_precision
    if ad:
        program.lower_grad, program.upper_grad = 1, 1
    program.apply(reset_ad_children)
    program.evaluate(precision, ad)

    error = program.upper - program.lower
    refinement_steps = 0
    while error > precision_bound:
        program.apply(reset_ad_children)
        precision += 3
        program.evaluate(precision, ad)
        error = program.upper - program.lower
        refinement_steps += 1
    return refinement_steps


def reset_ad_children(program: ExactRealProgram):
    program.ad_lower_children, program.ad_upper_children = [], []


def evaluate_using_derivatives(program: ExactRealProgram,
                               precision_bound: float,
                               initial_precisions: List[int]) -> ExactRealProgram:
    """ Uniformly refine the program until the given precision bound is met. """
    precisions = initial_precisions
    program.lower_grad, program.upper_grad = 1, 1
    program.apply(reset_ad_children)
    program.evaluate_at(precisions, ad=True)

    error = program.upper - program.lower
    refinement_steps = 0
    while error > precision_bound:
        grads = []
        program.apply(lambda program: grads.append(program.grad()))
        # TODO precision updates
        precisions = precision_from_grads(program, precisions, grads)
        program.apply(reset_ad_children)
        program.evaluate_at(precisions, ad=True)
        error = program.upper - program.lower
        refinement_steps += 1
    return refinement_steps


def precision_from_grads(program: ExactRealProgram,
                         prev_precisions: List[int],
                         grads: List[float]) -> List[int]:
    # Find the maximum gradient node in the computation graph
    argmax = max([(sum(grad), i) for i, grad in enumerate(grads)], key=lambda x: x[0])[1]

    # Get the set of nodes on the path from the maximum gradient node to the root
    nodes = []
    program.apply(lambda node: nodes.append(node))
    max_gradient_node = nodes[argmax]

    critical_path = set()
    node_to_ind = {node: i for i, node in enumerate(nodes)}
    node = max_gradient_node
    while node is not None:
        critical_path.add(node_to_ind[node])
        node = node.parent

    # Refine the largest precision by an extra step
    return [prev_prec + 6 if i in critical_path else prev_prec + 3 for i, prev_prec in enumerate(prev_precisions)]
