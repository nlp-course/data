# Adapted from https://github.com/szagoruyko/pytorchviz/blob/master/torchviz/dot.py
from collections import namedtuple
from graphviz import Digraph
import torch
from torch.autograd import Variable

Node = namedtuple('Node', ('name', 'inputs', 'attr', 'op'))

def make_dot(var, params=None):
    """ Produces Graphviz representation of PyTorch autograd graph.

    Blue nodes are the Variables that require grad, orange are Tensors
    saved for backward in torch.autograd.Function

    Args:
        var: output Variable
        params: dict of (name, Variable) to add names to node that
            require grad (TODO: make optional)
    """
    if params is not None:
        assert all(isinstance(p, Variable) for p in params.values())
        param_size_map = {}
        for k in params:
            v = params[k]
            param_size_map[k] = v.size()
            if hasattr(v, 'grad_fn') and (v.grad_fn is not None):
                params[k] = v.grad_fn
        param_map = {id(v): k for k, v in params.items()}

    node_attr = dict(style='filled',
                     shape='box',
                     align='left',
                     fontsize='12',
                     ranksep='0.1',
                     height='0.2')
    dot = Digraph(node_attr=node_attr, graph_attr=dict(size="12,12"))
    seen = set()

    def size_to_str(size):
        return '(' + (', ').join(['%d' % v for v in size]) + ')'

    output_nodes = (var.grad_fn,) if not isinstance(var, tuple) else tuple(v.grad_fn for v in var)

    def get_name(name):
        name = name.replace('Backward0', '')
        name = name.replace('Backward', '')
        if name == 'Mv':
            name = 'Mul'
        return name

    def add_nodes(var):
        if var not in seen:
            if hasattr(var, 'variable'):
                u = var.variable
                name = param_map[id(u)] if params is not None else ''
                node_name = '%s\n %s' % (name, size_to_str(u.size()))
                dot.node(str(id(var)), node_name, fillcolor='lightblue')
            elif var in output_nodes:
                if id(var) in param_map:
                    name = param_map[id(var)] if params is not None else ''
                    node_name = '%s\n%s\n %s' % (get_name(str(type(var).__name__)), name, size_to_str(param_size_map[name]))
                    node_name = '%s\n%s' % (get_name(str(type(var).__name__)), name)
                    dot.node(str(id(var)), node_name, fillcolor='darkolivegreen1')
                else:
                    dot.node(str(id(var)), get_name(str(type(var).__name__)), fillcolor='darkolivegreen1')
            else:
                node_name = get_name(str(type(var).__name__))
                if id(var) in param_map:
                    name = param_map[id(var)]
                    node_name = '%s\n%s\n %s' % (get_name(str(type(var).__name__)), name, size_to_str(param_size_map[name]))
                    node_name = '%s\n%s' % (get_name(str(type(var).__name__)), name)
                    dot.node(str(id(var)), node_name, fillcolor='lightblue')
                else:
                    #dot.node(str(id(var)), get_name(str(type(var).__name__)))
                    dot.node(str(id(var)), node_name)
            seen.add(var)
            if hasattr(var, 'next_functions'):
                for u in var.next_functions:
                    if u[0] is not None:
                        dot.edge(str(id(u[0])), str(id(var)))
                        add_nodes(u[0])
            if hasattr(var, 'saved_tensors'):
                for t in var.saved_tensors:
                    dot.edge(str(id(t)), str(id(var)))
                    add_nodes(t)

    # handle multiple outputs
    if isinstance(var, tuple):
        for v in var:
            add_nodes(v.grad_fn)
    else:
        add_nodes(var.grad_fn)

    resize_graph(dot)

    return dot

def resize_graph(dot, size_per_element=0.15, min_size=12):
    """Resize the graph according to how much content it contains.

    Modify the graph in place.
    """
    # Get the approximate number of nodes and edges
    num_rows = len(dot.body)
    content_size = num_rows * size_per_element
    size = max(min_size, content_size)
    size_str = str(size) + "," + str(size)
    dot.graph_attr.update(size=size_str)
