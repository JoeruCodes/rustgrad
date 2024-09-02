  if isinstance(lhs, SumNode):
    return create_node(LtNode(lhs, b)) if isinstance(lhs, SumNode) else create_lt_node(lhs, b)