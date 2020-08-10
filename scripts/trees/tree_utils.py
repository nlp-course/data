#!/usr/bin/env python

from nltk.grammar import CFG, Nonterminal, Production
from nltk.tree import Tree
from nltk import treetransforms


# Convert a grammar to CNF form
# To convert a grammar to CNF:
# `cnf_grammar, cnf_grammar_wunaries = get_cnf_grammar(grammar)`
def get_cnf_grammar(grammar):
  # First, binarize grammar by introducing new non-terminals
  # Second, remove mixed productions A -> b C or A -> B c
  # Finally, remove unary nonterminal productions A -> B
  return remove_unary_rules(remove_mixing(binarize(grammar)))

# Convert back to the original grammar
# To convert a tree output from CKY back to the original form of the grammar:
# `un_cnf(tree, cnf_grammar_wunaries)`
def un_cnf(tree, old_grammar):
  reinsert_unary_chains(tree, old_grammar)
  treetransforms.un_chomsky_normal_form(tree)
  nodeList = [(tree, [])]
  while nodeList != []:
    node, parent = nodeList.pop()
    if isinstance(node, Tree):
      if '$' in node.label():
        nodeIndex = parent.index(node)
        parent.remove(parent[nodeIndex])
        parent.insert(nodeIndex, node[0])
      else:
        for child in node:
          nodeList.append((child, node))

# Binarize grammar by introducing new nonterminals
def binarize(grammar):
  result = []

  for rule in grammar.productions():
    if len(rule.rhs()) > 2:
      # this rule needs to be broken down
      left_side = rule.lhs()
      symbol_names = [tsym.symbol() if not isinstance(tsym, str) else '@'+tsym for tsym in rule.rhs()]
      for k in range(1, len(rule.rhs())-1):
        new_rhs_name = rule.lhs().symbol()+'|<' + '-'.join(symbol_names[k:])+'>'
        new_sym = Nonterminal(new_rhs_name)
        new_production = Production(left_side, (rule.rhs()[k-1], new_sym))
        left_side = new_sym
        result.append(new_production)
      last_prd = Production(left_side, rule.rhs()[-2:])
      result.append(last_prd)
    else:
      result.append(rule)

  n_grammar = CFG(grammar.start(), result)
  return n_grammar

# Remove mixed productions A -> b C or A -> B c
def remove_mixing(grammar):
  result = []
  for rule in grammar.productions():
    if len(rule.rhs()) == 2 and (isinstance(rule.rhs()[0], str) or isinstance(rule.rhs()[1], str)):
      new_rhs = []
      for k in range(2):
        if isinstance(rule.rhs()[k], str):
          new_sym = Nonterminal('$'+rule.rhs()[k])
          new_production = Production(new_sym, (rule.rhs()[k],))
          result.append(new_production)
          new_rhs.append(new_sym)
        else:
          new_rhs.append(rule.rhs()[k])
      new_production = Production(rule.lhs(), new_rhs)
      result.append(new_production)
    else:
      result.append(rule)

  n_grammar = CFG(grammar.start(), result)
  return n_grammar

# Remove unary nonterminal productions A -> B
def remove_unary_rules(grammar):
  result = []
  unary = []
  fake_rules = []
  removed_rules = []
  for rule in grammar.productions():
    if len(rule) == 1 and rule.is_nonlexical():
      unary.append(rule)
    else:
      result.append(rule)

  while unary:
    rule = unary.pop(0)
    removed_rules.append(rule)
    for item in grammar.productions(lhs=rule.rhs()[0]):
      new_rule = Production(rule.lhs(), item.rhs())
      if len(new_rule) != 1 or new_rule.is_lexical():
        result.append(new_rule)
        fake_rules.append(new_rule)
      else:
        unary.append(new_rule)

  n_grammar = CFG(grammar.start(), result)
  return n_grammar, grammar

# Add the previously removed unary productions A -> B back
def reinsert_unary_chains(tree, old_grammar):
  old_unary_productions = [p for p in old_grammar.productions() if len(p) == 1 and p.is_nonlexical()]

  nodeList = [tree]
  while nodeList != []:
    node = nodeList.pop()
    if not isinstance(node, Tree):
      continue
    
    assert len(node) <= 2

    nodeCopy = node.copy()
    children_rhs = [Nonterminal(child.label()) if not isinstance(child, str) else child for child in node]

    possibilities = []
    possibility = [Nonterminal(node.label())]
    query = Production(possibility[-1], children_rhs)
    while query not in old_grammar.productions():
      new_possibilities = [possibility + [p.rhs()[0]] for p in old_unary_productions if p.lhs() == possibility[-1]]
      possibilities.extend(new_possibilities)
      possibility = possibilities.pop(0)
      query = Production(possibility[-1], children_rhs)
      
    # Once a chain has been found, add it back in:
    node[0:] = [] # remove children
    lastnode = node
    for nt in possibility[1:]:
      newnode = Tree(nt.symbol(), [])
      lastnode[0:] = [newnode]
      lastnode = newnode
    lastnode[0:] = [child for child in nodeCopy]

    for child in lastnode:
      nodeList.append(child)

