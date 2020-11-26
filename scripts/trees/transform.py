### Grammar and tree transformation functions for NLTK
###
### Prepared for CS187

import collections
import re

import nltk

from nltk import treetransforms
from nltk.corpus import treebank
from nltk.grammar import ProbabilisticProduction, CFG, PCFG, induce_pcfg, Nonterminal, Production
from nltk.parse import pchart
from nltk.tree import Tree

###......................................................................
### Augmented grammar specification format
###
### We provide a gramamr specification format that extends the NLTK string
### specification format. Like NLTK, grammar rules are in the format
### `LHS -> RHS`, where the `RHS` can have multiple alternations with `|`. 
### However, the following extensions are allowed:
###    
###     * Alternatives can be on their own line.
###     * Blank lines and lines with leading whitespace and `#` are ignored.
###     * Unparsable lines are skipped with a warning.
###     * Semantic augmentations can be added at the end of the line after a colon.
###       Augmentations are arbitrary Python expressions.  If no augmentation is 
###       provided but a previous rule had an augmentation, it is used as the 
###       current rule's augmentation.

def parse_augmented_grammar(spec, globals=globals()):
  """Parses the `spec` as a grammar specification in an NLTK-extended format.

  Returns an NLTK CFG grammar and a dictionary from productions to augmentations.
  """
  rules = []                      # accumulating rules
  augments = []                   # ... and their augmentations
  lhs = 'S'                       # current lhs, defaults to S
  augment = (lambda *args : None) # default augmentation does nothing

  augment_str = None
  prev_lhs = ''

  # read in the grammar from the string
  for line in spec.split('\n'):
    # skip blank lines and comment lines
    if re.match(r"\s*(\#.*)?$", line):
      next
    else: 
      match = (re.match(r"\s*(?P<lhs>\w+)\s*->\s*(?P<rhsides>[^:]*)(\:(?P<augment>.*))?$", line)
               or re.match(r"\s*(?P<lhs>)\|\s*(?P<rhsides>[^:]*)(\:(?P<augment>.*))?$", line))
      # skip ill-formed lines with warning
      if not match:
        print(f"Warning - ill-formed: {line}")
        next
      else: 
        rhsides = match.group('rhsides').strip()
        lhs = match.group('lhs').strip() or lhs
        if lhs != prev_lhs:
            augment_str = None
            prev_lhs = lhs
        if match.group('augment'):
          augment_str = match.group('augment').strip()
        # add rules for each rhs
        for rhs in rhsides.split('|'):
          if augment_str is not None:
            if '_RHS' in augment_str:
              #print (augment_str)
              rhs_items = rhs.strip().split()
              rhs_items = [item if (item[0]=='"' or item[0]=="'") else f'"{item}"' for item in rhs_items]
              augment_str_new = augment_str.replace('_RHS', '[' + ','.join(rhs_items) +']')
              #print (augment_str_new)
              augment = eval(augment_str_new, globals)
            else:
              augment = eval(augment_str, globals)
          #print(f"found {lhs} -> {rhs}: {augment}")
          rules.append(f"{lhs} -> {rhs}")
          augments.append(augment)
        
  # build the grammar from the rules
  grammar = nltk.CFG.fromstring(rules)
  # build an augmentation dictionary
  #   *** assumes that grammar.productions() doesn't reorder productions.
  augmentations = collections.defaultdict(lambda x: None)
  for rule, augment in zip(grammar.productions(), augments):
    augmentations[rule] = augment
  return grammar, augmentations

def read_augmented_grammar(file, path='.', globals=globals()):
  """Reads an augmented grammar spec from the file at the path, and 
  returns an NLTK grammar and augmentation dictionary built from it.
  """
  with open(path + '/' + file,'r') as fh:
     spec = fh.read()
  gr, aug = parse_augmented_grammar(spec, globals)
  return (gr, aug)

###......................................................................
### Conversion to and from Chomsky normal form (CNF)
### 
### Assumes that the input grammar has no epsilon productions.

# Convert a grammar to CNF form
def get_cnf_grammar(grammar):
  # First, binarize grammar by introducing new non-terminals
  # Second, remove mixed productions like A -> b C or A -> B c
  # Finally, remove unary nonterminal productions A -> B
  return remove_unary_rules(remove_mixing(binarize(grammar)))

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

# Convert back to the original grammar
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
