#!/usr/bin/env python

import sys
import collections
import tree
from six import iteritems, itervalues
from six.moves import zip

try:
    _, parsefilename, goldfilename = sys.argv
except:
    sys.stderr.write("usage: evalb.py <parse-file> <gold-file>\n")
    sys.exit(1)

def _brackets_helper(node, i, result):
    i0 = i
    if len(node.children) > 0:
        for child in node.children:
            i = _brackets_helper(child, i, result)
        j0 = i
        if len(node.children[0].children) > 0: # don't count preterminals
            result[node.label, i0, j0] += 1
    else:
        j0 = i0 + 1
    return j0

def brackets(t):
    result = collections.defaultdict(int)
    _brackets_helper(t.root, 0, result)
    return result

matchcount = parsecount = goldcount = 0

for parseline, goldline in zip(open(parsefilename), open(goldfilename)):
    gold = tree.Tree.from_str(goldline)
    goldbrackets = brackets(gold)
    goldcount += sum(itervalues(goldbrackets))

    if parseline.strip() in ["0", ""]:
        continue
    
    parse = tree.Tree.from_str(parseline)
    parsebrackets = brackets(parse)
    parsecount += sum(itervalues(parsebrackets))

    for bracket,count in iteritems(parsebrackets):
        matchcount += min(count,goldbrackets[bracket])

print("%s\t%d brackets" % (parsefilename, parsecount))
print("%s\t%d brackets" % (goldfilename, goldcount))
print("matching\t%d brackets" % matchcount)
print("precision\t%s" % (float(matchcount)/parsecount))
print("recall\t%s" % (float(matchcount)/goldcount))
print("F1\t%s" % (2./(goldcount/float(matchcount) + parsecount/float(matchcount))))
