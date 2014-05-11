from collections import defaultdict as hashtable
from pprint import pprint
from blessings import Terminal as T
from functools import partial
from itertools import combinations, chain
from sqlite3 import connect
from string import ascii_letters, digits
from time import clock
from psutil import Process as P; P = P()
from os.path import exists
from shutil import rmtree
from leveldb import LevelDB, WriteBatch
from operator import itemgetter
from multiprocessing import Pool
from sys import stderr as err
from sys import argv
from sh import du

# config

if len(argv) == 2:
    n_records = int(argv[-1])
else:
    n_records = 500

print >>err, '--- STARTING: n =', n_records, '---'

bad_values = set(list(ascii_letters + digits))

time_started = clock()
stats = {'Records.N': n_records,
         #'t_start': time_started,
         }

# helpers

def _merge(a):
    if len(a) == 2:
        return a[0].union(a[1])
    else:
        return a[0]

class timer(object):
    def __init__(self, name='<block>'):
        self.name = name
        self.start_sys  = 0.0
        self.start_user = 0.0
        self.start_rss = 0L
        self.start_disk = 0L
    def __enter__(self):
        cput = P.cpu_times()
        memi = P.memory_info_ex()
        self.start_sys  = cput.system
        self.start_user = cput.user
        self.start_rss  = memi.rss
        self.start_disk = 0L
    def __exit__(self, *args):
        cput = P.cpu_times()
        memi = P.memory_info_ex()
        self.stop_sys  = cput.system
        self.stop_user = cput.user
        self.stop_rss  = memi.rss
        self.stop_disk = 0L
        t_elapsed_sys  = self.stop_sys - self.start_sys
        t_elapsed_user = self.stop_user - self.start_user
        t_elapsed = t_elapsed_sys + t_elapsed_user
        print >>err, T().yellow('timer: {} took {} (user: {}, sys: {}) seconds.'.format(self.name, t_elapsed, t_elapsed_user, t_elapsed_sys))
        print >>err, T().yellow('timer: rss = {} MiB. (change: {} MiB).'.format(self.stop_rss/1048576.0, (self.stop_rss-self.start_rss)/1048576.0))
        print >>err, T().yellow('timer: disk = {} MiB. (change: {} MiB.'.format(self.stop_disk/1048576.0, (self.stop_disk-self.start_disk)/1048576.0))
        print >>err
        #stats[self.name+'.user']  = t_elapsed_user
        #stats[self.name+'.sys']   = t_elapsed_sys
        #stats[self.name+'.total'] = t_elapsed
        #stats[self.name+'.rss'] = self.stop_rss
        #stats[self.name+'.disk'] = self.stop_disk
        stats[self.name+'.Memory'] = self.stop_rss + self.stop_disk
        stats[self.name+'.Runtime'] = t_elapsed

def all_combinations(entities):
    return combinations(entities, 2)

c = lambda v: T().bold_bright_black(str(v))
b = lambda v: T().bold_bright_red(str(v))
e = lambda v: T().underline_white(str(v))

def show(d, f1, f2):
    for k, s in d.items():
        k = str(k)
        print ' +', f1(k), '.'*(20-len(k)), '[', ' '.join(map(f2, sorted(s))), ']'
    return

# load the table into memory

print >>err, c('# step 0: reading the table into memory and encoding attributes')
print >>err, c('#         through numbers to increase performance')
print >>err, c('#         (this is not part of metablocking)')
print >>err

with timer('Setup'):
    block_keys = {}
    block_to_value = {}
    table = hashtable(set)
    clusters = hashtable(set)

    db = connect('cleaned.sqlite3')
    cu = db.cursor()

    ok_chars  = ascii_letters + digits + ' '

    sane_str = lambda c: c in ok_chars

    for record in cu.execute('SELECT id, cluster, name, sort_name, type, area, gender, comment, begin_year, end_year FROM artist_sample ORDER BY cluster, id LIMIT {};'.format(n_records)):
        _id = int(record[0])
        _cl = int(record[1])

        clusters[_cl].add(_id)

        for value in record[2:]:
            if value:
                value = unicode(value).strip()

                if value:
                    values = u''.join(filter(sane_str, value)).lower().split()

                    for value in values:

                        if value in bad_values:
                            continue

                        block = block_keys.get(value, None)

                        if block == None:
                            block = len(block_keys)
                            block_keys[value] = block
                            block_to_value[block] = value

                        table[_id].add(block)

    cu.close()
    db.close()
    del cu, db


    print >>err, c('# step 1: transform the table into a collection of blocks')
    print >>err, c('#         (this is not part of metablocking)')
    print >>err

    def extract_blocks(table):
        blocks = hashtable(set)
        for entity, attributes in table.items(): # do entities need to be sorted in block?
            for attribute in attributes:
                blocks[attribute].add(entity)
        for block, entities in blocks.items(): # yes they do!!!
            entities = list(entities)
            entities.sort()
            blocks[block] = entities
        return blocks

    blocks = extract_blocks(table)

    del table

print >>err, c('# meta 1: create the reverse index from all blocks')
print >>err, c('#         (this is where metablocking starts)')
print >>err, c('#')
print >>err, c('#         the blocks in the reverse index have to be')
print >>err, c('#         in the same order as we process the blocks')
print >>err, c('#         for the sum calculation to work.')
print >>err

def build_rev_idx(blocks):
    rev_idx = hashtable(list) # must be a hashtable of SORTED lists
    for block, entities in sorted(blocks.items()): # add blocks in SORTED order.
        for entity in entities:
            rev_idx[entity].append(block)
    return rev_idx

with timer('RevIdx'):
    rev_idx = build_rev_idx(blocks)

#print 'REVERSE INDEX:'
#show(rev_idx, e, b)
#print

print >>err, c('# meta 2: calculate the "total_weight", "n_distinct_edges",')
print >>err, c('#         and "avg_weight" by iterating through all blocks')
print >>err, c('#         in sorted order.')
print >>err

def get_weight(block, e1, e2):
    #print '    \- get_weight:', '(current:', b(block), ')', \
    #                            e(e1), ' '*(10-len(str(e1))), '-', \
    #                            e(e2), ' '*(10-len(str(e2)))

    blocks_e1 = rev_idx[e1]
    blocks_e2 = rev_idx[e2]

    #print '           \- rev_idx[e1]:', e(e1), '.'*(15-len(str(e1))), '[', \
    #                                    ' '.join(map(b, blocks_e1)), ']'
    #print '           \- rev_idx[e2]:', e(e2), '.'*(15-len(str(e2))), '[', \
    #                                    ' '.join(map(b, blocks_e2)), ']'
    #print '           |'

    common_blocks = 0L
    first_common = False
    for b1 in blocks_e1:
        for b2 in blocks_e2:
            #print '           |', b(b1), ' '*(10-len(str(b1))), '==', \
            #                      b(b2), ' '*(10-len(str(b2))),


            if b1 == b2:
                common_blocks += 1

                if not first_common:
                    # print '&     first common',
                    first_common = True
                    if b1 != block:
                        # print '& NOT current block => return -1.'
                        return -1 # error code
                    else:
                        # print '&     current block => continue.'
                        pass
                else:
                    # print '& NOT first common => continue.'
                    pass
            else:
                # print '=> skip.'
                pass

    #print '           | return', common_blocks
    return common_blocks

with timer('Graph'):
    print >>err, 'CALCULATING total_weight, n_distinct_edges, average_weight'
    total_weight = 0L
    n_distinct_edges = 0L

    for block, entities in sorted(blocks.items()):
        for e1, e2 in all_combinations(blocks[block]):
            weight = get_weight(block, e1, e2)
            if weight != -1:
                total_weight += weight
                n_distinct_edges += 1

    average_weight = float(total_weight) / n_distinct_edges
    print >>err, ' - total_weight:    ', total_weight
    print >>err, ' - n_distinct_edges:', n_distinct_edges
    print >>err, ' - average_weight:  ', average_weight
    print >>err
    stats['Total Weight.N'] = total_weight
    stats['Distinct Edges.N'] = n_distinct_edges
    stats['Average Weight.N'] = average_weight

print >>err, c('# meta 3: re-iterate through all blocks and apply the pruning')
print >>err, c('#         criterion. create the output blocks.')
print >>err


with timer('Pruning'):
    # print 'APPLY PRUNING CRITERION AND OUTPUT NEW BLOCKS'
    new_blocks = hashtable(set)

    for block, entities in sorted(blocks.items()):
        #print ' - working on:', b(block)
        for e1, e2 in all_combinations(blocks[block]):
            weight = get_weight(block, e1, e2)
            #print '           \- weight:', weight,
            if weight < average_weight:
                #print '< ', average_weight, '=> skip.'
                pass
            else:
                new_blocks[block].add((e1, e2))
                #print '>=', average_weight, '=> add to block.'
                #print
        #print '    \- new block:', b(block), '[', ' '.join(map(e, new_blocks[block])), ']'
        #print

    #print 'NEW BLOCKS:'
    #show(new_blocks, b, e)
    #print

print >>err, c('# post 1: measure stuff')
print >>err, c('#         (this is not part of metablocking anymore.)')
print >>err

with timer('Scoring'):

    ground_truth = set()

    n_true_positive = 0L

    for cluster, entities in clusters.items():
        if len(entities) > 1:
            ground_truth = ground_truth.union(sorted(all_combinations(entities)))
        #else:
        #    # single record entity
        #    n_true_positive += 1

    print >>err, '# ground_truth:', len(ground_truth)
    stats['Ground Truth Entity Pairs.N'] = len(ground_truth)

    #all_comparisons = set()
    #for block, comparisons in new_blocks.items():
    #    all_comparisons = all_comparisons.union(comparisons)

    all_comparisons = list(new_blocks.values())
    while len(all_comparisons) > 1:
        for _ in xrange(len(all_comparisons)/2):
            tmp = all_comparisons.pop(0)
            tmp = tmp.union(all_comparisons.pop(0))
            all_comparisons.append(tmp)
    all_comparisons = all_comparisons[0]
    stats['Output Entity Pairs.N'] = len(all_comparisons)

    #all_comparisons = list(new_blocks.values())
    #chunks = lambda l, n: [l[x: x+n] for x in xrange(0, len(l), n)]

    #while len(all_comparisons) > 1:
    #    all_comparisons = map(_merge, chunks(all_comparisons, 2))
    #all_comparisons = all_comparisons[0]

    print >>err, '# all_comparisons:', len(all_comparisons)
    print >>err

    n_true_positive += len(ground_truth.intersection(all_comparisons))
    n_false_positive = len(all_comparisons - ground_truth)
    n_false_negative = len(ground_truth - all_comparisons)

    stats['True Positives.N'] = n_true_positive
    stats['False Positives.N'] = n_false_positive
    stats['False Negatives.N'] = n_false_negative

    recall = float(n_true_positive) / (n_true_positive + n_false_negative)
    precision = float(n_true_positive) / (n_true_positive + n_false_positive)
    f_measure = 2 * precision * recall / (precision + recall)

    print >>err, 'MEASURING QUALITY'
    print >>err, ' - recall:   ', recall
    print >>err, ' - precision:', precision
    print >>err, ' - f-measure:', f_measure
    print >>err

    stats['Recall.Recall'] = recall
    stats['Precision.Precision'] = precision
    stats['F-Measure.F-Measure'] = f_measure

time_stopped = clock()
#stats['time_stopped'] = time_stopped
stats['Overall Runtime.Runtime'] = time_stopped - time_started

print 'REVIDX', stats
