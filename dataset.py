import sqlite3
import collections
import string
import tabulate

# helpers

ok_chars  = string.ascii_letters + string.digits + ' '
sane_str = lambda c: c in ok_chars
bad_values = set(string.ascii_letters + string.digits) # single letters/digits

query_string = '''
    SELECT id,
           cluster,
           name,
           sort_name,
           type,
           area,
           gender,
           comment,
           begin_year,
           end_year
      FROM artist_sample
  ORDER BY cluster, id;
'''

def extract_stats(ht):
    n_ht = len(ht)
    s_min =  999999999
    s_max = -999999999
    s_sum = 0L
    for k, s in ht.items():
        s_min = min(s_min, len(s))
        s_max = max(s_max, len(s))
        s_sum += len(s)
    s_avg = float(s_sum) / n_ht

    return n_ht, s_min, s_max, s_avg

step_size =  [1000, 2000, 5000, 10000, 20000, 30000]
stop_size = max(step_size)

# connect to database
db = sqlite3.connect('cleaned.sqlite3')
cu = db.cursor()

# value->bid and bid->value can stay the same across subsets
value_to_bid = {}
bid_to_value = {}

# output statistics / helpers
stats = collections.defaultdict(list)

def dpt(k, y):
    stats[k].append(y)

# associations... kept globally for incremental approach.
entity2block = collections.defaultdict(set)
block2entity = collections.defaultdict(set)
entity2clust = collections.defaultdict(set)
clust2entity = collections.defaultdict(set)
block2clustr = collections.defaultdict(set)
clustr2block = collections.defaultdict(set)

for record in cu.execute(query_string):
    _id = int(record[0])
    _cl = int(record[1])

    # add cluster-entity associations
    clust2entity[_cl].add(_id)
    entity2clust[_id].add(_cl)

    for value in record[2:]:
        if value:
            # value is not none

            value = unicode(value).strip()

            if value:
                # value is not an empty string

                values = u''.join(filter(sane_str, value)).lower().split()

                for value in values:

                    if value in bad_values:
                        continue

                    bid = value_to_bid.get(value, None)

                    if bid == None:
                        bid = len(value_to_bid)
                        value_to_bid[value] = bid
                        bid_to_value[bid] = value

                    # add entity-block, and cluster-block associations
                    entity2block[_id].add(bid)
                    block2entity[bid].add(_id)
                    clustr2block[_cl].add(bid)
                    block2clustr[bid].add(_cl)

    n_records = len(entity2block)

    if n_records in step_size:
        # calculate statistics
        print 'calculating statistics... n_records =', n_records

        #EC = extract_stats(entity2clust)
        CE = extract_stats(clust2entity)
        EB = extract_stats(entity2block)
        BE = extract_stats(block2entity)
        #CB = extract_stats(clustr2block)
        #BC = extract_stats(block2clustr)

        # 1. table of input blocks
        # ------------------------

        #  - n_records
        dpt('n-records', n_records)

        #  - n_blocks
        dpt('n-blocks', BE[0])

        #  - n_clusters
        dpt('n-clusters', CE[0])

        #  - block size: min, max, avg
        dpt('blocksize-min', BE[1])
        dpt('blocksize-max', BE[2])
        dpt('blocksize-avg', "{0:.2f}".format(BE[3]))

        #  - n_sebs (single entity blocks)
        dpt('n-sebs', len(filter(lambda (k, v): len(v)==1, block2entity.items())))

        #  - bpe: min, max, avg (blocks per entity)
        dpt('bpe-min', EB[1])
        dpt('bpe-max', EB[2])
        dpt('bpe-avg', "{0:.2f}".format(EB[3]))

    if n_records == stop_size:
        break

cu.close()
db.close()

# output data for rendering
n_records = stats['n-records']

for k in stats:
    with file('report/dataset-stats/'+k, 'w') as fh:
        for i, v in enumerate(stats[k]):
            print >>fh, n_records[i], v

table = []

headers = [
    'n-records',
    'n-clusters',
    'n-blocks',
    'n-sebs',
    'blocksize-min',
    'blocksize-max',
    'blocksize-avg',
    'bpe-min',
    'bpe-max',
    'bpe-avg',
]
print_header = [
    'Records',
    'Clusters',
    'Blocks',
    '1-E. Blocks.',
    'Min.',
    'Max.',
    'Avg.',
    'Min.',
    'Max.',
    'Avg.',
]

for i in xrange(len(n_records)):
    row = []
    for k in headers:
        row.append(stats[k][i])
    table.append(row)

with file('report/dataset-table.tex', 'w') as fh:
    print >>fh, tabulate.tabulate(table, headers=print_header, tablefmt='latex')
