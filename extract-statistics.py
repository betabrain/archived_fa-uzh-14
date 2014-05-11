import sys
import collections
import pprint

def read_stats(path):
    with file(path) as fh:
        for line in fh:
            d = eval(line[line.find(' '):])
            yield d

def to_timelines(stats):
    tls = collections.defaultdict(list)

    for dpt in sorted(stats, key=lambda dpt: dpt['Records.N']):
        for k, v in dpt.items():
            tls[k].append(v)

    return dict(tls)

def write_data_files(prefix, keys, timelines):
    for key in keys:
        with file(prefix+key.replace(' ', '-'), 'w') as fh:
            for n, val in zip(timelines['Records.N'], timelines[key]):
                print >>fh, n, val

#pprint.pprint(list(read_stats('statistics.batch.txt')))

batch_stats  = to_timelines(read_stats('statistics.batch.txt'))
revidx_stats = to_timelines(read_stats('statistics.revidx.txt'))

#pprint.pprint(batch_stats)

batch_keys = list(batch_stats.keys())
batch_keys.sort()

write_data_files('report/statistics/BA-', batch_keys, batch_stats)

revidx_keys = list(revidx_stats.keys())
revidx_keys.sort()

write_data_files('report/statistics/RI-', revidx_keys, revidx_stats)

shared_keys = list(set(batch_keys).intersection(set(revidx_keys)))
shared_keys.sort()

pprint.pprint(shared_keys)

#plot_tex = '''
#\\begin{tikzpicture}
#  \\begin{axis}[title=%s, xlabel={$Records$}, ylabel={$%s$}, ]
#    \\addplot[red, mark=+] table{statistics/%s};
#    \\addplot[blue, mark=+] table{statistics/%s};

#  \\end{axis};
#\\end{tikzpicture};
#'''

