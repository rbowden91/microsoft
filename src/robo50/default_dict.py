from collections import defaultdict

# test, root_transition, transitions (true/false)
root_transitions_dict = lambda f: defaultdict(lambda: { 'true': f(), 'false': f() }) # type:ignore

# default 'null' test
data_dict = lambda f: defaultdict((lambda: root_transitions_dict(f)), null=root_transitions_dict(f))

def get_dict(d, *args):
    for arg in args:
        if arg not in d:
            d[arg] = {}
        d = d[arg]
    return d

def get_dict_default(d, *args):
    for i in range(len(args)-2):
        arg = args[i]
        if arg not in d:
            d[arg] = {}
        d = d[arg]
    if args[-2] not in d:
        d[args[-2]] = args[-1]
    return d
