from collections import defaultdict

# test, root_transition, transitions (true/false)
root_transitions_dict = lambda f: defaultdict(lambda: { 'true': f(), 'false': f() }) # type:ignore

# default 'null' test
data_dict = lambda f: defaultdict((lambda: root_transitions_dict(f)), null=root_transitions_dict(f))
