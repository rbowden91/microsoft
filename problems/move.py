import sys, json, os, shutil

problem = sys.argv[1]
uuid = sys.argv[2]

with open(problem + '/' + uuid + '/output.json', "r") as text_file:
    output = json.load(text_file)

if output['passes']:
    passed = None
    for check in output['checks']:
        for test in output['checks'][check]:
            if not test['status']:
                break
        else:
            passed = os.path.basename(check)
            break
    if passed is None:
        print('uh oh')
        sys.exit(1)

    # correct
    if output['parse1']:
        directory = 'correct_' + passed

    # correct after fixing headers
    elif output['parse2']:
        directory = 'correct_fixed_' + passed
    # only parsed with empty headers
    elif output['parse3']:
        directory = 'correct_empty_' + passed
    else:
        directory = 'correct_invalid'
elif output['compiles']:
    # incorrect, but compiles
    if output['parse1']:
        directory = 'compiles'
    # incorrect, but compiles after fixing headers
    elif output['parse2']:
        directory = 'compiles_fixed'
    # only compiled with empty headers
    elif output['parse3']:
        directory = 'compiles_empty'
    else:
        directory = 'compiles_invalid'
# incorrect, but parses
elif output['parse1']:
    directory = 'parses'
# incorrect, but parses after fixing headers
elif output['parse2']:
    directory = 'parses_fixed'
# parses with empty headers
elif output['parse3']:
    directory = 'parses_empty'
# doesn't parse
else:
    directory = 'invalid'

shutil.copytree(problem + '/' + uuid, 'sorted/' + problem + '/' + directory + '/' + uuid)
