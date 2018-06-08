# TODO: strings should include \0...
def GetString(interpreter):
    counter = 0
    def helper(args):
        stdin = interpreter.stdin.split('\n')
        interpreter.stdin = stdin[1:]
        stdin = bytearray(stdin[0], 'latin-1')
        nonlocal counter
        name = 'GetString ' + str(counter)
        counter += 1
        interpreter.memory[name] = (['char'], len(stdin), stdin, 'heap')
        return ['string'], name

    # TODO: technically const?
    type_ = [('(builtin)', ['string'], [], [])]
    return type_, helper

def isalpha(interpreter):
    def helper(args):
        return ['int'], bytes(args).decode('latin-1').isalpha()

    type_ = [('(builtin)', ['int'], [None], [['int']])]
    return type_, helper

def islower(interpreter):
    def helper(args):
        return ['int'], bytes(args).decode('latin-1').islower()

    type_ = [('(builtin)', ['int'], [None], [['int']])]
    return type_, helper

def isupper(interpreter):
    def helper(args):
        return ['int'], bytes(args).decode('latin-1').isupper()

    type_ = [('(builtin)', ['int'], [None], [['int']])]
    return type_, helper

def printf(interpreter):
    def helper(args):
        #TODO: is this more complicated than it seems?
        if len(args) == 1:
            interpreter.stdout += args[0]
        else:
            fmt = args[0]
            args = args[1:]
            for i in range(len(args)):
                # TODO: we don't want to do this if the flag is %p and not %s
                if args[i] in interpreter.memory:
                    args[i] = interpreter.memory[args[i]][2].decode('latin-1')
            interpreter.stdout += args[0] % args[1:]
        return ['int'], 1

    # TODO: technically const?
    type_ = [('(builtin)', ['int'], [None], [['*', 'char'], ['...']])]
    return type_, helper

def strlen(interpreter):
    def helper(args):
        # TODO: could iterate over memory til we hit \0
        return ['size_t'], interpreter.memory[args[0]][1]

    # TODO: technically const?
    type_ = [('(builtin)', ['size_t'], [None], [['*', 'char']])]
    return type_, helper


def tolower(interpreter):
    def helper(args):
        return ['int'], bytes(args).decode('latin-1').lower().encode('latin-1')[0]

    type_ = [('(builtin)', ['int'], [None], [['int']])]
    return type_, helper

def toupper(interpreter):
    def helper(args):
        return ['int'], bytes(args).decode('latin-1').upper().encode('latin-1')[0]

    type_ = [('(builtin)', ['int'], [None], [['int']])]
    return type_, helper
