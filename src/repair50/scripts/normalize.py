def main():
    generator = RemoveDecls()
    parser = c_parser.CParser()
    try:
        cfile = preprocess_file(sys.argv[1], cpp_path='cpp', cpp_args=[r'-I../fake_libc_include'])
        ast = parser.parse(cfile)
    except Exception as e:
        print('uh oh2', e)
        sys.exit(1)
    renamed_code = generator.visit(ast)
    cgen = c_generator.CGenerator()
    print(cgen.visit(renamed_code))
