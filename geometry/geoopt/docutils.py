import re


def insert_docs(doc, pattern=None, repl=None):
    doc = "" if doc is None else doc
    def wrapper(fn):
        
        if pattern is not None:
            if repl is None:
                raise RuntimeError("need repl parameter")
            fn.__doc__ = re.sub(pattern, repl, doc)
        else:
            fn.__doc__ = doc
        return fn

    return wrapper
