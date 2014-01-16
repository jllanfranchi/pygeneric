import re, os, sys

def findFiles(directory, regex):
    if isinstance(regex, (str, unicode)):
        regex = re.compile(regex)
    for root, dirs, files in os.walk(directory):
        for basename in files:
            if len(regex.findall(basename)) > 0:
                filename = os.path.join(root, basename)
                yield (root, basename) #filename

def wstdout(x):
    sys.stdout.write(x)
    sys.stdout.flush()

def wstderr(x):
    sys.stderr.write(x)
    sys.stderr.flush()


