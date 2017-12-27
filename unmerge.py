#!/usr/bin/env python

"""
Split a file that git has formatted with merge conflict notation into two
separate files.
"""


from __future__ import absolute_import, division, print_function

from argparse import ArgumentParser
from os.path import isfile, splitext


S_BOTH = 0
S_F0_ONLY = 1
S_F1_ONLY = 2


def unmerge(fpath):
    """Take a file that has been marked up by git with merge conflicts and
    write the two original (conflicting) source files.

    The new files will be named like the original, but suffixed by
    "_<commit/branch><ext>" where the extension is taken from the original
    filename.

    Parameters
    ----------
    fpath : string
        Path to the file to "unmerge"

    Returns
    -------
    f0_path : string
        Path to the new file created, file 0

    f0 : list of strings
        Each line of file 0 (including newlines)

    f1_path : string
        Path to the new file created, file 1

    f1 : list of strings
        Each line of file 1 (including newlines)

    """
    basepath, ext = splitext(fpath)
    with open(fpath, 'r') as fhandle:
        contents = fhandle.readlines()

    f0, f1 = [], []
    f0_name, f1_name = None, None
    state = S_BOTH
    for line_no, line in enumerate(contents):
        if line.startswith('<<<<<<< '):
            if state != S_BOTH:
                raise ValueError('Line {}: got "<<<<<<< " but not in S_BOTH'
                                 .format(line_no))
            state = S_F0_ONLY
            if f0_name is None:
                f0_name = line.lstrip('<<<<<<< ').strip()
            continue
        elif line.startswith('======='):
            if state != S_F0_ONLY:
                raise ValueError('Line {}: got "=======" but not in S_F0_ONLY'
                                 .format(line_no))
            state = S_F1_ONLY
            continue
        elif line.startswith('>>>>>>> '):
            if state != S_F1_ONLY:
                raise ValueError('Line {}: got ">>>>>>> but not in S_F1_ONLY'
                                 .format(line_no))
            state = S_BOTH
            if f1_name is None:
                f1_name = line.lstrip('>>>>>>> ').strip()
            continue

        if state in (S_BOTH, S_F0_ONLY):
            f0.append(line)
        if state in (S_BOTH, S_F1_ONLY):
            f1.append(line)

    new_f0_path = basepath + '_' + f0_name + ext
    new_f1_path = basepath + '_' + f1_name + ext

    for new_fpath, new_contents in zip((new_f0_path, new_f1_path), (f0, f1)):
        if isfile(new_fpath):
            print('"{}" already exists, not overwriting.'.format(new_fpath))
        else:
            with open(new_fpath, 'w') as outfile:
                outfile.writelines(new_contents)

    return new_f0_path, f0, new_f1_path, f1


def parse_args(descr=__doc__):
    """Parse command line arguments"""
    arg_parser = ArgumentParser(description=descr)
    arg_parser.add_argument('fpath')
    args = arg_parser.parse_args()
    return args


def main():
    """Main function if calling as script"""
    args = parse_args()
    unmerge(args.fpath)


if __name__ == '__main__':
    main()
