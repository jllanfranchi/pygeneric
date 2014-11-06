#!/usr/bin/env python


def readJustinTSV(pathToFile):
    """
    Justin's tab-separated-value files have a single header line:
     * Starts with #
     * No spaces
     * Each field name is followed by a colon and the converter to make the
       string a Python object
     * Currently supported are:
         * int
         * float
         * str
     * Tabs separate each field
     An example of a header line:
           #name:str\tx:float\ty:float\tn:int\n

    The data in the file is tab-separated values, each line must have the same
    number of fields as the header.

    TODO: Use already-built JSON exporter instead!
    """
    with open(pathToFile) as f:
        header = f.readline()[1:].strip().split('\t')
        fileData = []
        while True:
            line = f.readline()
            if line == "":
                break
            line = line.strip().split('\t')
            d = {}
            for (field, value) in zip(header, line):
                name, converter = field.split(':')
                if converter in ['int', 'str', 'float']:
                    d[name] = eval( converter+'("'+value+'")' )
                else:
                    try:
                        val_int = int(value)
                        val_float = float(value)
                        if val_int == val_float:
                            val = val_int
                        else:
                            val = val_float
                    except:
                        val = value
                    finally:
                        d[name] = val
            fileData.append(d)
    return fileData


