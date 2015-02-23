# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

import re
import numpy as np
from IPython.display import display as D
%rehashx

# <codecell>

%cd ~/courses/phys572_lasers/lecture/

# <codecell>

pdfnum_re = re.compile(r"lecture([0-9]{1,2})\.pdf", re.IGNORECASE)

# <codecell>

files = !ls lecture*
D(files)

# <codecell>

newfiles = []
idxnames = []
for f in files:
    filenum = int(pdfnum_re.findall(f)[0])
    pdfname = r"Lecture\ "+format(filenum, "02d")+".pdf" 
    idxname = r"Lecture "+format(filenum, "d")
    newfiles.append(pdfname)
    idxnames.append(idxname)
    !cp $f $pdfname
    print filenum, pdfname, idxname

# <markdowncell>

# Merge the PDF's into one; note that the --rotateoversize false keeps large pages from being rotated, which happened in these notes.
# 
# **TODO**: Modify to figure out the (or each) page size(s) automatically and use that, or something reasonable given the measurements.

# <codecell>

tempfile = "lectures.tmp.pdf"
outfile = "PHYS_572_lectures.pdf"
!pdfjoin --paper a4paper --rotateoversize false Lecture*.pdf --outfile $tempfile

# <markdowncell>

# Sort the files and correspondig index entries

# <codecell>

sortind = np.argsort(newfiles)
newfiles = [ newfiles[i] for i in sortind ]
idxnames = [ idxnames[i] for i in sortind ]
D(idxnames)

# <codecell>

r = re.compile(r"Pages:\s*([0-9]+)")
idxentries = []
pagenum = 1
for (f, name) in zip(files, idxnames):
    idxentry = r"[/Page " + format(pagenum,"d") \
        + r" /View [/XYZ null null null] /Title (" \
        + name \
        + r") /OUT pdfmark"
    idxentries.append(idxentry)
    pages_s = !pdfinfo $f | grep Pages
    pages = int(r.findall(pages_s[0])[0])
    pagenum += pages
idx = "\n".join(idxentries)

# <codecell>

with file("index.info", "w") as idxinfo:
    idxinfo.write(idx)

# <markdowncell>

# Add the index to the pdf; instructions on creating index in PDF file obtained here:
# 
# http://linproject.blogspot.com/2012/06/adding-index-to-your-pdf-file.html

# <codecell>

!gs -sDEVICE=pdfwrite -q -dBATCH -dNOPAUSE \
    -sOutputFile=$outfile index.info -f $tempfile 

# <codecell>

!rm -f $tempfile

