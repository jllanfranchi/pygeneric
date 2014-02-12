# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

import re
import numpy as np
from IPython.display import display as D
%rehashx

# <codecell>

%cd ~/courses/phys518_critical_phenomena/lecture/

# <codecell>

pdfnum_re = re.compile(r"518_lecture_([0-9]{1,2})\.pdf")

# <codecell>

files = !ls 518_lecture_*.pdf
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

# <markdowncell>

# Merge the PDF's into one

# <codecell>

tempfile = "PHYS_518_lectures.tmp.pdf"
outfile = "PHYS_518_lectures.pdf"
!pdfjoin Lecture*.pdf --outfile $tempfile

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
for name in idxnames:
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

