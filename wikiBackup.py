#!/usr/bin/env python

import sys
import os
import time
import subprocess
from genericUtils import wstderr, wstdout


#==============================================================================
# 1. Create a new directory to store the snapshot, named with a label and
#    datetime
#==============================================================================

startTimeSec = time.time()

startTime = time.strftime("%Y-%m-%dT%H%M%S") + \
        "{0:+05d}".format(-int(round(time.timezone/3600)*100))

#homeDir = os.path.expanduser("~")
homeDir = os.path.join("home", "justin")
backupBaseDir = os.path.join(homeDir, "wiki_archive")
backupSubDir = os.path.join(dataBaseDir, time.strftime("%Y"),
                          time.strftime("%m-%b"))

#-- Create directory if necessary (and if possible)
wstderr("\nCreating data directory structure...\n")
if os.path.exists(dataDir):
    createdDataDir = False
    if not os.path.isdir(dataDir):
        raise Exception("Wanted to create directory " + dataDir +
                        " but this path is a file.")
else:
    createdDataDir = True
    os.makedirs(dataDir, mode=0770)

#==============================================================================
# 2. Take wiki offline: sudo service apache2 stop
#==============================================================================

subprocess.call("service apache2 stop")

#==============================================================================
#-- 3. Backup wiki text and files and place dump in backup directory
#      a. /etc/mediawiki (apache.conf, cherokee, conf, LocalSettings.php)
#      b. /var/lib/mediawiki/images (uploaded wiki content files)
#      b. /usr/share/mediawiki-extensions (extensions)
#==============================================================================

subprocess.call("php /usr/share/mediawiki/maintenance/dumpBackup.php --full --uploads > ~/wiki_backups/t1.xml")
subprocess.call("cp -arL /home/wiki " + dataDir)
subprocess.call("cp -arL /home/wiki " + dataDir)
subprocess.call("cp -arL /etc/mediawiki " +
                os.path.join(dataDir, "etc", "mediawiki"))
subprocess.call("cp -arL /etc/mediawiki-extensions " +
                os.path.join(dataDir, "etc", "mediawiki-extensions"))


#==============================================================================
#-- 4. Add an identity file in the directory
#==============================================================================

#-- 5. tar and lzma compress (with plzip) (tar cfa archive.tar.lzma stuff) the contents of the backup directory

#-- 6. generate a sha512 hash from the compressed archive

#-- 7. remove original files

#-- 8. 


