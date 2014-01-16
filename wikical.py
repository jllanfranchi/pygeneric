#!/usr/bin/env python

import time, calendar, re, sys
import numpy as np

firstweekday = calendar.SUNDAY

width = 20
offset0 = 0
offset1 = 0
alignment = 'alignc'
monthNameSep = ' '
weekdaySep = ' '
labels = {0:'Mo', 1:'Tu', 2:'We', 3:'Th', 4:'Fr', 5:'Sa', 6:'Su'}
todayColor = '${color5}'
weekdayNamesColor = '${color #bbbbbb}'
todayStringColor = '${color5}'
separator = ' '
todayString = time.strftime('%a %b ') + str(int(time.strftime('%d'))) + \
        time.strftime(', %Y')
todayString2 = time.strftime('%Y.%m.%d %H:%M')

calendar.setfirstweekday(firstweekday)
calGen = calendar.Calendar()
calGen.setfirstweekday(firstweekday)

weekendDays = [calendar.SATURDAY, calendar.SUNDAY]
weekdays = [calendar.MONDAY, calendar.TUESDAY, calendar.WEDNESDAY,
            calendar.THURSDAY, calendar.FRIDAY]

weekdayOrder = [ np.mod(n,7) for n in range(firstweekday,firstweekday+7) ]
weekdayNames = [ labels[n] for n in weekdayOrder ]
weekdayNames = weekdaySep.join(weekdayNames)

if len(sys.argv) >= 2:
    color1 = sys.argv[1]
else:
    color1 = 'color6'

if len(sys.argv) >= 3:
    color2 = sys.argv[2]
else:
    color2 = 'color2'

if len(sys.argv) >= 4:
    color3 = sys.argv[3]
else:
    color3 = 'color4'

localtime = time.localtime(time.time())
today = int(localtime[2])
thisMonth = (localtime[0], localtime[1])

if thisMonth[1] == 1:
    lastMonth = (thisMonth[0]-1, 12)
else:
    lastMonth = (thisMonth[0], thisMonth[1]-1)

if thisMonth[1] == 12:
    nextMonth = (thisMonth[0]+1, 1)
else:
    nextMonth = (thisMonth[0], thisMonth[1]+1)

#-- User-defined
months = [{'year': lastMonth[0], 'month': lastMonth[1],
           'weekdayColor': '${color #686868}',
           'weekendColor': '${color #888888}',
           'abbrevColor': '${color #bbbbbb}' },

          {'year': thisMonth[0], 'month': thisMonth[1],
           'weekdayColor': '${color #5b6dad}',
           'weekendColor': '${color #7f8ed3}',
           'abbrevColor': '${color #bbbbbb}' },

          {'year': nextMonth[0], 'month': nextMonth[1],
           'weekdayColor': '${color #686868}',
           'weekendColor': '${color #888888}',
           'abbrevColor': '${color #bbbbbb}' }]

for monthNum in range(len(months)):
    m = months[monthNum]
    months[monthNum]['seqNum'] = monthNum
    months[monthNum]['genericCal'] = \
            calGen.monthdays2calendar(m['year'],m['month'])
    months[monthNum]['daysInMonth'] = np.max(
               calGen.monthdayscalendar(m['year'],m['month'])[-1])
    months[monthNum]['abbrev'] = \
            time.strftime('%b', (m['year'],m['month'],0,0,0,0,0,0,0))
    months[monthNum]['name'] = \
            time.strftime('%B', (m['year'],m['month'],0,0,0,0,0,0,0))

#-- Initialize calendar matrix
calMatrix = [[]]

#-- Initialize row and column pointers
row = 0
column = len(calMatrix[row])-1

#-- Fill initial blank days (only need first week of first month displayed)
for day in months[0]['genericCal'][0]:
    if day[0] == 0:
        calMatrix[0].append({'str':'  ', 'color': months[0]['weekdayColor']})
        #-- Incr. column ptr (ALWAYS on first row, so no need to incr. row ptr)
        column += 1
    else:
        break

#-- Loop through each month, recording the days & appropriate colors
for month in months:
    
    #-- Loop through each day of the month
    for week in month['genericCal']:
        for day in week:
            if day[0] == 0:
                pass
            else:
                #-- Increment row and column pointers
                column += 1
                if column > 6:
                    calMatrix.append([])
                    row += 1
                    column = 0

                #-- Record row and column pointers for FIRST day of the month
                if day[0] == 1:
                    month['fdotmPointer'] = (row, column)
                    month['fdotmDayOfTheWeek'] = (day[0], day[1])

                #-- Record row and column pointers for LAST day of the month
                if day[0] == month['daysInMonth']:
                    month['ldotmPointer'] = (row, column)
                    month['ldotmDayOfTheWeek'] = (day[0], day[1])

                #-- Set color if date is today
                if month['month'] == thisMonth[1] and day[0] == today:
                    color = todayColor
                #-- Set color if date is not today but on a weekend
                elif day[1] in weekendDays:
                    color = month['weekendColor']
                #-- Set color if date is not today but on a weekday
                else:
                    color = month['weekdayColor']
                    
                #-- Append date to calendar matrix
                calMatrix[row].append(
                    {'str': str(day[0]).rjust(2, ' '), 'color': color} )

#-- Fill final blank days (only need final week of final month displayed)
for day in months[-1]['genericCal'][-1][::-1]:
    if day[0] == 0:
        calMatrix[-1].append({'str':'  ', 'color': months[-1]['weekdayColor']})
    else:
        break

#-- Create month abbreviation matrix
monthAbbrevMat = []
for r in range(len(calMatrix)):
    monthAbbrevMat.append([{'str': ' ', 'color': None}])

for month in months:
    month['labelRow0'] = int(np.floor(float(
        month['ldotmPointer'][0] - month['fdotmPointer'][0] + 1 -
        len(month['abbrev']) ) / 2.0) + month['fdotmPointer'][0])
    for n in range(len(month['abbrev'])):
        row = month['labelRow0'] + n
        monthAbbrevMat[row][0]['str'] = str.upper(month['abbrev'])[n]
        monthAbbrevMat[row][0]['color'] = month['abbrevColor']

#-- Concatenate month abbreviations with calendar and add spaces
color = None
rowStrings = []
for row in range(len(calMatrix)):
    rowString = ''
    for entry in monthAbbrevMat[row] + calMatrix[row]:
        if color != entry['color'] and entry['color'] != None:
            color = entry['color']
            rowString += color
        rowString += entry['str'] + separator
    rowString = rowString[0:-len(separator)]
    rowStrings.append(rowString)

#sys.stdout.write('${alignc}' + ' ' + separator + todayStringColor + todayString2 + '\n\n')
sys.stdout.write('${alignc}' + ' ' + separator + todayStringColor + todayString + '\n')
sys.stdout.write('${alignc}' + ' ' + separator + weekdayNamesColor + weekdayNames + '\n')
sys.stdout.write('${alignc}' + '\n${alignc}'.join(rowStrings) + '\n')