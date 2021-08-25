import os
from pathlib import Path
from datetime import date
print(Path(os.getcwd()).parent.absolute())
print(os.path.join(Path(os.getcwd()).parent.absolute(),'dataset'))
print(os.path.isdir(os.path.join(Path(os.getcwd()).parent.absolute(),'dataset')))

today = date.today() # Today's date
folder_name = today.strftime("%Y_%m_%d") # YYYY_MM_DD
test = os.path.join(Path(os.getcwd()).parent.absolute(),'logs')
file = os.path.join(test,folder_name+".txt")
sourceFile = open(file, 'a')
print('Hello, Python!', file = sourceFile)
sourceFile.close()
