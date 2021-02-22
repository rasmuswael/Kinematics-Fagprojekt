import requests
subjects = 10
ntrials = 99
import os

originalpath = os.getcwd()

for subject in range(1, subjects):
    path = str(subject).zfill(2)
    os.mkdir(path)
    os.chdir(path)
    for trial in range(1, ntrials):
        url = 'http://mocap.cs.cmu.edu/subjects/'+str(subject).zfill(2)+'/'+str(subject).zfill(2)+'_'+str(trial).zfill(2)+'.amc'
        r = requests.get(url, allow_redirects=True)
        if '404 Not Found' not in r.text:
            open(str(subject).zfill(2)+'_'+str(trial).zfill(2)+'.amc', 'wb').write(r.content)

            url = 'http://mocap.cs.cmu.edu/subjects/' + str(subject).zfill(2) + '/' + str(subject).zfill(2) + '.asf'
            r = requests.get(url, allow_redirects=True)
            open(str(subject).zfill(2)+'.asf', 'wb').write(r.content)

            print("Subject " + str(subject) + ", Trial " + str(trial) + " downloaded")

    os.chdir(originalpath)