#!/bin/bash

RUN_FILE="do_raspivid.run"

sudo date --set="$(ssh pi@192.168.51.10 'date -u')" 1>/dev/null

cd /home/pi/bee_movies

if [ -f $RUN_FILE ]
then
	echo "Already Filming, exiting now"
	exit 1
fi
touch $RUN_FILE

#echo "Starting filming"
for i in $(seq 20)
do
	n=$(date +$(hostname)-%F-%H-%M-%S.h264)
	raspivid -w 800 -h 600 -o $n -fps 25 -t $((60 * 60 * 1000))
	scp $n pi@192.168.51.10:/mnt/movies/$n && rm $n
done

rm $RUN_FILE
