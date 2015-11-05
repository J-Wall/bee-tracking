#!/bin/bash

cd /home/pi/command_scripts

for i in $(seq 4)
do
	scp do_raspivid.sh 192.168.51.1$i:bee_movies/do_raspivid.sh &
	scp launch_raspivid.sh 192.168.51.1$i:bee_movies/launch_raspivid.sh &
#	scp timed_ping.sh 192.168.51.1$i:timed_ping.sh &
done
