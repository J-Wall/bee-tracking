#!/bin/bash

cd /mnt/movies
for n in $(ls raspberrypi*)
do
	if [ ! -h /home/pi/movies_symlinks/$n ]
	then
	ln -s $n /home/pi/movies_symlinks/$n
	fi
done
