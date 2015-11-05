#!/bin/bash


for i in $(seq 4)
do
	ssh 192.168.51.1$i bee_movies/launch_raspivid.sh &
done
