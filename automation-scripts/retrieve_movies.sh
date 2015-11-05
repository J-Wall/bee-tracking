#!/bin/bash

for i in $(seq 4)
do
	echo 192.168.51.1$i
	for n in $(ssh 192.168.51.1$i ls bee_movies/raspberrypi*)
	do
		echo $n
		if [ ! -h movies_symlinks/${n#"bee_movies/"} ]
		then
			echo copying and deleting
			scp 192.168.51.1$i:/home/pi/$n /mnt/movies/${n#"bee_movies/"} && ssh 192.168.51.1$i rm $n
		else
			echo file exists doing nothing
		fi
	done
done

/home/pi/command_scripts/update_symlinks.sh
