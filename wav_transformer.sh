#!/bin/bash

echo "Do you want to remove the original file(y/n)?"
read accept

for file in ./data/train/*
do
	if [ "${file##*.}" != "wav" ] 
	then 
		filename="${file%.*}.wav"
		sox ${file} ${filename} 
		if [ "$accept" == "y" ];then rm ${file};fi
	fi
done

for file in ./data/test/*
do
	if [ "${file##*.}" != "wav" ] 
	then 
		filename="${file%.*}.wav"
		sox ${file} ${filename} 
		if [ "$accept" == "y" ];then rm ${file};fi
	fi
done

echo "complete"
