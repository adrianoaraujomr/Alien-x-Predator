#!/bin/bash

train_alien=$(echo './org-data/train/alien/')
train_preda=$(echo './org-data/train/predator/')
tests_alien=$(echo './org-data/validation/alien/')
tests_preda=$(echo './org-data/validation/predator/')

if test -f "./train.csv"; then
	rm "./train.csv"
fi
if test -f "./test.csv"; then
	rm "./test.csv"
fi
if ! test -d "./data"; then
	mkdir "./data"
	mkdir "./data/train"
	mkdir "./data/test"

	for i in `ls $train_alien`
	do
		cp $train_alien$i "./data/train/0-"$i
	done

	for i in `ls $train_preda`
	do
		cp $train_preda$i "./data/train/1-"$i
	done

	for i in `ls $tests_alien`
	do
		cp $tests_alien$i "./data/test/0-"$i
	done

	for i in `ls $tests_preda`
	do
		cp $tests_preda$i "./data/test/1-"$i
	done
fi

touch "./train.csv"
echo "id,class" > "./train.csv"
for i in `ls ./data/train`
do
	idx=$(echo "$i" | tr -s " " | cut -d " " -f9 | cut -d "-" -f1)
	echo "$i,$idx" >> "./train.csv"
done

touch "./test.csv"
echo "id,class" > "./test.csv"
for i in `ls ./data/test`
do
	idx=$(echo "$i" | tr -s " " | cut -d " " -f9 | cut -d "-" -f1)
	echo "$i,$idx" >> "./test.csv"
done
