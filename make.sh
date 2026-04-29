#!/bin/bash
# echo "test"
for ((i=0;i<=1;i++))
do
for thread_num in 1 2 4 8 16 32 64 128
do
echo thread_num = ${thread_num}
cd /home/HGBTree
sudo numactl --cpunodebind=1 --membind=1 ./main-gu-zipfian -t ${thread_num} 
cd /pmem0/HGBTree
sudo rm pool
sleep 2
done
echo "-------"
done